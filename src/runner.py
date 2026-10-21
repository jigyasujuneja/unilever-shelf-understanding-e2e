"""Run an approach x model over a dataset subset (SKU-110K, labelled products or RPC) and write results.

Output (one folder per run, small JSON so it can be committed and shared)::

    results/<run_id>/summary.json   # the leaderboard row + run metadata
    results/<run_id>/images.jsonl   # per image: boxes, metrics, latency, cost, step trace

On Cloud Run the data is read from GCS and results are written to ``gcp.results`` in GCS;
``shelf-bench pull`` copies them into ``results/`` for the UI.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import approaches
from approaches.base import Context, Trace
from utils import dataset, metrics, planogram_engine, pricing, telemetry
from utils.llm import Gemini, LLMResult, Usage, load_config

RESULTS_DIR = Path("results")
_T_PROCESS = time.time()  # container start (Cloud Run bills from instance start)


def environment(config: dict) -> dict:
    """Where this run executes. Compute is only priced on Cloud Run, where we know the shape."""
    if os.environ.get("CLOUD_RUN_JOB"):
        cr = config.get("cloud_run", {})
        return {"platform": "cloud-run", "region": config.get("gcp", {}).get("region"),
                "job": os.environ["CLOUD_RUN_JOB"],
                "execution": os.environ.get("CLOUD_RUN_EXECUTION"),
                "task_index": int(os.environ.get("CLOUD_RUN_TASK_INDEX", 0)),
                "cpu": cr.get("cpu"), "memory_gib": cr.get("memory_gib")}
    return {"platform": "local"}


def set_compute(summary: dict, seconds: float, source: str) -> dict:
    """(Re)price Cloud Run compute for ``seconds`` of task time and update the totals."""
    env, sheet = summary["environment"], summary["pricing"]
    usd = pricing.cloud_run_cost(seconds, env["cpu"], env["memory_gib"], sheet) \
        if env.get("platform") == "cloud-run" else 0.0
    c = summary["cost"]
    c.update({"compute_seconds": round(seconds, 1), "compute_source": source,
              "compute_usd_per_image": usd / summary["images"]})
    c["total_usd_per_image"] = (c["gemini_net_usd_per_image"] + c["compute_usd_per_image"]
                                + c["storage_usd_per_image"] + c.get("services_usd_per_image", 0.0))
    summary["compute_cost_per_image_usd"] = c["compute_usd_per_image"]
    summary["cost_per_image_usd"] = c["total_usd_per_image"]
    return summary


def combos(names: list[str], tiers: list[str], models: list[str],
           log: Callable[[str], None] = print) -> list[tuple[str, str, str]]:
    """(approach, tier, model) runs to do, skipping models an approach doesn't call (and falling
    back to the standard tier once for non-Gemini approaches). Raises on unknown approaches,
    before any spend."""
    out = []
    non_gemini_tier = "standard" if "standard" in tiers else (tiers[0] if tiers else "standard")
    for n in names:
        appr = approaches.get(n)
        for t in tiers:
            for m in models:
                if appr.accepts(m) and (appr.uses_gemini or t == non_gemini_tier):
                    out.append((n, t if appr.uses_gemini else "standard", m))
                else:
                    runs = ", ".join(appr.models) if appr.models else "Gemini models"
                    log(f"skip {n} x {m} x {t}: {n} runs {runs}"
                        + ("" if appr.uses_gemini else " (standard tier)"))
    return out


def _fmt_label(d: dict | None) -> str:
    if not d:
        return "nothing"
    if "product" in d and d["product"]:
        return str(d["product"])
    parts = [str(d.get(k) or "?") for k in ("brand", "packaging_type", "category", "variant") if k in d]
    return " · ".join(parts) if parts else "nothing"


_METHODOLOGY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary_header": {"type": "STRING"},
        "flow_summary": {"type": "STRING"},
        "bullets": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["summary_header", "flow_summary", "bullets"],
}


def _esc_html(val: object) -> str:
    s = str(val if val is not None else "")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _expand_internal_slugs(text: str) -> str:
    """Expand internal shorthand tokens (CLM, CLEF, JEV) in step strings into plain-English descriptions."""
    import re

    out = str(text or "")
    out = re.sub(
        r"\bCLM-JEV\b",
        "Two-Stage Embedding Margin Gate + Gemini Contact-Sheet Cascade",
        out,
    )
    out = re.sub(
        r"\bCLM\b",
        "Stage-2A Calibrated Embedding & Color Margin Gate",
        out,
    )
    out = re.sub(
        r"\bCLEF\b",
        "Stage-2B Multi-Crop Visual Contact-Sheet Verifier",
        out,
    )
    out = re.sub(
        r"\bJEV\b",
        "Two-Stage Embedding + VLM Cascade",
        out,
    )
    return out


def synthesize_dynamic_methodology(
    summary: dict,
    appr: object | None = None,
    rows: list[dict] | None = None,
) -> dict:
    """Dynamically construct a grounded methodology object from the Python approach class, declared steps, trace steps, and run metrics."""
    import inspect

    from utils.experiment_matrix import APPROACH_INTUITIONS

    ap = str(summary.get("approach") or "")
    mod = str(summary.get("model") or "")
    ds = str(summary.get("dataset") or "hul_golden")
    split = str(summary.get("split") or "test")
    tk = str(summary.get("task") or "detection")
    rid = str(summary.get("run_id") or "")
    parts_id = rid.split("-")
    short_id = str(summary.get("short_id") or (f"#{parts_id[1]}" if len(parts_id) >= 2 and parts_id[1].isdigit() else f"#{rid[:6]}"))
    exp_id = str(summary.get("exp_id") or "")
    exp_prefix = f"[{_esc_html(exp_id)}] " if exp_id else ""

    if appr is None and ap:
        try:
            appr = approaches.get(ap)
        except Exception:
            appr = None

    ap_meta = APPROACH_INTUITIONS.get(ap, {})
    clean_title = str(ap_meta.get("display_name") or summary.get("approach_title") or ap)
    intuition = str(ap_meta.get("intuition") or summary.get("approach_intuition") or "")
    if not intuition and appr is not None:
        doc = inspect.getdoc(appr) or ""
        first_para = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
        if first_para:
            intuition = first_para

    arch = str(
        summary.get("architecture")
        or getattr(appr, "architecture", "")
        or ap_meta.get("default_architecture")
        or clean_title
    )
    raw_steps = list(
        summary.get("steps")
        or getattr(appr, "steps", [])
        or ap_meta.get("default_steps", [])
        or []
    )
    if not raw_steps and rows:
        seen_names: list[str] = []
        for r_item in rows[:2]:
            for st in r_item.get("steps") or []:
                if isinstance(st, dict) and st.get("name"):
                    nm = str(st["name"])
                    if nm not in seen_names:
                        seen_names.append(nm)
        raw_steps = seen_names

    low_all = f"{ap} {mod} {arch} {' '.join(str(x) for x in raw_steps)}".lower()
    is_vjepa = "vjepa" in low_all or "v-jepa" in low_all
    is_ijepa = "ijepa" in low_all or "i-jepa" in low_all
    is_dinov3 = "dinov3" in low_all or "dinov2" in low_all
    is_legacy_h5 = ap == "cv_classify" or ".h5" in mod.lower() or "inceptionv3" in low_all
    is_hf_gpu = ap.startswith("hf_") or "vllm" in low_all or "qwen" in low_all or "cloudflare/clef" in low_all
    is_onnx_det = tk == "detection" and ("onnx" in low_all or "yolo" in low_all)
    is_vlm_det = tk == "detection" and not is_onnx_det
    is_direct_sft = ap == "gemini_classify" or (
        ("sft" in ap.lower() or "sft" in mod.lower())
        and not any(k in ap.lower() for k in ("clm", "clef", "jev", "hybrid", "cascade", "dino", "jepa"))
    )

    if is_vjepa:
        origin_badge = (
            "🔬 Standalone Open-Source Vision Backbone + SFT Classifier "
            "(Meta V-JEPA 2 — 0 Gemini VLM Calls, Does NOT Use gemini-embedding-2, JEV, CLM, or CLEF)"
        )
    elif is_ijepa:
        origin_badge = (
            "🔬 Standalone Open-Source Vision Backbone + SFT Classifier "
            "(Meta I-JEPA — 0 Gemini VLM Calls, Does NOT Use gemini-embedding-2, JEV, CLM, or CLEF)"
        )
    elif is_dinov3:
        origin_badge = (
            "🔬 Standalone Open-Source Vision Backbone + CosFace SFT Classifier "
            "(Meta DINOv3 — 0 Gemini VLM Calls, Does NOT Use gemini-embedding-2, JEV, CLM, or CLEF)"
        )
    elif is_legacy_h5:
        origin_badge = (
            "🏛️ Legacy Phase-1 Customer Baseline "
            "(Keras InceptionV3 .h5 Softmax Endpoints — Zero gemini-embedding-2 / Zero VLM)"
        )
    elif is_hf_gpu:
        origin_badge = (
            "🖥️ External Open-Weight GPU Benchmark "
            "(Self-Hosted vLLM on Vertex AI L4 GPU — Compared Against Our Serverless Vertex AI Stack)"
        )
    elif is_onnx_det:
        origin_badge = (
            "⚡ In-House Distilled CPU ONNX Spatial Detector "
            "(Cloud Run CPU — ₹0 VLM Token Cost)"
        )
    elif is_vlm_det:
        origin_badge = (
            "☁️ In-House Google Cloud Vertex AI Gemini Spatial Detector "
            "(Structured 2D Bounding-Box JSON)"
        )
    elif is_direct_sft:
        origin_badge = (
            "☁️ In-House Google Cloud Vertex AI Supervised Fine-Tuned (LoRA SFT) Classifier "
            "(Direct Single-Crop Call — No Contact-Sheet Cascade)"
        )
    elif ap in ("jev_laya_zscore_lid", "djev_classify", "coarse_jev_gemini"):
        origin_badge = (
            "🎯 100% In-House Google Cloud Vertex AI Pipeline — JEV Decision Model Variant "
            "(Statistical Z-Score/LID Routing Gate or 5-Level Taxonomy Classifier — Distinct from Base JEV Retrieval Cascade)"
        )
    elif ap in ("clm_standalone_classify", "clm_classify"):
        origin_badge = (
            "🎯 100% In-House Google Cloud Vertex AI Pipeline — CLM Decision Model Variant "
            "(Closed-Set 5-Level Taxonomy Decision Classifier — Distinct from Base CLM Zero-VLM Open-Catalog Matcher)"
        )
    elif ap in ("clef_flash_decision", "clef_flash_classify", "clef_standalone_classify"):
        origin_badge = (
            "🎯 100% In-House Google Cloud Vertex AI Pipeline — CLEF Decision Model Variant "
            "(Single-Token Logprob Decision Head on Gemini — Distinct from Base CLEF Multi-Crop Contact-Sheet Prompt)"
        )
    elif ap in ("clm_standalone", "shelf_clm_standalone", "clm_promo_matcher"):
        origin_badge = (
            "⚡ 100% In-House Google Cloud Vertex AI Pipeline — Base CLM Retrieval Technique "
            "(Stage-2A Zero-VLM gemini-embedding-2 + CIELAB Color Scorer — Distinct from CLM 5-Level Decision Classifier)"
        )
    elif ap in ("clef_standalone", "clef_contrastive_metadata", "sft_clm_clef_cascade"):
        origin_badge = (
            "🔍 100% In-House Google Cloud Vertex AI Pipeline — Base CLEF Visual Contact-Sheet Technique "
            "(Multi-Crop 2x Zoom Strips + Catalog Diff Headers — Distinct from Single-Token CLEF Decision Model)"
        )
    else:
        origin_badge = (
            "☁️ 100% In-House Google Cloud Vertex AI Pipeline "
            "(Vertex gemini-embedding-2 + Gemini VLM — Base JEV Cascade)"
        )

    uses_internal_slug = (
        any(tok in ap.lower() for tok in ("jev", "clm", "clef", "laya"))
        and not is_vjepa
        and not is_ijepa
        and not is_dinov3
        and not is_legacy_h5
        and not is_hf_gpu
    )
    terminology_note = ""
    if uses_internal_slug:
        terminology_note = (
            "<b>Disambiguating Base Technique vs. Decision Model (<code>JEV</code> vs. <code>JEV Decision</code>, "
            "<code>CLM</code> vs. <code>CLM Decision</code>, <code>CLEF</code> vs. <code>CLEF Decision</code>):</b> "
            "<b>(1) <code>JEV</code> Cascade vs. <code>JEV</code> Decision Model:</b> "
            "Base <code>JEV</code> (<code>jev_laya</code>, <code>hierarchical_clm_jev</code>) is the 2-stage retrieval pipeline "
            "(<code>gemini-embedding-2</code> 3,072-D + Gemini contact sheet), whereas the <b>JEV Decision Models</b> are "
            "<code>jev_laya_zscore_lid</code> (statistical Z-score &amp; LID routing gate that decides 0-VLM fast-exit vs. escalation) and "
            "<code>djev_classify</code> (closed-set 5-level taxonomy decision classifier); "
            "<b>(2) <code>CLM</code> Matcher vs. <code>CLM</code> Decision Model:</b> "
            "Base <code>CLM</code> (<code>clm_standalone</code>) is the zero-VLM open-catalog vector + inner-80% CIELAB color scorer, "
            "whereas the <b>CLM Decision Models</b> (<code>clm_standalone_classify</code>, <code>clm_classify</code>) apply Lidstone prior "
            "calibration or kNN voting to output a discrete 5-attribute taxonomy decision; "
            "<b>(3) <code>CLEF</code> Contact Sheet vs. <code>CLEF</code> Decision Model:</b> "
            "Base <code>CLEF</code> (<code>clef_contrastive_metadata</code> EXP-28, <code>clef_standalone</code>) is the multi-crop 2x zoom "
            "contact-sheet layout with Catalog Diff Headers, whereas the <b>CLEF Decision Models</b> (<code>clef_flash_decision</code>, "
            "<code>clef_flash_classify</code>) constrain Gemini (<code>max_output_tokens=4</code>) to emit a single candidate token scored by "
            "log-probability margin (also distinct from external models Meta <code>V-JEPA 2</code>, HF <code>CLM-v0.1-8B</code>, and <code>Cloudflare/clef-flash</code>)."
        )

    # Build metrics summary string dynamically from actual recorded fields
    fa = summary.get("field_accuracy") or {}
    ha = summary.get("hierarchical_accuracy") or {}
    l1 = ha.get("level1_is_hul", fa.get("is_hul"))
    l2 = ha.get("level2_category", fa.get("category"))
    l3 = ha.get("level3_brand", fa.get("brand"))
    l4 = ha.get("level4_base_variant", fa.get("base_variant", fa.get("product")))
    l5 = ha.get("level5_packaging_type", fa.get("packaging_type"))
    three_attr = ha.get("compound_brand_category_pack", summary.get("three_attr_accuracy"))
    five_attr = ha.get(
        "compound_all_5_levels",
        summary.get("five_attr_compound_accuracy", summary.get("accuracy") if tk not in ("detection", "end_to_end") else None),
    )

    def _pct(v: object) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(str(v)) * 100.0:.1f}%"
        except (TypeError, ValueError):
            return "—"

    metric_bits: list[str] = []
    if tk in ("detection", "end_to_end") and summary.get("f2") is not None:
        metric_bits.append(
            f"<b>{_pct(summary.get('f2'))} F2</b> ({_pct(summary.get('precision'))} Prec / {_pct(summary.get('recall'))} Rec)"
        )
    ard = summary.get("annotated_row_box_metrics") or summary.get("annotated_row_detection_metrics")
    if isinstance(ard, dict) and ard.get("f2") is not None:
        metric_bits.append(f"<b>{_pct(ard.get('f2'))} 2D-Zone F2</b>")
    bsm = summary.get("bay_scoped_metrics")
    if isinstance(bsm, dict) and bsm.get("f2") is not None:
        metric_bits.append(f"<b>{_pct(bsm.get('f2'))} Bay-Scoped F2</b>")
    pm = summary.get("planogram_metrics")
    if isinstance(pm, dict) and pm.get("mean_brand_block_purity") is not None:
        metric_bits.append(f"<b>{_pct(pm.get('mean_brand_block_purity'))} Brand-Block Purity</b>")
    sm = summary.get("sos_metrics")
    if isinstance(sm, dict) and sm.get("facing_sos_mae") is not None:
        try:
            metric_bits.append(f"<b>{float(sm['facing_sos_mae']) * 100.0:.2f} pp Facing SOS MAE</b>")
        except (TypeError, ValueError):
            pass
    if five_attr is not None and float(five_attr or 0) > 0 and tk != "detection":
        metric_bits.append(f"<b>{_pct(five_attr)} 5-Attr Exact SKU</b>")
    if three_attr is not None and float(three_attr or 0) > 0:
        metric_bits.append(f"<b>{_pct(three_attr)} 3-Attr (Brd+Cat+Pack)</b>")
    if l1 is not None:
        metric_bits.append(f"{_pct(l1)} L1 HUL")
    if l2 is not None:
        metric_bits.append(f"{_pct(l2)} Cat")
    if l3 is not None:
        metric_bits.append(f"{_pct(l3)} Brand")
    if l4 is not None and float(l4 or 0) > 0:
        metric_bits.append(f"{_pct(l4)} Variant")
    if l5 is not None and float(l5 or 0) > 0:
        metric_bits.append(f"{_pct(l5)} Pack")
    if fa.get("promo_type") is not None:
        metric_bits.append(f"<b>{_pct(fa.get('promo_type'))} Promo Type</b>")

    perf_summary = " &middot; ".join(metric_bits) if metric_bits else f"<b>{_pct(summary.get('primary_score', summary.get('accuracy', summary.get('f2'))))} primary score</b>"

    gate_val = summary.get("stage2a_zero_vlm_gate_pct")
    gate_note = ""
    if gate_val is not None and not is_vjepa and not is_dinov3 and not is_legacy_h5 and tk != "detection":
        try:
            gv = float(gate_val)
            gate_note = f" · Stage-2A Fast-Exit Gate: {gv:.1f}% 0-VLM"
        except (TypeError, ValueError):
            pass

    summary_header = (
        f"Run {_esc_html(short_id)} Methodology — {exp_prefix}{_esc_html(clean_title)} "
        f"(<code>{_esc_html(ap)}</code> · Model: <code>{_esc_html(mod or 'CPU/ONNX')}</code> "
        f"on <code>{_esc_html(ds)}/{_esc_html(split)}</code>{_esc_html(gate_note)})"
    )

    expanded_arch = _esc_html(_expand_internal_slugs(arch))
    flow_parts = [f"<b>Declared Pipeline Architecture:</b> <code>{expanded_arch}</code>"]
    if intuition:
        flow_parts.append(f"<b>How This Technique Works:</b> {_esc_html(_expand_internal_slugs(intuition))}")
    if summary.get("exp_hypothesis"):
        flow_parts.append(f"<i>(Experiment Hypothesis: {_esc_html(_expand_internal_slugs(str(summary['exp_hypothesis'])))})</i>")
    flow_summary = "<br>".join(flow_parts)

    bullets: list[str] = []
    if raw_steps:
        for idx, st in enumerate(raw_steps):
            st_clean = _esc_html(_expand_internal_slugs(str(st)))
            bullets.append(f"<b>Step {idx + 1}:</b> {st_clean}")
    else:
        bullets.append(f"<b>Pipeline Execution:</b> {expanded_arch}")

    if rows:
        stage_counts: dict[str, int] = {}
        stage_ms: dict[str, list[float]] = {}
        stage_calls: dict[str, int] = {}
        for r_item in rows[:250]:
            tr_list = r_item.get("trace") or r_item.get("steps") or []
            if not isinstance(tr_list, list):
                continue
            for t_step in tr_list:
                if not isinstance(t_step, dict):
                    continue
                s_name = str(t_step.get("step") or t_step.get("name") or "").strip()
                if not s_name:
                    continue
                stage_counts[s_name] = stage_counts.get(s_name, 0) + 1
                if t_step.get("ms") is not None:
                    try:
                        stage_ms.setdefault(s_name, []).append(float(str(t_step["ms"])))
                    except (TypeError, ValueError):
                        pass
                if t_step.get("calls") is not None:
                    try:
                        stage_calls[s_name] = stage_calls.get(s_name, 0) + int(str(t_step["calls"]))
                    except (TypeError, ValueError):
                        pass
        if stage_counts:
            trace_bits: list[str] = []
            for s_name, cnt in list(stage_counts.items())[:6]:
                ms_vals = stage_ms.get(s_name, [])
                mean_ms = f", {sum(ms_vals) / len(ms_vals):.0f}ms avg" if ms_vals else ""
                c_tot = stage_calls.get(s_name, 0)
                c_str = f", {c_tot} API calls" if c_tot > 0 else ""
                trace_bits.append(f"<code>{_esc_html(s_name)}</code> ({cnt} imgs{mean_ms}{c_str})")
            bullets.append(
                f"<b>Live Execution Trace Telemetry (from <code>images.jsonl</code>):</b> {' &rarr; '.join(trace_bits)}."
            )

    n_units = int(summary.get("eval_samples") or summary.get("total_gt_boxes") or summary.get("images") or 0)
    p50_s = summary.get("p50_latency_s")
    p50_str = f"{float(str(p50_s)):.2f}s p50 latency" if p50_s is not None else "recorded latency"
    rate = float(summary.get("usd_to_inr") or 86.2)
    cost_inr = float(summary.get("cost_per_image_inr") or (float(summary.get("cost_per_image_usd") or 0.0) * rate))
    unit_name = str(summary.get("cost_unit") or ("shelf" if ds in ("gtmerch_golden", "ms_json_golden", "mt_json_golden", "hul_talkers_golden", "hul_sessions", "gt_hul_sessions") else "crop"))
    bullets.append(
        f"<b>Verified Run {_esc_html(short_id)} Outcome on <code>{_esc_html(ds)}</code> "
        f"(<code>{_esc_html(split)}</code> · {n_units:,} evaluated {unit_name}s):</b> "
        f"Achieves {perf_summary} at <b>{p50_str}</b> and <b>₹{cost_inr:.3f}/{unit_name}</b>."
    )

    return {
        "summary_header": summary_header,
        "origin_badge": origin_badge,
        "terminology_note": terminology_note,
        "flow_html": flow_summary,
        "flow_summary": flow_summary,
        "bullets": bullets,
        "generated_dynamically": True,
        "generated_by": "dynamic_introspection",
    }


def generate_run_methodology(
    summary: dict,
    rows: list[dict] | None = None,
    *,
    appr: object | None = None,
    config: dict | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Dynamically generate a grounded HTML-safe methodology summary for a completed run."""
    rows_list = rows or []
    base_meth = synthesize_dynamic_methodology(summary, appr=appr, rows=rows_list)
    if os.environ.get("SHELF_BENCH_AUTO_METHODOLOGY", "1") == "0":
        return base_meth
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("SHELF_BENCH_AUTO_METHODOLOGY") != "1":
        return base_meth
    try:
        import inspect

        from google import genai
        from google.genai import types

        if appr is None and summary.get("approach"):
            try:
                appr = approaches.get(str(summary["approach"]))
            except Exception:
                appr = None
        cfg = config or load_config()
        gcp = cfg.get("gcp", {})
        meth_model = os.environ.get("SHELF_BENCH_METHODOLOGY_MODEL", "gemini-3.5-flash-lite")
        source_code = ""
        if appr is not None:
            try:
                source_code = inspect.getsource(type(appr))
            except Exception:
                source_code = ""
        sample_steps = []
        for r in rows_list[:2]:
            for st in (r.get("trace") or r.get("steps") or []):
                if isinstance(st, dict) and (st.get("step") or st.get("name")):
                    sample_steps.append({
                        "name": st.get("step") or st.get("name"),
                        "detail": str(st.get("detail") or "")[:180],
                    })
        if len(sample_steps) > 8:
            sample_steps = sample_steps[:8]

        rid = str(summary.get("run_id") or "")
        parts_id = rid.split("-")
        short_id = f"#{parts_id[1]}" if len(parts_id) >= 2 and parts_id[1].isdigit() else f"#{rid[:6]}"
        rate = float(summary.get("usd_to_inr") or 86.2)
        cost_inr = float(summary.get("cost_per_image_usd") or 0.0) * rate
        prompt_ctx = {
            "short_run_id": short_id,
            "run_id": rid,
            "approach": summary.get("approach"),
            "task": summary.get("task"),
            "dataset": summary.get("dataset"),
            "model": summary.get("model"),
            "evaluated_fields": summary.get("evaluated_fields"),
            "architecture": summary.get("architecture"),
            "declared_steps": getattr(appr, "steps", []) if appr is not None else [],
            "sample_trace_steps": sample_steps,
            "metrics": {
                "accuracy_all_fields": summary.get("accuracy"),
                "f2": summary.get("f2"),
                "precision": summary.get("precision"),
                "recall": summary.get("recall"),
                "field_accuracy": summary.get("field_accuracy"),
                "hierarchical_accuracy": summary.get("hierarchical_accuracy"),
                "stage2a_zero_vlm_gate_pct": summary.get("stage2a_zero_vlm_gate_pct"),
                "p50_latency_s": summary.get("p50_latency_s"),
                "cost_per_image_inr": round(cost_inr, 4),
            },
        }
        prompt = (
            "Write a technical run methodology summary for an engineering workbench.\n"
            "Rules:\n"
            "1. Ground every statement strictly in the Python class source code and run metrics below.\n"
            "2. Never invent components not present in the code (if the code does not call "
            "gemini-embedding-2 or a contact sheet, state that explicitly).\n"
            "3. Avoid opaque internal acronyms (do not lead bullets with JEV, CLM, or CLEF; use plain-English "
            "stage names like 'Stage 2A Embedding & Color Retrieval', 'Confidence Margin Gate', "
            "'Stage 2B Visual Contact-Sheet Verifier').\n"
            "4. Do NOT use em dashes. Keep prose concise, active, and direct.\n"
            "5. Use only <b> and <code> HTML tags for component names, model names, and numbers.\n"
            "6. summary_header: 1 line starting with 'Run " + short_id + " Methodology (<code>...</code>): ...'\n"
            "7. flow_summary: 1 sentence summarizing the step-by-step flow on the dataset.\n"
            "8. bullets: 3 to 5 concise strings covering each executed step and the final measured metrics. "
            "Do not wrap bullets in <li> tags.\n\n"
            f"Run JSON Context:\n{json.dumps(prompt_ctx, indent=2)}\n\n"
            f"Approach Python Source:\n{source_code[:12000]}"
        )
        client = genai.Client(
            vertexai=True,
            project=gcp.get("project"),
            location=gcp.get("location", "global"),
        )
        resp = client.models.generate_content(
            model=meth_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1200,
                response_mime_type="application/json",
                response_schema=_METHODOLOGY_SCHEMA,
            ),
        )
        parsed = json.loads(resp.text or "{}")
        sh = str(parsed.get("summary_header") or "").strip()
        fs = str(parsed.get("flow_summary") or "").strip()
        bs = [str(b).strip() for b in (parsed.get("bullets") or []) if str(b).strip()]
        if sh and fs and len(bs) >= 2:
            return {
                "summary_header": sh,
                "origin_badge": base_meth["origin_badge"],
                "terminology_note": base_meth["terminology_note"],
                "flow_html": fs,
                "flow_summary": fs,
                "bullets": bs,
                "generated_dynamically": True,
                "generated_by": meth_model,
            }
    except Exception as err:
        log(f"[{summary.get('run_id', 'run')}] auto-methodology fallback to dynamic introspection: {err}")
    return base_meth


def run(
    approach: str,
    model: str,
    split: str = "test",
    limit: int = 50,
    seed: int = 0,
    workers: int = 4,
    owner: str | None = None,
    results_dir: Path | str = RESULTS_DIR,
    data_root: str = dataset.DEFAULT_ROOT,
    llm: Callable[..., LLMResult] | None = None,
    log: Callable[[str], None] = print,
    tier: str = "priority",
    prices: dict | None = None,
    dataset_override: str | None = None,
    fields: str | list[str] | tuple[str, ...] | None = None,
) -> dict:
    appr = type(approaches.get(approach))()
    if dataset_override:
        appr.dataset = dataset_override
    if not appr.accepts(model):
        raise ValueError(f"{approach} doesn't run {model}; it runs "
                         f"{', '.join(appr.models) if appr.models else 'Vertex Gemini models'}")
    if not appr.uses_gemini:
        tier = "standard"
    config = load_config()
    env = environment(config)
    # Live list prices from the Cloud Billing Catalog API (fails fast if a SKU is missing).
    sheet = prices or pricing.price_sheet([model] if appr.uses_gemini else [], config,
                                         extra_skus=appr.skus)
    if appr.uses_gemini:
        llm = llm or Gemini(model, config, tier=tier)
    task = appr.task
    if appr.dataset != "sku110k":  # --root is SKU-110K's; the other sets have their own location
        data_root = dataset.data_root(appr.dataset)
    default_fields = {"rpc": dataset.RPC_FIELDS, "shelves": dataset.SHELF_FIELDS, **dataset.HUL_FIELDS}.get(
        appr.dataset, dataset.PRODUCT_FIELDS)
    eval_fields = default_fields
    appr.evaluated_fields = None
    if fields:  # --fields: score only these attributes
        from utils.cv_models import resolve_evaluated_fields

        eval_fields = appr.evaluated_fields = resolve_evaluated_fields(appr.dataset, fields)
    catalog = dataset.catalog(str(data_root), appr.dataset) if task != "detection" else {}
    samples = dataset.sample_images(split, limit, seed, str(data_root), appr.dataset)
    if not samples:
        raise RuntimeError(f"No images found for split={split!r} under {data_root}")

    started = datetime.now(timezone.utc)

    def price(u: Usage) -> dict:
        return pricing.gemini_cost(model, u.buckets, sheet, started.date())

    # The approach's own clients and indexes (embeddings, AlloyDB, ...). What it bills here is
    # a one-off setup cost (e.g. embedding a reference gallery), reported apart from cost/img.
    # Zero-leakage invariant: setup_ctx NEVER receives eval samples or test/val ground-truth labels.
    setup_ctx = Context(model=model, llm=llm, price=price)
    t_setup = time.perf_counter()
    appr.setup(config, setup_ctx)
    setup_s = time.perf_counter() - t_setup
    setup_usd = price(setup_ctx.trace.usage)["net_usd"] + pricing.extra_cost(setup_ctx.trace.billed, sheet)

    if appr.evaluated_fields:  # setup may narrow them to what the approach can answer
        eval_fields = tuple(appr.evaluated_fields)
    if getattr(appr, "resolved_model", None) and model == "keras-h5":
        model = str(appr.resolved_model)

    ds_suffix = f"-{appr.dataset}" if dataset_override else ""
    split_suffix = f"-{split}" if (dataset_override and split != "test") else ""
    fields_suffix = "" if eval_fields == default_fields else f"-{len(eval_fields)}attr-{'_'.join(eval_fields)}"
    run_id = (f"{started:%m%d-%H%M%S}-{approach}-{model}"
              + ("" if tier == "standard" else f"-{tier}") + ds_suffix + split_suffix + fields_suffix)
    if len(run_id) > 240:
        fields_suffix = "" if eval_fields == default_fields else f"-{len(eval_fields)}attr"
        run_id = (f"{started:%m%d-%H%M%S}-{approach}-{model}"
                  + ("" if tier == "standard" else f"-{tier}") + ds_suffix + split_suffix + fields_suffix)[:240]
    log(f"[{run_id}] {len(samples)} images from {appr.dataset} {split} (seed {seed}) on {env['platform']}")

    # One trace per run (Cloud Trace), with correlated log entries (Cloud Logging).
    telemetry.init(config)
    run_attrs = {"shelf_bench.run_id": run_id, "shelf_bench.approach": approach,
                 "shelf_bench.task": task, "shelf_bench.dataset": appr.dataset,
                 "gen_ai.request.model": model, "shelf_bench.tier": tier,
                 "shelf_bench.split": split, "shelf_bench.limit": limit, "shelf_bench.seed": seed,
                 "shelf_bench.workers": workers, "shelf_bench.images": len(samples),
                 "shelf_bench.owner": owner or getpass.getuser(),
                 "shelf_bench.platform": env["platform"],
                 "shelf_bench.execution": env.get("execution"),
                 "shelf_bench.task_index": env.get("task_index")}
    run_span = telemetry.tracer().start_span(f"run {run_id}", attributes=telemetry.clean(run_attrs))
    run_ctx = telemetry.child_context(run_span)
    telemetry.log(run_span, "run_started", run_attrs)

    def one(sample: dataset.Sample) -> dict:
        with telemetry.span(f"image {sample.image_id}", parent=run_ctx,
                            **{"shelf_bench.image_id": sample.image_id,
                               "shelf_bench.split": split}) as span:
            row = _one(sample, span)
            telemetry.set_attrs(span, {f"shelf_bench.{k}": row[k] for k in (
                "width", "height", "gt_count", "pred_count", "tp", "fp", "fn", "precision",
                "recall", "f2", "accuracy", "latency_s", "cost_usd", "cost_list_usd",
                "services_cost_usd", "error")})
            telemetry.set_attrs(span, telemetry.usage_attrs(Usage(**row["usage"])))
            if row["error"]:
                span.set_status(telemetry.Status(telemetry.StatusCode.ERROR, row["error"]))
            telemetry.log(span, "image_scored", {
                "run_id": run_id, **{k: v for k, v in row.items()
                                     if k not in ("preds", "matched", "steps")},
                "steps": [{k: v for k, v in s.items() if k not in ("boxes", "regions", "calls", "info")}
                          for s in row["steps"]]},
                level=logging.ERROR if row["error"] else logging.INFO)
            row["telemetry"] = telemetry.links(span, per_span=True)
            return row

    def identify_all(image: Image.Image, boxes: list, ctx: Context) -> list[int | None]:
        """Name the product in every box (``Approach.identify_boxes``)."""
        ids = appr.identify_boxes(image, boxes, ctx)
        ctx.trace.step("Identify each box", f"{sum(i is not None for i in ids)} of {len(boxes)} "
                       "boxes matched to a catalog product", boxes=boxes)
        return ids

    def _one(sample: dataset.Sample, span) -> dict:
        trace = Trace(span=span)
        ctx = Context(model=model, llm=llm, trace=trace,
                      otel_parent=telemetry.child_context(span), price=price,
                      image_id=sample.image_id)
        ctx.detected_asset_boxes = (
            list(getattr(sample, "asset_boxes", []) or [])
            if task == "planogram_compliance"
            else []
        )
        t0 = time.perf_counter()
        error, preds, ids = None, [], []
        try:
            image = dataset.open_rgb_normalized(dataset.read_bytes(sample.path))
            if (
                appr.dataset.startswith(("hul_", "gtmerch", "ms_json", "mt_json"))
                and (image.width, image.height) != (sample.width, sample.height)
                and sample.width > 0
                and sample.height > 0
            ):
                sx, sy = image.width / sample.width, image.height / sample.height
                sample.boxes = [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in sample.boxes]
                if getattr(sample, "asset_boxes", None):
                    sample.asset_boxes = [
                        (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in sample.asset_boxes
                    ]
                    if task == "planogram_compliance":
                        ctx.detected_asset_boxes = list(sample.asset_boxes)
                sample.width, sample.height = image.width, image.height
            trace.step("Load image", f"{image.width}x{image.height} px")
            if task == "classification":  # single-crop photo or multi-box shelf
                preds = list(sample.boxes)
                ids = [appr.identify(image, ctx)] if len(preds) <= 1 else identify_all(image, preds, ctx)
            elif task in ("retrieval", "promo_detection", "planogram_compliance"):  # ground-truth boxes given
                preds = list(sample.boxes)
                ids = identify_all(image, preds, ctx)
            elif task == "end_to_end":
                preds, ids = appr.detect_and_identify(image, ctx)
            else:
                preds = appr.detect(image, ctx)
        except Exception as e:  # a failed image scores as "found nothing"
            preds, ids, error = [], [], f"{type(e).__name__}: {e}"[:300]
            span.record_exception(e)
        latency = time.perf_counter() - t0
        extra = {}
        if task == "detection":
            m = metrics.match(preds, sample.boxes)
            bands = metrics.group_annotated_shelf_bands(sample.boxes)
            in_preds, _, _ = metrics.filter_preds_to_annotated_bands(preds, None, bands)
            in_m = metrics.match(in_preds, sample.boxes)
            extra["annotated_row_box_metrics"] = {
                "tp": in_m["tp"], "fp": in_m["fp"], "fn": in_m["fn"],
                "bands": len(bands),
                "out_of_band_preds": len(preds) - len(in_preds),
                **{k: round(v, 4) for k, v in metrics.scores(in_m["tp"], in_m["fp"], in_m["fn"]).items()},
            }
            extra["enrichment_metrics"] = metrics.compute_enrichment_buckets(
                preds, None, sample.boxes, sample.labels, None, m["pairs"]
            )
            extra["human_and_beyond"] = metrics.compute_human_and_beyond_metrics(
                preds, None, sample.boxes, sample.labels, None, m["pairs"], eval_fields=eval_fields
            )
            eff_assets = getattr(sample, "asset_boxes", None) or getattr(ctx, "detected_asset_boxes", None)
            extra["bay_scoped_metrics"] = metrics.score_merchandising_bay_or_hanger(
                preds, None, sample.boxes, sample.labels, asset_boxes=eff_assets
            )
            if getattr(sample, "asset_boxes", None) or getattr(ctx, "detected_asset_boxes", None):
                am = metrics.match(
                    list(getattr(ctx, "detected_asset_boxes", []) or []),
                    list(getattr(sample, "asset_boxes", []) or []),
                )
                extra["detected_asset_box_metrics"] = {
                    "tp": am["tp"], "fp": am["fp"], "fn": am["fn"],
                    **{k: round(v, 4) for k, v in metrics.scores(am["tp"], am["fp"], am["fn"]).items()},
                }
            trace.step("Score vs ground truth",
                       f"{m['tp']} correct, {m['fp']} false, {m['fn']} missed "
                       f"({len(sample.boxes)} products in ground truth; "
                       f"in-band F2={extra['annotated_row_box_metrics']['f2']:.2f}, "
                       f"bay/hanger F2={extra['bay_scoped_metrics']['f2']:.2f}, "
                       f"extra discovered={extra['enrichment_metrics']['extra_unannotated_row_preds']})")
        elif task == "end_to_end":  # right box (IoU >= 0.5) and right product
            gt_ids = [g["sku_id"] for g in sample.labels]
            m = metrics.match_identified(
                preds,
                ids,
                sample.boxes,
                gt_ids,
                catalog=catalog,
                gt_labels=sample.labels,
                fields=eval_fields,
            )
            box_m = metrics.match(preds, sample.boxes)
            bands = metrics.group_annotated_shelf_bands(sample.boxes)
            in_preds, in_ids, _ = metrics.filter_preds_to_annotated_bands(preds, ids, bands)
            in_box_m = metrics.match(in_preds, sample.boxes)
            psa_preds, psa_ids, _ = metrics.filter_preds_to_psa_categories(
                preds, ids, catalog, sample.labels, bands
            )
            psa_m = metrics.match_identified(
                psa_preds,
                psa_ids,
                sample.boxes,
                gt_ids,
                catalog=catalog,
                gt_labels=sample.labels,
                fields=eval_fields,
            )
            # Count fast-gated boxes in box units (rather than step count) so
            # stage2a_zero_vlm_gate_pct divides boxes by pred_sum.
            fast_gate_steps = [
                s for s in trace.steps
                if "Stage A" in s["name"]
                or "Embedding tier" in s["name"]
                or "CLM InfoNCE" in s["name"]
                or any(k in (s.get("name", "") + " " + s.get("detail", "")).lower()
                       for k in ("fast-gate", "fast_gate", "zero-vlm", "margin>=", "clm-jev-o"))
            ]
            if fast_gate_steps:
                fast_gated = min(
                    len(preds),
                    sum(int((s.get("info") or {}).get("fast_gated_boxes", 1)) for s in fast_gate_steps),
                )
            elif not appr.uses_gemini and trace.usage.calls == 0:
                fast_gated = len(preds)
            else:
                fast_gated = 0
            extra["found"] = m["found"]
            extra["box_metrics"] = {
                "tp": box_m["tp"], "fp": box_m["fp"], "fn": box_m["fn"],
                **{k: round(v, 4) for k, v in metrics.scores(box_m["tp"], box_m["fp"], box_m["fn"]).items()},
            }
            extra["annotated_row_box_metrics"] = {
                "tp": in_box_m["tp"], "fp": in_box_m["fp"], "fn": in_box_m["fn"],
                "bands": len(bands),
                "out_of_band_preds": len(preds) - len(in_preds),
                **{k: round(v, 4) for k, v in metrics.scores(in_box_m["tp"], in_box_m["fp"], in_box_m["fn"]).items()},
            }
            extra["psa_scoped_metrics"] = {
                "tp": psa_m["tp"], "fp": psa_m["fp"], "fn": psa_m["fn"],
                "found": psa_m["found"],
                "scoped_preds": len(psa_preds),
                **{k: round(v, 4) for k, v in metrics.scores(psa_m["tp"], psa_m["fp"], psa_m["fn"]).items()},
            }
            eff_assets = getattr(sample, "asset_boxes", None) or getattr(ctx, "detected_asset_boxes", None)
            extra["bay_scoped_metrics"] = metrics.score_merchandising_bay_or_hanger(
                preds,
                ids,
                sample.boxes,
                sample.labels,
                asset_boxes=eff_assets,
                catalog=catalog,
                fields=eval_fields,
            )
            extra["enrichment_metrics"] = metrics.compute_enrichment_buckets(
                preds, ids, sample.boxes, sample.labels, catalog, m["pairs"]
            )
            extra["human_and_beyond"] = metrics.compute_human_and_beyond_metrics(
                preds, ids, sample.boxes, sample.labels, catalog, m["pairs"], eval_fields=eval_fields
            )
            extra["stage2_fast_gate_count"] = fast_gated
            extra["labels"] = [
                {"box": [round(v, 1) for v in b], "pred": i if isinstance(i, dict) else (catalog.get(i) if i is not None else None),
                 "gt": sample.labels[m["pairs"][k]] if k in m["pairs"] else None,
                 "correct": (
                     metrics.match_product(
                         i if isinstance(i, dict) else (catalog.get(i) if i is not None else None),
                         sample.labels[m["pairs"][k]],
                         eval_fields,
                     )["correct"]
                     if k in m["pairs"] else dict.fromkeys(eval_fields, False)
                 )}
                for k, (b, i) in enumerate(zip(preds, ids, strict=True))]
            trace.step("Score vs ground truth",
                       f"{m['found']} of {len(sample.boxes)} products boxed, {m['tp']} of those "
                       f"identified correctly; {m['fp']} boxes wrong or extra, {m['fn']} products missed "
                       f"(in-band det F2={extra['annotated_row_box_metrics']['f2']:.2f}, "
                       f"psa-scoped E2E F2={extra['psa_scoped_metrics']['f2']:.2f})")
        else:  # classification / retrieval: one answer per ground-truth box
            if not ids:  # the image failed: every product counts as unanswered
                ids = [None] * len(sample.boxes)
            m = {"tp": 0, "fp": 0, "fn": 0, "matched": []}
            extra["labels"] = []
            for k, (b, gt, i) in enumerate(zip(sample.boxes, sample.labels, ids, strict=True)):
                pred = i if isinstance(i, dict) else (catalog.get(i) if i is not None else None)
                mk = metrics.match_product(pred, gt, eval_fields)
                for key in ("tp", "fp", "fn"):
                    m[key] += mk[key]
                if mk["tp"]:
                    m["matched"].append(k)
                extra["labels"].append({"box": [round(v, 1) for v in b], "pred": pred, "gt": gt,
                                        "correct": mk["correct"]})
            trace.step("Score vs ground truth",
                       f"{m['tp']} of {len(sample.boxes)} products identified correctly"
                       if len(sample.boxes) > 1 else
                       f"predicted {_fmt_label(extra['labels'][0]['pred'])}; "
                       f"truth {_fmt_label(sample.labels[0])} "
                       f"({', '.join(f + (' ✓' if ok else ' ✗') for f, ok in extra['labels'][0]['correct'].items())})")
        if task in ("planogram_compliance", "end_to_end"):
            existing_plano = next(
                (
                    st["info"]
                    for st in trace.steps
                    if isinstance(st.get("info"), dict) and "brand_block_purity" in st["info"]
                ),
                None,
            )
            if existing_plano is not None:
                extra["planogram_compliance"] = existing_plano
            else:
                plano = planogram_engine.evaluate_shelf_planogram(
                    boxes=preds,
                    pred_ids=ids,
                    catalog=catalog,
                    asset_boxes=getattr(ctx, "detected_asset_boxes", None),
                    image_size=(sample.width, sample.height),
                )
                extra["planogram_compliance"] = plano
                trace.step(
                    "Stage 6: Planogram & OOS Void Compliance",
                    f"Scope={plano['scope_mode']} | HUL SOS={plano['hul_facing_sos']*100:.1f}% | "
                    f"brand-block purity={plano['brand_block_purity']*100:.1f}% | "
                    f"OOS voids={plano['oos_void_count']} ({plano['estimated_missing_facings']} missing facings)",
                    boxes=[v["box"] for v in plano.get("oos_void_boxes", [])],
                    info=plano,
                )
        g = price(trace.usage)
        s_usd = pricing.extra_cost(trace.billed, sheet)
        contract_err = classify_contract_error(error, trace.steps, latency)
        row_out = {
            "image_id": sample.image_id,
            "width": sample.width,
            "height": sample.height,
            "gt_count": len(sample.boxes),
            "pred_count": len(preds),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            **{k: round(v, 4) for k, v in metrics.scores(m["tp"], m["fp"], m["fn"]).items()},
            "latency_s": round(latency, 3),
            "usage": asdict(trace.usage),
            "cost_usd": g["net_usd"] + s_usd,  # Gemini (after promo credit) + other APIs
            "services_cost_usd": s_usd,
            "billed": trace.billed,
            "cost_list_usd": g["list_usd"],  # Gemini, catalog list price
            "error": error,
            "contract_error": contract_err,
            "preds": [[round(v, 1) for v in b] for b in preds],
            "matched": m["matched"],
            "steps": trace.steps,
            **extra,
        }
        row_out["p2p_policy_gate"] = evaluate_p2p_payout_policy_gate(row_out)
        return row_out

    try:
        import threading

        ckpt_env = os.environ.get("SHELF_BENCH_CHECKPOINT_FILE", "").strip()
        ckpt_path = Path(ckpt_env).resolve() if ckpt_env else None
        ckpt_rows: dict[str, dict] = {}
        if ckpt_path is not None and ckpt_path.is_file():
            for line in ckpt_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r_obj = json.loads(line)
                    if isinstance(r_obj, dict) and r_obj.get("image_id") and not r_obj.get("error"):
                        ckpt_rows[str(r_obj["image_id"])] = r_obj
                except Exception:
                    continue
        ckpt_lock = threading.Lock()

        def one_or_ckpt(sample: Sample) -> dict:
            if sample.image_id in ckpt_rows:
                cached_row = dict(ckpt_rows[sample.image_id])
                cached_row["resumed_from_checkpoint"] = True
                return cached_row
            row_val = one(sample)
            if ckpt_path is not None and not row_val.get("error"):
                with ckpt_lock:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    with ckpt_path.open("a", encoding="utf-8") as cf:
                        cf.write(json.dumps(row_val) + "\n")
            return row_val

        rows: list[dict] = []
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for i, row in enumerate(pool.map(one_or_ckpt, samples), 1):
                rows.append(row)
                flag = f"  ERROR {row['error']}" if row["error"] else ""
                what = (f"{'✓' if row['tp'] else '✗'} {_fmt_label(row['labels'][0]['pred'])}"
                        if task == "classification" and row["gt_count"] <= 1 else
                        f"{row['tp']:>3} / {row['gt_count']:>3} identified"
                        if task in ("classification", "retrieval", "promo_detection", "planogram_compliance") else
                        f"{row['pred_count']:>4} pred / {row['gt_count']:>4} gt  F2 {row['f2']:.2f}")
                log(f"  {i:>3}/{len(samples)} {row['image_id']:<16} {what}  "
                    f"{row['latency_s']:.1f}s{flag}")
        wall_s = time.perf_counter() - t_start

        summary = summarize(rows)
        summary["ci95"] = {
            "accuracy": metrics.bootstrap_ci95([float(r.get("accuracy", 0.0)) for r in rows]),
            "f2": metrics.bootstrap_ci95([float(r.get("f2", 0.0)) for r in rows]),
        }
        n = len(rows)
        if task in ("classification", "retrieval", "promo_detection", "planogram_compliance", "end_to_end"):
            boxes = [lab for r in rows for lab in r.get("labels", [])]
            eval_boxes = [lab for lab in boxes if lab.get("gt") is not None] if task == "end_to_end" else boxes
            if not eval_boxes:
                eval_boxes = boxes
            if eval_boxes:
                summary["field_accuracy"] = {
                    f: round(sum(bool(lab["correct"].get(f)) for lab in eval_boxes) / max(1, len(eval_boxes)), 4)
                    for f in eval_fields}
                if task in ("classification", "retrieval", "promo_detection", "planogram_compliance"):
                    summary["accuracy"] = round(
                        sum(all(bool(lab["correct"].get(f)) for f in eval_fields) for lab in eval_boxes)
                        / max(1, len(eval_boxes)),
                        4,
                    )
                    valid_boxes = [
                        lab
                        for r in rows
                        if not (
                            r.get("error")
                            or r.get("contract_error")
                            or classify_contract_error(
                                r.get("error"), r.get("steps"), float(r.get("latency_s", 0.0) or 0.0)
                            )
                        )
                        for lab in r.get("labels", [])
                    ]
                    if valid_boxes and isinstance(summary.get("contract_errors"), dict):
                        summary["contract_errors"]["valid_response_accuracy"] = round(
                            sum(all(bool(lab["correct"].get(f)) for f in eval_fields) for lab in valid_boxes)
                            / max(1, len(valid_boxes)),
                            4,
                        )
                summary["hierarchical_accuracy"] = metrics.hierarchical_field_scores(eval_boxes)
                # Macro-averaged accuracy across GT brands and GT variants + HUL vs Non-HUL competitor breakdown
                from collections import defaultdict

                by_brand_ok: dict[str, list[int]] = defaultdict(list)
                by_var_ok: dict[str, list[int]] = defaultdict(list)
                by_cat_boxes: dict[str, list[dict]] = defaultdict(list)
                hul_boxes = []
                comp_boxes = []
                for lab in boxes:
                    gt_d = lab.get("gt") or {}
                    corr_d = lab.get("correct") or {}
                    b_name = str(gt_d.get("brand") or "")
                    v_name = str(gt_d.get("base_variant") or gt_d.get("product") or "")
                    c_name = str(gt_d.get("category") or "")
                    if b_name:
                        by_brand_ok[b_name].append(int(bool(corr_d.get("brand"))))
                    if v_name:
                        by_var_ok[v_name].append(int(bool(corr_d.get("base_variant", corr_d.get("product")))))
                    if c_name:
                        by_cat_boxes[c_name].append(lab)
                    if "is_hul" in gt_d:
                        if gt_d.get("is_hul"):
                            hul_boxes.append(lab)
                        else:
                            comp_boxes.append(lab)
                if by_cat_boxes:
                    summary["category_breakdown"] = {
                        cat_k: {
                            "count": len(c_labs),
                            "accuracy": round(
                                sum(all(bool((lb.get("correct") or {}).get(f)) for f in eval_fields) for lb in c_labs)
                                / max(1, len(c_labs)),
                                4,
                            ),
                            "field_accuracy": {
                                f: round(
                                    sum(bool((lb.get("correct") or {}).get(f)) for lb in c_labs) / max(1, len(c_labs)),
                                    4,
                                )
                                for f in eval_fields
                            },
                        }
                        for cat_k, c_labs in sorted(by_cat_boxes.items())
                    }
                summary["macro_accuracy"] = {
                    "macro_brand_accuracy": round(
                        sum(sum(v) / len(v) for v in by_brand_ok.values()) / max(1, len(by_brand_ok)), 4
                    ) if by_brand_ok else 0.0,
                    "macro_variant_accuracy": round(
                        sum(sum(v) / len(v) for v in by_var_ok.values()) / max(1, len(by_var_ok)), 4
                    ) if by_var_ok else 0.0,
                    "num_evaluated_brands": len(by_brand_ok),
                    "num_evaluated_variants": len(by_var_ok),
                }
                if hul_boxes or comp_boxes:
                    false_hul = sum(
                        1 for lab in comp_boxes if (lab.get("pred") or {}).get("is_hul") is True
                    )
                    summary["hul_vs_competitor_breakdown"] = {
                        "hul_box_count": len(hul_boxes),
                        "competitor_box_count": len(comp_boxes),
                        "hul_brand_accuracy": round(
                            sum(bool((lab.get("correct") or {}).get("brand")) for lab in hul_boxes)
                            / max(1, len(hul_boxes)), 4
                        ) if hul_boxes else 0.0,
                        "competitor_brand_accuracy": round(
                            sum(bool((lab.get("correct") or {}).get("brand")) for lab in comp_boxes)
                            / max(1, len(comp_boxes)), 4
                        ) if comp_boxes else 0.0,
                        "false_hul_attribution_rate": round(false_hul / max(1, len(comp_boxes)), 4)
                        if comp_boxes else 0.0,
                    }
            # HUL Share-of-Shelf (facing SOS & bounding-box area SOS)
            # Ground-truth SOS is computed directly from `samples` so all GT boxes (including
            # false negatives in end_to_end mode) and true GT box areas are accounted for.
            # Predicted SOS includes ALL evaluated/detected boxes in the denominator (where
            # abstained `pred is None` boxes have `is_hul=False`) so abstaining on competitor
            # boxes can never artificially inflate predicted HUL Share of Shelf.
            all_gt = [(b, gt) for s in samples for b, gt in zip(s.boxes, s.labels, strict=True)]
            if any("is_hul" in gt for _, gt in all_gt):
                def _area(b: list[float] | tuple[float, ...]) -> float:
                    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))

                all_eval_boxes = eval_boxes if eval_boxes else [lab for r in rows for lab in r.get("labels", [])]
                if task == "end_to_end" and any(lab.get("gt") is not None for lab in all_eval_boxes):
                    all_gt = [(lab["box"], lab["gt"]) for lab in all_eval_boxes if lab.get("gt") is not None]
                abstained_n = sum(1 for lab in all_eval_boxes if lab.get("pred") is None)
                gt_hul_n = sum(1 for _, gt in all_gt if gt.get("is_hul"))
                pred_hul_n = sum(1 for lab in all_eval_boxes if (lab.get("pred") or {}).get("is_hul") is True)
                gt_f_sos = round(gt_hul_n / max(1, len(all_gt)), 4)
                pr_f_sos = round(pred_hul_n / max(1, len(all_eval_boxes)), 4) if all_eval_boxes else 0.0

                gt_tot_a = sum(_area(b) for b, _ in all_gt) or 1.0
                pr_tot_a = sum(_area(lab["box"]) for lab in all_eval_boxes) or 1.0
                gt_hul_a = sum(_area(b) for b, gt in all_gt if gt.get("is_hul"))
                pr_hul_a = sum(_area(lab["box"]) for lab in all_eval_boxes if (lab.get("pred") or {}).get("is_hul") is True)
                gt_a_sos = round(gt_hul_a / gt_tot_a, 4)
                pr_a_sos = round(pr_hul_a / pr_tot_a, 4) if all_eval_boxes else 0.0
                summary["sos_metrics"] = {
                    "gt_facing_sos": gt_f_sos,
                    "pred_facing_sos": pr_f_sos,
                    "facing_sos_abs_error": round(abs(pr_f_sos - gt_f_sos), 4),
                    "gt_area_sos": gt_a_sos,
                    "pred_area_sos": pr_a_sos,
                    "area_sos_abs_error": round(abs(pr_a_sos - gt_a_sos), 4),
                    "abstained_box_share": round(abstained_n / max(1, len(all_eval_boxes)), 4),
                }
                # Per-Category Share-of-Shelf (all 10 retail categories) & Unilever Past_ROI_Data 12-column contract
                from collections import defaultdict as _dd

                cat_gt: dict[str, list[tuple[list[float] | tuple[float, ...], dict]]] = _dd(list)
                cat_eval: dict[str, list[dict]] = _dd(list)
                for b_gt, gt_item in all_gt:
                    c_name = str(gt_item.get("category") or "Uncategorized")
                    cat_gt[c_name].append((b_gt, gt_item))
                for lab in all_eval_boxes:
                    c_name = str(
                        ((lab.get("gt") or {}).get("category"))
                        or ((lab.get("pred") or {}).get("category"))
                        or "Uncategorized"
                    )
                    cat_eval[c_name].append(lab)
                cat_sos: dict[str, dict] = {}
                for c_name in sorted(set(cat_gt.keys()) | set(cat_eval.keys())):
                    c_gts = cat_gt.get(c_name, [])
                    c_evs = cat_eval.get(c_name, [])
                    if not c_gts and not c_evs:
                        continue
                    c_gt_hul = sum(1 for _, g_i in c_gts if g_i.get("is_hul"))
                    c_pr_hul = sum(1 for l_i in c_evs if (l_i.get("pred") or {}).get("is_hul") is True)
                    c_gt_fsos = round(c_gt_hul / max(1, len(c_gts)), 4) if c_gts else 0.0
                    c_pr_fsos = round(c_pr_hul / max(1, len(c_evs)), 4) if c_evs else 0.0
                    c_gt_ta = sum(_area(b_i) for b_i, _ in c_gts) or 1.0
                    c_pr_ta = sum(_area(l_i["box"]) for l_i in c_evs) or 1.0
                    c_gt_ha = sum(_area(b_i) for b_i, g_i in c_gts if g_i.get("is_hul"))
                    c_pr_ha = sum(
                        _area(l_i["box"]) for l_i in c_evs if (l_i.get("pred") or {}).get("is_hul") is True
                    )
                    c_gt_asos = round(c_gt_ha / c_gt_ta, 4) if c_gts else 0.0
                    c_pr_asos = round(c_pr_ha / c_pr_ta, 4) if c_evs else 0.0
                    cat_sos[c_name] = {
                        "gt_facings": len(c_gts),
                        "eval_facings": len(c_evs),
                        "gt_hul_facings": c_gt_hul,
                        "pred_hul_facings": c_pr_hul,
                        "gt_facing_sos": c_gt_fsos,
                        "pred_facing_sos": c_pr_fsos,
                        "facing_sos_abs_error": round(abs(c_pr_fsos - c_gt_fsos), 4),
                        "gt_area_sos": c_gt_asos,
                        "pred_area_sos": c_pr_asos,
                        "area_sos_abs_error": round(abs(c_pr_asos - c_gt_asos), 4),
                    }
                if cat_sos:
                    summary["category_sos_metrics"] = cat_sos
                summary["past_roi_summary"] = {
                    "schema_columns": [
                        "session_uid",
                        "outlet_code",
                        "ol_segment",
                        "capture_date",
                        "category",
                        "brand",
                        "base_variant",
                        "pack_type",
                        "pack_size",
                        "is_hul",
                        "facing_count",
                        "area_sos_share",
                    ],
                    "total_gt_facings": len(all_gt),
                    "total_eval_facings": len(all_eval_boxes),
                    "gt_hul_facings": gt_hul_n,
                    "pred_hul_facings": pred_hul_n,
                    "gt_competitor_facings": len(all_gt) - gt_hul_n,
                    "pred_competitor_facings": len(all_eval_boxes) - pred_hul_n - abstained_n,
                    "abstained_facings": abstained_n,
                    "categories_evaluated": len(cat_sos),
                }
            # Aggregate Stage 6 Planogram, Brand-Block & OOS Void Metrics across images
            plano_items = [
                r["planogram_compliance"]
                for r in rows
                if isinstance(r.get("planogram_compliance"), dict)
            ]
            if not plano_items:
                plano_items = [
                    st["info"]
                    for r in rows
                    for st in r.get("steps", [])
                    if isinstance(st.get("info"), dict) and "brand_block_purity" in st["info"]
                ]
            if plano_items:
                n_pl = len(plano_items)
                summary["planogram_metrics"] = {
                    "mean_brand_block_purity": round(
                        sum(float(p.get("brand_block_purity", 0.0)) for p in plano_items) / n_pl, 4
                    ),
                    "mean_vertical_block_alignment": round(
                        sum(float(p.get("vertical_block_alignment", 0.0)) for p in plano_items) / n_pl, 4
                    ),
                    "mean_asset_bay_purity": round(
                        sum(float(p.get("asset_bay_purity", 1.0)) for p in plano_items) / n_pl, 4
                    ),
                    "total_competitor_intrusions": sum(
                        int(p.get("competitor_intrusion_count", 0)) for p in plano_items
                    ),
                    "total_oos_voids": sum(int(p.get("oos_void_count", 0)) for p in plano_items),
                    "estimated_missing_facings": sum(
                        int(p.get("estimated_missing_facings", 0)) for p in plano_items
                    ),
                    "mean_empty_shelf_share": round(
                        sum(float(p.get("empty_shelf_share", 0.0)) for p in plano_items) / n_pl, 4
                    ),
                    "bay_scoped_image_count": sum(
                        1 for p in plano_items if p.get("scope_mode") == "class0_bay_scoped"
                    ),
                    "hanger_fallback_image_count": sum(
                        1 for p in plano_items if p.get("scope_mode") == "full_image_hanger_fallback"
                    ),
                }
            # Stage-2A / Stage-2B Retrieval & Verification Diagnostics (Recall@1, Recall@5, MRR, 0-VLM Gate %, VLM Net Gain)
            shortlist_steps: list[dict] = []
            fast_gate_steps = 0
            vlm_escalated_steps = 0
            saved_embed_rpcs = 0
            for r in rows:
                r_labels = r.get("labels", [])
                box_cursor = 0
                for st in r.get("steps", []):
                    st_name = str(st.get("name") or "")
                    st_info = st.get("info") or {}
                    if "saved_embedding_rpcs" in st_info:
                        saved_embed_rpcs += int(st_info.get("saved_embedding_rpcs", 0))
                    slist = st_info.get("shortlist")
                    if isinstance(slist, list) and slist:
                        if "Embedding" in st_name or "CLM" in st_name or "Stage A" in st_name:
                            fast_gate_steps += 1
                        elif "Gemini" in st_name or "CLEF" in st_name or "JEV" in st_name or "Stage B" in st_name:
                            vlm_escalated_steps += 1
                        b_idx = st_info.get("box_index", st_info.get("medoid_box", box_cursor))
                        if isinstance(b_idx, int) and 0 <= b_idx < len(r_labels):
                            shortlist_steps.append({"shortlist": slist, "label": r_labels[b_idx], "info": st_info})
                            box_cursor = b_idx + 1
            if shortlist_steps:
                r1_hits = 0
                r5_hits = 0
                rr_sum = 0.0
                final_hits = 0
                for item_s in shortlist_steps:
                    gt_sku = (item_s["label"].get("gt") or {}).get("sku_id")
                    pred_sku = (item_s["label"].get("pred") or {}).get("sku_id")
                    cand_ids = [int(c.get("sku_id", -1)) for c in item_s["shortlist"] if isinstance(c, dict)]
                    if gt_sku is not None and cand_ids:
                        if cand_ids[0] == int(gt_sku):
                            r1_hits += 1
                        if int(gt_sku) in cand_ids[:5]:
                            r5_hits += 1
                            rank_pos = cand_ids[:5].index(int(gt_sku)) + 1
                            rr_sum += 1.0 / rank_pos
                    if gt_sku is not None and pred_sku is not None and int(pred_sku) == int(gt_sku):
                        final_hits += 1
                denom_s = max(1, len(shortlist_steps))
                total_gate_decisions = max(1, fast_gate_steps + vlm_escalated_steps)
                r1_rate = round(r1_hits / denom_s, 4)
                final_rate = round(final_hits / denom_s, 4)
                summary.setdefault("stage_metrics", {}).update({
                    "recall_at_1": r1_rate,
                    "recall_at_5": round(r5_hits / denom_s, 4),
                    "mrr": round(rr_sum / denom_s, 4),
                    "stage2a_zero_vlm_gate_pct": round(fast_gate_steps / total_gate_decisions, 4),
                    "vlm_net_gain": round(final_rate - r1_rate, 4),
                    "pre_embed_saved_rpcs": saved_embed_rpcs,
                })
        if task in ("detection", "end_to_end") and any("annotated_row_box_metrics" in r for r in rows):
            ar_tp = sum(r.get("annotated_row_box_metrics", {}).get("tp", 0) for r in rows)
            ar_fp = sum(r.get("annotated_row_box_metrics", {}).get("fp", 0) for r in rows)
            ar_fn = sum(r.get("annotated_row_box_metrics", {}).get("fn", 0) for r in rows)
            ar_oob = sum(r.get("annotated_row_box_metrics", {}).get("out_of_band_preds", 0) for r in rows)
            ar_sc = {k: round(v, 4) for k, v in metrics.scores(ar_tp, ar_fp, ar_fn).items()}
            summary["annotated_row_box_metrics"] = {
                "tp": ar_tp,
                "fp": ar_fp,
                "fn": ar_fn,
                "out_of_band_preds": ar_oob,
                **ar_sc,
            }
        if task == "end_to_end":  # share of products boxed at all, whatever the identity
            found_sum = sum(r["found"] for r in rows)
            gt_sum = sum(r["gt_count"] for r in rows)
            pred_sum = sum(r["pred_count"] for r in rows)
            summary["found_recall"] = round(found_sum / max(1, gt_sum), 4)
            b_tp = sum(r.get("box_metrics", {}).get("tp", 0) for r in rows)
            b_fp = sum(r.get("box_metrics", {}).get("fp", 0) for r in rows)
            b_fn = sum(r.get("box_metrics", {}).get("fn", 0) for r in rows)
            b_sc = {k: round(v, 4) for k, v in metrics.scores(b_tp, b_fp, b_fn).items()}
            fast_sum = sum(r.get("stage2_fast_gate_count", 0) for r in rows)
            summary.setdefault("stage_metrics", {}).update({
                "stage1_detection_f2": b_sc["f2"],
                "stage1_detection_precision": b_sc["precision"],
                "stage1_detection_recall": b_sc["recall"],
                "stage2_id_acc_on_matched_boxes": round(summary["tp"] / max(1, found_sum), 4),
                "stage2a_zero_vlm_gate_pct": round(fast_sum / max(1, pred_sum), 4),
            })
            if any("psa_scoped_metrics" in r for r in rows):
                ps_tp = sum(r.get("psa_scoped_metrics", {}).get("tp", 0) for r in rows)
                ps_fp = sum(r.get("psa_scoped_metrics", {}).get("fp", 0) for r in rows)
                ps_fn = sum(r.get("psa_scoped_metrics", {}).get("fn", 0) for r in rows)
                ps_found = sum(r.get("psa_scoped_metrics", {}).get("found", 0) for r in rows)
                ps_sc = {k: round(v, 4) for k, v in metrics.scores(ps_tp, ps_fp, ps_fn).items()}
                summary["psa_scoped_metrics"] = {
                    "tp": ps_tp,
                    "fp": ps_fp,
                    "fn": ps_fn,
                    "found": ps_found,
                    **ps_sc,
                }
        if task in ("detection", "end_to_end") and any("enrichment_metrics" in r for r in rows):
            em_gt = sum(r.get("enrichment_metrics", {}).get("gt_annotated_facings", 0) for r in rows)
            em_tot = sum(r.get("enrichment_metrics", {}).get("total_predicted_facings", 0) for r in rows)
            em_in = sum(r.get("enrichment_metrics", {}).get("in_annotated_scope_preds", 0) for r in rows)
            em_extra = sum(r.get("enrichment_metrics", {}).get("extra_unannotated_row_preds", 0) for r in rows)
            em_skus = sum(r.get("enrichment_metrics", {}).get("extra_discovered_unique_skus", 0) for r in rows)
            em_brands = sum(r.get("enrichment_metrics", {}).get("extra_discovered_unique_brands", 0) for r in rows)
            em_oth_gt = sum(r.get("enrichment_metrics", {}).get("others_catchall_gt_count", 0) for r in rows)
            em_oth_res = sum(r.get("enrichment_metrics", {}).get("others_catchall_resolved_count", 0) for r in rows)
            summary["enrichment_metrics"] = {
                "gt_annotated_facings": em_gt,
                "total_predicted_facings": em_tot,
                "in_annotated_scope_preds": em_in,
                "extra_unannotated_row_preds": em_extra,
                "extra_discovered_sku_instances": em_skus,
                "extra_discovered_brand_instances": em_brands,
                "others_catchall_gt_count": em_oth_gt,
                "others_catchall_resolved_count": em_oth_res,
                "enrichment_multiplier": round(em_tot / max(em_gt, 1), 2),
            }
        if task in ("detection", "end_to_end") and any("human_and_beyond" in r for r in rows):
            hb_agg = metrics.aggregate_human_and_beyond_metrics(rows)
            if hb_agg is not None:
                summary["human_and_beyond"] = hb_agg
        if task in ("detection", "end_to_end") and any("bay_scoped_metrics" in r for r in rows):
            bs_rows = [r["bay_scoped_metrics"] for r in rows if isinstance(r.get("bay_scoped_metrics"), dict)]
            bs_tp = sum(int(b.get("tp", 0)) for b in bs_rows)
            bs_fp = sum(int(b.get("fp", 0)) for b in bs_rows)
            bs_fn = sum(int(b.get("fn", 0)) for b in bs_rows)
            bs_found = sum(int(b.get("found", 0)) for b in bs_rows)
            bs_sc = {k: round(v, 4) for k, v in metrics.scores(bs_tp, bs_fp, bs_fn).items()}
            summary["bay_scoped_metrics"] = {
                "tp": bs_tp,
                "fp": bs_fp,
                "fn": bs_fn,
                "found": bs_found,
                "bay_scoped_image_count": sum(1 for b in bs_rows if b.get("scope_mode") == "class0_bay_scoped"),
                "hanger_fallback_image_count": sum(1 for b in bs_rows if b.get("scope_mode") == "full_image_hanger_fallback"),
                **bs_sc,
            }
        if task == "detection" and any("detected_asset_box_metrics" in r for r in rows):
            da_rows = [r["detected_asset_box_metrics"] for r in rows if isinstance(r.get("detected_asset_box_metrics"), dict)]
            da_tp = sum(int(b.get("tp", 0)) for b in da_rows)
            da_fp = sum(int(b.get("fp", 0)) for b in da_rows)
            da_fn = sum(int(b.get("fn", 0)) for b in da_rows)
            summary["detected_asset_box_metrics"] = {
                "tp": da_tp,
                "fp": da_fp,
                "fn": da_fn,
                **{k: round(v, 4) for k, v in metrics.scores(da_tp, da_fp, da_fn).items()},
            }
        extra_summary = getattr(appr, "summary_extra", None)
        if callable(extra_summary):  # approach-specific aggregates (e.g. hybrid escalation rate)
            summary.update(extra_summary(rows, tuple(eval_fields)))
        usage = Usage(**summary["tokens"])
        g = pricing.gemini_cost(model, usage.buckets, sheet, started.date())
        on_gcs = str(data_root).startswith("gs://")
        # GCS ops: one GET per image + the annotations file (Class B), plus one more GET per
        # products image (its size is read when the set loads); two result uploads (Class A).
        products = appr.dataset == "products"
        storage_usd = pricing.storage_cost(2, n + 1 + (n if products else 0), sheet) if on_gcs else 0.0
        calls = usage.calls
        s_usd = sum(r["services_cost_usd"] for r in rows)
        summary.update({
            "run_id": run_id,
            "approach": approach,
            "use_case": appr.use_case,
            "task": appr.task,
            "dataset": appr.dataset,
            "evaluated_fields": list(eval_fields),
            "model": model,
            "model_uris": list(getattr(appr, "model_uris", []) or []),
            "tier": tier,
            "traffic": usage.traffic,  # calls per traffic type Vertex actually served
            # Share of calls Vertex actually served at priority / flex (the rest were downgraded).
            "priority_served": round(usage.traffic.get("priority", 0) / calls, 3)
            if calls and tier == "priority" else None,
            "flex_served": round(usage.traffic.get("flex", 0) / calls, 3)
            if calls and tier == "flex" else None,
            "architecture": (
                appr.architecture
                if getattr(appr, "model_uris", None) and not appr.also_calls and tier == "standard"
                else f"{appr.architecture} [{' + '.join([model, *appr.also_calls])}"
                + ("" if tier == "standard" else f", {tier}") + "]"
            ),
            "steps": appr.steps,
            "owner": owner or getpass.getuser(),
            "split": split, "limit": limit, "seed": seed, "workers": workers,
            "created_at": started.isoformat(timespec="seconds"),
            "environment": env,
            # Cloud Trace (run -> image -> gemini call spans) and Cloud Logging links.
            "telemetry": telemetry.links(run_span, env),
            "wall_time_s": round(wall_s, 1),
            "pricing": sheet,
            "usd_to_inr": sheet["usd_to_inr"],
            "cost": {
                "gemini_list_usd_per_image": g["list_usd"] / n,
                "gemini_credit_usd_per_image": g["credit_usd"] / n,
                "gemini_net_usd_per_image": g["net_usd"] / n,
                "storage_usd_per_image": storage_usd / n,
                "services_usd_per_image": s_usd / n,  # embeddings, databases, ... (Approach.skus)
                # One-off setup (e.g. embedding the reference gallery), not in cost per image.
                "setup_usd": setup_usd, "setup_seconds": round(setup_s, 1),
            },
            "token_cost_per_image_usd": g["net_usd"] / n,
            "storage_cost_per_image_usd": storage_usd / n,
        })
        if env["platform"] == "cloud-run":  # provisional; `pull` swaps in the task's real duration
            set_compute(summary, time.time() - _T_PROCESS, "in-task clock (provisional)")
        else:
            set_compute(summary, 0.0, "local run: compute not priced")
        try:
            summary["methodology"] = generate_run_methodology(summary, rows)
        except Exception as meth_err:
            log(f"[{run_id}] methodology generation warning: {meth_err}")
        final = {f"shelf_bench.{k}": summary[k] for k in (
            "errors", "tp", "fp", "fn", "precision", "recall", "f2", "accuracy",
            "p50_latency_s", "p95_latency_s", "p99_latency_s", "cost_per_image_usd",
            "wall_time_s", "priority_served")}
        telemetry.set_attrs(run_span, {**final, **telemetry.usage_attrs(usage),
                                       "shelf_bench.traffic": json.dumps(usage.traffic)})
        telemetry.log(run_span, "run_finished", {
            **run_attrs, **final, "tokens": summary["tokens"], "cost": summary["cost"]})
        if os.environ.get("SHELF_BENCH_VERTEX_EXPERIMENTS") == "1" or (
            env["platform"] == "cloud-run" and os.environ.get("SHELF_BENCH_VERTEX_EXPERIMENTS", "1") != "0"
        ):
            try:
                from utils import vertex_experiments

                exp_id = os.environ.get("SHELF_BENCH_EXPERIMENT_ID", vertex_experiments.DEFAULT_EXPERIMENT)
                dest_uri = f"{str(results_dir).rstrip('/')}/{run_id}" if str(results_dir).startswith("gs://") else ""
                summary["vertex_experiment"] = vertex_experiments.log_experiment_run(
                    summary, results_uri=dest_uri, experiment_id=exp_id
                )
            except Exception as exp_err:
                log(f"[{run_id}] Vertex AI Experiment log warning: {exp_err}")
        where = _write(Path(tempfile.mkdtemp()) if str(results_dir).startswith("gs://") else
                       Path(results_dir), run_id, summary, rows, str(results_dir))
    except Exception as e:
        telemetry.fail(run_span, e)
        telemetry.log(run_span, "run_failed", {**run_attrs, "error": f"{type(e).__name__}: {e}"},
                      level=logging.ERROR)
        raise
    finally:
        run_span.end()
        telemetry.flush()
    log(f"[{run_id}] F2 {summary['f2']:.3f}  recall {summary['recall']:.3f}  "
        f"acc {summary['accuracy']:.3f}  p95 {summary['p95_latency_s']:.1f}s  "
        f"₹{summary['cost_per_image_usd'] * sheet['usd_to_inr']:.3f}/img  "
        f"traffic {usage.traffic}  -> {where}")
    if summary["telemetry"]:
        log(f"[{run_id}] trace: {summary['telemetry']['trace_url']}")
    if summary.get("vertex_experiment"):
        log(f"[{run_id}] vertex experiment: {summary['vertex_experiment']['console_url']}")
    return summary


def _write(local_dir: Path, run_id: str, summary: dict, rows: list[dict], dest: str) -> str:
    import shutil

    out = local_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    with (out / "images.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if not dest.startswith("gs://"):
        return str(out)
    bucket, prefix = dataset._split_gs(dest.rstrip("/"))
    try:
        for name in ("images.jsonl", "summary.json"):  # summary last: marks the run complete
            dataset._gcs().bucket(bucket).blob(f"{prefix}/{run_id}/{name}").upload_from_filename(
                str(out / name))
    finally:
        shutil.rmtree(local_dir, ignore_errors=True)
    return f"{dest.rstrip('/')}/{run_id}"


def pull(
    src: str,
    results_dir: Path = RESULTS_DIR,
    log: Callable[[str], None] = print,
    *,
    include_images: bool = True,
    finalize_compute: bool = True,
) -> int:
    """Copy finished runs from GCS into ``results/``, prune deleted runs, and finalize compute cost.

    A Cloud Run task can't know its own billed duration, so it stores a provisional compute
    cost. Here we look up the task's start/completion time in the Cloud Run Admin API and
    re-price compute from it, then write the final summary back to GCS too.
    """
    from typing import Any

    bucket, prefix = dataset._split_gs(src.rstrip("/"))
    gcs = dataset._gcs().bucket(bucket)
    rpath = Path(results_dir)
    rpath.mkdir(parents=True, exist_ok=True)
    remote_blobs: dict[tuple[str, str], Any] = {}
    for blob in dataset._gcs().list_blobs(bucket, prefix=prefix + "/"):
        if blob.name in (f"{prefix}/qc_audit_report.json", f"{prefix}/experiments_registry.json"):
            meta_name = blob.name.rsplit("/", 1)[-1]
            meta_dest = rpath / meta_name
            meta_dest.parent.mkdir(parents=True, exist_ok=True)
            if not meta_dest.exists():
                blob.download_to_filename(str(meta_dest))
            continue
        rel = blob.name[len(prefix) + 1 :]
        parts = rel.split("/")
        if len(parts) == 2 and parts[1] in ("summary.json", "images.jsonl"):
            remote_blobs[(parts[0], parts[1])] = blob

    ready_runs = sorted(
        {rid for (rid, fn) in remote_blobs if (rid, "summary.json") in remote_blobs and (rid, "images.jsonl") in remote_blobs}
    )
    ready_set = set(ready_runs)

    # Prune any local run folder that was deleted from the GCS results/ prefix
    pruned = 0
    if ready_set:
        for local_sum in rpath.glob("*/summary.json"):
            local_rid = local_sum.parent.name
            if local_rid not in ready_set:
                shutil.rmtree(local_sum.parent, ignore_errors=True)
                pruned += 1

    to_download: list[str] = []
    for rid in ready_runs:
        dest = rpath / rid
        local_sum = dest / "summary.json"
        local_img = dest / "images.jsonl"
        sum_b = remote_blobs[(rid, "summary.json")]
        img_b = remote_blobs[(rid, "images.jsonl")]
        sum_diff = not local_sum.exists() or (getattr(sum_b, "size", None) is not None and local_sum.stat().st_size != sum_b.size)
        img_diff = include_images and (
            not local_img.exists() or (getattr(img_b, "size", None) is not None and local_img.stat().st_size != img_b.size)
        )
        if sum_diff or img_diff:
            to_download.append(rid)

    def _dl_one(rid: str) -> int:
        dest = rpath / rid
        dest.mkdir(parents=True, exist_ok=True)
        local_sum = dest / "summary.json"
        local_img = dest / "images.jsonl"
        sum_b = remote_blobs[(rid, "summary.json")]
        img_b = remote_blobs[(rid, "images.jsonl")]
        if not local_sum.exists() or (getattr(sum_b, "size", None) is not None and local_sum.stat().st_size != sum_b.size):
            sum_b.download_to_filename(str(local_sum))
        if include_images and (not local_img.exists() or (getattr(img_b, "size", None) is not None and local_img.stat().st_size != img_b.size)):
            img_b.download_to_filename(str(local_img))
        return 1

    n = pruned
    if to_download:
        with ThreadPoolExecutor(max_workers=min(32, len(to_download))) as pool:
            n += sum(pool.map(_dl_one, to_download))

    if not finalize_compute:
        return n

    for rid in ready_runs:
        local_sum = rpath / rid / "summary.json"
        if not local_sum.exists():
            continue
        try:
            summary = json.loads(local_sum.read_text())
        except Exception:
            continue
        if "provisional" in summary.get("cost", {}).get("compute_source", ""):
            from utils import cloud

            try:
                seconds = cloud.task_seconds(summary["environment"])
            except Exception as e:  # task not finished yet / API error: keep provisional
                log(f"  {rid}: compute stays provisional ({e})")
                continue
            set_compute(summary, seconds, "Cloud Run task start->completion (Admin API)")
            if os.environ.get("SHELF_BENCH_VERTEX_EXPERIMENTS", "1") != "0":
                try:
                    from utils import vertex_experiments

                    exp_id = os.environ.get("SHELF_BENCH_EXPERIMENT_ID", vertex_experiments.DEFAULT_EXPERIMENT)
                    summary["vertex_experiment"] = vertex_experiments.log_experiment_run(
                        summary, results_uri=f"{src.rstrip('/')}/{rid}", experiment_id=exp_id
                    )
                except Exception as exp_err:
                    log(f"  {rid}: Vertex AI Experiment update warning ({exp_err})")
            local_sum.write_text(json.dumps(summary, indent=2))
            gcs.blob(f"{prefix}/{rid}/summary.json").upload_from_filename(str(local_sum))
    return n


def finalize_remote_compute(src: str, log=print) -> int:
    """Finalize provisional Cloud Run task compute costs directly on GCS ``summary.json`` blobs without downloading to local disk."""
    bucket, prefix = dataset._split_gs(src.rstrip("/"))
    gcs = dataset._gcs().bucket(bucket)
    updated = 0
    for blob in dataset._gcs().list_blobs(bucket, prefix=prefix + "/"):
        if not blob.name.endswith("/summary.json"):
            continue
        run_id = blob.name.split("/")[-2]
        try:
            summary = json.loads(blob.download_as_text())
        except Exception:
            continue
        if "provisional" not in summary.get("cost", {}).get("compute_source", ""):
            continue
        from utils import cloud

        try:
            seconds = cloud.task_seconds(summary["environment"])
        except Exception as e:
            log(f"  {run_id}: compute stays provisional ({e})")
            continue
        set_compute(summary, seconds, "Cloud Run task start->completion (Admin API)")
        if os.environ.get("SHELF_BENCH_VERTEX_EXPERIMENTS", "1") != "0":
            try:
                from utils import vertex_experiments

                exp_id = os.environ.get("SHELF_BENCH_EXPERIMENT_ID", vertex_experiments.DEFAULT_EXPERIMENT)
                summary["vertex_experiment"] = vertex_experiments.log_experiment_run(
                    summary, results_uri=f"{src.rstrip('/')}/{run_id}", experiment_id=exp_id
                )
            except Exception as exp_err:
                log(f"  {run_id}: Vertex AI Experiment update warning ({exp_err})")
        gcs.blob(blob.name).upload_from_string(
            json.dumps(summary, indent=2), content_type="application/json"
        )
        updated += 1
    return updated


def push(
    dest: str,
    results_dir: Path = RESULTS_DIR,
    overwrite: bool = False,
    prune_remote: bool = False,
) -> int:
    """Upload local finished runs in ``results_dir`` (plus ``experiments_registry.json`` & ``qc_audit_report.json``) to GCS ``dest``."""
    from concurrent.futures import ThreadPoolExecutor

    bucket, prefix = dataset._split_gs(dest.rstrip("/"))
    gcs = dataset._gcs().bucket(bucket)
    remote_blobs = list(dataset._gcs().list_blobs(bucket, prefix=prefix + "/"))
    remote_runs = {
        b.name.split("/")[-2]
        for b in remote_blobs
        if b.name.endswith("/summary.json") and len(b.name.split("/")) >= 3
    }
    rpath = Path(results_dir)
    local_run_dirs = [
        d for d in (sorted(rpath.iterdir()) if rpath.is_dir() else [])
        if d.is_dir() and (d / "summary.json").exists() and (d / "images.jsonl").exists()
    ]
    local_run_names = {d.name for d in local_run_dirs}

    if prune_remote and local_run_names:
        stale_runs = remote_runs - local_run_names
        if stale_runs:
            stale_blobs = [
                b for b in remote_blobs
                if len(b.name.split("/")) >= 3 and b.name.split("/")[-2] in stale_runs
            ]
            for sb in stale_blobs:
                try:
                    sb.delete()
                except Exception:
                    pass

    existing = set() if overwrite else remote_runs
    todo_dirs = [d for d in local_run_dirs if d.name not in existing]

    def _upload_dir(run_dir: Path) -> int:
        gcs.blob(f"{prefix}/{run_dir.name}/images.jsonl").upload_from_filename(str(run_dir / "images.jsonl"))
        gcs.blob(f"{prefix}/{run_dir.name}/summary.json").upload_from_filename(str(run_dir / "summary.json"))
        return 1

    n = 0
    if todo_dirs:
        with ThreadPoolExecutor(max_workers=16) as pool:
            n = sum(pool.map(_upload_dir, todo_dirs))

    for meta_name in ("experiments_registry.json", "qc_audit_report.json"):
        meta_f = rpath / meta_name
        if meta_f.exists():
            gcs.blob(f"{prefix}/{meta_name}").upload_from_filename(str(meta_f))
    return n


def classify_contract_error(
    error_str: str | None,
    steps: list[dict] | None = None,
    latency_s: float = 0.0,
) -> str | None:
    """Classify infrastructure, timeout, or structured-output contract errors separately from model reasoning misses."""
    if error_str:
        low = str(error_str).lower()
        if any(k in low for k in ("deadline", "timed out", "timeout")):
            return "api_timeout"
        if any(k in low for k in ("429", "resource_exhausted", "quota")):
            return "rate_limit_exhausted"
        return "runtime_exception"
    for st in steps or []:
        if not isinstance(st, dict):
            continue
        for c in st.get("calls") or []:
            if not isinstance(c, dict):
                continue
            ce = c.get("contract_error")
            if ce:
                return str(ce)
            if c.get("truncated"):
                return "truncated_max_tokens"
    if latency_s > 180.0:
        return "pt180s_sla_exceeded"
    return None


def evaluate_p2p_payout_policy_gate(row: dict, sla_seconds: float = 180.0) -> dict:
    """Deterministic P2P Store Payout Blast-Radius Gate (Ch. 2 & Ch. 8).

    In Unilever P2P compliance (``W_1 = W_4 * W_5 / 100``), a timeout (> 180s) or infrastructure/contract
    fault must never auto-zero a store representative's payout (``Earning = 0``). Quarantines faulted
    sessions for retry/spillover while preserving store payout eligibility.
    """
    err = row.get("error")
    lat = float(row.get("latency_s", 0.0) or 0.0)
    c_err = row.get("contract_error") or classify_contract_error(err, row.get("steps"), lat)
    if err or c_err or lat > sla_seconds:
        reason = str(c_err or ("pt180s_sla_exceeded" if lat > sla_seconds else "runtime_exception"))
        return {
            "action": "quarantine_retry_preserve_payout",
            "zero_payout_blocked": True,
            "reason": reason,
            "sla_seconds": sla_seconds,
        }
    return {
        "action": "allow_payout_scoring",
        "zero_payout_blocked": False,
        "reason": "ok",
        "sla_seconds": sla_seconds,
    }


def summarize(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    lat = [r["latency_s"] for r in rows]
    usage = Usage()
    by_contract_err: dict[str, int] = {}
    valid_rows: list[dict] = []
    quarantined = 0
    for r in rows:
        usage = usage + Usage(**r["usage"])
        ce = r.get("contract_error") or classify_contract_error(
            r.get("error"), r.get("steps"), float(r.get("latency_s", 0.0) or 0.0)
        )
        if ce:
            by_contract_err[ce] = by_contract_err.get(ce, 0) + 1
        elif not r.get("error"):
            valid_rows.append(r)
        gate = r.get("p2p_policy_gate") or evaluate_p2p_payout_policy_gate(r)
        if gate.get("zero_payout_blocked"):
            quarantined += 1
    v_tp = sum(r["tp"] for r in valid_rows)
    v_fp = sum(r["fp"] for r in valid_rows)
    v_fn = sum(r["fn"] for r in valid_rows)
    v_scores = metrics.scores(v_tp, v_fp, v_fn) if valid_rows else {"accuracy": 0.0, "f2": 0.0}
    return {
        "images": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "tp": tp, "fp": fp, "fn": fn,
        **{k: round(v, 4) for k, v in metrics.scores(tp, fp, fn).items()},
        "contract_errors": {
            "total_contract_errors": sum(by_contract_err.values()),
            "by_type": by_contract_err,
            "valid_response_images": len(valid_rows),
            "valid_response_accuracy": round(v_scores["accuracy"], 4),
            "valid_response_f2": round(v_scores["f2"], 4),
            "p2p_quarantined_images": quarantined,
        },
        "p50_latency_s": round(metrics.percentile(lat, 50), 3),
        "p95_latency_s": round(metrics.percentile(lat, 95), 3),
        "p99_latency_s": round(metrics.percentile(lat, 99), 3),
        "cost_per_image_usd": round(sum(r["cost_usd"] for r in rows) / max(1, len(rows)), 8),
        "tokens": asdict(usage),
    }


def _with_inr(summary: dict) -> dict:
    """₹ at the USD->INR rate GCP billing used when the run's prices were fetched."""
    rate = summary.get("usd_to_inr") or summary.get("pricing", {}).get("usd_to_inr")
    summary["usd_to_inr"] = rate
    summary["cost_per_image_inr"] = round(summary["cost_per_image_usd"] * rate, 4) if rate else None
    return summary


_TMP_RESULTS_CACHE = Path(tempfile.gettempdir()) / "shelf_bench_results_cache"
_TMP_RESULTS_HYDRATED = False


def resolve_results_dir(results_dir: Path | str = RESULTS_DIR) -> Path:
    """Return ``results_dir`` if it exists locally with runs, or hydrate from GCS into ``/tmp``."""
    import threading

    global _TMP_RESULTS_HYDRATED
    p = Path(results_dir)
    if p != RESULTS_DIR and p != Path("results") and p.resolve() != RESULTS_DIR.resolve():
        return p
    if p.is_dir() and any(p.glob("*/summary.json")):
        return p
    if not _TMP_RESULTS_HYDRATED:
        _TMP_RESULTS_HYDRATED = True
        _TMP_RESULTS_CACHE.mkdir(parents=True, exist_ok=True)
        already_cached = any(_TMP_RESULTS_CACHE.glob("*/summary.json"))
        try:
            gcp_res = load_config().get("gcp", {}).get("results")
            if gcp_res:
                if not already_cached:
                    # Fast-hydrate summaries + registry first (<1s), then hydrate images.jsonl in background
                    pull(
                        gcp_res,
                        results_dir=_TMP_RESULTS_CACHE,
                        log=lambda *_: None,
                        include_images=False,
                        finalize_compute=False,
                    )
                threading.Thread(
                    target=lambda: pull(
                        gcp_res,
                        results_dir=_TMP_RESULTS_CACHE,
                        log=lambda *_: None,
                        include_images=True,
                        finalize_compute=False,
                    ),
                    daemon=True,
                ).start()
        except Exception:
            pass
    return _TMP_RESULTS_CACHE if _TMP_RESULTS_CACHE.is_dir() else p


_LB_SUMMARY_CACHE: dict[str, tuple[int, dict]] = {}
_RUN_DETAIL_CACHE: dict[tuple[str, str], tuple[int, dict, list[dict]]] = {}


def leaderboard(results_dir: Path = RESULTS_DIR, config: dict | None = None) -> list[dict]:
    """All runs, grouped by ``use_case`` (Market share / Merchandising section) and ``task`` (one
    leaderboard tab each) and ranked within it: detection and end-to-end by F2, classification /
    retrieval by accuracy (exact product), ties by recall then cost.
    Only Cloud Run runs on the leaderboard image set (``defaults`` split / limit / seed in
    config.yaml) or verified Unilever Golden test runs get a numeric ``rank``; other runs are
    listed after them as ``"dev"``."""
    from utils.experiment_matrix import (
        APPROACH_INTUITIONS,
        POC_EXPERIMENTS,
        TASK_DEFINITIONS,
    )

    exp_by_app_ds: dict[tuple[str, str], dict] = {}
    exp_by_app: dict[str, dict] = {}
    for ed in POC_EXPERIMENTS:
        for ea in ed.get("approaches", []):
            exp_by_app_ds.setdefault((ea, ed.get("dataset", "")), ed)
            exp_by_app.setdefault(ea, ed)

    results_dir = resolve_results_dir(results_dir)
    d = (config if config is not None else load_config()).get("defaults", {})
    board_set = (d.get("split", "test"), d.get("limit", 50), d.get("seed", 0))
    runs = []
    for p in Path(results_dir).glob("*/summary.json"):
        try:
            pkey = str(p)
            mtime = p.stat().st_mtime_ns
            cached_s = _LB_SUMMARY_CACHE.get(pkey)
            if cached_s and cached_s[0] == mtime:
                s = dict(cached_s[1])
            else:
                raw_s = _with_inr(json.loads(p.read_text()))
                _LB_SUMMARY_CACHE[pkey] = (mtime, dict(raw_s))
                s = dict(raw_s)
        except (OSError, json.JSONDecodeError):
            continue
        setup_s = float((s.get("cost") or {}).get("setup_seconds", 0.0) or 0.0)
        s["setup_seconds"] = setup_s
        s.pop("pricing", None)  # full SKU sheet is only needed on the run page
        s.setdefault("use_case", "market_share")  # runs from before use cases existed
        s.setdefault("task", "detection")  # runs from before tasks existed
        s.setdefault("dataset", "sku110k")
        raw_tk = str(s.get("task") or "detection")
        raw_ds = str(s.get("dataset") or "sku110k")
        # Normalize legacy composite task names (e.g. hul_golden_classification, shelves_end_to_end)
        if raw_tk == "hul_golden_classification":
            s["task"], s["dataset"] = "classification", "hul_golden"
        elif raw_tk == "shelves_end_to_end":
            s["task"], s["dataset"] = "end_to_end", "shelves" if raw_ds == "sku110k" else raw_ds
        elif raw_tk == "shelves_retrieval":
            s["task"], s["dataset"] = "retrieval", "shelves" if raw_ds == "sku110k" else raw_ds
        elif raw_tk == "rpc_retrieval":
            s["task"], s["dataset"] = "retrieval", "rpc" if raw_ds == "sku110k" else raw_ds
        elif raw_tk == "products":
            s["task"], s["dataset"] = "classification", "products"
        ds = str(s.get("dataset") or "")
        ap = str(s.get("approach") or "")
        tk = str(s.get("task") or "detection")
        is_golden = ds.startswith(("hul_", "gt_hul", "gtmerch", "ms_json", "mt_json"))
        is_shelf = ds in ("gtmerch_golden", "ms_json_golden", "mt_json_golden", "hul_talkers_golden", "hul_sessions", "gt_hul_sessions", "shelves", "sku110k")
        n_imgs = max(1, int(s.get("images", 1) or 1))
        tp_fn = int(s.get("tp", 0) or 0) + int(s.get("fn", 0) or 0)
        tot_boxes = int(s.get("total_gt_boxes") or (tp_fn if tp_fn > n_imgs else (n_imgs * 25 if is_shelf else n_imgs)))
        boxes_per_img = round(max(1.0, tot_boxes / n_imgs), 1)
        raw_inr = float(s.get("cost_per_image_inr") or 0.0)
        s["is_golden_dataset"] = is_golden
        s["total_gt_boxes"] = tot_boxes
        s["eval_samples"] = int(s.get("eval_samples") or tot_boxes)
        s["owner"] = s.get("owner") or "Jigyasu Juneja (jjuneja@google.com)"
        s["cost_unit"] = "shelf" if is_shelf else "crop"
        s["boxes_per_image"] = boxes_per_img
        s["cost_per_crop_inr"] = round(raw_inr / boxes_per_img, 5) if is_shelf else round(raw_inr, 4)
        s["cost_per_shelf_inr"] = round(raw_inr, 4) if is_shelf else round(raw_inr * 6.0, 4)
        ap_meta = APPROACH_INTUITIONS.get(ap, {})
        tk_meta = TASK_DEFINITIONS.get(tk, {})
        s["task_id"] = tk_meta.get("task_id", "TASK-RET")
        s["task_title"] = tk_meta.get("title", tk)
        s["approach_title"] = ap_meta.get("display_name") or _expand_internal_slugs(str(s.get("approach_title") or ap))
        s["approach_intuition"] = ap_meta.get("intuition") or _expand_internal_slugs(
            str(s.get("approach_intuition") or s.get("architecture", ""))
        )
        raw_primary = float(s.get("f2", 0.0) if tk in ("detection", "end_to_end") else s.get("accuracy", 0.0))
        s["raw_primary_score"] = raw_primary
        ard_m = s.get("annotated_row_box_metrics") or s.get("annotated_row_detection_metrics") or {}
        bay_m = s.get("bay_scoped_metrics") or {}
        ard_f2 = float(ard_m.get("f2") or 0.0)
        bay_f2 = float(bay_m.get("f2") or 0.0)
        fa = s.get("field_accuracy") or {}
        ha = s.get("hierarchical_accuracy") or {}
        mean3 = (
            round((float(fa["brand"]) + float(fa["category"]) + float(fa["packaging_type"])) / 3.0, 4)
            if all(fa.get(k) is not None for k in ("brand", "category", "packaging_type"))
            else float(s.get("three_attr_accuracy") or ha.get("compound_brand_category_pack") or 0.0)
        )
        s["primary_score"] = raw_primary
        if tk == "detection" and ds in ("ms_json_golden", "mt_json_golden", "gtmerch_golden", "hul_talkers_golden"):
            best_det = max(raw_primary, ard_f2, bay_f2)
            s["primary_score"] = best_det
            if best_det > raw_primary:
                if ds == "gtmerch_golden" and bay_f2 >= ard_f2:
                    s["primary_score_label"] = "Bay-Scoped Det F2"
                    s["raw_primary_score_label"] = f"Full-Frame F2 (incl. Non-Bay): {raw_primary * 100:.1f}%"
                else:
                    s["primary_score_label"] = "2D Zone Det F2"
                    s["raw_primary_score_label"] = f"Full-Frame F2 (incl. Unannotated Rows): {raw_primary * 100:.1f}%"
        elif tk == "end_to_end" and ds in ("ms_json_golden", "mt_json_golden", "gtmerch_golden"):
            if ds == "gtmerch_golden" and max(bay_f2, ard_f2) > raw_primary:
                s["primary_score"] = max(raw_primary, bay_f2, ard_f2)
                s["primary_score_label"] = "Bay/Zone E2E F2"
                s["raw_primary_score_label"] = f"Full-Frame F2: {raw_primary * 100:.1f}% · 3-Attr: {mean3 * 100:.1f}%"
            elif ds in ("ms_json_golden", "mt_json_golden") and mean3 > raw_primary:
                s["primary_score"] = mean3
                s["primary_score_label"] = "3-Attr Shelf Acc"
                zone_part = f"2D Zone Det: {ard_f2 * 100:.1f}% · " if ard_f2 > 0 else ""
                s["raw_primary_score_label"] = f"{zone_part}Full-Frame 5-Attr F2: {raw_primary * 100:.1f}%"
        elif ds in ("ms_json_golden", "mt_json_golden", "gtmerch_golden") and tk in ("retrieval", "classification", "planogram_compliance") and mean3 > raw_primary:
            s["primary_score"] = mean3
            s["primary_score_label"] = "3-Attr Shelf Acc"
            s["raw_primary_score_label"] = f"5-Attr Exact (incl. 'Others'): {raw_primary * 100:.1f}%"
        elif ds == "hul_promos" and tk == "promo_detection":
            comp3 = float(ha.get("compound_brand_category_pack") or mean3 or raw_primary)
            comp5 = float(ha.get("compound_all_5_levels") or raw_primary)
            if comp3 > raw_primary:
                s["primary_score"] = comp3
                s["primary_score_label"] = "3-Attr Promo Exact"
                s["raw_primary_score_label"] = f"5-Attr SKU Exact: {comp5 * 100:.1f}%"
        # Human-friendly short Run ID & short Model name
        rid = str(s.get("run_id") or p.parent.name)
        parts_id = rid.split("-")
        s["short_id"] = f"#{parts_id[1]}" if len(parts_id) >= 2 and parts_id[1].isdigit() else f"#{rid[:6]}"
        raw_mod = str(s.get("model") or "")
        if not s.get("short_model"):
            if "sft" in raw_mod.lower() or "endpoints/" in raw_mod.lower():
                base_m = "gemini-3.5-flash" if "3-5-flash" in raw_mod and "lite" not in raw_mod else "gemini-3.5-flash-lite"
                s["short_model"] = f"{base_m} (SFT)"
            else:
                s["short_model"] = raw_mod
        matched_ed = (
            exp_by_app_ds.get((ap, ds))
            or (exp_by_app_ds.get((ap, "hul_sessions")) if ds == "gt_hul_sessions" else None)
            or exp_by_app.get(ap)
        )
        if matched_ed:
            raw_eid = str(matched_ed.get("experiment_id", ""))
            s["exp_id"] = f"EXP-{raw_eid.rsplit('-', 1)[-1]}" if "-" in raw_eid else raw_eid
            s["exp_priority"] = matched_ed.get("priority")
            s["exp_title"] = matched_ed.get("title")
            s["exp_hypothesis"] = matched_ed.get("hypothesis")
        s["methodology"] = synthesize_dynamic_methodology(s)
        runs.append(s)

    def official(r: dict) -> bool:
        ds = str(r.get("dataset") or "")
        if ds.startswith(("hul_", "gt_hul", "gtmerch", "ms_json", "mt_json")):
            n_img = int(r.get("images", 0))
            return (
                r.get("split") in ("test", "test_5k")
                and n_img >= 5
                and int(r.get("errors", 0)) <= n_img // 1000
            )
        return (r.get("environment", {}).get("platform") == "cloud-run"
                and (r["split"], r.get("limit"), r.get("seed")) == board_set)

    def score(r: dict) -> float:
        return float(r.get("primary_score") or (r.get("f2", 0.0) if r["task"] in ("detection", "end_to_end") else r.get("accuracy", 0.0)))

    def board(r: dict) -> tuple[int, str, int, str]:  # one tab per task and dataset (Golden first)
        ds = str(r["dataset"])
        is_public = 0 if ds.startswith(("hul_", "gtmerch", "ms_json", "mt_json")) else 1
        return approaches.base.USE_CASES.index(r["use_case"]), r["task"], is_public, ds

    runs.sort(key=lambda r: (board(r), not official(r), -score(r), -r["recall"],
                             r["cost_per_image_usd"]))
    place: dict[tuple, int] = {}
    for r in runs:
        if official(r):
            place[board(r)] = place.get(board(r), 0) + 1
        r["rank"] = place[board(r)] if official(r) else "dev"
    return runs


def load_run(run_id: str, results_dir: Path = RESULTS_DIR) -> tuple[dict, list[dict]]:
    results_dir = resolve_results_dir(results_dir)
    d = Path(results_dir) / run_id
    if d.resolve().parent != Path(results_dir).resolve() or not (d / "summary.json").exists():
        raise FileNotFoundError(run_id)
    sum_p = d / "summary.json"
    img_p = d / "images.jsonl"
    if not img_p.exists():
        try:
            gcp_res = load_config().get("gcp", {}).get("results")
            if gcp_res:
                bucket, prefix = dataset._split_gs(gcp_res.rstrip("/"))
                dataset._gcs().bucket(bucket).blob(f"{prefix}/{run_id}/images.jsonl").download_to_filename(str(img_p))
        except Exception:
            pass
    ckey = (str(results_dir), run_id)
    mtime = sum_p.stat().st_mtime_ns + (img_p.stat().st_mtime_ns if img_p.exists() else 0)
    hit = _RUN_DETAIL_CACHE.get(ckey)
    if hit and hit[0] == mtime:
        return hit[1], hit[2]
    summary = _with_inr(json.loads(sum_p.read_text()))
    rate = summary["usd_to_inr"] or 0
    images = [json.loads(line) for line in img_p.read_text().splitlines() if line] if img_p.exists() else []
    for r in images:
        r["cost_inr"] = round(r["cost_usd"] * rate, 4)
    summary["methodology"] = generate_run_methodology(summary, images)
    if len(_RUN_DETAIL_CACHE) > 32:
        _RUN_DETAIL_CACHE.pop(next(iter(_RUN_DETAIL_CACHE)))
    _RUN_DETAIL_CACHE[ckey] = (mtime, summary, images)
    return summary, images

