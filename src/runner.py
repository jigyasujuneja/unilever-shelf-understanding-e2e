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
from utils import dataset, metrics, pricing, telemetry
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
    fields = {"rpc": dataset.RPC_FIELDS, "shelves": dataset.SHELF_FIELDS, **dataset.HUL_FIELDS}.get(
        appr.dataset, dataset.PRODUCT_FIELDS)
    catalog = dataset.catalog(str(data_root), appr.dataset) if task != "detection" else {}
    samples = dataset.sample_images(split, limit, seed, str(data_root), appr.dataset)
    if not samples:
        raise RuntimeError(f"No images found for split={split!r} under {data_root}")

    started = datetime.now(timezone.utc)
    ds_suffix = f"-{appr.dataset}" if dataset_override else ""
    split_suffix = f"-{split}" if (dataset_override and split != "test") else ""
    run_id = (f"{started:%m%d-%H%M%S}-{approach}-{model}"
              + ("" if tier == "standard" else f"-{tier}") + ds_suffix + split_suffix)
    log(f"[{run_id}] {len(samples)} images from {appr.dataset} {split} (seed {seed}) on {env['platform']}")

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
                      otel_parent=telemetry.child_context(span), price=price)
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
                appr.dataset.startswith(("hul_", "gtmerch", "ms_json"))
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
            if task == "classification":  # the photo is the product
                ids = [appr.identify(image, ctx)]
                preds = list(sample.boxes)
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
            trace.step("Score vs ground truth",
                       f"{m['tp']} correct, {m['fp']} false, {m['fn']} missed "
                       f"({len(sample.boxes)} products in ground truth)")
        elif task == "end_to_end":  # right box (IoU >= 0.5) and right product
            gt_ids = [g["sku_id"] for g in sample.labels]
            m = metrics.match_identified(preds, ids, sample.boxes, gt_ids)
            box_m = metrics.match(preds, sample.boxes)
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
            extra["stage2_fast_gate_count"] = fast_gated
            extra["labels"] = [
                {"box": [round(v, 1) for v in b], "pred": catalog.get(i),
                 "gt": sample.labels[m["pairs"][k]] if k in m["pairs"] else None,
                 "correct": (
                     metrics.match_product(catalog.get(i), sample.labels[m["pairs"][k]], fields)["correct"]
                     if k in m["pairs"] else dict.fromkeys(fields, False)
                 )}
                for k, (b, i) in enumerate(zip(preds, ids, strict=True))]
            trace.step("Score vs ground truth",
                       f"{m['found']} of {len(sample.boxes)} products boxed, {m['tp']} of those "
                       f"identified correctly; {m['fp']} boxes wrong or extra, {m['fn']} products missed")
        else:  # classification / retrieval: one answer per ground-truth box
            if not ids:  # the image failed: every product counts as unanswered
                ids = [None] * len(sample.boxes)
            m = {"tp": 0, "fp": 0, "fn": 0, "matched": []}
            extra["labels"] = []
            for k, (b, gt, i) in enumerate(zip(sample.boxes, sample.labels, ids, strict=True)):
                pred = catalog.get(i) if i is not None else None
                mk = metrics.match_product(pred, gt, fields)
                for key in ("tp", "fp", "fn"):
                    m[key] += mk[key]
                if mk["tp"]:
                    m["matched"].append(k)
                extra["labels"].append({"box": [round(v, 1) for v in b], "pred": pred, "gt": gt,
                                        "correct": mk["correct"]})
            trace.step("Score vs ground truth",
                       f"{m['tp']} of {len(sample.boxes)} products identified correctly"
                       if len(sample.boxes) > 1 else
                       f"predicted {(extra['labels'][0]['pred'] or {}).get('product', 'nothing')}; "
                       f"truth {sample.labels[0]['product']} "
                       f"({', '.join(f + (' ✓' if ok else ' ✗') for f, ok in extra['labels'][0]['correct'].items())})")
        g = price(trace.usage)
        s_usd = pricing.extra_cost(trace.billed, sheet)
        return {
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
            "preds": [[round(v, 1) for v in b] for b in preds],
            "matched": m["matched"],
            "steps": trace.steps,
            **extra,
        }

    try:
        rows: list[dict] = []
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for i, row in enumerate(pool.map(one, samples), 1):
                rows.append(row)
                flag = f"  ERROR {row['error']}" if row["error"] else ""
                what = (f"{'✓' if row['tp'] else '✗'} {(row['labels'][0]['pred'] or {}).get('product', '-')}"
                        if task == "classification" else
                        f"{row['tp']:>3} / {row['gt_count']:>3} identified"
                        if task in ("retrieval", "promo_detection", "planogram_compliance") else
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
            if boxes:
                summary["field_accuracy"] = {
                    f: round(sum(bool(lab["correct"].get(f)) for lab in boxes) / max(1, len(boxes)), 4)
                    for f in fields}
                if task in ("classification", "retrieval", "promo_detection", "planogram_compliance"):
                    summary["accuracy"] = summary["field_accuracy"]["product"]
                summary["hierarchical_accuracy"] = metrics.hierarchical_field_scores(boxes)
                # Macro-averaged accuracy across GT brands and GT variants + HUL vs Non-HUL competitor breakdown
                from collections import defaultdict

                by_brand_ok: dict[str, list[int]] = defaultdict(list)
                by_var_ok: dict[str, list[int]] = defaultdict(list)
                hul_boxes = []
                comp_boxes = []
                for lab in boxes:
                    gt_d = lab.get("gt") or {}
                    corr_d = lab.get("correct") or {}
                    b_name = str(gt_d.get("brand") or "")
                    v_name = str(gt_d.get("base_variant") or gt_d.get("product") or "")
                    if b_name:
                        by_brand_ok[b_name].append(int(bool(corr_d.get("brand"))))
                    if v_name:
                        by_var_ok[v_name].append(int(bool(corr_d.get("base_variant", corr_d.get("product")))))
                    if "is_hul" in gt_d:
                        if gt_d.get("is_hul"):
                            hul_boxes.append(lab)
                        else:
                            comp_boxes.append(lab)
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

                all_eval_boxes = [lab for r in rows for lab in r.get("labels", [])]
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
            # Aggregate Merchandising Planogram & OOS Void Metrics across images
            plano_steps = [
                st["info"]
                for r in rows
                for st in r.get("steps", [])
                if isinstance(st.get("info"), dict) and "brand_block_purity" in st["info"]
            ]
            if plano_steps:
                n_pl = len(plano_steps)
                summary["planogram_metrics"] = {
                    "mean_brand_block_purity": round(
                        sum(float(p.get("brand_block_purity", 0.0)) for p in plano_steps) / n_pl, 4
                    ),
                    "mean_asset_bay_purity": round(
                        sum(float(p.get("asset_bay_purity", 1.0)) for p in plano_steps) / n_pl, 4
                    ),
                    "total_competitor_intrusions": sum(
                        int(p.get("competitor_intrusion_count", 0)) for p in plano_steps
                    ),
                    "total_oos_voids": sum(int(p.get("oos_void_count", 0)) for p in plano_steps),
                    "estimated_missing_facings": sum(
                        int(p.get("estimated_missing_facings", 0)) for p in plano_steps
                    ),
                    "mean_empty_shelf_share": round(
                        sum(float(p.get("empty_shelf_share", 0.0)) for p in plano_steps) / n_pl, 4
                    ),
                }
            # Stage-2A / Stage-2B Retrieval & Verification Diagnostics (Recall@1, Recall@5, MRR, 0-VLM Gate %, VLM Net Gain)
            shortlist_steps: list[dict] = []
            fast_gate_steps = 0
            vlm_escalated_steps = 0
            for r in rows:
                r_labels = r.get("labels", [])
                box_cursor = 0
                for st in r.get("steps", []):
                    st_name = str(st.get("name") or "")
                    st_info = st.get("info") or {}
                    slist = st_info.get("shortlist")
                    if isinstance(slist, list) and slist:
                        if "Embedding" in st_name or "CLM" in st_name:
                            fast_gate_steps += 1
                        elif "Gemini" in st_name or "CLEF" in st_name or "JEV" in st_name:
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
                })
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
            "model": model,
            "tier": tier,
            "traffic": usage.traffic,  # calls per traffic type Vertex actually served
            # Share of calls Vertex actually served at priority / flex (the rest were downgraded).
            "priority_served": round(usage.traffic.get("priority", 0) / calls, 3)
            if calls and tier == "priority" else None,
            "flex_served": round(usage.traffic.get("flex", 0) / calls, 3)
            if calls and tier == "flex" else None,
            "architecture": f"{appr.architecture} [{' + '.join([model, *appr.also_calls])}"
                            + ("" if tier == "standard" else f", {tier}") + "]",
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


def pull(src: str, results_dir: Path = RESULTS_DIR, log: Callable[[str], None] = print) -> int:
    """Copy finished runs from GCS into ``results/`` and finalize their compute cost.

    A Cloud Run task can't know its own billed duration, so it stores a provisional compute
    cost. Here we look up the task's start/completion time in the Cloud Run Admin API and
    re-price compute from it, then write the final summary back to GCS too.
    """
    bucket, prefix = dataset._split_gs(src.rstrip("/"))
    gcs = dataset._gcs().bucket(bucket)
    n = 0
    for blob in dataset._gcs().list_blobs(bucket, prefix=prefix + "/"):
        if blob.name in (f"{prefix}/qc_audit_report.json", f"{prefix}/experiments_registry.json"):
            meta_name = blob.name.rsplit("/", 1)[-1]
            meta_dest = Path(results_dir) / meta_name
            meta_dest.parent.mkdir(parents=True, exist_ok=True)
            if not meta_dest.exists():
                blob.download_to_filename(str(meta_dest))
            continue
        if not blob.name.endswith("/summary.json"):
            continue
        run_id = blob.name.split("/")[-2]
        dest = Path(results_dir) / run_id
        local_sum = dest / "summary.json"
        need_dl = not local_sum.exists() or (blob.size is not None and local_sum.stat().st_size != blob.size)
        if need_dl:
            img_blob = gcs.blob(f"{prefix}/{run_id}/images.jsonl")
            if not img_blob.exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            img_blob.download_to_filename(str(dest / "images.jsonl"))
            blob.download_to_filename(str(local_sum))
            n += 1
        summary = json.loads(local_sum.read_text())
        if "provisional" in summary.get("cost", {}).get("compute_source", ""):
            from utils import cloud

            try:
                seconds = cloud.task_seconds(summary["environment"])
            except Exception as e:  # task not finished yet / API error: keep provisional
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
            (dest / "summary.json").write_text(json.dumps(summary, indent=2))
            gcs.blob(blob.name).upload_from_filename(str(dest / "summary.json"))
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


def summarize(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    lat = [r["latency_s"] for r in rows]
    usage = Usage()
    for r in rows:
        usage = usage + Usage(**r["usage"])
    return {
        "images": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "tp": tp, "fp": fp, "fn": fn,
        **{k: round(v, 4) for k, v in metrics.scores(tp, fp, fn).items()},
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
    global _TMP_RESULTS_HYDRATED
    p = Path(results_dir)
    if p != RESULTS_DIR and p != Path("results"):
        return p
    if p.is_dir() and any(p.glob("*/summary.json")):
        return p
    if not _TMP_RESULTS_HYDRATED:
        _TMP_RESULTS_HYDRATED = True
        _TMP_RESULTS_CACHE.mkdir(parents=True, exist_ok=True)
        try:
            gcp_res = load_config().get("gcp", {}).get("results")
            if gcp_res:
                pull(gcp_res, results_dir=_TMP_RESULTS_CACHE, log=lambda *_: None)
        except Exception:
            pass
    return _TMP_RESULTS_CACHE if _TMP_RESULTS_CACHE.is_dir() else p


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
            s = _with_inr(json.loads(p.read_text()))
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
        is_golden = ds.startswith(("hul_", "gtmerch", "ms_json"))
        is_shelf = ds in ("gtmerch_golden", "ms_json_golden", "hul_sessions", "shelves", "sku110k")
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
        s["approach_title"] = ap_meta.get("display_name", ap)
        s["approach_intuition"] = ap_meta.get("intuition", s.get("architecture", ""))
        s["primary_score"] = float(s.get("f2", 0.0) if tk in ("detection", "end_to_end") else s.get("accuracy", 0.0))
        # Human-friendly short Run ID & short Model name
        rid = str(s.get("run_id") or p.parent.name)
        parts_id = rid.split("-")
        s["short_id"] = f"#{parts_id[1]}" if len(parts_id) >= 2 and parts_id[1].isdigit() else f"#{rid[:6]}"
        raw_mod = str(s.get("model") or "")
        if "sft" in raw_mod.lower() or "endpoints/" in raw_mod.lower():
            base_m = "gemini-3.5-flash" if "3-5-flash" in raw_mod and "lite" not in raw_mod else "gemini-3.5-flash-lite"
            s["short_model"] = f"{base_m} (SFT)"
        else:
            s["short_model"] = raw_mod
        matched_ed = exp_by_app_ds.get((ap, ds)) or exp_by_app.get(ap)
        if matched_ed:
            raw_eid = str(matched_ed.get("experiment_id", ""))
            s["exp_id"] = f"EXP-{raw_eid.rsplit('-', 1)[-1]}" if "-" in raw_eid else raw_eid
            s["exp_priority"] = matched_ed.get("priority")
            s["exp_title"] = matched_ed.get("title")
            s["exp_hypothesis"] = matched_ed.get("hypothesis")
        runs.append(s)

    def official(r: dict) -> bool:
        ds = str(r.get("dataset") or "")
        if ds.startswith(("hul_", "gtmerch", "ms_json")):
            return r.get("split") == "test" and int(r.get("images", 0)) >= 5 and int(r.get("errors", 0)) == 0
        return (r.get("environment", {}).get("platform") == "cloud-run"
                and (r["split"], r.get("limit"), r.get("seed")) == board_set)

    def score(r: dict) -> float:
        return float(r.get("f2", 0.0) if r["task"] in ("detection", "end_to_end") else r.get("accuracy", 0.0))

    def board(r: dict) -> tuple[int, str, int, str]:  # one tab per task and dataset (Golden first)
        ds = str(r["dataset"])
        is_public = 0 if ds.startswith(("hul_", "gtmerch", "ms_json")) else 1
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
    summary = _with_inr(json.loads((d / "summary.json").read_text()))
    rate = summary["usd_to_inr"] or 0
    images = [json.loads(line) for line in (d / "images.jsonl").read_text().splitlines() if line]
    for r in images:
        r["cost_inr"] = round(r["cost_usd"] * rate, 4)
    return summary, images

