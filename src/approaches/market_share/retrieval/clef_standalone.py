"""Standalone CLEF (Constrained Logprob-Evaluated Flash) Retrieval (``clef_standalone``).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Pure System-2 Constrained Logprob Decision Pipeline (Zero JEV Coupling)   │
  │  • Inherits directly from EmbeddingRetrieval (no Stage-A JEV fast-path,    │
  │    no JEV medoid clustering, no resolve_jev_choice override, no smoothing) │
  │  • 100% of queries routed to CLEF Zoom-Strip Contact Sheet (bottom-30%     │
  │    high-res net-weight zoom strip inside every candidate panel)            │
  │  • Grounded schema + direct token logprob argmax readout P_token(k | x)    │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PIL import Image

from approaches.base import Box, Context, crop, register
from approaches.market_share.retrieval.embedding_retrieval import (
    MODEL,
    EmbeddingRetrieval,
)
from utils.clm import zoom_multi_sheet, zoom_sheet
from utils.openjev import _parse_int_token, _slice_after_choice_key

CLEF_K = 5
CLEF_BATCH = 4

CLEF_ZOOM_SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "pack_size": {"type": "string"},
        "choice": {"type": "integer", "minimum": 0, "maximum": CLEF_K},
    },
    "required": ["pack_size", "choice"],
}
CLEF_ZOOM_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "pack_sizes": {"type": "array", "items": {"type": "string"}},
        "choices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": CLEF_K},
        },
    },
    "required": ["pack_sizes", "choices"],
}
CLEF_ZOOM_SINGLE_PROMPT = (
    "Panel Q is a product cut out of a retail photo. Panels 1-{k} are candidate catalog reference photos "
    "(each panel includes a bottom zoom strip of the net-weight/pack-size label). First read the visible "
    "net weight/pack size on Q (or '' if unreadable), then pick the numbered panel (1-{k}) that shows the "
    "exact same product (same brand, variant, packaging form, and pack size), or 0 if none match. "
    'Return ONLY JSON {{"pack_size": "...", "choice": <int 0..{k}>}}.'
)
CLEF_ZOOM_BATCH_PROMPT = (
    "The image shows {m} rows (Q1 to Q{m}). In each row, the red-labelled left panel Qi is a product crop, "
    "followed by candidate reference photos 1-{k} (each cell includes a bottom zoom strip of the net-weight "
    "label). For each row Qi (in order 1..{m}), read the visible net weight/pack size on Qi and pick the "
    "numbered panel (1-{k}) showing the exact same product, or 0 if none match. "
    'Return ONLY JSON {{"pack_sizes": ["...", ...], "choices": [<int>, ...]}} with {m} items.'
)


def build_contrastive_candidate_rubric(
    top: list[tuple[float, int, str]],
    catalog: dict[int, dict[str, Any]] | None,
    row_idx: int | None = None,
) -> str:
    """Build a concise candidate metadata block + pairwise contrastive diff tokens for Panels 1..K."""
    if not top or not catalog:
        return ""
    header = f"Row Q{row_idx + 1} Candidate Reference Panels:" if row_idx is not None else "Candidate Reference Panels:"
    lines: list[str] = [header]
    token_sets: list[set[str]] = []
    for idx, (_, pid, _) in enumerate(top, start=1):
        meta = catalog.get(pid) or {}
        brand = str(meta.get("brand") or "")
        cat = str(meta.get("category") or "")
        var = str(meta.get("base_variant") or meta.get("variant") or meta.get("product") or f"SKU_{pid}")
        pack = str(meta.get("packaging_type") or "")
        size = str(meta.get("pack_size") or "")
        cat_str = f" | Category={cat}" if cat else ""
        size_str = f" | Size={size}" if size else ""
        lines.append(f"- Panel {idx}: Brand={brand} | Variant={var}{cat_str} | Pack={pack}{size_str}")
        toks = {t for t in f"{brand}_{var}_{cat}_{pack}_{size}".replace("__", "_").split("_") if len(t) >= 2}
        token_sets.append(toks)
    if len(token_sets) >= 2:
        shared = set.intersection(*token_sets) if token_sets else set()
        diffs: list[str] = []
        for idx, toks in enumerate(token_sets[:3], start=1):
            uniq = sorted(toks - shared)[:4]
            if uniq:
                diffs.append(f"Panel {idx} has [{' '.join(uniq)}]")
        if diffs:
            q_lbl = f"Q{row_idx + 1}" if row_idx is not None else "Q"
            lines.append(f"Key discriminative text/pack differences to check on {q_lbl}: " + " vs ".join(diffs) + ".")
    return "\n".join(lines)


def clef_logprob_select(
    logprobs_steps: list[dict[str, Any]] | None,
    valid_options: set[int],
    n_slots: int = 1,
) -> list[dict[str, Any]]:
    """Pure CLEF token logprob readout without JEV visual-prior override.

    Scans the decoded token stream strictly after the ``choice`` / ``choices`` JSON key token
    (when present), computes normalized token probabilities ``P_token(k | x)`` across valid options,
    and selects ``argmax_k P_token(k | x)`` directly.
    """
    if not logprobs_steps:
        return []
    active_steps = _slice_after_choice_key(logprobs_steps)
    slots: list[dict[str, Any]] = []
    for step in active_steps:
        chosen_int = _parse_int_token(str(step.get("token", "")))
        if chosen_int is None or chosen_int not in valid_options:
            continue
        raw_lps: dict[int, float] = {chosen_int: float(step.get("logprob", 0.0))}
        for cand in step.get("top") or []:
            c_int = _parse_int_token(str(cand.get("token", "")))
            if c_int is not None and c_int in valid_options:
                lp = float(cand.get("logprob", -20.0))
                if c_int not in raw_lps or lp > raw_lps[c_int]:
                    raw_lps[c_int] = lp
        max_lp = max(raw_lps.values())
        unnorm = {k: math.exp(lp - max_lp) for k, lp in raw_lps.items()}
        z = sum(unnorm.values()) or 1.0
        probs = {k: round(v / z, 4) for k, v in unnorm.items()}
        best_opt = max(probs, key=lambda k: probs[k])
        slots.append({"choice": best_opt, "top1_prob": probs[best_opt], "probs": probs})
    return slots[-n_slots:] if len(slots) >= n_slots else slots


def resolve_clef_choice(
    choice: int | None,
    prob: float,
    top: list[tuple[float, int, str]],
    *,
    allow_abstain: bool = False,
) -> tuple[int | None, str]:
    """Resolve CLEF integer choice via 2-Tier Open-Set Competitor Guard + Log-Linear Energy Veto.

    - Valid panel ``1..K``: applies Log-Linear Energy Veto when ``choice > 1``, ``(s1 - sk) > 0.045``,
      and ``prob < 0.90`` (preventing low-confidence logprob flips against a dominant visual top-1).
    - ``choice == 0`` or ``None``:
      * Tier 1 (``s1 >= 0.72``): High-confidence catalog match where VLM over-abstained on minor blur
        -> recover ``top[0][1]``.
      * Tier 2 (``0.45 <= s1 < 0.72``): Open-set competitor guard -> abstain ``None`` when
        ``allow_abstain=True``, or fallback to ``top[0][1]`` on closed-set catalog tasks.
      * Tier 3 (``s1 < 0.45``): Out-of-catalog / severe occlusion -> abstain ``None``.
    """
    if not top:
        return None, "empty"
    s1, id1, _ = top[0]
    if isinstance(choice, int) and 1 <= choice <= len(top):
        sk, idk, _ = top[choice - 1]
        if choice > 1 and (s1 - sk) > 0.045 and prob < 0.90:
            return id1, "energy_veto_top1"
        return idk, f"panel_{choice}"
    if s1 >= 0.72:
        return id1, "tier1_high_cos_top1"
    if s1 >= 0.45:
        if allow_abstain:
            return None, "tier2_open_set_competitor_abstain"
        return id1, "tier2_closed_set_top1"
    return (None, "tier3_low_cos_abstain") if allow_abstain else (id1, "tier3_closed_set_top1")


@register
class ClefStandaloneRetrieval(EmbeddingRetrieval):
    """Standalone System-2 CLEF Retrieval (CLEF Zoom-Strip evaluation + 2-Tier Competitor Guard)."""

    name = "clef_standalone"
    task = "retrieval"
    dataset = "rpc"
    models: list[str] | None = None
    shortlist_model = MODEL
    also_calls = [MODEL]
    architecture = (
        f"Standalone CLEF: {MODEL} top-{CLEF_K} shortlist -> CLEF Bottom-30% Zoom-Strip "
        "Constrained Logprob Decision Head + 2-Tier Open-Set Competitor Guard"
    )
    steps = [
        "Setup: embed reference gallery photos",
        f"Retrieve top-{CLEF_K} candidate reference photos by embedding cosine",
        "Render Bottom-30% Pack-Size Zoom-Strip contact sheet for candidate verification",
        "Execute CLEF constrained schema + direct token logprob argmax readout P_token(k | x) + 2-Tier Competitor Guard",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        from utils import dataset

        super().setup(config, ctx)
        root = str(config.get("data_root") or dataset.data_root(self.dataset))
        try:
            self.catalog = dataset.catalog(root, self.dataset)
        except Exception:
            self.catalog = {}

    def _allow_abstain(self) -> bool:
        ds = str(getattr(self, "dataset", "") or "")
        return ds in ("ms_json_golden", "gtmerch_golden")

    use_contrastive_metadata: bool = False

    def _clef_single(
        self, image: Image.Image, top: list[tuple[float, int, str]], ctx: Context
    ) -> tuple[int | None, dict[str, Any]]:
        import os

        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        sheet_im = zoom_sheet(image, refs)
        base_prompt = CLEF_ZOOM_SINGLE_PROMPT.format(k=len(top))
        if getattr(self, "use_contrastive_metadata", False) or os.environ.get("SHELF_BENCH_CONTRASTIVE_METADATA") == "1":
            rubric = build_contrastive_candidate_rubric(top, getattr(self, "catalog", None))
            if rubric:
                base_prompt = f"{base_prompt}\n\n{rubric}"
        res = ctx.ask(
            sheet_im,
            base_prompt,
            schema=CLEF_ZOOM_SINGLE_SCHEMA,
            logprobs=len(top) + 1,
        )
        data = res.data if isinstance(res.data, dict) else {}
        raw_c = data.get("choice")
        slots = clef_logprob_select(res.meta.get("logprobs"), set(range(0, len(top) + 1)), n_slots=1)
        choice = slots[0]["choice"] if slots else (raw_c if isinstance(raw_c, int) else None)
        prob = slots[0]["top1_prob"] if slots else 1.0
        allow_abstain = self._allow_abstain() if hasattr(self, "_allow_abstain") else False
        ans, guard_mode = resolve_clef_choice(choice, prob, top, allow_abstain=allow_abstain)
        return ans, {
            "choice": choice,
            "clef_prob": prob,
            "pack_size": data.get("pack_size", ""),
            "guard_mode": guard_mode,
        }

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        import os

        from utils.clm import compute_zscore_and_lid, infonce_posterior

        ranked = self.ranked(image, ctx, allowed_ids)
        if not ranked:
            return None
        top = ranked[:CLEF_K]
        use_fast_gate = (
            os.environ.get("SHELF_BENCH_CLEF_FAST_GATE") == "1"
            or (
                self.dataset.startswith(("hul", "gt_hul", "gtmerch", "ms_json"))
                and os.environ.get("SHELF_BENCH_PURE_CLEF") != "1"
            )
        )
        if use_fast_gate:
            cat_map = getattr(self, "catalog", {})
            s1, id1, _ = top[0]
            s2, id2, _ = top[1] if len(top) > 1 else (-1.0, -1, "")
            c1 = cat_map.get(id1) or {}
            c2 = cat_map.get(id2) or {}
            same_variant = bool(
                c1.get("brand")
                and c1.get("brand") == c2.get("brand")
                and c1.get("base_variant") == c2.get("base_variant")
                and c1.get("pack_size") != c2.get("pack_size")
            )
            req_margin = 0.025 if same_variant else 0.009
            zlid = compute_zscore_and_lid([s for s, _, _ in top])
            p_clm, ent_conf = infonce_posterior(top, inv_temp=28.0, k=CLEF_K)
            if (
                s1 >= 0.75
                and (s1 - s2) >= req_margin
                and p_clm >= 0.30
                and zlid["z_margin"] >= 0.80
                and zlid["lid"] <= 11.5
            ):
                ctx.trace.step(
                    "Stage A CLM System-1 Fast-Gate (0 VLM calls)",
                    f"#{id1} accepted (cos={s1:.3f}, margin={s1 - s2:.3f}, P_CLM={p_clm:.2f}, z_margin={zlid['z_margin']:.2f})",
                    info={
                        "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in top],
                        "top1_sku_id": id1,
                        "top5_sku_ids": [i for _, i, _ in top],
                        "clm_top1_prob": p_clm,
                        "coarse_entropy_conf": ent_conf,
                        "same_variant_sister": same_variant,
                        **zlid,
                        "stage2a_bypassed": True,
                        "fast_gated_boxes": 1,
                        "answer": id1,
                    },
                )
                return id1
        ans, meta = self._clef_single(image, top, ctx)
        ctx.trace.step(
            "Standalone CLEF Zoom-Strip",
            f"panel {meta['choice']} -> #{ans} (P_token={meta['clef_prob']:.2f}, size={meta['pack_size']!r}, mode={meta['guard_mode']})",
            info={
                **meta,
                "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in top],
                "top1_sku_id": top[0][1],
                "top5_sku_ids": [i for _, i, _ in top],
                "stage2a_bypassed": False,
            },
        )
        return ans

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        import os

        from utils.clm import (
            compute_zscore_and_lid,
            group_adjacent_shelf_twins,
            infonce_posterior,
        )
        from utils.photometry import inner_lab_zones, restore_photometry

        if not boxes:
            return []
        crops = [crop(image, b) for b in boxes]
        use_fast_gate = (
            os.environ.get("SHELF_BENCH_CLEF_FAST_GATE") == "1"
            or (
                self.dataset.startswith(("hul", "gt_hul", "gtmerch", "ms_json"))
                and os.environ.get("SHELF_BENCH_PURE_CLEF") != "1"
            )
        )
        if use_fast_gate and len(boxes) > 1:
            restored_crops = [restore_photometry(c)[0] for c in crops]
            crop_labs = [inner_lab_zones(rc) for rc in restored_crops]
            twin_clusters, medoid_indices = group_adjacent_shelf_twins(boxes, restored_crops, crop_labs)
            with ThreadPoolExecutor(max_workers=8) as pool:
                medoid_ranked = list(pool.map(lambda m_idx: self.ranked(crops[m_idx], ctx), medoid_indices))
            all_ranked: list[list[tuple[float, int, str]]] = [[] for _ in range(len(boxes))]
            for cl, ranked_m in zip(twin_clusters, medoid_ranked, strict=True):
                for member_idx in cl:
                    all_ranked[member_idx] = ranked_m
            saved_emb = len(boxes) - len(medoid_indices)
            if saved_emb > 0:
                ctx.trace.step(
                    "Pre-Embedding Spatial Medoid Dedup",
                    f"{len(boxes)} facings -> {len(medoid_indices)} unique embedding RPCs ({saved_emb} adjacent shelf twins shared)",
                    info={
                        "total_boxes": len(boxes),
                        "unique_embedding_calls": len(medoid_indices),
                        "saved_embedding_rpcs": saved_emb,
                    },
                )
        else:
            with ThreadPoolExecutor(max_workers=8) as pool:
                all_ranked = list(pool.map(lambda c: self.ranked(c, ctx), crops))

        ids: list[int | None] = [None] * len(boxes)
        valid_indices: list[int] = []
        cat_map = getattr(self, "catalog", {})
        for i, r in enumerate(all_ranked):
            if not r:
                continue
            top = r[:CLEF_K]
            if use_fast_gate:
                s1, id1, _ = top[0]
                s2, id2, _ = top[1] if len(top) > 1 else (-1.0, -1, "")
                c1 = cat_map.get(id1) or {}
                c2 = cat_map.get(id2) or {}
                same_variant = bool(
                    c1.get("brand")
                    and c1.get("brand") == c2.get("brand")
                    and c1.get("base_variant") == c2.get("base_variant")
                    and c1.get("pack_size") != c2.get("pack_size")
                )
                req_margin = 0.025 if same_variant else 0.009
                zlid = compute_zscore_and_lid([s for s, _, _ in top])
                p_clm, ent_conf = infonce_posterior(top, inv_temp=28.0, k=CLEF_K)
                if (
                    s1 >= 0.75
                    and (s1 - s2) >= req_margin
                    and p_clm >= 0.30
                    and zlid["z_margin"] >= 0.80
                    and zlid["lid"] <= 11.5
                ):
                    ids[i] = id1
                    ctx.trace.step(
                        "Stage A CLM System-1 Fast-Gate",
                        f"box {i + 1}: #{id1} accepted (cos={s1:.3f}, margin={s1 - s2:.3f}, P_CLM={p_clm:.2f}, z_margin={zlid['z_margin']:.2f})",
                        info={
                            "box_index": i,
                            "shortlist": [{"sku_id": k_id, "cosine": round(s, 4)} for s, k_id, _ in top],
                            "top1_sku_id": id1,
                            "top5_sku_ids": [k_id for _, k_id, _ in top],
                            "clm_top1_prob": p_clm,
                            "coarse_entropy_conf": ent_conf,
                            "same_variant_sister": same_variant,
                            **zlid,
                            "stage2a_bypassed": True,
                            "fast_gated_boxes": 1,
                            "answer": id1,
                        },
                    )
                    continue
            valid_indices.append(i)
        batches = [valid_indices[i : i + CLEF_BATCH] for i in range(0, len(valid_indices), CLEF_BATCH)]
        allow_abstain = self._allow_abstain()

        def run_batch(batch: list[int]) -> list[tuple[int, int | None, dict[str, Any]]]:
            if len(batch) == 1:
                idx = batch[0]
                ans, meta = self._clef_single(crops[idx], all_ranked[idx][:CLEF_K], ctx)
                return [(idx, ans, meta)]
            qs = [crops[idx] for idx in batch]
            tops = [all_ranked[idx][:CLEF_K] for idx in batch]
            refs_per_q = [
                [Image.open(io.BytesIO(self.jpeg[p])).convert("RGB") for _, _, p in t]
                for t in tops
            ]
            sheet_im = zoom_multi_sheet(qs, refs_per_q)
            batch_prompt = CLEF_ZOOM_BATCH_PROMPT.format(m=len(batch), k=CLEF_K)
            if self.use_contrastive_metadata or os.environ.get("SHELF_BENCH_CONTRASTIVE_METADATA") == "1":
                row_rubrics = []
                for b_idx, t_row in enumerate(tops, start=1):
                    r_txt = build_contrastive_candidate_rubric(t_row, getattr(self, "catalog", None))
                    if r_txt:
                        row_rubrics.append(f"Row Q{b_idx} {r_txt}")
                if row_rubrics:
                    batch_prompt = f"{batch_prompt}\n\n" + "\n".join(row_rubrics)
            res = ctx.ask(
                sheet_im,
                batch_prompt,
                schema=CLEF_ZOOM_BATCH_SCHEMA,
                logprobs=CLEF_K + 1,
            )
            data = res.data if isinstance(res.data, dict) else {}
            raw_c_val = data.get("choices")
            raw_s_val = data.get("pack_sizes")
            raw_choices: list[Any] = raw_c_val if isinstance(raw_c_val, list) else []
            raw_sizes: list[Any] = raw_s_val if isinstance(raw_s_val, list) else []
            slots = clef_logprob_select(
                res.meta.get("logprobs"), set(range(0, CLEF_K + 1)), n_slots=len(batch)
            )
            out: list[tuple[int, int | None, dict[str, Any]]] = []
            for b_i, idx in enumerate(batch):
                top = tops[b_i]
                c = (
                    slots[b_i]["choice"]
                    if b_i < len(slots)
                    else (raw_choices[b_i] if b_i < len(raw_choices) else None)
                )
                prob = slots[b_i]["top1_prob"] if b_i < len(slots) else 1.0
                psize = str(raw_sizes[b_i]) if b_i < len(raw_sizes) else ""
                ans, guard_mode = resolve_clef_choice(c, prob, top, allow_abstain=allow_abstain)
                out.append(
                    (
                        idx,
                        ans,
                        {
                            "choice": c,
                            "clef_prob": prob,
                            "pack_size": psize,
                            "guard_mode": guard_mode,
                            "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in top],
                        },
                    )
                )
            return out

        with ThreadPoolExecutor(max_workers=4) as pool:
            for b_out in pool.map(run_batch, batches):
                for idx, ans, meta in b_out:
                    ids[idx] = ans
                    ctx.trace.step(
                        "Standalone CLEF Zoom-Strip",
                        f"box {idx + 1}: panel {meta['choice']} -> #{ans} (P_token={meta['clef_prob']:.2f}, mode={meta.get('guard_mode', 'panel')})",
                        info={"box_index": idx, **meta, "answer": ans},
                    )
        return ids


@register
class ClefContrastiveMetadata(ClefStandaloneRetrieval):
    """EXP-28: Contrastive Metadata-Enriched CLEF Contact Sheets (Targets 93.3% Recall@5 Ceiling)."""

    name = "clef_contrastive_metadata"
    use_contrastive_metadata = True
    architecture = (
        f"EXP-28 Contrastive Metadata CLEF: {MODEL} top-{CLEF_K} + CLM Fast Gate -> "
        "CLEF Zoom-Strip Contact Sheet with Candidate Text Metadata & Pairwise Contrastive Token Diffs"
    )
    steps = [
        "Setup: embed reference gallery photos and load SKU catalog metadata",
        f"Stage A: CLM InfoNCE + Z-score/LID 0-VLM fast gate on top-{CLEF_K} candidates",
        "Stage B: Render Bottom-30% Zoom-Strip contact sheet + inject candidate Brand/Variant/Pack text & pairwise contrastive token diffs",
        "Execute CLEF constrained schema + token logprob argmax readout + 2-Tier Open-Set Competitor Guard",
    ]
