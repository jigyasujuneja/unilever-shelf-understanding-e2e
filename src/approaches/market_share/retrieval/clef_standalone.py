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
    "net weight/pack size or distinctive variant/pack keyword on Q (or '' if unreadable), then pick the numbered panel (1-{k}) that shows the "
    "exact same product (same brand, variant, packaging form, and pack size), or 0 if none match. "
    'Return ONLY JSON {{"pack_size": "...", "choice": <int 0..{k}>}}.'
)
CLEF_ZOOM_BATCH_PROMPT = (
    "The image shows {m} rows (Q1 to Q{m}). In each row, the red-labelled left panel Qi is a product crop, "
    "followed by candidate reference photos 1-{k} (each cell includes a bottom zoom strip of the net-weight "
    "label). For each row Qi (in order 1..{m}), read the visible net weight/pack size or distinctive variant/pack keyword on Qi and pick the "
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


_PACK_SIZE_TOKEN_RE = None


def normalize_pack_size_token(raw: str | None) -> str:
    """Normalize visible or catalog pack-size text (e.g. '100 gm', '0.5 kg', '250ML') into a canonical token."""
    import re

    global _PACK_SIZE_TOKEN_RE
    if not raw:
        return ""
    if _PACK_SIZE_TOKEN_RE is None:
        _PACK_SIZE_TOKEN_RE = re.compile(
            r"(?<!\d)(\d+(?:\.\d+)?)\s*(ml|ltrs?|litres?|liters?|l|gms?|grams?|g|kg|oz|pcs|pieces?|sachets?|tabs?|cubes?)\b",
            re.I,
        )
    cleaned = str(raw).replace("__", " ").replace("_", " ")
    m = _PACK_SIZE_TOKEN_RE.search(cleaned)
    if not m:
        return ""
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("gm", "gms", "gram", "grams", "g"):
        unit = "g"
    elif unit == "kg":
        if val <= 10.0:
            val *= 1000.0
            unit = "g"
    elif unit in ("ltr", "ltrs", "litre", "litres", "liter", "liters", "l"):
        if val <= 10.0:
            val *= 1000.0
            unit = "ml"
        else:
            unit = "l"
    elif unit in ("piece", "pieces", "pcs"):
        unit = "pcs"
    elif unit in ("sachet", "sachets"):
        unit = "sachet"
    elif unit in ("tab", "tabs"):
        unit = "tabs"
    elif unit in ("cube", "cubes"):
        unit = "cubes"
    num_str = str(int(round(val))) if abs(val - round(val)) < 1e-4 else f"{val:.1f}".rstrip("0").rstrip(".")
    return f"{num_str}{unit}"


def _catalog_sister_tokens(meta: dict[str, Any]) -> set[str]:
    import re

    raw = "_".join(
        str(meta.get(k) or "")
        for k in ("product", "base_variant", "variant", "packaging_type", "pack_size", "size")
    )
    return {t.lower() for t in re.split(r"[^a-zA-Z0-9]+", raw) if len(t) >= 4}


def verify_pack_size_sister_sku(
    chosen_id: int | None,
    extracted_pack_size: str | None,
    ranked_candidates: list[tuple[float, int]] | list[tuple[float, int, str]],
    catalog: dict[int, dict[str, Any]] | None,
    *,
    max_cos_drop: float = 0.08,
) -> tuple[int | None, bool]:
    """Layer-5 deterministic post-generation verifier for sister-SKU pack-size & variant disambiguation.

    When the VLM extracts a visible ``pack_size`` token (e.g. ``'100g'``) that conflicts with the
    chosen candidate's catalog size (e.g. ``'50g'``), checks the shortlist for a same-brand,
    same-variant sister SKU whose catalog size matches ``extracted_pack_size`` within ``max_cos_drop``.
    When no numeric size is present in the catalog, checks whether ``extracted_pack_size`` contains a
    unique discriminative sister-SKU variant/packaging keyword among same-brand shortlist candidates.
    """
    import re

    if chosen_id is None or not extracted_pack_size or not catalog or not ranked_candidates:
        return chosen_id, False
    chosen_meta = catalog.get(chosen_id) or {}
    chosen_brand = str(chosen_meta.get("brand") or "").strip().lower()
    if not chosen_brand:
        return chosen_id, False
    pairs = [(float(item[0]), int(item[1])) for item in ranked_candidates]
    cos_by_id = {cid: cos for cos, cid in pairs}
    chosen_cos = cos_by_id.get(chosen_id, pairs[0][0])

    target_sz = normalize_pack_size_token(extracted_pack_size)
    if target_sz:
        chosen_sz = normalize_pack_size_token(
            str(chosen_meta.get("pack_size") or chosen_meta.get("size") or chosen_meta.get("product") or "")
        )
        if chosen_sz == target_sz:
            return chosen_id, False
        chosen_var = str(
            chosen_meta.get("base_variant") or chosen_meta.get("variant") or ""
        ).strip().lower()
        for cos, cid in pairs:
            if cid == chosen_id:
                continue
            if (chosen_cos - cos) > max_cos_drop:
                continue
            c_meta = catalog.get(cid) or {}
            c_brand = str(c_meta.get("brand") or "").strip().lower()
            if c_brand != chosen_brand:
                continue
            c_var = str(c_meta.get("base_variant") or c_meta.get("variant") or "").strip().lower()
            if chosen_var and c_var and c_var != chosen_var:
                continue
            c_sz = normalize_pack_size_token(
                str(c_meta.get("pack_size") or c_meta.get("size") or c_meta.get("product") or "")
            )
            if c_sz == target_sz:
                return cid, True

    # Fallback for catalogs where sister SKUs differ by discriminative variant/packaging token
    stopwords = {"pack", "size", "none", "null", "unknown", "unreadable", "same", "brand", "product", chosen_brand}
    ext_toks = {
        t.lower()
        for t in re.split(r"[^a-zA-Z0-9]+", str(extracted_pack_size))
        if len(t) >= 4 and t.lower() not in stopwords
    }
    if not ext_toks:
        return chosen_id, False
    chosen_toks = _catalog_sister_tokens(chosen_meta)
    missing_toks = ext_toks - chosen_toks
    if not missing_toks:
        return chosen_id, False
    matching_sisters: list[int] = []
    for cos, cid in pairs:
        if cid == chosen_id or (chosen_cos - cos) > min(max_cos_drop, 0.035):
            continue
        c_meta = catalog.get(cid) or {}
        if str(c_meta.get("brand") or "").strip().lower() != chosen_brand:
            continue
        c_toks = _catalog_sister_tokens(c_meta)
        if c_toks & missing_toks:
            matching_sisters.append(cid)
    if len(matching_sisters) == 1:
        return matching_sisters[0], True
    return chosen_id, False


def resolve_clef_choice(
    choice: int | None,
    prob: float,
    top: list[tuple[float, int, str]],
    *,
    allow_abstain: bool = False,
    extracted_pack_size: str | None = None,
    catalog: dict[int, dict[str, Any]] | None = None,
) -> tuple[int | None, str]:
    """Resolve CLEF integer choice via 2-Tier Open-Set Competitor Guard + Log-Linear Energy Veto + Layer-5 Pack-Size Verifier.

    - Valid panel ``1..K``: applies Log-Linear Energy Veto when ``choice > 1``, ``(s1 - sk) > 0.045``,
      and ``prob < 0.90`` (preventing low-confidence logprob flips against a dominant visual top-1).
    - ``choice == 0`` or ``None``:
      * Tier 1 (``s1 >= 0.72``): High-confidence catalog match where VLM over-abstained on minor blur
        -> recover ``top[0][1]``.
      * Tier 2 (``0.45 <= s1 < 0.72``): Open-set competitor guard -> abstain ``None`` when
        ``allow_abstain=True``, or fallback to ``top[0][1]`` on closed-set catalog tasks.
      * Tier 3 (``s1 < 0.45``): Out-of-catalog / severe occlusion -> abstain ``None``.
    - Layer 5 Post-Generation Pack-Size Verifier: when ``extracted_pack_size`` and ``catalog`` are
      provided, deterministically verifies the resolved SKU against same-brand/same-variant sisters.
    """
    if not top:
        return None, "empty"
    s1, id1, _ = top[0]
    ans_id: int | None
    mode: str
    if isinstance(choice, int) and 1 <= choice <= len(top):
        sk, idk, _ = top[choice - 1]
        if choice > 1 and (s1 - sk) > 0.045 and prob < 0.90:
            ans_id, mode = id1, "energy_veto_top1"
        else:
            ans_id, mode = idk, f"panel_{choice}"
    elif s1 >= 0.72:
        ans_id, mode = id1, "tier1_high_cos_top1"
    elif s1 >= 0.45:
        if allow_abstain:
            ans_id, mode = None, "tier2_open_set_competitor_abstain"
        else:
            ans_id, mode = id1, "tier2_closed_set_top1"
    else:
        ans_id, mode = (None, "tier3_low_cos_abstain") if allow_abstain else (id1, "tier3_closed_set_top1")

    if ans_id is not None and extracted_pack_size and catalog:
        verified_id, overridden = verify_pack_size_sister_sku(
            ans_id,
            extracted_pack_size,
            [(s, pid) for s, pid, _ in top],
            catalog,
        )
        if overridden:
            return verified_id, f"{mode}+layer5_pack_size_verified"
    return ans_id, mode


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
        psize_str = str(data.get("pack_size") or "")
        ans, guard_mode = resolve_clef_choice(
            choice,
            prob,
            top,
            allow_abstain=allow_abstain,
            extracted_pack_size=psize_str,
            catalog=getattr(self, "catalog", None),
        )
        return ans, {
            "choice": choice,
            "clef_prob": prob,
            "pack_size": psize_str,
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
                ans, guard_mode = resolve_clef_choice(
                    c,
                    prob,
                    top,
                    allow_abstain=allow_abstain,
                    extracted_pack_size=psize,
                    catalog=getattr(self, "catalog", None),
                )
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


@register
class ClefAspectScalePrior(ClefContrastiveMetadata):
    """EXP-50: Aspect-Preserving Cap/Rim Context Crop (+12% Top Pad) + Projective Container Scale Prior (h/w, crown_taper)."""

    name = "clef_aspect_scale_prior"
    use_saliency_diff_zoom: bool = False
    use_lidstone_pack_crown: bool = False
    architecture = (
        f"EXP-50 Aspect-Preserving Cap/Rim Context Crop (+12% top pad) + Projective Scale Prior (h/w, crown_taper) -> "
        f"{MODEL} top-{CLEF_K} + Contrastive Metadata CLEF Contact Sheet"
    )
    steps = [
        "Setup: embed reference gallery photos and load SKU catalog metadata",
        "Stage 1.5 Crop: Extract aspect-preserving crop with +12% top cap/nozzle/crimp context padding and measure (h/w, crown_taper, h/h_plank)",
        "Stage 2A: Re-score top-2K candidates with projective container scale prior (slender Bottle/Spray/Tube vs squat Jar/Tub/Bar)",
        "Stage 2B: Render CLEF Zoom-Strip contact sheet with physical container geometry & contrastive catalog rubric",
    ]

    def _crop_box(self, image: Image.Image, box: Box) -> Image.Image:
        from utils.clm import letterbox_cap_crop

        return letterbox_cap_crop(image, box, top_pad=0.12, side_pad=0.06, bottom_pad=0.08)

    def _rank_with_geometry(
        self,
        image: Image.Image,
        ctx: Context,
        allowed_ids: set[int] | None = None,
        box: Box | None = None,
        plank_height: float | None = None,
    ) -> tuple[list[tuple[float, int, str]], dict[str, float], dict[str, float]]:
        from utils.clm import (
            apply_lidstone_pack_fusion,
            apply_projective_scale_prior,
            compute_container_geometry_features,
        )

        base_ranked = super().ranked(image, ctx, allowed_ids)
        geom = compute_container_geometry_features(image, box=box, plank_height=plank_height)
        cat_map = getattr(self, "catalog", None)
        reranked = apply_projective_scale_prior(base_ranked, geom, cat_map, weight=0.003, k=CLEF_K)
        pack_probs: dict[str, float] = {}
        if self.use_lidstone_pack_crown:
            reranked, pack_probs = apply_lidstone_pack_fusion(reranked, geom, cat_map, lam=0.002, eps=0.01, k=CLEF_K)
        return reranked, geom, pack_probs

    def _build_rubric_with_geom(
        self,
        top: list[tuple[float, int, str]],
        geom: dict[str, float] | None = None,
        pack_probs: dict[str, float] | None = None,
        diff_band: tuple[float, float, float] | None = None,
        row_idx: int | None = None,
    ) -> str:
        base = build_contrastive_candidate_rubric(top, getattr(self, "catalog", None), row_idx=row_idx)
        extras: list[str] = []
        if geom:
            ar = geom.get("aspect_ratio", 1.0)
            ct = geom.get("crown_taper", 0.85)
            if ar >= 1.75:
                shape_hint = "tall-slender container"
            elif ar <= 0.72:
                shape_hint = "horizontally-oriented container (bar, tub, or horizontal tube/carton)"
            elif ar <= 1.20:
                shape_hint = "wide-squat container"
            else:
                shape_hint = "medium-upright container"
            extras.append(f"Physical Crop Geometry: h/w={ar:.2f} ({shape_hint}), top_crown_taper={ct:.2f}")
        if diff_band is not None:
            extras.append(
                f"Self-Supervised Reference-Diff Zoom Strip covers vertical band y={int(diff_band[0]*100)}%..{int(diff_band[1]*100)}% (variance peak {diff_band[2]:.2f}x)"
            )
        if pack_probs:
            extras.append(
                "Top-22% Crown/Nozzle/Lid/Crimp Zoom Strip included: verify physical closure & container form directly on Q"
            )
        if not extras:
            return base
        return f"{base}\n" + " | ".join(extras) if base else " | ".join(extras)

    def _stage_a_accept(
        self,
        top: list[tuple[float, int, str]],
        p_clm: float,
        zlid: dict[str, float],
    ) -> bool:
        if not top:
            return False
        cat_map = getattr(self, "catalog", {}) or {}
        s1, id1, _ = top[0]
        s2, id2, _ = top[1] if len(top) > 1 else (-1.0, -1, "")
        c1 = cat_map.get(id1) or {}
        c2 = cat_map.get(id2) or {}
        b1 = str(c1.get("brand") or "")
        b2 = str(c2.get("brand") or "")
        p1 = str(c1.get("packaging_type") or "")
        prod1 = str(c1.get("product") or "")
        brand_only = bool(b1 and prod1 == b1)
        same_brand = bool(b1 and b1 == b2)
        same_variant = bool(
            same_brand
            and c1.get("base_variant") == c2.get("base_variant")
            and c1.get("pack_size") != c2.get("pack_size")
        )
        same_brand_pack_conflict = any(
            str((cat_map.get(cid) or {}).get("brand") or "") == b1
            and str((cat_map.get(cid) or {}).get("packaging_type") or "") not in ("", p1)
            for sc, cid, _ in top[1:3]
            if (s1 - sc) < 0.06
        )
        if brand_only or same_brand_pack_conflict:
            return False
        req_margin = 0.030 if (same_brand or same_variant) else 0.009
        req_p = 0.42 if same_brand else 0.30
        return bool(
            s1 >= 0.75
            and (s1 - s2) >= req_margin
            and p_clm >= req_p
            and zlid["z_margin"] >= 0.80
            and zlid["lid"] <= 11.5
        )

    def _clef_single_geom(
        self,
        image: Image.Image,
        top: list[tuple[float, int, str]],
        ctx: Context,
        geom: dict[str, float] | None = None,
        pack_probs: dict[str, float] | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        from utils.clm import (
            compute_reference_diff_band,
            zoom_sheet,
            zoom_sheet_saliency_diff,
        )

        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        cat_map = getattr(self, "catalog", {}) or {}
        p1 = str((cat_map.get(top[0][1]) or {}).get("packaging_type") or "") if top else ""
        other_packs = {
            str((cat_map.get(cid) or {}).get("packaging_type") or "")
            for _, cid, _ in top[1:3]
            if (cat_map.get(cid) or {}).get("packaging_type")
        }
        pack_conflict = bool(p1 and any(op != p1 for op in other_packs))
        include_crown = bool(self.use_lidstone_pack_crown and pack_conflict)
        diff_band: tuple[float, float, float] | None = None

        if self.use_saliency_diff_zoom or include_crown:
            diff_band = compute_reference_diff_band(refs)
            sheet_im = zoom_sheet_saliency_diff(image, refs, include_crown=include_crown)
        else:
            sheet_im = zoom_sheet(image, refs)

        base_prompt = CLEF_ZOOM_SINGLE_PROMPT.format(k=len(top))
        rubric = self._build_rubric_with_geom(
            top,
            geom=geom,
            pack_probs=pack_probs if include_crown else None,
            diff_band=diff_band,
        )
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
        psize_str = str(data.get("pack_size") or "")
        ans, guard_mode = resolve_clef_choice(
            choice,
            prob,
            top,
            allow_abstain=allow_abstain,
            extracted_pack_size=psize_str,
            catalog=getattr(self, "catalog", None),
        )
        return ans, {
            "choice": choice,
            "clef_prob": prob,
            "pack_size": psize_str,
            "guard_mode": guard_mode,
            "geometry": geom,
            "diff_band": list(diff_band) if diff_band else None,
            "include_crown": include_crown,
            "lidstone_pack_probs": pack_probs or None,
            "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in top],
        }

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        from utils.clm import compute_zscore_and_lid, infonce_posterior

        ranked, geom, pack_probs = self._rank_with_geometry(image, ctx, allowed_ids)
        if not ranked:
            return None
        top = ranked[:CLEF_K]
        s1, id1, _ = top[0]
        s2, _, _ = top[1] if len(top) > 1 else (-1.0, -1, "")
        zlid = compute_zscore_and_lid([s for s, _, _ in top])
        p_clm, ent_conf = infonce_posterior(top, inv_temp=28.0, k=CLEF_K)
        if self._stage_a_accept(top, p_clm, zlid):
            ctx.trace.step(
                "Stage A CLM + Projective Scale Fast-Gate (0 VLM calls)",
                f"#{id1} accepted (cos={s1:.3f}, margin={s1 - s2:.3f}, P_CLM={p_clm:.2f}, h/w={geom['aspect_ratio']:.2f})",
                info={
                    "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in top],
                    "top1_sku_id": id1,
                    "top5_sku_ids": [i for _, i, _ in top],
                    "clm_top1_prob": p_clm,
                    "coarse_entropy_conf": ent_conf,
                    "geometry": geom,
                    "lidstone_pack_probs": pack_probs or None,
                    **zlid,
                    "stage2a_bypassed": True,
                    "fast_gated_boxes": 1,
                    "answer": id1,
                },
            )
            return id1

        ans, meta = self._clef_single_geom(image, top, ctx, geom=geom, pack_probs=pack_probs)
        ctx.trace.step(
            f"Stage B {self.name} Contact-Sheet",
            f"panel {meta['choice']} -> #{ans} (P_token={meta['clef_prob']:.2f}, h/w={geom['aspect_ratio']:.2f}, mode={meta['guard_mode']})",
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
        from utils.clm import (
            compute_zscore_and_lid,
            group_adjacent_shelf_twins,
            infonce_posterior,
            zoom_multi_sheet,
            zoom_multi_sheet_saliency_diff,
        )
        from utils.photometry import inner_lab_zones, restore_photometry

        if not boxes:
            return []
        crops = [self._crop_box(image, b) for b in boxes]
        heights = [max(1.0, b[3] - b[1]) for b in boxes]
        median_plank_h = float(sorted(heights)[len(heights) // 2]) if heights else None

        restored_crops = [restore_photometry(c)[0] for c in crops]
        crop_labs = [inner_lab_zones(rc) for rc in restored_crops]
        twin_clusters, medoid_indices = group_adjacent_shelf_twins(boxes, restored_crops, crop_labs)
        with ThreadPoolExecutor(max_workers=8) as pool:
            medoid_res = list(
                pool.map(
                    lambda m_idx: self._rank_with_geometry(
                        crops[m_idx], ctx, box=boxes[m_idx], plank_height=median_plank_h
                    ),
                    medoid_indices,
                )
            )
        all_ranked: list[list[tuple[float, int, str]]] = [[] for _ in range(len(boxes))]
        all_geoms: list[dict[str, float]] = [{} for _ in range(len(boxes))]
        all_pack_probs: list[dict[str, float]] = [{} for _ in range(len(boxes))]
        for cl, (ranked_m, geom_m, pprobs_m) in zip(twin_clusters, medoid_res, strict=True):
            for member_idx in cl:
                all_ranked[member_idx] = ranked_m
                all_geoms[member_idx] = geom_m
                all_pack_probs[member_idx] = pprobs_m

        ids: list[int | None] = [None] * len(boxes)
        valid_indices: list[int] = []
        for i, r in enumerate(all_ranked):
            if not r:
                continue
            top = r[:CLEF_K]
            s1, id1, _ = top[0]
            s2, _, _ = top[1] if len(top) > 1 else (-1.0, -1, "")
            zlid = compute_zscore_and_lid([s for s, _, _ in top])
            p_clm, ent_conf = infonce_posterior(top, inv_temp=28.0, k=CLEF_K)
            if self._stage_a_accept(top, p_clm, zlid):
                ids[i] = id1
                ctx.trace.step(
                    "Stage A CLM + Projective Scale Fast-Gate",
                    f"box {i + 1}: #{id1} accepted (cos={s1:.3f}, margin={s1 - s2:.3f}, P_CLM={p_clm:.2f})",
                    info={
                        "box_index": i,
                        "shortlist": [{"sku_id": k_id, "cosine": round(s, 4)} for s, k_id, _ in top],
                        "top1_sku_id": id1,
                        "top5_sku_ids": [k_id for _, k_id, _ in top],
                        "clm_top1_prob": p_clm,
                        "coarse_entropy_conf": ent_conf,
                        "geometry": all_geoms[i],
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
                ans, meta = self._clef_single_geom(
                    crops[idx],
                    all_ranked[idx][:CLEF_K],
                    ctx,
                    geom=all_geoms[idx],
                    pack_probs=all_pack_probs[idx],
                )
                return [(idx, ans, meta)]
            qs = [crops[idx] for idx in batch]
            tops = [all_ranked[idx][:CLEF_K] for idx in batch]
            refs_per_q = [
                [Image.open(io.BytesIO(self.jpeg[p])).convert("RGB") for _, _, p in t]
                for t in tops
            ]
            if self.use_saliency_diff_zoom or self.use_lidstone_pack_crown:
                sheet_im = zoom_multi_sheet_saliency_diff(
                    qs, refs_per_q, include_crown=self.use_lidstone_pack_crown
                )
            else:
                sheet_im = zoom_multi_sheet(qs, refs_per_q)
            batch_prompt = CLEF_ZOOM_BATCH_PROMPT.format(m=len(batch), k=CLEF_K)
            row_rubrics = []
            for b_idx, (idx, t_row) in enumerate(zip(batch, tops, strict=True)):
                r_txt = self._build_rubric_with_geom(
                    t_row,
                    geom=all_geoms[idx],
                    pack_probs=all_pack_probs[idx] if self.use_lidstone_pack_crown else None,
                    row_idx=b_idx,
                )
                if r_txt:
                    row_rubrics.append(r_txt)
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
                ans, guard_mode = resolve_clef_choice(
                    c,
                    prob,
                    top,
                    allow_abstain=allow_abstain,
                    extracted_pack_size=psize,
                    catalog=getattr(self, "catalog", None),
                )
                out.append(
                    (
                        idx,
                        ans,
                        {
                            "choice": c,
                            "clef_prob": prob,
                            "pack_size": psize,
                            "guard_mode": guard_mode,
                            "geometry": all_geoms[idx],
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
                        f"Stage B {self.name} Contact-Sheet",
                        f"box {idx + 1}: panel {meta['choice']} -> #{ans} (P_token={meta['clef_prob']:.2f}, mode={meta.get('guard_mode', 'panel')})",
                        info={"box_index": idx, **meta, "answer": ans},
                    )
        return ids


@register
class ClefSaliencyDiffZoom(ClefAspectScalePrior):
    """EXP-51: Self-Supervised Reference-Difference Saliency Zoom Strip (V(x,y) = Var_k(R_k(x,y)))."""

    name = "clef_saliency_diff_zoom"
    use_saliency_diff_zoom = True
    use_lidstone_pack_crown = False
    architecture = (
        f"EXP-51 Self-Supervised Reference-Difference Saliency Zoom (Var_k(R_k(x,y))) + EXP-50 Projective Scale Prior -> "
        f"{MODEL} top-{CLEF_K} + 3-Band Saliency-Diff CLEF Contact Sheet"
    )
    steps = [
        "Setup: embed reference gallery photos and load SKU catalog metadata",
        "Stage 2A: Re-score top-2K candidates with projective container scale prior (h/w, crown_taper)",
        "Stage 2B (EXP-51 Saliency-Diff Zoom): Compute pixel variance V(x,y) = Var_k(R_k(x,y)) across aligned reference packshots to locate the max-variance horizontal band [y1*, y2*]",
        "Render 2.5x Lanczos Saliency-Diff Zoom Strip + Bottom-30% Weight Strip and execute CLEF logprob readout",
    ]


@register
class ClefPackCrownFusion(ClefSaliencyDiffZoom):
    """EXP-52: Lidstone-Calibrated 3D Container Specialist Fusion + Top-22% Crown/Nozzle Zoom Strip."""

    name = "clef_pack_crown_fusion"
    use_saliency_diff_zoom = True
    use_lidstone_pack_crown = True
    architecture = (
        f"EXP-52 Lidstone-Calibrated 3D Container Prior (-lambda ln P_Lidstone(pack)) + Top-22% Crown/Nozzle Strip "
        f"+ EXP-51 Saliency-Diff Zoom + EXP-50 Projective Scale Prior ({MODEL} top-{CLEF_K})"
    )
    steps = [
        "Setup: embed reference gallery photos and load SKU catalog metadata",
        "Stage 2A (EXP-50 + EXP-52): Fuse projective container geometry (h/w, crown_taper) with Lidstone-smoothed (eps=0.01) 16-class packaging_type log-prior",
        "Stage 2B (EXP-51 + EXP-52): Render Top-22% Crown/Nozzle/Lid/Crimp Zoom Strip + Self-Supervised Reference-Diff Band + Bottom-30% Weight Strip when candidates conflict on packaging_type",
        "Execute CLEF constrained schema + token logprob argmax readout + 2-Tier Open-Set Competitor Guard",
    ]
