"""Retrieval Ablation Studies (``jev_laya_global_wb``, ``jev_laya_zoom_strip``, ``clef_flash_decision``).

Kept in ``experiments/ablations/`` for reproducibility and Vertex AI Experiments lineage.
See ``experiments/ablations/__init__.py`` for the per-image root-cause postmortem explaining
why ``jev_laya_zscore_lid`` (0.9571 F2) and ``l8_hul_ultimate`` (0.9510 F2) outperformed these ablations.
"""

from __future__ import annotations

import io

from PIL import Image

from approaches.base import Box, Context, register
from approaches.market_share.classification.embedding_text_match import unit
from approaches.market_share.retrieval.coarse_jev_laya import CoarseJevLaya
from approaches.market_share.retrieval.jev_laya_hybrid import (
    BATCH_PROMPT,
    BATCH_SCHEMA,
    LAB_WEIGHT,
    SHORTLIST_MODEL,
    K,
    _dot,
)
from approaches.market_share.retrieval.tiered_hybrid import PROMPT
from utils.clm import zoom_multi_sheet, zoom_sheet
from utils.openjev import extract_slot_distributions, resolve_jev_choice
from utils.photometry import (
    apply_illuminant,
    estimate_shelf_illuminant,
    inner_lab_zones,
    lab_similarity,
    restore_photometry,
)

CLEF_SINGLE_SCHEMA = {"type": "integer", "minimum": 0, "maximum": K}
CLEF_BATCH_SCHEMA = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": K},
}
CLEF_SINGLE_PROMPT = (
    "Panel Q is a product cut out of a retail photo. Panels 1-{k} are reference photos of candidate catalog products. "
    "Pick the numbered panel (1-{k}) that shows the same product (same brand, variant, packaging form, and pack size; "
    "each panel includes a bottom zoom strip of the net-weight zone). The reference may be photographed from another angle. "
    "If none of them is the same product, answer 0. Return ONLY the single integer (0-{k})."
)
CLEF_BATCH_PROMPT = (
    "The image shows {m} rows (Q1 to Q{m}). In each row, the red-labelled left panel Qi is a product cut out of a photo, "
    "followed by reference photos 1-{k} of candidate catalog products (each cell includes a bottom zoom strip of the "
    "net-weight/pack-size zone). For each row Qi (in order 1..{m}), pick the numbered panel (1-{k}) that shows the same "
    "product (same brand, product, flavour or variant, and pack size). The reference may be photographed from another angle. "
    "If none of them is the same product, answer 0. Return ONLY a JSON array of {m} integers like [2, 1]."
)


@register
class JevLayaGlobalWb(CoarseJevLaya):
    """Ablation #4 (0.8857 / 0.9078 F2): Full-shelf Shades-of-Gray (p=6) illuminant without Z-Score gate."""

    name = "jev_laya_global_wb"
    architecture = (
        f"Full-Shelf Shades-of-Gray (p=6) Global Illuminant + Inner-80% Foreground CIELAB + "
        f"{SHORTLIST_MODEL} -> Coarse JEV triage & Gemini 3.5 contact sheets"
    )
    steps = [
        "Setup: embed reference gallery + inner-80% foreground CIELAB descriptors",
        "Estimate store illuminant once on the full shelf image (Shades-of-Gray Minkowski p=6) and store on ctx",
        "Apply global shelf white-balance gains to every crop before de-glare, embedding & inner-CIELAB ranking",
        "Stage A Coarse JEV triage -> Stage B batched Gemini 3.5 contact sheets & row smoothing",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        super().setup(config, ctx)
        for _, path, _ in self.index:
            if path in self.jpeg:
                with Image.open(io.BytesIO(self.jpeg[path])) as im:
                    rgb = im.convert("RGB")
                    ref_gains = estimate_shelf_illuminant(rgb, p=6.0)
                    self.lab_index[path] = inner_lab_zones(apply_illuminant(rgb, ref_gains))

    def _rank_crop(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> tuple[list[tuple[float, int, str]], list[float], Image.Image]:
        restored, _ = restore_photometry(image)
        q = unit(self.emb.image(restored, ctx))
        pool = (
            [(pid, path, v) for pid, path, v in self.index if pid in allowed_ids]
            if allowed_ids
            else []
        ) or self.index
        best: dict[int, tuple[float, int, str]] = {}
        for pid, path, v in pool:
            s = _dot(q, v)
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, pid, path)
        coarse = sorted(best.values(), reverse=True)
        if not coarse:
            return [], q, restored

        gains = getattr(ctx, "shelf_illuminant", None)
        wb_crop = apply_illuminant(restored, gains) if gains else restored
        q_lab = inner_lab_zones(wb_crop)
        head = []
        for cos, pid, path in coarse[: 2 * K]:
            ref_lab = self.lab_index.get(path)
            s_lab = lab_similarity(q_lab, ref_lab) if ref_lab else 0.5
            joint = cos + LAB_WEIGHT * (s_lab - 0.5)
            head.append((joint, cos, pid, path))
        head.sort(reverse=True)
        reranked = [(cos, pid, path) for _, cos, pid, path in head] + coarse[2 * K :]
        return reranked, q, restored

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        ctx.shelf_illuminant = estimate_shelf_illuminant(image, p=6.0)
        ctx.trace.step(
            "Global shelf illuminant (Shades-of-Gray p=6)",
            f"RGB gains={ctx.shelf_illuminant}",
            info={"shelf_illuminant_gains": ctx.shelf_illuminant},
        )
        return super().identify_boxes(image, boxes, ctx)


@register
class JevLayaZoomStrip(CoarseJevLaya):
    """Ablation #5 (0.9000 / 0.9078 F2): Bottom-30% Pack-Size Zoom-Crop Strip inside each Laya cell."""

    name = "jev_laya_zoom_strip"
    architecture = (
        "Stage A Coarse JEV triage -> Stage B Bottom-30% Pack-Size Zoom-Strip Contact Sheets "
        "(max_side=1024, +10.5 dB net-weight PSNR at 0 extra image tokens)"
    )
    steps = [
        "Setup: embed reference gallery, CIELAB zones & SKU family centroids",
        "Stage A: Coarse JEV family triage accepts distinct-family crops in 0 VLM calls",
        "Stage B: Build multi-row contact sheets with a high-res Bottom-30% Pack-Size Zoom Strip in every cell",
        "Verify medoids with Gemini 3.5 logprob calibration and 1D shelf-row smoothing",
    ]

    def pick_with_confidence(
        self, image: Image.Image, top: list[tuple[float, int, str]], ctx: Context
    ) -> tuple[int | None, float, dict]:
        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        sheet_im = zoom_sheet(image, refs)
        res = ctx.ask(
            sheet_im,
            PROMPT,
            schema={"type": "object", "properties": {"choice": {"type": "integer"}}, "required": ["choice"]},
            logprobs=len(top) + 1,
        )
        data = res.data
        raw_choice = data.get("choice") if isinstance(data, dict) else None
        valid_opts = set(range(0, len(top) + 1))
        slots = extract_slot_distributions(
            res.meta.get("logprobs"), valid_options=valid_opts, n_options=len(top) + 1
        )
        return resolve_jev_choice(
            raw_choice,
            slots[0] if slots else None,
            top,
            fallback_cosine=self.fallback_cosine,
            min_confidence=self.min_jev_confidence,
        )

    def pick_batch_with_confidence(
        self,
        queries: list[Image.Image],
        tops: list[list[tuple[float, int, str]]],
        ctx: Context,
    ) -> list[tuple[int | None, float, dict]]:
        if len(queries) == 1:
            return [self.pick_with_confidence(queries[0], tops[0], ctx)]
        refs_per_q = [
            [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
            for top in tops
        ]
        prompt = BATCH_PROMPT.format(m=len(queries), k=K)
        sheet_im = zoom_multi_sheet(queries, refs_per_q)
        res = ctx.ask(sheet_im, prompt, schema=BATCH_SCHEMA, logprobs=K + 1)
        data = res.data
        raw = data.get("choices") if isinstance(data, dict) else None
        choices = raw if isinstance(raw, list) else []
        valid_opts = set(range(0, K + 1))
        slots = extract_slot_distributions(
            res.meta.get("logprobs"), valid_options=valid_opts, n_options=K + 1
        )
        return [
            resolve_jev_choice(
                choices[idx] if idx < len(choices) else None,
                slots[idx] if idx < len(slots) else None,
                top,
                fallback_cosine=self.fallback_cosine,
                min_confidence=self.min_jev_confidence,
            )
            for idx, top in enumerate(tops)
        ]


@register
class ClefFlashDecision(CoarseJevLaya):
    """Ablation #6 (0.8714 / 0.9222 F2): Clef-Flash (/v1/systemone) Bare Integer Array Decision Head."""

    name = "clef_flash_decision"
    architecture = (
        "Stage A Coarse JEV triage -> Stage B Clef-Flash (/v1/systemone) Typed Constrained "
        "Decision Head (integer array schema + 6-way token logprob distribution)"
    )
    steps = [
        "Setup: embed reference gallery, CIELAB zones & SKU family centroids",
        "Stage A: Coarse JEV family triage resolves distinct-family facings in 0 VLM calls",
        "Stage B: Clef-Flash typed decision head outputs bare integer array [c1..cm] with 6-way logprob readout",
        "Calibrate decision posterior via 1 - H(p)/ln(N) entropy fusion + 1D shelf-row smoothing",
    ]

    def pick_with_confidence(
        self, image: Image.Image, top: list[tuple[float, int, str]], ctx: Context
    ) -> tuple[int | None, float, dict]:
        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        sheet_im = zoom_sheet(image, refs)
        prompt = CLEF_SINGLE_PROMPT.format(k=len(top))
        res = ctx.ask(sheet_im, prompt, schema=CLEF_SINGLE_SCHEMA, logprobs=len(top) + 1)
        raw_choice = res.data if isinstance(res.data, int) else (
            res.data.get("choice") if isinstance(res.data, dict) else None
        )
        valid_opts = set(range(0, len(top) + 1))
        slots = extract_slot_distributions(
            res.meta.get("logprobs"), valid_options=valid_opts, n_options=len(top) + 1
        )
        return resolve_jev_choice(
            raw_choice,
            slots[0] if slots else None,
            top,
            fallback_cosine=self.fallback_cosine,
            min_confidence=self.min_jev_confidence,
        )

    def pick_batch_with_confidence(
        self,
        queries: list[Image.Image],
        tops: list[list[tuple[float, int, str]]],
        ctx: Context,
    ) -> list[tuple[int | None, float, dict]]:
        if len(queries) == 1:
            return [self.pick_with_confidence(queries[0], tops[0], ctx)]
        refs_per_q = [
            [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
            for top in tops
        ]
        prompt = CLEF_BATCH_PROMPT.format(m=len(queries), k=K)
        sheet_im = zoom_multi_sheet(queries, refs_per_q)
        res = ctx.ask(sheet_im, prompt, schema=CLEF_BATCH_SCHEMA, logprobs=K + 1)
        choices = res.data if isinstance(res.data, list) else (
            res.data.get("choices", []) if isinstance(res.data, dict) else []
        )
        valid_opts = set(range(0, K + 1))
        slots = extract_slot_distributions(
            res.meta.get("logprobs"), valid_options=valid_opts, n_options=K + 1
        )
        return [
            resolve_jev_choice(
                choices[idx] if idx < len(choices) else None,
                slots[idx] if idx < len(slots) else None,
                top,
                fallback_cosine=self.fallback_cosine,
                min_confidence=self.min_jev_confidence,
            )
            for idx, top in enumerate(tops)
        ]
