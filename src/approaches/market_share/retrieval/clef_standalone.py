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


@register
class ClefStandaloneRetrieval(EmbeddingRetrieval):
    """Standalone System-2 CLEF Retrieval (100% CLEF Zoom-Strip evaluation, zero JEV coupling)."""

    name = "clef_standalone"
    task = "retrieval"
    dataset = "rpc"
    models: list[str] | None = None
    shortlist_model = MODEL
    also_calls = [MODEL]
    architecture = (
        f"Standalone CLEF: {MODEL} top-{CLEF_K} shortlist -> 100% CLEF Bottom-30% Zoom-Strip "
        "Constrained Logprob Decision Head (zero Stage-A JEV bypass)"
    )
    steps = [
        "Setup: embed reference gallery photos",
        f"Retrieve top-{CLEF_K} candidate reference photos by embedding cosine",
        "Render Bottom-30% Pack-Size Zoom-Strip contact sheet for 100% of queries (no Stage A bypass)",
        "Execute CLEF constrained schema + direct token logprob argmax readout P_token(k | x)",
    ]

    def _clef_single(
        self, image: Image.Image, top: list[tuple[float, int, str]], ctx: Context
    ) -> tuple[int | None, dict[str, Any]]:
        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        sheet_im = zoom_sheet(image, refs)
        res = ctx.ask(
            sheet_im,
            CLEF_ZOOM_SINGLE_PROMPT.format(k=len(top)),
            schema=CLEF_ZOOM_SINGLE_SCHEMA,
            logprobs=len(top) + 1,
        )
        data = res.data if isinstance(res.data, dict) else {}
        raw_c = data.get("choice")
        slots = clef_logprob_select(res.meta.get("logprobs"), set(range(0, len(top) + 1)), n_slots=1)
        choice = slots[0]["choice"] if slots else (raw_c if isinstance(raw_c, int) else None)
        prob = slots[0]["top1_prob"] if slots else 1.0
        ans = top[choice - 1][1] if isinstance(choice, int) and 1 <= choice <= len(top) else (top[0][1] if top else None)
        return ans, {"choice": choice, "clef_prob": prob, "pack_size": data.get("pack_size", "")}

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        ranked = self.ranked(image, ctx, allowed_ids)
        if not ranked:
            return None
        top = ranked[:CLEF_K]
        ans, meta = self._clef_single(image, top, ctx)
        ctx.trace.step(
            "Standalone CLEF Zoom-Strip",
            f"panel {meta['choice']} -> #{ans} (P_token={meta['clef_prob']:.2f}, size={meta['pack_size']!r})",
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
        if not boxes:
            return []
        crops = [crop(image, b) for b in boxes]
        with ThreadPoolExecutor(max_workers=8) as pool:
            all_ranked = list(pool.map(lambda c: self.ranked(c, ctx), crops))

        ids: list[int | None] = [None] * len(boxes)
        valid_indices = [i for i, r in enumerate(all_ranked) if r]
        batches = [valid_indices[i : i + CLEF_BATCH] for i in range(0, len(valid_indices), CLEF_BATCH)]

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
            res = ctx.ask(
                sheet_im,
                CLEF_ZOOM_BATCH_PROMPT.format(m=len(batch), k=CLEF_K),
                schema=CLEF_ZOOM_BATCH_SCHEMA,
                logprobs=CLEF_K + 1,
            )
            data = res.data if isinstance(res.data, dict) else {}
            raw_choices = data.get("choices") if isinstance(data.get("choices"), list) else []
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
                ans = top[c - 1][1] if isinstance(c, int) and 1 <= c <= len(top) else (top[0][1] if top else None)
                out.append((idx, ans, {"choice": c, "clef_prob": prob}))
            return out

        with ThreadPoolExecutor(max_workers=4) as pool:
            for b_out in pool.map(run_batch, batches):
                for idx, ans, meta in b_out:
                    ids[idx] = ans
                    ctx.trace.step(
                        "Standalone CLEF Zoom-Strip",
                        f"box {idx + 1}: panel {meta['choice']} -> #{ans} (P_token={meta['clef_prob']:.2f})",
                        info={"box_index": idx, **meta, "answer": ans},
                    )
        return ids
