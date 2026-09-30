"""Embedding first, Gemini only when the embedding isn't sure (Retrieval and End-to-end tabs).

Ported from Jigyasu's ``tiered_hybrid_scann`` routing rule: accept the nearest-reference answer
when it is clearly ahead, otherwise escalate the crop to the ``gemini_rerank`` contact sheet.
His version reported a fixed routing share; here the share is whatever the crops produce, and
the trace shows each decision.

The thresholds were tuned on the RPC **val** split (264 crops, ``multimodalembedding@001``):
accepting when the top product's cosine is >= 0.70 and at least 0.045 ahead of the runner-up
answered 39% of crops without Gemini, 95% of them correctly (embedding alone: 74%). The
similarity floor barely matters; the margin does. His 0.82 / 0.045 were for another embedding.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Box, Context, register
from approaches.detect_identify import gemini_detect
from approaches.gemini_rerank import GeminiRerank, K

MIN_COSINE = 0.70
MIN_MARGIN = 0.045


@register
class TieredHybrid(GeminiRerank):
    name = "tiered_hybrid"
    architecture = (f"Embedding answer if clearly ahead (cos >= {MIN_COSINE}, margin >= "
                    f"{MIN_MARGIN}), else Gemini picks from the top {K}")
    steps = [
        "Setup (once per run, reported apart from cost/img): embed every reference photo",
        "Per crop: embed it and rank the products by their most similar reference photo",
        f"Top product at cosine >= {MIN_COSINE} and >= {MIN_MARGIN} ahead of the next: accept it",
        f"Otherwise one Gemini call: the crop next to the top {K}; Gemini picks one (or none)",
    ]

    def identify(self, image: Image.Image, ctx: Context) -> int | None:
        ranked = self.ranked(image, ctx)
        (s1, pid, _), s2 = ranked[0], ranked[1][0] if len(ranked) > 1 else -1.0
        if s1 >= MIN_COSINE and s1 - s2 >= MIN_MARGIN:
            ctx.trace.step("Embedding tier", f"#{pid} accepted (cos {s1:.3f}, margin {s1 - s2:.3f})")
            return pid
        ctx.trace.step("Gemini tier", f"escalated (cos {s1:.3f}, margin {s1 - s2:.3f})")
        return super().identify(image, ctx)


@register
class DetectTiered(TieredHybrid):
    name = "detect_tiered"
    task = "end_to_end"
    architecture = "Gemini detects boxes -> " + TieredHybrid.architecture
    steps = [
        "Setup (once per run, reported apart from cost/img): embed every reference photo",
        "One Gemini call on the photo returns every product box",
        *TieredHybrid.steps[1:],
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        return gemini_detect(image, ctx)
