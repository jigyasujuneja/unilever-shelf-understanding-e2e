"""Embedding first, Gemini only when the embedding isn't sure (Retrieval and End-to-end tabs).

Ported from Jigyasu's ``tiered_hybrid_scann`` routing rule: accept the nearest-reference answer
when it is clearly ahead, otherwise escalate the crop to Gemini. His version reported a fixed
routing share; here the share is whatever the crops produce, and the trace shows each decision.

Escalation: the ``K`` products whose reference photos are closest to the crop go on one numbered
sheet next to the crop (Q), and Gemini picks the matching panel or none. Embeddings are cheap and
good at "roughly this kind of pack"; Gemini is better at reading the label (brand, flavour, size).
The crop is embedded once; the escalation reuses that ranking.

Always escalating (the former ``gemini_rerank`` approach, and ``detect_rerank`` /
``shelf_detect_rerank``) cost more than this tiered version for no better F2, so only this one is
kept.

The thresholds were tuned on the RPC **val** split (264 crops, ``multimodalembedding@001``):
accepting when the top product's cosine is >= 0.70 and at least 0.045 ahead of the runner-up
answered 39% of crops without Gemini, 95% of them correctly (embedding alone: 74%). The
similarity floor barely matters; the margin does. His 0.82 / 0.045 were for another embedding.
Because the thresholds belong to that embedder's cosine scale, this approach keeps
``multimodalembedding@001`` as its shortlist model until they are re-tuned for
``gemini-embedding-2-preview`` (the default of the other pipelines).
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from approaches.base import Context, register
from approaches.market_share.retrieval.embedding_retrieval import EmbeddingRetrieval
from utils.embeddings import MULTIMODAL

MIN_COSINE = 0.70
MIN_MARGIN = 0.045
SHORTLIST_MODEL = MULTIMODAL  # the embedder MIN_COSINE / MIN_MARGIN were tuned for
K = 5       # candidates shown to Gemini
CELL = 320  # px per image on the sheet

SCHEMA = {"type": "object", "properties": {"choice": {"type": "integer"}}, "required": ["choice"]}
PROMPT = (
    f"The image is a row of {K + 1} panels. Panel Q is one product cut out of a checkout-counter "
    f"photo. Panels 1-{K} are reference photos of {K} different products from the catalog. "
    "Which numbered panel shows the same product as Q: same brand, product, flavour or variant, "
    "and pack size? The reference may be photographed from another angle. If none of them is "
    'the same product, answer 0. Return ONLY JSON like {"choice": 2}.'
)


def sheet(query: Image.Image, refs: list[Image.Image]) -> Image.Image:
    """Q followed by the numbered candidates, each fitted into a CELL x CELL panel."""
    out = Image.new("RGB", (CELL * (len(refs) + 1), CELL + 30), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for k, im in enumerate([query, *refs]):
        im = im.copy()
        im.thumbnail((CELL - 10, CELL - 10))
        x = k * CELL
        out.paste(im, (x + (CELL - im.width) // 2, 30 + (CELL - im.height) // 2))
        draw.rectangle((x, 0, x + CELL - 1, CELL + 29), outline="#999")
        draw.text((x + 6, 2), "Q" if k == 0 else str(k), fill="red" if k == 0 else "black", font=font)
    return out


@register
class TieredHybrid(EmbeddingRetrieval):
    name = "tiered_hybrid"
    models: list[str] | None = None  # any Gemini for the escalations
    shortlist_model = SHORTLIST_MODEL
    also_calls = [SHORTLIST_MODEL]
    architecture = (f"Embedding answer if clearly ahead (cos >= {MIN_COSINE}, margin >= "
                    f"{MIN_MARGIN}), else Gemini picks from the top {K}")
    steps = [
        "Setup (once per run, reported apart from cost/img): embed every reference photo",
        "Per crop: embed it and rank the products by their most similar reference photo",
        f"Top product at cosine >= {MIN_COSINE} and >= {MIN_MARGIN} ahead of the next: accept it",
        f"Otherwise one Gemini call: the crop next to the top {K}; Gemini picks one (or none)",
    ]

    def pick(self, image: Image.Image, top: list[tuple[float, int, str]], ctx: Context) -> int | None:
        """Gemini picks one of the shortlisted ``top`` products for the crop, or None."""
        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        data = ctx.ask(sheet(image, refs), PROMPT, schema=SCHEMA).data
        choice = data.get("choice") if isinstance(data, dict) else None
        return top[choice - 1][1] if isinstance(choice, int) and 1 <= choice <= len(top) else None

    def identify(self, image: Image.Image, ctx: Context,
                 allowed_ids: set[int] | None = None) -> int | None:
        ranked = self.ranked(image, ctx, allowed_ids)  # the crop's only embedding call
        if not ranked:
            return None
        (s1, pid, _), s2 = ranked[0], ranked[1][0] if len(ranked) > 1 else -1.0
        info = {"shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in ranked[:K]],
                "thresholds": {"min_cosine": MIN_COSINE, "min_margin": MIN_MARGIN}}
        if s1 >= MIN_COSINE and s1 - s2 >= MIN_MARGIN:
            ctx.trace.step("Embedding tier", f"#{pid} accepted (cos {s1:.3f}, margin {s1 - s2:.3f})",
                           info={**info, "answer": pid})
            return pid
        choice = self.pick(image, ranked[:K], ctx)
        ctx.trace.step("Gemini tier", f"escalated (cos {s1:.3f}, margin {s1 - s2:.3f}) -> "
                       + (f"#{choice}" if choice is not None else "none"),
                       info={**info, "answer": choice})
        return choice
