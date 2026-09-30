"""Detect, then identify (the End-to-end tab): the whole pipeline on RPC checkout photos.

Gemini finds every product box in the photo (one call); the runner crops each box and the
approach identifies it against the reference gallery, either by embedding alone
(``detect_retrieve``) or with the Gemini rerank (``detect_rerank``). A box only counts if it
overlaps the real product (IoU >= 0.5) **and** names the right product.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import BOX_LIST_SCHEMA, Box, Context, register, to_pixels
from approaches.embedding_retrieval import MODEL, EmbeddingRetrieval
from approaches.gemini_rerank import GeminiRerank, K

DETECT_PROMPT = (
    "Detect every individual retail product in this photo - every box, bottle, can, bag, cup "
    "and package, including partly hidden ones. Each physical item gets its own tight box; do "
    "not merge touching items. Return ONLY a JSON array of boxes, each [ymin, xmin, ymax, xmax] "
    "normalized to 0-1000."
)


def gemini_detect(image: Image.Image, ctx: Context) -> list[Box]:
    res = ctx.ask(image, DETECT_PROMPT, schema=BOX_LIST_SCHEMA, max_side=1400)
    boxes = to_pixels(res.data, 0, 0, *image.size)
    ctx.trace.step("Gemini detects products", f"{len(boxes)} boxes", boxes=boxes)
    return boxes


@register
class DetectRetrieve(EmbeddingRetrieval):
    name = "detect_retrieve"
    task = "end_to_end"
    models: list[str] | None = None  # the Gemini detector
    also_calls = [MODEL]
    architecture = "Gemini detects boxes -> crop embedding -> nearest reference photo"
    steps = [
        "Setup (once per run, reported apart from cost/img): embed every reference photo",
        "One Gemini call on the photo returns every product box",
        "Embed each box crop; the product of the most similar reference photo wins",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        return gemini_detect(image, ctx)


@register
class DetectRerank(GeminiRerank):
    name = "detect_rerank"
    task = "end_to_end"
    architecture = f"Gemini detects boxes -> embedding shortlist (top {K}) -> Gemini picks"
    steps = [
        "Setup (once per run, reported apart from cost/img): embed every reference photo",
        "One Gemini call on the photo returns every product box",
        f"Per box: embedding shortlist of {K} products, then one Gemini call picks one (or none)",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        return gemini_detect(image, ctx)
