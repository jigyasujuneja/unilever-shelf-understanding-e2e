"""End-to-end pipelines composed from detection + retrieval/classification steps (End-to-end tab).

Any detection approach (in ``market_share/detection/``) and one or more classification or
retrieval approaches (in ``market_share/classification/`` or ``market_share/retrieval/``) can be
composed into a registered ``end_to_end`` approach with :func:`approaches.base.compose`:

* ``detect -> retrieve`` (``detect_retrieve``, ``shelf_detect_retrieve``)
* ``detect -> tiered`` (``detect_tiered``, ``shelf_detect_tiered``): embedding answer, Gemini only
  for the crops the embedding is unsure about
* ``detect -> classify`` (e.g. ``compose("detect_classify_e2e", SinglePassDedup, HierarchyClassify)``)
* ``detect -> classify -> retrieve`` (coarse classifier narrows the catalog via ``narrow()``,
  then retrieval ranks only the matching reference photos)

Always sending every crop to Gemini (``detect_rerank`` / ``shelf_detect_rerank``) cost more than
the tiered pipelines for no better F2 and was removed.

For a single-invocation model that performs both detection and SKU classification in one call,
override ``Approach.detect_and_identify(image, ctx)`` instead — see
``_single_call_end_to_end_template.py``.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import BOX_LIST_SCHEMA, Approach, Box, Context, compose, to_pixels
from approaches.market_share.detection.single_pass_dedup import SinglePassDedup
from approaches.market_share.retrieval.embedding_retrieval import MODEL, EmbeddingRetrieval
from approaches.market_share.retrieval.tiered_hybrid import (
    MIN_COSINE,
    MIN_MARGIN,
    K,
    TieredHybrid,
)

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


class GeminiBoxDetector(Approach):
    """Lightweight single-call Gemini box detector for RPC checkout photos (used by ``compose``)."""

    name = "gemini_box_detector"
    architecture = "Gemini detects boxes"
    steps = ["One Gemini call on the photo returns every product box"]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        return gemini_detect(image, ctx)


# ---- RPC checkout photos (200 products, 4 studio reference photos each) -----------------------

DetectRetrieve = compose(
    "detect_retrieve",
    GeminiBoxDetector,
    EmbeddingRetrieval,
    dataset="rpc",
    also_calls=[MODEL],
    architecture="Gemini detects boxes -> crop embedding -> nearest reference photo",
    steps=[
        "Setup (once per run, reported apart from cost/img): embed every reference photo",
        "One Gemini call on the photo returns every product box",
        "Embed each box crop; the product of the most similar reference photo wins",
    ],
)

DetectTiered = compose(
    "detect_tiered",
    GeminiBoxDetector,
    TieredHybrid,
    dataset="rpc",
    architecture="Gemini detects boxes -> embedding match, Gemini rerank only when unsure",
    steps=[
        "Setup (once per run, reported apart from cost/img): embed every reference photo",
        "One Gemini call on the photo returns every product box",
        f"Per box: embedding match; if top >= {MIN_COSINE} and margin >= {MIN_MARGIN}, keep it, "
        f"else Gemini picks from the top {K}",
    ],
)


# ---- Real store shelves (Shelf Images parent-level, 96 products, 16-39 facings/photo) ---------

ShelfDetectRetrieve = compose(
    "shelf_detect_retrieve",
    SinglePassDedup,
    EmbeddingRetrieval,
    dataset="shelves",
    also_calls=[MODEL],
    architecture="single_pass_dedup boxes -> crop embedding -> nearest shelf reference crop",
    steps=[
        "Setup (once per run, reported apart from cost/img): embed every reference crop",
        "single_pass_dedup: one Gemini call + container, NMS and depth-ghost filters",
        "Embed each box crop; the product of the most similar reference crop wins",
    ],
)

ShelfDetectTiered = compose(
    "shelf_detect_tiered",
    SinglePassDedup,
    TieredHybrid,
    dataset="shelves",
    architecture="single_pass_dedup boxes -> embedding match, Gemini rerank only when unsure",
    steps=[
        "Setup (once per run, reported apart from cost/img): embed every reference crop",
        "single_pass_dedup: one Gemini call + container, NMS and depth-ghost filters",
        f"Per box: embedding match; if top >= {MIN_COSINE} and margin >= {MIN_MARGIN}, keep it, "
        f"else Gemini picks from the top {K}",
    ],
)
