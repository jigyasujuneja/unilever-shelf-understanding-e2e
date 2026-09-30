"""TEMPLATE (not registered: files starting with ``_`` are skipped).

Shows both ways to build an ``end_to_end`` approach in ``src/approaches/market_share/end_to_end/``:

1. **Single-invocation detect + classify** — override ``detect_and_identify(image, ctx)`` when a
   single model call (e.g. a fine-tuned Gemini model on Vertex AI or an Agent Platform endpoint)
   returns both bounding boxes and ``sku_id``s in one pass.
2. **Modular step composition** — call ``compose(name, detector, *identifiers, dataset=...)`` to
   chain any detector with one or more classification/retrieval steps (e.g.
   ``detect -> classify``, ``detect -> retrieve``, or ``detect -> classify -> retrieve``).
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, compose, register, to_pixels
from approaches.market_share.classification.hierarchy_classify import HierarchyClassify
from approaches.market_share.detection.single_pass_dedup import SinglePassDedup
from approaches.market_share.retrieval.embedding_retrieval import EmbeddingRetrieval
from utils import dataset

ITEM_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "box_2d": {"type": "array", "items": {"type": "integer"}},
            "sku_id": {"type": "integer"},
        },
        "required": ["box_2d", "sku_id"],
    },
}

PROMPT = (
    "Detect every retail product on this shelf and identify its catalog sku_id (or -1 if not "
    "in the catalog). Return ONLY JSON [{'box_2d': [ymin, xmin, ymax, xmax], 'sku_id': id}].\n\n"
    "Catalog:\n{catalog}"
)


# 1. Single-invocation detect + classify in one call
@register
class SingleCallShelfIdentify(Approach):
    name = "single_call_shelf_identify"
    task = "end_to_end"
    dataset = "shelves"  # or "rpc"
    architecture = "Single call: model detects every product box and assigns its catalog sku_id"
    steps = [
        "One call returns [{box_2d, sku_id}] for every product on the shelf",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.catalog = dataset.catalog(name=self.dataset)
        self.prompt = PROMPT.format(catalog="\n".join(
            f"{i}: {p['product']}" for i, p in self.catalog.items()
        ))

    def detect_and_identify(
        self, image: Image.Image, ctx: Context
    ) -> tuple[list[Box], list[int | None]]:
        w, h = image.size
        res = ctx.ask(image, self.prompt, schema=ITEM_SCHEMA, max_side=2048)
        boxes: list[Box] = []
        ids: list[int | None] = []
        for item in res.data if isinstance(res.data, list) else []:
            if not isinstance(item, dict):
                continue
            b = to_pixels([item.get("box_2d")], 0, 0, w, h)
            if b:
                boxes.append(b[0])
                sku = item.get("sku_id")
                ids.append(sku if sku in self.catalog else None)
        ctx.trace.step(
            "Single-call detect + classify",
            f"{len(boxes)} boxes, {sum(i is not None for i in ids)} matched to catalog",
            boxes=boxes,
        )
        return boxes, ids


# 2. Multi-step composition: detect -> coarse classify -> vector retrieve
ShelfDetectClassifyRetrieve = compose(
    "shelf_detect_classify_retrieve",
    SinglePassDedup,      # step 1: detect boxes
    HierarchyClassify,    # step 2: narrow catalog to matching brand/size
    EmbeddingRetrieval,   # step 3: rank reference crops within that shortlist
    dataset="shelves",
)
