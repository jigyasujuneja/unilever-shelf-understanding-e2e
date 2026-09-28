"""Fine-Tuned Gemini 3.1 Flash Lite for Compound Outputs #2: Category, Brand & Package Type.

Epic: ``MT Market Share - Other (Category, Brand and Package Type) Classifiers`` (``task = "classification"``)
Implements Task #2 from the team's roadmap: Fine-Tuned Gemini 3.1 Flash Lite (``gemini-3.1-flash-lite``)
predicting structured compound JSON outputs ``{"category", "brand", "packaging_type", "is_hul"}`` for each
shelf crop. Can be benchmarked standalone in Epic 2 or chained into ``ft_gemini31_variant_compound`` in
``compound_pipeline_1_plus_2`` (Epic 4).
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, label_counts, register
from utils import hul_domain

COMPOUND_CAT_BRAND_PKG_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "category": {"type": "string"},
            "brand": {"type": "string"},
            "packaging_type": {"type": "string"},
            "is_hul": {"type": "boolean"},
        },
        "required": ["id", "category", "brand", "packaging_type"],
    },
}


@register
class FTGemini31CatBrandPkgClassifier(Approach):
    name = "ft_gemini31_cat_brand_pkg"
    task = "classification"
    epic = "MT Market Share - Other (Category, Brand and Package Type) Classifiers"
    target_field = "compound"
    architecture = (
        "Fine-Tuned Gemini 3.1 Flash Lite for Compound Outputs #2: "
        "Constrained JSON Schema for (Category, Brand, Package Type, Is-HUL)"
    )
    steps = [
        "Step 1: Contact-Sheet Crop Batching & High-Purity Clustering",
        "Step 2: Fine-Tuned Gemini 3.1 Flash Lite Compound (Category + Brand + Package Type) Decode",
    ]

    def classify(
        self,
        image: Image.Image,
        boxes: list[Box],
        ctx: Context,
        prior: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        ctx.trace.step(
            "Step 1: Contact-Sheet Crop Batching",
            f"Packing {len(boxes)} crops into clustered contact sheets for Fine-Tuned Gemini 3.1 Flash Lite",
            boxes=boxes,
        )
        preds = hul_domain.classify_shelf_boxes_7dim(
            image, boxes, ctx=ctx, mode="ft_gemini31_cat_brand_pkg", prior=prior
        )
        ctx.trace.labels = preds
        ctx.trace.step(
            "Step 2: FT Gemini 3.1 Flash Lite Compound Decode (Cat + Brand + Pkg)",
            f"Decoded {len(preds)} compound (Category, Brand, Package Type) predictions ({label_counts(preds)})",
            boxes=boxes,
            labels=preds,
        )
        return preds
