"""Fine-Tuned Gemini 3.1 Flash Lite for Compound Outputs #1: Fine-Grained Variant Classification.

Epic: ``MT Market Share - Variant Classification`` (``task = "classification"``)
Implements Task #1 from the team's roadmap: Fine-Tuned Gemini 3.1 Flash Lite (``gemini-3.1-flash-lite``)
for Compound Outputs predicting fine-grained SKU ``variant`` alongside ``category``, ``brand``, and
``packaging_type``. Supports optional ``prior`` conditioning from Task #2 (``ft_gemini31_cat_brand_pkg``)
when chained inside ``compound_pipeline_1_plus_2``.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, label_counts, register
from utils import hul_domain


@register
class FTGemini31VariantCompoundClassifier(Approach):
    name = "ft_gemini31_variant_compound"
    task = "classification"
    epic = "MT Market Share - Variant Classification"
    target_field = "variant"
    architecture = (
        "Fine-Tuned Gemini 3.1 Flash Lite for Compound Outputs #1: "
        "Fine-Grained Variant & Shade Disambiguation (Supports Hierarchical Prior Conditioning)"
    )
    steps = [
        "Step 1: Sub-ROI Shade & Claim Zone Crop Preparation",
        "Step 2: Fine-Tuned Gemini 3.1 Flash Lite Compound Variant Decode",
    ]

    def classify(
        self,
        image: Image.Image,
        boxes: list[Box],
        ctx: Context,
        prior: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        prior_msg = "conditioned on Stage-1 (Category, Brand, Package Type) prior" if prior else "unconditioned standalone mode"
        ctx.trace.step(
            "Step 1: Sub-ROI Shade & Claim Crop Preparation",
            f"Preparing {len(boxes)} crops ({prior_msg})",
            boxes=boxes,
        )
        preds = hul_domain.classify_shelf_boxes_7dim(
            image, boxes, ctx=ctx, mode="ft_gemini31_variant_compound", prior=prior
        )
        ctx.trace.labels = preds
        ctx.trace.step(
            "Step 2: FT Gemini 3.1 Flash Lite Variant Compound Decode",
            f"Decoded {len(preds)} fine-grained variants ({label_counts(preds)})",
            boxes=boxes,
            labels=preds,
        )
        return preds
