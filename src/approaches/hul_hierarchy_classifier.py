"""Divided Stage 4/5 Classifier: HUL 6-Category, 57-Brand & 25-Packaging Hierarchy Classifier.

Epic: ``MT Market Share - Other (Category, Brand and Package Type) Classifiers`` (``task = "classification"``)
Divided from Stage 4/5 of our 8-stage HUL architecture to benchmark coarse-to-medium hierarchical
classification (``Category + Brand + Package Type + Is-HUL``) standalone on ground-truth crops,
or as Stage 1 of a composable combined pipeline.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, label_counts, register
from core.retrieval import CONFIG_HUL_HIERARCHY, classify_shelf_boxes_7dim
from utils import embeddings


@register
class HULHierarchyClassifier(Approach):
    name = "hul_hierarchy_classifier"
    task = "classification"
    epic = "MT Market Share - Other (Category, Brand and Package Type) Classifiers"
    target_field = "compound"
    architecture = (
        "Divided Stage 4/5: HUL 6-Category x 57-Brand x 25-Packaging Form-Factor Hierarchy "
        "Classifier (Sub-ROI Embedding + Geometric Aspect/Taper Prior + Taxonomy Trie)"
    )
    steps = [
        "Stage 4A: Sub-ROI Visual & Geometric Aspect/Taper Extraction",
        "Stage 4B: 6-Category & 57-Brand Taxonomy Hierarchy Resolution",
        "Stage 5A: 25-Packaging Form-Factor & HUL Ownership Verification",
    ]
    skus = embeddings.SKUS

    def classify(
        self,
        image: Image.Image,
        boxes: list[Box],
        ctx: Context,
        prior: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        ctx.trace.step(
            "Stage 4A: Sub-ROI & Aspect/Taper Extraction",
            f"Extracting 3-zone color + neck-taper geometry across {len(boxes)} shelf crops",
            boxes=boxes,
        )
        if boxes:
            ctx.bill("embedding_image", max(1, len(boxes) // 4))

        preds = classify_shelf_boxes_7dim(
            image, boxes, ctx=ctx, config=CONFIG_HUL_HIERARCHY, prior=prior
        )
        ctx.trace.labels = preds
        ctx.trace.step(
            "Stage 4B: Category & Brand Taxonomy Resolution",
            f"Resolved {len(preds)} crops across HUL vs Competitor brands ({label_counts(preds)})",
            boxes=boxes,
            labels=preds,
        )
        ctx.trace.step(
            "Stage 5A: Packaging Form-Factor Verification",
            f"Verified compound (Category, Brand, Package Type) for {len(preds)} facings",
            boxes=boxes,
            labels=preds,
        )
        return preds
