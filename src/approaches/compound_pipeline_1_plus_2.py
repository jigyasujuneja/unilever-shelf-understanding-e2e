"""Combined Classification Pipeline (1 + 2): Chains FT Gemini 3.1 Cat/Brand/Pkg (#2) -> FT Gemini 3.1 Variant (#1).

Epic: ``MT Market Share - Combined Classification`` (``task = "combined"``)
Implements "Pipeline of 1 + 2" from the team's roadmap without duplicating a single line of model logic:
  1. Runs Stage-1 Detector (default ``rtdetr_shelf_rail_detector``, or ``yolo_n26_sku110k``)
  2. Runs Classifier #2 (``ft_gemini31_cat_brand_pkg``) to predict ``(Category, Brand, Package Type)``
  3. Passes those predictions as the ``prior`` into Classifier #1 (``ft_gemini31_variant_compound``)
     to resolve the exact ``Variant`` and 7-Dim SKU.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, get, label_counts, register


@register
class CompoundPipeline1Plus2(Approach):
    name = "compound_pipeline_1_plus_2"
    task = "combined"
    epic = "MT Market Share - Combined Classification"
    target_field = "variant"
    architecture = (
        "Pipeline of 1 + 2: Stage-1 Detector -> FT Gemini 3.1 Flash Lite (Category, Brand, Package Type) "
        "-> Hierarchical Prior Conditioning -> FT Gemini 3.1 Flash Lite (Variant Compound)"
    )
    steps = [
        "Stage 1: Dense SKU Facing Detection (RT-DETR-v2 + Shelf-Rail NMS)",
        "Stage 2 (Task #2): FT Gemini 3.1 Flash Lite Compound (Category, Brand, Package Type)",
        "Stage 3 (Task #1 | Prior): FT Gemini 3.1 Flash Lite Compound Variant Classification",
    ]

    def __init__(
        self,
        detector_name: str = "rtdetr_shelf_rail_detector",
        attr_classifier_name: str = "ft_gemini31_cat_brand_pkg",
        variant_classifier_name: str = "ft_gemini31_variant_compound",
    ) -> None:
        self.detector_name = detector_name
        self.attr_classifier_name = attr_classifier_name
        self.variant_classifier_name = variant_classifier_name

    def setup(self, config: dict) -> None:
        self.detector = get(self.detector_name)
        self.attr_classifier = get(self.attr_classifier_name)
        self.variant_classifier = get(self.variant_classifier_name)
        self.detector.setup(config)
        self.attr_classifier.setup(config)
        self.variant_classifier.setup(config)

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        boxes, _ = self.detect_and_classify(image, ctx)
        return boxes

    def detect_and_classify(
        self, image: Image.Image, ctx: Context
    ) -> tuple[list[Box], list[Any]]:
        if not hasattr(self, "detector"):
            self.setup({})
        boxes = self.detector.detect(image, ctx)
        coarse_prior = self.attr_classifier.classify(image, boxes, ctx)
        final_labels = self.variant_classifier.classify(image, boxes, ctx, prior=coarse_prior)
        ctx.trace.labels = final_labels
        ctx.trace.step(
            "Pipeline 1+2 Summary",
            f"{len(boxes)} facings classified via 1+2 hierarchical chain ({label_counts(final_labels)})",
            boxes=boxes,
            labels=final_labels,
        )
        return boxes, final_labels
