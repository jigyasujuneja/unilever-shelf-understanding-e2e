"""TEMPLATE (not registered: files starting with ``_`` are skipped).
Copy to ``src/approaches/my_combined_pipeline.py`` to chain multiple approaches together
(for example: Pipeline 1+2 chaining Category/Brand/Package-Type -> Variant Classification,
or chaining any Detector + any Classifier).

Run locally or on Cloud Run:
    shelf-bench run -a my_combined_pipeline -m gemini-3.1-flash-lite --split val --limit 25 --owner riley
    shelf-bench cloud-run -a my_combined_pipeline -m gemini-3.1-flash-lite --owner riley
"""

from __future__ import annotations

from typing import Any

from PIL import Image

import approaches
from approaches.base import Approach, Box, Context, register


@register
class MyCombinedPipeline(Approach):
    name = "my_combined_pipeline"
    task = "combined"
    epic = "MT Market Share - Combined Classification"
    target_field = "variant"
    architecture = "Composable Pipeline: Detector -> Coarse (Category/Brand/Pkg) -> Fine Variant"
    steps = [
        "Stage 1: Detect product boxes on the shelf image",
        "Stage 2: Predict coarse hierarchy (Category, Brand, Package Type)",
        "Stage 3: Predict fine-grained Variant conditioned on coarse hierarchy",
    ]

    def setup(self, config: dict) -> None:
        self.detector = approaches.get("rtdetr_shelf_rail_detector")
        self.coarse = approaches.get("ft_gemini31_cat_brand_pkg")
        self.fine = approaches.get("ft_gemini31_variant_compound")
        for m in (self.detector, self.coarse, self.fine):
            m.setup(config)

    def detect_and_classify(
        self, image: Image.Image, ctx: Context
    ) -> tuple[list[Box], list[Any]]:
        boxes = self.detector.detect(image, ctx)
        coarse_attrs = self.coarse.classify(image, boxes, ctx)
        final_preds = self.fine.classify(image, boxes, ctx, prior=coarse_attrs)
        return boxes, final_preds
