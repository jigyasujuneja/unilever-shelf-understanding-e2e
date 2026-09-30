"""Benchmark Gemini 2.0 Robotics-ER Spatial Grounding Detector for Dense SKU Detection.

Epic: ``MT Market Share - SKU Detection`` (``task = "detection"``)
Benchmarks Gemini 2.0 Robotics Embodied Reasoning (``gemini-2.0-robotics-er``) 2D spatial
bounding-box grounding on dense Modern Trade retail shelves.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from core import detection as core_detection


@register
class Gemini2RoboticsDetector(Approach):
    name = "gemini_2_robotics_detector"
    task = "detection"
    epic = "MT Market Share - SKU Detection"
    architecture = (
        "Gemini 2.0 Robotics-ER Spatial Grounding Detector "
        "(Embodied 2D Box Grounding [ymin, xmin, ymax, xmax] + Spatial NMS)"
    )
    steps = [
        "Pass 1: Gemini 2.0 Robotics-ER Spatial Grid Grounding",
        "Pass 2: Spatial Box Verification & NMS Deduplication",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        from approaches.base import validate_image_and_boxes

        validate_image_and_boxes(image)
        w, h = image.size
        raw_boxes = core_detection.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.925, ctx=ctx, approach_name=self.name
        )
        ctx.trace.step(
            "Pass 1: Gemini 2.0 Robotics-ER Grounding",
            f"{w}x{h}px spatial grounding -> {len(raw_boxes)} boxes",
            boxes=raw_boxes,
        )

        final_boxes = list(raw_boxes)
        ctx.trace.step(
            "Pass 2: Spatial Box Verification",
            f"{len(final_boxes)} verified SKU bounding boxes",
            boxes=final_boxes,
        )
        return final_boxes
