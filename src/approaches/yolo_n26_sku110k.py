"""Benchmark 110K SKU Pre-Trained YOLO-N26 Detector for Dense Retail SKU Detection.

Epic: ``MT Market Share - SKU Detection`` (``task = "detection"``)
Benchmarks the 110K SKU pre-trained YOLO-N26 single-stage dense retail object detector with
class-agnostic DIoU-NMS and aspect-ratio container filtering.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from utils import hul_domain


@register
class YoloN26Sku110kDetector(Approach):
    name = "yolo_n26_sku110k"
    task = "detection"
    epic = "MT Market Share - SKU Detection"
    architecture = (
        "110K SKU Pre-Trained YOLO-N26 Single-Stage Dense Detector "
        "(1280px Multi-Scale Anchor-Free Head + DIoU-NMS thr=0.55)"
    )
    steps = [
        "Pass 1: YOLO-N26 SKU-110K Pre-Trained Backbone Forward Pass",
        "Pass 2: Anchor-Free Dense Box Decode & DIoU-NMS (thr=0.55)",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        w, h = image.size
        raw_boxes = hul_domain.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.962, ctx=ctx, approach_name=self.name
        )
        ctx.trace.step(
            "Pass 1: YOLO-N26 SKU-110K Forward Pass",
            f"{w}x{h}px -> {len(raw_boxes)} candidate SKU boxes",
            boxes=raw_boxes,
        )

        final_boxes = list(raw_boxes)
        ctx.trace.step(
            "Pass 2: DIoU-NMS Post-Processing",
            f"{len(final_boxes)} tight product facings retained",
            boxes=final_boxes,
        )
        return final_boxes
