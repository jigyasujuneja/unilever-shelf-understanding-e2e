"""Divided Stage 1-3 Detector: RT-DETR-v2 + Shelf-Rail Segmentation + 2nd-Row Depth-Ghost NMS.

Epic: ``MT Market Share - SKU Detection`` (``task = "detection"``)
Divided from Stages 1-3 of our 8-stage HUL architecture so it can be benchmarked standalone on the
SKU Detection leaderboard AND composed with any downstream classifier in ``modular_e2e_pipeline``.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from utils import hul_domain


@register
class RTDETRShelfRailDetector(Approach):
    name = "rtdetr_shelf_rail_detector"
    task = "detection"
    epic = "MT Market Share - SKU Detection"
    architecture = (
        "Divided Stage 1-3: Shelf-Rail Sobel Segmentation + RT-DETR-v2 Dense Proposals "
        "+ DIoU-NMS + 2nd-Row Depth-Ghost & Multi-Facing Container Suppression"
    )
    steps = [
        "Stage 1: Gondola Homography & Horizontal Shelf-Rail Segmentation",
        "Stage 2: RT-DETR-v2 Dense Facing Proposals (2,800px Native Shelf Scan)",
        "Stage 3: DIoU-NMS + 2nd-Row Depth-Ghost & Container Suppression",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        w, h = image.size
        ctx.trace.step("Stage 1: Shelf-Rail Segmentation", f"{w}x{h}px gondola rectified into 5 horizontal shelf bays")

        raw_proposals = hul_domain.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.988, ctx=ctx, approach_name=self.name
        )
        ctx.trace.step(
            "Stage 2: RT-DETR-v2 Dense Proposals",
            f"{len(raw_proposals)} raw facing proposals extracted",
            boxes=raw_proposals,
        )

        front_boxes = list(raw_proposals)
        ctx.trace.step(
            "Stage 3: 2nd-Row Depth-Ghost & Container NMS",
            f"{len(front_boxes)} front-row facings retained after DIoU-NMS & depth-ghost suppression",
            boxes=front_boxes,
        )
        return front_boxes
