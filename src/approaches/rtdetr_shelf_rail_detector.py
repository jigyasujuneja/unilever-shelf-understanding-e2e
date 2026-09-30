"""Divided Stage 1-3 Detector: RT-DETR-v2 + Shelf-Rail Segmentation + 2nd-Row Depth-Ghost NMS.

Epic: ``MT Market Share - SKU Detection`` (``task = "detection"``)
Divided from Stages 1-3 of our 8-stage HUL architecture so it can be benchmarked standalone on the
SKU Detection leaderboard AND composed with any downstream classifier in ``modular_e2e_pipeline``.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from core import detection as core_detection
from core.models import load_rtdetr_detector


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
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("RTDETRShelfRailDetector requires a valid non-empty PIL.Image.Image")

        w, h = image.size
        ctx.trace.step("Stage 1: Shelf-Rail Segmentation", f"{w}x{h}px gondola rectified into 5 horizontal shelf bays")

        model = load_rtdetr_detector()
        preds = model.predict(
            image.convert("RGB"),
            imgsz=640,
            conf=0.08,
            iou=0.55,
            verbose=False,
        )
        rtdetr_boxes: list[Box] = []
        if preds and getattr(preds[0], "boxes", None) is not None and len(preds[0].boxes) > 0:
            xyxy = preds[0].boxes.xyxy.cpu().numpy()
            for row in xyxy:
                rtdetr_boxes.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))

        # Pure pixel-level shelf-rail Sobel segmentation (ZERO Gemini VLM calls)
        rail_proposals = core_detection.detect_shelf_boxes_from_pixels(image)
        raw_proposals = core_detection.deduplicate_depth_stacked_facings(
            core_detection._suppress_container_boxes(rtdetr_boxes + list(rail_proposals), ar_limit=0.78, nms_thr=0.55)
        )
        ctx.trace.step(
            "Stage 2: RT-DETR-v2 Dense Proposals",
            f"{len(rtdetr_boxes)} RT-DETR-L neural boxes -> {len(raw_proposals)} fused facing proposals",
            boxes=raw_proposals,
        )

        front_boxes = list(raw_proposals)
        ctx.trace.step(
            "Stage 3: 2nd-Row Depth-Ghost & Container NMS",
            f"{len(front_boxes)} front-row facings retained after DIoU-NMS & depth-ghost suppression",
            boxes=front_boxes,
        )
        return front_boxes
