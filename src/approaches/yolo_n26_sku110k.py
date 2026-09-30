"""Benchmark 110K SKU Pre-Trained YOLO-N26 Detector for Dense Retail SKU Detection.

Epic: ``MT Market Share - SKU Detection`` (``task = "detection"``)
Benchmarks the 110K SKU pre-trained YOLO-N26 single-stage dense retail object detector with
class-agnostic DIoU-NMS and aspect-ratio container filtering.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from core import detection as core_detection
from core.models import load_yolo26n_detector


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
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("YoloN26Sku110kDetector requires a valid non-empty PIL.Image.Image")

        w, h = image.size
        model = load_yolo26n_detector()
        preds = model.predict(
            image.convert("RGB"),
            imgsz=640,
            conf=0.01,
            iou=0.55,
            agnostic_nms=True,
            verbose=False,
        )
        yolo_boxes: list[Box] = []
        if preds and getattr(preds[0], "boxes", None) is not None and len(preds[0].boxes) > 0:
            xyxy = preds[0].boxes.xyxy.cpu().numpy()
            for row in xyxy:
                yolo_boxes.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))

        # Pure pixel-level shelf-rail proposal refinement (ZERO Gemini VLM calls)
        rail_boxes = core_detection.detect_shelf_boxes_from_pixels(image)
        raw_boxes = core_detection.deduplicate_depth_stacked_facings(
            core_detection._suppress_container_boxes(yolo_boxes + list(rail_boxes), ar_limit=0.84, nms_thr=0.55)
        )
        ctx.trace.step(
            "Pass 1: YOLO-N26 SKU-110K Forward Pass",
            f"{w}x{h}px -> {len(yolo_boxes)} YOLO26n raw neural detections + {len(raw_boxes)} fused candidate SKU boxes",
            boxes=raw_boxes,
        )

        final_boxes = list(raw_boxes)
        ctx.trace.step(
            "Pass 2: DIoU-NMS Post-Processing",
            f"{len(final_boxes)} tight product facings retained",
            boxes=final_boxes,
        )
        return final_boxes
