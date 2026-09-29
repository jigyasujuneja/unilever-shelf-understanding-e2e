"""Pillar 1 of EPIC (src/core/detection.py): High-Recall Shelf SKU Localization.

Decouples 2D bounding-box extraction (`RT-DETR-v2` / `YOLOv11` / tiled spatial proposals)
and Distance-IoU Non-Maximum Suppression (`DIoU-NMS`) from downstream vector retrieval
and VLM classification.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import BOX_LIST_SCHEMA, DETECT_PROMPT, Box, Context, to_pixels
from stages.stage2_3_detection import propose_shelf_boxes, run_post_detection
from utils import metrics


def detect_shelf_skus(
    image: Image.Image,
    ctx: Context,
    *,
    post_detector_mode: str = "shelf_rail_soft_nms",
) -> list[Box]:
    """Localize product bounding boxes on a retail shelf image with post-detection NMS."""
    raw = ctx.ask(image, DETECT_PROMPT, schema=BOX_LIST_SCHEMA, max_side=2048)
    candidates: list[Box] = to_pixels(raw.data, 0, 0, *image.size)
    filtered = run_post_detection(image, candidates, mode=post_detector_mode)
    ctx.trace.step(
        "core.detection.detect_shelf_skus",
        f"{len(candidates)} raw proposals -> {len(filtered)} shelf facings ({post_detector_mode})",
        boxes=filtered,
    )
    return filtered


def merge_overlapping_detections(boxes: list[Box], iou_threshold: float = 0.55) -> list[Box]:
    """Apply non-maximum suppression to a list of pixel-space bounding boxes."""
    if not boxes:
        return []
    kept = metrics.nms(boxes, thr=iou_threshold)
    return [boxes[i] for i in kept]


__all__ = [
    "DETECT_PROMPT",
    "detect_shelf_skus",
    "merge_overlapping_detections",
    "propose_shelf_boxes",
    "run_post_detection",
]
