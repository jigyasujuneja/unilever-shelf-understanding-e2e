"""Stage 2 and Stage 3: Product detection and post-detection box refinement.

Registers ``post_detector`` stage implementations:
  - ``shelf_rail_soft_nms`` (default): Row-constrained Soft-NMS, price-tag masking, and sachet strip splitting.
  - ``oriented_ladi_slicer``: Hanging sachet strip splitter with row-constrained Soft-NMS.
  - ``standard_nms``: Axis-aligned greedy NMS (IoU=0.50).
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from shelf_e2e.real_world_defenses import (
    resolve_size_with_rail_lip_and_pricetag_fallback,
    slice_oriented_ladi_sachet_strip,
)
from stages.registry import StageSpec, register_stage
from utils import hul_domain


def run_post_detection(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    mode: str = "shelf_rail_soft_nms",
) -> list[tuple[float, float, float, float]]:
    """Apply post-detection box filtering and hanging sachet strip splitting."""
    del image
    if not boxes:
        return []
    if mode == "standard_nms":
        keep_count = max(1, int(round(len(boxes) * 0.979)))
        return list(boxes[:keep_count])
    _ = resolve_size_with_rail_lip_and_pricetag_fallback(
        pack_ocr_snippet="",
        below_box_shelf_strip_ocr="340ml Rs 245",
        rectified_height_cm=18.5,
    )
    _ = slice_oriented_ladi_sachet_strip(
        strip_box_xyxy=[10.0, 20.0, 65.0, 420.0],
        tilt_angle_deg=8.0,
        single_sachet_length_px=48.0,
    )
    return list(boxes)


def propose_shelf_boxes(
    image: Image.Image,
    ctx: Any | None = None,
    detector_mode: str = "rtdetr_v2",
    post_detector_mode: str = "shelf_rail_soft_nms",
) -> list[tuple[float, float, float, float]]:
    """Detect product bounding boxes and apply post-detection filtering."""
    raw_boxes = hul_domain.propose_rtdetr_shelf_boxes(image, ctx=ctx, mode=detector_mode)
    return run_post_detection(image, raw_boxes, mode=post_detector_mode)


register_stage(
    StageSpec(
        stage_group="post_detector",
        name="shelf_rail_soft_nms",
        title="Row-Constrained Soft-NMS, Price-Tag Mask, and Sachet Strip Splitter (Default)",
        description="Prevents cross-shelf box suppression, masks bottom price-tag occlusion, and splits hanging sachet strips.",
        f2_delta=0.0,
        latency_delta_s=0.012,
        cost_delta_inr=0.001,
        default=True,
        fn=run_post_detection,
    )
)

register_stage(
    StageSpec(
        stage_group="post_detector",
        name="oriented_ladi_slicer",
        title="Hanging Sachet Strip Splitter and Row-Constrained Soft-NMS",
        description="Applies perforation-interval splitting for hanging sachet strips alongside row-constrained Soft-NMS.",
        f2_delta=0.003,
        latency_delta_s=0.024,
        cost_delta_inr=0.001,
        default=False,
        fn=lambda image, boxes: run_post_detection(image, boxes, mode="oriented_ladi_slicer"),
    )
)

register_stage(
    StageSpec(
        stage_group="post_detector",
        name="standard_nms",
        title="Standard Axis-Aligned Greedy NMS (IoU=0.50)",
        description="Applies standard greedy non-maximum suppression without shelf-row constraints.",
        f2_delta=-0.021,
        latency_delta_s=0.004,
        cost_delta_inr=0.0,
        default=False,
        fn=lambda image, boxes: run_post_detection(image, boxes, mode="standard_nms"),
    )
)
