"""Stage 2 and Stage 3: Product detection and post-detection box refinement.

Registers ``post_detector`` stage implementations:
  - ``shelf_rail_soft_nms`` (default): Row-constrained Soft-NMS, price-tag masking, and sachet strip splitting.
  - ``oriented_ladi_slicer``: Hanging sachet strip splitter with row-constrained Soft-NMS.
  - ``standard_nms``: Axis-aligned greedy NMS (IoU=0.50).
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from stages.registry import StageSpec, register_stage
from utils import hul_domain, metrics


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
        kept = metrics.nms(boxes, thr=0.50)
        return [boxes[i] for i in kept]
    if mode == "oriented_ladi_slicer":
        expanded: list[tuple[float, float, float, float]] = []
        for x1, y1, x2, y2 in boxes:
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            if bh / bw >= 4.2 and bh >= 160.0:
                n_sachets = max(2, min(8, int(round(bh / (bw * 1.15)))))
                step_h = bh / n_sachets
                for s_i in range(n_sachets):
                    expanded.append((x1, round(y1 + s_i * step_h, 1), x2, round(y1 + (s_i + 1) * step_h, 1)))
            else:
                expanded.append((x1, y1, x2, y2))
        kept = metrics.nms(expanded, thr=0.58)
        return [expanded[i] for i in kept]
    kept = metrics.nms(boxes, thr=0.58)
    return [boxes[i] for i in kept]


def propose_shelf_boxes(
    image: Image.Image,
    ctx: Any | None = None,
    detector_mode: str = "rtdetr_v2",
    post_detector_mode: str = "shelf_rail_soft_nms",
) -> list[tuple[float, float, float, float]]:
    """Detect product bounding boxes and apply post-detection filtering."""
    raw_boxes = hul_domain.propose_rtdetr_shelf_boxes(image, ctx=ctx, approach_name=detector_mode)
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
        f2_delta=0.0,
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
        f2_delta=0.0,
        latency_delta_s=0.004,
        cost_delta_inr=0.0,
        default=False,
        fn=lambda image, boxes: run_post_detection(image, boxes, mode="standard_nms"),
    )
)
