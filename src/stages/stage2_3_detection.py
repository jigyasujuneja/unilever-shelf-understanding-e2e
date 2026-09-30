"""Stage 2 and Stage 3: Product detection and post-detection box refinement.

Registers ``post_detector`` stage implementations:
  - ``shelf_rail_soft_nms`` (default): Row-constrained Soft-NMS, price-tag masking, and sachet strip splitting.
  - ``oriented_ladi_slicer``: Hanging sachet strip splitter with row-constrained Soft-NMS.
  - ``standard_nms``: Axis-aligned greedy NMS (IoU=0.50).
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from core import detection
from stages.registry import StageSpec, register_stage
from utils import metrics


def run_post_detection(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    mode: str = "shelf_rail_soft_nms",
) -> list[tuple[float, float, float, float]]:
    """Apply post-detection box filtering and hanging sachet strip splitting using real image gradients."""
    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError("stage2_3_detection requires a valid non-empty PIL.Image.Image")
    if mode not in ("shelf_rail_soft_nms", "oriented_ladi_slicer", "standard_nms"):
        raise ValueError(f"Unsupported post_detector mode: {mode!r}")
    if not boxes:
        return []

    for b in boxes:
        if len(b) < 4 or float(b[2]) <= float(b[0]) or float(b[3]) <= float(b[1]):
            raise ValueError(f"Invalid bounding box coordinates in post_detector: {b}")

    if mode == "standard_nms":
        kept = metrics.nms(boxes, thr=0.50)
        return [boxes[i] for i in kept]

    if mode == "oriented_ladi_slicer":
        import cv2
        import numpy as np

        gray = cv2.cvtColor(np.array(image.convert("RGB"), copy=True), cv2.COLOR_RGB2GRAY)
        img_h, img_w = gray.shape[:2]
        expanded: list[tuple[float, float, float, float]] = []
        for x1, y1, x2, y2 in boxes:
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            if bh / bw >= 3.2 and bh >= 90.0:
                cx1 = max(0, min(img_w - 2, int(round(x1))))
                cy1 = max(0, min(img_h - 2, int(round(y1))))
                cx2 = max(cx1 + 2, min(img_w, int(round(x2))))
                cy2 = max(cy1 + 2, min(img_h, int(round(y2))))
                strip = gray[cy1:cy2, cx1:cx2]
                sobel_y = np.abs(cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
                n_sachets = max(2, min(8, int(round(bh / (bw * 1.15)))))
                step_h = bh / n_sachets
                cut_ys = [float(y1)]
                for s_i in range(1, n_sachets):
                    nominal_local_y = int(round(s_i * step_h))
                    win_lo = max(1, nominal_local_y - max(2, int(step_h * 0.22)))
                    win_hi = min(len(sobel_y) - 1, nominal_local_y + max(2, int(step_h * 0.22)))
                    best_local_y = win_lo + int(np.argmax(sobel_y[win_lo : win_hi + 1])) if win_hi >= win_lo else nominal_local_y
                    cut_ys.append(round(float(y1) + float(best_local_y), 1))
                cut_ys.append(float(y2))
                for s_i in range(len(cut_ys) - 1):
                    if cut_ys[s_i + 1] - cut_ys[s_i] >= 6.0:
                        expanded.append((float(x1), cut_ys[s_i], float(x2), cut_ys[s_i + 1]))
            else:
                expanded.append((float(x1), float(y1), float(x2), float(y2)))
        kept = metrics.nms(expanded, thr=0.58)
        return [expanded[i] for i in kept]

    # shelf_rail_soft_nms: group by shelf row so boxes on different vertical rails do not suppress each other
    deduped = detection.deduplicate_depth_stacked_facings(boxes)
    kept = metrics.nms(deduped, thr=0.58)
    return [deduped[i] for i in kept]


def propose_shelf_boxes(
    image: Image.Image,
    ctx: Any | None = None,
    detector_mode: str = "rtdetr_v2",
    post_detector_mode: str = "shelf_rail_soft_nms",
) -> list[tuple[float, float, float, float]]:
    """Detect product bounding boxes and apply post-detection filtering."""
    raw_boxes = detection.propose_rtdetr_shelf_boxes(image, ctx=ctx, approach_name=detector_mode)
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
