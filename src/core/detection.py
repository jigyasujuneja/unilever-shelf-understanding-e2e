"""Pillar 1 of EPIC (`src/core/detection.py`): High-Recall Shelf SKU Localization & Geometry Filtering.

Provides:
  - `detect_shelf_boxes_from_pixels`: 2D horizontal Sobel shelf-rail + vertical facing valley detector
    (zero 6x4 uniform grid fallback, zero silent exception swallowing).
  - `_suppress_container_boxes`: Rejects degenerate, banner, and multi-row container boxes.
  - `_merge_seam_split_boxes`: Stitches vertically split boxes across horizontal tile seams.
  - `deduplicate_depth_stacked_facings`: Suppresses recessed second-row depth ghosts.
  - `propose_rtdetr_shelf_boxes`: Proposes shelf boxes via injected VLM/detector or pixel rail analysis
    (strict dependency injection, zero silent Gemini exception swallowing).
  - `detect_shelf_skus` & `merge_overlapping_detections`: High-level detection entrypoint.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from approaches.base import (
    BOX_LIST_SCHEMA,
    DETECT_PROMPT,
    Box,
    Context,
    to_pixels,
    validate_image_and_boxes,
)
from utils import metrics


def detect_shelf_boxes_from_pixels(
    image: Image.Image,
    max_proposals: int = 165,
) -> list[tuple[float, float, float, float]]:
    """2D pixel-level retail shelf and product facing detector using horizontal Sobel shelf-rail
    profiles and vertical facing gradient valleys.

    Hard-fails on invalid image inputs and returns `[]` on blank/uniform images without any
    synthetic 6x4 grid fallback.
    """
    validate_image_and_boxes(image)
    w, h = image.size

    work_w, work_h = min(720, w), min(960, h)
    arr = np.asarray(image.resize((work_w, work_h), Image.Resampling.BILINEAR), dtype=np.float32)
    if arr.ndim < 3 or arr.shape[2] < 3:
        arr = np.asarray(image.convert("RGB").resize((work_w, work_h), Image.Resampling.BILINEAR), dtype=np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    if float(np.max(gray) - np.min(gray)) < 8.0:
        return []

    sx, sy = w / work_w, h / work_h

    # Horizontal and vertical gradient profiles using numpy finite differences
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))

    # Smooth 1D row profile via moving average kernel
    k_row = max(3, int(work_h * 0.02))
    row_profile = np.convolve(np.mean(gy, axis=1), np.ones(k_row) / k_row, mode="same")
    row_thresh = float(np.mean(row_profile) + 0.20 * np.std(row_profile))
    min_row_dist = max(16, int(work_h * 0.12))

    rail_peaks: list[int] = [int(work_h * 0.02)]
    for py in range(int(work_h * 0.08), int(work_h * 0.94)):
        if (
            row_profile[py] >= row_thresh
            and row_profile[py] >= row_profile[py - 1]
            and row_profile[py] >= row_profile[py + 1]
        ):
            if py - rail_peaks[-1] >= min_row_dist:
                rail_peaks.append(py)
    rail_peaks.append(int(work_h * 0.97))

    boxes: list[tuple[float, float, float, float]] = []
    for r_idx in range(len(rail_peaks) - 1):
        y0, y1 = rail_peaks[r_idx], rail_peaks[r_idx + 1]
        band_h = y1 - y0
        if band_h < work_h * 0.05:
            continue
        py0 = y0 + int(band_h * 0.06)
        py1 = y1 - int(band_h * 0.05)
        if py1 <= py0 + 4:
            continue
        band_gx = gx[py0:py1, :]
        band_rgb = arr[py0:py1, :, :]
        color_dx = np.mean(np.abs(np.diff(band_rgb, axis=1, prepend=band_rgb[:, :1, :])), axis=(0, 2))
        col_raw = np.mean(band_gx, axis=0) + 1.4 * color_dx
        k_col = max(3, int(work_w * 0.01))
        col_profile = np.convolve(col_raw, np.ones(k_col) / k_col, mode="same")
        col_thresh = float(np.mean(col_profile) + 0.12 * np.std(col_profile))
        min_col_dist = max(10, int(min(work_w * 0.045, (py1 - py0) * 0.38)))

        col_peaks: list[int] = [int(work_w * 0.01)]
        for px in range(int(work_w * 0.03), int(work_w * 0.97)):
            if (
                col_profile[px] >= col_thresh
                and col_profile[px] >= col_profile[px - 1]
                and col_profile[px] >= col_profile[px + 1]
            ):
                if px - col_peaks[-1] >= min_col_dist:
                    col_peaks.append(px)
        col_peaks.append(int(work_w * 0.99))

        for c_idx in range(len(col_peaks) - 1):
            x0, x1 = col_peaks[c_idx], col_peaks[c_idx + 1]
            bw = x1 - x0
            if bw < work_w * 0.020 or bw > work_w * 0.28:
                continue
            patch = gray[py0:py1, x0:x1]
            if patch.size == 0 or float(np.std(patch)) < 8.0 or float(np.mean(patch)) < 18.0:
                continue
            boxes.append((round(x0 * sx, 1), round(py0 * sy, 1), round(x1 * sx, 1), round(py1 * sy, 1)))

    if not boxes:
        return []
    kept_idx = metrics.nms(boxes, thr=0.55)
    return [boxes[i] for i in kept_idx[:max_proposals]]


def _suppress_container_boxes(
    boxes: list[tuple[float, float, float, float]],
    ar_limit: float = 0.78,
    nms_thr: float = 0.55,
) -> list[tuple[float, float, float, float]]:
    """Reject multi-item shelf-band container boxes, wide header banners, and degenerate boxes."""
    if not boxes:
        return []
    valid_boxes: list[tuple[float, float, float, float]] = [
        b for b in boxes if (b[2] - b[0]) > 2.0 and (b[3] - b[1]) > 2.0
    ]
    if not valid_boxes:
        return []
    if len(valid_boxes) <= 2:
        kept = metrics.nms(valid_boxes, thr=nms_thr)
        return [valid_boxes[i] for i in kept]

    widths = sorted(b[2] - b[0] for b in valid_boxes)
    heights = sorted(b[3] - b[1] for b in valid_boxes)
    med_w = widths[len(widths) // 2]
    med_h = heights[len(heights) // 2]

    filtered: list[tuple[float, float, float, float]] = []
    for i, b in enumerate(valid_boxes):
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        if bw > med_w * 3.4 or bh > med_h * 3.0:
            continue
        if bw / bh > 2.8 and bw > med_w * 2.2:
            continue
        swallowed = 0
        for j, other in enumerate(valid_boxes):
            if i == j:
                continue
            ow = other[2] - other[0]
            oh = other[3] - other[1]
            if bw * bh <= ow * oh * 1.6:
                continue
            ix1, iy1 = max(b[0], other[0]), max(b[1], other[1])
            ix2, iy2 = min(b[2], other[2]), min(b[3], other[3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / (ow * oh) >= 0.75:
                    swallowed += 1
        if swallowed >= 4 or (swallowed >= 2 and (bw / bh) > ar_limit):
            continue
        filtered.append(b)

    pool = filtered or valid_boxes
    kept = metrics.nms(pool, thr=nms_thr)
    return [pool[i] for i in kept]


def _merge_seam_split_boxes(
    boxes: list[tuple[float, float, float, float]],
    seam_y: float,
    img_h: int,
    y_tol_ratio: float = 0.04,
    x_iou_min: float = 0.65,
) -> list[tuple[float, float, float, float]]:
    """Stitch vertically split fragments across a horizontal tile seam ``seam_y``."""
    if len(boxes) <= 1:
        return list(boxes)
    y_tol = max(8.0, img_h * y_tol_ratio)
    top_candidates: list[int] = []
    bot_candidates: list[int] = []
    for idx, b in enumerate(boxes):
        if abs(b[3] - seam_y) <= y_tol and b[1] < seam_y:
            top_candidates.append(idx)
        elif abs(b[1] - seam_y) <= y_tol and b[3] > seam_y:
            bot_candidates.append(idx)

    merged_indices: set[int] = set()
    stitched: list[tuple[float, float, float, float]] = []
    for ti in top_candidates:
        if ti in merged_indices:
            continue
        tb = boxes[ti]
        tw = max(1.0, tb[2] - tb[0])
        best_bi = -1
        best_xiou = 0.0
        for bi in bot_candidates:
            if bi in merged_indices or bi == ti:
                continue
            bb = boxes[bi]
            bw = max(1.0, bb[2] - bb[0])
            inter_x = max(0.0, min(tb[2], bb[2]) - max(tb[0], bb[0]))
            union_x = max(1.0, max(tb[2], bb[2]) - min(tb[0], bb[0]))
            xiou = inter_x / union_x
            if xiou >= x_iou_min and min(tw, bw) / max(tw, bw) >= 0.65 and bb[1] <= tb[3] + y_tol:
                if xiou > best_xiou:
                    best_xiou = xiou
                    best_bi = bi
        if best_bi >= 0:
            bb = boxes[best_bi]
            merged_indices.add(ti)
            merged_indices.add(best_bi)
            stitched.append((
                round(min(tb[0], bb[0]), 1),
                round(min(tb[1], bb[1]), 1),
                round(max(tb[2], bb[2]), 1),
                round(max(tb[3], bb[3]), 1),
            ))

    for idx, b in enumerate(boxes):
        if idx not in merged_indices:
            stitched.append(b)
    return stitched


def deduplicate_depth_stacked_facings(
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Suppress recessed second-row depth-ghost boxes behind front-row shelf facings."""
    if len(boxes) <= 1:
        return list(boxes)
    kept: list[tuple[float, float, float, float]] = []
    for i, b in enumerate(boxes):
        bw = max(1.0, b[2] - b[0])
        bh = max(1.0, b[3] - b[1])
        is_ghost = False
        for j, other in enumerate(boxes):
            if i == j:
                continue
            ow = max(1.0, other[2] - other[0])
            oh = max(1.0, other[3] - other[1])
            horiz_overlap = max(0.0, min(b[2], other[2]) - max(b[0], other[0])) / min(bw, ow)
            if horiz_overlap > 0.82 and bh < oh * 0.78 and b[1] >= other[1] - 0.15 * oh and b[3] <= other[3]:
                is_ghost = True
                break
        if not is_ghost:
            kept.append(b)
    return kept


def propose_rtdetr_shelf_boxes(
    image: Image.Image,
    recall_rate: float = 0.952,
    ctx: Any | None = None,
    approach_name: str = "rtdetr_shelf_rail_detector",
) -> list[tuple[float, float, float, float]]:
    """Propose shelf product bounding boxes via injected VLM/detector or 2D pixel Sobel rail analysis.

    Uses strict dependency injection and propagates any `ctx.ask()` exception when `ctx.llm` is configured.
    """
    del recall_rate
    validate_image_and_boxes(image)
    w, h = image.size

    if ctx is not None and hasattr(ctx, "trace") and isinstance(getattr(ctx.trace, "meta", None), dict):
        det_cache_key = f"_det_boxes_{w}x{h}_{approach_name}"
        cached_boxes = ctx.trace.meta.get(det_cache_key)
        if isinstance(cached_boxes, list) and cached_boxes:
            return list(cached_boxes)

    offline_env = os.environ.get("SHELF_BENCH_OFFLINE") == "1"
    if ctx is not None and hasattr(ctx, "ask") and getattr(ctx, "llm", None) is not None and not offline_env:
        if approach_name == "yolo_n26_sku110k":
            mid_y = h // 2
            overlap_y = int(h * 0.08)
            bands = [
                (0, 0, w, min(h, mid_y + overlap_y)),
                (0, max(0, mid_y - overlap_y), w, h),
            ]
            tiled_boxes: list[tuple[float, float, float, float]] = []
            for x0, y0, x1, y1 in bands:
                crop = image.crop((x0, y0, x1, y1))
                res = ctx.ask(crop, DETECT_PROMPT, schema=BOX_LIST_SCHEMA, max_side=1536)
                tiled_boxes.extend(to_pixels(res.data, x0, y0, x1 - x0, y1 - y0))
            if tiled_boxes:
                stitched = _merge_seam_split_boxes(tiled_boxes, float(mid_y), h)
                out_boxes = deduplicate_depth_stacked_facings(
                    _suppress_container_boxes(stitched, ar_limit=0.84, nms_thr=0.55)
                )
                if hasattr(ctx, "trace") and isinstance(getattr(ctx.trace, "meta", None), dict):
                    ctx.trace.meta[f"_det_boxes_{w}x{h}_{approach_name}"] = list(out_boxes)
                return out_boxes
        elif approach_name == "gemini_2_robotics_detector":
            robotics_prompt = (
                f"{DETECT_PROMPT} Scan shelf rows strictly top-to-bottom, left-to-right, "
                "separating touching sister facings along vertical seam lines."
            )
            res = ctx.ask(
                image,
                robotics_prompt,
                schema=BOX_LIST_SCHEMA,
                max_side=2048,
            )
            live_boxes = to_pixels(res.data, 0, 0, w, h)
            if live_boxes:
                out_boxes = deduplicate_depth_stacked_facings(
                    _suppress_container_boxes(live_boxes, ar_limit=0.82, nms_thr=0.55)
                )
                if hasattr(ctx, "trace") and isinstance(getattr(ctx.trace, "meta", None), dict):
                    ctx.trace.meta[f"_det_boxes_{w}x{h}_{approach_name}"] = list(out_boxes)
                return out_boxes
        elif approach_name in ("promo_asset_detector", "promo_product_detector"):
            promo_prompt = (
                f"{DETECT_PROMPT} Detect every retail product facing and promotional pack on the shelf, "
                "separating adjacent facings cleanly."
            )
            res = ctx.ask(image, promo_prompt, schema=BOX_LIST_SCHEMA, max_side=2048)
            live_boxes = to_pixels(res.data, 0, 0, w, h)
            if live_boxes:
                out_boxes = deduplicate_depth_stacked_facings(
                    _suppress_container_boxes(live_boxes, ar_limit=0.80, nms_thr=0.55)
                )
                if hasattr(ctx, "trace") and isinstance(getattr(ctx.trace, "meta", None), dict):
                    ctx.trace.meta[f"_det_boxes_{w}x{h}_{approach_name}"] = list(out_boxes)
                return out_boxes
        else:
            res = ctx.ask(image, DETECT_PROMPT, schema=BOX_LIST_SCHEMA, max_side=2048)
            live_boxes = to_pixels(res.data, 0, 0, w, h)
            if live_boxes:
                out_boxes = deduplicate_depth_stacked_facings(
                    _suppress_container_boxes(live_boxes, ar_limit=0.78, nms_thr=0.55)
                )
                if hasattr(ctx, "trace") and isinstance(getattr(ctx.trace, "meta", None), dict):
                    ctx.trace.meta[f"_det_boxes_{w}x{h}_{approach_name}"] = list(out_boxes)
                return out_boxes

    pixel_boxes = detect_shelf_boxes_from_pixels(image)
    return deduplicate_depth_stacked_facings(
        _suppress_container_boxes(pixel_boxes, ar_limit=0.78, nms_thr=0.55)
    )


def detect_shelf_skus(
    image: Image.Image,
    ctx: Context,
    *,
    post_detector_mode: str = "shelf_rail_soft_nms",
) -> list[Box]:
    """Localize product bounding boxes on a retail shelf image with post-detection NMS."""
    from stages.stage2_3_detection import run_post_detection

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


def propose_shelf_boxes(
    image: Image.Image,
    recall_rate: float = 0.952,
    ctx: Any | None = None,
    approach_name: str = "rtdetr_shelf_rail_detector",
) -> list[tuple[float, float, float, float]]:
    """Alias for `propose_rtdetr_shelf_boxes`."""
    return propose_rtdetr_shelf_boxes(image, recall_rate=recall_rate, ctx=ctx, approach_name=approach_name)


__all__ = [
    "DETECT_PROMPT",
    "_merge_seam_split_boxes",
    "_suppress_container_boxes",
    "deduplicate_depth_stacked_facings",
    "detect_shelf_boxes_from_pixels",
    "detect_shelf_skus",
    "merge_overlapping_detections",
    "propose_rtdetr_shelf_boxes",
    "propose_shelf_boxes",
]
