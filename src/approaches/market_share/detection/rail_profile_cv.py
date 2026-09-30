"""Classical shelf detector, no model (Detection tab): rail and facing edges in pixel profiles.

Ported from Jigyasu's ``detect_shelf_boxes_from_pixels`` (the shelf-photo path). Shelf rails
are long horizontal edges, so peaks of the row-wise vertical-gradient profile split the photo
into shelf bands; inside each band, peaks of the column-wise horizontal-gradient + colour-change
profile split it into facings. No API call, so the only cost is Cloud Run compute: a floor for
what a model has to beat.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage, signal

from approaches.base import Approach, Box, Context, register
from utils import metrics

MODEL = "classical-cv"  # no model: numpy / scipy only
WORK_W, WORK_H = 720, 960  # profiles are computed on the photo downscaled to at most this
MAX_BOXES = 165


def rail_profile_boxes(image: Image.Image) -> tuple[list[Box], list[Box]]:
    """(facing boxes, shelf bands) in original-image pixels."""
    w, h = image.size
    ww, wh = min(WORK_W, w), min(WORK_H, h)
    arr = np.asarray(image.resize((ww, wh), Image.Resampling.BILINEAR), dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    sx, sy = w / ww, h / wh
    gy = np.abs(ndimage.sobel(gray, axis=0))
    gx = np.abs(ndimage.sobel(gray, axis=1))

    # Shelf rails: peaks of the smoothed mean |d/dy| per row.
    rows = ndimage.gaussian_filter1d(gy.mean(axis=1), sigma=max(2.0, wh * 0.012))
    peaks, _ = signal.find_peaks(rows, distance=max(14, int(wh * 0.085)), prominence=rows.std() * 0.14)
    ys = sorted({int(wh * 0.02), *(int(p) for p in peaks if wh * 0.05 < p < wh * 0.95), int(wh * 0.96)})

    boxes: list[Box] = []
    bands: list[Box] = []
    for y0, y1 in zip(ys, ys[1:], strict=False):
        band_h = y1 - y0
        py0, py1 = y0 + int(band_h * 0.06), y1 - int(band_h * 0.05)  # trim the rail itself
        if band_h < wh * 0.045 or py1 <= py0 + 4:
            continue
        bands.append((0.0, py0 * sy, float(w), py1 * sy))
        # Facing edges: peaks of mean |d/dx| + 1.5 x mean colour change per column.
        rgb = arr[py0:py1]
        colour_dx = np.abs(np.diff(rgb, axis=1, prepend=rgb[:, :1])).mean(axis=(0, 2))
        cols = ndimage.gaussian_filter1d(gx[py0:py1].mean(axis=0) + 1.5 * colour_dx,
                                         sigma=max(2.0, ww * 0.005))
        peaks, _ = signal.find_peaks(cols, distance=max(9, int(min(ww * 0.04, (py1 - py0) * 0.36))),
                                     prominence=cols.std() * 0.11)
        xs = sorted({int(ww * 0.01), *(int(p) for p in peaks if ww * 0.02 < p < ww * 0.98), int(ww * 0.99)})
        for x0, x1 in zip(xs, xs[1:], strict=False):
            if not ww * 0.02 <= x1 - x0 <= ww * 0.28:  # too thin or too wide for one facing
                continue
            patch = gray[py0:py1, x0:x1]
            if patch.mean() < 20 and patch.std() < 9:  # dark, flat: empty shelf space
                continue
            boxes.append((x0 * sx, py0 * sy, x1 * sx, py1 * sy))
    keep = metrics.nms(boxes, thr=0.55)[:MAX_BOXES]
    return [boxes[i] for i in keep], bands


@register
class RailProfileCV(Approach):
    name = "rail_profile_cv"
    models = [MODEL]
    architecture = "Classical CV: shelf rails from row edge profile, facings from column edge profile"
    steps = [
        "Downscale to at most 720x960; Sobel gradients",
        "Shelf bands: peaks of the smoothed row profile of vertical gradients",
        "Facings: peaks of the column profile of horizontal gradients + colour change in each band",
        "Drop too thin / too wide / empty-looking boxes; NMS at IoU 0.55",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        boxes, bands = rail_profile_boxes(image)
        ctx.trace.step("Shelf bands", f"{len(bands)} bands between rails", regions=bands)
        ctx.trace.step("Facings", f"{len(boxes)} boxes", boxes=boxes)
        return boxes
