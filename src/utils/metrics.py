"""Detection scoring for SKU-110K.

Per image we greedily match predictions to ground truth at IoU >= 0.5 (the standard
SKU-110K / COCO AP50 threshold), highest-IoU pairs first, one-to-one. From the TP/FP/FN
counts (summed over all images -- micro-averaged) we report:

* precision = TP / (TP + FP)
* recall    = TP / (TP + FN)
* F2        = 5PR / (4P + R)      -- weights recall 2x: a missed facing hurts more than a stray box
* accuracy  = TP / (TP + FP + FN) -- detection has no true negatives, so this is the
                                     share of all (predicted or real) boxes that were right
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Box = tuple[float, float, float, float]  # x1, y1, x2, y2
IOU_THRESHOLD = 0.5


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds: Sequence[Box], gts: Sequence[Box], thr: float = IOU_THRESHOLD) -> dict:
    """Greedy one-to-one matching. Returns tp/fp/fn and the matched pred indices."""
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            # cheap reject before computing IoU (dense scenes: ~150 x 150 pairs)
            if p[2] <= g[0] or g[2] <= p[0] or p[3] <= g[1] or g[3] <= p[1]:
                continue
            v = iou(p, g)
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    return {"tp": tp, "fp": len(preds) - tp, "fn": len(gts) - tp, "matched": sorted(used_p)}


def scores(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
    acc = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": p, "recall": r, "f2": f2, "accuracy": acc}


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100). Honest for small n: p99 of 25 values is the max."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(1, math.ceil(q / 100 * len(s)))
    return s[k - 1]


def nms(boxes: list[Box], thr: float = 0.5, scores_: list[float] | None = None) -> list[int]:
    """Plain greedy NMS; returns kept indices. Used to merge overlapping tiles."""
    order = sorted(range(len(boxes)), key=lambda i: -(scores_[i] if scores_ else 0.0))
    keep: list[int] = []
    for i in order:
        if all(iou(boxes[i], boxes[k]) < thr for k in keep):
            keep.append(i)
    return keep
