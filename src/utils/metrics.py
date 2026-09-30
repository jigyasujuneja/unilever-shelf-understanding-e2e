"""Scoring: detection on SKU-110K (below) and product identification (``match_product``).

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
    """Greedy one-to-one matching. Returns tp/fp/fn, the matched pred indices and the
    ``pairs`` {pred index: gt index}."""
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
    used: dict[int, int] = {}
    used_g = set()
    for _, i, j in pairs:
        if i not in used and j not in used_g:
            used[i] = j
            used_g.add(j)
    tp = len(used)
    return {"tp": tp, "fp": len(preds) - tp, "fn": len(gts) - tp, "matched": sorted(used),
            "pairs": used}


def match_identified(preds: Sequence[Box], pred_ids: Sequence, gts: Sequence[Box],
                     gt_ids: Sequence) -> dict:
    """End-to-end: a prediction is a TP only if it matches a GT box (IoU >= 0.5) **and** names
    that box's product. ``found`` counts box matches regardless of product."""
    m = match(preds, gts)
    hits = sorted(i for i, j in m["pairs"].items() if pred_ids[i] == gt_ids[j])
    tp = len(hits)
    return {"tp": tp, "fp": len(preds) - tp, "fn": len(gts) - tp, "matched": hits,
            "pairs": m["pairs"], "found": m["tp"]}


def match_product(pred: dict | None, gt: dict, fields: Sequence[str]) -> dict:
    """Score one identified product photo (the labelled products set).

    ``correct`` says per field (product, brand, category) whether the prediction agrees with the
    ground truth. The exact product is the hit: TP if right; a wrong product is FP + FN; no
    answer is FN. So precision = right / answered, recall = right / all photos.
    """
    correct = {f: pred is not None and pred.get(f) == gt.get(f) for f in fields}
    tp = int(correct[fields[0]])
    return {"tp": tp, "fp": int(pred is not None) - tp, "fn": 1 - tp,
            "matched": [0] if tp else [], "correct": correct}


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
