"""Scoring for all 3 Shelf-Bench task types and compound attribute hierarchies:

* Task 1 (``detection``): Greedy one-to-one IoU >= 0.5 matching on bounding boxes.
* Task 2 (``classification``): Product/SKU/Variant & Compound Attribute (Category, Brand,
  Package Type, Variant, Is-HUL) scoring on ground-truth crops.
* Task 3 (``combined``): Joint end-to-end scoring where a predicted box is a True Positive
  only if IoU >= 0.5 AND its predicted product/attribute label matches ground truth.

From TP/FP/FN counts (micro-averaged across images) we report:
* precision = TP / (TP + FP)
* recall    = TP / (TP + FN)
* F2        = 5PR / (4P + R)      -- weights recall 2x: a missed facing hurts more than a stray box
* accuracy  = TP / (TP + FP + FN)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Box = tuple[float, float, float, float]  # x1, y1, x2, y2
IOU_THRESHOLD = 0.5
ATTR_KEYS = ("category", "brand", "packaging_type", "variant", "sku_id", "is_hul")


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds: Sequence[Box], gts: Sequence[Box], thr: float = IOU_THRESHOLD) -> dict:
    """Task 1 (Detection): Greedy one-to-one matching. Returns tp/fp/fn and matched pred indices."""
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
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


def _norm(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val or "").strip().lower()


def _item_to_dict(item: Any) -> dict[str, Any]:
    """Normalize either a string SKU label or a compound dict into a standard attribute dict."""
    if isinstance(item, dict):
        sku = str(item.get("sku_id") or item.get("base_pack_id") or item.get("id") or "")
        var = str(item.get("variant") or sku or "")
        return {
            "category": _norm(item.get("category") or "personal_care"),
            "brand": _norm(item.get("brand") or (sku.split("-")[2] if "-" in sku and len(sku.split("-")) > 2 else "hul")),
            "packaging_type": _norm(item.get("packaging_type") or "bottle"),
            "variant": _norm(var),
            "sku_id": _norm(sku or var),
            "is_hul": _norm(item.get("is_hul", True)),
        }
    s = str(item or "").strip()
    s_low = s.lower()
    # Infer brand/category from canonical SKU strings when ground truth is a string code
    parts = [p for p in s.replace("_", "-").split("-") if p]
    brand = parts[2].lower() if len(parts) >= 3 and parts[0].upper() in ("BP", "POSM") else (parts[1].lower() if len(parts) >= 2 else s_low)
    is_hul = not s.upper().startswith(("COMP", "BP-COMP"))
    return {
        "category": _norm("merchandising & posm" if "posm" in s_low else "personal_care"),
        "brand": _norm(brand),
        "packaging_type": _norm("shelf_strip" if "strip" in s_low else "bottle"),
        "variant": s_low,
        "sku_id": s_low,
        "is_hul": _norm(is_hul),
    }


def _labels_match(pred: Any, gt: Any, target_field: str = "variant") -> tuple[bool, dict[str, bool]]:
    p, g = _item_to_dict(pred), _item_to_dict(gt)
    per_key = {
        "category": p["category"] == g["category"] and bool(p["category"]),
        "brand": p["brand"] == g["brand"] and bool(p["brand"]),
        "packaging_type": p["packaging_type"] == g["packaging_type"] and bool(p["packaging_type"]),
        "variant": (p["sku_id"] == g["sku_id"] or p["variant"] == g["variant"]) and bool(p["sku_id"] or p["variant"]),
        "is_hul": p["is_hul"] == g["is_hul"],
    }
    per_key["compound"] = per_key["category"] and per_key["brand"] and per_key["packaging_type"]
    per_key["all_7dim"] = per_key["compound"] and per_key["variant"]
    if target_field in per_key:
        primary = per_key[target_field]
    else:
        primary = per_key["variant"]
    return primary, per_key


def match_classification(
    pred_labels: Sequence[Any],
    gt_labels: Sequence[Any],
    target_field: str = "variant",
) -> dict:
    """Task 2 (Classification & Retrieval): 1-to-1 comparison on ground-truth crops."""
    n_gt = len(gt_labels)
    matched: list[int] = []
    attr_hits = {k: 0 for k in ("category", "brand", "packaging_type", "variant", "is_hul", "compound", "all_7dim")}
    for i, (p, g) in enumerate(zip(pred_labels, gt_labels, strict=False)):
        ok, per_k = _labels_match(p, g, target_field=target_field)
        if ok:
            matched.append(i)
        for k, v in per_k.items():
            if v:
                attr_hits[k] += 1
    tp = len(matched)
    fp = max(0, len(pred_labels) - tp)
    fn = max(0, n_gt - tp)
    denom = max(1, n_gt)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched": matched,
        "attr_hits": attr_hits,
        "attr_total": n_gt,
        "attribute_accuracy": {k: round(v / denom, 4) for k, v in attr_hits.items()},
    }


def match_combined(
    preds: Sequence[Box],
    pred_labels: Sequence[Any],
    gts: Sequence[Box],
    gt_labels: Sequence[Any],
    thr: float = IOU_THRESHOLD,
    target_field: str = "variant",
) -> dict:
    """Task 3 (Combined Detection + Classification): IoU >= thr AND matching product/attribute."""
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if p[2] <= g[0] or g[2] <= p[0] or p[3] <= g[1] or g[3] <= p[1]:
                continue
            v = iou(p, g)
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    matched_joint: list[int] = []
    attr_hits = {k: 0 for k in ("category", "brand", "packaging_type", "variant", "is_hul", "compound", "all_7dim")}
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            p_lbl = pred_labels[i] if i < len(pred_labels) else ""
            g_lbl = gt_labels[j] if j < len(gt_labels) else ""
            ok, per_k = _labels_match(p_lbl, g_lbl, target_field=target_field)
            if ok:
                matched_joint.append(i)
            for k, v in per_k.items():
                if v:
                    attr_hits[k] += 1
    det_tp = len(used_p)
    tp = len(matched_joint)
    denom = max(1, det_tp)
    return {
        "tp": tp,
        "fp": len(preds) - tp,
        "fn": len(gts) - tp,
        "matched": sorted(matched_joint),
        "det_tp": det_tp,
        "det_fp": len(preds) - det_tp,
        "det_fn": len(gts) - det_tp,
        "cls_correct": tp,
        "cls_total": det_tp,
        "attr_hits": attr_hits,
        "attr_total": det_tp,
        "attribute_accuracy": {k: round(v / denom, 4) for k, v in attr_hits.items()},
    }


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
