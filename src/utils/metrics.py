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


def _brand_matches(pred: dict, gt: dict) -> bool:
    """Return True if ``pred`` brand matches ``gt["brand"]`` OR its legacy ``ABrand`` mapping."""
    from utils.legacy_benchmark import resolve_legacy_brand_name
    from utils.taxonomy.brands import canonicalize_brand_name

    gt_b = str(gt.get("brand") or "").strip()
    if not gt_b:
        return False
    gt_leg = str(
        gt.get("legacy_brand_name")
        or resolve_legacy_brand_name(gt_b, is_hul=gt.get("is_hul"))
    ).strip()
    gt_canon = canonicalize_brand_name(gt_b).lower()
    valid_gt = {gt_b.lower(), gt_leg.lower(), gt_canon}

    for k in ("brand", "legacy_brand_name", "product"):
        raw_p = str(pred.get(k) or "").strip()
        if not raw_p:
            continue
        # Only use `product` if it's a single brand token (no `__` separator unless `A_3_Roses`)
        if k == "product" and "__" in raw_p:
            raw_p = raw_p.split("__", 1)[0].strip()
        if raw_p.lower() in valid_gt:
            return True
        if canonicalize_brand_name(raw_p).lower() in valid_gt:
            return True
    return False


def _field_matches(pred: dict | None, gt: dict, field: str) -> bool:
    """Granularity-aware, legacy-alias-aware, and multi-modal packaging-aware field matcher."""
    if pred is None:
        return False
    p_val = pred.get(field)
    g_val = gt.get(field)
    if p_val == g_val and p_val is not None:
        return True

    if field == "brand":
        return _brand_matches(pred, gt)

    if field == "category":
        from utils.legacy_benchmark import (
            LEGACY_PACK_CATEGORY_TO_CANONICAL,
            parse_legacy_pack_folder,
        )
        from utils.taxonomy.categories import PACK_CATEGORY_ALIAS, normalize_category_name

        gt_cat = normalize_category_name(PACK_CATEGORY_ALIAS.get(str(g_val or ""), str(g_val or "")))
        if p_val:
            p_str = str(p_val).strip()
            if "__" in p_str:
                p_cat = parse_legacy_pack_folder(p_str)["category"]
            else:
                p_cat = LEGACY_PACK_CATEGORY_TO_CANONICAL.get(
                    p_str.lower(), normalize_category_name(PACK_CATEGORY_ALIAS.get(p_str, p_str))
                )
            if p_cat == gt_cat:
                return True
        for k in ("legacy_pack_folder", "product"):
            raw_lp = str(pred.get(k) or "").strip()
            if "__" in raw_lp:
                head = raw_lp.split("__", 1)[0]
                if head.lower() in LEGACY_PACK_CATEGORY_TO_CANONICAL or head in PACK_CATEGORY_ALIAS:
                    if parse_legacy_pack_folder(raw_lp)["category"] == gt_cat:
                        return True
        return False

    if field == "packaging_type":
        from utils.legacy_benchmark import parse_legacy_pack_folder
        from utils.taxonomy.packaging import (
            normalize_packaging_type,
            project_to_valid_category_pack,
        )

        gt_cat_str = str(gt.get("category") or "Foods")
        p_str = str(p_val or "").strip() if p_val and p_val != "Unknown" else ""
        is_legacy_pred = ("__" in p_str) or bool(pred.get("legacy_pack_folder"))
        allowed = set(gt.get("allowed_packaging_types") or [])
        if g_val and g_val != "Unknown":
            allowed.add(str(g_val))
            if is_legacy_pred:
                allowed.add(project_to_valid_category_pack(gt_cat_str, str(g_val)))
        candidates: list[str] = []
        if p_str:
            if "__" in p_str:
                candidates.append(parse_legacy_pack_folder(p_str)["packaging_type"])
            else:
                candidates.append(p_str)
                norm_p = normalize_packaging_type(p_str, category=gt_cat_str)
                candidates.append(norm_p)
        for k in ("legacy_pack_folder", "product"):
            raw_lp = str(pred.get(k) or "").strip()
            if "__" in raw_lp:
                head = raw_lp.split("__", 1)[0]
                from utils.legacy_benchmark import LEGACY_PACK_CATEGORY_TO_CANONICAL
                from utils.taxonomy.categories import PACK_CATEGORY_ALIAS

                if k == "legacy_pack_folder" or head.lower() in LEGACY_PACK_CATEGORY_TO_CANONICAL or head in PACK_CATEGORY_ALIAS:
                    lp_pkg = parse_legacy_pack_folder(raw_lp)["packaging_type"]
                    candidates.append(lp_pkg)
        return any(c != "Unknown" and c in allowed for c in candidates)

    if field in ("base_variant", "variant", "product"):
        from utils.legacy_benchmark import resolve_legacy_brand_name, resolve_legacy_pack_folder

        gt_bv = str(gt.get("base_variant") or "")
        gt_brand = str(gt.get("brand") or "")
        gt_prod = str(gt.get("product") or "")
        gt_leg = str(
            gt.get("legacy_brand_name")
            or (resolve_legacy_brand_name(gt_brand, is_hul=gt.get("is_hul")) if gt_brand else "")
        )
        # 1. Brand-level dataset (e.g. `hul_brands` where `product == brand` or `base_variant == brand`)
        if gt_brand and g_val in (gt_brand, gt_leg) and _brand_matches(pred, gt):
            return True
        # 2. Pack-level dataset (e.g. `hul_packs` where `product` is `<Category>__<PackType>`)
        gt_lpf = str(
            gt.get("legacy_pack_folder")
            or (
                resolve_legacy_pack_folder(str(gt.get("category") or ""), str(gt.get("packaging_type") or ""))
                if gt.get("category") and gt.get("packaging_type")
                else ""
            )
        )
        if gt_lpf and g_val == gt_lpf:
            if _field_matches(pred, gt, "category") and _field_matches(pred, gt, "packaging_type"):
                return True
        # 3. Variant/Product string with legacy `ABrand__Variant` prefix
        if p_val and g_val and "__" in str(p_val) and "__" in str(g_val):
            p_b, p_v = str(p_val).split("__", 1)
            g_b, g_v = str(g_val).split("__", 1)
            if p_v.lower() == g_v.lower() and _brand_matches({"brand": p_b}, {"brand": g_b, "is_hul": gt.get("is_hul")}):
                return True
        # 4. If ground truth is a brand-only competitor record or catch-all `<Brand>__Others`
        # and the model predicts a specific variant of the exact same brand & category, credit it.
        if (
            gt_brand
            and _brand_matches(pred, gt)
            and _field_matches(pred, gt, "category")
            and (
                gt_bv in ("Non_HUL_Competitor", "Brand_Level", f"{gt_brand}__Brand_Level_Only")
                or gt_prod.endswith("__Brand_Level")
                or bool(gt.get("is_catchall_variant"))
            )
        ):
            return True
    return False


def match_product(pred: dict | None, gt: dict, fields: Sequence[str]) -> dict:
    """Score one identified product photo (the labelled products set).

    ``correct`` says per field (product, brand, category, ...) whether the prediction agrees with the
    ground truth (supporting ``legacy_brand_name``, ``legacy_pack_folder``, multi-modal
    ``allowed_packaging_types``, and brand-only competitor GT).
    The exact product is the hit: TP if right; a wrong product is FP + FN; no
    answer is FN. So precision = right / answered, recall = right / all photos.
    """
    correct = {f: _field_matches(pred, gt, f) for f in fields}
    # "product" ranks when scored; a --fields subset without it needs every field right.
    tp = int(correct[fields[0]]) if "product" in fields else int(bool(fields) and all(correct.values()))
    return {"tp": tp, "fp": int(pred is not None) - tp, "fn": 1 - tp,
            "matched": [0] if tp else [], "correct": correct}


def hierarchical_field_scores(labels: Sequence[dict]) -> dict[str, float]:
    """Compute 5-level hierarchical taxonomy accuracy, compound accuracy, and tree-consistent credit."""
    if not labels:
        return {
            "level1_is_hul": 0.0,
            "level2_category": 0.0,
            "level3_brand": 0.0,
            "level4_base_variant": 0.0,
            "level5_packaging_type": 0.0,
            "compound_brand_category_pack": 0.0,
            "compound_all_5_levels": 0.0,
            "hierarchical_tree_score": 0.0,
        }
    n = len(labels)
    l1 = l2 = l3 = l4 = l5 = 0
    bcp = all5 = 0
    tree_sum = 0.0
    for item in labels:
        pred = item.get("pred")
        gt = item.get("gt") or {}
        if pred is None:
            continue
        ok1 = bool(pred.get("is_hul")) == bool(gt.get("is_hul"))
        ok2 = _field_matches(pred, gt, "category")
        ok3 = _field_matches(pred, gt, "brand")
        ok4 = _field_matches(pred, gt, "base_variant") if "base_variant" in gt else _field_matches(pred, gt, "product")
        ok5 = _field_matches(pred, gt, "packaging_type")
        l1 += int(ok1)
        l2 += int(ok2)
        l3 += int(ok3)
        l4 += int(ok4)
        l5 += int(ok5)
        bcp += int(ok2 and ok3 and ok5)
        all5 += int(ok1 and ok2 and ok3 and ok4 and ok5)
        # Tree-consistent credit: prefix depth along (is_hul -> category -> brand -> base_variant -> packaging_type)
        c1 = 1.0 if ok1 else 0.0
        c2 = 1.0 if (ok1 and ok2) else 0.0
        c3 = 1.0 if (ok1 and ok2 and ok3) else 0.0
        c4 = 1.0 if (ok1 and ok2 and ok3 and ok4) else 0.0
        c5 = 1.0 if (ok1 and ok2 and ok3 and ok4 and ok5) else 0.0
        tree_sum += 0.15 * c1 + 0.20 * c2 + 0.30 * c3 + 0.25 * c4 + 0.10 * c5
    return {
        "level1_is_hul": round(l1 / n, 4),
        "level2_category": round(l2 / n, 4),
        "level3_brand": round(l3 / n, 4),
        "level4_base_variant": round(l4 / n, 4),
        "level5_packaging_type": round(l5 / n, 4),
        "compound_brand_category_pack": round(bcp / n, 4),
        "compound_all_5_levels": round(all5 / n, 4),
        "hierarchical_tree_score": round(tree_sum / n, 4),
    }


def bootstrap_ci95(values: Sequence[float], n_boot: int = 500, seed: int = 42) -> dict[str, float]:
    """Deterministic paired bootstrap 95% confidence interval [low, high, half_width] over image/session scores."""
    import random

    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "ci95_half_width": 0.0}
    vals = [float(v) for v in values]
    n = len(vals)
    mean_v = sum(vals) / n
    if n == 1:
        return {
            "mean": round(mean_v, 4),
            "ci95_low": round(mean_v, 4),
            "ci95_high": round(mean_v, 4),
            "ci95_half_width": 0.0,
        }
    rng = random.Random(seed)
    means = sorted(
        sum(vals[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(max(50, n_boot))
    )
    lo = means[max(0, int(0.025 * len(means)))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return {
        "mean": round(mean_v, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "ci95_half_width": round((hi - lo) / 2.0, 4),
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
