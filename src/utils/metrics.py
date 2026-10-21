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


def match_identified(
    preds: Sequence[Box],
    pred_ids: Sequence,
    gts: Sequence[Box],
    gt_ids: Sequence,
    *,
    catalog: dict[int, dict] | None = None,
    gt_labels: Sequence[dict] | None = None,
    fields: Sequence[str] | None = ("product",),
) -> dict:
    """End-to-end: a prediction is a TP only if it matches a GT box (IoU >= 0.5) **and** names
    that box's product. ``found`` counts box matches regardless of product."""
    resolved_fields = fields or ("product",)
    m = match(preds, gts)
    hits: list[int] = []
    for i, j in sorted(m["pairs"].items()):
        if pred_ids[i] == gt_ids[j]:
            hits.append(i)
        elif catalog is not None and gt_labels is not None and j < len(gt_labels):
            pred_entry = catalog.get(pred_ids[i]) if pred_ids[i] is not None else None
            if pred_entry is not None and match_product(pred_entry, gt_labels[j], resolved_fields)["tp"] > 0:
                hits.append(i)
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


def _variant_tokens_match(p_var: str, g_var: str, brand: str = "") -> bool:
    """Compare predicted vs GT variant strings across ``gt-mapping.yml`` and v2.2 folder conventions."""
    import re

    from utils.taxonomy.filenames import STRIP_PROMO_ART_SUFFIX_RE

    if not p_var or not g_var:
        return False

    def _extract_var(s: str) -> str:
        s = s.strip()
        parts = s.split("__")
        if len(parts) >= 5 and parts[0].upper() in ("HUL", "NON_HUL"):
            return "__".join(parts[4:])
        if len(parts) == 2:
            return parts[1]
        return s

    pv = _extract_var(str(p_var))
    gv = _extract_var(str(g_var))
    if pv.lower() == gv.lower():
        return True

    b_alnum = re.sub(r"[^a-z0-9]+", "", str(brand or "").replace("&", "and").lower())
    subcat_re = re.compile(
        r"^(?:shampoo|conditioner|toothpaste|paste|soaps?|bodywash|body|facewash|face|handwash|wash|hair|oil)$"
    )

    def _norm(v: str, strip_subcat: bool = False) -> str:
        base = STRIP_PROMO_ART_SUFFIX_RE.sub("", v) or v
        toks = [t for t in re.split(r"[^a-z0-9]+", base.replace("&", "and").lower()) if t]
        if strip_subcat and len(toks) > 1:
            filtered = [t for t in toks if not subcat_re.match(t)]
            if filtered:
                toks = filtered
        joined = "".join(toks)
        if b_alnum and joined.startswith(b_alnum) and len(joined) > len(b_alnum):
            joined = joined[len(b_alnum) :]
        return joined

    if _norm(pv, False) and _norm(pv, False) == _norm(gv, False):
        return True
    if _norm(pv, True) and _norm(pv, True) == _norm(gv, True):
        return True
    return False


def _pred_is_hul(pred: dict | None) -> bool:
    """Resolve HUL ownership from pred['is_hul'], falling back to canonical brand lookup when omitted."""
    if pred is None:
        return False
    raw_flag = pred.get("is_hul")
    if isinstance(raw_flag, bool):
        return raw_flag
    if raw_flag is not None and str(raw_flag).strip() != "":
        return str(raw_flag).strip().lower() in ("true", "1", "yes")
    from utils.taxonomy.brands import HUL_BRANDS, canonicalize_brand_name

    b_raw = str(pred.get("brand") or pred.get("legacy_brand_name") or "").strip()
    if not b_raw and pred.get("product") and "__" in str(pred.get("product")):
        b_raw = str(pred.get("product")).split("__", 1)[0].strip()
    if not b_raw:
        return False
    return canonicalize_brand_name(b_raw).lower() in HUL_BRANDS


def _field_matches(pred: dict | None, gt: dict, field: str) -> bool:
    """Granularity-aware, legacy-alias-aware, and multi-modal packaging-aware field matcher."""
    if pred is None:
        return False
    p_val = pred.get(field)
    g_val = gt.get(field)
    if p_val == g_val and p_val is not None:
        return True

    # Unbranded outside-bay product boxes in GT merchandising annotations (`product` with no brand/SKU):
    # credit any recognized SKU/brand prediction.
    if bool(gt.get("is_unbranded_outside_bay")):
        return pred.get("sku_id") is not None or pred.get("brand") is not None

    # Open-set competitor catch-all shelf boxes (`Others____Others____NA` in LabelMe annotations):
    # strictly require the model to predict a Non-HUL competitor (`is_hul is False`) when GT is not a backfilled HUL item.
    if bool(gt.get("is_others_catchall")) and not bool(gt.get("is_hul")):
        return _pred_is_hul(pred) is False

    if field == "is_hul":
        return _pred_is_hul(pred) == bool(g_val)

    is_bay_sister_match = (
        bool(gt.get("is_bay_inferred"))
        and _pred_is_hul(pred) == bool(gt.get("is_hul"))
        and (
            (pred.get("category") is not None and pred.get("category") == gt.get("category"))
            or (pred.get("packaging_type") is not None and pred.get("packaging_type") == gt.get("packaging_type"))
        )
    )

    if field == "brand":
        return _brand_matches(pred, gt) or is_bay_sister_match

    if bool(gt.get("is_catchall_variant")) and (_brand_matches(pred, gt) or is_bay_sister_match):
        if field in ("category", "packaging_type", "base_variant", "variant", "product"):
            return True

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
        # 3. Variant/Product string with legacy `ABrand__Variant` prefix or `gt-mapping.yml` 5-part label
        if p_val and g_val and "__" in str(p_val) and "__" in str(g_val):
            p_b, p_v = str(p_val).split("__", 1)
            g_b, g_v = str(g_val).split("__", 1)
            if (p_v.lower() == g_v.lower() or _variant_tokens_match(p_v, g_v, g_b)) and _brand_matches(
                {"brand": p_b}, {"brand": g_b, "is_hul": gt.get("is_hul")}
            ):
                return True
        if _brand_matches(pred, gt):
            p_cands = [pred.get(k) for k in (field, "base_variant", "variant", "gt_variant", "product") if pred.get(k)]
            g_cands = [gt.get(k) for k in (field, "base_variant", "variant", "product") if gt.get(k)]
            for pc in p_cands:
                for gc in g_cands:
                    if _variant_tokens_match(str(pc), str(gc), gt_brand):
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
    Every evaluated field in ``fields`` must match for ``tp = 1``; a partial or wrong answer is
    FP + FN, and no answer (``pred is None``) is FN.
    """
    correct = {f: _field_matches(pred, gt, f) for f in fields}
    tp = int(bool(fields) and all(correct.values()))
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
        ok1 = (_pred_is_hul(pred) == bool(gt.get("is_hul"))) or bool(gt.get("is_others_catchall"))
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


def group_annotated_shelf_bands(
    gt_boxes: Sequence[Box],
    y_pad_factor: float = 0.35,
    x_pad_factor: float = 0.35,
    x_gap_factor: float = 1.25,
) -> list[tuple[float, ...]]:
    """Cluster human-annotated GT boxes into 2D shelf-row segments ``[(y_min, y_max, x_min, x_max), ...]``.

    In ``ms_json_golden`` (``gs://ms-json-dataset``), human annotators boxed only the target
    audited pockets within each shelf row and left adjacent horizontal bays and other rows blank.
    Grouping GT boxes first by vertical shelf plank ``cy`` and then merging adjacent horizontal
    facings within ``1.25 * median_width`` isolates the exact human-annotated shelf zones.
    """
    if not gt_boxes:
        return []
    ordered = sorted(gt_boxes, key=lambda b: (b[1] + b[3]) / 2.0)
    planks: list[list[Box]] = []
    for b in ordered:
        cy = (b[1] + b[3]) / 2.0
        bh = max(1.0, b[3] - b[1])
        placed = False
        for p in planks:
            p_cy = sum((x[1] + x[3]) / 2.0 for x in p) / len(p)
            p_bh = sum(max(1.0, x[3] - x[1]) for x in p) / len(p)
            if abs(cy - p_cy) <= 0.60 * max(bh, p_bh):
                p.append(b)
                placed = True
                break
        if not placed:
            planks.append([b])

    segments: list[tuple[float, ...]] = []
    for p in planks:
        heights = sorted(max(1.0, x[3] - x[1]) for x in p)
        widths = sorted(max(1.0, x[2] - x[0]) for x in p)
        med_h = heights[len(heights) // 2]
        med_w = widths[len(widths) // 2]
        y_pad = y_pad_factor * med_h
        x_pad = x_pad_factor * med_w
        max_x_gap = x_gap_factor * med_w

        row_sorted = sorted(p, key=lambda x: x[0])
        clusters: list[list[Box]] = [[row_sorted[0]]]
        cur_x_max = row_sorted[0][2]
        for b in row_sorted[1:]:
            if b[0] - cur_x_max <= max_x_gap:
                clusters[-1].append(b)
                cur_x_max = max(cur_x_max, b[2])
            else:
                clusters.append([b])
                cur_x_max = b[2]

        for cl in clusters:
            y_min = max(0.0, min(x[1] for x in cl) - y_pad)
            y_max = max(x[3] for x in cl) + y_pad
            x_min = max(0.0, min(x[0] for x in cl) - x_pad)
            x_max = max(x[2] for x in cl) + x_pad
            segments.append((round(y_min, 2), round(y_max, 2), round(x_min, 2), round(x_max, 2)))

    segments.sort()
    return segments


def _box_in_any_band(box: Box, bands: Sequence[tuple[float, ...]]) -> bool:
    """Return True if ``box`` falls inside any 2D shelf-row segment (or 1D band)."""
    if not bands:
        return True
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    for band in bands:
        y0, y1 = band[0], band[1]
        y_ok = (y0 <= cy <= y1) or (max(0.0, min(box[3], y1) - max(box[1], y0)) / bh >= 0.50)
        if not y_ok:
            continue
        if len(band) < 4:
            return True
        x0, x1 = band[2], band[3]
        x_ok = (x0 <= cx <= x1) or (max(0.0, min(box[2], x1) - max(box[0], x0)) / bw >= 0.50)
        if x_ok:
            return True
    return False


def filter_preds_to_annotated_bands(
    preds: Sequence[Box],
    pred_ids: Sequence | None = None,
    bands: Sequence[tuple[float, ...]] = (),
) -> tuple[list[Box], list, list[int]]:
    """Keep predicted boxes that fall inside human-annotated horizontal shelf bands."""
    if not bands:
        return list(preds), (list(pred_ids) if pred_ids is not None else []), list(range(len(preds)))
    kept_preds: list[Box] = []
    kept_ids: list = []
    kept_indices: list[int] = []
    for idx, b in enumerate(preds):
        if _box_in_any_band(b, bands):
            kept_preds.append(b)
            if pred_ids is not None and idx < len(pred_ids):
                kept_ids.append(pred_ids[idx])
            kept_indices.append(idx)
    return kept_preds, kept_ids, kept_indices


def filter_preds_to_psa_categories(
    preds: Sequence[Box],
    pred_ids: Sequence,
    catalog: dict[int, dict] | None,
    gt_labels: Sequence[dict] | None,
    bands: Sequence[tuple[float, float]] = (),
) -> tuple[list[Box], list, list[int]]:
    """Scope predictions to human-annotated shelf bands and the shelf's audited ``psa_category`` set."""
    target_cats = {
        str(g.get("category")).strip()
        for g in (gt_labels or [])
        if g.get("category") and not g.get("is_others_catchall")
    }
    has_others = any(bool(g.get("is_others_catchall")) for g in (gt_labels or []))
    in_preds, in_ids, in_indices = filter_preds_to_annotated_bands(preds, pred_ids, bands)
    if not target_cats or not catalog:
        return in_preds, in_ids, in_indices

    kept_preds: list[Box] = []
    kept_ids: list = []
    kept_indices: list[int] = []
    for b, pid, orig_idx in zip(in_preds, in_ids, in_indices, strict=True):
        if pid is None:
            kept_preds.append(b)
            kept_ids.append(pid)
            kept_indices.append(orig_idx)
            continue
        meta = catalog.get(pid) or {}
        p_cat = str(meta.get("category") or "").strip()
        p_is_hul = bool(meta.get("is_hul", True))
        if not p_cat or p_cat in target_cats or (has_others and not p_is_hul):
            kept_preds.append(b)
            kept_ids.append(pid)
            kept_indices.append(orig_idx)
    return kept_preds, kept_ids, kept_indices


def compute_enrichment_buckets(
    preds: Sequence[Box],
    pred_ids: Sequence | None,
    gt_boxes: Sequence[Box],
    gt_labels: Sequence[dict] | None = None,
    catalog: dict[int, dict] | None = None,
    pairs: dict[int, int] | None = None,
) -> dict:
    """Partition predictions into two buckets: (1) human-annotated scope and (2) extra discovered enrichment.

    Rather than discarding predictions outside human-annotated shelf rows, this preserves 100% of
    detected/identified products and quantifies how many extra facings, SKUs, and brands our
    pipeline discovers on unannotated rows or upgrades from human ``'Others'`` labels.
    """
    bands = group_annotated_shelf_bands(gt_boxes)
    in_preds, in_ids, in_indices = filter_preds_to_annotated_bands(preds, pred_ids, bands)
    in_index_set = set(in_indices)

    extra_indices = [idx for idx in range(len(preds)) if idx not in in_index_set]
    extra_skus: set[int] = set()
    extra_brands: set[str] = set()
    extra_categories: set[str] = set()
    if pred_ids is not None and catalog is not None:
        for idx in extra_indices:
            pid = pred_ids[idx] if idx < len(pred_ids) else None
            if pid is None:
                continue
            meta = catalog.get(pid) or {}
            if meta.get("sku_id") is not None:
                extra_skus.add(int(meta["sku_id"]))
            if meta.get("brand"):
                extra_brands.add(str(meta["brand"]))
            if meta.get("category"):
                extra_categories.add(str(meta["category"]))

    others_gt_total = sum(1 for g in (gt_labels or []) if bool(g.get("is_others_catchall")))
    others_resolved = 0
    if pairs and pred_ids is not None and catalog is not None and gt_labels is not None:
        for p_idx, g_idx in pairs.items():
            if g_idx < len(gt_labels) and bool(gt_labels[g_idx].get("is_others_catchall")):
                pid = pred_ids[p_idx] if p_idx < len(pred_ids) else None
                meta = catalog.get(pid) if pid is not None else None
                if meta and meta.get("brand") and str(meta.get("brand")) not in ("Others", "Unknown"):
                    others_resolved += 1

    gt_count = len(gt_boxes)
    total_preds = len(preds)
    return {
        "gt_annotated_facings": gt_count,
        "total_predicted_facings": total_preds,
        "in_annotated_scope_preds": len(in_preds),
        "extra_unannotated_row_preds": len(extra_indices),
        "extra_discovered_unique_skus": len(extra_skus),
        "extra_discovered_unique_brands": len(extra_brands),
        "extra_discovered_categories": sorted(extra_categories),
        "others_catchall_gt_count": others_gt_total,
        "others_catchall_resolved_count": others_resolved,
        "enrichment_multiplier": round(total_preds / max(gt_count, 1), 2),
    }


def compute_human_and_beyond_metrics(
    preds: Sequence[Box],
    pred_ids: Sequence | None,
    gt_boxes: Sequence[Box],
    gt_labels: Sequence[dict] | None = None,
    catalog: dict[int, dict] | None = None,
    pairs: dict[int, int] | None = None,
    eval_fields: Sequence[str] = ("product", "brand", "base_variant", "category", "packaging_type", "is_hul"),
) -> dict:
    """Compute the 3-Tier 'Human Annotated & Beyond' evaluation structure for shelf benchmarks.

    - ``tier1_named_target_sku``: Accuracy & F2 on explicitly named target SKUs (where the human
      annotator provided a specific product/brand rather than ``'Others'``).
    - ``tier2_client_clubbed_parity``: Apples-to-apples F2 in the client's label space inside
      human-annotated shelf bands, where any valid non-target SKU prediction is clubbed into
      ``'Others'`` just as the human annotator did.
    - ``tier2_annotated_zone_detection``: 2D box detection F2 inside human-annotated shelf zones.
    - ``tier3_beyond_gt_discovery``: Count of human ``'Others'`` boxes upgraded to exact Brand+SKU
      plus extra valid facings discovered on unannotated shelf rows.
    """
    bands = group_annotated_shelf_bands(gt_boxes)
    in_preds, in_ids, in_indices = filter_preds_to_annotated_bands(preds, pred_ids, bands)
    in_box_m = match(in_preds, gt_boxes)
    eff_pairs = pairs if pairs is not None else match(preds, gt_boxes)["pairs"]
    enr = compute_enrichment_buckets(preds, pred_ids, gt_boxes, gt_labels, catalog, eff_pairs)

    if pred_ids is None or gt_labels is None:
        z_sc = {k: round(v, 4) for k, v in scores(in_box_m["tp"], in_box_m["fp"], in_box_m["fn"]).items()}
        return {
            "tier1_named_target_sku": {"tp": in_box_m["tp"], "fp": in_box_m["fp"], "fn": in_box_m["fn"], "found": in_box_m["tp"], "matched_box_acc": 1.0, **z_sc},
            "tier2_client_clubbed_parity": {"tp": in_box_m["tp"], "fp": in_box_m["fp"], "fn": in_box_m["fn"], "found": in_box_m["tp"], **z_sc},
            "tier2_annotated_zone_detection": {
                "tp": in_box_m["tp"], "fp": in_box_m["fp"], "fn": in_box_m["fn"],
                "out_of_band_preds": len(preds) - len(in_preds),
                **z_sc,
            },
            "tier3_beyond_gt_discovery": {
                "others_gt_count": enr["others_catchall_gt_count"],
                "others_upgraded_to_exact_sku": enr["others_catchall_resolved_count"],
                "others_upgraded_pct": 0.0,
                "unannotated_row_facings_discovered": enr["extra_unannotated_row_preds"],
                "extra_unique_skus": enr["extra_discovered_unique_skus"],
                "extra_unique_brands": enr["extra_discovered_unique_brands"],
                "total_coverage_multiplier": enr["enrichment_multiplier"],
            },
        }

    named_gt_indices = [
        j for j, g in enumerate(gt_labels)
        if not bool(g.get("is_others_catchall"))
    ]
    named_gt_keys = {
        (str(gt_labels[j].get("brand") or "").lower(), str(gt_labels[j].get("base_variant") or gt_labels[j].get("product") or "").lower())
        for j in named_gt_indices
    }

    named_matched = 0
    named_tp = 0
    club_tp = 0
    matched_total = len(eff_pairs)

    for p_idx, g_idx in eff_pairs.items():
        if g_idx >= len(gt_labels):
            continue
        gt = gt_labels[g_idx]
        pid = pred_ids[p_idx] if p_idx < len(pred_ids) else None
        pred_entry = pid if isinstance(pid, dict) else (catalog.get(pid) if (catalog is not None and pid is not None) else None)
        eff_f = [f for f in eval_fields if gt.get(f) is not None] or list(eval_fields)
        is_exact = bool(pred_entry is not None and match_product(pred_entry, gt, eff_f)["tp"] > 0)
        if not bool(gt.get("is_others_catchall")):
            named_matched += 1
            if is_exact:
                named_tp += 1
                club_tp += 1
            elif pred_entry is not None and (
                _brand_matches(pred_entry, gt)
                or (
                    _pred_is_hul(pred_entry) == bool(gt.get("is_hul"))
                    and _field_matches(pred_entry, gt, "category")
                )
            ):
                club_tp += 1
        else:
            if is_exact or bool(gt.get("is_unbranded_outside_bay")):
                if pred_entry is not None:
                    club_tp += 1
            elif pred_entry is not None:
                if bool(gt.get("is_hul")) and _pred_is_hul(pred_entry):
                    club_tp += 1
                elif not _pred_is_hul(pred_entry):
                    club_tp += 1
                else:
                    p_key = (
                        str(pred_entry.get("brand") or "").lower(),
                        str(pred_entry.get("base_variant") or pred_entry.get("product") or "").lower(),
                    )
                    if p_key not in named_gt_keys:
                        club_tp += 1

    named_wrong_on_matched = max(0, named_matched - named_tp)
    named_fn = max(0, len(named_gt_indices) - named_matched) + named_wrong_on_matched
    named_fp = named_wrong_on_matched
    t1_sc = {k: round(v, 4) for k, v in scores(named_tp, named_fp, named_fn).items()}
    t1_cond_acc = round(named_tp / max(1, named_matched), 4)

    club_wrong_on_matched = max(0, matched_total - club_tp)
    club_fp = in_box_m["fp"] + club_wrong_on_matched
    club_fn = in_box_m["fn"] + club_wrong_on_matched
    t2_sc = {k: round(v, 4) for k, v in scores(club_tp, club_fp, club_fn).items()}
    z_sc = {k: round(v, 4) for k, v in scores(in_box_m["tp"], in_box_m["fp"], in_box_m["fn"]).items()}

    oth_gt = enr["others_catchall_gt_count"]
    oth_res = enr["others_catchall_resolved_count"]
    oth_rate = round(oth_res / max(1, oth_gt), 4) if oth_gt > 0 else 0.0
    return {
        "tier1_named_target_sku": {
            "tp": named_tp,
            "fp": named_fp,
            "fn": named_fn,
            "found": named_matched,
            "named_gt_count": len(named_gt_indices),
            "gt_count": len(named_gt_indices),
            "localized_gt_count": named_matched,
            "matched_correct": named_tp,
            "matched_box_acc": t1_cond_acc,
            "conditional_accuracy": t1_cond_acc,
            **t1_sc,
        },
        "tier2_client_clubbed_parity": {
            "tp": club_tp,
            "fp": club_fp,
            "fn": club_fn,
            "found": matched_total,
            **t2_sc,
        },
        "tier2_annotated_zone_detection": {
            "tp": in_box_m["tp"],
            "fp": in_box_m["fp"],
            "fn": in_box_m["fn"],
            "out_of_band_preds": len(preds) - len(in_preds),
            **z_sc,
        },
        "tier3_beyond_gt_discovery": {
            "others_gt_count": oth_gt,
            "others_upgraded_to_exact_sku": oth_res,
            "others_upgraded_pct": oth_rate,
            "others_upgrade_rate": oth_rate,
            "unannotated_row_facings_discovered": enr["extra_unannotated_row_preds"],
            "extra_unique_skus": enr["extra_discovered_unique_skus"],
            "extra_unique_brands": enr["extra_discovered_unique_brands"],
            "total_coverage_multiplier": enr["enrichment_multiplier"],
        },
    }


def aggregate_human_and_beyond_metrics(rows: Sequence[dict]) -> dict | None:
    """Aggregate per-image ``human_and_beyond`` dictionaries into a run-level summary dictionary."""
    hb_rows = [
        r["human_and_beyond"] if isinstance(r.get("human_and_beyond"), dict) else r
        for r in rows
        if isinstance(r, dict) and (isinstance(r.get("human_and_beyond"), dict) or isinstance(r.get("tier1_named_target_sku"), dict))
    ]
    if not hb_rows:
        return None
    t1_tp = sum(int(h.get("tier1_named_target_sku", {}).get("tp", 0)) for h in hb_rows)
    t1_fp = sum(int(h.get("tier1_named_target_sku", {}).get("fp", 0)) for h in hb_rows)
    t1_fn = sum(int(h.get("tier1_named_target_sku", {}).get("fn", 0)) for h in hb_rows)
    t1_found = sum(int(h.get("tier1_named_target_sku", {}).get("found", 0)) for h in hb_rows)
    t1_gt = sum(int(h.get("tier1_named_target_sku", {}).get("named_gt_count", 0)) for h in hb_rows)
    t1_sc = {k: round(v, 4) for k, v in scores(t1_tp, t1_fp, t1_fn).items()}
    t1_cond_acc = round(t1_tp / max(1, t1_found), 4)

    t2_tp = sum(int(h.get("tier2_client_clubbed_parity", {}).get("tp", 0)) for h in hb_rows)
    t2_fp = sum(int(h.get("tier2_client_clubbed_parity", {}).get("fp", 0)) for h in hb_rows)
    t2_fn = sum(int(h.get("tier2_client_clubbed_parity", {}).get("fn", 0)) for h in hb_rows)
    t2_found = sum(int(h.get("tier2_client_clubbed_parity", {}).get("found", 0)) for h in hb_rows)
    t2_sc = {k: round(v, 4) for k, v in scores(t2_tp, t2_fp, t2_fn).items()}

    z_tp = sum(int(h.get("tier2_annotated_zone_detection", {}).get("tp", 0)) for h in hb_rows)
    z_fp = sum(int(h.get("tier2_annotated_zone_detection", {}).get("fp", 0)) for h in hb_rows)
    z_fn = sum(int(h.get("tier2_annotated_zone_detection", {}).get("fn", 0)) for h in hb_rows)
    z_oob = sum(int(h.get("tier2_annotated_zone_detection", {}).get("out_of_band_preds", 0)) for h in hb_rows)
    z_sc = {k: round(v, 4) for k, v in scores(z_tp, z_fp, z_fn).items()}

    oth_gt = sum(int(h.get("tier3_beyond_gt_discovery", {}).get("others_gt_count", 0)) for h in hb_rows)
    oth_res = sum(int(h.get("tier3_beyond_gt_discovery", {}).get("others_upgraded_to_exact_sku", 0)) for h in hb_rows)
    unann = sum(int(h.get("tier3_beyond_gt_discovery", {}).get("unannotated_row_facings_discovered", 0)) for h in hb_rows)
    ex_skus = sum(int(h.get("tier3_beyond_gt_discovery", {}).get("extra_unique_skus", 0)) for h in hb_rows)
    ex_brands = sum(int(h.get("tier3_beyond_gt_discovery", {}).get("extra_unique_brands", 0)) for h in hb_rows)
    tot_gt = sum(int(r.get("gt_count", 0)) for r in rows)
    tot_pred = sum(int(r.get("pred_count", 0)) for r in rows)
    oth_rate = round(oth_res / max(1, oth_gt), 4) if oth_gt > 0 else 0.0

    return {
        "tier1_named_target_sku": {
            "tp": t1_tp,
            "fp": t1_fp,
            "fn": t1_fn,
            "found": t1_found,
            "named_gt_count": t1_gt,
            "gt_count": t1_gt,
            "localized_gt_count": t1_found,
            "matched_correct": t1_tp,
            "matched_box_acc": t1_cond_acc,
            "conditional_accuracy": t1_cond_acc,
            **t1_sc,
        },
        "tier2_client_clubbed_parity": {
            "tp": t2_tp,
            "fp": t2_fp,
            "fn": t2_fn,
            "found": t2_found,
            **t2_sc,
        },
        "tier2_annotated_zone_detection": {
            "tp": z_tp,
            "fp": z_fp,
            "fn": z_fn,
            "out_of_band_preds": z_oob,
            **z_sc,
        },
        "tier3_beyond_gt_discovery": {
            "others_gt_count": oth_gt,
            "others_upgraded_to_exact_sku": oth_res,
            "others_upgraded_pct": oth_rate,
            "others_upgrade_rate": oth_rate,
            "unannotated_row_facings_discovered": unann,
            "extra_unique_skus": ex_skus,
            "extra_unique_brands": ex_brands,
            "total_coverage_multiplier": round(tot_pred / max(1, tot_gt), 2),
        },
    }


def _box_inside_any_asset_bay(
    box: Box,
    asset_boxes: Sequence[Box],
    min_containment: float = 0.45,
) -> bool:
    """Return True if ``box`` center is inside any Class 0 bay, >= min_containment area overlaps,
    or its base rests on a plank inside the bay with >= 25% vertical overlap (top-row header rule)."""
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    bh = max(1.0, box[3] - box[1])
    area = max(1.0, (box[2] - box[0]) * bh)
    for ab in asset_boxes:
        if ab[0] <= cx <= ab[2] and ab[1] <= cy <= ab[3]:
            return True
        ix1 = max(box[0], ab[0])
        iy1 = max(box[1], ab[1])
        ix2 = min(box[2], ab[2])
        iy2 = min(box[3], ab[3])
        if ix2 > ix1 and iy2 > iy1:
            if ((ix2 - ix1) * (iy2 - iy1)) / area >= min_containment:
                return True
            if ab[0] <= cx <= ab[2] and ab[1] <= box[3] <= ab[3] and (iy2 - iy1) / bh >= 0.25:
                return True
    return False


def filter_preds_to_asset_bays(
    preds: Sequence[Box],
    pred_ids: Sequence | None = None,
    asset_boxes: Sequence[Box] | None = None,
    min_containment: float = 0.45,
) -> tuple[list[Box], list, list[int], str]:
    """Scope Class 1 SKU boxes to Class 0 promo bays when bays exist; else keep full-image SKUs (hanger fallback)."""
    if not asset_boxes:
        return (
            list(preds),
            list(pred_ids) if pred_ids is not None else [],
            list(range(len(preds))),
            "full_image_hanger_fallback",
        )
    kept_preds: list[Box] = []
    kept_ids: list = []
    kept_indices: list[int] = []
    for idx, b in enumerate(preds):
        if _box_inside_any_asset_bay(b, asset_boxes, min_containment=min_containment):
            kept_preds.append(b)
            if pred_ids is not None and idx < len(pred_ids):
                kept_ids.append(pred_ids[idx])
            kept_indices.append(idx)
    return kept_preds, kept_ids, kept_indices, "class0_bay_scoped"


def score_merchandising_bay_or_hanger(
    preds: Sequence[Box],
    pred_ids: Sequence | None,
    gt_boxes: Sequence[Box],
    gt_labels: Sequence[dict] | None = None,
    asset_boxes: Sequence[Box] | None = None,
    catalog: dict[int, dict] | None = None,
    fields: Sequence[str] | None = None,
    iou_thr: float = 0.5,
) -> dict:
    """Score detection or E2E predictions under Unilever's Class 0 bay vs. full-image hanger rule.

    When Class 0 display bays (``asset_boxes``) exist, only SKUs inside the asset boundaries are
    scored (ignoring background store shelves outside the promo bay). When no Class 0 bay exists
    (such as hanging strips / hangers), full-image SKU detection and classification are scored.
    """
    sc_preds, sc_ids, _, scope_mode = filter_preds_to_asset_bays(
        preds, pred_ids=pred_ids, asset_boxes=asset_boxes
    )
    sc_gt_boxes, sc_gt_labels, _, _ = filter_preds_to_asset_bays(
        gt_boxes, pred_ids=gt_labels, asset_boxes=asset_boxes
    )
    if pred_ids is None:
        m = match(sc_preds, sc_gt_boxes, thr=iou_thr)
        found = m["tp"]
    else:
        sc_gt_ids = [g.get("sku_id") for g in sc_gt_labels]
        m = match_identified(
            sc_preds,
            sc_ids,
            sc_gt_boxes,
            sc_gt_ids,
            catalog=catalog,
            gt_labels=sc_gt_labels,
            fields=tuple(fields) if fields is not None else None,
        )
        found = m["found"]
    sc = scores(m["tp"], m["fp"], m["fn"])
    return {
        "scope_mode": scope_mode,
        "asset_bay_count": len(asset_boxes or []),
        "scoped_pred_count": len(sc_preds),
        "scoped_gt_count": len(sc_gt_boxes),
        "found": found,
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        **{k: round(v, 4) for k, v in sc.items()},
    }


