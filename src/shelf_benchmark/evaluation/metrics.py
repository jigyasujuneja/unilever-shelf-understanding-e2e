"""Accuracy evaluation for Detection, Classification, Matching and Planogram Compliance.

Design rules (see `docs/EVALUATION_PROTOCOL.md` for the full rationale):

1. **No hidden thresholds.** The IoU threshold comes from `EvaluationConfig` and is echoed back in
   `AccuracyMetrics.iou_threshold`, so a precision number can never be read out of context.
2. **Localization and classification stay independent.** Predictions are paired to ground truth on
   geometry alone by default. Letting brand agreement influence which box matches which inflates
   detection and classification simultaneously.
3. **Unmatched predictions are false positives.** A prediction that overlaps nothing is not quietly
   handed the nearest spare ground-truth item.
4. **Strict string matching by default.** Lenient, category-specific keyword matching remains
   available as `product_matcher="fuzzy_demo"` but is explicitly not for reported numbers.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from shelf_benchmark.config import EvaluationConfig
from shelf_benchmark.models import (
    AccuracyMetrics,
    GroundTruthProductItem,
    ImageGroundTruth,
    RowLevelReportItem,
)

# Re-exported for backwards compatibility. The implementations live in one place now; there used
# to be a second, subtly different normalizer in `tasks/facing_utils.py` that rewrote brands
# before this module ever compared them. See `shelf_benchmark.text_normalization`.
from shelf_benchmark.text_normalization import canonical_brand, normalize_text

__all__ = ["canonical_brand", "normalize_text"]


def brands_match(
    pred_brand: Optional[str],
    gt_brand: Optional[str],
    config: Optional[EvaluationConfig] = None,
) -> bool:
    """Compare two brand strings under the configured matcher.

    `strict` (default): normalized equality (including apostrophe/accent normalization) plus
    optional `brand_aliases` folding, so `"Brand_A"` == `"Brand_A"` and `"Brand_F"` == `"Brand_F"`,
    while `"Lux"` != `"Deluxe"`.
    `fuzzy`: additionally accepts substring containment.
    """
    cfg = config or EvaluationConfig()
    p = canonical_brand(pred_brand, cfg.brand_aliases)
    g = canonical_brand(gt_brand, cfg.brand_aliases)
    if not p or not g:
        return False
    if p == g:
        return True
    if cfg.brand_matcher == "fuzzy":
        return p in g or g in p
    return False


def products_match(
    pred_name: Optional[str],
    pred_variant: Optional[str],
    gt_name: Optional[str],
    config: Optional[EvaluationConfig] = None,
) -> bool:
    """Compare a predicted product (name + variant) against the ground-truth product name."""
    cfg = config or EvaluationConfig()
    combined_pred = normalize_text(f"{pred_name or ''} {pred_variant or ''}")
    norm_gt = normalize_text(gt_name)
    if not combined_pred or not norm_gt:
        return False

    if combined_pred == norm_gt or norm_gt in combined_pred:
        return True
    if cfg.product_matcher == "strict":
        return False

    stopwords: Set[str] = {normalize_text(s) for s in cfg.product_stopwords if s}
    gt_tokens = {t for t in norm_gt.split() if len(t) > 2 and t not in stopwords}
    pred_tokens = set(combined_pred.split())

    if cfg.product_matcher == "token_overlap":
        if not gt_tokens:
            return False
        overlap = len(gt_tokens & pred_tokens) / len(gt_tokens)
        return overlap >= cfg.product_token_overlap_threshold

    if cfg.product_matcher == "fuzzy_demo":
        if not gt_tokens:
            return False
        return len(gt_tokens & pred_tokens) / len(gt_tokens) >= 0.5

    raise ValueError(
        f"Unknown product_matcher '{cfg.product_matcher}'. "
        f"Expected 'strict', 'token_overlap' or 'fuzzy_demo'."
    )


def skus_match(pred_sku: Optional[str], gt_sku: Optional[str]) -> bool:
    """Exact SKU/EAN comparison, ignoring case, whitespace and separators."""
    if not pred_sku or not gt_sku:
        return False

    def _normalize(value: object) -> str:
        return re.sub(r"[\s\-_]", "", str(value)).strip().upper()

    return _normalize(pred_sku) == _normalize(gt_sku)


def compute_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    """2D Intersection over Union for `[ymin, xmin, ymax, xmax]` boxes."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ya1, xa1, ya2, xa2 = box_a[:4]
    yb1, xb1, yb2, xb2 = box_b[:4]
    if ya2 <= ya1 or xa2 <= xa1 or yb2 <= yb1 or xb2 <= xb1:
        return 0.0

    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_area = inter_h * inter_w
    if inter_area <= 0:
        return 0.0

    area_a = (ya2 - ya1) * (xa2 - xa1)
    area_b = (yb2 - yb1) * (xb2 - xb1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def row_bbox(row: RowLevelReportItem) -> List[int]:
    """Row bounding box in canonical `[ymin, xmin, ymax, xmax]` order."""
    return [row.bbox_ymin, row.bbox_xmin, row.bbox_ymax, row.bbox_xmax]


def has_valid_bbox(row: RowLevelReportItem) -> bool:
    """True when the row carries a non-degenerate bounding box."""
    return row.bbox_ymax > row.bbox_ymin and row.bbox_xmax > row.bbox_xmin



def _optimal_bipartite_match(
    n_rows: int,
    n_gt: int,
    candidates: List[Tuple[float, float, int, int]],
) -> Dict[int, Tuple[int, float]]:
    """Maximum-weight bipartite matching (Hungarian / Kuhn-Munkres potentials).

    Greedy IoU pairing (`sorted(candidates, key=-score)`) can make a sub-optimal global choice when
    two adjacent shelf products overlap: assigning Pred 1 to GT 1 with IoU 0.62 can leave Pred 2
    with 0.0 IoU even when (Pred 1 -> GT 2 at 0.58, Pred 2 -> GT 1 at 0.60) yields two true
    positives (total 1.18 vs 0.62). This solver finds the exact global maximum-weight matching in
    pure Python.
    """
    if not candidates or n_rows == 0 or n_gt == 0:
        return {}

    score_map: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for score, iou, r_idx, g_idx in candidates:
        if (r_idx, g_idx) not in score_map or score > score_map[(r_idx, g_idx)][0]:
            score_map[(r_idx, g_idx)] = (score, iou)

    size = max(n_rows, n_gt)
    max_w = max((s for s, _ in score_map.values()), default=1.0)
    cost = [[max_w] * (size + 1) for _ in range(size + 1)]
    for (r_idx, g_idx), (score, _iou) in score_map.items():
        cost[r_idx + 1][g_idx + 1] = max_w - score

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0][j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    result: Dict[int, Tuple[int, float]] = {}
    for j in range(1, size + 1):
        r_idx = p[j] - 1
        g_idx = j - 1
        if 0 <= r_idx < n_rows and 0 <= g_idx < n_gt and (r_idx, g_idx) in score_map:
            _score, iou = score_map[(r_idx, g_idx)]
            result[r_idx] = (g_idx, round(iou, 4))
    return result


def compute_precision_recall_and_ap(
    rows: List[RowLevelReportItem],
    gt_items: List[GroundTruthProductItem],
    iou_threshold: float = 0.50,
) -> Tuple[Optional[float], List[Dict[str, float]]]:
    """Compute confidence-ranked Precision-Recall curve points and all-point interpolated AP."""
    if not gt_items:
        return None, []
    if not rows or not any(has_valid_bbox(r) for r in rows):
        return 0.0, []

    ranked = sorted(
        enumerate(rows),
        key=lambda pair: (-float(pair[1].confidence or 0.0), pair[0]),
    )
    matched_gt: Set[int] = set()
    tp_cum = 0
    fp_cum = 0
    n_gt = len(gt_items)
    curve: List[Dict[str, float]] = []

    for _orig_idx, row in ranked:
        if not has_valid_bbox(row):
            fp_cum += 1
        else:
            best_iou = 0.0
            best_g = -1
            rb = row_bbox(row)
            for g_idx, gt in enumerate(gt_items):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(rb, gt.bbox_2d)
                if iou > best_iou:
                    best_iou = iou
                    best_g = g_idx
            if best_g >= 0 and best_iou >= iou_threshold:
                matched_gt.add(best_g)
                tp_cum += 1
            else:
                fp_cum += 1

        prec = tp_cum / max(tp_cum + fp_cum, 1)
        rec = tp_cum / max(n_gt, 1)
        curve.append(
            {
                "confidence": round(float(row.confidence or 0.0), 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }
        )

    mrec = [0.0] + [pt["recall"] for pt in curve] + [1.0]
    mpre = [0.0] + [pt["precision"] for pt in curve] + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return round(ap, 4), curve


def compute_coco_map_50_95(
    rows: List[RowLevelReportItem],
    gt_items: List[GroundTruthProductItem],
) -> Optional[float]:
    """Compute COCO-style mAP averaged over IoU thresholds 0.50:0.05:0.95."""
    if not gt_items:
        return None
    if not rows or not any(has_valid_bbox(r) for r in rows):
        return 0.0
    thresholds = [round(0.50 + 0.05 * k, 2) for k in range(10)]
    aps = []
    for thr in thresholds:
        ap, _ = compute_precision_recall_and_ap(rows, gt_items, iou_threshold=thr)
        if ap is not None:
            aps.append(ap)
    return round(sum(aps) / len(aps), 4) if aps else None


def pair_predictions_with_gt(
    rows: List[RowLevelReportItem],
    gt_items: List[GroundTruthProductItem],
    config: Optional[EvaluationConfig] = None,
) -> List[Tuple[RowLevelReportItem, Optional[GroundTruthProductItem], float]]:
    """Pair predictions to ground-truth items, best-IoU-first.

    Returns one tuple per input row, in the original row order: `(row, gt_item_or_None, iou)`.

    Greedy assignment runs over the globally sorted candidate list, which is order-independent
    (unlike left-to-right greedy) and follows the standard detection-evaluation convention. With
    `require_iou_for_pairing` enabled, sub-threshold pairs are never formed, so a prediction that
    hits nothing stays unmatched and counts as a false positive.
    """
    cfg = config or EvaluationConfig()
    if not gt_items:
        return [(r, None, 0.0) for r in rows]

    use_brand_bonus = cfg.pairing_strategy == "iou_plus_brand"
    geometry_available = any(has_valid_bbox(r) for r in rows)

    candidates: List[Tuple[float, float, int, int]] = []  # (score, iou, row_idx, gt_idx)
    for r_idx, row in enumerate(rows):
        if not has_valid_bbox(row):
            continue
        for g_idx, gt in enumerate(gt_items):
            iou = compute_iou(row_bbox(row), gt.bbox_2d)
            if cfg.require_iou_for_pairing and iou < cfg.iou_threshold:
                continue
            score = iou
            if use_brand_bonus and brands_match(row.predicted_brand, gt.brand, cfg):
                score += 0.25
            if score <= 0.0:
                continue
            candidates.append((score, iou, r_idx, g_idx))

    paired: Dict[int, Tuple[Optional[GroundTruthProductItem], float]] = {}
    if cfg.pairing_strategy in ("optimal", "hungarian"):
        for r_idx, (g_idx, iou) in _optimal_bipartite_match(len(rows), len(gt_items), candidates).items():
            paired[r_idx] = (gt_items[g_idx], iou)
    else:
        used_rows: Set[int] = set()
        used_gt: Set[int] = set()
        for _score, iou, r_idx, g_idx in sorted(candidates, key=lambda c: (-c[0], c[2], c[3])):
            if r_idx in used_rows or g_idx in used_gt:
                continue
            used_rows.add(r_idx)
            used_gt.add(g_idx)
            paired[r_idx] = (gt_items[g_idx], round(iou, 4))

    if not geometry_available:
        # Classification-only output with no usable geometry: fall back to left-to-right positional
        # pairing so brand/product accuracy stays measurable. Reported IoU is 0.0, which makes the
        # absence of localization evidence visible rather than implied.
        ordered = sorted(range(len(rows)), key=lambda i: (rows[i].position_on_shelf or (i + 1), i))
        for slot, r_idx in enumerate(ordered):
            if slot >= len(gt_items):
                break
            paired[r_idx] = (gt_items[slot], 0.0)

    return [(row, *paired.get(idx, (None, 0.0))) for idx, row in enumerate(rows)]


# Attributes a given task does not predict, and must therefore not be scored on.
#
# `evaluate_task_accuracy` receives `task_type` but used to ignore it when computing
# `per_attribute_accuracy`, scoring all six core attributes on every row of every task. The
# detection task emits only boxes (plus a preliminary brand hint), so it was being graded on
# category, subcategory, variant, packaging_type, pack_type and product_name that it had never
# produced -- a guaranteed ~0% that then dragged down `macro_attribute_accuracy` and made
# detection look worse than classification for reasons unrelated to either.
#
# This is a deny-list rather than an allow-list on purpose: a task added later is scored on
# everything by default, which is visible and arguable, whereas an allow-list default of "nothing"
# would silently report no accuracy at all.
_UNPREDICTED_ATTRIBUTES_BY_TASK: Dict[str, frozenset] = {
    "detection": frozenset(
        {"category", "subcategory", "variant", "packaging_type", "pack_type", "size", "product_name"}
    ),
}


def evaluate_task_accuracy(
    task_type: str,
    rows: List[RowLevelReportItem],
    ground_truth: Optional[ImageGroundTruth],
    config: Optional[EvaluationConfig] = None,
) -> AccuracyMetrics:
    """Compute accuracy metrics for one run and annotate each `RowLevelReportItem` in place."""
    cfg = config or EvaluationConfig()

    if ground_truth is None or not ground_truth.items:
        for row in rows:
            row.gt_status = "PLACEHOLDER_AWAITING_GT"
        return AccuracyMetrics(
            ground_truth_available=False,
            predicted_count=len(rows),
            iou_threshold=cfg.iou_threshold,
            pairing_strategy=cfg.pairing_strategy,
            brand_matcher=cfg.brand_matcher,
            product_matcher=cfg.product_matcher,
        )

    gt_items = ground_truth.items
    gt_count = ground_truth.total_main_shelf_facings or len(gt_items)
    pred_count = len(rows)
    count_acc = max(0.0, 1.0 - (abs(pred_count - gt_count) / max(gt_count, 1)))

    pairings = pair_predictions_with_gt(rows, gt_items, cfg)

    all_ious: List[float] = []
    matched_ious: List[float] = []
    true_positives = 0
    matched_pairs = 0
    brand_hits = 0
    product_hits = 0
    sku_hits = 0
    sku_comparable = 0
    planogram_hits = 0
    planogram_total = 0

    for row, gt_item, iou in pairings:
        row.iou_with_gt = iou
        row.iou_threshold = cfg.iou_threshold
        row.gt_version = ground_truth.gt_version
        all_ious.append(iou)

        if gt_item is None:
            row.gt_status = "UNMATCHED_FALSE_POSITIVE"
            row.is_true_positive = False
            row.brand_correct = False
            row.product_correct = False
            continue

        matched_pairs += 1
        row.gt_status = "MATCHED"
        row.gt_item_id = gt_item.item_id
        row.gt_brand = gt_item.brand
        row.gt_product_name = gt_item.product_name
        row.gt_sku_id = gt_item.sku_id

        is_true_positive = iou >= cfg.iou_threshold
        row.is_true_positive = is_true_positive
        if is_true_positive:
            true_positives += 1
            matched_ious.append(iou)

        row.brand_correct = brands_match(row.predicted_brand, gt_item.brand, cfg)
        row.product_correct = products_match(
            row.predicted_product_name, row.predicted_variant, gt_item.product_name, cfg
        )
        brand_hits += int(row.brand_correct)
        product_hits += int(row.product_correct)

        if row.matched_sku_id and gt_item.sku_id:
            sku_comparable += 1
            row.sku_correct = skus_match(row.matched_sku_id, gt_item.sku_id)
            sku_hits += int(row.sku_correct)

        if row.planogram_compliant is not None:
            planogram_total += 1
            planogram_hits += int(row.planogram_compliant)

    # Compute per-attribute accuracy across core + unlimited extra_attributes (>8 attributes)
    attr_totals: Dict[str, int] = {}
    attr_hits: Dict[str, int] = {}

    def _record_attr(name: str, is_hit: bool) -> None:
        attr_totals[name] = attr_totals.get(name, 0) + 1
        if is_hit:
            attr_hits[name] = attr_hits.get(name, 0) + 1

    unpredicted = set(_UNPREDICTED_ATTRIBUTES_BY_TASK.get(task_type, frozenset()))
    has_any_brand_pred = any((r.predicted_brand or "").strip() for r in rows)
    has_any_prod_pred = any(
        (r.predicted_product_name or "").strip() or (r.predicted_variant or "").strip()
        for r in rows
    )
    if task_type == "detection" and not has_any_brand_pred:
        unpredicted.add("brand")

    for r, g, _ in pairings:
        if g is None:
            continue
        if g.brand and "brand" not in unpredicted:
            _record_attr("brand", bool(r.brand_correct))
        if g.product_name and "product_name" not in unpredicted:
            _record_attr("product_name", bool(r.product_correct))
        core_pairs = [
            ("category", r.predicted_category, g.category),
            ("subcategory", r.predicted_subcategory, g.subcategory),
            ("variant", r.predicted_variant, g.variant),
            ("packaging_type", r.predicted_packaging, g.packaging_type),
            ("pack_type", r.predicted_pack_type, g.pack_type),
            ("size", r.predicted_size, g.size),
        ]
        for attr_name, p_val, g_val in core_pairs:
            if attr_name in unpredicted:
                continue
            if g_val is not None and str(g_val).strip() != "":
                _record_attr(attr_name, normalize_text(str(p_val or "")) == normalize_text(str(g_val)))
        for attr_name, g_val in (g.extra_attributes or {}).items():
            if attr_name in unpredicted:
                continue
            if g_val is not None and str(g_val).strip() != "":
                extra_p_val = (r.extra_attributes or {}).get(attr_name)
                _record_attr(attr_name, normalize_text(str(extra_p_val or "")) == normalize_text(str(g_val)))

    per_attr_acc: Dict[str, float] = {
        k: round(attr_hits.get(k, 0) / tot, 4)
        for k, tot in attr_totals.items()
        if tot > 0
    }
    macro_attr_acc: Optional[float] = (
        round(sum(per_attr_acc.values()) / len(per_attr_acc), 4) if per_attr_acc else None
    )

    # Brand-set recall: did the run find every distinct brand present on the shelf?
    expected_brands = [b for b in ground_truth.expected_brands if b]
    predicted_brands = [r.predicted_brand for r in rows if r.predicted_brand]
    matched_expected = sum(
        1 for eb in expected_brands if any(brands_match(pb, eb, cfg) for pb in predicted_brands)
    )
    brand_set_recall = (
        (matched_expected / len(expected_brands))
        if (expected_brands and (task_type != "detection" or has_any_brand_pred))
        else None
    )

    # Localization counts are only meaningful when predictions carry geometry.
    geometry_available = any(has_valid_bbox(r) for r in rows)
    false_positives = pred_count - true_positives
    false_negatives = max(0, gt_count - true_positives)
    precision = true_positives / pred_count if pred_count else 0.0
    recall = true_positives / gt_count if gt_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Classification accuracy is per prediction, so unmatched predictions count against the score
    # instead of quietly disappearing from the denominator. For detection-only runs that emit no
    # brand or product predictions, report None rather than a fabricated 0.0.
    brand_acc = (
        (brand_hits / pred_count)
        if (pred_count and (task_type != "detection" or has_any_brand_pred))
        else None
    )
    prod_acc = (
        (product_hits / pred_count)
        if (pred_count and (task_type != "detection" or has_any_prod_pred))
        else None
    )
    sku_acc = (sku_hits / sku_comparable) if sku_comparable else None
    plano_rate = (planogram_hits / planogram_total) if planogram_total else None

    def _round(value: Optional[float]) -> Optional[float]:
        return round(value, 4) if value is not None else None

    return AccuracyMetrics(
        ground_truth_available=True,
        accuracy_status="EVALUATED_AGAINST_GT",
        gt_version=ground_truth.gt_version,
        iou_threshold=cfg.iou_threshold,
        pairing_strategy=cfg.pairing_strategy,
        brand_matcher=cfg.brand_matcher,
        product_matcher=cfg.product_matcher,
        ground_truth_count=gt_count,
        predicted_count=pred_count,
        matched_pairs=matched_pairs if geometry_available else None,
        true_positives=true_positives if geometry_available else None,
        false_positives=false_positives if geometry_available else None,
        false_negatives=false_negatives if geometry_available else None,
        count_accuracy=_round(count_acc),
        detection_precision=_round(precision) if geometry_available else None,
        detection_recall=_round(recall) if geometry_available else None,
        detection_f1=_round(f1) if geometry_available else None,
        average_precision_at_50=compute_precision_recall_and_ap(rows, gt_items, iou_threshold=cfg.iou_threshold)[0] if geometry_available else None,
        map_50_95=compute_coco_map_50_95(rows, gt_items) if geometry_available else None,
        pr_curve_points=compute_precision_recall_and_ap(rows, gt_items, iou_threshold=cfg.iou_threshold)[1] if geometry_available else [],
        mean_iou=_round(sum(all_ious) / len(all_ious)) if all_ious and geometry_available else None,
        mean_iou_matched=(_round(sum(matched_ious) / len(matched_ious)) if matched_ious else None),
        brand_classification_accuracy=_round(brand_acc),
        brand_set_recall=_round(brand_set_recall),
        product_classification_accuracy=_round(prod_acc),
        sku_matching_accuracy=_round(sku_acc),
        planogram_compliance_rate=_round(plano_rate),
        per_attribute_accuracy=per_attr_acc,
        macro_attribute_accuracy=macro_attr_acc,
    )
