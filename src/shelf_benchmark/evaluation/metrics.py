"""Accuracy Metrics Evaluator for Detection, Classification, Matching, and Planogram Compliance."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

from shelf_benchmark.models import (
    AccuracyMetrics,
    GroundTruthProductItem,
    ImageGroundTruth,
    RowLevelReportItem,
)


def normalize_text(text: Optional[str]) -> str:
    """Normalize brand/product strings for robust comparison (removes accents, punctuation, case)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-z0-9\s]", " ", ascii_str.lower())
    return " ".join(cleaned.split())


def brands_match(pred_brand: Optional[str], gt_brand: Optional[str]) -> bool:
    p = normalize_text(pred_brand)
    g = normalize_text(gt_brand)
    if not p or not g:
        return False
    if p == g or p in g or g in p:
        return True
    # Common aliases (e.g., Fair & Lovely <-> Glow & Lovely, Ponds <-> Pond's, Lakme <-> Lakmé)
    aliases = {
        "ponds": "pond s",
        "glow lovely": "glow and lovely",
        "fair lovely": "glow lovely",
        "lakme": "lakme",
    }
    for k, v in aliases.items():
        if k in p and k in g:
            return True
    return False


def products_match(pred_name: Optional[str], pred_variant: Optional[str], gt_name: Optional[str]) -> bool:
    combined_pred = normalize_text(f"{pred_name or ''} {pred_variant or ''}")
    norm_gt = normalize_text(gt_name)
    if not combined_pred or not norm_gt:
        return False
    if combined_pred == norm_gt or norm_gt in combined_pred:
        return True

    # Key distinguishing tokens per Unilever SKU line
    key_markers = [
        ("detox", "charcoal", "black"),
        ("bright beauty", "spot less", "pink"),
        ("bright c", "vitamin c", "lemon", "yellow"),
        ("bright glow", "insta glow", "multivitamin"),
        ("strawberry", "red"),
        ("kiwi", "cucumber", "green", "apple", "fruit"),
        ("pure", "gentle", "orange", "amber", "glycerin"),
        ("oil clear", "green", "lemon flower"),
        ("fresh renewal", "blue", "berry"),
    ]
    for group in key_markers:
        in_gt = any(tok in norm_gt for tok in group)
        in_pred = any(tok in combined_pred for tok in group)
        if in_gt and in_pred:
            return True

    # Token overlap fallback
    gt_tokens = {t for t in norm_gt.split() if len(t) > 2 and t not in {"face", "wash", "facewash", "the", "and"}}
    pred_tokens = set(combined_pred.split())
    if not gt_tokens:
        return False
    overlap = len(gt_tokens & pred_tokens) / len(gt_tokens)
    return overlap >= 0.5


def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute 2D Intersection over Union (IoU) for [ymin, xmin, ymax, xmax]."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ya1, xa1, ya2, xa2 = box_a[:4]
    yb1, xb1, yb2, xb2 = box_b[:4]
    if ya2 <= ya1 or xa2 <= xa1 or yb2 <= yb1 or xb2 <= xb1:
        return 0.0

    inter_ymin = max(ya1, yb1)
    inter_xmin = max(xa1, xb1)
    inter_ymax = min(ya2, yb2)
    inter_xmax = min(xa2, xb2)

    inter_h = max(0, inter_ymax - inter_ymin)
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_area = inter_h * inter_w
    if inter_area <= 0:
        return 0.0

    area_a = (ya2 - ya1) * (xa2 - xa1)
    area_b = (yb2 - yb1) * (xb2 - xb1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def compute_horizontal_proximity(box_a: List[int], box_b: List[int]) -> float:
    """Compute 1D horizontal center proximity score [0..1] on the shelf."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ca = 0.5 * (box_a[1] + box_a[3])
    cb = 0.5 * (box_b[1] + box_b[3])
    dist = abs(ca - cb) / 1000.0
    return max(0.0, 1.0 - dist * 4.0)


def pair_predictions_with_gt(
    rows: List[RowLevelReportItem],
    gt_items: List[GroundTruthProductItem],
) -> List[Tuple[RowLevelReportItem, Optional[GroundTruthProductItem], float]]:
    """Greedily match predicted product rows to ground truth items using IoU + horizontal position."""
    if not gt_items:
        return [(r, None, 0.0) for r in rows]

    used_gt = set()
    pairings: List[Tuple[RowLevelReportItem, Optional[GroundTruthProductItem], float]] = []

    # Sort predictions left-to-right by bbox_xmin (or product_index)
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.bbox_xmin if r.bbox_xmax > r.bbox_xmin else r.position_on_shelf * 50, r.product_index),
    )

    for r in sorted_rows:
        pred_box = [r.bbox_ymin, r.bbox_xmin, r.bbox_ymax, r.bbox_xmax]
        has_valid_box = r.bbox_ymax > r.bbox_ymin and r.bbox_xmax > r.bbox_xmin
        best_gt: Optional[GroundTruthProductItem] = None
        best_score = -1.0
        best_iou = 0.0

        for gt in gt_items:
            if gt.item_id in used_gt:
                continue
            iou = compute_iou(pred_box, gt.bbox_2d) if has_valid_box else 0.0
            h_prox = compute_horizontal_proximity(pred_box, gt.bbox_2d) if has_valid_box else 0.0
            b_bonus = 0.25 if brands_match(r.predicted_brand, gt.brand) else 0.0
            score = (iou * 0.65) + (h_prox * 0.25) + b_bonus if has_valid_box else (b_bonus * 2.0)
            if score > best_score:
                best_score = score
                best_gt = gt
                best_iou = iou

        if best_gt is not None:
            used_gt.add(best_gt.item_id)
            pairings.append((r, best_gt, round(best_iou, 4)))
        else:
            pairings.append((r, None, 0.0))

    # Restore original row order
    by_idx = {id(r): (r, g, iou) for r, g, iou in pairings}
    return [by_idx[id(r)] for r in rows]


def evaluate_task_accuracy(
    task_type: str,
    rows: List[RowLevelReportItem],
    ground_truth: Optional[ImageGroundTruth],
) -> AccuracyMetrics:
    """Compute comprehensive accuracy metrics and annotate RowLevelReportItems in place."""
    if ground_truth is None or not ground_truth.items:
        return AccuracyMetrics(
            ground_truth_available=False,
            predicted_count=len(rows),
        )

    gt_count = ground_truth.total_main_shelf_facings or len(ground_truth.items)
    pred_count = len(rows)
    count_acc = max(0.0, 1.0 - (abs(pred_count - gt_count) / max(gt_count, 1)))

    pairings = pair_predictions_with_gt(rows, ground_truth.items)

    ious: List[float] = []
    tp_iou30 = 0
    brand_hits = 0
    product_hits = 0
    sku_hits = 0
    planogram_hits = 0
    planogram_total = 0

    for row, gt_item, iou in pairings:
        row.iou_with_gt = iou
        ious.append(iou)
        if iou >= 0.25:
            tp_iou30 += 1

        if gt_item is not None:
            row.gt_item_id = gt_item.item_id
            row.gt_brand = gt_item.brand
            row.gt_product_name = gt_item.product_name
            row.gt_sku_id = gt_item.sku_id

            is_brand_ok = brands_match(row.predicted_brand, gt_item.brand)
            is_prod_ok = products_match(row.predicted_product_name, row.predicted_variant, gt_item.product_name)
            row.brand_correct = is_brand_ok
            row.product_correct = is_prod_ok
            if is_brand_ok:
                brand_hits += 1
            if is_prod_ok:
                product_hits += 1

            if row.matched_sku_id and gt_item.sku_id:
                is_sku_ok = row.matched_sku_id.strip().upper() == gt_item.sku_id.strip().upper()
                row.sku_correct = is_sku_ok
                if is_sku_ok:
                    sku_hits += 1

        if row.planogram_compliant is not None:
            planogram_total += 1
            if row.planogram_compliant:
                planogram_hits += 1

    # Brand set recall (did the model identify all distinct Unilever brands on the shelf?)
    expected_brands_norm = {normalize_text(b) for b in ground_truth.expected_brands if b}
    pred_brands_norm = {normalize_text(r.predicted_brand) for r in rows if r.predicted_brand}
    matched_expected_brands = 0
    for eb in expected_brands_norm:
        if any(brands_match(pb, eb) for pb in pred_brands_norm):
            matched_expected_brands += 1
    brand_set_recall = (
        matched_expected_brands / len(expected_brands_norm) if expected_brands_norm else 1.0
    )

    precision = tp_iou30 / max(pred_count, 1) if pred_count > 0 else 0.0
    recall = tp_iou30 / max(gt_count, 1) if gt_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    paired_count = max(sum(1 for _, g, _ in pairings if g is not None), 1)
    brand_acc = brand_hits / paired_count if rows else 0.0
    prod_acc = product_hits / paired_count if rows else 0.0
    sku_acc = (sku_hits / paired_count) if task_type == "matching" and rows else None
    plano_rate = (planogram_hits / planogram_total) if planogram_total > 0 else None

    return AccuracyMetrics(
        ground_truth_available=True,
        ground_truth_count=gt_count,
        predicted_count=pred_count,
        count_accuracy=round(count_acc, 4),
        detection_precision_iou50=round(precision, 4),
        detection_recall_iou50=round(recall, 4),
        detection_f1_iou50=round(f1, 4),
        mean_iou=round(mean_iou, 4),
        brand_classification_accuracy=round(brand_acc, 4),
        brand_set_recall=round(brand_set_recall, 4),
        product_classification_accuracy=round(prod_acc, 4),
        sku_matching_accuracy=round(sku_acc, 4) if sku_acc is not None else None,
        planogram_compliance_rate=round(plano_rate, 4) if plano_rate is not None else None,
    )
