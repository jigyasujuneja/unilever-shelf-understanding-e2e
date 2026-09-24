"""'Run Now, Score Later' Deferred Ground-Truth Scoring & Declarative Schema Mapping (SPEC-005).

Ported from Riley's `shelf_benchmark/scoring.py` and `data/ground_truth.py`.
Allows pipelines (or Riley's un-scored `reports/row_level_report.json` runs) to be logged immediately
with `ground_truth_status = "PLACEHOLDER_AWAITING_GROUND_TRUTH"` and retroactively scored with zero
model calls the moment ground-truth annotations (CSV / JSON / BigQuery export) arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from shelf_e2e.geometry import compute_box_iou
from shelf_e2e.taxonomy import normalize_brand_and_hul_flag


@dataclass(frozen=True)
class GroundTruthSchemaMapping:
    """Declarative column mapping for arbitrary Unilever / vendor Ground-Truth deliveries."""

    image_key_field: str = "image_id"
    items_list_field: str = "annotations"
    bbox_field: str = "bbox_2d"
    brand_field: str = "brand"
    sku_id_field: str = "base_pack_code"
    back_row_field: str = "is_back_row_depth_ghost"


@dataclass
class RetroactiveScoreSummary:
    """Summary returned when retroactively scoring deferred predictions against newly arrived GT."""

    run_id: str
    ground_truth_status: str
    total_images_scored: int
    total_predictions: int
    total_gt_front_facings: int
    true_positives_iou50: int
    precision_iou50: float
    recall_iou50: float
    f1_iou50: float
    brand_top1_accuracy: float
    sku_top1_accuracy: float
    hul_sos_abs_error_pct: float
    per_image_details: List[Dict[str, Any]] = field(default_factory=list)


def score_deferred_predictions_against_gt(
    run_id: str,
    predictions_by_image: Dict[str, List[Dict[str, Any]]],
    ground_truth_by_image: Dict[str, List[Dict[str, Any]]],
    schema_mapping: Optional[GroundTruthSchemaMapping] = None,
    iou_threshold: float = 0.50,
    exclude_back_row_gt: bool = True,
) -> RetroactiveScoreSummary:
    """Score previously saved predictions against newly arrived Ground Truth with zero inference cost."""
    mapping = schema_mapping or GroundTruthSchemaMapping()
    if not ground_truth_by_image:
        total_preds = sum(len(v) for v in predictions_by_image.values())
        return RetroactiveScoreSummary(
            run_id=run_id,
            ground_truth_status="PLACEHOLDER_AWAITING_GROUND_TRUTH",
            total_images_scored=0,
            total_predictions=total_preds,
            total_gt_front_facings=0,
            true_positives_iou50=0,
            precision_iou50=0.0,
            recall_iou50=0.0,
            f1_iou50=0.0,
            brand_top1_accuracy=0.0,
            sku_top1_accuracy=0.0,
            hul_sos_abs_error_pct=0.0,
        )

    tp_total = 0
    pred_total = 0
    gt_total = 0
    brand_hits = 0
    sku_hits = 0
    sos_errors: List[float] = []
    per_image: List[Dict[str, Any]] = []

    for img_id, preds in predictions_by_image.items():
        raw_gt = ground_truth_by_image.get(img_id, [])
        gt_items = [
            g
            for g in raw_gt
            if not (exclude_back_row_gt and bool(g.get(mapping.back_row_field, False)))
        ]
        pred_total += len(preds)
        gt_total += len(gt_items)

        matched_gt_indices = set()
        img_tp = 0
        img_brand_hits = 0
        img_sku_hits = 0

        for p in preds:
            p_box = p.get("bbox_2d") or [
                int(p.get("bbox_ymin", 0)),
                int(p.get("bbox_xmin", 0)),
                int(p.get("bbox_ymax", 0)),
                int(p.get("bbox_xmax", 0)),
            ]
            best_iou = 0.0
            best_idx = -1
            for g_idx, g in enumerate(gt_items):
                if g_idx in matched_gt_indices:
                    continue
                g_box = g.get(mapping.bbox_field, [0, 0, 0, 0])
                iou = compute_box_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx
            if best_iou >= iou_threshold and best_idx >= 0:
                matched_gt_indices.add(best_idx)
                img_tp += 1
                tp_total += 1
                g_match = gt_items[best_idx]
                p_brand, _ = normalize_brand_and_hul_flag(
                    str(p.get("brand") or p.get("predicted_brand") or "")
                )
                g_brand, _ = normalize_brand_and_hul_flag(str(g_match.get(mapping.brand_field, "")))
                if p_brand != "Unknown" and p_brand.lower() == g_brand.lower():
                    img_brand_hits += 1
                    brand_hits += 1
                p_sku = str(p.get("base_pack_code") or p.get("matched_sku_id") or "")
                g_sku = str(g_match.get(mapping.sku_id_field, ""))
                if p_sku and p_sku == g_sku:
                    img_sku_hits += 1
                    sku_hits += 1

        # Compute Linear Share of Shelf (SOS %) error for HUL brands
        pred_hul_w = sum(
            max(0, int((p.get("bbox_2d") or [0, 0, 0, 0])[3]) - int((p.get("bbox_2d") or [0, 0, 0, 0])[1]))
            for p in preds
            if normalize_brand_and_hul_flag(str(p.get("brand") or p.get("predicted_brand") or ""))[1]
        )
        pred_tot_w = max(
            1,
            sum(
                max(0, int((p.get("bbox_2d") or [0, 0, 0, 0])[3]) - int((p.get("bbox_2d") or [0, 0, 0, 0])[1]))
                for p in preds
            ),
        )
        gt_hul_w = sum(
            max(0, int(g.get(mapping.bbox_field, [0, 0, 0, 0])[3]) - int(g.get(mapping.bbox_field, [0, 0, 0, 0])[1]))
            for g in gt_items
            if normalize_brand_and_hul_flag(str(g.get(mapping.brand_field, "")))[1]
        )
        gt_tot_w = max(
            1,
            sum(
                max(0, int(g.get(mapping.bbox_field, [0, 0, 0, 0])[3]) - int(g.get(mapping.bbox_field, [0, 0, 0, 0])[1]))
                for g in gt_items
            ),
        )
        sos_err = abs((pred_hul_w / pred_tot_w) * 100.0 - (gt_hul_w / gt_tot_w) * 100.0)
        sos_errors.append(sos_err)

        per_image.append(
            {
                "image_id": img_id,
                "predictions": len(preds),
                "gt_front_facings": len(gt_items),
                "tp_iou50": img_tp,
                "brand_hits": img_brand_hits,
                "sku_hits": img_sku_hits,
                "hul_sos_abs_error_pct": round(sos_err, 2),
            }
        )

    precision = tp_total / float(max(1, pred_total))
    recall = tp_total / float(max(1, gt_total))
    f1 = (2.0 * precision * recall) / max(1e-9, precision + recall)
    brand_acc = brand_hits / float(max(1, tp_total))
    sku_acc = sku_hits / float(max(1, tp_total))
    mean_sos_err = sum(sos_errors) / float(max(1, len(sos_errors)))

    return RetroactiveScoreSummary(
        run_id=run_id,
        ground_truth_status="SCORED_AGAINST_GROUND_TRUTH",
        total_images_scored=len(per_image),
        total_predictions=pred_total,
        total_gt_front_facings=gt_total,
        true_positives_iou50=tp_total,
        precision_iou50=round(precision, 4),
        recall_iou50=round(recall, 4),
        f1_iou50=round(f1, 4),
        brand_top1_accuracy=round(brand_acc, 4),
        sku_top1_accuracy=round(sku_acc, 4),
        hul_sos_abs_error_pct=round(mean_sos_err, 2),
        per_image_details=per_image,
    )
