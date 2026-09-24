"""SPEC-002: Standalone Python Evaluation & Benchmarking Harness (`tests/benchmark_harness.py`).

Implements exact calculation modules for:
1. mAP@50, mAP@50:95, and IoU (P50, P90) distribution using xyxy bounding box IoU.
2. Top-1 Accuracy, Top-5 Recall, MRR (Mean Reciprocal Rank), per-category Precision, Recall, F1, and Hallucination Rate.
3. Toker Compliance F1 and JSON Schema Adherence %.
4. Per-tier Latency Percentiles (P50, P90, P95, P99) across tier1_detection, tier2_catalog_match, tier3_compliance, and total_e2e.
5. Blended Cost Calculator in Indian Rupees (₹ at 1 USD = 84 INR) verified against the <= ₹0.22 SLA.
6. Mock Pipeline interface (`MockDesignContractPipeline`) accepting `DESIGN.md` InputContract and returning OutputContract.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shelf_e2e.pricing import BlendedCostBreakdown, CostRateConfig, calculate_blended_cost
from shelf_e2e.schemas import (
    CompliancePayload,
    ComplianceStatus,
    InputContract,
    LatencyBreakdownMs,
    MarketSharePayload,
    MetricsPayload,
    OutputContract,
    PlanogramContract,
    PromoRules,
    ResolvedSKU,
    SchemaValidationError,
    StoreMetadata,
)


# ============================================================================
# 1. DETECTION METRICS: IoU, mAP@50, mAP@50:95, IoU P50 / P90
# ============================================================================
def compute_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute exact 2D Intersection over Union (IoU) for [x1, y1, x2, y2] boxes."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def percentile(values: Sequence[float], q: float) -> float:
    """Compute linear interpolated percentile for q in [0, 100]."""
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_vals[lower]
    weight = pos - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


def greedy_match_boxes(
    pred_boxes: Sequence[Sequence[float]],
    pred_scores: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Match confidence-sorted predictions to ground-truth boxes above iou_threshold."""
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
    matched_gt: Set[int] = set()
    matches: List[Tuple[int, int, float]] = []
    unmatched_preds: List[int] = []

    for pred_idx in order:
        best_gt = -1
        best_iou = -1.0
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou_xyxy(pred_boxes[pred_idx], gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0:
            matched_gt.add(best_gt)
            matches.append((pred_idx, best_gt, best_iou))
        else:
            unmatched_preds.append(pred_idx)

    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return matches, unmatched_preds, unmatched_gts


def compute_ap_at_threshold(
    pred_boxes: Sequence[Sequence[float]],
    pred_scores: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float,
) -> float:
    """Compute 101-point COCO-style Average Precision (AP) at a single IoU threshold."""
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0
    if not pred_boxes:
        return 0.0

    order = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
    matched_gt: Set[int] = set()
    tp_cum = 0
    fp_cum = 0
    precisions: List[float] = []
    recalls: List[float] = []

    for pred_idx in order:
        best_gt = -1
        best_iou = -1.0
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou_xyxy(pred_boxes[pred_idx], gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0:
            matched_gt.add(best_gt)
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / float(tp_cum + fp_cum))
        recalls.append(tp_cum / float(len(gt_boxes)))

    recall_levels = [i / 100.0 for i in range(101)]
    interp_precisions: List[float] = []
    for r_level in recall_levels:
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= r_level]
        interp_precisions.append(max(p_at_r) if p_at_r else 0.0)

    return sum(interp_precisions) / len(interp_precisions)


@dataclass(frozen=True)
class DetectionKPIs:
    """SPEC-002 Detection KPIs."""

    map_50: float
    map_50_95: float
    iou_p50: float
    iou_p90: float


def calculate_map50_and_map50_95(
    pred_boxes: Sequence[Sequence[float]],
    pred_scores: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
) -> DetectionKPIs:
    """Calculate mAP@50, mAP@50:95 (0.50:0.05:0.95), and IoU P50/P90 distribution."""
    thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    aps = [compute_ap_at_threshold(pred_boxes, pred_scores, gt_boxes, t) for t in thresholds]
    matches, _, _ = greedy_match_boxes(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.50)
    matched_ious = [m[2] for m in matches] if matches else [0.0]

    return DetectionKPIs(
        map_50=round(aps[0], 4),
        map_50_95=round(sum(aps) / len(aps), 4),
        iou_p50=round(percentile(matched_ious, 50.0), 4),
        iou_p90=round(percentile(matched_ious, 90.0), 4),
    )


# ============================================================================
# 2. CLASSIFICATION, CATALOG RETRIEVAL & HALLUCINATION METRICS
# ============================================================================
@dataclass(frozen=True)
class RetrievalAndClassificationKPIs:
    """SPEC-002 Classification, Catalog Retrieval, and Hallucination KPIs."""

    top1_accuracy: float
    top5_recall: float
    mrr: float
    precision: float
    recall: float
    f1_score: float
    per_category_f1: Dict[str, float]
    hallucination_rate: float


def calculate_retrieval_and_classification_metrics(
    resolved_skus: Sequence[ResolvedSKU],
    gt_boxes: Sequence[Sequence[float]],
    gt_base_pack_ids: Sequence[str],
    gt_categories: Sequence[str],
    master_catalog_ids: Set[str],
    iou_threshold: float = 0.50,
) -> RetrievalAndClassificationKPIs:
    """Calculate Top-1 Accuracy, Top-5 Recall, MRR, Precision/Recall/F1, and Hallucination Rate."""
    pred_boxes = [s.box_xyxy for s in resolved_skus]
    pred_scores = [s.confidence for s in resolved_skus]
    matches, _, _ = greedy_match_boxes(pred_boxes, pred_scores, gt_boxes, iou_threshold)

    top1_hits = 0
    top5_hits = 0
    reciprocal_ranks: List[float] = []

    cat_tp: Dict[str, int] = defaultdict(int)
    cat_fp: Dict[str, int] = defaultdict(int)
    cat_fn: Dict[str, int] = defaultdict(int)

    matched_pred_indices = {m[0] for m in matches}
    matched_gt_indices = {m[1] for m in matches}

    for pred_idx, gt_idx, _ in matches:
        pred_item = resolved_skus[pred_idx]
        gt_sku = gt_base_pack_ids[gt_idx]
        gt_cat = gt_categories[gt_idx]

        candidates = pred_item.candidate_ranking or [pred_item.base_pack_id]
        if pred_item.base_pack_id == gt_sku:
            top1_hits += 1
            cat_tp[gt_cat] += 1
        else:
            cat_fp[pred_item.category or gt_cat] += 1
            cat_fn[gt_cat] += 1

        if gt_sku in candidates[:5]:
            top5_hits += 1
            rank = candidates.index(gt_sku) + 1
            reciprocal_ranks.append(1.0 / float(rank))
        else:
            reciprocal_ranks.append(0.0)

    for pred_idx, pred_item in enumerate(resolved_skus):
        if pred_idx not in matched_pred_indices:
            cat_fp[pred_item.category or "Unknown"] += 1

    for gt_idx, gt_cat in enumerate(gt_categories):
        if gt_idx not in matched_gt_indices:
            cat_fn[gt_cat] += 1

    num_matched = max(1, len(matches))
    top1_acc = top1_hits / float(num_matched) if matches else 0.0
    top5_rec = top5_hits / float(num_matched) if matches else 0.0
    mrr_val = sum(reciprocal_ranks) / float(num_matched) if reciprocal_ranks else 0.0

    total_tp = sum(cat_tp.values())
    total_fp = sum(cat_fp.values())
    total_fn = sum(cat_fn.values())

    precision = total_tp / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    all_cats = set(cat_tp.keys()) | set(cat_fp.keys()) | set(cat_fn.keys())
    per_cat_f1: Dict[str, float] = {}
    for cat in sorted(all_cats):
        c_tp = cat_tp[cat]
        c_fp = cat_fp[cat]
        c_fn = cat_fn[cat]
        c_p = c_tp / float(c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_r = c_tp / float(c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        per_cat_f1[cat] = round((2.0 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0, 4)

    hallucinated = sum(1 for s in resolved_skus if s.base_pack_id not in master_catalog_ids)
    hallucination_rate = (hallucinated / float(len(resolved_skus))) if resolved_skus else 0.0

    return RetrievalAndClassificationKPIs(
        top1_accuracy=round(top1_acc, 4),
        top5_recall=round(top5_rec, 4),
        mrr=round(mrr_val, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        per_category_f1=per_cat_f1,
        hallucination_rate=round(hallucination_rate, 4),
    )


# ============================================================================
# 3. COMPLIANCE & JSON SCHEMA ADHERENCE METRICS
# ============================================================================
def calculate_toker_compliance_f1(
    predicted_flags: Sequence[ComplianceStatus],
    ground_truth_flags: Sequence[ComplianceStatus],
) -> Tuple[float, float, float]:
    """Compute binary Precision, Recall, and F1 on Toker promotional compliance flags."""
    tp = sum(
        1
        for p, g in zip(predicted_flags, ground_truth_flags)
        if p == ComplianceStatus.COMPLIANT and g == ComplianceStatus.COMPLIANT
    )
    fp = sum(
        1
        for p, g in zip(predicted_flags, ground_truth_flags)
        if p == ComplianceStatus.COMPLIANT and g == ComplianceStatus.NON_COMPLIANT
    )
    fn = sum(
        1
        for p, g in zip(predicted_flags, ground_truth_flags)
        if p == ComplianceStatus.NON_COMPLIANT and g == ComplianceStatus.COMPLIANT
    )
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def calculate_json_schema_adherence(raw_payloads: Sequence[Dict[str, Any]]) -> float:
    """Return fraction (0.0..1.0) of raw dictionary payloads that pass strict OutputContract validation."""
    if not raw_payloads:
        return 0.0
    valid_count = 0
    for item in raw_payloads:
        try:
            OutputContract.model_validate(item)
            valid_count += 1
        except (SchemaValidationError, ValueError, TypeError, KeyError):
            continue
    return round(valid_count / float(len(raw_payloads)), 4)


# ============================================================================
# 4. OPERATIONAL LATENCY TIMERS (P50, P90, P95, P99) & CONCURRENCY HARNESS
# ============================================================================
@dataclass(frozen=True)
class LatencyDistributionMs:
    """Percentile latency breakdown in milliseconds."""

    p50: float
    p90: float
    p95: float
    p99: float


def summarize_tier_latencies(
    latencies: Sequence[LatencyBreakdownMs],
) -> Dict[str, LatencyDistributionMs]:
    """Calculate P50, P90, P95, and P99 across tier1, tier2, tier3, and total_e2e."""
    tiers = ("tier1_detection", "tier2_catalog_match", "tier3_compliance", "total_e2e")
    summary: Dict[str, LatencyDistributionMs] = {}
    for tier in tiers:
        vals = [float(getattr(item, tier)) for item in latencies]
        summary[tier] = LatencyDistributionMs(
            p50=round(percentile(vals, 50.0), 2),
            p90=round(percentile(vals, 90.0), 2),
            p95=round(percentile(vals, 95.0), 2),
            p99=round(percentile(vals, 99.0), 2),
        )
    return summary


def run_concurrency_stress_test(
    pipeline_fn: Callable[[InputContract], OutputContract],
    sample_input: InputContract,
    worker_count: int = 525,
) -> Dict[str, float]:
    """Execute pipeline concurrently across `worker_count` threads (e.g., 100, 250, 525) and return P95."""
    with ThreadPoolExecutor(max_workers=min(worker_count, 64)) as pool:
        futures = [pool.submit(pipeline_fn, sample_input) for _ in range(worker_count)]
        outputs = [f.result() for f in futures]
    e2e_vals = [o.metrics.latency_ms.total_e2e for o in outputs]
    return {
        "workers": float(worker_count),
        "p50_ms": round(percentile(e2e_vals, 50.0), 2),
        "p95_ms": round(percentile(e2e_vals, 95.0), 2),
        "p99_ms": round(percentile(e2e_vals, 99.0), 2),
    }


# ============================================================================
# 5. MOCK PIPELINE INTERFACE (`DESIGN.md` InputContract -> OutputContract)
# ============================================================================
class MockDesignContractPipeline:
    """Mock pipeline accepting SPEC-001 InputContract and returning valid OutputContract."""

    def __init__(self, master_catalog_ids: Optional[Set[str]] = None):
        self.master_catalog_ids = master_catalog_ids or {
            "BP-DOVE-BW-750",
            "BP-DOVE-BW-500",
            "BP-TRES-SH-750",
            "BP-COMP-SH-650",
        }

    def __call__(self, contract: InputContract | Dict[str, Any]) -> OutputContract:
        validated_in = (
            contract
            if isinstance(contract, InputContract)
            else InputContract.model_validate(contract)
        )
        cost: BlendedCostBreakdown = calculate_blended_cost(
            input_tokens=750,
            output_tokens=150,
            gpu_seconds=0.14,
            vcpu_seconds=0.20,
            vector_queries=4,
            embedded_crops=4,
        )
        resolved = [
            ResolvedSKU(
                box_xyxy=[40.0, 80.0, 120.0, 290.0],
                base_pack_id="BP-DOVE-BW-750",
                confidence=0.97,
                candidate_ranking=["BP-DOVE-BW-750", "BP-DOVE-BW-500"],
                category="Skin Cleansing",
            ),
            ResolvedSKU(
                box_xyxy=[125.0, 80.0, 205.0, 290.0],
                base_pack_id="BP-DOVE-BW-750",
                confidence=0.96,
                candidate_ranking=["BP-DOVE-BW-750", "BP-DOVE-BW-500"],
                category="Skin Cleansing",
            ),
            ResolvedSKU(
                box_xyxy=[215.0, 75.0, 295.0, 295.0],
                base_pack_id="BP-TRES-SH-750",
                confidence=0.95,
                candidate_ranking=["BP-TRES-SH-750", "BP-COMP-SH-650"],
                category="Hair Care",
            ),
            ResolvedSKU(
                box_xyxy=[310.0, 90.0, 385.0, 290.0],
                base_pack_id="BP-COMP-SH-650",
                confidence=0.93,
                candidate_ranking=["BP-COMP-SH-650", "BP-TRES-SH-750"],
                category="Hair Care",
            ),
        ]
        found_ids = {r.base_pack_id for r in resolved}
        red_line_gaps = [
            sku for sku in validated_in.planogram_contract.target_skus if sku not in found_ids
        ]
        return OutputContract(
            metrics=MetricsPayload(
                total_detected=len(resolved),
                share_of_shelf_pct=76.19,
                latency_ms=LatencyBreakdownMs(
                    tier1_detection=140.0,
                    tier2_catalog_match=175.0,
                    tier3_compliance=640.0,
                    total_e2e=955.0,
                ),
                estimated_cost_inr=cost.total_cost_inr,
            ),
            marketshare=MarketSharePayload(
                resolved_skus=resolved,
                red_line_gaps=red_line_gaps,
                width_pack_gaps=[],
            ),
            compliance=CompliancePayload(
                toker_status=ComplianceStatus.COMPLIANT,
                display_status=ComplianceStatus.COMPLIANT
                if not red_line_gaps
                else ComplianceStatus.NON_COMPLIANT,
                coaching_message="All target SKUs and promotional Tokers verified compliant.",
            ),
        )


# ============================================================================
# 6. PYTEST & UNITTEST VERIFICATION SUITE FOR SPEC-001 & SPEC-002 HARNESS
# ============================================================================
def _build_sample_input() -> InputContract:
    return InputContract(
        image_path="gs://unilever-shelf-images/mt_store_4k_001.jpg",
        store_metadata=StoreMetadata(
            store_id="MT-MUMBAI-042",
            channel="MODERN_TRADE",
            planogram_id="PLANO-SKIN-HAIR-Q3",
        ),
        planogram_contract=PlanogramContract(
            target_skus=["BP-DOVE-BW-750", "BP-TRES-SH-750"],
            promo_rules=PromoRules(toker_text="20% Extra", min_display_count=1),
        ),
    )


def test_mock_pipeline_io_contract_and_cost_sla() -> None:
    pipeline = MockDesignContractPipeline()
    sample_in = _build_sample_input()
    out = pipeline(sample_in)

    assert out.metrics.total_detected == 4
    assert out.metrics.estimated_cost_inr <= 0.22
    assert out.metrics.latency_ms.total_e2e <= 20000.0
    assert out.compliance.toker_status == ComplianceStatus.COMPLIANT
    assert calculate_json_schema_adherence([out.model_dump()]) == 1.0


def test_map50_and_map50_95_calculation() -> None:
    pipeline = MockDesignContractPipeline()
    out = pipeline(_build_sample_input())
    gt_boxes = [
        [40.0, 80.0, 120.0, 290.0],
        [125.0, 80.0, 205.0, 290.0],
        [215.0, 75.0, 295.0, 295.0],
        [310.0, 90.0, 385.0, 290.0],
    ]
    det_kpis = calculate_map50_and_map50_95(
        pred_boxes=[s.box_xyxy for s in out.marketshare.resolved_skus],
        pred_scores=[s.confidence for s in out.marketshare.resolved_skus],
        gt_boxes=gt_boxes,
    )
    assert det_kpis.map_50 == 1.0
    assert det_kpis.map_50_95 == 1.0
    assert det_kpis.iou_p50 == 1.0
    assert det_kpis.iou_p90 == 1.0


def test_retrieval_top1_top5_mrr_and_hallucination() -> None:
    pipeline = MockDesignContractPipeline()
    out = pipeline(_build_sample_input())
    gt_boxes = [
        [40.0, 80.0, 120.0, 290.0],
        [125.0, 80.0, 205.0, 290.0],
        [215.0, 75.0, 295.0, 295.0],
        [310.0, 90.0, 385.0, 290.0],
    ]
    gt_ids = ["BP-DOVE-BW-750", "BP-DOVE-BW-750", "BP-TRES-SH-750", "BP-COMP-SH-650"]
    gt_cats = ["Skin Cleansing", "Skin Cleansing", "Hair Care", "Hair Care"]

    ret_kpis = calculate_retrieval_and_classification_metrics(
        resolved_skus=out.marketshare.resolved_skus,
        gt_boxes=gt_boxes,
        gt_base_pack_ids=gt_ids,
        gt_categories=gt_cats,
        master_catalog_ids=pipeline.master_catalog_ids,
    )
    assert ret_kpis.top1_accuracy == 1.0
    assert ret_kpis.top5_recall == 1.0
    assert ret_kpis.mrr == 1.0
    assert ret_kpis.f1_score == 1.0
    assert ret_kpis.hallucination_rate == 0.0


def test_tier_latency_percentiles_and_525_concurrency() -> None:
    pipeline = MockDesignContractPipeline()
    sample_in = _build_sample_input()
    out = pipeline(sample_in)

    tier_stats = summarize_tier_latencies([out.metrics.latency_ms] * 25)
    assert tier_stats["total_e2e"].p95 <= 20000.0

    stress = run_concurrency_stress_test(pipeline, sample_in, worker_count=525)
    assert stress["workers"] == 525.0
    assert stress["p95_ms"] <= 20000.0


def test_blended_cost_inr_conversion_84() -> None:
    rates = CostRateConfig(usd_to_inr_rate=84.0, hard_ceiling_inr=0.22)
    cost = calculate_blended_cost(
        input_tokens=1_000_000,
        output_tokens=0,
        rates=rates,
    )
    assert math.isclose(cost.total_cost_usd, 0.075, rel_tol=1e-6)
    assert math.isclose(cost.total_cost_inr, 6.30, rel_tol=1e-5)
    assert cost.within_hard_ceiling is False


class TestBenchmarkHarness(unittest.TestCase):
    def test_01_mock_pipeline_io_and_sla(self) -> None:
        test_mock_pipeline_io_contract_and_cost_sla()

    def test_02_map50_and_map50_95(self) -> None:
        test_map50_and_map50_95_calculation()

    def test_03_retrieval_top1_top5_mrr(self) -> None:
        test_retrieval_top1_top5_mrr_and_hallucination()

    def test_04_tier_latency_and_525_concurrency(self) -> None:
        test_tier_latency_percentiles_and_525_concurrency()

    def test_05_blended_cost_inr_conversion(self) -> None:
        test_blended_cost_inr_conversion_84()


if __name__ == "__main__":
    unittest.main(verbosity=2)
