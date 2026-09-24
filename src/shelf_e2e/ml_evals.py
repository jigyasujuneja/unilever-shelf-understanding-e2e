"""SPEC-004: Data Science, Statistical Rigor & Metamorphic ML Evaluations (`src/shelf_e2e/ml_evals.py`).

Implements:
1. Non-parametric Bootstrap 95% Confidence Intervals (`[ci_lower, ci_upper]`) & Paired Bootstrap Significance Tests (`p_value`).
2. Slice-Based Subpopulation Error Analysis across 4 retail dimensions:
   - Optical Glare (`high_glare` vs `low_glare`)
   - Pack Geometry (`small_narrow_packs` vs `large_bottles`)
   - Shelf Density (`dense_bay` vs `standard_bay`)
   - Brand Ownership (`hul_brands` vs `competitor_brands`)
3. Confidence Calibration (`Expected Calibration Error / ECE`, `Brier Score`) & Pairwise SKU Confusion Matrix.
4. Automated Data Quality Profiling & Train/Public/Private Split Leakage Verification (`data-autocleaning` standard).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Sequence, Set, Tuple

from shelf_e2e.datasets import RPCCatalogAdapter
from shelf_e2e.schemas import ResolvedSKU
from tests.benchmark_harness import greedy_match_boxes, percentile


# ============================================================================
# 1. BOOTSTRAP 95% CONFIDENCE INTERVALS & PAIRED SIGNIFICANCE TESTING
# ============================================================================
@dataclass(frozen=True)
class BootstrapCIResult:
    """Point estimate with non-parametric 95% Bootstrap Confidence Interval."""

    mean: float
    ci_95_lower: float
    ci_95_upper: float
    std_error: float


def bootstrap_confidence_interval(
    sample_scores: Sequence[float],
    n_resamples: int = 500,
    seed: int = 42,
) -> BootstrapCIResult:
    """Compute non-parametric 95% bootstrap confidence interval for any per-item metric."""
    if not sample_scores:
        return BootstrapCIResult(mean=0.0, ci_95_lower=0.0, ci_95_upper=0.0, std_error=0.0)
    vals = [float(v) for v in sample_scores]
    n = len(vals)
    point_mean = sum(vals) / float(n)
    if n == 1:
        return BootstrapCIResult(
            mean=round(point_mean, 4),
            ci_95_lower=round(point_mean, 4),
            ci_95_upper=round(point_mean, 4),
            std_error=0.0,
        )

    rng = random.Random(seed)
    boot_means: List[float] = []
    for _ in range(n_resamples):
        resample = [vals[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(resample) / float(n))

    ci_low = percentile(boot_means, 2.5)
    ci_high = percentile(boot_means, 97.5)
    mean_of_boots = sum(boot_means) / float(len(boot_means))
    variance = sum((m - mean_of_boots) ** 2 for m in boot_means) / float(max(1, len(boot_means) - 1))

    return BootstrapCIResult(
        mean=round(point_mean, 4),
        ci_95_lower=round(ci_low, 4),
        ci_95_upper=round(ci_high, 4),
        std_error=round(math.sqrt(variance), 4),
    )


def paired_bootstrap_significance_test(
    scores_candidate: Sequence[float],
    scores_baseline: Sequence[float],
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap test evaluating H0: Mean(Candidate) <= Mean(Baseline)."""
    if len(scores_candidate) != len(scores_baseline) or not scores_candidate:
        raise ValueError("Paired significance test requires equal non-empty score sequences")

    n = len(scores_candidate)
    diffs = [float(c) - float(b) for c, b in zip(scores_candidate, scores_baseline)]
    observed_delta = sum(diffs) / float(n)

    rng = random.Random(seed)
    non_positive_count = 0
    boot_deltas: List[float] = []
    for _ in range(n_resamples):
        sample_diffs = [diffs[rng.randrange(n)] for _ in range(n)]
        delta = sum(sample_diffs) / float(n)
        boot_deltas.append(delta)
        if delta <= 0.0:
            non_positive_count += 1

    p_value = non_positive_count / float(n_resamples)
    return {
        "observed_delta": round(observed_delta, 4),
        "delta_ci_95_lower": round(percentile(boot_deltas, 2.5), 4),
        "delta_ci_95_upper": round(percentile(boot_deltas, 97.5), 4),
        "p_value": round(p_value, 4),
        "statistically_significant_at_0_05": p_value < 0.05 and observed_delta > 0.0,
    }


# ============================================================================
# 2. SLICE-BASED SUBPOPULATION ERROR ANALYSIS (4 RETAIL DIMENSIONS)
# ============================================================================
def evaluate_optical_and_category_slices(
    resolved_skus: Sequence[ResolvedSKU],
    gt_records: Sequence[Dict[str, Any]],
    catalog: RPCCatalogAdapter,
    iou_threshold: float = 0.50,
) -> Dict[str, Dict[str, float]]:
    """Disaggregate Top-1 Accuracy across Optical Glare, Pack Width, Shelf Density, and HUL Brand slices."""
    pred_boxes = [s.box_xyxy for s in resolved_skus]
    pred_scores = [s.confidence for s in resolved_skus]
    gt_boxes = [[float(v) for v in r["box_xyxy"]] for r in gt_records]
    matches, _, _ = greedy_match_boxes(pred_boxes, pred_scores, gt_boxes, iou_threshold)

    pred_by_gt: Dict[int, ResolvedSKU] = {gt_idx: resolved_skus[pred_idx] for pred_idx, gt_idx, _ in matches}
    slice_hits: Dict[str, List[int]] = defaultdict(list)
    is_dense_bay = len(gt_records) >= 15

    for gt_idx, gt_item in enumerate(gt_records):
        gt_sku = str(gt_item["gt_base_pack_id"])
        box = [float(v) for v in gt_item["box_xyxy"]]
        width_px = box[2] - box[0]
        glare = float(gt_item.get("glare_intensity", 0.0))
        is_hul = catalog.is_hul_sku(gt_sku)

        matched_pred = pred_by_gt.get(gt_idx)
        hit = 1 if (matched_pred is not None and matched_pred.base_pack_id == gt_sku) else 0

        # 1. Optical Glare Slice
        slice_hits["high_glare" if glare >= 0.35 else "low_glare"].append(hit)
        # 2. Pack Geometry Slice
        slice_hits["small_narrow_packs" if width_px < 65.0 else "large_bottles"].append(hit)
        # 3. Shelf Density Slice
        slice_hits["dense_bay" if is_dense_bay else "standard_bay"].append(hit)
        # 4. Brand Ownership Slice
        slice_hits["hul_brands" if is_hul else "competitor_brands"].append(hit)

    report: Dict[str, Dict[str, float]] = {}
    for slice_name, hits in sorted(slice_hits.items()):
        acc = sum(hits) / float(len(hits)) if hits else 0.0
        report[slice_name] = {
            "sample_count": float(len(hits)),
            "top1_accuracy": round(acc, 4),
            "error_rate": round(1.0 - acc, 4),
        }
    return report


# ============================================================================
# 3. CONFIDENCE CALIBRATION (ECE, BRIER SCORE) & SKU CONFUSION MATRIX
# ============================================================================
@dataclass(frozen=True)
class CalibrationAndConfusionReport:
    """Confidence calibration metrics and SKU confusion matrix."""

    expected_calibration_error: float
    brier_score: float
    mean_confidence: float
    empirical_accuracy: float
    confusion_matrix: Dict[str, Dict[str, int]]


def compute_calibration_ece_and_brier(
    resolved_skus: Sequence[ResolvedSKU],
    gt_boxes: Sequence[Sequence[float]],
    gt_base_pack_ids: Sequence[str],
    num_bins: int = 10,
    iou_threshold: float = 0.50,
) -> CalibrationAndConfusionReport:
    """Calculate Expected Calibration Error (ECE), Brier Score, and Pairwise SKU Confusion Matrix."""
    pred_boxes = [s.box_xyxy for s in resolved_skus]
    pred_scores = [s.confidence for s in resolved_skus]
    matches, _, _ = greedy_match_boxes(pred_boxes, pred_scores, gt_boxes, iou_threshold)

    if not matches:
        return CalibrationAndConfusionReport(
            expected_calibration_error=0.0,
            brier_score=0.0,
            mean_confidence=0.0,
            empirical_accuracy=0.0,
            confusion_matrix={},
        )

    confidences: List[float] = []
    correctness: List[float] = []
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for pred_idx, gt_idx, _ in matches:
        pred = resolved_skus[pred_idx]
        gt_sku = gt_base_pack_ids[gt_idx]
        is_correct = 1.0 if pred.base_pack_id == gt_sku else 0.0
        confidences.append(float(pred.confidence))
        correctness.append(is_correct)
        confusion[gt_sku][pred.base_pack_id] += 1

    n = float(len(confidences))
    brier = sum((c - y) ** 2 for c, y in zip(confidences, correctness)) / n

    ece = 0.0
    for b in range(num_bins):
        low = b / float(num_bins)
        high = (b + 1) / float(num_bins)
        in_bin = [
            (c, y)
            for c, y in zip(confidences, correctness)
            if (low <= c < high) or (b == num_bins - 1 and c == 1.0)
        ]
        if in_bin:
            bin_conf = sum(x[0] for x in in_bin) / float(len(in_bin))
            bin_acc = sum(x[1] for x in in_bin) / float(len(in_bin))
            ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)

    serializable_cm = {gt: dict(preds) for gt, preds in sorted(confusion.items())}
    return CalibrationAndConfusionReport(
        expected_calibration_error=round(ece, 4),
        brier_score=round(brier, 4),
        mean_confidence=round(sum(confidences) / n, 4),
        empirical_accuracy=round(sum(correctness) / n, 4),
        confusion_matrix=serializable_cm,
    )


# ============================================================================
# 4. DATA QUALITY PROFILING & SPLIT LEAKAGE CHECK (`data-autocleaning`)
# ============================================================================
@dataclass(frozen=True)
class DatasetQualityProfile:
    """Automated dataset hygiene and split contamination report."""

    total_images: int
    total_annotations: int
    null_or_missing_sku_count: int
    degenerate_bbox_count: int
    unknown_catalog_sku_count: int
    public_private_split_leakage_count: int
    is_clean_and_leak_free: bool


def profile_dataset_and_check_leakage(
    slice_data: Dict[str, Any],
    public_split_images: Sequence[str],
    private_split_images: Sequence[str],
    catalog: RPCCatalogAdapter,
) -> DatasetQualityProfile:
    """Verify 0% split contamination, 0 null SKUs, 0 degenerate boxes, and 100% catalog integrity."""
    images_map: Dict[str, List[Dict[str, Any]]] = slice_data.get("images", {})
    valid_skus: Set[str] = catalog.valid_base_pack_ids()

    null_sku = 0
    bad_bbox = 0
    unknown_sku = 0
    total_ann = 0

    for _, records in images_map.items():
        for rec in records:
            total_ann += 1
            sku = str(rec.get("gt_base_pack_id", "")).strip()
            if not sku:
                null_sku += 1
            elif sku not in valid_skus:
                unknown_sku += 1

            box = rec.get("box_xyxy", [])
            if len(box) != 4 or float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
                bad_bbox += 1

    pub_set = set(public_split_images)
    priv_set = set(private_split_images)
    leakage_count = len(pub_set & priv_set)

    is_clean = (
        null_sku == 0
        and bad_bbox == 0
        and unknown_sku == 0
        and leakage_count == 0
        and total_ann > 0
    )
    return DatasetQualityProfile(
        total_images=len(images_map),
        total_annotations=total_ann,
        null_or_missing_sku_count=null_sku,
        degenerate_bbox_count=bad_bbox,
        unknown_catalog_sku_count=unknown_sku,
        public_private_split_leakage_count=leakage_count,
        is_clean_and_leak_free=is_clean,
    )
