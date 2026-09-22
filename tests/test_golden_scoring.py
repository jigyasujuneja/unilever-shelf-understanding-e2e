"""Golden tests: pin down the scoring maths against hand-verifiable cases.

Every test here asserts a specific number that a human can verify by inspection. They exist
because the previous scorer produced plausible-looking metrics that were wrong in ways no
smoke test could catch:

* `detection_precision_iou50` actually thresholded at IoU 0.25.
* Product matching used face-wash-specific keyword groups, so "kiwi" matched "cucumber".
* Unmatched predictions were dropped instead of counting as false positives.
* A prediction could be paired with ground truth on brand agreement alone, coupling localization
  and classification so that a model with bad boxes but good guesses scored well on both.

If a future change breaks the scorer, one of these will move off its asserted value.

Run with: `.venv/bin/pytest tests/test_golden_scoring.py -v`
"""

from __future__ import annotations

import pytest

from shelf_benchmark.config import EvaluationConfig
from shelf_benchmark.evaluation.metrics import (
    brands_match,
    compute_iou,
    evaluate_task_accuracy,
    products_match,
)
from shelf_benchmark.models import RowLevelReportItem
from shelf_benchmark.testing import (
    OFFLINE_IMAGE_URI,
    make_offline_sdk,
    perfect_prediction_payload,
    sample_ground_truth,
    shifted_prediction_payload,
    wrong_brand_payload,
)

pytestmark = pytest.mark.offline


def _rows_from_payload(payload: dict) -> list[RowLevelReportItem]:
    """Convert a classification payload into report rows, as the task layer would."""
    rows = []
    for item in payload["classified_products"]:
        rows.append(
            RowLevelReportItem(
                run_id="golden-run",
                trace_id="0" * 32,
                span_id="0" * 16,
                task_type="classification",
                separation_approach="single_pass_full_shelf",
                model_name="golden-test",
                shelf_image_uri=OFFLINE_IMAGE_URI,
                start_time="2024-01-01T00:00:00.000000Z",
                end_time="2024-01-01T00:00:01.000000Z",
                image_latency_ms=1000.0,
                product_index=item["product_index"],
                bbox_ymin=item["bbox_2d"][0],
                bbox_xmin=item["bbox_2d"][1],
                bbox_ymax=item["bbox_2d"][2],
                bbox_xmax=item["bbox_2d"][3],
                predicted_brand=item["brand"],
                predicted_product_name=item["product_name"],
                shelf_row=item.get("shelf_row", ""),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# IoU geometry
# ---------------------------------------------------------------------------

def test_iou_identical_boxes_is_one():
    assert compute_iou([100, 100, 300, 200], [100, 100, 300, 200]) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert compute_iou([0, 0, 100, 100], [500, 500, 600, 600]) == pytest.approx(0.0)


def test_iou_half_overlap_is_one_third():
    """Two equal boxes overlapping on exactly half their area: intersection/union = 1/3."""
    a = [0, 0, 100, 100]
    b = [0, 50, 100, 150]
    assert compute_iou(a, b) == pytest.approx(1.0 / 3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The IoU threshold must be the configured one, not a hidden constant.
# ---------------------------------------------------------------------------

def test_detection_honours_configured_iou_threshold():
    """A pair at IoU ~0.33 is a miss at threshold 0.50 and a hit at threshold 0.25.

    The old scorer hardcoded 0.25 while naming the metric 'iou50', so this exact case was
    silently reported as a true positive at the stricter threshold.
    """
    gt = sample_ground_truth()
    # Shift one box sideways by half its width; leave the others exact.
    payload = perfect_prediction_payload()
    payload["classified_products"][0]["bbox_2d"] = [100, 150, 300, 250]
    rows = _rows_from_payload(payload)

    strict = evaluate_task_accuracy(
        task_type="classification", rows=rows, ground_truth=gt,
        config=EvaluationConfig(iou_threshold=0.50),
    )
    lenient = evaluate_task_accuracy(
        task_type="classification", rows=rows, ground_truth=gt,
        config=EvaluationConfig(iou_threshold=0.25),
    )
    assert strict.true_positives == 2, "the half-shifted box must NOT count at IoU 0.50"
    assert lenient.true_positives == 3, "the half-shifted box must count at IoU 0.25"
    assert strict.iou_threshold == 0.50
    assert lenient.iou_threshold == 0.25


# ---------------------------------------------------------------------------
# End-to-end golden cases through the real task pipeline.
# ---------------------------------------------------------------------------

def test_perfect_prediction_scores_one(tmp_path):
    """An exact reproduction of ground truth must score 1.0 on every metric."""
    acc = _run_offline(tmp_path, "perfect", perfect_prediction_payload())
    assert acc.detection_precision == pytest.approx(1.0)
    assert acc.detection_recall == pytest.approx(1.0)
    assert acc.detection_f1 == pytest.approx(1.0)
    assert acc.mean_iou_matched == pytest.approx(1.0)
    assert acc.brand_classification_accuracy == pytest.approx(1.0)
    assert acc.product_classification_accuracy == pytest.approx(1.0)
    assert acc.count_accuracy == pytest.approx(1.0)
    assert (acc.true_positives, acc.false_positives, acc.false_negatives) == (3, 0, 0)


def test_shifted_boxes_score_zero_detection(tmp_path):
    """Boxes far from the truth must produce zero recall and three false positives.

    Unmatched predictions counting as false positives is what stops a model from improving its
    precision by emitting extra boxes.
    """
    acc = _run_offline(tmp_path, "shifted", shifted_prediction_payload())
    assert acc.detection_precision == pytest.approx(0.0)
    assert acc.detection_recall == pytest.approx(0.0)
    assert acc.detection_f1 == pytest.approx(0.0)
    assert (acc.true_positives, acc.false_positives, acc.false_negatives) == (0, 3, 3)
    # It still found the right NUMBER of things, and we report that honestly.
    assert acc.count_accuracy == pytest.approx(1.0)


def test_localization_and_naming_are_scored_independently(tmp_path):
    """Perfect boxes with wrong brands: detection 1.0, classification 0.0.

    This is the whole reason pairing is geometry-only by default. If pairing rewarded brand
    agreement, this case would lose true positives and the two capabilities would be entangled.
    """
    acc = _run_offline(tmp_path, "wrongbrand", wrong_brand_payload())
    assert acc.detection_precision == pytest.approx(1.0)
    assert acc.detection_recall == pytest.approx(1.0)
    assert acc.mean_iou_matched == pytest.approx(1.0)
    assert acc.brand_classification_accuracy == pytest.approx(0.0)
    assert acc.product_classification_accuracy == pytest.approx(0.0)


def _run_offline(tmp_path, label, payload):
    """Run one offline classification pass and return its AccuracyMetrics."""
    from shelf_benchmark.sdk import UniversalModelSpec

    sdk = make_offline_sdk(tmp_path / label)
    sdk.register_model(
        UniversalModelSpec(
            model_id=f"golden-{label}",
            display_name=f"golden-{label}",
            provider_family="custom_callable",
            custom_handler=lambda p, u, s: payload,
        )
    )
    summary = sdk.run_suite(
        models=[f"golden-{label}"],
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
        ground_truth=sample_ground_truth(),
    )
    result = summary["results"][0]
    assert result.status == "SUCCESS", f"offline run failed: {result.error_message}"
    return result.accuracy


# ---------------------------------------------------------------------------
# Matchers: strict by default, no category-specific magic.
# ---------------------------------------------------------------------------

def test_strict_brand_matcher_is_not_substring_based():
    """'Dove' must not match 'Dover Soap'. Substring containment used to make it match."""
    cfg = EvaluationConfig(brand_matcher="strict")
    assert brands_match("Dove", "Dove", cfg) is True
    assert brands_match("Dove", "Dover Soap", cfg) is False


def test_brand_aliases_fold_known_renames():
    """Curated aliases are explicit data, not inferred similarity."""
    cfg = EvaluationConfig(brand_matcher="strict")
    assert brands_match("Fair and Lovely", "Glow and Lovely", cfg) is True
    assert brands_match("Pond's", "Ponds", cfg) is True


def test_strict_product_matcher_rejects_unrelated_products():
    """The old keyword-group matcher treated any two 'green' products as the same product."""
    cfg = EvaluationConfig(product_matcher="strict")
    assert products_match(
        "Himalaya Purifying Neem Face Wash", "", "Himalaya Purifying Neem Face Wash", cfg
    ) is True
    assert products_match(
        "Himalaya Cucumber Face Wash", "", "Himalaya Kiwi Face Wash", cfg
    ) is False


def test_no_ground_truth_is_reported_as_placeholder_not_zero():
    """Absent ground truth must be distinguishable from a model that scored zero."""
    rows = _rows_from_payload(perfect_prediction_payload())
    acc = evaluate_task_accuracy(
        task_type="classification", rows=rows, ground_truth=None, config=EvaluationConfig()
    )
    assert acc.ground_truth_available is False
    assert acc.detection_f1 is None, "no ground truth must yield None, never 0.0"
    assert acc.brand_classification_accuracy is None
    assert "NO_GROUND_TRUTH" in acc.accuracy_status or "PLACEHOLDER" in acc.accuracy_status


def test_confusion_matrix_counts_are_none_when_unmeasured():
    """TP/FP/FN must be None, not 0, when there is nothing to measure against.

    "0 true positives" is a real result meaning the model matched nothing. Emitting it for an
    unscored run makes a benchmark nobody has ground truth for look like a total failure, and
    it silently poisons any average computed over those runs.
    """
    rows = _rows_from_payload(perfect_prediction_payload())
    acc = evaluate_task_accuracy(
        task_type="classification", rows=rows, ground_truth=None, config=EvaluationConfig()
    )
    assert acc.true_positives is None
    assert acc.false_positives is None
    assert acc.false_negatives is None
    assert acc.matched_pairs is None


def test_confusion_matrix_counts_are_populated_when_scored(tmp_path):
    """The counterpart: once ground truth exists, the counts are real integers."""
    acc = _run_offline(tmp_path, "confusion-perfect", perfect_prediction_payload())
    assert acc.true_positives == 3
    assert acc.false_positives == 0
    assert acc.false_negatives == 0

    shifted = _run_offline(tmp_path, "confusion-shifted", shifted_prediction_payload())
    assert shifted.true_positives == 0, "a fully shifted prediction matches nothing"
    assert shifted.false_positives == 3
    assert shifted.false_negatives == 3
