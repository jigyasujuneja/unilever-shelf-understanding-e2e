"""Metric aggregation: turning `TaskExecutionResult`s into records that reports render.

This module is deliberately free of I/O and of any notion of output format. It answers
"what are the numbers?"; `tabular` and `markdown` answer "how are they written down?".

The invariant it exists to protect: a metric that was never measured is `None`, and `None`
is *skipped*, never coerced to `0.0`. Averaging an unmeasured metric as zero silently turns
"we have no ground truth" into "the model scored zero", which is the single easiest way for
this suite to report a confident wrong answer.
"""

from __future__ import annotations

import getpass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from shelf_benchmark.models import TaskExecutionResult
from shelf_benchmark.reporting.formatting import UNKNOWN

__all__ = [
    "LEADERBOARD_AVERAGED_METRICS",
    "LEADERBOARD_SUMMED_COUNTERS",
    "build_leaderboard_records",
    "build_provenance",
    "build_summary_records",
    "distinct_labels",
    "leaderboard_fieldnames",
    "leaderboard_sort_key",
    "mean_ignoring_none",
]

# Accuracy metrics that are averaged (skipping `None`) when building the leaderboard.
LEADERBOARD_AVERAGED_METRICS: Sequence[str] = (
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "average_precision_at_50",
    "map_50_95",
    "mean_iou_matched",
    "count_accuracy",
    "brand_classification_accuracy",
    "brand_set_recall",
    "product_classification_accuracy",
    "macro_attribute_accuracy",
    "sku_matching_accuracy",
    "planogram_compliance_rate",
)

# Accuracy counters that are summed across images.
LEADERBOARD_SUMMED_COUNTERS: Sequence[str] = (
    "matched_pairs",
    "true_positives",
    "false_positives",
    "false_negatives",
    "depth_duplicates_filtered",
)


def mean_ignoring_none(values: Iterable[Optional[float]]) -> Tuple[Optional[float], int]:
    """Average only the measured values.

    Returns `(mean, n_contributing)`. `None` entries are skipped rather than treated as
    `0.0`, and a metric measured on zero images returns `(None, 0)` so the report shows
    "not measured" instead of a fabricated zero.
    """
    measured = [float(v) for v in values if v is not None]
    if not measured:
        return None, 0
    return sum(measured) / len(measured), len(measured)


def distinct_labels(values: Iterable[Any]) -> str:
    """Join distinct non-empty provenance labels, e.g. two GT versions in one batch.

    A semicolon is used rather than a pipe so the result stays safe inside a Markdown table cell.
    """
    seen: List[str] = []
    for v in values:
        if v is None or v == "":
            continue
        text = str(v)
        if text not in seen:
            seen.append(text)
    return "; ".join(seen) if seen else UNKNOWN


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown-user"


# ---------------------------------------------------------------------------
# Per-run summary records
# ---------------------------------------------------------------------------


def build_summary_records(results: Sequence[TaskExecutionResult]) -> List[Dict[str, Any]]:
    """One flat record per run, the shared input to the summary CSV/JSON and Markdown."""
    records: List[Dict[str, Any]] = []
    for r in results:
        acc = r.accuracy
        cost = r.cost
        records.append(
            {
                "run_id": r.run_id,
                "trace_id": r.trace_id,
                "span_id": r.span_id,
                "task_type": r.task_type,
                "separation_approach": r.separation_approach,
                "model_name": r.model_name,
                "shelf_image_uri": r.shelf_image_uri,
                "status": r.status,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "latency_ms": r.latency_ms,
                "front_facings_count": cost.product_count,
                "depth_duplicates_filtered": acc.depth_duplicates_filtered,
                "latency_per_facing_ms": round(
                    r.latency_ms / max(cost.product_count, 1), 2
                ),
                "input_tokens": r.tokens.input_tokens,
                "thinking_tokens": r.tokens.thinking_tokens,
                "output_tokens": r.tokens.output_tokens,
                "total_tokens": r.tokens.total_tokens,
                "cost_per_shelf_image_usd": cost.cost_per_shelf_image_usd,
                "cost_per_product_usd": cost.cost_per_product_usd,
                "vertex_ai_payg_tokens_usd": cost.vertex_ai_payg_tokens_usd,
                "vertex_ai_provisioned_throughput_usd": cost.vertex_ai_provisioned_throughput_usd,
                "vertex_ai_embeddings_and_vision_usd": cost.vertex_ai_embeddings_and_vision_usd,
                "cloud_run_compute_usd": cost.cloud_run_compute_usd,
                "gcs_and_observability_usd": cost.gcs_and_observability_usd,
                "traffic_type": cost.traffic_type,
                # Cost provenance: where the rates came from and what is inside the total.
                "billing_source": cost.billing_source,
                "rates_from_live_catalog": cost.rates_from_live_catalog,
                "includes_modelled_infrastructure": cost.includes_modelled_infrastructure,
                "ground_truth_available": acc.ground_truth_available,
                "accuracy_status": (
                    "EVALUATED_AGAINST_GT"
                    if acc.ground_truth_available
                    else acc.accuracy_status
                ),
                # Scoring provenance: no accuracy number below can be read out of context.
                "gt_version": acc.gt_version,
                "iou_threshold": acc.iou_threshold,
                "pairing_strategy": acc.pairing_strategy,
                "brand_matcher": acc.brand_matcher,
                "product_matcher": acc.product_matcher,
                "gt_count": acc.ground_truth_count,
                "predicted_count": acc.predicted_count,
                "matched_pairs": acc.matched_pairs,
                "true_positives": acc.true_positives,
                "false_positives": acc.false_positives,
                "false_negatives": acc.false_negatives,
                "count_accuracy": acc.count_accuracy,
                "mean_iou_matched": acc.mean_iou_matched,
                "detection_precision": acc.detection_precision,
                "detection_recall": acc.detection_recall,
                "detection_f1": acc.detection_f1,
                "average_precision_at_50": acc.average_precision_at_50,
                "map_50_95": acc.map_50_95,
                "brand_classification_accuracy": acc.brand_classification_accuracy,
                "brand_set_recall": acc.brand_set_recall,
                "product_classification_accuracy": acc.product_classification_accuracy,
                "macro_attribute_accuracy": acc.macro_attribute_accuracy,
                "sku_matching_accuracy": acc.sku_matching_accuracy,
                "planogram_compliance_rate": acc.planogram_compliance_rate,
                "execution_environment": (r.execution_trace or {}).get(
                    "execution_environment", "local_non_cloud_run_live_vertex_ai"
                ),
                "execution_trace": r.execution_trace or {},
                "error_message": r.error_message,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def build_provenance(results: Sequence[TaskExecutionResult]) -> Dict[str, Any]:
    """Collects the scoring and billing provenance shared by this batch of results."""
    accs = [r.accuracy for r in results]
    costs = [r.cost for r in results]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by_user": _current_user(),
        "result_count": len(results),
        "ground_truth_available_any": any(a.ground_truth_available for a in accs),
        "ground_truth_available_all": bool(accs) and all(a.ground_truth_available for a in accs),
        "gt_version": distinct_labels(a.gt_version for a in accs),
        "iou_threshold": distinct_labels(a.iou_threshold for a in accs),
        "pairing_strategy": distinct_labels(a.pairing_strategy for a in accs),
        "brand_matcher": distinct_labels(a.brand_matcher for a in accs),
        "product_matcher": distinct_labels(a.product_matcher for a in accs),
        "accuracy_status": distinct_labels(a.accuracy_status for a in accs),
        "billing_source": distinct_labels(c.billing_source for c in costs),
        "rates_from_live_catalog_all": bool(costs) and all(c.rates_from_live_catalog for c in costs),
        "rates_from_live_catalog_any": any(c.rates_from_live_catalog for c in costs),
        "includes_modelled_infrastructure_any": any(
            c.includes_modelled_infrastructure for c in costs
        ),
        "includes_modelled_infrastructure_all": bool(costs)
        and all(c.includes_modelled_infrastructure for c in costs),
    }


# ---------------------------------------------------------------------------
# Aggregate leaderboard
# ---------------------------------------------------------------------------


def build_leaderboard_records(
    results: Sequence[TaskExecutionResult],
) -> List[Dict[str, Any]]:
    """Averages per-image metrics for each `(approach_id, model_name)` pair.

    Metrics that are `None` on an image are skipped for that image rather than counted
    as `0.0`, and the number of images that actually contributed to each average is
    emitted alongside it (`<metric>_n_images`).
    """
    groups: Dict[Tuple[str, str], List[TaskExecutionResult]] = {}
    for r in results:
        groups.setdefault((r.separation_approach, r.model_name), []).append(r)

    records: List[Dict[str, Any]] = []
    for (approach_id, model_name), grouped in groups.items():
        accs = [g.accuracy for g in grouped]
        costs = [g.cost for g in grouped]
        record: Dict[str, Any] = {
            "approach_id": approach_id,
            "model_name": model_name,
            "task_types": distinct_labels(sorted({g.task_type for g in grouped})),
            "images_scored": len(grouped),
            "images_with_ground_truth": sum(1 for a in accs if a.ground_truth_available),
            "successful_runs": sum(1 for g in grouped if g.status == "SUCCESS"),
        }

        for metric in LEADERBOARD_AVERAGED_METRICS:
            mean_value, n_contributing = mean_ignoring_none(getattr(a, metric) for a in accs)
            record[metric] = None if mean_value is None else round(mean_value, 4)
            record[f"{metric}_n_images"] = n_contributing

        for counter in LEADERBOARD_SUMMED_COUNTERS:
            # Sum only the images that were actually scored. If none were, the total
            # stays None: reporting "0 false negatives" for an unscored group would
            # read as a perfect recall result.
            contributing = [
                getattr(a, counter) for a in accs if getattr(a, counter, None) is not None
            ]
            record[counter] = sum(contributing) if contributing else None

        avg_latency, _ = mean_ignoring_none(g.latency_ms for g in grouped)
        avg_cost_image, _ = mean_ignoring_none(c.cost_per_shelf_image_usd for c in costs)
        avg_cost_product, _ = mean_ignoring_none(c.cost_per_product_usd for c in costs)
        avg_tokens, _ = mean_ignoring_none(g.tokens.total_tokens for g in grouped)

        record.update(
            {
                "avg_latency_ms": None if avg_latency is None else round(avg_latency, 1),
                "avg_cost_per_shelf_image_usd": (
                    None if avg_cost_image is None else round(avg_cost_image, 8)
                ),
                "total_cost_usd": round(sum(c.cost_per_shelf_image_usd for c in costs), 8),
                "avg_cost_per_product_usd": (
                    None if avg_cost_product is None else round(avg_cost_product, 8)
                ),
                "avg_total_tokens": None if avg_tokens is None else round(avg_tokens, 1),
                "gt_version": distinct_labels(a.gt_version for a in accs),
                "iou_threshold": distinct_labels(a.iou_threshold for a in accs),
                "pairing_strategy": distinct_labels(a.pairing_strategy for a in accs),
                "brand_matcher": distinct_labels(a.brand_matcher for a in accs),
                "product_matcher": distinct_labels(a.product_matcher for a in accs),
                "accuracy_status": distinct_labels(a.accuracy_status for a in accs),
                "billing_source": distinct_labels(c.billing_source for c in costs),
                "rates_from_live_catalog": bool(costs)
                and all(c.rates_from_live_catalog for c in costs),
                "includes_modelled_infrastructure": any(
                    c.includes_modelled_infrastructure for c in costs
                ),
            }
        )
        records.append(record)

    records.sort(key=leaderboard_sort_key)
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    return records


def leaderboard_sort_key(record: Dict[str, Any]) -> tuple:
    """Sort by detection F1 desc, then product classification accuracy desc.

    Unmeasured (`None`) metrics rank after every measured value instead of being treated
    as zero, so a model that was never scored does not appear to have lost.
    """
    f1 = record.get("detection_f1")
    product_acc = record.get("product_classification_accuracy")
    return (
        0 if f1 is not None else 1,
        -(f1 if f1 is not None else 0.0),
        0 if product_acc is not None else 1,
        -(product_acc if product_acc is not None else 0.0),
        str(record.get("approach_id", "")),
        str(record.get("model_name", "")),
    )


def leaderboard_fieldnames(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Stable leaderboard CSV column order, even when there are no records."""
    fieldnames: List[str] = [
        "rank",
        "approach_id",
        "model_name",
        "task_types",
        "images_scored",
        "images_with_ground_truth",
        "successful_runs",
    ]
    for metric in LEADERBOARD_AVERAGED_METRICS:
        fieldnames.append(metric)
        fieldnames.append(f"{metric}_n_images")
    fieldnames.extend(LEADERBOARD_SUMMED_COUNTERS)
    fieldnames.extend(
        [
            "avg_latency_ms",
            "avg_cost_per_shelf_image_usd",
            "avg_cost_per_product_usd",
            "total_cost_usd",
            "avg_total_tokens",
            "gt_version",
            "iou_threshold",
            "pairing_strategy",
            "brand_matcher",
            "product_matcher",
            "accuracy_status",
            "billing_source",
            "rates_from_live_catalog",
            "includes_modelled_infrastructure",
        ]
    )
    for rec in records:
        for key in rec:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames
