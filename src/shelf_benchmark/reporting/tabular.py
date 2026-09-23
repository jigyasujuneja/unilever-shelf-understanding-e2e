"""CSV and JSON report writers.

Every writer here goes through `shelf_benchmark.artifacts`, so a crash or a full disk
cannot leave a truncated-but-parseable report behind: readers see either the previous
complete file or the new one. See `artifacts` for why that matters for this particular
set of files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from shelf_benchmark.artifacts import atomic_text_writer, atomic_write_json
from shelf_benchmark.models import RowLevelReportItem, TaskExecutionResult
from shelf_benchmark.reporting.aggregation import leaderboard_fieldnames

__all__ = [
    "SUMMARY_FIELDNAMES",
    "write_leaderboard_csv",
    "write_row_level_csv",
    "write_row_level_json",
    "write_summary_csv",
    "write_summary_json",
]

# Canonical column order for `benchmark_summary.csv` (kept stable for downstream consumers).
SUMMARY_FIELDNAMES: Sequence[str] = (
    "run_id",
    "trace_id",
    "span_id",
    "task_type",
    "separation_approach",
    "model_name",
    "shelf_image_uri",
    "status",
    "start_time",
    "end_time",
    "latency_ms",
    "front_facings_count",
    "depth_duplicates_filtered",
    "latency_per_facing_ms",
    "input_tokens",
    "thinking_tokens",
    "output_tokens",
    "total_tokens",
    "cost_per_shelf_image_usd",
    "cost_per_product_usd",
    "vertex_ai_payg_tokens_usd",
    "vertex_ai_provisioned_throughput_usd",
    "vertex_ai_embeddings_and_vision_usd",
    "cloud_run_compute_usd",
    "gcs_and_observability_usd",
    "traffic_type",
    "billing_source",
    "rates_from_live_catalog",
    "includes_modelled_infrastructure",
    "ground_truth_available",
    "accuracy_status",
    "gt_version",
    "iou_threshold",
    "pairing_strategy",
    "brand_matcher",
    "product_matcher",
    "gt_count",
    "predicted_count",
    "matched_pairs",
    "true_positives",
    "false_positives",
    "false_negatives",
    "count_accuracy",
    "mean_iou_matched",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "brand_classification_accuracy",
    "brand_set_recall",
    "product_classification_accuracy",
    "sku_matching_accuracy",
    "planogram_compliance_rate",
    "error_message",
)


def write_row_level_csv(rows: Sequence[RowLevelReportItem], path: str | Path) -> Path:
    fieldnames = list(RowLevelReportItem.model_fields.keys())
    with atomic_text_writer(path, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.model_dump())
    return Path(path)


def write_row_level_json(rows: Sequence[RowLevelReportItem], path: str | Path) -> Path:
    return atomic_write_json(path, [r.model_dump() for r in rows])


def write_summary_csv(records: Sequence[Dict[str, Any]], path: str | Path) -> Path:
    fieldnames = list(SUMMARY_FIELDNAMES)
    if records:
        # Tolerate extra keys that a caller-built record might carry.
        for key in records[0]:
            if key not in fieldnames:
                fieldnames.append(key)
    with atomic_text_writer(path, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    return Path(path)


def write_summary_json(
    summary_records: Sequence[Dict[str, Any]],
    full_results: Sequence[TaskExecutionResult],
    path: str | Path,
    leaderboard_records: Optional[Sequence[Dict[str, Any]]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Path:
    # Imported lazily to keep this module's import graph free of the aggregation layer
    # for the common case where the caller already computed provenance.
    if provenance is None:
        from shelf_benchmark.reporting.aggregation import build_provenance

        provenance = build_provenance(full_results)

    payload = {
        "provenance": provenance,
        "summary": list(summary_records),
        "leaderboard": list(leaderboard_records) if leaderboard_records is not None else [],
        "detailed_runs": [r.model_dump() for r in full_results],
    }
    return atomic_write_json(path, payload)


def write_leaderboard_csv(records: Sequence[Dict[str, Any]], path: str | Path) -> Path:
    fieldnames: List[str] = leaderboard_fieldnames(records)
    with atomic_text_writer(path, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    return Path(path)
