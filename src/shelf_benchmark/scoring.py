"""Re-score an existing benchmark run against ground truth, without re-running inference.

Why this module exists
----------------------
Inference costs money and takes minutes; scoring is free and takes milliseconds. Coupling them
means that every time you fix a matcher, adjust an IoU threshold, or receive a corrected
annotation batch, you pay for the whole run again. Worse, it makes it impossible to run the
benchmark *before* ground truth exists -- which is exactly the situation this project is in.

So a run writes `predictions.json` (pure model output, no scoring), and this module turns that
file plus a ground-truth source into a fresh set of metrics and reports.

Workflow
--------
1.  Today, with no annotations:
        shelf-benchmark run --approaches single_pass_full_shelf
    produces `reports/predictions.json` with `accuracy_status="NO_GROUND_TRUTH"` everywhere.

2.  When annotations arrive:
        shelf-benchmark score --predictions reports/predictions.json \\
            --ground-truth-uri gs://bucket/annotations_v1.json --gt-version v1
    produces fully populated metrics for every run you have ever done, at zero inference cost.

3.  When the annotations are revised, re-run step 2 with `--gt-version v2`. Both result sets are
    stamped with their version, so they can never be silently compared.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from shelf_benchmark.config import BenchmarkConfig, EvaluationConfig
from shelf_benchmark.evaluation.metrics import evaluate_task_accuracy
from shelf_benchmark.models import (
    AccuracyMetrics,
    CostMetrics,
    RowLevelReportItem,
    TaskExecutionResult,
    TokenUsageMetrics,
)

logger = logging.getLogger(__name__)

PREDICTIONS_SCHEMA_VERSION = 1


class ScoringError(RuntimeError):
    """Raised when a predictions file cannot be read or scored."""


def write_predictions_file(
    results: List[TaskExecutionResult],
    path: str | Path,
    config: Optional[BenchmarkConfig] = None,
) -> Path:
    """Serialize raw model output so the run can be re-scored later.

    Deliberately stores predictions only -- no accuracy numbers. Anything derived from ground
    truth belongs to the scoring step, and baking a stale score into this file would defeat the
    purpose.
    """
    payload: Dict[str, Any] = {
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "runs": [],
    }
    if config is not None:
        payload["models"] = list(config.models)
        payload["approaches"] = list(config.approaches)

    for res in results:
        payload["runs"].append(
            {
                "run_id": res.run_id,
                "trace_id": res.trace_id,
                "span_id": res.span_id,
                "task_type": res.task_type,
                "separation_approach": res.separation_approach,
                "model_name": res.model_name,
                "shelf_image_uri": res.shelf_image_uri,
                "start_time": res.start_time,
                "end_time": res.end_time,
                "latency_ms": res.latency_ms,
                "status": res.status,
                "error_message": res.error_message,
                "tokens": res.tokens.model_dump(),
                "cost": res.cost.model_dump(),
                "rows": [r.model_dump() for r in res.row_level_items],
            }
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %d run(s) of raw predictions to '%s'.", len(payload["runs"]), out)
    return out


def load_predictions_file(path: str | Path) -> Dict[str, Any]:
    """Read and validate a predictions file."""
    p = Path(path)
    if not p.exists():
        raise ScoringError(
            f"Predictions file not found: '{p}'. Produce one with a benchmark run "
            f"(reporting.write_predictions_file must be enabled)."
        )
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScoringError(f"Predictions file '{p}' is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or "runs" not in payload:
        raise ScoringError(
            f"Predictions file '{p}' is missing the top-level 'runs' key. "
            f"Expected a file written by `shelf_benchmark.scoring.write_predictions_file`."
        )
    version = int(payload.get("schema_version", 0))
    if version > PREDICTIONS_SCHEMA_VERSION:
        raise ScoringError(
            f"Predictions file '{p}' uses schema version {version}, but this build understands "
            f"at most {PREDICTIONS_SCHEMA_VERSION}. Upgrade the package to score it."
        )
    return payload


def score_predictions(
    predictions_path: str | Path,
    gt_provider: Any,
    evaluation_config: Optional[EvaluationConfig] = None,
    gt_version: str = "unversioned",
) -> List[TaskExecutionResult]:
    """Re-score stored predictions against a ground-truth provider.

    Args:
        predictions_path: A file written by `write_predictions_file`.
        gt_provider: Any object with `get_ground_truth(shelf_image_uri, ground_truth_id)`.
        evaluation_config: Scoring policy. Defaults to the built-in policy.
        gt_version: Version label stamped on every rescored row.

    Returns:
        Rebuilt `TaskExecutionResult` objects with freshly computed accuracy. Token counts, costs,
        and latencies are carried through unchanged from the original run, because those were
        measured then and re-scoring does not change them.
    """
    payload = load_predictions_file(predictions_path)
    eval_cfg = evaluation_config or EvaluationConfig()
    results: List[TaskExecutionResult] = []
    matched_images = 0

    for run in payload.get("runs", []):
        rows = [RowLevelReportItem.model_validate(r) for r in run.get("rows", [])]
        image_uri = run.get("shelf_image_uri", "")
        gt = gt_provider.get_ground_truth(
            shelf_image_uri=image_uri,
            ground_truth_id=Path(image_uri).name if image_uri else None,
        )
        if gt is not None:
            matched_images += 1

        accuracy: AccuracyMetrics = evaluate_task_accuracy(
            task_type=run.get("task_type", "classification"),
            rows=rows,
            ground_truth=gt,
            config=eval_cfg,
        )
        for r in rows:
            r.gt_version = accuracy.gt_version or gt_version
            r.iou_threshold = accuracy.iou_threshold

        results.append(
            TaskExecutionResult(
                run_id=run.get("run_id", ""),
                trace_id=run.get("trace_id", ""),
                span_id=run.get("span_id", ""),
                task_type=run.get("task_type", "classification"),
                separation_approach=run.get("separation_approach", ""),
                model_name=run.get("model_name", ""),
                shelf_image_uri=image_uri,
                start_time=run.get("start_time", ""),
                end_time=run.get("end_time", ""),
                latency_ms=float(run.get("latency_ms", 0.0) or 0.0),
                tokens=TokenUsageMetrics.model_validate(run.get("tokens", {})),
                cost=CostMetrics.model_validate(run.get("cost", {})),
                accuracy=accuracy,
                raw_output={"rescored_from": str(predictions_path)},
                row_level_items=rows,
                status=run.get("status", "SUCCESS"),
                error_message=run.get("error_message"),
            )
        )

    total = len(results)
    if total and matched_images == 0:
        raise ScoringError(
            f"None of the {total} run(s) in '{predictions_path}' matched a ground-truth entry. "
            f"This usually means the ground-truth key field does not correspond to the image URIs "
            f"used at run time. Check `image_key_field` in your schema mapping, and compare the "
            f"keys in your annotations against the `shelf_image_uri` values in the predictions "
            f"file. Scoring aborted rather than reporting zeros."
        )
    logger.info(
        "Re-scored %d run(s); %d matched a ground-truth entry (IoU threshold %.2f).",
        total, matched_images, eval_cfg.iou_threshold,
    )
    return results
