#!/usr/bin/env python3
"""Sample 01: Quickstart - run the benchmark suite end to end.

Run it (no GCP project, no credentials, no network):

    .venv/bin/python code_samples/01_quickstart_run_full_suite.py

Run it against live Vertex AI models (needs `gcloud auth application-default login`
and read access to the configured GCS buckets):

    .venv/bin/python code_samples/01_quickstart_run_full_suite.py --live

The offline path uses the bundled 600x400 fixture image and a canned model response, so the
plumbing exercised here (approach dispatch, taxonomy, cost maths, scoring, report writing) is
exactly the plumbing used in a live run. Only the model call is faked.

Accuracy note: no ground truth is configured, so every accuracy metric comes back as `None`
with `accuracy_status="PLACEHOLDER_AWAITING_GROUND_TRUTH"`. `None` means "not measured"; it is
deliberately not 0.0, which would mean "measured, and the model got everything wrong".
See `docs/EVALUATION_PROTOCOL.md`.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from shelf_benchmark import ShelfBenchmarkSDK
from shelf_benchmark.testing import OFFLINE_IMAGE_URI, benchmark_harness

# Built-in single-step and multi-stage approaches:
# 1. single_pass_full_shelf               -> 1-Step VLM: Detect + 7-Dim Classify in 1 API call
# 2. single_step_detect_classify_and_match -> 1-Step End-to-End VLM: Detect + 7-Dim Classify + SKU Matching in 1 API call
# 3. two_stage_bbox_guided_nms            -> 2-Stage VLM: Stage 1 Detect + 1D-NMS -> Stage 2 BBox-Guided Classify
# 4. two_stage_physical_crop_per_facing   -> 2-Stage VLM: Stage 1 Detect + 1D-NMS + Physical Crop -> Stage 2 Classify
OFFLINE_APPROACHES = [
    "single_pass_full_shelf",
    "single_step_detect_classify_and_match",
    "two_stage_bbox_guided_nms",
    "two_stage_physical_crop_per_facing",
]


def run_offline() -> None:
    work = Path(tempfile.mkdtemp(prefix="shelf-quickstart-"))
    sdk = benchmark_harness(work, with_ground_truth=False)

    summary = sdk.run_suite(
        models=["offline-demo-model"],
        tasks=["detection", "classification"],
        approaches=OFFLINE_APPROACHES,
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )

    print(f"Completed {len(summary['results'])} benchmark runs (offline).")
    for res in summary["results"]:
        acc = res.accuracy
        trace = res.execution_trace
        print(
            f"  {res.separation_approach:<38} "
            f"calls={trace.get('api_calls_count')} "
            f"facings={len(res.row_level_items)} "
            f"cost=${res.cost.cost_per_shelf_image_usd:.6f} "
            f"accuracy_status={acc.accuracy_status} "
            f"detection_f1={acc.detection_f1}"
        )
    print("\nArtifacts:")
    for name, path in summary["artifacts"].items():
        print(f"  {name:<16} {path}")
    print(f"  {'otel_log':<16} {summary['otel_log_path']}")
    print(
        "\nEvery accuracy number above is None on purpose: no ground-truth annotations were connected.\n"
        "When ground truth arrives, re-score this run for free against\n"
        f"  {summary['artifacts']['predictions_json']}\n"
        "using `shelf-benchmark score` (see code_samples/10_run_now_score_later_ground_truth.py)."
    )


def run_live() -> None:
    # Loads configs/default_config.yaml; the taxonomy comes from the installed package.
    sdk = ShelfBenchmarkSDK(
        config_path="configs/default_config.yaml",
        output_dir="reports/sample_01_quickstart",
    )

    summary = sdk.run_suite(
        models=[
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
        ],
        tasks=["detection", "classification", "matching"],
        approaches=[
            "single_pass_full_shelf",
            "two_stage_bbox_guided_nms",
            "two_stage_physical_crop_per_facing",
        ],
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
    )

    print(f"Completed {len(summary['results'])} benchmark runs (live).")
    print(f"OpenTelemetry JSONL log: {summary['otel_log_path']}")
    for name, path in summary["artifacts"].items():
        print(f"Generated {name}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call real Vertex AI models instead of the offline fixture lane.",
    )
    args = parser.parse_args()
    run_live() if args.live else run_offline()


if __name__ == "__main__":
    main()
