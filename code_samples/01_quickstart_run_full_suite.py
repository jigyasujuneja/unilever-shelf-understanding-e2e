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
from pathlib import Path
import tempfile

from shelf_benchmark import ModelPricing, ShelfBenchmarkSDK, UniversalModelSpec
from shelf_benchmark.testing import (
    OFFLINE_IMAGE_URI,
    make_offline_sdk,
    perfect_prediction_payload,
)

# Two built-in approaches that need only a JSON-returning model call.
# `two_stage_physical_crop_per_facing` is left out on purpose: it physically crops each facing,
# which needs the image cropping helper (see the note in ENGINEER_ONBOARDING_GUIDE.md).
OFFLINE_APPROACHES = [
    "single_pass_full_shelf",
    "two_stage_bbox_guided_nms",
]

# Stage 1 of the two-stage approach asks for boxes only. Same three facings as the classification
# payload, so the two approaches are directly comparable.
DETECTION_PAYLOAD = {
    "total_detected_products": 3,
    "detected_products": [
        {"product_index": 1, "bbox_2d": [100, 100, 300, 200], "shelf_row": "top",
         "position_on_shelf": 1, "is_front_facing": True, "confidence": 0.95},
        {"product_index": 2, "bbox_2d": [100, 220, 300, 320], "shelf_row": "top",
         "position_on_shelf": 2, "is_front_facing": True, "confidence": 0.93},
        {"product_index": 3, "bbox_2d": [400, 100, 600, 200], "shelf_row": "middle",
         "position_on_shelf": 1, "is_front_facing": True, "confidence": 0.91},
    ],
}


def offline_model_handler(prompt: str, image_uri: str, response_schema=None):
    """Stand in for Vertex AI, returning the payload the current stage asked for.

    The suite passes the Pydantic response schema it wants, which is how a single handler can
    serve a multi-stage approach: boxes for the detection stage, attributes for the
    classification stage.
    """
    schema_name = getattr(response_schema, "__name__", "")
    if schema_name == "ProductDetectionOutput":
        return DETECTION_PAYLOAD
    return perfect_prediction_payload()


def run_offline() -> None:
    work = Path(tempfile.mkdtemp(prefix="shelf-quickstart-"))
    sdk = make_offline_sdk(work)

    # Register a stand-in model. `custom_callable` is the supported way to keep `run_suite` off the
    # network: the suite calls your handler instead of Vertex AI. Naming a real model id such as
    # "gemini-3.8-flash" here would build a real Vertex AI client and issue a billable call, even
    # with `offline.enabled = True`, because offline mode gates GCS rather than model calls.
    sdk.register_model(
        UniversalModelSpec(
            model_id="offline-demo-model",
            display_name="offline-demo-model",
            provider_family="custom_callable",
            custom_handler=offline_model_handler,
            # Without this the suite warns and falls back to the 'default' rate card.
            pricing=ModelPricing(input=0.30, thinking=0.30, output=2.50),
        )
    )

    summary = sdk.run_suite(
        models=["offline-demo-model"],
        tasks=["detection", "classification"],
        approaches=OFFLINE_APPROACHES,
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )

    print(f"Completed {len(summary['results'])} benchmark runs (offline).")
    for res in summary["results"]:
        acc = res.accuracy
        print(
            f"  {res.separation_approach:<28} status={res.status} "
            f"facings={len(res.row_level_items)} "
            f"tokens={res.tokens.total_tokens} "
            f"cost=${res.cost.cost_per_shelf_image_usd:.6f} "
            f"accuracy_status={acc.accuracy_status} "
            f"detection_f1={acc.detection_f1}"
        )
    print("\nArtifacts:")
    for name, path in summary["artifacts"].items():
        print(f"  {name:<16} {path}")
    print(f"  {'otel_log':<16} {summary['otel_log_path']}")
    print(
        "\nEvery accuracy number above is None on purpose: no annotations exist yet.\n"
        "When they arrive, re-score this run for free against\n"
        f"  {summary['artifacts']['predictions_json']}\n"
        "using `shelf-benchmark score` (see code_samples/10_run_now_score_later_ground_truth.py)."
    )


def run_live() -> None:
    # Loads configs/default_config.yaml and configs/taxonomy.yaml.
    sdk = ShelfBenchmarkSDK(
        config_path="configs/default_config.yaml",
        taxonomy_path="configs/taxonomy.yaml",
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
