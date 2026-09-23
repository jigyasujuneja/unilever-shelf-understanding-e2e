#!/usr/bin/env python3
"""Sample 04: Supervised fine-tuning (SFT) dataset generation and tuned-endpoint benchmarking.

Run it (dataset shape only, no network):

    .venv/bin/python code_samples/04_fine_tuning_sft_and_tuned_endpoints.py

Run it for real (writes the JSONL to GCS, optionally submits a Vertex AI tuning job):

    .venv/bin/python code_samples/04_fine_tuning_sft_and_tuned_endpoints.py --live
    .venv/bin/python code_samples/04_fine_tuning_sft_and_tuned_endpoints.py --live --submit-tuning-job

Three things happen here:
  1. `GeminiFineTuningTask` builds a Vertex AI SFT JSONL dataset, where each line is a `contents`
     array pairing a `fileData` GCS image URI with the structured JSON target.
  2. With `submit_live_tuning_job=True` it calls `client.tunings.tune`. That is the real
     kwarg name; `submit_tuning_job=` would be swallowed by `**kwargs` and silently ignored.
  3. A deployed tuned endpoint is registered as just another model, so it can be benchmarked
     side by side with the base models using the same approaches and the same scoring.
"""

from __future__ import annotations

import argparse
import json

from shelf_benchmark import ModelPricing, ShelfBenchmarkSDK, UniversalModelSpec

TUNED_ENDPOINT_SPEC = UniversalModelSpec(
    model_id="unilever-shelf-sft-v1",
    display_name="gemini-3.8-flash-sft-unilever-v1",
    provider_family="vertex_tuned_endpoint",
    # Replace with the tuned endpoint URI once your tuning job completes, for example
    # "projects/unilever-shelf-understanding/locations/us-central1/endpoints/1234567890".
    endpoint_uri="gemini-3.8-flash",
    pricing=ModelPricing(input=0.30, thinking=0.30, output=2.50),
)

# One line of the SFT dataset the task generates, shown so the shape is reviewable offline.
EXAMPLE_SFT_LINE = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {"fileData": {
                    "mimeType": "image/png",
                    "fileUri": "gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
                }},
                {"text": "Classify every product facing on this shelf."},
            ],
        },
        {
            "role": "model",
            "parts": [
                {"text": json.dumps({
                    "total_classified_products": 1,
                    "classified_products": [
                        {
                            "product_index": 1,
                            "bbox_2d": [535, 386, 795, 440],
                            "brand": "Brand_A",
                            "product_name": "Brand_A Radiance Daily Cleanser",
                            "category": "Skin Care",
                            "subcategory": "Face Wash",
                            "variant": "Bright Beauty",
                            "packaging_type": "tube",
                            "pack_type": "Single",
                            "size": "100g",
                        }
                    ],
                })},
            ],
        },
    ]
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Generate and upload the SFT dataset and benchmark the tuned endpoint (needs GCP).",
    )
    parser.add_argument(
        "--submit-tuning-job", action="store_true",
        help="With --live, also submit a real Vertex AI tuning job. This costs money.",
    )
    args = parser.parse_args()

    if not args.live:
        print("SFT dataset line shape (one JSONL record per training example):")
        print(json.dumps(EXAMPLE_SFT_LINE, indent=2)[:900])
        print(
            "\nTuned endpoint registration:\n"
            f"  display_name    : {TUNED_ENDPOINT_SPEC.display_name}\n"
            f"  provider_family : {TUNED_ENDPOINT_SPEC.provider_family}\n"
            f"  endpoint_uri    : {TUNED_ENDPOINT_SPEC.endpoint_uri}\n"
            "\nStopping here. Add --live to generate and upload the dataset, which needs GCP\n"
            "credentials and write access to the configured bucket."
        )
        return

    from shelf_benchmark.tasks.fine_tuning import GeminiFineTuningTask

    sdk = ShelfBenchmarkSDK(output_dir="reports/sample_04_fine_tuning")

    ft_task = GeminiFineTuningTask(
        config=sdk.config,
        storage=sdk.storage,
        telemetry=sdk.telemetry,
    )
    ft_result = ft_task.execute(
        model_name="gemini-3.8-flash",
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
        # The real kwarg. Passing `submit_tuning_job=` instead does nothing at all.
        submit_live_tuning_job=args.submit_tuning_job,
    )
    print(
        f"[SFT task] run_id={ft_result.run_id} status={ft_result.status} "
        f"latency={ft_result.latency_ms:.1f}ms "
        f"cost=${ft_result.cost.cost_per_shelf_image_usd:.6f} trace={ft_result.trace_id}"
    )

    sdk.register_model(TUNED_ENDPOINT_SPEC)
    suite_summary = sdk.run_suite(
        models=[TUNED_ENDPOINT_SPEC.display_name],
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
    )
    print(f"Benchmark reports written to: {suite_summary['artifacts']['markdown_report']}")


if __name__ == "__main__":
    main()
