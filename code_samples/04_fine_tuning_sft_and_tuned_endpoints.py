"""Sample 04: Supervised Fine-Tuning (SFT) Dataset Generation, Job Submission & Tuned Endpoint Benchmarking.

Demonstrates how developers can:
  1. Generate and upload a Vertex AI Supervised Fine-Tuning (SFT) JSONL dataset (`sft_training_examples.jsonl`)
  2. Optionally submit a live Vertex AI tuning job (`submit_tuning_job=True`)
  3. Register a deployed Fine-Tuned Vertex AI Endpoint (`projects/.../locations/.../endpoints/...`)
     and benchmark it side-by-side against base Gemini/Gemma models with OpenTelemetry logging.
"""

from shelf_benchmark import ModelPricing, ShelfBenchmarkSDK, UniversalModelSpec
from shelf_benchmark.tasks.fine_tuning import GeminiFineTuningTask


def main() -> None:
    sdk = ShelfBenchmarkSDK(output_dir="reports/sample_04_fine_tuning")

    # -------------------------------------------------------------------------
    # Step 1: Run the Fine-Tuning Task (Generates & uploads SFT JSONL to GCS
    # and benchmarks SFT structured inference with OpenTelemetry logging)
    # -------------------------------------------------------------------------
    ft_task = GeminiFineTuningTask(
        config=sdk.config,
        storage=sdk.storage,
        telemetry=sdk.telemetry,
    )
    ft_result = ft_task.execute(
        model_name="gemini-3.8-flash",
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
        submit_tuning_job=False,  # Set to True to trigger `client.tunings.tune(...)` on Vertex AI
    )
    print(
        f"[SFT Task Complete] RunID={ft_result.run_id} | "
        f"Latency={ft_result.latency_ms:.1f}ms | "
        f"Cost=${ft_result.cost.cost_per_shelf_image_usd:.6f} | "
        f"OTel Trace={ft_result.trace_id}"
    )

    # -------------------------------------------------------------------------
    # Step 2: Register a Fine-Tuned Vertex AI Model Endpoint & Benchmark It
    # -------------------------------------------------------------------------
    tuned_endpoint_spec = UniversalModelSpec(
        model_id="unilever-shelf-sft-v1",
        display_name="gemini-3.8-flash-sft-unilever-v1",
        provider_family="vertex_tuned_endpoint",
        # Replace with your tuned model endpoint URI when your Vertex AI tuning job completes:
        endpoint_uri="gemini-3.8-flash",
        pricing=ModelPricing(input=0.30, thinking=0.30, output=2.50),
    )
    sdk.register_model(tuned_endpoint_spec)

    suite_summary = sdk.run_suite(
        models=["gemini-3.8-flash-sft-unilever-v1"],
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
    )
    print(f"Benchmark reports written to: {suite_summary['artifacts']['markdown_report']}")


if __name__ == "__main__":
    main()
