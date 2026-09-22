"""Sample 01: Out-of-the-Box Quickstart — Run the Entire Benchmark Suite with OpenTelemetry Logging.

Demonstrates how to run any combination of tasks (`detection`, `classification`, `matching`, `fine_tuning`)
and any of the 4 architectural approaches across Vertex AI models with a single SDK call.
"""

from shelf_benchmark import ShelfBenchmarkSDK


def main() -> None:
    # 1. Initialize the SDK (automatically loads configs/default_config.yaml and configs/taxonomy.yaml)
    sdk = ShelfBenchmarkSDK(
        config_path="configs/default_config.yaml",
        taxonomy_path="configs/taxonomy.yaml",
        output_dir="reports/sample_01_quickstart",
    )

    # 2. Execute the benchmark suite on your target models and approaches
    summary = sdk.run_suite(
        models=[
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
        ],
        tasks=["detection", "classification", "matching", "fine_tuning"],
        approaches=[
            "single_pass_full_shelf",
            "two_stage_bbox_guided_nms",
            "two_stage_physical_crop_per_facing",
            "class_agnostic_visual_embedding",
        ],
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
    )

    print(f"Completed {len(summary['results'])} benchmark runs.")
    print(f"OpenTelemetry JSONL Log: {summary['otel_log_path']}")
    for name, path in summary["artifacts"].items():
        print(f"Generated {name}: {path}")


if __name__ == "__main__":
    main()
