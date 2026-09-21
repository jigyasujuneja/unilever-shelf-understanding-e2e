"""CLI Entrypoint (`shelf-benchmark`) for out-of-the-box execution."""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shelf-benchmark",
        description="Unilever Shelf Understanding GCP & Vertex AI Gemini Benchmark Suite",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML benchmark configuration file.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["detection", "classification", "matching", "fine_tuning"],
        choices=["detection", "classification", "matching", "fine_tuning", "all"],
        help="Tasks to benchmark (default: all four separated tasks).",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        default=[
            "single_pass_full_shelf",
            "two_stage_bbox_guided_nms",
            "two_stage_physical_crop_per_facing",
        ],
        choices=[
            "single_pass_full_shelf",
            "two_stage_bbox_guided_nms",
            "two_stage_physical_crop_per_facing",
        ],
        help="Bounding-box separation approaches to benchmark during classification.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Gemini models to benchmark (e.g., gemini-3.8-flash gemini-3.7-flash gemini-3.5-flash-lite).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Specific shelf image (`gs://...` URI or local path) to benchmark.",
    )
    parser.add_argument(
        "--upload-to-gcs",
        action="store_true",
        help="If --image is a local file, upload it automatically to the configured shelf_images_bucket.",
    )
    parser.add_argument(
        "--shelf-bucket",
        type=str,
        default=None,
        help="Override shelf images GCS bucket URI.",
    )
    parser.add_argument(
        "--catalog-bucket",
        type=str,
        default=None,
        help="Override product catalog GCS bucket URI.",
    )
    parser.add_argument(
        "--planogram-bucket",
        type=str,
        default=None,
        help="Override optional planograms GCS bucket URI (set to 'none' to disable planograms).",
    )
    parser.add_argument(
        "--submit-tuning-job",
        action="store_true",
        help="Submit a live Vertex AI SFT tuning job during fine_tuning task.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for generated reports.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = BenchmarkConfig.from_yaml(args.config)
    if args.shelf_bucket:
        config.buckets.shelf_images_bucket = args.shelf_bucket
    if args.catalog_bucket:
        config.buckets.catalog_images_bucket = args.catalog_bucket
    if args.planogram_bucket:
        config.buckets.planograms_bucket = None if args.planogram_bucket.lower() == "none" else args.planogram_bucket
    if args.output_dir:
        config.reporting.output_dir = args.output_dir

    selected_tasks = args.tasks
    if "all" in selected_tasks:
        selected_tasks = ["detection", "classification", "matching", "fine_tuning"]

    runner = BenchmarkRunner(config=config)
    summary = runner.run_benchmark(
        tasks=selected_tasks,
        models=args.models,
        classification_approaches=args.approaches,
        shelf_image_uri=args.image,
        upload_to_gcs=args.upload_to_gcs,
        submit_live_tuning_job=args.submit_tuning_job,
    )

    print("\n=== Benchmark Completed Successfully ===")
    print(json.dumps(summary["report_paths"], indent=2))
    print(f"OpenTelemetry JSONL Log: {summary['otel_log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
