"""CLI entrypoint (`shelf-benchmark`).

Subcommands
-----------
* `run`             Execute the benchmark and write reports (plus `predictions.json`).
* `score`           Re-score a previous run against ground truth, with no inference cost.
* `list-approaches` Show every registered approach, including plugins added by engineers.
* `validate-gt`     Load a ground-truth file and report what was parsed, without running anything.

`validate-gt` is the one to reach for first when annotations arrive: it answers "did my schema
mapping and bbox format actually work?" in a second, rather than after a full paid run that
returns all zeros.

Invoking with no subcommand defaults to `run`, so existing scripts keep working.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from shelf_benchmark.config import BenchmarkConfig

_TASK_CHOICES = ["detection", "classification", "matching", "fine_tuning", "all"]
_DEFAULT_TASKS = ["detection", "classification", "matching", "fine_tuning"]


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config", type=str, default="configs/default_config.yaml",
        help="Path to the YAML benchmark configuration file.",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Override the output directory for generated reports.",
    )
    p.add_argument(
        "--isolated", action="store_true",
        help=(
            "Write reports into a per-user, per-timestamp subdirectory. Use this when several "
            "engineers run concurrently, so nobody overwrites anyone else's results."
        ),
    )
    p.add_argument(
        "--offline", action="store_true",
        help="Disable every network call (GCS, BigQuery, Cloud Billing, Cloud Logging).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable INFO-level logging (shows ground-truth load stats and pricing provenance).",
    )


def _add_ground_truth_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--ground-truth-uri", type=str, default=None,
        help="Ground-truth annotations file (local path or GCS URI).",
    )
    p.add_argument(
        "--gt-provider", type=str, default=None,
        choices=["json", "jsonl", "csv", "coco", "bigquery", "none"],
        help="Ground-truth source format.",
    )
    p.add_argument(
        "--gt-version", type=str, default=None,
        help=(
            "Version label for this annotation set. Stamped on every row and span so results "
            "from different annotation revisions are never silently compared."
        ),
    )
    p.add_argument(
        "--bbox-format", type=str, default=None,
        choices=["ymin_xmin_ymax_xmax_1000", "coco_xywh_px", "xyxy_px", "xyxy_norm", "yxyx_norm"],
        help=(
            "Bounding-box convention used by the annotations. Declare this explicitly: an "
            "undetected mismatch produces detection metrics that look plausible but mean nothing."
        ),
    )
    p.add_argument(
        "--iou-threshold", type=float, default=None,
        help="IoU at which a predicted box counts as a true positive (default 0.50).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shelf-benchmark",
        description="Shelf Understanding benchmark suite for GCP / Vertex AI models.",
    )
    sub = parser.add_subparsers(dest="command")

    # -- run -----------------------------------------------------------------
    run_p = sub.add_parser("run", help="Execute the benchmark and write reports.")
    _add_config_args(run_p)
    _add_ground_truth_args(run_p)
    run_p.add_argument(
        "--tasks", type=str, nargs="+", default=_DEFAULT_TASKS, choices=_TASK_CHOICES,
        help="Tasks to benchmark.",
    )
    run_p.add_argument(
        "--approaches", type=str, nargs="+", default=None,
        help=(
            "Approach ids to benchmark during classification. Any registered plugin id is valid; "
            "run `shelf-benchmark list-approaches` to see them. Unknown ids fail loudly."
        ),
    )
    run_p.add_argument("--models", type=str, nargs="+", default=None, help="Models to benchmark.")
    run_p.add_argument(
        "--image", type=str, default=None,
        help="Specific shelf image (GCS URI or local path) to benchmark.",
    )
    run_p.add_argument(
        "--upload-to-gcs", action="store_true",
        help="If --image is a local file, upload it to the configured shelf images bucket.",
    )
    run_p.add_argument("--shelf-bucket", type=str, default=None, help="Override shelf images bucket.")
    run_p.add_argument("--catalog-bucket", type=str, default=None, help="Override catalog bucket.")
    run_p.add_argument(
        "--planogram-bucket", type=str, default=None,
        help="Override planograms bucket ('none' disables planograms).",
    )
    run_p.add_argument(
        "--submit-tuning-job", action="store_true",
        help="Submit a live Vertex AI supervised fine-tuning job during the fine_tuning task.",
    )

    # -- score ---------------------------------------------------------------
    score_p = sub.add_parser(
        "score",
        help="Re-score a previous run against ground truth, with no inference cost.",
    )
    _add_config_args(score_p)
    _add_ground_truth_args(score_p)
    score_p.add_argument(
        "--predictions", type=str, required=True,
        help="Path to a predictions.json written by a previous run.",
    )

    # -- list-approaches -----------------------------------------------------
    list_p = sub.add_parser("list-approaches", help="List every registered approach.")
    list_p.add_argument("--verbose", "-v", action="store_true", help="Enable INFO logging.")
    list_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    # -- validate-gt ---------------------------------------------------------
    val_p = sub.add_parser(
        "validate-gt",
        help="Load ground truth and report what was parsed, without running the benchmark.",
    )
    _add_config_args(val_p)
    _add_ground_truth_args(val_p)
    val_p.add_argument(
        "--sample", type=int, default=3,
        help="How many parsed ground-truth entries to print for eyeball verification.",
    )

    return parser


def _apply_common(config: BenchmarkConfig, args: argparse.Namespace) -> BenchmarkConfig:
    if getattr(args, "output_dir", None):
        config.reporting.output_dir = args.output_dir
    if getattr(args, "isolated", False):
        config.reporting.isolate_runs = True
    if getattr(args, "offline", False):
        config.offline.enabled = True
        config.billing.use_live_cloud_billing_catalog_api = False
        config.telemetry.export_to_gcp_cloud_logging = False
        config.telemetry.sync_otel_logs_to_gcs = False
        config.telemetry.sync_otel_jsonl_to_gcs = False
        config.reporting.sync_reports_to_gcs = False
    return config


def _apply_ground_truth(config: BenchmarkConfig, args: argparse.Namespace) -> BenchmarkConfig:
    if getattr(args, "ground_truth_uri", None):
        config.ground_truth.source_uri = args.ground_truth_uri
        # Pointing at a file without naming a provider clearly means "use it".
        if not getattr(args, "gt_provider", None) and config.ground_truth.provider_type == "none":
            config.ground_truth.provider_type = "json"
    if getattr(args, "gt_provider", None):
        config.ground_truth.provider_type = args.gt_provider
    if getattr(args, "gt_version", None):
        config.ground_truth.gt_version = args.gt_version
    if getattr(args, "bbox_format", None):
        config.ground_truth.schema_mapping.bbox_format = args.bbox_format
    if getattr(args, "iou_threshold", None) is not None:
        config.evaluation.iou_threshold = args.iou_threshold
    return config


def _cmd_run(args: argparse.Namespace) -> int:
    from shelf_benchmark.runner import BenchmarkRunner

    config = _apply_ground_truth(_apply_common(BenchmarkConfig.from_yaml(args.config), args), args)
    if args.shelf_bucket:
        config.buckets.shelf_images_bucket = args.shelf_bucket
    if args.catalog_bucket:
        config.buckets.catalog_images_bucket = args.catalog_bucket
    if args.planogram_bucket:
        config.buckets.planograms_bucket = (
            None if args.planogram_bucket.lower() == "none" else args.planogram_bucket
        )

    selected_tasks = args.tasks
    if "all" in selected_tasks:
        selected_tasks = list(_DEFAULT_TASKS)
    approaches = args.approaches if args.approaches is not None else list(config.approaches)

    runner = BenchmarkRunner(config=config)
    summary = runner.run_benchmark(
        tasks=selected_tasks,
        models=args.models,
        classification_approaches=approaches,
        shelf_image_uri=args.image,
        upload_to_gcs=args.upload_to_gcs,
        submit_live_tuning_job=args.submit_tuning_job,
    )

    print("\n=== Benchmark completed ===")
    print(json.dumps(summary.get("report_paths", {}), indent=2))
    print(f"OpenTelemetry JSONL log: {summary.get('otel_log_path')}")
    if config.ground_truth.provider_type == "none":
        print(
            "\nNOTE: no ground truth was configured, so accuracy columns are placeholders, not "
            "zeros. When annotations arrive, re-score this run for free with:\n"
            f"  shelf-benchmark score --predictions "
            f"{config.reporting.output_dir}/predictions.json --ground-truth-uri <file>"
        )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from shelf_benchmark.data.ground_truth import create_ground_truth_provider
    from shelf_benchmark.data.storage import StorageManager
    from shelf_benchmark.reporting.generator import BenchmarkReportGenerator
    from shelf_benchmark.scoring import score_predictions

    config = _apply_ground_truth(_apply_common(BenchmarkConfig.from_yaml(args.config), args), args)
    if config.ground_truth.provider_type == "none":
        print(
            "ERROR: `score` needs ground truth. Pass --ground-truth-uri (and --gt-provider if the "
            "format is not JSON).",
            file=sys.stderr,
        )
        return 2

    storage = StorageManager(
        project_id=config.gcp.project_id,
        bucket_config=config.buckets,
        offline=config.offline.enabled,
    )
    gt_provider = create_ground_truth_provider(
        gt_config=config.ground_truth,
        storage_manager=storage,
        project_id=config.gcp.project_id,
    )
    print(f"Ground truth: {gt_provider.describe()}")

    results = score_predictions(
        predictions_path=args.predictions,
        gt_provider=gt_provider,
        evaluation_config=config.evaluation,
        gt_version=config.ground_truth.gt_version,
    )
    generator = BenchmarkReportGenerator(output_dir=config.reporting.output_dir)
    paths = generator.generate_all_reports(results)

    print(f"\n=== Re-scored {len(results)} run(s) at IoU {config.evaluation.iou_threshold} ===")
    print(json.dumps(paths, indent=2, default=str))
    return 0


def _cmd_list_approaches(args: argparse.Namespace) -> int:
    from shelf_benchmark.approaches.registry import (
        BUILTIN_VLM_CLASSIFICATION_APPROACHES,
        GLOBAL_APPROACH_REGISTRY,
    )

    GLOBAL_APPROACH_REGISTRY.ensure_discovered()
    entries = []
    for approach_id in sorted(
        set(GLOBAL_APPROACH_REGISTRY.list_ids()) | set(BUILTIN_VLM_CLASSIFICATION_APPROACHES)
    ):
        plugin = GLOBAL_APPROACH_REGISTRY.get(approach_id)
        entries.append(
            {
                "approach_id": approach_id,
                "display_name": getattr(plugin, "display_name", "built-in VLM classification task"),
                "category": getattr(plugin, "category", "builtin_vlm"),
                "demo_only": bool(getattr(plugin, "is_demo_only", False)),
                "source": "plugin" if plugin is not None else "builtin",
            }
        )

    if getattr(args, "json", False):
        print(json.dumps({"approaches": entries,
                          "discovery_errors": GLOBAL_APPROACH_REGISTRY.discovery_errors}, indent=2))
        return 0

    print(f"{len(entries)} registered approach(es):\n")
    for e in entries:
        flag = "  [DEMO ONLY]" if e["demo_only"] else ""
        print(f"  {e['approach_id']}{flag}")
        print(f"      {e['display_name']}")
        print(f"      source={e['source']} category={e['category']}")
    if GLOBAL_APPROACH_REGISTRY.discovery_errors:
        print("\nWARNING: some plugin modules failed to import:")
        for mod, err in GLOBAL_APPROACH_REGISTRY.discovery_errors.items():
            print(f"  {mod}: {err}")
    return 0


def _cmd_validate_gt(args: argparse.Namespace) -> int:
    from shelf_benchmark.data.ground_truth import create_ground_truth_provider
    from shelf_benchmark.data.storage import StorageManager

    config = _apply_ground_truth(_apply_common(BenchmarkConfig.from_yaml(args.config), args), args)
    if config.ground_truth.provider_type == "none":
        print("ERROR: pass --ground-truth-uri (and --gt-provider) to validate.", file=sys.stderr)
        return 2

    storage = StorageManager(
        project_id=config.gcp.project_id,
        bucket_config=config.buckets,
        offline=config.offline.enabled,
    )
    provider = create_ground_truth_provider(
        gt_config=config.ground_truth,
        storage_manager=storage,
        project_id=config.gcp.project_id,
    )

    print(f"OK: {provider.describe()}")
    print(f"    bbox_format : {config.ground_truth.schema_mapping.bbox_format}")
    # A ground-truth file may declare its own `gt_version`, in which case it wins over the
    # config. Print what is actually stamped onto scored rows, not what was requested.
    effective_version = getattr(provider, "gt_version", config.ground_truth.gt_version)
    if effective_version != config.ground_truth.gt_version:
        print(
            f"    gt_version  : {effective_version} "
            f"(declared in the file; overrides config value "
            f"{config.ground_truth.gt_version!r})"
        )
    else:
        print(f"    gt_version  : {effective_version}")
    stats = getattr(provider, "load_stats", {}) or {}
    for k, v in stats.items():
        print(f"    {k:12}: {v}")

    entries = provider.parsed_entries()[: max(0, args.sample)]
    if entries:
        print(f"\nFirst {len(entries)} parsed entries (verify the boxes look sane):")
        for key, gt in entries:
            print(f"  key={key!r} facings={gt.total_main_shelf_facings} version={gt.gt_version}")
            for item in gt.items[:3]:
                print(f"      [{item.item_id}] {item.brand} | {item.product_name} | bbox={item.bbox_2d}")
    print(
        "\nBoxes are shown in suite-native [ymin, xmin, ymax, xmax] on a 0-1000 scale. "
        "If these do not correspond to where the products actually are, your --bbox-format is wrong."
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Default to `run` so pre-existing invocations without a subcommand keep working.
    known = {"run", "score", "list-approaches", "validate-gt"}
    if not raw or (raw[0] not in known and raw[0] not in ("-h", "--help")):
        raw = ["run"] + raw

    parser = build_parser()
    args = parser.parse_args(raw)

    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    handlers = {
        "run": _cmd_run,
        "score": _cmd_score,
        "list-approaches": _cmd_list_approaches,
        "validate-gt": _cmd_validate_gt,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
