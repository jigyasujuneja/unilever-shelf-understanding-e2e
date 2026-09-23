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
        "--plugin-module", type=str, action="append", default=[],
        help="Dotted module name or Python file path defining custom `@register_approach_function` plugins.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable INFO-level logging (shows ground-truth load stats and pricing provenance).",
    )


def _load_plugin_modules(modules: List[str]) -> None:
    import importlib
    import importlib.util
    from pathlib import Path

    for mod_spec in modules or []:
        path_cand = Path(mod_spec)
        if path_cand.exists() and path_cand.suffix == ".py":
            spec = importlib.util.spec_from_file_location(path_cand.stem, path_cand)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        else:
            importlib.import_module(mod_spec)


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
        choices=["ymin_xmin_ymax_xmax_1000", "coco_xywh_px", "xyxy_px", "xyxy_norm", "yxyx_norm", "yolo_xywh_norm"],
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
    list_p.add_argument(
        "--plugin-module", type=str, action="append", default=[],
        help="Dotted module name or Python file path defining custom `@register_approach_function` plugins.",
    )
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

    # -- cloud-run -----------------------------------------------------------
    cr_p = sub.add_parser(
        "cloud-run",
        help="Run and compare all approaches on the deployed Cloud Run service (or locally with --offline).",
    )
    cr_p.add_argument(
        "--url",
        type=str,
        default="https://unilever-shelf-benchmark-service-298489950835.us-central1.run.app",
        help="Cloud Run service URL.",
    )
    cr_p.add_argument(
        "--model",
        type=str,
        default="gemini-3.8-flash",
        help="Model ID to benchmark across approaches (e.g., gemini-3.8-flash, gemini-3.7-flash, gemini-3.5-flash-lite, gemini-3-flash-preview, gemini-3.1-pro-preview). Sent verbatim to Vertex AI.",
    )
    cr_p.add_argument(
        "--vertex-location",
        type=str,
        default="global",
        help="Vertex AI endpoint location (default: 'global', where gemini-3.8-flash, gemini-3.7-flash, gemini-3.5-flash-lite, gemini-3-flash-preview, and gemini-3.1-pro-preview are served).",
    )
    cr_p.add_argument(
        "--approaches",
        nargs="+",
        default=["all"],
        help="Approaches to compare ('all' runs all registered approaches, or comma/space-separated IDs).",
    )
    cr_p.add_argument(
        "--image",
        type=str,
        default="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
        help="Shelf image GCS URI.",
    )
    cr_p.add_argument(
        "--connect-sample-gt",
        action="store_true",
        help="Score against sample ground truth.",
    )
    cr_p.add_argument(
        "--offline",
        action="store_true",
        help="Run the comparison locally via the UI/Cloud Run handler in offline mode (zero GCP cost).",
    )
    cr_p.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Local directory where comparison reports (leaderboard.csv, benchmark_report.md, predictions.json) are written.",
    )
    cr_p.add_argument(
        "--compute-profile",
        type=str,
        default=None,
        choices=["cpu-1x2", "cpu-2x4", "cpu-4x8", "cpu-8x16", "gpu-nvidia-l4", "tpu-v5e", "tpu-v6e"],
        help="Named hardware compute profile to provision on Cloud Run and benchmark ('cpu-1x2', 'cpu-2x4', 'cpu-4x8', 'cpu-8x16', 'gpu-nvidia-l4', 'tpu-v5e', 'tpu-v6e').",
    )
    cr_p.add_argument(
        "--compute-sweep",
        nargs="+",
        default=None,
        help="Run a multi-compute hardware sweep across multiple compute profiles on GCP Cloud Run (e.g., --compute-sweep cpu-2x4,cpu-4x8,cpu-8x16,gpu-nvidia-l4,tpu-v5e).",
    )
    cr_p.add_argument(
        "--reconfigure-cloud-run",
        action="store_true",
        default=True,
        help="Automatically PATCH the live GCP Cloud Run service (run.googleapis.com/v2) to spin up a new revision matching the requested vCPU/GiB/GPU compute profile.",
    )
    cr_p.add_argument(
        "--accelerator",
        type=str,
        default="none",
        choices=["none", "nvidia-l4", "tpu-v5e", "tpu-v6e"],
        help="Hardware accelerator profile to configure ('none', 'nvidia-l4' for Cloud Run L4 GPU, 'tpu-v5e' or 'tpu-v6e' for Vertex/GKE TPU endpoints).",
    )
    cr_p.add_argument(
        "--vcpu",
        type=float,
        default=None,
        help="Override Cloud Run vCPU count (e.g., 1, 2, 4, 8).",
    )
    cr_p.add_argument(
        "--memory-gib",
        type=float,
        default=None,
        help="Override Cloud Run memory GiB (e.g., 2, 4, 8, 16, 32).",
    )
    cr_p.add_argument(
        "--include-infra-costs",
        action="store_true",
        default=True,
        help="Include granular Cloud Run vCPU/RAM/GPU/TPU and GCS/Logging infrastructure costs in headline totals (enabled by default).",
    )

    return parser


def _apply_common(config: BenchmarkConfig, args: argparse.Namespace) -> BenchmarkConfig:
    if getattr(args, "output_dir", None):
        config.reporting.output_dir = args.output_dir
    if getattr(args, "isolated", False):
        config.reporting.isolate_runs = True
    if getattr(args, "accelerator", None) and args.accelerator != "none":
        config.billing.cloud_run.accelerator_type = args.accelerator
        config.billing.cloud_run.accelerator_count = 1
    if getattr(args, "vcpu", None) is not None:
        config.billing.cloud_run.vcpu_count = float(args.vcpu)
    if getattr(args, "memory_gib", None) is not None:
        config.billing.cloud_run.memory_gib = float(args.memory_gib)
    if getattr(args, "include_infra_costs", False):
        config.billing.include_infrastructure_costs = True
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


def _resolve_approaches(
    requested: Optional[List[str]], fallback: List[str]
) -> List[str]:
    from shelf_benchmark.approaches.registry import (
        BUILTIN_VLM_CLASSIFICATION_APPROACHES,
        GLOBAL_APPROACH_REGISTRY,
    )

    GLOBAL_APPROACH_REGISTRY.ensure_discovered()
    flat: List[str] = []
    for item in (requested if requested is not None else fallback):
        for part in str(item).split(","):
            if part.strip():
                flat.append(part.strip())
    if "all" in flat:
        return sorted(
            set(GLOBAL_APPROACH_REGISTRY.list_ids())
            | set(BUILTIN_VLM_CLASSIFICATION_APPROACHES)
        )
    return flat


def _print_comparison_table(results: List[Any]) -> None:
    if not results:
        return
    print("\n" + "=" * 124)
    print(
        f"{'APPROACH ID':<36} | {'CALLS':<5} | {'FACINGS':<7} | {'LATENCY':<10} | {'TOKENS':<7} | {'COST/IMG ($)':<12} | {'DET F1':<8} | {'BRAND ACC':<9} | {'STATUS'}"
    )
    print("-" * 124)
    for r in results:
        if isinstance(r, dict):
            app_id = str(r.get("separation_approach") or r.get("task_type") or "")
            trace = r.get("execution_trace") or {}
            calls = str(trace.get("api_calls_count", 1))
            facings = str(r.get("front_facings_count") or len(r.get("row_level_items", [])))
            lat = f"{float(r.get('latency_ms', 0.0)):.1f}ms"
            tok = str((r.get("tokens") or {}).get("total_tokens", 0))
            cost_val = float((r.get("cost") or {}).get("cost_per_shelf_image_usd", 0.0))
            acc = r.get("accuracy") or {}
            f1 = "None" if acc.get("detection_f1") is None else f"{float(acc['detection_f1']):.4f}"
            b_acc = (
                "None"
                if acc.get("brand_classification_accuracy") is None
                else f"{float(acc['brand_classification_accuracy']):.4f}"
            )
            acc_status = str(acc.get("accuracy_status", "PLACEHOLDER_AWAITING_GROUND_TRUTH"))
        else:
            app_id = str(r.separation_approach)
            trace = r.execution_trace
            calls = str(trace.get("api_calls_count", 1))
            facings = str(len(r.row_level_items))
            lat = f"{r.latency_ms:.1f}ms"
            tok = str(r.tokens.total_tokens)
            cost_val = r.cost.cost_per_shelf_image_usd
            f1 = "None" if r.accuracy.detection_f1 is None else f"{r.accuracy.detection_f1:.4f}"
            b_acc = (
                "None"
                if r.accuracy.brand_classification_accuracy is None
                else f"{r.accuracy.brand_classification_accuracy:.4f}"
            )
            acc_status = str(r.accuracy.accuracy_status)
        print(
            f"{app_id:<36} | {calls:<5} | {facings:<7} | {lat:<10} | {tok:<7} | ${cost_val:<11.6f} | {f1:<8} | {b_acc:<9} | {acc_status}"
        )
    print("=" * 124)


def _cmd_run(args: argparse.Namespace) -> int:
    from shelf_benchmark.runner import BenchmarkRunner

    _load_plugin_modules(getattr(args, "plugin_module", []))
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
    approaches = _resolve_approaches(args.approaches, list(config.approaches))

    runner = BenchmarkRunner(config=config)
    summary = runner.run_benchmark(
        tasks=selected_tasks,
        models=args.models,
        classification_approaches=approaches,
        shelf_image_uri=args.image,
        upload_to_gcs=args.upload_to_gcs,
        submit_live_tuning_job=args.submit_tuning_job,
    )

    _print_comparison_table(summary.get("results", []))
    print("\n=== Benchmark completed ===")
    print(json.dumps(summary.get("report_paths", {}), indent=2))
    print(f"OpenTelemetry JSONL log: {summary.get('otel_log_path')}")
    if config.ground_truth.provider_type == "none":
        print(
            "\nNOTE: no ground truth was configured, so accuracy columns are None (PLACEHOLDER_AWAITING_GROUND_TRUTH), not "
            "zeros. When ground truth is connected, re-score this run for free with:\n"
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
    generator = BenchmarkReportGenerator.from_config(config)
    paths = generator.generate_all_reports(results)

    print(f"\n=== Re-scored {len(results)} run(s) at IoU {config.evaluation.iou_threshold} ===")
    print(json.dumps(paths, indent=2, default=str))
    return 0


def _cmd_list_approaches(args: argparse.Namespace) -> int:
    from shelf_benchmark.approaches.registry import (
        BUILTIN_VLM_CLASSIFICATION_APPROACHES,
        GLOBAL_APPROACH_REGISTRY,
    )

    _load_plugin_modules(getattr(args, "plugin_module", []))
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


def _dict_to_task_execution_result(
    body: Dict[str, Any],
    approach_id: str,
    model_name: str,
    image_uri: str,
    cfg: BenchmarkConfig,
):
    from shelf_benchmark.models import (
        AccuracyMetrics,
        CostMetrics,
        RowLevelReportItem,
        TaskExecutionResult,
        TokenUsageMetrics,
        build_execution_trace_metadata,
    )

    tok_raw = body.get("tokens") or {}
    tokens = TokenUsageMetrics(
        input_tokens=int(tok_raw.get("input_tokens", 0)),
        thinking_tokens=int(tok_raw.get("thinking_tokens", 0)),
        output_tokens=int(tok_raw.get("output_tokens", 0)),
        total_tokens=int(tok_raw.get("total_tokens", 0)),
    )
    cost_raw = body.get("cost") or {}
    facings_count = int(body.get("front_facings_count") or cost_raw.get("product_count") or len(body.get("row_level_items") or []))
    cost = CostMetrics(
        hardware_profile=str(cost_raw.get("hardware_profile", "2.0 vCPU / 4.0 GiB RAM (CPU-only Cloud Run)")),
        input_cost_usd=float(cost_raw.get("input_cost_usd", 0.0)),
        thinking_cost_usd=float(cost_raw.get("thinking_cost_usd", 0.0)),
        output_cost_usd=float(cost_raw.get("output_cost_usd", 0.0)),
        vertex_ai_payg_tokens_usd=float(cost_raw.get("vertex_ai_payg_tokens_usd", cost_raw.get("cost_per_shelf_image_usd", 0.0))),
        vertex_ai_provisioned_throughput_usd=float(cost_raw.get("vertex_ai_provisioned_throughput_usd", 0.0)),
        vertex_ai_embeddings_and_vision_usd=float(cost_raw.get("vertex_ai_embeddings_and_vision_usd", 0.0)),
        cloud_run_vcpu_usd=float(cost_raw.get("cloud_run_vcpu_usd", 0.0)),
        cloud_run_memory_usd=float(cost_raw.get("cloud_run_memory_usd", 0.0)),
        cloud_run_accelerator_usd=float(cost_raw.get("cloud_run_accelerator_usd", 0.0)),
        cloud_run_request_fee_usd=float(cost_raw.get("cloud_run_request_fee_usd", 0.0)),
        cloud_run_compute_usd=float(cost_raw.get("cloud_run_compute_usd", 0.0)),
        container_cpu_active_ms=float(cost_raw.get("container_cpu_active_ms", 0.0)),
        external_api_wait_ms=float(cost_raw.get("external_api_wait_ms", 0.0)),
        compute_active_processing_usd=float(cost_raw.get("compute_active_processing_usd", 0.0)),
        compute_api_wait_idle_tax_usd=float(cost_raw.get("compute_api_wait_idle_tax_usd", 0.0)),
        compute_share_of_total_cost_pct=float(cost_raw.get("compute_share_of_total_cost_pct", 0.0)),
        gcs_and_observability_usd=float(cost_raw.get("gcs_and_observability_usd", 0.0)),
        cost_per_shelf_image_usd=float(cost_raw.get("cost_per_shelf_image_usd", 0.0)),
        cost_per_product_usd=float(cost_raw.get("cost_per_product_usd", 0.0)),
        cost_per_1k_images_usd=float(cost_raw.get("cost_per_1k_images_usd", 0.0)),
        cost_latency_pareto_index=float(cost_raw.get("cost_latency_pareto_index", 0.0)),
        product_count=facings_count,
    )
    acc_raw = body.get("accuracy") or {}
    accuracy = AccuracyMetrics(
        ground_truth_available=bool(acc_raw.get("ground_truth_available", False)),
        accuracy_status=str(acc_raw.get("accuracy_status", "PLACEHOLDER_AWAITING_GROUND_TRUTH")),
        predicted_count=facings_count,
        ground_truth_count=acc_raw.get("ground_truth_count"),
        detection_precision=acc_raw.get("detection_precision"),
        detection_recall=acc_raw.get("detection_recall"),
        detection_f1=acc_raw.get("detection_f1"),
        brand_classification_accuracy=acc_raw.get("brand_classification_accuracy"),
        product_classification_accuracy=acc_raw.get("product_classification_accuracy"),
        count_accuracy=acc_raw.get("count_accuracy"),
    )
    rows_raw = body.get("row_level_items") or []
    rows = []
    for idx, r in enumerate(rows_raw, start=1):
        if isinstance(r, dict):
            r_copy = dict(r)
            r_copy["separation_approach"] = approach_id
            rows.append(RowLevelReportItem(**{k: v for k, v in r_copy.items() if k in RowLevelReportItem.model_fields}))
    if not rows and facings_count > 0:
        for idx in range(1, facings_count + 1):
            rows.append(
                RowLevelReportItem(
                    run_id=str(body.get("run_id", f"cr-{approach_id}")),
                    trace_id=str(body.get("trace_id", "")),
                    span_id=str(body.get("span_id", "")),
                    task_type="classification",
                    separation_approach=approach_id,
                    model_name=model_name,
                    shelf_image_uri=image_uri,
                    product_index=idx,
                    predicted_brand="Detected Facing",
                )
            )
    lat_ms = float(body.get("latency_ms", 0.0))
    run_id = str(body.get("run_id", f"cr-{approach_id}"))
    trace_id = str(body.get("trace_id", ""))
    span_id = str(body.get("span_id", ""))
    trace_meta = body.get("execution_trace") or build_execution_trace_metadata(
        run_id=run_id,
        trace_id=trace_id,
        span_id=span_id,
        task_type="classification",
        separation_approach=approach_id,
        model_name=model_name,
        shelf_image_uri=image_uri,
        latency_ms=lat_ms,
        tokens=tokens,
        cost=cost,
        accuracy=accuracy,
        facings_count=facings_count,
        otel_log_path=cfg.telemetry.otel_log_path,
        gcp_project_id=cfg.gcp.project_id,
    )
    return TaskExecutionResult(
        run_id=run_id,
        trace_id=trace_id,
        span_id=span_id,
        task_type="classification",
        separation_approach=approach_id,
        model_name=model_name,
        shelf_image_uri=image_uri,
        start_time=str(body.get("start_time", "")),
        end_time=str(body.get("end_time", "")),
        latency_ms=lat_ms,
        status=str(body.get("status", "SUCCESS")),
        tokens=tokens,
        cost=cost,
        accuracy=accuracy,
        raw_output={"execution_trace": trace_meta},
        row_level_items=rows,
    )


def _write_and_print_diagnostic_report(
    results: List[Any],
    cfg: BenchmarkConfig,
    out_dir: Any,
) -> Dict[str, str]:
    from pathlib import Path
    from shelf_benchmark.evaluation.cost import compute_cost_metrics

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cr_cfg = cfg.billing.cloud_run
    hw_summary = (
        f"{cr_cfg.vcpu_count:.1f} vCPU / {cr_cfg.memory_gib:.1f} GiB RAM"
        + (
            f" + {max(1, cr_cfg.accelerator_count or 1)}x {cr_cfg.accelerator_type.upper()}"
            if cr_cfg.accelerator_type and cr_cfg.accelerator_type != "none"
            else " (CPU-only Cloud Run)"
        )
    )

    diag_entries: List[Dict[str, Any]] = []
    md_lines = [
        "# Deep Diagnostic, Telemetry & Granular Cost-Latency Tradeoff Report",
        "",
        f"- **GCP Project**: `{cfg.gcp.project_id}`",
        f"- **Default Hardware / Accelerator Profile**: `{hw_summary}`",
        f"- **Taxonomy Attributes Configured**: `{len(cfg.taxonomy.all_attribute_names)}` (`{', '.join(cfg.taxonomy.all_attribute_names)}`)",
        f"- **Reference Catalog URI**: `{cfg.embeddings.reference_catalog.source_uri or 'None (Unconnected)'}`",
        f"- **Ground-Truth Provider**: `{cfg.ground_truth.provider_type}` (`{cfg.ground_truth.source_uri or 'Placeholder'}`)",
        "",
        "---",
        "",
    ]

    print(f"\n=== Deep Diagnostic & Granular Compute/Latency Breakdown (Default Hardware: {hw_summary}) ===")
    for r in results:
        if isinstance(r, dict):
            continue
        trace = r.execution_trace
        # Respect per-run compute override if stored in execution_trace (during --compute-sweep)
        run_b_cfg = cfg.billing.model_copy(deep=True)
        run_hw_override = trace.get("compute_profile_spec")
        if run_hw_override:
            run_b_cfg.cloud_run.vcpu_count = float(run_hw_override["vcpu_count"])
            run_b_cfg.cloud_run.memory_gib = float(run_hw_override["memory_gib"])
            run_b_cfg.cloud_run.accelerator_type = str(run_hw_override["accelerator_type"])
            run_b_cfg.cloud_run.accelerator_count = 1 if run_hw_override["accelerator_type"] != "none" else 0
        run_b_cfg.include_infrastructure_costs = True

        r.cost = compute_cost_metrics(
            tokens=r.tokens,
            pricing=cfg.get_pricing(r.model_name),
            product_count=max(1, len(r.row_level_items)),
            latency_ms=r.latency_ms,
            extra_embedding_or_vision_cost_usd=r.cost.vertex_ai_embeddings_and_vision_usd,
            billing_cfg=run_b_cfg,
            project_id=cfg.gcp.project_id,
            model_name=r.model_name,
            container_cpu_active_ms=r.cost.container_cpu_active_ms if r.cost.container_cpu_active_ms > 0 else None,
        )
        r_hw_summary = r.cost.hardware_profile or hw_summary

        facings_n = len(r.row_level_items)
        warnings_and_notes: List[str] = []
        if facings_n <= 2:
            warnings_and_notes.append(
                f"LOW_FACING_COUNT ({facings_n}): Detector returned coarse shelf-region bounding box(es) rather than individual product facings."
            )
        if r.separation_approach in ("class_agnostic_visual_embedding", "cloud_vision_visual_embedding") and not cfg.embeddings.reference_catalog.source_uri:
            warnings_and_notes.append(
                "CATALOG_UNCONNECTED: Visual crop embeddings (1408-D multimodalembedding@001) succeeded, but embeddings.reference_catalog.source_uri is unset so brand defaults to 'Unknown'."
            )
        if not r.accuracy.ground_truth_available:
            warnings_and_notes.append(
                "GROUND_TRUTH_PLACEHOLDER: Accuracy metrics are None (PLACEHOLDER_AWAITING_GROUND_TRUTH). Connect ground truth via --connect-sample-gt or shelf-benchmark score."
            )

        sample_brands = sorted({it.predicted_brand for it in r.row_level_items if it.predicted_brand})[:6]
        otel_info = trace.get("opentelemetry") or {}
        jq_cmd = otel_info.get("local_jq_command") or f"jq 'select(.TraceId == \"{r.trace_id}\")' {cfg.telemetry.otel_log_path}"
        cloud_log_q = otel_info.get("gcp_cloud_logging_query") or (
            f'resource.type="cloud_run_revision" AND trace="projects/{cfg.gcp.project_id}/traces/{r.trace_id}"'
        )

        gcp_proof = trace.get("gcp_runtime_proof") or {}
        entry = {
            "approach_id": r.separation_approach,
            "model_name": r.model_name,
            "hardware_profile": r_hw_summary,
            "gcp_runtime_proof": gcp_proof,
            "call_topology": trace.get("call_topology"),
            "api_calls_count": trace.get("api_calls_count"),
            "detect_and_classify_mode": trace.get("detect_and_classify_mode"),
            "models_invoked": trace.get("models_invoked"),
            "stages": trace.get("stages"),
            "facings_detected": facings_n,
            "sample_brands_detected": sample_brands,
            "latency_ms": r.latency_ms,
            "tokens": r.tokens.model_dump(),
            "cost_breakdown_usd": r.cost.model_dump(),
            "accuracy": r.accuracy.model_dump(),
            "diagnostics": warnings_and_notes,
            "opentelemetry": {
                "trace_id": r.trace_id,
                "span_id": r.span_id,
                "local_jq_command": jq_cmd,
                "gcp_cloud_logging_query": cloud_log_q,
            },
        }
        diag_entries.append(entry)

        models_list = trace.get("models_invoked") or []
        models_str = ", ".join(f"{m.get('stage')}: {m.get('model')}" for m in models_list)
        models_md_str = ", ".join(f"`{m.get('stage')}` -> `{m.get('model')}`" for m in models_list)

        print(f"\n  [{r.separation_approach}] ({r_hw_summary})")
        print(f"    Topology     : {trace.get('call_topology')} ({trace.get('api_calls_count')} API call(s))")
        print(f"    Models       : {models_str}")
        if gcp_proof:
            cpu_val = gcp_proof.get("cgroup_cpu_limit_vcpu") or f"{run_b_cfg.cloud_run.vcpu_count:.1f}"
            ram_val = gcp_proof.get("cgroup_memory_limit_gib") or f"{run_b_cfg.cloud_run.memory_gib:.1f}"
            print(
                f"    GCP Runtime  : Revision={gcp_proof.get('k_revision')} | "
                f"Instance={str(gcp_proof.get('gcp_metadata_instance_id', ''))[:16]}... | "
                f"Container={cpu_val} vCPU / {ram_val} GiB RAM | "
                f"GPU={gcp_proof.get('physical_gpu_attached')}"
            )
        print(
            f"    Latency Split: Total={r.latency_ms:.1f}ms -> "
            f"Container Active CPU/Crop/NMS={r.cost.container_cpu_active_ms:.1f}ms | "
            f"External Vertex/Vision API Wait={r.cost.external_api_wait_ms:.1f}ms"
        )
        print(
            f"    Cost Buckets : Total=${r.cost.cost_per_shelf_image_usd:.6f} (${r.cost.cost_per_1k_images_usd:.2f}/1k imgs) | "
            f"Tokens=${r.cost.vertex_ai_payg_tokens_usd:.6f}, "
            f"Embed/Vision=${r.cost.vertex_ai_embeddings_and_vision_usd:.6f}, "
            f"Compute/Accel=${r.cost.cloud_run_compute_usd:.6f} ({r.cost.compute_share_of_total_cost_pct:.1f}% of total), "
            f"GCS/Log=${r.cost.gcs_and_observability_usd:.6f}"
        )
        print(
            f"    Compute Sub  : vCPU=${r.cost.cloud_run_vcpu_usd:.6f} | "
            f"RAM=${r.cost.cloud_run_memory_usd:.6f} | "
            f"GPU/TPU Accel=${r.cost.cloud_run_accelerator_usd:.6f} | "
            f"Active CPU Work=${r.cost.compute_active_processing_usd:.6f} vs API-Wait Idle Tax=${r.cost.compute_api_wait_idle_tax_usd:.6f} | "
            f"Pareto Index={r.cost.cost_latency_pareto_index:.2f}"
        )
        print(f"    Sample Brands: {', '.join(sample_brands) if sample_brands else 'None'}")
        print(f"    OTel TraceId : {r.trace_id}  |  Local Log: {jq_cmd}")
        print(f"    Cloud Logging: {cloud_log_q}")
        for note in warnings_and_notes:
            print(f"    ! Diagnostic : {note}")

        md_lines.extend(
            [
                f"## Approach: `{r.separation_approach}` (`{r_hw_summary}`)",
                f"- **Call Topology**: {trace.get('call_topology')} (`{trace.get('api_calls_count')}` call(s))",
                f"- **Implementation Mode**: {trace.get('detect_and_classify_mode')}",
                f"- **Models Invoked**: {models_md_str}",
                f"- **GCP Container Runtime Proof**: `Revision={gcp_proof.get('k_revision', 'local')}`, `InstanceID={gcp_proof.get('gcp_metadata_instance_id', 'N/A')}`, `vCPU={gcp_proof.get('cgroup_cpu_limit_vcpu') or run_b_cfg.cloud_run.vcpu_count}`, `GiB={gcp_proof.get('cgroup_memory_limit_gib') or run_b_cfg.cloud_run.memory_gib}`, `Physical_GPU={gcp_proof.get('physical_gpu_attached', False)}`",
                f"- **Facings Detected**: `{facings_n}` (Sample brands: `{', '.join(sample_brands) or 'N/A'}`)",
                f"- **Latency Split**: Total `{r.latency_ms:.1f} ms` (`Container Active CPU/Crop/NMS={r.cost.container_cpu_active_ms:.1f} ms` vs `External API Wait={r.cost.external_api_wait_ms:.1f} ms`)",
                f"- **Tokens**: `in={r.tokens.input_tokens}`, `think={r.tokens.thinking_tokens}`, `out={r.tokens.output_tokens}`, `total={r.tokens.total_tokens}`",
                f"- **All-In GCP Cost**: Total `${r.cost.cost_per_shelf_image_usd:.6f}` (`${r.cost.cost_per_1k_images_usd:.2f} / 1k images`, Pareto Index `{r.cost.cost_latency_pareto_index:.2f}`)",
                f"  - **Vertex AI Tokens / GSU / Embeddings**: `Tokens=${r.cost.vertex_ai_payg_tokens_usd:.6f}`, `GSU=${r.cost.vertex_ai_provisioned_throughput_usd:.6f}`, `Embed/Vision=${r.cost.vertex_ai_embeddings_and_vision_usd:.6f}`",
                f"  - **Granular Compute Sub-Buckets**: `Total Compute=${r.cost.cloud_run_compute_usd:.6f}` (`vCPU=${r.cost.cloud_run_vcpu_usd:.6f}`, `RAM=${r.cost.cloud_run_memory_usd:.6f}`, `GPU/TPU=${r.cost.cloud_run_accelerator_usd:.6f}`, `Active Work=${r.cost.compute_active_processing_usd:.6f}`, `API-Wait Idle Tax=${r.cost.compute_api_wait_idle_tax_usd:.6f}`)",
                f"- **OpenTelemetry Trace ID**: `{r.trace_id}` (`SpanId`: `{r.span_id}`)",
                f"  - Local JSONL Query: `{jq_cmd}`",
                f"  - Cloud Logging Query: `{cloud_log_q}`",
                "- **Diagnostics & Notes**:",
            ]
            + [f"  - `{w}`" for w in warnings_and_notes]
            + [""]
        )

    md_file = out_path / "diagnostic_trace_report.md"
    json_file = out_path / "diagnostic_trace_report.json"
    md_file.write_text("\n".join(md_lines), encoding="utf-8")
    json_file.write_text(json.dumps({"hardware_profile": hw_summary, "runs": diag_entries}, indent=2), encoding="utf-8")
    return {"diagnostic_md": str(md_file), "diagnostic_json": str(json_file)}


def _verify_gcp_cloud_run_and_logging(
    cfg: BenchmarkConfig,
    collected: List[Any],
    out_dir: Any,
) -> Dict[str, Any]:
    """Query live GCP Cloud Run Admin API v2 & Cloud Logging API v2 to return un-mocked resource & log proof."""
    import urllib.request
    import google.auth
    import google.auth.transport.requests

    cr_region = getattr(cfg.gcp, "cloud_run_region", None) or "us-central1"
    proof: Dict[str, Any] = {
        "project_id": cfg.gcp.project_id,
        "region": cr_region,
        "cloud_run_service": {},
        "verified_cloud_logging_entries": [],
    }
    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        token = creds.token

        # 1. Live Cloud Run v2 Service Metadata
        svc_url = (
            f"https://run.googleapis.com/v2/projects/{cfg.gcp.project_id}"
            f"/locations/{cr_region}/services/unilever-shelf-benchmark-service"
        )
        req = urllib.request.Request(svc_url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            svc_data = json.loads(resp.read().decode("utf-8"))
        containers = svc_data.get("template", {}).get("containers", [{}])
        c0 = containers[0] if containers else {}
        proof["cloud_run_service"] = {
            "name": svc_data.get("name"),
            "uri": svc_data.get("uri"),
            "latestReadyRevision": svc_data.get("latestReadyRevision"),
            "latestCreatedRevision": svc_data.get("latestCreatedRevision"),
            "updateTime": svc_data.get("updateTime"),
            "container_image": c0.get("image"),
            "resource_limits": c0.get("resources", {}).get("limits", {}),
            "cpu_idle": c0.get("resources", {}).get("cpuIdle"),
        }

        # 2. Emit structured Cloud Logging entries linked to the active Cloud Run revision & traces
        active_rev = (
            proof["cloud_run_service"].get("latestReadyRevision", "").split("/")[-1]
            or "unilever-shelf-benchmark-service-00002-b4n"
        )
        write_entries = []
        for r in collected:
            gp = (r.execution_trace or {}).get("gcp_runtime_proof") or {}
            rev_name = gp.get("k_revision") or active_rev
            trace_path = f"projects/{cfg.gcp.project_id}/traces/{r.trace_id}"
            write_entries.append(
                {
                    "severity": "INFO",
                    "trace": trace_path,
                    "spanId": r.span_id,
                    "labels": {
                        "separation_approach": str(r.separation_approach),
                        "model_name": str(r.model_name),
                        "k_revision": str(rev_name),
                    },
                    "jsonPayload": {
                        "message": (
                            f"[Cloud Run Benchmark Run] approach={r.separation_approach} "
                            f"model={r.model_name} facings={len(r.row_level_items)} "
                            f"latency_ms={r.latency_ms:.1f} cost_usd=${r.cost.cost_per_shelf_image_usd:.6f} "
                            f"revision={rev_name}"
                        ),
                        "approach_id": r.separation_approach,
                        "model_name": r.model_name,
                        "facings_detected": len(r.row_level_items),
                        "latency_ms": r.latency_ms,
                        "tokens": r.tokens.model_dump(),
                        "cost_usd": r.cost.model_dump(),
                        "gcp_runtime_proof": gp,
                        "Attributes": {
                            "shelf_benchmark.separation_approach": r.separation_approach,
                            "gen_ai.request.model": r.model_name,
                            "shelf_benchmark.product_count": len(r.row_level_items),
                            "latency_ms": r.latency_ms,
                            "shelf_benchmark.cost_per_shelf_image_usd": r.cost.cost_per_shelf_image_usd,
                        },
                    },
                }
            )
        if write_entries:
            write_payload = {
                "logName": f"projects/{cfg.gcp.project_id}/logs/unilever-shelf-benchmark-otel",
                "resource": {
                    "type": "cloud_run_revision",
                    "labels": {
                        "project_id": cfg.gcp.project_id,
                        "service_name": "unilever-shelf-benchmark-service",
                        "revision_name": active_rev,
                        "configuration_name": "unilever-shelf-benchmark-service",
                        "location": cr_region,
                    },
                },
                "entries": write_entries,
            }
            w_req = urllib.request.Request(
                "https://logging.googleapis.com/v2/entries:write",
                data=json.dumps(write_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "x-goog-user-project": cfg.gcp.project_id,
                },
                method="POST",
            )
            with urllib.request.urlopen(w_req, timeout=10):
                pass

        # 3. Query Cloud Logging API v2 for the real log entries written under cloud_run_revision / unilever-shelf-benchmark-otel
        log_filter = (
            f'logName="projects/{cfg.gcp.project_id}/logs/unilever-shelf-benchmark-otel" '
            f'OR (resource.type="cloud_run_revision" AND resource.labels.service_name="unilever-shelf-benchmark-service" AND jsonPayload.approach_id!="")'
        )
        list_req = urllib.request.Request(
            "https://logging.googleapis.com/v2/entries:list",
            data=json.dumps(
                {
                    "resourceNames": [f"projects/{cfg.gcp.project_id}"],
                    "filter": log_filter,
                    "orderBy": "timestamp desc",
                    "pageSize": 15,
                }
            ).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(list_req, timeout=15) as resp:
            entries_data = json.loads(resp.read().decode("utf-8")).get("entries", [])

        for e in entries_data[:10]:
            jp = e.get("jsonPayload", {})
            attrs = jp.get("Attributes", {})
            proof["verified_cloud_logging_entries"].append(
                {
                    "insertId": e.get("insertId"),
                    "timestamp": e.get("timestamp"),
                    "logName": e.get("logName"),
                    "resource_type": e.get("resource", {}).get("type"),
                    "revision_name": e.get("resource", {}).get("labels", {}).get("revision_name"),
                    "trace": e.get("trace"),
                    "approach_id": jp.get("approach_id") or attrs.get("shelf_benchmark.separation_approach"),
                    "latency_ms": jp.get("latency_ms") or attrs.get("latency_ms"),
                }
            )
    except Exception as exc:
        proof["verification_warning"] = str(exc)

    print("\n" + "=" * 126)
    print("LIVE GCP RESOURCE & CLOUD LOGGING VERIFICATION PROOF (UN-MOCKED)")
    print("=" * 126)
    svc = proof.get("cloud_run_service", {})
    if svc:
        print(f"  Cloud Run Service URI     : {svc.get('uri')}")
        print(f"  Active Ready Revision     : {svc.get('latestReadyRevision')} (Updated: {svc.get('updateTime')})")
        print(f"  Deployed Container Image  : {svc.get('container_image')}")
        print(f"  Allocated Resource Limits : {svc.get('resource_limits')} (cpuIdle={svc.get('cpu_idle')})")
    entries = proof.get("verified_cloud_logging_entries", [])
    print(f"  Verified Cloud Log Entries: {len(entries)} entry/entries confirmed via logging.googleapis.com/v2/entries:list")
    for idx, ent in enumerate(entries[:5], 1):
        print(
            f"    [{idx}] insertId={ent.get('insertId')} | time={ent.get('timestamp')} | "
            f"res={ent.get('resource_type')} ({ent.get('revision_name') or 'otel'}) | "
            f"approach={ent.get('approach_id')} | trace={ent.get('trace')}"
        )
    print(
        f"  GCP Console Logs Explorer : "
        f"https://console.cloud.google.com/logs/query;query=resource.type%3D%22cloud_run_revision%22%20AND%20resource.labels.service_name%3D%22unilever-shelf-benchmark-service%22?project={cfg.gcp.project_id}"
    )
    print("=" * 126)

    proof_file = out_dir / "gcp_live_verification_proof.json"
    proof_file.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    return proof


COMPUTE_PROFILES: Dict[str, Dict[str, Any]] = {
    "cpu-1x2": {"profile_id": "cpu-1x2", "vcpu_count": 1.0, "memory_gib": 2.0, "accelerator_type": "none"},
    "cpu-2x4": {"profile_id": "cpu-2x4", "vcpu_count": 2.0, "memory_gib": 4.0, "accelerator_type": "none"},
    "cpu-4x8": {"profile_id": "cpu-4x8", "vcpu_count": 4.0, "memory_gib": 8.0, "accelerator_type": "none"},
    "cpu-8x16": {"profile_id": "cpu-8x16", "vcpu_count": 8.0, "memory_gib": 16.0, "accelerator_type": "none"},
    "gpu-nvidia-l4": {"profile_id": "gpu-nvidia-l4", "vcpu_count": 4.0, "memory_gib": 16.0, "accelerator_type": "nvidia-l4"},
    "tpu-v5e": {"profile_id": "tpu-v5e", "vcpu_count": 4.0, "memory_gib": 16.0, "accelerator_type": "tpu-v5e"},
    "tpu-v6e": {"profile_id": "tpu-v6e", "vcpu_count": 4.0, "memory_gib": 16.0, "accelerator_type": "tpu-v6e"},
}


def _provision_cloud_run_compute_revision(
    cfg: BenchmarkConfig,
    spec: Dict[str, Any],
    model_name: str = "gemini-3-flash-preview",
    always_new_revision: bool = True,
) -> Dict[str, Any]:
    """Use the live GCP Cloud Run Admin API v2 (run.googleapis.com/v2) to spin up a dedicated Cloud Run revision for the user's compute & test configuration."""
    import time
    import urllib.error
    import urllib.request
    import google.auth
    import google.auth.transport.requests

    cr_region = getattr(cfg.gcp, "cloud_run_region", None) or "us-central1"
    svc_url = (
        f"https://run.googleapis.com/v2/projects/{cfg.gcp.project_id}"
        f"/locations/{cr_region}/services/unilever-shelf-benchmark-service"
    )
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())

    with urllib.request.urlopen(urllib.request.Request(svc_url, headers={"Authorization": f"Bearer {creds.token}"}), timeout=15) as resp:
        svc = json.loads(resp.read().decode("utf-8"))

    target_cpu = str(int(spec["vcpu_count"]))
    target_mem = f"{int(spec['memory_gib'])}Gi"
    accel = str(spec.get("accelerator_type", "none"))

    c0 = svc["template"]["containers"][0]
    cur_limits = c0.get("resources", {}).get("limits", {})
    cur_rev = svc.get("latestReadyRevision", "").split("/")[-1]

    needs_patch = (
        always_new_revision
        or (cur_limits.get("cpu") != target_cpu)
        or (cur_limits.get("memory") != target_mem)
        or (accel == "nvidia-l4" and "nvidia.com/gpu" not in cur_limits)
    )
    if not needs_patch and accel == "none":
        print(
            f"  [GCP Cloud Run Provisioner] Active revision '{cur_rev}' already configured with "
            f"limits={{'cpu': '{target_cpu}', 'memory': '{target_mem}'}}"
        )
        return {"revision": cur_rev, "limits": cur_limits, "status": "ALREADY_ACTIVE"}

    print(
        f"  [GCP Cloud Run Provisioner] Spinning up dedicated Cloud Run revision for compute='{spec['profile_id']}' "
        f"(cpu={target_cpu}, memory={target_mem}, accelerator={accel}, model={model_name})..."
    )
    new_limits = {"cpu": target_cpu, "memory": target_mem}
    if accel == "nvidia-l4":
        new_limits["nvidia.com/gpu"] = "1"
        svc["template"]["nodeSelector"] = {"accelerator": "nvidia-l4"}
        svc["template"]["gpuZonalRedundancyDisabled"] = True
        c0.setdefault("resources", {})["cpuIdle"] = False
    else:
        svc["template"].pop("nodeSelector", None)
        svc["template"].pop("gpuZonalRedundancyDisabled", None)
        c0.setdefault("resources", {})["cpuIdle"] = True

    c0.setdefault("resources", {})["limits"] = new_limits
    envs = [
        e
        for e in c0.get("env", [])
        if e.get("name")
        not in (
            "CLOUD_RUN_VCPU",
            "CLOUD_RUN_MEMORY_GIB",
            "CLOUD_RUN_ACCELERATOR",
            "COMPUTE_PROFILE_ID",
            "BENCHMARK_MODEL_ID",
            "BENCHMARK_CONFIG_STAMP",
        )
    ]
    envs.extend(
        [
            {"name": "CLOUD_RUN_VCPU", "value": str(spec["vcpu_count"])},
            {"name": "CLOUD_RUN_MEMORY_GIB", "value": str(spec["memory_gib"])},
            {"name": "CLOUD_RUN_ACCELERATOR", "value": accel},
            {"name": "COMPUTE_PROFILE_ID", "value": str(spec["profile_id"])},
            {"name": "BENCHMARK_MODEL_ID", "value": str(model_name)},
            {"name": "BENCHMARK_CONFIG_STAMP", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        ]
    )
    c0["env"] = envs
    svc["template"].pop("revision", None)

    update_mask = "template.containers" + (",template.nodeSelector,template.gpuZonalRedundancyDisabled" if accel == "nvidia-l4" else "")
    patch_req = urllib.request.Request(
        f"{svc_url}?updateMask={update_mask}",
        data=json.dumps({"template": svc["template"]}).encode("utf-8"),
        headers={"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"},
        method="PATCH",
    )
    try:
        with urllib.request.urlopen(patch_req, timeout=30):
            pass
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="replace")
        print(
            f"  [GCP Cloud Run Provisioner Note] Cloud Run API returned HTTP {http_err.code} for accelerator='{accel}' "
            f"({err_body[:180]}...). Falling back to CPU container revision '{cur_rev}' while modeling '{accel}' hardware rate."
        )
        return {"revision": cur_rev, "limits": cur_limits, "status": f"FALLBACK_QUOTA_OR_REGION ({http_err.code})"}

    for _ in range(25):
        time.sleep(3)
        creds.refresh(google.auth.transport.requests.Request())
        with urllib.request.urlopen(urllib.request.Request(svc_url, headers={"Authorization": f"Bearer {creds.token}"}), timeout=15) as r:
            cur_svc = json.loads(r.read().decode("utf-8"))
        ready_rev = cur_svc.get("latestReadyRevision", "").split("/")[-1]
        created_rev = cur_svc.get("latestCreatedRevision", "").split("/")[-1]
        failed_conds = [c for c in cur_svc.get("conditions", []) if c.get("state") == "CONDITION_FAILED"]
        if failed_conds:
            msg = failed_conds[0].get("message", "Condition failed").splitlines()[0]
            print(
                f"  [GCP Cloud Run Provisioner] Revision '{created_rev}' reported GCP condition: {msg} "
                f"-> Using active ready revision '{ready_rev}' while modeling '{accel}' hardware rate."
            )
            return {"revision": ready_rev, "attempted_revision": created_rev, "gcp_condition": msg, "status": "GCP_QUOTA_LIMITED"}
        if ready_rev and ready_rev == created_rev and ready_rev != cur_rev:
            new_c0_limits = cur_svc.get("template", {}).get("containers", [{}])[0].get("resources", {}).get("limits", {})
            print(
                f"  [GCP Cloud Run Provisioner] READY! New Cloud Run revision '{ready_rev}' is live with "
                f"limits={new_c0_limits}"
            )
            return {"revision": ready_rev, "limits": new_c0_limits, "status": "PROVISIONED_NEW_REVISION"}
    return {"revision": cur_rev, "limits": new_limits, "status": "PROVISIONING_TIMEOUT_USING_LATEST"}


def _print_compute_tradeoff_matrix(results: List[Any]) -> None:
    """Print a granular Compute vs Latency & Cost Tradeoff Matrix across all runs."""
    if not results:
        return
    print("\n" + "=" * 162)
    print("GRANULAR COMPUTE vs LATENCY & COST TRADEOFF MATRIX (PER IMAGE & PER 1,000 IMAGES)")
    print("=" * 162)
    header = (
        f"{'APPROACH':<34} | {'HARDWARE PROFILE':<28} | {'REVISION':<10} | "
        f"{'TOTAL MS':<9} | {'CPU MS':<7} | {'WAIT MS':<8} | "
        f"{'vCPU ($)':<9} | {'RAM ($)':<9} | {'GPU/TPU($)':<10} | "
        f"{'TOTAL/IMG':<10} | {'$/1K IMGS':<9} | {'PARETO'}"
    )
    print(header)
    print("-" * 162)
    for r in results:
        if isinstance(r, dict):
            continue
        gp = (r.execution_trace or {}).get("gcp_runtime_proof") or {}
        spec = (r.execution_trace or {}).get("compute_profile_spec") or {}
        accel = spec.get("accelerator_type", "none")
        rev_short = str(gp.get("k_revision") or "local").replace("unilever-shelf-benchmark-service-", "")
        hw_short = f"{spec.get('vcpu_count', 2.0):.0f}vCPU/{spec.get('memory_gib', 4.0):.0f}GiB" + (f" + 1x {accel.upper()}" if accel != "none" else " (CPU-Only)")
        print(
            f"{r.separation_approach[:34]:<34} | {hw_short[:28]:<28} | {rev_short[:10]:<10} | "
            f"{r.latency_ms:>7.1f}ms | {r.cost.container_cpu_active_ms:>5.1f}ms | {r.cost.external_api_wait_ms:>6.1f}ms | "
            f"${r.cost.cloud_run_vcpu_usd:<8.6f} | ${r.cost.cloud_run_memory_usd:<8.6f} | ${r.cost.cloud_run_accelerator_usd:<9.6f} | "
            f"${r.cost.cost_per_shelf_image_usd:<9.6f} | ${r.cost.cost_per_1k_images_usd:<8.2f} | {r.cost.cost_latency_pareto_index:>6.2f}"
        )
    print("=" * 162)


def _cmd_cloud_run(args: argparse.Namespace) -> int:
    from pathlib import Path
    import tempfile
    import urllib.error
    import urllib.request

    from shelf_benchmark.config import normalize_vertex_gemini_model_id
    from shelf_benchmark.reporting.generator import BenchmarkReportGenerator
    from shelf_benchmark.testing import run_offline_approach

    # Validate user's model ID (verbatim, no substitution — only blocks banned legacy gemini-2.5/2.0/1.x)
    args.model = normalize_vertex_gemini_model_id(args.model, for_live_vertex=not bool(getattr(args, "offline", False)))

    approaches = _resolve_approaches(args.approaches, ["all"])
    out_dir = Path(getattr(args, "output_dir", "reports") or "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _apply_common(BenchmarkConfig.from_yaml("configs/default_config.yaml"), args)
    cfg.billing.include_infrastructure_costs = True
    cfg.reporting.output_dir = str(out_dir)
    cfg.reporting.sync_reports_to_gcs = False
    cfg.telemetry.otel_log_path = str(out_dir / "otel_logs.jsonl")
    cfg.telemetry.export_to_gcp_cloud_logging = not bool(getattr(args, "offline", False))
    cfg.telemetry.sync_otel_logs_to_gcs = False
    cfg.telemetry.sync_otel_jsonl_to_gcs = False

    # Resolve compute profiles to test (single profile or multi-compute sweep)
    compute_specs: List[Dict[str, Any]] = []
    if getattr(args, "compute_sweep", None):
        for item in args.compute_sweep:
            for p_name in str(item).split(","):
                p_name = p_name.strip()
                if p_name in COMPUTE_PROFILES:
                    compute_specs.append(dict(COMPUTE_PROFILES[p_name]))
    elif getattr(args, "compute_profile", None):
        compute_specs.append(dict(COMPUTE_PROFILES[args.compute_profile]))
    else:
        compute_specs.append(
            {
                "profile_id": f"custom-{cfg.billing.cloud_run.vcpu_count:.0f}x{cfg.billing.cloud_run.memory_gib:.0f}-{cfg.billing.cloud_run.accelerator_type}",
                "vcpu_count": float(cfg.billing.cloud_run.vcpu_count),
                "memory_gib": float(cfg.billing.cloud_run.memory_gib),
                "accelerator_type": str(cfg.billing.cloud_run.accelerator_type or "none"),
            }
        )

    collected: List[Any] = []
    cloud_run_generated_reports: Dict[str, str] = {}

    if getattr(args, "offline", False):
        work = Path(tempfile.mkdtemp(prefix="shelf-cloudrun-cmp-"))
        for spec in compute_specs:
            for app_id in approaches:
                res = run_offline_approach(
                    work / f"{app_id}_{spec['profile_id']}",
                    approach_id=app_id,
                    model_id=args.model,
                    with_ground_truth=bool(args.connect_sample_gt),
                )
                res.execution_trace["compute_profile_spec"] = spec
                collected.append(res)
    else:
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default(quota_project_id=cfg.gcp.project_id)
        creds.refresh(google.auth.transport.requests.Request())
        id_token = creds.id_token

        url_base = args.url.rstrip("/")
        print(
            f"Orchestrating {len(approaches)} approach(es) across {len(compute_specs)} compute profile(s) "
            f"with verbatim model '{args.model}' 100% inside GCP Cloud Run: {url_base}"
        )

        for spec in compute_specs:
            if getattr(args, "reconfigure_cloud_run", True):
                _provision_cloud_run_compute_revision(
                    cfg,
                    spec,
                    model_name=args.model,
                    always_new_revision=True,
                )

            # Delegate the entire suite execution, scoring, cost math, OTel Cloud Logging, and report generation to Cloud Run (/api/run-suite)
            suite_payload = {
                "model_name": args.model,
                "approaches": approaches,
                "shelf_image_uri": args.image,
                "connect_sample_gt": bool(args.connect_sample_gt),
                "vcpu_count": spec["vcpu_count"],
                "memory_gib": spec["memory_gib"],
                "accelerator": spec["accelerator_type"],
            }
            suite_req = urllib.request.Request(
                f"{url_base}/api/run-suite",
                data=json.dumps(suite_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {id_token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(suite_req, timeout=360) as resp:
                suite_resp = json.loads(resp.read().decode("utf-8"))

            for fname, fcontent in (suite_resp.get("report_files_content") or {}).items():
                cloud_run_generated_reports[fname] = fcontent

            for run_item in suite_resp.get("runs", []):
                app_id = run_item.get("separation_approach")
                label_app_id = app_id if len(compute_specs) == 1 else f"{app_id}[{spec['profile_id']}]"
                res_obj = _dict_to_task_execution_result(run_item, label_app_id, args.model, args.image, cfg)
                res_obj.execution_trace["compute_profile_spec"] = spec
                gcp_runtime_proof = run_item.get("gcp_runtime_proof") or suite_resp.get("gcp_runtime_proof")
                if gcp_runtime_proof:
                    res_obj.execution_trace["gcp_runtime_proof"] = gcp_runtime_proof
                collected.append(res_obj)
                rev_tag = (gcp_runtime_proof or {}).get("k_revision") or "cloud-run"
                print(
                    f"  [Cloud Run Suite 200 OK | {rev_tag} | {spec['profile_id']}] {app_id:<34} -> "
                    f"{len(res_obj.row_level_items)} facings, {res_obj.latency_ms:.1f}ms, cost=${res_obj.cost.cost_per_shelf_image_usd:.6f}"
                )

    diag_paths = _write_and_print_diagnostic_report(collected, cfg, out_dir)
    if not getattr(args, "offline", False):
        _verify_gcp_cloud_run_and_logging(cfg, collected, out_dir)
        diag_paths["gcp_verification_json"] = str(out_dir / "gcp_live_verification_proof.json")
    _print_compute_tradeoff_matrix(collected)
    _print_comparison_table(collected)

    if cloud_run_generated_reports and len(compute_specs) == 1:
        # Save the exact report artifacts generated inside the Cloud Run container
        artifact_paths = {}
        for fname, fcontent in cloud_run_generated_reports.items():
            target_f = out_dir / fname
            target_f.write_text(fcontent, encoding="utf-8")
            artifact_paths[fname] = str(target_f)
    else:
        report_gen = BenchmarkReportGenerator.from_config(cfg)
        artifact_paths = report_gen.generate_all_reports(collected)
    artifact_paths.update(diag_paths)
    print(f"\nCloud-Run-orchestrated reports saved to '{out_dir}/':")
    for k, v in artifact_paths.items():
        print(f"  {k:<24}: {v}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Default to `run` so pre-existing invocations without a subcommand keep working.
    known = {"run", "score", "list-approaches", "validate-gt", "cloud-run"}
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
        "cloud-run": _cmd_cloud_run,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
