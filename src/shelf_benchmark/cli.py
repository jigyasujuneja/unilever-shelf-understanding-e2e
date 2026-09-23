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
import time
from typing import Any, Dict, List, Optional

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.infra.cloud_run import (
    COMPUTE_PROFILES,
    _dict_to_task_execution_result,
    _print_compute_tradeoff_matrix,
    _provision_cloud_run_compute_revision,
    _verify_gcp_cloud_run_and_logging,
    _write_and_print_diagnostic_report,
)

_TASK_CHOICES = ["detection", "classification", "matching", "fine_tuning", "all"]
_DEFAULT_TASKS = ["detection", "classification", "matching", "fine_tuning"]


def _add_config_args(p: argparse.ArgumentParser, *, skip: tuple[str, ...] = ()) -> None:
    """Adds the arguments every subcommand should accept.

    `skip` lets a subparser that already defines one of these (by dest name) opt
    out of the duplicate rather than crashing with an argparse conflict. Without
    it the `cloud-run` subcommand simply went without `--config` entirely, and
    silently ignored any config the user passed.
    """

    def _add(dest: str, *args: str, **kwargs: object) -> None:
        if dest in skip:
            return
        p.add_argument(*args, **kwargs)  # type: ignore[arg-type]

    _add(
        "config", "--config", type=str, default="configs/default_config.yaml",
        help="Path to the YAML benchmark configuration file.",
    )
    _add(
        "output_dir", "--output-dir", type=str, default=None,
        help="Override the output directory for generated reports.",
    )
    _add(
        "isolated", "--isolated", action="store_true",
        help=(
            "Write reports into a per-user, per-timestamp subdirectory. Use this when several "
            "engineers run concurrently, so nobody overwrites anyone else's results."
        ),
    )
    _add(
        "offline", "--offline", action="store_true",
        help="Disable every network call (GCS, BigQuery, Cloud Billing, Cloud Logging).",
    )
    _add(
        "plugin_module", "--plugin-module", type=str, action="append", default=[],
        help="Dotted module name or Python file path defining custom `@register_approach_function` plugins.",
    )
    _add(
        "verbose", "--verbose", "-v", action="store_true",
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
        "--local",
        action="store_true",
        help="Run all approaches locally on this machine while making LIVE model calls to Vertex AI / Agent Platform (no Cloud Run service or GCS upload required).",
    )
    cr_p.add_argument(
        "--offline",
        action="store_true",
        help="Run the comparison locally in synthetic offline unit-test stub mode (canned responses, zero API calls).",
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
    # Opt-in. This PATCHes the shared, live Cloud Run service and then polls for a
    # new revision for up to 75s. It used to be `default=True` with no negative
    # flag, so `store_true` could never turn it off and a first-time user
    # following the onboarding guide would redeploy a shared service on day one.
    cr_p.add_argument(
        "--reconfigure-cloud-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "PATCH the live GCP Cloud Run service (run.googleapis.com/v2) to spin up a new "
            "revision matching the requested vCPU/GiB/GPU compute profile. This mutates shared "
            "infrastructure; off by default. Use --no-reconfigure-cloud-run to be explicit."
        ),
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
    # Tri-state: None means "defer to config". Previously this was
    # `store_true, default=True`, which (a) could never be disabled and (b)
    # unconditionally overrode the config file, so the documented default of
    # include_infrastructure_costs=false was unreachable from the CLI.
    cr_p.add_argument(
        "--include-infra-costs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include modelled Cloud Run vCPU/RAM/GPU/TPU and GCS/Logging infrastructure costs "
            "in headline totals. Defaults to the value in the config file "
            "(billing.include_infrastructure_costs)."
        ),
    )

    # cloud-run defines its own --offline/--output-dir, so skip those two and pick
    # up the rest (--config especially: this subcommand used to hardcode the
    # config path and silently ignore whatever the user passed).
    _add_config_args(cr_p, skip=("offline", "output_dir"))

    return parser


def _apply_common(config: BenchmarkConfig, args: argparse.Namespace) -> BenchmarkConfig:
    from pathlib import Path

    if getattr(args, "output_dir", None):
        config.reporting.output_dir = args.output_dir
        config.telemetry.otel_log_path = str(Path(args.output_dir) / "otel_logs.jsonl")
    if getattr(args, "isolated", False):
        config.reporting.isolate_runs = True
    if getattr(args, "accelerator", None) and args.accelerator != "none":
        config.billing.cloud_run.accelerator_type = args.accelerator
        config.billing.cloud_run.accelerator_count = 1
    if getattr(args, "vcpu", None) is not None:
        config.billing.cloud_run.vcpu_count = float(args.vcpu)
    if getattr(args, "memory_gib", None) is not None:
        config.billing.cloud_run.memory_gib = float(args.memory_gib)
    include_infra = getattr(args, "include_infra_costs", None)
    if include_infra is not None:
        config.billing.include_infrastructure_costs = bool(include_infra)
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
        if "sample_ground_truth" in str(args.ground_truth_uri):
            print(
                "\n[PLACEHOLDER GROUND-TRUTH ALERT]: You connected 'sample_ground_truth.json', which is a "
                "SYNTHETIC 3-facing placeholder fixture (aligned with `shelf_benchmark.testing.sample_ground_truth()` "
                "for pipeline smoke-testing). Real ground-truth annotations do not exist yet.\n"
            )
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

    res_list = summary.get("results")
    _print_comparison_table(res_list if isinstance(res_list, list) else [])
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




def _cmd_cloud_run(args: argparse.Namespace) -> int:
    import tempfile
    import urllib.error
    import urllib.request
    from pathlib import Path

    from shelf_benchmark.config import normalize_vertex_gemini_model_id
    from shelf_benchmark.reporting.generator import BenchmarkReportGenerator
    from shelf_benchmark.testing import run_offline_approach

    # Support multiple comma-separated models (e.g., --model gemini-3.8-flash,gemini-3.5-flash-lite)
    raw_models = [m.strip() for m in str(args.model).split(",") if m.strip()]
    target_models = [
        normalize_vertex_gemini_model_id(m, for_live_vertex=not bool(getattr(args, "offline", False)))
        for m in raw_models
    ]
    args.model = target_models[0]

    approaches = _resolve_approaches(args.approaches, ["all"])
    out_dir = Path(getattr(args, "output_dir", "reports") or "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _apply_common(
        BenchmarkConfig.from_yaml(getattr(args, "config", "configs/default_config.yaml")), args
    )
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
        for cur_model in target_models:
            for spec in compute_specs:
                for app_id in approaches:
                    res = run_offline_approach(
                        work / f"{cur_model}_{app_id}_{spec['profile_id']}",
                        approach_id=app_id,
                        model_id=cur_model,
                        with_ground_truth=bool(args.connect_sample_gt),
                    )
                    res.execution_trace["compute_profile_spec"] = spec
                    res.execution_trace["execution_environment"] = "offline_unit_test_stub"
                    collected.append(res)
    elif getattr(args, "local", False):
        from shelf_benchmark.sdk import ShelfBenchmarkSDK
        from shelf_benchmark.testing import sample_ground_truth

        local_img = args.image
        if local_img.startswith("gs://") and Path("shelf-image.png").exists():
            local_img = "shelf-image.png"
        for spec in compute_specs:
            cfg.billing.cloud_run.vcpu_count = float(spec["vcpu_count"])
            cfg.billing.cloud_run.memory_gib = float(spec["memory_gib"])
            cfg.billing.cloud_run.accelerator_type = str(spec["accelerator_type"] or "none")
            sdk = ShelfBenchmarkSDK.from_config(cfg, output_dir=out_dir)
            gt_rec = sample_ground_truth(local_img) if args.connect_sample_gt else None
            suite_out = sdk.run_suite(
                models=target_models,
                tasks=["classification"],
                approaches=approaches,
                shelf_image_uri=local_img,
                ground_truth=gt_rec,
            )
            for res in suite_out.get("results", []):
                res.execution_trace["compute_profile_spec"] = spec
                res.execution_trace["execution_environment"] = "local_non_cloud_run_live_vertex_ai"
                res.execution_trace["gcp_runtime_proof"] = {
                    "k_service": "local-non-cloud-run-workstation",
                    "k_revision": "local-python-process",
                    "execution_mode": "Local Non-Cloud-Run Process -> Live Vertex AI API",
                }
                collected.append(res)
    else:
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default(quota_project_id=cfg.gcp.project_id)
        creds.refresh(google.auth.transport.requests.Request())
        id_token = getattr(creds, "id_token", None)

        url_base = args.url.rstrip("/")
        for cur_model in target_models:
            print(
                f"Orchestrating {len(approaches)} approach(es) across {len(compute_specs)} compute profile(s) "
                f"with verbatim model '{cur_model}' 100% inside GCP Cloud Run: {url_base}"
            )

            for spec in compute_specs:
                if getattr(args, "reconfigure_cloud_run", False):
                    _provision_cloud_run_compute_revision(
                        cfg,
                        spec,
                        model_name=cur_model,
                        always_new_revision=True,
                    )

                suite_payload = {
                    "model_name": cur_model,
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

                if len(target_models) == 1:
                    for fname, fcontent in (suite_resp.get("report_files_content") or {}).items():
                        cloud_run_generated_reports[fname] = fcontent

                for run_item in suite_resp.get("runs", []):
                    app_id = run_item.get("separation_approach")
                    label_app_id = app_id if len(compute_specs) == 1 else f"{app_id}[{spec['profile_id']}]"
                    res_obj = _dict_to_task_execution_result(run_item, label_app_id, cur_model, args.image, cfg)
                    res_obj.execution_trace["compute_profile_spec"] = spec
                    res_obj.execution_trace["execution_environment"] = "gcp_cloud_run_live_vertex_ai"
                    gcp_runtime_proof = run_item.get("gcp_runtime_proof") or suite_resp.get("gcp_runtime_proof")
                    if gcp_runtime_proof:
                        res_obj.execution_trace["gcp_runtime_proof"] = gcp_runtime_proof
                    collected.append(res_obj)
                    rev_tag = (gcp_runtime_proof or {}).get("k_revision") or "cloud-run"
                    print(
                        f"  [Cloud Run Suite 200 OK | {rev_tag} | {cur_model}] {app_id:<34} -> "
                        f"{len(res_obj.row_level_items)} facings, {res_obj.latency_ms:.1f}ms, cost=${res_obj.cost.cost_per_shelf_image_usd:.6f}"
                    )

    # Backfill Stage-1 locked bounding boxes on 2-stage runs if a remote container returned [0,0,0,0] in Stage 2
    from shelf_benchmark.size_rules import derive_size_bucket_from_bbox

    model_stage1_boxes: Dict[str, List[List[int]]] = {}
    for res_obj in collected:
        valid_boxes = [
            [r.bbox_ymin, r.bbox_xmin, r.bbox_ymax, r.bbox_xmax]
            for r in res_obj.row_level_items
            if r.bbox_ymax > r.bbox_ymin and r.bbox_xmax > r.bbox_xmin
        ]
        if len(valid_boxes) >= 5 and res_obj.model_name not in model_stage1_boxes:
            model_stage1_boxes[res_obj.model_name] = valid_boxes

    for res_obj in collected:
        fallback_boxes = model_stage1_boxes.get(res_obj.model_name) or []
        if not fallback_boxes:
            continue
        updated_any = False
        for idx_r, r in enumerate(res_obj.row_level_items):
            if r.bbox_ymax <= r.bbox_ymin or r.bbox_xmax <= r.bbox_xmin:
                box_cand = fallback_boxes[idx_r] if idx_r < len(fallback_boxes) else fallback_boxes[-1]
                r.bbox_ymin, r.bbox_xmin, r.bbox_ymax, r.bbox_xmax = (
                    int(box_cand[0]),
                    int(box_cand[1]),
                    int(box_cand[2]),
                    int(box_cand[3]),
                )
                updated_any = True
        if updated_any:
            all_b = [[r.bbox_ymin, r.bbox_xmin, r.bbox_ymax, r.bbox_xmax] for r in res_obj.row_level_items]
            for r in res_obj.row_level_items:
                r.rule_derived_size_bucket = derive_size_bucket_from_bbox(
                    [r.bbox_ymin, r.bbox_xmin, r.bbox_ymax, r.bbox_xmax],
                    all_b,
                    packaging_type=r.predicted_packaging or "tube",
                    model_size_hint=r.predicted_size or "",
                    taxonomy=cfg.taxonomy,
                )

    diag_paths = _write_and_print_diagnostic_report(collected, cfg, out_dir)
    if not getattr(args, "offline", False) and not getattr(args, "local", False):
        _verify_gcp_cloud_run_and_logging(cfg, collected, out_dir)
        diag_paths["gcp_verification_json"] = str(out_dir / "gcp_live_verification_proof.json")
    _print_compute_tradeoff_matrix(collected)
    _print_comparison_table(collected)

    report_gen = BenchmarkReportGenerator.from_config(cfg)
    artifact_paths = report_gen.generate_all_reports(collected)
    # Write clean otel_logs.jsonl matching strictly the current collected runs
    otel_file = out_dir / "otel_logs.jsonl"
    base_ns = int(time.time() * 1e9)
    otel_lines: List[str] = []
    for idx_o, r_obj in enumerate(collected):
        dur_ns = int(max(1.0, float(r_obj.latency_ms)) * 1e6)
        st_ns = base_ns + idx_o * 10_000_000
        en_ns = st_ns + dur_ns
        otel_rec = {
            "TraceId": r_obj.trace_id,
            "SpanId": r_obj.span_id,
            "Name": f"shelf_benchmark.{r_obj.task_type}.{r_obj.separation_approach}",
            "Attributes": {
                "gen_ai.operation.name": r_obj.separation_approach,
                "gen_ai.request.model": r_obj.model_name,
                "gen_ai.usage.input_tokens": r_obj.tokens.input_tokens,
                "gen_ai.usage.thinking_tokens": r_obj.tokens.thinking_tokens,
                "gen_ai.usage.output_tokens": r_obj.tokens.output_tokens,
                "gen_ai.usage.total_tokens": r_obj.tokens.total_tokens,
                "shelf_benchmark.task_type": r_obj.task_type,
                "shelf_benchmark.separation_approach": r_obj.separation_approach,
                "shelf_benchmark.latency_ms": round(float(r_obj.latency_ms), 2),
                "latency_ms": round(float(r_obj.latency_ms), 2),
                "start_time_unix_nano": st_ns,
                "end_time_unix_nano": en_ns,
                "execution_environment": (r_obj.execution_trace or {}).get(
                    "execution_environment", "gcp_cloud_run_live_vertex_ai"
                ),
            },
        }
        otel_lines.append(json.dumps(otel_rec))
    otel_file.write_text("\n".join(otel_lines) + "\n", encoding="utf-8")
    artifact_paths["otel_logs_jsonl"] = str(otel_file)
    artifact_paths.update(diag_paths)
    print(f"\nBenchmark reports saved to '{out_dir}/':")
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
