"""Command-line interface for ``shelf-bench``.

Examples:
    shelf-bench download                         # fetch and extract SKU-110K (~12 GB)
    shelf-bench list                             # list approaches, stages, and models
    shelf-bench run -a detect_classify -m gemini-3.8-flash
    shelf-bench run -a single_pass detect_classify -m gemini-3.8-flash gemini-3.5-flash-lite
    shelf-bench cloud-run -a single_pass -m gemini-3.8-flash -t standard priority
    shelf-bench leaderboard
    shelf-bench serve                            # serve leaderboard UI on http://localhost:8080
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import approaches  # noqa: E402
import runner  # noqa: E402
from utils import dataset  # noqa: E402
from utils.llm import TIERS, load_config  # noqa: E402


def _add_pipeline_override_flags(subparser: argparse.ArgumentParser) -> None:
    """Attach task and stage override arguments for ``modular_e2e_pipeline``."""
    subparser.add_argument(
        "--with-detector",
        default=None,
        help="override detector approach in modular_e2e_pipeline",
    )
    subparser.add_argument(
        "--with-attr-classifier",
        default=None,
        help="override category/brand/packaging classifier in modular_e2e_pipeline",
    )
    subparser.add_argument(
        "--with-variant-classifier",
        default=None,
        help="override variant classifier in modular_e2e_pipeline",
    )
    subparser.add_argument(
        "--with-rectifier",
        default=None,
        help="override Stage 1 rectifier (hough_rail_homography, depth_anything_v2, none)",
    )
    subparser.add_argument(
        "--with-post-detector",
        default=None,
        help="override Stage 3 post-detector (shelf_rail_soft_nms, oriented_ladi_slicer, standard_nms)",
    )
    subparser.add_argument(
        "--with-clusterer",
        default=None,
        help="override Stage 3.5 crop clusterer (connected_component_adjacency, maxvit_agglomerative, none)",
    )
    subparser.add_argument(
        "--with-retriever",
        default=None,
        help="override Stage 4 catalog retriever (ijepa_scann_entropy_prefilter, siglip_multiprototype, pure_scann_cosine)",
    )
    subparser.add_argument(
        "--with-tiebreaker",
        default=None,
        help="override Stage 5 shade tiebreaker (cielab_delta_e_and_systemone, cielab_delta_e_only, direct_vlm_only)",
    )
    subparser.add_argument(
        "--with-shelf-metrics",
        "--with-gondola-kpi",
        dest="with_shelf_metrics",
        default=None,
        help="override Stage 6 shelf metrics evaluator (dual_mt_marketshare_and_merchandising, panorama_6img_stitch_sos, single_img_planogram_oos)",
    )


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch the requested ``shelf-bench`` subcommand."""
    config = load_config()
    defaults = config.get("defaults", {})
    parser = argparse.ArgumentParser(
        prog="shelf-bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project",
        default=None,
        help="override GCP project ID (sets SHELF_BENCH_PROJECT)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="override GCP region (sets SHELF_BENCH_REGION)",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="provision GCP project APIs, GCS buckets, Artifact Registry, and datasets",
    )
    bootstrap_parser.add_argument("--project", dest="boot_project", default=None, help="target GCP project ID")
    bootstrap_parser.add_argument("--region", dest="boot_region", default=None, help="target GCP region (default: us-central1)")
    bootstrap_parser.add_argument("--skip-upload", action="store_true", help="provision GCP resources without uploading dataset files")

    download_parser = subparsers.add_parser("download", help="download and extract SKU-110K")
    download_parser.add_argument("--root", default=dataset.LOCAL_ROOT)

    upload_parser = subparsers.add_parser("upload", help="copy datasets (SKU-110K, HUL labeled, catalog) to GCS")
    upload_parser.add_argument("--root", default=dataset.LOCAL_ROOT)
    upload_parser.add_argument("--to", default=None)
    upload_parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "sku110k", "hul_labeled", "catalog"],
        help="dataset bundle to upload to GCS",
    )

    subparsers.add_parser("splits", help="build and verify the SHA-256 train/val/test split manifest")

    list_parser = subparsers.add_parser("list", help="list registered approaches, stages, and models")
    list_parser.add_argument("--task", default=None, help="filter approaches by task (detection, classification, combined)")
    list_parser.add_argument("--epic", default=None, help="filter approaches by epic")

    run_parser = subparsers.add_parser("run", help="evaluate approach and model combinations locally")
    run_parser.add_argument("-a", "--approach", nargs="+", required=True)
    run_parser.add_argument("-m", "--model", nargs="+", required=True)
    run_parser.add_argument(
        "-t",
        "--tier",
        nargs="+",
        choices=TIERS,
        default=defaults.get("tier", ["standard"]),
        help="Gemini PayGo tier(s): standard and/or priority",
    )
    run_parser.add_argument("--split", default=defaults.get("split", "test"), choices=dataset.SPLITS)
    run_parser.add_argument("--limit", type=int, default=defaults.get("limit", 50), help="0 = full split")
    run_parser.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    run_parser.add_argument("--workers", type=int, default=defaults.get("workers", 4))
    run_parser.add_argument("--owner", default=None, help="defaults to $USER")
    run_parser.add_argument("--root", default=dataset.DEFAULT_ROOT, help="local directory or gs:// URI")
    run_parser.add_argument("--results", default=str(runner.RESULTS_DIR), help="local directory or gs:// URI")
    _add_pipeline_override_flags(run_parser)

    for cloud_cmd in ("cloud-run", "cloud"):
        cloud_run_parser = subparsers.add_parser(cloud_cmd, help="run approach and model combinations on Cloud Run")
        cloud_run_parser.add_argument("-a", "--approach", nargs="+", required=True)
        cloud_run_parser.add_argument("-m", "--model", nargs="+", required=True)
        cloud_run_parser.add_argument(
            "-t",
            "--tier",
            nargs="+",
            choices=TIERS,
            default=defaults.get("tier", ["standard"]),
            help="Gemini PayGo tier(s): standard and/or priority",
        )
        cloud_run_parser.add_argument("--split", default=defaults.get("split", "test"), choices=dataset.SPLITS)
        cloud_run_parser.add_argument("--limit", type=int, default=defaults.get("limit", 50), help="0 = full split")
        cloud_run_parser.add_argument("--seed", type=int, default=defaults.get("seed", 0))
        cloud_run_parser.add_argument("--workers", type=int, default=defaults.get("workers", 4))
        cloud_run_parser.add_argument("--owner", default=None, help="defaults to $USER")
        _add_pipeline_override_flags(cloud_run_parser)

    vertex_job_parser = subparsers.add_parser(
        "vertex-job",
        help="run approach and model combinations on Vertex AI Custom Jobs",
    )
    vertex_job_parser.add_argument("-a", "--approach", nargs="+", required=True)
    vertex_job_parser.add_argument("-m", "--model", nargs="+", required=True)
    vertex_job_parser.add_argument(
        "-t",
        "--tier",
        nargs="+",
        choices=TIERS,
        default=defaults.get("tier", ["standard"]),
        help="Gemini PayGo tier(s): standard and/or priority",
    )
    vertex_job_parser.add_argument("--split", default=defaults.get("split", "test"), choices=dataset.SPLITS)
    vertex_job_parser.add_argument("--limit", type=int, default=defaults.get("limit", 50), help="0 = full split")
    vertex_job_parser.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    vertex_job_parser.add_argument("--workers", type=int, default=defaults.get("workers", 4))
    vertex_job_parser.add_argument("--owner", default=None, help="defaults to $USER")
    vertex_job_parser.add_argument("--machine-type", default=None, help="Vertex AI machine type (default: n1-standard-4)")
    vertex_job_parser.add_argument("--accelerator-type", default=None, help="optional GPU type (e.g. NVIDIA_L4)")
    vertex_job_parser.add_argument("--accelerator-count", type=int, default=None, help="optional GPU count")
    _add_pipeline_override_flags(vertex_job_parser)

    vertex_train_parser = subparsers.add_parser(
        "vertex-train",
        help="submit a Vertex AI Supervised Fine-Tuning job on the train split",
    )
    vertex_train_parser.add_argument("-m", "--model", default="gemini-2.5-flash", help="base Gemini model to fine-tune")
    vertex_train_parser.add_argument("--train-uri", default=None, help="GCS URI of training JSONL dataset")
    vertex_train_parser.add_argument("--val-uri", default=None, help="GCS URI of validation JSONL dataset")
    vertex_train_parser.add_argument("--display-name", default=None, help="tuned model display name")
    vertex_train_parser.add_argument("--epochs", type=int, default=4, help="number of training epochs")
    vertex_train_parser.add_argument("--dry-run", action="store_true", help="print payload without calling Vertex AI")

    vertex_deploy_parser = subparsers.add_parser(
        "vertex-deploy",
        help="deploy the Shelf Intelligence Agent on Vertex AI Agent Engine (ReasoningEngine)",
    )
    vertex_deploy_parser.add_argument("--display-name", default=None, help="Agent Engine display name")
    vertex_deploy_parser.add_argument("--dry-run", action="store_true", help="print payload without calling Vertex AI")

    cloud_service_parser = subparsers.add_parser("cloud-service", help="deploy the Cloud Run web service")
    cloud_service_parser.add_argument("--port", type=int, default=8080)

    cloud_proxy_parser = subparsers.add_parser(
        "cloud-proxy",
        help="start an authenticated local proxy to the live Cloud Run web service",
    )
    cloud_proxy_parser.add_argument("--port", type=int, default=8080)
    cloud_proxy_parser.add_argument("--host", default="127.0.0.1")
    cloud_proxy_parser.add_argument("--url", default=None, help="override Cloud Run service URL")

    subparsers.add_parser("pull", help="copy finished Vertex AI / Cloud Run results from GCS into results/")

    leaderboard_parser = subparsers.add_parser("leaderboard", help="print the benchmark leaderboard")
    leaderboard_parser.add_argument("--results", type=Path, default=runner.RESULTS_DIR)
    leaderboard_parser.add_argument("--task", default=None, help="filter by task (detection, classification, combined)")
    leaderboard_parser.add_argument("--epic", default=None, help="filter by epic")
    leaderboard_parser.add_argument(
        "--attribute",
        default=None,
        help="rank by attribute (category, brand, packaging_type, variant, compound)",
    )

    serve_parser = subparsers.add_parser("serve", help="serve the leaderboard web UI")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--results", type=Path, default=runner.RESULTS_DIR)
    serve_parser.add_argument("--root", default=dataset.DEFAULT_ROOT, help="local directory or gs:// URI")

    args = parser.parse_args(argv)
    if getattr(args, "project", None):
        os.environ["SHELF_BENCH_PROJECT"] = args.project
    if getattr(args, "region", None):
        os.environ["SHELF_BENCH_REGION"] = args.region
    config = load_config()

    if args.cmd == "bootstrap":
        from utils import cloud

        target_project = args.boot_project or args.project or config["gcp"]["project"]
        target_region = args.boot_region or args.region or config["gcp"]["region"]
        cloud.bootstrap_argolis_project(
            project=target_project,
            region=target_region,
            upload_datasets=not args.skip_upload,
        )
    elif args.cmd == "download":
        dataset.download(args.root)
    elif args.cmd == "upload":
        dataset.upload(args.root, args.to or config["gcp"]["data"], dataset_target=args.dataset)
    elif args.cmd == "splits":
        manifest = dataset.build_and_verify_splits_manifest()
        print(
            f"Verified SHA-256 Split Manifest ({manifest['split_sha256'][:16]}): "
            f"train={manifest['splits']['train']['image_count']} imgs, "
            f"val={manifest['splits']['val']['image_count']} imgs, "
            f"test={manifest['splits']['test']['image_count']} imgs "
            f"(zero_leakage={manifest['zero_leakage_verified']})"
        )
    elif args.cmd == "list":
        import stages

        print("Approaches (by Epic and Task):")
        for name, approach_obj in approaches.all_approaches().items():
            if getattr(args, "task", None) and approach_obj.task != args.task:
                continue
            if getattr(args, "epic", None) and approach_obj.epic != args.epic:
                continue
            print(f"  {name:<30} [{approach_obj.task:<14} | {approach_obj.epic}]  {approach_obj.architecture}")
        print("\nPipeline Stages (for modular_e2e_pipeline):")
        for group_name, specs in stages.all_stages().items():
            stage_names = ", ".join(spec["name"] + ("*" if spec["default"] else "") for spec in specs)
            print(f"  --with-{group_name.replace('_', '-'):<16} {stage_names}")
        print("\nModels (any Vertex Gemini ID is supported):")
        for model_name in config.get("models", []):
            print(f"  {model_name}")
    elif args.cmd == "run":
        for name in args.approach:
            approaches.get(name)  # Validate approach names before starting runs.
        modular_overrides = {
            key: val
            for key, val in (
                ("detector", getattr(args, "with_detector", None)),
                ("attr_classifier", getattr(args, "with_attr_classifier", None)),
                ("variant_classifier", getattr(args, "with_variant_classifier", None)),
            )
            if val
        }
        stage_overrides = {
            key: val
            for key, val in (
                ("rectifier", getattr(args, "with_rectifier", None)),
                ("post_detector", getattr(args, "with_post_detector", None)),
                ("clusterer", getattr(args, "with_clusterer", None)),
                ("retriever", getattr(args, "with_retriever", None)),
                ("tiebreaker", getattr(args, "with_tiebreaker", None)),
                ("shelf_metrics", getattr(args, "with_shelf_metrics", None)),
            )
            if val
        }
        combos = [(name, tier, model) for name in args.approach for tier in args.tier for model in args.model]
        if int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1")) > 1:
            combos = [combos[int(os.environ["CLOUD_RUN_TASK_INDEX"])]]
        failed = 0
        for name, tier, model in combos:
            try:
                runner.run(
                    name,
                    model,
                    args.split,
                    args.limit,
                    args.seed,
                    args.workers,
                    args.owner,
                    results_dir=args.results,
                    data_root=args.root,
                    tier=tier,
                    modular_overrides=modular_overrides or None,
                    stage_overrides=stage_overrides or None,
                )
            except Exception as err:
                failed += 1
                print(f"[{name} x {model} x {tier}] FAILED: {err}", file=sys.stderr)
        return 1 if failed else 0
    elif args.cmd in ("cloud-run", "cloud"):
        from utils import cloud

        for name in args.approach:
            approaches.get(name)
        return cloud.run_on_cloud(
            args.approach,
            args.model,
            args.tier,
            args.split,
            args.limit,
            args.seed,
            args.workers,
            args.owner or getpass.getuser(),
        )
    elif args.cmd == "vertex-job":
        from utils import vertex_platform

        for name in args.approach:
            approaches.get(name)
        return vertex_platform.run_on_vertex(
            args.approach,
            args.model,
            args.tier,
            args.split,
            args.limit,
            args.seed,
            args.workers,
            args.owner or getpass.getuser(),
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
        )
    elif args.cmd == "vertex-train":
        import json

        from utils import vertex_platform

        res = vertex_platform.submit_vertex_tuning_job(
            base_model=args.model,
            train_dataset_uri=args.train_uri,
            validation_dataset_uri=args.val_uri,
            tuned_model_display_name=args.display_name,
            epoch_count=args.epochs,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(json.dumps(res, indent=2))
    elif args.cmd == "vertex-deploy":
        import json

        from utils import vertex_platform

        res = vertex_platform.deploy_vertex_agent(
            display_name=args.display_name,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(json.dumps(res, indent=2))
    elif args.cmd == "cloud-service":
        from utils import cloud

        cloud.deploy_cloud_service(port=args.port)
    elif args.cmd == "cloud-proxy":
        from utils import cloud

        cloud.proxy_cloud_service(port=args.port, host=args.host, target_url=args.url)
    elif args.cmd == "pull":
        print(f"Pulled {runner.pull(config['gcp']['results'])} new run(s) into results/")
    elif args.cmd == "leaderboard":
        print_leaderboard(
            runner.leaderboard(
                args.results,
                task=getattr(args, "task", None),
                epic=getattr(args, "epic", None),
                attribute=getattr(args, "attribute", None),
            )
        )
    elif args.cmd == "serve":
        from utils.server import serve

        serve(args.host, args.port, args.results, args.root)
    return 0


def print_leaderboard(rows: list[dict]) -> None:
    """Format and print leaderboard rows to stdout."""
    if not rows:
        print("No runs yet. Try: shelf-bench run -a single_pass -m gemini-3.8-flash")
        return
    header = (
        f"{'#':>3}  {'task':<14} {'epic':<44} {'run id':<46} "
        f"{'owner':<9} {'acc':>5} {'rec':>5} {'F2':>5} {'p95':>6} {'INR/img':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        task_name = str(row.get("task", "detection"))[:14]
        epic_name = str(row.get("epic", "MT Market Share - SKU Detection"))[:44]
        print(
            f"{row['rank']:>3}  {task_name:<14} {epic_name:<44} {row['run_id'][:46]:<46} "
            f"{row['owner'][:9]:<9} {row['accuracy']:>5.3f} "
            f"{row['recall']:>5.3f} {row['f2']:>5.3f} {row['p95_latency_s']:>5.1f}s "
            f"{row['cost_per_image_inr']:>8.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
