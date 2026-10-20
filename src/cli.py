"""``shelf-bench`` command line.

    shelf-bench bootstrap                        # new GCP project: APIs, bucket, registry, data
    shelf-bench download                         # fetch + extract SKU-110K (~12 GB)
    shelf-bench list                             # approaches and models
    shelf-bench run -a single_pass_dedup -m gemini-3.5-flash-lite
    shelf-bench run -a single_pass_dedup tiled_dedup -m gemini-3.5-flash-lite gemini-3.8-flash
    shelf-bench run -a hierarchy_classify embedding_retrieval -m gemini-3.5-flash-lite gemini-embedding-2
    shelf-bench cloud-run -a shelf_detect_retrieve -m gemini-3.5-flash-lite -t priority flex
    shelf-bench run -a cv_classify -d hul_golden # legacy .h5 models on Vertex AI Endpoints
    shelf-bench deploy-cv-endpoints              # (re)deploy those endpoints
    shelf-bench leaderboard
    shelf-bench serve                            # leaderboard UI on http://localhost:8080
    shelf-bench serve-cloud                      # the same UI as a private Cloud Run service
    shelf-bench proxy                            # open that private service on http://localhost:8083

Every approach runs only the models it really calls (``list`` shows them); other
approach x model combinations are skipped.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

import approaches
import runner
from utils import dataset
from utils.llm import TIERS, load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    d = cfg.get("defaults", {})
    p = argparse.ArgumentParser(prog="shelf-bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="download and extract SKU-110K")
    s.add_argument("--root", default=dataset.LOCAL_ROOT)

    s = sub.add_parser("upload", help="copy the local dataset to GCS (resumable)")
    s.add_argument("--root", default=dataset.LOCAL_ROOT)
    s.add_argument("--to", default=cfg.get("gcp", {}).get("data"))

    s = sub.add_parser("rpc", help="fetch the RPC checkout subset (~40 MB) and copy it to GCS")
    s.add_argument("--no-upload", action="store_true", help="keep it local only")
    s = sub.add_parser("shelves", help="build the HoloSelecta shelves set from its photos + VOC "
                                       "files and copy it to GCS")
    s.add_argument("raw", help="folder with HoloSelecta's *.jpg and *.xml (its Drive folder)")
    s.add_argument("--no-upload", action="store_true", help="keep it local only")

    s = sub.add_parser("hul", help="build and publish v2.1 hul_golden, gtmerch_golden, hul_promos, hul_brands, and hul_sessions manifests to GCS")
    s.add_argument("--max-train", type=int, default=16000, help="max train samples per Golden manifest (50/50 HUL/Non-HUL)")
    s.add_argument("--max-eval", type=int, default=2000, help="max val/test samples per Golden manifest (50/50 HUL/Non-HUL)")

    sub.add_parser("list", help="list approaches and models")

    s = sub.add_parser("run", help="evaluate approach x model combinations")
    s.add_argument("-a", "--approach", nargs="+", required=True)
    s.add_argument("-m", "--model", nargs="+", default=None, help="required (cv_classify: keras-h5)")
    s.add_argument("-t", "--tier", nargs="+", choices=TIERS, default=d.get("tier", ["priority"]),
                   help="Gemini PayGo tier(s): priority (1.8x, default), standard and/or flex (0.5x)")
    s.add_argument("-d", "--dataset", default=None,
                   choices=("sku110k", "products", *dataset.RPC_FORMAT_SETS),
                   help="override dataset for approaches (e.g. hul_golden, gtmerch_golden, hul_variants, hul_brands, hul_packs, hul_promos)")
    s.add_argument("--fields", default=None,
                   help="score only these attributes: 1 / 2 / 3 / 4 / 5 (HUL presets) or names, e.g. brand,packaging_type")
    s.add_argument("--brand-model", default=None,
                   help="cv_classify: Vertex AI Endpoint or .h5 instead of config cv_models.brand_endpoint")
    s.add_argument("--pack-model", default=None,
                   help="cv_classify: Vertex AI Endpoint or .h5 instead of config cv_models.pack_endpoint")
    for v_role in ("wash_variant", "skin_variant", "p5_variant", "oral_variant", "gen_variant", "nonhul_p6_variant", "nonhul_gen_variant"):
        s.add_argument(f"--{v_role.replace('_', '-')}-model", default=None,
                       help=f"cv_classify: Vertex AI Endpoint or .h5 instead of config cv_models.{v_role}_endpoint")
    s.add_argument("--split", default=d.get("split", "test"), choices=dataset.SPLITS)
    s.add_argument("--limit", type=int, default=d.get("limit", 50), help="0 = whole split")
    s.add_argument("--seed", type=int, default=d.get("seed", 0))
    s.add_argument("--workers", type=int, default=d.get("workers", 4))
    s.add_argument("--owner", default=None, help="defaults to $USER")
    s.add_argument("--root", default=dataset.DEFAULT_ROOT, help="SKU-110K: local dir or gs:// URI")
    s.add_argument("--results", default=str(runner.RESULTS_DIR), help="local dir or gs:// URI")

    s = sub.add_parser("cloud-run", help="run approach x model combinations on Cloud Run")
    s.add_argument("-a", "--approach", nargs="+", required=True)
    s.add_argument("-m", "--model", nargs="+", default=None, help="required (cv_classify: keras-h5)")
    s.add_argument("-t", "--tier", nargs="+", choices=TIERS, default=d.get("tier", ["priority"]),
                   help="Gemini PayGo tier(s): priority (1.8x, default), standard and/or flex (0.5x)")
    s.add_argument("-d", "--dataset", default=None,
                   choices=("sku110k", "products", *dataset.RPC_FORMAT_SETS),
                   help="override dataset for approaches (e.g. hul_golden, gtmerch_golden, hul_variants, hul_brands, hul_packs, hul_promos)")
    s.add_argument("--fields", default=None,
                   help="score only these attributes: 1 / 2 / 3 / 4 / 5 (HUL presets) or names, e.g. brand,packaging_type")
    s.add_argument("--brand-model", default=None,
                   help="cv_classify: Vertex AI Endpoint or .h5 instead of config cv_models.brand_endpoint")
    s.add_argument("--pack-model", default=None,
                   help="cv_classify: Vertex AI Endpoint or .h5 instead of config cv_models.pack_endpoint")
    for v_role in ("wash_variant", "skin_variant", "p5_variant", "oral_variant", "gen_variant", "nonhul_p6_variant", "nonhul_gen_variant"):
        s.add_argument(f"--{v_role.replace('_', '-')}-model", default=None,
                       help=f"cv_classify: Vertex AI Endpoint or .h5 instead of config cv_models.{v_role}_endpoint")
    s.add_argument("--split", default=d.get("split", "test"), choices=dataset.SPLITS)
    s.add_argument("--limit", type=int, default=d.get("limit", 50), help="0 = whole split")
    s.add_argument("--seed", type=int, default=d.get("seed", 0))
    s.add_argument("--workers", type=int, default=d.get("workers", 4))
    s.add_argument("--owner", default=None, help="defaults to $USER")
    s.add_argument("--image", default=None, help="reuse pre-built Artifact Registry image URI instead of rebuilding")
    s.add_argument("--job", default=None, help="override Cloud Run job name (e.g. shelf-bench-5k-exp28)")
    s.add_argument("--pull", action="store_true", help="pull finished runs into local results/ (default: finalize directly on GCS with zero local files)")
    s.add_argument("--no-pull", action="store_true", help="keep results 100%% on GCS without pulling to local results/")

    s = sub.add_parser("deploy-cv-endpoints",
                       help="register the legacy .h5 models in Vertex AI Model Registry and deploy their Vertex AI Endpoints")
    s.add_argument("--brand-model", default=None, help="gs:// .h5 (default: config cv_models.brand_model)")
    s.add_argument("--pack-model", default=None, help="gs:// .h5 (default: config cv_models.pack_model)")
    for v_role in ("wash_variant", "skin_variant", "p5_variant", "oral_variant", "gen_variant", "nonhul_p6_variant", "nonhul_gen_variant"):
        s.add_argument(f"--{v_role.replace('_', '-')}-model", default=None,
                       help=f"gs:// .h5 (default: config cv_models.{v_role}_model)")
    s.add_argument("--roles", nargs="*", default=None,
                   help="subset of roles to deploy (e.g. wash_variant skin_variant p5_variant oral_variant nonhul_p6_variant nonhul_gen_variant)")
    s.add_argument("--image", default=None, help="reuse a pre-built image (must include the cv extra)")

    sub.add_parser("pull", help="copy finished Cloud Run results from GCS into results/")

    s = sub.add_parser("push", help="upload local runs, experiments_registry.json, and qc_audit_report.json to GCS")
    s.add_argument("--results", type=Path, default=runner.RESULTS_DIR)
    s.add_argument("--to", default=cfg.get("gcp", {}).get("results"))
    s.add_argument("--overwrite", action="store_true", help="overwrite existing run summaries in GCS")
    s.add_argument("--prune", action="store_true", help="remove stale remote runs not present in results/")
    s.add_argument("--clean-local", action="store_true", help="remove local results/ dir after verified GCS upload")

    s = sub.add_parser("leaderboard", help="print the leaderboard")
    s.add_argument("--results", type=Path, default=runner.RESULTS_DIR)

    s = sub.add_parser("serve", help="serve the leaderboard UI")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--results", type=Path, default=runner.RESULTS_DIR)
    s.add_argument("--root", default=dataset.DEFAULT_ROOT, help="local dir or gs:// URI")
    s.add_argument("--pull", action="store_true", help="first pull finished runs from GCS")
    s.add_argument("--mode", choices=("engineering", "cio", "exec"), default="engineering",
                   help="default landing portal ('engineering' serves index.html, 'exec'/'cio' serves cio.html)")

    sub.add_parser("serve-cloud", help="deploy the leaderboard UI (with results/) to Cloud Run")
    s = sub.add_parser("proxy", help="open the private Cloud Run UI on localhost (authenticated)")
    s.add_argument("--port", type=int, default=8083)
    s.add_argument("--host", default="127.0.0.1")
    sub.add_parser("bootstrap", help="prepare a new GCP project (APIs, bucket, registry, data)")

    s = sub.add_parser("tune", help="build zero-leakage SFT JSONL data and launch/list Vertex AI LoRA tuning jobs")
    s.add_argument("--dataset", default="shelves",
                   choices=("shelves", "rpc", "pack", "brand", "variant", "hul_golden", "gtmerch_golden", "hul_variants", "hul_brands", "hul_packs", "hul_promos"))
    s.add_argument("--task", default="all3", choices=("all3", "category", "brand", "variant", "clef_contrastive"),
                   help="target task when --dataset=variant or 'clef_contrastive' for contact-sheet SFT")
    s.add_argument("--fields", type=int, default=5, choices=(3, 5),
                   help="HUL Golden SFT attribute count: 5 (product, brand, base_variant, packaging_type, category) or 3 (brand, packaging_type, category)")
    s.add_argument("--base-model", default="gemini-3.5-flash-lite")
    s.add_argument("--display-name", default="shelf-bench-jev-laya-sft")
    s.add_argument("--region", default=None, help="Vertex AI region for SFT job (e.g. us-central1, europe-west4)")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--adapter-size", type=int, default=4, choices=(1, 4, 8, 16))
    s.add_argument("--lr-multiplier", type=float, default=1.0)
    s.add_argument("--out-dir", default="data/tuning")
    s.add_argument("--build-only", action="store_true", help="only build local train/val JSONL files")
    s.add_argument("--upload-only", action="store_true", help="build and upload SFT JSONL files to GCS without launching a tuning job")
    s.add_argument("--launch-only", action="store_true", help="launch Vertex AI SFT job directly from pre-built GCS JSONL URIs")
    s.add_argument("--train-uri", default=None, help="override GCS train.jsonl URI when launching SFT")
    s.add_argument("--val-uri", default=None, help="override GCS val.jsonl URI when launching SFT")
    s.add_argument("--max-train", type=int, default=None, help="create a temporary GCS subset of at most N training examples before launching SFT")
    s.add_argument("--max-val", type=int, default=None, help="create a temporary GCS subset of at most M validation examples before launching SFT")
    s.add_argument("--clean-hul-golden", action="store_true", help="clean and stratify hul_golden train/val/test splits in GCS (zero local files) and optionally launch SFT")
    s.add_argument("--clean-local", action="store_true", help="remove local JSONL output dir after uploading to GCS")
    s.add_argument("--status", action="store_true", help="list Vertex AI tuning jobs and endpoints")

    s = sub.add_parser("sync-brands", help="crawl and update the live HUL vs Non-HUL Brand & SKU registry JSON")
    s.add_argument("--output", default="data/hul_golden/brand_sku_live_registry.json")
    s.add_argument("--online", action="store_true", help="query live Wikipedia/MediaWiki and OpenFoodFacts endpoints")

    s = sub.add_parser("experiments", help="list P0/P1/P2/Moonshot experiment matrix and sync runs/models to GCP Vertex AI")
    s.add_argument("--priority", default="all", choices=("all", "P0", "P1", "P2", "Moonshot"))
    s.add_argument("--results", default=str(runner.RESULTS_DIR))
    s.add_argument("--sync-gcp", action="store_true", help="log runs to Vertex AI Experiments and register 6 model families in Vertex AI Model Registry")

    s = sub.add_parser("qc", help="run 12-gate automated labelling, annotation, and pipeline QC verification")
    s.add_argument("--verify-all", action="store_true", help="verify all 8 Golden manifests, detector source, and results/")
    s.add_argument("--backfill-results", action="store_true", help="synchronize existing results/ runs with v2.2 canonical taxonomy before verifying")
    s.add_argument("--results", default=str(runner.RESULTS_DIR))

    s = sub.add_parser("hf-gpu-job", help="launch a Vertex AI GPU CustomJob loading Hugging Face Cloudflare/clef-flash, Cloudflare/clef-27b, or Contrastive-LM/CLM-v0.1-8B across val+test (1,000+ samples)")
    s.add_argument("-a", "--approach", nargs="+", default=["clef_standalone", "clef_standalone_classify"])
    s.add_argument("-m", "--model", default="Cloudflare/clef-flash",
                   choices=("all", "Cloudflare/clef-flash", "Cloudflare/clef-27b", "Contrastive-LM/CLM-v0.1-8B"))
    s.add_argument("-d", "--dataset", default="hul_golden",
                   choices=("hul_golden", "hul_variants", "hul_packs", "hul_brands", "hul_promos", "hul_sessions", "gt_hul_sessions", "gtmerch_golden", "ms_json_golden"))
    s.add_argument("--splits", nargs="+", default=["val", "test"], choices=("val", "test"))
    s.add_argument("--limit", type=int, default=1000, help="samples per split (default 1000 => 2000 across val+test)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--owner", default=None)
    s.add_argument("--dry-run", action="store_true", help="output Vertex AI CustomJob JSON payload without submitting")

    s = sub.add_parser(
        "legacy-eval",
        help="export standalone legacy_mapping_table.jsonl or score joined legacy Brand (ABrand) + Pack (Category__PackType) + Variant predictions",
    )
    s.add_argument("-d", "--dataset", default="hul_golden", help="Golden dataset name (default: hul_golden)")
    s.add_argument("--split", default="test", choices=dataset.SPLITS)
    s.add_argument("--root", default=None, help="Override dataset root directory or GCS URI")
    s.add_argument(
        "--export-mapping",
        default=None,
        help="Write standalone legacy_mapping_table.jsonl to this local path",
    )
    s.add_argument("--brand-preds", default=None, help="JSON/JSONL file mapping image_id -> raw legacy brand (e.g. ADove)")
    s.add_argument("--pack-preds", default=None, help="JSON/JSONL file mapping image_id -> raw legacy Category__PackType (e.g. Hair_Care__Bottle)")
    s.add_argument("--variant-preds", default=None, help="JSON/JSONL file mapping image_id -> raw legacy variant (e.g. ADove__Hair_Fall_Rescue)")

    s = sub.add_parser("load-test", help="run national peak load test & autoscaling analysis on Google Cloud Run")
    s.add_argument("--duration-minutes", type=float, default=5.0, help="duration in minutes (default: 5.0)")
    s.add_argument("--shelves-per-min", type=float, default=2500.0, help="ingress shelves per minute (default: 2500.0)")
    s.add_argument("--products-per-shelf", type=float, default=50.0, help="products per shelf (default: 50.0)")
    s.add_argument("--approach", default="sft_clm_clef_cascade", help="pipeline approach (Option A default: sft_clm_clef_cascade)")
    s.add_argument("--dry-run-vlm", action="store_true", default=True, help="simulate Gemini calls with calibrated latency/tokens")
    s.add_argument("--cloud-run", action="store_true", help="execute 100%% on GCP Cloud Run Job instead of locally")
    s.add_argument("--image", default=None, help="Artifact Registry container image URI to reuse")
    s.add_argument("--job", default="shelf-bench-loadtest", help="override Cloud Run job name (default: shelf-bench-loadtest)")
    s.add_argument("--cpu", type=int, default=4, help="Cloud Run container vCPU count (default: 4)")
    s.add_argument("--memory-gib", type=int, default=16, help="Cloud Run container RAM GiB (default: 16)")

    a = p.parse_args(argv)
    if a.cmd in ("run", "cloud-run"):
        if not a.model:
            if set(a.approach) != {"cv_classify"}:
                p.error("the following arguments are required: -m/--model")
            a.model = ["keras-h5"]
        from utils.cv_models import ROLES

        for role in ROLES:  # read by utils/cv_models.CVPipeline.from_config
            if getattr(a, f"{role}_model", None):
                os.environ[f"SHELF_BENCH_CV_{role.upper()}_MODEL"] = getattr(a, f"{role}_model")

    if a.cmd == "download":
        dataset.download(a.root)
    elif a.cmd == "upload":
        dataset.upload(a.root, a.to)
    elif a.cmd == "rpc":
        dataset.prepare_rpc()
    elif a.cmd == "shelves":
        dataset.prepare_shelves(a.raw)
    elif a.cmd == "hul":
        import json

        from utils.hul_builder import publish_v2_golden_manifests_to_gcs

        summary = publish_v2_golden_manifests_to_gcs(
            project=cfg.get("gcp", {}).get("project", "ul-fs-s-sandbx-16-prj"),
            max_train_per_split=a.max_train,
            max_eval_per_split=a.max_eval,
        )
        print(json.dumps(summary, indent=2))
    elif a.cmd == "list":
        print("Approaches (use case / task / dataset / models it runs):")
        for name, ap in approaches.all_approaches().items():
            print(f"  {name:<20} {ap.use_case:<13} {ap.task:<15} {ap.dataset:<9} "
                  f"{' + '.join([', '.join(ap.models) if ap.models else 'any Gemini', *ap.also_calls])}  {ap.architecture}")
        print("Gemini models:")
        for m in cfg.get("models", []):
            print(f"  {m}")
    elif a.cmd == "run":
        combos = runner.combos(a.approach, a.tier, a.model)  # fails fast on typos before spending
        if int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1")) > 1:  # one combo per Cloud Run task
            combos = [combos[int(os.environ["CLOUD_RUN_TASK_INDEX"])]]
        failed = 0
        for name, tier, model in combos:
            try:
                runner.run(name, model, a.split, a.limit, a.seed, a.workers, a.owner,
                           results_dir=a.results, data_root=a.root, tier=tier,
                           dataset_override=a.dataset, fields=a.fields)
            except Exception as e:
                failed += 1
                print(f"[{name} x {model} x {tier}] FAILED: {e}", file=sys.stderr)
        return 1 if failed or not combos else 0
    elif a.cmd == "cloud-run":
        from utils import cloud
        from utils.cv_models import VARIANT_ROLES

        return cloud.run_on_cloud(
            a.approach, a.model, a.tier, a.split, a.limit, a.seed, a.workers,
            a.owner or getpass.getuser(),
            dataset_override=a.dataset,
            fields=a.fields,
            brand_model=a.brand_model,
            pack_model=a.pack_model,
            image=a.image,
            job_name=a.job,
            pull_local=bool(a.pull and not a.no_pull),
            **{f"{r}_model": getattr(a, f"{r}_model", None) for r in VARIANT_ROLES},
        )
    elif a.cmd == "deploy-cv-endpoints":
        import json

        from utils import cloud
        from utils.cv_models import VARIANT_ROLES

        print(json.dumps(cloud.deploy_cv_endpoints(
            a.brand_model,
            a.pack_model,
            a.image,
            roles=a.roles,
            **{r: getattr(a, f"{r}_model", None) for r in VARIANT_ROLES},
        ), indent=2))
    elif a.cmd == "pull":
        print(f"Pulled {runner.pull(cfg['gcp']['results'])} new run(s) into results/")
    elif a.cmd == "push":
        import shutil

        dest = a.to or cfg["gcp"]["results"]
        pushed = runner.push(dest, results_dir=a.results, overwrite=a.overwrite, prune_remote=a.prune)
        print(f"Pushed {pushed} run(s) + registry & QC reports from {a.results}/ to {dest}")
        if a.clean_local and Path(a.results).is_dir():
            shutil.rmtree(a.results)
            print(f"Cleaned local directory: {a.results}/")
    elif a.cmd == "leaderboard":
        print_leaderboard(runner.leaderboard(a.results))
    elif a.cmd == "serve":
        import threading

        from utils.server import serve

        eff_results = runner.resolve_results_dir(a.results)
        if a.pull:
            def _bg_initial_pull() -> None:
                try:
                    n = runner.pull(
                        cfg["gcp"]["results"],
                        eff_results,
                        include_images=True,
                        finalize_compute=False,
                    )
                    if n:
                        print(f"Pulled {n} updated/pruned run(s) into {eff_results}/")
                except Exception:
                    pass

            threading.Thread(target=_bg_initial_pull, daemon=True).start()
        serve(a.host, a.port, eff_results, a.root, pull_src=cfg["gcp"]["results"] if a.pull else None, mode=a.mode)
    elif a.cmd == "serve-cloud":
        from utils import cloud

        cloud.deploy_ui()
    elif a.cmd == "proxy":
        from utils import cloud

        cloud.proxy_ui(port=a.port, host=a.host)
    elif a.cmd == "bootstrap":
        from utils import cloud

        cloud.bootstrap()
    elif a.cmd == "tune":
        import json

        from utils.gemini_fine_tuning import tuning

        if a.status:
            for j in tuning.list_sft_jobs(cfg, region=a.region):
                print(f"  {j['name']}  state={j['state']}  base={j['base_model']}  endpoint={j['endpoint']}")
            return 0
        if a.clean_hul_golden:
            from utils.gemini_fine_tuning.hul_golden_sft import (
                DEFAULT_5ATTR_PROMPT_TEXT,
                DEFAULT_PROMPT_TEXT,
                HulGoldenSftConfig,
                build_clean_splits_in_gcp,
            )

            f_mode = int(getattr(a, "fields", 5) or 5)
            out_pref = "derived_manifests/hul_golden/sft_5attr" if f_mode >= 5 else "derived_manifests/hul_golden/sft_3attr"
            default_disp = (
                "marketshare-hul-golden-5attr-sft-euw4-gemini-3-5-flash-lite"
                if f_mode >= 5
                else "marketshare-hul-golden-clean-sft-euw4-gemini-3-5-flash-lite"
            )
            hcfg = HulGoldenSftConfig(
                output_prefix=out_pref,
                fields_mode=f_mode,
                region=a.region or HulGoldenSftConfig.region,
                base_model=a.base_model,
                tuned_model_display_name=a.display_name if a.display_name != "shelf-bench-jev-laya-sft" else default_disp,
                epoch_count=a.epochs,
                learning_rate_multiplier=a.lr_multiplier,
                adapter_size=a.adapter_size,
                max_train=HulGoldenSftConfig.max_train if a.max_train is None else a.max_train,
                max_val=HulGoldenSftConfig.max_val if a.max_val is None else a.max_val,
                prompt_text=DEFAULT_5ATTR_PROMPT_TEXT if f_mode >= 5 else DEFAULT_PROMPT_TEXT,
            )
            stats = build_clean_splits_in_gcp(hcfg, launch_sft=not (a.build_only or a.upload_only), app_config=cfg)
            print(json.dumps(stats, indent=2))
            return 0
        if a.launch_only or a.train_uri:
            resolved = {} if a.train_uri else tuning.resolve_sft_dataset_uris(a.dataset, task=a.task)
            train_uri = a.train_uri or resolved["train_uri"]
            val_uri = a.val_uri or resolved.get("val_uri")
        else:
            built = tuning.build_sft_jsonl(
                a.dataset,
                a.out_dir,
                upload=not a.build_only,
                task=a.task,
                clean_local=a.clean_local,
            )
            print(f"Built {built['train_count']} train / {built['val_count']} val examples in {built['out_dir']}")
            if a.build_only or a.upload_only:
                return 0
            train_uri, val_uri = built["train_uri"], built["val_uri"]
        if a.max_train or a.max_val:
            sub_res = tuning.subset_gcs_sft_jsonl(train_uri, val_uri, max_train=a.max_train, max_val=a.max_val)
            train_uri, val_uri = sub_res["train_uri"], sub_res["val_uri"]
        job = tuning.launch_sft_job(
            train_uri,
            val_uri,
            base_model=a.base_model,
            display_name=a.display_name,
            epochs=a.epochs,
            adapter_size=a.adapter_size,
            learning_rate_multiplier=a.lr_multiplier,
            config=cfg,
            region=a.region,
        )
        print(f"Launched Vertex AI SFT job: {job['name']} (state={job['state']}, endpoint={job['endpoint']})")
    elif a.cmd == "sync-brands":
        from utils.brand_registry_crawler import write_live_brand_sku_registry

        reg = write_live_brand_sku_registry(output_path=a.output, crawl_online=a.online)
        s_info = reg["summary"]
        print(
            f"Synced {s_info['total_brands']} brands ({s_info['hul_brands']} HUL, {s_info['non_hul_brands']} Non-HUL) "
            f"and {s_info['total_skus']} SKUs ({s_info['hul_skus']} HUL, {s_info['non_hul_skus']} Non-HUL) -> {a.output}"
        )
    elif a.cmd == "experiments":
        from utils.experiment_matrix import (
            build_experiments_and_models_catalog,
            sync_experiments_and_models_to_gcp,
        )

        cat = (
            sync_experiments_and_models_to_gcp(results_dir=a.results)
            if a.sync_gcp
            else build_experiments_and_models_catalog(results_dir=a.results)
        )
        exps = cat["experiments"]
        if a.priority != "all":
            exps = [e for e in exps if e["priority"] == a.priority]
        hdr = (
            f"{'Prio':<10} {'ID':<11} {'Use Case':<13} {'Task':<20} {'Dataset':<14} "
            f"{'Est ₹/img':>9} {'Est ₹/15':>8} {'Est ₹/50':>8}  "
            f"{'L1 HUL':>6} {'L2 Cat/Pk':>9} {'L3 Brd':>6} {'L4 Var':>6} {'L5 SKU':>6}  {'Best Run'}"
        )
        print(hdr)
        print("-" * len(hdr))
        for e in exps:
            c = e["cost_estimate"]
            best = e.get("best_run")
            if best:
                l1_s = f"{best.get('is_hul_accuracy', 0) * 100:>5.1f}%"
                l2_s = f"{best.get('category_accuracy', 0) * 100:>3.0f}/{best.get('packaging_accuracy', 0) * 100:>3.0f}%"
                l3_s = f"{best.get('brand_accuracy', 0) * 100:>5.1f}%"
                l4_s = f"{best.get('base_variant_accuracy', 0) * 100:>5.1f}%"
                l5_s = f"{best.get(e['primary_metric'], 0) * 100:>5.1f}%"
                best_str = f"{best['run_id']} (₹{best.get('cost_per_image_inr', 0):.3f}/img)"
            else:
                l1_s = l2_s = l3_s = l4_s = l5_s = "  —  "
                best_str = "planned (not run yet)"
            print(
                f"[{e['priority']:<8}] {e['experiment_id']:<11} {e['use_case']:<13} {e['task']:<20} {e['dataset']:<14} "
                f"₹{c['est_cost_per_image_inr']:>8.3f} ₹{c['est_15_images_inr']:>7.2f} ₹{c['est_50_images_inr']:>7.2f}  "
                f"{l1_s:>6} {l2_s:>9} {l3_s:>6} {l4_s:>6} {l5_s:>6}  {best_str}"
            )
        print(f"\nRegistered Model Families ({len(cat['models'])}):")
        for m in cat["models"]:
            print(f"  {m['display_name']:<40} {m['version_alias']:<20} {m['model_resource']}")
    elif a.cmd == "qc":
        import json

        from utils.qc import backfill_results_taxonomy, run_12_gate_qc

        if a.backfill_results:
            updated = backfill_results_taxonomy(results_dir=a.results)
            print(f"Backfilled canonical taxonomy on {updated} existing run(s) in {a.results}/")
        report = run_12_gate_qc(results_dir=a.results)
        print(json.dumps(report, indent=2))
        return 0 if report["all_passed"] else 1
    elif a.cmd == "hf-gpu-job":
        import json

        from utils import cloud

        target_models = (
            ["Cloudflare/clef-flash", "Cloudflare/clef-27b", "Contrastive-LM/CLM-v0.1-8B"]
            if a.model == "all"
            else [a.model]
        )
        results = [
            cloud.submit_hf_gpu_custom_job(
                approaches=a.approach,
                hf_model=m,
                dataset_name=a.dataset,
                splits=tuple(a.splits),
                limit=a.limit,
                seed=a.seed,
                workers=a.workers,
                owner=a.owner or getpass.getuser(),
                dry_run=a.dry_run,
            )
            for m in target_models
        ]
        print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
    elif a.cmd == "legacy-eval":
        import json

        from utils.legacy_benchmark import (
            evaluate_joined_legacy_models,
            write_legacy_mapping_table_jsonl,
        )

        ds_root = a.root or dataset.data_root(a.dataset)
        if a.export_mapping:
            mf = dataset._rpc_json(ds_root)
            n_rows = write_legacy_mapping_table_jsonl(mf.get("classes", {}), a.export_mapping)
            print(f"Exported {n_rows} legacy mapping rows -> {a.export_mapping}")
        if a.brand_preds or a.pack_preds or a.variant_preds:
            def _load_pred_map(path_str: str | None, field_hint: str) -> dict[str, str]:
                if not path_str:
                    return {}
                raw_txt = Path(path_str).read_text().strip()
                try:
                    parsed = json.loads(raw_txt)
                    if isinstance(parsed, dict):
                        return {str(k): str(v) for k, v in parsed.items()}
                except Exception:
                    pass
                out_map: dict[str, str] = {}
                for line in raw_txt.splitlines():
                    if not line.strip():
                        continue
                    row_obj = json.loads(line)
                    img_k = str(row_obj.get("image") or row_obj.get("image_id") or "")
                    val_s = str(row_obj.get(field_hint) or row_obj.get("prediction") or row_obj.get("label") or "")
                    if img_k:
                        out_map[img_k] = val_s
                return out_map

            samples = list(dataset.load_rpc(a.split, ds_root).values())
            report = evaluate_joined_legacy_models(
                samples=samples,
                brand_preds=_load_pred_map(a.brand_preds, "brand"),
                pack_preds=_load_pred_map(a.pack_preds, "packaging_type"),
                variant_preds=_load_pred_map(a.variant_preds, "base_variant"),
            )
            print(json.dumps({k: v for k, v in report.items() if k != "labels"}, indent=2))
    elif a.cmd == "load-test":
        if a.cloud_run:
            from utils.cloud import run_cli_on_cloud
            cli_args = [
                "load-test",
                "--duration-minutes", str(a.duration_minutes),
                "--shelves-per-min", str(a.shelves_per_min),
                "--products-per-shelf", str(a.products_per_shelf),
                "--approach", a.approach,
            ]
            if a.dry_run_vlm:
                cli_args.append("--dry-run-vlm")
            run_cli_on_cloud(cli_args, image=a.image, tasks=1, cpu=a.cpu, memory_gib=a.memory_gib, job_name=a.job, log=print)
        else:
            from utils.load_tester import LoadTestConfig, run_load_test
            cfg = LoadTestConfig(
                duration_minutes=a.duration_minutes,
                shelves_per_min=a.shelves_per_min,
                products_per_shelf=a.products_per_shelf,
                approach=a.approach,
                dry_run_vlm=a.dry_run_vlm,
                container_cpu=a.cpu,
                container_ram_gib=a.memory_gib,
            )
            run_load_test(cfg)
    return 0



def print_leaderboard(rows: list[dict]) -> None:
    if not rows:
        print("No runs yet. Try: shelf-bench run -a single_pass_dedup -m gemini-3.5-flash-lite")
        return
    hdr = f"{'#':>3}  {'run id':<56} {'owner':<10} {'acc':>5} {'rec':>5} {'F2':>5} {'p95':>6} {'p99':>6} {'₹/img':>8}"
    use_case = task = None
    for r in rows:  # sorted by use case and task, ranked within each
        if r["use_case"] != use_case:
            use_case, task = r["use_case"], None
            print(f"\n=== {use_case.replace('_', ' ').title()} ===")
        if (r["task"], r["dataset"]) != task:
            task = (r["task"], r["dataset"])
            if r["dataset"] == "shelves":
                label = f"shelves_{r['task']}"
            elif r["dataset"].startswith(("hul_", "gtmerch", "ms_json")):
                label = f"{r['dataset']}_{r['task']}"
            else:
                label = r["task"]
            by = "ranked by F2" if r["task"] in ("detection", "end_to_end") else "ranked by exact product accuracy"
            print(f"\n{label} ({by})")
            print(hdr)
            print("-" * len(hdr))
        print(f"{r['rank']:>3}  {r['run_id']:<56} {r['owner'][:10]:<10} {r['accuracy']:>5.3f} "
              f"{r['recall']:>5.3f} {r['f2']:>5.3f} {r['p95_latency_s']:>5.1f}s "
              f"{r['p99_latency_s']:>5.1f}s {r['cost_per_image_inr']:>8.3f}")


if __name__ == "__main__":
    sys.exit(main())
