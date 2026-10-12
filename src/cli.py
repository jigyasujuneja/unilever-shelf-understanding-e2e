"""``shelf-bench`` command line.

    shelf-bench bootstrap                        # new GCP project: APIs, bucket, registry, data
    shelf-bench download                         # fetch + extract SKU-110K (~12 GB)
    shelf-bench list                             # approaches and models
    shelf-bench run -a single_pass_dedup -m gemini-3.5-flash-lite
    shelf-bench run -a single_pass_dedup tiled_dedup -m gemini-3.5-flash-lite gemini-3.8-flash
    shelf-bench run -a hierarchy_classify embedding_retrieval -m gemini-3.5-flash-lite gemini-embedding-2
    shelf-bench cloud-run -a shelf_detect_retrieve -m gemini-3.5-flash-lite -t priority flex
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
    s.add_argument("-m", "--model", nargs="+", required=True)
    s.add_argument("-t", "--tier", nargs="+", choices=TIERS, default=d.get("tier", ["priority"]),
                   help="Gemini PayGo tier(s): priority (1.8x, default), standard and/or flex (0.5x)")
    s.add_argument("-d", "--dataset", default=None,
                   choices=("sku110k", "products", *dataset.RPC_FORMAT_SETS),
                   help="override dataset for approaches (e.g. hul_golden, gtmerch_golden, hul_variants, hul_brands, hul_packs, hul_promos)")
    s.add_argument("--split", default=d.get("split", "test"), choices=dataset.SPLITS)
    s.add_argument("--limit", type=int, default=d.get("limit", 50), help="0 = whole split")
    s.add_argument("--seed", type=int, default=d.get("seed", 0))
    s.add_argument("--workers", type=int, default=d.get("workers", 4))
    s.add_argument("--owner", default=None, help="defaults to $USER")
    s.add_argument("--root", default=dataset.DEFAULT_ROOT, help="SKU-110K: local dir or gs:// URI")
    s.add_argument("--results", default=str(runner.RESULTS_DIR), help="local dir or gs:// URI")

    s = sub.add_parser("cloud-run", help="run approach x model combinations on Cloud Run")
    s.add_argument("-a", "--approach", nargs="+", required=True)
    s.add_argument("-m", "--model", nargs="+", required=True)
    s.add_argument("-t", "--tier", nargs="+", choices=TIERS, default=d.get("tier", ["priority"]),
                   help="Gemini PayGo tier(s): priority (1.8x, default), standard and/or flex (0.5x)")
    s.add_argument("--split", default=d.get("split", "test"), choices=dataset.SPLITS)
    s.add_argument("--limit", type=int, default=d.get("limit", 50), help="0 = whole split")
    s.add_argument("--seed", type=int, default=d.get("seed", 0))
    s.add_argument("--workers", type=int, default=d.get("workers", 4))
    s.add_argument("--owner", default=None, help="defaults to $USER")

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
    s.add_argument("--mode", choices=("engineering", "cio"), default="engineering",
                   help="default landing portal ('engineering' serves index.html, 'cio' serves cio.html)")

    sub.add_parser("serve-cloud", help="deploy the leaderboard UI (with results/) to Cloud Run")
    s = sub.add_parser("proxy", help="open the private Cloud Run UI on localhost (authenticated)")
    s.add_argument("--port", type=int, default=8083)
    s.add_argument("--host", default="127.0.0.1")
    sub.add_parser("bootstrap", help="prepare a new GCP project (APIs, bucket, registry, data)")

    s = sub.add_parser("tune", help="build zero-leakage SFT JSONL data and launch/list Vertex AI LoRA tuning jobs")
    s.add_argument("--dataset", default="shelves",
                   choices=("shelves", "rpc", "pack", "brand", "variant", "hul_golden", "gtmerch_golden", "hul_variants", "hul_brands", "hul_packs", "hul_promos"))
    s.add_argument("--task", default="all3", choices=("all3", "category", "brand", "variant"),
                   help="target task when --dataset=variant")
    s.add_argument("--base-model", default="gemini-3.5-flash-lite")
    s.add_argument("--display-name", default="shelf-bench-jev-laya-sft")
    s.add_argument("--epochs", type=int, default=4)
    s.add_argument("--adapter-size", type=int, default=4, choices=(1, 4, 8, 16))
    s.add_argument("--lr-multiplier", type=float, default=1.0)
    s.add_argument("--out-dir", default="data/tuning")
    s.add_argument("--build-only", action="store_true", help="only build local train/val JSONL files")
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
                   choices=("hul_golden", "hul_variants", "hul_packs", "hul_brands", "hul_promos", "hul_sessions", "gtmerch_golden", "ms_json_golden"))
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

    a = p.parse_args(argv)

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
                           dataset_override=a.dataset)
            except Exception as e:
                failed += 1
                print(f"[{name} x {model} x {tier}] FAILED: {e}", file=sys.stderr)
        return 1 if failed or not combos else 0
    elif a.cmd == "cloud-run":
        from utils import cloud

        return cloud.run_on_cloud(a.approach, a.model, a.tier, a.split, a.limit, a.seed, a.workers,
                                  a.owner or getpass.getuser())
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
        from utils.server import serve

        if a.pull:
            print(f"Pulled {runner.pull(cfg['gcp']['results'], a.results)} new run(s) into {a.results}/")
        serve(a.host, a.port, a.results, a.root, pull_src=cfg["gcp"]["results"] if a.pull else None, mode=a.mode)
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
        from utils import tuning

        if a.status:
            for j in tuning.list_sft_jobs(cfg):
                print(f"  {j['name']}  state={j['state']}  base={j['base_model']}  endpoint={j['endpoint']}")
            return 0
        built = tuning.build_sft_jsonl(a.dataset, a.out_dir, upload=not a.build_only, task=a.task)
        print(f"Built {built['train_count']} train / {built['val_count']} val examples in {built['out_dir']}")
        if a.build_only:
            return 0
        job = tuning.launch_sft_job(
            built["train_uri"],
            built["val_uri"],
            base_model=a.base_model,
            display_name=a.display_name,
            epochs=a.epochs,
            adapter_size=a.adapter_size,
            learning_rate_multiplier=a.lr_multiplier,
            config=cfg,
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
