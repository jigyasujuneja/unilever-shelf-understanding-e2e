"""100% Cloud Run orchestrator for Unilever Golden .jsonl datasets (High-Sample-Size Wave + Full Matrix).

Can be run either:
1. Directly inside a Cloud Run Job (`--launch-on-cloud-run`), where it authenticates via the
   GCP Metadata Server (`663686437672-compute@developer.gserviceaccount.com`) and never suffers
   from 4-hour local OAuth RAPT token expiration.
2. From the workstation after `make auth`, where it reuses the pre-built Cloud Build image and
   skips any (dataset, split, approach, model, limit) combination that already has a valid
   completed `summary.json` in `gs://ul-fs-s-sandbx-16-prj-shelf-images/results/`.
"""

from __future__ import annotations

import argparse
import sys
import time

from utils import cloud
from utils.llm import load_config

MODELS_ALL = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-embedding-2",
    "multimodalembedding@001",
    "yolo11n-distilled-onnx",
    "yolov12s-sku110k-onnx",
]

# Fast models for high-sample-size (N=500-1000 per split) statistical power wave
MODELS_HIGH_N = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-embedding-2",
    "yolo11n-distilled-onnx",
    "yolov12s-sku110k-onnx",
]

# High-Sample-Size Anchor & Production Approaches (run at N=500-1000 per split -> ±0.7%-1.5% 95% CI)
HIGH_N_CROP_APPS = [
    "clm_standalone_classify",
    "embedding_text_match",
    "clm_standalone",
    "clm_contrastive",
    "embedding_retrieval",
    "coarse_jev_laya",
    "cold_start_lid_discovery",
    "hierarchical_clm_jev",
    "jev_laya_global_wb",
    "jev_laya_zscore_lid",
    "l8_hul_ultimate",
    "outlet_scoped_clm_jev",
    "row_neighbor_prior_jev",
    "session_overlap_dedup_jev",
    "shelf_height_size_prior_jev",
    "tube_midband_jev",
]

CLS_APPS = [
    "clef_flash_classify",
    "clef_standalone_classify",
    "clm_classify",
    "clm_standalone_classify",
    "coarse_jev_gemini",
    "djev_classify",
    "embedding_text_match",
    "hierarchy_classify",
]

RET_APPS = [
    "clef_flash_decision",
    "clef_standalone",
    "clm_contrastive",
    "clm_standalone",
    "coarse_jev_laya",
    "cold_start_lid_discovery",
    "embedding_retrieval",
    "hierarchical_clm_jev",
    "jev_laya_global_wb",
    "jev_laya_hybrid",
    "jev_laya_zoom_strip",
    "jev_laya_zscore_lid",
    "l8_hul_ultimate",
    "outlet_scoped_clm_jev",
    "row_neighbor_prior_jev",
    "session_overlap_dedup_jev",
    "shelf_height_size_prior_jev",
    "tiered_hybrid",
    "tube_midband_jev",
]

DET_APPS = [
    "distilled_yolo_960_softnms",
    "distilled_yolo_onnx",
    "single_pass_dedup",
    "tiled_dedup",
    "yolo_sku110k_onnx",
]

E2E_APPS = [
    "detect_clef_flash",
    "detect_clef_standalone",
    "detect_clm_contrastive",
    "detect_clm_standalone",
    "detect_coarse_jev_gemini",
    "detect_jev_laya",
    "detect_l8_ultimate",
    "detect_onnx_jev_laya",
    "detect_retrieve",
    "detect_tiered",
    "session_overlap_dedup_e2e",
    "shelf_960_softnms_e2e",
    "shelf_clef_standalone",
    "shelf_clm_standalone",
    "shelf_coarse_jev_gemini",
    "shelf_detect_retrieve",
    "shelf_detect_tiered",
    "shelf_height_size_prior",
    "shelf_hierarchical_clm_jev",
    "shelf_jev_laya",
    "shelf_l8_ultimate",
    "shelf_onnx_jev_laya",
    "shelf_outlet_scoped_clm_jev",
    "shelf_row_neighbor_prior",
    "shelf_tube_midband_jev",
]

PLANO_APPS = ["bay_planogram_verifier"]
PROMO_APPS = ["clm_promo_matcher", "jev_promo_verifier"]

DEFAULT_IMAGE = "us-central1-docker.pkg.dev/ul-fs-s-sandbx-16-prj/cloud-run-source-deploy/shelf-bench:20261012-202859"


def ulog(*args: object) -> None:
    print(*args, flush=True)


def get_completed_keys() -> set[tuple[str, str, str, str, int]]:
    """Return (dataset, split, approach, model, limit) tuples already completed in GCS results."""
    import json
    from pathlib import Path

    import runner

    cfg = load_config()
    done: set[tuple[str, str, str, str, int]] = set()
    try:
        cache_dir = Path("/tmp/shelf_bench_results_cache")
        runner.pull(src=cfg["gcp"]["results"], results_dir=cache_dir, log=lambda *a: None, include_images=False)
        for p in cache_dir.glob("*/summary.json"):
            try:
                s = json.loads(p.read_text())
            except Exception:
                continue
            if float(s.get("accuracy") or 0.0) <= 0.0 and float(s.get("f2") or 0.0) <= 0.0:
                continue
            ds = str(s.get("dataset") or "")
            # Force fresh Cloud Run execution on newly expanded shelf datasets if run before 2026-10-13T23:20:00
            if ds in ("gt_hul_sessions", "hul_sessions", "gtmerch_golden", "ms_json_golden"):
                if str(s.get("created_at") or "") < "2026-10-13T23:20:00":
                    continue
            sp = str(s.get("split") or "")
            app = str(s.get("approach") or "")
            mod = str(s.get("model") or "")
            lim = int(s.get("limit") or s.get("images") or 0)
            done.add((ds, sp, app, mod, lim))
    except Exception as exc:
        ulog(f"Warning: could not pre-scan GCS summaries ({exc}); running full batch.")
    return done


def run_batch_resilient(
    label: str,
    ds: str,
    sp: str,
    lim: int,
    apps: list[str],
    models: list[str],
    image: str,
    workers: int,
    completed: set[tuple[str, str, str, str, int]],
) -> None:
    import runner

    # Filter out approaches where all compatible models in `models` are already done at >= lim
    pending_apps: list[str] = []
    for app in apps:
        c = runner.combos([app], ["priority"], models, lambda *a: None)
        if any((ds, sp, app, m, lim) not in completed for _, _, m in c):
            pending_apps.append(app)

    if not pending_apps:
        ulog(f"=== [{label}] SKIPPED (all {len(apps)} approaches for {ds}/{sp} limit={lim} already in GCS) ===")
        return

    t0 = time.time()
    ulog(
        f"\n=== [{label}] dataset={ds} split={sp} limit={lim} "
        f"pending_approaches={len(pending_apps)}/{len(apps)} models={len(models)} workers={workers} ==="
    )
    try:
        cloud.run_on_cloud(
            approaches=pending_apps,
            models=models,
            tiers=["priority"],
            split=sp,
            limit=lim,
            seed=0,
            workers=workers,
            owner="jjuneja",
            dataset_override=ds,
            image=image,
            pull_local=False,
            log=ulog,
        )
        ulog(f"=== [{label}] COMPLETED 100% in {time.time() - t0:.1f}s ===")
    except RuntimeError as exc:
        ulog(
            f"=== [{label}] COMPLETED with partial task timeout after {time.time() - t0:.1f}s "
            f"(all succeeded tasks finalized in GCS): {exc} ==="
        )


def sync_expanded_manifests_and_cache_to_gcs() -> None:
    """Upload expanded data/* manifests (.json + .jsonl) and gallery_path_vec_cache.pkl to GCS."""
    from pathlib import Path

    from google.cloud import storage  # type: ignore[attr-defined]

    from utils.llm import ensure_gcloud_auth_fallback

    ensure_gcloud_auth_fallback()
    client = storage.Client(project="ul-fs-s-sandbx-16-prj")
    b_ms = client.bucket("marketshare-datasets")
    b_gt = client.bucket("gtmerch-marketshare")
    b_res = client.bucket("ul-fs-s-sandbx-16-prj-shelf-images")

    gal_cache = Path("/tmp/shelf_bench_emb_cache/gallery_path_vec_cache.pkl")
    if gal_cache.exists():
        ulog(f"Uploading {gal_cache} ({gal_cache.stat().st_size / 1e6:.1f} MB) -> gs://ul-fs-s-sandbx-16-prj-shelf-images/cache/gallery_path_vec_cache.pkl ...")
        b_res.blob("cache/gallery_path_vec_cache.pkl").upload_from_filename(str(gal_cache))

    for ds in (
        "gt_hul_sessions",
        "hul_sessions",
        "gtmerch_golden",
        "ms_json_golden",
        "hul_golden",
        "hul_variants",
        "hul_packs",
        "hul_brands",
        "hul_promos",
    ):
        ds_dir = Path("data") / ds
        if not ds_dir.exists():
            continue
        for fname in ("rpc.json", "train.jsonl", "val.jsonl", "test.jsonl"):
            fpath = ds_dir / fname
            if fpath.exists():
                b_ms.blob(f"derived_manifests/{ds}/{fname}").upload_from_filename(str(fpath))
                if ds == "gtmerch_golden":
                    b_gt.blob(f"derived_manifests/{ds}/{fname}").upload_from_filename(str(fpath))
        ulog(f"Synced expanded manifest & splits for {ds} to GCS.")


def launch_orchestrator_on_cloud_run(image: str, cfg: dict, only_exp28_30: bool = False) -> int:
    """Deploy and trigger shelf-bench-orchestrator as a 24-hour Cloud Run Job using GCP service account."""
    s = cloud._session()
    project = cfg["gcp"]["project"]
    region = cfg["gcp"]["region"]
    job = "shelf-bench-orchestrator"
    base = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/jobs"
    c_args = ["-u", "/app/scripts/run_all_golden_cloud_batches.py", "--image", image]
    if only_exp28_30:
        c_args.append("--only-exp28-30")
    body = {
        "template": {
            "taskCount": 1,
            "parallelism": 1,
            "template": {
                "maxRetries": 0,
                "timeout": "86400s",
                "containers": [
                    {
                        "image": image,
                        "command": ["python3"],
                        "args": c_args,
                        "env": [{"name": "PYTHONPATH", "value": "/app/src"}],
                        "resources": {"limits": {"cpu": "2", "memory": "8Gi"}},
                    }
                ],
            },
        }
    }
    exists = s.get(f"{base}/{job}", timeout=30).status_code == 200
    op = cloud._ok(
        s.patch(f"{base}/{job}", json=body, timeout=60)
        if exists
        else s.post(f"{base}?jobId={job}", json=body, timeout=60)
    )
    cloud._wait_op(s, op["name"])
    run_op = cloud._ok(s.post(f"{base}/{job}:run", json={}, timeout=60))
    exec_id = (run_op.get("metadata", {}).get("name") or "").rsplit("/", 1)[-1]
    ulog(
        f"Launched 24h Cloud Run Orchestrator Job {job} ({exec_id}):\n"
        f"  https://console.cloud.google.com/run/jobs/executions/details/{region}/{exec_id}?project={project}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Golden Cloud Run batches with GCS checkpointing.")
    parser.add_argument("--image", default=None, help="Optional pre-built Artifact Registry image URI (builds fresh image if omitted)")
    parser.add_argument("--sync-gcs", action="store_true", default=False, help="Sync expanded manifests & gallery cache to GCS before launching")
    parser.add_argument(
        "--launch-on-cloud-run",
        action="store_true",
        default=False,
        help="Deploy and run this orchestrator itself inside a 24h Cloud Run Job (eliminates 4-hour local RAPT expiry)",
    )
    parser.add_argument(
        "--only-exp28-30",
        action="store_true",
        default=False,
        help="Run strictly EXP-28 (clef_contrastive_metadata, l8_hul_ultimate), EXP-29 (sft_clm_clef_cascade Base & SFT), and EXP-30 (distilled_yolo_960_softnms, shelf_960_softnms_e2e) on val and test",
    )
    args = parser.parse_args()

    if args.sync_gcs:
        try:
            sync_expanded_manifests_and_cache_to_gcs()
        except Exception as exc:
            ulog(f"Warning: GCS manifest sync skipped ({exc})")

    cfg = load_config()
    image = args.image or cloud.build_image(
        cloud._session(), cfg["gcp"]["project"], cfg["gcp"]["region"], log=ulog
    )
    ulog("USING_CLOUD_RUN_IMAGE:", image)
    if args.launch_on_cloud_run:
        return launch_orchestrator_on_cloud_run(image, cfg, only_exp28_30=args.only_exp28_30)
    if args.only_exp28_30:
        exp28_29_apps = ["clef_contrastive_metadata", "sft_clm_clef_cascade", "l8_hul_ultimate"]
        exp28_29_models = [
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash-lite-sft",
        ]
        exp30_apps = ["distilled_yolo_960_softnms", "shelf_960_softnms_e2e"]
        exp30_models = ["yolo11n-distilled-onnx", "gemini-3.5-flash-lite"]
        exp_batches = [
            ("EXP-28 & EXP-29 (hul_variants Val 200)", "hul_variants", "val", 200, exp28_29_apps, exp28_29_models, 12),
            ("EXP-28 & EXP-29 (hul_variants Test 1000)", "hul_variants", "test", 1000, exp28_29_apps, exp28_29_models, 12),
            ("EXP-28 & EXP-29 (hul_golden Val 200)", "hul_golden", "val", 200, exp28_29_apps, exp28_29_models, 12),
            ("EXP-28 & EXP-29 (hul_golden Test 1000)", "hul_golden", "test", 1000, exp28_29_apps, exp28_29_models, 12),
            ("EXP-30 (ms_json_golden Val 60)", "ms_json_golden", "val", 60, exp30_apps, exp30_models, 4),
            ("EXP-30 (ms_json_golden Test 100)", "ms_json_golden", "test", 100, exp30_apps, exp30_models, 4),
            ("EXP-30 (gtmerch_golden Val 100)", "gtmerch_golden", "val", 100, exp30_apps, exp30_models, 4),
            ("EXP-30 (gtmerch_golden Test 200)", "gtmerch_golden", "test", 200, exp30_apps, exp30_models, 4),
        ]
        for label, ds, sp, lim, apps, models, w in exp_batches:
            run_batch_resilient(label, ds, sp, lim, apps, models, image, workers=w, completed=set())
        return 0

    completed = get_completed_keys()
    ulog(f"Found {len(completed)} existing completed (dataset, split, approach, model, limit) runs in GCS.")

    # WAVE 0: Winner Priority Wave — Ensure ALL winning approaches have gemini-3.5-flash-lite (and gemini-3.8-flash) on Cloud Run first
    winner_shelf_apps = [
        "distilled_yolo_onnx",
        "distilled_yolo_960_softnms",
        "tiled_dedup",
        "single_pass_dedup",
        "shelf_l8_ultimate",
        "shelf_hierarchical_clm_jev",
        "shelf_960_softnms_e2e",
        "session_overlap_dedup_e2e",
        "shelf_clm_standalone",
        "bay_planogram_verifier",
    ]
    winner_crop_apps = [
        "clef_contrastive_metadata",
        "sft_clm_clef_cascade",
        "l8_hul_ultimate",
        "jev_laya_zscore_lid",
        "clef_standalone",
        "jev_laya_global_wb",
        "hierarchical_clm_jev",
        "clm_standalone",
        "clef_standalone_classify",
        "hierarchy_classify",
        "clm_standalone_classify",
    ]
    winner_priority_batches = [
        ("Wave 0A (GT Sessions Winners Test 100 — Flash-Lite & 3.8-Flash)", "gt_hul_sessions", "test", 100, winner_shelf_apps, MODELS_HIGH_N, 4),
        ("Wave 0B (GT Sessions Winners Val 50 — Flash-Lite & 3.8-Flash)", "gt_hul_sessions", "val", 50, winner_shelf_apps, MODELS_HIGH_N, 4),
        ("Wave 0C (MT Market Share Shelves Winners Test 100 — Flash-Lite & 3.8-Flash)", "ms_json_golden", "test", 100, winner_shelf_apps, MODELS_HIGH_N, 4),
        ("Wave 0D (MT Market Share Shelves Winners Val 60 — Flash-Lite & 3.8-Flash)", "ms_json_golden", "val", 60, winner_shelf_apps, MODELS_HIGH_N, 4),
        ("Wave 0E (GT Merchandising Winners Test 200 — Flash-Lite & 3.8-Flash)", "gtmerch_golden", "test", 200, winner_shelf_apps, MODELS_HIGH_N, 4),
        ("Wave 0F (GT Merchandising Winners Val 100 — Flash-Lite & 3.8-Flash)", "gtmerch_golden", "val", 100, winner_shelf_apps, MODELS_HIGH_N, 4),
        ("Wave 0G (HUL Golden Winners Test 1000 — Flash-Lite & 3.8-Flash)", "hul_golden", "test", 1000, winner_crop_apps, MODELS_HIGH_N, 12),
        ("Wave 0H (HUL Golden Winners Val 1000 — Flash-Lite & 3.8-Flash)", "hul_golden", "val", 1000, winner_crop_apps, MODELS_HIGH_N, 12),
    ]
    for label, ds, sp, lim, apps, models, w in winner_priority_batches:
        run_batch_resilient(label, ds, sp, lim, apps, models, image, workers=w, completed=completed)
    completed = get_completed_keys()

    # WAVE 1: Expanded High-Sample-Size Statistical Power Wave (Batches 1-6 across MT Market Share, GT Merchandising, Detection & E2E)
    high_n_batches = [
        ("Batch 1a (GT Sessions Test)", "gt_hul_sessions", "test", 100, DET_APPS + HIGH_N_CROP_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("Batch 1b (GT Sessions Val)", "gt_hul_sessions", "val", 50, DET_APPS + HIGH_N_CROP_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("Batch 1c (HUL Sessions Alias Test)", "hul_sessions", "test", 100, DET_APPS + HIGH_N_CROP_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("Batch 1d (HUL Sessions Alias Val)", "hul_sessions", "val", 50, DET_APPS + HIGH_N_CROP_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("Batch 2a (GT Merchandising Test)", "gtmerch_golden", "test", 200, DET_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("Batch 2b (GT Merchandising Val)", "gtmerch_golden", "val", 100, DET_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("Batch 3a (MT Market Share Shelves Test)", "ms_json_golden", "test", 100, DET_APPS + HIGH_N_CROP_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("Batch 3b (MT Market Share Shelves Val)", "ms_json_golden", "val", 60, DET_APPS + HIGH_N_CROP_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("Batch 4a (MT Golden 5-Level Test)", "hul_golden", "test", 1000, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("Batch 4b (MT Golden 5-Level Val)", "hul_golden", "val", 1000, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("Batch 5a (MT Packs Test)", "hul_packs", "test", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("Batch 5b (MT Packs Val)", "hul_packs", "val", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("Batch 5c (MT Brands Test)", "hul_brands", "test", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("Batch 5d (MT Brands Val)", "hul_brands", "val", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("Batch 5e (MT Variants Test)", "hul_variants", "test", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("Batch 5f (MT Variants Val)", "hul_variants", "val", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("Batch 6a (MT Toker Promos Test)", "hul_promos", "test", 500, PROMO_APPS, MODELS_HIGH_N, 12),
        ("Batch 6b (MT Toker Promos Val)", "hul_promos", "val", 500, PROMO_APPS, MODELS_HIGH_N, 12),
    ]

    for label, ds, sp, lim, apps, models, w in high_n_batches:
        run_batch_resilient(label, ds, sp, lim, apps, models, image, workers=w, completed=completed)

    ulog("\n=== Running Interim 12-Gate QC Backfill + Vertex AI Sync after High-N Wave ===")
    cloud.run_cli_on_cloud(
        ["qc", "--backfill-results", "--verify-all"],
        image=image, tasks=1, cpu=4, memory_gib=8, log=ulog,
    )
    cloud.run_cli_on_cloud(
        ["experiments", "--sync-gcp"],
        image=image, tasks=1, cpu=4, memory_gib=8, log=ulog,
    )

    # WAVE 2: Remaining Full-Matrix VLM Ablation Batches (limit=50 per split for slow per-crop VLM baselines)
    vlm_ablation_apps = [a for a in (CLS_APPS + RET_APPS) if a not in HIGH_N_CROP_APPS]
    ablation_batches = [
        ("Ablation 1/6", "hul_variants", "test", 50, vlm_ablation_apps, MODELS_ALL, 12),
        ("Ablation 2/6", "hul_variants", "val", 50, vlm_ablation_apps, MODELS_ALL, 12),
        ("Ablation 3/6", "hul_packs", "test", 50, vlm_ablation_apps, MODELS_ALL, 12),
        ("Ablation 4/6", "hul_packs", "val", 50, vlm_ablation_apps, MODELS_ALL, 12),
        ("Ablation 5/6", "hul_brands", "test", 50, vlm_ablation_apps, MODELS_ALL, 12),
        ("Ablation 6/6", "hul_brands", "val", 50, vlm_ablation_apps, MODELS_ALL, 12),
    ]

    for label, ds, sp, lim, apps, models, w in ablation_batches:
        run_batch_resilient(label, ds, sp, lim, apps, models, image, workers=w, completed=completed)

    ulog("\n=== Running Final 12-Gate QC + Backfill + Vertex AI Experiments Sync on Cloud Run ===")
    cloud.run_cli_on_cloud(
        ["qc", "--backfill-results", "--verify-all"],
        image=image, tasks=1, cpu=4, memory_gib=8, log=ulog,
    )
    cloud.run_cli_on_cloud(
        ["experiments", "--sync-gcp"],
        image=image, tasks=1, cpu=4, memory_gib=8, log=ulog,
    )
    ulog("ALL_GOLDEN_CLOUD_BATCHES_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
