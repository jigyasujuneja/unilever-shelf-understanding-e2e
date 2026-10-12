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
    "multimodalembedding@001",
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
    import runner

    cfg = load_config()
    done: set[tuple[str, str, str, str, int]] = set()
    try:
        for s in runner.load_summaries(cfg["gcp"]["results"]):
            if float(s.get("primary_score") or 0.0) <= 0.0:
                continue
            ds = str(s.get("dataset") or "")
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
        c = runner.combos([app], ["flex"], models, lambda *a: None)
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
            tiers=["flex"],
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Golden Cloud Run batches with GCS checkpointing.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Pre-built Artifact Registry image URI")
    args = parser.parse_args()

    image = args.image
    ulog("REUSING_BUILT_IMAGE:", image)
    completed = get_completed_keys()
    ulog(f"Found {len(completed)} existing completed (dataset, split, approach, model, limit) runs in GCS.")

    # WAVE 1: High-Sample-Size Statistical Power Wave (workers=4 on full-shelf, workers=12 on single-crop)
    high_n_batches = [
        ("High-N 1/16", "gtmerch_golden", "test", 100, DET_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("High-N 2/16", "gtmerch_golden", "val", 72, DET_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("High-N 3/16", "ms_json_golden", "test", 31, DET_APPS + HIGH_N_CROP_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("High-N 4/16", "ms_json_golden", "val", 21, DET_APPS + HIGH_N_CROP_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("High-N 5/16", "hul_sessions", "test", 25, DET_APPS + HIGH_N_CROP_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("High-N 6/16", "hul_sessions", "val", 20, DET_APPS + HIGH_N_CROP_APPS + E2E_APPS + PLANO_APPS, MODELS_HIGH_N, 4),
        ("High-N 7/16", "hul_golden", "test", 1000, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("High-N 8/16", "hul_golden", "val", 1000, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("High-N 9/16", "hul_packs", "test", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("High-N 10/16", "hul_packs", "val", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("High-N 11/16", "hul_brands", "test", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("High-N 12/16", "hul_brands", "val", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("High-N 13/16", "hul_variants", "test", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("High-N 14/16", "hul_variants", "val", 500, HIGH_N_CROP_APPS, MODELS_HIGH_N, 12),
        ("High-N 15/16", "hul_promos", "test", 500, PROMO_APPS, MODELS_HIGH_N, 12),
        ("High-N 16/16", "hul_promos", "val", 500, PROMO_APPS, MODELS_HIGH_N, 12),
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
