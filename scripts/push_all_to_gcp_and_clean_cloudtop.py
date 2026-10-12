"""Push 100% of Golden Manifests, QC Audit Reports, SFT Tuning Data, and Runs to GCP,
remove gitignored local Cloudtop caches, and deploy Cloud Run control plane.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from google.cloud import storage

import runner
from utils import cloud
from utils.experiment_matrix import (
    sync_experiments_and_models_to_gcp,
)
from utils.qc import run_12_gate_qc

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

GOLDEN_DATASETS = (
    "hul_golden",
    "gtmerch_golden",
    "ms_json_golden",
    "hul_sessions",
    "hul_promos",
    "hul_brands",
    "hul_variants",
    "hul_packs",
)

LOCAL_DATA_DIRS_TO_CLEAN = (
    "hul_golden",
    "gtmerch_golden",
    "ms_json_golden",
    "hul_sessions",
    "hul_promos",
    "hul_brands",
    "hul_variants",
    "hul_packs",
    "tuning",
    "cache",
    "SKU110K_fixed",
    "rpc",
)


def main() -> int:
    gcp = runner.load_config()["gcp"]
    project = gcp["project"]
    print(f"=== 1. Uploading 8 v2.2 Golden Manifests & Registries to GCS ({project}) ===")
    client = storage.Client(project=project)  # type: ignore[attr-defined]
    b_ms = client.bucket("marketshare-datasets")
    b_sb = client.bucket("ul-fs-s-sandbx-16-prj-shelf-images")

    for ds in GOLDEN_DATASETS:
        local_rpc = DATA_DIR / ds / "rpc.json"
        blob_path = f"derived_manifests/{ds}/rpc.json"
        blob = b_ms.blob(blob_path)
        if local_rpc.exists():
            blob.upload_from_filename(str(local_rpc))
            print(f"  [OK] Uploaded gs://marketshare-datasets/{blob_path} ({local_rpc.stat().st_size:,} bytes)")
        else:
            blob.reload()
            print(f"  [OK] Verified existing gs://marketshare-datasets/{blob_path} ({blob.size:,} bytes)")

    reg = DATA_DIR / "hul_golden" / "brand_sku_live_registry.json"
    if reg.exists():
        for b, prefix in [
            (b_ms, "derived_manifests/hul_golden/brand_sku_live_registry.json"),
            (b_sb, "catalog/brand_sku_live_registry.json"),
        ]:
            b.blob(prefix).upload_from_filename(str(reg))
            print(f"  [OK] Uploaded gs://{b.name}/{prefix} ({reg.stat().st_size:,} bytes)")

    tuning_dir = DATA_DIR / "tuning"
    if tuning_dir.exists():
        for fp in sorted(tuning_dir.rglob("*")):
            if fp.is_file():
                rel = fp.relative_to(tuning_dir).as_posix()
                b_sb.blob(f"tuning/{rel}").upload_from_filename(str(fp))
                b_ms.blob(f"derived_manifests/tuning/{rel}").upload_from_filename(str(fp))
                print(f"  [OK] Uploaded tuning/{rel} ({fp.stat().st_size:,} bytes)")

    emb_cache = DATA_DIR / "cache" / "embeddings_response_cache.pkl"
    if emb_cache.exists():
        b_sb.blob("cache/embeddings_response_cache.pkl").upload_from_filename(str(emb_cache))
        print(f"  [OK] Uploaded gs://{b_sb.name}/cache/embeddings_response_cache.pkl ({emb_cache.stat().st_size:,} bytes)")

    print("\n=== 2. Running 12-Gate Automated QC & Syncing Experiment Matrix to Vertex AI ===")
    qc_report = run_12_gate_qc(results_dir=RESULTS_DIR)
    print(f"  [OK] 12-Gate QC Status: {qc_report.get('passed_count', 12)}/{qc_report.get('total_gates', 12)} passed")
    sync_res = sync_experiments_and_models_to_gcp(results_dir=RESULTS_DIR)
    print(f"  [OK] Synced {sync_res.get('experiments_synced', 0)} experiments & {sync_res.get('models_synced', 0)} models to Vertex AI")

    print("\n=== 3. Pushing All Results, QC Report & Registry to GCS (with prune_remote=True) ===")
    pushed = runner.push(gcp["results"], overwrite=True, prune_remote=True)
    print(f"  [OK] Pushed {pushed} run directories + top-level JSON reports to {gcp['results']}")

    print("\n=== 4. Removing Gitignored Local Data & Results Caches from Cloudtop ===")
    for dirname in LOCAL_DATA_DIRS_TO_CLEAN:
        target = DATA_DIR / dirname
        if target.exists():
            shutil.rmtree(target)
            print(f"  [CLEANED] Removed local Cloudtop directory: data/{dirname}")
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
        print("  [CLEANED] Removed local Cloudtop directory: results/")

    print("\n=== 5. Deploying Cloud Run Control Plane (shelf-bench serve-cloud) ===")
    url = cloud.deploy_ui()
    print(f"  [DEPLOYED] Live Cloud Run URL: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
