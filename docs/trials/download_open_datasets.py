"""Unified Open-Source Dataset Loader (`SKU-110k` CVPR19 + `Smart-Retail-Shelf-Auditing-v1`).

Delegates directly to `scripts/download_and_ingest_real_datasets.py` so every test and
benchmarking engine (`ShelfBench Arena`, `Track A-D`) runs against the 25 REAL open-source
retail shelf photographs (`3,649` human-annotated product bounding boxes).
"""

from __future__ import annotations

from pathlib import Path

from scripts.download_and_ingest_real_datasets import main as ingest_real_25_images


def build_sku110k_rpc_benchmark_slice() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    slice_path = repo_root / "data" / "sku110k" / "sku110k_benchmark_slice.json"
    if not slice_path.exists() or slice_path.stat().st_size < 100_000:
        ingest_real_25_images()
    return slice_path


if __name__ == "__main__":
    build_sku110k_rpc_benchmark_slice()
