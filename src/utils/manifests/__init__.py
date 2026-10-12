"""Modular Golden Manifest Builders & Verifiers Package."""

from __future__ import annotations

from utils.manifests.crop_verifier import verify_gtmerch_shelf_boxes
from utils.manifests.golden_builder import (
    compute_split_hashes,
    publish_clean_golden_manifests_to_gcs,
    sanitize_and_decollide_catalog,
    stratified_hul_non_hul_sample,
)
from utils.manifests.ms_json_loader import (
    build_ms_json_golden_manifest,
    dedup_shelf_boxes,
    has_burned_in_cv2_overlay,
    is_valid_product_crop,
)

__all__ = [
    "build_ms_json_golden_manifest",
    "compute_split_hashes",
    "dedup_shelf_boxes",
    "has_burned_in_cv2_overlay",
    "is_valid_product_crop",
    "publish_clean_golden_manifests_to_gcs",
    "sanitize_and_decollide_catalog",
    "stratified_hul_non_hul_sample",
    "verify_gtmerch_shelf_boxes",
]
