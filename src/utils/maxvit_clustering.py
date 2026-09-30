"""Backward-compatible facade re-exporting from `src/core/{models,features,clustering,catalog}` (`src/utils/maxvit_clustering.py`)."""

from __future__ import annotations

from core.catalog import load_dynamic_hul_catalog_index
from core.clustering import (
    ClusteringSummary,
    ShelfFacingCluster,
    cluster_shelf_facings_high_purity,
)
from core.features import (
    _delta_e_cie76,
    _extract_subroi_from_array,
    _rgb_to_cielab_fast,
    _run_maxvit_t_latent_64d,
    extract_gemini_subroi_embedding,
    extract_maxvit_multiscale_features,
)
from core.models import (
    load_efficientnet_b4_backbone,
    load_maxvit_t_backbone,
    load_mobilesam_segmenter,
    load_owlv2_detector,
    load_rtdetr_detector,
    load_siglip_classifier,
    load_yolo26n_detector,
)

__all__ = [
    "ClusteringSummary",
    "ShelfFacingCluster",
    "_delta_e_cie76",
    "_extract_subroi_from_array",
    "_rgb_to_cielab_fast",
    "_run_maxvit_t_latent_64d",
    "cluster_shelf_facings_high_purity",
    "extract_gemini_subroi_embedding",
    "extract_maxvit_multiscale_features",
    "load_dynamic_hul_catalog_index",
    "load_efficientnet_b4_backbone",
    "load_maxvit_t_backbone",
    "load_mobilesam_segmenter",
    "load_owlv2_detector",
    "load_rtdetr_detector",
    "load_siglip_classifier",
    "load_yolo26n_detector",
]
