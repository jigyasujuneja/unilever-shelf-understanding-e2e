"""Pluggable pipeline stages (``src/stages/``).

Importing ``stages`` registers all six pipeline stage groups:
  1. ``rectifier``     (Stage 1: Image quality check and perspective rectification)
  2. ``post_detector`` (Stage 3: Post-detection box filtering and strip splitting)
  3. ``clusterer``     (Stage 3.5: Adjacent crop clustering and deduplication)
  4. ``retriever``     (Stage 4: Glare compensation and vector catalog retrieval)
  5. ``tiebreaker``    (Stage 5: Perceptual color and VLM variant disambiguation)
  6. ``shelf_metrics`` (Stage 6: Share-of-shelf, out-of-stock, and planogram metrics)
"""

from __future__ import annotations

from stages import (
    stage1_rectification,
    stage2_3_detection,
    stage3_5_clustering,
    stage4_retrieval,
    stage5_compound_vlm,
    stage6_gondola_kpis,
    stage6_shelf_metrics,
)
from stages.registry import (
    STAGE_REGISTRY,
    MicroStageSpec,
    StageSpec,
    all_stages,
    default_stage_config,
    get_stage,
    register_stage,
)

__all__ = [
    "STAGE_REGISTRY",
    "MicroStageSpec",
    "StageSpec",
    "all_stages",
    "default_stage_config",
    "get_stage",
    "register_stage",
    "stage1_rectification",
    "stage2_3_detection",
    "stage3_5_clustering",
    "stage4_retrieval",
    "stage5_compound_vlm",
    "stage6_gondola_kpis",
    "stage6_shelf_metrics",
]
