"""Backward-compatible domain facade re-exporting from `src/core/` (`src/utils/hul_domain.py`).

All domain logic has been decomposed into single-responsibility modules under `src/core/`:
  - `core.catalog`: Canonical SKU catalog, real reference crop prototype bank, and anti-hallucination validator.
  - `core.models`: Thread-safe neural model registry (`YOLO26n`, `RT-DETR`, `MaxViT`, `EfficientNet`, `OWLv2`, `SigLIP`, `SAM`).
  - `core.imaging`: Specular glare Telea inpainting (`IJEPASpecularGlarePredictor`) and contact sheet builder.
  - `core.features`: 64-D visual crop descriptor, sub-ROI extraction, and CIELAB `compute_ciede2000_approx`.
  - `core.clustering`: Complete-linkage high-purity facing clustering and 1D row Markov smoothing.
  - `core.detection`: Pixel Sobel shelf-rail detection, container/depth-ghost suppression, and seam stitching.
  - `core.retrieval`: Explicit `ClassificationConfig` pipelines, ScaNN vector lookup, and VLM tie-breaking.
  - `core.analytics`: Real geometric Share-of-Shelf, Out-of-Stock void, and Brand-Block Purity evaluation.
"""

from __future__ import annotations

from core.analytics import (
    HULEndToEndShelfProcessor,
    compute_geometric_shelf_kpis,
    compute_hul_7dim_and_gondola_summary,
    compute_modern_trade_gondola_kpis,
    evaluate_shelf_metrics,
    evaluate_shelf_summary,
)
from core.catalog import (
    CANONICAL_7DIM_CATALOG,
    CANONICAL_BY_CODE,
    _build_real_catalog_prototype_bank,
    build_real_catalog_prototype_bank,
    validate_canonical_7dim_prediction,
)
from core.clustering import smooth_shelf_row_predictions
from core.detection import (
    _merge_seam_split_boxes,
    _suppress_container_boxes,
    deduplicate_depth_stacked_facings,
    detect_shelf_boxes_from_pixels,
    propose_rtdetr_shelf_boxes,
)
from core.features import (
    _cosine_sim,
    _rgb_to_cielab,
    compute_ciede2000_approx,
    extract_real_crop_features,
)
from core.imaging import (
    _CONTACT_SHEET_CLASSIFY_SCHEMA,
    GlareCompensationResult,
    IJEPASpecularGlarePredictor,
    _build_contact_sheet,
)
from core.retrieval import (
    ClassificationConfig,
    DjevSystemOneClient,
    SisterCandidateProfile,
    SisterShadeResolution,
    SystemOneResolution,
    ThreeTaskPrefilterResult,
    classify_shelf_boxes_7dim,
    disambiguate_sister_shade_roi,
    scann_vector_lookup,
    score_hierarchical_7dim_candidates,
)

__all__ = [
    "CANONICAL_7DIM_CATALOG",
    "CANONICAL_BY_CODE",
    "ClassificationConfig",
    "DjevSystemOneClient",
    "GlareCompensationResult",
    "HULEndToEndShelfProcessor",
    "IJEPASpecularGlarePredictor",
    "SisterCandidateProfile",
    "SisterShadeResolution",
    "SystemOneResolution",
    "ThreeTaskPrefilterResult",
    "_CONTACT_SHEET_CLASSIFY_SCHEMA",
    "_build_contact_sheet",
    "_build_real_catalog_prototype_bank",
    "_cosine_sim",
    "_merge_seam_split_boxes",
    "_rgb_to_cielab",
    "_suppress_container_boxes",
    "build_real_catalog_prototype_bank",
    "classify_shelf_boxes_7dim",
    "compute_ciede2000_approx",
    "compute_geometric_shelf_kpis",
    "compute_hul_7dim_and_gondola_summary",
    "compute_modern_trade_gondola_kpis",
    "deduplicate_depth_stacked_facings",
    "detect_shelf_boxes_from_pixels",
    "disambiguate_sister_shade_roi",
    "evaluate_shelf_metrics",
    "evaluate_shelf_summary",
    "extract_real_crop_features",
    "propose_rtdetr_shelf_boxes",
    "scann_vector_lookup",
    "score_hierarchical_7dim_candidates",
    "smooth_shelf_row_predictions",
    "validate_canonical_7dim_prediction",
]
