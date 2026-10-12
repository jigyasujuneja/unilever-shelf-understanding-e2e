"""Standalone End-to-End CLM and CLEF Pipelines (Zero JEV Coupling).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  1. detect_clm_standalone : GeminiBoxDetector -> ClmStandaloneRetrieval    │
  │     (Stage 1 Gemini Boxes + Stage 2 Pure 0-VLM CLM InfoNCE)                │
  │  2. shelf_clm_standalone  : DistilledYoloOnnx -> ClmStandaloneRetrieval    │
  │     (100% Zero-VLM E2E: Stage 1 ONNX YOLO + Stage 2 Pure CLM InfoNCE)      │
  │  3. detect_clef_standalone: GeminiBoxDetector -> ClefStandaloneRetrieval   │
  │     (Stage 1 Gemini Boxes + Stage 2 100% CLEF Zoom-Strip Decision Head)    │
  │  4. shelf_clef_standalone : DistilledYoloOnnx -> ClefStandaloneRetrieval   │
  │     (Stage 1 ONNX YOLO + Stage 2 100% CLEF Zoom-Strip Decision Head)       │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from approaches.base import compose
from approaches.market_share.detection.onnx_yolo_detector import (
    DistilledYolo960SoftNms,
    DistilledYoloOnnx,
)
from approaches.market_share.end_to_end.detect_identify import GeminiBoxDetector
from approaches.market_share.retrieval.clef_standalone import ClefStandaloneRetrieval
from approaches.market_share.retrieval.clm_info_nce import (
    HierarchicalClmJev,
    L8HulUltimate,
    OutletScopedClmJev,
    RowNeighborPriorClmJev,
    SessionOverlapDedupClmJev,
    ShelfHeightSizePriorClmJev,
    TubeMidbandJev,
)
from approaches.market_share.retrieval.clm_standalone import ClmStandaloneRetrieval
from approaches.market_share.retrieval.embedding_retrieval import MODEL
from utils.embeddings import MODELS

ONNX_MODEL_ID = (DistilledYoloOnnx.models or ["distilled-yolo11n-onnx"])[0]

DetectClmStandalone = compose(
    "detect_clm_standalone",
    GeminiBoxDetector,
    ClmStandaloneRetrieval,
    dataset="rpc",
    architecture=(
        f"Gemini detects boxes -> Standalone CLM InfoNCE ({MODEL} multi-view + SKU centroid + "
        "inner-80% CIELAB energy, 0 Stage-2 VLM calls)"
    ),
    steps=[
        "Setup: embed reference gallery views, SKU centroids & inner-80% CIELAB descriptors",
        "Stage 1: One Gemini call on the photo returns every product box",
        "Stage 2: Standalone CLM InfoNCE scores every box in 0 VLM calls (zero JEV coupling)",
    ],
)

ShelfClmStandalone = compose(
    "shelf_clm_standalone",
    DistilledYoloOnnx,
    ClmStandaloneRetrieval,
    dataset="rpc",
    models=MODELS,
    also_calls=[ONNX_MODEL_ID],
    architecture=(
        f"100% Zero-VLM E2E: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> "
        "Stage 2 Standalone CLM InfoNCE (0 Gemini calls total)"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery views, SKU centroids & inner-80% CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: Standalone CLM InfoNCE identifies all boxes in 0 VLM calls (zero JEV coupling)",
    ],
)

DetectClefStandalone = compose(
    "detect_clef_standalone",
    GeminiBoxDetector,
    ClefStandaloneRetrieval,
    dataset="rpc",
    architecture=(
        f"Gemini detects boxes -> Standalone CLEF ({MODEL} shortlist -> 100% CLEF Zoom-Strip "
        "Constrained Logprob Decision Head, zero JEV coupling)"
    ),
    steps=[
        "Setup: embed reference gallery photos",
        "Stage 1: One Gemini call on the photo returns every product box",
        "Stage 2: 100% of detected crops evaluated on CLEF Bottom-30% Zoom-Strip contact sheets",
    ],
)

ShelfClefStandalone = compose(
    "shelf_clef_standalone",
    DistilledYoloOnnx,
    ClefStandaloneRetrieval,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> Stage 2 Standalone CLEF "
        "(100% CLEF Zoom-Strip Constrained Logprob Decision Head, zero JEV coupling)"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery photos",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: 100% of detected crops evaluated on CLEF Bottom-30% Zoom-Strip contact sheets",
    ],
)

ShelfHierarchicalClmJev = compose(
    "shelf_hierarchical_clm_jev",
    DistilledYoloOnnx,
    HierarchicalClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> Stage 2 Hierarchical Brand-Gate (0 VLM) -> "
        "Allowed-ID CLM InfoNCE + Z-Score Gate -> Batched JEV Contact Sheets"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + build brand centroids, SKU centroids & inner-80% CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: Hierarchical Brand-Gate restricts allowed_ids -> CLM InfoNCE + JEV contact-sheet fallback",
    ],
)

ShelfOutletScopedClmJev = compose(
    "shelf_outlet_scoped_clm_jev",
    DistilledYoloOnnx,
    OutletScopedClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-20: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> Store-Segment & Scene Category "
        "Gallery Scoping -> CLM InfoNCE + Batched JEV Contact Sheets"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + build category centroids, SKU centroids & inner-80% CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: Store-Segment & dominant scene category prior prunes gallery -> CLM InfoNCE + JEV fallback",
    ],
)

ShelfTubeMidbandJev = compose(
    "shelf_tube_midband_jev",
    DistilledYoloOnnx,
    TubeMidbandJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-21: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> CLM InfoNCE -> Middle-30% "
        "Cylindrical Squeeze-Tube Unwarped Variant Strip + Bottom-30% CLEF Weight Strip JEV"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery views, SKU centroids & CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: CLM InfoNCE + Middle-30% Unwarped Variant Strip + Bottom-30% CLEF Weight Strip JEV",
    ],
)

ShelfRowNeighborPrior = compose(
    "shelf_row_neighbor_prior",
    DistilledYoloOnnx,
    RowNeighborPriorClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-22: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> CLM InfoNCE + JEV -> "
        "1D Horizontal Shelf-Row Neighbor Brand/Category Consistency Prior"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery views, SKU centroids & CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: CLM InfoNCE + JEV + 1D Horizontal Shelf-Row Neighbor Category Prior",
    ],
)

ShelfHeightSizePrior = compose(
    "shelf_height_size_prior",
    DistilledYoloOnnx,
    ShelfHeightSizePriorClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-23: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> CLM InfoNCE + JEV -> "
        "Shelf-Plank Relative Box-Height Physical Size Prior (h_box / h_row)"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery views, SKU centroids & CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: CLM InfoNCE + JEV + Shelf-Plank Relative Box-Height Physical Size Prior",
    ],
)

Shelf960SoftNmsE2E = compose(
    "shelf_960_softnms_e2e",
    DistilledYolo960SoftNms,
    L8HulUltimate,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-24: Stage 1 960px High-Res + Gaussian Soft-NMS ONNX Detector ({ONNX_MODEL_ID}) -> "
        "Stage 2 Gray-World CLM InfoNCE + Batched JEV Contact Sheets"
    ),
    steps=[
        "Setup: load 960px Soft-NMS ONNX detector + embed reference gallery views, SKU centroids & CIELAB",
        "Stage 1: 960px High-Res + Aspect-Aware Gaussian Soft-NMS ONNX detection ($0 token cost)",
        "Stage 2: Gray-World CLM InfoNCE + Batched JEV Contact Sheets",
    ],
)

SessionOverlapDedupE2E = compose(
    "session_overlap_dedup_e2e",
    DistilledYoloOnnx,
    SessionOverlapDedupClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-25: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> Stage 2 Multi-Photo Store-Visit "
        "Session Overlap Homography Deduplication + CLM InfoNCE + JEV"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery views, SKU centroids & CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: Multi-Photo Store-Visit Session Overlap Deduplication + CLM InfoNCE + JEV",
    ],
)


