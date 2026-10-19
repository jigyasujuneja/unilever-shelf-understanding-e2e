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

from PIL import Image

from approaches.base import Box, Context, compose, register
from approaches.market_share.detection.onnx_yolo_detector import (
    DistilledYolo960SoftNms,
    DistilledYoloOnnx,
)
from approaches.market_share.detection.single_pass_dedup import SinglePassDedup
from approaches.market_share.end_to_end.detect_identify import GeminiBoxDetector
from approaches.market_share.retrieval.clef_standalone import (
    ClefContrastiveMetadata,
    ClefStandaloneRetrieval,
)
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
from utils.photometry import dual_buffer_shelf_photometry

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


@register
class DistilledYoloStage0Clahe(DistilledYoloOnnx):
    """EXP-49a Stage 0 + Stage 1: Dual-Buffer CLAHE Photometry -> Distilled ONNX YOLO."""

    name = "distilled_yolo_stage0_clahe"
    architecture = (
        f"EXP-49a: Stage 0 Dual-Buffer Photometry (CIELAB L* CLAHE) -> "
        f"Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID})"
    )
    steps = [
        "Setup: load Vertex AI Model Registry ONNX graphs into CPUExecutionProvider",
        "Stage 0: Build dual photometric buffers (CIELAB L* CLAHE detector_image + linear Shades-of-Gray color_image)",
        "Stage 1: Run dual-head ONNX CPU detection on detector_image while preserving linear color_image for Stage 2",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        det_im, color_im, stats = dual_buffer_shelf_photometry(image)
        ctx.detector_image = det_im
        ctx.color_image = color_im
        ctx.trace.step(
            "Stage 0 Dual-Buffer Photometry",
            f"Shades-of-Gray gains={stats['gains']}, CIELAB L* CLAHE std {stats['l_std_before']:.1f} -> {stats['l_std_after']:.1f}",
            info=stats,
        )
        return super().detect(det_im, ctx)


ShelfStage0ClaheE2E = compose(
    "shelf_stage0_clahe_e2e",
    DistilledYoloStage0Clahe,
    HierarchicalClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-49a (Stage 0 Swap): Stage 0 Dual-Buffer CLAHE -> Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) "
        "-> Stage 2 Hierarchical Brand-Gate CLM InfoNCE + Batched JEV Contact Sheets (linear color_image)"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + build brand centroids, SKU centroids & inner-80% CIELAB",
        "Stage 0: Build CIELAB L* CLAHE detector_image and linear Shades-of-Gray color_image",
        "Stage 1: Single-pass ONNX YOLO detects shelf boxes on detector_image (~40ms, 0 VLM calls)",
        "Stage 2: Hierarchical Brand-Gate + CLM InfoNCE + JEV contact-sheet fallback on linear color_image crops",
    ],
)

ShelfGeminiDetectE2E = compose(
    "shelf_gemini_detect_e2e",
    SinglePassDedup,
    HierarchicalClmJev,
    dataset="rpc",
    also_calls=[MODEL],
    architecture=(
        "EXP-49b (Stage 1 Swap): Stage 1 Gemini single_pass_dedup -> Stage 2 Hierarchical "
        "Brand-Gate CLM InfoNCE + Batched JEV Contact Sheets"
    ),
    steps=[
        "Setup: build brand centroids, SKU centroids & inner-80% CIELAB descriptors",
        "Stage 1: Gemini single_pass_dedup detects all shelf product boxes + container/NMS/ghost filters",
        "Stage 2: Hierarchical Brand-Gate restricts allowed_ids -> CLM InfoNCE + JEV contact-sheet fallback",
    ],
)

ShelfClefMetadataE2E = compose(
    "shelf_clef_metadata_e2e",
    DistilledYoloOnnx,
    ClefContrastiveMetadata,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-49c (Stage 2 Swap): Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> "
        "Stage 2 Contrastive Metadata CLEF Zoom-Strip Contact Sheets + 2-Tier Open-Set Guard"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery photos and load SKU catalog metadata",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: CLM InfoNCE fast gate -> Contrastive Metadata CLEF Bottom-30% Zoom-Strip Contact Sheets",
    ],
)



