"""Modular End-to-End Shelf Detection and Identification Pipelines (``detect_l8_ultimate``, ``shelf_l8_ultimate``, ``detect_onnx_jev_laya``).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Stage 1 (Detection):                                                      │
  │    • GeminiBoxDetector (RPC) / SinglePassDedup (Shelves)                   │
  │    • OR DistilledYoloOnnx ($0-token CPU ONNX detector in ~55 ms)           │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  Stage 0 (Photometry): Global Shelf Shades-of-Gray (p=6) Illuminant        │
  │  Stage 2a (Fast Gate): CLM InfoNCE Posterior + Z-Score/LID Margin Gate     │
  │  Stage 2b (Escalation): Medoid Clustering -> Batched Top-5 Contact Sheets  │
  │  Stage 2c (Post-Proc): Visually-Gated 1D Shelf-Row Continuity Smoothing    │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from approaches.base import compose
from approaches.market_share.detection.onnx_yolo_detector import DistilledYoloOnnx
from approaches.market_share.detection.single_pass_dedup import SinglePassDedup
from approaches.market_share.end_to_end.detect_identify import GeminiBoxDetector
from approaches.market_share.retrieval.clm_info_nce import L8HulUltimate
from approaches.market_share.retrieval.coarse_jev_laya import (
    COARSE_DISTINCT_MARGIN,
    CoarseJevLaya,
)
from approaches.market_share.retrieval.embedding_retrieval import MODEL
from approaches.market_share.retrieval.tiered_hybrid import K

DetectL8Ultimate = compose(
    "detect_l8_ultimate",
    GeminiBoxDetector,
    L8HulUltimate,
    dataset="rpc",
    architecture=(
        "Gemini detects boxes -> Global Shelf Illuminant (p=6) + Inner-80% CIELAB + "
        "CLM InfoNCE & Z-Score Gate + Batched Contact Sheets & Row Smoothing"
    ),
    steps=[
        "Setup: embed multi-view gallery, SKU centroids & inner-80% CIELAB + shoulder-taper descriptors",
        "Stage 1: One Gemini call on the photo returns every product box",
        "Stage 0: Estimate full-shelf Shades-of-Gray (p=6) store illuminant once and normalize crops",
        "Stage A: Intersect CLM InfoNCE posterior and Z-score normalized margin gate",
        "Stage B: Batched top-5 Gemini contact sheets & 1D shelf-row continuity smoothing",
    ],
)

ShelfL8Ultimate = compose(
    "shelf_l8_ultimate",
    SinglePassDedup,
    L8HulUltimate,
    dataset="shelves",
    architecture=(
        "single_pass_dedup boxes -> Global Shelf Illuminant (p=6) + Inner-80% CIELAB + "
        "CLM InfoNCE & Z-Score Gate + Batched Contact Sheets & Row Smoothing"
    ),
    steps=[
        "Setup: embed multi-view gallery, SKU centroids & inner-80% CIELAB + shoulder-taper descriptors",
        "Stage 1: single_pass_dedup returns shelf product boxes",
        "Stage 0: Estimate full-shelf Shades-of-Gray (p=6) store illuminant once and normalize crops",
        "Stage A: Intersect CLM InfoNCE posterior and Z-score normalized margin gate",
        "Stage B: Batched top-5 Gemini contact sheets & 1D shelf-row continuity smoothing",
    ],
)

DetectOnnxJevLaya = compose(
    "detect_onnx_jev_laya",
    DistilledYoloOnnx,
    CoarseJevLaya,
    dataset="rpc",
    architecture=(
        f"Distilled ONNX CPU detector ($0 token cost) -> Stage A Coarse JEV family triage ({MODEL}) "
        f"-> Stage B Gemini 3.8 batched top-{K} contact sheets"
    ),
    steps=[
        "Setup: load Vertex AI Model Registry ONNX graphs + embed reference gallery & SKU family centroids",
        "Stage 1: Distilled ONNX CPU detector returns product boxes in ~55 ms ($0 token cost)",
        f"Stage A: Coarse JEV family triage; accept when distinct-family margin >= {COARSE_DISTINCT_MARGIN}",
        f"Stage B: Cluster uncertain crops and verify medoids in batched top-{K} Gemini 3.8 contact sheets",
    ],
)

ShelfOnnxJevLaya = compose(
    "shelf_onnx_jev_laya",
    DistilledYoloOnnx,
    CoarseJevLaya,
    dataset="shelves",
    architecture=(
        f"Distilled ONNX CPU detector ($0 token cost) -> Stage A Coarse JEV family triage ({MODEL}) "
        f"-> Stage B Gemini 3.8 batched top-{K} contact sheets"
    ),
    steps=[
        "Setup: load Vertex AI Model Registry ONNX graphs + embed reference gallery & SKU family centroids",
        "Stage 1: Distilled ONNX CPU detector returns shelf product boxes in ~55 ms ($0 token cost)",
        f"Stage A: Coarse JEV family triage; accept when distinct-family margin >= {COARSE_DISTINCT_MARGIN}",
        f"Stage B: Cluster uncertain crops and verify medoids in batched top-{K} Gemini 3.8 contact sheets",
    ],
)
