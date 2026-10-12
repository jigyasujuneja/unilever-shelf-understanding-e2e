"""End-to-End Ablation Studies (``detect_clef_flash``, ``detect_clm_contrastive``).

Kept in ``experiments/ablations/`` for reproducibility and Vertex AI Experiments lineage.
See ``experiments/ablations/__init__.py`` for the per-image root-cause postmortem explaining
why ``detect_l8_ultimate`` (0.9429 / 0.9687 F2) outperformed these two E2E ablations.
"""

from __future__ import annotations

from approaches.base import compose
from approaches.market_share.end_to_end.detect_identify import GeminiBoxDetector
from approaches.market_share.retrieval.clm_info_nce import ClmContrastive
from approaches.market_share.retrieval.embedding_retrieval import MODEL
from experiments.ablations.retrieval_ablations import ClefFlashDecision

DetectClefFlash = compose(
    "detect_clef_flash",
    GeminiBoxDetector,
    ClefFlashDecision,
    dataset="rpc",
    architecture=(
        "Gemini detects boxes -> Stage A Coarse JEV triage -> Stage B Clef-Flash (/v1/systemone) "
        "Typed Single-Token Decision Head"
    ),
    steps=[
        "Setup: embed reference gallery, CIELAB zones & SKU family centroids",
        "Stage 1: One Gemini call on the photo returns every product box",
        "Stage A: Coarse JEV family triage accepts distinct-family crops in 0 VLM calls",
        "Stage B: Clef-Flash typed decision head + Bottom-30% Zoom Strip resolves sister SKUs",
    ],
)

DetectClmContrastive = compose(
    "detect_clm_contrastive",
    GeminiBoxDetector,
    ClmContrastive,
    dataset="rpc",
    architecture=(
        f"Gemini detects boxes -> CLM (Contrastive-LM) InfoNCE Multi-View Scorer ({MODEL}) "
        f"-> Gemini 3.5 contact-sheet fallback"
    ),
    steps=[
        "Setup: embed multi-view gallery, SKU centroids & inner-80% CIELAB descriptors",
        "Stage 1: One Gemini call on the photo returns every product box",
        "Stage A: CLM InfoNCE softmax posterior P_CLM(y|x) resolves confident crops in 0 VLM calls",
        "Stage B: Batched contact sheets resolve high-entropy crops",
    ],
)
