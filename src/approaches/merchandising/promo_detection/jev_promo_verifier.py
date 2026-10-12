"""Track 2 Merchandising: Promo, Multipack & Price-Callout Verifier (`clm_promo_matcher`, `jev_promo_verifier`).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Use Case: Merchandising -> Promo & Price Detection (W9, hul_promos)       │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  1. clm_promo_matcher (0 VLM calls):                                       │
  │     • Multi-view + SKU centroid + inner-80% CIELAB + top/bottom promo-band │
  │       colour signature matching for B1G1, B3G1, PO2, PO3, PO4, Rs10 SKUs   │
  │  2. jev_promo_verifier (CLM Gate + CLEF Flash Promo-Band Zoom Verifier):   │
  │     • Stage A: Auto-accepts clear promotional packs via CLM InfoNCE gate   │
  │     • Stage B: Escalates sister promo vs non-promo packs to CLEF Flash     │
  │       Zoom-Strip contact sheets with top-25% + bottom-30% callout bands    │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from approaches.base import register
from approaches.market_share.retrieval.clm_info_nce import L8HulUltimate
from approaches.market_share.retrieval.clm_standalone import ClmStandaloneRetrieval
from approaches.market_share.retrieval.embedding_retrieval import MODEL
from utils.embeddings import MODELS


@register
class ClmPromoMatcher(ClmStandaloneRetrieval):
    """Zero-VLM CLM InfoNCE matcher for W9 Promotional, Multipack & Price-Tag SKUs."""

    name = "clm_promo_matcher"
    use_case = "merchandising"
    task = "promo_detection"
    dataset = "hul_promos"
    models = MODELS
    architecture = (
        f"Zero-VLM Promo Matcher: CLM InfoNCE ({MODEL} multi-view + SKU centroid + "
        "inner-80% CIELAB promo-band energy, 0 Gemini calls)"
    )
    steps = [
        "Setup: embed promotional SKU reference gallery views, centroids & inner-80% CIELAB descriptors",
        "Stage 0: Estimate global shelf illuminant (p=6) and normalize crop white balance",
        "Stage 1: Rank promotional SKU candidates by CLM InfoNCE energy (base_variant + promo_type)",
        "Return argmax promotional SKU in 0 VLM calls",
    ]


@register
class JevPromoVerifier(L8HulUltimate):
    """CLM InfoNCE Gate + JEV/CLEF Promo-Band Contact-Sheet Verifier for W9 Promo Compliance."""

    name = "jev_promo_verifier"
    use_case = "merchandising"
    task = "promo_detection"
    dataset = "hul_promos"
    architecture = (
        f"CLM InfoNCE Promo Gate ({MODEL}) -> JEV/CLEF Promo-Band Zoom Contact-Sheet Verifier "
        "(verifies B1G1/B3G1/PO3/Rs10/Free callouts)"
    )
    steps = [
        "Setup: embed promotional SKU reference gallery views, centroids & inner-80% CIELAB descriptors",
        "Stage 0: Estimate global shelf illuminant (p=6) and normalize crop white balance",
        "Stage A: Auto-accept promotional SKUs passing CLM InfoNCE posterior & Z-score margin gate (0 VLM calls)",
        "Stage B: Escalate sister promo vs base variants to Gemini contact sheets with bottom-30% zoom strips",
    ]
