"""Unit tests for Hierarchical Routing (`hierarchical_clm_jev`) and Track 2 Merchandising approaches."""

from __future__ import annotations

import approaches
from approaches.merchandising.planogram_compliance.bay_planogram_verifier import (
    compute_planogram_metrics,
)


def test_merchandising_and_hierarchical_approaches_register_cleanly():
    reg = approaches.all_approaches()
    for name in (
        "hierarchical_clm_jev",
        "shelf_hierarchical_clm_jev",
        "clm_promo_matcher",
        "jev_promo_verifier",
        "bay_planogram_verifier",
    ):
        assert name in reg, f"{name} missing from registry"

    assert reg["clm_promo_matcher"].use_case == "merchandising"
    assert reg["clm_promo_matcher"].task == "promo_detection"
    assert reg["clm_promo_matcher"].dataset == "hul_promos"

    assert reg["jev_promo_verifier"].use_case == "merchandising"
    assert reg["jev_promo_verifier"].task == "promo_detection"
    assert reg["jev_promo_verifier"].dataset == "hul_promos"

    assert reg["bay_planogram_verifier"].use_case == "merchandising"
    assert reg["bay_planogram_verifier"].task == "planogram_compliance"
    assert reg["bay_planogram_verifier"].dataset == "hul_sessions"


def test_compute_planogram_metrics_measures_sos_and_brand_block_purity():
    catalog = {
        1: {"sku_id": 1, "brand": "Dove", "product": "Dove__Bar", "is_hul": True},
        2: {"sku_id": 2, "brand": "Lux", "product": "Lux__Bar", "is_hul": True},
        3: {"sku_id": 3, "brand": "Colgate", "product": "Colgate", "is_hul": False},
    }
    boxes = [
        (0.0, 0.0, 100.0, 100.0),
        (100.0, 0.0, 200.0, 100.0),
        (200.0, 0.0, 300.0, 100.0),
        (300.0, 0.0, 400.0, 100.0),
    ]
    # Contiguous blocks: [Dove, Dove, Lux, Colgate] -> 0 extra transitions -> purity 1.0, HUL SOS = 3/4 = 0.75
    m_clean = compute_planogram_metrics(boxes, [1, 1, 2, 3], catalog)
    assert m_clean["hul_facing_sos"] == 0.75
    assert m_clean["brand_block_purity"] == 1.0
    assert m_clean["distinct_brands"] == 3

    # Interleaved blocks: [Dove, Lux, Dove, Colgate] -> 3 transitions for 3 brands (1 extra) -> purity < 1.0
    m_interleaved = compute_planogram_metrics(boxes, [1, 2, 1, 3], catalog)
    assert m_interleaved["hul_facing_sos"] == 0.75
    assert m_interleaved["brand_block_purity"] < 1.0

    # Adversarial all-None abstained predictions must NEVER yield 100% HUL SOS or 100% brand-block purity
    m_abstained = compute_planogram_metrics(boxes, [None, None, None, None], catalog)
    assert m_abstained["hul_facing_sos"] == 0.0
    assert m_abstained["brand_block_purity"] == 0.0
    assert m_abstained["unidentified_facings"] == 4

    # Class 0 display asset bay compliance: Colgate (id=3, is_hul=False) inside HUL bay flags 1 competitor intrusion
    asset_bays = [(0.0, 0.0, 400.0, 100.0)]
    m_bay = compute_planogram_metrics(boxes, [1, 1, 2, 3], catalog, asset_boxes=asset_bays)
    assert m_bay["asset_bays"] == 1
    assert m_bay["competitor_intrusion_count"] == 1
    assert m_bay["asset_bay_purity"] == 0.75
