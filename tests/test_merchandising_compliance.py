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


def test_wave2_wave3_wave4_approaches_register_cleanly():
    reg = approaches.all_approaches()
    expected = (
        "promo_asset_sku_detect",
        "promo_asset_bay_first_detect",
        "promo_asset_stage0_clahe_detect",
        "promo_asset_stage15_clutter_detect",
        "promo_asset_awp_router_detect",
        "promo_robotics_spatial_detect",
        "talker_price_tag_detect",
        "promo_ocr_zoom_verifier",
        "distilled_yolo_stage0_clahe",
        "shelf_stage0_clahe_e2e",
        "shelf_gemini_detect_e2e",
        "shelf_clef_metadata_e2e",
    )
    for name in expected:
        assert name in reg, f"{name} missing from registry"


def test_dual_buffer_photometry_preserves_linear_color_image():
    import numpy as np
    from PIL import Image

    from utils.photometry import apply_illuminant, dual_buffer_shelf_photometry

    arr = np.full((64, 64, 3), [215, 178, 190], dtype=np.uint8)
    arr[:32, :32] = [35, 28, 32]
    im = Image.fromarray(arr, mode="RGB")

    det_im, col_im, stats = dual_buffer_shelf_photometry(im)
    expected_col = apply_illuminant(im, stats["gains"])

    col_diff = float(np.abs(np.asarray(col_im, dtype=np.float32) - np.asarray(expected_col, dtype=np.float32)).max())
    det_diff = float(np.abs(np.asarray(det_im, dtype=np.float32) - np.asarray(col_im, dtype=np.float32)).mean())

    assert col_diff <= 1.0, f"color_image must remain strictly linear (got max diff {col_diff})"
    assert det_diff >= 1.0, f"detector_image must apply L* CLAHE contrast lift (got mean diff {det_diff})"


def test_bottom_plank_containment_rule_keeps_header_occluded_packs():
    from utils.metrics import _box_inside_any_asset_bay

    bay = (100.0, 200.0, 500.0, 800.0)
    # Top-row bottle: center y=190 is slightly above bay top y=200 (area overlap = 35/100 = 0.35 < 0.45),
    # but horizontal center cx=250 is inside [100, 500] and base y2=235 rests on top plank inside [200, 800]!
    top_row_bottle = (220.0, 135.0, 280.0, 235.0)
    assert _box_inside_any_asset_bay(top_row_bottle, [bay], min_containment=0.45) is True

    # Background store shelf bottle beside the bay (cx=60 outside [100, 500]) must still be rejected
    outside_bottle = (30.0, 135.0, 90.0, 235.0)
    assert _box_inside_any_asset_bay(outside_bottle, [bay], min_containment=0.45) is False


def test_stage15_clutter_filter_and_awp_hanger_router():
    import numpy as np
    from PIL import Image

    from approaches.merchandising.promo_detection.promo_asset_sku_detect import (
        filter_stage15_clutter,
        parse_awp_router_boxes,
    )

    im = Image.fromarray(np.full((1000, 1000, 3), 180, dtype=np.uint8), mode="RGB")
    bay = (100.0, 100.0, 900.0, 900.0)
    sku_boxes = [
        (300.0, 400.0, 420.0, 620.0),  # 3D pack inside bay -> KEEP
        (200.0, 630.0, 320.0, 655.0),  # flat shelf-rail price tag (120x25, aspect=4.8) -> DROP
        (220.0, 110.0, 680.0, 195.0),  # top header sign graphic (460x85, aspect=5.4, top 14%) -> DROP
        (102.0, 180.0, 145.0, 820.0),  # vertical wing pillar strip -> DROP
    ]
    kept, removed = filter_stage15_clutter(sku_boxes, [bay], im)
    assert kept == [(300.0, 400.0, 420.0, 620.0)]
    assert len(removed) == 3

    # AWP router must exclude frameless 'HAN' hanging strips from Class 0 asset_boxes
    raw_router = [
        {"box_2d": [100, 200, 850, 350], "label": "promo_asset", "fixture_type": "HAN"},
        {"box_2d": [200, 220, 320, 330], "label": "product_sku"},
    ]
    assets, ftypes, skus = parse_awp_router_boxes(raw_router, 1000.0, 1000.0)
    assert assets == []
    assert ftypes == []
    assert len(skus) == 1


def test_resolve_awp_planogram_spec_matches_lakme_window_bay():
    from utils.awp_template_index import resolve_awp_planogram_spec

    spec = resolve_awp_planogram_spec("PPLAKWINPROMOC10_store_12345.jpg")
    assert spec is not None
    assert spec["awp_code"] == "PPLAKWINPROMOC10"
    assert spec["fixture_type"] == "WIN"
    assert spec["required_brands"] == ["Lakme"]
    assert spec["forbid_competitors_in_bay"] is True

