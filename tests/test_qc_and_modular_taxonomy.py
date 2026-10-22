"""Unit tests for the modular taxonomy/manifests packages and the 12-Gate Automated QC Engine."""

from __future__ import annotations

import numpy as np

from utils.manifests.ms_json_loader import (
    dedup_shelf_boxes,
    has_burned_in_cv2_overlay,
    is_valid_product_crop,
)
from utils.qc import run_12_gate_qc
from utils.taxonomy import (
    CANONICAL_CATEGORIES,
    CANONICAL_PACKAGING_TYPES,
    normalize_brand_folder,
    normalize_category_name,
    normalize_packaging_type,
    parse_ms_json_shape_label,
)


def test_12_gate_qc_all_pass(tmp_path):
    report = run_12_gate_qc(results_dir=tmp_path, allow_offline_fixture=True)
    assert report["all_passed"] is True
    assert report["passed_gates"] == 15
    assert report["total_gates"] == 15
    for gate in report["gates"]:
        assert gate["passed"] is True, f"Gate {gate['gate_id']} ({gate['name']}) failed: {gate}"
        assert gate["violations_count"] == 0


def test_modular_taxonomy_zero_unknowns_and_typos():
    vwash = normalize_brand_folder("ppvwash")
    assert vwash is not None and vwash["brand"] == "V_Wash" and vwash["is_hul"] is True

    tetley = normalize_brand_folder("tetely")
    assert tetley is not None and tetley["brand"] == "Tetley" and tetley["is_hul"] is False

    twinings = normalize_brand_folder("twinnings")
    assert twinings is not None and twinings["brand"] == "Twinings" and twinings["is_hul"] is False

    cat = normalize_category_name("Unknown")
    assert cat in CANONICAL_CATEGORIES and cat != "Unknown"

    pkg = normalize_packaging_type("Unknown", category="Hair_Care", brand="Dove", variant="Shampoo")
    assert pkg in CANONICAL_PACKAGING_TYPES and pkg != "Unknown"


def test_ms_json_loader_overlay_and_crop_quarantine():
    clean_img = np.full((200, 200, 3), 220, dtype=np.uint8)
    shapes = [{"points": [[20, 20], [120, 140]], "label": "S-T-P-S-T_0.99_Dove____Dove_Shampoo____NA"}]
    assert has_burned_in_cv2_overlay(clean_img, shapes) is False

    # Burn in a red cv2.rectangle line along y=20
    burned_img = clean_img.copy()
    burned_img[20, 20:120] = [255, 0, 0]
    assert has_burned_in_cv2_overlay(burned_img, shapes) is True

    # Parse LabelMe shape label
    p_info = parse_ms_json_shape_label("S-T-P-S-T_0.99_Dove____Dove_Shampoo____NA")
    assert p_info["is_valid_format"] is True
    assert p_info["is_others_catchall"] is False
    assert p_info["brand"] == "Dove"

    # Dedup nested/overlapping boxes
    deduped = dedup_shelf_boxes([
        ([20.0, 20.0, 120.0, 140.0], p_info),
        ([22.0, 22.0, 118.0, 138.0], p_info),
    ])
    assert len(deduped) == 1
    # Valid product crop on a 1000x1000 shelf vs giant ceiling-appliance box near top edge
    assert is_valid_product_crop([20.0, 20.0, 120.0, 140.0], 1000, 1000, p_info, clean_img[20:140, 20:120]) is True
    assert is_valid_product_crop([20.0, 20.0, 400.0, 400.0], 1000, 1000, p_info, clean_img) is False

