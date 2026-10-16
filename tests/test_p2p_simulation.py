"""Tests for P2P impact simulation (ROI_Pipeline.json, NR_Brand_Hair_Beverages.py, NR_final_PW2.py)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils import p2p_simulation  # noqa: E402


def test_p2p_weights_w1_formula_matches_nr_final_pw2() -> None:
    oral = p2p_simulation.P2PWeights.from_category_dict(
        p2p_simulation.LEGACY_CATEGORY_WEIGHTS["Oral_Care"]
    )
    # W1 = (68.87 * 46.25) / 100 = 31.8524
    assert abs(oral.W1_basepack - 31.8524) < 1e-3

    ours_oral = p2p_simulation.P2PWeights.from_category_dict(
        p2p_simulation.OURS_CATEGORY_WEIGHTS["Oral_Care"]
    )
    # W1 = (86.20 * 84.50) / 100 = 72.839
    assert abs(ours_oral.W1_basepack - 72.839) < 1e-3
    assert ours_oral.W1_basepack > 2.0 * oral.W1_basepack


def test_cosine_similarity_and_score_cal() -> None:
    v1 = {"OUT_1": 2.0, "OUT_2": 1.0}
    v2 = {"OUT_1": 2.0, "OUT_2": 1.0}
    assert abs(p2p_simulation.cosine_similarity(v1, v2) - 1.0) < 1e-6
    assert abs(p2p_simulation.score_cal(v1) - 0.5) < 1e-6  # 1 of 2 outlets has occ >= 2


def test_build_past_roi_rows_produces_exact_12_columns_and_scenarios() -> None:
    samples = [
        {
            "session_id": "msjson_HUL-431134D-P42003_800112364_2026-09-18_1",
            "labels": [
                {
                    "sku_id": 10,
                    "brand": "Pepsodent",
                    "variant": "Germicheck",
                    "base_variant": "Germicheck",
                    "category": "Oral_Care",
                    "packaging_type": "Carton",
                    "is_hul": True,
                    "is_others_catchall": False,
                },
                {
                    "sku_id": 999,
                    "brand": "Others",
                    "variant": "Others",
                    "base_variant": "Others",
                    "category": "Oral_Care",
                    "packaging_type": "Carton",
                    "is_hul": False,
                    "is_others_catchall": True,
                },
            ],
        }
    ]
    rows_leg = p2p_simulation.build_past_roi_rows_for_scenario(samples, "legacy_h5", seed=1)
    rows_oth = p2p_simulation.build_past_roi_rows_for_scenario(samples, "ours_others_recovered", seed=1)
    rows_enr = p2p_simulation.build_past_roi_rows_for_scenario(samples, "ours_full_shelf_enriched", seed=1)

    for r in rows_leg + rows_oth + rows_enr:
        assert tuple(r.keys()) == p2p_simulation.PAST_ROI_COLUMNS

    res_leg = p2p_simulation.evaluate_p2p_scenario(rows_leg, "legacy_h5")
    res_oth = p2p_simulation.evaluate_p2p_scenario(rows_oth, "ours_others_recovered")
    res_enr = p2p_simulation.evaluate_p2p_scenario(rows_enr, "ours_full_shelf_enriched")

    assert res_oth["usable_p2p_facings"] > res_leg["usable_p2p_facings"]
    assert res_enr["usable_p2p_facings"] > res_oth["usable_p2p_facings"]
