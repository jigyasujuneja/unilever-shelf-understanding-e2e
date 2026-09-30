"""Modular tests for Stage 5 (`src/stages/stage5_compound_vlm.py`) on real shelf images."""

from __future__ import annotations

import pytest
from conftest import RealShelfFixture, load_real_pack_image

from stages import stage5_compound_vlm


def test_stage5_real_cielab_delta_e_and_systemone_tiebreaker(
    shelf_fixture_a: RealShelfFixture,
) -> None:
    pack_a = load_real_pack_image("labeled_sku_0098_dove.jpg")
    pack_b = load_real_pack_image("labeled_sku_0102_sunsilk.jpg")
    candidates = ["UL-DOVE-BW-500ML", "UL-TRES-KR-340ML", "UL-SUNS-BL-180ML"]

    res_a = stage5_compound_vlm.run_sister_shade_tiebreaker(
        box_xyxy=[10.0, 10.0, float(pack_a.width - 10), float(pack_a.height - 10)],
        candidate_skus=candidates,
        mode="cielab_delta_e_and_systemone",
        image=pack_a,
    )
    res_b = stage5_compound_vlm.run_sister_shade_tiebreaker(
        box_xyxy=[10.0, 10.0, float(pack_b.width - 10), float(pack_b.height - 10)],
        candidate_skus=candidates,
        mode="cielab_delta_e_only",
        image=pack_b,
    )
    res_shelf = stage5_compound_vlm.run_sister_shade_tiebreaker(
        box_xyxy=list(shelf_fixture_a.boxes[0]),
        candidate_skus=candidates,
        mode="direct_vlm_only",
        image=shelf_fixture_a.image,
    )

    assert res_a["resolved_sku_id"] in candidates
    assert res_b["resolved_sku_id"] in candidates
    assert res_shelf["resolved_sku_id"] in candidates
    assert res_a["cielab_delta_e00"] != res_b["cielab_delta_e00"]
    assert res_b["vlm_invoked"] is False


def test_stage5_hard_fail_on_invalid_inputs_or_empty_candidates(
    shelf_fixture_a: RealShelfFixture,
) -> None:
    with pytest.raises(ValueError):
        stage5_compound_vlm.run_sister_shade_tiebreaker(
            box_xyxy=[90.0, 20.0, 30.0, 80.0],
            mode="cielab_delta_e_and_systemone",
            image=shelf_fixture_a.image,
        )
    with pytest.raises(ValueError):
        stage5_compound_vlm.run_sister_shade_tiebreaker(
            box_xyxy=list(shelf_fixture_a.boxes[0]),
            mode="unknown_tiebreaker_mode",
            image=shelf_fixture_a.image,
        )
