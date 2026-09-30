"""Modular tests for Stage 4 (`src/stages/stage4_retrieval.py`) on real shelf images."""

from __future__ import annotations

import pytest
from conftest import RealShelfFixture

from stages import stage4_retrieval


def test_stage4_real_workflow_telea_deglare_and_scann_lookup(
    shelf_fixture_a: RealShelfFixture,
    shelf_fixture_b: RealShelfFixture,
) -> None:
    res_ijepa = stage4_retrieval.run_retrieval_stage(
        predicted_brand="Dove",
        predicted_packaging="bottle",
        glare_intensity=0.25,
        mode="ijepa_scann_entropy_prefilter",
        image=shelf_fixture_a.image,
        boxes=shelf_fixture_a.boxes,
    )
    res_siglip = stage4_retrieval.run_retrieval_stage(
        predicted_brand="Sunsilk",
        predicted_packaging="bottle",
        glare_intensity=0.25,
        mode="siglip_multiprototype",
        image=shelf_fixture_b.image,
        boxes=shelf_fixture_b.boxes,
    )
    res_pure = stage4_retrieval.run_retrieval_stage(
        mode="pure_scann_cosine",
        image=shelf_fixture_a.image,
        boxes=shelf_fixture_a.boxes,
    )

    assert res_ijepa["ijepa_deglare_applied"] is True
    assert res_ijepa["scann_pool_after"] < res_ijepa["scann_pool_before"]
    assert res_ijepa["cosine_gain"] > 0.0
    assert res_siglip["mode"] == "siglip_multiprototype"
    assert res_pure["ijepa_deglare_applied"] is False and res_pure["cosine_gain"] == 0.0
    assert res_ijepa["top1_sim"] != res_siglip["top1_sim"]


def test_stage4_hard_fail_on_invalid_inputs(shelf_fixture_a: RealShelfFixture) -> None:
    with pytest.raises(ValueError):
        stage4_retrieval.run_retrieval_stage(glare_intensity=1.4, mode="ijepa_scann_entropy_prefilter")
    with pytest.raises(ValueError):
        stage4_retrieval.run_retrieval_stage(
            mode="ijepa_scann_entropy_prefilter",
            image=shelf_fixture_a.image,
            boxes=[(100.0, 50.0, 20.0, 90.0)],
        )
    with pytest.raises(ValueError):
        stage4_retrieval.run_retrieval_stage(mode="invalid_retriever_mode")
