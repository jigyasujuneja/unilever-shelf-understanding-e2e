"""Modular tests for Stage 3.5 (`src/stages/stage3_5_clustering.py`) on real shelf images."""

from __future__ import annotations

import pytest
from conftest import RealShelfFixture
from PIL import Image

from stages import stage3_5_clustering


def test_stage3_5_real_workflow_and_neural_maxvit_vs_subroi(
    shelf_fixture_a: RealShelfFixture,
    shelf_fixture_b: RealShelfFixture,
) -> None:
    res_cc = stage3_5_clustering.run_crop_clustering(
        shelf_fixture_a.boxes, mode="connected_component_adjacency", image=shelf_fixture_a.image
    )
    res_mv = stage3_5_clustering.run_crop_clustering(
        shelf_fixture_a.boxes, mode="maxvit_agglomerative", image=shelf_fixture_a.image
    )
    res_b = stage3_5_clustering.run_crop_clustering(
        shelf_fixture_b.boxes, mode="connected_component_adjacency", image=shelf_fixture_b.image
    )
    res_none = stage3_5_clustering.run_crop_clustering(
        shelf_fixture_a.boxes, mode="none", image=shelf_fixture_a.image
    )

    assert res_cc["input_crops"] == len(shelf_fixture_a.boxes)
    assert res_mv["input_crops"] == len(shelf_fixture_a.boxes)
    assert len(res_cc["crop_feats"]) == len(shelf_fixture_a.boxes)
    assert len(res_mv["crop_feats"]) == len(shelf_fixture_a.boxes)
    # Verify neural MaxViT-T embeddings differ from gemini_subroi embeddings and across images
    assert res_cc["crop_feats"][0]["embedding"] != res_mv["crop_feats"][0]["embedding"]
    assert res_cc["crop_feats"][0]["embedding"] != res_b["crop_feats"][0]["embedding"]
    assert res_none["medoid_calls"] == len(shelf_fixture_a.boxes)


def test_stage3_5_hard_fail_on_invalid_inputs(shelf_fixture_a: RealShelfFixture) -> None:
    with pytest.raises(ValueError):
        stage3_5_clustering.run_crop_clustering(
            shelf_fixture_a.boxes, mode="maxvit_agglomerative", image=Image.new("RGB", (0, 0))
        )
    with pytest.raises(ValueError):
        stage3_5_clustering.run_crop_clustering(
            shelf_fixture_a.boxes, mode="unknown_cluster_mode", image=shelf_fixture_a.image
        )
