"""Modular tests for Stage 1 (`src/stages/stage1_rectification.py`) on real shelf images."""

from __future__ import annotations

import pytest
from conftest import RealShelfFixture
from PIL import Image

from stages import stage1_rectification


def test_stage1_real_workflow_and_input_sensitivity(
    shelf_fixture_a: RealShelfFixture,
    shelf_fixture_b: RealShelfFixture,
) -> None:
    res_a = stage1_rectification.run_rectification(
        shelf_fixture_a.image, shelf_fixture_a.boxes, mode="hough_rail_homography"
    )
    tilted_a = shelf_fixture_a.image.rotate(5.5, fillcolor=(230, 230, 230))
    res_tilted = stage1_rectification.run_rectification(
        tilted_a, shelf_fixture_a.boxes, mode="hough_rail_homography"
    )
    res_depth = stage1_rectification.run_rectification(
        shelf_fixture_b.image, shelf_fixture_b.boxes, mode="depth_anything_v2"
    )
    res_none = stage1_rectification.run_rectification(
        shelf_fixture_a.image, shelf_fixture_a.boxes, mode="none"
    )

    assert isinstance(res_a["rectified_image"], Image.Image)
    assert len(res_a["rectified_boxes"]) == len(shelf_fixture_a.boxes)
    assert res_a["yaw_corrected_deg"] != 4.2, "Must not return static hardcoded 4.2 yaw"
    assert res_a["yaw_corrected_deg"] != res_tilted["yaw_corrected_deg"]
    assert res_depth["mode"] == "depth_anything_v2"
    assert 0.85 <= res_depth["aspect_compensation_factor"] <= 1.18
    assert res_none["homography_applied"] is False and res_none["yaw_corrected_deg"] == 0.0


def test_stage1_hard_fail_on_invalid_inputs(shelf_fixture_a: RealShelfFixture) -> None:
    with pytest.raises(ValueError):
        stage1_rectification.run_rectification(None, shelf_fixture_a.boxes, mode="hough_rail_homography")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        stage1_rectification.run_rectification(Image.new("RGB", (0, 0)), shelf_fixture_a.boxes)
    with pytest.raises(ValueError):
        stage1_rectification.run_rectification(shelf_fixture_a.image, shelf_fixture_a.boxes, mode="invalid_mode")
