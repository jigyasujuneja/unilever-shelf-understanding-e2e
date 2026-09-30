"""Modular tests for Stage 2 & 3 (`src/stages/stage2_3_detection.py`) on real shelf images."""

from __future__ import annotations

import pytest
from conftest import RealShelfFixture
from PIL import ImageDraw

from stages import stage2_3_detection


def test_stage2_3_real_workflow_and_ladi_slicing(
    shelf_fixture_a: RealShelfFixture,
    shelf_fixture_b: RealShelfFixture,
) -> None:
    out_a = stage2_3_detection.run_post_detection(
        shelf_fixture_a.image, shelf_fixture_a.boxes, mode="oriented_ladi_slicer"
    )
    out_b = stage2_3_detection.run_post_detection(
        shelf_fixture_b.image, shelf_fixture_b.boxes, mode="shelf_rail_soft_nms"
    )
    out_std = stage2_3_detection.run_post_detection(
        shelf_fixture_a.image, shelf_fixture_a.boxes, mode="standard_nms"
    )
    assert len(out_a) > 0 and len(out_b) > 0 and len(out_std) > 0
    assert out_a != out_b

    # Verify vertical sachet strip slicing on a real shelf image with a tall hanging strip box
    ladi_img = shelf_fixture_a.image.copy()
    draw = ImageDraw.Draw(ladi_img)
    draw.rectangle([15, 20, 45, 230], fill=(195, 35, 40))
    for seal_y in (70, 120, 175):
        draw.line([(15, seal_y), (45, seal_y)], fill=(250, 250, 250), width=3)
    sliced = stage2_3_detection.run_post_detection(
        ladi_img, [(15.0, 20.0, 45.0, 230.0)], mode="oriented_ladi_slicer"
    )
    assert len(sliced) > 1


def test_stage2_3_hard_fail_on_invalid_inputs(shelf_fixture_a: RealShelfFixture) -> None:
    with pytest.raises(ValueError):
        stage2_3_detection.run_post_detection(None, shelf_fixture_a.boxes, mode="oriented_ladi_slicer")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        stage2_3_detection.run_post_detection(
            shelf_fixture_a.image, [(90.0, 20.0, 40.0, 80.0)], mode="oriented_ladi_slicer"
        )
    with pytest.raises(ValueError):
        stage2_3_detection.run_post_detection(shelf_fixture_a.image, shelf_fixture_a.boxes, mode="invalid_mode")
