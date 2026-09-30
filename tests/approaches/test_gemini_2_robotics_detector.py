"""Modular tests for `gemini_2_robotics_detector` (`src/approaches/gemini_2_robotics_detector.py`)."""

from __future__ import annotations

import pytest
from conftest import RealShelfFixture, make_real_context
from PIL import Image

import approaches


def test_gemini_2_robotics_detector_real_workflow_and_row_ordering(
    shelf_fixture_a: RealShelfFixture,
    shelf_fixture_b: RealShelfFixture,
) -> None:
    det = approaches.get("gemini_2_robotics_detector")
    det.setup({})

    ctx_a = make_real_context()
    ctx_b = make_real_context()
    boxes_a = det.detect(shelf_fixture_a.image, ctx_a)
    boxes_b = det.detect(shelf_fixture_b.image, ctx_b)

    assert len(boxes_a) > 0 and len(boxes_b) > 0
    assert boxes_a != boxes_b
    assert len(ctx_a.trace.steps) == 2


def test_gemini_2_robotics_detector_hard_fail_on_invalid_inputs() -> None:
    det = approaches.get("gemini_2_robotics_detector")
    ctx = make_real_context()
    with pytest.raises(ValueError):
        det.detect(None, ctx)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        det.detect(Image.new("RGB", (0, 0)), ctx)
