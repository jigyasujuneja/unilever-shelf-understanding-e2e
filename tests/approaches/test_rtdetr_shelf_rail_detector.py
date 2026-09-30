"""Modular tests for `rtdetr_shelf_rail_detector` (`src/approaches/rtdetr_shelf_rail_detector.py`)."""

from __future__ import annotations

import pytest
from conftest import RealShelfFixture, make_real_context
from PIL import Image

import approaches


def test_rtdetr_shelf_rail_detector_real_workflow_and_technique(
    shelf_fixture_a: RealShelfFixture,
    shelf_fixture_b: RealShelfFixture,
) -> None:
    det = approaches.get("rtdetr_shelf_rail_detector")
    det.setup({})

    ctx_a = make_real_context(model="none", trap_vlm=True)
    ctx_b = make_real_context(model="none", trap_vlm=True)

    boxes_a = det.detect(shelf_fixture_a.image, ctx_a)
    boxes_b = det.detect(shelf_fixture_b.image, ctx_b)

    assert len(boxes_a) > 0 and len(boxes_b) > 0
    assert boxes_a != boxes_b
    assert ctx_a.trace.usage.calls == 0 and ctx_b.trace.usage.calls == 0
    assert len(ctx_a.trace.steps) == 3
    assert "RT-DETR" in ctx_a.trace.steps[1]["name"]


def test_rtdetr_shelf_rail_detector_hard_fail_on_invalid_inputs() -> None:
    det = approaches.get("rtdetr_shelf_rail_detector")
    ctx = make_real_context(model="none", trap_vlm=True)
    with pytest.raises(ValueError):
        det.detect(None, ctx)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        det.detect(Image.new("RGB", (0, 0)), ctx)
