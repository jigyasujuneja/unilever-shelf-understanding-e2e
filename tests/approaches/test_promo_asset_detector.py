"""Modular tests for `promo_asset_detector` (`src/approaches/promo_asset_detector.py`)."""

from __future__ import annotations

import pytest
from conftest import RealShelfFixture, make_real_context
from PIL import Image

import approaches


def test_promo_asset_detector_real_workflow_and_aspect_filter(
    shelf_fixture_a: RealShelfFixture,
    smart_retail_fixture: RealShelfFixture,
) -> None:
    det = approaches.get("promo_asset_detector")
    det.setup({})

    ctx_a = make_real_context()
    ctx_sr = make_real_context()
    boxes_a = det.detect(shelf_fixture_a.image, ctx_a)
    boxes_sr = det.detect(smart_retail_fixture.image, ctx_sr)

    assert len(boxes_a) > 0 and len(boxes_sr) > 0
    assert boxes_a != boxes_sr
    assert len(ctx_a.trace.steps) == 2


def test_promo_asset_detector_hard_fail_on_invalid_inputs() -> None:
    det = approaches.get("promo_asset_detector")
    ctx = make_real_context()
    with pytest.raises(ValueError):
        det.detect(None, ctx)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        det.detect(Image.new("RGB", (0, 0)), ctx)
