"""Dedicated real-data tests for `detect_classify` (`DetectClassify`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import CATEGORIES, NOT_PRODUCT, Context
from approaches.detect_classify import CELL, COLS, ROWS, contact_sheet


def test_detect_classify_contact_sheet_and_two_pass_pipeline(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify DetectClassify builds real contact sheets and runs Pass 1 + Pass 2 on real shelf images."""
    sheet = contact_sheet(
        real_shelf_sample_0["image"], real_shelf_sample_0["gt_boxes"][:12], start=0
    )
    assert sheet.size == (COLS * CELL, ROWS * CELL)

    app = load("detect_classify", {})
    ctx = Context(llm=real_signal_vlm)
    boxes = app.detect(real_shelf_sample_0["image"], ctx)
    assert len(boxes) >= 8
    assert len(ctx.trace.labels) == len(boxes)
    assert len(ctx.trace.steps) == 2
    for label in ctx.trace.labels:
        assert label in CATEGORIES
        assert label != NOT_PRODUCT


def test_detect_classify_hard_fails_on_invalid_image(real_signal_vlm) -> None:
    """Ensure DetectClassify hard-fails when given None instead of a real PIL.Image."""
    app = load("detect_classify", {})
    with pytest.raises(ValueError, match="Expected a valid PIL.Image.Image"):
        app.detect(None, Context(llm=real_signal_vlm))  # type: ignore[arg-type]
