"""Dedicated real-data tests for `single_pass` (`SinglePass`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import CATEGORIES, NOT_PRODUCT, Context


def test_single_pass_real_image_detection_and_classification(
    real_shelf_sample_0: dict,
    real_shelf_sample_1: dict,
    real_signal_vlm,
) -> None:
    """Verify SinglePass detects and classifies real shelf images and drops NOT_PRODUCT labels."""
    app = load("single_pass", {})
    assert app.task == "detection"

    ctx0 = Context(llm=real_signal_vlm)
    boxes0 = app.detect(real_shelf_sample_0["image"], ctx0)
    assert len(boxes0) >= 8
    assert len(ctx0.trace.labels) == len(boxes0)
    for label in ctx0.trace.labels:
        assert label in CATEGORIES
        assert label != NOT_PRODUCT

    ctx1 = Context(llm=real_signal_vlm)
    boxes1 = app.detect(real_shelf_sample_1["image"], ctx1)
    assert len(boxes1) >= 8
    assert boxes0 != boxes1


def test_single_pass_hard_fails_on_invalid_image(real_signal_vlm) -> None:
    """Ensure SinglePass hard-fails when given None instead of a real PIL.Image."""
    app = load("single_pass", {})
    with pytest.raises(ValueError, match="Expected a valid PIL.Image.Image"):
        app.detect(None, Context(llm=real_signal_vlm))  # type: ignore[arg-type]
