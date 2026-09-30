"""Dedicated real-data tests for `track_a_cascading_vit` (`CascadingViTTrackA`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_track_a_cascading_vit_efficientnet_b4_and_maxvit_t_cascade(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify CascadingViTTrackA executes real torchvision EfficientNet-B4 (1792-D) -> MaxViT-T (512-D) cascade."""
    app = load("track_a_cascading_vit", {})
    assert app.task == "combined"

    ctx = Context(llm=real_signal_vlm)
    boxes = app.detect(real_shelf_sample_0["image"], ctx)
    assert len(boxes) >= 12
    assert "EfficientNet-B4 (1792D) -> MaxViT-T (512D)" in ctx.trace.steps[-1]["detail"]
    for pred in ctx.trace.labels:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_track_a_cascading_vit_hard_fails_on_invalid_image(real_signal_vlm) -> None:
    """Ensure CascadingViTTrackA raises ValueError when given None instead of a real PIL.Image."""
    app = load("track_a_cascading_vit", {})
    with pytest.raises(ValueError, match="CascadingViTTrackA requires a valid non-empty PIL.Image.Image"):
        app.detect(None, Context(llm=real_signal_vlm))  # type: ignore[arg-type]
