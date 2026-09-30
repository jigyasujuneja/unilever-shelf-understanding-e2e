"""Dedicated real-data tests for `track_f_sam2_scann` (`Sam2MaskScannTrackF`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_track_f_sam2_scann_mobilesam_instance_segmentation(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify Sam2MaskScannTrackF runs real MobileSAM (`mobile_sam.pt`) instance masks on real shelf images."""
    app = load("track_f_sam2_scann", {})
    assert app.task == "combined"

    ctx = Context(llm=real_signal_vlm)
    boxes = app.detect(real_shelf_sample_0["image"], ctx)
    assert len(boxes) >= 12
    assert "MobileSAM instance masks applied for background suppression" in ctx.trace.steps[-1]["detail"]
    for pred in ctx.trace.labels:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_track_f_sam2_scann_hard_fails_on_invalid_image(real_signal_vlm) -> None:
    """Ensure Sam2MaskScannTrackF raises ValueError when given None instead of a real PIL.Image."""
    app = load("track_f_sam2_scann", {})
    with pytest.raises(ValueError, match="Sam2MaskScannTrackF requires a valid non-empty PIL.Image.Image"):
        app.detect(None, Context(llm=real_signal_vlm))  # type: ignore[arg-type]
