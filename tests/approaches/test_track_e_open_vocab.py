"""Dedicated real-data tests for `track_e_open_vocab` (`OpenVocabGroundingTrackE`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_track_e_open_vocab_owlv2_and_siglip_zero_vlm_calls(
    real_shelf_sample_0: dict,
    trap_vlm,
) -> None:
    """Verify OpenVocabGroundingTrackE executes real HuggingFace OWL-v2 + SigLIP with zero Gemini VLM calls."""
    app = load("track_e_open_vocab", {})
    assert app.task == "combined"

    ctx = Context(llm=trap_vlm)
    boxes = app.detect(real_shelf_sample_0["image"], ctx)
    assert len(boxes) >= 10
    assert "OWL-v2 open-vocab proposals" in ctx.trace.steps[-1]["detail"]
    assert "SigLIP top prompt:" in ctx.trace.steps[-1]["detail"]
    for pred in ctx.trace.labels:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_track_e_open_vocab_hard_fails_on_invalid_image(trap_vlm) -> None:
    """Ensure OpenVocabGroundingTrackE raises ValueError when given None instead of a real PIL.Image."""
    app = load("track_e_open_vocab", {})
    with pytest.raises(ValueError, match="OpenVocabGroundingTrackE requires a valid non-empty PIL.Image.Image"):
        app.detect(None, Context(llm=trap_vlm))  # type: ignore[arg-type]
