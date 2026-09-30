"""Dedicated real-data tests for `hul_8stage_gemini38_hybrid` (`HUL8StageGemini38Hybrid`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_hul_8stage_gemini38_hybrid_full_pipeline_real_shelf(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify HUL8StageGemini38Hybrid executes all 8 stages (RT-DETR-v2 + MaxViT Clustering + ScaNN + SystemOne + Stage 6 KPIs)."""
    app = load("hul_8stage_gemini38_hybrid", {})
    assert app.task == "combined"

    ctx = Context(llm=real_signal_vlm)
    boxes = app.detect(real_shelf_sample_0["image"], ctx)
    assert len(boxes) >= 15
    assert len(ctx.trace.steps) == 6
    assert len(ctx.trace.labels) == len(boxes)
    for pred in ctx.trace.labels:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_hul_8stage_gemini38_hybrid_hard_fails_on_hallucination(
    real_shelf_sample_0: dict,
    hallucinating_vlm,
) -> None:
    """Ensure hul_8stage_gemini38_hybrid raises ValueError on out-of-catalog SKUs."""
    app = load("hul_8stage_gemini38_hybrid", {})
    with pytest.raises(ValueError, match="Hallucinated or out-of-catalog SKU"):
        app.detect(real_shelf_sample_0["image"], Context(llm=hallucinating_vlm))
