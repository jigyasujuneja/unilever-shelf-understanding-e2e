"""Dedicated real-data tests for `tiered_hybrid_scann` (`TieredHybridScann`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_tiered_hybrid_scann_end_to_end_real_shelf(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify TieredHybridScann runs RT-DETR-v2 + Complete-Linkage Clustering + ScaNN + 7-Dim Classification."""
    app = load("tiered_hybrid_scann", {})
    assert app.task == "combined"

    ctx = Context(llm=real_signal_vlm)
    boxes = app.detect(real_shelf_sample_0["image"], ctx)
    assert len(boxes) >= 15
    assert len(ctx.trace.labels) == len(boxes)
    for pred in ctx.trace.labels:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_tiered_hybrid_scann_hard_fails_on_hallucination(
    real_shelf_sample_0: dict,
    hallucinating_vlm,
) -> None:
    """Ensure tiered_hybrid_scann raises ValueError on out-of-catalog SKUs."""
    app = load("tiered_hybrid_scann", {})
    with pytest.raises(ValueError, match="Hallucinated or out-of-catalog SKU"):
        app.detect(real_shelf_sample_0["image"], Context(llm=hallucinating_vlm))
