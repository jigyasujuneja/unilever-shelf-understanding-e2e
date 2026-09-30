"""Dedicated real-data tests for `maxvit_clustered_djev` (`MaxViTClusteredDjevApproach`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_maxvit_clustered_djev_real_maxvit_t_clustering_and_scann(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify MaxViTClusteredDjevApproach executes real MaxViT-T Block+Grid clustering + ScaNN + SystemOne."""
    app = load("maxvit_clustered_djev", {})
    assert app.task == "combined"

    ctx = Context(llm=real_signal_vlm)
    boxes = app.detect(real_shelf_sample_0["image"], ctx)
    assert len(boxes) >= 15
    assert len(ctx.trace.steps) == 3
    assert len(ctx.trace.labels) == len(boxes)
    for pred in ctx.trace.labels:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_maxvit_clustered_djev_hard_fails_on_hallucination(
    real_shelf_sample_0: dict,
    hallucinating_vlm,
) -> None:
    """Ensure maxvit_clustered_djev raises ValueError on hallucinated SKUs."""
    app = load("maxvit_clustered_djev", {})
    with pytest.raises(ValueError, match="Hallucinated or out-of-catalog SKU"):
        app.detect(real_shelf_sample_0["image"], Context(llm=hallucinating_vlm))
