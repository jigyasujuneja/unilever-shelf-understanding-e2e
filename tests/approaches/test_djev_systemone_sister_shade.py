"""Dedicated real-data tests for `djev_systemone_sister_shade` (`DjevSystemOneSisterShade`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_djev_systemone_sister_shade_end_to_end_real_shelf(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify DjevSystemOneSisterShade executes all 5 stages on real shelf images."""
    app = load("djev_systemone_sister_shade", {})
    assert app.task == "combined"

    ctx = Context(llm=real_signal_vlm)
    boxes = app.detect(real_shelf_sample_0["image"], ctx)
    assert len(boxes) >= 15
    assert len(ctx.trace.steps) == 4
    assert len(ctx.trace.labels) == len(boxes)
    for pred in ctx.trace.labels:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_djev_systemone_sister_shade_hard_fails_on_hallucination(
    real_shelf_sample_0: dict,
    hallucinating_vlm,
) -> None:
    """Ensure djev_systemone_sister_shade raises ValueError on hallucinated SKUs."""
    app = load("djev_systemone_sister_shade", {})
    with pytest.raises(ValueError, match="Hallucinated or out-of-catalog SKU"):
        app.detect(real_shelf_sample_0["image"], Context(llm=hallucinating_vlm))
