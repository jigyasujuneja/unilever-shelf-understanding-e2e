"""Dedicated real-data tests for `compound_pipeline_1_plus_2` (`CompoundPipeline1Plus2`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_compound_pipeline_1_plus_2_hierarchical_chain_real_shelf(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify CompoundPipeline1Plus2 chains detector -> ft_gemini31_cat_brand_pkg -> ft_gemini31_variant_compound."""
    app = load("compound_pipeline_1_plus_2", {})
    assert app.task == "combined"

    ctx = Context(llm=real_signal_vlm)
    boxes, labels = app.detect_and_classify(real_shelf_sample_0["image"], ctx)
    assert len(boxes) >= 15
    assert len(labels) == len(boxes)
    for pred in labels:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_compound_pipeline_1_plus_2_hard_fails_on_hallucination(
    real_shelf_sample_0: dict,
    hallucinating_vlm,
) -> None:
    """Ensure compound_pipeline_1_plus_2 raises ValueError on hallucinated SKUs."""
    app = load("compound_pipeline_1_plus_2", {})
    with pytest.raises(ValueError, match="Hallucinated or out-of-catalog SKU"):
        app.detect_and_classify(real_shelf_sample_0["image"], Context(llm=hallucinating_vlm))
