"""Dedicated real-data tests for `promo_product_detector` (`PromoProductDetector`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_promo_product_detector_end_to_end_real_shelf(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify PromoProductDetector detects and classifies promotional SKU facings on real shelf images."""
    app = load("promo_product_detector", {})
    assert app.task == "combined"
    assert app.epic == "MT Merchandising - Promotion Product Detection"

    ctx = Context(llm=real_signal_vlm)
    boxes, labels = app.detect_and_classify(real_shelf_sample_0["image"], ctx)
    assert len(boxes) >= 15
    assert len(labels) == len(boxes)
    for pred in labels:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_promo_product_detector_hard_fails_on_hallucination(
    real_shelf_sample_0: dict,
    hallucinating_vlm,
) -> None:
    """Ensure promo_product_detector raises ValueError on hallucinated SKUs."""
    app = load("promo_product_detector", {})
    with pytest.raises(ValueError, match="Hallucinated or out-of-catalog SKU"):
        app.detect_and_classify(real_shelf_sample_0["image"], Context(llm=hallucinating_vlm))
