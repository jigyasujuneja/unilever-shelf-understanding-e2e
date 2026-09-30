"""Dedicated real-data tests for `sister_shade_systemone` (`SisterShadeSystemOneClassifier`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_sister_shade_systemone_clustering_and_cielab_disambiguation(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify Complete-Linkage visual clustering + 3x Sub-ROI CIELAB Delta-E variant classification."""
    cls_app = load("sister_shade_systemone", {})
    assert cls_app.task == "classification"
    assert cls_app.target_field == "variant"

    boxes = real_shelf_sample_0["gt_boxes"][:8]
    ctx = Context(llm=real_signal_vlm)
    preds = cls_app.classify(real_shelf_sample_0["image"], boxes, ctx)

    assert len(preds) == len(boxes)
    assert float(ctx.trace.meta.get("compression_ratio", 0.0)) >= 1.0
    for pred in preds:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_sister_shade_systemone_hard_fails_on_hallucination(
    real_shelf_sample_0: dict,
    hallucinating_vlm,
) -> None:
    """Ensure sister_shade_systemone raises ValueError when VLM emits non-canonical SKU or brand."""
    cls_app = load("sister_shade_systemone", {})
    ctx = Context(llm=hallucinating_vlm)
    with pytest.raises(ValueError, match="Hallucinated or out-of-catalog SKU"):
        cls_app.classify(real_shelf_sample_0["image"], real_shelf_sample_0["gt_boxes"][:4], ctx)
