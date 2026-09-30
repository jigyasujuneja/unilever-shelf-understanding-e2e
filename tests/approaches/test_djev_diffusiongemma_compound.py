"""Dedicated real-data tests for `djev_diffusiongemma_compound` (`DjevDiffusionGemmaCompoundClassifier`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_djev_diffusiongemma_compound_real_3task_prefilter(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify 64-token canvas compound classification and 3-task ScaNN prefiltering on real shelf crops."""
    cls_app = load("djev_diffusiongemma_compound", {})
    assert cls_app.task == "classification"

    boxes = real_shelf_sample_0["gt_boxes"][:6]
    ctx = Context(llm=real_signal_vlm)
    preds = cls_app.classify(real_shelf_sample_0["image"], boxes, ctx)

    assert len(preds) == len(boxes)
    for pred in preds:
        hul_domain.validate_canonical_7dim_prediction(pred)
        assert "djev_routing" in pred
        assert int(pred["scann_pool_after_3task_filter"]) < 50000
        assert isinstance(pred["filtered_candidate_skus"], list)
        assert len(pred["filtered_candidate_skus"]) > 0


def test_djev_diffusiongemma_compound_hard_fails_on_hallucinated_sku(
    real_shelf_sample_0: dict,
    hallucinating_vlm,
) -> None:
    """Ensure djev_diffusiongemma_compound raises ValueError on out-of-catalog SKUs."""
    cls_app = load("djev_diffusiongemma_compound", {})
    ctx = Context(llm=hallucinating_vlm)
    with pytest.raises(ValueError, match="Hallucinated or out-of-catalog SKU"):
        cls_app.classify(real_shelf_sample_0["image"], real_shelf_sample_0["gt_boxes"][:4], ctx)
