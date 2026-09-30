"""Dedicated real-data tests for `ft_gemini31_variant_compound` (`FTGemini31VariantCompoundClassifier`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_ft_gemini31_variant_compound_with_and_without_prior(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify FTGemini31VariantCompoundClassifier runs standalone and conditioned on Stage-1 prior."""
    cls_app = load("ft_gemini31_variant_compound", {})
    assert cls_app.task == "classification"
    assert cls_app.target_field == "variant"
    assert cls_app.sft_lora_endpoint.startswith("projects/")

    boxes = real_shelf_sample_0["gt_boxes"][:5]
    ctx_standalone = Context(llm=real_signal_vlm)
    preds_standalone = cls_app.classify(real_shelf_sample_0["image"], boxes, ctx_standalone)
    assert len(preds_standalone) == len(boxes)

    # Now condition on a Stage-1 prior from ft_gemini31_cat_brand_pkg
    stage1_app = load("ft_gemini31_cat_brand_pkg", {})
    prior = stage1_app.classify(real_shelf_sample_0["image"], boxes, Context(llm=real_signal_vlm))

    ctx_conditioned = Context(llm=real_signal_vlm)
    preds_conditioned = cls_app.classify(
        real_shelf_sample_0["image"], boxes, ctx_conditioned, prior=prior
    )
    assert len(preds_conditioned) == len(boxes)
    for pred, prior_item in zip(preds_conditioned, prior, strict=True):
        hul_domain.validate_canonical_7dim_prediction(pred)
        assert pred["category"] == prior_item["category"]
        assert pred["brand"] == prior_item["brand"]
        assert pred["packaging_type"] == prior_item["packaging_type"]


def test_ft_gemini31_variant_compound_hard_fails_on_hallucination(
    real_shelf_sample_0: dict,
    hallucinating_vlm,
) -> None:
    """Ensure ft_gemini31_variant_compound raises ValueError on out-of-catalog SKUs."""
    cls_app = load("ft_gemini31_variant_compound", {})
    ctx = Context(llm=hallucinating_vlm)
    with pytest.raises(ValueError, match="Hallucinated or out-of-catalog SKU"):
        cls_app.classify(real_shelf_sample_0["image"], real_shelf_sample_0["gt_boxes"][:4], ctx)
