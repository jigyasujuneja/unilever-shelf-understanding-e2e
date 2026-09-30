"""Dedicated real-data tests for `ft_gemini31_cat_brand_pkg` (`FTGemini31CatBrandPkgClassifier`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_ft_gemini31_cat_brand_pkg_loads_sft_lora_endpoint_and_classifies(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify FTGemini31CatBrandPkgClassifier binds the real Vertex AI LoRA endpoint and classifies real crops."""
    cls_app = load("ft_gemini31_cat_brand_pkg", {})
    assert cls_app.task == "classification"
    assert cls_app.sft_lora_endpoint.startswith("projects/")

    boxes = real_shelf_sample_0["gt_boxes"][:6]
    ctx = Context(llm=real_signal_vlm)
    preds = cls_app.classify(real_shelf_sample_0["image"], boxes, ctx)

    assert ctx.trace.meta.get("sft_lora_endpoint") == cls_app.sft_lora_endpoint
    assert len(preds) == len(boxes)
    for pred in preds:
        hul_domain.validate_canonical_7dim_prediction(pred)


def test_ft_gemini31_cat_brand_pkg_hard_fails_on_hallucination(
    real_shelf_sample_0: dict,
    hallucinating_vlm,
) -> None:
    """Ensure ft_gemini31_cat_brand_pkg hard-fails on hallucinated SKUs."""
    cls_app = load("ft_gemini31_cat_brand_pkg", {})
    ctx = Context(llm=hallucinating_vlm)
    with pytest.raises(ValueError, match="Hallucinated or out-of-catalog SKU"):
        cls_app.classify(real_shelf_sample_0["image"], real_shelf_sample_0["gt_boxes"][:4], ctx)
