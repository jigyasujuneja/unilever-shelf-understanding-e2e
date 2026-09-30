"""Dedicated real-data tests for `modular_e2e_pipeline` (`ModularEndToEndPipeline`)."""

from __future__ import annotations

import pytest

from approaches import load
from approaches.base import Context
from utils import hul_domain


def test_modular_e2e_pipeline_stage_override_sensitivity_on_real_shelf(
    real_shelf_sample_0: dict,
    real_signal_vlm,
) -> None:
    """Verify swapping stage overrides in modular_e2e_pipeline changes real clustering and retrieval metrics."""
    app_a = load(
        "modular_e2e_pipeline",
        {
            "modular_overrides": {
                "detector": "rtdetr_shelf_rail_detector",
                "attr_classifier": "hul_hierarchy_classifier",
                "variant_classifier": "sister_shade_systemone",
            },
            "stage_overrides": {
                "rectifier": "ransac_4dof_homography",
                "post_detector": "shelf_rail_soft_nms",
                "clusterer": "connected_component_adjacency",
                "retriever": "ijepa_scann_entropy_prefilter",
                "tiebreaker": "cielab_delta_e_and_systemone",
                "shelf_metrics": "dual_mt_marketshare_and_merchandising",
            },
        },
    )
    ctx_a = Context(llm=real_signal_vlm)
    boxes_a, labels_a = app_a.detect_and_classify(real_shelf_sample_0["image"], ctx_a)

    app_b = load(
        "modular_e2e_pipeline",
        {
            "modular_overrides": {
                "detector": "yolo_n26_sku110k",
                "attr_classifier": "djev_diffusiongemma_compound",
                "variant_classifier": "ft_gemini31_variant_compound",
            },
            "stage_overrides": {
                "rectifier": "depth_anything_v2",
                "post_detector": "standard_nms",
                "clusterer": "none",
                "retriever": "pure_scann_cosine",
                "tiebreaker": "cielab_delta_e_only",
                "shelf_metrics": "single_img_planogram_oos",
            },
        },
    )
    ctx_b = Context(llm=real_signal_vlm)
    boxes_b, labels_b = app_b.detect_and_classify(real_shelf_sample_0["image"], ctx_b)

    assert len(boxes_a) >= 10 and len(boxes_b) >= 10
    assert len(labels_a) == len(boxes_a) and len(labels_b) == len(boxes_b)
    for pred in labels_a + labels_b:
        hul_domain.validate_canonical_7dim_prediction(pred)

    meta_a = ctx_a.trace.meta["composed_modules"]
    meta_b = ctx_b.trace.meta["composed_modules"]
    assert meta_a["clustering"]["compression_ratio"] != meta_b["clustering"]["compression_ratio"]
    assert meta_a["retrieval"]["scann_pool_after"] < meta_b["retrieval"]["scann_pool_after"]
    assert labels_a[0]["tiebreaker_mode"] != labels_b[0]["tiebreaker_mode"]


def test_modular_e2e_pipeline_hard_fails_on_invalid_image(real_signal_vlm) -> None:
    """Ensure modular_e2e_pipeline raises ValueError when given None instead of a real PIL.Image."""
    app = load("modular_e2e_pipeline", {})
    with pytest.raises(ValueError, match="Expected a valid PIL.Image.Image"):
        app.detect_and_classify(None, Context(llm=real_signal_vlm))  # type: ignore[arg-type]
