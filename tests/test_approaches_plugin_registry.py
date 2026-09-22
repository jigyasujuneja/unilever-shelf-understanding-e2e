"""Tests for the Pluggable Approach Registry (`src/shelf_benchmark/approaches/`)."""

import pytest

from shelf_benchmark.approaches import (
    GLOBAL_APPROACH_REGISTRY,
    BaseShelfApproachPlugin,
)

# Everything in this module runs without network or GCP credentials.
pytestmark = pytest.mark.offline


def test_approach_registry_auto_discovers_all_plugins() -> None:
    """Verify that GLOBAL_APPROACH_REGISTRY discovers both existing VLM wrappers and the new 3-Stage Class-Agnostic Visual Embedding approach."""
    ids = GLOBAL_APPROACH_REGISTRY.list_ids()
    assert "class_agnostic_visual_embedding" in ids
    assert "cloud_vision_visual_embedding" in ids
    assert "single_pass_full_shelf" in ids
    assert "two_stage_bbox_guided_nms" in ids
    assert "two_stage_physical_crop_per_facing" in ids

    plugin = GLOBAL_APPROACH_REGISTRY.get("class_agnostic_visual_embedding")
    assert isinstance(plugin, BaseShelfApproachPlugin)
    assert plugin.category == "classic_cv_metric_learning"
    assert len(plugin.stages_description) == 3


def test_centralized_taxonomy_config_and_prompts() -> None:
    """Verify TaxonomyConfig loads from configs/taxonomy.yaml and builds prompts dynamically without hardcoded counts."""
    from shelf_benchmark.config import BenchmarkConfig, TaxonomyConfig
    from shelf_benchmark.tasks.classification import build_classification_prompt
    from shelf_benchmark.tasks.facing_utils import check_is_hul_brand, derive_size_bucket_from_bbox

    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    assert "Oral Care" in cfg.taxonomy.categories
    assert "Toothpaste" in cfg.taxonomy.subcategories
    assert "box" in cfg.taxonomy.packaging_types
    assert cfg.taxonomy.hul_brands == []
    assert cfg.taxonomy.non_hul_brands == []

    prompt = build_classification_prompt(cfg.taxonomy)
    assert "56 HUL" not in prompt
    assert "123 non-HUL" not in prompt
    assert "10 categories" not in prompt
    assert "960" not in prompt
    assert "953" not in prompt
    assert "no predefined brand list required" in prompt
    assert "Oral Care" in prompt

    # Custom overridden TaxonomyConfig works seamlessly
    custom_taxonomy = TaxonomyConfig(
        categories=["Custom Category A"],
        subcategories=["Custom Subcategory B"],
        packaging_types=["Custom Pouch"],
        pack_types=["Bundle"],
        hul_brands=["CustomBrandX"],
        non_hul_brands=["OtherBrandY"],
    )
    assert check_is_hul_brand("CustomBrandX", taxonomy=custom_taxonomy) is True
    assert check_is_hul_brand("OtherBrandY", taxonomy=custom_taxonomy) is False
    assert (
        derive_size_bucket_from_bbox(
            [100, 100, 200, 150],
            [[100, 100, 200, 150]],
            packaging_type="box",
            model_size_hint="40g",
            taxonomy=custom_taxonomy,
        )
        == custom_taxonomy.size_buckets.small_label
    )


def test_developer_sdk_universal_model_and_custom_approach_decorator(tmp_path) -> None:
    """Verify ShelfBenchmarkSDK supports custom Gemma/GEAP models and @register_approach_function with full OTel logging."""
    from pathlib import Path

    from shelf_benchmark import (
        ModelPricing,
        ShelfBenchmarkSDK,
        UniversalModelSpec,
        register_approach_function,
    )

    @register_approach_function(
        approach_id="unit_test_custom_approach",
        display_name="Unit Test Custom Approach",
    )
    def my_custom_approach(ctx, model_name, shelf_image_uri):
        return [
            {
                "bbox_2d": [535, 386, 795, 440],
                "shelf_row": "middle",
                "category": "Skin Care",
                "subcategory": "Face Wash",
                "brand": "Pond's",
                "variant": "Bright Beauty",
                "packaging_type": "tube",
                "pack_type": "Single",
                "size": "50g",
                "confidence": 0.96,
                "_input_tokens": 300,
                "_thinking_tokens": 20,
                "_output_tokens": 100,
            }
        ]

    sdk = ShelfBenchmarkSDK(output_dir=tmp_path / "sdk_reports")
    sdk.register_model(
        UniversalModelSpec(
            model_id="gemma-3-27b-it-test",
            provider_family="custom_callable",
            pricing=ModelPricing(input=0.08, thinking=0.0, output=0.24),
            custom_handler=lambda prompt, img, schema: {
                "total_classified_products": 1,
                "distinct_brands_found": ["Pepsodent"],
                "classified_products": [
                    {
                        "product_index": 1,
                        "bbox_2d": [171, 676, 487, 813],
                        "shelf_row": "top",
                        "position_on_shelf": 1,
                        "category": "Oral Care",
                        "subcategory": "Toothpaste",
                        "brand": "Pepsodent",
                        "is_hul_brand": True,
                        "variant": "Germi Check",
                        "packaging_type": "box",
                        "pack_type": "Single",
                        "size": "150g",
                        "product_name": "Pepsodent Germi Check",
                        "confidence": 0.98,
                    }
                ],
                "_token_usage": {"input_tokens": 400, "thinking_tokens": 0, "output_tokens": 120},
            },
        )
    )

    summary = sdk.run_suite(
        models=["gemma-3-27b-it-test"],
        tasks=["classification"],
        approaches=["single_pass_full_shelf", "unit_test_custom_approach"],
    )
    assert len(summary["results"]) == 2
    assert Path(summary["otel_log_path"]).exists()
    assert Path(summary["artifacts"]["markdown_report"]).exists()

    # Verify plugging in Ground Truth at runtime scores the custom approach automatically
    import json

    gt_file = tmp_path / "sample_gt.json"
    gt_file.write_text(
        json.dumps(
            {
                "shelf-image.png": {
                    "image_id": "shelf-image.png",
                    "total_main_shelf_facings": 1,
                    "expected_brands": ["Pond's"],
                    "items": [
                        {
                            "item_id": 1,
                            "brand": "Pond's",
                            "product_name": "Pond's Bright Beauty Face Wash",
                            "sku_id": "HUL-PONDS-50G",
                            "bbox_2d": [535, 386, 795, 440],
                            "shelf_row": "middle",
                        }
                    ],
                }
            }
        )
    )
    sdk.connect_ground_truth(provider_type="json", source_uri=str(gt_file))
    gt_summary = sdk.run_suite(
        models=["gemma-3-27b-it-test"],
        tasks=["classification"],
        approaches=["unit_test_custom_approach"],
    )
    gt_res = gt_summary["results"][0]
    assert gt_res.accuracy.ground_truth_available is True
    assert gt_res.accuracy.count_accuracy == 1.0
    assert gt_res.accuracy.mean_iou == 1.0
    assert gt_res.accuracy.brand_classification_accuracy == 1.0
    assert gt_res.row_level_items[0].brand_correct is True
