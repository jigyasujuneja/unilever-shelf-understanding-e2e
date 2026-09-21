"""Tests for the Pluggable Approach Registry (`src/shelf_benchmark/approaches/`)."""

from shelf_benchmark.approaches import (
    BaseShelfApproachPlugin,
    CommonLayerContext,
    GLOBAL_APPROACH_REGISTRY,
)


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
    assert "Pepsodent" in cfg.taxonomy.hul_brands

    prompt = build_classification_prompt(cfg.taxonomy)
    assert "56 HUL" not in prompt
    assert "123 non-HUL" not in prompt
    assert "10 categories" not in prompt
    assert "960" not in prompt
    assert "953" not in prompt
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



