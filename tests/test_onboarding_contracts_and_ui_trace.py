"""Contract tests pinning onboarding readiness, config integrity, bbox formats, null-vs-zero serialization, SimpleShelfApproachPlugin, and UI/Cloud Run execution trace metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from shelf_benchmark import (
    BenchmarkConfig,
    CommonLayerContext,
    SimpleShelfApproachPlugin,
)
from shelf_benchmark.approaches import GLOBAL_APPROACH_REGISTRY
from shelf_benchmark.data.ground_truth import SUPPORTED_BBOX_FORMATS, convert_bbox
from shelf_benchmark.evaluation.metrics import brands_match
from shelf_benchmark.models import ShelfAssociationRecord
from shelf_benchmark.testing import (
    OFFLINE_IMAGE_URI,
    benchmark_harness,
    run_offline_approach,
)

pytestmark = pytest.mark.offline


def test_shipped_default_config_matches_contracts():
    """Verify configs/default_config.yaml has zero hardcoded brand aliases/stopwords, preserves per-facing rates, and documents valid bbox_format names."""
    yaml_path = Path("configs/default_config.yaml")
    cfg = BenchmarkConfig.from_yaml(yaml_path)

    assert cfg.evaluation.brand_aliases == {}, "configs/default_config.yaml must remain free of hardcoded brand aliases"
    assert cfg.evaluation.product_stopwords == [], "configs/default_config.yaml must remain free of hardcoded stopwords"
    assert brands_match("Brand's", "Brands", cfg.evaluation) is True
    assert brands_match("CaféBrand", "CafeBrand", cfg.evaluation) is True

    # 2. Per-approach embedding/vision surcharges must be preserved
    per_app = cfg.billing.embeddings_and_vision.per_approach_per_facing_usd
    assert per_app.get("two_stage_physical_crop_per_facing") == pytest.approx(0.000125)
    assert per_app.get("class_agnostic_visual_embedding") == pytest.approx(0.000125)

    # 3. Every bbox_format option documented in the YAML comments must exist in SUPPORTED_BBOX_FORMATS
    yaml_text = yaml_path.read_text(encoding="utf-8")
    commented_formats = re.findall(
        r"#\s+(ymin_xmin_ymax_xmax_1000|coco_xywh_px|xyxy_px|xyxy_norm|yxyx_norm|yolo_xywh_norm|xmin_ymin_xmax_ymax_\w+)\b",
        yaml_text,
    )
    assert commented_formats
    for fmt in commented_formats:
        assert fmt in SUPPORTED_BBOX_FORMATS, f"YAML documents unsupported bbox_format: {fmt}"


@pytest.mark.parametrize(
    ("bbox_format", "raw_box", "w", "h"),
    [
        ("ymin_xmin_ymax_xmax_1000", [250, 250, 750, 375], None, None),
        ("yxyx_norm", [0.25, 0.25, 0.75, 0.375], None, None),
        ("xyxy_norm", [0.25, 0.25, 0.375, 0.75], None, None),
        ("xyxy_px", [200, 150, 300, 450], 800, 600),
        ("coco_xywh_px", [200, 150, 100, 300], 800, 600),
        ("yolo_xywh_norm", [0.3125, 0.50, 0.125, 0.50], None, None),
    ],
)
def test_all_bbox_formats_convert_to_canonical_1000(bbox_format, raw_box, w, h):
    """Golden test for the 6 bounding-box conventions in docs/GROUND_TRUTH_CONTRACT.md."""
    converted = convert_bbox(raw_box, bbox_format=bbox_format, image_width=w, image_height=h)
    assert converted == [250, 250, 750, 375]


def test_swapping_xyxy_norm_and_yxyx_norm_produces_distinct_geometry():
    """Demonstrates why declaring bbox_format explicitly is critical."""
    box_xyxy = convert_bbox([0.25, 0.25, 0.375, 0.75], bbox_format="xyxy_norm")
    box_misread_as_yxyx = convert_bbox([0.25, 0.25, 0.375, 0.75], bbox_format="yxyx_norm")
    assert box_xyxy == [250, 250, 750, 375]
    assert box_misread_as_yxyx == [250, 250, 375, 750]
    assert box_xyxy != box_misread_as_yxyx


def test_unscored_reports_serialize_null_not_zero(tmp_path: Path):
    """Headline contract: unmeasured accuracy must serialize as null (None), never 0.0, in reports."""
    sdk = benchmark_harness(tmp_path, with_ground_truth=False)
    summary = sdk.run_suite(
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )
    artifacts = summary["artifacts"]
    assert Path(artifacts["leaderboard_csv"]).exists()

    summary_json = json.loads(Path(artifacts["summary_json"]).read_text(encoding="utf-8"))
    rec = summary_json["summary"][0]
    assert rec["accuracy_status"] == "PLACEHOLDER_AWAITING_GROUND_TRUTH"
    assert rec["detection_f1"] is None
    assert rec["detection_precision"] is None
    assert rec["detection_recall"] is None
    assert rec["brand_classification_accuracy"] is None


@pytest.mark.parametrize(
    "approach_id",
    [
        "single_pass_full_shelf",
        "configurable_multi_attribute_vlm",
        "single_step_detect_classify_and_match",
        "two_stage_bbox_guided_nms",
        "two_stage_physical_crop_per_facing",
        "class_agnostic_visual_embedding",
        "cloud_vision_visual_embedding",
    ],
)
def test_all_registered_approaches_and_crop_helper_run_offline(
    tmp_path: Path, approach_id: str
):
    """Every advertised approach in list-approaches must run offline without TypeError and emit full execution_trace."""
    res = run_offline_approach(
        tmp_path / approach_id,
        approach_id=approach_id,
        with_ground_truth=True,
    )
    assert res.status == "SUCCESS"
    assert len(res.row_level_items) >= 1
    assert res.accuracy.detection_f1 == pytest.approx(1.0)

    trace = res.execution_trace
    assert trace["separation_approach"] == approach_id
    assert trace["api_calls_count"] >= 1
    assert trace["models_invoked"]
    assert trace["opentelemetry"]["trace_id"] == res.trace_id
    assert "jq" in trace["opentelemetry"]["local_jq_command"]


def test_simple_shelf_approach_plugin_base_class(tmp_path: Path):
    """Verify a new engineer can write a complete class-based plugin in ~15 lines with SimpleShelfApproachPlugin."""

    class MinimalEngineerPlugin(SimpleShelfApproachPlugin):
        @property
        def approach_id(self) -> str:
            return "minimal_engineer_plugin"

        @property
        def display_name(self) -> str:
            return "Minimal 15-Line Engineer Plugin"

        @property
        def category(self) -> str:
            return "two_stage_vlm"

        @property
        def stages_description(self) -> List[str]:
            return ["Stage 1: Detect & Deduplicate", "Stage 2: Classify"]

        def detect_and_classify(
            self,
            ctx: CommonLayerContext,
            model_name: str,
            record: ShelfAssociationRecord,
        ) -> List[Dict[str, Any]]:
            candidates = [
                {
                    "bbox_2d": [100, 100, 300, 200],
                    "brand": "Brand_A",
                    "product_name": "Brand_A Radiance Daily Cleanser",
                    "packaging_type": "tube",
                    "size": "100g",
                    "is_hul_brand": True,
                }
            ]
            kept, filtered = ctx.deduplicate_depth_stacked_facings(candidates)
            if kept:
                kept[0]["_depth_filtered"] = filtered
            return kept

    GLOBAL_APPROACH_REGISTRY.register(MinimalEngineerPlugin())
    res = run_offline_approach(tmp_path, "minimal_engineer_plugin", with_ground_truth=True)
    assert res.separation_approach == "minimal_engineer_plugin"
    assert len(res.row_level_items) == 1
    assert res.row_level_items[0].is_hul_brand is True
    assert res.execution_trace["call_topology"] == "Custom Plugin Pipeline (minimal_engineer_plugin)"


def test_universal_schema_adapters_for_datasets_and_ground_truth(tmp_path: Path):
    """Verify sdk.connect_dataset and sdk.connect_ground_truth adapt to arbitrary schemas (nested dot-paths, 4-column CSV boxes, polygon vertices)."""
    sdk = benchmark_harness(tmp_path, with_ground_truth=False)

    # 1. Manifest with custom nested dot-paths and relative image path resolved to GCS bucket
    manifest_file = tmp_path / "custom_manifest.json"
    manifest_file.write_text(
        json.dumps(
            [
                {
                    "audit_row_id": "row-99",
                    "photo_filename": "store_42/aisle_3.jpg",
                    "context": {"retail_store_code": "S-42"},
                }
            ]
        ),
        encoding="utf-8",
    )
    ds_stats = sdk.connect_dataset(
        provider_type="json",
        source_uri=str(manifest_file),
        shelf_images_bucket="gs://custom-retail-bucket/photos",
        schema_mapping={
            "association_id_field": "audit_row_id",
            "shelf_image_uri_field": "photo_filename",
            "store_id_field": "context.retail_store_code",
        },
    )
    assert ds_stats["records_loaded"] == 1
    assert ds_stats["image_uris"] == ["gs://custom-retail-bucket/photos/store_42/aisle_3.jpg"]
    assert ds_stats["records"][0].store_id == "S-42"

    # 2. Ground truth with 4-column CSV boxes (y_min,x_min,y_max,x_max) and custom column names
    csv_gt = tmp_path / "custom_4col_gt.csv"
    csv_gt.write_text(
        "photo_uri,y_min,x_min,y_max,x_max,maker,sku_label,pack_shape\n"
        "gs://custom-retail-bucket/photos/store_42/aisle_3.jpg,0.10,0.20,0.50,0.40,Brand_A,Brand_A Cleansing Bar,box\n",
        encoding="utf-8",
    )
    gt_stats = sdk.connect_ground_truth(
        provider_type="csv",
        source_uri=str(csv_gt),
        bbox_format="yxyx_norm",
        schema_mapping={
            "image_uri_field": "photo_uri",
            "bbox_field": "y_min,x_min,y_max,x_max",
            "brand_field": "maker",
            "product_name_field": "sku_label",
            "packaging_type_field": "pack_shape",
        },
    )
    assert gt_stats["images"] == 1
    gt_rec = sdk.gt_provider.get_ground_truth("gs://custom-retail-bucket/photos/store_42/aisle_3.jpg")
    assert gt_rec is not None
    assert gt_rec.items[0].bbox_2d == [100, 200, 500, 400]
    assert gt_rec.items[0].brand == "Brand_A"

    # 3. Ground truth with Vertex AI AutoML Vision polygon vertices and nested dot-paths
    json_gt = tmp_path / "vertex_automl_gt.json"
    json_gt.write_text(
        json.dumps(
            [
                {
                    "imageGcsUri": "gs://custom-retail-bucket/photos/store_42/aisle_3.jpg",
                    "annotations": [
                        {
                            "boundingPoly": {
                                "normalizedVertices": [
                                    {"x": 0.25, "y": 0.10},
                                    {"x": 0.50, "y": 0.10},
                                    {"x": 0.50, "y": 0.60},
                                    {"x": 0.25, "y": 0.60},
                                ]
                            },
                            "attributes": {"brand_name": "Brand_A", "title": "Brand_A Body Wash"},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    sdk.connect_ground_truth(
        provider_type="json",
        source_uri=str(json_gt),
        bbox_format="xyxy_norm",
        schema_mapping={
            "image_uri_field": "imageGcsUri",
            "items_list_field": "annotations",
            "bbox_field": "boundingPoly.normalizedVertices",
            "brand_field": "attributes.brand_name",
            "product_name_field": "attributes.title",
        },
    )
    gt_poly = sdk.gt_provider.get_ground_truth("gs://custom-retail-bucket/photos/store_42/aisle_3.jpg")
    assert gt_poly is not None
    assert gt_poly.items[0].bbox_2d == [100, 250, 600, 500]
    assert gt_poly.items[0].brand == "Brand_A"


def test_ui_server_offline_and_sample_gt_traceability():
    """Verify the UI/Cloud Run backend endpoint supports both placeholder and sample-GT offline execution with trace metadata."""
    from ui.server import execute_live_benchmark_task

    res_placeholder = execute_live_benchmark_task(
        task_type="classification",
        model_name="gemini-3.8-flash",
        separation_approach="single_pass_full_shelf",
        mode="offline",
        connect_sample_gt=False,
    )
    assert res_placeholder["execution_mode"] == "offline_local_fixture"
    assert res_placeholder["accuracy"]["accuracy_status"] == "PLACEHOLDER_AWAITING_GROUND_TRUTH"
    assert res_placeholder["accuracy"]["detection_f1"] is None
    assert res_placeholder["execution_trace"]["api_calls_count"] == 1

    res_scored = execute_live_benchmark_task(
        task_type="classification",
        model_name="gemini-3.8-flash",
        separation_approach="two_stage_bbox_guided_nms",
        mode="offline",
        connect_sample_gt=True,
    )
    assert res_scored["accuracy"]["accuracy_status"] == "EVALUATED_AGAINST_GT"
    assert res_scored["accuracy"]["detection_f1"] == pytest.approx(1.0)
    assert res_scored["execution_trace"]["api_calls_count"] == 2


def test_more_than_eight_attributes_and_grouped_vlm_calls(tmp_path: Path):
    """Verify >8 attributes (e.g. 12 attributes) can be configured, predicted in 1 call or grouped VLM calls, and scored in per_attribute_accuracy."""
    from shelf_benchmark import CustomAttributeSpec
    from shelf_benchmark.models import GroundTruthProductItem, ImageGroundTruth

    def multi_attr_handler(prompt: str, image_uri: str, response_schema: Any = None) -> Dict[str, Any]:
        return {
            "total_classified_products": 1,
            "distinct_brands_found": ["Brand_A"],
            "classified_products": [
                {
                    "product_index": 1,
                    "bbox_2d": [100, 100, 300, 200],
                    "category": "Skin Cleansing",
                    "subcategory": "Body Wash",
                    "brand": "Brand_A",
                    "product_name": "Brand_A Moisture Body Wash",
                    "variant": "Deep Moisture",
                    "packaging_type": "bottle",
                    "pack_type": "Single",
                    "size": "250ml",
                    "extra_attributes": {
                        "price_tag_visible": "true",
                        "promo_callout": "20% Extra",
                        "facing_orientation": "front_straight",
                        "shelf_talker_present": "false",
                    },
                }
            ],
        }

    sdk = benchmark_harness(tmp_path, handler=multi_attr_handler, with_ground_truth=False)
    sdk.config.taxonomy.custom_attributes = {
        "price_tag_visible": CustomAttributeSpec(description="Price tag visible", value_type="boolean"),
        "promo_callout": CustomAttributeSpec(description="Promo badge text", value_type="string"),
        "facing_orientation": CustomAttributeSpec(description="Facing orientation", value_type="string"),
        "shelf_talker_present": CustomAttributeSpec(description="Shelf talker attached", value_type="boolean"),
    }
    sdk.config.taxonomy.attribute_call_groups = [
        ["category", "subcategory", "brand", "product_name", "variant"],
        ["packaging_type", "pack_type", "size", "price_tag_visible", "promo_callout", "facing_orientation", "shelf_talker_present"],
    ]
    assert len(sdk.config.taxonomy.all_attribute_names) == 12

    gt = ImageGroundTruth(
        image_id=OFFLINE_IMAGE_URI,
        total_main_shelf_facings=1,
        expected_brands=["Brand_A"],
        items=[
            GroundTruthProductItem(
                item_id=1,
                brand="Brand_A",
                product_name="Brand_A Moisture Body Wash",
                category="Skin Cleansing",
                subcategory="Body Wash",
                variant="Deep Moisture",
                packaging_type="bottle",
                pack_type="Single",
                size="250ml",
                bbox_2d=[100, 100, 300, 200],
                extra_attributes={
                    "price_tag_visible": "true",
                    "promo_callout": "20% Extra",
                    "facing_orientation": "front_straight",
                    "shelf_talker_present": "false",
                },
            )
        ],
    )

    summary = sdk.run_suite(
        tasks=["classification"],
        approaches=["configurable_multi_attribute_vlm"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
        ground_truth=gt,
    )
    res = summary["results"][0]
    assert res.row_level_items[0].extra_attributes["promo_callout"] == "20% Extra"
    assert res.accuracy.per_attribute_accuracy["promo_callout"] == pytest.approx(1.0)
    assert res.accuracy.per_attribute_accuracy["facing_orientation"] == pytest.approx(1.0)
    assert len(res.accuracy.per_attribute_accuracy) == 12
    assert res.accuracy.macro_attribute_accuracy == pytest.approx(1.0)


def test_two_thousand_brands_scale_without_prompt_bloat():
    """Verify 2,000+ brands never bloat the VLM prompt and resolve in O(1) via resolve_brand_against_catalog."""
    from shelf_benchmark import TaxonomyConfig
    from shelf_benchmark.tasks.classification import build_classification_prompt
    from shelf_benchmark.tasks.facing_utils import check_is_hul_brand, resolve_brand_against_catalog

    brands_2000 = [f"CatalogBrand_{i:04d}" for i in range(1, 2001)]
    tax = TaxonomyConfig(
        brand_extraction_mode="open_vocabulary_generative",
        hul_brands=brands_2000[:1000],
        non_hul_brands=brands_2000[1000:],
    )
    prompt = build_classification_prompt(tax)
    assert "CatalogBrand_1500" not in prompt, "2,000 brands must never be dumped into the VLM prompt"
    assert "open-vocabulary generative extraction" in prompt

    # Punctuation variants fold: "CatalogBrand-0742" / "CatalogBrand 0742" all normalize to the
    # same token run as the catalog's "CatalogBrand_0742".
    for variant in ("CatalogBrand_0742", "CatalogBrand-0742", "catalogbrand 0742", "CatalogBrand.0742"):
        assert resolve_brand_against_catalog(variant, tax) == "CatalogBrand_0742", variant

    # Brand extraction from a longer generated string folds to the catalog entry.
    assert resolve_brand_against_catalog("CatalogBrand_0742 Extra Moisturising", tax) == "CatalogBrand_0742"

    # Known, deliberate limitation of the single shared normalizer: apostrophes are deleted rather
    # than treated as possessive markers, because real brands are spelled both ways ("Pond's" and
    # "Ponds" must match, and they do). The cost is that a spurious possessive inside a
    # multi-token brand does NOT fold, and the resolver leaves the string alone rather than
    # guessing. The resolver and the scorer now agree on that, which is the point -- previously
    # the resolver rewrote the prediction under one rule and the scorer graded it under another.
    assert resolve_brand_against_catalog("CatalogBrand's 0742", tax) == "CatalogBrand's 0742"

    from shelf_benchmark.text_normalization import normalize_text

    assert normalize_text("Pond's") == normalize_text("Ponds")

    # Whole-token matching: a catalog brand must not be inferred from an incidental substring.
    narrow = TaxonomyConfig(hul_brands=["Lux"], non_hul_brands=[])
    assert resolve_brand_against_catalog("Deluxe", narrow) == "Deluxe"
    assert check_is_hul_brand("Deluxe", narrow) is False

    assert check_is_hul_brand("CatalogBrand_0742", tax) is True
    assert check_is_hul_brand("CatalogBrand_1850", tax) is False


def test_onboarding_audit_fixes_contract(tmp_path: Path):
    """Pin audit fixes: plugin token bookkeeping keys, ctx.compute_cost, multi-shelf vertical depth NMS, and sample_ground_truth.json placeholder alignment."""
    from shelf_benchmark.approaches.class_agnostic_visual_embedding.stage3_vector_search_matcher import (
        clear_catalog_cache,
    )
    from shelf_benchmark.geometry import deduplicate_depth_stacked_facings
    from shelf_benchmark.models import TokenUsageMetrics

    # 1. Multi-shelf vertical check in Depth NMS: upper shelf item (y=100..300) and lower shelf item
    #    (y=400..600) in the exact same X column (x=100..200) must BOTH survive even if both omit shelf_row.
    multi_row_boxes = [
        {"bbox_2d": [100, 100, 300, 200], "shelf_row": "middle", "brand": "TopShelfBrand"},
        {"bbox_2d": [400, 100, 600, 200], "shelf_row": "middle", "brand": "MiddleShelfBrand"},
        {"bbox_2d": [360, 102, 460, 198], "shelf_row": "middle", "brand": "BackRowBehindMiddle"},
    ]
    kept, filtered = deduplicate_depth_stacked_facings(multi_row_boxes)
    assert len(kept) == 2
    assert filtered == 1
    assert {b["brand"] for b in kept} == {"TopShelfBrand", "MiddleShelfBrand"}

    # 2. Plugin bookkeeping keys (_input_tokens, _thinking_tokens, _output_tokens) and ctx.compute_cost()
    class TokenBookkeepingPlugin(SimpleShelfApproachPlugin):
        @property
        def approach_id(self) -> str:
            return "token_bookkeeping_plugin"

        @property
        def display_name(self) -> str:
            return "Token Bookkeeping Plugin"

        @property
        def category(self) -> str:
            return "two_stage_vlm"

        @property
        def stages_description(self) -> List[str]:
            return ["Stage 1: Test"]

        def detect_and_classify(
            self,
            ctx: CommonLayerContext,
            model_name: str,
            record: ShelfAssociationRecord,
        ) -> List[Dict[str, Any]]:
            cost_check = ctx.compute_cost(
                TokenUsageMetrics(input_tokens=100, thinking_tokens=20, output_tokens=50, total_tokens=170),
                model_name=model_name,
                product_count=1,
            )
            assert cost_check.cost_per_shelf_image_usd > 0
            return [
                {
                    "bbox_2d": [100, 100, 300, 200],
                    "brand": "Brand_A",
                    "product_name": "Brand_A Radiance Daily Cleanser",
                    "_input_tokens": 220,
                    "_thinking_tokens": 30,
                    "_output_tokens": 85,
                    "_depth_filtered": 1,
                }
            ]

    GLOBAL_APPROACH_REGISTRY.register(TokenBookkeepingPlugin())
    res = run_offline_approach(tmp_path, "token_bookkeeping_plugin", with_ground_truth=True)
    assert res.tokens.input_tokens == 220
    assert res.tokens.thinking_tokens == 30
    assert res.tokens.output_tokens == 85
    assert res.tokens.total_tokens == 335
    assert res.cost.vertex_ai_payg_tokens_usd > 0
    assert res.accuracy.depth_duplicates_filtered == 1

    # 3. Offline reference catalog support in class_agnostic_visual_embedding
    clear_catalog_cache()
    sdk = benchmark_harness(tmp_path / "cat_offline", with_ground_truth=True)
    sdk.config.embeddings.reference_catalog.source_uri = "configs/demo_visual_prototypes.json"
    cat_summary = sdk.run_suite(
        tasks=["classification"],
        approaches=["class_agnostic_visual_embedding"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )
    assert cat_summary["results"][0].status == "SUCCESS"
    assert cat_summary["results"][0].raw_output["catalog_status"] == "MATCHED"

    # 4. configs/sample_ground_truth.json carries _placeholder_alert and aligns with perfect_prediction_payload()
    sample_gt_raw = json.loads(Path("configs/sample_ground_truth.json").read_text(encoding="utf-8"))
    assert "_placeholder_alert" in sample_gt_raw
    sdk_gt = benchmark_harness(tmp_path / "sample_gt_align", with_ground_truth=False)
    sdk_gt.connect_ground_truth(
        provider_type="json",
        source_uri="configs/sample_ground_truth.json",
    )
    gt_summary = sdk_gt.run_suite(
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )
    assert gt_summary["results"][0].accuracy.detection_f1 == pytest.approx(1.0)
    assert gt_summary["results"][0].accuracy.brand_classification_accuracy == pytest.approx(1.0)


def test_pure_detection_plugin_and_decoupled_stage1_stage2_pairing(tmp_path: Path):
    """Verify pure bounding-box detector plugins are not penalized on classification attributes and can feed `prior_detection` into Stage-2 classifiers."""
    from shelf_benchmark import register_approach_function

    @register_approach_function(
        approach_id="unit_test_pure_box_detector",
        display_name="Unit Test Pure BBox Detector",
        task_type="detection",
    )
    def run_pure_detector(ctx: CommonLayerContext, model_name: str, shelf_image_uri: str):
        raw_boxes = [
            {"bbox_2d": [100, 100, 300, 200], "confidence": 0.95, "_input_tokens": 110, "_output_tokens": 40},
            {"bbox_2d": [100, 220, 300, 320], "confidence": 0.93},
            {"bbox_2d": [400, 100, 600, 200], "confidence": 0.91},
        ]
        kept, filtered = ctx.deduplicate_depth_stacked_facings(raw_boxes)
        if kept:
            kept[0]["_depth_filtered"] = filtered
        return kept

    # 1. Pure detector plugin runs with task_type='detection' and has None (not 0.0) for unpredicted attributes
    det_res = run_offline_approach(tmp_path / "det_only", "unit_test_pure_box_detector", with_ground_truth=True)
    assert det_res.task_type == "detection"
    assert det_res.accuracy.detection_f1 == pytest.approx(1.0)
    assert det_res.accuracy.per_attribute_accuracy == {}
    assert det_res.accuracy.macro_attribute_accuracy is None
    assert det_res.accuracy.brand_classification_accuracy is None
    assert det_res.accuracy.product_classification_accuracy is None
    assert det_res.row_level_items[0].shelf_row == "top"
    assert det_res.row_level_items[2].shelf_row == "middle"

    # 2. Pair Stage-1 custom detector (`detector_approach`) with Stage-2 classifier (`two_stage_bbox_guided_nms`)
    sdk = benchmark_harness(tmp_path / "decoupled_pair", with_ground_truth=True)
    paired_summary = sdk.run_suite(
        tasks=["detection", "classification"],
        detector_approach="unit_test_pure_box_detector",
        approaches=["two_stage_bbox_guided_nms"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )
    assert len(paired_summary["results"]) == 2
    s1_res, s2_res = paired_summary["results"]
    assert s1_res.separation_approach == "unit_test_pure_box_detector"
    assert s2_res.separation_approach == "two_stage_bbox_guided_nms"
    assert s2_res.accuracy.detection_f1 == pytest.approx(1.0)
    # Stage 2 folds Stage 1's tokens (110+40=150) into its own Stage-2 classification tokens (350+25+160=535 -> 685)
    assert s2_res.tokens.total_tokens == s1_res.tokens.total_tokens + 535


def test_plugin_retry_policy_graceful_error_and_unreported_callable_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify plugins run through PipelineExecutor.run() (retries + ERROR status parity), custom_callable does not fabricate tokens when omitted, depth-NMS handles mislabeled row strings, and SHELF_BENCH_ISOLATE_RUNS works."""
    from shelf_benchmark import UniversalModelSpec
    from shelf_benchmark.geometry import deduplicate_depth_stacked_facings
    from shelf_benchmark.pipeline import RetryPolicy
    from shelf_benchmark.reporting.generator import BenchmarkReportGenerator

    # 1. Plugin retry policy succeeds on 2nd attempt
    attempts = {"count": 0}

    class FlakyRetryPlugin(SimpleShelfApproachPlugin):
        retry_policy = RetryPolicy(max_attempts=2, backoff_base_seconds=0.0)

        @property
        def approach_id(self) -> str:
            return "unit_test_flaky_retry_plugin"

        @property
        def display_name(self) -> str:
            return "Flaky Retry Plugin"

        @property
        def category(self) -> str:
            return "two_stage_vlm"

        @property
        def stages_description(self) -> List[str]:
            return ["Stage 1: Retryable call"]

        def detect_and_classify(
            self, ctx: CommonLayerContext, model_name: str, record: ShelfAssociationRecord
        ) -> List[Dict[str, Any]]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("transient network glitch")
            return [{"bbox_2d": [100, 100, 300, 200], "confidence": 0.95}]

    GLOBAL_APPROACH_REGISTRY.register(FlakyRetryPlugin())
    retry_res = run_offline_approach(tmp_path / "retry_ok", "unit_test_flaky_retry_plugin", with_ground_truth=True)
    assert retry_res.status == "SUCCESS"
    assert retry_res.task_type == "detection"
    assert attempts["count"] == 2

    # 2. Custom callable that omits `_token_usage` reports 0 tokens and `token_usage_reported=False`
    sdk = benchmark_harness(tmp_path / "unreported_tok", with_ground_truth=False)
    sdk.register_model(
        UniversalModelSpec(
            model_id="external-no-token-model",
            provider_family="custom_callable",
            custom_handler=lambda prompt, uri, schema: {
                "total_classified_products": 1,
                "distinct_brands_found": ["Brand_A"],
                "classified_products": [
                    {
                        "product_index": 1,
                        "bbox_2d": [100, 100, 300, 200],
                        "brand": "Brand_A",
                        "product_name": "Brand_A Radiance Daily Cleanser",
                        "confidence": 0.92,
                    }
                ],
            },
        )
    )
    unrep_summary = sdk.run_suite(
        models=["external-no-token-model"],
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
        shelf_image_uris=[OFFLINE_IMAGE_URI, OFFLINE_IMAGE_URI],
        max_workers=2,
    )
    assert len(unrep_summary["results"]) == 2
    assert unrep_summary["results"][0].tokens.total_tokens == 0
    assert unrep_summary["results"][0].raw_output["token_usage_reported"] is False

    # 3. 1D Depth-NMS suppresses back-row unit even when VLM mislabels its shelf_row as 'top' behind a 'middle' front unit
    kept, removed = deduplicate_depth_stacked_facings(
        [
            {"bbox_2d": [320, 100, 520, 200], "shelf_row": "middle", "brand": "FrontUnit"},
            {"bbox_2d": [300, 105, 480, 195], "shelf_row": "top", "brand": "BackUnitMislabeledRow"},
            {"bbox_2d": [50, 100, 220, 200], "shelf_row": "top", "brand": "GenuineUpperShelfUnit"},
        ]
    )
    assert removed == 1
    assert [it["brand"] for it in kept] == ["GenuineUpperShelfUnit", "FrontUnit"]

    # 4. SHELF_BENCH_ISOLATE_RUNS=1 enables run isolation via environment variable
    monkeypatch.setenv("SHELF_BENCH_ISOLATE_RUNS", "1")
    gen = BenchmarkReportGenerator.from_config(BenchmarkConfig())
    assert gen.isolate_runs is True


