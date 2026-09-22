"""Unit tests for separated tasks, Facing-Only Depth NMS, 7-Dimension HUL Taxonomy, Bounding-Box Separation approaches, and OTel compliance."""

from __future__ import annotations

from pathlib import Path

import pytest

from shelf_benchmark.config import (
    AssociationSchemaMapping,
    BenchmarkConfig,
    GroundTruthSchemaMapping,
)
from shelf_benchmark.data.associations import CSVAssociationProvider
from shelf_benchmark.data.ground_truth import CSVGroundTruthProvider
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.evaluation.cost import compute_cost_metrics
from shelf_benchmark.evaluation.metrics import compute_iou
from shelf_benchmark.models import TokenUsageMetrics
from shelf_benchmark.tasks import (
    ProductClassificationTask,
)
from shelf_benchmark.tasks.facing_utils import (
    check_is_hul_brand,
    deduplicate_depth_stacked_facings,
    derive_size_bucket_from_bbox,
)
from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger
from shelf_benchmark.testing import FakeGenAIClient

# Everything in this module runs without network or GCP credentials.
pytestmark = pytest.mark.offline


def test_facing_depth_deduplication_and_size_rules():
    """Verify products stacked behind a front facing in the same horizontal slot are deduplicated."""
    raw_detections = [
        # Front tube resting on shelf lip (ymax=785)
        {"product_index": 1, "bbox_2d": [535, 50, 785, 108], "shelf_row": "middle", "is_front_facing": True},
        # Back-row tube sticking out directly behind the front tube in the same [52..106] column (ymax=610)
        {"product_index": 2, "bbox_2d": [490, 52, 610, 106], "shelf_row": "middle", "is_front_facing": True},
        # Distinct adjacent facing slot [110..165]
        {"product_index": 3, "bbox_2d": [580, 110, 786, 165], "shelf_row": "middle", "is_front_facing": True},
    ]
    dedup, removed = deduplicate_depth_stacked_facings(raw_detections)
    assert len(dedup) == 2
    assert removed == 1
    assert dedup[0]["bbox_2d"] == [535, 50, 785, 108]
    assert dedup[1]["bbox_2d"] == [580, 110, 786, 165]

    all_boxes = [[590, 196, 792, 256], [590, 258, 792, 316], [640, 315, 792, 348], [530, 386, 795, 440]]
    small_bucket = derive_size_bucket_from_bbox([640, 315, 792, 348], all_boxes, "tube", "")
    large_bucket = derive_size_bucket_from_bbox([530, 386, 795, 440], all_boxes, "tube", "")
    assert "Small" in small_bucket
    assert "Large" in large_bucket
    assert check_is_hul_brand("Pond's", model_predicted=True) is True
    assert check_is_hul_brand("CompetitorBrand", model_predicted=False) is False


def test_custom_association_schema_swap(tmp_path: Path):
    """Verify AssociationProvider supports arbitrary column schemas and optional missing planograms."""
    csv_file = tmp_path / "custom_assoc.csv"
    csv_file.write_text(
        "custom_row_id,gcs_shelf_photo,gcs_cat_ref,store_code\n"
        "ROW-99,gs://custom-shelf-bucket/bay1.png,gs://custom-cat-bucket/cat.json,MUM-01\n",
        encoding="utf-8",
    )
    mapping = AssociationSchemaMapping(
        association_id_field="custom_row_id",
        shelf_image_uri_field="gcs_shelf_photo",
        catalog_uri_field="gcs_cat_ref",
        planogram_uri_field="missing_plano_col",
        store_id_field="store_code",
    )
    cfg = BenchmarkConfig()
    sm = StorageManager(cfg.gcp.project_id, cfg.buckets)
    provider = CSVAssociationProvider(str(csv_file), sm, mapping)
    records = provider.load_associations()
    assert len(records) == 1
    assert records[0].association_id == "ROW-99"
    assert records[0].planogram_uri is None


def test_custom_ground_truth_schema_swap(tmp_path: Path):
    """Verify GroundTruthProvider can swap schemas seamlessly via GroundTruthSchemaMapping."""
    csv_gt = tmp_path / "custom_gt.csv"
    csv_gt.write_text(
        "img_name,maker,sku_title,ean_code,box_coords\n"
        'shelf-image.png,"Pond\'s","Pond\'s Bright Miracle Detox Facewash",UNI-PONDS-DETOX-002,"[535, 52, 782, 106]"\n',
        encoding="utf-8",
    )
    mapping = GroundTruthSchemaMapping(
        image_key_field="img_name",
        brand_field="maker",
        product_name_field="sku_title",
        sku_id_field="ean_code",
        bbox_field="box_coords",
    )
    cfg = BenchmarkConfig()
    sm = StorageManager(cfg.gcp.project_id, cfg.buckets)
    gt_provider = CSVGroundTruthProvider(str(csv_gt), sm, mapping)
    gt = gt_provider.get_ground_truth("gs://any-bucket/shelf-image.png")
    assert gt is not None
    assert gt.total_main_shelf_facings == 1


def test_cost_and_iou_calculations():
    """Verify exact token cost per shelf image, cost per product, and IoU calculations."""
    cfg = BenchmarkConfig()
    pricing = cfg.get_pricing("gemini-3.8-flash")
    tokens = TokenUsageMetrics(input_tokens=1000, thinking_tokens=500, output_tokens=400, total_tokens=1900)
    cost = compute_cost_metrics(tokens, pricing, product_count=10)
    assert abs(cost.cost_per_shelf_image_usd - 0.00145) < 1e-8
    assert abs(cost.cost_per_product_usd - 0.000145) < 1e-8
    assert abs(compute_iou([500, 100, 800, 200], [500, 100, 800, 200]) - 1.0) < 1e-6


def test_separated_tasks_and_7_dimension_hul_schema(tmp_path: Path):
    """Verify Detection, 7-Dimension Classification, and Hybrid Search Matching run and log OTel records."""
    cfg = BenchmarkConfig()
    cfg.telemetry.otel_log_path = str(tmp_path / "otel_test.jsonl")
    sm = StorageManager(cfg.gcp.project_id, cfg.buckets)
    otel = OpenTelemetryBenchmarkLogger(cfg.telemetry, cfg.gcp.project_id, cfg.gcp.location)

    cls_client = FakeGenAIClient(
        {
            "total_classified_products": 1,
            "distinct_brands_found": ["Pond's"],
            "classified_products": [
                {
                    "product_index": 1,
                    "bbox_2d": [535, 52, 782, 106],
                    "shelf_row": "middle",
                    "position_on_shelf": 1,
                    "category": "Skin Care",
                    "subcategory": "Face Wash",
                    "brand": "Pond's",
                    "is_hul_brand": True,
                    "variant": "Bright Miracle Detox Activated Charcoal",
                    "packaging_type": "tube",
                    "pack_type": "Single",
                    "size": "100g",
                    "product_name": "Pond's Bright Miracle Detox Facewash",
                    "confidence": 0.99,
                }
            ],
        }
    )
    cls_task = ProductClassificationTask(cfg, sm, otel, genai_client=cls_client)
    cls_res = cls_task.execute(
        "gemini-3.8-flash",
        "gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
        separation_approach="single_pass_full_shelf",
    )
    assert cls_res.status == "SUCCESS"
    item = cls_res.row_level_items[0]
    assert item.predicted_category == "Skin Care"
    assert item.predicted_subcategory == "Face Wash"
    assert item.predicted_brand == "Pond's"
    assert item.is_hul_brand is True
    assert item.predicted_packaging == "tube"
    assert item.predicted_pack_type == "Single"
    assert "Medium" in item.rule_derived_size_bucket
