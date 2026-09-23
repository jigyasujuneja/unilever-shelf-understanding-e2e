"""Unit tests for BenchmarkReportGenerator and OpenTelemetryBenchmarkLogger."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from shelf_benchmark.models import (
    AccuracyMetrics,
    CostMetrics,
    RowLevelReportItem,
    TaskExecutionResult,
    TokenUsageMetrics,
)
from shelf_benchmark.reporting.generator import BenchmarkReportGenerator

# Everything in this module runs without network or GCP credentials.
pytestmark = pytest.mark.offline


def test_report_generator_outputs_all_formats(tmp_path: Path):
    """Verify row-level CSV/JSON, summary CSV/JSON, and Markdown report generation."""
    row = RowLevelReportItem(
        run_id="run-01",
        trace_id="trace-1234",
        span_id="span-5678",
        task_type="classification",
        model_name="gemini-3.8-flash",
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
        store_id="STORE-01",
        start_time="2026-09-21T11:00:00Z",
        end_time="2026-09-21T11:00:05Z",
        image_latency_ms=5000.0,
        product_index=1,
        shelf_row="middle",
        position_on_shelf=1,
        bbox_ymin=535,
        bbox_xmin=52,
        bbox_ymax=782,
        bbox_xmax=106,
        predicted_brand="Brand_A",
        predicted_product_name="Brand_A Bright Miracle Detox Facewash",
        predicted_variant="Charcoal Black Tube",
        predicted_category="Face Wash",
        predicted_packaging="Tube",
        confidence=0.99,
        gt_item_id=3,
        gt_brand="Brand_A",
        gt_product_name="Brand_A Bright Miracle Detox Facewash",
        gt_sku_id="UNI-PONDS-DETOX-002",
        iou_with_gt=0.92,
        brand_correct=True,
        product_correct=True,
        input_tokens=2279,
        thinking_tokens=2191,
        output_tokens=2515,
        total_tokens=6985,
        cost_per_shelf_image_usd=0.007628,
        cost_per_product_usd=0.000424,
    )
    result = TaskExecutionResult(
        run_id="run-01",
        trace_id="trace-1234",
        span_id="span-5678",
        task_type="classification",
        model_name="gemini-3.8-flash",
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
        start_time="2026-09-21T11:00:00Z",
        end_time="2026-09-21T11:00:05Z",
        latency_ms=5000.0,
        tokens=TokenUsageMetrics(
            input_tokens=2279,
            thinking_tokens=2191,
            output_tokens=2515,
            total_tokens=6985,
        ),
        cost=CostMetrics(
            input_cost_usd=0.000684,
            thinking_cost_usd=0.000657,
            output_cost_usd=0.006288,
            cost_per_shelf_image_usd=0.007628,
            cost_per_product_usd=0.000424,
            product_count=18,
        ),
        accuracy=AccuracyMetrics(
            ground_truth_available=True,
            ground_truth_count=18,
            predicted_count=18,
            count_accuracy=1.0,
            brand_classification_accuracy=1.0,
            product_classification_accuracy=1.0,
        ),
        row_level_items=[row],
    )

    gen = BenchmarkReportGenerator(output_dir=tmp_path)
    paths = gen.generate_all_reports([result])

    assert Path(paths["row_level_csv"]).exists()
    assert Path(paths["row_level_json"]).exists()
    assert Path(paths["summary_csv"]).exists()
    assert Path(paths["summary_json"]).exists()
    assert Path(paths["markdown_report"]).exists()

    with open(paths["row_level_csv"], encoding="utf-8") as f:
        rows_read = list(csv.DictReader(f))
    assert len(rows_read) == 1
    assert rows_read[0]["predicted_brand"] == "Brand_A"
    assert float(rows_read[0]["cost_per_product_usd"]) == 0.000424

    summary_data = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert summary_data["summary"][0]["model_name"] == "gemini-3.8-flash"
