"""Separated Task 1: Front-Facing Product Detection (`ProductDetectionTask`).

Detects ONLY distinct front-facing product slots on a retail shelf image (`[ymin, xmin, ymax, xmax]`
in 0..1000 coordinates) and applies deterministic Geometric Facing Depth De-duplication
(`deduplicate_depth_stacked_facings`) so products stacked behind the front unit in the same
facing column are NEVER counted as multiple products.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from google.genai import types

from shelf_benchmark.evaluation.cost import extract_token_usage
from shelf_benchmark.models import (
    ProductDetectionOutput,
    RowLevelReportItem,
    TokenUsageMetrics,
)
from shelf_benchmark.tasks.base import BaseBenchmarkTask
from shelf_benchmark.tasks.facing_utils import (
    check_is_hul_brand,
    deduplicate_depth_stacked_facings,
    derive_size_bucket_from_bbox,
)

DETECTION_PROMPT = """You are an expert retail computer vision system for shelf facing detection.
Analyze this shelf image and detect every distinct FRONT-FACING product slot on the main middle shelf (ordered strictly from left to right).

CRITICAL FACING RULE:
- Do NOT count multiple products stocked in depth (one behind another) in the same facing column as separate detections!
- We ONLY care about the FRONT-MOST visible product unit in each horizontal facing slot (`1 horizontal facing slot = 1 bounding box`).
- Ignore partial tops or caps of back-row products peeking out from behind a front-row product.

For each distinct front-facing product slot on the main middle shelf:
1. `product_index`: 1-based facing slot index ordered strictly from left to right.
2. `bbox_2d`: Normalized 2D coordinates `[ymin, xmin, ymax, xmax]` (0 to 1000) tightly enclosing ONLY the front-most unit in that facing.
3. `shelf_row`: `"middle"` and `position_on_shelf`: 1-based horizontal facing slot index.
4. `is_front_facing`: `true` (must be front-most unit).
5. `visual_description`: Packaging form factor, color, and visible graphic cues.
6. `preliminary_brand_hint`: Visible brand text read directly from packaging if legible, else `"Unknown"`.
7. `confidence`: Detection confidence (0.0 to 1.0).

Return structured JSON matching the response schema."""


class ProductDetectionTask(BaseBenchmarkTask):
    """Standalone Front-Facing-Only Product Detection benchmark task with Depth NMS."""

    task_type = "detection"

    def invoke_model(
        self,
        model_name: str,
        shelf_image_uri: str,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], TokenUsageMetrics, List[RowLevelReportItem]]:
        client = self.get_client()
        image_part = self.storage.to_genai_part(shelf_image_uri)
        custom_prompt = kwargs.get("prompt") or DETECTION_PROMPT

        response = client.models.generate_content(
            model=model_name,
            contents=[image_part, custom_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProductDetectionOutput,
                temperature=0.1,
            ),
        )

        tokens = extract_token_usage(response)
        raw_text = response.text or "{}"
        parsed_dict = json.loads(raw_text)
        validated = ProductDetectionOutput.model_validate(parsed_dict)

        raw_items = [it.model_dump() for it in validated.detected_products]
        dedup_items, depth_filtered_count = deduplicate_depth_stacked_facings(raw_items)
        all_boxes = [it.get("bbox_2d", [0, 0, 0, 0]) for it in dedup_items]

        rows: List[RowLevelReportItem] = []
        for idx, item in enumerate(dedup_items, start=1):
            box = item.get("bbox_2d") or [0, 0, 0, 0]
            brand_hint = item.get("preliminary_brand_hint") or ""
            rule_size = derive_size_bucket_from_bbox(box, all_boxes, "tube", "")
            rows.append(
                RowLevelReportItem(
                    run_id="",
                    trace_id="",
                    span_id="",
                    task_type=self.task_type,
                    separation_approach="single_pass_facing_nms",
                    model_name=model_name,
                    shelf_image_uri=shelf_image_uri,
                    start_time="",
                    end_time="",
                    image_latency_ms=0.0,
                    product_index=idx,
                    shelf_row=item.get("shelf_row", "middle"),
                    position_on_shelf=idx,
                    bbox_ymin=box[0],
                    bbox_xmin=box[1],
                    bbox_ymax=box[2],
                    bbox_xmax=box[3],
                    predicted_category="Detected Facing",
                    predicted_subcategory="Shelf Facing",
                    predicted_brand=brand_hint,
                    is_hul_brand=check_is_hul_brand(brand_hint),
                    predicted_variant=item.get("visual_description", ""),
                    predicted_packaging="tube",
                    predicted_pack_type="Single",
                    predicted_size=rule_size,
                    rule_derived_size_bucket=rule_size,
                    predicted_product_name=item.get("visual_description", ""),
                    confidence=float(item.get("confidence", 0.95)),
                )
            )

        out_dump = {
            "total_raw_detections": len(raw_items),
            "depth_duplicates_filtered": depth_filtered_count,
            "total_detected_products": len(dedup_items),
            "detected_products": dedup_items,
        }
        return out_dump, tokens, rows
