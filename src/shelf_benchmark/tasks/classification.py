"""Separated Task 2: 7-Dimension HUL Product Classification & Multi-Approach Bounding-Box Separation (`ProductClassificationTask`).

Supports the configurable 7-Dimension Retail Classification Taxonomy (configured in `configs/taxonomy.yaml`):
  1. `Category`: Configured retail categories (e.g., Hair Care, Oral Care, Laundry, Skin Care, Skin Cleansing)
  2. `Subcategory`: Configured subcategories (e.g., Shampoo, Mouthwash, Soaps, Face Wash, Body Wash, Detergent)
  3. `Brand`: Configured HUL portfolio brands & non-HUL brands (+ `is_hul_brand` flag)
  4. `Variant`: Product variant / active ingredient / line read zero-shot from packaging
  5. `Packaging type`: Configured packaging formats (e.g., box, jar, sachet, tube, bottle, pouch, bar)
  6. `Pack type`: Configured pack types (e.g., Single or Multiple)
  7. `Size`: Rule-derived size bucket (combining OCR + bounding-box geometry relative to shelf row median)

Supports benchmarking multiple Bounding-Box Separation Approaches against each other:
  - `single_pass_full_shelf`: Single-pass full-shelf detection + 7-dimension classification + Facing Depth-NMS.
  - `two_stage_bbox_guided_nms`: Stage 1 detects & depth-deduplicates front-facing bounding boxes; Stage 2 classifies each locked-in facing box on the full image.
  - `two_stage_physical_crop_per_facing`: Stage 1 detects & depth-deduplicates front-facing bounding boxes, physically crops each facing (`PIL.Image.crop`) into `reports/crops/` and a numbered crop strip, and Stage 2 classifies the high-res separated crops.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from shelf_benchmark.config import TaxonomyConfig
from shelf_benchmark.evaluation.cost import extract_token_usage
from shelf_benchmark.models import (
    ProductClassificationOutput,
    ProductDetectionOutput,
    RowLevelReportItem,
    TokenUsageMetrics,
)
from shelf_benchmark.tasks.base import BaseBenchmarkTask
from shelf_benchmark.tasks.detection import DETECTION_PROMPT
from shelf_benchmark.tasks.facing_utils import (
    check_is_hul_brand,
    crop_detected_facings,
    deduplicate_depth_stacked_facings,
    derive_size_bucket_from_bbox,
)


def build_classification_prompt(taxonomy: Optional[TaxonomyConfig] = None) -> str:
    """Dynamically builds the 7-Dimension classification prompt from centralized `TaxonomyConfig`."""
    tax = taxonomy or TaxonomyConfig.from_yaml_or_defaults()
    cats_str = ", ".join(f"`{c}`" for c in tax.categories)
    subcats_str = ", ".join(f"`{s}`" for s in tax.subcategories[:10])
    hul_sample_str = ", ".join(tax.hul_brands[:12])
    pkg_str = ", ".join(f"`{p}`" for p in tax.packaging_types)
    pack_str = " or ".join(f'`"{pt}"`' for pt in tax.pack_types)
    sizes_str = ", ".join(f"`{sb}`" for sb in tax.size_bucket_labels)

    return f"""You are an expert retail shelf product classification model using the configurable 7-Dimension Retail Taxonomy.
Visually inspect this shelf image and classify EVERY distinct FRONT-FACING product slot on the main middle shelf (ordered strictly from left to right).

CRITICAL FACING RULE:
- Do NOT count products stacked in depth (behind the front-most product in the same facing column) as separate items!
- Only classify the FRONT-MOST visible unit in each horizontal facing slot (`1 facing slot = 1 product entry`).

For EVERY front-facing product slot on the main middle shelf, extract all 7 Taxonomy Dimensions:
1. `category`: Choose from configured categories ({cats_str}).
2. `subcategory`: Choose or infer subcategory (e.g., {subcats_str}).
3. `brand`: Brand name read from the package (identifying whether it belongs to the configured HUL portfolio such as {hul_sample_str}, etc. vs. non-HUL brands) and set `is_hul_brand` (`true`/`false`).
4. `variant`: Specific variant / active ingredient / product line read zero-shot from the packaging.
5. `packaging_type`: Choose from configured packaging types ({pkg_str}).
6. `pack_type`: {pack_str}.
7. `size`: Visible size cue or configured size bucket ({sizes_str}).
Also provide `bbox_2d` (`[ymin, xmin, ymax, xmax]` 0..1000), `product_name`, and `confidence` (0.0 to 1.0)."""


CLASSIFICATION_PROMPT = build_classification_prompt()


class ProductClassificationTask(BaseBenchmarkTask):
    """7-Dimension HUL Product Classification task supporting multiple Bounding-Box Separation approaches."""

    task_type = "classification"

    def _sum_tokens(self, t1: TokenUsageMetrics, t2: TokenUsageMetrics) -> TokenUsageMetrics:
        return TokenUsageMetrics(
            input_tokens=t1.input_tokens + t2.input_tokens,
            thinking_tokens=t1.thinking_tokens + t2.thinking_tokens,
            output_tokens=t1.output_tokens + t2.output_tokens,
            total_tokens=t1.total_tokens + t2.total_tokens,
            cached_tokens=t1.cached_tokens + t2.cached_tokens,
        )

    def invoke_model(
        self,
        model_name: str,
        shelf_image_uri: str,
        separation_approach: str = "single_pass_full_shelf",
        detected_boxes: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], TokenUsageMetrics, List[RowLevelReportItem]]:
        client = self.get_client()
        image_part = self.storage.to_genai_part(shelf_image_uri)
        stage1_tokens = TokenUsageMetrics()
        crop_paths: List[str] = []
        depth_filtered = 0

        # Stage 1 for two-stage separation approaches: run explicit Front-Facing Detection + Depth-NMS first
        if separation_approach in ("two_stage_bbox_guided_nms", "two_stage_physical_crop_per_facing") and not detected_boxes:
            det_resp = client.models.generate_content(
                model=model_name,
                contents=[image_part, DETECTION_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ProductDetectionOutput,
                    temperature=0.1,
                ),
            )
            stage1_tokens = extract_token_usage(det_resp)
            det_parsed = ProductDetectionOutput.model_validate(json.loads(det_resp.text or "{}"))
            raw_det = [d.model_dump() for d in det_parsed.detected_products]
            detected_boxes, depth_filtered = deduplicate_depth_stacked_facings(raw_det)

        contents_list: List[Any] = [image_part]
        prompt = kwargs.get("prompt") or CLASSIFICATION_PROMPT

        if separation_approach == "two_stage_bbox_guided_nms" and detected_boxes:
            compact_boxes = [
                {"facing_index": i + 1, "bbox_2d": b.get("bbox_2d"), "hint": b.get("preliminary_brand_hint")}
                for i, b in enumerate(detected_boxes)
            ]
            prompt += (
                f"\n\nSTAGE 1 DETECTED & DEPTH-DEDUPLICATED {len(compact_boxes)} FRONT FACINGS:\n"
                + json.dumps(compact_boxes)
                + "\nClassify each of these exact front-facing bounding boxes in order from 1 to "
                + str(len(compact_boxes))
                + " into the 7 HUL Taxonomy dimensions."
            )
        elif separation_approach == "two_stage_physical_crop_per_facing" and detected_boxes:
            crop_paths, montage_bytes = crop_detected_facings(
                storage=self.storage,
                shelf_image_uri=shelf_image_uri,
                detected_items=detected_boxes,
                model_tag=f"{model_name}_{separation_approach}",
            )
            montage_part = types.Part.from_bytes(data=montage_bytes, mime_type="image/png")
            contents_list.append(montage_part)
            compact_boxes = [
                {"facing_index": i + 1, "bbox_2d": b.get("bbox_2d")}
                for i, b in enumerate(detected_boxes)
            ]
            prompt += (
                f"\n\nSTAGE 1 PHYSICALLY CROPPED {len(compact_boxes)} FRONT FACINGS (see second image showing high-resolution numbered crops #1 through #{len(compact_boxes)} alongside their shelf bounding boxes):\n"
                + json.dumps(compact_boxes)
                + "\nUse the zoomed-in numbered crops (#1..#"
                + str(len(compact_boxes))
                + ") to read fine-print packaging text and classify each facing into all 7 HUL Taxonomy dimensions."
            )

        contents_list.append(prompt)

        response = client.models.generate_content(
            model=model_name,
            contents=contents_list,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProductClassificationOutput,
                temperature=0.1,
            ),
        )

        stage2_tokens = extract_token_usage(response)
        total_tokens = self._sum_tokens(stage1_tokens, stage2_tokens)

        raw_text = response.text or "{}"
        parsed_dict = json.loads(raw_text)
        validated = ProductClassificationOutput.model_validate(parsed_dict)

        raw_cls_items = [it.model_dump() for it in validated.classified_products]
        if separation_approach == "single_pass_full_shelf":
            dedup_cls_items, depth_filtered = deduplicate_depth_stacked_facings(raw_cls_items)
        else:
            dedup_cls_items = raw_cls_items

        all_boxes = [it.get("bbox_2d", [0, 0, 0, 0]) for it in dedup_cls_items]

        rows: List[RowLevelReportItem] = []
        for idx, item in enumerate(dedup_cls_items, start=1):
            box = item.get("bbox_2d") or [0, 0, 0, 0]
            if len(box) < 4:
                box = [0, 0, 0, 0]
            pkg = item.get("packaging_type") or "tube"
            model_size = item.get("size") or ""
            rule_size = derive_size_bucket_from_bbox(box, all_boxes, pkg, model_size)
            brand_val = item.get("brand") or ""
            hul_flag = check_is_hul_brand(brand_val)
            crop_path = crop_paths[idx - 1] if idx - 1 < len(crop_paths) else None

            rows.append(
                RowLevelReportItem(
                    run_id="",
                    trace_id="",
                    span_id="",
                    task_type=self.task_type,
                    separation_approach=separation_approach,
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
                    crop_image_path=crop_path,
                    predicted_category=item.get("category", "Skin Care"),
                    predicted_subcategory=item.get("subcategory", "Face Wash"),
                    predicted_brand=brand_val,
                    is_hul_brand=hul_flag,
                    predicted_variant=item.get("variant", ""),
                    predicted_packaging=pkg,
                    predicted_pack_type=item.get("pack_type", "Single"),
                    predicted_size=model_size,
                    rule_derived_size_bucket=rule_size,
                    predicted_product_name=item.get("product_name") or f"{brand_val} {item.get('variant', '')}".strip(),
                    confidence=float(item.get("confidence", 0.95)),
                )
            )

        out_dump = validated.model_dump()
        out_dump["separation_approach"] = separation_approach
        out_dump["depth_duplicates_filtered"] = depth_filtered
        out_dump["classified_products"] = dedup_cls_items
        return out_dump, total_tokens, rows
