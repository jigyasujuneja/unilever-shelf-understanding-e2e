"""Separated Task 2: 7-Dimension HUL Product Classification & Multi-Approach Bounding-Box Separation (`ProductClassificationTask`).

Supports the configurable 7-Dimension Retail Classification Taxonomy (configured in the packaged `shelf_benchmark/_resources/taxonomy.yaml`):
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
from pathlib import Path
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
    resolve_brand_against_catalog,
)


def build_classification_prompt(
    taxonomy: Optional[TaxonomyConfig] = None,
    attribute_subset: Optional[List[str]] = None,
) -> str:
    """Dynamically builds the N-Dimension classification prompt (8 core + unlimited custom_attributes) from `TaxonomyConfig`."""
    tax = taxonomy or TaxonomyConfig.from_yaml_or_defaults()
    cats_str = ", ".join(f"`{c}`" for c in tax.categories)
    subcats_str = ", ".join(f"`{s}`" for s in tax.subcategories[:10])
    pkg_str = ", ".join(f"`{p}`" for p in tax.packaging_types)
    pack_str = " or ".join(f'`"{pt}"`' for pt in tax.pack_types)
    sizes_str = ", ".join(f"`{sb}`" for sb in tax.size_bucket_labels)

    total_catalog_brands = len(tax.hul_brands or []) + len(tax.non_hul_brands or [])
    if (
        tax.brand_extraction_mode == "closed_set_taxonomy"
        and 0 < total_catalog_brands <= tax.max_prompt_brands
    ):
        all_b = list(tax.hul_brands or []) + list(tax.non_hul_brands or [])
        brand_instruction = (
            f"Brand name classified from configured taxonomy brands ({', '.join(all_b)}) and set `is_hul_brand` (`true`/`false`)."
        )
    else:
        brand_instruction = (
            "Determine and generate the exact brand name directly from the visible package logo and typography "
            "(open-vocabulary generative extraction; no predefined brand list required, scaling to 2,000+ brands) "
            "and set `is_hul_brand` (`true`/`false`)."
        )

    extra_lines: List[str] = []
    idx_num = 8
    for attr_name, spec in (tax.custom_attributes or {}).items():
        if attribute_subset is not None and attr_name not in attribute_subset:
            continue
        allowed_str = (
            f" (allowed values: {', '.join(f'`{v}`' for v in spec.allowed_values)})"
            if spec.allowed_values
            else ""
        )
        extra_lines.append(
            f"{idx_num}. `extra_attributes.{attr_name}` ({spec.value_type}): {spec.description}{allowed_str}"
        )
        idx_num += 1

    extra_block = (
        "\nAdditional Configured Attributes (populate inside `extra_attributes` dict on each product):\n"
        + "\n".join(extra_lines)
        if extra_lines
        else ""
    )
    subset_note = (
        f"\nTARGET ATTRIBUTE GROUP FOR THIS PASS: Focus on accurately extracting {attribute_subset}."
        if attribute_subset
        else ""
    )

    return f"""You are an expert retail shelf product classification model using the configurable N-Dimension Retail Taxonomy.
Visually inspect this shelf image and classify EVERY distinct FRONT-FACING product slot on the main middle shelf (ordered strictly from left to right).

CRITICAL FACING RULE:
- Do NOT count products stacked in depth (behind the front-most product in the same facing column) as separate items!
- Only classify the FRONT-MOST visible unit in each horizontal facing slot (`1 facing slot = 1 product entry`).

For EVERY front-facing product slot on the main middle shelf, extract all configured Taxonomy Dimensions:
1. `category`: Choose from configured categories ({cats_str}).
2. `subcategory`: Choose or infer subcategory (e.g., {subcats_str}).
3. `brand`: {brand_instruction}
4. `variant`: Specific variant / active ingredient / product line read zero-shot from the packaging.
5. `packaging_type`: Choose from configured packaging types ({pkg_str}).
6. `pack_type`: {pack_str}.
7. `size`: Visible size cue or configured size bucket ({sizes_str}).{extra_block}{subset_note}
Also provide `bbox_2d` (`[ymin, xmin, ymax, xmax]` 0..1000), `product_name`, `extra_attributes`, and `confidence` (0.0 to 1.0)."""





class ProductClassificationTask(BaseBenchmarkTask):
    """N-Dimension Product Classification task supporting single-step, two-stage, and configurable multi-attribute VLM approaches."""

    task_type = "classification"
    default_separation_approach = "single_pass_full_shelf"

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
            detected_boxes, depth_filtered = deduplicate_depth_stacked_facings(
                raw_det,
                x_overlap_threshold=self.config.depth_deduplication.x_overlap_threshold,
            )

        call_groups = list(self.config.taxonomy.attribute_call_groups or [])
        if separation_approach == "configurable_multi_attribute_vlm" and len(call_groups) >= 2:
            # Multi-call grouped attribute extraction: run 1 VLM call per attribute group and merge per facing
            merged_items: List[Dict[str, Any]] = []
            total_tokens = stage1_tokens
            for g_idx, group_attrs in enumerate(call_groups, start=1):
                group_prompt = build_classification_prompt(
                    self.config.taxonomy, attribute_subset=list(group_attrs)
                )
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[image_part, group_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ProductClassificationOutput,
                        temperature=0.1,
                    ),
                )
                total_tokens = self._sum_tokens(total_tokens, extract_token_usage(resp))
                parsed = ProductClassificationOutput.model_validate(json.loads(resp.text or "{}"))
                group_items = [it.model_dump() for it in parsed.classified_products]
                if g_idx == 1:
                    merged_items = group_items
                else:
                    for idx_item, g_item in enumerate(group_items):
                        if idx_item < len(merged_items):
                            target = merged_items[idx_item]
                            for attr_key in group_attrs:
                                if attr_key in target and g_item.get(attr_key):
                                    target[attr_key] = g_item[attr_key]
                                elif g_item.get("extra_attributes", {}).get(attr_key) is not None:
                                    target.setdefault("extra_attributes", {})[attr_key] = g_item["extra_attributes"][attr_key]
            dedup_cls_items, depth_filtered = deduplicate_depth_stacked_facings(
                merged_items,
                x_overlap_threshold=self.config.depth_deduplication.x_overlap_threshold,
            )
            validated = ProductClassificationOutput(
                total_classified_products=len(dedup_cls_items),
                distinct_brands_found=sorted({str(it.get("brand", "")) for it in dedup_cls_items if it.get("brand")}),
                classified_products=[],
            )
        else:
            contents_list: List[Any] = [image_part]
            prompt = kwargs.get("prompt") or build_classification_prompt(self.config.taxonomy)

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
                    + " into all configured Taxonomy dimensions."
                )
            elif separation_approach == "two_stage_physical_crop_per_facing" and detected_boxes:
                crop_paths, montage_bytes = crop_detected_facings(
                    storage=self.storage,
                    shelf_image_uri=shelf_image_uri,
                    detected_items=detected_boxes,
                    output_crop_dir=Path(self.config.reporting.output_dir) / "crops",
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
                    + ") to read fine-print packaging text and classify each facing into all configured Taxonomy dimensions."
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
            if separation_approach in (
                "single_pass_full_shelf",
                "open_vocab_brand_plus_catalog_resolver",
                "configurable_multi_attribute_vlm",
            ):
                dedup_cls_items, depth_filtered = deduplicate_depth_stacked_facings(
                    raw_cls_items,
                    x_overlap_threshold=self.config.depth_deduplication.x_overlap_threshold,
                )
            else:
                dedup_cls_items = raw_cls_items

        all_boxes = [it.get("bbox_2d", [0, 0, 0, 0]) for it in dedup_cls_items]

        rows: List[RowLevelReportItem] = []
        for idx, item in enumerate(dedup_cls_items, start=1):
            box = item.get("bbox_2d") or [0, 0, 0, 0]
            if len(box) < 4:
                box = [0, 0, 0, 0]
            pkg = item.get("packaging_type") or ""
            model_size = item.get("size") or ""
            rule_size = derive_size_bucket_from_bbox(
                box, all_boxes, pkg, model_size, taxonomy=self.config.taxonomy
            )
            brand_val = resolve_brand_against_catalog(
                item.get("brand") or "", taxonomy=self.config.taxonomy
            )
            hul_flag = check_is_hul_brand(
                brand_val,
                taxonomy=self.config.taxonomy,
                model_predicted=item.get("is_hul_brand"),
            )
            # Align the crop by the facing number the prompt actually numbered (#1..#N), not by
            # position in the response list. The crops are produced in `detected_boxes` order
            # (which `crop_detected_facings` re-sorts), while these rows iterate the model's
            # response order, and `two_stage_physical_crop_per_facing` does not dedup -- so
            # `crop_paths[idx - 1]` attached the wrong crop image to a row whenever the model
            # reordered or dropped a facing.
            facing_no = item.get("product_index")
            crop_slot = (int(facing_no) - 1) if isinstance(facing_no, int) and facing_no > 0 else (idx - 1)
            crop_path = crop_paths[crop_slot] if 0 <= crop_slot < len(crop_paths) else None
            extra_attrs = dict(item.get("extra_attributes") or {})

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
                    # No invented fallbacks. These used to default to "Skin Care" / "Face Wash" /
                    # "Single" / a synthesised "<brand> <variant>" product name, all of which the
                    # scorer then compared against ground truth as if the model had said them.
                    predicted_category=item.get("category") or "",
                    predicted_subcategory=item.get("subcategory") or "",
                    predicted_brand=brand_val,
                    is_hul_brand=hul_flag,
                    predicted_variant=item.get("variant") or "",
                    predicted_packaging=pkg,
                    predicted_pack_type=item.get("pack_type") or "",
                    predicted_size=model_size,
                    rule_derived_size_bucket=rule_size,
                    predicted_product_name=item.get("product_name") or "",
                    confidence=float(item.get("confidence") or 0.0),
                    extra_attributes=extra_attrs,
                )
            )

        out_dump = validated.model_dump()
        out_dump["separation_approach"] = separation_approach
        out_dump["depth_duplicates_filtered"] = depth_filtered
        out_dump["classified_products"] = dedup_cls_items
        return out_dump, total_tokens, rows
