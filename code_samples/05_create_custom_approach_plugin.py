#!/usr/bin/env python3
"""Sample 05: Write and benchmark your own approach with `@register_approach_function`.

Run it (no GCP project, no credentials, no network):

    .venv/bin/python code_samples/05_create_custom_approach_plugin.py

The decorator is the fastest way in: one function, and the suite handles taxonomy validation,
rule-derived size buckets, HUL brand attribution, depth deduplication, cost, OpenTelemetry spans
and every report. Your function signature must be:

    f(ctx: CommonLayerContext, model_name: str, shelf_image_uri: str) -> List[Dict[str, Any]]

Return one dict per detected facing. Recognized keys: `bbox_2d`, `brand`, `product_name`,
`variant`, `category`, `subcategory`, `packaging_type`, `pack_type`, `size`, `matched_sku_id`,
`shelf_row`, `confidence`, `is_hul_brand`, plus the bookkeeping keys `_input_tokens`,
`_thinking_tokens`, `_output_tokens` and `_depth_filtered`.

A function registered this way is a first-class approach: `shelf-benchmark list-approaches`
shows it and `--approaches` accepts it, as long as the module that defines it is imported.
For a permanent approach, promote it to `src/shelf_benchmark/approaches/<your_id>/plugin.py`
(see ENGINEER_ONBOARDING_GUIDE.md).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Dict, List

from shelf_benchmark import register_approach_function
from shelf_benchmark.approaches import CommonLayerContext
from shelf_benchmark.testing import OFFLINE_IMAGE_URI, make_offline_sdk


@register_approach_function(
    approach_id="custom_yolo_plus_gemma_verifier",
    display_name="Custom 2-Stage Detector + Verifier Pipeline",
    category="two_stage_vlm",
    stages_description=[
        "Stage 1: Custom bounding-box proposal generator + front-facing depth deduplication",
        "Stage 2: Custom attribute verifier + rule-derived size bucketing",
    ],
)
def run_custom_yolo_plus_gemma_pipeline(
    ctx: CommonLayerContext,
    model_name: str,
    shelf_image_uri: str,
) -> List[Dict[str, Any]]:
    """Your approach goes here.

    In a real approach this is where you would call your detector and your verifier. The two
    hard-coded candidates below stand in for that, so the sample is deterministic.

    `ctx` gives you the shared helpers, so a plugin and a built-in task cannot disagree about
    geometry or taxonomy:
      ctx.deduplicate_depth_stacked_facings(items, x_overlap_threshold=None)
      ctx.derive_size_bucket_from_bbox(bbox_2d, all_bboxes_on_shelf, packaging_type, model_size_hint)
      ctx.check_is_hul_brand(brand_name, model_predicted=None)
      ctx.load_shelf_image(record)
      ctx.extract_tokens(response)
    """
    raw_candidate_boxes = [
        # Front-most facing on the middle shelf (ymax=795).
        {
            "bbox_2d": [535, 386, 795, 440],
            "shelf_row": "middle",
            "category": "Skin Care",
            "subcategory": "Face Wash",
            "brand": "Brand_A",
            "product_name": "Brand_A Radiance Daily Cleanser",
            "variant": "Bright Beauty Spot-less Glow",
            "packaging_type": "tube",
            "pack_type": "Single",
            "size": "100g",
            "confidence": 0.97,
            "_input_tokens": 220,
            "_thinking_tokens": 30,
            "_output_tokens": 85,
        },
        # Back-row unit stacked in the same column [388..438]. Depth dedup removes it, because a
        # unit hidden behind another unit is not a facing and must not be counted twice.
        {
            "bbox_2d": [495, 388, 610, 438],
            "shelf_row": "middle",
            "category": "Skin Care",
            "subcategory": "Face Wash",
            "brand": "Brand_A",
            "product_name": "Brand_A Radiance Daily Cleanser",
            "variant": "Bright Beauty Spot-less Glow",
            "packaging_type": "tube",
            "pack_type": "Single",
            "size": "100g",
            "confidence": 0.89,
        },
    ]

    # Use the shared helper on `ctx` rather than importing `facing_utils` directly: the helper
    # picks up the run's configured `depth_deduplication.x_overlap_threshold`.
    front_facings, depth_filtered_count = ctx.deduplicate_depth_stacked_facings(raw_candidate_boxes)
    if front_facings:
        front_facings[0]["_depth_filtered"] = depth_filtered_count
    return front_facings


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="shelf-custom-approach-"))
    sdk = make_offline_sdk(work)

    summary = sdk.run_suite(
        models=["gemini-3.8-flash"],  # Unused by this approach; recorded on the row for grouping.
        tasks=["classification"],
        approaches=["custom_yolo_plus_gemma_verifier"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )

    res = summary["results"][0]
    print(f"approach                 : {res.separation_approach}")
    print(f"front facings kept       : {len(res.row_level_items)}")
    print(f"depth duplicates filtered: {res.accuracy.depth_duplicates_filtered}")
    print(f"rule-derived size bucket : {res.row_level_items[0].rule_derived_size_bucket}")
    print(f"is_hul_brand (taxonomy)  : {res.row_level_items[0].is_hul_brand}")
    print(f"tokens                   : {res.tokens.total_tokens}")
    print(f"cost / shelf image       : ${res.cost.cost_per_shelf_image_usd:.6f}")
    print(f"accuracy_status          : {res.accuracy.accuracy_status}")
    print(f"OTel trace               : {res.trace_id}")
    print(f"markdown report          : {summary['artifacts']['markdown_report']}")
    print(
        "\nThe approach is registered globally, so it is also visible to the CLI once this module\n"
        "has been imported:  shelf-benchmark list-approaches"
    )


if __name__ == "__main__":
    main()
