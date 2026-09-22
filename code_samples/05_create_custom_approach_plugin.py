"""Sample 05: Create & Benchmark a Custom CV / VLM / Hybrid Approach in 25 Lines.

Demonstrates how any developer can write a custom shelf-understanding pipeline function
using `@register_approach_function` and immediately run it through the full benchmark suite
with automatic:
  - OpenTelemetry Span & JSONL logging (`start_time`, `end_time`, `input/thinking/output_tokens`)
  - Centralized 7-dimension taxonomy & HUL brand attribution (`configs/taxonomy.yaml`)
  - Rule-derived size bucket calculation (`derive_size_bucket_from_bbox`)
  - Row-level CSV/JSON and Markdown report generation
"""

from typing import Any, Dict, List

from shelf_benchmark import ShelfBenchmarkSDK, register_approach_function
from shelf_benchmark.approaches import CommonLayerContext
from shelf_benchmark.tasks.facing_utils import deduplicate_depth_stacked_facings


@register_approach_function(
    approach_id="custom_yolo_plus_gemma_verifier",
    display_name="Custom 2-Stage Detector + Verifier Pipeline",
    category="two_stage_vlm",
    stages_description=[
        "Stage 1: Custom Bounding-Box Proposal Generator + Front-Facing Depth Deduplication",
        "Stage 2: Custom Attribute Verifier + Rule-Derived Size Bucketing",
    ],
)
def run_custom_yolo_plus_gemma_pipeline(
    ctx: CommonLayerContext,
    model_name: str,
    shelf_image_uri: str,
) -> List[Dict[str, Any]]:
    """Your custom approach logic!

    Return a list of dicts (one per detected facing). The SDK automatically applies
    `configs/taxonomy.yaml` brand checks, size bucket rules, cost calculation, and OpenTelemetry logging.
    """
    raw_candidate_boxes = [
        # Front-most facing on middle shelf (ymax=795)
        {
            "bbox_2d": [535, 386, 795, 440],
            "shelf_row": "middle",
            "category": "Skin Care",
            "subcategory": "Face Wash",
            "brand": "Pond's",
            "variant": "Bright Beauty Spot-less Glow",
            "packaging_type": "tube",
            "pack_type": "Single",
            "size": "100g",
            "confidence": 0.97,
            "_input_tokens": 220,
            "_thinking_tokens": 30,
            "_output_tokens": 85,
        },
        # Back-row depth duplicate stacked in the exact same column [388..438] (will be filtered out!)
        {
            "bbox_2d": [495, 388, 610, 438],
            "shelf_row": "middle",
            "category": "Skin Care",
            "subcategory": "Face Wash",
            "brand": "Pond's",
            "variant": "Bright Beauty Spot-less Glow",
            "packaging_type": "tube",
            "pack_type": "Single",
            "size": "100g",
            "confidence": 0.89,
        },
    ]

    # Reuse the shared common-layer depth deduplication helper
    front_facings, depth_filtered_count = deduplicate_depth_stacked_facings(raw_candidate_boxes)
    if front_facings:
        front_facings[0]["_depth_filtered"] = depth_filtered_count
    return front_facings


def main() -> None:
    sdk = ShelfBenchmarkSDK(output_dir="reports/sample_05_custom_approach")

    # Run your newly registered custom approach across any model!
    summary = sdk.run_suite(
        models=["gemini-3.8-flash"],
        tasks=["classification"],
        approaches=["custom_yolo_plus_gemma_verifier"],
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
    )

    res = summary["results"][0]
    print(
        f"[Custom Approach Run] Approach={res.separation_approach} | "
        f"Front Facings={len(res.row_level_items)} | "
        f"Depth Duplicates Filtered={res.accuracy.depth_duplicates_filtered} | "
        f"Rule Size Bucket='{res.row_level_items[0].rule_derived_size_bucket}' | "
        f"OTel Trace={res.trace_id}"
    )


if __name__ == "__main__":
    main()
