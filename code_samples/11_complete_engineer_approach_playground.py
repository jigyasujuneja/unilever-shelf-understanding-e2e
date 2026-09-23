#!/usr/bin/env python3
"""Sample 11: Complete Engineer Approach & Configuration Playground (CLI Flags + Interactive Menu + Python Reference).

You can use this file in THREE ways:
  1. Read/edit the Python code below to see how a custom approach and all config levers are wired.
  2. Pass CLI flags to toggle any lever directly from the command line:
       .venv/bin/python code_samples/11_complete_engineer_approach_playground.py \
         --brand-mode open_vocabulary_plus_catalog_resolver \
         --catalog-brand-count 2000 \
         --attribute-call-mode grouped_calls \
         --extra-attributes price_tag_visible,promo_callout,facing_orientation,shelf_talker_present \
         --accelerator tpu-v5e \
         --model gemini-3.8-flash
  3. Run with `--interactive` for an interactive step-by-step terminal prompt that lets you choose each lever:
       .venv/bin/python code_samples/11_complete_engineer_approach_playground.py --interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any, Dict, List

from shelf_benchmark import (
    CommonLayerContext,
    CustomAttributeSpec,
    ShelfBenchmarkSDK,
    SimpleShelfApproachPlugin,
)
from shelf_benchmark.approaches import GLOBAL_APPROACH_REGISTRY
from shelf_benchmark.cli import _print_comparison_table, _write_and_print_diagnostic_report
from shelf_benchmark.models import ShelfAssociationRecord
from shelf_benchmark.testing import OFFLINE_IMAGE_URI, benchmark_harness


# ============================================================================
# 1. EXAMPLE CUSTOM APPROACH (~15 Lines with SimpleShelfApproachPlugin)
# ============================================================================
class EngineerPlaygroundPlugin(SimpleShelfApproachPlugin):
    """Example custom approach showing how to use `ctx` helpers and return >8 attributes."""

    @property
    def approach_id(self) -> str:
        return "engineer_custom_12attr_approach"

    @property
    def display_name(self) -> str:
        return "Engineer Custom 12-Attribute Detector + Classifier"

    @property
    def category(self) -> str:
        return "two_stage_vlm"

    @property
    def stages_description(self) -> List[str]:
        return [
            "Stage 1: Detect front-row candidate boxes + 1D-NMS depth deduplication",
            "Stage 2: Extract 8 core + N custom merchandising attributes",
        ]

    def detect_and_classify(
        self,
        ctx: CommonLayerContext,
        model_name: str,
        record: ShelfAssociationRecord,
    ) -> List[Dict[str, Any]]:
        candidates = [
            {
                "bbox_2d": [100, 100, 300, 200],
                "category": "Skin Cleansing",
                "subcategory": "Face Wash",
                "brand": "Brand_A",
                "product_name": "Brand_A Radiance Daily Cleanser",
                "variant": "Bright Beauty",
                "packaging_type": "tube",
                "pack_type": "Single",
                "size": "100g",
                "is_hul_brand": True,
                "price_tag_visible": "true",
                "promo_callout": "20% Extra",
                "facing_orientation": "front_straight",
                "shelf_talker_present": "false",
            },
            {
                "bbox_2d": [100, 220, 300, 320],
                "category": "Skin Cleansing",
                "subcategory": "Face Wash",
                "brand": "Brand_B",
                "product_name": "Brand_B Herbal Purifying Cleanser",
                "variant": "Purifying Neem",
                "packaging_type": "tube",
                "pack_type": "Single",
                "size": "100g",
                "is_hul_brand": False,
                "price_tag_visible": "true",
                "promo_callout": "None",
                "facing_orientation": "front_straight",
                "shelf_talker_present": "false",
            },
            {
                "bbox_2d": [400, 350, 600, 450],
                "category": "Skin Cleansing",
                "subcategory": "Face Wash",
                "brand": "Brand_C",
                "product_name": "Brand_C Vitamin Gel Cleanser",
                "variant": "Blush and Glow",
                "packaging_type": "tube",
                "pack_type": "Single",
                "size": "100g",
                "is_hul_brand": True,
                "price_tag_visible": "false",
                "promo_callout": "New Pack",
                "facing_orientation": "front_straight",
                "shelf_talker_present": "true",
            },
        ]
        front_facings, depth_filtered = ctx.deduplicate_depth_stacked_facings(candidates)
        if front_facings:
            front_facings[0]["_depth_filtered"] = depth_filtered
        return front_facings


GLOBAL_APPROACH_REGISTRY.register(EngineerPlaygroundPlugin())


def _ask_choice(prompt_text: str, options: List[str], default: str) -> str:
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, start=1):
        marker = " (default)" if opt == default else ""
        print(f"  [{i}] {opt}{marker}")
    raw = input(f"Select 1-{len(options)} or press Enter for '{default}': ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return raw if raw in options else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive wizard to toggle every lever step-by-step.")
    parser.add_argument("--live", action="store_true", help="Run against live Vertex AI instead of offline fixture.")
    parser.add_argument(
        "--brand-mode",
        default="open_vocabulary_plus_catalog_resolver",
        choices=["open_vocabulary_generative", "open_vocabulary_plus_catalog_resolver", "closed_set_taxonomy"],
        help="How the LLM extracts brand (default: open_vocabulary_plus_catalog_resolver).",
    )
    parser.add_argument(
        "--catalog-brand-count",
        type=int,
        default=2000,
        help="Number of brands in the master catalog for O(1) post-hoc resolution (default: 2000).",
    )
    parser.add_argument(
        "--attribute-call-mode",
        default="grouped_calls",
        choices=["single_call", "grouped_calls"],
        help="Whether to extract all >8 attributes in 1 VLM call ('single_call') or split into multiple VLM calls by attribute group ('grouped_calls').",
    )
    parser.add_argument(
        "--extra-attributes",
        default="price_tag_visible,promo_callout,facing_orientation,shelf_talker_present",
        help="Comma-separated list of custom attributes beyond the 8 core dimensions.",
    )
    parser.add_argument(
        "--accelerator",
        default="tpu-v5e",
        choices=["none", "nvidia-l4", "tpu-v5e", "tpu-v6e"],
        help="Hardware accelerator profile to model (default: tpu-v5e).",
    )
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model ID to test.")
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=[
            "engineer_custom_12attr_approach",
            "single_pass_full_shelf",
            "open_vocab_brand_plus_catalog_resolver",
            "configurable_multi_attribute_vlm",
            "single_step_detect_classify_and_match",
            "two_stage_bbox_guided_nms",
        ],
        help="Approaches to benchmark.",
    )
    args = parser.parse_args()

    if args.interactive:
        print("=" * 95)
        print("  INTERACTIVE ENGINEER CONFIGURATION WIZARD")
        print("=" * 95)
        args.brand_mode = _ask_choice(
            "1. Choose Brand Extraction Strategy (for 0 to 2,000+ brands):",
            ["open_vocabulary_generative", "open_vocabulary_plus_catalog_resolver", "closed_set_taxonomy"],
            args.brand_mode,
        )
        args.attribute_call_mode = _ask_choice(
            "2. Choose >8 Attribute VLM Call Strategy:",
            ["single_call", "grouped_calls"],
            args.attribute_call_mode,
        )
        args.accelerator = _ask_choice(
            "3. Choose Hardware / Accelerator Profile:",
            ["none", "nvidia-l4", "tpu-v5e", "tpu-v6e"],
            args.accelerator,
        )
        args.model = _ask_choice(
            "4. Choose Model ID:",
            ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite", "gemma-3-27b-it"],
            args.model,
        )

    extra_attr_names = [a.strip() for a in args.extra_attributes.split(",") if a.strip()]

    out_dir = Path(tempfile.mkdtemp(prefix="shelf-engineer-playground-"))
    if args.live:
        sdk = ShelfBenchmarkSDK(output_dir=out_dir)
    else:
        sdk = benchmark_harness(out_dir, model_id=args.model, with_ground_truth=True)

    # Apply Lever 1: Brand Extraction Mode & 2,000-Brand Master Catalog
    sdk.config.taxonomy.brand_extraction_mode = args.brand_mode
    half = max(1, args.catalog_brand_count // 2)
    sdk.config.taxonomy.hul_brands = ["Brand_A", "Brand_C"] + [f"CatalogBrand_{i:04d}" for i in range(1, half + 1)]
    sdk.config.taxonomy.non_hul_brands = ["Brand_B"] + [
        f"CatalogBrand_{i:04d}" for i in range(half + 1, args.catalog_brand_count + 1)
    ]

    # Apply Lever 2: Custom Attributes (>8 Attributes)
    sdk.config.taxonomy.custom_attributes = {
        attr_name: CustomAttributeSpec(
            description=f"Extract custom attribute '{attr_name}' from the product facing",
            value_type="boolean" if "visible" in attr_name or "present" in attr_name else "string",
        )
        for attr_name in extra_attr_names
    }

    # Apply Lever 3: Single-Call vs Grouped Multi-Call VLM Strategy
    if args.attribute_call_mode == "single_call":
        sdk.config.taxonomy.attribute_call_groups = []
    else:
        sdk.config.taxonomy.attribute_call_groups = [
            ["category", "subcategory", "brand", "product_name", "variant"],
            ["packaging_type", "pack_type", "size"] + extra_attr_names,
        ]

    # Apply Lever 4: Hardware / Accelerator Profile
    sdk.config.billing.cloud_run.accelerator_type = args.accelerator
    sdk.config.billing.cloud_run.accelerator_count = 1 if args.accelerator != "none" else 0
    sdk.config.billing.include_infrastructure_costs = True

    # Print explicit Configuration Levers Banner so engineers see every active setting & how to toggle it
    print("\n" + "=" * 110)
    print("  ACTIVE ENGINEER CONFIGURATION LEVERS (Pass CLI flags or `--interactive` to toggle any lever)")
    print("=" * 110)
    print(f"  1. Brand Extraction Mode   (--brand-mode)          : {sdk.config.taxonomy.brand_extraction_mode}")
    print(f"  2. Master Brand Catalog    (--catalog-brand-count) : {len(sdk.config.taxonomy.hul_brands) + len(sdk.config.taxonomy.non_hul_brands)} brands (O(1) post-hoc resolution; 0 in prompt)")
    print(f"  3. Total Attributes (>8)   (--extra-attributes)    : {len(sdk.config.taxonomy.all_attribute_names)} attributes (8 core + {len(extra_attr_names)} custom: {', '.join(extra_attr_names)})")
    print(f"  4. VLM Attribute Call Mode (--attribute-call-mode) : {args.attribute_call_mode} ({'1 VLM call for all attributes' if args.attribute_call_mode == 'single_call' else f'{len(sdk.config.taxonomy.attribute_call_groups)} grouped VLM calls by attribute type'})")
    print(f"  5. Hardware / Accelerator  (--accelerator)         : {args.accelerator.upper()} (vCPU={sdk.config.billing.cloud_run.vcpu_count}, RAM={sdk.config.billing.cloud_run.memory_gib}GiB)")
    print(f"  6. Model & Execution Lane  (--model / --live)      : {args.model} ({'LIVE Vertex AI' if args.live else 'OFFLINE Fixture (pass --live for Vertex AI)'})")
    print(f"  7. Approaches Compared     (--approaches)          : {', '.join(args.approaches)}")
    print("=" * 110)

    summary = sdk.run_suite(
        models=[args.model],
        tasks=["classification"],
        approaches=args.approaches,
        shelf_image_uri=OFFLINE_IMAGE_URI if not args.live else "gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
    )

    diag_paths = _write_and_print_diagnostic_report(summary["results"], sdk.config, out_dir)
    _print_comparison_table(summary["results"])
    print(f"\nDiagnostic Markdown Report: {diag_paths['diagnostic_md']}")
    print(f"Diagnostic JSON Report    : {diag_paths['diagnostic_json']}")


if __name__ == "__main__":
    main()
