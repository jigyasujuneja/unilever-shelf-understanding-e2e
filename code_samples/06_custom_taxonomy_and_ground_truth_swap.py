#!/usr/bin/env python3
"""Sample 06: Configurable N-Attribute Taxonomy (>8 attributes), Grouped VLM Calls & Universal Schema Swap.

Run it (no GCP project, no credentials, no network):

    .venv/bin/python code_samples/06_custom_taxonomy_and_ground_truth_swap.py

What this sample shows:
  Part 1  Configure >8 attributes (8 core dimensions + 4 custom attributes = 12 attributes total)
          and configure `attribute_call_groups` so you can extract all 12 attributes in 1 VLM call
          OR split them across multiple targeted VLM calls by attribute type (`configurable_multi_attribute_vlm`).
  Part 2  Connect a ground-truth dataset (`configs/sample_ground_truth.json`). Note: this benchmark
          suite does NOT produce annotations; it ingests the ground-truth dataset provided externally.
  Part 3  Universal GCP Dataset (`sdk.connect_dataset`) & Ground-Truth (`sdk.connect_ground_truth`)
          schema adaptation for multi-image GCS buckets, BigQuery tables, CSVs, and nested JSONs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from shelf_benchmark import BenchmarkConfig, CustomAttributeSpec, TaxonomyConfig
from shelf_benchmark.config import SizeBucketRulesConfig
from shelf_benchmark.data.ground_truth import create_ground_truth_provider
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.tasks.classification import build_classification_prompt
from shelf_benchmark.tasks.facing_utils import check_is_hul_brand, derive_size_bucket_from_bbox
from shelf_benchmark.testing import benchmark_harness

SAMPLE_GT_FILE = "configs/sample_ground_truth.json"


def main() -> None:
    # ---------------------------------------------------------------- Part 1: 12-Attribute Taxonomy (>8 attributes)
    custom_taxonomy = TaxonomyConfig(
        categories=["Oral Care", "Skin Care", "Hair Care", "Home & Hygiene"],
        subcategories=["Toothpaste", "Mouthwash", "Face Wash", "Shampoo", "Laundry Bar"],
        packaging_types=["box", "tube", "bottle", "jar", "sachet", "pouch"],
        pack_types=["Single", "Multipack", "Promo Bundle"],
        size_buckets=SizeBucketRulesConfig(
            sachet_label="Trial Sachet (<25g/ml)",
            small_label="Compact Pack (25-55g/ml)",
            medium_label="Standard Pack (56-110g/ml)",
            large_label="Family Pack (>110g/ml)",
            sachet_max_grams=25,
            small_max_grams=55,
            medium_max_grams=110,
        ),
        hul_brands=["Brand_A", "Brand_B", "Brand_A", "Brand_A", "Brand_C", "Brand_D"],
        non_hul_brands=["Competitor_1", "Competitor_2", "Brand_B", "Competitor_3", "Competitor_4"],
        # Add 4 custom attributes on top of the 8 core dimensions -> 12 total attributes!
        custom_attributes={
            "price_tag_visible": CustomAttributeSpec(
                description="Whether a retail price tag is visible below the facing",
                value_type="boolean",
                allowed_values=["true", "false"],
            ),
            "promo_callout": CustomAttributeSpec(
                description="Promotional badge or discount text printed on the package",
                value_type="string",
            ),
            "facing_orientation": CustomAttributeSpec(
                description="Physical orientation of the product facing on the shelf",
                value_type="string",
                allowed_values=["front_straight", "tilted", "sideways", "upside_down"],
            ),
            "shelf_talker_present": CustomAttributeSpec(
                description="Whether a promotional shelf talker is attached near this slot",
                value_type="boolean",
            ),
        },
        # Configure whether attributes are extracted in 1 VLM call (leave empty `[]`) or
        # grouped into separate VLM calls by attribute type (`configurable_multi_attribute_vlm`):
        attribute_call_groups=[
            ["category", "subcategory", "brand", "product_name", "variant"],
            ["packaging_type", "pack_type", "size", "price_tag_visible", "promo_callout", "facing_orientation", "shelf_talker_present"],
        ],
    )

    print(f"Total configured attributes ({len(custom_taxonomy.all_attribute_names)}): {custom_taxonomy.all_attribute_names}")
    prompt = build_classification_prompt(custom_taxonomy)
    print("\n=== Generated 12-Attribute Classification Prompt (Last 12 Lines) ===")
    print("\n".join(prompt.splitlines()[-12:]))

    size_label = derive_size_bucket_from_bbox(
        bbox_2d=[535, 386, 795, 440],
        all_bboxes_on_row=[[535, 386, 795, 440]],
        packaging_type="tube",
        model_size_hint="45g",
        taxonomy=custom_taxonomy,
    )
    print(f"\nsize bucket for a 45g tube      : {size_label}")
    print(f"is 'Brand_A' a HUL brand?     : {check_is_hul_brand('Brand_A', taxonomy=custom_taxonomy)}")
    print(f"is 'Competitor_2' a HUL brand?     : {check_is_hul_brand('Competitor_2', taxonomy=custom_taxonomy)}")

    # ------------------------------------------------------- Part 2: Ingest Provided Ground Truth
    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    cfg.taxonomy = custom_taxonomy
    print(f"\nconfigured provider out of the box: {cfg.ground_truth.provider_type}")

    # Connect a provided ground-truth source (json, jsonl, csv, coco, bigquery, none).
    cfg.ground_truth.provider_type = "json"
    cfg.ground_truth.source_uri = SAMPLE_GT_FILE
    cfg.ground_truth.gt_version = "sample-v1"
    cfg.ground_truth.schema_mapping.bbox_format = "ymin_xmin_ymax_xmax_1000"
    cfg.offline.enabled = True

    storage = StorageManager(
        project_id=cfg.gcp.project_id,
        bucket_config=cfg.buckets,
        offline=cfg.offline.enabled,
    )
    provider = create_ground_truth_provider(
        gt_config=cfg.ground_truth,
        storage_manager=storage,
        project_id=cfg.gcp.project_id,
    )
    print(f"loaded provider                   : {provider.describe()}")

    record = provider.get_ground_truth(
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf_sample_01.png",
        ground_truth_id="shelf_sample_01.png",
    )
    print(f"facings in ground truth           : {record.total_main_shelf_facings}")
    print(f"items after back-row exclusion    : {len(record.items)}")
    print(f"gt_version stamped on every row   : {record.gt_version}")

    # --------------------------------------- Part 3: Universal GCP Dataset & Schema Adaptation
    work = Path(tempfile.mkdtemp(prefix="shelf-schema-adapter-"))
    sdk = benchmark_harness(work)
    sdk.connect_ground_truth(
        provider_type="json",
        source_uri=SAMPLE_GT_FILE,
        bbox_format="ymin_xmin_ymax_xmax_1000",
        gt_version="sample-v1",
    )
    print(
        "\nUniversal Schema Adapter Summary:\n"
        "  1. sdk.connect_dataset(provider_type='bigquery'|'gcs_bucket'|'csv'|'json', source_uri=..., schema_mapping=...)\n"
        "     - Supports multi-image GCS buckets of any image resolution + BigQuery tables/SQL queries\n"
        "     - Supports nested dot-paths (e.g., 'context.retail_store_code')\n"
        "     - Resolves relative image filenames automatically against shelf_images_bucket\n"
        "  2. sdk.connect_ground_truth(provider_type='bigquery'|'coco'|'csv'|'json'|'jsonl', source_uri=..., bbox_format=...)\n"
        "     - Supports 4-column CSV/BQ boxes ('ymin,xmin,ymax,xmax' or 'x,y,width,height')\n"
        "     - Supports polygon vertices ('boundingPoly.normalizedVertices')\n"
        "     - Supports >8 attributes via nested dot-paths or extra_attributes ('per_attribute_accuracy')\n"
        f"  Validate any new file with:\n"
        f"    .venv/bin/shelf-benchmark validate-gt --gt-provider json --ground-truth-uri {SAMPLE_GT_FILE}"
    )


if __name__ == "__main__":
    main()
