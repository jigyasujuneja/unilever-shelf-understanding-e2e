#!/usr/bin/env python3
"""Sample 06: Customise the taxonomy, then swap in a real ground-truth source.

Run it (no GCP project, no credentials, no network):

    .venv/bin/python code_samples/06_custom_taxonomy_and_ground_truth_swap.py

Part 1 shows that `categories`, `subcategories`, `packaging_types`, `pack_types`, `size_buckets`,
`hul_brands` and `non_hul_brands` live in exactly one place (`configs/taxonomy.yaml`, or a
`TaxonomyConfig` built in Python). Prompts, the size-bucket rule and the HUL brand check all read
from it, so there are no hard-coded category or brand lists anywhere in the pipeline.

Part 2 shows the ground-truth swap. It is a config change, not a code change. The file used here,
`configs/sample_ground_truth.json`, is the canonical worked example of the suite-native schema;
`docs/GROUND_TRUTH_CONTRACT.md` specifies it field by field.
"""

from __future__ import annotations

from shelf_benchmark import BenchmarkConfig, TaxonomyConfig
from shelf_benchmark.config import SizeBucketRulesConfig
from shelf_benchmark.data.ground_truth import create_ground_truth_provider
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.tasks.classification import build_classification_prompt
from shelf_benchmark.tasks.facing_utils import check_is_hul_brand, derive_size_bucket_from_bbox

SAMPLE_GT_FILE = "configs/sample_ground_truth.json"


def main() -> None:
    # ---------------------------------------------------------------- Part 1: taxonomy
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
        hul_brands=["Pepsodent", "Closeup", "Pond's", "Dove", "Sunsilk", "Surf Excel"],
        non_hul_brands=["Colgate", "Sensodyne", "Himalaya", "Garnier", "Ariel"],
    )

    prompt = build_classification_prompt(custom_taxonomy)
    print("=== first 10 lines of the generated classification prompt ===")
    print("\n".join(prompt.splitlines()[:10]))
    print(f"[prompt is {len(prompt.splitlines())} lines, {len(prompt)} characters]")

    size_label = derive_size_bucket_from_bbox(
        bbox_2d=[535, 386, 795, 440],
        all_bboxes_on_row=[[535, 386, 795, 440]],
        packaging_type="tube",
        model_size_hint="45g",
        taxonomy=custom_taxonomy,
    )
    print(f"\nsize bucket for a 45g tube      : {size_label}")
    print(f"is 'Pepsodent' a HUL brand?     : {check_is_hul_brand('Pepsodent', taxonomy=custom_taxonomy)}")
    print(f"is 'Sensodyne' a HUL brand?     : {check_is_hul_brand('Sensodyne', taxonomy=custom_taxonomy)}")

    # ------------------------------------------------------- Part 2: ground-truth swap
    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    cfg.taxonomy = custom_taxonomy
    print(f"\nconfigured provider out of the box: {cfg.ground_truth.provider_type}")

    # Swap to a real source. Supported provider types: json, jsonl, csv, coco, bigquery, none.
    cfg.ground_truth.provider_type = "json"
    cfg.ground_truth.source_uri = SAMPLE_GT_FILE
    cfg.ground_truth.gt_version = "sample-v1"
    # Declare the coordinate convention of the incoming boxes. Getting this wrong produces
    # plausible-looking but wrong numbers rather than an error, so it is stated explicitly.
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

    # A vendor delivering different column names needs a schema_mapping, not a code change:
    cfg.ground_truth.schema_mapping.brand_field = "annotated_brand_name"
    cfg.ground_truth.schema_mapping.bbox_field = "annotated_bbox_2d"
    print(
        "\nFor arbitrary vendor field names, set ground_truth.schema_mapping.* "
        "(see docs/GROUND_TRUTH_CONTRACT.md).\n"
        "Verify any new annotation file before running a benchmark against it:\n"
        f"  .venv/bin/shelf-benchmark validate-gt --gt-provider json --ground-truth-uri {SAMPLE_GT_FILE}"
    )


if __name__ == "__main__":
    main()
