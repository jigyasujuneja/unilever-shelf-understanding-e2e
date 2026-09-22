"""Sample 06: Customizing Product Taxonomy (`TaxonomyConfig`) & Swapping Ground Truth Schemas.

Demonstrates how developers can:
  1. Override `categories`, `subcategories`, `packaging_types`, `pack_types`, `size_buckets`,
     and `hul_brands` in one place (`configs/taxonomy.yaml` or via `TaxonomyConfig` in Python)
     without any hardcoded brand or variant counts.
  2. Configure custom BigQuery / CSV / JSON Ground Truth and Association Table schema mappings
     when real annotated datasets become available on GCP.
"""

from shelf_benchmark import BenchmarkConfig, ShelfBenchmarkSDK, TaxonomyConfig
from shelf_benchmark.config import SizeBucketRulesConfig
from shelf_benchmark.tasks.classification import build_classification_prompt
from shelf_benchmark.tasks.facing_utils import check_is_hul_brand, derive_size_bucket_from_bbox


def main() -> None:
    # 1. Define or override a custom taxonomy programmatically (or load from a custom YAML file)
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

    # 2. Verify dynamic prompt & size rule generation (zero hardcoded counts!)
    prompt = build_classification_prompt(custom_taxonomy)
    print("=== Dynamically Generated Classification Prompt Preview ===")
    print("\n".join(prompt.splitlines()[:10]))
    print("...")

    size_label = derive_size_bucket_from_bbox(
        bbox_2d=[535, 386, 795, 440],
        all_bboxes_on_row=[[535, 386, 795, 440]],
        packaging_type="tube",
        model_size_hint="45g",
        taxonomy=custom_taxonomy,
    )
    print(f"\nDerived Size Bucket for 45g tube: {size_label}")
    print(f"Is 'Pepsodent' HUL Brand? {check_is_hul_brand('Pepsodent', taxonomy=custom_taxonomy)}")
    print(f"Is 'Sensodyne' HUL Brand? {check_is_hul_brand('Sensodyne', taxonomy=custom_taxonomy)}")

    # 3. Show how Ground Truth & Association Table schemas can be swapped in 5 lines when ready on GCP
    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    cfg.taxonomy = custom_taxonomy
    cfg.ground_truth.provider_type = "none"  # Change to "bigquery", "csv", or "json" when GT table is ready
    cfg.ground_truth.schema_mapping.brand_field = "annotated_brand_name"
    cfg.ground_truth.schema_mapping.bbox_field = "annotated_bbox_2d"
    print(f"\nConfigured Ground Truth Provider: {cfg.ground_truth.provider_type} (Ready to swap to BigQuery/CSV/JSON)")


if __name__ == "__main__":
    main()
