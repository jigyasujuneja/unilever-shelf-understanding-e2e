"""Configuration definitions for the Shelf Understanding Benchmark Suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class GCPConfig(BaseModel):
    project_id: str = Field(default="unilever-shelf-understanding")
    location: str = Field(default="global")
    tuning_location: str = Field(default="us-central1")


class BucketConfig(BaseModel):
    shelf_images_bucket: str = Field(default="gs://unilever-shelf-understanding-shelf-images")
    catalog_images_bucket: str = Field(default="gs://unilever-shelf-understanding-catalog-images")
    planograms_bucket: Optional[str] = Field(default="gs://unilever-shelf-understanding-planograms")
    artifacts_bucket: Optional[str] = Field(default="gs://unilever-shelf-understanding-shelf-images")


class ModelPricing(BaseModel):
    input: float = Field(default=0.30, description="USD per 1M input tokens")
    thinking: float = Field(default=0.30, description="USD per 1M thinking tokens")
    output: float = Field(default=2.50, description="USD per 1M output tokens")


class AssociationSchemaMapping(BaseModel):
    association_id_field: str = "association_id"
    shelf_image_uri_field: str = "shelf_image_uri"
    catalog_uri_field: str = "catalog_uri"
    planogram_uri_field: str = "planogram_uri"
    store_id_field: str = "store_id"
    ground_truth_id_field: str = "ground_truth_id"


class AssociationConfig(BaseModel):
    provider_type: str = Field(default="json", description="json | csv | bigquery | bucket_discovery")
    source_uri: Optional[str] = Field(default="configs/sample_associations.json")
    schema_mapping: AssociationSchemaMapping = Field(default_factory=AssociationSchemaMapping)


class GroundTruthSchemaMapping(BaseModel):
    image_key_field: str = "image_id"
    items_list_field: str = "items"
    brand_field: str = "brand"
    product_name_field: str = "product_name"
    sku_id_field: str = "sku_id"
    bbox_field: str = "bbox_2d"
    shelf_row_field: str = "shelf_row"


class GroundTruthConfig(BaseModel):
    provider_type: str = Field(default="json", description="json | csv | bigquery | coco | none")
    source_uri: Optional[str] = Field(default="configs/sample_ground_truth.json")
    schema_mapping: GroundTruthSchemaMapping = Field(default_factory=GroundTruthSchemaMapping)


class TelemetryConfig(BaseModel):
    service_name: str = Field(default="unilever-shelf-benchmark")
    otel_log_path: str = Field(default="reports/otel_logs.jsonl")
    export_to_console: bool = Field(default=False)


class ReportingConfig(BaseModel):
    output_dir: str = Field(default="reports")


class SizeBucketRulesConfig(BaseModel):
    sachet_label: str = "Sachet / Trial (<25g/ml)"
    small_label: str = "Small / Compact (25-55g/ml)"
    medium_label: str = "Medium / Regular (56-110g/ml)"
    large_label: str = "Large / Family (>110g/ml)"
    sachet_max_grams: int = 25
    small_max_grams: int = 55
    medium_max_grams: int = 110
    small_bbox_height_ratio: float = 0.82
    large_bbox_height_ratio: float = 1.08


class TaxonomyConfig(BaseModel):
    """Centralized, configurable taxonomy for Categories, Subcategories, Brands, Packaging, Pack Types, and Size Buckets."""

    taxonomy_file: str = Field(default="configs/taxonomy.yaml")
    categories: List[str] = Field(
        default_factory=lambda: [
            "Hair Care",
            "Oral Care",
            "Laundry",
            "Skin Care",
            "Skin Cleansing",
            "Deodorants & Fragrances",
            "Home & Surface Care",
            "Tea & Coffee",
            "Packaged Foods",
            "Health & Nutrition",
        ]
    )
    subcategories: List[str] = Field(
        default_factory=lambda: [
            "Shampoo",
            "Conditioner",
            "Hair Oil",
            "Mouthwash",
            "Toothpaste",
            "Soaps",
            "Body Wash",
            "Face Wash",
            "Moisturizer / Cream",
            "Facial Gel / Serum",
            "Detergent Bar",
            "Detergent Powder",
            "Liquid Detergent",
            "Fabric Conditioner",
        ]
    )
    packaging_types: List[str] = Field(
        default_factory=lambda: [
            "box",
            "jar",
            "sachet",
            "tube",
            "bottle",
            "pouch",
            "bar",
        ]
    )
    pack_types: List[str] = Field(default_factory=lambda: ["Single", "Multiple"])
    size_buckets: SizeBucketRulesConfig = Field(default_factory=SizeBucketRulesConfig)
    hul_brands: List[str] = Field(
        default_factory=lambda: [
            "Pond's",
            "Glow & Lovely",
            "Fair & Lovely",
            "Lakme",
            "Pears",
            "Dove",
            "Lifebuoy",
            "Lux",
            "Sunsilk",
            "Clinic Plus",
            "TRESemmé",
            "Indulgeo",
            "Vaseline",
            "Rexona",
            "Axe",
            "Pepsodent",
            "Closeup",
            "Surf Excel",
            "Rin",
            "Wheel",
            "Sunlight",
            "Comfort",
            "Vim",
            "Domex",
            "Cif",
            "Brooke Bond",
            "Lipton",
            "Bru",
            "Kissan",
            "Knorr",
            "Kwality Wall's",
            "Horlicks",
            "Boost",
            "Breeze",
            "Hamam",
            "Motee",
            "Liril",
            "Ayush",
            "Love Beauty and Planet",
            "Simple",
            "St. Ives",
            "Clear",
        ]
    )
    non_hul_brands: List[str] = Field(
        default_factory=lambda: [
            "Himalaya",
            "Nivea",
            "Clean & Clear",
            "Everyuth",
            "Garnier",
            "Olay",
            "Neutrogena",
            "Cetaphil",
            "Biotique",
            "Mamaearth",
            "Colgate",
            "Sensodyne",
            "Oral-B",
            "Ariel",
            "Tide",
            "Pantene",
            "Head & Shoulders",
            "L'Oréal",
            "Dettol",
            "Santoor",
        ]
    )

    @property
    def size_bucket_labels(self) -> List[str]:
        return [
            self.size_buckets.sachet_label,
            self.size_buckets.small_label,
            self.size_buckets.medium_label,
            self.size_buckets.large_label,
        ]

    @classmethod
    def from_yaml_or_defaults(cls, path: str | Path = "configs/taxonomy.yaml") -> "TaxonomyConfig":
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = yaml.safe_load(f) or {}
            return cls.model_validate(data)
        return cls()


class BenchmarkConfig(BaseModel):
    gcp: GCPConfig = Field(default_factory=GCPConfig)
    buckets: BucketConfig = Field(default_factory=BucketConfig)
    models: List[str] = Field(
        default_factory=lambda: [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
        ]
    )
    pricing_per_million_tokens: Dict[str, ModelPricing] = Field(
        default_factory=lambda: {
            "gemini-3.8-flash": ModelPricing(input=0.30, thinking=0.30, output=2.50),
            "gemini-3.7-flash": ModelPricing(input=0.25, thinking=0.25, output=2.00),
            "gemini-3.5-flash-lite": ModelPricing(input=0.10, thinking=0.10, output=0.40),
            "gemini-2.5-flash": ModelPricing(input=0.30, thinking=0.30, output=2.50),
            "gemini-2.5-flash-lite": ModelPricing(input=0.10, thinking=0.10, output=0.40),
            "gemini-2.5-pro": ModelPricing(input=1.25, thinking=1.25, output=10.00),
            "default": ModelPricing(input=0.30, thinking=0.30, output=2.50),
        }
    )
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig.from_yaml_or_defaults)
    associations: AssociationConfig = Field(default_factory=AssociationConfig)
    ground_truth: GroundTruthConfig = Field(default_factory=GroundTruthConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        if "taxonomy" not in raw:
            tax_file = raw.get("taxonomy_file", "configs/taxonomy.yaml")
            if Path(tax_file).exists():
                with open(tax_file, "r", encoding="utf-8") as tf:
                    raw["taxonomy"] = yaml.safe_load(tf) or {}
        return cls.model_validate(raw)

    def get_pricing(self, model_name: str) -> ModelPricing:
        clean_name = model_name.split("/")[-1]
        if clean_name in self.pricing_per_million_tokens:
            return self.pricing_per_million_tokens[clean_name]
        for key, val in self.pricing_per_million_tokens.items():
            if key in clean_name:
                return val
        return self.pricing_per_million_tokens.get("default", ModelPricing())
