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
    export_to_gcp_cloud_logging: bool = Field(default=True)
    gcp_log_name: str = Field(default="unilever-shelf-benchmark-otel")
    sync_otel_jsonl_to_gcs: bool = Field(default=True)
    gcs_otel_subpath: str = Field(default="otel/otel_logs.jsonl")


class ReportingConfig(BaseModel):
    output_dir: str = Field(default="reports")
    sync_reports_to_gcs: bool = Field(default=True)
    gcs_reports_prefix: str = Field(default="reports")


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
    hul_brands: List[str] = Field(default_factory=list)
    non_hul_brands: List[str] = Field(default_factory=list)

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


class ProvisionedThroughputConfig(BaseModel):
    """Configuration for Vertex AI Provisioned Throughput (GSU — Generative AI Scale Units) cost calculation."""

    enabled: bool = False
    gsu_count: int = 1
    reserved_gsus: int = 1
    hourly_rate_per_gsu_usd: float = 22.00
    gsu_hourly_rate_usd: float = 2.70  # Official GCP hourly rate per GSU under monthly commitment
    monthly_commitment_discount_pct: float = 0.0
    target_images_per_hour_per_gsu: int = 1800
    concurrent_request_slots_per_gsu: int = 4
    allow_payg_spillover: bool = True


class CloudRunCostConfig(BaseModel):
    """Official GCP Cloud Run compute & request pricing parameters for 100% separated infrastructure cost tracking."""

    enabled: bool = True
    service_name: str = "unilever-shelf-benchmark-service"
    region: str = "us-central1"
    vcpu_count: float = 2.0
    memory_gib: float = 4.0
    concurrency: int = 1
    cpu_allocation: str = "cpu_always_allocated"
    vcpu_per_second_usd: float = 0.00002400
    memory_gib_per_second_usd: float = 0.00000250
    vcpu_second_rate_usd: float = 0.00002400  # Official GCP Cloud Run rate ($0.000024 / vCPU-sec)
    gib_second_rate_usd: float = 0.00000250   # Official GCP Cloud Run rate ($0.0000025 / GiB-sec)
    per_million_requests_usd: float = 0.40    # Official GCP Cloud Run rate ($0.40 / 1M requests)


class GCPBillingConfig(BaseModel):
    """Master GCP Billing & True Cost Calculation configuration."""

    pricing_source: str = "gcp_billing_catalog_api"
    use_live_cloud_billing_catalog_api: bool = True
    vertex_ai_service_id: str = "6F81-5844-456A"
    cloud_run_service_id: str = "152E-C115-5142"
    traffic_mode: str = "auto"  # "auto" (detects ON_DEMAND vs PROVISIONED_THROUGHPUT from Vertex AI response), "on_demand", or "provisioned_throughput"
    attach_gcp_billing_labels: bool = True
    billing_label_prefix: str = "unilever_shelf"
    bigquery_billing_export_table: Optional[str] = None  # e.g. "unilever-shelf-understanding.billing_export.gcp_billing_export_resource_v1_XXXXXX"
    provisioned_throughput: ProvisionedThroughputConfig = Field(default_factory=ProvisionedThroughputConfig)
    cloud_run: CloudRunCostConfig = Field(default_factory=CloudRunCostConfig)
    gcs_and_logging_overhead_per_image_usd: float = 0.000015
    gcs_class_a_per_thousand_ops_usd: float = 0.005  # Official GCS Standard Class A rate ($0.005 / 1,000 ops)
    gcs_class_b_per_thousand_ops_usd: float = 0.0004 # Official GCS Standard Class B rate ($0.0004 / 1,000 ops)
    cloud_logging_per_gib_usd: float = 0.50          # Official GCP Cloud Logging rate ($0.50 / GiB)


class EmbeddingsConfig(BaseModel):
    """YAML-configurable embedding models and hybrid search weights."""

    text_embedding_model: str = "gemini-embedding-001"
    text_embedding_dimensions: int = 3072
    visual_embedding_model: str = "multimodalembedding@001"
    visual_embedding_dimensions: int = 1408
    hybrid_dense_weight: float = 0.65
    hybrid_sparse_weight: float = 0.35


class DepthDeduplicationConfig(BaseModel):
    """YAML-configurable front-facing depth deduplication geometry rules."""

    enabled: bool = True
    x_overlap_threshold: float = 0.55


class FineTuningConfig(BaseModel):
    """YAML-configurable Supervised Fine-Tuning (SFT) job parameters."""

    submit_live_tuning_job: bool = False
    tuned_model_display_name: str = "unilever-shelf-classifier-tuned"
    epochs: int = 3
    num_synthetic_replicas: int = 16
    dataset_gcs_subpath: str = "tuning/shelf_sft_train.jsonl"


class ModelEndpointYamlConfig(BaseModel):
    """YAML-configurable routing for GEAP, Gemma, or Fine-Tuned Vertex AI Endpoints."""

    provider_family: str = "vertex_gemini"  # "vertex_gemini", "vertex_geap", "vertex_gemma", "vertex_tuned_endpoint"
    model_id: Optional[str] = None
    endpoint_uri: Optional[str] = None
    api_version: Optional[str] = None
    location: Optional[str] = None


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
    tasks: List[str] = Field(
        default_factory=lambda: [
            "detection",
            "classification",
            "matching",
            "fine_tuning",
        ]
    )
    approaches: List[str] = Field(
        default_factory=lambda: [
            "single_pass_full_shelf",
            "two_stage_bbox_guided_nms",
            "two_stage_physical_crop_per_facing",
            "class_agnostic_visual_embedding",
        ]
    )
    model_endpoints: Dict[str, ModelEndpointYamlConfig] = Field(default_factory=dict)
    billing: GCPBillingConfig = Field(default_factory=GCPBillingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    depth_deduplication: DepthDeduplicationConfig = Field(default_factory=DepthDeduplicationConfig)
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)
    pricing_per_million_tokens: Dict[str, ModelPricing] = Field(
        default_factory=lambda: {
            "gemini-3.8-flash": ModelPricing(input=0.30, thinking=0.30, output=2.50),
            "gemini-3.7-flash": ModelPricing(input=0.25, thinking=0.25, output=2.00),
            "gemini-3.5-flash-lite": ModelPricing(input=0.10, thinking=0.10, output=0.40),
            "gemini-2.5-flash": ModelPricing(input=0.30, thinking=0.30, output=2.50),
            "gemini-2.5-flash-lite": ModelPricing(input=0.10, thinking=0.10, output=0.40),
            "gemini-2.5-pro": ModelPricing(input=1.25, thinking=1.25, output=10.00),
            "gemma-3-27b-it": ModelPricing(input=0.08, thinking=0.00, output=0.24),
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

