"""Configuration definitions for the Shelf Understanding Benchmark Suite."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

# The taxonomy ships inside the package and is resolved relative to *this file*, never relative to
# the current working directory. Resolving it against the CWD used to mean `shelf-benchmark run`
# produced a different set of categories depending on where you happened to launch it from, and the
# mismatch was silent because the loader fell back to a second, divergent copy hardcoded in Python.
PACKAGED_TAXONOMY_PATH = Path(__file__).resolve().parent / "_resources" / "taxonomy.yaml"


@lru_cache(maxsize=1)
def _packaged_taxonomy_data() -> Dict[str, Any]:
    """Parse the bundled taxonomy YAML once per process."""
    if not PACKAGED_TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            f"The bundled taxonomy is missing from the installed package: "
            f"'{PACKAGED_TAXONOMY_PATH}'. This means the wheel was built without "
            f"[tool.setuptools.package-data] shelf_benchmark = ['_resources/*.yaml']."
        )
    with open(PACKAGED_TAXONOMY_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"'{PACKAGED_TAXONOMY_PATH}' must contain a YAML mapping.")
    return data


def _packaged_default(key: str, fallback: Any) -> Callable[[], Any]:
    """Build a Pydantic `default_factory` that reads `key` from the bundled taxonomy YAML.

    This is what keeps the YAML the single source of truth: the Python model declares *which*
    fields exist, the YAML declares their values, and the two can no longer drift apart.
    """

    def factory() -> Any:
        return copy.deepcopy(_packaged_taxonomy_data().get(key, fallback))

    return factory



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
    """Field-name mapping from an arbitrary annotator schema onto the canonical `ImageGroundTruth` model.

    `bbox_format` is explicit on purpose: annotation vendors most commonly deliver COCO
    `[x, y, width, height]` in absolute pixels, while this suite works internally in
    `[ymin, xmin, ymax, xmax]` normalized to 0..1000. Guessing that conversion is the single
    most common source of silently-wrong detection metrics, so it must be declared.
    """

    image_key_field: str = "image_id"
    items_list_field: str = "items"
    brand_field: str = "brand"
    product_name_field: str = "product_name"
    sku_id_field: str = "sku_id"
    bbox_field: str = "bbox_2d"
    shelf_row_field: str = "shelf_row"
    item_id_field: str = "item_id"
    category_field: str = "category"
    subcategory_field: str = "subcategory"
    variant_field: str = "variant"
    packaging_type_field: str = "packaging_type"
    pack_type_field: str = "pack_type"
    size_field: str = "size"
    back_row_field: str = "back_row"
    occluded_field: str = "occluded"

    bbox_format: str = Field(
        default="ymin_xmin_ymax_xmax_1000",
        description=(
            "One of: 'ymin_xmin_ymax_xmax_1000' (suite-native), 'coco_xywh_px', "
            "'xyxy_px', 'xyxy_norm', 'yxyx_norm'."
        ),
    )
    image_width_field: str = Field(
        default="image_width",
        description="Per-image pixel width field, required when bbox_format uses absolute pixels.",
    )
    image_height_field: str = Field(
        default="image_height",
        description="Per-image pixel height field, required when bbox_format uses absolute pixels.",
    )
    default_image_width: Optional[int] = Field(
        default=None,
        description="Fallback pixel width when the per-image field is absent (pixel formats only).",
    )
    default_image_height: Optional[int] = Field(
        default=None,
        description="Fallback pixel height when the per-image field is absent (pixel formats only).",
    )


class GroundTruthConfig(BaseModel):
    provider_type: str = Field(default="none", description="json | jsonl | csv | bigquery | coco | none")
    source_uri: Optional[str] = Field(default=None)
    schema_mapping: GroundTruthSchemaMapping = Field(default_factory=GroundTruthSchemaMapping)
    gt_version: str = Field(
        default="unversioned",
        description="Ground-truth dataset version, stamped onto every report row and OTel span.",
    )
    strict: bool = Field(
        default=True,
        description=(
            "Fail loudly when a configured ground-truth source cannot be read or yields zero "
            "images, instead of silently degrading to 'no ground truth available'."
        ),
    )
    exclude_back_row_items: bool = Field(
        default=True,
        description="Drop ground-truth items flagged as depth-stacked back-row units (front facings only).",
    )
    min_matched_image_ratio: float = Field(
        default=0.0,
        description=(
            "When >0 and strict=True, raise if fewer than this fraction of benchmarked images "
            "resolve to a ground-truth entry."
        ),
    )



class EvaluationConfig(BaseModel):
    """Scoring policy. Every number that can change a reported metric lives here, not in code.

    See `docs/EVALUATION_PROTOCOL.md` for the rationale behind each default.
    """

    iou_threshold: float = Field(
        default=0.50,
        description="IoU at which a predicted box counts as a true positive for precision/recall/F1.",
    )
    pairing_strategy: str = Field(
        default="iou_greedy",
        description=(
            "'iou_greedy' pairs purely on geometry (recommended: keeps localization and "
            "classification independent). 'iou_plus_brand' also rewards brand agreement when "
            "pairing, which couples the two and inflates both."
        ),
    )
    require_iou_for_pairing: bool = Field(
        default=True,
        description=(
            "Only pair a prediction with a ground-truth item when IoU >= iou_threshold. "
            "Unpaired predictions count as false positives rather than being force-matched."
        ),
    )
    brand_matcher: str = Field(
        default="strict",
        description="'strict' (normalized equality + curated alias table) or 'fuzzy' (substring/token overlap).",
    )
    product_matcher: str = Field(
        default="strict",
        description=(
            "'strict' (normalized equality or full containment of the GT name), "
            "'token_overlap' (configurable Jaccard-style threshold), or "
            "'fuzzy_demo' (legacy category-specific keyword groups - NOT for reporting)."
        ),
    )
    product_token_overlap_threshold: float = Field(
        default=0.8,
        description="Fraction of significant ground-truth tokens required by the 'token_overlap' matcher.",
    )
    brand_aliases: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional normalized-brand alias table (empty by default; punctuation/apostrophe normalization is automatic).",
    )
    product_stopwords: List[str] = Field(
        default_factory=list,
        description="Optional tokens ignored by the 'token_overlap' product matcher.",
    )


class TelemetryConfig(BaseModel):
    service_name: str = Field(default="unilever-shelf-benchmark")
    otel_log_path: str = Field(default="reports/otel_logs.jsonl")
    export_to_console: bool = Field(default=False)
    export_to_gcp_cloud_logging: bool = Field(default=True)
    gcp_log_name: str = Field(default="unilever-shelf-benchmark-otel")
    sync_otel_jsonl_to_gcs: bool = Field(default=True)
    sync_otel_logs_to_gcs: bool = Field(default=True)
    gcs_otel_subpath: str = Field(default="otel/otel_logs.jsonl")
    gcs_otel_logs_prefix: str = Field(default="otel")


class ReportingConfig(BaseModel):
    output_dir: str = Field(default="reports")
    sync_reports_to_gcs: bool = Field(default=True)
    gcs_reports_prefix: str = Field(default="reports")
    isolate_runs: bool = Field(
        default=False,
        description=(
            "Write each run into `<output_dir>/<user>-<timestamp>-<run_id>/` so parallel "
            "engineers never overwrite one another's reports."
        ),
    )
    write_predictions_file: bool = Field(
        default=True,
        description=(
            "Emit `predictions.json` (model output only, no scoring) so runs can be re-scored "
            "later with `shelf-benchmark score` when ground truth arrives - without re-paying "
            "for inference."
        ),
    )



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


class CustomAttributeSpec(BaseModel):
    """Schema definition for an additional product or shelf attribute beyond the base 8 dimensions."""

    description: str = "Extract this attribute from the product facing."
    allowed_values: List[str] = Field(default_factory=list)
    value_type: str = "string"  # "string" | "boolean" | "number" | "list"


class TaxonomyConfig(BaseModel):
    """Centralized, configurable taxonomy for Categories, Subcategories, Brands, Packaging, Pack Types, Size Buckets, and Custom Attributes (>8 attributes)."""

    taxonomy_file: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to a YAML file that overrides the taxonomy bundled with the package. "
            "`None` (the default) means 'use the bundled taxonomy'. Keys omitted from an override "
            "file fall back to the bundled values."
        ),
    )
    # Every default below is read from the bundled YAML (see `_packaged_default`) rather than
    # restated here. Keeping a second copy in Python is what allowed the two to drift into
    # completely different category sets.
    categories: List[str] = Field(default_factory=_packaged_default("categories", []))
    subcategories: List[str] = Field(default_factory=_packaged_default("subcategories", []))
    packaging_types: List[str] = Field(default_factory=_packaged_default("packaging_types", []))
    pack_types: List[str] = Field(
        default_factory=_packaged_default("pack_types", ["Single", "Multiple"])
    )
    size_buckets: SizeBucketRulesConfig = Field(
        default_factory=lambda: SizeBucketRulesConfig.model_validate(
            _packaged_taxonomy_data().get("size_buckets", {})
        )
    )
    brand_extraction_mode: str = Field(
        default="open_vocabulary_generative",
        description=(
            "How brand is extracted: "
            "'open_vocabulary_generative' (default: LLM determines and generates brand directly from visual/OCR text on package; scales to 2,000+ brands with zero prompt bloat), "
            "'open_vocabulary_plus_catalog_resolver' (LLM generates brand openly, then post-hoc canonical resolver snaps it to a 2,000+ brand catalog), or "
            "'closed_set_taxonomy' (injects hul_brands/non_hul_brands into prompt when <= max_prompt_brands)."
        ),
    )
    max_prompt_brands: int = Field(
        default=50,
        description="Safety cap: if total configured brands exceed this (e.g. 2,000 brands), never inject the brand list into the VLM prompt; use open-vocabulary generation + post-hoc catalog resolution instead.",
    )
    hul_brands: List[str] = Field(default_factory=list)
    non_hul_brands: List[str] = Field(default_factory=list)
    custom_attributes: Dict[str, CustomAttributeSpec] = Field(
        default_factory=dict,
        description=(
            "Arbitrary additional attributes beyond the core 8 dimensions (e.g., price_tag_visible, "
            "promo_callout, flavor_or_fragrance, facing_orientation, damage_or_dent, shelf_talker_present). "
            "Automatically added to VLM prompts, predictions, and ground-truth scoring."
        ),
    )
    attribute_call_groups: List[List[str]] = Field(
        default_factory=list,
        description=(
            "Optional grouping of attributes into separate VLM calls (used by configurable_multi_attribute_vlm). "
            "If empty or 1 group, all attributes (8 core + N custom) are predicted in a single VLM call. "
            "If 2+ groups, runs 1 targeted VLM call per attribute group and merges the results per facing."
        ),
    )

    @property
    def core_attribute_names(self) -> List[str]:
        return [
            "category",
            "subcategory",
            "brand",
            "product_name",
            "variant",
            "packaging_type",
            "pack_type",
            "size",
        ]

    @property
    def all_attribute_names(self) -> List[str]:
        return self.core_attribute_names + list(self.custom_attributes.keys())

    @property
    def size_bucket_labels(self) -> List[str]:
        return [
            self.size_buckets.sachet_label,
            self.size_buckets.small_label,
            self.size_buckets.medium_label,
            self.size_buckets.large_label,
        ]

    @property
    def taxonomy_source(self) -> str:
        """Absolute path of the YAML this taxonomy came from, for report/UI provenance."""
        return str(Path(self.taxonomy_file).resolve()) if self.taxonomy_file else str(PACKAGED_TAXONOMY_PATH)

    @classmethod
    def from_yaml_or_defaults(cls, path: Optional[str | Path] = None) -> "TaxonomyConfig":
        """Load a taxonomy, defaulting to the one bundled with the package.

        `path=None` returns the bundled taxonomy. A non-`None` `path` that does not exist raises
        `FileNotFoundError`: quietly falling back to a different taxonomy after a path typo used to
        produce a run whose categories had nothing to do with the file the engineer had just edited.
        """
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Taxonomy file not found: '{p}'. Pass an existing YAML path, or omit it to use "
                f"the taxonomy bundled with the package ('{PACKAGED_TAXONOMY_PATH}')."
            )
        with open(p, encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Taxonomy file '{p}' must contain a YAML mapping, got {type(data).__name__}.")
        # Keys the override omits fall back to the bundled defaults via each field's default_factory.
        data.setdefault("taxonomy_file", str(p))
        return cls.model_validate(data)


class ProvisionedThroughputConfig(BaseModel):
    """Configuration for Vertex AI Provisioned Throughput (GSU — Generative AI Scale Units) cost calculation.

    There is exactly one field per quantity here. This model previously carried two names for the
    GSU count and two for the hourly rate (`hourly_rate_per_gsu_usd=22.00` alongside
    `gsu_hourly_rate_usd=2.70`, an 8.1x difference). The cost engine read them as
    `getattr(a, X, None) or getattr(a, Y, default)`, which on a Pydantic model always resolves to
    the first name because it is always present and non-zero. Setting the second one therefore did
    nothing at all, silently.
    """

    enabled: bool = False
    gsu_count: int = 1
    hourly_rate_per_gsu_usd: float = Field(
        default=22.00,
        description=(
            "List on-demand price per GSU-hour. Committed-use pricing is substantially lower; "
            "express that with `monthly_commitment_discount_pct` rather than by editing this rate, "
            "so reports can still show list vs. effective cost."
        ),
    )
    monthly_commitment_discount_pct: float = 0.0
    target_images_per_hour_per_gsu: int = 1800
    concurrent_request_slots_per_gsu: int = 4
    allow_payg_spillover: bool = True


class CloudRunCostConfig(BaseModel):
    """Official GCP Cloud Run & Vertex AI/GKE Accelerator (CPU / NVIDIA L4 GPU / Cloud TPU v5e & v6e) pricing parameters."""

    enabled: bool = True
    service_name: str = "unilever-shelf-benchmark-service"
    region: str = "us-central1"
    vcpu_count: float = 2.0
    memory_gib: float = 4.0
    concurrency: int = 1
    cpu_allocation: str = "cpu_always_allocated"
    accelerator_type: str = Field(
        default="none",
        description=(
            "Hardware accelerator profile: 'none' (CPU-only Cloud Run), 'nvidia-l4' (Cloud Run native L4 GPU), "
            "'tpu-v5e' (Vertex AI / GKE Cloud TPU v5e endpoint), or 'tpu-v6e' (Vertex AI / GKE Cloud TPU v6e Trillium endpoint)."
        ),
    )
    accelerator_count: int = Field(default=0, description="Number of attached GPUs or TPU chips (e.g., 1).")
    accelerator_hourly_rates_usd: Dict[str, float] = Field(
        default_factory=lambda: {
            "none": 0.0,
            "nvidia-l4": 0.67,
            "tpu-v5e": 1.20,
            "tpu-v6e": 2.70,
        }
    )
    # One name per rate. `vcpu_second_rate_usd` / `gib_second_rate_usd` were duplicate aliases of
    # these two; the cost engine only ever read the names below, so the aliases were inert.
    vcpu_per_second_usd: float = 0.00002400  # Official GCP Cloud Run rate ($0.000024 / vCPU-sec)
    memory_gib_per_second_usd: float = 0.00000250  # Official GCP Cloud Run rate ($0.0000025 / GiB-sec)
    per_million_requests_usd: float = 0.40  # Official GCP Cloud Run rate ($0.40 / 1M requests)

    def accelerator_per_second_usd(self) -> float:
        acc_key = (self.accelerator_type or "none").strip().lower()
        if acc_key in ("none", "cpu", ""):
            return 0.0
        eff_count = max(1, self.accelerator_count or 1)
        hourly = float(self.accelerator_hourly_rates_usd.get(acc_key, 1.20))
        return (hourly / 3600.0) * eff_count


class EmbeddingAndVisionCostConfig(BaseModel):
    """Per-facing auxiliary API cost estimates (embeddings / Vision), by task and approach.

    These are *estimates*, not metered charges. They previously lived as bare literals inside
    `tasks/base.py`, which meant the cost ranking between approaches was partly hand-assigned.
    Keeping them here makes the assumption visible and overridable, and `billing_source` on every
    row reports whether a run used estimates or metered billing-export data.
    """

    enabled: bool = True
    default_per_facing_usd: float = 0.000025
    per_approach_per_facing_usd: Dict[str, float] = Field(
        default_factory=lambda: {
            "two_stage_physical_crop_per_facing": 0.000125,
            "class_agnostic_visual_embedding": 0.000125,
            # Was "demo_prototype_visual_embedding", which is not a registered
            # approach -- so this override never matched anything and the real
            # crop-per-facing Cloud Vision pipeline silently fell through to the
            # cheaper per-task rate. Keep this list in sync with the ids from
            # `shelf-benchmark list-approaches`.
            "cloud_vision_visual_embedding": 0.000125,
        },
        description="Approach-specific override (crop-per-facing pipelines embed one image per facing).",
    )
    per_task_per_facing_usd: Dict[str, float] = Field(
        default_factory=lambda: {
            "classification": 0.000025,
            "matching": 0.000050,
            "detection": 0.000020,
            "fine_tuning": 0.000020,
        },
    )

    def per_facing_usd(self, task_type: str, approach_id: str) -> float:
        if not self.enabled:
            return 0.0
        if approach_id in self.per_approach_per_facing_usd:
            return float(self.per_approach_per_facing_usd[approach_id])
        if task_type in self.per_task_per_facing_usd:
            return float(self.per_task_per_facing_usd[task_type])
        return float(self.default_per_facing_usd)


class GCPBillingConfig(BaseModel):
    """Master GCP Billing & Cost Calculation configuration.

    Cost honesty rules enforced by `GCPBillingAndCostEngine`:
    - Token cost is derived from measured `usage_metadata` and a rate table; `billing_source`
      records whether the rate came from the live Cloud Billing Catalog API or the YAML table.
    - Infrastructure components (Cloud Run, GCS, Cloud Logging) are *modelled*, not measured,
      and are excluded unless `include_infrastructure_costs` is true.
    - Only `bigquery_billing_export_table` reconciliation yields invoice-exact numbers.
    """

    pricing_source: str = "gcp_billing_catalog_api"
    pricing_last_verified_date: str = Field(
        default="2026-09-23",
        description="ISO date (YYYY-MM-DD) when the YAML rate table was last verified against GCP pricing.",
    )
    max_pricing_age_days: Optional[int] = Field(
        default=None,
        description="If set, `check_pricing_staleness` raises when `pricing_last_verified_date` is older than this many days.",
    )
    use_live_cloud_billing_catalog_api: bool = True
    # Vertex AI ("Cloud AI Platform") service ID in the Cloud Billing Catalog API.
    vertex_ai_service_id: str = "6F81-5844-456A"
    cloud_run_service_id: str = "152E-C115-5142"
    catalog_api_timeout_seconds: float = 2.5
    traffic_mode: str = "auto"  # "auto" (detect from usage_metadata), "on_demand", "provisioned_throughput"
    attach_gcp_billing_labels: bool = True
    billing_label_prefix: str = "unilever_shelf"
    bigquery_billing_export_table: Optional[str] = None  # "project.billing_export.gcp_billing_export_resource_v1_XXXXXX"
    include_infrastructure_costs: bool = Field(
        default=False,
        description=(
            "Include Cloud Run vCPU/GiB/GPU/TPU-seconds and GCS/Cloud Logging overhead in "
            "cost_per_shelf_image_usd (enabled in default_config.yaml and Cloud Run CLI)."
        ),
    )
    provisioned_throughput: ProvisionedThroughputConfig = Field(default_factory=ProvisionedThroughputConfig)
    cloud_run: CloudRunCostConfig = Field(default_factory=CloudRunCostConfig)
    embeddings_and_vision: EmbeddingAndVisionCostConfig = Field(
        default_factory=EmbeddingAndVisionCostConfig
    )
    gcs_and_logging_overhead_per_image_usd: float = 0.000015
    gcs_class_a_per_thousand_ops_usd: float = 0.005  # GCS Standard Class A ($0.005 / 1,000 ops)
    gcs_class_b_per_thousand_ops_usd: float = 0.0004  # GCS Standard Class B ($0.0004 / 1,000 ops)
    cloud_logging_per_gib_usd: float = 0.50  # Cloud Logging ingestion ($0.50 / GiB)
    estimated_log_bytes_per_run: int = 4096


class ReferenceCatalogConfig(BaseModel):
    """Reference product catalog used by the visual-embedding approach's vector-search stage.

    The catalog is the thing that turns a class-agnostic detector into a product recognizer.
    Until one is indexed, the pipeline can localize facings but genuinely cannot name them, and
    it must say so rather than emit the nearest entry from a handful of built-in examples.

    Set `source_uri` to a JSON file (local path or GCS URI) containing a list of entries:

        [{"sku_id": "...", "brand": "...", "category": "...", "subcategory": "...",
          "variant": "...", "packaging": "...", "is_hul": true,
          "prompt": "text description used to build the reference vector"}]

    `configs/demo_visual_prototypes.json` ships as an illustrative example only. It describes a
    single sample face-wash shelf, so any accuracy measured against it is meaningless. It is NOT
    loaded unless you point `source_uri` at it explicitly.
    """

    source_uri: Optional[str] = Field(
        default=None,
        description="JSON catalog of reference products. None means no catalog is indexed.",
    )
    min_match_similarity: float = Field(
        default=0.10,
        description=(
            "Cosine similarity floor in the 1408-D space. Below this the facing is reported as "
            "'Unknown' rather than being snapped to the least-bad catalog entry."
        ),
    )
    unknown_brand_label: str = Field(
        default="Unknown",
        description="Value written to predicted_brand when no catalog entry clears the threshold.",
    )


class EmbeddingsConfig(BaseModel):
    """YAML-configurable embedding models and hybrid search weights."""

    text_embedding_model: str = "gemini-embedding-001"
    text_embedding_dimensions: int = 3072
    visual_embedding_model: str = "multimodalembedding@001"
    visual_embedding_dimensions: int = 1408
    hybrid_dense_weight: float = 0.65
    hybrid_sparse_weight: float = 0.35
    reference_catalog: ReferenceCatalogConfig = Field(default_factory=ReferenceCatalogConfig)


class DepthDeduplicationConfig(BaseModel):
    """YAML-configurable front-facing depth deduplication geometry rules."""

    enabled: bool = True
    x_overlap_threshold: float = 0.55


class FineTuningConfig(BaseModel):
    """YAML-configurable Supervised Fine-Tuning (SFT) job parameters.

    Every field here is read by `GeminiFineTuningTask`. `epochs`, `tuned_model_display_name` and
    the example count were previously declared but never passed to the tuning call, so editing
    them in YAML changed nothing about the submitted job.
    """

    submit_live_tuning_job: bool = False
    tuned_model_display_name: str = "unilever-shelf-classifier-tuned"
    epochs: int = 3
    duplicate_single_example_times: int = Field(
        default=1,
        description=(
            "Number of times to write the SAME single SFT example, purely to clear Vertex AI's "
            "minimum-row check. Nothing is synthesised and no supervision is added. Was named "
            "`num_synthetic_replicas` and defaulted to 16, which made a one-example fixture look "
            "like a 16-example dataset. Supply multiple annotated images for a real dataset."
        ),
    )
    dataset_gcs_subpath: str = "tuning/shelf_sft_train.jsonl"


class ModelEndpointYamlConfig(BaseModel):
    """YAML-configurable routing for GEAP, Gemma, or Fine-Tuned Vertex AI Endpoints."""

    provider_family: str = "vertex_gemini"  # "vertex_gemini", "vertex_geap", "vertex_gemma", "vertex_tuned_endpoint"
    model_id: Optional[str] = None
    endpoint_uri: Optional[str] = None
    api_version: Optional[str] = None
    location: Optional[str] = None


class OfflineConfig(BaseModel):
    """Switches every network dependency off so the suite runs on a laptop with no GCP access.

    Used by `shelf_benchmark.testing.offline_config()` and `shelf-benchmark --offline`.
    """

    enabled: bool = False
    fail_on_network_access: bool = Field(
        default=True,
        description="Raise instead of silently degrading if code attempts GCS/BigQuery/Vertex while offline.",
    )


class BenchmarkConfig(BaseModel):
    gcp: GCPConfig = Field(default_factory=GCPConfig)
    buckets: BucketConfig = Field(default_factory=BucketConfig)
    models: List[str] = Field(
        default_factory=lambda: [
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
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
            "open_vocab_brand_plus_catalog_resolver",
            "configurable_multi_attribute_vlm",
            "single_step_detect_classify_and_match",
            "two_stage_bbox_guided_nms",
            "two_stage_physical_crop_per_facing",
        ]
    )
    model_endpoints: Dict[str, ModelEndpointYamlConfig] = Field(default_factory=dict)
    billing: GCPBillingConfig = Field(default_factory=GCPBillingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    depth_deduplication: DepthDeduplicationConfig = Field(default_factory=DepthDeduplicationConfig)
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)
    pricing_per_million_tokens: Dict[str, ModelPricing] = Field(
        default_factory=lambda: {
            "gemini-3-flash-preview": ModelPricing(input=0.30, thinking=0.30, output=2.50),
            "gemini-3.1-pro-preview": ModelPricing(input=1.25, thinking=1.25, output=10.00),
            "gemini-3.0-flash": ModelPricing(input=0.30, thinking=0.30, output=2.50),
            "gemini-3.8-flash": ModelPricing(input=0.30, thinking=0.30, output=2.50),
            "gemini-3.7-flash": ModelPricing(input=0.25, thinking=0.25, output=2.00),
            "gemini-3.5-flash-lite": ModelPricing(input=0.10, thinking=0.10, output=0.40),
            "gemma-3-27b-it": ModelPricing(input=0.08, thinking=0.00, output=0.24),
            "default": ModelPricing(input=0.30, thinking=0.30, output=2.50),
        }
    )
    warn_on_unpriced_model: bool = Field(
        default=True,
        description="Print a warning when a benchmarked model falls back to the 'default' rate card.",
    )
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig.from_yaml_or_defaults)
    associations: AssociationConfig = Field(default_factory=AssociationConfig)
    ground_truth: GroundTruthConfig = Field(default_factory=GroundTruthConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    offline: OfflineConfig = Field(default_factory=OfflineConfig)

    @classmethod
    def from_yaml(cls, path: str | Path, strict: bool = True) -> "BenchmarkConfig":
        """Load configuration from YAML.

        Raises `FileNotFoundError` when `strict` (the default): silently falling back to built-in
        defaults after a path typo used to produce a run that looked fine but ignored every
        setting the engineer had just edited.
        """
        p = Path(path)
        if not p.exists():
            if strict:
                raise FileNotFoundError(
                    f"Benchmark config not found: '{p}'. Pass an existing YAML path, or call "
                    f"BenchmarkConfig() directly for built-in defaults."
                )
            return cls()
        with open(p, encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Benchmark config '{p}' must contain a YAML mapping, got {type(raw).__name__}.")
        if "taxonomy" not in raw:
            tax_file = raw.get("taxonomy_file")
            if tax_file:
                # Resolve relative to the config file that declares it, not to the process CWD --
                # otherwise `shelf-benchmark --config /abs/path/cfg.yaml` picked up (or silently
                # missed) a taxonomy depending on where the command happened to be launched from.
                tax_path = Path(tax_file)
                if not tax_path.is_absolute():
                    tax_path = p.resolve().parent / tax_path
                raw["taxonomy"] = TaxonomyConfig.from_yaml_or_defaults(tax_path).model_dump()
        return cls.model_validate(raw)

    def has_explicit_pricing(self, model_name: str) -> bool:
        """True when the model has its own rate card (i.e. cost is not a 'default' guess)."""
        clean_name = model_name.split("/")[-1]
        if clean_name in self.pricing_per_million_tokens:
            return True
        return any(
            key != "default" and key in clean_name for key in self.pricing_per_million_tokens
        )

    def get_pricing(self, model_name: str) -> ModelPricing:
        """Resolve token pricing for a model.

        Matching is exact first, then longest-substring, so `gemini-3.5-flash-lite` can never be
        priced as `gemini-3-flash-preview` just because of dictionary insertion order.
        """
        clean_name = model_name.split("/")[-1]
        if clean_name in self.pricing_per_million_tokens:
            return self.pricing_per_million_tokens[clean_name]
        candidates = [
            key
            for key in self.pricing_per_million_tokens
            if key != "default" and key in clean_name
        ]
        if candidates:
            return self.pricing_per_million_tokens[max(candidates, key=len)]
        return self.pricing_per_million_tokens.get("default", ModelPricing())


def normalize_vertex_gemini_model_id(model_name: str, for_live_vertex: bool = False) -> str:
    """Enforce the no-legacy-models policy (rejecting 1.x / 2.0 / 2.5) while keeping model_name 100% verbatim under user control."""
    raw = str(model_name or "gemini-3-flash-preview").strip()
    lower = raw.lower()
    if any(legacy in lower for legacy in ("gemini-2.5", "gemini-2.0", "gemini-1.5", "gemini-1.0")):
        raise ValueError(
            f"Legacy model '{raw}' is prohibited by benchmark policy (too old). "
            "Pass the exact Gemini 3+ or custom model ID you want to test (e.g., 'gemini-3-flash-preview', 'gemini-3.1-pro-preview')."
        )
    # Return the exact model ID specified by the user — never substitute or rewrite it.
    return raw


