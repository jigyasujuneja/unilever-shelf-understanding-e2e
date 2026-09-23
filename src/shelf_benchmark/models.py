"""Canonical Pydantic Data Schemas for Detection (Facings-Only), 7-Dimension HUL Classification, Hybrid Search Matching, Fine-Tuning, Telemetry, and Reports."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from shelf_benchmark.config import PACKAGED_TAXONOMY_PATH

# =====================================================================
# 1. Structured Gemini Output Schemas (used in response_schema)
# =====================================================================

class DetectedProductItem(BaseModel):
    """Single detected front-facing product slot on a shelf image (excludes depth-stacked duplicates)."""
    product_index: int = Field(
        description="1-based index of the front-facing product slot on the shelf (ordered left-to-right)"
    )
    bbox_2d: List[int] = Field(
        description="Normalized 2D bounding box [ymin, xmin, ymax, xmax] scaled 0 to 1000 of ONLY the front-most unit in this facing"
    )
    shelf_row: str = Field(
        default="",
        description="Shelf row identifier (top, middle, bottom); \"\" if the model did not report one"
    )
    position_on_shelf: int = Field(
        default=1,
        description="1-based horizontal facing slot index from left to right on that shelf row"
    )
    is_front_facing: bool = Field(
        default=True,
        description="True ONLY if this is the front-most visible product in the facing slot (do NOT include products stacked behind it in depth)"
    )
    visual_description: str = Field(
        default="",
        description="Visual description of packaging color, shape, and visible text"
    )
    preliminary_brand_hint: str = Field(
        default="",
        description="Visible brand text read directly from packaging if legible, else Unknown"
    )
    confidence: float = Field(
        default=0.0,
        description="Detection confidence score between 0.0 and 1.0. 0.0 means not reported.",
    )


class ProductDetectionOutput(BaseModel):
    """Structured output from ProductDetectionTask."""
    total_detected_products: int = Field(
        description="Total count of distinct front-facing product slots (excluding depth-stacked units behind front facings)"
    )
    detected_products: List[DetectedProductItem] = Field(default_factory=list)


class ClassifiedProductItem(BaseModel):
    """Single classified product facing following the 7-Dimension HUL Classification Taxonomy."""
    product_index: int = Field(description="1-based index of the front-facing product from left to right")
    bbox_2d: List[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="Normalized 2D bounding box [ymin, xmin, ymax, xmax] scaled 0 to 1000"
    )
    shelf_row: str = Field(default="", description="Shelf row (top, middle, bottom); \"\" if not reported")
    position_on_shelf: int = Field(default=1, description="1-based horizontal facing slot from left to right")
    # Defaults are "" (= not reported), never a plausible value. These fields are part of the
    # Gemini `response_schema`, so a value the model omits is filled in by Pydantic and then
    # reported and *scored* as if the model had predicted it. The previous defaults were
    # "Skin Care" / "Face Wash" / "tube" / "Single" / "Medium / Regular (50-100g)" (a size label
    # that does not even exist in the taxonomy), and `is_hul_brand=True`, i.e. an unanswered
    # question silently became a confident claim about a Unilever brand.
    category: str = Field(
        default="",
        description="Dimension 1 (Category): Configured retail category (e.g., Hair Care, Oral Care, Laundry, Skin Care, Skin Cleansing, etc.)"
    )
    subcategory: str = Field(
        default="",
        description="Dimension 2 (Subcategory): Configured subcategory (e.g., Shampoo, Mouthwash, Soaps, Face Wash, Body Wash, Detergent, Cream, Gel)"
    )
    brand: str = Field(
        description="Dimension 3 (Brand): Exact brand name read from packaging (HUL portfolio or non-HUL brand)"
    )
    is_hul_brand: Optional[bool] = Field(
        default=None,
        description="True if brand belongs to the configured Hindustan Unilever (HUL) brand portfolio"
    )
    variant: str = Field(
        default="",
        description="Dimension 4 (Variant): Specific product line + variant/ingredient/claim read from packaging"
    )
    packaging_type: str = Field(
        default="",
        description="Dimension 5 (Packaging type): e.g., box, jar, sachet, tube, bottle, pouch, bar"
    )
    pack_type: str = Field(
        default="",
        description="Dimension 6 (Pack type): 'Single' or 'Multiple' (multipack/bundled)"
    )
    size: str = Field(
        default="",
        description="Dimension 7 (Size): Rule-derived size bucket (e.g., Sachet/Trial <25g, Small/Compact 25-50g, Medium/Regular 51-100g, Large/Family >100g)"
    )
    product_name: str = Field(
        default="",
        description="Full synthesized product display name (Brand + Sub-brand + Variant + Subcategory)"
    )
    confidence: float = Field(
        default=0.0,
        description=(
            "Classification confidence between 0.0 and 1.0. 0.0 means the model did not report "
            "one -- it previously defaulted to 0.95, so an unanswered question was recorded as "
            "high confidence."
        ),
    )
    extra_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional product/shelf attributes beyond the base 8 dimensions (>8 attributes support)"
    )


class ProductClassificationOutput(BaseModel):
    """Structured output from ProductClassificationTask."""
    total_classified_products: int = Field(description="Total count of front-facing products classified")
    distinct_brands_found: List[str] = Field(default_factory=list, description="List of unique brands visually identified")
    classified_products: List[ClassifiedProductItem] = Field(default_factory=list)


class MatchedProductItem(BaseModel):
    """Single product prepared for Hybrid Search (Vector + Lexical) with all 7 HUL Taxonomy filters."""
    product_index: int = Field(description="1-based index of the product facing")
    bbox_2d: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    shelf_row: str = Field(default="", description="Shelf row (top, middle, bottom); \"\" if not reported")
    position_on_shelf: int = Field(default=1)
    # As with ClassifiedProductItem, these are part of the Gemini `response_schema`: a field the
    # model omits is filled in by Pydantic and then reported and *scored* as a prediction. The
    # previous defaults ("Skin Care" / "Face Wash" / "tube" / "Single" / "Medium / Regular")
    # meant an unanswered question became a confident, wrong, graded answer.
    category: str = Field(default="")
    subcategory: str = Field(default="")
    brand: str = Field(description="Visually extracted brand filter for hybrid search")
    variant: str = Field(default="", description="Visually extracted variant")
    packaging_type: str = Field(default="")
    pack_type: str = Field(default="")
    size: str = Field(default="")
    product_name: str = Field(description="Visually extracted product title")
    lexical_search_keywords: List[str] = Field(
        default_factory=list,
        description="Sparse keyword tokens (OCR text, category, subcategory, brand, variant, packaging_type, pack_type, size) for BM25/keyword search"
    )
    dense_embedding_text: str = Field(
        default="",
        description="Rich semantic & visual passage synthesized for dense vector embedding search against a product catalog"
    )
    matched_sku_id: str = Field(
        default="HYBRID_SEARCH_READY",
        description="Matched catalog SKU ID if a catalog is provided, otherwise HYBRID_SEARCH_READY"
    )
    match_confidence: float = Field(
        default=0.0,
        description="Confidence in extracted search attributes (0.0 to 1.0). 0.0 means not reported.",
    )
    planogram_compliant: Optional[bool] = Field(default=None, description="True/False if planogram provided, else null")
    extra_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional product/shelf attributes beyond the base 8 dimensions"
    )


class ProductMatchingOutput(BaseModel):
    """Structured output from ProductMatchingTask."""
    total_matched_products: int = Field(description="Number of shelf facings processed for hybrid search / catalog matching")
    overall_planogram_compliance_rate: Optional[float] = Field(
        default=None, description="Planogram compliance rate if planogram provided, else null"
    )
    matched_products: List[MatchedProductItem] = Field(default_factory=list)


# =====================================================================
# 2. Canonical Association, Ground Truth, Telemetry & Report Models
# =====================================================================

class ShelfAssociationRecord(BaseModel):
    """Canonical record associating a shelf image with catalog, optional planogram, and ground truth."""
    association_id: str
    shelf_image_uri: str
    local_shelf_image_path: Optional[str] = None
    store_id: Optional[str] = None
    aisle_category: Optional[str] = None
    catalog_uri: Optional[str] = None
    planogram_uri: Optional[str] = None
    ground_truth_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroundTruthProductItem(BaseModel):
    """Canonical ground truth item for a single product facing.

    `bbox_2d` is always `[ymin, xmin, ymax, xmax]` normalized to 0..1000 by the provider layer,
    whatever format the annotation vendor delivered (see `GroundTruthSchemaMapping.bbox_format`).
    """

    item_id: int
    brand: str
    product_name: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    variant: Optional[str] = None
    packaging_type: Optional[str] = None
    pack_type: Optional[str] = None
    size: Optional[str] = None
    sku_id: Optional[str] = None
    bbox_2d: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    shelf_row: str = ""  # "" = the annotation did not record a row
    back_row: bool = Field(
        default=False,
        description="True if this unit sits behind a front facing (excluded from front-facing scoring).",
    )
    occluded: bool = Field(default=False, description="True if substantially occluded by another product.")
    extra_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional ground-truth attributes beyond the base 8 dimensions (>8 attributes support)"
    )


class ImageGroundTruth(BaseModel):
    """Canonical ground truth for a shelf image."""

    image_id: str
    total_main_shelf_facings: int
    expected_brands: List[str] = Field(default_factory=list)
    items: List[GroundTruthProductItem] = Field(default_factory=list)
    gt_version: str = "unversioned"
    source_key: Optional[str] = Field(
        default=None, description="Key this entry was looked up by, for join debugging."
    )
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class TokenUsageMetrics(BaseModel):
    """Token counts extracted from Vertex AI Gemini response."""
    input_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


class CostMetrics(BaseModel):
    """Separated GCP cost breakdown for a single shelf image and per front-facing product.

    Includes granular compute sub-buckets (vCPU, Memory, GPU/TPU Accelerator, Request fee),
    Active Container Compute vs External API-Wait Idle Tax attribution, and per-1,000-image
    Cost-Latency Pareto efficiency metrics so engineers can evaluate hardware tradeoffs.
    """

    billing_source: str = "yaml_rate_table"
    rates_from_live_catalog: bool = False
    includes_modelled_infrastructure: bool = True
    traffic_type: str = "ON_DEMAND"  # "ON_DEMAND", "PROVISIONED_THROUGHPUT", or "HYBRID_SPILLOVER"
    hardware_profile: str = "2.0 vCPU / 4.0 GiB RAM (CPU-only Cloud Run)"
    input_cost_usd: float = 0.0
    thinking_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    vertex_ai_payg_tokens_usd: float = 0.0
    vertex_ai_provisioned_throughput_usd: float = 0.0
    vertex_ai_embeddings_and_vision_usd: float = 0.0
    # Granular Cloud Run / Accelerator Compute Sub-Buckets
    cloud_run_vcpu_usd: float = 0.0
    cloud_run_memory_usd: float = 0.0
    cloud_run_accelerator_usd: float = 0.0
    cloud_run_request_fee_usd: float = 0.0
    cloud_run_compute_usd: float = 0.0
    # Active Local Compute vs External API-Wait Cost Attribution (Cloud Run bills for full request wall-clock)
    container_cpu_active_ms: float = 0.0
    external_api_wait_ms: float = 0.0
    compute_active_processing_usd: float = 0.0
    compute_api_wait_idle_tax_usd: float = 0.0
    compute_share_of_total_cost_pct: float = 0.0
    # Storage & Observability + Totals + Scaling Metrics
    gcs_and_observability_usd: float = 0.0
    cost_per_shelf_image_usd: float = 0.0
    cost_per_product_usd: float = 0.0
    cost_per_1k_images_usd: float = 0.0
    cost_latency_pareto_index: float = 0.0
    product_count: int = 0
    gcp_billing_labels: Dict[str, str] = Field(default_factory=dict)


class AccuracyMetrics(BaseModel):
    """Accuracy metrics evaluated against Ground Truth (or placeholder status when GT is absent).

    Field names deliberately do NOT hardcode a threshold: the operative threshold is reported in
    `iou_threshold`, and the matchers used are reported in `brand_matcher` / `product_matcher`, so
    a number can never be read out of context. See `docs/EVALUATION_PROTOCOL.md`.
    """

    ground_truth_available: bool = False
    accuracy_status: str = "PLACEHOLDER_AWAITING_GROUND_TRUTH"
    gt_version: str = "unversioned"
    # Scoring policy actually applied (provenance for every number below).
    iou_threshold: Optional[float] = None
    pairing_strategy: Optional[str] = None
    brand_matcher: Optional[str] = None
    product_matcher: Optional[str] = None

    ground_truth_count: Optional[int] = None
    predicted_count: int = 0
    depth_duplicates_filtered: int = 0
    # Confusion-matrix counts. These are Optional for the same reason the ratios below are:
    # "0 true positives" is a measurement meaning the model matched nothing, whereas None
    # means no ground truth existed to match against. Collapsing the two is how an unscored
    # benchmark comes to look like a failing one.
    matched_pairs: Optional[int] = None
    true_positives: Optional[int] = None
    false_positives: Optional[int] = None
    false_negatives: Optional[int] = None

    count_accuracy: Optional[float] = None
    detection_precision: Optional[float] = None
    detection_recall: Optional[float] = None
    detection_f1: Optional[float] = None
    average_precision_at_50: Optional[float] = None
    map_50_95: Optional[float] = None
    pr_curve_points: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Confidence-ranked Precision-Recall curve points [{'confidence', 'precision', 'recall'}].",
    )
    mean_iou: Optional[float] = None
    mean_iou_matched: Optional[float] = None
    brand_classification_accuracy: Optional[float] = None
    brand_set_recall: Optional[float] = None
    product_classification_accuracy: Optional[float] = None
    sku_matching_accuracy: Optional[float] = None
    planogram_compliance_rate: Optional[float] = None
    per_attribute_accuracy: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-attribute accuracy across all core and custom (>8) attributes when ground truth is connected."
    )
    macro_attribute_accuracy: Optional[float] = Field(
        default=None,
        description="Mean accuracy across all evaluated attributes (core + custom attributes)."
    )



class RowLevelReportItem(BaseModel):
    """Detailed row-level report entry (one row per detected/classified/matched front-facing product)."""
    run_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    # Both default to "" and are stamped by the pipeline. They used to default to
    # "classification" / "single_pass_full_shelf", and `tasks/base.py` did
    # `if rows[0].separation_approach: approach = rows[0].separation_approach` -- an
    # always-true test, so any task whose rows did not set the field had its real approach id
    # silently replaced by a *classification* id in reports, cost records and billing labels.
    task_type: str = ""
    separation_approach: str = ""
    model_name: str = ""
    shelf_image_uri: str = ""
    store_id: Optional[str] = None
    start_time: str = ""
    end_time: str = ""
    image_latency_ms: float = 0.0
    product_index: int = 1
    shelf_row: str = ""  # "" = not predicted; never guess a shelf position
    position_on_shelf: int = 1
    is_front_facing: bool = True
    bbox_ymin: int = 0
    bbox_xmin: int = 0
    bbox_ymax: int = 0
    bbox_xmax: int = 0
    crop_image_path: Optional[str] = None
    # 7-Dimension HUL Taxonomy Fields + Unlimited Custom Attributes (>8 attributes support)
    predicted_category: str = ""
    predicted_subcategory: str = ""
    predicted_brand: str = ""
    is_hul_brand: Optional[bool] = None
    predicted_variant: str = ""
    predicted_packaging: str = ""
    predicted_pack_type: str = ""
    predicted_size: str = ""
    rule_derived_size_bucket: str = ""
    predicted_product_name: str = ""
    confidence: float = 0.0
    extra_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional predicted product/shelf attributes beyond the base 8 dimensions"
    )
    # Hybrid Search Fields
    lexical_search_keywords: str = ""
    dense_embedding_text: str = ""
    embedding_vector_dim: Optional[int] = None
    matched_sku_id: Optional[str] = None
    planogram_compliant: Optional[bool] = None
    # Ground Truth Evaluation Fields (populated only when ground truth is connected)
    gt_status: str = "PLACEHOLDER_AWAITING_GT"  # PLACEHOLDER_AWAITING_GT | MATCHED | UNMATCHED_FALSE_POSITIVE
    gt_version: str = "unversioned"
    gt_item_id: Optional[int] = None
    gt_brand: Optional[str] = None
    gt_product_name: Optional[str] = None
    gt_sku_id: Optional[str] = None
    iou_with_gt: Optional[float] = None
    iou_threshold: Optional[float] = None
    is_true_positive: Optional[bool] = None
    brand_correct: Optional[bool] = None
    product_correct: Optional[bool] = None
    sku_correct: Optional[bool] = None
    # Token & 100% Separated All-In GCP Cost Fields
    input_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    billing_source: str = "yaml_rate_table"
    traffic_type: str = "ON_DEMAND"
    vertex_ai_payg_tokens_usd: float = 0.0
    vertex_ai_provisioned_throughput_usd: float = 0.0
    vertex_ai_embeddings_and_vision_usd: float = 0.0
    cloud_run_compute_usd: float = 0.0
    gcs_and_observability_usd: float = 0.0
    cost_per_shelf_image_usd: float = 0.0
    cost_per_product_usd: float = 0.0


def build_execution_trace_metadata(
    *,
    run_id: str,
    trace_id: str,
    span_id: str,
    task_type: str,
    separation_approach: str,
    model_name: str,
    shelf_image_uri: str,
    latency_ms: float,
    tokens: TokenUsageMetrics,
    cost: CostMetrics,
    accuracy: AccuracyMetrics,
    facings_count: int,
    otel_log_path: str = "reports/otel_logs.jsonl",
    gcp_project_id: str = "unilever-shelf-understanding",
    gcp_log_name: str = "unilever-shelf-benchmark-otel",
    taxonomy_source: str = str(PACKAGED_TAXONOMY_PATH),
    ground_truth_provider: str = "none",
    reference_catalog_uri: Optional[str] = None,
    custom_stages: Optional[List[str]] = None,
    api_calls_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a comprehensive, human- and UI-readable trace explanation for any benchmark execution."""
    approach_blueprints: Dict[str, Dict[str, Any]] = {
        "single_pass_full_shelf": {
            "call_topology": "1 API Call (Single-Step Open-Vocab Detection + LLM-Generated Brand & N-Dim Classification)",
            "api_calls_count": 1,
            "detect_and_classify_mode": "Simultaneous in 1 step (Full shelf image -> BBoxes + Open-Vocabulary Generated Brand & N-Dim Taxonomy JSON in 1 VLM call)",
            "models_invoked": [
                {"stage": "Stage 1 (Joint Detect + Open-Vocab Classify)", "model": model_name, "type": "Vertex AI Multimodal VLM"},
            ],
            "stages": [
                "Stage 1: Single VLM call localizes front-facing [ymin,xmin,ymax,xmax] boxes and generates brand/attributes openly from package text",
                "Stage 2: Post-hoc geometric Depth NMS (`deduplicate_depth_stacked_facings`) + rule-derived size bucketing",
            ],
        },
        "open_vocab_brand_plus_catalog_resolver": {
            "call_topology": "1 VLM Call (Open-Vocab Generation) + Post-Hoc O(1) 2,000-Brand Catalog Resolver",
            "api_calls_count": 1,
            "detect_and_classify_mode": "Open-Vocabulary LLM Brand Generation -> O(1) Master Catalog Canonical Resolver (`resolve_brand_against_catalog`)",
            "models_invoked": [
                {"stage": "Stage 1 (Open-Vocabulary VLM Generation)", "model": model_name, "type": "Vertex AI Multimodal VLM"},
                {"stage": "Stage 2 (Canonical Brand Resolver)", "model": "O(1) Normalized Master Brand Index (2,000+ Brands)", "type": "Deterministic Catalog Resolver"},
            ],
            "stages": [
                "Stage 1: VLM determines & generates brand + N-Dim attributes openly from packaging (zero brand list in prompt)",
                "Stage 2: Post-hoc O(1) catalog resolver (`resolve_brand_against_catalog`) snaps generated brand to 2,000+ brand catalog + Depth NMS",
            ],
        },
        "configurable_multi_attribute_vlm": {
            "call_topology": "Configurable N-Attribute VLM (Single Call or Grouped Multi-Call by Attribute Type)",
            "api_calls_count": api_calls_count or 1,
            "detect_and_classify_mode": "Configurable N-Attribute Extraction (predicts >8 attributes in 1 VLM call or split across taxonomy.attribute_call_groups)",
            "models_invoked": [
                {"stage": "Stage 1..G (Configurable Attribute Groups)", "model": model_name, "type": "Vertex AI Multimodal VLM"},
            ],
            "stages": [
                "Stage 1: Extract core + custom attributes (`taxonomy.custom_attributes`) either in 1 VLM call or grouped via `taxonomy.attribute_call_groups`",
                "Stage 2: Merge all attribute groups per facing (`extra_attributes`) + Front-Facing Column Depth NMS",
            ],
        },
        "single_step_detect_classify_and_match": {
            "call_topology": "1 Unified VLM Call + Dense Vector Embedding (Single-Step Detect + 7-Dim Classify + Hybrid SKU Match)",
            "api_calls_count": 1,
            "detect_and_classify_mode": "Simultaneous in 1 step (1 VLM call extracts BBoxes + 7-Dim Taxonomy + BM25/Dense SKU passages)",
            "models_invoked": [
                {"stage": "Stage 1 (Joint Detect + Classify + SKU Passage)", "model": model_name, "type": "Vertex AI Multimodal VLM"},
                {"stage": "Stage 2 (3072-D Vector Index)", "model": "gemini-embedding-001 (3072-D)", "type": "Vertex AI Text Embeddings"},
            ],
            "stages": [
                "Stage 1: Single VLM call extracts front-facing [ymin,xmin,ymax,xmax] boxes, 7-Dim taxonomy, and hybrid SKU search passages",
                "Stage 2: Geometric Depth NMS (`deduplicate_depth_stacked_facings`) + 3072-D vector embedding (`gemini-embedding-001`)",
            ],
        },
        "two_stage_bbox_guided_nms": {
            "call_topology": "2 Separate API Calls (Stage 1 Detector -> Stage 2 Coordinate-Guided Classifier)",
            "api_calls_count": 2,
            "detect_and_classify_mode": "Separated into 2 calls (Call 1 localizes boxes + Depth NMS; Call 2 classifies locked coordinates)",
            "models_invoked": [
                {"stage": "Stage 1 (Front-Facing Detection)", "model": model_name, "type": "Vertex AI VLM Detector"},
                {"stage": "Stage 2 (Coordinate-Conditioned Classification)", "model": model_name, "type": "Vertex AI VLM Classifier"},
            ],
            "stages": [
                "Stage 1: VLM Detection extracts candidate boxes -> Geometric Depth NMS suppresses back-row duplicates",
                "Stage 2: Full shelf image + locked front-facing [ymin,xmin,ymax,xmax] coordinates passed to VLM for 7-Dim classification",
            ],
        },
        "two_stage_physical_crop_per_facing": {
            "call_topology": "2 VLM API Calls + Physical PIL Cropping & Montage Strip",
            "api_calls_count": 2,
            "detect_and_classify_mode": "Separated with physical cropping (Call 1 detects boxes; Pillow crops each facing PNG + numbered montage; Call 2 reads fine print)",
            "models_invoked": [
                {"stage": "Stage 1 (Front-Facing Detection)", "model": model_name, "type": "Vertex AI VLM Detector"},
                {"stage": "Stage 2 (Physical Crop + Montage Fine-Print Classification)", "model": model_name, "type": "Vertex AI VLM Classifier"},
            ],
            "stages": [
                "Stage 1: VLM Detection + Front-Facing Column Depth NMS",
                "Stage 2: Physical PIL cropping per facing (`facing_01..N.png`) + numbered visual montage (`montage_all_facings.png`)",
                "Stage 3: VLM fine-print reading on physical crops + montage for 7-Dim HUL Taxonomy & size rules",
            ],
        },
        "class_agnostic_visual_embedding": {
            "call_topology": "3-Stage CV Detector + 1408-D Visual Embedding + Vector Search (Zero Generative VLM Classification)",
            "api_calls_count": 2,
            "detect_and_classify_mode": "Separated non-generative pipeline (Call 1 detects class='product' boxes; Call 2 embeds crops via multimodalembedding@001; Stage 3 matches vectors)",
            "models_invoked": [
                {"stage": "Stage 1 (Class-Agnostic 'product' Detector)", "model": model_name, "type": "Single-Class Spatial Detector"},
                {"stage": "Stage 2 (1408-D Visual Crop Embedder)", "model": "multimodalembedding@001", "type": "Vertex AI Contrastive Vision Embedder (1408-D)"},
                {"stage": "Stage 3 (Catalog Matcher)", "model": "ScaNN / Cosine ANN Vector Search", "type": "Vector Similarity Index"},
            ],
            "stages": [
                "Stage 1: Class-agnostic object detector (`class='product'`) + Front-Facing Column Depth NMS",
                "Stage 2: Physical PIL crop extraction + 1408-D visual embedding via `multimodalembedding@001`",
                "Stage 3: Cosine / ScaNN nearest-neighbor lookup against reference catalog embeddings",
            ],
        },
        "cloud_vision_visual_embedding": {
            "call_topology": "3-Stage Cloud Vision API + 1408-D Visual Embedding + Vector Search",
            "api_calls_count": 2,
            "detect_and_classify_mode": "Separated Cloud Vision + Embedding pipeline (Cloud Vision OBJECT_LOCALIZATION -> multimodalembedding@001 -> Vector Search)",
            "models_invoked": [
                {"stage": "Stage 1 (Cloud Vision Detector)", "model": "vision.googleapis.com (OBJECT_LOCALIZATION)", "type": "Google Cloud Vision API"},
                {"stage": "Stage 2 (1408-D Visual Crop Embedder)", "model": "multimodalembedding@001", "type": "Vertex AI Contrastive Vision Embedder (1408-D)"},
                {"stage": "Stage 3 (Catalog Matcher)", "model": "ScaNN / Cosine ANN Vector Search", "type": "Vector Similarity Index"},
            ],
            "stages": [
                "Stage 1: Google Cloud Vision `OBJECT_LOCALIZATION` + Front-Facing Column Depth NMS",
                "Stage 2: Physical PIL crop extraction + 1408-D visual embedding via `multimodalembedding@001`",
                "Stage 3: Cosine / ScaNN nearest-neighbor lookup against reference catalog embeddings",
            ],
        },
    }

    if separation_approach in approach_blueprints:
        bp = approach_blueprints[separation_approach]
    elif task_type == "detection":
        bp = {
            "call_topology": "1 API Call (Standalone Front-Facing Spatial Detection)",
            "api_calls_count": 1,
            "detect_and_classify_mode": "Detection only (1 call returns [ymin,xmin,ymax,xmax] boxes + Depth NMS; no brand classification)",
            "models_invoked": [{"stage": "Stage 1 (Spatial Detection)", "model": model_name, "type": "Vertex AI VLM Detector"}],
            "stages": [
                "Stage 1: Zero-shot front-facing bounding box detection `[ymin, xmin, ymax, xmax]` (0..1000)",
                "Stage 2: Geometric Front-Facing Column Depth NMS (`deduplicate_depth_stacked_facings`)",
            ],
        }
    elif task_type == "matching":
        bp = {
            "call_topology": "2 API Calls (VLM Structured Extraction -> 3072-D Hybrid Embedding + BM25 RRF)",
            "api_calls_count": 2,
            "detect_and_classify_mode": "VLM extracts structured product passages + `gemini-embedding-001` (3072-D) dense vectors + BM25 lexical RRF",
            "models_invoked": [
                {"stage": "Stage 1 (Structured Passage Extraction)", "model": model_name, "type": "Vertex AI VLM"},
                {"stage": "Stage 2 (Dense Text Embedding)", "model": "gemini-embedding-001 (3072-D)", "type": "Vertex AI Text Embeddings"},
            ],
            "stages": [
                "Stage 1: VLM extracts product attributes + sparse lexical BM25 keywords + dense embedding passage",
                "Stage 2: `gemini-embedding-001` generates 3072-D vectors combined via Reciprocal Rank Fusion (RRF)",
            ],
        }
    elif task_type == "fine_tuning":
        bp = {
            "call_topology": "1 API Call (Zero-shot structured inference) -> Vertex AI SFT dataset build",
            "api_calls_count": 1,
            "detect_and_classify_mode": (
                "Zero-shot structured extraction used as distillation input, then serialised to a "
                "Vertex AI supervised fine-tuning (SFT) JSONL dataset and uploaded to GCS"
            ),
            "models_invoked": [
                {"stage": "Stage 1 (Zero-shot Structured Inference)", "model": model_name, "type": "Vertex AI VLM"},
            ],
            "stages": [
                "Stage 1: Zero-shot structured extraction over the shelf image",
                "Stage 2: Build Vertex AI SFT JSONL examples and upload the dataset to GCS",
            ],
        }
    else:
        bp = {
            "call_topology": f"Custom Plugin Pipeline ({separation_approach})",
            "api_calls_count": 1,
            "detect_and_classify_mode": f"Custom plugin execution (`{separation_approach}`) with shared CommonLayerContext",
            "models_invoked": [{"stage": "Custom Pipeline", "model": model_name, "type": "Plugin / Custom Callable"}],
            "stages": custom_stages or [f"Stage 1: Custom approach `{separation_approach}` execution"],
        }

    return {
        "run_id": run_id,
        "task_type": task_type,
        "separation_approach": separation_approach,
        "call_topology": bp["call_topology"],
        "api_calls_count": bp["api_calls_count"],
        "detect_and_classify_mode": bp["detect_and_classify_mode"],
        "models_invoked": bp["models_invoked"],
        "stages": custom_stages or bp["stages"],
        "data_used": {
            "shelf_image_uri": shelf_image_uri,
            "taxonomy_source": taxonomy_source,
            "ground_truth_provider": ground_truth_provider,
            "ground_truth_available": accuracy.ground_truth_available,
            "gt_version": accuracy.gt_version,
            "reference_catalog_uri": reference_catalog_uri,
            "front_facings_detected": facings_count,
            "depth_duplicates_filtered": accuracy.depth_duplicates_filtered,
        },
        "cost_summary": {
            "cost_per_shelf_image_usd": cost.cost_per_shelf_image_usd,
            "cost_per_product_usd": cost.cost_per_product_usd,
            "billing_source": cost.billing_source,
            "traffic_type": cost.traffic_type,
            "includes_modelled_infrastructure": cost.includes_modelled_infrastructure,
            "buckets_usd": {
                "vertex_ai_payg_tokens_usd": cost.vertex_ai_payg_tokens_usd,
                "vertex_ai_provisioned_throughput_usd": cost.vertex_ai_provisioned_throughput_usd,
                "vertex_ai_embeddings_and_vision_usd": cost.vertex_ai_embeddings_and_vision_usd,
                "cloud_run_compute_usd": cost.cloud_run_compute_usd,
                "gcs_and_observability_usd": cost.gcs_and_observability_usd,
            },
            "tokens": {
                "input_tokens": tokens.input_tokens,
                "thinking_tokens": tokens.thinking_tokens,
                "output_tokens": tokens.output_tokens,
                "total_tokens": tokens.total_tokens,
            },
        },
        "accuracy_summary": {
            "accuracy_status": accuracy.accuracy_status,
            "ground_truth_available": accuracy.ground_truth_available,
            "gt_version": accuracy.gt_version,
            "iou_threshold": accuracy.iou_threshold,
            "detection_precision": accuracy.detection_precision,
            "detection_recall": accuracy.detection_recall,
            "detection_f1": accuracy.detection_f1,
            "mean_iou_matched": accuracy.mean_iou_matched,
            "brand_classification_accuracy": accuracy.brand_classification_accuracy,
            "product_classification_accuracy": accuracy.product_classification_accuracy,
            "count_accuracy": accuracy.count_accuracy,
            "note": (
                "Metrics are None (not 0.0) because ground truth is not connected yet. "
                "Re-score this run for free anytime with `shelf-benchmark score`."
                if not accuracy.ground_truth_available
                else f"Evaluated against ground truth ({accuracy.gt_version}) at IoU >= {accuracy.iou_threshold}."
            ),
        },
        "opentelemetry": {
            "trace_id": trace_id,
            "span_id": span_id,
            "otel_log_path": otel_log_path,
            "local_jq_command": f"jq 'select(.TraceId == \"{trace_id}\")' {otel_log_path}",
            "gcp_cloud_logging_query": (
                f'logName="projects/{gcp_project_id}/logs/{gcp_log_name}" '
                f'AND trace="projects/{gcp_project_id}/traces/{trace_id}"'
            ),
        },
    }


class TaskExecutionResult(BaseModel):
    """Complete result of executing one task/approach on one image with one model."""
    run_id: str
    trace_id: str
    span_id: str
    task_type: str
    separation_approach: str = "single_pass_full_shelf"
    model_name: str
    shelf_image_uri: str
    start_time: str
    end_time: str
    latency_ms: float
    tokens: TokenUsageMetrics
    cost: CostMetrics
    accuracy: AccuracyMetrics
    raw_output: Dict[str, Any] = Field(default_factory=dict)
    row_level_items: List[RowLevelReportItem] = Field(default_factory=list)
    status: str = "SUCCESS"
    error_message: Optional[str] = None

    @property
    def execution_trace(self) -> Dict[str, Any]:
        """Return structured trace metadata describing calls, models, cost, accuracy, data, and OTel lookup."""
        if "execution_trace" in self.raw_output and isinstance(self.raw_output["execution_trace"], dict):
            return self.raw_output["execution_trace"]
        return build_execution_trace_metadata(
            run_id=self.run_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            task_type=self.task_type,
            separation_approach=self.separation_approach,
            model_name=self.model_name,
            shelf_image_uri=self.shelf_image_uri,
            latency_ms=self.latency_ms,
            tokens=self.tokens,
            cost=self.cost,
            accuracy=self.accuracy,
            facings_count=len(self.row_level_items),
        )
