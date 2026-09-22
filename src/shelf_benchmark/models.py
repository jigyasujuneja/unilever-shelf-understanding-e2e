"""Canonical Pydantic Data Schemas for Detection (Facings-Only), 7-Dimension HUL Classification, Hybrid Search Matching, Fine-Tuning, Telemetry, and Reports."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
        default="middle",
        description="Shelf row identifier (e.g., top, middle, bottom)"
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
        default=0.95,
        description="Detection confidence score between 0.0 and 1.0"
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
    shelf_row: str = Field(default="middle", description="Shelf row (top, middle, bottom)")
    position_on_shelf: int = Field(default=1, description="1-based horizontal facing slot from left to right")
    category: str = Field(
        default="Skin Care",
        description="Dimension 1 (Category): Configured retail category (e.g., Hair Care, Oral Care, Laundry, Skin Care, Skin Cleansing, etc.)"
    )
    subcategory: str = Field(
        default="Face Wash",
        description="Dimension 2 (Subcategory): Configured subcategory (e.g., Shampoo, Mouthwash, Soaps, Face Wash, Body Wash, Detergent, Cream, Gel)"
    )
    brand: str = Field(
        description="Dimension 3 (Brand): Exact brand name read from packaging (HUL portfolio or non-HUL brand)"
    )
    is_hul_brand: bool = Field(
        default=True,
        description="True if brand belongs to the configured Hindustan Unilever (HUL) brand portfolio"
    )
    variant: str = Field(
        default="",
        description="Dimension 4 (Variant): Specific product line + variant/ingredient/claim read from packaging"
    )
    packaging_type: str = Field(
        default="tube",
        description="Dimension 5 (Packaging type): e.g., box, jar, sachet, tube, bottle, pouch, bar"
    )
    pack_type: str = Field(
        default="Single",
        description="Dimension 6 (Pack type): 'Single' or 'Multiple' (multipack/bundled)"
    )
    size: str = Field(
        default="Medium / Regular (50-100g)",
        description="Dimension 7 (Size): Rule-derived size bucket (e.g., Sachet/Trial <25g, Small/Compact 25-50g, Medium/Regular 51-100g, Large/Family >100g)"
    )
    product_name: str = Field(
        default="",
        description="Full synthesized product display name (Brand + Sub-brand + Variant + Subcategory)"
    )
    confidence: float = Field(default=0.95, description="Classification confidence between 0.0 and 1.0")


class ProductClassificationOutput(BaseModel):
    """Structured output from ProductClassificationTask."""
    total_classified_products: int = Field(description="Total count of front-facing products classified")
    distinct_brands_found: List[str] = Field(default_factory=list, description="List of unique brands visually identified")
    classified_products: List[ClassifiedProductItem] = Field(default_factory=list)


class MatchedProductItem(BaseModel):
    """Single product prepared for Hybrid Search (Vector + Lexical) with all 7 HUL Taxonomy filters."""
    product_index: int = Field(description="1-based index of the product facing")
    bbox_2d: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    shelf_row: str = Field(default="middle")
    position_on_shelf: int = Field(default=1)
    category: str = Field(default="Skin Care")
    subcategory: str = Field(default="Face Wash")
    brand: str = Field(description="Visually extracted brand filter for hybrid search")
    variant: str = Field(default="", description="Visually extracted variant")
    packaging_type: str = Field(default="tube")
    pack_type: str = Field(default="Single")
    size: str = Field(default="Medium / Regular")
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
    match_confidence: float = Field(default=0.95, description="Confidence in extracted search attributes (0.0 to 1.0)")
    planogram_compliant: Optional[bool] = Field(default=None, description="True/False if planogram provided, else null")


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
    """Canonical ground truth item for a single product facing (placeholder schema)."""
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
    shelf_row: str = "middle"


class ImageGroundTruth(BaseModel):
    """Canonical ground truth for a shelf image."""
    image_id: str
    total_main_shelf_facings: int
    expected_brands: List[str] = Field(default_factory=list)
    items: List[GroundTruthProductItem] = Field(default_factory=list)


class TokenUsageMetrics(BaseModel):
    """Token counts extracted from Vertex AI Gemini response."""
    input_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


class CostMetrics(BaseModel):
    """100% Separated All-In GCP Cost Breakdown for a single shelf image and per front-facing product."""
    billing_source: str = "gcp_cloud_billing_catalog_live"
    traffic_type: str = "ON_DEMAND"  # "ON_DEMAND", "PROVISIONED_THROUGHPUT", or "HYBRID_SPILLOVER"
    input_cost_usd: float = 0.0
    thinking_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    vertex_ai_payg_tokens_usd: float = 0.0
    vertex_ai_provisioned_throughput_usd: float = 0.0
    vertex_ai_embeddings_and_vision_usd: float = 0.0
    cloud_run_compute_usd: float = 0.0
    gcs_and_observability_usd: float = 0.0
    cost_per_shelf_image_usd: float = 0.0
    cost_per_product_usd: float = 0.0
    product_count: int = 0
    gcp_billing_labels: Dict[str, str] = Field(default_factory=dict)


class AccuracyMetrics(BaseModel):
    """Accuracy metrics evaluated against Ground Truth (or placeholder status when GT is not yet connected)."""
    ground_truth_available: bool = False
    accuracy_status: str = "PLACEHOLDER_AWAITING_GROUND_TRUTH"
    ground_truth_count: Optional[int] = None
    predicted_count: int = 0
    depth_duplicates_filtered: int = 0
    count_accuracy: Optional[float] = None
    detection_precision_iou50: Optional[float] = None
    detection_recall_iou50: Optional[float] = None
    detection_f1_iou50: Optional[float] = None
    mean_iou: Optional[float] = None
    brand_classification_accuracy: Optional[float] = None
    brand_set_recall: Optional[float] = None
    product_classification_accuracy: Optional[float] = None
    sku_matching_accuracy: Optional[float] = None
    planogram_compliance_rate: Optional[float] = None


class RowLevelReportItem(BaseModel):
    """Detailed row-level report entry (one row per detected/classified/matched front-facing product)."""
    run_id: str
    trace_id: str
    span_id: str
    task_type: str
    separation_approach: str = "single_pass_full_shelf"
    model_name: str
    shelf_image_uri: str
    store_id: Optional[str] = None
    start_time: str
    end_time: str
    image_latency_ms: float
    product_index: int
    shelf_row: str = "middle"
    position_on_shelf: int = 1
    bbox_ymin: int = 0
    bbox_xmin: int = 0
    bbox_ymax: int = 0
    bbox_xmax: int = 0
    crop_image_path: Optional[str] = None
    # 7-Dimension HUL Taxonomy Fields
    predicted_category: str = ""
    predicted_subcategory: str = ""
    predicted_brand: str = ""
    is_hul_brand: Optional[bool] = None
    predicted_variant: str = ""
    predicted_packaging: str = ""
    predicted_pack_type: str = "Single"
    predicted_size: str = ""
    rule_derived_size_bucket: str = ""
    predicted_product_name: str = ""
    confidence: float = 0.0
    # Hybrid Search Fields
    lexical_search_keywords: str = ""
    dense_embedding_text: str = ""
    embedding_vector_dim: Optional[int] = None
    matched_sku_id: Optional[str] = None
    planogram_compliant: Optional[bool] = None
    # Ground Truth Placeholder Fields
    gt_status: str = "PLACEHOLDER_AWAITING_GT"
    gt_item_id: Optional[int] = None
    gt_brand: Optional[str] = None
    gt_product_name: Optional[str] = None
    gt_sku_id: Optional[str] = None
    iou_with_gt: Optional[float] = None
    brand_correct: Optional[bool] = None
    product_correct: Optional[bool] = None
    sku_correct: Optional[bool] = None
    # Token & 100% Separated All-In GCP Cost Fields
    input_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    billing_source: str = "gcp_cloud_billing_catalog_live"
    traffic_type: str = "ON_DEMAND"
    vertex_ai_payg_tokens_usd: float = 0.0
    vertex_ai_provisioned_throughput_usd: float = 0.0
    vertex_ai_embeddings_and_vision_usd: float = 0.0
    cloud_run_compute_usd: float = 0.0
    gcs_and_observability_usd: float = 0.0
    cost_per_shelf_image_usd: float = 0.0
    cost_per_product_usd: float = 0.0


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
