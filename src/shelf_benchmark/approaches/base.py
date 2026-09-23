"""Base plugin interface and the shared "common layer" context for Shelf Understanding approaches.

Any engineer can create `src/shelf_benchmark/approaches/<my_approach>/plugin.py` with a
`BaseShelfApproachPlugin` subclass (or a `get_plugins()` factory). It is discovered automatically
and receives a `CommonLayerContext` giving access to storage, facing geometry helpers, telemetry,
cost and accuracy evaluation - without duplicating infrastructure or editing core files.

Every helper on the context routes through the *run's configuration*, so a plugin and a built-in
task always apply the same thresholds, taxonomy and scoring policy to the same input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shelf_benchmark.config import BenchmarkConfig, ModelPricing
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.evaluation.cost import compute_cost_metrics, extract_token_usage
from shelf_benchmark.evaluation.metrics import evaluate_task_accuracy
from shelf_benchmark.models import (
    AccuracyMetrics,
    CostMetrics,
    ImageGroundTruth,
    RowLevelReportItem,
    ShelfAssociationRecord,
    TaskExecutionResult,
    TokenUsageMetrics,
    build_execution_trace_metadata,
)
from shelf_benchmark.tasks import facing_utils
from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger


@dataclass
class CommonLayerContext:
    """Unified API surface exposing all shared infrastructure layers to an approach plugin."""

    config: BenchmarkConfig
    storage: StorageManager
    telemetry: OpenTelemetryBenchmarkLogger
    reports_dir: Path
    genai_client: Optional[Any] = None

    def get_client(self, location: Optional[str] = None) -> Any:
        """Return the injected GenAI client/adapter (for offline fakes, GEAP, Gemma, Tuned Endpoints) or a Vertex AI client."""
        if self.genai_client is not None and location is None:
            return self.genai_client
        from shelf_benchmark.auth import create_genai_client

        return create_genai_client(
            project_id=self.config.gcp.project_id,
            location=location or self.config.gcp.location,
        )

    # --- Shared Geometry & Facing Utilities (`facing_utils.py`) ---
    def deduplicate_depth_stacked_facings(
        self,
        items: List[Dict[str, Any]],
        x_overlap_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Suppress back-row units stacked behind the front facing in the same shelf column.

        Defaults to the configured `depth_deduplication.x_overlap_threshold` so plugins and
        built-in tasks apply identical geometry.
        """
        threshold = (
            x_overlap_threshold
            if x_overlap_threshold is not None
            else self.config.depth_deduplication.x_overlap_threshold
        )
        return facing_utils.deduplicate_depth_stacked_facings(
            items, x_overlap_threshold=threshold
        )

    def crop_facing_images(
        self,
        shelf_image_uri: str,
        facings: List[Dict[str, Any]],
        model_tag: str,
        local_fallback: Optional[str] = None,
    ) -> Tuple[List[Tuple[int, bytes, str]], Optional[bytes], Optional[str]]:
        """Physically crop each detected facing into PNGs plus a numbered montage.

        Returns `(crops, montage_bytes, montage_path)`, where `crops` is a list of
        `(facing_index, png_bytes, file_path)` tuples ordered left-to-right.
        """
        crop_paths, montage_bytes = facing_utils.crop_detected_facings(
            storage=self.storage,
            shelf_image_uri=shelf_image_uri,
            detected_items=facings,
            output_crop_dir=self.reports_dir / "crops",
            model_tag=model_tag,
            local_fallback=local_fallback,
        )
        crops: List[Tuple[int, bytes, str]] = []
        for idx, path_str in enumerate(crop_paths, start=1):
            path = Path(path_str)
            crops.append((idx, path.read_bytes() if path.exists() else b"", path_str))
        montage_path = str(
            self.reports_dir / "crops" / model_tag.replace("/", "_") / "montage_all_facings.png"
        )
        return crops, montage_bytes, montage_path

    def derive_size_bucket_from_bbox(
        self,
        bbox_2d: List[int],
        all_bboxes_on_shelf: List[List[int]],
        packaging_type: str = "tube",
        model_size_hint: str = "",
    ) -> str:
        """Compute the size bucket from packaging, OCR size hint and relative box geometry.

        Uses the run's configured taxonomy, so a plugin and a built-in task cannot disagree about
        the bucket for the same product.
        """
        return facing_utils.derive_size_bucket_from_bbox(
            bbox_2d=bbox_2d,
            all_bboxes_on_row=all_bboxes_on_shelf,
            packaging_type=packaging_type,
            model_size_hint=model_size_hint,
            taxonomy=self.config.taxonomy,
        )

    def check_is_hul_brand(
        self, brand_name: str, model_predicted: Optional[bool] = None
    ) -> bool:
        """Check whether a predicted brand belongs to the configured HUL portfolio."""
        return facing_utils.check_is_hul_brand(
            brand_name, taxonomy=self.config.taxonomy, model_predicted=model_predicted
        )

    def load_shelf_image(self, record: ShelfAssociationRecord):
        """Load the record's shelf image as a PIL image, honouring its declared local path."""
        return facing_utils.load_pil_image(
            self.storage,
            record.shelf_image_uri,
            local_fallback=record.local_shelf_image_path,
        )

    # --- Shared Cost & Accuracy Evaluation APIs (`cost.py` & `metrics.py`) ---
    def extract_tokens(self, response: Any) -> TokenUsageMetrics:
        """Extract input/thinking/output/cached token counts from a model response."""
        return extract_token_usage(response)

    def compute_cost(
        self,
        tokens: TokenUsageMetrics,
        model_name: str,
        product_count: int,
        extra_api_cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        run_id: Optional[str] = None,
        approach_id: str = "custom_approach",
        task_type: str = "classification",
    ) -> CostMetrics:
        """Compute the separated GCP cost breakdown for one approach execution."""
        from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine

        pricing: ModelPricing = self.config.get_pricing(model_name)
        engine = GCPBillingAndCostEngine(
            project_id=self.config.gcp.project_id,
            billing_cfg=self.config.billing,
        )
        gcp_labels = engine.build_gcp_billing_labels(
            run_id=run_id or "plugin-run",
            approach_id=approach_id,
            task_type=task_type,
            model_name=model_name,
        )
        return compute_cost_metrics(
            tokens=tokens,
            pricing=pricing,
            product_count=product_count,
            latency_ms=latency_ms,
            extra_embedding_or_vision_cost_usd=extra_api_cost_usd,
            billing_cfg=self.config.billing,
            project_id=self.config.gcp.project_id,
            model_name=model_name,
            gcp_labels=gcp_labels,
        )

    def evaluate_accuracy(
        self,
        task_type: str,
        rows: List[RowLevelReportItem],
        gt_record: Optional[ImageGroundTruth],
        depth_duplicates_filtered: int = 0,
    ) -> AccuracyMetrics:
        """Score rows against ground truth using the run's configured evaluation policy."""
        accuracy = evaluate_task_accuracy(
            task_type, rows, gt_record, config=self.config.evaluation
        )
        accuracy.depth_duplicates_filtered = depth_duplicates_filtered
        return accuracy

    def finalize(
        self,
        *,
        approach_id: str,
        model_name: str,
        record: ShelfAssociationRecord,
        raw_outputs: List[Dict[str, Any]],
        start_dt: Any,
        end_dt: Any,
        gt_record: Optional[ImageGroundTruth] = None,
        task_type: str = "classification",
        run_id: Optional[str] = None,
        tokens: Optional[TokenUsageMetrics] = None,
        extra_api_cost_usd: float = 0.0,
        depth_duplicates_filtered: Optional[int] = None,
        stages_description: Optional[List[str]] = None,
        raw_output_extra: Optional[Dict[str, Any]] = None,
    ) -> TaskExecutionResult:
        """Turn raw facing prediction dicts into a fully scored, OTel-logged `TaskExecutionResult`.

        Eliminates ~180 lines of boilerplate per plugin by applying the shared taxonomy, size rules,
        5-bucket GCP cost model, ground-truth evaluation, OpenTelemetry span logging, and
        structured execution trace generation in one call.
        """
        effective_run_id = run_id or f"run-{approach_id}-{model_name}"
        latency_ms = max(0.1, round((end_dt - start_dt).total_seconds() * 1000.0, 3))

        all_bboxes = [item.get("bbox_2d", [0, 0, 0, 0]) for item in raw_outputs]
        if tokens is None:
            total_in = sum(int(item.get("_input_tokens", 180)) for item in raw_outputs) or 350
            total_think = sum(int(item.get("_thinking_tokens", 0)) for item in raw_outputs)
            total_out = sum(int(item.get("_output_tokens", 90)) for item in raw_outputs) or 160
            tokens = TokenUsageMetrics(
                input_tokens=total_in,
                thinking_tokens=total_think,
                output_tokens=total_out,
                total_tokens=total_in + total_think + total_out,
            )

        cost = self.compute_cost(
            tokens=tokens,
            model_name=model_name,
            product_count=max(len(raw_outputs), 1),
            extra_api_cost_usd=extra_api_cost_usd,
            latency_ms=latency_ms,
            run_id=effective_run_id,
            approach_id=approach_id,
            task_type=task_type,
        )
        per_facing_cost = round(
            cost.cost_per_shelf_image_usd / max(len(raw_outputs), 1), 8
        )
        depth_filtered = (
            depth_duplicates_filtered
            if depth_duplicates_filtered is not None
            else (int(raw_outputs[0].get("_depth_filtered", 0)) if raw_outputs else 0)
        )

        row_items: List[RowLevelReportItem] = []
        for idx, item in enumerate(raw_outputs, start=1):
            bbox = item.get("bbox_2d", [500, 100, 800, 200])
            brand = str(item.get("brand", "Unknown"))
            pkg = str(item.get("packaging_type", "box"))
            size_hint = str(item.get("size", ""))
            is_hul = item.get(
                "is_hul_brand",
                self.check_is_hul_brand(brand),
            )
            rule_size = self.derive_size_bucket_from_bbox(
                bbox,
                all_bboxes,
                packaging_type=pkg,
                model_size_hint=size_hint,
            )
            std_keys = {
                "bbox_2d", "brand", "product_name", "category", "subcategory",
                "variant", "packaging_type", "pack_type", "size", "matched_sku_id",
                "crop_image_path", "shelf_row", "position_on_shelf", "confidence",
                "is_hul_brand", "extra_attributes",
            }
            extra_attrs: Dict[str, Any] = dict(item.get("extra_attributes") or {})
            for k, v in item.items():
                if not str(k).startswith("_") and k not in std_keys and v is not None:
                    extra_attrs[str(k)] = v

            row_items.append(
                RowLevelReportItem(
                    run_id=effective_run_id,
                    start_time=self.telemetry.format_iso(start_dt),
                    end_time=self.telemetry.format_iso(end_dt),
                    image_latency_ms=latency_ms,
                    task_type=task_type,
                    separation_approach=approach_id,
                    model_name=model_name,
                    shelf_image_uri=record.shelf_image_uri,
                    store_id=record.store_id,
                    product_index=idx,
                    predicted_category=str(item.get("category", "Personal Care")),
                    predicted_subcategory=str(item.get("subcategory", "General")),
                    predicted_brand=brand,
                    is_hul_brand=bool(is_hul),
                    predicted_product_name=str(item.get("product_name", f"{brand} Product")),
                    predicted_variant=str(item.get("variant", "Standard")),
                    predicted_packaging=pkg,
                    predicted_pack_type=str(item.get("pack_type", "Single")),
                    predicted_size=size_hint,
                    rule_derived_size_bucket=rule_size,
                    matched_sku_id=item.get("matched_sku_id"),
                    crop_image_path=item.get("crop_image_path"),
                    bbox_ymin=int(bbox[0]),
                    bbox_xmin=int(bbox[1]),
                    bbox_ymax=int(bbox[2]),
                    bbox_xmax=int(bbox[3]),
                    shelf_row=str(item.get("shelf_row", "middle")),
                    position_on_shelf=int(item.get("position_on_shelf", idx)),
                    is_front_facing=True,
                    confidence=float(item.get("confidence", 0.95)),
                    extra_attributes=extra_attrs,
                    input_tokens=tokens.input_tokens // max(len(raw_outputs), 1),
                    thinking_tokens=tokens.thinking_tokens // max(len(raw_outputs), 1),
                    output_tokens=tokens.output_tokens // max(len(raw_outputs), 1),
                    total_tokens=tokens.total_tokens // max(len(raw_outputs), 1),
                    billing_source=cost.billing_source,
                    traffic_type=cost.traffic_type,
                    vertex_ai_payg_tokens_usd=cost.vertex_ai_payg_tokens_usd,
                    vertex_ai_provisioned_throughput_usd=cost.vertex_ai_provisioned_throughput_usd,
                    vertex_ai_embeddings_and_vision_usd=cost.vertex_ai_embeddings_and_vision_usd,
                    cloud_run_compute_usd=cost.cloud_run_compute_usd,
                    gcs_and_observability_usd=cost.gcs_and_observability_usd,
                    cost_per_product_usd=per_facing_cost,
                    cost_per_shelf_image_usd=cost.cost_per_shelf_image_usd,
                )
            )

        accuracy = self.evaluate_accuracy(
            task_type=task_type,
            rows=row_items,
            gt_record=gt_record,
            depth_duplicates_filtered=depth_filtered,
        )
        trace_id, span_id, _ = self.telemetry.log_task_execution(
            run_id=effective_run_id,
            task_type=task_type,
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            start_dt=start_dt,
            end_dt=end_dt,
            tokens=tokens,
            cost=cost,
            accuracy=accuracy,
            extra_attributes={"shelf_benchmark.separation_approach": approach_id},
        )
        for r in row_items:
            r.trace_id = trace_id
            r.span_id = span_id

        exec_trace = build_execution_trace_metadata(
            run_id=effective_run_id,
            trace_id=trace_id,
            span_id=span_id,
            task_type=task_type,
            separation_approach=approach_id,
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            latency_ms=latency_ms,
            tokens=tokens,
            cost=cost,
            accuracy=accuracy,
            facings_count=len(row_items),
            otel_log_path=str(self.config.telemetry.otel_log_path),
            gcp_project_id=self.config.gcp.project_id,
            gcp_log_name=self.config.telemetry.gcp_log_name,
            taxonomy_source=self.config.taxonomy.taxonomy_file,
            ground_truth_provider=self.config.ground_truth.provider_type,
            reference_catalog_uri=self.config.embeddings.reference_catalog.source_uri,
            custom_stages=stages_description,
        )
        raw_payload: Dict[str, Any] = {
            "total_classified_products": len(row_items),
            "depth_duplicates_filtered": depth_filtered,
            "execution_trace": exec_trace,
        }
        if raw_output_extra:
            raw_payload.update(raw_output_extra)

        return TaskExecutionResult(
            run_id=effective_run_id,
            trace_id=trace_id,
            span_id=span_id,
            task_type=task_type,
            separation_approach=approach_id,
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            start_time=self.telemetry.format_iso(start_dt),
            end_time=self.telemetry.format_iso(end_dt),
            latency_ms=latency_ms,
            tokens=tokens,
            cost=cost,
            accuracy=accuracy,
            raw_output=raw_payload,
            row_level_items=row_items,
        )


class BaseShelfApproachPlugin(ABC):
    """Abstract base class for all modular Shelf Understanding approaches."""

    #: Set to True for illustrative approaches that must never appear in a decision-making
    #: comparison (e.g. pipelines whose labels come from a hardcoded prototype list).
    is_demo_only: bool = False

    @property
    @abstractmethod
    def approach_id(self) -> str:
        """Unique machine identifier (e.g. 'two_stage_physical_crop_per_facing')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-friendly title for executive reports and the UI."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Pipeline paradigm ('vlm_multimodal' | 'two_stage_vlm' | 'classic_cv_metric_learning')."""

    @property
    @abstractmethod
    def stages_description(self) -> List[str]:
        """Ordered description of the stages this approach executes."""

    @abstractmethod
    def execute(
        self,
        ctx: CommonLayerContext,
        model_name: str,
        record: ShelfAssociationRecord,
        gt_record: Optional[ImageGroundTruth] = None,
        prior_detection: Optional[TaskExecutionResult] = None,
    ) -> TaskExecutionResult:
        """Execute this approach on one shelf image and return standardized telemetry and rows."""


class SimpleShelfApproachPlugin(BaseShelfApproachPlugin):
    """Streamlined ~15-line base class for approach plugins.

    Subclasses only implement `detect_and_classify(ctx, model_name, record) -> List[Dict[str, Any]]`
    and `ctx.finalize(...)` automatically computes size buckets, HUL brand attribution, 5-bucket
    GCP cost, ground-truth accuracy, OpenTelemetry spans, and UI execution traces.
    """

    @abstractmethod
    def detect_and_classify(
        self,
        ctx: CommonLayerContext,
        model_name: str,
        record: ShelfAssociationRecord,
    ) -> List[Dict[str, Any]]:
        """Return one dictionary per front-facing product (`bbox_2d`, `brand`, `product_name`, etc.)."""

    def execute(
        self,
        ctx: CommonLayerContext,
        model_name: str,
        record: ShelfAssociationRecord,
        gt_record: Optional[ImageGroundTruth] = None,
        prior_detection: Optional[TaskExecutionResult] = None,
    ) -> TaskExecutionResult:
        start_dt = ctx.telemetry.now_utc()
        raw_outputs = self.detect_and_classify(ctx, model_name, record)
        end_dt = ctx.telemetry.now_utc()
        return ctx.finalize(
            approach_id=self.approach_id,
            model_name=model_name,
            record=record,
            raw_outputs=raw_outputs,
            start_dt=start_dt,
            end_dt=end_dt,
            gt_record=gt_record,
            stages_description=self.stages_description,
        )
