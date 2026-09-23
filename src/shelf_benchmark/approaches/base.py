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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.cropping import crop_detected_facings, load_pil_image
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.evaluation.cost import extract_token_usage
from shelf_benchmark.evaluation.metrics import evaluate_task_accuracy
from shelf_benchmark.geometry import deduplicate_depth_stacked_facings
from shelf_benchmark.models import (
    AccuracyMetrics,
    ImageGroundTruth,
    RowLevelReportItem,
    ShelfAssociationRecord,
    TaskExecutionResult,
    TokenUsageMetrics,
)
from shelf_benchmark.pipeline import (
    NO_RETRY,
    InvocationContext,
    PipelineExecutor,
    RawInvocation,
    RetryPolicy,
)
from shelf_benchmark.size_rules import derive_size_bucket_from_bbox
from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger

#: Keys a plugin may return that map onto a first-class `RowLevelReportItem` column. Anything else
#: a plugin returns (that is not underscore-prefixed) is preserved in `extra_attributes`, which is
#: how the suite supports more than the 8 core attributes without a schema change.
_STANDARD_FACING_KEYS = frozenset({
    "bbox_2d", "brand", "product_name", "category", "subcategory",
    "variant", "packaging_type", "pack_type", "size", "matched_sku_id",
    "crop_image_path", "shelf_row", "position_on_shelf", "product_index",
    "confidence", "is_hul_brand", "extra_attributes",
})

_CLASSIFICATION_ATTR_KEYS = frozenset({
    "brand", "product_name", "category", "subcategory",
    "variant", "packaging_type", "pack_type", "size", "matched_sku_id",
    "extra_attributes",
})


@dataclass
class CommonLayerContext:
    """Unified API surface exposing all shared infrastructure layers to an approach plugin."""

    config: BenchmarkConfig
    storage: StorageManager
    telemetry: OpenTelemetryBenchmarkLogger
    reports_dir: Path
    genai_client: Optional[Any] = None
    prior_detection: Optional[TaskExecutionResult] = None
    _executor_cache: Optional[PipelineExecutor] = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def _executor(self) -> PipelineExecutor:
        """The one finalisation pipeline, shared with the built-in task path."""
        if self._executor_cache is None:
            self._executor_cache = PipelineExecutor(
                config=self.config, telemetry=self.telemetry
            )
        return self._executor_cache

    def get_prior_detected_boxes(
        self, prior_detection: Optional[TaskExecutionResult] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract front-facing bounding-box dicts from a prior Stage-1 detection result (`prior_detection`).

        Allows any 2-stage classification plugin to consume boxes produced by any standalone
        Stage-1 detector without re-running detection inference.
        """
        det = prior_detection or self.prior_detection
        if det is None:
            return None
        raw_boxes = (det.raw_output or {}).get("detected_products")
        if isinstance(raw_boxes, list) and raw_boxes:
            return [dict(b) for b in raw_boxes if isinstance(b, dict)]
        if det.row_level_items:
            return [
                {
                    "product_index": r.product_index,
                    "bbox_2d": [r.bbox_ymin, r.bbox_xmin, r.bbox_ymax, r.bbox_xmax],
                    "shelf_row": r.shelf_row or "middle",
                    "position_on_shelf": r.position_on_shelf,
                    "is_front_facing": True,
                    "preliminary_brand_hint": r.predicted_brand or "",
                    "confidence": r.confidence,
                }
                for r in det.row_level_items
            ]
        return None

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
        return deduplicate_depth_stacked_facings(
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
        crop_paths, montage_bytes = crop_detected_facings(
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
        return derive_size_bucket_from_bbox(
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
        from shelf_benchmark.tasks.facing_utils import check_is_hul_brand

        return check_is_hul_brand(
            brand_name, taxonomy=self.config.taxonomy, model_predicted=model_predicted
        )

    def load_shelf_image(self, record: ShelfAssociationRecord):
        """Load the record's shelf image as a PIL image, honouring its declared local path."""
        return load_pil_image(
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
    ) -> Any:
        """Compute separated 5-bucket GCP cost metrics using the run's configured billing rules."""
        from shelf_benchmark.evaluation.cost import compute_cost_metrics
        from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine

        pricing = self.config.get_pricing(model_name)
        engine = GCPBillingAndCostEngine(
            project_id=self.config.gcp.project_id,
            billing_cfg=self.config.billing,
        )
        gcp_labels = engine.build_gcp_billing_labels(
            run_id=run_id or f"run-{approach_id}-{model_name}",
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

    def rows_from_facing_dicts(
        self,
        raw_outputs: List[Dict[str, Any]],
        *,
        task_type: str = "classification",
    ) -> List[RowLevelReportItem]:
        """Map a plugin's list of per-facing dicts onto report rows.

        Only fields the plugin actually supplied are populated. An absent field stays at the
        schema default (empty string / `None`), which downstream scoring treats as "not
        predicted".

        This used to invent values for anything the plugin omitted -- `"Personal Care"`,
        `"General"`, `"Standard"`, `"Single"`, `"box"`, `"Unknown"`, `f"{brand} Product"`,
        `confidence=0.95`, and a bounding box of `[500, 100, 800, 200]`. Those were then written
        into the report and **graded against ground truth**, so a plugin that predicted nothing
        but a brand still scored on six other attributes, and a plugin that returned no geometry
        was scored against a box in the middle of the shelf.
        """
        all_bboxes = [item.get("bbox_2d") or [0, 0, 0, 0] for item in raw_outputs]
        rows: List[RowLevelReportItem] = []
        for idx, item in enumerate(raw_outputs, start=1):
            bbox = item.get("bbox_2d")
            brand = str(item.get("brand", ""))
            pkg = str(item.get("packaging_type", ""))
            size_hint = str(item.get("size", ""))
            is_hul = item.get("is_hul_brand")
            if is_hul is None and brand:
                is_hul = self.check_is_hul_brand(brand)

            extra_attrs: Dict[str, Any] = dict(item.get("extra_attributes") or {})
            for k, v in item.items():
                if not str(k).startswith("_") and k not in _STANDARD_FACING_KEYS and v is not None:
                    extra_attrs[str(k)] = v

            rows.append(
                RowLevelReportItem(
                    task_type=task_type,
                    product_index=int(item.get("product_index", idx)),
                    predicted_category=str(item.get("category", "")),
                    predicted_subcategory=str(item.get("subcategory", "")),
                    predicted_brand=brand,
                    is_hul_brand=None if is_hul is None else bool(is_hul),
                    predicted_product_name=str(item.get("product_name", "")),
                    predicted_variant=str(item.get("variant", "")),
                    predicted_packaging=pkg,
                    predicted_pack_type=str(item.get("pack_type", "")),
                    predicted_size=size_hint,
                    rule_derived_size_bucket=self.derive_size_bucket_from_bbox(
                        bbox or [0, 0, 0, 0],
                        all_bboxes,
                        packaging_type=pkg,
                        model_size_hint=size_hint,
                    ),
                    matched_sku_id=item.get("matched_sku_id"),
                    crop_image_path=item.get("crop_image_path"),
                    bbox_ymin=int(bbox[0]) if bbox else 0,
                    bbox_xmin=int(bbox[1]) if bbox else 0,
                    bbox_ymax=int(bbox[2]) if bbox else 0,
                    bbox_xmax=int(bbox[3]) if bbox else 0,
                    shelf_row=str(item.get("shelf_row", "")),
                    position_on_shelf=int(item.get("position_on_shelf", idx)),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    extra_attributes=extra_attrs,
                )
            )
        return rows

    @staticmethod
    def infer_effective_task_type(
        raw_outputs: Optional[List[Dict[str, Any]]],
        declared_task_type: str = "classification",
    ) -> str:
        """Return `'detection'` if a plugin explicitly declared `task_type='detection'` or returned only bounding-box geometry with zero classification attributes."""
        if declared_task_type != "classification" or not raw_outputs:
            return declared_task_type
        has_any_bbox = any(bool(it.get("bbox_2d")) for it in raw_outputs if isinstance(it, dict))
        if not has_any_bbox:
            return declared_task_type
        for it in raw_outputs:
            if not isinstance(it, dict):
                continue
            for k, v in it.items():
                if k in _CLASSIFICATION_ATTR_KEYS and v not in (None, "", {}, []):
                    return declared_task_type
                if (
                    not str(k).startswith("_")
                    and k not in _STANDARD_FACING_KEYS
                    and v not in (None, "", {}, [])
                ):
                    return declared_task_type
        return "detection"

    def _build_raw_invocation(
        self,
        *,
        raw_outputs: Optional[List[Dict[str, Any]]] = None,
        rows: Optional[List[RowLevelReportItem]] = None,
        task_type: str = "classification",
        tokens: Optional[TokenUsageMetrics] = None,
        extra_api_cost_usd: Optional[float] = None,
        depth_duplicates_filtered: Optional[int] = None,
        stages_description: Optional[List[str]] = None,
        raw_output_extra: Optional[Dict[str, Any]] = None,
        span_attributes: Optional[Dict[str, Any]] = None,
        prior_detection: Optional[TaskExecutionResult] = None,
    ) -> RawInvocation:
        if (raw_outputs is None) == (rows is None):
            raise ValueError(
                "finalize() takes exactly one of `raw_outputs=` or `rows=`; "
                f"got raw_outputs={'set' if raw_outputs is not None else 'None'}, "
                f"rows={'set' if rows is not None else 'None'}."
            )
        if rows is None:
            assert raw_outputs is not None
            rows = self.rows_from_facing_dicts(raw_outputs, task_type=task_type)
        if depth_duplicates_filtered is None and raw_outputs:
            depth_duplicates_filtered = sum(
                int(it.get("_depth_filtered", 0) or 0) for it in raw_outputs
            )
        if depth_duplicates_filtered is None and prior_detection is not None:
            depth_duplicates_filtered = int(
                prior_detection.accuracy.depth_duplicates_filtered or 0
            )
        if tokens is None and raw_outputs:
            in_tok = sum(int(it.get("_input_tokens", 0) or 0) for it in raw_outputs)
            think_tok = sum(int(it.get("_thinking_tokens", 0) or 0) for it in raw_outputs)
            out_tok = sum(int(it.get("_output_tokens", 0) or 0) for it in raw_outputs)
            cached_tok = sum(int(it.get("_cached_tokens", 0) or 0) for it in raw_outputs)
            if in_tok > 0 or think_tok > 0 or out_tok > 0:
                tokens = TokenUsageMetrics(
                    input_tokens=in_tok,
                    thinking_tokens=think_tok,
                    output_tokens=out_tok,
                    total_tokens=in_tok + think_tok + out_tok,
                    cached_tokens=cached_tok,
                )
        if prior_detection is not None and prior_detection.tokens.total_tokens > 0:
            p_tok = prior_detection.tokens
            if tokens is None:
                tokens = p_tok.model_copy(deep=True)
            else:
                tokens = TokenUsageMetrics(
                    input_tokens=tokens.input_tokens + p_tok.input_tokens,
                    thinking_tokens=tokens.thinking_tokens + p_tok.thinking_tokens,
                    output_tokens=tokens.output_tokens + p_tok.output_tokens,
                    total_tokens=tokens.total_tokens + p_tok.total_tokens,
                    cached_tokens=tokens.cached_tokens + p_tok.cached_tokens,
                )
        if (
            extra_api_cost_usd is None
            and raw_outputs
            and any("_extra_api_cost_usd" in it for it in raw_outputs)
        ):
            extra_api_cost_usd = round(
                sum(float(it.get("_extra_api_cost_usd", 0.0) or 0.0) for it in raw_outputs), 8
            )

        return RawInvocation(
            rows=rows,
            tokens=tokens,
            raw_output=dict(raw_output_extra or {}),
            extra_api_cost_usd=extra_api_cost_usd,
            depth_duplicates_filtered=depth_duplicates_filtered,
            stages_description=stages_description,
            span_attributes=dict(span_attributes or {}),
        )

    def execute_with_pipeline(
        self,
        *,
        approach_id: str,
        model_name: str,
        record: ShelfAssociationRecord,
        invoke_fn: Any,
        gt_record: Optional[ImageGroundTruth] = None,
        task_type: str = "classification",
        run_id: Optional[str] = None,
        stages_description: Optional[List[str]] = None,
        retry_policy: Optional[RetryPolicy] = None,
        prior_detection: Optional[TaskExecutionResult] = None,
    ) -> TaskExecutionResult:
        """Execute a plugin callable through `PipelineExecutor.run()` for retry, CPU-time, and error parity with built-in tasks."""
        prev_prior = self.prior_detection
        self.prior_detection = prior_detection
        effective_policy = retry_policy or (
            NO_RETRY if self.config.offline.enabled else RetryPolicy(max_attempts=3)
        )
        resolved_task_type_holder = [task_type]

        def _wrapped(live_model: str) -> RawInvocation:
            raw_outputs = invoke_fn(live_model)
            effective_task = self.infer_effective_task_type(raw_outputs, task_type)
            resolved_task_type_holder[0] = effective_task
            return self._build_raw_invocation(
                raw_outputs=raw_outputs,
                task_type=effective_task,
                stages_description=stages_description,
                prior_detection=prior_detection,
            )

        try:
            res = self._executor.run(
                _wrapped,
                ctx=InvocationContext(
                    task_type=resolved_task_type_holder[0],
                    approach_id=approach_id,
                    model_name=model_name,
                    shelf_image_uri=record.shelf_image_uri,
                    run_id=run_id or f"run-{approach_id}-{model_name}",
                    store_id=record.store_id,
                    ground_truth=gt_record,
                ),
                retry_policy=effective_policy,
            )
            if resolved_task_type_holder[0] != res.task_type:
                from shelf_benchmark.evaluation.metrics import evaluate_task_accuracy

                res.task_type = resolved_task_type_holder[0]
                for r in res.row_level_items:
                    r.task_type = resolved_task_type_holder[0]
                res.accuracy = evaluate_task_accuracy(
                    task_type=resolved_task_type_holder[0],
                    rows=res.row_level_items,
                    ground_truth=gt_record,
                    config=self.config.evaluation,
                )
                res.accuracy.depth_duplicates_filtered = int(
                    res.raw_output.get("depth_duplicates_filtered", 0) or 0
                )
                if "execution_trace" in res.raw_output and isinstance(
                    res.raw_output["execution_trace"], dict
                ):
                    res.raw_output["execution_trace"]["task_type"] = resolved_task_type_holder[0]
            return res
        finally:
            self.prior_detection = prev_prior

    def finalize(
        self,
        *,
        approach_id: str,
        model_name: str,
        record: ShelfAssociationRecord,
        raw_outputs: Optional[List[Dict[str, Any]]] = None,
        rows: Optional[List[RowLevelReportItem]] = None,
        start_dt: Any = None,
        end_dt: Any,
        gt_record: Optional[ImageGroundTruth] = None,
        task_type: str = "classification",
        run_id: Optional[str] = None,
        tokens: Optional[TokenUsageMetrics] = None,
        extra_api_cost_usd: Optional[float] = None,
        depth_duplicates_filtered: Optional[int] = None,
        stages_description: Optional[List[str]] = None,
        raw_output_extra: Optional[Dict[str, Any]] = None,
        span_attributes: Optional[Dict[str, Any]] = None,
        cpu_active_ms: Optional[float] = None,
    ) -> TaskExecutionResult:
        """Turn raw facing prediction dicts into a fully scored, OTel-logged `TaskExecutionResult`."""
        invocation = self._build_raw_invocation(
            raw_outputs=raw_outputs,
            rows=rows,
            task_type=task_type,
            tokens=tokens,
            extra_api_cost_usd=extra_api_cost_usd,
            depth_duplicates_filtered=depth_duplicates_filtered,
            stages_description=stages_description,
            raw_output_extra=raw_output_extra,
            span_attributes=span_attributes,
            prior_detection=self.prior_detection,
        )
        return self._executor.finalize(
            invocation,
            ctx=InvocationContext(
                task_type=task_type,
                approach_id=approach_id,
                model_name=model_name,
                shelf_image_uri=record.shelf_image_uri,
                run_id=run_id or f"run-{approach_id}-{model_name}",
                store_id=record.store_id,
                ground_truth=gt_record,
            ),
            start_dt=start_dt,
            end_dt=end_dt,
            cpu_active_ms=cpu_active_ms,
        )


class BaseShelfApproachPlugin(ABC):
    """Abstract base class for all modular Shelf Understanding approaches."""

    #: Set to True for illustrative approaches that must never appear in a decision-making
    #: comparison (e.g. pipelines whose labels come from a hardcoded prototype list).
    is_demo_only: bool = False

    #: Primary benchmark task type ('classification', 'detection', or 'matching').
    task_type: str = "classification"

    #: Optional custom retry policy (defaults to RetryPolicy(max_attempts=3) in live mode and NO_RETRY offline).
    retry_policy: Optional[RetryPolicy] = None

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
    and `ctx.execute_with_pipeline(...)` automatically measures process CPU time, applies retry
    policy, computes size buckets, HUL brand attribution, 5-bucket GCP cost, ground-truth accuracy,
    OpenTelemetry spans, and UI execution traces.
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
        return ctx.execute_with_pipeline(
            approach_id=self.approach_id,
            model_name=model_name,
            record=record,
            invoke_fn=lambda live_model: self.detect_and_classify(ctx, live_model, record),
            gt_record=gt_record,
            task_type=self.task_type,
            stages_description=self.stages_description,
            retry_policy=self.retry_policy,
            prior_detection=prior_detection,
        )
