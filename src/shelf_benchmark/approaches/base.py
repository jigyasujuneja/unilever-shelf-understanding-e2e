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
    InvocationContext,
    PipelineExecutor,
    RawInvocation,
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


@dataclass
class CommonLayerContext:
    """Unified API surface exposing all shared infrastructure layers to an approach plugin."""

    config: BenchmarkConfig
    storage: StorageManager
    telemetry: OpenTelemetryBenchmarkLogger
    reports_dir: Path
    genai_client: Optional[Any] = None
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
        """Turn raw facing prediction dicts into a fully scored, OTel-logged `TaskExecutionResult`.

        This is now a thin adapter over `shelf_benchmark.pipeline.PipelineExecutor`, which the
        built-in tasks also use, so a plugin and a built-in task fed identical predictions produce
        identical cost, token and accuracy figures. They previously did not: see the `pipeline`
        module docstring.

        `extra_api_cost_usd=None` means "derive embeddings/vision cost from the run's billing
        config", which is what the built-in tasks have always done. It used to default to `0.0`
        here, so a plugin's crops were free and a task's were not.

        Pass **either** `raw_outputs=` (a list of per-facing dicts, mapped by
        `rows_from_facing_dicts`) **or** `rows=` (rows you built yourself, for approaches that
        populate columns the dict mapping does not cover, such as the embedding fields). Passing
        neither or both is a programming error and raises.
        """
        if (raw_outputs is None) == (rows is None):
            raise ValueError(
                "finalize() takes exactly one of `raw_outputs=` or `rows=`; "
                f"got raw_outputs={'set' if raw_outputs is not None else 'None'}, "
                f"rows={'set' if rows is not None else 'None'}."
            )
        if rows is None:
            assert raw_outputs is not None  # narrowed by the check above
            rows = self.rows_from_facing_dicts(raw_outputs, task_type=task_type)
        if depth_duplicates_filtered is None and raw_outputs:
            depth_duplicates_filtered = int(raw_outputs[0].get("_depth_filtered", 0) or 0)

        invocation = RawInvocation(
            rows=rows,
            tokens=tokens,
            raw_output=dict(raw_output_extra or {}),
            extra_api_cost_usd=extra_api_cost_usd,
            depth_duplicates_filtered=depth_duplicates_filtered,
            stages_description=stages_description,
            span_attributes=dict(span_attributes or {}),
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
