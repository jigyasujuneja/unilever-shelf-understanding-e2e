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
