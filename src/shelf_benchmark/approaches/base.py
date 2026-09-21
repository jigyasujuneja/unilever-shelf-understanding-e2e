"""Base Plugin Interface and Shared Common Layer API Context for Shelf Understanding Approaches.

Any engineer can create a new subdirectory under `src/shelf_benchmark/approaches/<my_approach>/`
with a `plugin.py` subclassing `BaseShelfApproachPlugin`. It automatically receives access to all
common layers (`StorageManager`, `facing_utils`, `OpenTelemetryBenchmarkLogger`, `cost`, `metrics`,
and `ground_truth`) without duplicating infrastructure code or modifying existing approaches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    """Unified API surface exposing all shared infrastructure layers to any Approach Plugin."""

    config: BenchmarkConfig
    storage: StorageManager
    telemetry: OpenTelemetryBenchmarkLogger
    reports_dir: Path

    # --- Shared Geometry & Facing Utilities (`facing_utils.py`) ---
    def deduplicate_depth_stacked_facings(
        self,
        items: List[Dict[str, Any]],
        x_overlap_threshold: float = 0.55,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Suppresses back-row units stacked behind the front facing in the same shelf column."""
        return facing_utils.deduplicate_depth_stacked_facings(
            items, x_overlap_threshold=x_overlap_threshold
        )

    def crop_facing_images(
        self,
        shelf_image_uri: str,
        facings: List[Dict[str, Any]],
        model_tag: str,
    ) -> Tuple[List[Tuple[int, bytes, str]], Optional[bytes], Optional[str]]:
        """Physically crops each detected `[ymin, xmin, ymax, xmax]` facing into PNGs + numbered montage."""
        crop_paths, montage_bytes = facing_utils.crop_detected_facings(
            storage=self.storage,
            shelf_image_uri=shelf_image_uri,
            detected_items=facings,
            output_crop_dir=self.reports_dir / "crops",
            model_tag=model_tag,
        )
        tuples: List[Tuple[int, bytes, str]] = []
        for idx, p_str in enumerate(crop_paths, start=1):
            p = Path(p_str)
            b = p.read_bytes() if p.exists() else b""
            tuples.append((idx, b, p_str))
        montage_path = str(
            self.reports_dir / "crops" / model_tag.replace("/", "_") / "montage_all_facings.png"
        )
        return tuples, montage_bytes, montage_path

    def derive_size_bucket_from_bbox(
        self,
        bbox_2d: List[int],
        all_bboxes_on_shelf: List[List[int]],
        packaging_type: str = "tube",
    ) -> str:
        """Deterministically computes the HUL size bucket from relative bounding box dimensions."""
        return facing_utils.derive_size_bucket_from_bbox(
            bbox_2d=bbox_2d,
            all_bboxes_on_row=all_bboxes_on_shelf,
            packaging_type=packaging_type,
        )

    def check_is_hul_brand(self, brand_name: str) -> bool:
        """Checks whether a predicted brand belongs to the configured HUL portfolio brands."""
        return facing_utils.check_is_hul_brand(brand_name, taxonomy=self.config.taxonomy)

    # --- Shared Cost & Accuracy Evaluation APIs (`cost.py` & `metrics.py`) ---
    def extract_tokens(self, response: Any) -> TokenUsageMetrics:
        return extract_token_usage(response)

    def compute_cost(
        self,
        tokens: TokenUsageMetrics,
        model_name: str,
        product_count: int,
        extra_api_cost_usd: float = 0.0,
    ) -> CostMetrics:
        pricing: ModelPricing = self.config.get_pricing(model_name)
        base_cost = compute_cost_metrics(tokens, pricing, product_count)
        if extra_api_cost_usd > 0.0:
            total_shelf = round(base_cost.cost_per_shelf_image_usd + extra_api_cost_usd, 8)
            eff_n = max(1, product_count)
            return CostMetrics(
                input_cost_usd=round(base_cost.input_cost_usd + extra_api_cost_usd, 8),
                thinking_cost_usd=base_cost.thinking_cost_usd,
                output_cost_usd=base_cost.output_cost_usd,
                cost_per_shelf_image_usd=total_shelf,
                cost_per_product_usd=round(total_shelf / eff_n, 8),
                product_count=product_count,
            )
        return base_cost

    def evaluate_accuracy(
        self,
        task_type: str,
        rows: List[RowLevelReportItem],
        gt_record: Optional[ImageGroundTruth],
        depth_duplicates_filtered: int = 0,
    ) -> AccuracyMetrics:
        acc = evaluate_task_accuracy(task_type, rows, gt_record)
        acc.depth_duplicates_filtered = depth_duplicates_filtered
        return acc


class BaseShelfApproachPlugin(ABC):
    """Abstract Base Class for all modular Shelf Understanding Approaches."""

    @property
    @abstractmethod
    def approach_id(self) -> str:
        """Unique machine identifier (e.g., 'class_agnostic_visual_embedding')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-friendly title for executive reports and UI."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Pipeline paradigm ('vlm_multimodal' | 'classic_cv_metric_learning' | 'hybrid')."""

    @property
    @abstractmethod
    def stages_description(self) -> List[str]:
        """Ordered description of the stages executed by this approach."""

    @abstractmethod
    def execute(
        self,
        ctx: CommonLayerContext,
        model_name: str,
        record: ShelfAssociationRecord,
        gt_record: Optional[ImageGroundTruth] = None,
        prior_detection: Optional[TaskExecutionResult] = None,
    ) -> TaskExecutionResult:
        """Execute the approach on a single shelf image record and return standardized telemetry & row items."""
