"""Abstract Base Task providing standardized execution, token extraction, cost calculation, accuracy scoring, and OpenTelemetry logging."""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from google import genai

from shelf_benchmark.auth import create_genai_client
from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.evaluation.cost import compute_cost_metrics, extract_token_usage
from shelf_benchmark.evaluation.metrics import evaluate_task_accuracy
from shelf_benchmark.models import (
    ImageGroundTruth,
    RowLevelReportItem,
    TaskExecutionResult,
    TokenUsageMetrics,
)
from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger


class BaseBenchmarkTask(ABC):
    """Base class for separated Shelf Understanding tasks."""

    task_type: str = "base"

    def __init__(
        self,
        config: BenchmarkConfig,
        storage_manager: StorageManager,
        telemetry_logger: OpenTelemetryBenchmarkLogger,
        genai_client: Optional[genai.Client] = None,
    ):
        self.config = config
        self.storage = storage_manager
        self.telemetry = telemetry_logger
        self._genai_client = genai_client

    def get_client(self, location: Optional[str] = None) -> genai.Client:
        if self._genai_client is not None and location is None:
            return self._genai_client
        return create_genai_client(
            project_id=self.config.gcp.project_id,
            location=location or self.config.gcp.location,
        )

    def _get_billing_engine(self):
        from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine

        return GCPBillingAndCostEngine(
            project_id=self.config.gcp.project_id,
            billing_cfg=self.config.billing,
        )

    @abstractmethod
    def invoke_model(
        self,
        model_name: str,
        shelf_image_uri: str,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], TokenUsageMetrics, List[RowLevelReportItem]]:
        """Execute the task logic against the model and return (parsed_output_dict, token_usage, row_items)."""

    def execute(
        self,
        model_name: str,
        shelf_image_uri: str,
        run_id: Optional[str] = None,
        store_id: Optional[str] = None,
        ground_truth: Optional[ImageGroundTruth] = None,
        **kwargs: Any,
    ) -> TaskExecutionResult:
        """Standardized execution wrapper recording start_time, end_time, OTel telemetry, cost, accuracy, and row-level records."""
        active_run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        separation_approach = str(kwargs.get("separation_approach") or "single_pass_full_shelf")

        status = "SUCCESS"
        error_message: Optional[str] = None
        raw_output: Dict[str, Any] = {}
        tokens = TokenUsageMetrics()
        rows: List[RowLevelReportItem] = []

        start_dt = self.telemetry.now_utc()
        start_iso = self.telemetry.format_iso(start_dt)

        max_attempts = int(kwargs.pop("max_attempts", 3))
        for attempt in range(1, max_attempts + 1):
            start_dt = self.telemetry.now_utc()
            start_iso = self.telemetry.format_iso(start_dt)
            try:
                raw_output, tokens, rows = self.invoke_model(
                    model_name=model_name,
                    shelf_image_uri=shelf_image_uri,
                    ground_truth=ground_truth,
                    **kwargs,
                )
                status = "SUCCESS"
                error_message = None
                break
            except Exception as exc:
                status = "ERROR"
                error_message = str(exc)
                if attempt < max_attempts:
                    time.sleep(2.0 * attempt)

        end_dt = self.telemetry.now_utc()
        end_iso = self.telemetry.format_iso(end_dt)
        latency_ms = round((end_dt - start_dt).total_seconds() * 1000.0, 3)

        if rows and rows[0].separation_approach:
            separation_approach = rows[0].separation_approach

        pricing = self.config.get_pricing(model_name)
        billing_engine = self._get_billing_engine()
        gcp_labels = billing_engine.build_gcp_billing_labels(
            run_id=active_run_id,
            approach_id=separation_approach,
            task_type=self.task_type,
            model_name=model_name,
        )
        cost = compute_cost_metrics(
            tokens=tokens,
            pricing=pricing,
            product_count=len(rows),
            latency_ms=latency_ms,
            billing_cfg=self.config.billing,
            project_id=self.config.gcp.project_id,
            model_name=model_name,
            gcp_labels=gcp_labels,
        )

        for r in rows:
            r.run_id = active_run_id
            r.task_type = self.task_type
            r.separation_approach = separation_approach
            r.model_name = model_name
            r.shelf_image_uri = shelf_image_uri
            r.store_id = store_id
            r.start_time = start_iso
            r.end_time = end_iso
            r.image_latency_ms = latency_ms
            r.input_tokens = tokens.input_tokens
            r.thinking_tokens = tokens.thinking_tokens
            r.output_tokens = tokens.output_tokens
            r.total_tokens = tokens.total_tokens
            r.cost_per_shelf_image_usd = cost.cost_per_shelf_image_usd
            r.cost_per_product_usd = cost.cost_per_product_usd
            r.vertex_ai_payg_tokens_usd = cost.vertex_ai_payg_tokens_usd
            r.vertex_ai_provisioned_throughput_usd = cost.vertex_ai_provisioned_throughput_usd
            r.vertex_ai_embeddings_and_vision_usd = cost.vertex_ai_embeddings_and_vision_usd
            r.cloud_run_compute_usd = cost.cloud_run_compute_usd
            r.gcs_and_observability_usd = cost.gcs_and_observability_usd
            r.traffic_type = cost.traffic_type
            r.billing_source = cost.billing_source

        accuracy = evaluate_task_accuracy(
            task_type=self.task_type,
            rows=rows,
            ground_truth=ground_truth,
        )
        accuracy.depth_duplicates_filtered = int(raw_output.get("depth_duplicates_filtered", 0) or 0)

        trace_id, span_id, _ = self.telemetry.log_task_execution(
            run_id=active_run_id,
            task_type=self.task_type,
            model_name=model_name,
            shelf_image_uri=shelf_image_uri,
            start_dt=start_dt,
            end_dt=end_dt,
            tokens=tokens,
            cost=cost,
            accuracy=accuracy,
            status=status,
            error_message=error_message,
            extra_attributes={
                "shelf_benchmark.separation_approach": separation_approach,
                "shelf_benchmark.depth_duplicates_filtered": accuracy.depth_duplicates_filtered,
            },
        )

        for r in rows:
            r.trace_id = trace_id
            r.span_id = span_id

        return TaskExecutionResult(
            run_id=active_run_id,
            trace_id=trace_id,
            span_id=span_id,
            task_type=self.task_type,
            separation_approach=separation_approach,
            model_name=model_name,
            shelf_image_uri=shelf_image_uri,
            start_time=start_iso,
            end_time=end_iso,
            latency_ms=latency_ms,
            tokens=tokens,
            cost=cost,
            accuracy=accuracy,
            raw_output=raw_output,
            row_level_items=rows,
            status=status,
            error_message=error_message,
        )
