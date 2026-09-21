"""OpenTelemetry-Compliant Logging & Tracing for Vertex AI Gemini Shelf Benchmarks.

Always logs:
- start_time (ISO-8601 UTC & Unix epoch nanoseconds)
- end_time (ISO-8601 UTC & Unix epoch nanoseconds)
- duration_ms / latency_ms
- input_token_count (`gen_ai.usage.input_tokens`)
- thinking_token_count (`gen_ai.usage.thinking_tokens`)
- output_token_count (`gen_ai.usage.output_tokens`)
- total_token_count (`gen_ai.usage.total_tokens`)
- cached_token_count (`gen_ai.usage.cached_tokens`)
- cost_per_shelf_image_usd & cost_per_product_usd
- trace_id, span_id, resource & gen_ai.* semantic attributes
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from shelf_benchmark.config import TelemetryConfig
from shelf_benchmark.models import AccuracyMetrics, CostMetrics, TokenUsageMetrics


class OpenTelemetryBenchmarkLogger:
    """Manages OpenTelemetry tracing and emits OTel-compliant JSONL log records."""

    def __init__(self, config: TelemetryConfig, project_id: str = "unilever-shelf-understanding", location: str = "global"):
        self.config = config
        self.project_id = project_id
        self.location = location

        resource = Resource.create(
            {
                "service.name": config.service_name,
                "service.version": "0.1.0",
                "cloud.provider": "gcp",
                "cloud.account.id": project_id,
                "cloud.region": location,
            }
        )
        self.provider = TracerProvider(resource=resource)
        if config.export_to_console:
            self.provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer(config.service_name, "0.1.0")

        self.log_path = Path(config.otel_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def format_iso(dt: datetime) -> str:
        return dt.isoformat().replace("+00:00", "Z")

    @staticmethod
    def to_unix_nano(dt: datetime) -> int:
        return int(dt.timestamp() * 1_000_000_000)

    def log_task_execution(
        self,
        *,
        run_id: str,
        task_type: str,
        model_name: str,
        shelf_image_uri: str,
        start_dt: datetime,
        end_dt: datetime,
        tokens: TokenUsageMetrics,
        cost: CostMetrics,
        accuracy: Optional[AccuracyMetrics] = None,
        status: str = "SUCCESS",
        error_message: Optional[str] = None,
        extra_attributes: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str, Dict[str, Any]]:
        """Create an OTel span and write an OpenTelemetry Data Model compliant JSONL record."""
        start_iso = self.format_iso(start_dt)
        end_iso = self.format_iso(end_dt)
        start_nano = self.to_unix_nano(start_dt)
        end_nano = self.to_unix_nano(end_dt)
        latency_ms = round((end_dt - start_dt).total_seconds() * 1000.0, 3)

        with self.tracer.start_as_current_span(f"gen_ai.{task_type}") as span:
            ctx = span.get_span_context()
            trace_id_hex = f"{ctx.trace_id:032x}" if ctx and ctx.trace_id else uuid.uuid4().hex
            span_id_hex = f"{ctx.span_id:016x}" if ctx and ctx.span_id else uuid.uuid4().hex[:16]

            attributes: Dict[str, Any] = {
                "run_id": run_id,
                "start_time": start_iso,
                "end_time": end_iso,
                "start_time_unix_nano": start_nano,
                "end_time_unix_nano": end_nano,
                "latency_ms": latency_ms,
                "input_token_count": tokens.input_tokens,
                "thinking_token_count": tokens.thinking_tokens,
                "output_token_count": tokens.output_tokens,
                "total_token_count": tokens.total_tokens,
                "cached_token_count": tokens.cached_tokens,
                "gen_ai.system": "vertex_ai",
                "gen_ai.operation.name": task_type,
                "gen_ai.request.model": model_name,
                "gen_ai.response.model": model_name,
                "gen_ai.usage.input_tokens": tokens.input_tokens,
                "gen_ai.usage.thinking_tokens": tokens.thinking_tokens,
                "gen_ai.usage.output_tokens": tokens.output_tokens,
                "gen_ai.usage.total_tokens": tokens.total_tokens,
                "gen_ai.usage.cached_tokens": tokens.cached_tokens,
                "shelf_benchmark.task_type": task_type,
                "shelf_benchmark.shelf_image_uri": shelf_image_uri,
                "shelf_benchmark.product_count": cost.product_count,
                "shelf_benchmark.cost_per_shelf_image_usd": round(cost.cost_per_shelf_image_usd, 8),
                "shelf_benchmark.cost_per_product_usd": round(cost.cost_per_product_usd, 8),
                "shelf_benchmark.status": status,
            }
            if error_message:
                attributes["shelf_benchmark.error_message"] = error_message
            if accuracy and accuracy.ground_truth_available:
                if accuracy.count_accuracy is not None:
                    attributes["shelf_benchmark.accuracy.count_accuracy"] = round(accuracy.count_accuracy, 4)
                if accuracy.brand_classification_accuracy is not None:
                    attributes["shelf_benchmark.accuracy.brand_accuracy"] = round(
                        accuracy.brand_classification_accuracy, 4
                    )
                if accuracy.product_classification_accuracy is not None:
                    attributes["shelf_benchmark.accuracy.product_accuracy"] = round(
                        accuracy.product_classification_accuracy, 4
                    )
                if accuracy.detection_f1_iou50 is not None:
                    attributes["shelf_benchmark.accuracy.detection_f1_iou50"] = round(
                        accuracy.detection_f1_iou50, 4
                    )
                if accuracy.sku_matching_accuracy is not None:
                    attributes["shelf_benchmark.accuracy.sku_matching_accuracy"] = round(
                        accuracy.sku_matching_accuracy, 4
                    )
            if extra_attributes:
                for k, v in extra_attributes.items():
                    if v is not None:
                        attributes[k] = v

            for k, v in attributes.items():
                if isinstance(v, (str, int, float, bool)):
                    span.set_attribute(k, v)

        otel_record = {
            "Timestamp": end_nano,
            "ObservedTimestamp": self.to_unix_nano(self.now_utc()),
            "TraceId": trace_id_hex,
            "SpanId": span_id_hex,
            "SeverityText": "INFO" if status == "SUCCESS" else "ERROR",
            "SeverityNumber": 9 if status == "SUCCESS" else 17,
            "Body": (
                f"[{task_type.upper()}] model={model_name} image={shelf_image_uri} "
                f"start={start_iso} end={end_iso} latency_ms={latency_ms} "
                f"tokens(in={tokens.input_tokens}, think={tokens.thinking_tokens}, out={tokens.output_tokens}, total={tokens.total_tokens}) "
                f"cost_image=${cost.cost_per_shelf_image_usd:.6f} cost_product=${cost.cost_per_product_usd:.6f}"
            ),
            "Resource": {
                "service.name": self.config.service_name,
                "cloud.provider": "gcp",
                "cloud.account.id": self.project_id,
                "cloud.region": self.location,
            },
            "InstrumentationScope": {
                "Name": "shelf_benchmark.telemetry",
                "Version": "0.1.0",
            },
            "Attributes": attributes,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(otel_record) + "\n")

        return trace_id_hex, span_id_hex, otel_record
