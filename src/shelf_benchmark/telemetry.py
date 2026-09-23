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

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from shelf_benchmark.config import TelemetryConfig
from shelf_benchmark.models import AccuracyMetrics, CostMetrics, TokenUsageMetrics


class OpenTelemetryBenchmarkLogger:
    """Manages OpenTelemetry tracing and emits OTel-compliant JSONL log records locally and to GCP Cloud Logging & GCS."""

    def __init__(
        self,
        config: TelemetryConfig,
        project_id: str = "unilever-shelf-understanding",
        location: str = "global",
        bucket_name: str = "unilever-shelf-understanding-shelf-images",
    ):
        self.config = config
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name

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
        if not isinstance(trace.get_tracer_provider(), TracerProvider):
            trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer(config.service_name, "0.1.0")

        self.log_path = Path(config.otel_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_cloud_logging_status: Optional[str] = None
        self._last_gcs_otel_uri: Optional[str] = None
        self._gcs_dirty: bool = False
        self._gcs_upload_count: int = 0
        self._cached_creds: Any = None
        self._cached_storage_client: Any = None

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def format_iso(dt: datetime) -> str:
        return dt.isoformat().replace("+00:00", "Z")

    @staticmethod
    def to_unix_nano(dt: datetime) -> int:
        return int(dt.timestamp() * 1_000_000_000)

    def _export_to_cloud_logging(
        self,
        otel_record: Dict[str, Any],
        trace_id_hex: str,
        span_id_hex: str,
        gcp_labels: Dict[str, str],
    ) -> Optional[str]:
        """Writes the OpenTelemetry structured entry directly to Google Cloud Logging API (`logging.googleapis.com`) and Cloud Run structured stdout."""
        import os
        import sys

        k_service = os.environ.get("K_SERVICE", "unilever-shelf-benchmark-service")
        k_revision = os.environ.get("K_REVISION", "")
        k_config = os.environ.get("K_CONFIGURATION", "unilever-shelf-benchmark-service")
        trace_path = f"projects/{self.project_id}/traces/{trace_id_hex}"

        if os.environ.get("K_SERVICE"):
            cloud_run_stdout_entry = {
                "severity": otel_record.get("SeverityText", "INFO"),
                "message": (
                    f"[OTel Benchmark] approach={otel_record.get('Attributes', {}).get('shelf_benchmark.separation_approach')} "
                    f"model={otel_record.get('Attributes', {}).get('gen_ai.request.model')} "
                    f"facings={otel_record.get('Attributes', {}).get('shelf_benchmark.product_count')} "
                    f"latency_ms={otel_record.get('Attributes', {}).get('latency_ms')} "
                    f"cost_usd=${otel_record.get('Attributes', {}).get('shelf_benchmark.cost_per_shelf_image_usd')} "
                    f"trace_id={trace_id_hex}"
                ),
                "logging.googleapis.com/trace": trace_path,
                "logging.googleapis.com/spanId": span_id_hex,
                "logging.googleapis.com/labels": {str(k): str(v) for k, v in (gcp_labels or {}).items()},
                "otel_span": otel_record,
            }
            print(json.dumps(cloud_run_stdout_entry), file=sys.stdout, flush=True)

        if not getattr(self.config, "export_to_gcp_cloud_logging", True):
            return None
        try:
            import urllib.request

            import google.auth.transport.requests

            from shelf_benchmark.auth import get_gcp_credentials

            creds = get_gcp_credentials(self.project_id)
            if not getattr(creds, "token", None):
                auth_req = google.auth.transport.requests.Request()
                creds.refresh(auth_req)
            token = creds.token
            if not token:
                return None

            log_name = f"projects/{self.project_id}/logs/{getattr(self.config, 'gcp_log_name', 'unilever-shelf-benchmark-otel')}"
            if k_revision:
                resource_obj = {
                    "type": "cloud_run_revision",
                    "labels": {
                        "project_id": self.project_id,
                        "service_name": k_service,
                        "revision_name": k_revision,
                        "configuration_name": k_config,
                        "location": "us-central1",
                    },
                }
            else:
                resource_obj = {
                    "type": "global",
                    "labels": {"project_id": self.project_id},
                }
            payload = {
                "logName": log_name,
                "resource": resource_obj,
                "labels": {str(k): str(v) for k, v in (gcp_labels or {}).items()},
                "entries": [
                    {
                        "severity": otel_record.get("SeverityText", "INFO"),
                        "trace": trace_path,
                        "spanId": span_id_hex,
                        "jsonPayload": otel_record,
                    }
                ],
            }
            req = urllib.request.Request(
                "https://logging.googleapis.com/v2/entries:write",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "x-goog-user-project": self.project_id,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                if resp.status in (200, 201):
                    self._last_cloud_logging_status = f"EXPORTED ({log_name})"
                    return log_name
        except Exception as exc:
            self._last_cloud_logging_status = f"FALLBACK_LOCAL_AND_GCS ({type(exc).__name__}: {exc})"
        return None

    def target_gcs_uri(self) -> Optional[str]:
        """Return the canonical `gs://` destination for the OTel JSONL log without performing I/O."""
        if not getattr(self.config, "sync_otel_logs_to_gcs", True):
            return None
        clean_bucket = self.bucket_name.replace("gs://", "").strip("/").split("/")[0]
        if not clean_bucket:
            return None
        prefix = getattr(self.config, "gcs_otel_logs_prefix", "otel").strip("/")
        return f"gs://{clean_bucket}/{prefix}/{self.log_path.name}"

    def _get_credentials(self) -> Any:
        if self._cached_creds is None:
            from shelf_benchmark.auth import get_gcp_credentials

            self._cached_creds = get_gcp_credentials(self.project_id)
        return self._cached_creds

    def _sync_to_gcs(self) -> Optional[str]:
        """Uploads the OpenTelemetry JSONL log file to Google Cloud Storage (`gs://<bucket>/otel/otel_logs.jsonl`)."""
        if not getattr(self.config, "sync_otel_logs_to_gcs", True):
            return None
        try:
            import google.cloud.storage as storage

            if self._cached_storage_client is None:
                creds = self._get_credentials()
                self._cached_storage_client = storage.Client(
                    project=self.project_id, credentials=creds
                )
            clean_bucket = self.bucket_name.replace("gs://", "").strip("/").split("/")[0]
            bucket = self._cached_storage_client.bucket(clean_bucket)
            prefix = getattr(self.config, "gcs_otel_logs_prefix", "otel").strip("/")
            blob_path = f"{prefix}/{self.log_path.name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(self.log_path), content_type="application/jsonl")
            gcs_uri = f"gs://{clean_bucket}/{blob_path}"
            self._last_gcs_otel_uri = gcs_uri
            self._gcs_dirty = False
            self._gcs_upload_count += 1
            return gcs_uri
        except Exception:
            return None

    def flush(self) -> Optional[str]:
        """Upload the accumulated JSONL log to GCS once if new spans were written since the last sync.

        Previously `_sync_to_gcs()` ran inside `log_task_execution` on every span, re-uploading the
        entire JSONL file N times (O(N^2) bytes transferred). Buffering per-span writes and flushing
        at run completion makes GCS sync O(N) in bytes and O(1) in upload requests.
        """
        if not self._gcs_dirty:
            return self._last_gcs_otel_uri
        return self._sync_to_gcs()

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
        """Create an OTel span and write an OpenTelemetry Data Model compliant JSONL record + export to GCP Cloud Logging & GCS."""
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
                "shelf_benchmark.cost.vertex_ai_payg_tokens_usd": round(cost.vertex_ai_payg_tokens_usd, 8),
                "shelf_benchmark.cost.vertex_ai_provisioned_throughput_usd": round(
                    cost.vertex_ai_provisioned_throughput_usd, 8
                ),
                "shelf_benchmark.cost.vertex_ai_embeddings_and_vision_usd": round(
                    cost.vertex_ai_embeddings_and_vision_usd, 8
                ),
                "shelf_benchmark.cost.cloud_run_compute_usd": round(cost.cloud_run_compute_usd, 8),
                "shelf_benchmark.cost.gcs_and_observability_usd": round(cost.gcs_and_observability_usd, 8),
                "shelf_benchmark.cost.traffic_type": cost.traffic_type,
                "shelf_benchmark.cost.billing_source": cost.billing_source,
                "shelf_benchmark.status": status,
            }
            if cost.gcp_billing_labels:
                for lk, lv in cost.gcp_billing_labels.items():
                    attributes[f"gcp.billing.label.{lk}"] = lv

            if error_message:
                attributes["shelf_benchmark.error_message"] = error_message
            if accuracy and accuracy.ground_truth_available:
                attributes["shelf_benchmark.accuracy.gt_version"] = accuracy.gt_version
                attributes["shelf_benchmark.accuracy.iou_threshold"] = accuracy.iou_threshold
                attributes["shelf_benchmark.accuracy.pairing_strategy"] = accuracy.pairing_strategy
                attributes["shelf_benchmark.accuracy.brand_matcher"] = accuracy.brand_matcher
                attributes["shelf_benchmark.accuracy.product_matcher"] = accuracy.product_matcher
                # Now Optional: omit rather than export None, which OTel rejects and which
                # would otherwise be exported as a zero count by a lenient exporter.
                if accuracy.true_positives is not None:
                    attributes["shelf_benchmark.accuracy.true_positives"] = accuracy.true_positives
                if accuracy.false_positives is not None:
                    attributes["shelf_benchmark.accuracy.false_positives"] = (
                        accuracy.false_positives
                    )
                if accuracy.false_negatives is not None:
                    attributes["shelf_benchmark.accuracy.false_negatives"] = (
                        accuracy.false_negatives
                    )
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
                if accuracy.detection_precision is not None:
                    attributes["shelf_benchmark.accuracy.detection_precision"] = round(
                        accuracy.detection_precision, 4
                    )
                if accuracy.detection_recall is not None:
                    attributes["shelf_benchmark.accuracy.detection_recall"] = round(
                        accuracy.detection_recall, 4
                    )
                if accuracy.detection_f1 is not None:
                    attributes["shelf_benchmark.accuracy.detection_f1"] = round(
                        accuracy.detection_f1, 4
                    )
                if accuracy.mean_iou_matched is not None:
                    attributes["shelf_benchmark.accuracy.mean_iou_matched"] = round(
                        accuracy.mean_iou_matched, 4
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
                f"cost_image=${cost.cost_per_shelf_image_usd:.6f} "
                f"(payg=${cost.vertex_ai_payg_tokens_usd:.6f}, pt_gsu=${cost.vertex_ai_provisioned_throughput_usd:.6f}, "
                f"embed=${cost.vertex_ai_embeddings_and_vision_usd:.6f}, cloud_run=${cost.cloud_run_compute_usd:.6f}, "
                f"gcs_obs=${cost.gcs_and_observability_usd:.6f})"
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

        import os

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_line = (json.dumps(otel_record) + "\n").encode("utf-8")
        fd = os.open(str(self.log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, encoded_line)
            os.fsync(fd)
        finally:
            os.close(fd)
        self._gcs_dirty = True

        # Export directly to Google Cloud Logging; record target GCS URI and defer full-file upload to flush()
        cloud_log_name = self._export_to_cloud_logging(
            otel_record=otel_record,
            trace_id_hex=trace_id_hex,
            span_id_hex=span_id_hex,
            gcp_labels=cost.gcp_billing_labels,
        )
        gcs_otel_uri = self.target_gcs_uri()
        rec_attrs = otel_record.get("Attributes")
        if isinstance(rec_attrs, dict):
            if cloud_log_name:
                rec_attrs["gcp.cloud_logging.log_name"] = cloud_log_name
            if gcs_otel_uri:
                rec_attrs["gcp.gcs.otel_log_uri"] = gcs_otel_uri

        return trace_id_hex, span_id_hex, otel_record
