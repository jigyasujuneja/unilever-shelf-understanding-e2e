"""OpenTelemetry traces (Cloud Trace) + correlated structured logs (Cloud Logging).

Every run is one trace::

    run <run_id>                   run config; final scores, tokens and cost
      image <image_id>             gt/pred/tp/fp/fn, F2, latency, cost, error; one event per step
        gemini <model>             tokens by kind (input / image / text / cached / output /
                                   thinking), traffic type served, retries, finish reason, cost

Each span also writes one Cloud Logging entry with the same fields (Gemini calls add the prompt
and response text), attached to that span. So "Logs" on a span in Cloud Trace shows its entry,
and ``summary.json["telemetry"]["logs_url"]`` shows every entry of the run.

Approach authors don't need to touch this: ``ctx.ask`` and ``ctx.trace.step`` record
everything. Turn it off with ``telemetry.enabled: false`` in config.yaml or
``SHELF_BENCH_TELEMETRY=0``. Needs ``roles/cloudtrace.agent`` and ``roles/logging.logWriter``
(the default compute service account used by the Cloud Run job has both).
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

SERVICE = "shelf-bench"
LOG = logging.getLogger("shelf_bench.telemetry")
LOG.setLevel(logging.INFO)
MAX_TEXT = 64_000  # chars of prompt / response kept per log entry (entries are capped at 256 KB)

_lock = threading.Lock()
_provider = None  # opentelemetry.sdk.trace.TracerProvider once enabled
_handler: logging.Handler | None = None
_project: str | None = None


def enabled_in(config: dict) -> bool:
    env = os.environ.get("SHELF_BENCH_TELEMETRY")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "off", "no", "")
    return bool(config.get("telemetry", {}).get("enabled", True))


def init(config: dict) -> bool:
    """Export spans to Cloud Trace and logs to Cloud Logging (once per process)."""
    global _provider, _handler, _project
    if not enabled_in(config):
        return False
    with _lock:
        if _provider is not None:
            return True
        _project = config.get("gcp", {}).get("project")
        provider = _new_provider()
        try:
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider.add_span_processor(BatchSpanProcessor(_Loud(CloudTraceSpanExporter(project_id=_project))))
            _handler = _cloud_log_handler(_project)
            LOG.addHandler(_handler)
            LOG.propagate = False  # keep the JSON entries out of the console output
        except ImportError:
            from opentelemetry.sdk.trace.export import InMemorySpanExporter, SimpleSpanProcessor

            provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
        _provider = provider
    return True


class _Loud:
    """SpanExporter wrapper: say once, clearly, when spans can't be written (usually IAM)."""

    def __init__(self, inner):
        self.inner, self.warned = inner, False

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult

        res = self.inner.export(spans)
        if res != SpanExportResult.SUCCESS and not self.warned:
            self.warned = True
            import sys

            print(f"WARNING: Cloud Trace export failed, so this run's trace will be missing (logs "
                  f"are unaffected). The credentials need roles/cloudtrace.agent on {_project}; "
                  f"see the error above.", file=sys.stderr)
        return res

    def shutdown(self):
        return self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return getattr(self.inner, "force_flush", lambda *_: True)(timeout_millis)


def use_exporter(exporter, project: str | None = "test-project") -> None:
    """Tests: send spans synchronously to ``exporter`` (e.g. InMemorySpanExporter)."""
    global _provider, _project
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = _new_provider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _provider, _project = provider, project


def reset() -> None:
    global _provider, _handler, _project
    if _handler is not None:
        LOG.removeHandler(_handler)
        LOG.propagate = True
    _provider = _handler = _project = None


def flush() -> None:
    if _provider is not None:
        _provider.force_flush(30_000)
    if _handler is not None:
        _handler.flush()


def _new_provider():
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    attrs = {"service.name": SERVICE}
    for env, key in (("CLOUD_RUN_JOB", "cloud_run.job"), ("CLOUD_RUN_EXECUTION", "cloud_run.execution"),
                     ("CLOUD_RUN_TASK_INDEX", "cloud_run.task_index")):
        if os.environ.get(env):
            attrs[key] = os.environ[env]
    return TracerProvider(resource=Resource.create(attrs))


def _cloud_log_handler(project: str | None) -> logging.Handler:
    from google.cloud.logging_v2.handlers import CloudLoggingHandler, StructuredLogHandler

    if os.environ.get("CLOUD_RUN_JOB"):  # Cloud Run ships JSON lines on stdout to Cloud Logging
        return StructuredLogHandler(project_id=project)
    import google.cloud.logging
    from google.cloud.logging import Resource

    client = google.cloud.logging.Client(project=project)
    return CloudLoggingHandler(client, name=SERVICE, resource=Resource(type="global", labels={}))


def tracer() -> trace.Tracer:
    return _provider.get_tracer(SERVICE) if _provider is not None else trace.NoOpTracer()


def ids(span: Span | None) -> tuple[str | None, str | None]:
    """(trace_id, span_id) as hex, or (None, None) when telemetry is off."""
    c = span.get_span_context() if span is not None else None
    if not c or not c.is_valid:
        return None, None
    return trace.format_trace_id(c.trace_id), trace.format_span_id(c.span_id)


# ---- console links ---------------------------------------------------------------------------

def _logs_url(project: str, query: str) -> str:
    return (f"https://console.cloud.google.com/logs/query;query={quote(query, safe='')}"
            f"?project={project}")


def _window(span: Span) -> str:
    """Timestamp bounds for a log query. Logs Explorer otherwise searches only the last hour
    (and the API pages through everything), so links to older runs would show nothing."""
    ns = getattr(span, "start_time", None)
    t0 = datetime.fromtimestamp(ns / 1e9, timezone.utc) if ns else datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (f'timestamp>="{(t0 - timedelta(minutes=5)).strftime(fmt)}" '
            f'timestamp<="{(t0 + timedelta(days=1)).strftime(fmt)}"')


def links(span: Span | None, env: dict | None = None, per_span: bool = False) -> dict | None:
    """Where to look in the console. ``per_span`` narrows the logs link to this span only."""
    trace_id, span_id = ids(span)
    if not trace_id or not _project:
        return None
    when = _window(span)
    q = f'trace="projects/{_project}/traces/{trace_id}" {when}'
    out = {
        "trace_id": trace_id,
        "span_id": span_id,
        "trace_url": f"https://console.cloud.google.com/traces/explorer;traceId={trace_id}"
                     + (f";spanId={span_id}" if per_span else "") + f"?project={_project}",
        "logs_url": _logs_url(_project, q + (f' spanId="{span_id}"' if per_span else "")),
    }
    if env and env.get("platform") == "cloud-run":  # the task's whole stdout / stderr
        out["task_logs_url"] = _logs_url(_project, "\n".join([
            'resource.type="cloud_run_job"',
            f'resource.labels.job_name="{env["job"]}"',
            f'labels."run.googleapis.com/execution_name"="{env["execution"]}"',
            f'labels."run.googleapis.com/task_index"="{env["task_index"]}"',
            when,
        ]))
    return out


# ---- spans + logs ----------------------------------------------------------------------------

def clean(fields: dict[str, Any]) -> dict[str, Any]:
    """OTel attribute values must be str/bool/int/float (or lists of one type); drop None."""
    out: dict[str, Any] = {}
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, (str, bool, int, float)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out


def set_attrs(span: Span, fields: dict[str, Any]) -> None:
    span.set_attributes(clean(fields))


def log(span: Span | None, message: str, fields: dict[str, Any],
        level: int = logging.INFO) -> None:
    """One Cloud Logging entry attached to ``span`` (jsonPayload = ``fields``)."""
    trace_id, span_id = ids(span)
    extra: dict[str, Any] = {"json_fields": {"event": message, **fields}}
    if trace_id and _project:
        extra.update(trace=f"projects/{_project}/traces/{trace_id}", span_id=span_id,
                     trace_sampled=True)
    LOG.log(level, message, extra=extra)


@contextmanager
def span(name: str, parent=None, **attrs):
    """Start a span under ``parent`` (an OTel Context; needed across threads)."""
    with tracer().start_as_current_span(name, context=parent,
                                        attributes=clean(attrs)) as s:
        yield s


def child_context(s: Span):
    return trace.set_span_in_context(s)


def fail(s: Span, e: BaseException) -> None:
    s.record_exception(e)
    s.set_status(Status(StatusCode.ERROR, f"{type(e).__name__}: {e}"[:300]))


def usage_attrs(u) -> dict[str, int]:
    """Token counts of a ``utils.llm.Usage``, split every way pricing cares about."""
    b = u.buckets

    def total(kind: str) -> int:
        return sum(v for k, v in b.items() if k.split("/", 1)[-1] == kind)

    return {
        "gen_ai.usage.input_tokens": u.input_tokens,
        "gen_ai.usage.output_tokens": u.output_tokens,
        "shelf_bench.usage.thinking_tokens": u.thinking_tokens,
        "shelf_bench.usage.billed_output_tokens": u.output_tokens + u.thinking_tokens,
        "shelf_bench.usage.image_input_tokens": total("image_input") + total("cached_image_input"),
        "shelf_bench.usage.text_input_tokens": total("text_input") + total("cached_text_input"),
        "shelf_bench.usage.cached_input_tokens": total("cached_image_input") + total("cached_text_input"),
        "shelf_bench.usage.total_tokens": u.input_tokens + u.output_tokens + u.thinking_tokens,
        "shelf_bench.usage.calls": u.calls,
    }


def truncate(text: str | None) -> str | None:
    if text is None or len(text) <= MAX_TEXT:
        return text
    return text[:MAX_TEXT] + f"... [{len(text) - MAX_TEXT} more chars]"
