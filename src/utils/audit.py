"""Audit view of one scored image: its spans (Cloud Trace) and log entries (Cloud Logging).

The UI's "Load Cloud Trace + Logging" button calls this for an image. It returns what GCP
recorded under the image's span: every ``gemini <model>`` child span (all attributes: tokens by
kind, traffic type, retries, finish reason, cost) joined with that call's ``gemini_call`` /
``gemini_call_failed`` log entry (the full jsonPayload, prompt and response included), plus the
image's own entries (``image_scored``). Times are ms from the image span's start, so the UI can
place each call on the step it belongs to (exactly, by ``span_id``, for runs that record it;
by timestamp for older runs).

Works for local and Cloud Run runs alike (Cloud Run's structured stdout lands in the same trace).
Needs read access: ``roles/cloudtrace.user`` and ``roles/logging.viewer`` (or the viewer role).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from urllib.parse import quote

TRACE_API = "https://cloudtrace.googleapis.com/v1/projects/{project}/traces/{trace}"
LOGGING_API = "https://logging.googleapis.com/v2/entries:list"
_CACHE_SIZE = 8  # traces kept in memory: every image of a run shares its run's trace

_lock = threading.Lock()
_traces: OrderedDict[str, dict] = OrderedDict()
_session = None


def _http():
    global _session
    with _lock:
        if _session is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            _session = AuthorizedSession(creds)
        return _session


def _ts(s: str) -> datetime:
    """RFC 3339 with nanoseconds -> datetime (microseconds)."""
    head, _, frac = s.rstrip("Z").partition(".")
    return datetime.fromisoformat(f"{head}.{(frac + '000000')[:6]}+00:00")


def _hex(span_id: str) -> str:
    """Cloud Trace v1 span ids are decimal; logs and OTel use 16 hex digits."""
    return format(int(span_id), "016x")


def _trace(project: str, trace_id: str) -> dict:
    with _lock:
        if trace_id in _traces:
            _traces.move_to_end(trace_id)
            return _traces[trace_id]
    r = _http().get(TRACE_API.format(project=project, trace=trace_id), timeout=60)
    r.raise_for_status()
    data = r.json()
    with _lock:
        _traces[trace_id] = data
        while len(_traces) > _CACHE_SIZE:
            _traces.popitem(last=False)
    return data


def _entries(project: str, trace_id: str, span_ids: list[str], t0: datetime, t1: datetime) -> list[dict]:
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
    ids = " OR ".join(f'spanId="{s}"' for s in span_ids)
    flt = (f'trace="projects/{project}/traces/{trace_id}" AND ({ids}) '
           f'AND timestamp>="{(t0 - timedelta(minutes=1)).strftime(fmt)}" '
           f'AND timestamp<="{(t1 + timedelta(minutes=10)).strftime(fmt)}"')
    out, token = [], None
    while True:
        body = {"resourceNames": [f"projects/{project}"], "filter": flt, "pageSize": 200,
                "orderBy": "timestamp asc", **({"pageToken": token} if token else {})}
        r = _http().post(LOGGING_API, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        out += data.get("entries", [])
        token = data.get("nextPageToken")
        if not token:
            return out


def _links(project: str, trace_id: str, span_id: str, t0: datetime) -> dict:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    q = (f'trace="projects/{project}/traces/{trace_id}" spanId="{span_id}" '
         f'timestamp>="{(t0 - timedelta(minutes=5)).strftime(fmt)}" '
         f'timestamp<="{(t0 + timedelta(days=1)).strftime(fmt)}"')
    return {"trace_url": f"https://console.cloud.google.com/traces/explorer;traceId={trace_id};"
                         f"spanId={span_id}?project={project}",
            "logs_url": f"https://console.cloud.google.com/logs/query;query={quote(q, safe='')}"
                        f"?project={project}"}


def _entry(e: dict) -> dict:
    return {"timestamp": e.get("timestamp"), "severity": e.get("severity", "DEFAULT"),
            "log_name": e.get("logName", "").rsplit("/", 1)[-1],
            "resource": e.get("resource", {}).get("type"),
            "payload": e.get("jsonPayload") or e.get("textPayload")}


def image_audit(project: str, trace_id: str, span_id: str) -> dict:
    """Spans + log entries under one image span (see the module docstring)."""
    spans = _trace(project, trace_id).get("spans", [])
    img = next((s for s in spans if _hex(s["spanId"]) == span_id), None)
    if img is None:
        raise FileNotFoundError(f"span {span_id} not in trace {trace_id} (not exported yet?)")
    t0, t1 = _ts(img["startTime"]), _ts(img["endTime"])
    ms = lambda s: round((_ts(s) - t0).total_seconds() * 1000, 1)  # noqa: E731
    kids = sorted((s for s in spans if s.get("parentSpanId") == img["spanId"]),
                  key=lambda s: s["startTime"])
    by_span: dict[str, list[dict]] = {}
    for e in _entries(project, trace_id, [span_id] + [_hex(s["spanId"]) for s in kids], t0, t1):
        by_span.setdefault(e.get("spanId", ""), []).append(_entry(e))
    calls = [{
        "span_id": _hex(s["spanId"]),
        "name": s["name"],
        "start_ms": ms(s["startTime"]),
        "end_ms": ms(s["endTime"]),
        "attributes": {k: v for k, v in sorted(s.get("labels", {}).items()) if k != "g.co/agent"},
        "entries": by_span.get(_hex(s["spanId"]), []),
        **_links(project, trace_id, _hex(s["spanId"]), t0),
    } for s in kids]
    return {
        "image": {"span_id": span_id, "name": img["name"], "start": img["startTime"],
                  "duration_ms": ms(img["endTime"]),
                  "attributes": {k: v for k, v in sorted(img.get("labels", {}).items())
                                 if k != "g.co/agent"},
                  **_links(project, trace_id, span_id, t0)},
        "entries": by_span.get(span_id, []),
        "calls": calls,
    }
