"""Lightweight stdlib-backed OpenTelemetry implementation (`src/opentelemetry/trace.py`) for local & test execution."""

from __future__ import annotations

import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StatusCode(Enum):
    UNSET = 0
    OK = 1
    ERROR = 2


@dataclass
class Status:
    status_code: StatusCode = StatusCode.UNSET
    description: str | None = None


@dataclass
class SpanContext:
    trace_id: int
    span_id: int
    is_valid: bool = True


@dataclass
class Event:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


class Span:
    def __init__(
        self,
        name: str = "noop",
        trace_id: int = 0,
        span_id: int = 0,
        parent: SpanContext | None = None,
        attributes: dict[str, Any] | None = None,
        processors: list[Any] | None = None,
        valid: bool = False,
    ) -> None:
        self.name = name
        self._ctx = SpanContext(trace_id=trace_id, span_id=span_id, is_valid=valid)
        self.parent = parent
        self.attributes: dict[str, Any] = dict(attributes or {})
        self.events: list[Event] = []
        self.status = Status(StatusCode.UNSET)
        self.start_time = int(time.time() * 1e9)
        self._processors = processors or []

    def get_span_context(self) -> SpanContext:
        return self._ctx

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self.attributes.update(attrs)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(Event(name=name, attributes=dict(attributes or {})))

    def set_status(self, status: Status, description: str | None = None) -> None:
        if isinstance(status, StatusCode):
            self.status = Status(status, description)
        else:
            self.status = status

    def record_exception(self, exc: BaseException) -> None:
        self.add_event("exception", {"exception.type": type(exc).__name__, "exception.message": str(exc)})

    def end(self) -> None:
        for p in self._processors:
            p.on_end(self)

    def __enter__(self) -> Span:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.end()


class NoOpTracer:
    def start_span(self, name: str, context: Any = None, attributes: dict[str, Any] | None = None) -> Span:
        return Span(name=name, valid=False, attributes=attributes)

    @contextmanager
    def start_as_current_span(self, name: str, context: Any = None, attributes: dict[str, Any] | None = None):
        s = self.start_span(name, context=context, attributes=attributes)
        try:
            yield s
        finally:
            s.end()


class Tracer(NoOpTracer):
    def __init__(self, processors: list[Any]) -> None:
        self._processors = processors

    def start_span(self, name: str, context: Any = None, attributes: dict[str, Any] | None = None) -> Span:
        parent_ctx: SpanContext | None = None
        if isinstance(context, Span):
            parent_ctx = context.get_span_context()
        elif isinstance(context, SpanContext):
            parent_ctx = context
        trace_id = parent_ctx.trace_id if (parent_ctx and parent_ctx.is_valid) else secrets.randbits(128)
        span_id = secrets.randbits(64)
        return Span(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent=parent_ctx,
            attributes=attributes,
            processors=self._processors,
            valid=True,
        )


def format_trace_id(trace_id: int) -> str:
    return f"{trace_id:032x}"


def format_span_id(span_id: int) -> str:
    return f"{span_id:016x}"


def set_span_in_context(span: Span) -> SpanContext:
    return span.get_span_context()
