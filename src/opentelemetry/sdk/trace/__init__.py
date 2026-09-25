"""OpenTelemetry SDK TracerProvider (`src/opentelemetry/sdk/trace/__init__.py`)."""

from __future__ import annotations

from typing import Any

from opentelemetry.trace import Tracer


class TracerProvider:
    def __init__(self, resource: Any = None) -> None:
        self.resource = resource
        self._processors: list[Any] = []

    def add_span_processor(self, processor: Any) -> None:
        self._processors.append(processor)

    def get_tracer(self, name: str) -> Tracer:
        return Tracer(self._processors)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        for p in self._processors:
            if hasattr(p, "force_flush"):
                p.force_flush(timeout_millis)
        return True
