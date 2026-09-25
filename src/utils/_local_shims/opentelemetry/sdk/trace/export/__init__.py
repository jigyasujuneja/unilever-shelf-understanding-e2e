"""OpenTelemetry SDK Span Exporters & Processors (`src/opentelemetry/sdk/trace/export/__init__.py`)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class SpanExportResult(Enum):
    SUCCESS = 0
    FAILURE = 1


class SimpleSpanProcessor:
    def __init__(self, exporter: Any) -> None:
        self.exporter = exporter

    def on_end(self, span: Any) -> None:
        self.exporter.export([span])

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class BatchSpanProcessor(SimpleSpanProcessor):
    pass


class InMemorySpanExporter:
    def __init__(self) -> None:
        self._spans: list[Any] = []

    def export(self, spans: list[Any]) -> SpanExportResult:
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[Any]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()

    def shutdown(self) -> None:
        pass
