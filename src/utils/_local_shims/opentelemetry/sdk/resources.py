"""OpenTelemetry SDK Resource & TracerProvider (`src/opentelemetry/sdk/resources.py` & `trace`)."""

from __future__ import annotations

from typing import Any


class Resource:
    def __init__(self, attributes: dict[str, Any] | None = None) -> None:
        self.attributes = dict(attributes or {})

    @classmethod
    def create(cls, attributes: dict[str, Any] | None = None) -> Resource:
        return cls(attributes)
