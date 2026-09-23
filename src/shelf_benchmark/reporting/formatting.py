"""Value formatters shared by every report surface.

Isolated so that "not measured" renders identically everywhere. The single most damaging
thing this reporting layer can do is print `0.0%` for a metric that was never computed, so
the `None` handling lives in one place rather than being re-decided at each call site.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "UNKNOWN",
    "fmt_count",
    "fmt_float",
    "fmt_pct",
    "fmt_usd",
    "slugify",
]

UNKNOWN = "unknown"


def fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def fmt_float(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_count(value: Optional[int]) -> str:
    """Render a confusion-matrix count, keeping "not measured" distinct from "zero"."""
    return "n/a" if value is None else str(value)


def fmt_usd(value: Optional[float]) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def slugify(value: str) -> str:
    """Filesystem-safe fragment for run directory names."""
    return "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in str(value)).strip("-")
