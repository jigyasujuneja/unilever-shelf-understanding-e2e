"""Backward-compatible re-export of ``stages.stage6_shelf_metrics``."""

from __future__ import annotations

from stages.stage6_shelf_metrics import (
    compute_dynamic_unilever_kpis,
    evaluate_shelf_metrics,
)

__all__ = [
    "compute_dynamic_unilever_kpis",
    "evaluate_shelf_metrics",
]
