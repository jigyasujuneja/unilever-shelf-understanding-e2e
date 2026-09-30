"""Stage 6: Shelf share, out-of-stock, and planogram compliance metrics (`src/stages/stage6_shelf_metrics.py`).

Registers ``shelf_metrics`` stage implementations backed by `core.analytics`:
  - ``dual_mt_marketshare_and_merchandising`` (default): Computes both multi-image market share
    and single-image merchandising compliance metrics from predicted boxes and labels.
  - ``panorama_6img_stitch_sos``: Multi-image shelf stitching and linear/area share-of-shelf.
  - ``single_img_planogram_oos``: Single-image out-of-stock gap and planogram sequence checks.
"""

from __future__ import annotations

from core.analytics import compute_geometric_shelf_kpis, evaluate_shelf_metrics
from stages.registry import StageSpec, register_stage

# Backward-compatible alias for existing callers.
compute_dynamic_unilever_kpis = evaluate_shelf_metrics


register_stage(
    StageSpec(
        stage_group="shelf_metrics",
        name="dual_mt_marketshare_and_merchandising",
        title="Market Share (6-Image <=30s) and Merchandising (1-Image <=10s) Metrics (Default)",
        description=(
            "Computes linear/area share-of-shelf, 6-image overlap deduplication, out-of-stock "
            "gaps, and planogram compliance from predicted boxes and attributes."
        ),
        f2_delta=0.0,
        latency_delta_s=0.010,
        cost_delta_inr=0.0,
        default=True,
        fn=evaluate_shelf_metrics,
    )
)

register_stage(
    StageSpec(
        stage_group="shelf_metrics",
        name="panorama_6img_stitch_sos",
        title="Market Share Only (6-Image Stitch and Share-of-Shelf)",
        description="Computes 6-image overlap deduplication and linear/area share-of-shelf.",
        f2_delta=0.0,
        latency_delta_s=0.006,
        cost_delta_inr=0.0,
        default=False,
        fn=evaluate_shelf_metrics,
    )
)

register_stage(
    StageSpec(
        stage_group="shelf_metrics",
        name="single_img_planogram_oos",
        title="Merchandising Only (Single-Image Out-of-Stock and Planogram Audit)",
        description="Computes single-image out-of-stock gaps, brand-block grouping, and planogram compliance.",
        f2_delta=0.0,
        latency_delta_s=0.005,
        cost_delta_inr=0.0,
        default=False,
        fn=evaluate_shelf_metrics,
    )
)

__all__ = [
    "compute_dynamic_unilever_kpis",
    "compute_geometric_shelf_kpis",
    "evaluate_shelf_metrics",
]
