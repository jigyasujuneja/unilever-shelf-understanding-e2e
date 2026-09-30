"""Modular tests for Stage 6 (`src/stages/stage6_shelf_metrics.py`) on real shelf detections."""

from __future__ import annotations

from conftest import RealShelfFixture

from stages import stage6_shelf_metrics
from utils import hul_domain


def test_stage6_dynamic_shelf_metrics_from_real_detections(
    shelf_fixture_a: RealShelfFixture,
    shelf_fixture_b: RealShelfFixture,
) -> None:
    preds_a = hul_domain.classify_shelf_boxes_7dim(
        shelf_fixture_a.image, shelf_fixture_a.boxes, mode="sister_shade_systemone"
    )
    preds_b = hul_domain.classify_shelf_boxes_7dim(
        shelf_fixture_b.image, shelf_fixture_b.boxes, mode="scann_vector_retriever"
    )
    rows_a = [
        {"preds": [list(b) for b in shelf_fixture_a.boxes], "pred_labels": preds_a, "width": shelf_fixture_a.width, "height": shelf_fixture_a.height}
    ]
    rows_b = [
        {"preds": [list(b) for b in shelf_fixture_b.boxes[:6]], "pred_labels": preds_b[:6], "width": shelf_fixture_b.width, "height": shelf_fixture_b.height}
    ]

    m_a = stage6_shelf_metrics.evaluate_shelf_metrics(
        total_boxes=len(shelf_fixture_a.boxes),
        box_f2=0.96,
        box_recall=0.97,
        p95_latency_s=1.8,
        cost_per_image_inr=0.04,
        attribute_accuracy={"compound": 0.97, "variant": 0.95, "all_7dim": 0.94},
        rows=rows_a,
        stage_overrides={"shelf_metrics": "dual_mt_marketshare_and_merchandising"},
    )
    m_b = stage6_shelf_metrics.evaluate_shelf_metrics(
        total_boxes=6,
        box_f2=0.91,
        box_recall=0.92,
        p95_latency_s=2.1,
        cost_per_image_inr=0.05,
        attribute_accuracy={"compound": 0.90, "variant": 0.88, "all_7dim": 0.87},
        rows=rows_b,
        stage_overrides={"shelf_metrics": "panorama_6img_stitch_sos"},
    )
    m_oos = stage6_shelf_metrics.evaluate_shelf_metrics(
        total_boxes=6,
        box_f2=0.91,
        box_recall=0.92,
        rows=rows_b,
        stage_overrides={"shelf_metrics": "single_img_planogram_oos"},
    )

    assert m_a["hul_7dim_sku_f2"] != m_b["hul_7dim_sku_f2"]
    assert m_a["linear_sos_pct"] > 0.0 and m_a["area_sos_pct"] > 0.0
    assert "mt_market_share_kpis" in m_a and "mt_merchandising_kpis" in m_oos
