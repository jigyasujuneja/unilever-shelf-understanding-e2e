"""Tests for annotated-row shelf-band grouping, psa_category scoping, and two-bucket enrichment KPIs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils import metrics  # noqa: E402


def test_group_annotated_shelf_bands_merges_same_row_and_splits_other_rows() -> None:
    gt_boxes = [
        (10.0, 100.0, 50.0, 200.0),
        (60.0, 105.0, 100.0, 202.0),
        (15.0, 500.0, 55.0, 600.0),
    ]
    bands = metrics.group_annotated_shelf_bands(gt_boxes)
    assert len(bands) == 2
    assert bands[0][0] < 100.0 and bands[0][1] > 202.0
    assert bands[1][0] < 500.0 and bands[1][1] > 600.0


def test_filter_preds_to_annotated_bands_excludes_unannotated_rows() -> None:
    gt_boxes = [(10.0, 100.0, 50.0, 200.0), (60.0, 105.0, 100.0, 202.0)]
    bands = metrics.group_annotated_shelf_bands(gt_boxes)
    preds = [
        (12.0, 102.0, 48.0, 198.0),  # inside band
        (62.0, 106.0, 98.0, 200.0),  # inside band
        (20.0, 450.0, 70.0, 550.0),  # unannotated lower shelf row
        (15.0, 10.0, 55.0, 50.0),    # unannotated top shelf row
    ]
    pred_ids = [101, 102, 201, 202]
    kept_preds, kept_ids, kept_indices = metrics.filter_preds_to_annotated_bands(preds, pred_ids, bands)
    assert len(kept_preds) == 2
    assert kept_ids == [101, 102]
    assert kept_indices == [0, 1]


def test_compute_enrichment_buckets_preserves_both_buckets() -> None:
    gt_boxes = [(10.0, 100.0, 50.0, 200.0), (60.0, 105.0, 100.0, 202.0)]
    gt_labels = [
        {"sku_id": 101, "brand": "Dove", "category": "Hair_Care", "is_others_catchall": False},
        {"sku_id": 999, "brand": "Others", "category": "Hair_Care", "is_others_catchall": True},
    ]
    preds = [
        (12.0, 102.0, 48.0, 198.0),  # matches GT 0
        (62.0, 106.0, 98.0, 200.0),  # matches GT 1 ('Others' -> resolved to Pantene)
        (20.0, 450.0, 70.0, 550.0),  # unannotated row extra SKU 1
        (80.0, 455.0, 130.0, 552.0), # unannotated row extra SKU 2
    ]
    pred_ids = [101, 301, 401, 402]
    catalog = {
        101: {"sku_id": 101, "brand": "Dove", "category": "Hair_Care", "is_hul": True},
        301: {"sku_id": 301, "brand": "Pantene", "category": "Hair_Care", "is_hul": False},
        401: {"sku_id": 401, "brand": "Colgate", "category": "Oral_Care", "is_hul": False},
        402: {"sku_id": 402, "brand": "Pepsodent", "category": "Oral_Care", "is_hul": True},
    }
    m = metrics.match(preds, gt_boxes)
    enr = metrics.compute_enrichment_buckets(preds, pred_ids, gt_boxes, gt_labels, catalog, m["pairs"])
    assert enr["gt_annotated_facings"] == 2
    assert enr["total_predicted_facings"] == 4
    assert enr["in_annotated_scope_preds"] == 2
    assert enr["extra_unannotated_row_preds"] == 2
    assert enr["extra_discovered_unique_skus"] == 2
    assert enr["extra_discovered_unique_brands"] == 2
    assert enr["others_catchall_gt_count"] == 1
    assert enr["others_catchall_resolved_count"] == 1
    assert enr["enrichment_multiplier"] == 2.0


def test_compute_human_and_beyond_metrics_three_tiers_and_backfilled_hul_parity() -> None:
    gt_boxes = [
        (10.0, 100.0, 50.0, 200.0),   # GT 0: explicitly named target SKU (Dove)
        (60.0, 105.0, 100.0, 202.0),  # GT 1: non-HUL 'Others' catch-all upgraded to Pantene
        (110.0, 104.0, 150.0, 201.0), # GT 2: backfilled HUL catch-all (is_others_catchall=True, is_hul=True)
    ]
    gt_labels = [
        {"sku_id": 101, "product": "Dove Hair Fall Rescue", "brand": "Dove", "category": "Hair_Care", "is_hul": True, "is_others_catchall": False},
        {"sku_id": 999, "product": "Others", "brand": "Others", "category": "Hair_Care", "is_hul": False, "is_others_catchall": True},
        {"sku_id": 102, "product": "Sunsilk Black Shine", "brand": "Sunsilk", "category": "Hair_Care", "is_hul": True, "is_others_catchall": True},
    ]
    preds = [
        (12.0, 102.0, 48.0, 198.0),   # Matches GT 0 -> Dove
        (62.0, 106.0, 98.0, 200.0),   # Matches GT 1 -> Pantene (Non-HUL upgraded from 'Others')
        (112.0, 105.0, 148.0, 199.0), # Matches GT 2 -> Sunsilk (Backfilled HUL catch-all)
        (20.0, 450.0, 70.0, 550.0),   # Unannotated lower shelf row -> Colgate
    ]
    pred_ids = [101, 301, 102, 401]
    catalog = {
        101: {"sku_id": 101, "product": "Dove Hair Fall Rescue", "brand": "Dove", "category": "Hair_Care", "is_hul": True},
        102: {"sku_id": 102, "product": "Sunsilk Black Shine", "brand": "Sunsilk", "category": "Hair_Care", "is_hul": True},
        301: {"sku_id": 301, "product": "Pantene Advanced Hairfall", "brand": "Pantene", "category": "Hair_Care", "is_hul": False},
        401: {"sku_id": 401, "product": "Colgate Strong Teeth", "brand": "Colgate", "category": "Oral_Care", "is_hul": False},
    }
    hb = metrics.compute_human_and_beyond_metrics(preds, pred_ids, gt_boxes, gt_labels, catalog)
    assert hb["tier1_named_target_sku"]["gt_count"] == 1
    assert hb["tier1_named_target_sku"]["matched_correct"] == 1
    assert hb["tier1_named_target_sku"]["f2"] == 1.0
    assert hb["tier2_client_clubbed_parity"]["tp"] == 3
    assert hb["tier2_client_clubbed_parity"]["f2"] == 1.0
    assert hb["tier3_beyond_gt_discovery"]["others_upgraded_to_exact_sku"] == 2
    assert hb["tier3_beyond_gt_discovery"]["unannotated_row_facings_discovered"] == 1
    agg = metrics.aggregate_human_and_beyond_metrics([hb])
    assert agg is not None
    assert agg["tier2_client_clubbed_parity"]["f2"] == 1.0
    assert agg["tier3_beyond_gt_discovery"]["others_upgraded_to_exact_sku"] == 2

