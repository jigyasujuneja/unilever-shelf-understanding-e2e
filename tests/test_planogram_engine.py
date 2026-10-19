"""Unit tests for Stage 6 Planogram Compliance Engine (`planogram_engine`) and `promo_asset_sku_detect`."""

from __future__ import annotations

import json

from PIL import Image

import approaches
from approaches.base import Context
from utils import planogram_engine
from utils.llm import LLMResult, Usage


def _catalog() -> dict[int, dict]:
    return {
        1: {"sku_id": 1, "brand": "Dove", "product": "Dove__Bar", "is_hul": True},
        2: {"sku_id": 2, "brand": "Lux", "product": "Lux__Bar", "is_hul": True},
        3: {"sku_id": 3, "brand": "Sunsilk", "product": "Sunsilk__Bottle", "is_hul": True},
        4: {"sku_id": 4, "brand": "Colgate", "product": "Colgate__Paste", "is_hul": False},
    }


def test_cluster_shelf_planks_separates_rows_and_orders_left_to_right():
    boxes = [
        (200.0, 120.0, 280.0, 210.0),  # Row 1, right
        (20.0, 10.0, 90.0, 95.0),      # Row 0, left
        (100.0, 125.0, 180.0, 212.0),  # Row 1, left
        (100.0, 12.0, 170.0, 96.0),    # Row 0, right
    ]
    planks = planogram_engine.cluster_shelf_planks(boxes)
    assert planks == [[1, 3], [2, 0]]


def test_multirow_brand_purity_does_not_penalize_vertical_blocks_across_rows():
    cat = _catalog()
    # Row 0: [Dove, Dove, Lux]
    # Row 1: [Dove, Dove, Lux] directly below Row 0
    # Flat 1D concatenation would see [Dove, Dove, Lux, Dove, Dove, Lux] and falsely penalize!
    boxes = [
        (0.0, 0.0, 80.0, 90.0),
        (85.0, 0.0, 165.0, 90.0),
        (170.0, 0.0, 250.0, 90.0),
        (0.0, 120.0, 80.0, 210.0),
        (85.0, 120.0, 165.0, 210.0),
        (170.0, 120.0, 250.0, 210.0),
    ]
    res = planogram_engine.evaluate_shelf_planogram(boxes, [1, 1, 2, 1, 1, 2], cat)
    assert res["shelf_plank_count"] == 2
    assert res["brand_block_purity"] == 1.0
    assert res["vertical_block_alignment"] == 1.0


def test_vertical_block_alignment_detects_misaligned_adjacent_rows():
    cat = _catalog()
    # Row 0: Dove on far left (x=0..80), Lux on right (x=200..280)
    # Row 1: Lux on far left (x=0..80), Dove on right (x=200..280) -> 0% horizontal overlap!
    boxes = [
        (0.0, 0.0, 80.0, 90.0),
        (200.0, 0.0, 280.0, 90.0),
        (0.0, 120.0, 80.0, 210.0),
        (200.0, 120.0, 280.0, 210.0),
    ]
    res = planogram_engine.evaluate_shelf_planogram(boxes, [1, 2, 2, 1], cat)
    assert res["brand_block_purity"] == 1.0
    assert res["vertical_block_alignment"] == 0.0


def test_per_row_oos_voids_handles_mixed_pack_widths_and_bay_edges():
    cat = _catalog()
    # Row 0 has narrow 40px sachets with an 80px middle gap (2x row_med_w=40 -> 1 inter-pack OOS void)
    # Row 1 has wide 150px bottles with no gap
    # Global median width would be inflated by Row 1, but per-row median width catches Row 0's void!
    boxes = [
        (50.0, 10.0, 90.0, 70.0),     # Row 0 sachet 1 (w=40)
        (95.0, 10.0, 135.0, 70.0),    # Row 0 sachet 2 (w=40)
        (215.0, 10.0, 255.0, 70.0),   # Row 0 sachet 3 (w=40, gap=80 from 135..215)
        (50.0, 100.0, 200.0, 250.0),  # Row 1 bottle 1 (w=150)
        (205.0, 100.0, 355.0, 250.0), # Row 1 bottle 2 (w=150)
    ]
    # Bay bounds from x=0 to x=360 -> Row 0 also has a left-edge gap (50px < 1.5*40=60, no) and right-edge gap (360-255=105px >= 60 -> 1 bay_edge void!)
    bay = [(0.0, 0.0, 360.0, 260.0)]
    res = planogram_engine.evaluate_shelf_planogram(
        boxes, [1, 1, 1, 3, 3], cat, asset_boxes=bay
    )
    assert res["scope_mode"] == "class0_bay_scoped"
    assert res["oos_void_count"] == 2
    kinds = [v["kind"] for v in res["oos_void_boxes"]]
    assert "inter_pack" in kinds
    assert "bay_edge" in kinds
    inter = next(v for v in res["oos_void_boxes"] if v["kind"] == "inter_pack")
    assert inter["box"] == [135.0, 10.0, 215.0, 70.0]
    assert inter["missing_facings"] == 2


def test_bay_scope_router_vs_hanger_fallback():
    cat = _catalog()
    # 2 HUL packs inside Class 0 bay, plus 2 competitor packs outside the bay on the open counter
    boxes = [
        (20.0, 20.0, 80.0, 100.0),    # inside bay (Dove)
        (85.0, 20.0, 145.0, 100.0),   # inside bay (Dove)
        (300.0, 20.0, 360.0, 100.0),  # outside bay (Colgate)
        (370.0, 20.0, 430.0, 100.0),  # outside bay (Colgate)
    ]
    bay = [(0.0, 0.0, 200.0, 120.0)]

    # With Class 0 bay: scores ONLY SKUs inside the bay (2 HUL facings, 100% SOS, 0 intrusions)
    scoped = planogram_engine.evaluate_shelf_planogram(
        boxes, [1, 1, 4, 4], cat, asset_boxes=bay
    )
    assert scoped["scope_mode"] == "class0_bay_scoped"
    assert scoped["total_facings"] == 2
    assert scoped["full_shelf_facings"] == 4
    assert scoped["hul_facing_sos"] == 1.0
    assert scoped["competitor_intrusion_count"] == 0
    assert scoped["asset_bay_purity"] == 1.0

    # Without Class 0 bay (hanger / open shelf fallback): scores full image (4 facings, 50% SOS)
    fallback = planogram_engine.evaluate_shelf_planogram(
        boxes, [1, 1, 4, 4], cat, asset_boxes=[]
    )
    assert fallback["scope_mode"] == "full_image_hanger_fallback"
    assert fallback["total_facings"] == 4
    assert fallback["hul_facing_sos"] == 0.5


def test_check_planogram_spec_audits_outlet_plg_rules():
    cat = _catalog()
    boxes = [
        (0.0, 0.0, 80.0, 90.0),
        (90.0, 0.0, 170.0, 90.0),
        (300.0, 0.0, 380.0, 90.0),  # OOS gap (130px) + Colgate competitor intrusion
    ]
    bay = [(0.0, 0.0, 400.0, 100.0)]
    spec = {
        "spec_id": "PLG_SKIN_01",
        "required_brands": ["Dove", "Lux"],
        "min_hul_sos": 0.80,
        "forbid_competitors_in_bay": True,
        "max_oos_voids": 0,
    }
    res = planogram_engine.evaluate_shelf_planogram(
        boxes, [1, 1, 4], cat, asset_boxes=bay, planogram_spec=spec
    )
    sc = res["spec_compliance"]
    assert sc["has_spec"] is True
    assert sc["spec_id"] == "PLG_SKIN_01"
    assert sc["is_compliant"] is False
    assert any(v.startswith("missing_required_brand:Lux") for v in sc["violations"])
    assert any(v.startswith("hul_sos_below_target:") for v in sc["violations"])
    assert any(v.startswith("competitor_intrusion_in_bay:") for v in sc["violations"])
    assert any(v.startswith("oos_voids_exceed_limit:") for v in sc["violations"])


def test_promo_asset_sku_detect_separates_class0_bays_and_class1_skus():
    reg = approaches.all_approaches()
    assert "promo_asset_sku_detect" in reg
    appr = reg["promo_asset_sku_detect"]
    assert appr.use_case == "merchandising"
    assert appr.task == "detection"
    assert appr.dataset == "gtmerch_golden"

    fake_payload = [
        {"box_2d": [50, 50, 950, 950], "label": "promo_asset"},
        {"box_2d": [100, 100, 300, 250], "label": "product_sku"},
        {"box_2d": [100, 270, 300, 420], "label": "product_sku"},
        {"box_2d": [960, 100, 990, 200], "label": "not_a_product"},
    ]

    def fake_llm(image, prompt, **kw):
        return LLMResult(
            text=json.dumps(fake_payload),
            data=fake_payload,
            usage=Usage(input_tokens=100, output_tokens=50, calls=1),
            seconds=0.25,
        )

    img = Image.new("RGB", (1000, 1000), (240, 240, 240))
    ctx = Context(model="gemini-3.5-flash-lite", llm=fake_llm)
    sku_boxes = appr.detect(img, ctx)

    # Class 0 bay is saved to ctx.detected_asset_boxes and does NOT get dropped as a container
    assert len(ctx.detected_asset_boxes) == 1
    assert ctx.detected_asset_boxes[0] == (50.0, 50.0, 950.0, 950.0)
    # Class 1 SKUs inside the bay are returned cleanly (and not_a_product is filtered out)
    assert len(sku_boxes) == 2


def test_score_merchandising_bay_or_hanger_filters_outside_bay_and_keeps_hanger_fallback():
    from utils import metrics

    cat = _catalog()
    # 2 GT boxes inside the Class 0 bay (0..200, 0..200)
    gt_boxes = [
        (20.0, 20.0, 80.0, 100.0),
        (90.0, 20.0, 150.0, 100.0),
    ]
    # 2 exact predictions inside the bay + 2 background predictions outside the bay (x=300..450)
    pred_boxes = [
        (20.0, 20.0, 80.0, 100.0),
        (90.0, 20.0, 150.0, 100.0),
        (300.0, 20.0, 360.0, 100.0),
        (370.0, 20.0, 430.0, 100.0),
    ]
    pred_ids = [1, 2, 4, 4]
    gt_labels = [{"sku_id": 1, "brand": "Dove"}, {"sku_id": 2, "brand": "Lux"}]
    bay = [(0.0, 0.0, 200.0, 200.0)]

    kept_b, kept_ids, kept_idx, mode = metrics.filter_preds_to_asset_bays(
        pred_boxes, pred_ids=pred_ids, asset_boxes=bay
    )
    assert len(kept_b) == 2
    assert kept_ids == [1, 2]
    assert kept_idx == [0, 1]
    assert mode == "class0_bay_scoped"

    # Case A: With Class 0 bay -> outside predictions are not penalized as false positives (F2 = 1.0)
    bay_res = metrics.score_merchandising_bay_or_hanger(
        preds=pred_boxes,
        pred_ids=pred_ids,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        asset_boxes=bay,
        catalog=cat,
        fields=("brand",),
    )
    assert bay_res["scope_mode"] == "class0_bay_scoped"
    assert bay_res["asset_bay_count"] == 1
    assert bay_res["scoped_pred_count"] == 2
    assert bay_res["scoped_gt_count"] == 2
    assert bay_res["tp"] == 2
    assert bay_res["fp"] == 0
    assert bay_res["f2"] == 1.0

    # Case B: No Class 0 bay (hanger / open shelf fallback) -> all 4 predictions are scored (2 TP, 2 FP)
    hanger_res = metrics.score_merchandising_bay_or_hanger(
        preds=pred_boxes,
        pred_ids=pred_ids,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        asset_boxes=[],
        catalog=cat,
        fields=("brand",),
    )
    assert hanger_res["scope_mode"] == "full_image_hanger_fallback"
    assert hanger_res["asset_bay_count"] == 0
    assert hanger_res["scoped_pred_count"] == 4
    assert hanger_res["scoped_gt_count"] == 2
    assert hanger_res["tp"] == 2
    assert hanger_res["fp"] == 2
    assert hanger_res["precision"] == 0.5


def test_build_unified_brand_sft_jsonl_excludes_test_and_session_leakage(tmp_path):
    from utils.gemini_fine_tuning.tuning import build_unified_brand_sft_jsonl

    fake_manifest = {
        "classes": {
            "1": {"brand": "Dove", "is_hul": True},
            "2": {"brand": "Colgate", "is_hul": False},
            "3": {"brand": "Others", "is_hul": False},
        },
        "gallery": {
            "1": [
                "gs://bkt/gal/dove_0.jpg",
                "gs://bkt/gal/dove_leak_test.jpg",
            ],
            "2": ["gs://bkt/gal/colgate_0.jpg"],
        },
        "splits": {
            "train": [
                {"uri": "gs://bkt/tr/sess1_001.jpg", "session_id": "sess_train_1", "products": [1]},
                {"uri": "gs://bkt/tr/sess_val_leak.jpg", "session_id": "sess_heldout_val", "products": [1]},
                {"uri": "gs://bkt/tr/others_001.jpg", "session_id": "sess_train_2", "products": [3]},
            ],
            "val": [
                {"uri": "gs://bkt/val/dove_val.jpg", "session_id": "sess_heldout_val", "products": [1]},
            ],
            "test": [
                {"uri": "gs://bkt/gal/dove_leak_test.jpg", "session_id": "sess_test_1", "products": [1]},
            ],
            "test_5k": [
                {"uri": "gs://bkt/t5k/colgate_t5k.jpg", "session_id": "sess_test_5k", "products": [2]},
            ],
        },
    }

    res = build_unified_brand_sft_jsonl(
        out_dir=tmp_path,
        upload=False,
        manifests=[fake_manifest],
    )
    assert res["class_count"] == 3
    assert res["hul_brand_count"] == 1
    assert res["non_hul_brand_count"] == 2

    train_lines = (tmp_path / "train_stratified.jsonl").read_text(encoding="utf-8").splitlines()
    val_lines = (tmp_path / "val_stratified.jsonl").read_text(encoding="utf-8").splitlines()
    train_uris = {json.loads(ln)["contents"][0]["parts"][0]["fileData"]["fileUri"] for ln in train_lines}
    val_uris = {json.loads(ln)["contents"][0]["parts"][0]["fileData"]["fileUri"] for ln in val_lines}

    # Strict URI disjointness: zero test_leak URI, zero session-leaked train URI, zero train/val overlap
    assert "gs://bkt/gal/dove_leak_test.jpg" not in train_uris
    assert "gs://bkt/tr/sess_val_leak.jpg" not in train_uris
    assert not (train_uris & val_uris)
    assert "gs://bkt/tr/others_001.jpg" in train_uris

    # Verify JSON target schema has {"brand": ..., "is_hul": ...}
    targets = [json.loads(json.loads(ln)["contents"][1]["parts"][0]["text"]) for ln in train_lines]
    by_b = {t["brand"]: t["is_hul"] for t in targets}
    assert by_b["Dove"] is True
    assert by_b["Colgate"] is False
    assert by_b["Others"] is False

