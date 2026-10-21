"""Unit tests for EXP-50..53 generalizable, non-overfitting primitives and approaches."""

from __future__ import annotations

from PIL import Image, ImageDraw

import approaches
from approaches.merchandising.promo_detection.promo_asset_sku_detect import (
    filter_plank_support_and_shoulders,
)
from utils.clm import (
    apply_lidstone_pack_fusion,
    apply_projective_scale_prior,
    compute_container_geometry_features,
    compute_reference_diff_band,
    letterbox_cap_crop,
    lidstone_pack_posterior,
    zoom_sheet_saliency_diff,
)
from utils.experiment_matrix import APPROACH_INTUITIONS, POC_EXPERIMENTS


def _make_bottle_img(width: int = 80, height: int = 200) -> Image.Image:
    """Create a synthetic bottle image with a narrow top neck and wide body."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Narrow cap/neck in top 22%
    draw.rectangle([width // 2 - 10, 5, width // 2 + 10, int(height * 0.22)], fill=(20, 60, 180))
    # Wide bottle body in middle
    draw.rectangle([8, int(height * 0.22), width - 8, height - 10], fill=(30, 90, 200))
    return img


def test_exp50_letterbox_cap_crop_and_geometry_features():
    shelf = Image.new("RGB", (300, 400), (240, 240, 240))
    padded = letterbox_cap_crop(shelf, (100.0, 100.0, 180.0, 300.0), top_pad=0.12, bottom_pad=0.08)
    assert padded.width > 80
    assert padded.height > 200

    img = _make_bottle_img(80, 200)
    feats = compute_container_geometry_features(img, box=(100.0, 100.0, 180.0, 300.0), plank_height=200.0)
    assert feats["aspect_ratio"] == 2.5
    assert 0.25 <= feats["crown_taper"] <= 1.35
    assert abs(feats["plank_height_ratio"] - 1.0) < 0.05


def test_exp50_projective_scale_prior_penalizes_aspect_mismatch():
    query_img = _make_bottle_img(80, 200)  # tall bottle: h/w = 2.5
    q_feats = compute_container_geometry_features(query_img)
    ranked = [
        (0.855, 101, "gs://ref/tub_101.jpg"),
        (0.850, 202, "gs://ref/bottle_202.jpg"),
    ]
    catalog = {
        101: {"sku_id": 101, "packaging_type": "Jar"},
        202: {"sku_id": 202, "packaging_type": "Bottle"},
    }
    reranked = apply_projective_scale_prior(ranked, q_feats, catalog, weight=0.018)
    assert reranked[0][1] == 202
    assert reranked[0][0] > reranked[1][0]


def test_exp51_self_supervised_reference_diff_band():
    # Two reference packshots that are identical at top & bottom, but differ strongly around y=0.50
    ref1 = Image.new("RGB", (128, 128), (200, 200, 200))
    ref2 = Image.new("RGB", (128, 128), (200, 200, 200))
    d1 = ImageDraw.Draw(ref1)
    d2 = ImageDraw.Draw(ref2)
    d1.rectangle([10, 56, 118, 76], fill=(240, 20, 20))
    d2.rectangle([10, 56, 118, 76], fill=(20, 20, 240))

    y1, y2, peak_ratio = compute_reference_diff_band([ref1, ref2], band_frac=0.25)
    band_center = 0.5 * (y1 + y2)
    assert 0.35 <= band_center <= 0.68
    assert peak_ratio > 1.0

    sheet = zoom_sheet_saliency_diff(ref1, [ref1, ref2], include_crown=True)
    assert sheet.width > 300
    assert sheet.height > 200


def test_exp52_lidstone_pack_posterior_and_fusion():
    query_img = _make_bottle_img(80, 220)
    q_feats = compute_container_geometry_features(query_img)
    ranked = [
        (0.88, 1, "gs://ref/s1.jpg"),
        (0.87, 2, "gs://ref/s2.jpg"),
        (0.80, 3, "gs://ref/s3.jpg"),
    ]
    catalog = {
        1: {"sku_id": 1, "packaging_type": "Spray"},
        2: {"sku_id": 2, "packaging_type": "Bottle"},
        3: {"sku_id": 3, "packaging_type": "Jar"},
    }
    post = lidstone_pack_posterior(ranked, q_feats, catalog, eps=0.01)
    assert len(post) == 3
    assert all(p > 0.0 for p in post.values())

    fused, pack_probs = apply_lidstone_pack_fusion(ranked, q_feats, catalog, eps=0.01)
    assert len(fused) == 3
    assert len(pack_probs) == 3


def test_exp53_physical_plank_support_and_shoulder_filter():
    # 4 products resting on a horizontal shelf plank at y_max ~= 400,
    # plus 1 back-row shoulder slice above item 0, and 1 floating banner noise box in the top header zone.
    bay = (50.0, 50.0, 950.0, 900.0)
    plank_items = [
        (100.0, 240.0, 180.0, 400.0),
        (200.0, 235.0, 280.0, 402.0),
        (300.0, 242.0, 380.0, 398.0),
        (400.0, 238.0, 480.0, 401.0),
    ]
    # Upper shelf tier of 2 items sitting above plank_items (y_max = 220 < fy1 = 240)
    upper_tier_items = [
        (100.0, 150.0, 180.0, 220.0),
        (200.0, 150.0, 280.0, 221.0),
    ]
    # Back-row depth shoulder peeking just above plank_items[0] (y2 = 255 inside [240, 307])
    back_row_shoulder = (105.0, 205.0, 175.0, 255.0)
    # Floating graphic/header inside the bay far from any plank
    floating_fp = (600.0, 90.0, 690.0, 140.0)

    filtered, removed = filter_plank_support_and_shoulders(
        plank_items + upper_tier_items + [back_row_shoulder, floating_fp],
        asset_boxes=[bay],
        fixture_types=["WIN"],
    )
    assert len(filtered) == 6
    assert len(removed) == 2
    for b in plank_items + upper_tier_items:
        assert b in filtered


def test_exp50_to_53_registered_in_registry_and_matrix():
    all_ap = approaches.all_approaches()
    expected_approaches = [
        "clef_aspect_scale_prior",
        "clef_saliency_diff_zoom",
        "clef_pack_crown_fusion",
        "hybrid_pack_specialist_clef",
        "promo_asset_plank_tiling_detect",
        "shelf_plank_tiling_e2e",
        "shelf_generalizable_spine_e2e",
    ]
    for name in expected_approaches:
        assert name in all_ap, f"{name} missing from approaches.all_approaches()"
        assert name in APPROACH_INTUITIONS, f"{name} missing from APPROACH_INTUITIONS"

    exp_ids = {exp["experiment_id"] for exp in POC_EXPERIMENTS}
    for exp_id in ("P0-EXP-50", "P0-EXP-51", "P0-EXP-52", "P0-EXP-53"):
        assert exp_id in exp_ids, f"{exp_id} missing from POC_EXPERIMENTS"
