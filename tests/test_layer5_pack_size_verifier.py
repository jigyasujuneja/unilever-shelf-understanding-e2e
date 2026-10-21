"""Unit tests for Layer-5 post-generation deterministic pack_size sister-SKU verifier (PR-14)."""

from __future__ import annotations

from approaches.market_share.retrieval.clef_standalone import (
    normalize_pack_size_token,
    resolve_clef_choice,
    verify_pack_size_sister_sku,
)


def test_normalize_pack_size_token_canonicalizes_units():
    assert normalize_pack_size_token("100 g") == "100g"
    assert normalize_pack_size_token("100gm") == "100g"
    assert normalize_pack_size_token("100 Gms") == "100g"
    assert normalize_pack_size_token("0.5 kg") == "500g"
    assert normalize_pack_size_token("1 L") == "1000ml"
    assert normalize_pack_size_token("250 ML") == "250ml"
    assert normalize_pack_size_token("ADove__Hair_Fall_Rescue__180ml") == "180ml"
    assert normalize_pack_size_token("") == ""
    assert normalize_pack_size_token(None) == ""


def test_verify_pack_size_sister_sku_overrides_same_brand_same_variant_sister():
    catalog = {
        10: {
            "sku_id": 10,
            "brand": "Dove",
            "base_variant": "Cream Beauty Bathing Bar",
            "pack_size": "50g",
            "product": "Dove Cream Beauty Bathing Bar 50g",
        },
        11: {
            "sku_id": 11,
            "brand": "Dove",
            "base_variant": "Cream Beauty Bathing Bar",
            "pack_size": "100g",
            "product": "Dove Cream Beauty Bathing Bar 100g",
        },
        12: {
            "sku_id": 12,
            "brand": "Lux",
            "base_variant": "Soft Touch",
            "pack_size": "100g",
            "product": "Lux Soft Touch 100g",
        },
    }
    ranked_candidates = [(0.86, 10), (0.83, 11), (0.85, 12)]

    # VLM picked 10 (50g) but extracted visible pack_size "100 gm" -> overrides to sister SKU 11 (100g)
    verified_id, overridden = verify_pack_size_sister_sku(
        10, "100 gm", ranked_candidates, catalog
    )
    assert overridden is True
    assert verified_id == 11

    # When extracted size already matches chosen SKU (50g), no override occurs
    same_id, overridden_same = verify_pack_size_sister_sku(
        10, "50g", ranked_candidates, catalog
    )
    assert overridden_same is False
    assert same_id == 10

    # Does not cross-override to a different brand (Lux 100g) if sister Dove 100g is absent
    only_diff_brand = [(0.86, 10), (0.85, 12)]
    kept_id, overridden_diff = verify_pack_size_sister_sku(
        10, "100g", only_diff_brand, catalog
    )
    assert overridden_diff is False
    assert kept_id == 10


def test_resolve_clef_choice_integrates_layer5_pack_size_verifier():
    top = [
        (0.87, 101, "gs://ref/dove_50g.jpg"),
        (0.84, 102, "gs://ref/dove_100g.jpg"),
    ]
    catalog = {
        101: {"brand": "Dove", "base_variant": "Intense Repair", "pack_size": "50g"},
        102: {"brand": "Dove", "base_variant": "Intense Repair", "pack_size": "100g"},
    }
    # Panel 1 (101, 50g) selected by choice=1, but extracted pack_size='100g' matches Panel 2 (102, 100g)
    ans, mode = resolve_clef_choice(
        1,
        0.95,
        top,
        allow_abstain=False,
        extracted_pack_size="100g",
        catalog=catalog,
    )
    assert ans == 102
    assert mode == "panel_1+layer5_pack_size_verified"


def test_verify_pack_size_sister_sku_discriminative_token_fallback():
    catalog = {
        365: {
            "sku_id": 365,
            "brand": "Dove",
            "base_variant": "Glycolic_Hydration",
            "packaging_type": "Bottle",
            "product": "ADove__Glycolic_Hydration",
        },
        366: {
            "sku_id": 366,
            "brand": "Dove",
            "base_variant": "Glycolic_Hydration_Conditioner",
            "packaging_type": "Bottle",
            "product": "ADove__Glycolic_Hydration_Conditioner",
        },
    }
    ranked = [(0.876, 365, "p1"), (0.855, 366, "p2")]
    ans, overridden = verify_pack_size_sister_sku(365, "Conditioner", ranked, catalog)
    assert overridden is True
    assert ans == 366

