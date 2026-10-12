"""Unit tests for the standalone legacy mapping table, dual-accept grading, and JSONL split exports."""

from __future__ import annotations

import json
from pathlib import Path

from utils import dataset, metrics
from utils.dataset import Sample, export_rpc_splits_jsonl, load_split_jsonl, parse_split_jsonl
from utils.legacy_benchmark import (
    build_legacy_mapping_table,
    evaluate_joined_legacy_models,
    export_legacy_mapping_jsonl,
    normalize_legacy_prediction,
    resolve_legacy_brand_name,
    resolve_legacy_pack_folder,
)
from utils.taxonomy.filenames import normalize_brand_folder, parse_variant_folder
from utils.taxonomy.packaging import parse_pack_folder


def test_resolve_legacy_brand_and_pack_in_standalone_table(tmp_path: Path) -> None:
    """Verify legacy brand/pack mappings live in the standalone table while core taxonomy stays clean."""
    # 1. Core taxonomy dicts stay 100% clean (no legacy fields mixed in)
    b_clinic = normalize_brand_folder("AClinic")
    assert b_clinic is not None
    assert "legacy_brand_name" not in b_clinic
    v_clinic = parse_variant_folder("variant_hair", "Clinic_Plus__Strong_Long_Bottle")
    assert "legacy_brand_name" not in v_clinic
    assert "legacy_pack_folder" not in v_clinic
    p_hair = parse_pack_folder("Hair__Bottle")
    assert "legacy_pack_folder" not in p_hair

    # 2. Standalone mapping functions resolve irregular HUL, regular HUL, and Non-HUL brands
    assert resolve_legacy_brand_name("Clinic_Plus", is_hul=True) == "AClinic"
    assert resolve_legacy_brand_name("3_Roses", is_hul=True) == "A_3_Roses"
    assert resolve_legacy_brand_name("Brooke_Bond", is_hul=True) == "ABrook_Bond"
    assert resolve_legacy_brand_name("Dove", is_hul=True) == "ADove"
    assert resolve_legacy_brand_name("Dr_Wash", is_hul=True) == "ADr_Wash"
    assert resolve_legacy_brand_name("Pantene", is_hul=False) == "Pantene"
    assert resolve_legacy_brand_name("Colgate", is_hul=False) == "Colgate"

    assert resolve_legacy_pack_folder("Hair_Care", "Bottle") == "Hair__Bottle"
    assert resolve_legacy_pack_folder("Skin_Cleansing", "Bar") == "PW__Wrapper"
    assert resolve_legacy_pack_folder("DMT", "Roll_On") == "DMT__Bottle"
    assert resolve_legacy_pack_folder("Oral_Care", "Tube") == "Oral_Care__Tube"

    # 3. Export standalone mapping table as JSONL (one JSON object per line)
    jsonl_path = export_legacy_mapping_jsonl(
        tmp_path / "legacy_mapping_table.jsonl",
        extra_brands=[{"brand": "Pantene", "is_hul": False, "category": "Hair_Care"}],
    )
    rows = parse_split_jsonl(jsonl_path.read_text())
    table = build_legacy_mapping_table(
        extra_brands=[{"brand": "Pantene", "is_hul": False, "category": "Hair_Care"}],
    )
    assert len(table["category_packs"]) == 46
    assert len(rows) == len(table["brands"]) + len(table["category_packs"])
    clinic_row = next(r for r in rows if r.get("canonical_brand") == "Clinic_Plus")
    assert clinic_row["legacy_brand_name"] == "AClinic"
    pantene_row = next(r for r in rows if r.get("canonical_brand") == "Pantene")
    assert pantene_row["legacy_brand_name"] == "Pantene"


def test_dual_accept_grading_on_clean_ground_truth() -> None:
    """Verify metrics.match_product accepts both canonical and legacy outputs against clean GT dicts."""
    gt_clinic = {
        "product": "Clinic_Plus__Strong_Long",
        "brand": "Clinic_Plus",
        "variant": "Strong_Long",
        "base_variant": "Strong_Long",
        "category": "Hair_Care",
        "packaging_type": "Bottle",
        "is_hul": True,
    }
    fields = ("product", "brand", "base_variant", "category", "packaging_type", "is_hul")

    # 1. Modern unified model prediction (clean Brand / Category / PackType)
    modern_pred = {
        "product": "Clinic_Plus__Strong_Long",
        "brand": "Clinic_Plus",
        "base_variant": "Strong_Long",
        "category": "Hair_Care",
        "packaging_type": "Bottle",
        "is_hul": True,
    }
    m_mod = metrics.match_product(modern_pred, gt_clinic, fields)
    assert all(m_mod["correct"].values())

    # 2. Legacy model raw prediction (ABrand + Category__PackType)
    legacy_raw_pred = {
        "product": "AClinic__Strong_Long",
        "brand": "AClinic",
        "base_variant": "Strong_Long",
        "category": "Hair__Bottle",
        "packaging_type": "Hair__Bottle",
        "is_hul": True,
    }
    m_leg = metrics.match_product(legacy_raw_pred, gt_clinic, fields)
    assert all(m_leg["correct"].values())

    # 3. Wrong brand must still fail
    wrong_pred = {
        "product": "ADove__Strong_Long",
        "brand": "ADove",
        "base_variant": "Strong_Long",
        "category": "Hair__Bottle",
        "packaging_type": "Hair__Bottle",
        "is_hul": True,
    }
    m_wrong = metrics.match_product(wrong_pred, gt_clinic, fields)
    assert m_wrong["correct"]["brand"] is False
    assert m_wrong["correct"]["category"] is True
    assert m_wrong["correct"]["packaging_type"] is True


def test_joined_legacy_models_compound_accuracy() -> None:
    """Verify joining legacy Brand + Pack models computes individual & compound accuracy."""
    samples = [
        Sample(
            image_id="val_00001.jpg",
            path="val_00001.jpg",
            width=256,
            height=256,
            boxes=[(0.0, 0.0, 256.0, 256.0)],
            labels=[
                {
                    "product": "Dove__Cream_Beauty_Bar",
                    "brand": "Dove",
                    "base_variant": "Cream_Beauty_Bar",
                    "category": "Skin_Cleansing",
                    "packaging_type": "Bar",
                    "is_hul": True,
                }
            ],
        ),
        Sample(
            image_id="val_00002.jpg",
            path="val_00002.jpg",
            width=256,
            height=256,
            boxes=[(0.0, 0.0, 256.0, 256.0)],
            labels=[
                {
                    "product": "Clinic_Plus__Strong_Long",
                    "brand": "Clinic_Plus",
                    "base_variant": "Strong_Long",
                    "category": "Hair_Care",
                    "packaging_type": "Bottle",
                    "is_hul": True,
                }
            ],
        ),
        Sample(
            image_id="val_00003.jpg",
            path="val_00003.jpg",
            width=256,
            height=256,
            boxes=[(0.0, 0.0, 256.0, 256.0)],
            labels=[
                {
                    "product": "Pantene__Hair_Fall_Control",
                    "brand": "Pantene",
                    "base_variant": "Hair_Fall_Control",
                    "category": "Hair_Care",
                    "packaging_type": "Bottle",
                    "is_hul": False,
                }
            ],
        ),
    ]

    brand_preds = {
        "val_00001.jpg": "ADove",
        "val_00002.jpg": "AClinic",
        "val_00003.jpg": "Pantene",
    }
    pack_preds = {
        "val_00001.jpg": "PW__Bar",
        "val_00002.jpg": "Hair__Bottle",
        "val_00003.jpg": "Hair__Pouch",  # Wrong pack type on 3rd crop
    }

    res = evaluate_joined_legacy_models(
        samples=samples,
        brand_preds=brand_preds,
        pack_preds=pack_preds,
    )
    assert res["evaluated_images"] == 3
    assert res["evaluated_boxes"] == 3
    assert res["field_accuracy"]["brand"] == 1.0
    assert res["field_accuracy"]["category"] == 1.0
    assert res["field_accuracy"]["packaging_type"] == round(2 / 3, 4)
    assert res["compound_brand_category_pack"] == round(2 / 3, 4)
    assert res["hierarchical_accuracy"]["compound_brand_category_pack"] == round(2 / 3, 4)

    norm = normalize_legacy_prediction(brand_pred="A_3_Roses", pack_pred="TEA__Carton")
    assert norm["brand"] == "3_Roses"
    assert norm["legacy_brand_name"] == "A_3_Roses"
    assert norm["category"] == "Beverages"
    assert norm["packaging_type"] == "Carton"


def test_export_and_load_splits_jsonl(tmp_path: Path) -> None:
    """Verify splits are exported and loaded as line-by-line JSONL (train.jsonl, val.jsonl, test.jsonl)."""
    manifest = {
        "classes": {
            "1": {
                "sku_id": 1,
                "product": "Dove__Cream_Beauty_Bar",
                "brand": "Dove",
                "variant": "Cream_Beauty_Bar",
                "base_variant": "Cream_Beauty_Bar",
                "category": "Skin_Cleansing",
                "packaging_type": "Bar",
                "is_hul": True,
            }
        },
        "gallery": {"1": ["gallery/1_0.jpg"]},
        "splits": {
            "train": [
                {
                    "image": "train_00000.jpg",
                    "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Dove__Cream_Beauty_Bar/img1.jpg",
                    "width": 256,
                    "height": 256,
                    "boxes": [[0.0, 0.0, 256.0, 256.0]],
                    "products": [1],
                    "labels": [{"sku_id": 1, "brand": "Dove", "category": "Skin_Cleansing", "packaging_type": "Bar"}],
                }
            ],
            "val": [
                {
                    "image": "val_00000.jpg",
                    "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Dove__Cream_Beauty_Bar/img2.jpg",
                    "width": 256,
                    "height": 256,
                    "boxes": [[0.0, 0.0, 256.0, 256.0]],
                    "products": [1],
                    "labels": [{"sku_id": 1, "brand": "Dove", "category": "Skin_Cleansing", "packaging_type": "Bar"}],
                }
            ],
            "test": [
                {
                    "image": "test_00000.jpg",
                    "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Dove__Cream_Beauty_Bar/img3.jpg",
                    "width": 256,
                    "height": 256,
                    "boxes": [[0.0, 0.0, 256.0, 256.0]],
                    "products": [1],
                    "labels": [{"sku_id": 1, "brand": "Dove", "category": "Skin_Cleansing", "packaging_type": "Bar"}],
                }
            ],
        },
    }
    (tmp_path / "rpc.json").write_text(json.dumps(manifest))
    paths = export_rpc_splits_jsonl(manifest, tmp_path)
    assert set(paths.keys()) == {"train", "val", "test"}

    for sp in ("train", "val", "test"):
        p = Path(paths[sp])
        assert p.exists()
        assert p.suffix == ".jsonl"
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["image"] == f"{sp}_00000.jpg"

        loaded = load_split_jsonl(p, catalog=dataset.rpc_catalog(str(tmp_path)), root=str(tmp_path))
        assert f"{sp}_00000.jpg" in loaded
        assert loaded[f"{sp}_00000.jpg"].labels[0]["brand"] == "Dove"
