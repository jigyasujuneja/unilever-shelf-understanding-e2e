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


def test_dynamic_methodology_across_all_approach_families() -> None:
    """Verify dynamic methodology synthesis introspects each approach class and never stamps JEV/CLM/CLEF onto V-JEPA 2 or DINOv3."""
    import runner

    # 1. Meta V-JEPA 2 (Run #115616 from user screenshot) — must reflect V-JEPA 2 ViT-L/16 + CosFace head, NEVER JEV/CLM/CLEF/gemini-embedding-2
    vjepa_sum = {
        "run_id": "1017-115616-vjepa2_hierarchical-vjepa2_vitl16-hul_golden-test",
        "short_id": "#115616",
        "approach": "vjepa2_hierarchical",
        "model": "vjepa2_vitl16",
        "dataset": "hul_golden",
        "split": "test",
        "task": "classification",
        "images": 1000,
        "eval_samples": 1000,
        "accuracy": 0.453,
        "three_attr_accuracy": 0.659,
        "p50_latency_s": 0.08,
        "cost_per_image_inr": 0.0002,
        "field_accuracy": {
            "is_hul": 0.870,
            "category": 0.875,
            "brand": 0.779,
            "packaging_type": 0.858,
            "base_variant": 0.506,
            "product": 0.453,
        },
    }
    m_vjepa = runner.synthesize_dynamic_methodology(vjepa_sum)
    assert m_vjepa["generated_dynamically"] is True
    assert "V-JEPA 2" in m_vjepa["summary_header"]
    assert "Standalone Open-Source Vision Backbone" in m_vjepa["origin_badge"]
    assert "Meta V-JEPA 2" in m_vjepa["origin_badge"]
    assert "CLM Gate: 68.0% 0-VLM" not in m_vjepa["summary_header"]
    assert "CLEF Zoom Sheet" not in m_vjepa["summary_header"]
    joined_vjepa_bullets = " ".join(m_vjepa["bullets"])
    assert "V-JEPA 2" in joined_vjepa_bullets or "1024-d" in joined_vjepa_bullets
    assert "JEV (Joint Embedding + VLM Cascade" not in joined_vjepa_bullets
    assert "45.3% 5-Attr Exact SKU" in joined_vjepa_bullets

    # 2. Meta DINOv3 — must reflect DINOv3 + CosFace head
    dino_sum = {
        "run_id": "1015-081200-dinov3_classify-dinov3_vits16-hul_golden-test",
        "short_id": "#081200",
        "approach": "dinov3_classify",
        "model": "dinov3_vits16",
        "dataset": "hul_golden",
        "split": "test",
        "task": "classification",
        "images": 1000,
        "accuracy": 0.562,
        "p50_latency_s": 0.05,
        "cost_per_image_inr": 0.0001,
    }
    m_dino = runner.synthesize_dynamic_methodology(dino_sum)
    assert "Standalone Open-Source Vision Backbone" in m_dino["origin_badge"]
    assert "Meta DINOv3" in m_dino["origin_badge"]
    assert "CLM Gate" not in m_dino["summary_header"]

    # 3. In-House Vertex AI Hybrid Cascade (clef_contrastive_metadata) — must expand internal slugs and include terminology_note + live trace stages
    clef_sum = {
        "run_id": "1016-092456-clef_contrastive_metadata-gemini-3.5-flash-lite-priority-hul_golden-test_5k",
        "short_id": "#092456",
        "approach": "clef_contrastive_metadata",
        "model": "gemini-3.5-flash-lite",
        "dataset": "hul_golden",
        "split": "test_5k",
        "task": "retrieval",
        "images": 5000,
        "eval_samples": 5000,
        "accuracy": 0.823,
        "three_attr_accuracy": 0.916,
        "p50_latency_s": 1.38,
        "cost_per_image_inr": 0.044,
    }
    sample_rows = [
        {
            "image_id": "test_00001.jpg",
            "trace": [
                {"step": "embed_and_gate", "calls": 1, "ms": 75.0, "detail": "gate_accepted=True"},
            ],
        },
        {
            "image_id": "test_00002.jpg",
            "trace": [
                {"step": "embed_and_gate", "calls": 1, "ms": 80.0, "detail": "gate_accepted=False"},
                {"step": "contact_sheet_verify", "calls": 1, "ms": 940.0, "detail": "top-6 sheet"},
            ],
        },
    ]
    m_clef = runner.generate_run_methodology(clef_sum, sample_rows)
    assert "100% In-House Google Cloud Vertex AI Pipeline" in m_clef["origin_badge"]
    assert "Base CLEF Visual Contact-Sheet Technique" in m_clef["origin_badge"]
    assert "Multi-Crop Visual Contact Sheet" in m_clef["summary_header"]
    assert "Disambiguating Base Technique vs. Decision Model" in m_clef["terminology_note"]
    assert "CLM-v0.1-8B" in m_clef["terminology_note"]
    assert "Cloudflare/clef-flash" in m_clef["terminology_note"]
    assert any("Live Execution Trace Telemetry" in b for b in m_clef["bullets"])

    # 4. Decision Model Variant (clef_flash_decision & jev_laya_zscore_lid) — must explicitly distinguish Decision Model from Base Technique
    m_clef_dec = runner.synthesize_dynamic_methodology({
        "run_id": "1015-120000-clef_flash_decision-gemini-3.5-flash-lite-hul_golden-test",
        "approach": "clef_flash_decision",
        "model": "gemini-3.5-flash-lite",
        "dataset": "hul_golden",
        "split": "test",
        "task": "retrieval",
        "accuracy": 0.812,
    })
    assert "CLEF Decision Model Variant" in m_clef_dec["origin_badge"]
    assert "Distinct from Base CLEF Multi-Crop Contact-Sheet Prompt" in m_clef_dec["origin_badge"]


