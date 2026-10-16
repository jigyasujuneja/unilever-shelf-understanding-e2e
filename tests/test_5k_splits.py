from __future__ import annotations

from pathlib import Path

from utils import dataset
from utils.manifests.golden_builder import build_leakage_free_5k_split, compute_split_hashes


def _fixture_manifest() -> dict:
    classes = {
        "1": {
            "sku_id": 1,
            "product": "Dove__Hair_Fall_Rescue",
            "brand": "Dove",
            "variant": "Hair_Fall_Rescue",
            "base_variant": "Hair_Fall_Rescue",
            "category": "Hair_Care",
            "packaging_type": "Bottle",
            "is_hul": True,
            "has_promo": False,
        },
        "2": {
            "sku_id": 2,
            "product": "Pantene__Total_Damage_Care",
            "brand": "Pantene",
            "variant": "Total_Damage_Care",
            "base_variant": "Total_Damage_Care",
            "category": "Hair_Care",
            "packaging_type": "Bottle",
            "is_hul": False,
            "has_promo": False,
        },
        "3": {
            "sku_id": 3,
            "product": "Dove",
            "brand": "Dove",
            "variant": "Dove",
            "base_variant": "Dove",
            "category": "Hair_Care",
            "packaging_type": "Bottle",
            "is_hul": True,
            "has_promo": False,
        },
    }
    gallery = {
        "1": ["gs://b/gal_1.jpg"],
        "2": ["gs://b/gal_2.jpg"],
        "3": ["gs://b/gal_3.jpg"],
    }

    def _row(idx: int, cid: int, sp: str, sess: str) -> dict:
        return {
            "image": f"{sp}_{idx:04d}.jpg",
            "uri": f"gs://b/{sp}_{idx:04d}.jpg",
            "session_id": sess,
            "width": 200,
            "height": 200,
            "boxes": [[0.0, 0.0, 200.0, 200.0]],
            "products": [cid],
            "labels": [dict(classes[str(cid)])],
        }

    test_rows = [
        _row(0, 1, "test", "sess_test_1"),
        _row(1, 2, "test", "sess_test_2"),
    ]
    val_rows = [
        _row(0, 1, "val", "sess_val_1"),
        _row(1, 2, "val", "sess_val_2"),
    ]
    train_rows = [
        _row(0, 1, "train", "sess_sft_1"),       # SFT-trained URI/session -> must be excluded
        _row(1, 1, "train", "sess_sft_1"),       # Same session as SFT -> must be excluded
        _row(2, 2, "train", "sess_val_1"),       # Same session as val -> must be excluded
        _row(3, 3, "train", "sess_ambig"),       # Brand-only ambiguous class -> must be excluded
        {"image": "train_0004.jpg", "uri": "gs://b/gal_1.jpg", "session_id": "sess_gal", "width": 200, "height": 200, "boxes": [[0, 0, 200, 200]], "products": [1], "labels": [dict(classes["1"])]},  # Gallery URI -> must be excluded
        _row(5, 1, "train", "sess_clean_1"),     # Clean HUL
        _row(6, 2, "train", "sess_clean_2"),     # Clean Non-HUL
        _row(7, 1, "train", "sess_clean_3"),     # Clean HUL
        _row(8, 2, "train", "sess_clean_4"),     # Clean Non-HUL
    ]
    return {
        "classes": classes,
        "gallery": gallery,
        "splits": {"train": train_rows, "val": val_rows, "test": test_rows},
    }


def test_splits_constant_and_jsonl_export(tmp_path: Path):
    assert "test_5k" in dataset.SPLITS
    mf = _fixture_manifest()
    mf["splits"]["test_5k"] = mf["splits"]["test"]
    written = dataset.export_rpc_splits_jsonl(mf, tmp_path)
    assert "test_5k" in written
    assert (tmp_path / "test_5k.jsonl").exists()


def test_build_leakage_free_5k_split_excludes_gallery_val_sft_and_ambiguous():
    mf = _fixture_manifest()
    excluded_uris = {"gs://b/train_0000.jpg"}
    excluded_sessions = {"sess_sft_1"}

    rows_5k = build_leakage_free_5k_split(
        mf,
        excluded_uris=excluded_uris,
        excluded_sessions=excluded_sessions,
        target=6,
        seed=5000,
    )
    assert len(rows_5k) == 6

    uris = {r["uri"] for r in rows_5k}
    sessions = {r["session_id"] for r in rows_5k}
    products = {r["products"][0] for r in rows_5k}

    assert "gs://b/gal_1.jpg" not in uris
    assert "gs://b/val_0000.jpg" not in uris
    assert "gs://b/train_0000.jpg" not in uris
    assert "gs://b/train_0001.jpg" not in uris
    assert "sess_val_1" not in sessions
    assert "sess_sft_1" not in sessions
    assert 3 not in products

    # Verify 50/50 HUL / Non-HUL balance (3 HUL, 3 Non-HUL)
    hul_count = sum(1 for r in rows_5k if r["labels"][0]["is_hul"] is True)
    assert hul_count == 3

    # Verify compute_split_hashes preserves train/val/test hashes while adding test_5k
    hashes_before = compute_split_hashes(mf["splits"], mf["gallery"])
    mf["splits"]["test_5k"] = rows_5k
    hashes_after = compute_split_hashes(mf["splits"], mf["gallery"])
    assert hashes_after["test_sha256"] == hashes_before["test_sha256"]
    assert hashes_after["val_sha256"] == hashes_before["val_sha256"]
    assert hashes_after["train_sha256"] == hashes_before["train_sha256"]
    assert "test_5k_sha256" in hashes_after
