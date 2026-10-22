"""Unit tests for Wave 1 Golden Manifests: ``mt_json_golden``, ``hul_talkers_golden``, and ``awp_code_templates``."""

from __future__ import annotations

import json
from pathlib import Path

from utils.awp_template_index import (
    CATEGORY_PREFIXES,
    FIXTURE_TYPES,
    load_awp_template_index,
    parse_awp_template_filename,
)
from utils.dataset import HUL_FIELDS, RPC_FORMAT_SETS, load_rpc, parse_split_jsonl
from utils.llm import load_config
from utils.manifests.talker_loader import TALKER_CLASSES, extract_talker_outlet_code


def test_config_and_dataset_registration() -> None:
    assert "mt_json_golden" in RPC_FORMAT_SETS
    assert "hul_talkers_golden" in RPC_FORMAT_SETS
    assert "mt_json_golden" in HUL_FIELDS
    assert "hul_talkers_golden" in HUL_FIELDS

    cfg = load_config().get("gcp", {})
    assert cfg.get("mt_json_golden") == "gs://marketshare-datasets/derived_manifests/mt_json_golden"
    assert cfg.get("hul_talkers_golden") == "gs://marketshare-datasets/derived_manifests/hul_talkers_golden"
    assert cfg.get("awp_code_templates") == "gs://marketshare-datasets/derived_manifests/awp_code_templates"


def test_awp_code_templates_index() -> None:
    idx = load_awp_template_index("data/awp_code_templates/index.json")
    assert idx["total_templates"] == 63
    templates = idx["templates"]
    assert len(templates) == 63
    assert len(idx["by_code"]) == 63

    seen_codes: set[str] = set()
    seen_brands: set[str] = set()
    for t in templates:
        code = t["awp_code"]
        assert code and code not in seen_codes
        seen_codes.add(code)
        assert t["uri"].startswith("gs://sos-datasets/awp_vectors/awp_code/")
        assert t["width"] > 0 and t["height"] > 0
        assert t["aspect_ratio"] > 0.0
        assert t["category_prefix"] in CATEGORY_PREFIXES
        assert t["fixture_type"] in FIXTURE_TYPES
        assert t["brand_hint"]
        seen_brands.add(t["brand_hint"])

    expected_brands = {
        "LUX", "DOVE", "PONDS", "LAKME", "VASELINE", "PEPSODENT",
        "SUNSILK", "SIMPLE", "TRESEMME", "INDULEKHA", "HORLICKS", "BOOST",
    }
    assert expected_brands.issubset(seen_brands)

    meta = parse_awp_template_filename("PPPONDSMB2WINGTMOC2-HORIZONTAL.jpg")
    assert meta["category_prefix"] == "PP"
    assert meta["fixture_type"] == "MB2"
    assert meta["brand_hint"] == "PONDS"


def test_hul_talkers_golden_zero_outlet_overlap_and_valid_boxes() -> None:
    from utils.dataset import data_root, read_bytes

    root_str = str(Path("data/hul_talkers_golden")) if Path("data/hul_talkers_golden/rpc.json").exists() else data_root("hul_talkers_golden")
    mf = json.loads(read_bytes(f"{root_str}/rpc.json").decode("utf-8"))

    assert set(mf["classes"].keys()) == {"1", "2", "3"}
    for cid, expected in TALKER_CLASSES.items():
        assert mf["classes"][cid]["product"] == expected["product"]
        assert mf["classes"][cid]["promo_type"] == expected["promo_type"]

    outlets_by_split: dict[str, set[str]] = {}
    md5s_by_split: dict[str, set[str]] = {}
    classes_by_split: dict[str, set[int]] = {}

    for sp in ("train", "val", "test"):
        rows = parse_split_jsonl(read_bytes(f"{root_str}/{sp}.jsonl").decode("utf-8"))
        assert len(rows) == len(mf["splits"][sp])
        assert len(rows) > 0

        outlets: set[str] = set()
        md5s: set[str] = set()
        cls_seen: set[int] = set()
        for r in rows:
            oc = r["outlet_code"]
            assert oc == extract_talker_outlet_code(r["uri"])
            outlets.add(oc)
            assert r["md5"] not in md5s
            md5s.add(r["md5"])

            w, h = r["width"], r["height"]
            assert w > 0 and h > 0
            assert len(r["boxes"]) == len(r["products"]) == len(r["labels"])
            assert len(r["boxes"]) > 0
            for (x1, y1, x2, y2), pid in zip(r["boxes"], r["products"], strict=True):
                assert x2 > x1 and y2 > y1
                assert 0.0 <= x1 < x2 <= float(w) + 0.5
                assert 0.0 <= y1 < y2 <= float(h) + 0.5
                assert pid in (1, 2, 3)
                cls_seen.add(pid)

        outlets_by_split[sp] = outlets
        md5s_by_split[sp] = md5s
        classes_by_split[sp] = cls_seen
        loaded = load_rpc(sp, root_str)
        assert len(loaded) == len(rows)

    assert not (outlets_by_split["train"] & outlets_by_split["val"])
    assert not (outlets_by_split["train"] & outlets_by_split["test"])
    assert not (outlets_by_split["val"] & outlets_by_split["test"])
    assert len(outlets_by_split["train"] | outlets_by_split["val"] | outlets_by_split["test"]) == 532

    assert not (md5s_by_split["train"] & md5s_by_split["val"])
    assert not (md5s_by_split["train"] & md5s_by_split["test"])
    assert not (md5s_by_split["val"] & md5s_by_split["test"])

    for sp in ("train", "val", "test"):
        assert classes_by_split[sp] == {1, 2, 3}, f"Split {sp} missing talker class: {classes_by_split[sp]}"


def test_mt_json_golden_zero_outlet_overlap_and_valid_boxes() -> None:
    from utils.dataset import data_root, read_bytes

    root_str = str(Path("data/mt_json_golden")) if Path("data/mt_json_golden/rpc.json").exists() else data_root("mt_json_golden")
    mf = json.loads(read_bytes(f"{root_str}/rpc.json").decode("utf-8"))
    classes = mf["classes"]
    assert len(classes) > 100

    outlets_by_split: dict[str, set[str]] = {}
    md5s_by_split: dict[str, set[str]] = {}
    total_rows = 0

    for sp in ("train", "val", "test"):
        rows = parse_split_jsonl(read_bytes(f"{root_str}/{sp}.jsonl").decode("utf-8"))
        assert len(rows) == len(mf["splits"][sp])
        assert len(rows) > 0
        total_rows += len(rows)

        outlets: set[str] = set()
        md5s: set[str] = set()
        for r in rows:
            outlets.add(r["outlet_code"])
            assert r["md5"] not in md5s
            md5s.add(r["md5"])

            w, h = r["width"], r["height"]
            assert w > 0 and h > 0
            assert len(r["boxes"]) == len(r["products"]) == len(r["labels"])
            assert len(r["boxes"]) >= 2
            for (x1, y1, x2, y2), pid in zip(r["boxes"], r["products"], strict=True):
                assert x2 > x1 and y2 > y1
                assert 0.0 <= x1 < x2 <= float(w) + 0.5
                assert 0.0 <= y1 < y2 <= float(h) + 0.5
                assert str(pid) in classes

        outlets_by_split[sp] = outlets
        md5s_by_split[sp] = md5s
        loaded = load_rpc(sp, root_str)
        assert len(loaded) == len(rows)

    assert not (outlets_by_split["train"] & outlets_by_split["val"])
    assert not (outlets_by_split["train"] & outlets_by_split["test"])
    assert not (outlets_by_split["val"] & outlets_by_split["test"])
    assert not (md5s_by_split["train"] & md5s_by_split["val"])
    assert not (md5s_by_split["train"] & md5s_by_split["test"])
    assert not (md5s_by_split["val"] & md5s_by_split["test"])
    assert 950 <= total_rows <= 1300
