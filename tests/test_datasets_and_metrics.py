"""Tests for ``src/utils/dataset.py``, ``src/utils/metrics.py``, and ``src/utils/llm.py`` JSON parsing."""

from __future__ import annotations

import json

import pytest
from conftest import GT
from PIL import Image

from utils import dataset, metrics
from utils.llm import parse_json


def test_load_split_and_sampling(fake_root):
    s = dataset.load_split("test", fake_root)
    assert len(s) == 3 and len(s["test_0.jpg"].boxes) == 4
    a = dataset.sample_images("test", 2, 0, fake_root)
    b = dataset.sample_images("test", 2, 0, fake_root)
    assert [x.image_id for x in a] == [x.image_id for x in b]
    assert len(dataset.sample_images("test", 0, 0, fake_root)) == 3


def test_metrics():
    m = metrics.match(GT[:3] + [(0, 200, 5, 205)], GT)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 1, 1)
    s = metrics.scores(3, 1, 1)
    assert s["precision"] == 0.75 and s["recall"] == 0.75 and s["accuracy"] == 0.6
    assert s["f2"] == pytest.approx(0.75)
    assert metrics.percentile([1, 2, 3, 4], 95) == 4
    assert metrics.percentile(list(range(1, 101)), 95) == 95


def test_each_gt_matched_once():
    m = metrics.match([GT[0], GT[0]], [GT[0]])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)


def test_match_pairs_each_prediction_with_its_ground_truth_box():
    m = metrics.match([(70, 10, 120, 90), (0, 0, 5, 5), (10, 10, 60, 90)], GT)
    assert m["pairs"] == {0: 1, 2: 0} and m["matched"] == [0, 2]


def test_parse_json_recovers_truncated_list():
    assert parse_json("[[1,2,3,4],[5,6,7,8],[9,10") == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert parse_json("```json\n[[1,2,3,4]]\n```") == [[1, 2, 3, 4]]


def test_shelf_sessions_split_bursts_by_time_and_camera_counter():
    names = [
        "20180301_101500.jpg",
        "20180301_101700_HoloLens.jpg",
        "IMG_20180301_103000.jpg",
        "DSC01000.png",
        "DSC01003.png",
        "DSC01020.png",
    ]
    assert dataset.shelf_sessions(names) == [
        ["DSC01000.png", "DSC01003.png"],
        ["DSC01020.png"],
        ["20180301_101500.jpg", "20180301_101700_HoloLens.jpg"],
        ["IMG_20180301_103000.jpg"],
    ]
    with pytest.raises(ValueError):
        dataset.shelf_sessions(["photo.jpg"])


def test_prepare_shelves_keeps_sessions_apart_and_builds_gallery(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(10):  # ten sessions, ten minutes apart, one photo each
        name = f"20180301_1{i}0000.jpg"
        Image.new("RGB", (200, 100), (20 * i, 0, 0)).save(raw / name)
        (raw / f"{i}.xml").write_text(
            f"<annotation><filename>{name}</filename>"
            f"<object><name>fuse_peach__50__54{i % 2}</name>"
            "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>90</ymax></bndbox>"
            "</object></annotation>"
        )
    out = dataset.prepare_shelves(str(raw), str(tmp_path / "shelves"), log=lambda *_: None)
    data = json.loads((out / dataset.RPC_JSON).read_text())
    test, val = ({r["image"] for r in data["splits"][s]} for s in ("test", "val"))
    assert len(test) == 4 and len(val) == 1 and not test & val
    assert data["classes"]["1"] == {"sku_id": 1, "product": "fuse peach 50", "gtin": "540"}
    for files in data["gallery"].values():  # references only come from unscored photos
        assert files and all((out / f).exists() for f in files)
    assert data["splits"]["test"][0]["boxes"] == [[10.0, 10.0, 50.0, 90.0]]
    assert set(dataset.rpc_catalog(str(out))) == {1, 2}
