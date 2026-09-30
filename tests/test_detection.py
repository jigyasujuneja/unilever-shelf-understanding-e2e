"""Tests for ``src/approaches/market_share/detection/`` (the Detection tab).

Copy ``test_run_single_pass_end_to_end`` when adding a new detection approach.
"""

from __future__ import annotations

import json

import pytest
from conftest import GT, PER_M, H, OracleLLM, W, board_inr, usage
from PIL import Image

import runner
from utils import metrics
from utils.llm import LLMResult


def test_single_pass_parses_labelled_boxes():
    from approaches.market_share.detection.single_pass import labelled_boxes

    raw = [
        {"box_2d": [0, 0, 500, 500], "label": "food"},
        {"box_2d": [500, 500, 1000, 1000], "label": "not_a_product"},
        [0, 500, 500, 1000],  # truncated output falls back to bare boxes
    ]
    boxes, labels = labelled_boxes(raw, 100, 100)
    assert boxes == [(0, 0, 50, 50), (50, 50, 100, 100), (50, 0, 100, 50)]
    assert labels == ["food", "not_a_product", "other_product"]


def test_run_single_pass_end_to_end(fake_root, tmp_path):
    out = tmp_path / "results"
    s = runner.run(
        "single_pass",
        "gemini-fake",
        "test",
        0,
        0,
        2,
        "tester",
        results_dir=out,
        data_root=fake_root,
        llm=OracleLLM(),
        log=lambda *_: None,
    )
    assert s["images"] == 3 and s["errors"] == 0
    assert s["recall"] == pytest.approx(1.0, abs=0.01) and s["f2"] > 0.99
    assert s["owner"] == "tester" and s["cost_per_image_usd"] > 0
    assert board_inr(out) == pytest.approx(s["cost_per_image_usd"] * 95.5, abs=1e-3)
    # 1000 image-input + 250 output tokens per image at the fake sheet's prices; local runs
    # read the local dataset, so no compute or storage cost.
    assert s["cost_per_image_usd"] == pytest.approx((1000 * 0.3 + 250 * 2.5) * PER_M)
    assert s["cost"]["compute_source"].startswith("local") and s["traffic"] == {"standard": 3}
    cfg = {"defaults": {"split": "test", "limit": 0, "seed": 0}}
    board = runner.leaderboard(out, cfg)
    assert board[0]["rank"] == "dev" and board[0]["run_id"] == s["run_id"]  # local: unranked
    # A Cloud Run run on the configured image set is ranked, above dev runs even with lower F2;
    # one on a different image count is not.
    official = json.loads((out / s["run_id"] / "summary.json").read_text())
    for run_id, limit in (("cloud", 0), ("cloud-25", 25)):
        (out / run_id).mkdir()
        (out / run_id / "summary.json").write_text(
            json.dumps({
                **official,
                "run_id": run_id,
                "f2": 0.1,
                "limit": limit,
                "environment": {"platform": "cloud-run"},
            })
        )
    assert [(r["run_id"], r["rank"]) for r in runner.leaderboard(out, cfg)] == [
        ("cloud", 1),
        (s["run_id"], "dev"),
        ("cloud-25", "dev"),
    ]
    _, images = runner.load_run(s["run_id"], out)
    assert [st["name"] for st in images[0]["steps"]][-1] == "Score vs ground truth"


def test_detect_classify_drops_non_products(fake_root, tmp_path):
    junk = [int(260 / H * 1000), int(0 / W * 1000), int(299 / H * 1000), int(40 / W * 1000)]

    class TwoPass(OracleLLM):
        def __call__(self, image, prompt, **kw):
            with self.lock:
                self.calls += 1
            if "contact sheet" in prompt:  # pass 2: box 4 (the junk box) is not a product
                return LLMResult(
                    [{"id": i, "label": "not_a_product" if i == 4 else "food"} for i in range(5)],
                    usage(500, 50),
                    0.01,
                )
            boxes = [
                [int(y1 / H * 1000), int(x1 / W * 1000), int(y2 / H * 1000), int(x2 / W * 1000)]
                for x1, y1, x2, y2 in GT
            ]
            return LLMResult(boxes + [junk], usage(), 0.01)

    llm = TwoPass()
    s = runner.run(
        "detect_classify",
        "gemini-fake",
        "val",
        1,
        0,
        1,
        "t",
        results_dir=tmp_path,
        data_root=fake_root,
        llm=llm,
        log=lambda *_: None,
    )
    assert llm.calls == 2 and s["fp"] == 0 and s["recall"] == 1.0
    _, images = runner.load_run(s["run_id"], tmp_path)
    names = [st["name"] for st in images[0]["steps"]]
    assert names == [
        "Load image",
        "Pass 1: detection",
        "Pass 2: classification",
        "Score vs ground truth",
    ]
    assert len(images[0]["steps"][1]["boxes"]) == 5 and len(images[0]["steps"][2]["boxes"]) == 4


def test_rail_profile_cv_finds_facings_between_shelf_rails():
    from approaches.market_share.detection.rail_profile_cv import rail_profile_boxes

    im = Image.new("RGB", (600, 400), (235, 235, 235))
    colours = ["red", "blue", "yellow", "green", "purple", "orange"]
    for y0 in (20, 210):  # two shelves: a dark rail above each row of 6 coloured packs
        im.paste((30, 30, 30), (0, y0, 600, y0 + 12))
        for k, c in enumerate(colours):
            im.paste(c, (10 + k * 98, y0 + 25, 95 + k * 98, y0 + 180))
    boxes, bands = rail_profile_boxes(im)
    assert len(bands) >= 2 and 6 <= len(boxes) <= 24
    packs = [(10 + k * 98, y0 + 25, 95 + k * 98, y0 + 180) for y0 in (20, 210) for k in range(6)]
    assert sum(any(metrics.iou(b, p) > 0.3 for b in boxes) for p in packs) >= 8


def test_single_pass_dedup_drops_containers_duplicates_and_depth_ghosts():
    from approaches.market_share.detection.single_pass_dedup import dedup

    item, other = (100, 100, 200, 300), (300, 100, 400, 300)
    group = (90, 90, 410, 310)            # wide box around both items: container
    bottle = (500, 0, 560, 300)           # tall box with its label boxed inside: kept
    label = (505, 100, 555, 160)
    double = (102, 102, 200, 300)         # near-duplicate of item: NMS
    ghost = (130, 110, 190, 280)          # smaller box behind item: depth ghost
    a, b, kept = dedup([item, other, group, bottle, label, double, ghost])
    assert group not in a and bottle in a
    assert double not in b and ghost in b
    assert set(kept) == {item, other, bottle}


def test_tiled_dedup_merges_products_cut_by_the_seam():
    from approaches.base import Context, Trace
    from approaches.market_share.detection.tiled_dedup import TiledDedup, merge_seam

    top = [(100, 400, 150, 500), (300, 420, 360, 499), (500, 100, 560, 200)]
    bottom = [(101, 502, 149, 610), (330, 500, 420, 600), (700, 700, 760, 800)]
    boxes, n = merge_seam(top, bottom, 500, 1000)
    assert n == 1 and (100, 400, 150, 610) in boxes  # same columns, both edges on the seam
    assert (300, 420, 360, 499) in boxes and (330, 500, 420, 600) in boxes  # columns differ
    assert len(boxes) == 5

    calls = []

    def llm(image, prompt, **kw):  # each tile: one product at its centre, in 0-1000 coords
        calls.append(image.size)
        return LLMResult([{"box_2d": [400, 400, 600, 600], "label": "food"}], usage(), 0.01)

    ctx = Context(model="m", llm=llm, trace=Trace())
    kept = TiledDedup().detect(Image.new("RGB", (W, H)), ctx)
    assert calls == [(W, H // 2), (W, H - H // 2)]
    assert sorted(kept) == [(160, 60, 240, 90), (160, 210, 240, 240)]


def test_failed_image_counts_as_zero_detections(fake_root, tmp_path):
    def broken(*a, **k):
        raise RuntimeError("boom")

    s = runner.run(
        "single_pass",
        "gemini-t",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        data_root=fake_root,
        llm=broken,
        log=lambda *_: None,
    )
    assert s["errors"] == 3 and s["recall"] == 0.0
