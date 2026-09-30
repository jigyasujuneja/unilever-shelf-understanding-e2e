"""Tests for ``src/approaches/market_share/detection/`` (the Detection tab).

Copy ``test_run_single_pass_dedup_end_to_end`` when adding a new detection approach.
"""

from __future__ import annotations

import json

import pytest
from conftest import PER_M, H, OracleLLM, W, board_inr, usage
from PIL import Image

import runner
from utils.llm import LLMResult


def test_single_pass_dedup_parses_labelled_boxes():
    from approaches.market_share.detection.single_pass_dedup import labelled_boxes

    raw = [
        {"box_2d": [0, 0, 500, 500], "label": "food"},
        {"box_2d": [500, 500, 1000, 1000], "label": "not_a_product"},
        [0, 500, 500, 1000],  # truncated output falls back to bare boxes
    ]
    boxes, labels = labelled_boxes(raw, 100, 100)
    assert boxes == [(0, 0, 50, 50), (50, 50, 100, 100), (50, 0, 100, 50)]
    assert labels == ["food", "not_a_product", "other_product"]


def test_run_single_pass_dedup_end_to_end(fake_root, tmp_path):
    out = tmp_path / "results"
    s = runner.run(
        "single_pass_dedup",
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


def test_flex_tier_runs_are_labelled_and_billed_at_flex(fake_root, tmp_path):
    from utils.llm import TIER_HEADERS

    assert TIER_HEADERS["flex"] == {
        "X-Vertex-AI-LLM-Request-Type": "shared",
        "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
    }

    class FlexLLM(OracleLLM):  # Vertex reports ON_DEMAND_FLEX traffic -> flex token buckets
        def __call__(self, image, prompt, **kw):
            res = super().__call__(image, prompt, **kw)
            res.usage.traffic = {"flex": 1}
            res.usage.buckets = {"flex/image_input": 1000, "flex/output": 250}
            return res

    s = runner.run(
        "single_pass_dedup",
        "gemini-fake",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        data_root=fake_root,
        llm=FlexLLM(),
        log=lambda *_: None,
        tier="flex",
    )
    assert s["run_id"].endswith("-flex") and s["architecture"].endswith(", flex]")
    assert s["flex_served"] == 1.0 and s["priority_served"] is None
    assert s["cost_per_image_usd"] == pytest.approx(0.5 * (1000 * 0.3 + 250 * 2.5) * PER_M)


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


def test_dedup_traces_only_filters_that_removed_something():
    from approaches.base import Context, Trace
    from approaches.market_share.detection.single_pass_dedup import dedup_traced

    ctx = Context(model="m", llm=None, trace=Trace())
    item, ghost = (100, 100, 200, 300), (130, 110, 190, 280)
    assert dedup_traced([item, ghost], ctx) == [item]
    assert [s["name"] for s in ctx.trace.steps] == ["Depth ghosts removed"]
    ctx = Context(model="m", llm=None, trace=Trace())
    dedup_traced([item], ctx)
    assert ctx.trace.steps == []


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
        "single_pass_dedup",
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
