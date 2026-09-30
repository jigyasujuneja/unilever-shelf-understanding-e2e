"""Tests for ``src/approaches/market_share/end_to_end/``, ``compose()``, ``detect_and_identify()``,
and approach registry discovery.

Shows all three ways to build an ``end_to_end`` approach:
1. 2-step composition: ``compose(name, detector, retriever)`` (``test_end_to_end_needs_the_right_box_and_the_right_product``)
2. 3-step composition: ``compose(name, detector, classifier, retriever)`` (``test_compose_chains_detect_classify_and_retrieve``)
3. 1-step single invocation: override ``detect_and_identify(image, ctx)`` (``test_single_call_detect_and_identify_runs_end_to_end``)
"""

from __future__ import annotations

import json

import pytest
from conftest import RPC_GT, H, OracleLLM, W, usage

import approaches
import runner
from utils.llm import LLMResult


def test_registry_has_builtin_approaches():
    from pathlib import Path

    import tomllib

    cfg = tomllib.loads(Path("pyproject.toml").read_text())
    assert cfg["tool"]["setuptools"]["packages"]["find"] == {"where": ["src"]}
    names = set(approaches.all_approaches())
    assert {"single_pass", "detect_classify", "detect_retrieve", "shelf_detect_tiered"} <= names
    for ap in approaches.all_approaches().values():
        assert ap.architecture and ap.steps


def test_retrieval_template_is_not_registered():
    assert "detect_retrieve_alloydb" not in approaches.all_approaches()
    assert "single_call_shelf_identify" not in approaches.all_approaches()


def test_market_share_and_merchandising_are_ranked_separately(fake_root, tmp_path):
    from approaches.base import Approach, register

    s = runner.run(
        "single_pass",
        "gemini-t",
        "test",
        1,
        0,
        1,
        "t",
        results_dir=tmp_path,
        data_root=fake_root,
        llm=OracleLLM(),
        log=lambda *_: None,
    )
    assert s["use_case"] == "market_share"
    p = tmp_path / s["run_id"] / "summary.json"
    base = {**json.loads(p.read_text()), "limit": 50, "environment": {"platform": "cloud-run"}}
    old = {k: v for k, v in base.items() if k != "use_case"}  # runs from before use cases
    for run_id, summary in (("old", old), ("merch", {**base, "use_case": "merchandising"})):
        (tmp_path / run_id).mkdir()
        (tmp_path / run_id / "summary.json").write_text(json.dumps({**summary, "run_id": run_id}))
    board = runner.leaderboard(tmp_path, {"defaults": {"split": "test", "limit": 50, "seed": 0}})
    assert [(r["run_id"], r["use_case"], r["rank"]) for r in board] == [
        ("old", "market_share", 1),
        (s["run_id"], "market_share", "dev"),
        ("merch", "merchandising", 1),
    ]

    class Planogram(Approach):
        name, use_case = "planogram", "planograms"

    with pytest.raises(ValueError, match="use_case"):
        register(Planogram)


def test_end_to_end_needs_the_right_box_and_the_right_product(rpc_root, colour_embeddings, tmp_path):
    boxes = [g[:4] for g in RPC_GT] + [(300, 20, 390, 100)]  # + one box on empty counter

    def llm(image, prompt, **kw):
        return LLMResult(
            [
                [int(y1 / H * 1000), int(x1 / W * 1000), int(y2 / H * 1000), int(x2 / W * 1000)]
                for x1, y1, x2, y2 in boxes
            ],
            usage(),
            0.01,
        )

    s = runner.run(
        "detect_retrieve",
        "gemini-t",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        llm=llm,
        log=lambda *_: None,
    )
    # Boxes 1-2 right; box 3 is on a real product but names the wrong one (FP + FN); box 4 FP.
    assert (s["tp"], s["fp"], s["fn"]) == (2, 2, 1) and s["found_recall"] == 1.0
    _, rows = runner.load_run(s["run_id"], tmp_path)
    labels = rows[0]["labels"]
    assert labels[2]["gt"]["sku_id"] == 3 and labels[2]["correct"]["product"] is False
    assert labels[3]["gt"] is None and labels[3]["correct"] == {"product": False, "category": False}


def test_compose_chains_detect_classify_and_retrieve(
    rpc_root, colour_embeddings, tmp_path, monkeypatch
):
    from approaches.base import Approach, compose, to_pixels
    from approaches.market_share.retrieval.embedding_retrieval import EmbeddingRetrieval

    class BoxDet(Approach):
        name, architecture, steps = "box_det", "Detect boxes", ["Detect boxes"]

        def detect(self, image, ctx):
            return to_pixels(ctx.ask(image, "detect").data, 0, 0, *image.size)

    class OnlyId3Filter(Approach):
        name, architecture, steps = "id3_filter", "Narrow to SKU 3", ["Filter catalog to SKU 3"]

        def narrow(self, image, catalog, ctx):
            # For the green crop at x >= 150, restrict retrieval to SKU 3 so it beats SKU 2
            return {3} if image.getpixel((10, 10))[1] > 100 else {1, 2}

    cls = compose(
        "det_cls_ret",
        BoxDet,
        OnlyId3Filter,
        EmbeddingRetrieval,
        dataset="rpc",
        register_approach=False,
    )
    monkeypatch.setitem(approaches.base.REGISTRY, "det_cls_ret", cls())

    boxes = [g[:4] for g in RPC_GT]

    def llm(image, prompt, **kw):
        return LLMResult(
            [
                [int(y1 / H * 1000), int(x1 / W * 1000), int(y2 / H * 1000), int(x2 / W * 1000)]
                for x1, y1, x2, y2 in boxes
            ],
            usage(),
            0.01,
        )

    s = runner.run(
        "det_cls_ret",
        "gemini-t",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        llm=llm,
        log=lambda *_: None,
    )
    # Without OnlyId3Filter, product 3 (painted green) matched product 2; with narrow->{3} it matches 3.
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 1)
    _, rows = runner.load_run(s["run_id"], tmp_path)
    assert [lab["pred"]["sku_id"] for lab in rows[0]["labels"]] == [1, 3, 3]


def test_single_call_detect_and_identify_runs_end_to_end(rpc_root, tmp_path, monkeypatch):
    from approaches.base import Approach, to_pixels

    class SingleShotE2E(Approach):
        name = "single_shot_e2e"
        task = "end_to_end"
        dataset = "rpc"
        architecture = "Single call: detect + identify"
        steps = ["One call returns boxes and sku_ids"]

        def detect_and_identify(self, image, ctx):
            res = ctx.ask(image, "detect_and_identify")
            boxes = to_pixels([r["box_2d"] for r in res.data], 0, 0, *image.size)
            ids = [r["sku_id"] for r in res.data]
            ctx.trace.step("Single-shot detect + classify", f"{len(boxes)} items", boxes=boxes)
            return boxes, ids

    monkeypatch.setitem(approaches.base.REGISTRY, "single_shot_e2e", SingleShotE2E())

    def llm(image, prompt, **kw):
        return LLMResult(
            [
                {
                    "box_2d": [
                        int(y1 / H * 1000),
                        int(x1 / W * 1000),
                        int(y2 / H * 1000),
                        int(x2 / W * 1000),
                    ],
                    "sku_id": sku,
                }
                for x1, y1, x2, y2, sku, _ in RPC_GT
            ],
            usage(),
            0.01,
        )

    s = runner.run(
        "single_shot_e2e",
        "gemini-t",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        llm=llm,
        log=lambda *_: None,
    )
    assert (s["tp"], s["fp"], s["fn"]) == (3, 0, 0) and s["f2"] == pytest.approx(1.0)
