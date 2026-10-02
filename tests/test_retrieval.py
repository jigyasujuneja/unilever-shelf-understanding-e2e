"""Tests for ``src/approaches/market_share/retrieval/`` (the Retrieval tab).

Copy ``test_embedding_retrieval_identifies_each_ground_truth_crop`` when adding a new retrieval
approach.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from conftest import OracleLLM, usage

import runner
from utils.llm import LLMResult
from utils.server import Handler


def test_embedding_retrieval_identifies_each_ground_truth_crop(rpc_root, colour_embeddings, tmp_path):
    s = runner.run(
        "embedding_retrieval",
        "multimodalembedding@001",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        log=lambda *_: None,
    )
    assert (s["task"], s["dataset"], s["images"]) == ("retrieval", "rpc", 1)
    assert s["field_accuracy"] == {"product": pytest.approx(2 / 3, abs=1e-4), "category": 1.0}
    assert s["accuracy"] == s["field_accuracy"]["product"]
    # 3 crops embedded per photo; the 3 reference photos are the one-off setup cost.
    assert s["cost_per_image_usd"] == pytest.approx(3e-4)
    assert s["cost"]["setup_usd"] == pytest.approx(3e-4)
    _, rows = runner.load_run(s["run_id"], tmp_path)
    labels = rows[0]["labels"]
    assert [lab["pred"]["sku_id"] for lab in labels] == [1, 2, 2]
    assert labels[2]["box"] == [200, 150, 260, 280] and labels[2]["gt"]["sku_id"] == 3
    assert labels[2]["correct"] == {"product": False, "category": True}

    Handler.results_dir = tmp_path
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    get = lambda p: urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}{p}").read()  # noqa: E731
    try:
        detail = json.loads(get(f"/api/runs/{s['run_id']}/images/test_0.jpg"))
        assert detail["split"] == "rpc" and len(detail["gt"]) == 3
        assert get("/img/rpc/test_0.jpg")[:2] == get("/img/rpc-ref/3")[:2] == b"\xff\xd8"
    finally:
        httpd.shutdown()


def test_tiered_hybrid_escalation_takes_gemini_choice_and_none_means_none(
    rpc_root, colour_embeddings, tmp_path, monkeypatch
):
    from approaches.market_share.retrieval import tiered_hybrid
    from approaches.market_share.retrieval.tiered_hybrid import CELL

    def llm(sheet, prompt, **kw):  # picks candidate 1 for the red crop, "none of these" otherwise
        assert "Panel Q" in prompt and kw["schema"]
        r, g, b = sheet.getpixel((CELL // 2, 30 + CELL // 2))
        return LLMResult({"choice": 1 if r > 200 and g < 100 else 0}, usage(), 0.01)

    monkeypatch.setattr(tiered_hybrid, "MIN_MARGIN", 2.0)  # never sure: every crop escalates
    s = runner.run(
        "tiered_hybrid",
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
    assert s["accuracy"] == pytest.approx(1 / 3, abs=1e-4) and s["precision"] == 1.0
    # Thresholds were tuned for multimodalembedding@001, so tiered keeps it as its shortlist.
    assert s["architecture"].endswith("[gemini-t + multimodalembedding@001]")
    assert colour_embeddings == ["multimodalembedding@001"]
    _, rows = runner.load_run(s["run_id"], tmp_path)
    assert [lab["pred"] and lab["pred"]["sku_id"] for lab in rows[0]["labels"]] == [1, None, None]
    assert rows[0]["billed"] == {"embedding_image": 3}  # each escalated crop embedded once
    # Step details (UI (i) icon): each crop's step carries its own call, bill and shortlist,
    # even though crops run in worker threads.
    tiers = [st for st in rows[0]["steps"] if st["name"] == "Gemini tier"]
    assert len(tiers) == 3
    for st in tiers:
        assert len(st["calls"]) == 1 and st["billed"] == {"embedding_image": 1}
        assert "Panel Q" in st["calls"][0]["prompt"] and st["calls"][0]["input_tokens"] > 0
        assert st["info"]["shortlist"][0]["cosine"] >= st["info"]["shortlist"][-1]["cosine"]
    assert sorted(str(st["info"]["answer"]) for st in tiers) == ["1", "None", "None"]
    identify = next(st for st in rows[0]["steps"] if st["name"] == "Identify each box")
    assert "calls" not in identify and "billed" not in identify  # nothing left unclaimed


def test_tiered_hybrid_asks_gemini_only_when_the_embedding_is_unsure(
    rpc_root, colour_embeddings, tmp_path, monkeypatch
):
    from approaches.market_share.retrieval import tiered_hybrid

    llm = OracleLLM()  # never right here: any answer that isn't {"choice": k} means "none"
    s = runner.run(
        "tiered_hybrid",
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
    assert llm.calls == 0 and s["accuracy"] == pytest.approx(2 / 3, abs=1e-4)  # pure colours: sure
    monkeypatch.setattr(tiered_hybrid, "MIN_MARGIN", 2.0)  # never sure: every crop escalates
    s = runner.run(
        "tiered_hybrid",
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
    assert llm.calls == 3 and s["accuracy"] == 0.0
    _, rows = runner.load_run(s["run_id"], tmp_path)
    assert rows[0]["billed"] == {"embedding_image": 3}


def test_jev_laya_hybrid_clusters_twins_batches_medoids_and_smooths_rows(
    rpc_root, colour_embeddings, tmp_path, monkeypatch
):
    from approaches.market_share.retrieval import jev_laya_hybrid
    from approaches.market_share.retrieval.jev_laya_hybrid import (
        cluster_uncertain_crops,
        smooth_shelf_rows,
    )

    # 1. Verify medoid clustering merges adjacent uncertain crops on the same row with high cosine
    boxes = [(10.0, 10.0, 50.0, 80.0), (55.0, 12.0, 95.0, 82.0), (100.0, 10.0, 140.0, 80.0)]
    vecs = [[1.0, 0.0, 0.0], [0.99, 0.14, 0.0], [0.0, 1.0, 0.0]]
    clusters = cluster_uncertain_crops([0, 1, 2], boxes, vecs, [2, 2, 3])
    assert clusters == [[0, 1], [2]]

    # 2. Verify visually-gated 1D shelf-row Markov smoothing:
    #    [1, None, 1] smooths to [1, 1, 1] when visual cosine >= SMOOTH_COSINE,
    #    but does NOT overwrite when visual cosine is low!
    s_vecs = [[1.0, 0.0], [0.95, 0.31], [1.0, 0.0]]
    smoothed_ids, info = smooth_shelf_rows(
        boxes, [1, None, 1], s_vecs, [0.1, 0.01, 0.1], [{1, 2}, {1, 2}, {1, 2}]
    )
    assert smoothed_ids == [1, 1, 1] and len(info) == 1
    unsmoothed_ids, info2 = smooth_shelf_rows(
        boxes, [1, None, 1], [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], [0.1, 0.01, 0.1], [{1}, {1}, {1}]
    )
    assert unsmoothed_ids == [1, None, 1] and not info2

    # 3. Verify end-to-end JevLayaHybrid runner execution with batched escalation
    monkeypatch.setattr(jev_laya_hybrid, "MIN_MARGIN", 2.0)  # force all 3 boxes into escalation
    calls = []

    def llm(sheet_im, prompt, **kw):
        calls.append(prompt)
        if "choices" in kw["schema"]["properties"]:
            # Batched multi-row sheet: Q1 is red (choice 1), Q2 is green (choice 1 -> product 2,3)
            return LLMResult({"choices": [1, 1, 2]}, usage(), 0.01)
        return LLMResult({"choice": 1}, usage(), 0.01)

    s = runner.run(
        "jev_laya_hybrid",
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
    assert colour_embeddings == ["gemini-embedding-2-preview"]
    # 3 uncertain crops resolved in a single batched Gemini call (vs 3 calls in tiered_hybrid)!
    assert len(calls) == 1
    assert s["accuracy"] >= 2 / 3

