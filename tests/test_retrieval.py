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
    assert s["tier"] == "priority"
    assert s["accuracy"] == pytest.approx(1 / 3, abs=1e-4) and s["precision"] == 1.0
    # Thresholds were tuned for multimodalembedding@001, so tiered keeps it as its shortlist.
    assert s["architecture"].endswith("[gemini-t + multimodalembedding@001, priority]")
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
    #    but does NOT overwrite a verified prediction [1, 2, 1] or when visual cosine is low!
    s_vecs = [[1.0, 0.0], [0.95, 0.31], [1.0, 0.0]]
    smoothed_ids, info = smooth_shelf_rows(
        boxes, [1, None, 1], s_vecs, [{1, 2}, {1, 2}, {1, 2}]
    )
    assert smoothed_ids == [1, 1, 1] and len(info) == 1
    kept_verified, info_v = smooth_shelf_rows(
        boxes, [1, 2, 1], s_vecs, [{1, 2}, {1, 2}, {1, 2}]
    )
    assert kept_verified == [1, 2, 1] and not info_v
    unsmoothed_ids, info2 = smooth_shelf_rows(
        boxes, [1, None, 1], [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], [{1}, {1}, {1}]
    )
    assert unsmoothed_ids == [1, None, 1] and not info2

    # 3. Verify end-to-end JevLayaHybrid runner execution with batched escalation
    monkeypatch.setattr(jev_laya_hybrid.JevLayaHybrid, "min_margin", 2.0)  # force all 3 boxes into escalation
    monkeypatch.setattr(jev_laya_hybrid.JevLayaHybrid, "high_cosine_margin", 2.0)
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
    assert colour_embeddings == ["gemini-embedding-2"]
    # 3 uncertain crops resolved in a single batched Gemini call (vs 3 calls in tiered_hybrid)!
    assert len(calls) == 1
    assert s["accuracy"] >= 2 / 3


def test_djev_logprob_entropy_calibration_and_openjev_systemone_client(
    rpc_root, colour_embeddings, tmp_path, monkeypatch
):
    import math
    from http.server import BaseHTTPRequestHandler
    from types import SimpleNamespace as NS

    from approaches.market_share.retrieval import jev_laya_hybrid
    from approaches.market_share.retrieval.jev_laya_hybrid import smooth_shelf_rows
    from utils.openjev import (
         extract_slot_distributions,
        jev_entropy_confidence,
        resolve_jev_choice,
        serialize_logprobs,
    )

    # 1. Verify djev normalized Shannon entropy confidence: 1 - H(p)/ln(N)
    assert jev_entropy_confidence({1: 1.0, 2: 0.0}, n_options=6) == 1.0
    assert jev_entropy_confidence({0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}, n_options=4) == 0.0
    split_conf = jev_entropy_confidence({1: 0.49, 2: 0.51}, n_options=3)
    assert 0.30 < split_conf < 0.40

    # 2. Verify Vertex AI LogprobsResult serialization + slot extraction + Bayesian visual fusion
    fake_lp = NS(
        chosen_candidates=[
            NS(token='{"choices": [', log_probability=-0.001),
            NS(token="2", log_probability=math.log(0.51)),
            NS(token="]}", log_probability=-0.001),
        ],
        top_candidates=[
            NS(candidates=[NS(token='{"choices": [', log_probability=-0.001)]),
            NS(
                candidates=[
                    NS(token="2", log_probability=math.log(0.51)),
                    NS(token="1", log_probability=math.log(0.48)),
                    NS(token="0", log_probability=math.log(0.01)),
                ]
            ),
            NS(candidates=[NS(token="]}", log_probability=-0.001)]),
        ],
    )
    steps = serialize_logprobs(fake_lp)
    slots = extract_slot_distributions(steps, valid_options={0, 1, 2, 3}, n_options=3)
    assert len(slots) == 1 and slots[0]["choice"] == 2
    assert slots[0]["confidence"] < 0.40  # uncertain 51% vs 48% split
    # Because candidate 1 has a much stronger visual cosine (0.91 vs 0.78), Bayesian fusion picks SKU 101
    top_cands = [(0.91, 101, "g/1.jpg"), (0.78, 202, "g/2.jpg")]
    ans, conf, meta = resolve_jev_choice(2, slots[0], top_cands, min_confidence=0.45)
    assert ans == 101 and conf == slots[0]["confidence"] and meta["mode"] == "jev_visual_fusion"

    # 3. Verify smooth_shelf_rows smooths low-confidence interior slots but keeps confident ones
    boxes = [(10.0, 10.0, 50.0, 80.0), (55.0, 12.0, 95.0, 82.0), (100.0, 10.0, 140.0, 80.0)]
    s_vecs = [[1.0, 0.0], [0.95, 0.31], [1.0, 0.0]]
    smoothed_low, s_info = smooth_shelf_rows(
        boxes, [1, 2, 1], s_vecs, [{1, 2}, {1, 2}, {1, 2}], confidences=[1.0, 0.22, 1.0]
    )
    assert smoothed_low == [1, 1, 1] and s_info[0]["from"] == 2 and s_info[0]["mid_confidence"] == 0.22

    # 4. Verify direct /v1/systemone OpenJevClient integration
    systemone_requests = []

    class SystemOneHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            systemone_requests.append(body)
            results = {}
            for q_key in body["questions"]:
                results[q_key] = {
                    "choice": "1",
                    "confidence": 0.92,
                    "probabilities": {"0": 0.02, "1": 0.95, "2": 0.03},
                }
            resp = json.dumps({
                "results": results,
                "model": body.get("model", "openjev-latest"),
                "usage": {"prompt_tokens": 310, "output_tokens": 0},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SystemOneHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("OPENJEV_BASE_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setattr(jev_laya_hybrid.JevLayaHybrid, "min_margin", 2.0)
        monkeypatch.setattr(jev_laya_hybrid.JevLayaHybrid, "high_cosine_margin", 2.0)
        s_oj = runner.run(
            "jev_laya_hybrid",
            "gemini-t",
            "test",
            0,
            0,
            1,
            "t",
            results_dir=tmp_path,
            llm=OracleLLM(),
            log=lambda *_: None,
        )
        assert len(systemone_requests) == 1
        assert "questions" in systemone_requests[0] and "images" in systemone_requests[0]
        assert s_oj["tier"] == "priority" and s_oj["traffic"] == {"priority": 1}
        assert s_oj["tokens"]["buckets"] == {"priority/text_input": 310}
        assert s_oj["accuracy"] >= 2 / 3
    finally:
        server.shutdown()


def test_photometry_and_clm_math_primitives():
    import numpy as np
    from PIL import Image

    from approaches.market_share.retrieval.clef_standalone import clef_logprob_select
    from utils.clm import (
        compute_csls_hubness,
        compute_zscore_and_lid,
        infonce_posterior,
        infonce_rerank,
    )
    from utils.photometry import apply_illuminant, estimate_shelf_illuminant, inner_lab_zones

    im = Image.new("RGB", (64, 96), (210, 80, 60))
    gains = estimate_shelf_illuminant(im, p=6.0)
    assert len(gains) == 3 and all(0.70 <= g <= 1.40 for g in gains)
    wb = apply_illuminant(im, gains)
    zones = inner_lab_zones(wb)
    assert len(zones) == 13  # 3 inner LAB bands (9) + aspect/taper/promo-chroma features (4)

    zlid = compute_zscore_and_lid([0.92, 0.75, 0.73, 0.72, 0.71])
    assert zlid["z_margin"] > 1.5 and zlid["z_top1"] > 1.5 and zlid["lid"] > 0.0

    # Degenerate tied runner-ups must NOT produce inflated z_margin >= 1.42
    zlid_tied = compute_zscore_and_lid([0.751, 0.750, 0.750, 0.750, 0.750])
    assert zlid_tied["z_margin"] < 0.25

    # Open-set dustbin suppresses false confidence on low-similarity OOD crops
    p_ood, _ = infonce_posterior([(0.52, 10, "a.jpg"), (0.35, 20, "b.jpg")], inv_temp=28.0)
    assert p_ood < 0.05

    mat = np.asarray([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]], dtype=np.float32)
    csls = compute_csls_hubness(mat, k_nn=1)
    assert csls.shape == (3,)

    coarse = [(0.88, 10, "a.jpg"), (0.87, 20, "b.jpg")]
    reranked = infonce_rerank(
        coarse,
        q=[1.0, 0.0],
        q_lab=zones,
        lab_index={"a.jpg": [0.0] * 13, "b.jpg": zones},
        sku_centroids={10: [0.5, 0.5], 20: [1.0, 0.0]},
    )
    assert reranked[0][1] == 20
    p1, ent = infonce_posterior(reranked, inv_temp=28.0)
    assert 0.50 <= p1 <= 1.0 and 0.0 <= ent <= 1.0

    # Ensure digits inside "pack_size" before "choice" are ignored by clef_logprob_select
    lp_steps = [
        {"token": '"pack_size"', "logprob": -0.01, "top": []},
        {"token": "3", "logprob": -0.01, "top": [{"token": "3", "logprob": -0.01}]},
        {"token": '"choice"', "logprob": -0.01, "top": []},
        {"token": "2", "logprob": -0.05, "top": [{"token": "2", "logprob": -0.05}, {"token": "1", "logprob": -3.0}]},
    ]
    slots = clef_logprob_select(lp_steps, {0, 1, 2, 3, 4, 5}, n_slots=1)
    assert len(slots) == 1 and slots[0]["choice"] == 2


def test_standalone_clm_and_clef_have_zero_jev_inheritance(rpc_root, colour_embeddings, tmp_path):
    from approaches.market_share.classification.clef_standalone_classify import (
        ClefStandaloneClassify,
    )
    from approaches.market_share.classification.clm_standalone_classify import (
        ClmStandaloneClassify,
    )
    from approaches.market_share.classification.coarse_jev_gemini import CoarseJevGemini
    from approaches.market_share.classification.djev_classify import DjevClassify
    from approaches.market_share.retrieval.clef_standalone import ClefStandaloneRetrieval
    from approaches.market_share.retrieval.clm_standalone import ClmStandaloneRetrieval
    from approaches.market_share.retrieval.coarse_jev_laya import CoarseJevLaya
    from approaches.market_share.retrieval.jev_laya_hybrid import JevLayaHybrid

    assert not issubclass(ClmStandaloneRetrieval, (JevLayaHybrid, CoarseJevLaya))
    assert not issubclass(ClefStandaloneRetrieval, (JevLayaHybrid, CoarseJevLaya))
    assert not issubclass(ClmStandaloneClassify, (DjevClassify, CoarseJevGemini))
    assert not issubclass(ClefStandaloneClassify, (DjevClassify, CoarseJevGemini))

    s_clm = runner.run(
        "clm_standalone",
        "multimodalembedding@001",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        log=lambda *_: None,
    )
    assert s_clm["tokens"]["calls"] == 0
    assert s_clm["accuracy"] >= 2 / 3

    s_clef = runner.run(
        "clef_standalone",
        "gemini-t",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        llm=lambda im, pr, **kw: LLMResult({"pack_sizes": ["", "", ""], "choices": [1, 1, 2]}, usage(), 0.01),
        log=lambda *_: None,
    )
    assert s_clef["tokens"]["calls"] == 1
    assert s_clef["accuracy"] == 1.0


def test_two_tier_gallery_self_calibration_and_poe_fusion():
    import pytest

    from utils.clm import calibrate_gallery_geometry
    from utils.openjev import resolve_jev_choice

    # 1. Forbid calibrating on test split
    with pytest.raises(ValueError, match="Zero-leakage invariant violated"):
        calibrate_gallery_geometry({}, {}, split="test")

    # 2. Tier A + Tier B LOVO calibration on synthetic 4-SKU train gallery
    sku_centroids = {
        10: [1.0, 0.0, 0.0],
        20: [0.95, 0.3122, 0.0],
        30: [0.0, 1.0, 0.0],
        40: [0.0, 0.0, 1.0],
    }
    index = [
        (10, "g/10_1.jpg", [0.99, 0.141, 0.0]),
        (10, "g/10_2.jpg", [0.98, -0.199, 0.0]),
        (20, "g/20_1.jpg", [0.95, 0.3122, 0.0]),
        (20, "g/20_2.jpg", [0.94, 0.341, 0.0]),
        (30, "g/30_1.jpg", [0.0, 0.99, 0.141]),
        (30, "g/30_2.jpg", [0.0, 1.0, 0.0]),
        (40, "g/40_1.jpg", [0.0, 0.0, 1.0]),
        (40, "g/40_2.jpg", [0.0, 0.141, 0.99]),
    ]
    cal = calibrate_gallery_geometry(
        index,
        sku_centroids,
        target_system1_precision=0.95,
        split="train",
    )
    assert 12.0 <= cal["inv_temp"] <= 60.0
    assert 0.35 <= cal["conformal_min_prob"] <= 0.92
    assert 0.72 <= cal["intra_sku_cosine_floor"] <= 0.96
    # Close siblings (10 and 20) have higher sibling cosine and tighter gap than isolated SKU 40
    assert cal["sku_sibling_cosine"][10] > cal["sku_sibling_cosine"][40]
    assert cal["sku_required_margin"][10] < cal["sku_required_margin"][40]

    # 3. Continuous Entropy-Weighted Bayesian Product-of-Experts metadata
    slot = {
        "choice": 2,
        "confidence": 0.25,
        "probs": {"0": 0.02, "1": 0.47, "2": 0.51},
    }
    top_cands = [(0.92, 101, "g/1.jpg"), (0.76, 202, "g/2.jpg")]
    ans, conf, meta = resolve_jev_choice(2, slot, top_cands, min_confidence=0.45)
    assert ans == 101
    assert meta["mode"] == "jev_visual_fusion"
    assert "c1_confidence" in meta and "c2_confidence" in meta and "poe_weight_w2" in meta
    assert 0.0 <= meta["poe_weight_w2"] <= 1.0

