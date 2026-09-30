"""Tests for ``src/approaches/market_share/classification/`` (the Classification tab).

Copy ``test_classification_scores_product_brand_and_category`` when adding a new classification
approach.
"""

from __future__ import annotations

import pytest
from conftest import CATALOG, usage

import runner
from utils.llm import LLMResult


def test_classification_scores_product_brand_and_category(products_root, tmp_path):
    answers = {61: 1, 62: 1, 63: -1}  # p1 right; p2 -> sister size (same brand); p3 not listed

    def llm(image, prompt, **kw):
        if "Which of these brands" in prompt:  # brand unread: call 2 sees the whole catalog
            return LLMResult({"brand": "not listed", "size": ""}, usage(), 0.01)
        assert "2: Knorr | Knorr Cubes 60g | cooking" in prompt and kw["schema"]
        return LLMResult({"sku_id": answers[image.width]}, usage(), 0.01)

    s = runner.run(
        "hierarchy_classify",
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
    assert (s["task"], s["dataset"], s["images"]) == ("classification", "products", 3)
    assert s["accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["precision"] == pytest.approx(1 / 2, abs=1e-4)  # 1 right of the 2 answered
    assert s["field_accuracy"] == {
        "product": pytest.approx(1 / 3, abs=1e-4),
        "brand": pytest.approx(2 / 3, abs=1e-4),
        "category": pytest.approx(2 / 3, abs=1e-4),
    }
    _, rows = runner.load_run(s["run_id"], tmp_path)
    by_id = {r["image_id"]: r["labels"][0] for r in rows}  # one product per photo
    assert by_id["p2.jpg"]["pred"]["product"] == "Knorr Cubes 10g"
    assert by_id["p2.jpg"]["correct"] == {"product": False, "brand": True, "category": True}
    assert by_id["p3.jpg"]["pred"] is None


def test_hierarchy_classify_filters_by_brand_and_size_before_asking(products_root, tmp_path):
    calls = []
    reads = {61: ("Knorr", "10 g"), 62: ("Knorr", ""), 63: ("not listed", "")}
    picks = {62: 2, 63: -1}

    def llm(image, prompt, **kw):
        calls.append(image.width)
        if "Which of these brands" in prompt:
            assert "Dove, Knorr, Surf" in prompt
            brand, size = reads[image.width]
            return LLMResult({"brand": brand, "size": size}, usage(), 0.01)
        n = (
            prompt.count(" | cooking")
            + prompt.count(" | laundry")
            + prompt.count(" | personal_care")
        )
        assert n == (2 if image.width == 62 else 4)  # Knorr only / whole catalog
        return LLMResult({"sku_id": picks[image.width]}, usage(), 0.01)

    s = runner.run(
        "hierarchy_classify",
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
    assert sorted(calls) == [61, 62, 62, 63, 63]  # p1: brand + size leave one product
    assert s["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    _, rows = runner.load_run(s["run_id"], tmp_path)
    steps = {r["image_id"]: [st["detail"] for st in r["steps"]] for r in rows}
    assert any("not in the catalog: whole catalog" in d for d in steps["p3.jpg"])


def test_embedding_text_match_picks_nearest_catalog_text(products_root, tmp_path, monkeypatch):
    from approaches.market_share.classification import embedding_text_match

    names = [n for _, _, n, _ in CATALOG]

    class FakeEmbeddings:
        def __init__(self, config, model):
            assert model == "multimodalembedding@001"

        def text(self, text, ctx=None):
            ctx.bill("embedding_image", 1)  # setup: billed to the run's setup cost
            return [1.0 if n in text else 0.0 for n in names]

        def image(self, image, ctx=None):
            ctx.bill("embedding_image", 1)
            i = image.width - 61  # p1 -> product 1, p2 -> product 2, p3 -> product 4 (wrong)
            return [1.0 if k == (3 if i == 2 else i) else 0.0 for k in range(len(names))]

    monkeypatch.setattr(embedding_text_match, "VertexEmbeddings", FakeEmbeddings)
    s = runner.run(
        "embedding_text_match",
        "multimodalembedding@001",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        log=lambda *_: None,
    )
    assert s["task"] == "classification" and s["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["field_accuracy"]["category"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["cost"]["gemini_net_usd_per_image"] == 0
    assert s["cost_per_image_usd"] == pytest.approx(1e-4)
    assert s["cost"]["setup_usd"] == pytest.approx(4e-4)  # 4 catalog texts, not in cost/img
