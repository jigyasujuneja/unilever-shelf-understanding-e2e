"""Offline tests for ``hybrid_classify`` (embedding kNN gate -> LLM escalation -> taxonomy)."""

from __future__ import annotations

import json

import pytest
from conftest import usage
from PIL import Image

import runner
from approaches.market_share.classification.hybrid_classify import (
    FIELDS,
    field_votes,
    gate,
    hint_prompt,
    merge_fields,
    project_taxonomy,
    taxonomy_index,
)
from utils import dataset
from utils.gemini_fine_tuning.hul_golden_sft import (
    HUL_GOLDEN_5ATTR_INFERENCE_PROMPT,
    HUL_GOLDEN_INFERENCE_PROMPT,
)
from utils.llm import LLMResult

LUX_BAR = {
    "product": "Lux__Lux_Soft_Touch_Bar",
    "brand": "Lux",
    "base_variant": "Lux_Soft_Touch_Bar",
    "packaging_type": "Bar",
    "category": "PW",
}
LUX_WRAP = {
    "product": "Lux__Lux_Soft_Touch_Bar",
    "brand": "Lux",
    "base_variant": "Lux_Soft_Touch_Bar",
    "packaging_type": "Wrapper",
    "category": "PW",
}
DOVE_JAR = {
    "product": "Dove__Dove_Beauty_Cream_Jar",
    "brand": "Dove",
    "base_variant": "Dove_Beauty_Cream_Jar",
    "packaging_type": "Jar",
    "category": "Skin",
}


def test_field_votes_answer_is_top1_and_confidence_is_weighted_share():
    cands = [(0.90, LUX_BAR), (0.89, LUX_WRAP), (0.70, DOVE_JAR)]
    v = field_votes(cands, inv_temp=80.0, k=5)
    assert {f: v[f]["label"] for f in FIELDS} == LUX_BAR
    # brand / category / base_variant / product agree between the two close neighbours
    assert v["brand"]["conf"] == pytest.approx(v["category"]["conf"]) and v["brand"]["conf"] > 0.99
    assert v["base_variant"]["conf"] == pytest.approx(v["product"]["conf"]) and v["base_variant"]["conf"] > 0.99
    # packaging is split between Bar and Wrapper (exp(-0.8) = 0.45 relative weight)
    assert 0.6 < v["packaging_type"]["conf"] < 0.75
    assert v["packaging_type"]["margin"] == pytest.approx(
        v["packaging_type"]["votes"]["Bar"] - v["packaging_type"]["votes"]["Wrapper"], abs=1e-3)
    # a sharper temperature concentrates the vote on the top-1 neighbour
    assert field_votes(cands, inv_temp=400.0)["packaging_type"]["conf"] > 0.98
    assert field_votes([], 80.0)["brand"]["label"] is None


def test_gate_accepts_only_when_every_field_and_the_energy_are_confident():
    votes = field_votes([(0.95, LUX_BAR), (0.94, LUX_WRAP)], inv_temp=80.0)
    th = {
        "product": 0.7,
        "brand": 0.7,
        "base_variant": 0.7,
        "packaging_type": 0.8,
        "category": 0.7,
    }
    unsure, reason = gate(votes, 0.95, th, min_energy=0.85)
    assert unsure == ["packaging_type"] and "packaging_type vote" in reason
    assert gate(votes, 0.95, {**th, "packaging_type": 0.5}, 0.85) == ([], "all fields confident")
    unsure, reason = gate(votes, 0.80, {f: 0.0 for f in FIELDS}, 0.85)
    assert unsure == list(FIELDS) and "energy" in reason


def test_merge_keeps_confident_stage_a_fields():
    b = {
        "product": "Dove__Dove_Beauty_Cream_Jar",
        "brand": "Dove",
        "base_variant": "Dove_Beauty_Cream_Jar",
        "packaging_type": "Wrapper",
        "category": "Skin",
    }
    assert merge_fields(LUX_BAR, b, ["packaging_type"], "field") == LUX_WRAP
    assert merge_fields(LUX_BAR, b, ["packaging_type"], "all") == b
    assert merge_fields(LUX_BAR, {}, ["brand"], "field") == LUX_BAR  # failed call -> Stage A
    assert merge_fields(LUX_BAR, {"brand": ""}, ["brand"], "field")["brand"] == "Lux"


def test_taxonomy_projection_uses_catalog_combinations_first():
    catalog = {
        1: {**LUX_BAR, "allowed_packaging_types": ["Bar"]},
        2: LUX_WRAP,
        3: DOVE_JAR,
        4: {
            "product": "Dove__Dove_Hair_Fall_Rescue_Shampoo",
            "brand": "Dove",
            "base_variant": "Dove_Hair_Fall_Rescue_Shampoo",
            "packaging_type": "Bottle",
            "category": "Hair_Care",
        },
    }
    brand_cats, pairs, variants_by_brand, sku_meta_by_bv = taxonomy_index(catalog)
    assert brand_cats["Lux"] == {"PW"} and brand_cats["Dove"] == {"Skin", "Hair_Care"}
    # (PW, Bar) is outside utils.taxonomy's pack_data domain but is a real catalog combination: kept
    assert project_taxonomy(LUX_BAR, brand_cats, pairs, variants_by_brand, sku_meta_by_bv) == (LUX_BAR, [])
    # single-category brand fixes the category; then (PW, Tube) -> Bottle is not in the catalog, so kept
    out, changed = project_taxonomy(
        {
            "product": "Lux__Lux_Soft_Touch_Bar",
            "brand": "Lux",
            "base_variant": "Lux_Soft_Touch_Bar",
            "packaging_type": "Tube",
            "category": "Skin",
        },
        brand_cats,
        pairs,
        variants_by_brand,
        sku_meta_by_bv,
    )
    assert out["category"] == "PW" and changed == ["category"]
    # (Skin, Pump) -> Bottle via project_to_valid_category_pack only if (Skin, Bottle) is a catalog pair
    pairs2 = pairs | {("Skin", "Bottle")}
    out, changed = project_taxonomy(
        {
            "product": "Dove__Dove_Beauty_Cream_Jar",
            "brand": "Dove",
            "base_variant": "Dove_Beauty_Cream_Jar",
            "packaging_type": "Pump",
            "category": "Skin",
        },
        brand_cats,
        pairs2,
        variants_by_brand,
        sku_meta_by_bv,
    )
    assert out["packaging_type"] == "Bottle" and changed == ["packaging_type"]
    # multi-category brand resolves SKU-specific category via sku_meta_by_bv
    out, changed = project_taxonomy(
        {
            "product": "Dove__Dove_Hair_Fall_Rescue_Shampoo",
            "brand": "Dove",
            "base_variant": "Dove_Hair_Fall_Rescue_Shampoo",
            "packaging_type": "Bottle",
            "category": "Skin",
        },
        brand_cats,
        pairs,
        variants_by_brand,
        sku_meta_by_bv,
    )
    assert out["category"] == "Hair_Care" and "category" in changed


def test_hint_prompt_appends_distinct_knn_candidates_to_the_sft_prompt():
    p = hint_prompt([(0.91, LUX_BAR), (0.90, LUX_BAR), (0.85, LUX_WRAP), (0.8, DOVE_JAR), (0.7, LUX_WRAP)])
    assert p.startswith(HUL_GOLDEN_5ATTR_INFERENCE_PROMPT)
    assert (
        p.count("brand=") == 3
        and "product=Lux__Lux_Soft_Touch_Bar, brand=Lux, base_variant=Lux_Soft_Touch_Bar, packaging_type=Bar, category=PW (similarity 0.91)" in p
    )
    assert hint_prompt([]) == HUL_GOLDEN_5ATTR_INFERENCE_PROMPT
    assert hint_prompt([], fields=("brand", "packaging_type", "category")) == HUL_GOLDEN_INFERENCE_PROMPT


# ---- end to end through the runner (fake embeddings + fake Gemini) ------------------------------

COLOURS = {1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 160, 0)}
CLASSES = {
    1: LUX_BAR,
    2: DOVE_JAR,
    3: {
        "product": "Surf_Excel__Surf_Excel_Matic_Liquid",
        "brand": "Surf_Excel",
        "base_variant": "Surf_Excel_Matic_Liquid",
        "packaging_type": "Pouch",
        "category": "Laundry",
    },
}
CROPS = [("red.jpg", (255, 0, 0), 1), ("purple.jpg", (110, 0, 170), 1)]  # purple: nearest blue, then red


@pytest.fixture
def hul_root(tmp_path, monkeypatch):
    root = tmp_path / "hul_golden"
    (root / "gallery").mkdir(parents=True)
    (root / "images").mkdir()
    for pid, rgb in COLOURS.items():
        Image.new("RGB", (40, 60), rgb).save(root / "gallery" / f"{pid}_0.jpg")
    # a reference shot in the same store visit (session 3541) as a test crop must not be indexed
    Image.new("RGB", (40, 60), (110, 0, 170)).save(root / "gallery" / "3541_7.jpg")
    for name, rgb, _ in CROPS:
        Image.new("RGB", (40, 60), rgb).save(root / "images" / name)
    classes = {str(p): {"sku_id": p, **c, "is_hul": True,
                        "allowed_packaging_types": [c["packaging_type"]]} for p, c in CLASSES.items()}
    rows = [{"image": n, "uri": f"images/{n}", "session_id": "3541" if n == "purple.jpg" else f"s_{n}",
             "width": 40, "height": 60, "boxes": [[0, 0, 40, 60]], "products": [p]} for n, _, p in CROPS]
    (root / dataset.RPC_JSON).write_text(json.dumps({
        "classes": classes,
        "gallery": {**{str(p): [f"gallery/{p}_0.jpg"] for p in COLOURS}, "2": ["gallery/2_0.jpg", "gallery/3541_7.jpg"]},
        "splits": {"train": [], "val": [], "test": rows},
    }))
    monkeypatch.setenv("SHELF_BENCH_HUL_GOLDEN", str(root))
    dataset._rpc_json.cache_clear()
    dataset.load_rpc.cache_clear()
    yield root
    dataset._rpc_json.cache_clear()
    dataset.load_rpc.cache_clear()


@pytest.mark.parametrize("approach", ["hybrid_classify", "hybrid_classify_sft"])
def test_hybrid_gates_confident_crops_and_escalates_the_rest(hul_root, colour_embeddings, tmp_path, approach,
                                                            monkeypatch):
    monkeypatch.setenv("SHELF_BENCH_HYBRID_MIN_ENERGY", "0.85")  # purple's top-1 cosine is ~0.82
    prompts: list[str] = []

    def llm(image, prompt, **kw):
        prompts.append(prompt)
        if approach == "hybrid_classify":  # contact sheet: pick panel 2 (Lux, the 2nd neighbour)
            return LLMResult({"pack_size": "", "choice": 2}, usage(), 0.01)
        return LLMResult(dict(LUX_BAR), usage(), 0.01)

    s = runner.run(approach, "gemini-t", "test", 0, 0, 1, "t", results_dir=tmp_path, llm=llm,
                   log=lambda *_: None)
    assert s["evaluated_fields"] == list(FIELDS)
    assert s["images"] == 2 and s["accuracy"] == pytest.approx(1.0)
    assert len(prompts) == 1  # the red crop is answered by the embedding alone
    if approach == "hybrid_classify_sft":
        assert prompts[0].startswith(HUL_GOLDEN_5ATTR_INFERENCE_PROMPT) and "Visual-search hints" in prompts[0]
    else:
        assert "Panel Q" in prompts[0]
    h = s["hybrid"]
    assert h["escalation_rate"] == pytest.approx(0.5)
    assert h["accuracy_all_fields"]["gated"] == 1.0 and h["accuracy_all_fields"]["escalated"] == 1.0
    assert h["count"]["gated"] == 1 and h["count"]["escalated"] == 1
    assert h["accuracy_all_fields"]["escalated_stage_a_would_be"] == 0.0  # kNN top-1 was Dove
    assert h["gallery_excluded"] == 1  # the session-3541 reference (else purple would match it exactly)
    assert h["params"]["stage_b"] == ("contact_sheet" if approach == "hybrid_classify" else "sft_hints")
    _, rows = runner.load_run(s["run_id"], tmp_path)
    by_id = {r["image_id"]: r for r in rows}
    red, purple = by_id["red.jpg"], by_id["purple.jpg"]
    dec = {r: next(st["info"] for st in by_id[r]["steps"] if st["name"] == "Hybrid decision")
           for r in by_id}
    assert dec["red.jpg"]["answered_by"] == "A" and red["usage"]["calls"] == 0
    assert dec["purple.jpg"]["escalated"] and purple["usage"]["calls"] == 1
    assert "energy" in dec["purple.jpg"]["reason"]
    # both images pay for their embedding call; only the escalated one pays for Gemini
    assert red["billed"] == {"embedding_image": 1} and purple["billed"] == {"embedding_image": 1}
    assert purple["cost_usd"] > red["cost_usd"] > 0
    assert red["labels"][0]["pred"]["sku_id"] == 1


def test_hybrid_accepts_tuned_models_and_rejects_bad_params(monkeypatch):
    from approaches.base import get

    appr = get("hybrid_classify")
    assert appr.accepts("gemini-3.5-flash-lite")
    assert appr.accepts("marketshare-hul-golden-clean-sft-euw4-gemini-3-5-flash-lite")  # config tuned_models
    assert appr.accepts("projects/p/locations/eu/endpoints/123")
    assert not appr.accepts("gemini-embedding-2")
    assert get("gemini_classify").accepts("marketshare-hul-golden-clean-sft-euw4-gemini-3-5-flash-lite")
    assert not get("gemini_classify").accepts("not-a-model")
    monkeypatch.setenv("SHELF_BENCH_HYBRID_STAGE_B", "nope")
    with pytest.raises(ValueError, match="stage_b"):
        type(appr)()._load_params({})
    monkeypatch.setenv("SHELF_BENCH_HYBRID_STAGE_B", "sft")
    monkeypatch.setenv("SHELF_BENCH_HYBRID_SHEET_CANDIDATES", "nope")
    with pytest.raises(ValueError, match="sheet_candidates"):
        type(appr)()._load_params({})
    monkeypatch.delenv("SHELF_BENCH_HYBRID_SHEET_CANDIDATES")
    monkeypatch.setenv("SHELF_BENCH_HYBRID_TAU_BRAND", "0.55")
    inst = type(appr)()
    inst._load_params({"hybrid_classify": {"tau_brand": 0.9, "min_energy": 0.8}})
    assert (inst.stage_b, inst.tau_brand, inst.min_energy) == ("sft", 0.55, 0.8)
