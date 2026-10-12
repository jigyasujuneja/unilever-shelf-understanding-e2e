"""Unit tests for the POC Experiment Matrix, Pre-Run Cost Estimator, and GCP Model Registry catalog."""

from __future__ import annotations

import json
from pathlib import Path

from utils.experiment_matrix import (
    POC_EXPERIMENTS,
    POC_MODELS,
    build_experiments_and_models_catalog,
    estimate_experiment_cost,
)


def test_poc_experiments_and_models_schema():
    """Verify all 26 prioritized experiments (P0, P1, P2, Moonshot) and 6 model families are well-formed."""
    assert len(POC_EXPERIMENTS) == 26
    priorities = {e["priority"] for e in POC_EXPERIMENTS}
    assert priorities == {"P0", "P1", "P2", "Moonshot"}

    use_cases = {e["use_case"] for e in POC_EXPERIMENTS}
    assert use_cases == {"market_share", "merchandising"}

    tasks = {e["task"] for e in POC_EXPERIMENTS}
    assert tasks == {
        "detection",
        "classification",
        "retrieval",
        "end_to_end",
        "planogram_compliance",
        "promo_detection",
    }

    datasets = {e["dataset"] for e in POC_EXPERIMENTS}
    assert {"hul_golden", "gtmerch_golden", "hul_sessions", "hul_promos", "hul_brands", "hul_variants", "ms_json_golden"} == datasets

    assert len(POC_MODELS) == 6
    model_keys = {m["model_key"] for m in POC_MODELS}
    for exp in POC_EXPERIMENTS:
        assert exp["model_family"] in model_keys
        cost = estimate_experiment_cost(exp)
        assert cost["est_cost_per_image_inr"] >= 0
        assert cost["est_50_images_inr"] >= cost["est_15_images_inr"]
        assert cost["est_100_images_inr"] >= cost["est_50_images_inr"]
        assert "₹" in cost["breakdown"]


def test_build_experiments_and_models_catalog_matches_runs(tmp_path: Path):
    """Verify build_experiments_and_models_catalog joins live run summaries and cost estimates."""
    run_dir = tmp_path / "1008-120000-hierarchical_clm_jev-gemini-3.5-flash-priority-hul_golden"
    run_dir.mkdir(parents=True)
    summary = {
        "run_id": run_dir.name,
        "approach": "hierarchical_clm_jev",
        "model": "gemini-3.5-flash",
        "tier": "priority",
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "split": "test",
        "images": 15,
        "seed": 0,
        "accuracy": 0.9333,
        "precision": 0.9333,
        "recall": 0.9333,
        "f2": 0.9333,
        "p95_latency_s": 0.85,
        "p99_latency_s": 1.12,
        "cost_per_image_usd": 0.0004,
        "cost_per_image_inr": 0.036,
        "usd_to_inr": 90.0,
        "field_accuracy": {
            "product": 0.9333,
            "brand": 1.0,
            "base_variant": 0.9333,
            "category": 1.0,
            "packaging_type": 0.9333,
            "is_hul": 1.0,
            "promo_type": 1.0,
        },
        "stage_metrics": {
            "recall_at_1": 0.8667,
            "recall_at_5": 1.0,
            "mrr": 0.9167,
            "stage2a_zero_vlm_gate_pct": 66.7,
            "vlm_net_gain": 0.0666,
        },
        "architecture": "2-Stage Hierarchical Routing",
        "owner": "jjuneja",
        "steps": ["Stage 1", "Stage 2"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    (run_dir / "images.jsonl").write_text("")

    catalog = build_experiments_and_models_catalog(results_dir=tmp_path)
    assert catalog["gcp_project"] == "ul-fs-s-sandbx-16-prj"
    exp_by_id = {e["exp_id"]: e for e in catalog["experiments"]}
    p0_exp2 = exp_by_id["P0-EXP-02"]
    assert p0_exp2["status"] == "completed"
    assert p0_exp2["run_count"] == 1
    assert p0_exp2["best_run"]["run_id"] == run_dir.name
    assert p0_exp2["best_run"]["primary_score"] == 0.9333
    assert p0_exp2["cost_estimate"]["est_cost_per_image_inr"] > 0

    rollup = p0_exp2["best_run"]["taxonomy_rollup"]
    assert rollup["L1_is_hul"] == 1.0
    assert rollup["L2_category"] == 1.0
    assert rollup["L2_packaging_type"] == 0.9333
    assert rollup["L3_brand"] == 1.0
    assert rollup["L4_base_variant"] == 0.9333
    assert rollup["L5_exact_sku"] == 0.9333
    assert rollup["L5_promo_type"] == 1.0
    assert rollup["hierarchical_tree_score"] > 0.95

    stage = p0_exp2["best_run"]["stage_metrics"]
    assert stage["recall_at_1"] == 0.8667
    assert stage["recall_at_5"] == 1.0
    assert stage["zero_vlm_gate_pct"] == 66.7
    assert stage["vlm_net_gain"] == 0.0666

    models_by_key = {m["model_key"]: m for m in catalog["models"]}
    hier_model = models_by_key["hul-hierarchical-brand-pack-gate"]
    assert hier_model["status"] == "evaluated"
    assert hier_model["best_run"]["run_id"] == run_dir.name


def test_v21_golden_split_stratification_and_gtmerch_labels():
    """Verify 50/50 HUL vs. Non-HUL stratified sampling, split_hashes, and GT-Merch shelf brand inference."""
    from utils.hul_builder import (
        _compute_split_hashes,
        _stratified_hul_non_hul_sample,
        infer_gtmerch_shelf_label,
    )

    classes = {
        "1": {"brand": "Dove", "category": "Skin Cleansing", "packaging_type": "Bottles", "is_hul": True, "has_promo": False},
        "2": {"brand": "Sunsilk", "category": "Hair Care", "packaging_type": "Sachets", "is_hul": True, "has_promo": True},
        "3": {"brand": "Pantene", "category": "Hair Care", "packaging_type": "Bottles", "is_hul": False, "has_promo": False},
        "4": {"brand": "Nivea", "category": "Skin Cleansing", "packaging_type": "Bottles", "is_hul": False, "has_promo": True},
    }
    items = []
    for cid in ("1", "2", "3", "4"):
        for idx in range(25):
            items.append({
                "uri": f"gs://test/cls_{cid}_{idx}.jpg",
                "sku_id": int(cid),
                "products": [int(cid)],
                "bbox": [0, 0, 10, 10],
                "meta": {},
            })

    sampled = _stratified_hul_non_hul_sample(items, classes, cap=40, seed=42, split_name="test")
    assert len(sampled) == 40
    hul_count = sum(1 for x in sampled if classes[str(x["sku_id"])]["is_hul"])
    nh_count = sum(1 for x in sampled if not classes[str(x["sku_id"])]["is_hul"])
    assert hul_count == 20
    assert nh_count == 20

    hashes = _compute_split_hashes({"train": sampled[:20], "val": sampled[20:30], "test": sampled[30:]}, {"1": ["gs://ref1.jpg"]})
    assert len(hashes["train_sha256"]) == 64
    assert len(hashes["gallery_sha256"]) == 64

    cid_out, dove_meta = infer_gtmerch_shelf_label(
        "10935933_DA_Shampoo_38973997_1151034.jpg",
        classes=classes,
        brand_to_cid={"dove": 1, "sunsilk": 2},
        default_cid=1,
    )
    assert cid_out == 1
    assert dove_meta["brand"] == "Dove"
    assert dove_meta["is_hul"] is True


def test_openjev_closed_set_non_abstention_and_energy_veto():
    """Verify resolve_jev_choice enforces closed-set non-abstention and energy veto guardrails."""
    from utils.openjev import resolve_jev_choice

    top = [(0.820, 10, "SKU-10"), (0.805, 22, "SKU-22"), (0.710, 35, "SKU-35")]
    slot_uncertain = {"confidence": 0.65, "probabilities": {0: 0.55, 1: 0.45}, "top2_margin": 0.10}
    # 1. Closed-set non-abstention: raw_choice=0 defaults to top[0][1] when allow_abstain=False
    pred_id, conf1, meta1 = resolve_jev_choice(0, slot_uncertain, top, allow_abstain=False)
    assert pred_id == 10
    assert conf1 == 0.65
    assert meta1["mode"] == "closed_set_top1_fallback"

    # 2. Within energy margin (0.820 - 0.805 = 0.015 <= 0.045): VLM pick #2 is accepted
    slot_pick2 = {"confidence": 0.72, "probabilities": {2: 0.72, 1: 0.28}, "top2_margin": 0.44}
    pred_id2, _, meta2 = resolve_jev_choice(2, slot_pick2, top, allow_abstain=False, max_energy_drop=0.045)
    assert pred_id2 == 22
    assert meta2["mode"] == "jev_confident"

    # 3. Exceeds energy margin (0.820 - 0.710 = 0.110 > 0.045): VLM pick #3 is vetoed back to top[0][1]
    slot_pick3 = {"confidence": 0.72, "probabilities": {3: 0.72, 1: 0.28}, "top2_margin": 0.44}
    pred_id3, _, meta3 = resolve_jev_choice(3, slot_pick3, top, allow_abstain=False, max_energy_drop=0.045)
    assert pred_id3 == 10
    assert "energy_veto_top1" in meta3["mode"]
    assert abs(meta3["energy_drop"] - 0.110) < 1e-4


def test_real_results_catalog_and_v21_backfill():
    """Verify live results (from GCS or local) has v2.1+ val+test runs across all 26 experiments, 5-level rollups, stage metrics, and GCP Vertex AI Registry sync."""
    import runner

    real_results = runner.resolve_results_dir("results")
    if not real_results.is_dir() or not any(real_results.glob("*/summary.json")):
        return

    cat = build_experiments_and_models_catalog(real_results)
    assert len(cat["experiments"]) == 26
    assert len(cat["models"]) == 6
    for m in cat["models"]:
        assert "/locations/us-central1/models/" in m["model_resource"]
        assert "110k" not in m["version_alias"].lower()
    vlog = cat.get("vertex_logged_runs") or {}
    assert len(vlog) >= 20
    assert all("run_context" in v for v in vlog.values())
    for e in cat["experiments"]:
        assert e["cost_estimate"]["est_1000_images_inr"] >= e["cost_estimate"]["est_100_images_inr"]
        if e["run_count"] == 0:
            continue
        assert e["run_count"] >= 2, f"{e['experiment_id']} has <2 matched runs"
        assert e["best_val_run"] is not None, f"{e['experiment_id']} missing best_val_run"
        assert e["best_test_run"] is not None, f"{e['experiment_id']} missing best_test_run"
        best = e["best_run"]
        assert best is not None
        rollup = best["taxonomy_rollup"]
        if e["task"] != "detection":
            assert rollup["L1_is_hul"] >= 0.70, f"{e['experiment_id']} L1_is_hul={rollup['L1_is_hul']}"
            assert rollup["L2_category"] >= 0.70, f"{e['experiment_id']} L2_category={rollup['L2_category']}"
            assert rollup["L3_brand"] >= 0.70, f"{e['experiment_id']} L3_brand={rollup['L3_brand']}"
            assert rollup["hierarchical_tree_score"] >= 0.65
        stage = best["stage_metrics"]
        if e["task"] != "detection":
            assert stage["recall_at_1"] is not None
            assert stage["recall_at_5"] is not None
            assert stage["zero_vlm_gate_pct"] is not None


def test_oos_void_detection_and_cio_portal_endpoints():
    """Verify horizontal OOS void detection, MOON-EXP-15 cold-start binding, and parallel CIO portal HTTP routes."""
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from approaches.merchandising.planogram_compliance.bay_planogram_verifier import (
        compute_planogram_metrics,
    )
    from utils.server import Handler

    # 1. Verify horizontal OOS void gap detection (gap = 180px >= 1.5 * 60px)
    boxes = [
        (10.0, 20.0, 70.0, 120.0),
        (75.0, 20.0, 135.0, 120.0),
        (315.0, 20.0, 375.0, 120.0),
    ]
    catalog = {
        1: {"brand": "Dove", "is_hul": True},
        2: {"brand": "Dove", "is_hul": True},
    }
    plano = compute_planogram_metrics(boxes, [1, 1, 2], catalog)
    assert plano["oos_void_count"] == 1
    assert plano["estimated_missing_facings"] == 3
    assert plano["empty_shelf_share"] > 0.40
    assert plano["brand_block_purity"] == 1.0

    # 2. Verify MOON-EXP-15 binds to cold_start_lid_discovery
    import runner

    real_results = runner.resolve_results_dir("results")
    if real_results.is_dir():
        cat = build_experiments_and_models_catalog(real_results)
        moon15 = next(e for e in cat["experiments"] if e["experiment_id"] == "MOON-EXP-15")
        assert moon15["approach"] == "cold_start_lid_discovery"
        assert moon15["best_run"]["approach"] == "cold_start_lid_discovery"

    # 3. Verify parallel CIO Executive Portal and Engineering UI routes
    class _TestHandler(Handler):
        results_dir = real_results
        pull_src = None
        default_mode = "cio"

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _TestHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        for route in ("/", "/exec", "/executive", "/cio", "/engineering", "/static/cio.js", "/static/cio.css", "/api/exec-overview", "/api/cio-overview"):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=10) as resp:
                assert resp.status == 200
                body = resp.read()
                assert len(body) > 100
    finally:
        httpd.shutdown()
        httpd.server_close()






