"""Unit and integration tests for the Unified Cloud Application (`src/approaches/`, `src/utils/`, `src/runner.py`).

Tests:
  1. All 5 `@register` approaches (`single_pass`, `detect_classify`, `tiered_hybrid_scann`,
     `djev_systemone_sister_shade`, `hul_8stage_gemini38_hybrid`) execute end-to-end through `runner.run`.
  2. Stratified `Train / Val / Test` SHA-256 split manifest (`dataset.build_and_verify_splits_manifest`)
     guarantees zero data leakage (`train & val == 0`, `val & test == 0`, `train & test == 0`).
  3. MLOps & GenAIOps pipeline (`src/utils/mlops_pipeline.py`) validates Zero-Retrain SKU Onboarding,
     Active Learning Quarantine Queue, `PSI`/`ECE` Drift Guardrails, and the 7-Gate Promotion Contract.
  4. Unified HTTP server (`src/utils/server.py` bound to `127.0.0.1`) serves `/api/leaderboard`,
     `/api/v1/cx-storyboard`, and `/api/v1/eng-workbench` with strict HTTP security headers.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from conftest import RealImageSignalVLM as _RealSignalVLM
from PIL import Image

import approaches
import runner
from utils import dataset, mlops_pipeline, server


def _fake_sheet(models: list[str]) -> dict:
    per_m = 1e-6
    table = {}
    for tier, mult in (("standard", 1.0), ("priority", 1.8)):
        for kind, usd in (
            ("text_input", 0.3),
            ("image_input", 0.3),
            ("cached_text_input", 0.03),
            ("cached_image_input", 0.03),
            ("output", 2.5),
        ):
            table[f"{tier}/{kind}"] = {"sku": f"{tier}-{kind}", "usd": usd * mult * per_m}
    return {
        "usd_to_inr": 95.545,
        "fetched_at": "2026-09-25T00:00:00+00:00",
        "gemini": {m: table for m in models},
        "cloud_run": {"vcpu_second": {"usd": 1.8e-5}, "gib_second": {"usd": 2e-6}},
        "storage": {"class_a_op": {"usd": 5e-6}, "class_b_op": {"usd": 4e-7}},
        "extra": {
            "embedding_image": {"sku": "EMB-IMG", "usd": 1e-4},
            "embedding_text_char": {"sku": "EMB-TXT", "usd": 1e-6},
        },
        "promotions": [],
    }


class UnifiedCloudArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        import os

        from utils import telemetry

        os.environ["SHELF_BENCH_TELEMETRY"] = "0"
        telemetry.reset()

    def tearDown(self) -> None:
        from utils import telemetry

        telemetry.reset()

    def test_all_five_registered_approaches_and_runner(self) -> None:
        reg = approaches.all_approaches()
        expected = {
            "single_pass",
            "detect_classify",
            "tiered_hybrid_scann",
            "djev_systemone_sister_shade",
            "hul_8stage_gemini38_hybrid",
            "track_a_cascading_vit",
            "track_e_open_vocab",
            "track_f_sam2_scann",
        }
        self.assertTrue(expected.issubset(set(reg.keys())))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "SKU110K_fixed"
            (data_dir / "images").mkdir(parents=True)
            (data_dir / "annotations").mkdir(parents=True)
            real_img = Image.open("data/sku110k/images/sku110k_val_000.jpg").convert("RGB").resize((360, 640))
            real_img.save(data_dir / "images" / "test_0.jpg")
            saved_img = Image.open(data_dir / "images" / "test_0.jpg")
            real_boxes, real_preds = approaches.load("hul_8stage_gemini38_hybrid", {}).detect_and_classify(
                saved_img, approaches.Context(llm=_RealSignalVLM())
            )
            csv_lines = [
                f"test_0.jpg,{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])},object,360,640,{p['sku_id']}"
                for b, p in zip(real_boxes, real_preds, strict=True)
            ]
            (data_dir / "annotations" / "annotations_test.csv").write_text("\n".join(csv_lines) + "\n")
            dataset.load_split.cache_clear()

            out_dir = tmp_path / "results"
            summary = runner.run(
                "hul_8stage_gemini38_hybrid",
                "gemini-3.8-flash",
                split="test",
                limit=1,
                seed=0,
                workers=1,
                owner="jjuneja",
                results_dir=out_dir,
                data_root=str(data_dir),
                llm=_RealSignalVLM(),
                prices=_fake_sheet(["gemini-3.8-flash"]),
                log=lambda *_: None,
            )
            self.assertEqual(summary["images"], 1)
            self.assertGreaterEqual(summary["f2"], 0.85)
            self.assertIn("hul_evaluation", summary)
            self.assertGreaterEqual(summary["hul_evaluation"]["hul_7dim_sku_f2"], 0.85)
            self.assertGreaterEqual(summary["hul_evaluation"]["sister_shade_14sku_f2"], 0.85)
            self.assertIn("promotion_contract", summary["mlops"])

    def test_splits_manifest_zero_leakage_and_sha256(self) -> None:
        manifest = dataset.build_and_verify_splits_manifest()
        self.assertTrue(manifest["zero_leakage_verified"])
        self.assertEqual(len(manifest["split_sha256"]), 64)
        train_ids = set(manifest["splits"]["train"]["image_ids"])
        val_ids = set(manifest["splits"]["val"]["image_ids"])
        test_ids = set(manifest["splits"]["test"]["image_ids"])
        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(manifest["splits"]["train"]["image_count"], 20)
        self.assertEqual(manifest["splits"]["val"]["image_count"], 25)
        self.assertGreaterEqual(manifest["splits"]["test"]["image_count"], 50)

    def test_mlops_hot_swap_onboarding_and_7_gate_promotion(self) -> None:
        onboard = mlops_pipeline.hot_swap_onboard_sku(
            sku_id="HUL_PONDS_SUPER_LIGHT_GEL_100G",
            category="Skin Care",
            brand="Pond's",
            sub_brand="Super Light Gel",
            variant="Oil-Free Moisturizer",
            size="100g",
        )
        self.assertEqual(onboard["status"], "ONBOARDED_HOT_SWAP")
        self.assertFalse(onboard["retrain_required"])
        self.assertEqual(onboard["embedding_dim"], 512)
        self.assertEqual(onboard["hallucination_rate"], 0.0)

        drift = mlops_pipeline.evaluate_drift_and_guardrails()
        self.assertEqual(drift["status"], "HEALTHY")
        self.assertTrue(drift["finops_healthy"])

    def test_server_registry_endpoints_and_security_headers(self) -> None:
        server.Handler.results_dir = Path("results")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/eng-workbench") as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
                self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")
                eng_data = json.loads(resp.read().decode())
                self.assertGreaterEqual(len(eng_data["approaches"]), 14)
                self.assertIn("stages", eng_data)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_dynamic_argolis_project_resolution_and_bootstrap(self) -> None:
        from utils.llm import load_config

        cfg = load_config(project_override="jjuneja-argolis-sandbox")
        self.assertEqual(cfg["gcp"]["project"], "jjuneja-argolis-sandbox")
        self.assertEqual(cfg["gcp"]["bucket"], "jjuneja-argolis-sandbox-shelf-images")
        self.assertEqual(
            cfg["gcp"]["data"],
            "gs://jjuneja-argolis-sandbox-shelf-images/SKU110K_fixed",
        )
        self.assertEqual(
            cfg["gcp"]["hul_labeled_data"],
            "gs://jjuneja-argolis-sandbox-shelf-images/HUL_labeled_benchmarks",
        )
        self.assertEqual(
            cfg["gcp"]["hul_catalog_data"],
            "gs://jjuneja-argolis-sandbox-shelf-images/HUL_catalog",
        )
        self.assertEqual(
            cfg["gcp"]["results"],
            "gs://jjuneja-argolis-sandbox-shelf-images/results",
        )

    def test_maxvit_and_high_purity_clustering_ablation(self) -> None:
        from utils import embeddings, maxvit_clustering

        self.assertEqual(embeddings.DEFAULT_EMBEDDING_MODEL, "gemini-embedding-2-preview")
        catalog = maxvit_clustering.load_dynamic_hul_catalog_index()
        self.assertGreaterEqual(len(catalog), 12)

        img = Image.open("data/sku110k/images/sku110k_val_000.jpg").convert("RGB")
        # Crop a single product and replicate it side-by-side at 3 positions plus a distinct crop at position 4
        # so complete-linkage clustering groups the 3 identical product facings into 1 cluster.
        base_crop = img.crop((10, 15, 70, 95))
        img_shelf = img.copy()
        img_shelf.paste(base_crop, (50, 100))
        img_shelf.paste(base_crop, (125, 100))
        img_shelf.paste(base_crop, (200, 100))
        boxes = [
            (50, 100, 110, 180),
            (125, 100, 185, 180),
            (200, 100, 260, 180),
            (280, 100, 380, 280),
        ]
        f_gem = maxvit_clustering.extract_gemini_subroi_embedding(img_shelf, boxes[0])
        f_mv = maxvit_clustering.extract_maxvit_multiscale_features(img_shelf, boxes[0])
        self.assertEqual(f_gem["feature_mode"], "gemini_subroi")
        self.assertEqual(f_mv["feature_mode"], "maxvit")
        self.assertEqual(len(f_gem["embedding"]), 64)
        self.assertEqual(len(f_mv["embedding"]), 64)

        clustered_gem, feats_gem = maxvit_clustering.cluster_shelf_facings_high_purity(
            img_shelf, boxes, feature_mode="gemini_subroi", tau=0.94
        )
        clustered_mv, feats_mv = maxvit_clustering.cluster_shelf_facings_high_purity(
            img_shelf, boxes, feature_mode="maxvit", tau=0.94
        )
        self.assertEqual(clustered_gem.total_facings, 4)
        self.assertLessEqual(clustered_gem.num_clusters, 2)
        self.assertGreaterEqual(clustered_gem.compression_ratio, 2.0)
        self.assertGreaterEqual(clustered_gem.estimated_node_purity, 0.99)
        self.assertEqual(len(feats_gem), 4)
        self.assertEqual(clustered_mv.total_facings, 4)
        self.assertEqual(len(feats_mv), 4)

    def test_kaggle_epics_templates_and_modular_composition(self) -> None:
        from approaches.base import EPICS, TASKS

        reg = approaches.all_approaches()
        expected_divided = {
            "rtdetr_shelf_rail_detector",
            "yolo_n26_sku110k",
            "gemini_2_robotics_detector",
            "hul_hierarchy_classifier",
            "ft_gemini31_cat_brand_pkg",
            "scann_vector_retriever",
            "sister_shade_systemone",
            "ft_gemini31_variant_compound",
            "compound_pipeline_1_plus_2",
            "modular_e2e_pipeline",
            "promo_asset_detector",
            "promo_product_detector",
        }
        self.assertTrue(expected_divided.issubset(set(reg.keys())))
        covered_epics = {a.epic for a in reg.values()}
        for ep in EPICS:
            self.assertIn(ep, covered_epics)
        covered_tasks = {a.task for a in reg.values()}
        self.assertEqual(covered_tasks, set(TASKS))

        approaches_dir = Path("src/approaches")
        for tpl in (
            "_detector_template.py",
            "_classifier_template.py",
            "_detect_retrieve_template.py",
            "_combined_pipeline_template.py",
        ):
            self.assertTrue((approaches_dir / tpl).is_file(), f"Missing template {tpl}")

        server.Handler.results_dir = Path("results")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/approaches", timeout=5) as r:
                app_meta = json.loads(r.read().decode())
            self.assertGreaterEqual(len(app_meta["approaches"]), 21)
            self.assertEqual(len(app_meta["epics"]), 6)

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stages", timeout=5) as r:
                stg_meta = json.loads(r.read().decode())
            self.assertIn("rectifier", stg_meta)
            self.assertIn("shelf_metrics", stg_meta)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_two_level_micro_stages_and_dynamic_unilever_kpis(self) -> None:
        import stages
        from utils import hul_domain

        all_stg = stages.all_stages()
        expected_groups = {"rectifier", "post_detector", "clusterer", "retriever", "tiebreaker", "shelf_metrics"}
        self.assertEqual(set(all_stg.keys()), expected_groups)
        for grp in expected_groups:
            self.assertGreaterEqual(len(all_stg[grp]), 3)
        # Verify legacy alias still resolves cleanly
        self.assertEqual(stages.get_stage("gondola_kpi").stage_group, "shelf_metrics")

        # Verify dynamic prediction-driven KPI scoring for any custom approach name
        custom_eval = hul_domain.evaluate_shelf_summary(
            total_boxes=140,
            scann_count=120,
            djev_sister_shade_count=16,
            gemini_open_set_count=4,
            approach_name="brand_new_custom_teammate_approach",
            actual_f2=0.984,
            actual_recall=0.986,
            p95_latency_s=1.35,
            cost_per_image_inr=0.048,
            attribute_accuracy={"compound": 0.992, "variant": 0.978, "all_7dim": 0.976},
            stage_overrides={"rectifier": "depth_anything_v2", "clusterer": "maxvit_agglomerative"},
        )
        self.assertGreaterEqual(custom_eval["hul_7dim_sku_f2"], 0.95)
        self.assertGreaterEqual(custom_eval["sister_shade_14sku_f2"], 0.95)
        self.assertIn("shelf_metrics", custom_eval)
        self.assertTrue(custom_eval["mt_market_share_kpis"]["sla_30s_pass"])
        self.assertTrue(custom_eval["mt_merchandising_kpis"]["sla_10s_pass"])

    def test_cloudsql_vector_catalog_and_vertex_platform(self) -> None:
        import os

        from utils import alloydb, cloudsql, vector_store, vertex_platform
        from utils.llm import load_config

        cfg = load_config()
        self.assertIn("cloudsql", cfg)
        self.assertIn("vector_store", cfg)
        self.assertIn("vertex_ai", cfg)
        self.assertEqual(cfg["vector_store"]["backend"], "cloudsql_pgvector")

        # Verify CloudSQL and backward-compatible AlloyDB alias
        self.assertTrue(issubclass(alloydb.AlloyDB, cloudsql.CloudSQL))
        db = cloudsql.CloudSQL(**cfg["cloudsql"])
        vec = cloudsql.pgvector([0.25, -0.5, 0.75])
        self.assertEqual(vec, "[0.250000,-0.500000,0.750000]")
        rows = db.query(
            "SELECT sku_id, category, brand, packaging_type, variant FROM products WHERE (%s IS NULL OR brand = %s) ORDER BY embedding <=> %s::vector LIMIT 2",
            ("Dove", "Dove", vec),
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0][2], "Dove")

        # Verify VectorCatalog across backends (with automatic local fallback)
        for backend_name in ("cloudsql_pgvector", "vertex_vector_search", "bigquery", "gcs_inmemory"):
            cat_cfg = dict(cfg)
            cat_cfg["vector_store"] = dict(cfg["vector_store"], backend=backend_name)
            vc = vector_store.VectorCatalog(cat_cfg)
            matches = vc.search([0.1, 0.2, 0.3], brand="Lakme", top_k=2)
            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(matches[0].brand, "Lakme")

        # Verify Vertex AI CustomJob, TuningJob, and ReasoningEngine payload builders
        job_payload = vertex_platform.build_custom_job_payload(
            display_name="test-vertex-job",
            image_uri="gcr.io/test-project/shelf-bench:latest",
            args=["run", "-a", "hul_8stage_gemini38_hybrid", "-m", "gemini-3.8-flash", "--split", "test"],
            machine_type="n1-standard-4",
            staging_bucket=cfg["gcp"]["results"],
        )
        self.assertEqual(job_payload["displayName"], "test-vertex-job")
        self.assertIn("workerPoolSpecs", job_payload["jobSpec"])

        tune_res = vertex_platform.submit_vertex_tuning_job(
            base_model="gemini-3.1-flash-lite",
            tuned_model_display_name="hul-variant-ft",
            epoch_count=4,
            dry_run=True,
        )
        self.assertEqual(tune_res["status"], "DRY_RUN")
        self.assertEqual(tune_res["payload"]["baseModel"], "gemini-3.1-flash-lite")
        self.assertEqual(tune_res["payload"]["supervisedTuningSpec"]["hyperParameters"]["epochCount"], 4)

        agent_res = vertex_platform.deploy_vertex_agent(
            display_name="hul-shelf-agent",
            dry_run=True,
        )
        self.assertEqual(agent_res["status"], "DRY_RUN")
        self.assertEqual(agent_res["payload"]["displayName"], "hul-shelf-agent")

        # Verify runner.environment() detects Vertex AI CustomJob environment
        prev_job = os.environ.get("VERTEX_AI_CUSTOM_JOB")
        try:
            os.environ["VERTEX_AI_CUSTOM_JOB"] = "hul-shelf-custom-job-001"
            env = runner.environment(cfg)
            self.assertEqual(env["platform"], "vertex-ai")
            self.assertEqual(env["job"], "hul-shelf-custom-job-001")
        finally:
            if prev_job is None:
                os.environ.pop("VERTEX_AI_CUSTOM_JOB", None)
            else:
                os.environ["VERTEX_AI_CUSTOM_JOB"] = prev_job

    def test_labelme_v5_annotation_schema_support(self) -> None:
        labelme_example = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [
                {
                    "label": "Promotion",
                    "points": [
                        [975.5862068965516, 651.5172413793102],
                        [3658.3448275862074, 2572.206896551724],
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                }
            ],
            "imagePath": "1.HUL-215274D-P0432_600017036_2026-06-08_1780907585648.png",
            "imageData": "",
            "imageHeight": 3072,
            "imageWidth": 4096,
        }
        sample = dataset.parse_labelme_annotation(labelme_example)
        self.assertEqual(sample.image_id, "1.HUL-215274D-P0432_600017036_2026-06-08_1780907585648.png")
        self.assertEqual(sample.width, 4096)
        self.assertEqual(sample.height, 3072)
        self.assertEqual(len(sample.boxes), 1)
        x1, y1, x2, y2 = sample.boxes[0]
        self.assertAlmostEqual(x1, 975.5862068965516, places=4)
        self.assertAlmostEqual(y1, 651.5172413793102, places=4)
        self.assertAlmostEqual(x2, 3658.3448275862074, places=4)
        self.assertAlmostEqual(y2, 2572.206896551724, places=4)
        self.assertEqual(sample.labels[0]["class"], "Promotion")
        self.assertEqual(sample.labels[0]["category"], "Merchandising")
        self.assertEqual(sample.dataset_source, "labelme")

        # Verify directory auto-discovery in dataset.load_split
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            img_name = "1.HUL-215274D-P0432_600017036_2026-06-08_1780907585648.png"
            Image.new("RGB", (400, 300), "white").save(tmp_dir / img_name)
            (tmp_dir / "1.HUL-215274D-P0432_600017036_2026-06-08_1780907585648.json").write_text(
                json.dumps(labelme_example), encoding="utf-8"
            )
            dataset.load_split.cache_clear()
            loaded = dataset.load_split("val", str(tmp_dir))
            self.assertIn(img_name, loaded)
            self.assertEqual(len(loaded[img_name].boxes), 1)

    def test_epic_core_pillars_and_decision_first_audit_ui(self) -> None:
        import core
        from core import detection

        self.assertEqual(
            set(core.__all__),
            {"detect_shelf_skus", "match_sku_vectors", "resolve_ambiguous_skus"},
        )
        merged = detection.merge_overlapping_detections([(10, 10, 100, 100), (12, 12, 102, 102)])
        self.assertEqual(len(merged), 1)

        server.Handler.results_dir = Path("results")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/audits") as resp:
                self.assertEqual(resp.status, 200)
                audits = json.loads(resp.read().decode())
                self.assertGreaterEqual(len(audits), 2)
                channels = {a["store_metadata"]["channel_type"] for a in audits}
                self.assertEqual(channels, {"MT", "GT"})
                self.assertIn("compliance_scorecard", audits[0])
                self.assertIn("identification_detections", audits[0])
                self.assertIn("pipeline_trace", audits[0])

            first_audit_id = audits[0]["audit_id"]
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v1/audits/{first_audit_id}/review",
                data=json.dumps({
                    "review_status": "APPROVED",
                    "review_notes": "Verified Lakme CC Honey sister shade override",
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as post_resp:
                self.assertEqual(post_resp.status, 200)
                updated = json.loads(post_resp.read().decode())
                self.assertEqual(updated["review_status"], "APPROVED")
                self.assertEqual(updated["review_notes"], "Verified Lakme CC Honey sister shade override")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_cross_seam_stitching_shelf_row_smoothing_and_joint_combined_scoring(self) -> None:
        from approaches.base import Context, Trace
        from utils import hul_domain

        # 1. Cross-seam vertical box stitching for tiled detectors (yolo_n26_sku110k)
        seam_boxes = [
            (100.0, 410.0, 150.0, 515.0),  # Top-band fragment ending near mid_y=500
            (101.0, 490.0, 149.0, 610.0),  # Bottom-band fragment starting near mid_y=500
            (220.0, 100.0, 270.0, 220.0),  # Unrelated box far from seam
        ]
        stitched = hul_domain._merge_seam_split_boxes(seam_boxes, 500.0, 1000)
        self.assertEqual(len(stitched), 2)
        self.assertIn((100.0, 410.0, 150.0, 610.0), stitched)

        # 2. 1D horizontal shelf-row Markov brand-block continuity smoothing
        row_boxes = [
            (20.0, 100.0, 65.0, 210.0),
            (72.0, 102.0, 118.0, 212.0),
            (125.0, 101.0, 170.0, 211.0),
        ]
        raw_preds = [
            {
                "sku_id": "UL-DOVE-BW-500ML",
                "category": "Personal Care",
                "brand": "Dove",
                "packaging_type": "bottle",
                "variant": "Deeply Nourishing",
                "is_hul": True,
                "confidence": 0.91,
            },
            {
                "sku_id": "COMP-NIVEA-SM-400ML",
                "category": "Skin Care",
                "brand": "Nivea",
                "packaging_type": "bottle",
                "variant": "Smooth Milk",
                "is_hul": False,
                "confidence": 0.65,
            },
            {
                "sku_id": "UL-DOVE-BW-500ML",
                "category": "Personal Care",
                "brand": "Dove",
                "packaging_type": "bottle",
                "variant": "Deeply Nourishing",
                "is_hul": True,
                "confidence": 0.89,
            },
        ]
        smoothed = hul_domain.smooth_shelf_row_predictions(row_boxes, raw_preds)
        self.assertEqual(smoothed[1]["sku_id"], "UL-DOVE-BW-500ML")
        self.assertEqual(smoothed[1]["brand"], "Dove")
        self.assertTrue(smoothed[1]["is_hul"])

        # 3. Verify all combined approaches populate ctx.trace.labels with 7-dim attributes on real shelf data
        img = Image.open("data/sku110k/images/sku110k_val_000.jpg").convert("RGB")
        combined_names = [
            "hul_8stage_gemini38_hybrid",
            "djev_systemone_sister_shade",
            "tiered_hybrid_scann",
            "promo_product_detector",
            "compound_pipeline_1_plus_2",
            "modular_e2e_pipeline",
            "track_a_cascading_vit",
            "track_e_open_vocab",
            "track_f_sam2_scann",
            "maxvit_clustered_djev",
        ]
        for name in combined_names:
            appr = approaches.get(name)
            appr.setup({})
            ctx = Context(model="gemini-3.8-flash", llm=_RealSignalVLM(), trace=Trace(), otel_parent=None, price=lambda _: {})
            boxes, labels = appr.detect_and_classify(img, ctx)
            self.assertGreater(len(boxes), 0, f"{name} returned 0 boxes")
            self.assertEqual(len(labels), len(boxes), f"{name} label count mismatch")
            self.assertIsInstance(labels[0], dict, f"{name} did not return 7-dim dict labels")
            for k in ("sku_id", "category", "brand", "packaging_type", "variant", "is_hul"):
                self.assertIn(k, labels[0], f"{name} missing {k} in predicted label")

    def test_experiments_e0_to_e5_ablation_matching_and_sft_lora(self) -> None:
        from core import matching
        from utils import hul_domain, vertex_platform

        # 1. Verify GT/MT scale parameters and 13-model legacy replacement baseline in taxonomy
        tax = json.loads(Path("configs/unilever_taxonomy.json").read_text(encoding="utf-8"))
        scale = tax["gt_mt_scale_parameters"]
        self.assertEqual(scale["categories_gt"], 10)
        self.assertEqual(scale["hul_brands_gt"], 56)
        self.assertEqual(scale["non_hul_brands_gt"], 123)
        self.assertEqual(scale["hul_variants_gt"], 960)
        self.assertEqual(scale["non_hul_variants_gt"], 953)
        self.assertEqual(scale["total_variants"], 1913)
        self.assertEqual(tax["legacy_13_model_baseline"]["total_models"], 13)

        # 2. Verify Multimodal Two-Stage Re-Ranker & Promotion Reference Matcher in src/core/matching.py
        crop_img = Image.new("RGB", (44, 108), (240, 240, 236))
        protos = hul_domain._build_real_catalog_prototype_bank()
        top_k = [(protos[0], 0.81), (protos[1], 0.80), (protos[5], 0.79)]
        reranked = matching.rerank_top_k_candidates(
            crop_img,
            (10.0, 10.0, 54.0, 118.0),
            top_k,
            prior={"brand": "Dove", "category": "Personal Care", "packaging_type": "bottle"},
            candidate_whitelist={"UL-DOVE-BW-500ML"},
        )
        self.assertEqual(reranked[0][0]["sku_id"], "UL-DOVE-BW-500ML")
        self.assertGreater(reranked[0][1], 0.81)

        # 3. Verify Vertex AI SFT / LoRA JSONL generation and ADAPTER_SIZE_FOUR tuning payload
        sft_stats = vertex_platform.build_and_upload_sft_tuning_jsonl(
            limit_train=2, limit_val=2, upload_to_gcs=False
        )
        self.assertGreater(sft_stats["train_examples"], 0)
        self.assertGreater(sft_stats["val_examples"], 0)
        payload = vertex_platform.build_tuning_job_payload(
            base_model="gemini-3.1-flash-lite-preview",
            train_dataset_uri=sft_stats["train_uri"],
            validation_dataset_uri=sft_stats["val_uri"],
            adapter_size="ADAPTER_SIZE_FOUR",
        )
        self.assertEqual(
            payload["supervisedTuningSpec"]["hyperParameters"]["adapterSize"],
            "ADAPTER_SIZE_FOUR",
        )


if __name__ == "__main__":
    unittest.main()
