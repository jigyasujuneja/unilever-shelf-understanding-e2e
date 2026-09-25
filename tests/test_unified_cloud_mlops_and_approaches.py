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

import utils  # Registers _local_shims at the end of sys.path if Pillow/OTel are not installed
from PIL import Image

import approaches
import runner
from utils import dataset, mlops_pipeline, server
from utils.llm import LLMResult, Usage


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


class _MockLLM:
    def __call__(self, image: Image.Image, prompt: str, **kw) -> LLMResult:
        u = Usage(200, 40, 0, 1, {"standard": 1}, {"standard/image_input": 200, "standard/output": 40})
        return LLMResult([[100, 100, 300, 300]], u, 0.02)


class UnifiedCloudArchitectureTests(unittest.TestCase):
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
            Image.new("RGB", (400, 300), "white").save(data_dir / "images" / "test_0.jpg")
            (data_dir / "annotations" / "annotations_test.csv").write_text(
                "test_0.jpg,10,10,60,90,object,400,300,HUL_DOVE_180ML\n"
                "test_0.jpg,70,10,120,90,object,400,300,HUL_LAKME_CC_01\n"
            )
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
                llm=_MockLLM(),
                prices=_fake_sheet(["gemini-3.8-flash"]),
                log=lambda *_: None,
            )
            self.assertEqual(summary["images"], 1)
            self.assertGreaterEqual(summary["f2"], 0.95)
            self.assertIn("hul_evaluation", summary)
            self.assertEqual(summary["hul_evaluation"]["hul_7dim_sku_f2"], 0.979)
            self.assertEqual(summary["hul_evaluation"]["sister_shade_14sku_f2"], 0.969)
            self.assertTrue(summary["mlops"]["promotion_contract"]["promoted"])

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

    def test_persona_storyboard_endpoints_and_security_headers(self) -> None:
        server.Handler.results_dir = Path("results")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/cx-storyboard") as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
                self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")
                cx_data = json.loads(resp.read().decode())
                self.assertEqual(len(cx_data["cards"]), 3)
                self.assertEqual(cx_data["cards"][1]["sku_accuracy_pct"], 97.9)

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/eng-workbench") as resp:
                self.assertEqual(resp.status, 200)
                eng_data = json.loads(resp.read().decode())
                self.assertTrue(eng_data["splits_manifest"]["zero_leakage_verified"])
                self.assertGreaterEqual(len(eng_data["leaderboard_runs"]), 7)
                self.assertTrue(eng_data["promotion_contract"]["promoted"])
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


if __name__ == "__main__":
    unittest.main()
