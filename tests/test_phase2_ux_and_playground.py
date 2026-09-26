"""End-to-end verification tests for Phase 2: 3-Tab UX Command Center, 7 POC DoD Criteria, Multi-Image / GCS Playground & Gemini Enterprise."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from utils import server


class TestPhase2UXAndPlayground(unittest.TestCase):
    """Verify all Phase 2 API endpoints, 7 POC DoD criteria, Playground batch modes, and Gemini Enterprise co-pilot."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _get_json(self, path: str) -> dict | list:
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def test_01_pilot_dod_7_criteria_and_9_row_scope_matrix(self) -> None:
        data = self._get_json("/api/v1/pilot-dod-and-scope")
        self.assertIsInstance(data, dict)
        dod = data["pilot_definition_of_done"]
        self.assertEqual(len(dod), 7)
        for item in dod:
            self.assertEqual(item["status"], "EXCEEDED")

        # Verify the 9-row Master Scope-to-Win Traceability Matrix
        matrix = data["scope_traceability_matrix"]
        self.assertEqual(len(matrix), 9)

        # Verify Riley Cloud Billing Catalog + 5-Bucket GCP Billing comparison
        cost = data["cost_demo_comparison"]["riley_cloud_billing_catalog_summary"]
        self.assertLessEqual(cost["our_coarse_to_fine_hybrid_inr_per_img"], cost["pilot_target_cap_inr_per_img"])
        self.assertLessEqual(cost["track_d_systemone_canvas_inr_per_img"], 0.22)

    def test_02_playground_6_image_single_request_marketshare_slo(self) -> None:
        res = self._post_json(
            "/api/v1/playground/analyze",
            {
                "input_mode": "multi_image_6batch",
                "workflow": "MARKETSHARE",
                "image_count": 6,
                "classification_mode": "coarse_to_fine_3task_plus_prefiltered_scann",
                "h3_packaging_entropy_gate": 0.030,
                "scann_similarity_gate": 0.82,
                "enable_9_defenses": True,
            },
        )
        slo = res["slo_verification"]
        self.assertTrue(slo["within_slo"])
        self.assertLessEqual(slo["total_e2e_latency_s"], 30.0)
        self.assertLessEqual(slo["per_image_latency_s"], 5.0)
        self.assertLessEqual(slo["cost_per_image_inr"], 0.22)
        self.assertEqual(len(res["inspected_crops"]), 6)

        # Check Crop #02 (Dove Pouch under Glare -> Soft Equivalence Group Expansion)
        pouch_crop = res["inspected_crops"][1]
        self.assertIn("SOFT_EQUIVALENCE_GROUP_EXPANSION", pouch_crop["scann_prefilter_telemetry"]["filter_mode"])
        self.assertEqual(pouch_crop["resolved_base_pack_id"], "BP-HUL-DOVE-HW-500-POUCH")

        # Check Crop #05 (Competitor Pantene -> Stops at 3-Task)
        comp_crop = res["inspected_crops"][4]
        self.assertFalse(comp_crop["coarse_3task_output"]["is_hul_brand"])
        self.assertEqual(comp_crop["scann_prefilter_telemetry"]["filter_mode"], "COMPETITOR_STOP_AT_3TASK")

    def test_03_playground_gcs_bucket_uri_batch_mode(self) -> None:
        res = self._post_json(
            "/api/v1/playground/analyze",
            {
                "input_mode": "gcs_bucket_uri",
                "workflow": "MARKETSHARE",
                "image_count": 6,
                "gcs_uri": "gs://jjuneja-fde-sandbox-shelf-images/results/",
                "enable_9_defenses": True,
            },
        )
        self.assertEqual(res["input_mode"], "gcs_bucket_uri")
        self.assertEqual(res["gcs_uri_scanned"], "gs://jjuneja-fde-sandbox-shelf-images/results/")
        self.assertEqual(len(res["gcs_objects_discovered"]), 6)

    def test_04_gemini_enterprise_agentspace_copilot_queries(self) -> None:
        for q in [
            "Which stores have non-compliant promotional Tokers or Ghost Promo OOS?",
            "Show 6-asset branded window and backlight L* compliance",
            "How did the 9 defense layers handle the rotated bottle and glare pouch?",
            "Show Mumbai West MarketShare and 6-image panorama latency",
        ]:
            res = self._post_json("/api/v1/gemini-enterprise/query", {"query": q})
            self.assertIn("openapi_tool_invoked", res)
            self.assertIn("bigquery_grounding_sql", res)
            self.assertTrue(len(res["summary_cards"]) >= 4)

    def test_05_static_assets_and_gemini_enterprise_bundle(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/", timeout=10) as r:
            html = r.read().decode("utf-8")
            self.assertIn("Unilever Shelf Intelligence", html)
            self.assertIn("#/overview", html)
            self.assertIn("#/arena", html)
            self.assertIn("#/playground", html)

        with urllib.request.urlopen(f"{self.base_url}/img/val/val_000.jpg", timeout=10) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers.get("Content-Type"), "image/jpeg")

        topo = self._get_json("/api/v1/architecture/gcp-topology")
        self.assertTrue(Path(topo["gemini_enterprise_provisioning"]["openapi_spec_path"]).is_file())
        self.assertTrue(Path(topo["gemini_enterprise_provisioning"]["provision_script_path"]).is_file())

    def test_06_hul_examples_slide_preset_and_lipton_detection(self) -> None:
        presets_resp = self._get_json("/api/v1/playground/presets")
        preset_ids = [p["id"] for p in presets_resp["presets"]]
        self.assertIn("preset_hul_scope_slide_examples", preset_ids)

        res = self._post_json(
            "/api/v1/playground/analyze",
            {
                "input_mode": "preset",
                "preset_id": "preset_hul_scope_slide_examples",
                "workflow": "MERCHANDISING_6_ASSET",
                "image_count": 1,
                "classification_mode": "coarse_to_fine_3task_plus_prefiltered_scann",
                "h3_packaging_entropy_gate": 0.030,
                "scann_similarity_gate": 0.82,
                "enable_9_defenses": True,
            },
        )
        crops = res["inspected_crops"]
        self.assertEqual(len(crops), 28)
        brands = [c["coarse_3task_output"]["slot2_brand"] for c in crops]
        for expected_brand in ("Red Label", "Lux", "Clinic Plus", "Sunsilk", "Rin", "Surf Excel", "Pond's", "Glow & Lovely", "Lakme", "Pears", "Lipton"):
            self.assertIn(expected_brand, brands)

        resolved_ids = [c["resolved_base_pack_id"] for c in crops]
        self.assertIn("POSM-HUL-LIPTON-REF-ASSET", resolved_ids)
        self.assertIn("POSM-HUL-LIPTON-WINDOW-HEADER", resolved_ids)
        self.assertIn("BP-HUL-LIPTON-GREEN-TEA-25TB", resolved_ids)
        self.assertIn("BP-HUL-LIPTON-HONEY-LEMON-25TB", resolved_ids)
        self.assertIn("BP-HUL-LIPTON-TULSI-NATURO-25TB", resolved_ids)
        self.assertIn("POSM-HUL-LAKME-PONDS-SHELF-STRIP", resolved_ids)

        # Also verify image route for the HUL Examples slide
        with urllib.request.urlopen(f"{self.base_url}/img/sku110k/sku110k_hul_examples_slide.jpg", timeout=10) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers.get("Content-Type"), "image/jpeg")

    def test_07_full_unilever_6_domain_taxonomy_and_open_vocab_basepack_synthesis(self) -> None:
        from shelf_e2e.djev_client import DjevSystemOneClient
        from shelf_e2e.taxonomy import (
            CANONICAL_PACKAGING_TYPES,
            HUL_BRANDS_CANONICAL,
            MASTER_HUL_CATALOG,
            UNILEVER_VARIANT_DOMAINS,
            resolve_or_synthesize_base_pack,
        )

        self.assertGreaterEqual(len(HUL_BRANDS_CANONICAL), 85)
        self.assertEqual(len(CANONICAL_PACKAGING_TYPES), 25)
        self.assertGreaterEqual(len(MASTER_HUL_CATALOG), 65)
        self.assertEqual(len(UNILEVER_VARIANT_DOMAINS), 7)

        djev = DjevSystemOneClient()
        # Test across all 6 Unilever Variant Domains + Novel Open-Vocabulary SKU Synthesis
        domain_cases = [
            ("Hair Care - DMT", "Indulekha", "bottle", "Bringha Selfie Comb Hair Oil", "100ml", True, "BP-HUL-INDULEKHA-BRINGHA-OIL-100ML"),
            ("Skin Care", "Minimalist", "dropper_serum", "10% Niacinamide Face Serum", "30ml", True, "BP-HUL-MINIMALIST-NIACINAMIDE-10PCT-30ML"),
            ("Oral Care", "Closeup", "box", "Everfresh Red Hot Gel Toothpaste", "150g", True, "BP-HUL-CLOSEUP-EVERFRESH-RED-150G"),
            ("Personal Wash - Laundry", "Rexona", "roll_on", "Powder Dry Underarm Roll-On", "50ml", True, "BP-HUL-REXONA-ROLLON-50ML"),
            ("Foods - Beverages", "Horlicks", "jar", "Classic Malt Health & Nutrition Drink", "500g", True, "BP-HUL-HORLICKS-CLASSIC-MALT-500G"),
            ("Non-HUL", "Tata Tea", "box", "Gold Assam Leaf Tea", "250g", False, "NON-HUL-NONH-TATATE-250G"),
        ]
        for cat, brand, pkg, var, sz, exp_hul, exp_sku in domain_cases:
            res = djev.classify_3task_and_prefilter_scann(
                box_xyxy=[100, 200, 220, 500],
                hint_category=cat,
                hint_brand=brand,
                hint_packaging=pkg,
                ocr_snippet=sz,
                hint_variant=var,
            )
            self.assertEqual(res.is_hul_brand, exp_hul)
            self.assertEqual(res.brand, brand)
            self.assertEqual(res.packaging_type, pkg)
            self.assertEqual(res.resolved_base_pack_id, exp_sku)

        # Verify Open-Vocabulary Base-Pack synthesis for a newly launched regional HUL variant
        synth_sku, cands = resolve_or_synthesize_base_pack(
            brand="Boost",
            category="Foods - Beverages",
            packaging_type="sachet_strip_ladi",
            variant="Choco Almond Malt Ladi",
            size_text="15g",
        )
        self.assertTrue(synth_sku.startswith("BP-HUL-BOOST-"))
        self.assertIn(synth_sku, cands)


if __name__ == "__main__":
    unittest.main()


