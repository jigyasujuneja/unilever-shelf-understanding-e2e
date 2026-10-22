"""Unit and integration verification for v0.2 production endpoints and Locust load test report."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import runner
from utils import dataset
from utils.server import Handler, ProductionThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


def test_v2_shelf_analyze_and_timeout_quarantine_gate():
    Handler.results_dir = runner.RESULTS_DIR
    Handler.data_root = str(dataset.DEFAULT_ROOT)
    Handler.pull_src = None
    Handler.default_mode = "engineering"
    httpd = ProductionThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        # 1. Health check
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v2/health") as resp:
            assert resp.status == 200
            h = json.loads(resp.read().decode())
            assert h["status"] == "healthy"
            assert h["catalog_skus"] == 2659

        # 2. Option 2 Hybrid CLEF
        req2 = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v2/shelf/analyze",
            data=json.dumps({"option": "option_2_hybrid_clef", "facings_count": 35}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req2) as resp:
            assert resp.status == 200
            d2 = json.loads(resp.read().decode())
            assert d2["contract_status"] == "OK"
            assert d2["catalog_skus"] == 2659
            assert len(d2["catalog_sha256"]) == 16
            assert d2["medoid_groups"] < d2["raw_facings"]
            assert d2["layer5_verified_sku"] == 540
            assert d2["layer5_pack_size_corrected"] is True

        # 3. Option 1 Fast 0-VLM
        req1 = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v2/shelf/analyze",
            data=json.dumps({"option": "option_1_fast_clm", "facings_count": 35}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req1) as resp:
            assert resp.status == 200
            d1 = json.loads(resp.read().decode())
            assert d1["contract_status"] == "OK"
            assert d1["vlm_contact_sheet_batches"] == 0

        # 4. Level-4 Total Contract Timeout Quarantine Gate
        req_to = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v2/shelf/analyze",
            data=json.dumps({"option": "option_2_hybrid_clef", "simulate_timeout": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_to) as resp:
            assert resp.status == 200
            dto = json.loads(resp.read().decode())
            assert dto["contract_status"] == "RETRY_PENDING"
            assert dto["earning_override"] is None
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_locust_load_test_report_zero_failures():
    report_path = ROOT / "results" / "locust_load_test_report.json"
    assert report_path.exists(), "Missing results/locust_load_test_report.json"
    report = json.loads(report_path.read_text())
    tiers = report.get("tiers") or []
    assert len(tiers) == 3
    for tier in tiers:
        agg = tier["aggregated"]
        assert agg["requests"] > 0
        assert agg["failures"] == 0, f"Expected 0 failures in {tier['tier']}, got {agg['failures']}"
