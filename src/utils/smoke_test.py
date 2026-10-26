"""CLEF Contrastive Contact-Sheet Cascade Smoke Test Runner for Google Cloud Run.

Samples 5 crop images and 5 shelf images directly from GCS manifests,
starts an in-container HTTP REST server, and verifies the endpoints:
- GET /api/predict/health
- POST /api/predict/crop
- POST /api/predict/shelf
Writes the JSON report to GCS and prints to stdout.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from utils import dataset

logger = logging.getLogger("smoke_test")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _write_output_payload(payload_str: str, path_or_uri: str) -> None:
    if path_or_uri.startswith("gs://"):
        bucket, blob_name = dataset._split_gs(path_or_uri)
        dataset._gcs().bucket(bucket).blob(blob_name).upload_from_string(
            payload_str, content_type="application/json"
        )
    else:
        p = Path(path_or_uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload_str, encoding="utf-8")


def _http_request(url: str, method: str = "GET", data: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    body_bytes = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as he:
        raw_err = he.read().decode("utf-8")
        try:
            err_json = json.loads(raw_err)
        except Exception:
            err_json = {"error": raw_err}
        return he.code, err_json


def run_smoke_test_clef(
    crops_n: int = 5,
    shelves_n: int = 5,
    service_url: str | None = None,
    output_uri: str = "gs://ul-fs-s-sandbx-16-prj-shelf-images/results/clef_contrastive_smoke_test.json",
    log=print,
) -> dict:
    """Execute end-to-end smoke test on 5 crops and 5 shelves."""
    log(f"Starting CLEF Contrastive Contact-Sheet Smoke Test (crops={crops_n}, shelves={shelves_n}) ...")

    # 1. Sample crop images from hul_golden (test split)
    log("Sampling crop image URIs from hul_golden (test split) ...")
    crop_samples = dataset.sample_images(
        split="test",
        limit=crops_n,
        seed=0,
        root=dataset.data_root("hul_golden"),
        name="hul_golden",
    )
    crop_uris = [s.path for s in crop_samples[:crops_n]]
    log(f"  Found {len(crop_uris)} crop URIs: {crop_uris}")

    # 2. Sample shelf images from gtmerch_golden (test split)
    log("Sampling shelf image URIs from gtmerch_golden (test split) ...")
    shelf_samples = dataset.sample_images(
        split="test",
        limit=shelves_n,
        seed=0,
        root=dataset.data_root("gtmerch_golden"),
        name="gtmerch_golden",
    )
    shelf_uris = [s.path for s in shelf_samples[:shelves_n]]
    log(f"  Found {len(shelf_uris)} shelf URIs: {shelf_uris}")

    # 3. Setup test endpoint (local in-container server or remote service)
    server_thread = None
    httpd = None
    if service_url:
        base_url = service_url.rstrip("/")
        log(f"Using remote Cloud Run service URL: {base_url}")
    else:
        from utils.server import Handler, ThreadingHTTPServer

        port = _find_free_port()
        base_url = f"http://127.0.0.1:{port}"
        log(f"Starting in-container HTTP server on {base_url} ...")
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        time.sleep(1.0)

    crop_results: list[dict] = []
    shelf_results: list[dict] = []

    try:
        # Verify /api/predict/health
        log(f"Testing GET {base_url}/api/predict/health ...")
        status, health_resp = _http_request(f"{base_url}/api/predict/health")
        log(f"  Health status: {status} -> {health_resp}")

        # 4. Test crop predictions
        log(f"\n--- Testing {len(crop_uris)} Crop Predictions (POST /api/predict/crop) ---")
        for idx, uri in enumerate(crop_uris, 1):
            log(f"[{idx}/{len(crop_uris)}] Predicting crop: {uri} ...")
            t0 = time.perf_counter()
            payload = {
                "image_uri": uri,
                "model": "gemini-3.5-flash-lite",
                "tier": "priority",
            }
            status, resp = _http_request(f"{base_url}/api/predict/crop", method="POST", data=payload)
            dur = time.perf_counter() - t0
            sku_id = resp.get("sku_id")
            pred = resp.get("prediction") or {}
            stage = resp.get("stage")
            conf = resp.get("confidence")
            log(f"  Status {status} in {dur:.2f}s | SKU #{sku_id} | Brand={pred.get('brand')} | "
                f"Variant={pred.get('base_variant')} | Stage={stage} (conf={conf})")
            crop_results.append({
                "index": idx,
                "image_uri": uri,
                "status": status,
                "sku_id": sku_id,
                "prediction": pred,
                "stage": stage,
                "confidence": conf,
                "top1_cosine": resp.get("top1_cosine"),
                "latency_s": round(dur, 3),
            })

        # 5. Test shelf predictions
        log(f"\n--- Testing {len(shelf_uris)} Shelf Predictions (POST /api/predict/shelf) ---")
        for idx, uri in enumerate(shelf_uris, 1):
            log(f"[{idx}/{len(shelf_uris)}] Predicting shelf: {uri} ...")
            t0 = time.perf_counter()
            payload = {
                "image_uri": uri,
                "model": "gemini-3.5-flash-lite",
                "tier": "priority",
            }
            status, resp = _http_request(f"{base_url}/api/predict/shelf", method="POST", data=payload)
            dur = time.perf_counter() - t0
            total_facings = resp.get("total_facings", 0)
            hul_facings = resp.get("hul_facings", 0)
            sos = resp.get("hul_facing_sos", 0.0)
            fg_count = resp.get("stage2a_fast_gate_count", 0)
            vlm_count = resp.get("stage2b_vlm_count", 0)
            log(f"  Status {status} in {dur:.2f}s | Facings: {total_facings} total ({hul_facings} HUL, SoS={sos:.2%}) | "
                f"FastGate={fg_count} / VLM={vlm_count}")
            shelf_results.append({
                "index": idx,
                "image_uri": uri,
                "status": status,
                "total_facings": total_facings,
                "hul_facings": hul_facings,
                "competitor_facings": resp.get("competitor_facings", 0),
                "hul_facing_sos": sos,
                "hul_area_sos": resp.get("hul_area_sos", 0.0),
                "stage2a_fast_gate_count": fg_count,
                "stage2b_vlm_count": vlm_count,
                "brand_counts": resp.get("brand_counts", {}),
                "category_counts": resp.get("category_counts", {}),
                "sample_detections": (resp.get("detections") or [])[:5],
                "latency_s": round(dur, 3),
            })

    finally:
        if httpd:
            httpd.shutdown()

    # 6. Build summary report
    crops_passed = sum(1 for c in crop_results if c.get("status") == 200 and c.get("sku_id") is not None)
    shelves_passed = sum(1 for s in shelf_results if s.get("status") == 200 and s.get("total_facings", 0) > 0)
    all_passed = (crops_passed == len(crop_results) > 0) and (shelves_passed == len(shelf_results) > 0)

    report = {
        "status": "PASS" if all_passed else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": "clef_contrastive_contact_sheet",
        "approach": "clef_contrastive_metadata",
        "model": "gemini-3.5-flash-lite",
        "tier": "priority",
        "tested_endpoint": base_url,
        "summary": {
            "crops_tested": len(crop_results),
            "crops_passed": crops_passed,
            "shelves_tested": len(shelf_results),
            "shelves_passed": shelves_passed,
            "total_facings_detected": sum(s.get("total_facings", 0) for s in shelf_results),
            "avg_crop_latency_s": round(
                sum(c.get("latency_s", 0) for c in crop_results) / max(1, len(crop_results)), 3
            ),
            "avg_shelf_latency_s": round(
                sum(s.get("latency_s", 0) for s in shelf_results) / max(1, len(shelf_results)), 3
            ),
        },
        "crop_predictions": crop_results,
        "shelf_predictions": shelf_results,
    }

    # 7. Write report to output_uri
    if output_uri:
        log(f"\nWriting smoke test report to {output_uri} ...")
        _write_output_payload(json.dumps(report, indent=2), output_uri)
        log("  Report written successfully.")

    return report


# Backward-compatible aliases
run_smoke_test = run_smoke_test_clef
run_smoke_test_exp28 = run_smoke_test_clef
