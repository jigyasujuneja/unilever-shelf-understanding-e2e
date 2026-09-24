#!/usr/bin/env python3
"""ShelfBench Arena Web Server (MLflow Run Explorer + Kaggle Leaderboard + SKU-110k Visual Inspector).

Security Compliance (`mandatory-secure-web-skills`):
- Enforces strict `Content-Security-Policy`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Cache-Control: no-store`.
- Enforces synchronizer CSRF token verification (`X-CSRF-Token`) on all state-changing `POST` requests.
- Enforces strict allow-list & `Path.resolve()` directory boundary verification for static and image assets.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import secrets
import sys
from typing import Any, Dict
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stream_open_retail_benchmarks import (
    OPEN_RETAIL_DATASETS_CATALOG,
    export_open_retail_datasets_manifest,
)
from shelf_e2e.djev_client import DjevSystemOneClient
from shelf_e2e.platform.leaderboard import KaggleLeaderboardEngine
from shelf_e2e.pricing import compute_five_bucket_gcp_billing
from shelf_e2e.taxonomy import enrich_with_7dim_taxonomy
from shelf_e2e.tracks import (
    TrackACascadingViTPipeline,
    TrackB2TwoStageCropVLMPipeline,
    TrackBEndToEndVLMPipeline,
    TrackCTieredHybridPipeline,
    TrackDJevRoutingPipeline,
    TrackEIJEPALatentWorldModelPipeline,
    TrackFPaliGemma2LoRAPipeline,
)

STATIC_DIR = (Path(__file__).resolve().parent / "static").resolve()
IMAGES_DIR = (REPO_ROOT / "data" / "sku110k" / "images").resolve()
ALLOWED_IMAGES = (
    {f"sku110k_val_{i:03d}.jpg" for i in range(20)}
    | {f"smart_retail_val_{i:03d}.jpg" for i in range(5)}
    | {"sku110k_val_001.png", "sku110k_val_002.png", "sku110k_val_003_dense147.png"}
)
CSRF_TOKEN = secrets.token_hex(24)
export_open_retail_datasets_manifest()
ENGINE = KaggleLeaderboardEngine()
ENGINE.seed_default_arena_runs()


class ShelfBenchArenaHandler(BaseHTTPRequestHandler):
    """Secure HTTP handler for ShelfBench Arena."""

    def _send_security_headers(self, content_type: str, status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; script-src 'self'; frame-ancestors 'none'; object-src 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.end_headers()

    def _send_json(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self._send_security_headers("application/json; charset=utf-8", status_code=status_code)
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            self._serve_static_file("index.html", "text/html; charset=utf-8")
            return
        if route == "/styles.css":
            self._serve_static_file("styles.css", "text/css; charset=utf-8")
            return
        if route == "/app.js":
            self._serve_static_file("app.js", "application/javascript; charset=utf-8")
            return
        if route.startswith("/images/"):
            img_name = Path(route).name
            if img_name not in ALLOWED_IMAGES:
                self._send_json({"error": "Image not in allow-list"}, status_code=404)
                return
            candidate = (IMAGES_DIR / img_name).resolve()
            if not str(candidate).startswith(str(IMAGES_DIR) + "/") or not candidate.is_file():
                self._send_json({"error": "Image not found"}, status_code=404)
                return
            mime = "image/jpeg" if img_name.endswith(".jpg") else "image/png"
            self._send_security_headers(mime, status_code=200)
            self.wfile.write(candidate.read_bytes())
            return
        if route == "/api/state":
            runs = [r.to_dict() for r in ENGINE.registry.list_runs()]
            images_list = ENGINE.slice_data.get("images", [])
            img_dims = {}
            if isinstance(images_list, list):
                for row in images_list:
                    k = row.get("file_name") or row.get("image_name") or row.get("image_id")
                    if k:
                        img_dims[k] = [row.get("width", 2336), row.get("height", 4160)]
            billing_d = compute_five_bucket_gcp_billing(
                run_id="track_d_gemini_diffusion_as_jev",
                vertex_tokens_usd=0.00028,
                embeddings_and_vision_usd=0.00166,
                image_latency_ms=4120.0,
            )
            billing_b = compute_five_bucket_gcp_billing(
                run_id="track_b_e2e_vlm",
                vertex_tokens_usd=0.00338,
                embeddings_and_vision_usd=0.00010,
                image_latency_ms=6850.0,
            )
            tax_map = {}
            for sku_id, item in ENGINE.catalog.entries.items():
                attr = enrich_with_7dim_taxonomy(
                    {
                        "brand": item.brand,
                        "category": item.category,
                        "subcategory": getattr(item, "subcategory", "General"),
                        "product_name": getattr(item, "product_name", sku_id),
                        "variant": getattr(item, "variant", "Standard"),
                        "packaging_type": getattr(item, "packaging_type", "bottle"),
                        "pack_type": getattr(item, "pack_type", "Single"),
                        "size_bucket": getattr(item, "size_bucket", ""),
                    }
                )
                tax_map[sku_id] = asdict(attr)

            djev_client = DjevSystemOneClient()
            sample_canvas = djev_client.build_djev_64token_canvas(
                box_xyxy=[120.0, 240.0, 168.0, 395.0],
                scann_top5=["BP-DOVE-BW-750", "BP-DOVE-BW-500", "BP-TRES-SH-750"],
                ocr_snippet="Dove Deep Moisture 750ml",
                glare_intensity=0.38,
            )

            self._send_json(
                {
                    "csrf_token": CSRF_TOKEN,
                    "experiment_name": "unilever-mt-500k-daily-arena",
                    "sla_targets": {
                        "max_cost_inr": 0.22,
                        "max_p95_latency_ms": 20000.0,
                        "concurrency_workers": 525,
                        "daily_volume": 500000,
                    },
                    "ground_truth_images": ENGINE.slice_data.get("images_by_name")
                    or ENGINE.slice_data.get("images", {}),
                    "image_dimensions": img_dims,
                    "taxonomy_7dim": tax_map,
                    "spec005_summary": {
                        "total_real_images": 25,
                        "total_human_boxes": 3649,
                        "depth_ghosts_suppressed": 3,
                        "front_row_facings": 3646,
                        "track_d_billing": asdict(billing_d),
                        "track_b_billing": asdict(billing_b),
                    },
                    "spec006_djev_systemone": sample_canvas.to_dict(),
                    "open_retail_datasets": [asdict(d) for d in OPEN_RETAIL_DATASETS_CATALOG],
                    "runs": runs,
                }
            )
            return

        self._send_json({"error": "Not found"}, status_code=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run-track":
            self._send_json({"error": "Endpoint not found"}, status_code=404)
            return

        client_csrf = self.headers.get("X-CSRF-Token", "")
        if not secrets.compare_digest(client_csrf, CSRF_TOKEN):
            self._send_json({"error": "Invalid CSRF token"}, status_code=403)
            return

        content_len = int(self.headers.get("Content-Length", "0"))
        if content_len <= 0 or content_len > 16384:
            self._send_json({"error": "Invalid request body size"}, status_code=400)
            return

        try:
            body = json.loads(self.rfile.read(content_len).decode("utf-8"))
        except Exception:
            self._send_json({"error": "Malformed JSON payload"}, status_code=400)
            return

        track_choice = str(body.get("track_id", "track_d_jev_routing")).strip()
        engineer_ldap = str(body.get("engineer_ldap", "jjuneja")).strip()[:32] or "jjuneja"

        track_map = {
            "track_a_cascading_vit": lambda: TrackACascadingViTPipeline(ENGINE.catalog),
            "track_b_e2e_vlm": lambda: TrackBEndToEndVLMPipeline(ENGINE.catalog),
            "track_b2_two_stage_crop_vlm": lambda: TrackB2TwoStageCropVLMPipeline(
                ENGINE.catalog, sku110k_slice_path=ENGINE.slice_path
            ),
            "track_c_tiered_hybrid": lambda: TrackCTieredHybridPipeline(
                ENGINE.catalog, sku110k_slice_path=ENGINE.slice_path
            ),
            "track_d_jev_routing": lambda: TrackDJevRoutingPipeline(
                ENGINE.catalog,
                use_gemini_diffusion_as_jev=False,
                sku110k_slice_path=ENGINE.slice_path,
            ),
            "track_d_gemini_diffusion_as_jev": lambda: TrackDJevRoutingPipeline(
                ENGINE.catalog,
                use_gemini_diffusion_as_jev=True,
                sku110k_slice_path=ENGINE.slice_path,
            ),
            "track_e_ijepa_world_model": lambda: TrackEIJEPALatentWorldModelPipeline(
                ENGINE.catalog, sku110k_slice_path=ENGINE.slice_path
            ),
            "track_f_paligemma2_lora": lambda: TrackFPaliGemma2LoRAPipeline(
                ENGINE.catalog, sku110k_slice_path=ENGINE.slice_path
            ),
        }

        factory = track_map.get(track_choice)
        if factory is None:
            self._send_json({"error": "Unsupported track_id"}, status_code=400)
            return

        new_record = ENGINE.evaluate_and_log_track(
            factory(),
            engineer_ldap=engineer_ldap,
            custom_hyperparams={"triggered_from": "ShelfBench Arena Web UI"},
        )
        self._send_json(
            {
                "status": "ok",
                "new_run": new_record.to_dict(),
                "runs": [r.to_dict() for r in ENGINE.registry.list_runs()],
            }
        )

    def _serve_static_file(self, filename: str, content_type: str) -> None:
        target = (STATIC_DIR / Path(filename).name).resolve()
        if not str(target).startswith(str(STATIC_DIR) + "/") or not target.is_file():
            self._send_json({"error": "Asset not found"}, status_code=404)
            return
        self._send_security_headers(content_type, status_code=200)
        self.wfile.write(target.read_bytes())

    def do_HEAD(self) -> None:
        self._send_security_headers("text/html; charset=utf-8", status_code=200)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    from http.server import ThreadingHTTPServer

    parser = argparse.ArgumentParser(description="Start the ShelfBench Arena Benchmarking Platform")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface (default: 0.0.0.0 for Cloudtop + localhost)",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ShelfBenchArenaHandler)
    print(
        f"ShelfBench Arena live at:\n"
        f"  -> Cloudtop URL:  http://jjuneja.c.googlers.com:{args.port}\n"
        f"  -> Localhost URL: http://127.0.0.1:{args.port}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
