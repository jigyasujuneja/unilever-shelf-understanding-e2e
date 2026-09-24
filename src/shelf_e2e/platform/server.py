#!/usr/bin/env python3
"""ShelfBench Arena Web Server (MLflow Run Explorer + Kaggle Leaderboard + SKU-110k Visual Inspector).

Security Compliance (`mandatory-secure-web-skills`):
- Binds strictly to `127.0.0.1` (`localhost`), NEVER `0.0.0.0`.
- Enforces strict `Content-Security-Policy`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Cache-Control: no-store`.
- Enforces synchronizer CSRF token verification (`X-CSRF-Token`) on all state-changing `POST` requests.
- Enforces strict allow-list & `Path.resolve()` directory boundary verification for static and image assets.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
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

from shelf_e2e.platform.leaderboard import KaggleLeaderboardEngine
from shelf_e2e.tracks import (
    TrackACascadingViTPipeline,
    TrackBEndToEndVLMPipeline,
    TrackCTieredHybridPipeline,
    TrackDJevRoutingPipeline,
)

STATIC_DIR = (Path(__file__).resolve().parent / "static").resolve()
IMAGES_DIR = (REPO_ROOT / "data" / "sku110k" / "images").resolve()
ALLOWED_IMAGES = {
    "sku110k_val_001.png",
    "sku110k_val_002.png",
    "sku110k_val_003_dense147.png",
}
CSRF_TOKEN = secrets.token_hex(24)
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
            self._send_security_headers("image/png", status_code=200)
            self.wfile.write(candidate.read_bytes())
            return
        if route == "/api/state":
            runs = [r.to_dict() for r in ENGINE.registry.list_runs()]
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
                    "ground_truth_images": ENGINE.slice_data.get("images", {}),
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

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the ShelfBench Arena Benchmarking Platform")
    parser.add_argument("--port", type=int, default=8765, help="Port on 127.0.0.1 (default: 8765)")
    args = parser.parse_args()

    # Strictly bind to 127.0.0.1 per mandatory-secure-web-skills
    server = HTTPServer(("127.0.0.1", args.port), ShelfBenchArenaHandler)
    print(f"ShelfBench Arena (MLflow + Kaggle Platform) live at: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
