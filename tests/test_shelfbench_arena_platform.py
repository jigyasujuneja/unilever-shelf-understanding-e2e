"""Unit & Security Verification Tests for ShelfBench Arena (MLflow Registry + Kaggle Leaderboard + HTTP Server)."""

from __future__ import annotations

from http.server import HTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shelf_e2e.platform.leaderboard import KaggleLeaderboardEngine
from shelf_e2e.platform.server import ShelfBenchArenaHandler


class TestShelfBenchArenaPlatform(unittest.TestCase):
    def test_01_kaggle_public_and_private_splits_and_mlflow_sqlite(self) -> None:
        engine = KaggleLeaderboardEngine()
        runs = engine.seed_default_arena_runs()
        self.assertGreaterEqual(len(runs), 5)

        # Top ranked run should be Track D (Jev or GeminiDiffusion-as-Jev) with GOLD medal
        top_run = runs[0]
        self.assertEqual(top_run.medal_tier, "GOLD")
        self.assertEqual(top_run.public_metrics["top1_acc"], 1.0)
        self.assertEqual(top_run.private_metrics["top1_acc"], 1.0)
        self.assertTrue(top_run.meets_all_slas)
        self.assertLessEqual(top_run.cost_breakdown["cost_per_image_inr"], 0.22)

    def test_02_http_server_security_headers_and_csrf_enforcement(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), ShelfBenchArenaHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # 1. Verify GET /api/state returns security headers & CSRF token
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")
                self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
                self.assertIn("default-src 'self'", resp.headers.get("Content-Security-Policy", ""))
                state_data = json.loads(resp.read().decode("utf-8"))
                csrf_token = state_data["csrf_token"]
                self.assertTrue(len(csrf_token) >= 32)

            # 2. Verify POST /api/run-track WITHOUT CSRF token is rejected with 403
            bad_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/run-track",
                data=json.dumps({"track_id": "track_d_jev_routing"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(bad_req, timeout=5)
            self.assertEqual(ctx.exception.code, 403)

            # 3. Verify POST /api/run-track WITH valid CSRF token logs a new MLflow run
            good_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/run-track",
                data=json.dumps(
                    {"track_id": "track_d_gemini_diffusion_as_jev", "engineer_ldap": "jjuneja"}
                ).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-CSRF-Token": csrf_token,
                },
                method="POST",
            )
            with urllib.request.urlopen(good_req, timeout=20) as resp2:
                self.assertEqual(resp2.status, 200)
                res_json = json.loads(resp2.read().decode("utf-8"))
                self.assertEqual(res_json["status"], "ok")
                self.assertEqual(res_json["new_run"]["medal_tier"], "GOLD")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
