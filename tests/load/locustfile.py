"""Locust end-to-end load test suite for Unilever Shelf Understanding v0.2 (`PR06-PR08` Cloud Run workers).

Exercises five production traffic classes matching `Sales EDGE`, `Shikhar`, and `MTPC` workloads:
  1. Option 1 Fast Path (`option_1_fast_clm`): Stage 1 CPU Box Finder + Neighbor Medoid Grouping + Stage 2A Vector Catalog Lookup (`CLM` 0-VLM).
  2. Option 2 Hybrid Primary (`option_2_hybrid_clef`): Stage 1 CPU + 60% Stage 2A Vector Gate + 40% 8-Crop Batched Zoom Sheet (`CLEF`) + Layer-5 Pack-Size Verifier.
  3. Merchandising Bay-Scoped E2E (`merchandising_bay_e2e`): Dual-Class Bay/Pack Detection + Stage 1.5 Pre-Embedding Class-0 Bay Gate (`-61%` RPCs) + Planogram Purity.
  4. Level-4 Total Contract Timeout Quarantine (`simulate_timeout=True`): Verifies `PT180S` timeouts return `RETRY_PENDING` with `earning_override=None` (never `Earning='0'`).
  5. Control-Plane Health & Telemetry APIs (`/api/v2/health`, `/api/leaderboard`, `/api/exec-overview`).
"""

from __future__ import annotations

from locust import HttpUser, between, task


class UnileverShelfStoreWorker(HttpUser):
    """Simulates concurrent store photo uploads arriving via PR05 Backend -> PR19 Cloud Pub/Sub -> PR06-PR08 Cloud Run."""

    wait_time = between(0.01, 0.05)

    @task(5)
    def analyze_marketshare_option2_hybrid(self) -> None:
        """Primary Production Default (Option 2: Smart 2-Step Zoom & Crown Hybrid, 35 facings -> 11 medoid groups)."""
        with self.client.post(
            "/api/v2/shelf/analyze",
            json={"option": "option_2_hybrid_clef", "facings_count": 35},
            name="POST /api/v2/shelf/analyze [Option 2 Hybrid CLEF]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            data = resp.json()
            if data.get("contract_status") != "OK" or not data.get("layer5_pack_size_corrected"):
                resp.failure(f"Unexpected payload: {data}")

    @task(3)
    def analyze_kirana_option1_fast_clm(self) -> None:
        """Volume & Evening Peak Spillover Path (Option 1: Fast Vector Catalog Match, 0 VLM calls)."""
        with self.client.post(
            "/api/v2/shelf/analyze",
            json={"option": "option_1_fast_clm", "facings_count": 35},
            name="POST /api/v2/shelf/analyze [Option 1 Fast 0-VLM CLM]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            data = resp.json()
            if data.get("contract_status") != "OK" or data.get("vlm_contact_sheet_batches") != 0:
                resp.failure(f"Expected 0 VLM batches in Option 1: {data}")

    @task(3)
    def analyze_merchandising_bay_gated(self) -> None:
        """Merchandising Display-Bay E2E (Class-0 Pre-Embedding Bay Gate pruning out-of-bay clutter)."""
        with self.client.post(
            "/api/v2/shelf/analyze",
            json={"option": "merchandising_bay_e2e", "facings_count": 35},
            name="POST /api/v2/shelf/analyze [Merchandising Class-0 Bay Gate]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            data = resp.json()
            if data.get("scope_mode") != "class0_bay_scoped":
                resp.failure(f"Expected class0_bay_scoped: {data}")

    @task(1)
    def verify_p2p_timeout_quarantine_gate(self) -> None:
        """Level-4 Total Contract Payout Protection Gate (verifies PT180S timeouts emit RETRY_PENDING, never Earning=0)."""
        with self.client.post(
            "/api/v2/shelf/analyze",
            json={"option": "option_2_hybrid_clef", "simulate_timeout": True},
            name="POST /api/v2/shelf/analyze [Level-4 Timeout Quarantine Gate]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            data = resp.json()
            if data.get("contract_status") != "RETRY_PENDING" or data.get("earning_override") is not None:
                resp.failure(f"Payout gate violation: {data}")

    @task(2)
    def check_worker_health_and_leaderboard(self) -> None:
        """Cloud Run readiness probe and cached leaderboard telemetry fetch."""
        with self.client.get("/api/v2/health", name="GET /api/v2/health", catch_response=True) as resp:
            if resp.status_code != 200 or resp.json().get("status") != "healthy":
                resp.failure(f"Health probe failed: {resp.status_code}")
        with self.client.get("/api/leaderboard", name="GET /api/leaderboard", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Leaderboard failed: {resp.status_code}")
