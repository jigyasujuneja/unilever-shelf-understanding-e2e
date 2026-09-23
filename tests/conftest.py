"""Pytest configuration and shared fixtures for the Shelf Understanding Benchmark Suite.

Enforces two properties across all `@pytest.mark.offline` tests:
1. Working directory is pinned to the repository root so relative config paths always resolve.
2. Live GCP network exporters (Cloud Logging, GCS report/OTel sync, and live Cloud Billing Catalog
   HTTP calls) are stubbed out so offline tests run in <2 seconds and never touch real GCP
   resources on a credentialed developer workstation.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _pin_repo_root_and_enforce_offline(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Any:
    prev_cwd = Path.cwd()
    os.chdir(REPO_ROOT)

    is_offline = request.node.get_closest_marker("offline") is not None
    if is_offline and not os.environ.get("SHELF_BENCH_ALLOW_LIVE_IN_TESTS"):
        from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine
        from shelf_benchmark.reporting.generator import BenchmarkReportGenerator
        from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger

        orig_init = OpenTelemetryBenchmarkLogger.__init__

        def _isolated_otel_init(self: Any, config: Any, *args: Any, **kwargs: Any) -> None:
            if getattr(config, "otel_log_path", "") == "reports/otel_logs.jsonl":
                config.otel_log_path = str(tmp_path / "otel_logs.jsonl")
            orig_init(self, config, *args, **kwargs)

        monkeypatch.setattr(OpenTelemetryBenchmarkLogger, "__init__", _isolated_otel_init)
        monkeypatch.setattr(
            OpenTelemetryBenchmarkLogger,
            "_export_to_cloud_logging",
            lambda self, *args, **kwargs: None,
        )
        monkeypatch.setattr(
            OpenTelemetryBenchmarkLogger,
            "_sync_to_gcs",
            lambda self: None,
        )
        monkeypatch.setattr(
            BenchmarkReportGenerator,
            "_upload_reports_to_gcs",
            lambda self, local_paths: {},
        )
        monkeypatch.setattr(
            GCPBillingAndCostEngine,
            "fetch_live_gcp_sku_pricing",
            lambda self, model_name="gemini-3.8-flash": {
                "rates_found": False,
                "reason": "offline_test_lane",
            },
        )
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@pytest.fixture
def offline_harness(tmp_path: Path):
    """Ready-to-use offline `ShelfBenchmarkSDK` with a universal stub model registered."""
    from shelf_benchmark.testing import benchmark_harness

    return benchmark_harness(tmp_path)
