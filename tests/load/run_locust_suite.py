"""Execute headless Locust load tests across 10, 40, and 80 concurrent workers and emit results/locust_load_test_report.json."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import runner  # noqa: E402
from utils import dataset  # noqa: E402
from utils.server import (  # noqa: E402
    Handler,
    ProductionThreadingHTTPServer,
    _prewarm_server_caches,
)


def _start_local_server(port: int = 8899) -> ProductionThreadingHTTPServer:
    Handler.results_dir = runner.RESULTS_DIR
    Handler.data_root = str(dataset.DEFAULT_ROOT)
    Handler.pull_src = None
    Handler.default_mode = "engineering"
    Handler._last_pull = 0.0
    _prewarm_server_caches(Handler.results_dir, Handler.data_root)
    httpd = ProductionThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def _parse_locust_csv(csv_prefix: Path) -> list[dict]:
    stats_file = Path(f"{csv_prefix}_stats.csv")
    if not stats_file.exists():
        return []
    rows = []
    with stats_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "name": r.get("Name", ""),
                "type": r.get("Type", ""),
                "requests": int(float(r.get("Request Count") or 0)),
                "failures": int(float(r.get("Failure Count") or 0)),
                "median_ms": float(r.get("Median Response Time") or 0.0),
                "avg_ms": round(float(r.get("Average Response Time") or 0.0), 2),
                "p50_ms": float(r.get("50%") or 0.0),
                "p90_ms": float(r.get("90%") or 0.0),
                "p95_ms": float(r.get("95%") or 0.0),
                "p99_ms": float(r.get("99%") or 0.0),
                "max_ms": round(float(r.get("Max Response Time") or 0.0), 2),
                "rps": round(float(r.get("Requests/s") or 0.0), 2),
            })
    return rows


def main() -> None:
    port = 8899
    httpd = _start_local_server(port=port)
    time.sleep(0.5)

    tiers = [
        {"tier": "Tier 1: Daytime Steady State (10 Concurrent Workers)", "users": 10, "spawn_rate": 10, "run_time": "6s"},
        {"tier": "Tier 2: Evening Kirana Peak (40 Concurrent Workers)", "users": 40, "spawn_rate": 40, "run_time": "6s"},
        {"tier": "Tier 3: Max Cloud Run Pool Burst (80 Concurrent Workers)", "users": 80, "spawn_rate": 80, "run_time": "6s"},
    ]

    report_tiers = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for cfg in tiers:
            u = cfg["users"]
            csv_prefix = Path(tmpdir) / f"locust_u{u}"
            cmd = [
                "uv",
                "run",
                "--with",
                "locust",
                "locust",
                "-f",
                str(ROOT / "tests/load/locustfile.py"),
                "--headless",
                "--host",
                f"http://127.0.0.1:{port}",
                "-u",
                str(u),
                "-r",
                str(cfg["spawn_rate"]),
                "-t",
                cfg["run_time"],
                "--csv",
                str(csv_prefix),
            ]
            print(f"Running Locust: {cfg['tier']} ...")
            subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, text=True)
            rows = _parse_locust_csv(csv_prefix)
            agg = next((r for r in rows if r["name"] == "Aggregated"), {})
            report_tiers.append({
                **cfg,
                "aggregated": agg,
                "endpoints": [r for r in rows if r["name"] != "Aggregated"],
            })

    httpd.shutdown()

    out_report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "locust_version": "2.46.7",
        "target_daily_volume": {
            "baseline_daily_images": 506531,
            "peak_daily_images": 800000,
            "evening_peak_window_ist": "17:00-21:00 IST",
            "evening_peak_required_rps": 40.0,
        },
        "tiers": report_tiers,
        "end_to_end_pipeline_latency_bridge": {
            "single_thread_sequential_benchmark_s": 94.6,
            "medoid_grouped_sequential_s": 29.5,
            "batched_8crop_contact_sheet_p50_s": 2.69,
            "batched_8crop_contact_sheet_p95_s": 6.40,
            "option_1_fast_0vlm_p50_s": 0.90,
            "option_1_fast_0vlm_p95_s": 1.85,
        },
    }
    out_path = ROOT / "results" / "locust_load_test_report.json"
    out_path.write_text(json.dumps(out_report, indent=2))
    print(f"Saved Locust load test report to {out_path}")
    for t in report_tiers:
        a = t["aggregated"]
        print(
            f"  {t['tier']}: {a.get('requests')} reqs | {a.get('failures')} fails | "
            f"RPS={a.get('rps')} | P50={a.get('p50_ms')}ms | P95={a.get('p95_ms')}ms | P99={a.get('p99_ms')}ms"
        )


if __name__ == "__main__":
    main()
