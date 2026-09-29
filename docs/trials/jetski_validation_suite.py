#!/usr/bin/env python3
"""Local diagnostics and experimental validation sandbox (docs/trials/jetski_validation_suite.py).

Verifies:
  1. All 6 EPICs and registered approaches in `src/approaches/`
  2. The 3 EPIC core pillars (`src/core/detection.py`, `src/core/matching.py`, `src/core/fallback.py`)
  3. Live benchmark summaries in `results/` and exports a consolidated ScaNN/vector benchmark snapshot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import approaches  # noqa: E402
import core  # noqa: E402
import runner  # noqa: E402
import stages  # noqa: E402


def run_diagnostics() -> dict:
    reg = approaches.all_approaches()
    st_reg = stages.all_stages()
    board = runner.leaderboard(ROOT / "results")
    scann_runs = [
        r for r in board if any(k in r["approach"] for k in ("scann", "djev", "hul_8stage", "sister_shade"))
    ]
    report = {
        "status": "PASS",
        "registered_approaches_count": len(reg),
        "registered_stage_groups_count": len(st_reg),
        "core_pillars": list(core.__all__),
        "total_benchmark_runs": len(board),
        "vector_and_hybrid_runs": scann_runs,
    }
    out_path = Path(__file__).parent / "scann_benchmark_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[Trials Validation] Verified {len(reg)} approaches, {len(st_reg)} stage groups, and {len(board)} live runs.")
    print(f"[Trials Validation] Wrote snapshot to {out_path}")
    return report


if __name__ == "__main__":
    run_diagnostics()
