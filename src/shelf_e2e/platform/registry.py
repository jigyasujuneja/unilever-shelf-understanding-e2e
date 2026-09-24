"""MLflow-inspired Experiment, Hyperparameter, Metric, and Artifact Run Registry (SQLite + JSON)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ExperimentRunRecord:
    """Single logged benchmark run (MLflow Run + Kaggle Submission Metadata)."""

    run_id: str
    experiment_name: str
    track_id: str
    track_name: str
    engineer_ldap: str
    timestamp_utc: str
    hyperparameters: Dict[str, Any]
    public_metrics: Dict[str, float]
    private_metrics: Dict[str, float]
    latency_tiers_ms: Dict[str, float]
    cost_breakdown: Dict[str, float]
    pareto_score: float
    medal_tier: str
    meets_all_slas: bool
    predictions_by_image: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MLflowRunRegistry:
    """SQLite-backed experiment run store with automatic JSON snapshot export."""

    def __init__(self, db_path: Optional[Path] = None):
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        reports_dir = repo_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (reports_dir / "shelfbench_arena.sqlite")
        self.json_snapshot_path = reports_dir / "arena_runs_history.json"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    track_name TEXT NOT NULL,
                    engineer_ldap TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    pareto_score REAL NOT NULL,
                    medal_tier TEXT NOT NULL,
                    meets_all_slas INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def log_run(self, record: ExperimentRunRecord) -> ExperimentRunRecord:
        payload_str = json.dumps(record.to_dict())
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiment_runs (
                    run_id, experiment_name, track_id, track_name, engineer_ldap,
                    timestamp_utc, pareto_score, medal_tier, meets_all_slas, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.experiment_name,
                    record.track_id,
                    record.track_name,
                    record.engineer_ldap,
                    record.timestamp_utc,
                    float(record.pareto_score),
                    record.medal_tier,
                    1 if record.meets_all_slas else 0,
                    payload_str,
                ),
            )
            conn.commit()
        self._export_json_snapshot()
        return record

    def list_runs(self, experiment_name: Optional[str] = None) -> List[ExperimentRunRecord]:
        with sqlite3.connect(str(self.db_path)) as conn:
            if experiment_name:
                cursor = conn.execute(
                    "SELECT payload_json FROM experiment_runs WHERE experiment_name = ? ORDER BY pareto_score DESC, timestamp_utc DESC",
                    (experiment_name,),
                )
            else:
                cursor = conn.execute(
                    "SELECT payload_json FROM experiment_runs ORDER BY pareto_score DESC, timestamp_utc DESC"
                )
            rows = cursor.fetchall()
        return [ExperimentRunRecord(**json.loads(r[0])) for r in rows]

    def get_run(self, run_id: str) -> Optional[ExperimentRunRecord]:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT payload_json FROM experiment_runs WHERE run_id = ?", (run_id,)
            )
            row = cursor.fetchone()
        return ExperimentRunRecord(**json.loads(row[0])) if row else None

    def clear_runs(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM experiment_runs")
            conn.commit()
        self._export_json_snapshot()

    def _export_json_snapshot(self) -> None:
        runs = [r.to_dict() for r in self.list_runs()]
        self.json_snapshot_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")

    @staticmethod
    def new_run_id(track_id: str) -> str:
        short_uuid = uuid.uuid4().hex[:6]
        return f"run-{track_id[:14]}-{short_uuid}"

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
