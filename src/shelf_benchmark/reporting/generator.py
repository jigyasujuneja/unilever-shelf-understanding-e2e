"""Reporting Generator for Row-Level CSV/JSON, Model/Task/Approach Summaries, Aggregate Leaderboards, and Markdown Reports.

Every number emitted here carries its provenance: the ground truth version it was scored
against, the IoU threshold in force, the pairing strategy, and the brand/product matchers.
Metrics that were never measured are emitted as empty/`None` and are skipped (never coerced
to `0.0`) when aggregating, so "not measured" can never be misread as "scored zero".
"""

from __future__ import annotations

import getpass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from shelf_benchmark.artifacts import RunArtifacts, atomic_write_text
from shelf_benchmark.models import RowLevelReportItem, TaskExecutionResult
from shelf_benchmark.reporting.aggregation import (
    LEADERBOARD_AVERAGED_METRICS,
    LEADERBOARD_SUMMED_COUNTERS,
    build_leaderboard_records,
    build_provenance,
    build_summary_records,
)
from shelf_benchmark.reporting.formatting import (
    UNKNOWN as _UNKNOWN,
)
from shelf_benchmark.reporting.formatting import (
    fmt_count as _fmt_count,
)
from shelf_benchmark.reporting.formatting import (
    fmt_float as _fmt_float,
)
from shelf_benchmark.reporting.formatting import (
    fmt_pct as _fmt_pct,
)
from shelf_benchmark.reporting.formatting import (
    fmt_usd as _fmt_usd,
)
from shelf_benchmark.reporting.formatting import (
    slugify as _slugify,
)
from shelf_benchmark.reporting.tabular import (
    SUMMARY_FIELDNAMES,
    write_leaderboard_csv,
    write_row_level_csv,
    write_row_level_json,
    write_summary_csv,
    write_summary_json,
)

__all__ = [
    "BenchmarkReportGenerator",
    "LEADERBOARD_AVERAGED_METRICS",
    "LEADERBOARD_SUMMED_COUNTERS",
    "SUMMARY_FIELDNAMES",
    "_UNKNOWN",
    "_fmt_count",
    "_fmt_float",
    "_fmt_pct",
    "_fmt_usd",
    "_slugify",
]


class BenchmarkReportGenerator:
    """Produces row-level CSV/JSON reports, summary CSV/JSON, an aggregate leaderboard, and a rich Markdown report locally and synced to GCS."""

    def __init__(
        self,
        output_dir: str | Path = "reports",
        project_id: str = "unilever-shelf-understanding",
        bucket_name: str = "unilever-shelf-understanding-shelf-images",
        sync_to_gcs: bool = True,
        gcs_reports_prefix: str = "reports",
        isolate_runs: bool = False,
        write_predictions_file: bool = False,
    ):
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        # `output_dir` stays the public attribute existing callers already read.
        self.output_dir = self.base_output_dir
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.sync_to_gcs = sync_to_gcs
        self.gcs_reports_prefix = gcs_reports_prefix.strip("/")
        self.isolate_runs = isolate_runs
        self.write_predictions_file = write_predictions_file
        # Directory the most recent `generate_all_reports` call actually wrote into.
        self.last_run_output_dir = self.base_output_dir

    @classmethod
    def from_config(cls, config: Any) -> "BenchmarkReportGenerator":
        """Builds a generator straight from a `BenchmarkConfig`, honouring `reporting.isolate_runs`.

        Accepts any object exposing `reporting`, `gcp`, and `buckets` sections so this module
        does not need to import the config module (and cannot create an import cycle).
        """
        reporting = getattr(config, "reporting", None)
        gcp = getattr(config, "gcp", None)
        buckets = getattr(config, "buckets", None)
        return cls(
            output_dir=getattr(reporting, "output_dir", "reports"),
            project_id=getattr(gcp, "project_id", "unilever-shelf-understanding"),
            bucket_name=getattr(
                buckets, "shelf_images_bucket", "unilever-shelf-understanding-shelf-images"
            ),
            sync_to_gcs=bool(getattr(reporting, "sync_reports_to_gcs", True)),
            gcs_reports_prefix=getattr(reporting, "gcs_reports_prefix", "reports"),
            isolate_runs=bool(getattr(reporting, "isolate_runs", False)),
            write_predictions_file=bool(getattr(reporting, "write_predictions_file", False)),
        )

    # -----------------------------------------------------------------
    # Output directory isolation
    # -----------------------------------------------------------------

    def _resolve_run_output_dir(self, results: Sequence[TaskExecutionResult]) -> Path:
        """Returns the directory this invocation writes into.

        When `isolate_runs` is enabled, reports land in
        `<output_dir>/<user>-<UTC timestamp>-<run_id>/` so several engineers running the
        suite concurrently against the same `output_dir` never overwrite each other.
        """
        if not self.isolate_runs:
            return self.base_output_dir

        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown-user"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = results[0].run_id if results else "no-results"
        slug = "-".join(
            _slugify(part) for part in (user, timestamp, run_id) if _slugify(part)
        )
        run_dir = self.base_output_dir / (slug or "run")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    # -----------------------------------------------------------------
    # GCS sync
    # -----------------------------------------------------------------

    def _upload_reports_to_gcs(self, local_paths: Dict[str, str]) -> Dict[str, str]:
        """Uploads all generated benchmark report artifacts to Google Cloud Storage (`gs://<bucket>/reports/`)."""
        if not self.sync_to_gcs:
            return {}
        gcs_uris: Dict[str, str] = {}
        try:
            from google.cloud import storage

            from shelf_benchmark.auth import get_gcp_credentials

            creds = get_gcp_credentials(self.project_id)
            client = storage.Client(project=self.project_id, credentials=creds)
            clean_bucket = self.bucket_name.replace("gs://", "").strip("/").split("/")[0]
            bucket = client.bucket(clean_bucket)
            for key, local_p_str in local_paths.items():
                p = Path(local_p_str)
                if not p.is_file():
                    continue
                # Preserve the per-run subdirectory so isolated runs stay isolated in GCS too.
                try:
                    rel = p.relative_to(self.base_output_dir).as_posix()
                except ValueError:
                    rel = p.name
                blob_path = f"{self.gcs_reports_prefix}/{rel}"
                try:
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(str(p))
                    gcs_uris[f"gcs_{key}"] = f"gs://{clean_bucket}/{blob_path}"
                except Exception:
                    continue
        except Exception:
            pass
        return gcs_uris

    # -----------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------

    def generate_all_reports(
        self,
        results: List[TaskExecutionResult],
    ) -> Dict[str, str]:
        """Generate all report files locally and sync them to Google Cloud Storage."""
        results = list(results or [])
        run_dir = self._resolve_run_output_dir(results)
        # Expose the directory actually written to (isolated runs differ from the base dir).
        self.last_run_output_dir = run_dir

        all_rows: List[RowLevelReportItem] = []
        for res in results:
            all_rows.extend(res.row_level_items)

        row_csv_path = run_dir / "row_level_report.csv"
        run_artifacts = RunArtifacts(
            run_dir,
            base_dir=self.output_dir,
            run_id=results[0].run_id if results else None,
        )
        row_csv_path = run_artifacts.path_for("row_level_csv")
        row_json_path = run_artifacts.path_for("row_level_json")
        summary_csv_path = run_artifacts.path_for("summary_csv")
        summary_json_path = run_artifacts.path_for("summary_json")
        leaderboard_csv_path = run_artifacts.path_for("leaderboard_csv")
        md_report_path = run_artifacts.path_for("markdown_report")

        summary_records = self._build_summary_records(results)
        leaderboard_records = self._build_leaderboard_records(results)
        provenance = self._build_provenance(results)

        self._write_row_level_csv(all_rows, row_csv_path)
        self._write_row_level_json(all_rows, row_json_path)
        self._write_summary_csv(summary_records, summary_csv_path)
        self._write_summary_json(summary_records, results, summary_json_path, leaderboard_records, provenance)
        self._write_leaderboard_csv(leaderboard_records, leaderboard_csv_path)
        self._write_markdown_report(
            summary_records,
            all_rows,
            md_report_path,
            leaderboard_records=leaderboard_records,
            provenance=provenance,
        )

        for key, target in (
            ("row_level_csv", row_csv_path),
            ("row_level_json", row_json_path),
            ("summary_csv", summary_csv_path),
            ("summary_json", summary_json_path),
            ("leaderboard_csv", leaderboard_csv_path),
            ("markdown_report", md_report_path),
        ):
            run_artifacts.record(key, target)

        artifacts = {
            "row_level_csv": str(row_csv_path),
            "row_level_json": str(row_json_path),
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "leaderboard_csv": str(leaderboard_csv_path),
            "markdown_report": str(md_report_path),
        }

        if self.write_predictions_file:
            predictions_path = run_artifacts.path_for("predictions_json")
            self._write_predictions_file(results, predictions_path)
            run_artifacts.record("predictions_json", predictions_path)
            artifacts["predictions_json"] = str(predictions_path)

        manifest_path = run_artifacts.write_manifest()
        artifacts["manifest_json"] = str(manifest_path)

        gcs_artifacts = self._upload_reports_to_gcs(artifacts)
        artifacts.update(gcs_artifacts)
        # Reported last so it is never mistaken for an uploadable file.
        artifacts["output_dir"] = str(run_dir)
        return artifacts

    # -----------------------------------------------------------------
    # Delegated writers & aggregators (`reporting.tabular` / `reporting.aggregation`)
    # -----------------------------------------------------------------

    def _write_row_level_csv(self, rows: List[RowLevelReportItem], path: Path) -> None:
        write_row_level_csv(rows, path)

    def _write_row_level_json(self, rows: List[RowLevelReportItem], path: Path) -> None:
        write_row_level_json(rows, path)

    def _write_predictions_file(
        self, results: Sequence[TaskExecutionResult], path: Path
    ) -> None:
        from shelf_benchmark.scoring import write_predictions_file as _write

        _write(list(results), path)

    def _build_summary_records(self, results: List[TaskExecutionResult]) -> List[Dict[str, Any]]:
        return build_summary_records(results)

    def _write_summary_csv(self, records: List[Dict[str, Any]], path: Path) -> None:
        write_summary_csv(records, path)

    def _write_summary_json(
        self,
        summary_records: List[Dict[str, Any]],
        full_results: List[TaskExecutionResult],
        path: Path,
        leaderboard_records: Optional[List[Dict[str, Any]]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        write_summary_json(
            summary_records,
            full_results,
            path,
            leaderboard_records=leaderboard_records,
            provenance=provenance if provenance is not None else self._build_provenance(full_results),
        )

    def _build_provenance(self, results: Sequence[TaskExecutionResult]) -> Dict[str, Any]:
        return build_provenance(results)

    def _build_leaderboard_records(
        self, results: Sequence[TaskExecutionResult]
    ) -> List[Dict[str, Any]]:
        return build_leaderboard_records(results)

    def _write_leaderboard_csv(self, records: List[Dict[str, Any]], path: Path) -> None:
        write_leaderboard_csv(records, path)

    def _write_markdown_report(
        self,
        summary_records: List[Dict[str, Any]],
        all_rows: List[RowLevelReportItem],
        path: Path,
        leaderboard_records: Optional[List[Dict[str, Any]]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        leaderboard_records = leaderboard_records or []
        provenance = provenance or {}

        lines: List[str] = [
            "# Unilever Shelf Understanding - Vertex AI Gemini Benchmark Report",
            "",
        ]
        lines.extend(self._provenance_markdown(provenance, summary_records))
        lines.extend(self._leaderboard_markdown(leaderboard_records))
        lines.extend(self._executive_summary_markdown(summary_records))
        lines.extend(self._localization_markdown(summary_records))
        lines.extend(self._cost_breakdown_markdown(summary_records))
        lines.extend(self._otel_markdown(summary_records))
        lines.extend(self._row_level_markdown(all_rows))

        atomic_write_text(path, "\n".join(lines) + "\n")

    def _provenance_markdown(
        self, provenance: Dict[str, Any], summary_records: List[Dict[str, Any]]
    ) -> List[str]:
        gt_any = bool(provenance.get("ground_truth_available_any"))
        lines: List[str] = [
            "## 1. Run Provenance (read this before quoting any number)",
            "",
            f"- Report generated (UTC): `{provenance.get('generated_at_utc', _UNKNOWN)}`",
            f"- Generated by user: `{provenance.get('generated_by_user', _UNKNOWN)}`",
            f"- Output directory: `{self.last_run_output_dir}`",
            f"- Runs included: **{provenance.get('result_count', len(summary_records))}**",
            f"- Ground truth version: `{provenance.get('gt_version', _UNKNOWN)}`",
            f"- IoU threshold in force: `{provenance.get('iou_threshold', _UNKNOWN)}`",
            f"- Pairing strategy: `{provenance.get('pairing_strategy', _UNKNOWN)}`",
            f"- Brand matcher: `{provenance.get('brand_matcher', _UNKNOWN)}`",
            f"- Product matcher: `{provenance.get('product_matcher', _UNKNOWN)}`",
            f"- Accuracy status observed: `{provenance.get('accuracy_status', _UNKNOWN)}`",
            f"- Billing rate source: `{provenance.get('billing_source', _UNKNOWN)}`",
            (
                "- Cost rates came from the live GCP price catalog: "
                f"**{'yes' if provenance.get('rates_from_live_catalog_all') else 'NO (static rate table)'}**"
            ),
            (
                "- Costs include modelled (estimated, not billed) infrastructure: "
                f"**{'YES - totals contain modelled estimates' if provenance.get('includes_modelled_infrastructure_any') else 'no - measured token cost only'}**"
            ),
            "",
        ]

        if not gt_any:
            lines.extend(
                [
                    "> [!WARNING]",
                    "> **NO GROUND TRUTH WAS AVAILABLE FOR ANY RUN IN THIS REPORT.**",
                    "> Every accuracy column below is an **unpopulated placeholder**, not a measurement.",
                    "> Empty or `n/a` cells mean *not measured*. They do **not** mean the model scored zero.",
                    "> Connect ground truth and re-score before drawing any conclusion about accuracy.",
                    "",
                ]
            )
        elif not provenance.get("ground_truth_available_all"):
            lines.extend(
                [
                    "> [!IMPORTANT]",
                    "> Ground truth was available for only **some** runs in this report.",
                    "> Runs without ground truth show `n/a` accuracy cells, meaning *not measured* rather than zero.",
                    "",
                ]
            )
        return lines

    def _leaderboard_markdown(self, leaderboard_records: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = [
            "## 2. Aggregate Leaderboard (per approach and model, averaged across images)",
            "",
            (
                "Ranked by detection F1 (descending), then product classification accuracy (descending). "
                "Averages skip images where a metric was not measured; `n` shows how many images "
                "actually contributed to each average. Full data in `leaderboard.csv`."
            ),
            "",
        ]
        if not leaderboard_records:
            lines.extend(["_No results were supplied, so the leaderboard is empty._", ""])
            return lines

        lines.extend(
            [
                "| Rank | Approach | Model | Images | Detection F1 (n) | Precision | Recall | Mean IoU (matched) | Brand Acc | Product Acc | SKU Match | Cost / Shelf Image ($) | Avg Latency (ms) | GT Version | IoU Thr |",
                "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |",
            ]
        )
        for rec in leaderboard_records:
            lines.append(
                f"| {rec['rank']} | `{rec['approach_id']}` | `{rec['model_name']}` | "
                f"{rec['images_scored']} | "
                f"{_fmt_pct(rec['detection_f1'])} ({rec['detection_f1_n_images']}) | "
                f"{_fmt_pct(rec['detection_precision'])} | {_fmt_pct(rec['detection_recall'])} | "
                f"{_fmt_float(rec['mean_iou_matched'])} | "
                f"{_fmt_pct(rec['brand_classification_accuracy'])} | "
                f"{_fmt_pct(rec['product_classification_accuracy'])} | "
                f"{_fmt_pct(rec['sku_matching_accuracy'])} | "
                f"{_fmt_usd(rec['avg_cost_per_shelf_image_usd'])} | "
                f"{_fmt_float(rec['avg_latency_ms'], 1)} | "
                f"`{rec['gt_version']}` | `{rec['iou_threshold']}` |"
            )
        lines.append("")
        return lines

    def _executive_summary_markdown(self, summary_records: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = [
            "## 3. Per-Run Execution Summary (Tasks and Bounding-Box Separation Approaches)",
            "",
            "| Task | Separation Approach | Model | Status | Front Facings | Depth Filtered | Latency (ms) | Input Tokens | Thinking Tokens | Output Tokens | All-In Cost / Image ($) | All-In Cost / Facing ($) | GT Accuracy Status |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
        for rec in summary_records:
            gt_summary = f"`{rec['accuracy_status']}` (no GT, values not measured)"
            if rec["accuracy_status"] == "EVALUATED_AGAINST_GT":
                gt_summary = (
                    f"Brand: {_fmt_pct(rec['brand_classification_accuracy'])}, "
                    f"Det F1: {_fmt_pct(rec['detection_f1'])}"
                )
            lines.append(
                f"| `{rec['task_type']}` | `{rec['separation_approach']}` | `{rec['model_name']}` | {rec['status']} | "
                f"{rec['front_facings_count']} | {rec['depth_duplicates_filtered']} | {rec['latency_ms']:.1f} | "
                f"{rec['input_tokens']} | {rec['thinking_tokens']} | {rec['output_tokens']} | "
                f"${rec['cost_per_shelf_image_usd']:.6f} | ${rec['cost_per_product_usd']:.6f} | "
                f"{gt_summary} |"
            )
        lines.append("")
        return lines

    def _localization_markdown(self, summary_records: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = [
            "## 4. Localization, Classification and Matching Quality (per run)",
            "",
            (
                "`n/a` means the metric was not measured for that run (typically no ground truth), "
                "not that it scored zero. TP/FP/FN are counted at the reported IoU threshold."
            ),
            "",
            "| Task | Approach | Model | Mean IoU (matched) | Precision | Recall | F1 | Count Acc | Brand Acc | Brand Set Recall | Product Acc | SKU Match | Planogram | TP | FP | FN | IoU Thr | Pairing | Brand Matcher | Product Matcher | GT Version | Status |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- | :--- |",
        ]
        for rec in summary_records:
            lines.append(
                f"| `{rec['task_type']}` | `{rec['separation_approach']}` | `{rec['model_name']}` | "
                f"{_fmt_float(rec['mean_iou_matched'])} | {_fmt_pct(rec['detection_precision'])} | "
                f"{_fmt_pct(rec['detection_recall'])} | {_fmt_pct(rec['detection_f1'])} | "
                f"{_fmt_pct(rec['count_accuracy'])} | {_fmt_pct(rec['brand_classification_accuracy'])} | "
                f"{_fmt_pct(rec['brand_set_recall'])} | {_fmt_pct(rec['product_classification_accuracy'])} | "
                f"{_fmt_pct(rec['sku_matching_accuracy'])} | {_fmt_pct(rec['planogram_compliance_rate'])} | "
                f"{_fmt_count(rec['true_positives'])} | {_fmt_count(rec['false_positives'])} | "
                f"{_fmt_count(rec['false_negatives'])} | "
                f"{rec['iou_threshold'] if rec['iou_threshold'] is not None else 'n/a'} | "
                f"`{rec['pairing_strategy'] or _UNKNOWN}` | `{rec['brand_matcher'] or _UNKNOWN}` | "
                f"`{rec['product_matcher'] or _UNKNOWN}` | `{rec['gt_version']}` | `{rec['accuracy_status']}` |"
            )
        lines.append("")
        return lines

    def _cost_breakdown_markdown(self, summary_records: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = [
            "## 5. Separated All-In GCP Cost Breakdown (Vertex AI PAYG vs. Provisioned Throughput GSU vs. Embeddings vs. Cloud Run vs. GCS/Observability)",
            "",
            (
                "`Modelled Infra?` reports whether the total includes estimated (not billed) "
                "infrastructure components. `Live Rates?` reports whether unit prices were pulled "
                "from the live GCP price catalog rather than a static rate table."
            ),
            "",
            "| Approach | Model | Traffic Type | Vertex AI PAYG Tokens ($) | Vertex AI Prov. Throughput GSU ($) | Embeddings and Vision API ($) | Cloud Run vCPU + RAM ($) | GCS + Cloud Logging ($) | Total All-In / Image ($) | Billing Source | Live Rates? | Modelled Infra? |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |",
        ]
        for rec in summary_records:
            lines.append(
                f"| `{rec['separation_approach']}` | `{rec['model_name']}` | `{rec['traffic_type']}` | "
                f"${rec['vertex_ai_payg_tokens_usd']:.6f} | ${rec['vertex_ai_provisioned_throughput_usd']:.6f} | "
                f"${rec['vertex_ai_embeddings_and_vision_usd']:.6f} | ${rec['cloud_run_compute_usd']:.6f} | "
                f"${rec['gcs_and_observability_usd']:.6f} | **${rec['cost_per_shelf_image_usd']:.6f}** | "
                f"`{rec['billing_source']}` | {'yes' if rec['rates_from_live_catalog'] else 'no'} | "
                f"{'yes' if rec['includes_modelled_infrastructure'] else 'no'} |"
            )
        lines.append("")
        return lines

    def _otel_markdown(self, summary_records: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = [
            "## 6. OpenTelemetry Compliance and Timing Audit",
            "",
            "| Run ID | Trace ID | Span ID | Task | Approach | Model | Start Time (UTC) | End Time (UTC) | Total Tokens |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |",
        ]
        for rec in summary_records:
            lines.append(
                f"| `{rec['run_id']}` | `{rec['trace_id']}` | `{rec['span_id']}` | "
                f"`{rec['task_type']}` | `{rec['separation_approach']}` | `{rec['model_name']}` | "
                f"`{rec['start_time']}` | `{rec['end_time']}` | {rec['total_tokens']} |"
            )
        lines.append("")
        return lines

    def _row_level_markdown(self, all_rows: List[RowLevelReportItem]) -> List[str]:
        lines: List[str] = [
            "## 7. Detailed Row-Level Report (7-Dimension HUL Taxonomy and Hybrid Search)",
            "",
            (
                f"Total front-facing product rows logged across all models and approaches: "
                f"**{len(all_rows)}** (full dataset in `row_level_report.csv` and `row_level_report.json`)."
            ),
            "",
            "| Approach | Model | Facing # | BBox `[ymin, xmin, ymax, xmax]` | Category | Subcategory | Brand (HUL?) | Variant | Packaging | Pack Type | Rule-Derived Size | GT Status | IoU | TP? | Cost/Facing ($) |",
            "| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
        ]
        for r in all_rows:
            if r.task_type != "classification":
                continue
            bbox_str = f"`[{r.bbox_ymin}, {r.bbox_xmin}, {r.bbox_ymax}, {r.bbox_xmax}]`"
            hul_tag = "HUL" if r.is_hul_brand else "Non-HUL"
            tp_str = "n/a" if r.is_true_positive is None else ("yes" if r.is_true_positive else "no")
            lines.append(
                f"| `{r.separation_approach}` | `{r.model_name}` | {r.product_index} | {bbox_str} | "
                f"{r.predicted_category} | {r.predicted_subcategory} | {r.predicted_brand} ({hul_tag}) | "
                f"{r.predicted_variant[:32]} | {r.predicted_packaging} | {r.predicted_pack_type} | "
                f"{r.rule_derived_size_bucket} | `{r.gt_status}` | {_fmt_float(r.iou_with_gt)} | "
                f"{tp_str} | ${r.cost_per_product_usd:.6f} |"
            )
        return lines


