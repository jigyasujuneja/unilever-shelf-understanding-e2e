"""Reporting Generator for Row-Level CSV/JSON, Model/Task/Approach Summaries, and Markdown Reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from shelf_benchmark.models import RowLevelReportItem, TaskExecutionResult


class BenchmarkReportGenerator:
    """Produces row-level CSV/JSON reports, summary CSV/JSON, and a rich Markdown report locally and synced to GCS."""

    def __init__(
        self,
        output_dir: str | Path = "reports",
        project_id: str = "unilever-shelf-understanding",
        bucket_name: str = "unilever-shelf-understanding-shelf-images",
        sync_to_gcs: bool = True,
        gcs_reports_prefix: str = "reports",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.sync_to_gcs = sync_to_gcs
        self.gcs_reports_prefix = gcs_reports_prefix.strip("/")

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
                if not p.exists():
                    continue
                blob_path = f"{self.gcs_reports_prefix}/{p.name}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(p))
                gcs_uris[f"gcs_{key}"] = f"gs://{clean_bucket}/{blob_path}"
        except Exception:
            pass
        return gcs_uris

    def generate_all_reports(
        self,
        results: List[TaskExecutionResult],
    ) -> Dict[str, str]:
        """Generate all report files locally and sync them to Google Cloud Storage."""
        all_rows: List[RowLevelReportItem] = []
        for res in results:
            all_rows.extend(res.row_level_items)

        row_csv_path = self.output_dir / "row_level_report.csv"
        row_json_path = self.output_dir / "row_level_report.json"
        summary_csv_path = self.output_dir / "benchmark_summary.csv"
        summary_json_path = self.output_dir / "benchmark_summary.json"
        md_report_path = self.output_dir / "benchmark_report.md"

        self._write_row_level_csv(all_rows, row_csv_path)
        self._write_row_level_json(all_rows, row_json_path)
        summary_records = self._build_summary_records(results)
        self._write_summary_csv(summary_records, summary_csv_path)
        self._write_summary_json(summary_records, results, summary_json_path)
        self._write_markdown_report(summary_records, all_rows, md_report_path)

        artifacts = {
            "row_level_csv": str(row_csv_path),
            "row_level_json": str(row_json_path),
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "markdown_report": str(md_report_path),
        }
        gcs_artifacts = self._upload_reports_to_gcs(artifacts)
        artifacts.update(gcs_artifacts)
        return artifacts

    def _write_row_level_csv(self, rows: List[RowLevelReportItem], path: Path) -> None:
        fieldnames = list(RowLevelReportItem.model_fields.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r.model_dump())

    def _write_row_level_json(self, rows: List[RowLevelReportItem], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in rows], f, indent=2)

    def _build_summary_records(self, results: List[TaskExecutionResult]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for r in results:
            records.append(
                {
                    "run_id": r.run_id,
                    "trace_id": r.trace_id,
                    "span_id": r.span_id,
                    "task_type": r.task_type,
                    "separation_approach": r.separation_approach,
                    "model_name": r.model_name,
                    "shelf_image_uri": r.shelf_image_uri,
                    "status": r.status,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "latency_ms": r.latency_ms,
                    "front_facings_count": r.cost.product_count,
                    "depth_duplicates_filtered": r.accuracy.depth_duplicates_filtered,
                    "latency_per_facing_ms": round(
                        r.latency_ms / max(r.cost.product_count, 1), 2
                    ),
                    "input_tokens": r.tokens.input_tokens,
                    "thinking_tokens": r.tokens.thinking_tokens,
                    "output_tokens": r.tokens.output_tokens,
                    "total_tokens": r.tokens.total_tokens,
                    "cost_per_shelf_image_usd": r.cost.cost_per_shelf_image_usd,
                    "cost_per_product_usd": r.cost.cost_per_product_usd,
                    "vertex_ai_payg_tokens_usd": r.cost.vertex_ai_payg_tokens_usd,
                    "vertex_ai_provisioned_throughput_usd": r.cost.vertex_ai_provisioned_throughput_usd,
                    "vertex_ai_embeddings_and_vision_usd": r.cost.vertex_ai_embeddings_and_vision_usd,
                    "cloud_run_compute_usd": r.cost.cloud_run_compute_usd,
                    "gcs_and_observability_usd": r.cost.gcs_and_observability_usd,
                    "traffic_type": r.cost.traffic_type,
                    "billing_source": r.cost.billing_source,
                    "accuracy_status": (
                        "EVALUATED_AGAINST_GT"
                        if r.accuracy.ground_truth_available
                        else r.accuracy.accuracy_status
                    ),
                    "gt_count": r.accuracy.ground_truth_count,
                    "count_accuracy": r.accuracy.count_accuracy,
                    "brand_classification_accuracy": r.accuracy.brand_classification_accuracy,
                    "product_classification_accuracy": r.accuracy.product_classification_accuracy,
                    "error_message": r.error_message,
                }
            )
        return records

    def _write_summary_csv(self, records: List[Dict[str, Any]], path: Path) -> None:
        if not records:
            return
        fieldnames = list(records[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)

    def _write_summary_json(
        self,
        summary_records: List[Dict[str, Any]],
        full_results: List[TaskExecutionResult],
        path: Path,
    ) -> None:
        payload = {
            "summary": summary_records,
            "detailed_runs": [r.model_dump() for r in full_results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _write_markdown_report(
        self,
        summary_records: List[Dict[str, Any]],
        all_rows: List[RowLevelReportItem],
        path: Path,
    ) -> None:
        lines: List[str] = [
            "# Unilever Shelf Understanding — Vertex AI Gemini Benchmark Report",
            "",
            "## 1. Executive Summary (Tasks & Bounding-Box Separation Approaches)",
            "",
            "| Task | Separation Approach | Model | Status | Front Facings | Depth Filtered | Latency (ms) | Input Tokens | Thinking Tokens | Output Tokens | All-In Cost / Image ($) | All-In Cost / Facing ($) | GT Accuracy Status |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        for rec in summary_records:
            gt_summary = "`PLACEHOLDER (Ready for GCP GT)`"
            if rec["accuracy_status"] == "EVALUATED_AGAINST_GT":
                b_acc = f"{rec['brand_classification_accuracy']*100:.1f}%" if rec["brand_classification_accuracy"] is not None else "N/A"
                gt_summary = f"Brand: {b_acc}"

            lines.append(
                f"| `{rec['task_type']}` | `{rec['separation_approach']}` | `{rec['model_name']}` | {rec['status']} | "
                f"{rec['front_facings_count']} | {rec['depth_duplicates_filtered']} | {rec['latency_ms']:.1f} | "
                f"{rec['input_tokens']} | {rec['thinking_tokens']} | {rec['output_tokens']} | "
                f"${rec['cost_per_shelf_image_usd']:.6f} | ${rec['cost_per_product_usd']:.6f} | "
                f"{gt_summary} |"
            )

        lines.extend(
            [
                "",
                "## 1B. 100% Separated All-In GCP Cost Breakdown (Vertex AI PAYG vs. Provisioned Throughput GSU vs. Embeddings vs. Cloud Run vs. GCS/Observability)",
                "",
                "| Approach | Model | Traffic Type | Vertex AI PAYG Tokens ($) | Vertex AI Prov. Throughput GSU ($) | Embeddings & Vision API ($) | Cloud Run vCPU + RAM ($) | GCS + Cloud Logging ($) | Total All-In / Image ($) | Billing Source |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ]
        )
        for rec in summary_records:
            lines.append(
                f"| `{rec['separation_approach']}` | `{rec['model_name']}` | `{rec['traffic_type']}` | "
                f"${rec['vertex_ai_payg_tokens_usd']:.6f} | ${rec['vertex_ai_provisioned_throughput_usd']:.6f} | "
                f"${rec['vertex_ai_embeddings_and_vision_usd']:.6f} | ${rec['cloud_run_compute_usd']:.6f} | "
                f"${rec['gcs_and_observability_usd']:.6f} | **${rec['cost_per_shelf_image_usd']:.6f}** | `{rec['billing_source']}` |"
            )

        lines.extend(
            [
                "",
                "## 2. OpenTelemetry Compliance & Timing Audit",
                "",
                "| Run ID | Trace ID | Span ID | Task | Approach | Model | Start Time (UTC) | End Time (UTC) | Total Tokens |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |",
            ]
        )
        for rec in summary_records:
            lines.append(
                f"| `{rec['run_id']}` | `{rec['trace_id']}` | `{rec['span_id']}` | "
                f"`{rec['task_type']}` | `{rec['separation_approach']}` | `{rec['model_name']}` | "
                f"`{rec['start_time']}` | `{rec['end_time']}` | {rec['total_tokens']} |"
            )

        lines.extend(
            [
                "",
                "## 3. Detailed Row-Level Report (7-Dimension HUL Taxonomy & Hybrid Search)",
                "",
                f"Total front-facing product rows logged across all models and approaches: **{len(all_rows)}** (full dataset in `reports/row_level_report.csv` and `reports/row_level_report.json`).",
                "",
                "| Approach | Model | Facing # | BBox `[ymin, xmin, ymax, xmax]` | Category | Subcategory | Brand (HUL?) | Variant | Packaging | Pack Type | Rule-Derived Size | Cost/Facing ($) |",
                "| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |",
            ]
        )

        for r in all_rows:
            if r.task_type == "classification":
                bbox_str = f"`[{r.bbox_ymin}, {r.bbox_xmin}, {r.bbox_ymax}, {r.bbox_xmax}]`"
                hul_tag = "HUL" if r.is_hul_brand else "Non-HUL"
                lines.append(
                    f"| `{r.separation_approach}` | `{r.model_name}` | {r.product_index} | {bbox_str} | "
                    f"{r.predicted_category} | {r.predicted_subcategory} | {r.predicted_brand} ({hul_tag}) | "
                    f"{r.predicted_variant[:32]} | {r.predicted_packaging} | {r.predicted_pack_type} | "
                    f"{r.rule_derived_size_bucket} | ${r.cost_per_product_usd:.6f} |"
                )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
