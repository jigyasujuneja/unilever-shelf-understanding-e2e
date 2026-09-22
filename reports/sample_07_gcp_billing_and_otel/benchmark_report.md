# Unilever Shelf Understanding — Vertex AI Gemini Benchmark Report

## 1. Executive Summary (Tasks & Bounding-Box Separation Approaches)

| Task | Separation Approach | Model | Status | Front Facings | Depth Filtered | Latency (ms) | Input Tokens | Thinking Tokens | Output Tokens | All-In Cost / Image ($) | All-In Cost / Facing ($) | GT Accuracy Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `detection` | `single_pass_full_shelf` | `gemini-2.5-flash` | ERROR | 0 | 0 | 2205.6 | 0 | 0 | 0 | $0.000141 | $0.000141 | `PLACEHOLDER (Ready for GCP GT)` |

## 1B. 100% Separated All-In GCP Cost Breakdown (Vertex AI PAYG vs. Provisioned Throughput GSU vs. Embeddings vs. Cloud Run vs. GCS/Observability)

| Approach | Model | Traffic Type | Vertex AI PAYG Tokens ($) | Vertex AI Prov. Throughput GSU ($) | Embeddings & Vision API ($) | Cloud Run vCPU + RAM ($) | GCS + Cloud Logging ($) | Total All-In / Image ($) | Billing Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `single_pass_full_shelf` | `gemini-2.5-flash` | `ON_DEMAND` | $0.000000 | $0.000000 | $0.000000 | $0.000128 | $0.000012 | **$0.000141** | `gcp_cloud_billing_catalog_live` |

## 2. OpenTelemetry Compliance & Timing Audit

| Run ID | Trace ID | Span ID | Task | Approach | Model | Start Time (UTC) | End Time (UTC) | Total Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `run-10eef81b` | `351f00e416f8e3522f55c63413aa7139` | `633e776f94f5ae92` | `detection` | `single_pass_full_shelf` | `gemini-2.5-flash` | `2026-09-22T09:51:56.769937Z` | `2026-09-22T09:51:58.975525Z` | 0 |

## 3. Detailed Row-Level Report (7-Dimension HUL Taxonomy & Hybrid Search)

Total front-facing product rows logged across all models and approaches: **0** (full dataset in `reports/row_level_report.csv` and `reports/row_level_report.json`).

| Approach | Model | Facing # | BBox `[ymin, xmin, ymax, xmax]` | Category | Subcategory | Brand (HUL?) | Variant | Packaging | Pack Type | Rule-Derived Size | Cost/Facing ($) |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
