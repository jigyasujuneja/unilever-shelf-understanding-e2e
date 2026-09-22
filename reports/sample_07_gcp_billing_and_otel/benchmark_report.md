# Unilever Shelf Understanding — Vertex AI Gemini Benchmark Report

## 1. Executive Summary (Tasks & Bounding-Box Separation Approaches)

| Task | Separation Approach | Model | Status | Front Facings | Depth Filtered | Latency (ms) | Input Tokens | Thinking Tokens | Output Tokens | All-In Cost / Image ($) | All-In Cost / Facing ($) | GT Accuracy Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `detection` | `single_pass_facing_nms` | `gemini-3.8-flash` | SUCCESS | 17 | 1 | 23455.1 | 1977 | 1533 | 2202 | $0.007931 | $0.000467 | `PLACEHOLDER (Ready for GCP GT)` |
| `detection` | `single_pass_facing_nms` | `gemini-3.7-flash` | SUCCESS | 17 | 0 | 19917.5 | 1977 | 1281 | 2051 | $0.006084 | $0.000358 | `PLACEHOLDER (Ready for GCP GT)` |
| `detection` | `single_pass_facing_nms` | `gemini-3.5-flash-lite` | SUCCESS | 20 | 0 | 16530.1 | 1977 | 0 | 2424 | $0.002139 | $0.000107 | `PLACEHOLDER (Ready for GCP GT)` |

## 1B. 100% Separated All-In GCP Cost Breakdown (Vertex AI PAYG vs. Provisioned Throughput GSU vs. Embeddings vs. Cloud Run vs. GCS/Observability)

| Approach | Model | Traffic Type | Vertex AI PAYG Tokens ($) | Vertex AI Prov. Throughput GSU ($) | Embeddings & Vision API ($) | Cloud Run vCPU + RAM ($) | GCS + Cloud Logging ($) | Total All-In / Image ($) | Billing Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `single_pass_facing_nms` | `gemini-3.8-flash` | `ON_DEMAND` | $0.006558 | $0.000000 | $0.000000 | $0.001361 | $0.000012 | **$0.007931** | `gcp_cloud_billing_catalog_live` |
| `single_pass_facing_nms` | `gemini-3.7-flash` | `ON_DEMAND` | $0.004916 | $0.000000 | $0.000000 | $0.001156 | $0.000012 | **$0.006084** | `gcp_cloud_billing_catalog_live` |
| `single_pass_facing_nms` | `gemini-3.5-flash-lite` | `ON_DEMAND` | $0.001167 | $0.000000 | $0.000000 | $0.000959 | $0.000012 | **$0.002139** | `gcp_cloud_billing_catalog_live` |

## 2. OpenTelemetry Compliance & Timing Audit

| Run ID | Trace ID | Span ID | Task | Approach | Model | Start Time (UTC) | End Time (UTC) | Total Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `run-3a047a33` | `368415f8700309e398cf41368f4364c8` | `65004e938592aa6e` | `detection` | `single_pass_facing_nms` | `gemini-3.8-flash` | `2026-09-22T13:51:55.037844Z` | `2026-09-22T13:52:18.492906Z` | 5712 |
| `run-838d18aa` | `4e94ce195a7a9e7cc87f78599408ebb1` | `aa79b1fb96a3b3a9` | `detection` | `single_pass_facing_nms` | `gemini-3.7-flash` | `2026-09-22T13:52:21.631775Z` | `2026-09-22T13:52:41.549306Z` | 5309 |
| `run-1d1ef420` | `ef28b894373aba6587e28bc642fe681f` | `848731d59e3b995e` | `detection` | `single_pass_facing_nms` | `gemini-3.5-flash-lite` | `2026-09-22T13:52:44.028063Z` | `2026-09-22T13:53:00.558156Z` | 4401 |

## 3. Detailed Row-Level Report (7-Dimension HUL Taxonomy & Hybrid Search)

Total front-facing product rows logged across all models and approaches: **54** (full dataset in `reports/row_level_report.csv` and `reports/row_level_report.json`).

| Approach | Model | Facing # | BBox `[ymin, xmin, ymax, xmax]` | Category | Subcategory | Brand (HUL?) | Variant | Packaging | Pack Type | Rule-Derived Size | Cost/Facing ($) |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
