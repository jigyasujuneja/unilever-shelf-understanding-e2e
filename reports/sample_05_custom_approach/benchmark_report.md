# Unilever Shelf Understanding — Vertex AI Gemini Benchmark Report

## 1. Executive Summary (Tasks & Bounding-Box Separation Approaches)

| Task | Separation Approach | Model | Status | Front Facings | Depth Filtered | Latency (ms) | Input Tokens | Thinking Tokens | Output Tokens | Cost / Image ($) | Cost / Facing ($) | GT Accuracy Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `classification` | `custom_yolo_plus_gemma_verifier` | `gemini-3.8-flash` | SUCCESS | 1 | 1 | 0.1 | 220 | 30 | 85 | $0.000287 | $0.000287 | `PLACEHOLDER (Ready for GCP GT)` |

## 2. OpenTelemetry Compliance & Timing Audit

| Run ID | Trace ID | Span ID | Task | Approach | Model | Start Time (UTC) | End Time (UTC) | Total Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `custom-custom_yolo_plus_gemma_verifier-gemini-3.8-flash` | `f16e319bebbb714032302ee39e6c65e3` | `147e39aa0623c56a` | `classification` | `custom_yolo_plus_gemma_verifier` | `gemini-3.8-flash` | `2026-09-22T08:57:30.338284Z` | `2026-09-22T08:57:30.338329Z` | 335 |

## 3. Detailed Row-Level Report (7-Dimension HUL Taxonomy & Hybrid Search)

Total front-facing product rows logged across all models and approaches: **1** (full dataset in `reports/row_level_report.csv` and `reports/row_level_report.json`).

| Approach | Model | Facing # | BBox `[ymin, xmin, ymax, xmax]` | Category | Subcategory | Brand (HUL?) | Variant | Packaging | Pack Type | Rule-Derived Size | Cost/Facing ($) |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `custom_yolo_plus_gemma_verifier` | `gemini-3.8-flash` | 1 | `[535, 386, 795, 440]` | Skin Care | Face Wash | Pond's (HUL) | Bright Beauty Spot-less Glow | tube | Single | Medium / Regular (56-110g/ml) | $0.000287 |
