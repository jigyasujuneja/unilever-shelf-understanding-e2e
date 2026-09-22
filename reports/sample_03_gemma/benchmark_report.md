# Unilever Shelf Understanding — Vertex AI Gemini Benchmark Report

## 1. Executive Summary (Tasks & Bounding-Box Separation Approaches)

| Task | Separation Approach | Model | Status | Front Facings | Depth Filtered | Latency (ms) | Input Tokens | Thinking Tokens | Output Tokens | Cost / Image ($) | Cost / Facing ($) | GT Accuracy Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `classification` | `single_pass_full_shelf` | `gemma-3-27b-it-vllm` | SUCCESS | 2 | 0 | 40.3 | 640 | 0 | 210 | $0.000102 | $0.000051 | `PLACEHOLDER (Ready for GCP GT)` |

## 2. OpenTelemetry Compliance & Timing Audit

| Run ID | Trace ID | Span ID | Task | Approach | Model | Start Time (UTC) | End Time (UTC) | Total Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `run-b6b81978` | `d2ae3868fb507752d30cdae29bfebbc3` | `cce55e98e3125918` | `classification` | `single_pass_full_shelf` | `gemma-3-27b-it-vllm` | `2026-09-22T08:57:28.748617Z` | `2026-09-22T08:57:28.788954Z` | 850 |

## 3. Detailed Row-Level Report (7-Dimension HUL Taxonomy & Hybrid Search)

Total front-facing product rows logged across all models and approaches: **2** (full dataset in `reports/row_level_report.csv` and `reports/row_level_report.json`).

| Approach | Model | Facing # | BBox `[ymin, xmin, ymax, xmax]` | Category | Subcategory | Brand (HUL?) | Variant | Packaging | Pack Type | Rule-Derived Size | Cost/Facing ($) |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `single_pass_full_shelf` | `gemma-3-27b-it-vllm` | 1 | `[535, 386, 795, 440]` | Skin Care | Face Wash | Pond's (HUL) | Bright Beauty Spot-less Glow | tube | Single | Medium / Regular (56-110g/ml) | $0.000051 |
| `single_pass_full_shelf` | `gemma-3-27b-it-vllm` | 2 | `[171, 676, 487, 813]` | Oral Care | Toothpaste | Pepsodent (HUL) | Germi Check | box | Single | Large / Family (>110g/ml) | $0.000051 |
