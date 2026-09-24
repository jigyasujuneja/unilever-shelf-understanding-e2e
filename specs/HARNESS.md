# SPEC-002: Evaluation & Benchmarking Harness Contract (`HARNESS.md`)

## 1. Metrics to Calculate

### A. Computer Vision & Retrieval KPIs
- **Detection:**
  - `mAP@50` and `mAP@50:95` calculated against `SKU-110k` ground-truth bounding boxes (`[x1, y1, x2, y2]`).
  - `IoU` distribution (`P50`, `P90`).
- **Classification / Catalog Matching:**
  - `Top-1 Accuracy`: % of detected crops where nearest neighbor `base_pack_id` == ground truth.
  - `Top-5 Recall`: % where ground truth is within top-5 candidates.
  - `MRR (Mean Reciprocal Rank)`: Reciprocal rank score across catalog retrievals.
  - `Precision`, `Recall`, and `F1-Score` per product category.
- **Compliance & Reasoning:**
  - `Toker Compliance F1`: Precision/Recall/F1 on binary promotional pass/fail flags (`COMPLIANT` vs `NON_COMPLIANT`).
  - `JSON Schema Adherence`: % of runs returning 100% valid schema without fallback parsing.
  - `Hallucination Rate`: % of detected `base_pack_id`s not present in the `RPC` master catalog.

### B. Operational & Economic KPIs
- **Latency (`ms`):** `P50`, `P90`, `P95`, and `P99` latency measured per tier (`tier1_detection`, `tier2_catalog_match`, `tier3_compliance`, `total_e2e`) across continuous requests.
- **Concurrency Stress:** Sustained `P95` latency under `100`, `250`, and `525` concurrent workers.
- **Unit Cost (`$` and `₹`):**
  - Token cost calculated at Vertex AI `Gemini 2.5 Flash Lite` rates (`$0.075 / 1M` input tokens, `$0.30 / 1M` output tokens).
  - Compute cost calculated at Cloud Run `L4 GPU` / `vCPU` hourly rates divided by throughput.
  - Vector search query cost calculated at Vertex AI Vector Search (`ScaNN`) QPS rates.
  - Blended cost converted at `1 USD = 84 INR` and checked against the hard ceiling `<= ₹0.22`.
