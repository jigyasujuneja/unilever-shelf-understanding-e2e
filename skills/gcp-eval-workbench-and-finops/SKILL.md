---
name: gcp-eval-workbench-and-finops
description: >-
  Playbook for building a 100% GCS-streamed evaluation harness (zero local dataset
  or result storage), audited GCP Cloud Billing SKU unit-cost calculators, burst-traffic
  FinOps models (Serverless vs Dedicated GPUs), Locust load testing, and Dual-Persona
  (/engineering leaderboard + /exec P2P ROI simulator) Cloud Run control planes.
---

# GCP Serverless Evaluation Workbench & Burst FinOps Playbook

## 1. 100% GCS-Streamed Evaluation Architecture (Zero Local State)

1. **In-Memory Manifest & Image Streaming**:
   - Store all canonical golden datasets (`rpc.json` with `stats.split_hashes` and `catalog_sha256`) and reference galleries in Google Cloud Storage (`gs://...`).
   - Stream manifests and image crops directly into memory during evaluation runs (`cli run` or Cloud Run Jobs).
2. **Atomic Result Upload & Vertex AI Sync**:
   - Write `summary.json` and `images.jsonl` to a temporary directory, upload atomically to `gs://<bucket>/results/<run_id>/`, log parameters and 5-level metrics to **Vertex AI Experiments**, and delete local temp files.
3. **Auto-Polling Dual-Persona Control Plane (`PULL_TTL_S = 30.0`)**:
   - Run a stateless Cloud Run UI server polling GCS every 30 seconds to serve two persona-specific views from the same ground-truth artifacts:
     - **`/engineering` Portal**: Task leaderboards (`TASK-E2E`, `TASK-RET`, `TASK-CLS`, `TASK-DET`, `TASK-PLN`, `TASK-PRM`), 5-level accuracy breakdowns (`L1..L5`), `15/15` QC gate badges, and per-box shelf inspection modals.
     - **`/exec` Portal**: Executive transformation matrix, audited GCP Cloud Billing SKU cost simulator, and Pay-for-Performance (`P2P`) payout ROI simulator.

## 2. Audited GCP Cloud Billing SKU Cost Modeling & Burst FinOps (`EXP-26`)

1. **Exact Billing SKU Attribution**:
   - Compute unit economics from audited GCP Cloud Billing SKU rates rather than rough list-price guesses (e.g., `gemini-embedding-2` `768-D` Matryoshka slice SKU `330D-7FB0-87C5`, `gemini-3.5-flash-lite` input/output SKUs `8271-FF1A-6295` / `D47E-C6CE-1E1A`, and Cloud Run vCPU/RAM seconds).
2. **Diurnal Burst Traffic FinOps Formula (Serverless vs. Dedicated GPUs)**:
   - Field retail uploads arrive in sharp diurnal spikes (`60%` of `500,000` daily store photos uploaded between `17:00` and `21:00` local time, peaking at `20.8 shelves/sec`).
   - Dedicated self-hosted GPU clusters (`g2-standard-24` L4 replicas at `$50,700/month`, or 26B A100 clusters at `$96,000/month`) sit **83% idle for 20 hours/day**.
   - Combining **Stage 1 CPU ONNX (`$0` token cost)** + **`3.5x` Spatial Medoid Deduplication** + **`68%` Stage 2A `768-D` Embedding Fast-Exit (`0` generative VLM tokens)** + **Stage 1.5 Pre-Embedding Bay Gate (`-61%` background crops)** compresses serverless VLM calls by **`90.8%`**, achieving **`$0.00139/shelf` (`$21,100/month`)** and saving **`$355K/year` steady-state (`$630K/year` festive peak)**.
   - Under the linear stress cost model $C(r) = \$0.00055 + r \cdot \$0.00254$, even a **100% VLM escalation spike ($r = 1.00$)** caps at **`$0.00309/shelf`**, remaining below the **`$0.00380/shelf`** legacy ceiling.
3. **Net Cost-Negative Online Smart Sampling**:
   - Replace uniform `5%` human/pro-judge auditing with 3-tier asymmetric routing:
     - **Tier 0 (`100%` sample)**: Items triggering `0.045` Energy Vetoes, `PR-14` pack-size switches, or circuit-breaker fallbacks.
     - **Tier 1 (`20%` sample)**: Standard Stage 2B verified crops.
     - **Tier 2 (`0.2%` canary)**: High-margin Stage 2A fast-exits.
   - Reduces human/judge audit volume by **66.8%** while catching **3.1x** more production drift events.
