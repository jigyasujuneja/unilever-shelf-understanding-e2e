# SPEC-003: ShelfBench Arena — MLflow + Kaggle Benchmarking Platform (`PLATFORM_DESIGN.md`)

## 1. Architectural Intent ("Design is the New Code")
Unify **MLflow Experiment Tracking** (parameters, metrics, per-tier telemetry, artifacts, SQLite run registry) and **Kaggle Competition Benchmarking** (Public vs. Private holdout dataset splits, automated submission scoring, medal tiers, and SLA gatekeeping) into a zero-dependency, self-hosted platform (`ShelfBench Arena`).

```mermaid
flowchart TB
    subgraph SDK["1. MLflow-Style Tracking SDK (platform/registry.py)"]
        RUN["shelf_e2e.platform.log_benchmark_run()"]
        DB[("SQLite + JSON Run Store\nreports/shelfbench_arena.sqlite")]
        RUN --> DB
    end

    subgraph Evaluators["2. Kaggle-Style Public & Private Split Scorer (platform/leaderboard.py)"]
        PUB["Public Split Evaluator\n(sku110k_val_001.png + sku110k_val_002.png)"]
        PRIV["Private Holdout Split Evaluator\n(sku110k_val_003_dense147.png)"]
        PARETO["Composite Pareto Score (0..100)\n+ Hard SLA Gate (Cost <= ₹0.22, P95 <= 20s)"]
        PUB & PRIV --> PARETO
    end

    subgraph WebUI["3. ShelfBench Arena Web Platform (127.0.0.1:8765)"]
        TAB1["Tab 1: Kaggle Leaderboard\n(Public vs Private Rank, Medals, Live Run Trigger)"]
        TAB2["Tab 2: MLflow Run Explorer & Diff\n(Params, Per-Tier Latency, INR Cost Breakdown)"]
        TAB3["Tab 3: Pareto Frontier Plot\n(Cost ₹ vs Top-1 & mAP@50:95 with ₹0.22 Wall)"]
        TAB4["Tab 4: SKU-110k Visual Shelf Inspector\n(Interactive BBox Overlays & Top-5 Candidates)"]
    end

    PARETO --> DB
    DB --> WebUI
```

## 2. Composite Pareto Scoring Formula & SLA Gates
Every run is evaluated on both the **Public Split** and the **Private Holdout Split**:
- **Raw Quality Score (`0..100`):**
  $$\text{Quality} = 100 \times \left(0.35 \cdot \text{mAP}_{50:95} + 0.35 \cdot \text{Top1Acc} + 0.15 \cdot \text{MRR} + 0.15 \cdot (1 - \text{HallucinationRate})\right)$$
- **SLA Gatekeeping (`SPEC-001`):**
  - `within_cost_sla`: `estimated_cost_inr <= 0.22`
  - `within_latency_sla`: `p95_latency_ms <= 20000.0`
  - `schema_adherence == 1.0`
- **Pareto Composite Score (`0..100`):**
  $$\text{ParetoScore} = \text{Quality} - 25.0 \cdot \mathbb{I}(\text{Cost} > \text{₹}0.22) - 25.0 \cdot \mathbb{I}(\text{P95} > 20\text{s})$$
