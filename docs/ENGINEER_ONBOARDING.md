# Step-by-Step Engineering Onboarding Guide

Welcome to the **Unilever Shelf Understanding with Computer Vision & Multimodal
LLMs** repository! This guide is designed specifically for **new engineers who
have never seen this repository before**.

---

## 1. Visual Step-by-Step Onboarding Journey

Follow the 5 stages below in order. Every stage has a clear verification check
before moving to the next.

<!-- mdformat off(reason: preserving Mermaid onboarding journey diagram) -->
```mermaid
flowchart TD
    subgraph Stage0["Stage 0: 2-Minute Mental Model (Understand the Problem)"]
        S0_A["Input: 1..N Retail Shelf Photos<br/>(shelf-image.png or gs:// URIs)"]
        S0_B["3 Cumulative Use Cases:<br/>UC1: Bounding Boxes [ymin,xmin,ymax,xmax]<br/>UC2: Brand Classification (Closed / Open-Vocab)<br/>UC3: 7-Dim Attributes + Catalog SKU Match"]
        S0_C["8 Competing CV/VLM Paths<br/>(Path 1 Monolithic .. Path 8 Critic Cascade)<br/>× 2 Gemini Models = 16 Benchmark Runs"]
        S0_A --> S0_B --> S0_C
    end

    subgraph Stage1["Stage 1: Local Environment & Zero-Cost Verification (3 Mins)"]
        S1_A["Activate Virtualenv:<br/>source .venv/bin/activate"]
        S1_B["Run Offline Unit Test Suite (86 Tests, $0 Cost):<br/>.venv/bin/pytest -q"]
        S1_C["Run CLI Offline Smoke Test:<br/>.venv/bin/shelf-benchmark run"]
        S1_A --> S1_B --> S1_C
    end

    subgraph Stage2["Stage 2: Launch & Explore the Interactive UI (5 Mins)"]
        S2_A["Start Local Dashboard Server:<br/>.venv/bin/python ui/server.py"]
        S2_B["Open http://127.0.0.1:8080 in Browser"]
        S2_C["Explore Tabs:<br/>• Tab 1: 16-Run Matrix & Cost/Latency<br/>• Tabs 2-4: UC1, UC2, UC3 Canvas & Crops<br/>• Tab 5: Live Execution Studio"]
        S2_A --> S2_B --> S2_C
    end

    subgraph Stage3["Stage 3: Plug-and-Play Scaling (1..N Images & Ground Truth)"]
        S3_A["Add More Shelf Images:<br/>Edit configs/shelf_associations.json"]
        S3_B["Add / Update Ground Truth:<br/>Edit configs/sample_ground_truth.json"]
        S3_C["Re-Score Instantly Without Re-Running LLMs:<br/>.venv/bin/shelf-benchmark score --summary reports/benchmark_summary.json --gt configs/sample_ground_truth.json"]
        S3_A --> S3_B --> S3_C
    end

    subgraph Stage4["Stage 4: Build & Benchmark a New Approach on GCP Cloud Run"]
        S4_A["1. Add enum in src/shelf_benchmark/config.py<br/>2. Add handler in src/shelf_benchmark/runner.py<br/>3. Register card in ui/static/app.js"]
        S4_B["Run Live Benchmark on GCP Cloud Run:<br/>.venv/bin/shelf-benchmark cloud-run --approaches all --model gemini-3.8-flash,gemini-3.5-flash-lite"]
        S4_C["Inspect Updated Leaderboard & Row-Level Reports in reports/ and UI"]
        S4_A --> S4_B --> S4_C
    end

    Stage0 --> Stage1 --> Stage2 --> Stage3 --> Stage4
```
<!-- mdformat on -->

---

## 2. Quick-Reference Decision Map: "Which File Do I Touch?"

When you are asked to make a change, use this decision tree to jump straight to
the exact file responsible:

<!-- mdformat off(reason: preserving Mermaid file decision tree diagram) -->
```mermaid
flowchart LR
    START(["What do you want to do?"])

    START -->|"Add or modify shelf images (1..N)<br/>or candidate SKUs"| F_ASSOC["configs/shelf_associations.json<br/>src/shelf_benchmark/data/associations.py"]
    START -->|"Add or update Ground Truth<br/>bounding boxes & 7-Dim labels"| F_GT["configs/sample_ground_truth.json<br/>src/shelf_benchmark/data/ground_truth.py"]
    START -->|"Tune LLM system prompts or<br/>JSON output schemas"| F_PROMPT["src/shelf_benchmark/prompts/registry.py"]
    START -->|"Modify Gemini API calls,<br/>retries, or token tracking"| F_VERTEX["src/shelf_benchmark/models/vertex_client.py"]
    START -->|"Add a new CV / VLM<br/>pipeline approach (Path 9)"| F_RUNNER["1. src/shelf_benchmark/config.py (SeparationApproach)<br/>2. src/shelf_benchmark/runner.py (_execute_inference)<br/>3. ui/static/app.js (USE_CASE_PATHS)"]
    START -->|"Change IoU matching threshold,<br/>AP@0.50, or 7-Dim scoring math"| F_EVAL["src/shelf_benchmark/evaluation/metrics.py"]
    START -->|"Update UI visuals, canvas<br/>overlays, or Live Studio controls"| F_UI["ui/server.py<br/>ui/static/index.html<br/>ui/static/app.js"]
```
<!-- mdformat on -->

---

## 3. Step-by-Step Commands for Your First 15 Minutes

### Step 1: Verify Your Environment (`$0` API Cost)

Run the automated test suite. By default, tests use `VertexAIClient(offline_mode=True)`
so they never require live GCP credentials or spend Vertex AI quota:

```bash
.venv/bin/pytest -q
```

### Step 2: Launch the Visual Inspection UI (`http://127.0.0.1:8080`)

Start the dashboard server to inspect the 16 pre-computed Cloud Run runs
(`8 paths × 2 Gemini models`) stored in `reports/`:

```bash
.venv/bin/python ui/server.py
```

* **What to look for in the UI**:
  * **Tab 1 (`Executive Overview & Architecture`)**: Compares all 16 runs side
    by side on Latency (`s`), Estimated Cost (`$`), Detected Front Facings,
    Detection F1 / `AP@0.50` / `mAP@[0.50:0.95]`, and 7-Dimension Attribute
    Accuracy. Use the **Evaluated Shelf Image Manifest (`1..N`)** dropdown to
    filter across images.
  * **Tabs 2, 3, and 4 (`Use Case 1`, `Use Case 2`, `Use Case 3`)**: Click any
    bounding box on the shelf photo or any thumbnail in the crop strip to see
    its coordinates, confidence, matched Ground Truth IoU, and all 7 taxonomy
    attributes (`category`, `subcategory`, `brand`, `variant`, `packaging`,
    `pack_type`, `size`).
  * **Tab 5 (`Live Execution Studio`)**: Execute any of the 8 paths in real time
    using **Offline Unit-Test Stub**, **Local Non-Cloud-Run Process**, or
    **Remote GCP Cloud Run Service**.

### Step 3: Run a Live Benchmark Against GCP Cloud Run

To execute all 8 approaches across both `gemini-3.8-flash` and
`gemini-3.5-flash-lite` on the remote GCP Cloud Run service and refresh
`reports/`:

```bash
.venv/bin/shelf-benchmark cloud-run \
  --approaches all \
  --model gemini-3.8-flash,gemini-3.5-flash-lite
```

### Step 4: Re-Score Existing Runs When New Ground Truth Arrives

When new Ground Truth annotations are added to `configs/sample_ground_truth.json`,
you do **not** need to re-run Vertex AI inference. Run:

```bash
.venv/bin/shelf-benchmark score \
  --summary reports/benchmark_summary.json \
  --gt configs/sample_ground_truth.json
```
