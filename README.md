# Unilever Retail Shelf Understanding and Merchandising Benchmark (`shelf-bench` v2.2)

> 🚀 **Partner Quick-Start Guide**: See [docs/Partner_CLEF_Contrastive_QuickStart_Guide.md](docs/Partner_CLEF_Contrastive_QuickStart_Guide.md) for copy-pasteable `curl` and Python snippets to query the deployed CLEF Contrastive Contact-Sheet Cascade Cloud Run API (`clef-contrastive-api`) for individual crops (`POST /api/predict/crop`) and full shelves (`POST /api/predict/shelf`).

End-to-end computer vision and multimodal benchmark for **Unilever Market Share (Share of Shelf)** and **In-Store Merchandising Compliance** across Indian General Trade (Kirana) and Modern Trade supermarkets. All datasets, embedding caches, model registries, and experiment results run **100% on Google Cloud Platform (Google Cloud Storage, Vertex AI, BigQuery, Cloud SQL, and Cloud Run)** without storing datasets or results on local machines.

## 1. Executive Summary: What We Built (4 Bullets)

1. **8 Leakage-Free Unilever Golden Datasets in GCS (`2,745` Products, `12/12` Quality Checks Passed):** Consolidated raw store shelf photos, 2-class display bay and product annotations (`gtmerch`), and clean client full-shelf images (`gs://ms-json-dataset`, after filtering out images with pre-drawn red or green debug boxes) into 8 standardized GCS manifests (`gs://marketshare-datasets/derived_manifests/<dataset>/rpc.json`) with **0 `Unknown` categories or packaging types**, **0 `Others` placeholder labels**, and **100% catalog-to-box parity**.
2. **Two-Stage Shelf Pipeline (`₹0.095 / shelf` when Grouping Identical Side-by-Side Packs, `96.2%` Golden Accuracy):**
   - **Stage 1 (`distilled_yolo_onnx`):** CPU ONNX shelf detector (`<100 ms`, `₹0` LLM token cost) that separates branded display bays (`Class 0`) from individual product facings (`Class 1`).
   - **Stage 2A (`clm_standalone` / Embedding Brand Filter):** `gemini-embedding-2` (`₹0.0111/crop`) with white-balance store lighting correction and an embedding confidence filter that resolves **68% of shelf facings without calling an LLM**.
   - **Stage 2B (`hierarchical_clm_jev`):** 4-crop contact-sheet verification (`gemini-3.5-flash-lite` at `₹0.0674/call`) restricted to the matched brand's ~15 sister variants using neighboring shelf products as context.
3. **19-Experiment Benchmark Catalog, Sequential Phase-2 Roadmap (`EXP-20..25`), and 6 Vertex AI Registered Models:** Evaluated all 19 POC experiments (`P0` Core, `P1` Scale, `P2` Fast-Follow, and `Moonshot`) across validation and test splits with audited GCP Cloud Billing SKU rates (`330D-7FB0-87C5`, `8271-FF1A-6295`, and `D47E-C6CE-1E1A`), documented in [`docs/experiments_decision_log.md`](docs/experiments_decision_log.md), and synced all runs and models to Vertex AI Experiments and Model Registry (`us-central1`).
4. **Cloud Run Dashboards (`/engineering` and `/exec`) Backed 100% by GCS:** Deployed an interactive Engineering Leaderboard (`https://perfect-store-control-plane-jbn2qgbina-uc.a.run.app/engineering`, showing Task IDs, plain-English model descriptions, run durations, unit-labeled `₹/crop` vs `₹/shelf` costs, and 12-check data quality status) alongside an Executive Summary & Partner Readout Portal (`https://perfect-store-control-plane-jbn2qgbina-uc.a.run.app/exec`, with a live rollout ROI simulator, 5-level accuracy breakdown, bounding-box shelf inspector, and next week's Decision Maker vs. Partner readout playbook in [`docs/next_week_readout_playbook.md`](docs/next_week_readout_playbook.md)).

## 2. System Architecture and 100% GCS Data Flow

### A. Two-Stage Shelf Inference Pipeline
```mermaid
flowchart LR
    IMG["Store Shelf Photo\n(Kirana / Modern Trade)"] --> S1["Stage 1: CPU ONNX Detector\n• Class 0: Branded Display Bays\n• Class 1: Product Facings\n(₹0 Token Cost · <100 ms)"]
    S1 --> S2A{"Stage 2A: Embedding Filter\ngemini-embedding-2 + Confidence Gate\n(₹0.0111 / crop)"}
    S2A -->|"High Confidence (~68%)"| OUT["5-Level Product & Share of Shelf\nL1 Unilever vs Rival -> L2 Cat/Pack\n-> L3 Brand -> L4 Variant -> L5 SKU"]
    S2A -->|"Ambiguous Sister Variant (~32%)"| S2B["Stage 2B: 4-Crop Contact Sheet\ngemini-3.5-flash-lite Verifier\n(₹0.0674 / sheet + Neighbor Context)"]
    S2B --> OUT
```

### B. 100% GCS Experiment Execution and Live Dashboard Flow (Zero Local Storage)
```mermaid
flowchart TD
    subgraph GCS["Google Cloud Storage (Single Source of Truth)"]
        M["gs://marketshare-datasets/derived_manifests/\n• 8 Golden Datasets (rpc.json)\n• brand_sku_live_registry.json\n• tuning/ SFT JSONL"]
        R["gs://ul-fs-s-sandbx-16-prj-shelf-images/results/\n• <run_id>/summary.json & images.jsonl\n• experiments_registry.json\n• qc_audit_report.json"]
    end

    subgraph Compute["Developer CLI or Cloud Run Job"]
        CLI["shelf-bench run / cloud-run\n1. Streams rpc.json + images from GCS\n2. Logs run to Vertex AI Experiments\n3. Uploads summary.json + images.jsonl to GCS"]
    end

    subgraph Dash["Cloud Run Control Plane (Auto-Syncs Every 30s)"]
        ENG["/engineering Portal\n• Task Leaderboards (TASK-E2E..TASK-PRM)\n• Scannable 19-Experiment Table\n• 12-Gate QC & Vertex Model Registry"]
        EXEC["/exec Executive Summary\n• Audited GCP SKU Cost & ROI Simulator\n• Next Week Readout (Execs vs Partners)\n• 5-Level Taxonomy & Shelf Proof Modal"]
    end

    M -->|"Streamed in-memory"| CLI
    CLI -->|"Writes directly to gs://.../results/"| R
    R -->|"Polled every 30s (PULL_TTL_S)"| ENG
    R -->|"Polled every 30s (PULL_TTL_S)"| EXEC
```

## 3. Team Guide: Which Dataset to Use and How to Load It

Use the **8 Unilever Golden Datasets** listed below. All 8 stream directly from `gs://marketshare-datasets/derived_manifests/<dataset>/rpc.json` into memory with `SHA-256` split hashes (`stats.split_hashes`), **0 `Unknown` categories or packaging types**, **0 `Others` placeholder labels**, and **100% catalog-to-label parity**. Legacy public sets (`sku110k`, `rpc`, `products`, and `shelves`) are hidden behind an opt-in toggle in the UI and should not be used for Unilever accuracy or cost reporting.

| What You Are Building or Testing | Dataset Key (`-d`) | Task ID & Name | Standard Splits (`train / val / test`) | Classes | Canonical GCS Manifest Location |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full-Shelf Share of Shelf (SOS) on Kirana / Modern Trade** | **`gtmerch_golden`** | `TASK-E2E` (`end_to_end`) | `186` / `72` / `100` | `2,745` | `gs://marketshare-datasets/derived_manifests/gtmerch_golden/rpc.json` |
| **Full-Shelf SOS on Clean Client LabelMe Shelves** | **`ms_json_golden`** | `TASK-E2E` (`end_to_end`) | `38` / `25` / `27` | `2,745` | `gs://marketshare-datasets/derived_manifests/ms_json_golden/rpc.json` |
| **Multi-Photo Store Session Stitching & SOS** | **`hul_sessions`** | `TASK-E2E` / `TASK-PLN` | `30` / `20` / `25` | `2,745` | `gs://marketshare-datasets/derived_manifests/hul_sessions/rpc.json` |
| **Crop-Level 2,745-Product Retrieval or Classification** | **`hul_golden`** | `TASK-RET` / `TASK-CLS` | `16,000` / `1,900` / `1,907` | `2,745` (`1,108` HUL, `1,637` Competitor) | `gs://marketshare-datasets/derived_manifests/hul_golden/rpc.json` |
| **Fine-Grained Sister-Variant Disambiguation** | **`hul_variants`** | `TASK-RET` / `TASK-CLS` | `15,208` / `1,900` / `1,907` | `1,962` | `gs://marketshare-datasets/derived_manifests/hul_variants/rpc.json` |
| **Unilever vs Competitor Brand & M&A Ownership** | **`hul_brands`** | `TASK-CLS` (`classification`) | `8,000` / `1,000` / `1,000` | `953` | `gs://marketshare-datasets/derived_manifests/hul_brands/rpc.json` |
| **Packaging Form Factor × Category** | **`hul_packs`** | `TASK-CLS` (`classification`) | `6,000` / `1,000` / `1,000` | `18` | `gs://marketshare-datasets/derived_manifests/hul_packs/rpc.json` |
| **Promo Banner, Multipack & Price-Off Detection** | **`hul_promos`** | `TASK-PRM` (`promo_detection`) | `4,000` / `1,000` / `982` | `240` | `gs://marketshare-datasets/derived_manifests/hul_promos/rpc.json` |

### Loading Any Golden Dataset in Python (Streams Directly from GCS)
```python
from utils import dataset

# Streams rpc.json directly from gs://marketshare-datasets/derived_manifests/gtmerch_golden/rpc.json
samples = dataset.sample_images(
    split="test",
    limit=15,
    seed=0,
    root=dataset.data_root("gtmerch_golden"),
    name="gtmerch_golden",
)

# Inspect 5-level product ground truth on the first sample
print(samples[0].image_id, len(samples[0].boxes), samples[0].labels[0])
```

## 4. Team Guide: How to Set Up Experiments, Write to GCS, and Update the Dashboard

### Step 1: Choose or Build an Approach (`src/approaches/`)
| Task ID & Name | Embedding-Only Approach (`₹0` LLM Spend) | Highest-Accuracy Hybrid Approach | Plain-English Summary |
| :--- | :--- | :--- | :--- |
| **`TASK-E2E` (`end_to_end`)** | `shelf_clm_standalone` (`gemini-embedding-2`) | `shelf_hierarchical_clm_jev` (`gemini-3.5-flash-lite`) | CPU ONNX shelf detector (`Class 0` bays vs `Class 1` products) + Embedding Brand Filter (`68%` early exit) + 4-crop Gemini contact sheet. |
| **`TASK-RET` (`retrieval`)** | `clm_standalone` (`gemini-embedding-2`) | `hierarchical_clm_jev` / `l8_hul_ultimate` (`gemini-3.5-flash-lite`) | Matches each crop against the `2,745`-product gallery using embeddings and confidence filtering before optional contact-sheet verification. |
| **`TASK-CLS` (`classification`)** | `clm_standalone_classify` (`gemini-embedding-2`) | `djev_classify` (`gemini-3.5-flash-lite`) | Predicts all 5 catalog levels (`L1` Unilever/Rival -> `L2` Cat/Pack -> `L3` Brand -> `L4` Variant -> `L5` Product) with token probability calibration. |
| **`TASK-DET` (`detection`)** | `distilled_yolo_onnx` (`yolo11n-distilled-onnx`) | `distilled_yolo_onnx` (`yolo11n-distilled-onnx`) | Dual-head CPU ONNX detector (`<100 ms`, `₹0` token cost) trained on `gtmerch_golden` and `ms_json_golden`. |
| **`TASK-PLN` (`planogram_compliance`)** | None (requires bay & gap reasoning) | `bay_planogram_verifier` (`gemini-3.5-flash-lite`) | Checks `Class 0` branded bay purity, brand-block cohesion, horizontal out-of-stock shelf gaps, and Share of Shelf (`SOS`). |
| **`TASK-PRM` (`promo_detection`)** | `clm_promo_matcher` (`gemini-embedding-2`) | `jev_promo_verifier` (`gemini-3.5-flash-lite`) | Uses color-band cues and structured Gemini verification to read price-off bands, multipacks, and freebie callouts. |

To create a new approach:
1. Add `src/approaches/<use_case>/<task>/<your_approach>.py` subclassing `Approach` (from `src/approaches/base.py`) and register it with `@register` in `src/approaches/__init__.py`.
2. Add its plain-English title and 1-line summary to `APPROACH_INTUITIONS` (and optionally register an Experiment ID in `EXPERIMENT_DEFINITIONS`) inside [`src/utils/experiment_matrix.py`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/utils/experiment_matrix.py).

### Step 2: Run the Experiment Directly into GCS (Zero Local Files Left Behind)
Pass `--results gs://ul-fs-s-sandbx-16-prj-shelf-images/results` (or use `cloud-run`). `runner.py` writes `images.jsonl` and `summary.json` in a temporary directory, uploads them directly to GCS, logs the run to **Vertex AI Experiments**, and deletes the temporary directory:

```bash
# Option A: Run from terminal directly into GCS (zero local files kept)
PYTHONPATH=src .venv/bin/python -m cli run \
  -a shelf_hierarchical_clm_jev \
  -m gemini-3.5-flash-lite \
  -d gtmerch_golden \
  --split test --limit 15 \
  --results gs://ul-fs-s-sandbx-16-prj-shelf-images/results

# Option B: Run in parallel as a Cloud Run Job (reads & writes 100% in GCS)
PYTHONPATH=src .venv/bin/python -m cli cloud-run \
  -a shelf_hierarchical_clm_jev \
  -m gemini-3.5-flash-lite \
  -t priority \
  --split test --limit 50
```

If you run locally into `results/` during debugging, push everything to GCS and clean your local disk in one command:
```bash
PYTHONPATH=src .venv/bin/python -m cli qc --verify-all
PYTHONPATH=src .venv/bin/python -m cli experiments --sync-gcp
PYTHONPATH=src .venv/bin/python -m cli push --overwrite --prune --clean-local
```

### Step 3: How New Results Automatically Reflect in the Dashboards (`/engineering` and `/exec`)
1. **Automatic 30-Second GCS Polling (`PULL_TTL_S = 30.0`):** The live Cloud Run service (`https://perfect-store-control-plane-jbn2qgbina-uc.a.run.app`) runs `shelf-bench serve --pull`, which polls `gs://ul-fs-s-sandbx-16-prj-shelf-images/results` every 30 seconds.
2. **Where Your Run Appears Automatically:**
   - **Engineering Leaderboard (`/engineering#/board/market_share` or `#/board/merchandising`):** Appears under its Task tab (`TASK-E2E`, `TASK-RET`, etc.) and Golden Dataset chip with its Task ID badge, plain-English model summary, UTC timestamp, run duration (`wall_time_s` + `setup_seconds`), and unit-labeled cost (`₹/crop` and `₹/shelf`).
   - **Experiments Matrix (`/engineering#/experiments`):** Binds to its matching Experiment ID (`P0-EXP-01` through `P0-EXP-19`), updates the 5-level accuracy breakdown (`L1` through `L5`), and links the Vertex AI Experiment context.
   - **Executive Summary (`/exec`):** Updates executive KPI cards, the Audited GCP Cloud Billing SKU cost simulator, 5-level accuracy bars, and the interactive Shelf Proof modal.
3. **Redeploying After UI or Code Changes:**
   ```bash
   PYTHONPATH=src .venv/bin/python -m cli serve-cloud
   ```

## 5. 12-Check Automated Data and Pipeline Quality Gate (`cli qc --verify-all`)

Before publishing manifests or promoting runs, verify all 12 quality checks (`src/utils/qc.py`):

```bash
PYTHONPATH=src .venv/bin/python -m cli qc --verify-all
```

1. **`G1 catalog_label_parity`**: Every box's `sample.labels[k]` matches `classes[str(sku_id)]` (`0` drift).
2. **`G2 canonical_taxonomy_domain`**: `0` `"Unknown"` categories, `0` `"Unknown"` packaging types, `0` `"Others"` placeholders, `0` brand typos (`tetely`, `twinnings`, `ppvwash`).
3. **`G3 zero_class_collision`**: Unique `(brand, base_variant)` pairs across all `2,745` classes; `0` brand-only classes in `hul_variants`.
4. **`G4 multi_box_shelf_diversity`**: `100%` of multi-box shelves in `gtmerch_golden`, `ms_json_golden`, and `hul_sessions` have `>1` distinct SKU (`0` single-product fallback stamping).
5. **`G5 pixel_overlay_deleakage`**: `100%` quarantine of burned-in OpenCV red rectangles and green variant text in `gs://ms-json-dataset`.
6. **`G6 non_product_box_quarantine`**: Rejects ceiling fans, printed model faces on display borders, and tiny `<3px` artifacts.
7. **`G7 bbox_geometry_and_dedup`**: Valid `[x1, y1, x2, y2]` bounds and nested-box deduplication.
8. **`G8 split_and_session_deleakage`**: `0` image URI overlap across `train`, `val`, and `test`.
9. **`G9 visual_embedding_concordance`**: Every catalog class has verified reference gallery images.
10. **`G10 detector_deoracle_integrity`**: `0` ground-truth bay proposal shortcuts or hardcoded `220px` grid division in detectors.
11. **`G11 stage2_per_box_completeness`**: Complete 5-level `field_accuracy` across all evaluated runs.
12. **`G12 ci_confidence_and_rival_parity`**: Cryptographic `SHA-256` split hashes and Unilever vs Competitor parity.

## 6. Modular Codebase Structure

```text
src/
├── approaches/
│   ├── market_share/
│   │   ├── detection/       # CPU ONNX YOLO detectors (onnx_yolo_detector.py) & Gemini dedup detectors
│   │   ├── retrieval/       # Embedding search, Brand Filter + Contact Sheet, and Cold-Start discovery
│   │   ├── classification/  # 5-level catalog classifiers (embedding-only & Gemini contact-sheet)
│   │   └── end_to_end/      # Composed Stage 1 ONNX/Gemini + Stage 2 Shelf Attribution pipelines
│   └── merchandising/       # Display Bay Planogram & Out-of-Stock Verifier, Promo Verifiers
├── utils/
│   ├── taxonomy/            # Modular catalog package (brands.py, categories.py, packaging.py, filenames.py)
│   ├── manifests/           # Modular manifest builders (ms_json_loader.py, crop_verifier.py, golden_builder.py)
│   ├── qc.py                # 12-Check Automated Data & Pipeline Quality Engine
│   ├── dataset.py           # Unified GCS-first streaming loader for all 8 Golden & 4 Public datasets
│   ├── experiment_matrix.py # 19-Experiment Catalog & Audited GCP Billing SKU Cost Estimator
│   ├── server.py            # Dual-mode HTTP server (:8080 /engineering, :8090 / /exec Executive Summary)
│   └── static/              # Frontend UI assets (index.html, app.js, styles.css, cio.html, cio.js, cio.css)
├── cli.py                   # CLI entrypoints (run, cloud-run, push, pull, qc, experiments, serve, serve-cloud)
└── runner.py                # Parallel evaluation harness, 5-level scorer & GCS/Vertex telemetry
```
