# Unilever Shelf Understanding — CV & VLM Benchmark Suite on GCP

A modular, production-ready benchmark and experimentation suite for retail shelf computer vision on Google Cloud Platform (`Vertex AI`, `Cloud Run`, `Cloud Vision`, `BigQuery`, `Cloud Storage`, and `OpenTelemetry`).

---

## 1. Core CV Tasks & Scope

Given shelf photographs of any resolution (from a single quickstart fixture locally to multi-image GCS buckets `gs://...` or BigQuery tables), this suite benchmarks approaches across three foundational computer-vision tasks (which power downstream capabilities such as recommendations, assortment optimization, planogram compliance, and shelf analytics):

1. **Front-Facing Spatial Detection (`detection`)**:
   - Localize every front-most visible product unit in normalized `[ymin, xmin, ymax, xmax]` (`0..1000`) coordinates and suppress back-row depth-stacked units via geometric 1D-NMS (`deduplicate_depth_stacked_facings`).
2. **Configurable N-Attribute Classification (`classification` — 8 Core + Unlimited Custom Attributes)**:
   - Predict the 8 core dimensions (`category`, `subcategory`, `brand` + `is_hul_brand`, `product_name`, `variant`, `packaging_type`, `pack_type`, `size` + rule-derived size bucket) **plus any number of custom attributes (>8 attributes)** configured in `taxonomy.custom_attributes` (e.g., `price_tag_visible`, `promo_callout`, `facing_orientation`, `shelf_talker_present`).
   - Supports **Single-Call VLM extraction** (`single_pass_full_shelf` or `configurable_multi_attribute_vlm` with `attribute_call_groups: []`) and **Grouped Multi-Call VLM extraction** (`configurable_multi_attribute_vlm` with `attribute_call_groups: [[...], [...]]`).
3. **Hybrid SKU Matching (`matching`)**:
   - Match detected facings against a product catalog using sparse BM25 lexical keywords + dense vector embeddings (`gemini-embedding-001` 3072-D or `multimodalembedding@001` 1408-D), or single-step end-to-end `single_step_detect_classify_and_match`.

> [!IMPORTANT]
> **Ground-Truth Scope (Ingestion Only — We Do Not Produce Annotations):**
> - This benchmark suite **does not author or produce ground-truth annotations**. It ingests existing ground-truth datasets provided externally (`sdk.connect_ground_truth` supporting BigQuery, GCS, CSV, JSON/JSONL, and COCO).
> - Before ground truth is connected, accuracy metrics are reported as `None` with `accuracy_status="PLACEHOLDER_AWAITING_GROUND_TRUTH"` (never `0.0`).
> - Every run saves `predictions.json`, so once the ground-truth dataset is connected, saved predictions can be scored or re-scored offline (`shelf-benchmark score`) with zero inference cost.

---

## 2. Getting started

From a fresh clone, three commands:

```bash
make setup    # creates .venv and installs the package with dev extras
make test     # offline test lane: no network, no GCP credentials
make check    # what CI runs: lint + typecheck + tests + wheel packaging
```

Then follow [`ENGINEER_ONBOARDING_GUIDE.md`](ENGINEER_ONBOARDING_GUIDE.md). Everything under `docs/`
is reference material -- read it when you need it, not before.

> [!NOTE]
> Nothing above touches GCP. Credentials are only needed for `--live` runs and the `cloud-run`
> subcommand; see [`CONTRIBUTING.md`](CONTRIBUTING.md) §1.

---

## 3. Repository Directory Structure

```text
unilever-shelf-understanding-with-cv/
├── README.md                              # Project overview, directory map & configuration cheat sheet
├── ENGINEER_ONBOARDING_GUIDE.md           # Step-by-step 15-minute onboarding guide for new engineers
├── Dockerfile                             # Container image for Cloud Run service & headless benchmark workers
├── pyproject.toml                         # Package dependencies, pytest config & `shelf-benchmark` CLI entrypoint
│
├── configs/                               # YAML configuration files (zero hardcoded brands/aliases)
│   ├── default_config.yaml                # Master config: GCP project, buckets, models, billing, hardware/TPU, OTel
│   ├── demo_visual_prototypes.json        # Demo-only reference catalog for visual embedding lookup
│   ├── sample_associations.json           # Sample shelf-image manifest for local quickstarts
│   └── sample_ground_truth.json           # Synthetic placeholder ground-truth fixture (aligned with testing.py)
│
├── code_samples/                          # 11 self-contained runnable scripts (see code_samples/README.md)
│   ├── README.md                          # Cookbook & reference table for all 11 code samples
│   ├── 01_quickstart_run_full_suite.py    # End-to-end quickstart (offline by default, `--live` for Vertex AI)
│   ├── 02..09_*.py                        # Focused recipes: GEAP/Gemma models, SFT tuning, custom plugins, billing
│   ├── 10_run_now_score_later_ground_truth.py       # Placeholder run -> connect ground truth -> zero-cost re-scoring
│   └── 11_complete_engineer_approach_playground.py  # ALL-IN-ONE PLAYGROUND: custom plugin + 12 attrs + TPU/GPU + diagnostics
│
├── src/shelf_benchmark/                   # Core Python package
│   ├── __init__.py                        # Public SDK exports (ShelfBenchmarkSDK, SimpleShelfApproachPlugin, etc.)
│   ├── _resources/taxonomy.yaml           # Packaged N-attribute taxonomy (categories, sizes, custom_attributes, call groups)
│   ├── sdk.py                             # Developer facade (`connect_dataset`, `connect_ground_truth`, `run_suite`)
│   ├── cli.py                             # `shelf-benchmark` CLI (`run`, `cloud-run`, `score`, `list-approaches`, `validate-gt`)
│   ├── runner.py                          # Multi-model / multi-approach batch orchestrator
│   ├── config.py                          # Pydantic config schemas (`TaxonomyConfig`, `CustomAttributeSpec`, `CloudRunCostConfig`)
│   ├── models.py                          # Data contracts (`TaskExecutionResult`, `RowLevelReportItem`, execution traces)
│   ├── testing.py                         # 1-line offline test harness (`benchmark_harness`, `run_offline_approach`)
│   ├── telemetry.py                       # OpenTelemetry span logger (`otel_logs.jsonl` + GCP Cloud Logging exporter)
│   ├── scoring.py                         # Zero-cost offline re-scorer (`score_predictions` from `predictions.json`)
│   │
│   ├── approaches/                        # Auto-discovered modular approach plugins (`approaches/*/plugin.py`)
│   │   ├── base.py                        # `SimpleShelfApproachPlugin` (~15-line base class) & `CommonLayerContext`
│   │   ├── registry.py                    # `GLOBAL_APPROACH_REGISTRY` eager plugin discovery
│   │   ├── TEMPLATE_NEW_APPROACH.md       # Copy-paste guide for adding a new approach plugin folder
│   │   ├── vlm_existing_approaches/       # 6 VLM plugins (`single_pass_full_shelf`, `open_vocab_brand_plus_catalog_resolver`,
│   │   │                                  #   `configurable_multi_attribute_vlm`, `single_step_detect_classify_and_match`,
│   │   │                                  #   `two_stage_bbox_guided_nms`, `two_stage_physical_crop_per_facing`)
│   │   └── class_agnostic_visual_embedding/ # 2 CV/Embedding plugins (`class_agnostic_visual_embedding`, `cloud_vision_visual_embedding`)
│   │
│   ├── tasks/                             # Shared task implementations & geometric utilities
│   │   ├── detection.py                   # `ProductDetectionTask` (0..1000 bounding boxes + 1D-NMS)
│   │   ├── classification.py              # `ProductClassificationTask` (8 core + N custom attributes, 1-call or grouped)
│   │   ├── matching.py                    # `ProductMatchingTask` (BM25 lexical + 3072-D dense hybrid search)
│   │   ├── fine_tuning.py                 # `GeminiFineTuningTask` (Vertex AI SFT JSONL generator & job launcher)
│   │   └── facing_utils.py                # Geometric 1D-NMS depth deduplication, PIL cropping & size-bucket rules
│   │
│   ├── data/                              # Universal schema-agnostic GCP dataset & ground-truth adapters
│   │   ├── associations.py                # `connect_dataset`: BigQuery, GCS bucket discovery, CSV, JSON/JSONL + dot-paths
│   │   ├── ground_truth.py                # `connect_ground_truth`: 6 bbox formats, 4-col CSV/BQ boxes, polygon vertices
│   │   └── storage.py                     # GCS (`gs://`) and local file I/O manager
│   │
│   ├── evaluation/                        # Accuracy scoring & 5-bucket GCP cost attribution
│   │   ├── metrics.py                     # Greedy/Hungarian IoU pairing, brand/product/SKU & `per_attribute_accuracy`
│   │   ├── cost.py                        # Token usage extraction & per-image / per-facing cost computation
│   │   └── gcp_billing.py                 # 5-bucket GCP cost engine (Tokens, GSU, Vision/Embed, Cloud Run/GPU/TPU, GCS/Logs)
│   │
│   └── reporting/                         # Artifact & report generators
│       └── generator.py                   # Writes CSV/JSON/Markdown reports (`leaderboard.csv`, `benchmark_report.md`, `predictions.json`)
│
├── ui/                                    # Interactive Local Studio UI & Cloud Run HTTP Server
│   ├── server.py                          # HTTP server (`/api/run-live`, `/api/dashboard`, `/api/hybrid-search`)
│   └── static/                            # Interactive dashboard (`index.html`, `app.js`, `styles.css`)
│
├── docs/                                  # Technical specifications
│   ├── EVALUATION_PROTOCOL.md             # Exact mathematical definitions of IoU pairing, F1, and attribute accuracy
│   └── GROUND_TRUTH_CONTRACT.md           # Supported ground-truth formats, schema mappings & bbox coordinate conventions
│
├── tests/                                 # Fast offline contract & golden-scoring test suite (`86 passed`)
│   ├── conftest.py                        # Shared fixtures & temporary OTel/report directory isolation
│   ├── test_onboarding_contracts_and_ui_trace.py # Contracts for all 8 approaches, >8 attributes, schema adapters & UI
│   └── test_*.py                          # Golden scoring, depth deduplication, billing & ground-truth workflow tests
│
└── reports/                               # Default output directory for generated reports, crops & OTel logs
```

---

## 4. Engineer Cheat Sheet: How to Configure & Test Everything

| What You Want to Do | Command / File to Use |
| :--- | :--- |
| **See all configurable levers in 1 file** (custom approach, >8 attributes, 1-call vs grouped VLM calls, GPU/TPU profile, diagnostics) | `.venv/bin/python code_samples/11_complete_engineer_approach_playground.py` |
| **Test & compare ALL 8 approaches locally with LIVE Vertex AI / Agent Platform calls** (reads local `shelf-image.png`, writes local reports, zero Cloud Run/GCS setup required) | `.venv/bin/shelf-benchmark cloud-run --local --approaches all --model gemini-3.8-flash --image shelf-image.png --connect-sample-gt` |
| **Test & compare ALL 8 approaches on remote Cloud Run** (and write local reports + `diagnostic_trace_report.md`) | `.venv/bin/shelf-benchmark cloud-run --approaches all --model gemini-3.8-flash --image gs://unilever-shelf-understanding-shelf-images/shelf-image.png` |
| **Run instant synthetic unit-test stub mode** (canned responses, zero API calls for CI) | `.venv/bin/shelf-benchmark cloud-run --approaches all --offline --connect-sample-gt` |
| **Configure Cloud Run / Vertex AI Hardware & Accelerators** (`none` CPU, `nvidia-l4` GPU, `tpu-v5e`, `tpu-v6e` Trillium TPU) | Pass `--accelerator tpu-v5e --include-infra-costs` (or set `billing.cloud_run.accelerator_type` in `configs/default_config.yaml`) |
| **Predict >8 Attributes** (e.g., 12 attributes in 1 VLM call or split into attribute groups) | Add fields under `custom_attributes:` and `attribute_call_groups:` in [`src/shelf_benchmark/_resources/taxonomy.yaml`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/src/shelf_benchmark/_resources/taxonomy.yaml) (or your own file via `taxonomy_file:`) and run `--approaches configurable_multi_attribute_vlm` |
| **Write a custom Detection or Classification approach** (~15 lines) | Subclass `SimpleShelfApproachPlugin` (set `task_type = "detection"` for pure detectors, or `"classification"` for joint/multi-stage) or pass `--plugin-module path/to/my_plugin.py` |
| **Chain any Stage-1 Detector into any Stage-2 Classifier** (without re-running detection) | Pass `detector_approach="my_detector"` or `reuse_prior_detection=True` to `sdk.run_suite(...)` / `runner.run_benchmark(...)` and read boxes via `ctx.get_prior_detected_boxes()` |
| **Connect any GCP image dataset** (multi-image GCS bucket, BigQuery table/SQL, CSV, JSON) | Call `sdk.connect_dataset(provider_type="gcs_bucket" \| "bigquery" \| "csv" \| "json", source_uri=..., schema_mapping=...)` |
| **Connect any provided Ground-Truth dataset** & re-score saved predictions for free | `.venv/bin/shelf-benchmark score --predictions reports/predictions.json --gt-provider json --ground-truth-uri <path_or_gs_uri>` |
| **Launch the Interactive Local UI Studio** (defaults to Local Machine + Live Vertex AI / Agent Platform Calls) | `.venv/bin/python ui/server.py` (open `http://localhost:8080`) |
| **Diagnose runs & inspect OpenTelemetry traces** | Open `reports/diagnostic_trace_report.md` or run `jq 'select(.TraceId == "<trace_id>")' reports/otel_logs.jsonl` |

---

## 5. Built-in Approaches (`shelf-benchmark list-approaches`)

| Approach ID | Call Topology | Description |
| :--- | :--- | :--- |
| `single_pass_full_shelf` | **1 VLM Call (Detect + Classify)** | Detects front-row `[ymin, xmin, ymax, xmax]` boxes and classifies all configured attributes in 1 structured VLM call + 1D-NMS depth deduplication. |
| `open_vocab_brand_plus_catalog_resolver` | **1 VLM Call + $O(1)$ Brand Resolver** | Open-vocabulary LLM brand generation $\rightarrow$ post-hoc $O(1)$ canonical snap against a **2,000+ brand catalog** (`resolve_brand_against_catalog`). |
| `configurable_multi_attribute_vlm` | **1 Call or Grouped Multi-Call VLM (>8 Attributes)** | Extracts 8 core + $N$ `custom_attributes` in **1 VLM call** (`attribute_call_groups: []`) or **1 VLM call per attribute group** (`attribute_call_groups: [[...], [...]]`). |
| `single_step_detect_classify_and_match` | **1 VLM Call (Detect + Classify + SKU Match)** | Performs spatial detection, N-attribute classification, and hybrid SKU catalog matching in a single pass. |
| `two_stage_bbox_guided_nms` | **2 VLM Calls (Stage 1 Detect -> Stage 2 Classify)** | Stage 1 localizes & depth-deduplicates front-facing boxes; Stage 2 classifies the locked coordinates. |
| `two_stage_physical_crop_per_facing` | **2 VLM Calls + PIL Cropping** | Stage 1 detects boxes; physically crops each facing into `reports/crops/` + a numbered montage; Stage 2 reads fine print on crops. |
| `class_agnostic_visual_embedding` | **3 Stages (Detector + 1408-D Embed + ScaNN)** | Class-agnostic detection (`class='product'`) -> PIL crops -> `multimodalembedding@001` cosine search against reference catalog. |
| `cloud_vision_visual_embedding` | **3 Stages (Cloud Vision + 1408-D Embed + ScaNN)** | Google Cloud Vision `OBJECT_LOCALIZATION` -> `multimodalembedding@001` visual crop search against reference catalog. |
