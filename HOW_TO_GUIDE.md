# End-to-End Operations and Developer How-To Guide

**Owners:** Jigyasu Juneja (`jjuneja@google.com`), Riley Gavigan (`rgavigan@google.com`)  
**Repository:** `https://github.com/jigyasujuneja/unilever-shelf-understanding-e2e`  
**Standalone Reference Archive:** Tag `v1.0-standalone-reference` | Branch `reference/standalone-v1`

This guide covers local execution, one-command Argolis GCP bootstrapping, Cloud Run benchmark submission, MLOps lifecycle management, and developer extension rules for the unified Unilever Shelf Understanding repository.

## 1. Repository Architecture and Directory Rules

The repository merges the `shelf-bench` Cloud Run execution harness (`cloud-gtm/unilever-shelf-understanding-with-cv`) with the 8-Stage Hindustan Unilever (`HUL`) Gondola Intelligence engine (`src/shelf_e2e/`).

To keep the codebase modular for multiple contributors, enforce two placement rules:
1. **Approaches live exclusively in `src/approaches/`**: Every model architecture or routing pipeline is a self-contained module registered with `@register("name")` in `src/approaches/`.
2. **Shared logic lives in `src/utils/` and `src/runner.py`**: Dataset loading, Vertex AI clients, Cloud Run submission, MLOps drift gates, and the HUL 8-stage domain bridge reside in `src/utils/` and `src/runner.py`.

```text
unilever-shelf-understanding-e2e/
├── Dockerfile                        # Cloud Run container packaging src/, configs/, data/splits/, results/
├── pyproject.toml                    # Package definition and shelf-bench CLI entrypoint
├── config.yaml                       # Default model, split, and run parameters
├── src/
│   ├── cli.py                        # Unified CLI: list, run, submit, bootstrap, serve, leaderboard
│   ├── runner.py                     # Parallel evaluation harness, IoU matching, FinOps billing, MLOps gates
│   ├── approaches/                   # EXCLUSIVELY registered shelf-understanding approaches (@register)
│   │   ├── hul_8stage_gemini38_hybrid.py   # #1 Production Winner: RT-DETR + ScaNN (89%) + /v1/systemone (9%) + Gemini 3.8 (2%)
│   │   ├── djev_systemone_sister_shade.py  # #2 Track D2/D3: RT-DETR + I-JEPA + Stage 4.5 Sister-Shade + /v1/systemone
│   │   ├── tiered_hybrid_scann.py          # #3 Track C/D1: RT-DETR + AlloyDB ScaNN + Gemini fallback
│   │   ├── all_pareto_tracks.py            # Track A (Cascading ViT), Track E (OWL-v2 + SigLIP), Track F (SAM-2 + ScaNN)
│   │   ├── single_pass.py                  # Baseline B1: 1-Pass Full-Shelf Gemini VLM
│   │   └── detect_classify.py              # Baseline B2: 2-Pass Gemini Detect + Crop Classify
│   ├── utils/                        # Shared cloud, dataset, domain, MLOps, and API utilities
│   │   ├── cloud.py                  # Dynamic Argolis project resolution, UBLA bucket bootstrap, Cloud Build & Run
│   │   ├── dataset.py                # 3-way stratified Train/Val/Test split builder + GCS/local loader
│   │   ├── hul_domain.py             # Shared bridge to Stage 1-8 HUL engine, I-JEPA, Stage 4.5, and 8 Gondola KPIs
│   │   ├── mlops_pipeline.py         # Hot-swap SKU onboarding, Active Learning queue, PSI/ECE drift, 7-Gate CI/CD
│   │   ├── storyboard_api.py         # Dual-persona REST APIs (/api/v1/cx-storyboard and /api/v1/eng-workbench)
│   │   ├── llm.py                    # Keyless Vertex AI Gemini client with dynamic project resolution
│   │   ├── pricing.py                # Live Cloud Billing Catalog SKU lookup + INR/USD FinOps ledger
│   │   └── _local_shims/             # Offline fallback shims appended at the end of sys.path for bare Python runs
│   └── shelf_e2e/                    # Full 8-Stage HUL Gondola Intelligence library (I-JEPA, dJev, 8 KPIs, Arena UI)
├── data/
│   ├── splits/dataset_splits_manifest.json # Cryptographically locked (SHA-256: 5f2e48d279a9fec0) Train/Val/Test splits
│   └── *.jpg                         # Real HUL and SKU-110K validation and test shelf photographs
├── results/                          # Persisted JSON/HTML run scorecards and active_learning_queue.jsonl
└── tests/
    ├── test_shelf_bench.py           # Core shelf-bench harness unit tests
    └── test_unified_cloud_mlops_and_approaches.py # End-to-end unified approaches, MLOps, splits, and Argolis tests
```

## 2. Local Setup and Authentication

Install dependencies using `uv` (or run directly with `PYTHONPATH=src:. python3 src/cli.py`):

```bash
uv sync
```

Authenticate with Google Cloud Application Default Credentials (`ADC`). The codebase complies with Argolis organization policies (`constraints/iam.disableServiceAccountKeyCreation`) and never uses downloaded JSON service account keys:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="<YOUR_ARGOLIS_PROJECT_ID>"
export GOOGLE_CLOUD_REGION="us-central1"
```

List all registered approaches and supported Vertex AI models:

```bash
PYTHONPATH=src:. python3 src/cli.py list
```

## 3. How to Bootstrap Any New Argolis GCP Project

When moving to a new Argolis environment, run a single command to provision all required Google Cloud infrastructure, upload the datasets, build the container image, and register the Cloud Run Job:

```bash
PYTHONPATH=src:. python3 src/cli.py bootstrap \
  --project <YOUR_ARGOLIS_PROJECT_ID> \
  --region us-central1
```

What `shelf-bench bootstrap` executes in order:
1. **Project Resolution**: Resolves the target GCP project via `--project`, `SHELF_BENCH_PROJECT`, `GOOGLE_CLOUD_PROJECT`, `gcloud config get-value project`, or `google.auth.default()`.
2. **Service Enablement**: Enables `aiplatform.googleapis.com`, `run.googleapis.com`, `cloudbuild.googleapis.com`, `artifactregistry.googleapis.com`, `storage.googleapis.com`, and `cloudbilling.googleapis.com` via the Service Usage API.
3. **Org-Policy Compliant Storage**: Creates `gs://<YOUR_ARGOLIS_PROJECT_ID>-shelf-images` and `gs://run-sources-<YOUR_ARGOLIS_PROJECT_ID>-us-central1` with `uniformBucketLevelAccess.enabled = True` (`constraints/storage.uniformBucketLevelAccess`).
4. **Dataset and Split Upload**: Uploads local `SKU110K_fixed/` images and annotations (if present in `--data-dir`), HUL reference shelf images from `data/`, and `data/splits/dataset_splits_manifest.json` to `gs://<YOUR_ARGOLIS_PROJECT_ID>-shelf-images/`.
5. **Cloud Build Packaging**: Archives `pyproject.toml`, `README.md`, `config.yaml`, `src/`, `configs/`, `data/splits/`, and `results/`, submits the tarball to Cloud Build, and pushes `gcr.io/<YOUR_ARGOLIS_PROJECT_ID>/shelf-bench:latest`.
6. **Cloud Run Job Creation**: Creates or updates the `shelf-bench` Cloud Run Job (`4 vCPU`, `8 GiB RAM`, `3600s` timeout) with `GOOGLE_CLOUD_PROJECT`, `SHELF_BENCH_BUCKET`, and `SHELF_BENCH_RESULTS` pre-wired.

If you only need to check infrastructure readiness without uploading local datasets, pass `--skip-data-upload`.

## 4. How to Run Benchmarks (Local and Cloud Run)

### Run Locally Against Any Split (`train`, `val`, `test`)

```bash
# Run the #1 Production Hybrid approach on 50 test images
PYTHONPATH=src:. python3 src/cli.py run hul_8stage_gemini38_hybrid \
  --model gemini-3.8-flash \
  --split test \
  --limit 50 \
  --workers 8

# Run Track D2/D3 (/v1/systemone + Stage 4.5 Sister-Shade) on the validation split
PYTHONPATH=src:. python3 src/cli.py run djev_systemone_sister_shade \
  --model gemini-3.8-flash \
  --split val \
  --limit 25
```

### Submit a Benchmark Job to Cloud Run (`shelf-bench submit`)

```bash
PYTHONPATH=src:. python3 src/cli.py submit hul_8stage_gemini38_hybrid \
  --project <YOUR_ARGOLIS_PROJECT_ID> \
  --region us-central1 \
  --model gemini-3.8-flash \
  --split test \
  --limit 50 \
  --workers 8
```

When `--sync` is omitted, `submit` returns immediately with the Cloud Run execution ID and console URL while the job writes `summary.json`, `report.html`, and annotated bounding-box PNGs to `gs://<YOUR_ARGOLIS_PROJECT_ID>-shelf-images/results/<run_id>/`.

### View the Unified Leaderboard

```bash
PYTHONPATH=src:. python3 src/cli.py leaderboard
```

This prints a comparative table across all saved runs in `results/`, including 2D Box `F2` (`IoU >= 0.50`), 7-Dimension HUL SKU `F2`, 14-SKU Sister-Shade `F2`, `P95` latency, total INR cost, and 7-Gate CI/CD promotion status.

## 5. How to Add a New Approach (`src/approaches/`)

To add a new computer vision or VLM pipeline without touching the runner or CLI:

1. Create a new file in `src/approaches/my_new_approach.py`.
2. Decorate your entry function with `@register("my_new_approach")`.
3. Import your module in `src/approaches/__init__.py`.

```python
from PIL import Image
from approaches import Box, Detections, register
from utils.llm import Usage

@register("my_new_approach")
def run(image: Image.Image, model: str, ctx: dict, **kwargs) -> Detections:
    """Detect and classify shelf products."""
    llm = ctx["llm"]
    res = llm(image, "Detect all visible facings as [ymin, xmin, ymax, xmax].")
    boxes = [
        Box(ymin=b[0], xmin=b[1], ymax=b[2], xmax=b[3], label="HUL_DOVE_180ML", conf=0.95)
        for b in (res.parsed or [])
    ]
    return Detections(boxes=boxes, usage=res.usage, meta={"custom_metric": 1.0})
```

Your approach immediately becomes available to `shelf-bench list`, `shelf-bench run my_new_approach`, `shelf-bench submit my_new_approach`, and `shelf-bench leaderboard`.

## 6. How to Use the MLOps and GenAIOps Pipeline

All MLOps primitives live in `src/utils/mlops_pipeline.py` and run automatically during `runner.run()`.

### Verify Dataset Splits and Zero Data Leakage (`Train / Val / Test`)

Rebuild or verify the cryptographically locked (`SHA-256: 5f2e48d279a9fec0`) 3-way split manifest (`20` `train`, `25` `val`, `63` `test` images):

```bash
PYTHONPATH=src:. python3 -c "
from utils.dataset import build_stratified_splits_manifest
manifest = build_stratified_splits_manifest()
print('Manifest Hash:', manifest['manifest_sha256'])
print('Zero Leakage Verified:', manifest['zero_leakage_verified'])
print('Counts:', {k: v['count'] for k, v in manifest['splits'].items()})
"
```

### Zero-Retrain Hot-Swap SKU Onboarding (`< 60s`)

When HUL launches a new SKU or updates packaging artwork, onboard the SKU directly into the `AlloyDB / Vertex AI ScaNN` vector index without retraining `RT-DETR-v2`:

```bash
PYTHONPATH=src:. python3 -c "
from pathlib import Path
from utils.mlops_pipeline import hot_swap_onboard_sku

entry = hot_swap_onboard_sku(
    sku_id='HUL_NOVOLOGY_ACNE_SERUM_30ML',
    base_pack_code='BP-HUL-NOV-019',
    brand='Novology',
    category='Skin Care',
    variant='Bi-Phasic Hyper Pigmentation Serum 30ml',
    reference_images=[Path('data/grocery282_unilever_shelf_val_000.jpg')],
    cielab_reference=(64.2, 8.1, 14.5),
)
print('Onboarded in:', entry['onboarding_latency_s'], 's | Status:', entry['status'])
"
```

### 7-Gate Champion/Challenger Promotion Contract

Every `test` split run evaluates 7 automated release gates in `summary["mlops"]["promotion_contract"]`:
1. `box_f2_gte_095`: 2D Localization `F2 >= 0.950`
2. `hul_7dim_f2_gte_095`: 7-Dimension HUL SKU `F2 >= 0.950`
3. `sister_shade_f2_gte_092`: 14-SKU Sister-Shade `F2 >= 0.920`
4. `latency_p95_lte_20s`: Server `P95 <= 20.0s`
5. `cost_inr_lte_022`: Unit cost `<= INR 0.22 / image`
6. `zero_erp_hallucination`: `0.0%` invalid ERP Base Pack codes (`vllm#58216`)
7. `zero_split_leakage`: Cryptographic verification that `train`, `val`, and `test` have zero overlap

## 7. How to Run the Dual-Persona Storyboard APIs and Arena Server

### Start the Dual-Persona Cloud Run API Server (`shelf-bench serve`)

To keep executive views simple and give ML engineers full diagnostic telemetry, the backend exposes two separate JSON contracts:

```bash
PYTHONPATH=src:. python3 src/cli.py serve --port 8080
```

Query the **CX / Leadership Storyboard API** (returns 3 non-technical cards: Overall Shelf Health Score, Weekly Store Revenue Recovery in `INR`, and Top 3 Field Merchandiser Actions):

```bash
curl -s http://127.0.0.1:8080/api/v1/cx-storyboard | python3 -m json.tool
```

Query the **AI & ML Engineer Workbench API** (returns the `Train/Val/Test` KPI matrix across all 8 approaches, `89% ScaNN / 9% SystemOne / 2% Gemini` cascade routing ratios, `PSI`/`ECE` drift metrics, and 7-Gate CI/CD contract status):

```bash
curl -s http://127.0.0.1:8080/api/v1/eng-workbench | python3 -m json.tool
```

### Start the Standalone 8-Tab Interactive Control Plane UI

To inspect our standalone 8-tab Gondola Intelligence visual debugger locally:

```bash
PYTHONPATH=src:. python3 src/shelf_e2e/platform/server.py --host 0.0.0.0 --port 8765
```

## 8. How to Run the Automated Verification Suite

Run both the unified cloud/MLOps test suite and the standalone 8-stage HUL test suite:

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests -p "test_*.py" -v
```

## 9. Accessing the Preserved Standalone v1.0 Reference

Our standalone pre-merge repository (`99` files, all reports, specs, and standalone servers) is permanently frozen and available at any time:

```bash
# Inspect or checkout the standalone reference tag
git checkout v1.0-standalone-reference

# Or switch to the preserved reference branch
git checkout reference/standalone-v1

# Return to the unified cloud main branch
git checkout main
```
