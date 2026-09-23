# Engineer onboarding guide

From zero context to a benchmarked approach of your own. Seven steps (`Step 0` to `Step 6`), about half an hour.

Five engineers work in this repository in parallel. Nothing below requires editing a shared file,
so you should never need to resolve a merge conflict to run an experiment.

---

## Step 0: Core CV Tasks & Scope (60 Seconds)

Given shelf photographs of any resolution (from a single quickstart image locally to large multi-image GCS buckets like `gs://unilever-shelf-understanding-shelf-images` or BigQuery tables), this benchmark suite evaluates approaches across three core computer-vision tasks that power downstream retail capabilities (such as recommendations, assortment optimization, planogram compliance, and shelf analytics):

1. **Detect Front-Row Facings (`detection`)**:
   - Localize every front-most visible unit in `[ymin, xmin, ymax, xmax]` (`0` to `1000` normalized coordinates so any image resolution is supported seamlessly) and suppress depth-stacked back-row units (`deduplicate_depth_stacked_facings`).
2. **Classify N Attributes (`classification` — 8 Core + Unlimited Custom Attributes)**:
   - Extract the 8 core dimensions (`category`, `subcategory`, `brand` + `is_hul_brand`, `product_name`, `variant`, `packaging_type`, `pack_type`, `size` + rule-derived size bucket) **plus any number of additional attributes (>8 attributes)** configured under `taxonomy.custom_attributes` (e.g., `price_tag_visible`, `promo_callout`, `facing_orientation`, `shelf_talker_present`).
   - **Configurable VLM Call Strategy**: Extract all $8 + N$ attributes in a **single VLM call** (`single_pass_full_shelf` or `configurable_multi_attribute_vlm` with `attribute_call_groups: []`), or split attributes across **multiple targeted VLM calls** by attribute type (`configurable_multi_attribute_vlm` with `attribute_call_groups: [['brand', 'product_name', 'variant'], ['packaging_type', 'size', 'promo_callout']]`).
3. **Match to Catalog SKUs (`matching`)**:
   - Hybrid search combining sparse BM25 lexical keywords and dense vector embeddings (`gemini-embedding-001` 3072-D or `multimodalembedding@001` 1408-D), or single-step end-to-end `single_step_detect_classify_and_match`.

> [!IMPORTANT]
> **Ground-Truth Scope (Ingestion Only — We Do Not Produce Annotations):**
> - This benchmark suite does **not** author or produce ground-truth annotations; it ingests the pre-existing ground-truth dataset provided externally (via `sdk.connect_ground_truth` supporting BigQuery, GCS, CSV, JSON/JSONL, and COCO).
> - Before ground truth is connected, accuracy metrics are `None` with `accuracy_status="PLACEHOLDER_AWAITING_GROUND_TRUTH"` (never `0.0`).
> - Every run saves `predictions.json`, so once the ground-truth dataset is connected, you can score and re-score saved predictions immediately (`sdk.score_existing_predictions` / `shelf-benchmark score`) with zero additional inference cost.

---

## Step 1: Verify Your Environment & Run the All-in-One Playground (2 Minutes)

```bash
# 1. Run the offline test suite (86 tests in ~15s)
.venv/bin/pytest -q

# 2. List all 8 registered test approaches
.venv/bin/shelf-benchmark list-approaches

# 3. Run the All-in-One Engineer Playground (custom approach + 2,000-brand resolver + 12 attributes + TPU profile + diagnostics)
.venv/bin/python code_samples/11_complete_engineer_approach_playground.py

# 4a. Run & compare all 8 approaches LOCALLY on your machine with LIVE calls to Vertex AI / Agent Platform
.venv/bin/shelf-benchmark cloud-run \
  --local \
  --approaches all \
  --model gemini-3.8-flash \
  --image shelf-image.png

# 4b. Or orchestrate all 8 approaches remotely on the deployed GCP Cloud Run service
.venv/bin/shelf-benchmark cloud-run \
  --approaches all \
  --model gemini-3.8-flash \
  --image gs://unilever-shelf-understanding-shelf-images/shelf-image.png
```

### All 8 Registered Test Approaches (`.venv/bin/shelf-benchmark list-approaches`)

| Approach ID | Call Topology | Brand & Attribute Strategy |
| :--- | :--- | :--- |
| `single_pass_full_shelf` | **1 VLM Call** | Open-vocabulary LLM brand generation + N-dim attributes + 1D-NMS depth deduplication. |
| `open_vocab_brand_plus_catalog_resolver` | **1 VLM Call + $O(1)$ Resolver** | Open-vocabulary LLM brand generation $\rightarrow$ post-hoc $O(1)$ canonical snap against a **2,000+ brand catalog** (`resolve_brand_against_catalog`). |
| `configurable_multi_attribute_vlm` | **1 Call or Grouped Multi-Call** | Predicts **>8 attributes** (`custom_attributes`) either in 1 VLM call (`attribute_call_groups: []`) or split into multiple VLM calls by attribute group. |
| `single_step_detect_classify_and_match` | **1 VLM Call (End-to-End)** | Single-step spatial detection + N-dim classification + hybrid SKU catalog matching (`matched_sku_id`). |
| `two_stage_bbox_guided_nms` | **2 VLM Calls** | Stage 1 detects & depth-deduplicates boxes; Stage 2 classifies the locked coordinates. |
| `two_stage_physical_crop_per_facing` | **2 VLM Calls + PIL Crops** | Stage 1 detects boxes; crops each facing into `reports/crops/` + montage; Stage 2 reads fine print on crops. |
| `class_agnostic_visual_embedding` | **3 Stages (CV + 1408-D Embed)** | Class-agnostic detection (`class='product'`) $\rightarrow$ PIL crops $\rightarrow$ `multimodalembedding@001` cosine lookup against reference catalog. |
| `cloud_vision_visual_embedding` | **3 Stages (Cloud Vision + Embed)** | Cloud Vision `OBJECT_LOCALIZATION` $\rightarrow$ `multimodalembedding@001` visual crop lookup against reference catalog. |

### Complete Configuration Levers (`code_samples/11_complete_engineer_approach_playground.py`)

| Configuration Lever | Where to Set It (YAML or Python) | Options |
| :--- | :--- | :--- |
| **Brand Extraction (1 to 2,000+ Brands)** | `taxonomy.brand_extraction_mode` | `"open_vocabulary_generative"` *(default: LLM generates brand from package text)* \| `"open_vocabulary_plus_catalog_resolver"` *(O(1) snap to 2,000+ brand catalog)* \| `"closed_set_taxonomy"` *(only when $\le 50$ brands)* |
| **Predict >8 Attributes (12+ Attributes)** | `taxonomy.custom_attributes` | Dictionary of `CustomAttributeSpec(description=..., value_type="string"\|"boolean"\|"number", allowed_values=[...])` |
| **Single-Call vs Grouped Attribute Calls** | `taxonomy.attribute_call_groups` | `[]` *(1 VLM call for all attributes)* or `[['brand', 'product_name'], ['packaging_type', 'promo_callout']]` *(1 VLM call per attribute group)* |
| **Hardware / Accelerator Profile** | `--accelerator` or `billing.cloud_run.accelerator_type` | `"none"` *(CPU)* \| `"nvidia-l4"` *(Cloud Run L4 GPU)* \| `"tpu-v5e"` *(Cloud TPU v5e)* \| `"tpu-v6e"` *(Trillium TPU v6e)* |
| **Dataset & Ground-Truth Schema Adapter** | `sdk.connect_dataset(...)` & `sdk.connect_ground_truth(...)` | `provider_type="gcs_bucket"\|"bigquery"\|"csv"\|"json"\|"coco"` with dot-paths (`attributes.brand`), 4-col boxes (`ymin,xmin,ymax,xmax`), or polygon `vertices` |

---

## Step 2: Local Mode (Live Vertex AI / Agent Platform Calls) vs. Offline Unit-Test Stubs

The suite distinguishes between **Local Storage Mode** (running locally on your laptop with local images and local `reports/`, while **still making real live calls to Vertex AI / Agent Platform**) and **Offline Unit-Test Stub Mode** (deterministic canned responses for CI and unit tests):

```python
from shelf_benchmark import UniversalModelSpec
from shelf_benchmark.testing import (
    OFFLINE_IMAGE_URI, make_offline_sdk, perfect_prediction_payload,
)

# Local Storage Mode: disables GCS sync, Cloud Logging, and Billing Catalog API so all
# files stay under /tmp/my-experiment.
sdk = make_offline_sdk("/tmp/my-experiment")

# Option A — LIVE CALL TO VERTEX AI / AGENT PLATFORM from your local machine:
# Passing any real Vertex AI / Agent Platform model ID (e.g., "gemini-3.8-flash")
# builds a real Vertex AI client and issues a live model call against your local image:
# live_summary = sdk.run_suite(
#     models=["gemini-3.8-flash"],
#     tasks=["classification"],
#     approaches=["single_pass_full_shelf"],
#     shelf_image_uri="shelf-image.png",
# )

# Option B — OFFLINE UNIT-TEST STUB (Zero API calls, instant CI execution):
# Register an `offline-*` model ID or a `custom_callable` stand-in:
sdk.register_model(UniversalModelSpec(
    model_id="offline-demo-model",
    display_name="offline-demo-model",
    provider_family="custom_callable",
    custom_handler=lambda prompt, image_uri, schema: perfect_prediction_payload(),
))
summary = sdk.run_suite(
    models=["offline-demo-model"],
    tasks=["classification"],
    approaches=["single_pass_full_shelf"],
    shelf_image_uri=OFFLINE_IMAGE_URI,
)
```

What `shelf_benchmark.testing` gives you:

| Name | Use |
| :--- | :--- |
| `make_offline_sdk(tmp_dir, **overrides)` | An SDK with local-only I/O (no Cloud Logging, no GCS sync, no live billing catalog, artifacts under `tmp_dir`). Passing a real model ID (`gemini-3.8-flash`) still calls live Vertex AI / Agent Platform; passing `offline-demo-model` or a `custom_callable` runs 100% stubbed. |
| `benchmark_harness(tmp_dir, ...)` | Convenience wrapper around `make_offline_sdk` that pre-registers `offline-demo-model` with `offline_universal_payload_handler` for 1-line unit tests. |
| `offline_config(tmp_dir, **overrides)` | The same local-I/O config, if you want to edit it before building the SDK. |
| `OFFLINE_IMAGE_URI`, `fixture_image_path()` | The bundled 600x400 fixture shelf image. |
| `sample_ground_truth()`, `write_sample_ground_truth_file(path)` | Three annotated facings with known-correct answers. |
| `perfect_prediction_payload()` | A prediction that scores 1.0 against that ground truth. |
| `shifted_prediction_payload()`, `wrong_brand_payload()` | Predictions that should move detection and classification metrics respectively, and only those. |
| `FakeGenAIClient(payload)` | One canned response for every call. For tests that construct a task directly with `genai_client=`. |
| `ReplayGenAIClient(responses)`, `.from_file(path)` | A recorded sequence, replayed in order. For multi-stage approaches, and for turning one real run into a regression test. |

> [!TIP]
> **Why `make_offline_sdk` still supports live Vertex AI / Agent Platform calls:**
> `offline.enabled = True` gates **storage and observability side-effects** (Cloud Storage buckets, Cloud Logging, and Cloud Billing SKU lookups) so you can iterate locally with local files (`shelf-image.png` or `OFFLINE_IMAGE_URI`) without touching shared GCP buckets. When you pass a real model ID such as `gemini-3.8-flash` to `sdk.run_suite(models=["gemini-3.8-flash"], ...)` (or run `shelf-benchmark cloud-run --local`), the suite still connects to live Vertex AI / Agent Platform. When you want zero network calls in unit tests, pass `models=["offline-demo-model"]` or use `benchmark_harness()` / `run_offline_approach()`.

`tests/test_golden_scoring.py` and `tests/test_ground_truth_workflow.py` are short, they pass, and
they are the best templates to copy.

---

## Step 3: write your own approach (5 minutes)

You never need to modify `src/shelf_benchmark/tasks/` or `runner.py`. Decorate a function:

```python
from typing import Any, Dict, List

from shelf_benchmark import register_approach_function
from shelf_benchmark.approaches import CommonLayerContext


@register_approach_function(
    approach_id="my_new_cv_approach",
    display_name="My Custom Detector + VLM Verifier",
    category="two_stage_vlm",
    stages_description=["Stage 1: my detector", "Stage 2: my verifier"],
)
def run_my_new_cv_approach(
    ctx: CommonLayerContext,
    model_name: str,
    shelf_image_uri: str,
) -> List[Dict[str, Any]]:
    candidate_boxes = [
        {
            "bbox_2d": [535, 386, 795, 440],
            "shelf_row": "middle",
            "category": "Skin Care",
            "subcategory": "Face Wash",
            "brand": "Pond's",
            "product_name": "Pond's Bright Beauty Face Wash",
            "variant": "Bright Beauty",
            "packaging_type": "tube",
            "pack_type": "Single",
            "size": "100g",
            "matched_sku_id": "HUL-PONDS-BB-100G",
            "confidence": 0.97,
            "_input_tokens": 220,
            "_thinking_tokens": 25,
            "_output_tokens": 85,
        }
    ]
    front_facings, depth_filtered = ctx.deduplicate_depth_stacked_facings(candidate_boxes)
    if front_facings:
        front_facings[0]["_depth_filtered"] = depth_filtered
    return front_facings
```

Return one dict per facing. Recognized keys: `bbox_2d`, `brand`, `product_name`, `variant`,
`category`, `subcategory`, `packaging_type`, `pack_type`, `size`, `matched_sku_id`, `shelf_row`,
`confidence`, `is_hul_brand`. Bookkeeping keys the suite consumes and strips: `_input_tokens`,
`_thinking_tokens`, `_output_tokens`, `_depth_filtered`.

A working version is [`code_samples/05_create_custom_approach_plugin.py`](code_samples/05_create_custom_approach_plugin.py),
and one scored against annotations is [`code_samples/09_step_by_step_engineer_ground_truth_benchmark.py`](code_samples/09_step_by_step_engineer_ground_truth_benchmark.py).
Both run offline.

### The shared helpers on `ctx`

Use these rather than reimplementing the geometry, so that your approach and a built-in approach
cannot disagree about what a facing is. Signatures as of `src/shelf_benchmark/approaches/base.py`:

| Helper | Signature | Returns |
| :--- | :--- | :--- |
| Active GenAI Client | `get_client(location=None)` | Injected fake/UniversalModelSpec client or Vertex AI client |
| Prior Stage-1 Boxes | `get_prior_detected_boxes(prior_detection=None)` | `List[Dict[str, Any]]` of front-facing boxes from a chained Stage-1 detector |
| Depth dedup | `deduplicate_depth_stacked_facings(items, x_overlap_threshold=None)` | `(front_facings, filtered_count)`; threshold defaults to `depth_deduplication.x_overlap_threshold` |
| Size bucket | `derive_size_bucket_from_bbox(bbox_2d, all_bboxes_on_shelf, packaging_type="tube", model_size_hint="")` | bucket label from the configured taxonomy |
| HUL check | `check_is_hul_brand(brand_name, model_predicted=None)` | `bool` |
| Load the image | `load_shelf_image(record)` | PIL image, honouring `record.local_shelf_image_path` |
| Token extraction | `extract_tokens(response)` | `TokenUsageMetrics` |
| Cost | `compute_cost(tokens, model_name, product_count, extra_api_cost_usd=0.0, latency_ms=0.0, run_id=None, approach_id="custom_approach", task_type="classification")` | `CostMetrics` |
| Scoring | `evaluate_accuracy(task_type, rows, gt_record, depth_duplicates_filtered=0)` | `AccuracyMetrics` |
| Cropping | `crop_facing_images(shelf_image_uri, facings, model_tag, local_fallback=None)` | `(crops, montage_bytes, montage_path)` |
| Execute w/ Pipeline | `execute_with_pipeline(approach_id, model_name, record, invoke_fn, ...)` | `TaskExecutionResult` (with automatic retries, CPU time, cost, accuracy, OTel, and trace) |
| Finalize result | `finalize(approach_id, model_name, record, raw_outputs, start_dt, end_dt, gt_record=None, ...)` | `TaskExecutionResult` (rows + cost + accuracy + OTel + execution_trace) |

Dataclass fields on `ctx`: `config`, `storage`, `telemetry`, `reports_dir`, `genai_client`, `prior_detection`.

### Standalone Object Detection Plugins & Chaining Stage-1 → Stage-2

If you are benchmarking a **pure Object Detection approach** (which predicts only `[ymin, xmin, ymax, xmax]` boxes and `confidence`, with no brand/attribute labels), set `task_type="detection"` (or simply return dicts containing only `bbox_2d` — `ctx.infer_effective_task_type` automatically infers `"detection"` so your detector is scored purely on IoU / Precision / Recall / F1 / `AP@50` / `mAP@50:95` without being penalized on unpredicted brand or taxonomy attributes):

```python
@register_approach_function(
    approach_id="my_stage1_detector",
    display_name="My Custom Stage-1 Detector",
    task_type="detection",
)
def run_my_stage1_detector(ctx: CommonLayerContext, model_name: str, shelf_image_uri: str):
    raw_boxes = [{"bbox_2d": [100, 100, 300, 200], "confidence": 0.96}]
    front_facings, filtered = ctx.deduplicate_depth_stacked_facings(raw_boxes)
    if front_facings:
        front_facings[0]["_depth_filtered"] = filtered
    return front_facings
```

You can also **chain any Stage-1 detector into any Stage-2 classifier** without re-running detection:

```python
summary = sdk.run_suite(
    tasks=["detection", "classification"],
    detector_approach="my_stage1_detector",
    reuse_prior_detection=True,
    approaches=["two_stage_bbox_guided_nms", "two_stage_physical_crop_per_facing"],
)
```

### Running and testing your approach in 3 lines (Offline or CLI)

```python
from shelf_benchmark.testing import run_offline_approach

# Runs offline in <50ms, scores against sample ground truth, and returns full execution trace:
res = run_offline_approach("/tmp/my_experiment", "my_new_cv_approach", with_ground_truth=True)
print(res.accuracy.detection_f1, res.execution_trace["call_topology"])
```

Or from the command line using `--plugin-module` (no need to move your script):

```bash
.venv/bin/shelf-benchmark run --offline \
  --plugin-module code_samples/05_create_custom_approach_plugin.py \
  --approaches custom_yolo_plus_gemma_verifier
```

---

## Step 4: promote to a plugin folder & inspect in Local UI / Cloud Run

When the approach is worth keeping, create `src/shelf_benchmark/approaches/<your_approach_id>/plugin.py` subclassing `SimpleShelfApproachPlugin` (~15 lines — see [`src/shelf_benchmark/approaches/TEMPLATE_NEW_APPROACH.md`](src/shelf_benchmark/approaches/TEMPLATE_NEW_APPROACH.md)):

1. Only implement `detect_and_classify(self, ctx, model_name, record) -> List[Dict[str, Any]]`.
2. Confirm auto-discovery: `.venv/bin/shelf-benchmark list-approaches` shows it with `source=plugin`.
3. Launch the **Local Interactive UI & Trace Studio** (identical container to Cloud Run):
   ```bash
   .venv/bin/python ui/server.py
   # Open http://127.0.0.1:8080 -> "Run Live Studio" tab
   ```
   In **Run Live Studio**, your new plugin automatically appears in the Approach dropdown. Choose:
   - **Execution Environment**:
     - `Local Machine + Live Vertex AI / Agent Platform Calls` *(default: runs locally with local files while making live calls to Vertex AI / Agent Platform)*
     - `Live Vertex AI + GCP Cloud Run & Cloud Logging` *(full GCP cloud storage & logging integration)*
     - `Offline Unit-Test Stub (Canned Responses • Zero API Calls)` *(instant synthetic test harness)*
   - **Ground Truth Scoring**: `Placeholder Mode (Accuracy = None)` or `Connect Sample GT (Score Precision / Recall / F1 / IoU)`
   - Inspect the **Engineering & Onboarding Traceability Inspector** at the bottom of any run to see:
     1. **Call Pattern & Implementation**: 1 call vs 2 calls vs 3 stages, and which models/APIs (`gemini-3.8-flash`, `multimodalembedding@001`, `gemini-embedding-001`) were invoked per stage.
     2. **Separated 5-Bucket GCP Cost & Tokens**: Vertex AI PAYG tokens, Embeddings/Vision API, Provisioned Throughput GSU, Cloud Run compute, GCS/Observability, and `billing_source`.
     3. **Accuracy, Recall & Ground Truth Status**: Precision/Recall/F1, `AP@50`, `mAP@50:95`, Mean IoU, Brand/Product/Count Accuracy, and why unmeasured metrics are `None` (never `0.0`).
     4. **OpenTelemetry Trace Lookup**: Exact `TraceId`, `SpanId`, copyable `jq` command for `reports/otel_logs.jsonl`, and copyable GCP Cloud Logging query (`logName=... AND trace=...`).

---

## Step 5: test different models

```python
from shelf_benchmark import ModelPricing, UniversalModelSpec

sdk.register_model(
    UniversalModelSpec(
        model_id="gemma-3-27b-it",
        display_name="gemma-3-27b-it-endpoint",
        # "vertex_gemini" | "vertex_geap" | "vertex_gemma" | "vertex_tuned_endpoint" | "custom_callable"
        provider_family="vertex_gemma",
        endpoint_uri="projects/unilever-shelf-understanding/locations/us-central1/endpoints/YOUR_ENDPOINT_ID",
        pricing=ModelPricing(input=0.08, thinking=0.0, output=0.24),
    )
)
```

`sdk.swap_models(models)` takes a sequence of names or specs and returns the resulting model list.

Always pass `pricing=`. Vertex AI returns token counts, not dollars, so an unpriced model falls
back to the `default` rate card and its cost column becomes an estimate of an estimate.

Worked examples: [GEAP and preview models](code_samples/02_benchmark_geap_and_gemini_models.py),
[Gemma, Model Garden and any HTTP server](code_samples/03_benchmark_gemma_and_model_garden.py),
[fine-tuned endpoints](code_samples/04_fine_tuning_sft_and_tuned_endpoints.py).

---

## Step 6: connect ground truth when it arrives

No code changes. Full specification in
[`docs/GROUND_TRUTH_CONTRACT.md`](docs/GROUND_TRUTH_CONTRACT.md); a complete worked annotation file
ships as `configs/sample_ground_truth.json`.

### Option A: one call in Python

```python
sdk.connect_ground_truth(
    provider_type="json",          # "json" | "jsonl" | "csv" | "coco" | "bigquery" | "none"
    source_uri="gs://unilever-shelf-understanding-shelf-images/gt/ground_truth.json",
    gt_version="v1",
    bbox_format="ymin_xmin_ymax_xmax_1000",  # or "coco_xywh_px" | "xyxy_px" | "xyxy_norm" | "yxyx_norm" | "yolo_xywh_norm"
    strict=True,
    schema_mapping={               # only needed when the vendor uses other field names
        "image_key_field": "image_id",
        "items_list_field": "items",
        "brand_field": "brand",
        "product_name_field": "product_name",
        "sku_id_field": "sku_id",
        "bbox_field": "bbox_2d",
        "shelf_row_field": "shelf_row",
    },
)
```

It returns a summary dict (`images`, `items`, `excluded_back_row`, `gt_version`, `summary`).
A mistyped mapping key raises immediately rather than being ignored.

### Option B: configuration

Edit the `ground_truth:` block in
[`configs/default_config.yaml`](configs/default_config.yaml), which documents every field inline,
including all six `bbox_format` options (`ymin_xmin_ymax_xmax_1000`, `coco_xywh_px`, `xyxy_px`,
`xyxy_norm`, `yxyx_norm`, and `yolo_xywh_norm`), 4-column bounding-box fields, and polygon `vertices`.
Every CLI, UI and SDK run then scores automatically.

### Verify before you believe

```bash
.venv/bin/shelf-benchmark validate-gt \
  --gt-provider json \
  --ground-truth-uri configs/sample_ground_truth.json
```

It prints how many images and items parsed, how many back-row items were excluded, the
`gt_version`, and sample entries with their boxes. A declared bbox format that does not match the
data produces plausible-looking but wrong numbers rather than an error, which is why this step
exists.

### Re-score a run you already paid for

```bash
.venv/bin/shelf-benchmark score \
  --predictions reports/predictions.json \
  --gt-provider json --ground-truth-uri annotations_v1.json --gt-version v1
```

End-to-end walkthrough: [`code_samples/10_run_now_score_later_ground_truth.py`](code_samples/10_run_now_score_later_ground_truth.py).

### What gets computed once annotations are connected

| Metric | Meaning |
| :--- | :--- |
| `detection_precision`, `detection_recall`, `detection_f1` | Localization at the threshold reported separately as `iou_threshold` (default `0.50`), using `pairing_strategy` (`"iou_greedy"` or `"optimal"` / `"hungarian"`). |
| `average_precision_at_50`, `map_50_95`, `pr_curve_points` | All-point interpolated `AP@50`, COCO `mAP@[.50:.05:.95]`, and confidence-ranked PR curve points. |
| `mean_iou`, `mean_iou_matched` | Average IoU over all predictions, and over matched pairs only. |
| `count_accuracy` | Facing count after depth deduplication versus annotated count. |
| `brand_classification_accuracy`, `brand_set_recall` | Per-facing brand accuracy, and shelf-level brand recall. |
| `product_classification_accuracy`, `sku_matching_accuracy` | Product name and canonical SKU accuracy. |
| `per_attribute_accuracy`, `macro_attribute_accuracy` | Per-attribute accuracy across all 8 core dimensions + any `custom_attributes` (`>8` attributes), and their unweighted macro average. |
| Row-level columns | `gt_item_id`, `gt_brand`, `gt_product_name`, `gt_sku_id`, `iou_with_gt`, `iou_threshold`, `brand_correct`, `product_correct`, `sku_correct`, `gt_version`, `gt_status`. |

> [!NOTE]
> These were previously named `detection_precision_iou50`, `detection_recall_iou50` and
> `detection_f1_iou50`, and the `50` was wrong: the code compared at `0.25`. The threshold is now
> configurable (`evaluation.iou_threshold`), reported as data, and stamped on every row. Changing
> it invalidates comparison with runs scored at a different value.

> [!IMPORTANT]
> Unmeasured metrics are `None`, not `0.0`. That applies to the counts too:
> `true_positives`, `false_positives`, `false_negatives` and `matched_pairs` are `None` when there
> is no ground truth, or when the predictions carry no geometry to pair on.
