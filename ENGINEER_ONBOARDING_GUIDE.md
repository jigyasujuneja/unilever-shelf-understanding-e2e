# Engineer onboarding guide

From zero context to a benchmarked approach of your own. Five steps, about half an hour.

Five engineers work in this repository in parallel. Nothing below requires editing a shared file,
so you should never need to resolve a merge conflict to run an experiment.

---

## Step 0: the problem, in 60 seconds

Given a photograph of a retail shelf, find every product facing and say what it is.

A **facing** is the front-most visible unit in one horizontal slot. A unit stacked behind another
unit in the same column is not a facing, and counting it twice inflates share of shelf. Depth
deduplication is therefore part of the pipeline, not an afterthought.

1. **Detect facings** (`detection`): `[ymin, xmin, ymax, xmax]` boxes on a `0..1000` normalised
   scale, back-row duplicates suppressed.
2. **Classify seven dimensions** (`classification`): `category`, `subcategory`, `brand` (plus the
   `is_hul_brand` flag), `variant`, `packaging_type`, `pack_type`, `size` (OCR text plus a
   rule-derived size bucket). With `product_name` that is the eight attributes reported per row.
3. **Match to catalog SKUs** (`matching`): hybrid search, sparse BM25 keywords plus dense
   embeddings, and optional planogram compliance.
4. **Measure the three-way trade-off**: accuracy (detection precision / recall / F1 at a stated
   `iou_threshold`, count accuracy, brand / product / SKU accuracy) against latency (ms per image
   and per facing) against GCP cost (dollars per image and per facing).

> [!IMPORTANT]
> Ground truth does not exist yet. Until it does, accuracy metrics are `None` with
> `accuracy_status="PLACEHOLDER_AWAITING_GROUND_TRUTH"`, never `0.0`. Latency, tokens and cost are
> real from day one. Every run writes `predictions.json`, so when annotations arrive you re-score
> what you already ran instead of paying for inference again.

Before you trust any accuracy number, read
[`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md). It defines every metric and works
through the arithmetic on a three-facing example.

---

## Step 1: verify your environment (2 minutes)

```bash
# The test suite: plugin discovery, depth NMS, IoU, pairing, billing, ground-truth swap.
.venv/bin/pytest -q

# A benchmark run with no GCP project, no credentials and no network.
.venv/bin/python code_samples/01_quickstart_run_full_suite.py

# What approaches exist right now.
.venv/bin/shelf-benchmark list-approaches
```

The quickstart writes to a temp directory and prints the paths: `benchmark_report.md` (the summary
table), `row_level_report.csv` / `.json` (one row per facing with box, seven dimensions, latency,
tokens, cost and, once annotations exist, `iou_with_gt`, `gt_brand`, `brand_correct` and friends),
`predictions.json` (re-scorable later) and `otel_logs.jsonl` (spans, tokens, cost).

---

## Step 2: the offline test lane

Use it for everything except the runs whose point is the live model. It is deterministic, free,
and needs no credentials, which matters when five people are iterating at once.

```python
from shelf_benchmark import UniversalModelSpec
from shelf_benchmark.testing import (
    OFFLINE_IMAGE_URI, make_offline_sdk, perfect_prediction_payload,
)

sdk = make_offline_sdk("/tmp/my-experiment")   # every GCP switch off, artifacts under tmp_dir
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
| `make_offline_sdk(tmp_dir, **overrides)` | An SDK with offline config: no Cloud Logging, no GCS sync, no live billing catalog, artifacts under `tmp_dir`. |
| `offline_config(tmp_dir, **overrides)` | The same config, if you want to edit it before building the SDK. |
| `OFFLINE_IMAGE_URI`, `fixture_image_path()` | The bundled 600x400 fixture shelf image. |
| `sample_ground_truth()`, `write_sample_ground_truth_file(path)` | Three annotated facings with known-correct answers. |
| `perfect_prediction_payload()` | A prediction that scores 1.0 against that ground truth. |
| `shifted_prediction_payload()`, `wrong_brand_payload()` | Predictions that should move detection and classification metrics respectively, and only those. |
| `FakeGenAIClient(payload)` | One canned response for every call. For tests that construct a task directly with `genai_client=`. |
| `ReplayGenAIClient(responses)`, `.from_file(path)` | A recorded sequence, replayed in order. For multi-stage approaches, and for turning one real run into a regression test. |

> [!WARNING]
> Offline mode gates Cloud Storage, not model calls. If you pass a real model id such as
> `gemini-3.8-flash` to `run_suite`, the suite builds a real Vertex AI client and issues a
> billable request even with `offline.enabled = True`. Register a `custom_callable` stand-in as
> above. Assigning `sdk.client` to a fake does **not** intercept `run_suite`.

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
| Depth dedup | `deduplicate_depth_stacked_facings(items, x_overlap_threshold=None)` | `(front_facings, filtered_count)`; threshold defaults to `depth_deduplication.x_overlap_threshold` |
| Size bucket | `derive_size_bucket_from_bbox(bbox_2d, all_bboxes_on_shelf, packaging_type="tube", model_size_hint="")` | bucket label from the configured taxonomy |
| HUL check | `check_is_hul_brand(brand_name, model_predicted=None)` | `bool` |
| Load the image | `load_shelf_image(record)` | PIL image, honouring `record.local_shelf_image_path` |
| Token extraction | `extract_tokens(response)` | `TokenUsageMetrics` |
| Cost | `compute_cost(tokens, model_name, product_count, extra_api_cost_usd=0.0, latency_ms=0.0, run_id=None, approach_id="custom_approach", task_type="classification")` | `CostMetrics` |
| Scoring | `evaluate_accuracy(task_type, rows, gt_record, depth_duplicates_filtered=0)` | `AccuracyMetrics` |
| Cropping | `crop_facing_images(shelf_image_uri, facings, model_tag, local_fallback=None)` | `(crops, montage_bytes, montage_path)` |

Dataclass fields on `ctx`: `config`, `storage`, `telemetry`, `reports_dir`.

> [!CAUTION]
> `ctx.crop_facing_images` currently raises `TypeError` on every call: it forwards
> `local_fallback=` to `facing_utils.crop_detected_facings`, which does not accept that parameter.
> Until that is fixed, avoid the helper (and the `two_stage_physical_crop_per_facing` and
> `class_agnostic_visual_embedding` approaches that depend on it), or call
> `facing_utils.crop_detected_facings(storage, shelf_image_uri, detected_items,
> output_crop_dir, model_tag)` directly.

> [!NOTE]
> `derive_size_bucket_from_bbox` takes `all_bboxes_on_shelf` on `ctx`, but the underlying
> `facing_utils` function calls the same argument `all_bboxes_on_row`. Match whichever one you are
> calling.

### Running it

```python
from shelf_benchmark import ShelfBenchmarkSDK

sdk = ShelfBenchmarkSDK(output_dir="reports/my_experiment")   # isolated per engineer
summary = sdk.run_suite(
    models=["gemini-3.8-flash", "gemini-3.5-flash-lite"],
    tasks=["classification"],
    approaches=["single_pass_full_shelf", "my_new_cv_approach"],
    shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
)
```

Or from the command line, once the module defining the approach has been imported:

```bash
.venv/bin/shelf-benchmark run --approaches my_new_cv_approach --models gemini-3.8-flash
```

`--approaches` has no hardcoded list of valid values, so plugins work from the CLI. An unrecognised
id fails loudly with the list of registered ids rather than quietly running something else.

---

## Step 4: promote it to a plugin folder

When the approach is worth keeping, move it into the package so the CLI, the UI and everyone else
pick it up without importing your script.

1. Create `src/shelf_benchmark/approaches/<your_approach_id>/` with `__init__.py` and `plugin.py`.
2. In `plugin.py`, either define a non-abstract subclass of `BaseShelfApproachPlugin`, or define a
   `get_plugins()` function returning instances. The factory form is preferred when one class
   serves several approach ids; see
   [`src/shelf_benchmark/approaches/vlm_existing_approaches/plugin.py`](src/shelf_benchmark/approaches/vlm_existing_approaches/plugin.py),
   where a single class registers the three built-in VLM approaches. Start from
   [`src/shelf_benchmark/approaches/TEMPLATE_NEW_APPROACH.md`](src/shelf_benchmark/approaches/TEMPLATE_NEW_APPROACH.md).
3. Implement the abstract surface: properties `approach_id`, `display_name`, `category`,
   `stages_description`; optional class attribute `is_demo_only: bool = False`; and
   `execute(ctx, model_name, record, gt_record=None, prior_detection=None) -> TaskExecutionResult`.
4. Confirm discovery: `.venv/bin/shelf-benchmark list-approaches` should show it with
   `source=plugin`.

Discovery runs once, eagerly, through `GLOBAL_APPROACH_REGISTRY.ensure_discovered()`. Two details
are worth knowing, because both were bugs once:

* Discovery is no longer conditional on the registry being empty. It used to be, so a user
  approach registered at import time hid every built-in approach.
* A plugin that fails to import **raises** `ApproachPluginError` instead of being skipped. A
  silently missing approach looks exactly like an approach that scored badly. Set
  `SHELF_BENCH_TOLERANT_PLUGINS=1` if you deliberately want a broken plugin to be skipped.

Mark illustrative approaches with `is_demo_only = True` so they cannot be mistaken for a candidate
in a decision-making comparison.

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
    bbox_format="ymin_xmin_ymax_xmax_1000",
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
including all five `bbox_format` options. Every CLI, UI and SDK run then scores automatically.

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
| `detection_precision`, `detection_recall`, `detection_f1` | Localization, at the threshold reported separately as `iou_threshold` (default `0.50`). |
| `mean_iou`, `mean_iou_matched` | Average IoU over all predictions, and over matched pairs only. |
| `count_accuracy` | Facing count after depth deduplication versus annotated count. |
| `brand_classification_accuracy`, `brand_set_recall` | Per-facing brand accuracy, and shelf-level brand recall. |
| `product_classification_accuracy`, `sku_matching_accuracy` | Product name and canonical SKU accuracy. |
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
