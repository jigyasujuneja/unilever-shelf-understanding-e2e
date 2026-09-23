# Code Samples & Engineer Onboarding Cookbook

Eleven self-contained, runnable Python scripts that walk through every workflow in the Shelf Understanding Benchmark Suite.

- **Local Live Vertex AI / Agent Platform vs. Offline Unit-Test Stubs**: Seven samples run **100% offline with zero GCP credentials** against the bundled quickstart fixture (`shelf_sample_01.png`) when using `benchmark_harness()` or `offline-demo-model`, and seamlessly switch to **live Vertex AI / Agent Platform calls** locally when `--live` or a real model ID (`gemini-3.8-flash`) is passed to `make_offline_sdk()` or `ShelfBenchmarkSDK()`. In staging and production, point `sdk.connect_dataset` at multi-image GCS buckets or BigQuery tables of any image resolution.
- **Ground-Truth Scope (Ingestion Only)**: This benchmark suite does **not** author or produce ground-truth annotations. It ingests existing ground-truth datasets provided externally (`sdk.connect_ground_truth`). Before ground truth is connected, accuracy metrics report `None` with `accuracy_status="PLACEHOLDER_AWAITING_GROUND_TRUTH"`.

---

## 1. Quick Reference Table

| Script | Runs Offline | What You Learn |
| :--- | :---: | :--- |
| [`01_quickstart_run_full_suite.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/01_quickstart_run_full_suite.py) | **Yes** (`--live` optional) | Run single-step (`single_pass_full_shelf`, `single_step_detect_classify_and_match`) and two-stage (`two_stage_bbox_guided_nms`, `two_stage_physical_crop_per_facing`) approaches end-to-end with `benchmark_harness()`. |
| [`02_benchmark_geap_and_gemini_models.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/02_benchmark_geap_and_gemini_models.py) | Registration (`--live` to call API) | Register early-access (`vertex_geap`) and production Gemini models (`gemini-3.8-flash`, `gemini-3.8-pro`) with custom pricing cards and regional endpoints. |
| [`03_benchmark_gemma_and_model_garden.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/03_benchmark_gemma_and_model_garden.py) | **Yes** | Attach open-weights models (`gemma-3-27b-it`, PaliGemma 2) via Vertex Model Garden or any local/HTTP inference server using `custom_callable`. |
| [`04_fine_tuning_sft_and_tuned_endpoints.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/04_fine_tuning_sft_and_tuned_endpoints.py) | Shape preview (`--live` to submit) | Generate Vertex AI Supervised Fine-Tuning (SFT) JSONL datasets (`shelf_sft_train.jsonl`) and benchmark fine-tuned endpoints. |
| [`05_create_custom_approach_plugin.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/05_create_custom_approach_plugin.py) | **Yes** | Write a custom approach in ~15 lines with `@register_approach_function` or `SimpleShelfApproachPlugin`, using `ctx` helpers for depth deduplication and size bucketing. |
| [`06_custom_taxonomy_and_ground_truth_swap.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/06_custom_taxonomy_and_ground_truth_swap.py) | **Yes** | Configure **>8 attributes** (`custom_attributes` + `attribute_call_groups` for 1-call vs grouped VLM calls) and plug in any GCP dataset (`sdk.connect_dataset`) or ground-truth schema (`sdk.connect_ground_truth`). |
| [`07_gcp_billing_provisioned_throughput_and_cloud_run.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/07_gcp_billing_provisioned_throughput_and_cloud_run.py) | **Yes** (`--live` for SKU API) | Inspect the 5-bucket GCP cost breakdown (Token Usage, Provisioned Throughput, Embeddings/Vision, Cloud Run Compute, Storage/Egress) and BigQuery Billing Export SQL. |
| [`08_test_live_cloud_run_deployment.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/08_test_live_cloud_run_deployment.py) | Request preview (`--live` to invoke) | Authenticate with IAM (`gcloud auth print-identity-token`) and trigger the deployed Cloud Run benchmark service. |
| [`09_step_by_step_engineer_ground_truth_benchmark.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/09_step_by_step_engineer_ground_truth_benchmark.py) | **Yes** | Step-by-step walkthrough showing how Hungarian IoU matching separates bounding-box geometry (`detection_f1`) from attribute naming (`brand_accuracy`, `per_attribute_accuracy`). |
| [`10_run_now_score_later_ground_truth.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/10_run_now_score_later_ground_truth.py) | **Yes** | Run benchmarks while ground truth is on `PLACEHOLDER_AWAITING_GROUND_TRUTH`, save `predictions.json`, and score/re-score offline in <1s (`score_predictions`) once ground truth is connected. |
| [`11_complete_engineer_approach_playground.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/11_complete_engineer_approach_playground.py) | **Yes** (`--live` optional) | **All-in-One Engineer Playground**: Write a custom approach, swap models, configure >8 attributes (`12` attributes in 1-call or grouped VLM calls), select hardware/accelerator (`none`, `nvidia-l4`, `tpu-v5e`, `tpu-v6e`), and inspect `diagnostic_trace_report.md`. |

---

## 2. Built-in Approaches (Single-Step, Multi-Attribute & Multi-Stage)

All 8 registered approaches can be listed via `.venv/bin/shelf-benchmark list-approaches` and tested offline or live:

| Approach ID | Call Topology | What It Does |
| :--- | :--- | :--- |
| `single_pass_full_shelf` | **1 API Call (Single-Step Detect + Classify)** | Detects all front-row facings (`bbox_2d`) and extracts all configured attributes (8 core + any `custom_attributes`) in a single structured VLM call, followed by deterministic 1D-NMS depth deduplication. |
| `open_vocab_brand_plus_catalog_resolver` | **1 VLM Call + $O(1)$ Brand Resolver** | Open-vocabulary LLM brand generation followed by $O(1)$ canonical brand resolution (`resolve_brand_against_catalog`) against a 2,000+ brand catalog. |
| `configurable_multi_attribute_vlm` | **1 Call or Grouped Multi-Call VLM (>8 Attributes)** | Predicts arbitrarily many attributes (`taxonomy.custom_attributes`): runs in **1 VLM call** when `attribute_call_groups` is empty/1-group, or executes **1 targeted VLM call per attribute group** when `attribute_call_groups` splits attributes by type (e.g. OCR identity vs physical/promo attributes). |
| `single_step_detect_classify_and_match` | **1 API Call (Single-Step Detect + Classify + Match)** | Performs full-shelf facing detection, N-dimension classification, and hybrid SKU catalog matching (`matched_sku_id`, lexical + dense descriptors) in a single VLM pass. |
| `two_stage_bbox_guided_nms` | **2 API Calls (2-Stage VLM)** | Stage 1 detects bounding boxes and filters back-row duplicates via 1D-NMS; Stage 2 classifies only the surviving front-row coordinates. |
| `two_stage_physical_crop_per_facing` | **2 Stages (Detect + Crop + Classify)** | Stage 1 detects boxes; physically crops each facing (`crop_detected_facings`); Stage 2 classifies high-resolution crops. |
| `class_agnostic_visual_embedding` | **3 Stages (Detect + Embed + Match)** | Class-agnostic detection -> crops -> `multimodalembedding@001` cosine similarity against reference catalog. |
| `cloud_vision_visual_embedding` | **3 Stages (Cloud Vision + Embed + Match)** | Cloud Vision `OBJECT_LOCALIZATION` -> `multimodalembedding@001` catalog search. |

---

## 3. Predicting >8 Attributes (Single VLM Call vs Grouped VLM Calls)

Add any number of custom attributes in the packaged `shelf_benchmark/_resources/taxonomy.yaml` or Python (`TaxonomyConfig.custom_attributes`):

```python
from shelf_benchmark import CustomAttributeSpec, TaxonomyConfig

taxonomy = TaxonomyConfig(
    custom_attributes={
        "price_tag_visible": CustomAttributeSpec(description="Price tag visible below facing", value_type="boolean"),
        "promo_callout": CustomAttributeSpec(description="Promo badge or discount text", value_type="string"),
        "facing_orientation": CustomAttributeSpec(description="Orientation on shelf", allowed_values=["front_straight", "tilted", "sideways"]),
        "shelf_talker_present": CustomAttributeSpec(description="Promotional wobbler/talker attached", value_type="boolean"),
    },
    # Leave `attribute_call_groups=[]` to predict all 12 attributes in a SINGLE VLM call,
    # OR group attributes by type so `configurable_multi_attribute_vlm` runs 1 VLM call per group:
    attribute_call_groups=[
        ["category", "subcategory", "brand", "product_name", "variant"],
        ["packaging_type", "pack_type", "size", "price_tag_visible", "promo_callout", "facing_orientation", "shelf_talker_present"],
    ],
)
```

Every predicted custom attribute is saved on each row in `row.extra_attributes` and automatically evaluated in `result.accuracy.per_attribute_accuracy` and `result.accuracy.macro_attribute_accuracy` when present in your ground-truth dataset.
