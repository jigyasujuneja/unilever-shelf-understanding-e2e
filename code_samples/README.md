# Developer Code Samples — Unilever Shelf Understanding Benchmark SDK

This directory (`code_samples/`) provides generic, copy-pasteable Python scripts showing how any developer can plug in **new models (Gemini, GEAP Early-Access, Gemma / Model Garden, Fine-Tuned Endpoints)** or **new CV/VLM approaches** and run the full benchmark suite with **automatic OpenTelemetry compliance (`otel_logs.jsonl`)**, **7-dimension taxonomy validation (`configs/taxonomy.yaml`)**, and **row-level CSV/JSON/Markdown reporting**.

---

## Included Code Samples

| File | Use Case | What It Demonstrates |
| :--- | :--- | :--- |
| [`01_quickstart_run_full_suite.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/01_quickstart_run_full_suite.py) | **Out-of-the-Box Full Suite** | Run `detection`, `classification` (all 4 approaches), `matching`, and `fine_tuning` in 15 lines of Python with full OpenTelemetry logging. |
| [`02_benchmark_geap_and_gemini_models.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/02_benchmark_geap_and_gemini_models.py) | **GEAP & Preview Gemini Models** | Register and benchmark **Google Early Access Program (GEAP)** models (`api_version="v1beta1"` / `"v1alpha"`, preview model IDs, custom regional endpoints, custom token pricing). |
| [`03_benchmark_gemma_and_model_garden.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/03_benchmark_gemma_and_model_garden.py) | **Gemma 3 / PaliGemma / Model Garden / vLLM** | Benchmark open-weights **Gemma** models deployed on Vertex AI Model Garden / MaaS or any custom Cloud Run / vLLM / Python callable endpoint with automatic OTel token & cost tracking. |
| [`04_fine_tuning_sft_and_tuned_endpoints.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/04_fine_tuning_sft_and_tuned_endpoints.py) | **Supervised Fine-Tuning (SFT) & Tuned Endpoints** | Generate SFT JSONL datasets, launch Vertex AI `client.tunings.tune(...)` jobs, and benchmark deployed **Fine-Tuned Endpoints** (`projects/.../locations/.../endpoints/...`) against base models. |
| [`05_create_custom_approach_plugin.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/05_create_custom_approach_plugin.py) | **Plug In a New CV / VLM Approach** | Create a brand-new separation/detection/embedding approach in ~25 lines using `@register_approach_function` and run the full benchmark + OTel suite on it. |
| [`06_custom_taxonomy_and_ground_truth_swap.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/06_custom_taxonomy_and_ground_truth_swap.py) | **Custom Taxonomy & Ground Truth Schema Swap** | Override `TaxonomyConfig` (categories, subcategories, packaging types, size rules, brands) and connect BigQuery/CSV/JSON Ground Truth & Association tables without modifying code. |
| [`07_gcp_billing_provisioned_throughput_and_cloud_run.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/07_gcp_billing_provisioned_throughput_and_cloud_run.py) | **Live GCP Cloud Billing API, Provisioned Throughput (GSU), Cloud Run & BigQuery Cost Reconciliation** | Query Google Cloud Billing Catalog API live, compare On-Demand PAYG vs. Provisioned Throughput (GSU) vs. Cloud Run compute costs with 100% 5-bucket separation, generate BigQuery Billing Export SQL, and export OpenTelemetry logs directly to GCP Cloud Logging & GCS. |

---

## Running Any Sample

```bash
# 1. Run the Quickstart sample
.venv/bin/python code_samples/01_quickstart_run_full_suite.py

# 2. Run the Custom Approach Plugin sample
.venv/bin/python code_samples/05_create_custom_approach_plugin.py

# 3. Run the Gemma & Model Garden Adapter sample
.venv/bin/python code_samples/03_benchmark_gemma_and_model_garden.py
```

---

## Automatic OpenTelemetry Guarantees

Every sample uses [`ShelfBenchmarkSDK`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/src/shelf_benchmark/sdk.py) and [`OpenTelemetryBenchmarkLogger`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/src/shelf_benchmark/telemetry.py), which automatically records for **every single run**:
- `start_time` & `end_time` (ISO-8601 UTC + `start_time_unix_nano` & `end_time_unix_nano`)
- `TraceId` & `SpanId` (W3C TraceContext compliant)
- `gen_ai.usage.input_tokens`, `gen_ai.usage.thinking_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.total_tokens`
- `shelf_benchmark.cost_per_shelf_image_usd` & `shelf_benchmark.cost_per_product_usd`
- `shelf_benchmark.latency_ms` & `shelf_benchmark.product_count`
