# Code samples

Ten runnable scripts. Each one is a single file, prints what it did, and states plainly whether
it needs a GCP project.

Six of them run with **no GCP project, no credentials and no network**, against the 600x400
fixture image bundled in `src/shelf_benchmark/_fixtures/`. The four that genuinely need Google
Cloud print what they would do and exit unless you pass `--live`.

## The samples

| File | Runs offline | What it shows |
| :--- | :--- | :--- |
| [`01_quickstart_run_full_suite.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/01_quickstart_run_full_suite.py) | yes | Detection and classification across two built-in approaches, all artifacts written. Start here. `--live` runs the same thing against Vertex AI. |
| [`02_benchmark_geap_and_gemini_models.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/02_benchmark_geap_and_gemini_models.py) | registration only | Registering a GEAP / preview model: allowlisted model id, `api_version`, regional endpoint, its own rate card. `--live` to benchmark it. |
| [`03_benchmark_gemma_and_model_garden.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/03_benchmark_gemma_and_model_garden.py) | yes | Two ways to attach a non-Gemini model: a Model Garden endpoint, and any HTTP server wrapped in a `custom_callable` handler. The handler path runs for real. |
| [`04_fine_tuning_sft_and_tuned_endpoints.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/04_fine_tuning_sft_and_tuned_endpoints.py) | shape only | The Vertex AI SFT JSONL record shape and tuned-endpoint registration. `--live` generates and uploads the dataset; `--live --submit-tuning-job` submits the job. |
| [`05_create_custom_approach_plugin.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/05_create_custom_approach_plugin.py) | yes | Writing your own approach with `@register_approach_function`, including depth deduplication and rule-derived size buckets through the shared `ctx` helpers. |
| [`06_custom_taxonomy_and_ground_truth_swap.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/06_custom_taxonomy_and_ground_truth_swap.py) | yes | Taxonomy lives in one place and drives the prompt, the size rule and the HUL check. Then swaps in `configs/sample_ground_truth.json` and loads it. |
| [`07_gcp_billing_provisioned_throughput_and_cloud_run.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/07_gcp_billing_provisioned_throughput_and_cloud_run.py) | yes | The five cost buckets, `include_infrastructure_costs` off and on, provisioned throughput, and the BigQuery reconciliation SQL. `--live` resolves live SKU rates. |
| [`08_test_live_cloud_run_deployment.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/08_test_live_cloud_run_deployment.py) | request shape only | Calling the headless Cloud Run worker with an IAM ID token. Needs `roles/run.invoker`, so it is `--live` only. |
| [`09_step_by_step_engineer_ground_truth_benchmark.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/09_step_by_step_engineer_ground_truth_benchmark.py) | yes | Your approach scored against real annotations, with detection perfect and brand accuracy at 2/3, to show geometry and naming are scored separately. |
| [`10_run_now_score_later_ground_truth.py`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/code_samples/10_run_now_score_later_ground_truth.py) | yes | The workflow this project runs on: benchmark now with no annotations, re-score for free when they arrive, re-score again under a new `gt_version` when they are corrected. |

## Running them

```bash
.venv/bin/python code_samples/01_quickstart_run_full_suite.py
.venv/bin/python code_samples/10_run_now_score_later_ground_truth.py
.venv/bin/python code_samples/09_step_by_step_engineer_ground_truth_benchmark.py

# Live variants, which need `gcloud auth application-default login` and spend money:
.venv/bin/python code_samples/01_quickstart_run_full_suite.py --live
```

## How the offline lane works

`shelf_benchmark.testing.make_offline_sdk(tmp_dir)` builds a config with every GCP switch turned
off explicitly, writes all artifacts under `tmp_dir`, and points at the bundled fixture image
(`shelf_benchmark.testing.OFFLINE_IMAGE_URI`).

> [!IMPORTANT]
> Offline mode gates Google Cloud Storage, not model calls. If you pass a real model id such as
> `gemini-3.8-flash` to `run_suite`, the suite builds a real Vertex AI client and issues a
> billable request even with `offline.enabled = True`. To stay offline, register a stand-in model:
>
> ```python
> sdk.register_model(UniversalModelSpec(
>     model_id="offline-demo-model",
>     display_name="offline-demo-model",
>     provider_family="custom_callable",
>     custom_handler=lambda prompt, image_uri, schema: perfect_prediction_payload(),
> ))
> ```
>
> `FakeGenAIClient` and `ReplayGenAIClient` exist for tests that construct a task directly with
> `genai_client=`. Assigning `sdk.client` does not affect `run_suite`.

Approaches that register through `@register_approach_function` never call a model at all, so they
are offline by construction (samples 05 and 09).

## What every run records

Each execution creates an OpenTelemetry span and a JSONL record with `start_time` / `end_time`
(ISO-8601 UTC plus unix nanos), `TraceId` and `SpanId`, `gen_ai.usage.{input,thinking,output,
total}_tokens`, `shelf_benchmark.cost_per_shelf_image_usd`, `shelf_benchmark.cost_per_product_usd`,
`shelf_benchmark.latency_ms` and `shelf_benchmark.product_count`.

Accuracy is a separate matter: with no annotations connected, accuracy metrics are `None` with
`accuracy_status="PLACEHOLDER_AWAITING_GROUND_TRUTH"`, never `0.0`. See
[`docs/EVALUATION_PROTOCOL.md`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/docs/EVALUATION_PROTOCOL.md)
and
[`docs/GROUND_TRUTH_CONTRACT.md`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/docs/GROUND_TRUTH_CONTRACT.md).
