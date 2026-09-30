# Shelf Benchmark: reference

Details behind the [README](../README.md): onboarding cookbook for new approaches, how cost is
computed, the Priority and Flex tiers, how to back retrieval with a vector database, and what telemetry
records.

## Onboarding cookbook: 4 patterns for adding & benchmarking an approach

Every approach lives under `src/approaches/<use_case>/<task>/` and is auto-discovered by
[`approaches/base.py`](../src/approaches/base.py). Pick the pattern that matches your model:

| Pattern | Where to put it | What to implement | Copy from |
|---------|-----------------|-------------------|-----------|
| **1. Standalone detector / classifier / retriever** | `market_share/{detection,classification,retrieval}/` | `@register` subclass with `detect(image, ctx)` or `identify(image, ctx, allowed_ids=None)` | [`single_pass_dedup.py`](../src/approaches/market_share/detection/single_pass_dedup.py), [`hierarchy_classify.py`](../src/approaches/market_share/classification/hierarchy_classify.py), [`embedding_retrieval.py`](../src/approaches/market_share/retrieval/embedding_retrieval.py) |
| **2. Multi-step composition** (`detect -> retrieve`, `detect -> classify`, or `detect -> classify -> retrieve`) | `market_share/end_to_end/` | `compose("my_e2e", DetectorCls, *IdentifierClasses, dataset="shelves")` — intermediate classifiers filter the catalog via `narrow(image, catalog, ctx) -> set[int]` | [`detect_identify.py`](../src/approaches/market_share/end_to_end/detect_identify.py) |
| **3. Single-invocation detect + classify** (e.g. fine-tuned Gemini or Agent Platform endpoint returning boxes + `sku_id`s in 1 call) | `market_share/end_to_end/` | `@register` subclass with `task = "end_to_end"` and `detect_and_identify(image, ctx) -> (boxes, sku_ids)` | [`_single_call_end_to_end_template.py`](../src/approaches/market_share/end_to_end/_single_call_end_to_end_template.py) |
| **4. Custom / non-Gemini model** (e.g. YOLO, DiffusionGemma, Vertex endpoint) | Any task folder | Set `models = ["my-model-id"]` (and `also_calls = [...]` if hybrid), initialize your client/weights in `setup(config, ctx)`, and optionally bill non-Gemini SKUs with `ctx.bill(unit, amount)` | [`embedding_retrieval.py`](../src/approaches/market_share/retrieval/embedding_retrieval.py) |

**4-step verification & benchmark checklist:**
1. **Check registration:** `.venv/bin/shelf-bench list` (confirms use case, task, dataset, and accepted models).
2. **Smoke test on `val`:** `make run A=my_approach M=gemini-3.5-flash-lite ARGS="--split val --limit 5"` and inspect in `make ui`.
3. **Add an offline unit test:** Add a test in the matching file under [`tests/`](../tests/) (`test_detection.py`, `test_classification.py`, `test_retrieval.py`, or `test_end_to_end.py`) and run `make lint && make test`.
4. **Run official Cloud Run benchmark:** `make cloud A=my_approach M=gemini-3.5-flash-lite` (builds the container, runs on Cloud Run, prices every SKU from the Cloud Billing Catalog API, pulls `results/<run_id>/`, and ranks it on the leaderboard).

## How cost per image is calculated

Every number comes from GCP itself; no prices are hard-coded ([pricing.py](../src/utils/pricing.py)).

| Part | Quantity (measured) | Price (source) |
|------|---------------------|----------------|
| Gemini | Every response's `usage_metadata`: tokens split by **traffic type Vertex actually served** (standard / priority / flex), modality (text / image), cached vs uncached, and output + thinking | The matching SKU in the **Cloud Billing Catalog API**, fetched at the start of each run (e.g. `Gemini 3.8 Flash Global Image Input Priority - Predictions`) |
| Promotional credit | Gemini list cost | `promotions` in `config.yaml` (Gemini 3.x Flash: 50% credit back until 2026-12-31). Promotions are paid back as billing credits, so they are not in the catalog |
| Other APIs (embeddings) | What the approach bills with `ctx.bill(unit, amount)` | The Billing Catalog SKUs the approach lists in `skus` |
| Cloud Run | Task start → completion from the **Cloud Run Admin API**, rounded up to 100 ms, × vCPU and GiB | `Jobs CPU in <region>` / `Jobs Memory in <region>` catalog SKUs |
| Cloud Storage | 1 read per image + the annotations file, 2 result writes | `Regional Standard Class A/B Operations` SKUs (paid rate) |
| ₹ | — | The USD→INR rate GCP billing uses (catalog `currencyConversionRate`) |

A Cloud Run task can't know its own billed duration, so it stores a provisional compute figure;
`shelf-bench pull` (run automatically after `cloud-run`) replaces it with the real task
duration and writes the final summary back to GCS. Each run's summary stores the full price
sheet (SKU ids, prices, fetch time), so any cost can be audited.

An approach that bills a unit it didn't list in `skus` fails loudly rather than under-reporting.
What `setup()` bills (e.g. embedding the reference gallery once) is the run's setup cost, shown on
the run page and not in cost/img.

Not included: Cloud Build (one image build per `cloud-run` call, not per image), monthly free
tiers, taxes, negotiated discounts, and instance-hour services such as AlloyDB (a fixed monthly
cost, not per image). The exact invoiced amount is only in the Cloud Billing BigQuery export.
Local runs (`shelf-bench run`) price Gemini and other APIs only and are for development.

## Priority and Flex PayGo

Every run can call Gemini on the Standard, the [Priority PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo)
or the [Flex PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/flex-paygo) tier.
Priority costs 1.8x per token and aims for steadier latency; Flex costs 0.5x per token in exchange
for longer, less predictable latency and more throttling, which suits batch shelf processing
(nobody waits on a market-share report per photo). No setup is needed beyond the request headers,
which `llm.py` sends for `-t priority` / `-t flex`. Both work on the global endpoint only.

```bash
shelf-bench cloud-run -a shelf_detect_retrieve shelf_detect_tiered -m gemini-3.5-flash-lite \
                      -t standard flex              # 4 runs; flex runs end in -flex
```

Vertex can downgrade a priority or flex request to standard when capacity is short. Each call's
`trafficType` is recorded, and downgraded calls are billed at standard rates. The run summary's
`priority_served` / `flex_served` field is the share of calls actually served at that tier.

## Retrieval with a vector database (AlloyDB)

[embedding_retrieval.py](../src/approaches/market_share/retrieval/embedding_retrieval.py) keeps RPC's 800 reference
vectors in memory (built in `setup`; a Python loop is enough at that size).
[_detect_retrieve_template.py](../src/approaches/market_share/end_to_end/_detect_retrieve_template.py) is the same pipeline
with the vectors in AlloyDB, which a real catalog of thousands of products needs (the leading `_`
keeps it off the CLI until a database exists):

```python
class DetectRetrieveAlloyDB(EmbeddingRetrieval):
    task, also_calls = "end_to_end", [MODEL]
    skus = embeddings.SKUS                        # extra Billing Catalog SKUs priced at run start

    def setup(self, config, ctx):                 # once per run, before any image
        self.emb = embeddings.VertexEmbeddings(config, model=MODEL)
        self.db = AlloyDB(**config["alloydb"])

    def detect(self, image, ctx):
        return gemini_detect(image, ctx)          # boxes; the runner crops each one

    def identify(self, crop, ctx):
        v = pgvector(self.emb.image(crop, ctx))
        rows = self.db.query(VECTOR_SQL, (v, v))  # your SQL: vector, hybrid, anything
        return int(rows[0][0]) if rows else None
```

The approach owns everything that changes between experiments (embedder, where the vectors live,
the search); nothing outside `approaches/` needs to change. A hybrid (vector + full-text,
reciprocal-rank fusion) query is in the template's comments.
[utils/alloydb.py](../src/utils/alloydb.py) runs SQL through the
[AlloyDB Python Connector](https://cloud.google.com/alloydb/docs/connect-language-connectors) (IAM
auth, encrypted, no passwords); `pgvector(vec)` formats a vector parameter.

**To turn it on once an AlloyDB instance exists:**
1. Run the setup SQL in the docstring of [alloydb.py](../src/utils/alloydb.py): the `vector`
   extension, a `products (id, embedding vector(512), ...)` table, and a ScaNN index.
2. Grant the Cloud Run job's service account `roles/alloydb.client` and
   `roles/serviceusage.serviceUsageConsumer`, and create an IAM database user for it.
3. Load the RPC reference-photo embeddings made with the **same** embedder as the approach
   (512-d), with `id` = the RPC product id.
4. Fill the `alloydb:` section of `config.yaml`. For a private-IP instance also set
   `cloud_run.network` / `cloud_run.subnet` (Direct VPC egress).
5. Copy the template to `src/approaches/market_share/end_to_end/detect_retrieve_alloydb.py` (no `_`) and run it like any other approach.

## Telemetry (Cloud Trace + Cloud Logging)

Every run, local or Cloud Run, is **one OpenTelemetry trace** in Cloud Trace
([telemetry.py](../src/utils/telemetry.py)). Each span also writes one Cloud Logging entry
(log `shelf-bench`) with the same fields, linked to its span:

| Span | What it records |
|------|-----------------|
| `run <run_id>` | approach, model, tier, split / limit / seed, owner, Cloud Run execution + task; at the end F2, recall, precision, p50/p95/p99, cost/img, total tokens, traffic served |
| `image <image_id>` | gt / pred / tp / fp / fn, F2, latency, cost (net + list + other APIs), tokens, error (span status = ERROR); each `ctx.trace.step` is a span event |
| `gemini <model>` | `gen_ai.usage.input_tokens` / `output_tokens`, thinking tokens, image vs text vs cached input tokens, traffic type Vertex actually served, tier requested, attempts + one `retry` event per 429/5xx, finish reason, response id / model version, temperature / max tokens / thinking level, image size and JPEG bytes, model latency, cost. Its log entry also has the **prompt and response text** |
| `embedding <model>` | one per crop / text embedded while scoring an image: model, image or text, dimension, latency, attempts, billed units (`shelf_bench.billed.*`). Reference-photo embeddings in setup are billed but not traced |

Links are stored with the results and shown on the run page:

* `summary.json["telemetry"]`: `trace_id`, `trace_url` (Cloud Trace), `logs_url` (every
  entry of the run) and, on Cloud Run, `task_logs_url` (the task's full stdout/stderr).
* each row of `images.jsonl` has `telemetry` with that image's span (`span_id`, `trace_url`,
  `logs_url`).
* each call recorded on a step in `images.jsonl` carries its `span_id`, so the run page's
  **Load Cloud Trace + Logging into the steps** ([audit.py](../src/utils/audit.py)) puts every
  span and log entry on the exact step it belongs to (older runs: matched by timestamp).

Handy Logs Explorer queries: `logName="projects/unilever-shelf-understanding/logs/shelf-bench"
jsonPayload.event="gemini_call" jsonPayload."shelf_bench.attempts">1` (retried calls),
`jsonPayload."shelf_bench.truncated"=true` (output hit the token limit), `severity>=ERROR`.

Approaches get all of this for free through `ctx.ask` and `ctx.trace.step`. Writing needs
`roles/cloudtrace.agent` + `roles/logging.logWriter`, and viewing needs `roles/cloudtrace.user` +
`roles/logging.viewer`. Logs take up to a minute to show up. To turn it off, set
`telemetry.enabled: false` in `config.yaml` or `SHELF_BENCH_TELEMETRY=0`. Tests never export.
