# Shelf Benchmark: reference

Details behind the [README](../README.md): how cost is computed, the Priority tier, how to back
retrieval with a vector database, and what telemetry records.

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

## Priority PayGo

Every run can call Gemini on the Standard or the [Priority PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo)
tier. Priority costs 1.8x per token and aims for steadier latency. No setup is needed beyond the
request headers, which `llm.py` sends for `-t priority`. It works on the global endpoint only.

```bash
shelf-bench cloud-run -a single_pass detect_classify -m gemini-3.8-flash gemini-3.5-flash-lite \
                      -t standard priority          # 8 runs; priority runs end in -priority
```

Vertex can downgrade a priority request to standard when capacity is short. Each call's
`trafficType` is recorded, and downgraded calls are billed at standard rates. The run summary's
`priority_served` field is the share of calls actually served at priority.

## Retrieval with a vector database (AlloyDB)

[embedding_retrieval.py](../src/approaches/embedding_retrieval.py) keeps RPC's 800 reference
vectors in memory (built in `setup`; a Python loop is enough at that size).
[_detect_retrieve_template.py](../src/approaches/_detect_retrieve_template.py) is the same pipeline
with the vectors in AlloyDB, which a real catalog of thousands of products needs (the leading `_`
keeps it off the CLI until a database exists):

```python
class DetectRetrieveAlloyDB(EmbeddingRetrieval):
    task, also_calls = "end_to_end", [MODEL]
    skus = embeddings.SKUS                        # extra Billing Catalog SKUs priced at run start

    def setup(self, config, ctx):                 # once per run, before any image
        self.emb = embeddings.VertexEmbeddings(config)
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
5. Copy the template to `detect_retrieve_alloydb.py` (no `_`) and run it like any other approach.

## Telemetry (Cloud Trace + Cloud Logging)

Every run, local or Cloud Run, is **one OpenTelemetry trace** in Cloud Trace
([telemetry.py](../src/utils/telemetry.py)). Each span also writes one Cloud Logging entry
(log `shelf-bench`) with the same fields, linked to its span:

| Span | What it records |
|------|-----------------|
| `run <run_id>` | approach, model, tier, split / limit / seed, owner, Cloud Run execution + task; at the end F2, recall, precision, p50/p95/p99, cost/img, total tokens, traffic served |
| `image <image_id>` | gt / pred / tp / fp / fn, F2, latency, cost (net + list + other APIs), tokens, error (span status = ERROR); each `ctx.trace.step` is a span event |
| `gemini <model>` | `gen_ai.usage.input_tokens` / `output_tokens`, thinking tokens, image vs text vs cached input tokens, traffic type Vertex actually served, tier requested, attempts + one `retry` event per 429/5xx, finish reason, response id / model version, temperature / max tokens / thinking level, image size and JPEG bytes, model latency, cost. Its log entry also has the **prompt and response text** |

Links are stored with the results and shown on the run page:

* `summary.json["telemetry"]`: `trace_id`, `trace_url` (Cloud Trace), `logs_url` (every
  entry of the run) and, on Cloud Run, `task_logs_url` (the task's full stdout/stderr).
* each row of `images.jsonl` has `telemetry` with that image's span (`span_id`, `trace_url`,
  `logs_url`).

Handy Logs Explorer queries: `logName="projects/unilever-shelf-understanding/logs/shelf-bench"
jsonPayload.event="gemini_call" jsonPayload."shelf_bench.attempts">1` (retried calls),
`jsonPayload."shelf_bench.truncated"=true` (output hit the token limit), `severity>=ERROR`.

Approaches get all of this for free through `ctx.ask` and `ctx.trace.step`. Writing needs
`roles/cloudtrace.agent` + `roles/logging.logWriter`, and viewing needs `roles/cloudtrace.user` +
`roles/logging.viewer`. Logs take up to a minute to show up. To turn it off, set
`telemetry.enabled: false` in `config.yaml` or `SHELF_BENCH_TELEMETRY=0`. Tests never export.
