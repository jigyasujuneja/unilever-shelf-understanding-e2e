# Shelf Detection Benchmark (SKU-110K)

Compare approaches for **finding every product on a retail shelf** on the
[SKU-110K](https://docs.ultralytics.com/datasets/detect/sku-110k/) dataset, and rank every
approach x model combination on one leaderboard.

```bash
make test             # offline tests, no GCP needed (creates .venv on first use)
make auth             # once: gcloud Application Default Credentials
make ui               # leaderboard at http://localhost:8080
make run A=single_pass M=gemini-3.5-flash-lite ARGS="--split val --limit 5"   # quick dev run
make cloud            # evaluate all approaches x models on Cloud Run (leaderboard numbers)
make                  # list every target
```

No dataset download is needed: runs and the UI read SKU-110K from GCS unless a local copy exists
(`make data`). Only Cloud Run runs on the leaderboard image set (`defaults` in `config.yaml`:
the same 50 seeded `test` images) are ranked; other runs show up as `dev` rows below them.

Diagrams: [architecture](docs/images/architecture.png), [onboarding](docs/images/onboarding.png),
[sequence](docs/images/sequence.png) (sources in [docs/diagrams](docs/diagrams), `make docs`).

## Running on GCP (how leaderboard numbers are produced)

The dataset lives in GCS at `gs://unilever-shelf-understanding-shelf-images/SKU110K_fixed/`
(all 11,748 files). Benchmarks run as a **Cloud Run job**:

```bash
shelf-bench cloud-run -a single_pass detect_classify -m gemini-3.8-flash gemini-3.5-flash-lite
```

1. Cloud Build builds the container from this repo and pushes it to Artifact Registry.
2. The `shelf-bench` Cloud Run job gets one task per approach x tier x model (2 vCPU / 4 GiB each).
   Tasks read images from GCS, call Gemini on Vertex AI, and write results to
   `gs://unilever-shelf-understanding-shelf-images/results/<run_id>/`.
3. Finished runs are pulled into `results/` for the UI. Run `shelf-bench pull` to fetch again.

`cloud_run.parallelism` (default: the number of models) controls how many tasks run at once.
Tasks are ordered so that concurrent tasks use different models, so runs don't compete for the
same Vertex AI quota. Competing for quota would inflate latency and cause 429 errors.

### How cost per image is calculated

Every number comes from GCP itself; no prices are hard-coded ([pricing.py](src/utils/pricing.py)).

| Part | Quantity (measured) | Price (source) |
|------|---------------------|----------------|
| Gemini | Every response's `usage_metadata`: tokens split by **traffic type Vertex actually served** (standard / priority / flex), modality (text / image), cached vs uncached, and output + thinking | The matching SKU in the **Cloud Billing Catalog API**, fetched at the start of each run (e.g. `Gemini 3.8 Flash Global Image Input Priority - Predictions`) |
| Promotional credit | Gemini list cost | `promotions` in `config.yaml` (Gemini 3.x Flash: 50% credit back until 2026-12-31). Promotions are paid back as billing credits, so they are not in the catalog |
| Cloud Run | Task start → completion from the **Cloud Run Admin API**, rounded up to 100 ms, × vCPU and GiB | `Jobs CPU in <region>` / `Jobs Memory in <region>` catalog SKUs |
| Cloud Storage | 1 read per image + the annotations file, 2 result writes | `Regional Standard Class A/B Operations` SKUs (paid rate) |
| ₹ | — | The USD→INR rate GCP billing uses (catalog `currencyConversionRate`) |

A Cloud Run task can't know its own billed duration, so it stores a provisional compute figure;
`shelf-bench pull` (run automatically after `cloud-run`) replaces it with the real task
duration and writes the final summary back to GCS. Each run's summary stores the full price
sheet (SKU ids, prices, fetch time), so any cost can be audited.

Not included: Cloud Build (one image build per `cloud-run` call, not per image), monthly free
tiers, taxes and any negotiated discounts. The exact invoiced amount is only in the Cloud
Billing BigQuery export. Local runs (`shelf-bench run`) price Gemini only and are for development.

Only the gcloud project and Application Default Credentials are needed
(`gcloud auth application-default login`). The commands call the GCP APIs directly.

Local runs (`make run`) read images straight from GCS. For faster or offline iteration,
`make data` downloads a local copy (~12 GB), which is then used automatically.
`shelf-bench upload` copies a local dataset to GCS (resumable).

## Dataset and how we use it

SKU-110K (Trax Retail, CVPR'19): 11,743 densely packed shelf photos, one class ("object"),
~147 boxes per image.

| Split | Images | Used for |
|-------|-------:|----------|
| `test` | 2,936 | **Leaderboard.** Every run scores the same seeded subset (default 50 images, seed 0, set in `config.yaml`). |
| `val` | 588 | Iterating on prompts and parameters without touching test. |
| `train` | 8,219 | Available for approaches that need examples or fine-tuning. |

Use `--limit 0` to evaluate the whole split.

## Metrics

Predictions are matched one-to-one to ground truth at **IoU >= 0.5**. TP/FP/FN are summed over all
images (micro-averaged).

| Column | Definition |
|--------|------------|
| Accuracy | TP / (TP + FP + FN). Detection has no true negatives, so this is the share of all boxes (predicted or real) that were right. |
| Recall | TP / (TP + FN) |
| F2 | 5PR / (4P + R). Weights recall 2x, because a missed product costs more than a stray box. **Rank is by F2.** |
| p95 / p99 | End-to-end latency per image (nearest-rank percentile, so with 50 images p99 is the slowest image) |
| Cost / img | Gemini (net of promotional credit) + Cloud Run compute + Cloud Storage ops per image, in ₹. See [How cost per image is calculated](#how-cost-per-image-is-calculated) |

Precision is on each run's detail page. If an image errors (for example, after 6 retries), it scores as
"found nothing", so errors lower the score instead of being hidden.

## Running approaches

```bash
shelf-bench list                                             # approaches + models
shelf-bench run -a detect_classify -m gemini-3.8-flash        # one combination
shelf-bench run -a single_pass detect_classify \
                -m gemini-3.8-flash gemini-3.5-flash-lite     # every combination (4 runs)
shelf-bench run -a detect_classify -m gemini-3.8-flash --split val --limit 10   # quick iteration
shelf-bench leaderboard                                      # print in the terminal
```

Each run writes `results/<run_id>/summary.json` (the leaderboard row) and `images.jsonl`
(per-image boxes, metrics, and step trace). These are small, so commit them to share the leaderboard.

### Built-in approaches

| Name | Architecture |
|------|--------------|
| `single_pass` | One Gemini call on the full (downscaled) image returns every box **with its class** |
| `detect_classify` | Pass 1: one Gemini call detects every box. Pass 2: box crops are laid out on numbered contact sheets (48 per call) and Gemini labels each one |

Both label boxes as food, beverage, personal_care, home_care, other_product or not_a_product, and
drop not_a_product boxes. SKU-110K has a single class, so labels are shown in the step view but not
scored; classification only affects the score by removing false boxes.

### Priority PayGo

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

### Adding an approach

An approach is one Python file in `src/approaches/`. It gets a PIL image and returns product
boxes in original-image pixels. There's nothing to register by hand: any `@register` class in that
folder shows up in `shelf-bench list`, the CLI, Cloud Run and the leaderboard.

**1. Create `src/approaches/my_approach.py`**

```python
from approaches.base import (BOX_LIST_SCHEMA, DETECT_PROMPT, Approach, Context,
                             register, to_pixels)


@register
class MyApproach(Approach):
    name = "my_approach"                                  # CLI id (-a my_approach)
    architecture = "Gemini, my idea"                      # leaderboard text
    steps = ["Call Gemini", "Post-process"]               # pipeline shown on the run page

    def detect(self, image, ctx: Context):
        res = ctx.ask(image, DETECT_PROMPT, schema=BOX_LIST_SCHEMA)   # tokens + cost tracked
        boxes = to_pixels(res.data, 0, 0, *image.size)                 # -> [(x1, y1, x2, y2), ...]
        ctx.trace.step("Gemini", f"{len(boxes)} boxes", boxes=boxes)   # a clickable step in the UI
        return boxes
```

**2. Iterate on `val`** (fast, local, doesn't touch the leaderboard split):

```bash
shelf-bench run -a my_approach -m gemini-3.5-flash-lite --split val --limit 5
make ui    # open the run and click through the steps
```

**3. Add an offline test** in `tests/test_shelf_bench.py`: copy
`test_run_single_pass_end_to_end`, which runs an approach on a fake dataset with a fake model
(`OracleLLM`), so `make test` needs no GCP.

**4. Get leaderboard numbers on Cloud Run:**

```bash
make cloud A=my_approach M="gemini-3.8-flash gemini-3.5-flash-lite"
```

The container is rebuilt from your working tree on every `cloud-run`, so no deploy step is needed.
New pip dependencies go in `pyproject.toml`.

**What an approach can use**

| API | What it does |
|-----|--------------|
| `ctx.ask(image, prompt, schema=None, max_side=None)` | Calls the run's Gemini model on an image (a full shelf, a crop, a contact sheet...). `schema` = JSON response schema, `max_side` = downscale first. Returns `LLMResult(data, usage, seconds, text)`. Tokens, traffic type and cost are recorded automatically. Call it as often as you like, including from threads |
| `setup(self, config)` | Optional hook run once per run before any image, to create clients (embeddings, databases...) from `config.yaml` |
| `skus = {unit: catalog query}` + `ctx.bill(unit, amount)` | For non-Gemini paid APIs: list their Billing Catalog SKUs and record usage per image, and it's priced into cost/img. The helpers in `utils/` (e.g. embeddings) already do this. See [Product retrieval](#product-retrieval) |
| `ctx.trace.step(name, detail, boxes=None, regions=None)` | Adds a step to the run page. `boxes` are drawn as detections, `regions` as outlines (tiles, crops). Keep it to 2-4 meaningful steps |
| `ctx.model` | The model id for this run |
| `to_pixels(raw, x0, y0, w, h)` | Converts Gemini's `[ymin, xmin, ymax, xmax]` (0-1000) boxes from a region of the image into pixels |
| `DETECT_PROMPT`, `BOX_LIST_SCHEMA` | The shared detection prompt and schema |
| `CATEGORIES`, `NOT_PRODUCT`, `label_counts` | Shared product classes, and a helper that summarises labels for a step's detail text |

Rules of the game: return every product box (the score is F2 at IoU 0.5 against SKU-110K); don't
catch errors just to hide them (a failed image scores as "found nothing"); keep per-image state
local, because images run concurrently. Non-Gemini approaches (e.g. a YOLO model trained on the
`train` split) can ignore `ctx.ask` and return boxes directly. `-m` is still required and becomes
part of the run id.

## Product retrieval

To identify the specific product in each box, an approach embeds the crop and looks it up in a
vector database. **The approach owns everything that changes between experiments**: which
embedder, which database, and the SQL (pure vector, hybrid vector + keyword, filters...).
`src/utils/` only has thin clients, and the database does the actual searching:

* [utils/embeddings.py](src/utils/embeddings.py): `VertexEmbeddings(config).image(img, ctx)` /
  `.text(s, ctx)` call Vertex AI `multimodalembedding@001` (REST) and bill `ctx` for the call.
* [utils/alloydb.py](src/utils/alloydb.py): `AlloyDB(**config["alloydb"]).query(sql, params)` runs
  SQL through the [AlloyDB Python Connector](https://cloud.google.com/alloydb/docs/connect-language-connectors)
  (IAM auth, encrypted, no passwords). `pgvector(vec)` formats a vector parameter.

A ready-to-copy example is [_detect_retrieve_template.py](src/approaches/_detect_retrieve_template.py)
(the leading `_` keeps it off the CLI until a database exists):

```python
class DetectRetrieve(Approach):
    skus = embeddings.SKUS                        # extra Billing Catalog SKUs priced at run start

    def setup(self, config):                      # once per run, before any image
        self.embed = embeddings.VertexEmbeddings(config)
        self.db = AlloyDB(**config["alloydb"])

    def detect(self, image, ctx):
        boxes = ...                               # detect as usual
        for b in boxes:
            v = pgvector(self.embed.image(image.crop(b), ctx))
            rows = self.db.query(VECTOR_SQL, (v, v))   # your SQL: vector, hybrid, anything
```

To try a different embedder, database or search, write a new approach (or change the SQL); nothing
outside `approaches/` needs to change. A hybrid (vector + full-text, reciprocal-rank fusion) query
is in the template's comments.

**To turn it on once an AlloyDB instance exists:**
1. Run the setup SQL in the docstring of [alloydb.py](src/utils/alloydb.py): the `vector`
   extension, a `products (id, embedding vector(512), ...)` table, and a ScaNN index.
2. Grant the Cloud Run job's service account `roles/alloydb.client` and
   `roles/serviceusage.serviceUsageConsumer`, and create an IAM database user for it.
3. Load product embeddings made with the **same** embedder as the approach (512-d).
4. Fill the `alloydb:` section of `config.yaml`. For a private-IP instance also set
   `cloud_run.network` / `cloud_run.subnet` (Direct VPC egress).
5. Copy the template to `detect_retrieve.py` (no `_`) and run it like any other approach.

**Cost and latency.** Everything inside `detect()` counts toward the image's latency. Calls billed
with `ctx.bill(unit, amount)` (the embedding client does this for you) are priced from the Billing
Catalog like Gemini and shown on the leaderboard as "+ other APIs". An approach that bills a unit
it didn't list in `skus` fails loudly rather than under-reporting. AlloyDB is billed per
instance-hour, not per query, so it's a fixed monthly cost and not part of cost per image.

**Scoring.** SKU-110K has no product identities, so match accuracy isn't scored yet. That needs a
dataset labelled with catalog product ids.

## UI

`shelf-bench serve` hosts two pages:

* **Leaderboard** with Rank, Run ID, Architecture, Owner, Accuracy, Recall, F2, p95, p99, and Cost/img.
* **Run page** (click a row) with the approach's pipeline and each evaluated image. Click a step to
  overlay what it produced (tiles, raw boxes, merged boxes). The final step shows ground truth
  (green), correct predictions (blue), and false predictions (red).

## Layout

```
config.yaml                    GCP project/bucket, Cloud Run shape, models, promo credits
Dockerfile                     container for the Cloud Run job
src/
  cli.py                       shelf-bench command
  runner.py                    run approach x model -> results/<run_id>/ (local or GCS)
  approaches/                  base.py (interface) + one file per approach (_*.py = templates)
  utils/                       plumbing you rarely need to touch:
    dataset.py                   download, upload, load SKU-110K (local or gs://)
    metrics.py                   IoU matching, accuracy/recall/F2, percentiles
    llm.py                       Vertex Gemini client (retries, JSON parsing, token usage)
    embeddings.py                Vertex multimodal embeddings client (for retrieval approaches)
    alloydb.py                   AlloyDB connection + query (for retrieval approaches)
    pricing.py                   live prices from the Cloud Billing Catalog API
    cloud.py                     Cloud Build + Cloud Run job orchestration
    server.py, static/           leaderboard UI (stdlib, no framework)
docs/                          architecture, onboarding and sequence diagrams
tests/                         offline tests (fake dataset + fake model)
results/                       one folder per run (committed)
```

Live runs use Application Default Credentials (`gcloud auth application-default login`) against
the project in `config.yaml`.
