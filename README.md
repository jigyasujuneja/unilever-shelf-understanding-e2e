# Shelf Benchmark

Compare approaches for the parts of shelf understanding, each on its own leaderboard tab, and
rank every approach x model combination. Tabs sit in two sections, one per business question:
**Market share** (which products are on the shelf and how many: everything below) and
**Merchandising** (how they are displayed: planogram, shelf position, promo material, price tags;
no benchmarks yet). An approach picks its section with `use_case`.

| Tab | Task | Scored on | Ranked by |
|-----|------|-----------|-----------|
| Detection | Find every product box on a shelf photo | [SKU-110K](https://docs.ultralytics.com/datasets/detect/sku-110k/) (boxes only, no product labels) | F2 at IoU 0.5 |
| Classification | Say which catalog product a photo shows | [Labelled product photos](#labelled-product-photos) (25 photos, 184-product catalog given as text) | Exact product accuracy |
| Retrieval | Identify each product cut out of a multi-product photo by matching it to reference photos (image to image) | [RPC](#rpc-retail-product-checkout) checkout photos, ground-truth boxes given (200 products, 4 reference photos each) | Exact product accuracy |
| End-to-end | Find every product **and** identify it | [HoloSelecta](#holoselecta-shelves-end-to-end) shelves (115 products) and [RPC](#rpc-retail-product-checkout) checkout photos (ranked separately); a box counts only at IoU >= 0.5 **and** the right product | F2 |

RPC photos are products on a checkout counter and HoloSelecta shelves are Swiss vending machines;
no labelled HUL shelf photos exist yet (see [Limitations](#limitations)).

```bash
make test             # offline tests, no GCP needed (creates .venv on first use)
make auth             # once: gcloud Application Default Credentials
make ui               # leaderboard at http://localhost:8080
make run A=hierarchy_classify M=gemini-3.5-flash-lite ARGS="--limit 5"   # quick dev run
make cloud            # evaluate all approaches x models on Cloud Run (leaderboard numbers)
make ui-cloud         # deploy the UI as a private Cloud Run service
make ui-proxy         # open that service on http://localhost:8083
make bootstrap        # only for a new GCP project: APIs, buckets, registry, product photos, RPC
make rpc              # only to rebuild the RPC set from Hugging Face (it is already in GCS)
make                  # list every target
```

Live runs use Application Default Credentials against the project in `config.yaml`. No dataset
download is needed: runs and the UI read the data from GCS unless a local copy exists (`make data`
downloads SKU-110K, ~12 GB; the product photos are in the repo). Only Cloud Run runs on the
leaderboard image set (`defaults` in `config.yaml`) are ranked; other runs show up as `dev` rows.

Diagrams: [architecture](docs/images/architecture.png), [onboarding](docs/images/onboarding.png),
[sequence](docs/images/sequence.png) (sources in [docs/diagrams](docs/diagrams), `make docs`).
Cost, Priority PayGo, vector-database and telemetry details: [docs/reference.md](docs/reference.md).

## Running on GCP (how leaderboard numbers are produced)

```bash
shelf-bench cloud-run -a shelf_detect_retrieve shelf_detect_tiered -m gemini-3.5-flash-lite
```

1. Cloud Build builds the container from this repo and pushes it to Artifact Registry.
2. The `shelf-bench` Cloud Run job gets one task per approach x tier x model (2 vCPU / 4 GiB each).
   Tasks read images from GCS (`gs://unilever-shelf-understanding-shelf-images/`), call Gemini on
   Vertex AI, and write results to `.../results/<run_id>/`.
3. Finished runs are pulled into `results/` for the UI. Run `shelf-bench pull` to fetch again.

`cloud_run.parallelism` (default: the number of models) controls how many tasks run at once.
Concurrent tasks use different models, so runs don't compete for the same Vertex AI quota.

**Cost per image** is measured, not hard-coded: tokens from every Gemini response (by traffic type
actually served), other APIs the approach bills, the real Cloud Run task duration and GCS
operations, each priced from the Cloud Billing Catalog API at run start, net of promotional
credit ([details](docs/reference.md#how-cost-per-image-is-calculated)). Any run can also use the
Priority PayGo tier with `-t priority` (1.8x per token) or the Flex PayGo tier with `-t flex`
(0.5x per token, slower; [details](docs/reference.md#priority-and-flex-paygo)).

## Datasets

### SKU-110K (Detection)

SKU-110K (Trax Retail, CVPR'19): 11,743 densely packed shelf photos, one class ("object"),
~147 boxes per image.

| Split | Images | Used for |
|-------|-------:|----------|
| `test` | 2,936 | **Leaderboard.** Every run scores the same seeded subset (default 50 images, seed 0, set in `config.yaml`). |
| `val` | 588 | Iterating on prompts and parameters without touching test. |
| `train` | 8,219 | Available for approaches that need examples or fine-tuning. |

Use `--limit 0` to evaluate the whole split.

### Labelled product photos

[data/labeled_retail_benchmarks](data/labeled_retail_benchmarks) (Classification): 25 photos,
each of one product, labelled with its product (name with size), brand and category, plus the
184-product catalog they are identified against. Taken from the Hugging Face dataset
`kierth/retail-products-philippines`: catalog-style photos from the Philippine market, not HUL
shelf crops. The fine-grained cases are sizes of the same product (Lady's Choice mayonnaise in 7
sizes, Pond's 10 g vs 100 g). Cloud Run reads the copy in GCS (`gcp.products` in `config.yaml`,
uploaded by `make bootstrap`). It has a single `test` split and every run scores all 25 photos.
The file's `is_unilever` flag is not used: it is unreliable (it marks cigarettes and adhesive tape
as Unilever).

### RPC (Retail Product Checkout)

[RPC](https://rpc-dataset.github.io/) (Megvii, 2019; Hugging Face
`benjamintli/retail-product-checkout`, CC BY-NC-SA 2.0) (Retrieval and End-to-end): 200 grocery
products in 17 categories (drinks, snacks, instant noodles, tissues...). It has what the product
photos lack: **photos with many products, each box labelled with its product**, plus separate
photos of every product on its own.

`make rpc` (once, ~20 min; `make bootstrap` uploads it) builds a small benchmark from it in
`data/rpc` and `gcp.rpc` in GCS:

| Part | Content |
|------|---------|
| Reference gallery | 4 photos per product (800), from different turntable angles, cropped to the product. This is the "catalog" approaches search |
| `test` | 100 checkout photos (seeded sample of RPC test), resized to 1400 px; leaderboard runs score a seeded 50 of them (`defaults` in `config.yaml`) |
| `val` | 20 checkout photos from RPC validation, for iterating |

RPC only names products by id and category (e.g. `RPC #37 (drink)`), so the Category column is
the coarse check. Photos are taken from above on a white counter.

### HoloSelecta shelves (End-to-end)

[HoloSelecta](https://github.com/tobiagru/ObjectDetectionGroceryProducts) (Selecta / ETH Zurich,
CC BY 4.0): 295 photos of vending-machine shelves, every product boxed and labelled with its name,
size and GTIN (115 products, ~33 per photo). There are no catalog photos, so references are cut
from the photos themselves. Photos are grouped into sessions (bursts of one machine, by timestamp
in the file name) and whole sessions are split, so no reference is cut from the same burst as a
scored photo (the same machine may still appear in another session, as a real store would).
`shelf-bench shelves <download dir>` builds `data/shelves` and uploads it to `gcp.shelves`:

| Part | Content |
|------|---------|
| Reference gallery | Up to 4 crops per product from the 142 unscored photos (374 crops, 106 products; 9 products only appear in scored photos, so no approach can name them) |
| `test` | 123 photos (4,074 boxes), resized to 1400 px; leaderboard runs score a seeded 50 |
| `val` | 30 photos (904 boxes), for tuning |

## Metrics

**Detection.** Predictions are matched one-to-one to ground truth at **IoU >= 0.5**. TP/FP/FN are
summed over all images (micro-averaged).

| Column | Definition |
|--------|------------|
| Accuracy | TP / (TP + FP + FN). Detection has no true negatives, so this is the share of all boxes (predicted or real) that were right. |
| Recall | TP / (TP + FN) |
| F2 | 5PR / (4P + R). Weights recall 2x, because a missed product costs more than a stray box. **Rank is by F2.** |
| p95 / p99 | End-to-end latency per image (nearest-rank percentile, so with 50 images p99 is the slowest image) |
| Cost / img | Gemini (net of promotional credit) + Cloud Run compute + Cloud Storage ops + other APIs per image, in ₹ |

**Classification and retrieval.** Each product (a photo, or a ground-truth box of an RPC photo)
gets one catalog id (or none).

| Column | Definition |
|--------|------------|
| Product | Share of products where the exact catalog product (including size) is right. **Rank is by this.** |
| Brand / Category | Share of products where the predicted product has the right brand / category (coarser checks) |
| Precision (run page) | Right / answered. Answering "not in catalog" counts as a miss, not as a wrong product |

**End-to-end.** Boxes are matched to ground truth at IoU >= 0.5 as for detection, but a match is a
TP only if it also names that box's product; a matched box with the wrong product is both an FP
and an FN. Found = share of real products boxed at all (whatever the name), so Found vs Recall
separates detection misses from identification misses. **Rank is by F2.**

All tabs show p95, p99 and Cost/img. If an image errors (for example, after 6 retries), it scores
as "found nothing", so errors lower the score instead of being hidden.

## Running approaches

```bash
shelf-bench list                                             # approaches + models
shelf-bench run -a single_pass_dedup -m gemini-3.5-flash-lite # one combination
shelf-bench run -a single_pass_dedup tiled_dedup \
                -m gemini-3.5-flash-lite gemini-3.8-flash     # every combination (4 runs)
shelf-bench run -a shelf_detect_retrieve -m gemini-3.5-flash-lite --split val --limit 10   # quick iteration
shelf-bench leaderboard                                      # print in the terminal
```

Each run writes `results/<run_id>/summary.json` (the leaderboard row) and `images.jsonl`
(per-image boxes, metrics, and step trace). These are small, so commit them to share the leaderboard.

### Built-in approaches

| Name | Tab | Models | Architecture |
|------|-----|--------|--------------|
| `single_pass_dedup` | Detection | any `gemini-*` | One Gemini call on the full (downscaled) image returns every box with its class; drop `not_a_product`, container boxes, NMS at IoU 0.58 and depth ghosts (smaller boxes behind a front box). No extra call |
| `tiled_dedup` | Detection | any `gemini-*` | Two Gemini calls, one per half of the photo (top/bottom, each at full detail). A product cut by the seam is re-joined when both halves touch the seam and overlap by >= 80% in x, then `single_pass_dedup`'s filters |
| `hierarchy_classify` | Classification | any `gemini-*` | Call 1: Gemini picks the brand from the catalog's brand list and reads the pack size. The catalog is filtered to that brand and size; if one product is left that's the answer, otherwise call 2 picks from the short list |
| `embedding_text_match` | Classification | `multimodalembedding@001` or `gemini-embedding-2-preview` | Embeds the photo and returns the catalog product whose **text** embedding is closest (cosine). No Gemini call |
| `embedding_retrieval` | Retrieval | `multimodalembedding@001` or `gemini-embedding-2-preview` | Embeds each crop and returns the product whose closest **reference photo** is most similar (cosine). No Gemini call |
| `tiered_hybrid` | Retrieval | any `gemini-*` (+ `multimodalembedding@001`) | Accepts the embedding answer when it is clearly ahead (cosine >= 0.70 and >= 0.045 above the runner-up, tuned on val for `multimodalembedding@001`); otherwise one Gemini call sees the crop next to one reference photo of each of the top 5 products and picks one, or none |
| `detect_retrieve` | End-to-end (RPC) | any `gemini-*` (+ `gemini-embedding-2-preview`) | One Gemini call boxes every product, then `embedding_retrieval` on each box |
| `detect_tiered` | End-to-end (RPC) | any `gemini-*` (+ `multimodalembedding@001`) | One Gemini call boxes every product, then `tiered_hybrid` on each box |
| `shelf_detect_retrieve`, `shelf_detect_tiered` | End-to-end (Shelves) | as above | `single_pass_dedup` boxes, then the same identification as the two pipelines above, on HoloSelecta shelves (`tiered_hybrid`'s thresholds are the ones tuned on RPC) |

Models in brackets are fixed: they are called on every run whatever `-m` says (`also_calls`).
Detection approaches also label boxes (food, beverage, ...) and drop `not_a_product` ones; the
labels are shown on the run page but not scored, since SKU-110K has a single class.

**Removed because another approach beat them on both F2/accuracy and ₹/image** (leaderboard of
2026-09-30, in git history): `single_pass` (same call as `single_pass_dedup` without the free
filters), `detect_classify` (a second contact-sheet pass cost more and scored lower),
`rail_profile_cv` (F2 0.16), `gemini_classify` (no more accurate than `hierarchy_classify`, 31-69%
more per photo), `sister_shade_rerank` (below plain `embedding_retrieval` with
`gemini-embedding-2-preview`), and the always-escalate `gemini_rerank` / `detect_rerank` /
`shelf_detect_rerank` (more ₹ than the tiered versions for no better F2 on shelves).

`single_pass_dedup`, `tiled_dedup`, `hierarchy_classify`, `tiered_hybrid` and `detect_tiered`
are the working parts of Jigyasu Juneja's earlier pipeline, rebuilt here
(each module's docstring says what it comes from and what changed). The rest of that pipeline
didn't run the models it was named after, so it wasn't kept. His row smoothing (relabel an unsure
box between two boxes of one product) was tried on the HoloSelecta val split with embedding margins
as confidence: no threshold pair improved accuracy (0.683 at best, down to 0.671), so it isn't
included.

### Adding an approach

Approaches live under `src/approaches/<use_case>/<task>/`, matching the UI sections and tabs:
`market_share/{detection,classification,retrieval,end_to_end}/` and
`merchandising/{planogram_compliance,promo_detection}/`. Any `@register` class or `compose(...)`
call in that tree is discovered automatically and shows up in `shelf-bench list`, the CLI, Cloud
Run and the leaderboard.

**1a. Create a standalone approach** (e.g. `src/approaches/market_share/detection/my_approach.py`):

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

**1b. Or compose existing steps into an `end_to_end` pipeline** (`detect -> retrieve`, `detect -> classify`, or `detect -> classify -> retrieve`):

```python
from approaches.base import compose
from approaches.market_share.classification.hierarchy_classify import HierarchyClassify
from approaches.market_share.detection.single_pass_dedup import SinglePassDedup
from approaches.market_share.retrieval.embedding_retrieval import EmbeddingRetrieval

# 2-step: detect -> retrieve, or 3-step: detect -> coarse classify (narrow) -> retrieve
ShelfDetectClassifyRetrieve = compose(
    "shelf_detect_classify_retrieve",
    SinglePassDedup,
    HierarchyClassify,
    EmbeddingRetrieval,
    dataset="shelves",
)
```

**1c. Or run detection + classification in a single model invocation** by overriding `detect_and_identify(image, ctx) -> (boxes, sku_ids)` (see [`_single_call_end_to_end_template.py`](src/approaches/market_share/end_to_end/_single_call_end_to_end_template.py)).

**2. Iterate on `val`** (fast, local, doesn't touch the leaderboard split):

```bash
shelf-bench run -a my_approach -m gemini-3.5-flash-lite --split val --limit 5
make ui    # open the run and click through the steps
```

**3. Add an offline test** in the matching module under `tests/` (`test_detection.py`,
`test_classification.py`, `test_retrieval.py`, or `test_end_to_end.py`): shared fixtures in
`tests/conftest.py` provide fake datasets and a fake model (`OracleLLM`), so `make test` needs no
GCP. See the [Onboarding cookbook](docs/reference.md#onboarding-cookbook-4-patterns-for-adding--benchmarking-an-approach)
for all 4 implementation patterns (standalone, `compose`, single-call `detect_and_identify`, and
custom non-Gemini models).

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
| `setup(self, config, ctx)` | Optional hook run once per run before any image: create clients (embeddings, databases...) from `config.yaml`, build an index. What it bills to `ctx` is the run's **setup cost**, shown on the run page and not in cost/img |
| `compose(name, detector, *identifiers, dataset=...)` | Chains any detector approach with 1+ classification/retrieval steps into a registered `end_to_end` approach. Intermediate classifiers narrow the catalog via `narrow(image, catalog, ctx) -> set[int]` before the final step's `identify(image, ctx, allowed_ids=...)` |
| `detect_and_identify(self, image, ctx)` | Override on an `end_to_end` approach when a single model call returns both `boxes` and `sku_id`s in one pass |
| `VertexEmbeddings(config, model)` | [utils/embeddings.py](src/utils/embeddings.py): `.image(img, ctx)` / `.text(s, ctx)` with `multimodalembedding@001` or `gemini-embedding-2-preview`, billed to `ctx`. For a vector database see [docs/reference.md](docs/reference.md#retrieval-with-a-vector-database-alloydb) |
| `skus = {unit: catalog query}` + `ctx.bill(unit, amount)` | For non-Gemini paid APIs: list their Billing Catalog SKUs and record usage per image, and it's priced into cost/img. The embeddings client already does this |
| `ctx.trace.step(name, detail, boxes=None, regions=None, info=None)` | Adds a step to the run page. `boxes` are drawn as detections, `regions` as outlines (tiles, crops). `info` (any JSON: candidates, scores, `removed` boxes) plus the Gemini calls (tokens, latency, cost, prompt, response) and billed units since the previous step appear behind the step's ⓘ icon. Keep it to 2-4 meaningful steps |
| `ctx.model` | The model id for this run |
| `to_pixels(raw, x0, y0, w, h)` | Converts Gemini's `[ymin, xmin, ymax, xmax]` (0-1000) boxes from a region of the image into pixels |
| `DETECT_PROMPT`, `BOX_LIST_SCHEMA` | The shared detection prompt and schema |
| `CATEGORIES`, `NOT_PRODUCT`, `label_counts` | Shared product classes, and a helper that summarises labels for a step's detail text |

Rules of the game: don't catch errors just to hide them (a failed image scores as "found
nothing"); keep per-image state local, because images run concurrently.

**Which tab, which data, which models.** These class attributes decide where an approach runs:

| Attribute | Values | Default |
|-----------|--------|---------|
| `task` | `detection` (implement `detect`); `classification` or `retrieval` (implement `identify(image, ctx, allowed_ids=None) -> catalog id or None`; for retrieval the runner calls it on each ground-truth box crop); `end_to_end` (compose via `compose(detector, *identifiers)` or override `detect_and_identify(image, ctx)`) | `detection` |
| `dataset` | `sku110k`, `products` (the labelled product photos, `test` split only), `rpc` or `shelves` | `sku110k` |
| `models` | The model ids the approach actually calls, e.g. `["multimodalembedding@001"]`. `None` = any `gemini-*` model via `ctx.ask` | `None` |
| `also_calls` | Models called on every run whatever `-m` is (e.g. a fixed embedder), shown in the architecture | `[]` |
| `use_case` | Leaderboard section: `market_share` (find, identify, count products) or `merchandising` (planogram, shelf position, promo material, price tags). Ranked separately | `market_share` |

An approach only runs with the models it calls: other approach x model pairs in a `run` /
`cloud-run` are skipped, and asking for one explicitly fails. So a run id always names the model
that produced it. A non-Gemini approach sets `models` to its own id and ignores `ctx.ask`.

## UI

`shelf-bench serve` (`make ui`) hosts two pages:

* **Leaderboard**: a Market share and a Merchandising section, each with one tab per task
  (Detection, Classification, Retrieval, End-to-end), each ranked on its own metric (see
  [Metrics](#metrics)). Merchandising is empty for now.
* **Run page** (click a row) with the approach's pipeline and each evaluated image.
  * Detection: click a step to overlay what it produced (tiles, raw boxes, merged boxes). The final
    step shows ground truth (green), correct predictions (blue), and false predictions (red).
  * Classification: each photo is framed green (right product) or red (wrong or no answer), with
    the true and predicted product side by side and ✓/✗ per field.
  * Retrieval / End-to-end: every box is drawn green (right product) or red (wrong, or no product
    there) and numbered; the table lists truth vs prediction per box with each product's reference
    photo. End-to-end also draws the ground truth dashed, so missed products stand out.
  * Links to the run's trace and logs in Cloud Trace / Cloud Logging
    ([details](docs/reference.md#telemetry-cloud-trace--cloud-logging)).
  * An ⓘ on each step opens its details: every Gemini and embedding call made in that step
    (model, latency, tokens per billing bucket, ₹, traffic tier, finish reason, retries, prompt and
    response), other billed units, and what the approach recorded (shortlists with cosines,
    boxes a filter removed, drawn dashed red). The last step shows the image's totals.
  * **Load Cloud Trace + Logging into the steps** (per image, on demand, for auditing) fetches
    the image's spans and log entries (`gemini_call`, `embedding_call`, `image_scored`, from local
    and Cloud Run runs alike) and puts each on its step: every span attribute, the full log
    jsonPayload, and links to that exact span / entry in the console. Runs from now on are joined
    exactly by span id; older runs are matched by timestamp (parallel per-crop steps such as
    "Gemini tier" only approximately, as the UI says). Needs `roles/cloudtrace.user` and
    `roles/logging.viewer` for whoever runs the UI (the Cloud Run UI's service account too).

`shelf-bench serve-cloud` (`make ui-cloud`) deploys the same UI as a **private** Cloud Run service
that pulls every run from GCS on start; it is never made public. Open it with `make ui-proxy`
(an authenticated proxy on http://localhost:8083, using `gcloud auth print-identity-token`).

## Limitations

* **No labelled HUL shelf photos.** Detection is scored on SKU-110K shelves (no identities),
  identification on RPC checkout photos (Chinese grocery products on a white counter, shot from
  above) and on 25 Philippine catalog photos. None of these are HUL India shelves: use the tabs to
  compare approaches, not to predict production accuracy.
* **Small sets.** Classification: 25 photos (one photo = 4 points). Retrieval / End-to-end: 50
  leaderboard photos with roughly a dozen products each.
* **RPC is easier than a real shelf in one way:** the reference photos come from the same studio
  as the checkout photos (different photos, same lighting and camera), and products lie apart on a
  plain counter. Shelves have tighter packing, facings at angles and price tags.
* **RPC product names are unknown** (ids and categories only), so there is no brand check and
  prompts can't use product names.
* **Classification matches the photo to catalog text**, because that catalog has no product images.
* **Setup cost is not per image.** Embedding the reference gallery / catalog texts happens once
  per run; it is shown on the run page but not in cost/img.
* **Merchandising has no benchmark yet.** Planogram compliance, shelf position, POSM / promo
  detection and price tags need labelled shelf photos (and planograms) that we don't have.

## Layout

```
config.yaml                    GCP project/bucket, Cloud Run shape, models, promo credits
Dockerfile                     container for the Cloud Run job
src/
  cli.py                       shelf-bench command
  runner.py                    run approach x model -> results/<run_id>/ (local or GCS)
  approaches/                  base.py (Approach, compose, register) + modular hierarchy (_*.py = templates):
    market_share/
      detection/                 SKU-110K shelf product detectors (single_pass_dedup, tiled_dedup)
      classification/            closed-catalog classifiers (hierarchy_classify, embedding_text_match)
      retrieval/                 reference-photo retrievers (embedding_retrieval, tiered_hybrid)
      end_to_end/                composed pipelines (detect_identify.py) + single-call / AlloyDB templates
    merchandising/
      planogram_compliance/      shelf-row layout, facings & share-of-shelf approaches (awaiting labelled data)
      promo_detection/           POSM / promotional asset & price-tag detection (awaiting labelled data)
  utils/                       plumbing you rarely need to touch:
    dataset.py                   download, upload, load SKU-110K, the labelled products and RPC (local or gs://)
    metrics.py                   IoU matching, accuracy/recall/F2, product matching, percentiles
    llm.py                       Vertex Gemini client (retries, JSON parsing, token usage)
    embeddings.py                Vertex embeddings client (multimodalembedding@001, gemini-embedding-2-preview)
    alloydb.py                   AlloyDB connection + query (for retrieval approaches)
    pricing.py                   live prices from the Cloud Billing Catalog API
    telemetry.py                 OpenTelemetry traces (Cloud Trace) + linked logs (Cloud Logging)
    cloud.py                     Cloud Build + Cloud Run job, UI service deploy, project bootstrap
    server.py, static/           leaderboard UI (stdlib, no framework)
data/labeled_retail_benchmarks/ the 25 labelled product photos + catalog (committed)
docs/                          reference.md (onboarding cookbook, cost, Priority, vector DB, telemetry) + diagrams
tests/                         offline tests (no network / GCP), split by task & subsystem:
  conftest.py                    shared fake datasets (SKU-110K, products, RPC), OracleLLM & price sheet
  test_detection.py              tests for market_share/detection/ approaches
  test_classification.py         tests for market_share/classification/ approaches
  test_retrieval.py              tests for market_share/retrieval/ approaches
  test_end_to_end.py             tests for market_share/end_to_end/, compose(), detect_and_identify() & registry
  test_datasets_and_metrics.py   tests for dataset loading/splitting, IoU/F2 metrics & JSON recovery
  test_pricing_and_billing.py    tests for Billing Catalog pricing, promo credits, embeddings & AlloyDB
  test_cloud_ui_and_telemetry.py tests for Cloud Run task sharding/compute, UI server/proxy & OpenTelemetry
results/                       one folder per run (committed)
```
