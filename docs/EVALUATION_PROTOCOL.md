# Evaluation Protocol

This is the definitive statement of what every number the benchmark reports actually means and
how it is computed. If a metric appears in `reports/benchmark_summary.json`,
`reports/row_level_report.csv`, or an OpenTelemetry span, its definition is here.

The audience is an engineer who is about to use one of these numbers to argue that approach A is
better than approach B. Read the section for the metric you are about to quote.

Source of truth: [`evaluation/metrics.py`](../src/shelf_benchmark/evaluation/metrics.py),
[`evaluation/gcp_billing.py`](../src/shelf_benchmark/evaluation/gcp_billing.py), and the
scoring policy in [`config.py`](../src/shelf_benchmark/config.py) (`EvaluationConfig`).
Every claim below is pinned by a test in
[`tests/test_golden_scoring.py`](../tests/test_golden_scoring.py).

---

## 1. The one thing to read first: `None` is not `0.0`

Ground truth annotations **do not exist yet** for this project. A benchmark run today still
produces predictions, latency, token counts, and cost — but it cannot produce accuracy.

When there is no ground truth for an image, the suite reports:

| Field | Value |
| :--- | :--- |
| `accuracy_status` | `"PLACEHOLDER_AWAITING_GROUND_TRUTH"` |
| `ground_truth_available` | `false` |
| `detection_precision` / `detection_recall` / `detection_f1` | `null` |
| `mean_iou` / `mean_iou_matched` | `null` |
| `brand_classification_accuracy` / `product_classification_accuracy` | `null` |
| `count_accuracy` | `null` |
| `true_positives` / `false_positives` / `false_negatives` / `matched_pairs` | `null` |
| `predicted_count` | the real number of facings the model emitted |
| `iou_threshold` | the configured threshold, echoed back even though nothing was scored |
| per-row `gt_status` | `"PLACEHOLDER_AWAITING_GT"` |

> [!IMPORTANT]
> `null` means "not measured". `0.0` means "measured, and the model got nothing right".
> These are not the same claim, and the suite will never emit the second when it means the first.
> An earlier build emitted `0.0` for unscored runs, which made an unannotated benchmark look
> like a failing one and triggered a round of pointless model debugging.

The same rule applies to the counts, not only to the ratios. `true_positives = 0` is a measurement:
there were annotations, there were predictions, and none of them matched. `true_positives = null`
means no comparison happened at all, either because there is no ground truth or because the
predictions carry no geometry to pair on. A dashboard that sums a column of counts must skip the
nulls rather than coerce them to zero.

Verify this yourself:

```python
from shelf_benchmark.config import EvaluationConfig
from shelf_benchmark.evaluation.metrics import evaluate_task_accuracy
from shelf_benchmark.models import RowLevelReportItem

row = RowLevelReportItem(
    run_id="demo", trace_id="0" * 32, span_id="0" * 16,
    task_type="classification", model_name="demo-model", shelf_image_uri="demo.png",
    start_time="2024-01-01T00:00:00.000000Z", end_time="2024-01-01T00:00:01.000000Z",
    image_latency_ms=1000.0, product_index=1,
    bbox_ymin=100, bbox_xmin=100, bbox_ymax=300, bbox_xmax=200,
    predicted_brand="Pond's",
)
acc = evaluate_task_accuracy("classification", [row], None, EvaluationConfig())
print(acc.accuracy_status)        # PLACEHOLDER_AWAITING_GROUND_TRUTH
print(acc.detection_f1)           # None
print(acc.true_positives)         # None
print(acc.predicted_count)        # 1
print(acc.iou_threshold)          # 0.5
```

When annotations arrive, you do **not** re-run inference. See
[`docs/GROUND_TRUTH_CONTRACT.md`](GROUND_TRUTH_CONTRACT.md) and the `score` subcommand.

---

## 2. Coordinate system

All geometry, everywhere in the suite, is:

```
bbox_2d = [ymin, xmin, ymax, xmax]
```

normalized to an integer `0..1000` scale on both axes, independent of the source image's
pixel dimensions or aspect ratio.

This is Gemini's native 2D box convention, which is why it was chosen: the VLM emits it directly,
so there is no conversion step in the hot path where a bug could hide. Ground-truth annotations
in any other convention are converted once, explicitly, at load time — see
[`GROUND_TRUTH_CONTRACT.md`](GROUND_TRUTH_CONTRACT.md).

A box is **degenerate** (and is excluded from all localization maths) when
`ymax <= ymin` or `xmax <= xmin`. `has_valid_bbox()` is the predicate.

---

## 3. IoU

Intersection over Union between two boxes `A` and `B`:

```
inter_h    = max(0, min(A.ymax, B.ymax) - max(A.ymin, B.ymin))
inter_w    = max(0, min(A.xmax, B.xmax) - max(A.xmin, B.xmin))
inter_area = inter_h * inter_w
union_area = area(A) + area(B) - inter_area
IoU        = inter_area / union_area        (0.0 if union_area <= 0)
```

### Worked numeric example

Ground truth box `G = [100, 100, 300, 200]`. Height `300 - 100 = 200`, width `200 - 100 = 100`,
so `area(G) = 20000`.

Prediction `P = [100, 150, 300, 250]` — the same size, slid 50 units to the right.

```
inter_h    = min(300, 300) - max(100, 100) = 300 - 100 = 200
inter_w    = min(200, 250) - max(100, 150) = 200 - 150 =  50
inter_area = 200 * 50                                  = 10000
area(P)    = 200 * 100                                 = 20000
union_area = 20000 + 20000 - 10000                     = 30000
IoU        = 10000 / 30000                             = 0.3333
```

A box that is half a product-width off has an IoU of one third. That is a **miss** at the default
threshold of 0.50 and a **hit** at 0.25 — which is exactly why the threshold is now configurable
and reported, rather than baked into a metric name.

Four reference values against the same `G`:

| Prediction | IoU | True positive at 0.50? |
| :--- | ---: | :--- |
| `[100, 100, 300, 200]` (exact) | `1.0000` | yes |
| `[120, 100, 320, 200]` (20 units down) | `0.8182` | yes |
| `[100, 150, 300, 250]` (50 units right) | `0.3333` | no |
| `[600, 600, 800, 700]` (elsewhere) | `0.0000` | no |

```python
from shelf_benchmark.evaluation.metrics import compute_iou

gt = [100, 100, 300, 200]
for pred in ([100, 100, 300, 200], [120, 100, 320, 200],
             [100, 150, 300, 250], [600, 600, 800, 700]):
    print(pred, round(compute_iou(pred, gt), 4))
```

---

## 4. Pairing: global best-IoU-first greedy, geometry only

Before any metric can be computed, each prediction must be associated with at most one ground
truth item. `pair_predictions_with_gt()` does this.

### Algorithm

1. Build the full candidate list: every `(prediction, ground_truth_item)` combination where the
   prediction has a valid box.
2. Score each candidate. Under the default `pairing_strategy="iou_greedy"` the score **is** the
   IoU. Nothing else contributes.
3. Drop candidates below `iou_threshold` (because `require_iou_for_pairing` defaults to `true`)
   and candidates scoring zero.
4. Sort the entire candidate list by score, descending, with deterministic tie-breaking on
   `(row_index, gt_index)`.
5. Walk the sorted list. Accept a pair if neither its prediction nor its ground-truth item has
   already been taken. Mark both as taken.
6. Any prediction not paired at the end is an unmatched false positive; any ground-truth item not
   paired is a false negative.

### Why global sort, not left-to-right

A left-to-right greedy pass gives different answers depending on the order the model happened to
emit its boxes. If prediction #1 overlaps ground-truth item B at IoU 0.55 and prediction #2
overlaps B at IoU 0.95, left-to-right assigns B to #1 and leaves #2 unmatched. Sorting globally
assigns B to #2, which is the correct answer and does not depend on emission order. This is the
standard convention in detection evaluation and it makes the score **order independent**: shuffle
the prediction list and you get the same numbers.

### Why geometry only

`pairing_strategy` has a second option, `iou_plus_brand`, which adds `+0.25` to the score when the
predicted brand matches. It is **not** the default, and should not be used for reported numbers.

Letting brand agreement influence which box pairs with which couples localization and
classification. A model that puts boxes in roughly the wrong places but guesses popular brands
correctly will have its boxes snapped onto the ground-truth items with matching brands, inflating
detection *and* classification simultaneously. With geometry-only pairing, the two capabilities
move independently, which is the whole point of running a benchmark.

The golden test `test_localization_and_naming_are_scored_independently` pins this: perfect boxes
with every brand replaced by a wrong string yields detection precision/recall of `1.0` and brand
accuracy of `0.0`.

### Why unmatched predictions are false positives

They are not quietly handed the nearest spare ground-truth item. If they were, a model could
improve its recall for free by emitting extra boxes, since every extra box would eventually find
something to match. Counting them as false positives is what makes precision meaningful.

### The classification-only fallback

Some approaches return labels with no usable geometry at all (every box degenerate). Pairing then
falls back to positional order (`position_on_shelf`, then emission index) so that brand and
product accuracy remain measurable, and:

* every reported `iou_with_gt` is `0.0`, and
* `detection_precision`, `detection_recall`, `detection_f1`, and `mean_iou` are all `null`.

The absence of localization evidence is made visible rather than implied.

---

## 5. Detection metrics

Definitions, with `TP`, `FP`, `FN` as computed above:

| Term | Definition |
| :--- | :--- |
| True positive (`true_positives`) | A prediction paired with a ground-truth item at `IoU >= iou_threshold`. |
| False positive (`false_positives`) | `predicted_count - true_positives`. Every prediction that is not a true positive, whether it matched a lower-IoU pair or matched nothing at all. |
| False negative (`false_negatives`) | `max(0, ground_truth_count - true_positives)`. Every annotated facing that no prediction hit. |
| `matched_pairs` | Pairs formed by the pairing step. With the default `require_iou_for_pairing=true` this equals `true_positives`. It diverges only if you disable that switch. |

All four are `null`, not `0`, when no comparison took place: no ground truth, or predictions with
no geometry to pair on. `0` is a measured outcome; `null` is the absence of a measurement.

| Metric | Formula |
| :--- | :--- |
| `detection_precision` | `TP / predicted_count` |
| `detection_recall` | `TP / ground_truth_count`, or `0.0` if the annotation declares zero facings |
| `detection_f1` | `2 * P * R / (P + R)`, or `0.0` when `P + R == 0` |
| `mean_iou` | Mean IoU over **every** prediction, including unmatched ones contributing `0.0` |
| `mean_iou_matched` | Mean IoU over **true positives only** |

All five, and the four counts above, are gated on geometry: if not one prediction carries a valid
bounding box, they are `null`. A run that emitted labels but no boxes has not been localized
badly, it has not been localized at all, and reporting `precision = 0.0` for it would be a claim
the data does not support. `mean_iou_matched` is additionally `null` when there are no matches to
average.

`ground_truth_count` is `ImageGroundTruth.total_main_shelf_facings` when the annotation file
declares it, otherwise the number of ground-truth items that survived back-row filtering.

> [!NOTE]
> These three fields used to be called `detection_precision_iou50`, `detection_recall_iou50`, and
> `detection_f1_iou50`. The names were wrong: the scorer thresholded at **0.25** while the names
> said 50. Any historical report or dashboard using the old field names is reporting a lenient
> number under a strict label. The threshold is now a separate, explicit field
> (`iou_threshold`) and the metric names carry no number.

### `mean_iou` vs `mean_iou_matched`

They answer different questions and are both worth reading:

* `mean_iou_matched` — "when the model finds a product, how tightly does it box it?" It is a
  quality measure over successes and ignores misses entirely.
* `mean_iou` — "across everything the model emitted, how much of it landed on a product?" Spurious
  boxes drag it down.

A model with `mean_iou_matched = 0.95` and `mean_iou = 0.40` is precise when it is right and emits
a lot of junk. Quoting only the first would be misleading.

### Worked example

Ground truth: three facings.

| Item | Brand | Box |
| ---: | :--- | :--- |
| 1 | Pond's | `[100, 100, 300, 200]` |
| 2 | Himalaya | `[100, 220, 300, 320]` |
| 3 | Lakme | `[400, 100, 600, 200]` |

Predictions: four facings.

| # | Brand | Box | IoU with best GT |
| ---: | :--- | :--- | ---: |
| 1 | Ponds | `[100, 100, 300, 200]` | `1.0000` (item 1) |
| 2 | Himalaya | `[120, 220, 320, 320]` | `0.8182` (item 2) |
| 3 | Lakme | `[400, 150, 600, 250]` | `0.3333` (item 3) |
| 4 | Dove | `[700, 700, 900, 800]` | `0.0000` |

Scored at the default `iou_threshold = 0.50`:

| Metric | Value | Why |
| :--- | ---: | :--- |
| `matched_pairs` | `2` | Predictions 3 and 4 never clear 0.50, so no pair forms |
| `true_positives` | `2` | Predictions 1 and 2 |
| `false_positives` | `2` | `4 - 2`; predictions 3 and 4 |
| `false_negatives` | `1` | Ground-truth item 3 was never hit |
| `detection_precision` | `0.5` | `2 / 4` |
| `detection_recall` | `0.6667` | `2 / 3` |
| `detection_f1` | `0.5714` | `2 * 0.5 * 0.6667 / 1.1667` |
| `mean_iou` | `0.4546` | `(1.0 + 0.8182 + 0.0 + 0.0) / 4` |
| `mean_iou_matched` | `0.9091` | `(1.0 + 0.8182) / 2` |
| `count_accuracy` | `0.6667` | `1 - abs(4 - 3) / 3` |
| `brand_classification_accuracy` | `0.5` | 2 of 4 predictions are on a matched pair with the right brand |
| `brand_set_recall` | `1.0` | All three shelf brands appear somewhere in the predictions |

Per-row annotations in `row_level_report.csv`:

| `product_index` | `iou_with_gt` | `gt_status` | `is_true_positive` | `gt_item_id` |
| ---: | ---: | :--- | :--- | ---: |
| 1 | `1.0` | `MATCHED` | `true` | 1 |
| 2 | `0.8182` | `MATCHED` | `true` | 2 |
| 3 | `0.0` | `UNMATCHED_FALSE_POSITIVE` | `false` | `null` |
| 4 | `0.0` | `UNMATCHED_FALSE_POSITIVE` | `false` | `null` |

Note that prediction 3 reports `iou_with_gt = 0.0` even though its geometric IoU with item 3 is
`0.3333`. Below the threshold no pair forms, so there is nothing to report an IoU against. If you
want to see sub-threshold overlap, lower the threshold.

---

## 6. Changing the IoU threshold

```yaml
# configs/<your_config>.yaml
evaluation:
  iou_threshold: 0.75
```

The full `evaluation:` block, with every scoring knob commented inline, is in
[`configs/default_config.yaml`](../configs/default_config.yaml): `iou_threshold`,
`pairing_strategy`, `require_iou_for_pairing`, `brand_matcher`, `product_matcher` and the curated
`brand_aliases` table. Read that file rather than guessing at option names.

or per invocation:

```bash
.venv/bin/shelf-benchmark score \
  --predictions reports/predictions.json \
  --ground-truth-uri annotations_v1.json \
  --iou-threshold 0.75
```

The exact same four predictions above, scored at `iou_threshold = 0.25`:

| Metric | at 0.50 | at 0.25 |
| :--- | ---: | ---: |
| `true_positives` | `2` | `3` |
| `false_positives` | `2` | `1` |
| `false_negatives` | `1` | `0` |
| `detection_precision` | `0.5` | `0.75` |
| `detection_recall` | `0.6667` | `1.0` |
| `detection_f1` | `0.5714` | `0.8571` |
| `mean_iou` | `0.4546` | `0.5379` |
| `mean_iou_matched` | `0.9091` | `0.7172` |
| `brand_classification_accuracy` | `0.5` | `0.75` |

Nothing about the model changed. Recall went from 0.67 to 1.00 purely by moving a number in a
config file.

> [!WARNING]
> Changing `iou_threshold` invalidates comparison with every prior run scored at a different
> value. Precision, recall, F1, mean IoU, and both classification accuracies all move, because
> the threshold controls which pairs exist at all.
>
> The threshold is stamped onto `AccuracyMetrics.iou_threshold` and onto every row as
> `iou_threshold`, so you can always check. **Check it before putting two numbers in the same
> table.** Re-score the older run at the new threshold — it is free, see section 9.

Pick a threshold once, for the whole comparison, and write it down. 0.50 is the default because
it is the standard detection convention and because it is strict enough that a box which is
visibly on the wrong product cannot pass.

---

## 7. Classification metrics

All classification comparisons run over **normalized** strings. `normalize_text()` strips accents
(NFKD decomposition), lowercases, replaces every non-alphanumeric character with a space, and
collapses runs of whitespace. So `"Pond's"` and `"PONDS"` both normalize to `"ponds"`.

### `brand_classification_accuracy`

```
brand_classification_accuracy = brand_hits / predicted_count
```

The denominator is **every prediction**, not just the matched ones. A prediction that failed to
match any ground-truth item contributes `0` to the numerator and `1` to the denominator: it is a
wrong answer, not an absent one. Letting unmatched predictions vanish from the denominator would
mean a model could raise its brand accuracy by emitting confident garbage boxes.

A brand is a hit when `brands_match(predicted, ground_truth)` is true.

### `brands_match` — strict by default

| Matcher | Behaviour |
| :--- | :--- |
| `strict` (default) | Normalized equality, after folding both sides through the curated `brand_aliases` table. |
| `fuzzy` | Additionally accepts substring containment in either direction. |

Strict matching means `"Dove"` does **not** match `"Dover Soap"`. Under `fuzzy`, it does — which
is how a face-cream benchmark once credited a model for reading a floor-cleaner label.

Brand aliases are an **explicit curated table**, not inferred similarity:

```yaml
evaluation:
  brand_matcher: "strict"
  brand_aliases:
    ponds: "ponds"
    "pond s": "ponds"
    "fair and lovely": "glow and lovely"
    "fair lovely": "glow and lovely"
    "glow lovely": "glow and lovely"
    lakme: "lakme"
    hul: "hindustan unilever"
```

Keys and values are matched after normalization, so write them lowercase and unpunctuated. The
table exists for genuine renames (Fair & Lovely became Glow & Lovely) and for punctuation variants
the normalizer splits differently. It is deliberately hand-maintained: a similarity threshold
would silently decide that `"Lux"` and `"Luxe"` are the same brand on some shelves and not others.

Adding an alias changes reported accuracy. Treat the table as part of the protocol and record a
change to it the same way you would record a change to `iou_threshold`.

### `product_classification_accuracy`

```
product_classification_accuracy = product_hits / predicted_count
```

Same denominator rule as brand accuracy. The comparison concatenates the predicted
`product_name` and `variant`, normalizes, and compares against the normalized ground-truth
`product_name`.

| Matcher | Behaviour |
| :--- | :--- |
| `strict` (default) | Normalized equality, **or** the full normalized ground-truth name appearing inside the predicted string. Nothing else. |
| `token_overlap` | Fraction of significant ground-truth tokens present in the prediction must reach `product_token_overlap_threshold` (default `0.8`). Stopwords and generic tokens (`face`, `wash`, `gel`, `cream`, `pack`) are removed first. |
| `fuzzy_demo` | Legacy category-specific keyword groups. **Never use for reported numbers.** |

The containment rule in `strict` is there so that a model which returns
`"Himalaya Purifying Neem Face Wash 100g"` is not marked wrong against a ground truth of
`"Himalaya Purifying Neem Face Wash"`. It is one-directional: the prediction may be more
specific than the truth, never less.

> [!CAUTION]
> `fuzzy_demo` contains hardcoded keyword groups such as `("kiwi", "cucumber", "green", "apple",
> "fruit")`, tuned against a single face-wash shelf. Under it, a predicted "Cucumber Face Wash"
> matches a ground-truth "Kiwi Face Wash" because both are in the "green" group. It survives only
> so that old demo outputs can be reproduced. Using it on any other assortment over-reports
> product accuracy by an unknown amount.

### `sku_matching_accuracy`

```
sku_matching_accuracy = sku_hits / sku_comparable
```

`sku_comparable` counts only matched pairs where **both** the prediction and the ground truth
carry a SKU. If no pair is comparable the metric is `null`, not `0.0` — a run that never attempted
SKU matching has not failed at it.

SKU comparison is exact after stripping whitespace, hyphens and underscores, and uppercasing. It
is deliberately unforgiving: a SKU is an identifier, and a near-miss identifier is a wrong
identifier.

### `planogram_compliance_rate`

```
planogram_compliance_rate = planogram_hits / planogram_total
```

over matched rows where `planogram_compliant` is not `null`. `null` when no planogram was
connected.

---

## 8. Shelf-level metrics

### `count_accuracy`

```
count_accuracy = max(0.0, 1 - abs(predicted_count - ground_truth_count) / max(ground_truth_count, 1))
```

Purely a count comparison. It says nothing about *where* the facings are or *what* they are.

This is intentional, and it is worth reading alongside the detection metrics. A model whose boxes
are all badly placed but which finds the right number of products scores `count_accuracy = 1.0`
and `detection_recall = 0.0`. Both statements are true and both are useful: share-of-shelf by
count is a real business metric that does not require good localization, whereas planogram
compliance does.

Bounded below at `0.0`, so a model that hallucinates 30 facings on a 3-facing shelf scores `0.0`
rather than a large negative number.

### `brand_set_recall`

```
brand_set_recall = (number of expected brands matched by at least one prediction) / (number of expected brands)
```

`expected_brands` comes from the ground-truth entry. This is a **set** measure at the shelf level:
it asks "did the run notice that Lakme is on this shelf at all?", not "did it get every Lakme
facing right". A prediction anywhere on the shelf can satisfy an expected brand.

`null` when the ground truth declares no expected brands.

This is the right metric for share-of-shelf *presence* questions and the wrong one for share-of-
shelf *proportion* questions. For the latter use `brand_classification_accuracy` together with
the facing counts.

---

## 9. Run now, score later

Scoring is decoupled from inference. A run writes `predictions.json` containing raw model output
and no accuracy numbers at all; scoring turns that file plus a ground-truth source into metrics.

```bash
# Today: no annotations exist. Costs money, produces predictions.
.venv/bin/shelf-benchmark run --approaches single_pass_full_shelf

# Later: annotations land. Costs nothing, produces metrics.
.venv/bin/shelf-benchmark score \
  --predictions reports/predictions.json \
  --ground-truth-uri gs://bucket/annotations_v1.json \
  --gt-version v1

# Annotations get corrected. Still costs nothing.
.venv/bin/shelf-benchmark score \
  --predictions reports/predictions.json \
  --ground-truth-uri gs://bucket/annotations_v2.json \
  --gt-version v2
```

Token counts, latency, and cost are carried through re-scoring unchanged, because those were
measured at run time and re-scoring does not re-measure them.

`gt_version` is stamped onto every row and every span. Two result sets computed against different
annotation revisions can therefore never be silently compared. Use it.

> [!IMPORTANT]
> If none of the runs in a predictions file match a ground-truth entry, `score_predictions` raises
> `ScoringError` rather than reporting a table of zeros. A key mismatch is a configuration error,
> not a model failure, and must not be presented as one.

---

## 10. The cost model

Cost is separated into five buckets. They are not equally trustworthy, and the difference matters
more than the numbers.

| # | Bucket field | What it is | Measured or modelled? | In the total by default? |
| ---: | :--- | :--- | :--- | :--- |
| 1 | `vertex_ai_payg_tokens_usd` | On-demand token charge: input, thinking and output tokens times the rate card | **Measured** token counts (from the API's `usage_metadata`) times a rate that may be live or configured — see `billing_source` | Yes, when `traffic_type == "ON_DEMAND"` |
| 2 | `vertex_ai_provisioned_throughput_usd` | Amortized Vertex AI Provisioned Throughput (GSU) reservation cost for the request's occupancy of a slot | **Modelled** from the configured GSU rate, count, discount and slot count | Yes, when `traffic_type == "PROVISIONED_THROUGHPUT"`, and bucket 1 is then excluded so nothing is double counted |
| 3 | `vertex_ai_embeddings_and_vision_usd` | Per-facing embedding and Cloud Vision API charges the approach genuinely incurred | **Estimated** per-facing rate from `billing.embeddings_and_vision`, multiplied by the real facing count | Yes, always — this is a real API charge caused by the approach, so gating it would make crop-per-facing pipelines look free |
| 4 | `cloud_run_compute_usd` | Cloud Run vCPU-seconds, GiB-seconds and the per-request fee for the measured latency | **Modelled.** A laptop run incurs no Cloud Run charge at all | **No.** Only when `include_infrastructure_costs = true` |
| 5 | `gcs_and_observability_usd` | GCS Class A/B operations plus Cloud Logging ingestion | **Modelled** from fixed per-run op counts and an assumed log size | **No.** Only when `include_infrastructure_costs = true` |

`cost_per_shelf_image_usd` is the sum of whichever buckets are in scope.
`cost_per_product_usd` is that divided by the facing count.

### `include_infrastructure_costs` defaults to `false`

```python
# src/shelf_benchmark/config.py
class GCPBillingConfig(BaseModel):
    include_infrastructure_costs: bool = False
```

Buckets 4 and 5 are simulations of what a serving deployment *would* cost. Folding them into a
model-comparison run does two bad things: it inflates the totals with charges nobody was billed
for, and because they are driven by latency rather than by tokens, it compresses the real measured
differences between models into noise.

Turn it on when you are costing a deployment. Leave it off when you are ranking models. The flag
is echoed into every `CostMetrics` as `includes_modelled_infrastructure`, so a reader can always
tell which kind of number they are looking at.

Concretely, for one image at 1M input / 1M thinking / 1M output tokens on the
`gemini-3.8-flash` rate card (`$0.30 / $0.30 / $2.50` per 1M), 10 facings, 4.0 s latency, and
`$0.00125` of embedding calls:

| Field | `include_infrastructure_costs=false` | `include_infrastructure_costs=true` |
| :--- | ---: | ---: |
| `vertex_ai_payg_tokens_usd` | `3.1` | `3.1` |
| `vertex_ai_embeddings_and_vision_usd` | `0.00125` | `0.00125` |
| `cloud_run_compute_usd` | `0.0` | `0.0002324` |
| `gcs_and_observability_usd` | `0.0` | `0.00001231` |
| **`cost_per_shelf_image_usd`** | **`3.10125`** | **`3.10149471`** |
| `includes_modelled_infrastructure` | `false` | `true` |

### `billing_source` — how to tell a rate from an estimate

Every `CostMetrics` and every report row carries `billing_source`:

| Value | Meaning |
| :--- | :--- |
| `yaml_rate_table` | Rates came from `pricing_per_million_tokens` in the config. **This is the default.** |
| `gcp_cloud_billing_catalog_api` | Input *and* output rates were both genuinely parsed from real SKUs returned by `cloudbilling.googleapis.com`. |
| `bigquery_billing_export` | Reconciled against the actual invoice via BigQuery Billing Export. Only this is invoice-exact. |

`gcp_cloud_billing_catalog_api` is only written when the lookup actually succeeded and matched
confidently. If the API is unreachable, the credentials are missing, or no SKU description contains
every significant token of the model name, the engine falls back to the YAML table and says so.
A previous build reported `pricing_source: "gcp_billing_catalog_api"` from the *config*, which
described an intention rather than an outcome — the value was identical whether the live lookup
had worked or silently failed.

SKU matching is deliberately conservative: a SKU must mention every significant token of the model
name, and where several regional SKUs share a description the cheapest (base) rate wins. A wrong
SKU match is worse than no match, because it produces a confident number sourced from the wrong
product.

### Per-facing embedding rate defaults

| Key | Rate per facing (USD) |
| :--- | ---: |
| approach `two_stage_physical_crop_per_facing` | `0.000125` |
| approach `class_agnostic_visual_embedding` | `0.000125` |
| approach `demo_prototype_visual_embedding` | `0.000125` |
| task `detection` | `0.000020` |
| task `classification` | `0.000025` |
| task `matching` | `0.000050` |
| task `fine_tuning` | `0.000020` |
| fallback | `0.000025` |

Approach-specific overrides win over task rates. These were previously bare literals inside
`tasks/base.py`, which meant part of the cost ranking between approaches was hand-assigned in code
where nobody would look for it. They now live in
`billing.embeddings_and_vision.per_approach_per_facing_usd` and are overridable from YAML.

---

## 11. Regenerating the bundled fixture image

`shelf_benchmark.testing` ships a synthetic shelf image at
`src/shelf_benchmark/_fixtures/shelf_sample_01.png` so that offline runs exercise the real
image-loading, cropping and montage code paths instead of stubbing them out.
`fixture_image_path()` points at it, and `OFFLINE_IMAGE_URI` is its path as a string.

It is a 600x400 RGB PNG containing two shelf planks and three rectangles positioned to match
`sample_ground_truth()` exactly. `fixture_image_path()` raises a `FileNotFoundError` pointing at
this section if the file is missing.

To regenerate it byte-for-byte:

```python
from pathlib import Path

from PIL import Image, ImageDraw

from shelf_benchmark.testing import sample_ground_truth

W, H = 600, 400
BACKGROUND = (238, 236, 230)
SHELF_PLANK = (150, 120, 90)
OUTLINE = (60, 60, 60)
FILLS = [(215, 120, 160), (110, 170, 120), (225, 170, 80)]

img = Image.new("RGB", (W, H), BACKGROUND)
draw = ImageDraw.Draw(img)

# Shelf planks first, so a facing drawn on top occludes the plank behind it.
for plank_top_px in (200, 340):
    draw.rectangle([0, plank_top_px, W - 1, plank_top_px + 10], fill=SHELF_PLANK)

# One rectangle per fixture ground-truth item, converted from 0-1000 to pixels.
for item, fill in zip(sample_ground_truth().items, FILLS):
    ymin, xmin, ymax, xmax = item.bbox_2d
    draw.rectangle(
        [xmin * W // 1000, ymin * H // 1000, xmax * W // 1000, ymax * H // 1000],
        fill=fill,
        outline=OUTLINE,
        width=2,
    )

out = Path("src/shelf_benchmark/_fixtures/shelf_sample_01.png")
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out)
print(f"Wrote {out} ({img.size[0]}x{img.size[1]})")
```

The three boxes it draws are exactly the three in `sample_ground_truth()`:

| Item | Brand | `bbox_2d` (0-1000) | Pixel rectangle | Fill |
| ---: | :--- | :--- | :--- | :--- |
| 1 | Pond's | `[100, 100, 300, 200]` | `(60, 40)` to `(120, 120)` | pink `(215, 120, 160)` |
| 2 | Himalaya | `[100, 220, 300, 320]` | `(132, 40)` to `(192, 120)` | green `(110, 170, 120)` |
| 3 | Lakme | `[400, 100, 600, 200]` | `(60, 160)` to `(120, 240)` | amber `(225, 170, 80)` |

Because the boxes and the image agree, a prediction of `sample_ground_truth()`'s boxes against
this image scores exactly `1.0` on every metric, which is what the golden test
`test_perfect_prediction_scores_one` asserts. If you change the fixture geometry you must change
`sample_ground_truth()` in the same commit, or the golden tests stop being golden.

---

## 12. Field reference

Every field on `AccuracyMetrics`, in report order:

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `ground_truth_available` | bool | Whether any ground truth was found for this image |
| `accuracy_status` | str | `PLACEHOLDER_AWAITING_GROUND_TRUTH` or `EVALUATED_AGAINST_GT` |
| `gt_version` | str | Annotation revision the numbers were computed against |
| `iou_threshold` | float or null | Threshold actually applied |
| `pairing_strategy` | str or null | `iou_greedy` or `iou_plus_brand` |
| `brand_matcher` | str or null | `strict` or `fuzzy` |
| `product_matcher` | str or null | `strict`, `token_overlap` or `fuzzy_demo` |
| `ground_truth_count` | int or null | Annotated front facings |
| `predicted_count` | int | Facings the model emitted |
| `depth_duplicates_filtered` | int | Back-row units suppressed by depth deduplication |
| `matched_pairs` | int or null | Prediction/ground-truth pairs formed; `null` when nothing was measured |
| `true_positives` | int or null | Pairs at or above the threshold; `null` when nothing was measured |
| `false_positives` | int or null | `predicted_count - true_positives`; `null` when nothing was measured |
| `false_negatives` | int or null | `max(0, ground_truth_count - true_positives)`; `null` when nothing was measured |
| `count_accuracy` | float or null | Section 8 |
| `detection_precision` | float or null | Section 5 |
| `detection_recall` | float or null | Section 5 |
| `detection_f1` | float or null | Section 5 |
| `mean_iou` | float or null | Over all predictions |
| `mean_iou_matched` | float or null | Over true positives only |
| `brand_classification_accuracy` | float or null | Section 7 |
| `brand_set_recall` | float or null | Section 8 |
| `product_classification_accuracy` | float or null | Section 7 |
| `sku_matching_accuracy` | float or null | Section 7 |
| `planogram_compliance_rate` | float or null | Section 7 |

Row-level ground-truth fields on `RowLevelReportItem`:

| Field | Meaning |
| :--- | :--- |
| `gt_status` | `PLACEHOLDER_AWAITING_GT`, `MATCHED` or `UNMATCHED_FALSE_POSITIVE` |
| `gt_version` | Annotation revision |
| `gt_item_id`, `gt_brand`, `gt_product_name`, `gt_sku_id` | The paired ground-truth item, `null` when unmatched |
| `iou_with_gt` | IoU of the formed pair; `0.0` when no pair formed |
| `iou_threshold` | Threshold applied to this row |
| `is_true_positive`, `brand_correct`, `product_correct`, `sku_correct` | Per-row outcomes |

---

## 13. Scoring policy reference

Everything that can move a reported number, and its default:

| Setting | Default | Effect |
| :--- | :--- | :--- |
| `evaluation.iou_threshold` | `0.50` | True-positive cutoff, and (with `require_iou_for_pairing`) the pairing cutoff |
| `evaluation.pairing_strategy` | `"iou_greedy"` | `iou_plus_brand` couples localization to naming — do not use for reported numbers |
| `evaluation.require_iou_for_pairing` | `true` | Disabling it lets sub-threshold pairs form, so `matched_pairs > true_positives` and classification metrics become measurable on near-misses |
| `evaluation.brand_matcher` | `"strict"` | `fuzzy` adds substring containment |
| `evaluation.product_matcher` | `"strict"` | `token_overlap` and `fuzzy_demo` are looser |
| `evaluation.product_token_overlap_threshold` | `0.8` | Only used by `token_overlap` |
| `evaluation.brand_aliases` | curated table | Explicit renames; editing it changes reported accuracy |
| `evaluation.product_stopwords` | `["the", "and", "with", "new", "pack"]` | Only used by `token_overlap` |
| `ground_truth.exclude_back_row_items` | `true` | Drops depth-stacked units so the denominator is front facings only |
| `ground_truth.gt_version` | `"unversioned"` | Stamped on every row and span |
| `billing.include_infrastructure_costs` | `false` | Adds modelled Cloud Run and GCS/Logging to the totals |

All of these are echoed into the results, so a number in a report can always be traced back to
the policy that produced it. Nothing is hardcoded in the scorer.
