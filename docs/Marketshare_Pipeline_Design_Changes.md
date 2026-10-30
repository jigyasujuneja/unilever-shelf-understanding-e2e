# **Market Share Pipeline Design**

## **Context**

Field representatives upload store shelf photos from the Sales EDGE (Modern Trade) and Shikhar (General Trade) apps during daytime store operating hours (07:30 to 18:00 IST). As outlined in the [End-to-End Architecture Guide](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.erkk4n45aac9), the pipeline is sized for 500,000 store shelf photos per day (15.0M per month).

For every visible product, the pipeline predicts five attributes (`is_hul`, `category`, `brand`, `base_variant`, and `packaging_type`) to compute category Share of Shelf (SoS) and Path-to-Purchase (P2P) Basepack Weights (`W1..W6`).

[Existing Model Analysis](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.ipskbbbvu50h) and [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0) identified four problems with the existing nine-model Market Share chain and its training data:

- **Chained classifiers compound errors (42.32% 5-attribute SKU):** Even when given a 100% Oracle Brand, the downstream variant and pack classifiers drop 5-attribute accuracy to 52.00% (`29.60%` of crops get Brand, Category, and Variant right and then fail on the 46-folder `<Category>__<Pack>` classifier).
- **42.2% of catalog SKUs have zero training crops:** Out of 2,659 Unilever SKUs, 1,121 have no training crops and 356 have only 1 to 2 crops, causing closed-set classifiers to score `0.0%` on unseen SKUs and `27.9%` on rare SKUs.
- **Competitor brands are pooled into "Others":** All 59 classes of the existing brand model map to `is_hul = True`, leaving 874 rival brands without brand or variant tracking.
- **Overlapping aisle photos double-count facings:** Multi-photo store visits (2 to 5 overlapping shots per aisle) count the same shelf stretch multiple times when summed without spatial deduplication.

To replace that chain, we evaluated three model families on the store-disjoint splits built in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0):

- **Closed-set fine-tuned vision backbones (DINOv2/v3 and MaxViT):** Single-backbone classifiers trained on the 1,538 classes present in the training split.
- **Pure embedding retrieval (Stage 2A standalone):** Nearest-neighbor lookup using `gemini-embedding-2` and a learned 2.4 MB Residual InfoNCE[^c1] adapter against our 2,659-SKU reference gallery, making zero LLM calls.
- **Two-stage hybrid retrieval and visual tie-breaking (Stage 2A + Stage 2B):** Fast in-memory `gemini-embedding-2` matching for clear crops (68% direct match), followed by a batched `gemini-3.5-flash-lite` contact-sheet tie-breaker for the 32% lookalike sister variants and size twins.

### **Goals and Non-Goals**

- **Goals:** Lift held-out 5,000-crop 5-attribute SKU accuracy from `42.32%` to `> 82%` and P2P Basepack Weight (`W1`) from `34.82%` to `> 73%`; recognize all 2,659 Unilever SKUs and 874 competitor brands (including Day-0 launches in `< 60 s` without retraining); and deduplicate overlapping multi-photo store visits (`<= 2.25%` SoS error).
- **Non-Goals:** Changing downstream P2P Basepack Weight formulas (`W1..W6`) or Cloud SQL reporting schemas.

## **Key Findings**

All results below are reported on **held-out test splits** partitioned by store outlet and MD5 image digest in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0):

- **1. Two-stage hybrid retrieval nearly doubles 5,000-crop test accuracy (42.32% to 82.30%):** On the 5,000-crop held-out test set (892 stores), the hybrid pipeline reaches **82.30% exact 5-attribute SKU match** (`96.1%` Brand, `93.5%` Category, `91.7%` Pack, `89.3%` Variant) in `0.82 s`. On the 1,000-crop test set, single-pass zoom scores **84.0%** (`83.5%` Val) and the full hybrid production configuration reaches **95.20%**.
- **2. Closed-set fine-tuned classifiers fail on rare and unseen SKUs:** Stratifying test crops by training sample count shows that DINOv3 SFT scores `90.9%` on head SKUs (`16+` training crops) but drops to `27.9%` on rare SKUs (`1 to 2` crops) and **0.0%** on the 1,121 catalog SKUs with zero training crops. The gallery-backed hybrid scores **79.1%** on 1 to 2 crop SKUs and **94.4% (17/18)** on unseen Day-0 launches.
- **3. Catalog pack lookup plus crown zoom fixes packaging (54.86% to 91.7%–94.0%):** Replacing the standalone 46-folder `<Category>__<Pack>` classifier with a lookup in the 2,659-SKU catalog lifts the existing classifier chain from `42.32%` to `71.92%` (`+29.60 pp`). Pairing catalog lookup with top-cap crown zoom reaches **93.3% to 94.0%** pack accuracy on 1,000 test crops (`91.7%` on 5,000 crops).
- **4. An open-set competitor threshold cuts False Unilever attribution to 0.4%–1.0%:** Routing crops with intermediate similarity (`0.45` to `0.72`) to an open-set competitor prediction (`is_hul = False`) reaches **98.4% ownership accuracy** (`0.4%` False Unilever) and upgrades **66.4% of vague "Others" labels** on Kirana shelves into exact competitor SKUs.
- **5. Full-shelf calibration and session overlap deduplication hold Share-of-Shelf error to 0.93%–2.25%:** On 200 Modern Trade shelves, category Share-of-Shelf error drops to **0.93%** (`2.28%` facing error), lifting P2P Basepack Weight (`W1`) from **34.82% to 73.68%**. Across 2 to 5 overlapping aisle photos per store visit, spatial and embedding deduplication holds visit SoS error to **1.00% to 2.25%**.


*Figure 1: Four-stage Market Share pipeline from dual-buffer image preparation and 960px Distilled YOLO11n CPU ONNX detection (Stage 1) to in-memory vector retrieval (Stage 2A), visual tie-breaking (Stage 2B), and category Share-of-Shelf scoring (Stage 3).*

## **How the Pipeline Works**

### **Stage 1: Two Image Buffers, 960px Detection, and Twin Grouping**

**Data insight driving design:** In [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0), 66.1% of raw brand cutouts were under 64px, 35.5% had motion blur, and full-shelf annotations boxed only target stretches (~40% of visible packs).

Each store photo is normalized and split into product boxes on a `4 vCPU / 8 GiB` Cloud Run worker in `92 ms` (shared with [Merchandising Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.b5xo3ahiv7af)):

- **Two image buffers and cylindrical tube stretch (12 ms):** Shades-of-Gray white balance corrects warm store lighting and builds two copies in memory: a contrast-boosted copy (`L*`-channel CLAHE) so the detector finds dark shelf edges, and an unsharpened colour-true RGB copy so `gemini-embedding-2` distinguishes dark blue *Clinic Plus* sachets from black *Sunsilk* sachets (`+6.0 pp`), plus a horizontal unwarp on narrow tubes so curved edge text stays readable.
- **Distilling 960px YOLO11n on SKU-110K instead of partially boxed shelves (80 ms):** Fine-tuning YOLO on partially boxed store shelves penalizes real unboxed products as background. Instead, we distill a 960px YOLO11n CPU ONNX student from a 1280px YOLOv12s teacher pre-trained on exhaustively boxed SKU-110K (11,762 shelves, 1.73M boxes) and checkout layouts (800 images).
- **Shelf-plank filter and neighbouring twin grouping:** Fitting horizontal shelf lines across box bottoms rejects 74% of floating background noise. Along each shelf row, adjacent identical facings are grouped into a single representative crop, shrinking `35` facings to **~11 unique crops per shelf** before any API call is made.


| **Detector Configuration** | **Input Resolution** | **Pre-Training Corpus** | **Shelf mAP@50** | **Mean Crops / Shelf** | **Cloud Run 4-vCPU Latency** |
| --- | --- | --- | --- | --- | --- |
| **Existing YOLOv8n** | 640x640 | Internal partial shelf boxes | 90.2% | 46.8 (high false positives) | 380 ms |
| **YOLOv12s Teacher** | 1280x1280 | SKU-110K (11,762 images, 1.73M boxes) + checkout layouts | 99.3% | 35.0 | 1.85 s (offline teacher) |
| **Distilled YOLO11n Student** | 960x960 (ONNX FP32) | Distilled from YOLOv12s on SKU-110K + checkout layouts | 99.1% | 35.0 (~11 unique crops) | 80 ms (production worker) |

### **Stage 2A: In-Memory Embedding Lookup**

**Data insight driving design:** Because 781 raw variant folders omitted pack types and 42.2% of catalog SKUs lacked training crops ([Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0)), we linked all silos by MD5 byte-hash into a 2,659-SKU master catalog where every class has 1 to 4 clean reference packshots (`10,082` total).

The ~11 unique crops per shelf are embedded in one batched `gemini-embedding-2` call (`180 ms`) and matched against this in-RAM gallery (`< 25 MB`):

- **Multi-angle reference gallery and Residual InfoNCE[^c1] adapter:** Each of the 2,659 Unilever SKUs and 874 competitor brands is represented by up to 3 angle centroids in 768-D space, refined by a lightweight 2.4 MB Residual InfoNCE adapter and 3-zone colour reranking.
- **Hubness correction across product families:** Generic white bottles tend to appear in the top-5 list of many unrelated queries. Cross-Domain Similarity Local Scaling (CSLS) subtracts each gallery vector's average neighbourhood score while masking out same-SKU reference twins so single-pack brands are not penalized.
- **Direct match without an LLM call (68% of groups, < 1 ms):** A single matrix multiply in worker RAM scores all 11 crops against all 10,082 gallery vectors in `0.12 ms`. Confident, high-margin matches (68% of product groups) are accepted immediately with zero LLM tokens.


*Figure 2: Offline reference gallery construction (< 25 MB in-worker RAM) and online four-step vector lookup resolving 68% of product groups without an LLM call.*

### **Stage 2B: Contact Sheets for Lookalikes (32% Ties)**

**Data insight driving design:** Error trace analysis showed `gemini-embedding-2` Top-5 Recall is already `93.3%`, meaning `63.2%` of Stage 2A misses already contain the true SKU in the top-5 shortlist but need fine-print net-weight or cap-crown disambiguation.

For the remaining 32% of groups (~3.5 lookalike ties per shelf), Stage 2B sends one batched contact sheet to `gemini-3.5-flash-lite` (`88.4%` context prefix cache hit rate):

- **Four-panel zoom strip and candidate differences:** Each tied crop is paired with a zoomed bottom weight band, a saliency-diff band, and a top cap-crown strip alongside the top-5 candidate names and how they differ (`50g` vs. `100g`, `SPF 30` vs. `SPF 50`, `Tub` vs. `Jar`).
- **Pre-choice text reading and catalog pack lookup:** The model reads visible brand, variant words, and gram/ml weight inside isolated `<untrusted_pack_text>` tags before picking the winning candidate ID, which is combined with Stage 2A scores and mapped deterministically to its catalog pack shape.
- **Open-set competitor guard:** When a crop does not match a Unilever SKU with high confidence, intermediate embedding similarity routes it to a named or open-set competitor (`is_hul = False`) while low similarity abstains as background clutter.

### **Journey of a Store Image Across the 3 Core Market Share Tasks**

The table below traces how a store photo or crop moves through the four stages across the three core Market Share tasks (`MS 1: Full-Shelf SoS`, `MS 2: Single-Crop 5-Attr Match`, and `MS 3: Lookalike Sister-Variant Zoom`):

| **Market Share Task** | **Stage 1: Photometry & Optical Gate** | **Stage 2 (Spatial): Tiled / ONNX Detection** | **Stage 2A (Fast Path): Vector Gate (`68%–80%` Skipped)** | **Stage 2B (Hard Path) & Calibrated P2P Output** |
| --- | --- | --- | --- | --- |
| **MS 1 · Full-Shelf Gondola & Kirana SoS** (`mt_json_golden` `#1021-105454` & `ms_json_golden` `#1015-232902`) | Generates Buffer A (natural RGB) and Buffer B (`L*` CLAHE `clip = 2.5`) to recover dark back-of-shelf sachets without bleaching foils. | Runs `960x960` Distilled `YOLO11n` ONNX (`69.0%` F2, `0` tokens) or `2x2` Tiled Detector (`73.4%` F2) + Gaussian Soft-NMS (`σ = 0.5`). | Resolves `79.7%` of clear shelf facings directly via `gemini-embedding-2` (`cos >= 0.86`, `Δcos >= 0.06`) in `2.8 ms/crop`. | Sends only `20.3%` ambiguous crops to `3.5-Flash-Lite` contact sheet -> **82.7% MT 3-Attr**, **0.93% SoS area error**, **+66.4% `"Others"` upgraded**, **73.68% P2P Basepack W1**. |
| **MS 2 · Single-Crop 5-Attr Match & Fast-Skip** (`hul_golden` `test_5k` `#1016-092456`) | Verifies crop edge (`>= 28 px`) and Laplacian sharpness (`>= 8.5`); unreadable crops abstain (`UNVERIFIED_LOW_CONFIDENCE`). | Slices crop into Top-25% Brand Header, Middle-30% Variant Ribbon, and Bottom-30% Net-Weight Band. | Scores against `2,659`-SKU gallery (`K = 3` centroids + `512d` Residual InfoNCE Adapter, `93.3%` Top-5 Recall); skips LLM when `Δcos >= 0.06`. | Pre-Choice OCR + Bayesian PoE (`α = 0.35`) + Layer-5 deterministic pack lock -> **82.3% (5K) / 83.6% (1K) 5-Attr**, **98.6% HUL**, **0.0% `1.0`-saturation** (`[0.02, 0.985]`). |
| **MS 3 · Lookalike Sister Variants & Pack Twins** (`hul_variants` `test_5k` `#1016-092456`) | Preserves subtle pastel color shifts (`ΔE_ab < 4.0`) in Buffer A while sharpening 4pt–6pt gram/ml numerals in Buffer B. | Generates a `2x` Lanczos 4-panel inspection strip (Full Crop + Cap-Crown + Variant Ribbon + Saliency-Diff Heat Band). | Detects narrow sister margin (`Δcos < 0.04` between same-brand variants) and automatically escalates to Stage 2B with Top-5 candidate diffs. | Transcribes variant keywords and net-weight numerals before candidate selection -> **83.5% Test (85.0% Val)** (`+9.5 pp` Variant, `+11.7 pp` Pack Size), category-adaptive `τ_c` gate. |

## **How We Evaluated & Results**

Every benchmark dataset in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0) is split by store outlet and MD5 file digest so zero store shelves or duplicate crops leak between training, validation, and test sets.

On partially boxed store shelves, evaluation scores detections inside human-annotated shelf stretches while reporting valid extra detections outside those stretches separately.

### **Market Share Use-Case & Task Summary**


| **Market Share Task** | **Test Dataset & Scope** | **Existing Baseline** | **Test Result & Business Outcome** |
| --- | --- | --- | --- |
| **1. Full-Shelf Market Share (Modern Trade & Kirana)** | 200 Modern Trade & 151 General Trade full store shelves | 34.82% P2P Basepack W1; rival packs pooled into "Others" | **Modern Trade:** 0.93% SoS area error (2.28% facing error, 83.9% Brand, 4.1 s/photo). **Kirana:** 68.3% row F2, upgrades 66.4% of vague "Others" into exact SKUs, and lifts **P2P Basepack W1 from 34.82% to 73.68%**. |
| **2. Multi-Photo Store Visit Deduplication** | 100 to 150 store visits (2 to 5 overlapping aisle photos/visit) | Double-counts overlapping aisle facings across adjacent photos | **1.00% to 2.25% visit SoS error** (96.5% client-parity F2, 75.5% strict SKU F2, 0.0% False Unilever attribution, 2.9 s/visit). |
| **3. Single-Crop SKU Classification at Scale (5,000 Crops)** | 5,000 held-out test crops across 892 stores | 42.32% exact 5-attribute SKU (87.1% Brand, 54.9% Pack) | **82.30% exact 5-attribute SKU** (96.1% Brand, 93.5% Category, 91.7% Pack, 89.3% Variant) in 0.82 s (+40.0 pp, p < 10^-6). |
| **4. High-Precision Cap & Fine-Print Zoom** | 1,000 held-out test crops (4-panel zoom + crown strip) | 44.50% (52.00% with 100% Oracle Brand) | **84.0% Test** single-pass zoom (83.5% Val; 94.0% Pack, 96.5% Brand, 0.6% False Unilever) and **95.20%** on the locked hybrid production configuration. |
| **5. Zero-LLM & Fast-Skip Embedding Retrieval** | 5,000 held-out crops & 18 unseen Day-0 product launches | 0.0% on 1,121 unseen catalog SKUs; closed-set SFT fails on new launches | **81.2% with 50% LLM skip** (0.47 s) and **78.9% to 80.9% with zero LLM calls**; **94.4% (17/18) on unseen Day-0 launches in < 60 s**. |
| **6. Lookalike Sister Variant Disambiguation** | 1,000 lookalike test crops across 1,922 sister variants | 44.50% on 9-model classifier chain | **83.5% Test** (85.0% Val) across lookalike sister variants and gram/ml size twins in 0.56 s. |
| **7. Brand & Competitor Attribution** | 1,000 test crops spanning Unilever & 874 rival brands | 87.1% Brand; rivals grouped into "Others" | **95.9% exact Brand, 98.4% Unilever/rival ownership, 0.4% False Unilever** (94.4% Brand and 97.8% ownership in zero-LLM mode). |
| **8. 3D Packaging Shape Classification** | 1,000 & 5,000 test crops across 18 canonical container shapes | 54.86% on standalone pack-type classifier | **88.9% to 89.5%** standalone shape recognition; **93.3% to 94.0% (1K) and 91.7% (5K)** when combined with catalog lookup and top-cap crown zoom. |

## **Matched Silo Audit**

Comparing the existing nine-model classifier chain ([Existing Model Analysis](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.ipskbbbvu50h)) against the four-stage funnel on the same 5,000 held-out crops (892 stores) shows where the **+40.0 pp** accuracy gain comes from:


| **Cohort / Container Slice** | **Sample Size (N)** | **Existing v0.1 Accuracy** | **Target v0.2 Accuracy** | **Absolute Lift (Δ pp)** | **Primary Technical Driver** |
| --- | --- | --- | --- | --- | --- |
| **Skin & Hair Care Silo** | 1,140 | 31.8% | 91.6% | +59.8 pp | Unified 2,659-SKU gallery + top-cap crown twin-pack tie-breaking |
| **Oral Care Silo** | 680 | 32.9% | 83.2% | +50.3 pp | Eliminates cross-silo misrouting on multi-brand family packs |
| **Rival P6 Silo** | 1,420 | 29.1% | 77.0% | +47.9 pp | 874 named competitor brands + 3-tier open-set competitor guard |
| **P5 Foods & HFD Silo** | 890 | 50.9% | 89.1% | +38.2 pp | Deterministic catalog pack-shape lookup + bottom weight-band OCR |
| **Sachet & Wrapper Packs** | 910 | 18.4% | 91.8% | +73.4 pp | Stage 1 Dual-Buffer LAB CLAHE glare cleanup + 960px Distilled YOLO11n |
| **Pump & Tube Containers** | 640 | 28.6% | 90.4% | +61.8 pp | Stage 1 cylindrical unwarp + K=3 multi-angle centroids |
| **Total Matched 5K Benchmark** | **5,000** | **42.32%** | **82.30% to 84.00%** | **+40.0 pp** | **Paired McNemar p < 10^-6 (95.20% on 1,000-crop benchmark)** |

## **Accuracy by Category, Shape & Ownership**

### **1. Store Categories**

In Unilever's P2P scoring engine, each category blends six empirical reliability weights (`W1 = Basepack [W4 * W5 / 100]`, `W2 = SubCategory`, `W3 = Brand`, `W4 = Variant`, `W5 = Size`, `W6 = PackType`):


| **Store Category** | **Sample Size (N)** | **Live P2P W1 (W6 Pack)** | **Existing 9-Model SKU (Strict / Canon)** | **New v0.2 Hybrid SKU (5K)** | **New v0.2 Brand (W3)** | **New v0.2 Category** | **Net SKU Lift (Δ pp)** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Foods** | **630** | **70.55% (78.00%)** | **48.3% (52.22%)** | **90.0%** | **97.1%** | **99.7%** | **+41.7 pp** |
| **Skin** | **772** | **54.58% (65.00%)** | **39.9% (38.60%)** | **85.9%** | **96.9%** | **93.8%** | **+46.0 pp** |
| **Hair_Care** | **385** | **56.51% (84.00%)** | **42.3% (45.97%)** | **85.7% (89.9% 1K)** | **99.0%** | **99.2%** | **+43.4 pp** |
| **Beverages** | **569** | **54.00% (75.00%)** | **38.7% (41.48%)** | **80.5%** | **95.4%** | **98.6%** | **+41.8 pp** |
| **HHC (Household Care)** | **513** | **62.90% (84.00%)** | **33.7% (30.21%)** | **80.5%** | **96.9%** | **93.0%** | **+46.8 pp** |
| **PW (Personal Wash)** | **441** | **38.03% (81.00%)** | **56.7% (50.57%)** | **80.3% (87.5% 1K)** | **95.2%** | **93.7%** | **+23.6 pp** |
| **HFD (Health Food Drinks)** | **651** | **43.32% (91.00%)** | **36.3% (37.17%)** | **80.0%** | **98.0%** | **98.5%** | **+43.7 pp** |
| **Oral_Care** | **421** | **31.85% (80.00%)** | **37.1% (32.78%)** | **80.0%** | **94.3%** | **97.4%** | **+42.9 pp** |
| **Laundry** | **527** | **35.59% (84.00%)** | **52.0% (54.65%)** | **78.4%** | **93.7%** | **97.0%** | **+26.4 pp** |
| **DMT (Deodorants)** | **91** | **58.13% (84.00%)** | **35.2% (29.67%)** | **63.7%** | **83.5%** | **83.5%** | **+28.5 pp** ***(Can/Spray)*** |

### **2. Container Shapes**

Decoupling the 18 physical container shapes from store aisles in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0) lifts accuracy across every container form factor:


| **Container Shape** | **Sample Size (N)** | **Existing Pack Model** | **New v0.2 Pack** | **Existing 5-Attr SKU** | **New v0.2 5-Attr SKU** | **Net SKU Lift (Δ pp)** |
| --- | --- | --- | --- | --- | --- | --- |
| **Bottle** | **1,198** | **63.94%** | **93.7%** | **49.3%** | **85.3%** | **+36.0 pp** |
| **Pouch** | **747** | **65.86%** | **94.8%** | **51.3%** | **84.7%** | **+33.4 pp** |
| **Jar** | **599** | **59.60%** | **90.7%** | **43.9%** | **85.0%** | **+41.1 pp** |
| **Tube** | **513** | **55.56%** | **98.1%** | **35.3%** | **76.8% (89.0% unwarp)** | **+41.5 pp** |
| **Bar** | **408** | **81.13%** | **81.9%** | **65.2%** | **70.8%** | **+5.6 pp** |
| **Carton** | **375** | **42.93%** | **92.8%** | **31.5%** | **73.3%** | **+41.8 pp** |
| **Sachet** | **353** | **21.81%** | **94.3%** | **19.8%** | **94.3%** | **+74.5 pp** |
| **Wrapper** | **228** | **15.35%** | **97.4%** | **14.5%** | **86.4%** | **+71.9 pp** |
| **Can** | **152** | **60.53%** | **89.5%** | **44.7%** | **88.2%** | **+43.5 pp** |
| **Tub** | **144** | **25.69%** | **81.9%** | **23.6%** | **80.6%** | **+57.0 pp** |
| **Tetrapack** | **89** | **79.78%** | **83.1%** | **78.7%** | **82.0%** | **+3.3 pp** |
| **Tin** | **67** | **58.21%** | **98.5%** | **58.2%** | **95.5%** | **+37.3 pp** |
| **Roll_On** | **45** | **0.00%** | **71.1%** | **0.0%** | **66.7%** | **+66.7 pp** |
| **Pump** | **41** | **0.00%** | **95.1%** | **0.0%** | **95.1%** | **+95.1 pp** |
| **Spray (Aerosol Alias)** | 41 | 0.00% | 14.6% (97.6% Can/Spray) | 0.0% | 14.6% (85.4% Can/Spray) | Aerosol Spray/Can alias |

- **Why Bar (70.8%) and Tube (76.8% to 89.0% unwarp) account for most remaining errors:** Sideways soap bars and curved tubes suffer when flash glare obscures sub-8px variant print; even on those ties, Brand stays at 93.7% to 99.0%, `is_hul` stays at 98.6% to 99.0%, and single-shelf Share-of-Shelf error holds at 1.25%.

### **3. Ownership and Long-Tail Cohorts**


| **Cohort Slice** | **Scope** | **Existing Classifier / Closed-Set** | **New v0.2 Hybrid** | **Operational Impact** |
| --- | --- | --- | --- | --- |
| **Unilever (is_hul = True)** | N = 2,523 (5,000-crop split) | 48.51% SKU (87.08% Var, 94.45% Brand) | 88.8% SKU / 98.1% Brand / 98.4% is_hul | Drives W1 = 73.68% Basepack P2P weight (2.12x existing). |
| **Competitor (is_hul = False)** | N = 2,477 (5,000-crop split) | 36.01% SKU (65.16% Var, "Others") | 75.7% SKU / 94.1% Brand / 98.8% is_hul | Tracks Share-of-Shelf across 874 named rivals (0.4% to 1.0% False Unilever). |
| **Rare Long-Tail (1 to 2 Crops)** | 356 rare SKUs | 27.9% vision SFT (0.0% missing train) | 79.1% Exact SKU (+51.2 pp) | Studio packshot anchor matching works from 1 reference image. |
| **Day-0 Product Launches** | **18 unseen new SKUs** | **0.0% (0 / 18) across closed-set models** | **94.4% (17 / 18) in < 60 s** | **Eliminates multi-endpoint retraining lag on new launches.** |

### **3.3 Residual Forced-Choice Benchmark Gaps & Implemented Confidence-Abstention Fixes**

In forced Top-1 benchmark scoring (`0%` abstention allowed), auditing the 180 crops where at least one of the 5 attributes missed on the 1,000-crop `hul_golden` test run (`820 / 1,000` all-5-correct) shows that **`82.2%` (`148 / 180`) of all SKU-level misses still predict the exact `Brand` correctly**, and **`75.6%` (`136 / 180`) predict both `Brand` and `Category` correctly**. Every bucket below is already intercepted in production by our live `v0.2` calibration and abstention gates:

| **Residual Benchmark Gap (Forced Top-1 Scoring Across 180 Misses in 1,000 Crops)** | **Count (% of Misses / % of Total Crops)** | **Physical Store-Image or Annotation Root Cause** | **Implemented Pipeline Fix & Production Behavior (Live in v0.2)** |
| --- | --- | --- | --- |
| **1. Same-Brand & Same-Pack Sibling Variant Confusion** (`Brand`, `Category`, `Pack` all 100% correct) | **96 (`53.3%` / `9.6%`)** | Near-identical sister artwork (`cosine = 0.92–0.97`, top-2 margin `< 0.025`) where only a `< 6 px` sub-banner differs (*Lakme Sunscreen SPF 50* vs. *SPF 24*, *Nivea Honey* vs. *Turmeric*). | **Already Live in `v0.2`:** Sibling-margin factor in `jev_laya_hybrid.py` discounts `confidence["base_variant"]` while keeping `confidence["brand"] >= 0.90`; hierarchical abstention retains `Brand + Category + Pack` for SoS and sets `base_variant = "UNVERIFIED_LOW_CONFIDENCE"` so zero guessed variants enter P2P cross-sell. |
| **2. Same-Brand Container Format + Variant Shift** (`Brand` & `Category` 100% correct) | **40 (`22.2%` / `4.0%`)** | Same-brand SKU sold in both rigid and flexible formats (*Comfort Pink Lily Bottle* vs. *Pouch*, *Funfoods Mayonnaise Bottle* vs. *Poly Pouch*, *Boost Tetrapack* vs. *Bottle*). | **Already Live in `v0.2`:** Top-22% Crown/Nozzle/Crimp zoom strip + 16-class Lidstone 3D container geometry prior (`Bottle -> Tube: 10`, `Bar -> Pouch: 6`, `Bottle -> Pouch: 5`, `Jar -> Bottle: 5` remaining off-diagonal). |
| **3. Multi-Category Brand Boundary Split** (`Brand` 100% correct, `Category` split) | **12 (`6.7%` / `1.2%`)** | Cross-category master brands (*Dove*, *Vaseline*, *Pears*) spanning `Skin`, `PW`, and `Hair_Care` (`Skin -> PW: 3`, `PW -> HHC: 3`). | **Already Live in `v0.2`:** Stage 3 `sku_meta_by_bv` catalog projection resolves `(brand, base_variant)` to its deterministic SKU-specific category before falling back to brand-level category priors. |
| **4. Within-Category Cross-Brand Mismatch** (`Category` correct, `Brand` wrong) | **17 (`9.4%` / `1.7%`)** | Heavily angled or specular-glare `Oral_Care` tubes and `PW` soap bars where logo text is physically occluded. | **Already Live in `v0.2`:** Discounted by open-set InfoNCE dustbin (`tau_0 = 0.62`) and cosine clarity (`[0.42, 0.96]`); automatically abstains (`abstain_recommended = True`) when `joint_confidence < τ_cat`. |
| **5. Severe Blur / Occlusion / Stage-B Rejects** (`choice == 0`) | **15 (`8.3%` / `1.5%`)** | Motion-blurred or back-row dark cutouts (`< 38 px`) where Stage-B VLM reports `choice = 0` ("none match / unreadable"). | **Already Live in `v0.2`:** Optical Quality Gate (`< 28 px` or `Laplacian < 8.5`) and Stage-B reject gate cap `joint_confidence <= 0.36` with `abstain_recommended = True`, returning `UNVERIFIED_LOW_CONFIDENCE` instead of guessing. |

**Why We Use a Single Unified Threshold ($\tau = 0.78$) Instead of Separate Thresholds by Category:**
- **Empirical Category Uniformity (`0.922–0.941` cosine across all 10 categories):** Auditing Stage-A cosine similarity across all business categories (`Skin`, `Foods`, `HFD`, `Laundry`, `Beverages`, `PW`, `Oral_Care`, `HHC`, `Hair_Care`) proves that category-level mean similarity sits within a tight `0.922–0.941` band.
- **Difficulty Is Driven by 3D Pack Geometry & Sibling Margin—Not Category:** Remaining variant errors are caused by **3D container curvature** (curved `Jar` `72.6%` and `Bar` `75.6%` vs. flat `Wrapper` `88.6%`, `Sachet` `93.3%`, and `Can` `94.1%`) and **same-brand sibling margin ($\Delta_{\text{sib}} < 0.025$)**, both of which are already built directly into our per-crop calibrated confidence score (`calibrated_field_confidences`).
- **Ablation Proof (`88.0%` Unified vs. `87.8%` Per-Category):** As shown in the table below, we ran the per-category threshold experiment (`use_category_thresholds = True`, $\tau_{\text{cat}} \in [0.74, 0.80]$) and proved that our **Single Unified Global Threshold ($\tau = 0.78$, `use_category_thresholds = False` by default)** achieves **`98.8%` Brand Precision and `88.0%` Base Variant Precision (`87 / 180` errors removed)**—outperforming separate per-category thresholds (`87.8%`, `81 / 180` removed) without the operational complexity of maintaining 10–13 separate category thresholds.

| **Confidence Threshold ($\tau$)** | **Returned Coverage** | **Brand Precision** | **Pack Type Precision** | **Base Variant Precision** | **Variant Hallucinations Removed** | **Recommended Operating Mode** |
| --- | --- | --- | --- | --- | --- | --- |
| **Unfiltered ($\tau = 0.00$)** | `100.0%` (`1,000`) | `96.8%` | `93.4%` | `82.0%` | `0 / 180` (`0.0%`) | Closed-set benchmark evaluation (no abstention). |
| **Stage-B Reject Gate ($\tau = 0.40$)** | `98.8%` (`988`) | `97.4%` | `94.0%` | `82.8%` | `12 / 180` (`100%` of `choice=0`) | Drops all unreadable / Stage-B rejected crops (`choice = 0`). |
| **Balanced Unified Global Gate ($\tau = 0.78$, Default)** | `77.6%` (`776`) | **`98.8%`** | **`95.5%`** | **`88.0%`** | **`87 / 180` (`48.3%` removed)** | **Recommended Production Default (`use_category_thresholds = False`):** Single self-calibrating threshold across all categories. |
| **Category-Conditioned Ablation ($\tau_{\text{cat}} \in [0.74, 0.80]$)** | `81.4%` (`814`) | `98.8%` | `95.3%` | `87.8%` | `81 / 180` (`45.0%` removed) | Ablation only (`use_category_thresholds = True`): proves per-category tuning adds no precision gain over unified $\tau = 0.78$. |
| **High-Precision Cross-Sell ($\tau = 0.80–0.82$)** | `72.2%` (`722`) | `99.3%` | `96.1%` | `89.0%–89.2%` | `98–102 / 180` (`54.4%–56.7%` removed) | Recommended for strict variant-level cross-sell recommendations. |
| **Strict SKU Audit ($\tau = 0.85$)** | `65.5%` (`655`) | `99.4%` | `96.8%` | `90.7%` | `119 / 180` (`66.1%` removed) | High-purity SKU variant audit (paired with per-field abstention to keep `98.8%` Brand at `92%+` coverage). |

**How Confidence Calibration & Two-Level Abstention Work in Production (On by Default):**
1. **Open-Set Dustbin ($\tau_0 = 0.62$) & Cosine Clarity (`[0.08, 0.985]`):** Eliminates `1.0` confidence saturation (`89.3% -> 0.0%`) by competing every crop against a virtual `"None of the Above"` candidate at cosine `0.62`.
2. **Same-Brand Sibling Margin Penalty ($\Delta_{\text{sib}}$):** Discounts `confidence["base_variant"]` whenever two sister variants of the same brand have nearly identical similarity ($\Delta_{\text{sib}} < 0.025$), while keeping `confidence["brand"] >= 0.90`.
3. **Two-Level Abstention Flags Emitted on 100% of Crops by Default:**
   - **`variant_abstain_recommended = True`** (`confidence["base_variant"] < 0.78` while `brand >= 0.78`): Downstream keeps **`Brand + Category + Packaging_Type`** for Share-of-Shelf (`98.8%–99.3%` Brand precision) and abstains **only on `base_variant`** (`UNVERIFIED_LOW_CONFIDENCE`) for variant-level cross-sell.
   - **`abstain_recommended = True`** (Stage-B returns `choice == 0`, `joint_confidence < 0.38`, or `energy < 0.52`): Full crop is physically unreadable or out-of-catalog and is excluded from SKU compliance.

## **\[Optional Read\] Deploying on Google Cloud & Day-0 SKU Onboarding**

Stage 3 computes 10-category Share-of-Shelf and P2P Basepack Weights (`W1..W6`) with Rogan-Gladen prevalence adjustment and the production guardrails sized in [Cost Breakdown & Sizing](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.rnj5urhvxc2j) and [Model Whitelisting Report](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.m11c1bw64077):

- **Zero-training SKU onboarding (< 60 s hot-reload):** Adding a newly launched SKU or promotional wrapper requires only 3 to 5 reference packshots and a catalog row; updating the `< 25 MB` gallery artifact on Cloud Storage refreshes all running Cloud Run workers in under 60 seconds.
- **180 s SLA timeout protection (`RETRY_PENDING`) & 4-stage rollout:** Shelf text is isolated inside `<untrusted_pack_text>` tags, and any request exceeding the 180-second callback deadline emits `contract_status = "RETRY_PENDING"` via the dead-letter queue instead of a false zero share payout. Rollout progresses from Stage 1 shadow to Stage 4 cutover as detailed in the [End-to-End Architecture Guide](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.erkk4n45aac9).


*Figure 3: Zero-training Day-0 SKU onboarding workflow: adding 3 to 5 reference packshots updates the in-memory vector gallery across Cloud Run workers in under 60 seconds.*

### **New Variant Vector DB Enrichment Process**

- **Step 1 (Cold-Start Enrichment for New/Rare Variants):** When a new variant or promo pack launches with only 1–3 photos, our enricher automatically generates: (a) `2x` Lanczos high-res zoom strips of the Top-25% Brand Header, Middle-30% Variant Ribbon, and Bottom-30% Net-Weight Band, (b) synthetic store-lighting and perspective views to ensure single-photo variants maintain calibrated thresholds, and (c) a fused multimodal vector (`0.85 * visual + 0.15 * structured catalog text`).
- **Step 2 (Daytime Active-Learning Harvest):** During daytime store runs, whenever a real store crop is verified with high confidence (Stage-2B VLM probability `>= 0.95`), passes our Optical Quality Gate (`>= 96 px`, sharp text edges, zero glare), and represents a novel angle or lighting condition (`0.78 <= cosine <= 0.90` vs. existing medoids), it is logged to a daily staging table.
- **Step 3 (Nightly Medoid Re-Clustering & QC Gate):** A scheduled nightly Cloud Run job ingests any newly added brand or promo SKUs along with harvested store crops, recomputes the `K = 8` medoids and hard-negative diff pairs per variant, runs automated QC (ensuring leave-one-out validation accuracy for that variant improves), and publishes a versioned daily catalog snapshot (`catalog_YYYY_MM_DD`).
- **Step 4 (Work-Hours Serving):** At the start of the workday, Cloud Run workers (or our stoppable Cloud SQL / AlloyDB instance) load the latest validated daily catalog snapshot. This provides continuous improvement with zero daytime latency impact and ensures an instant rollback if an incorrect label enters the system.
- **Why we cluster all 381k–511k training crops into `K = 4 to 8` medoids instead of dumping raw crops into the live vector lookup (and how we use all of them):** We use all 381k–511k training records in the daily offline gallery build to compute `K = 4 to 8` cluster centroids and hard-negative boundaries rather than placing 511k raw crops directly into the live k-NN index:
  1. **Pose and lighting medoid clustering (`K = 4 to 8`):** Each variant's crops are clustered into `K = 4 to 8` pose and lighting groups. Clean embeddings within each cluster are averaged to reduce sensor noise, while the sharpest high-resolution crop (`>= 96 px`, high edge sharpness, low glare) is selected as the visual reference image for that cluster.
  2. **Intra-class variance and sister-variant diff mining:** Across all 381k+ crops, the offline build computes each variant's intra-class variance and identifies its top confused "sister variants", automatically generating the catalog diff headers (`[DIFF vs #1: Variant | Pack | Size]`) used by Stage 2B.

### **Embedding Refinement for Specific Products / Classes**

Because Stage 2A uses a non-parametric reference gallery rather than fixed classifier weights, improving a specific product or class does not require retraining a model and takes under 60 seconds:

- **1. Curate the reference views for that SKU:** Replace blurry, glare-heavy, or occluded reference cutouts for that class with 3 to 5 clean, representative angles (front-facing, slight side angle, and real shelf lighting) and remove any mislabeled reference crop.
- **2. Re-embed only the updated SKU (`< 60 s`):** Recompute the `gemini-embedding-2` vectors, the mean SKU centroid, and the inner-80% CIELAB colour descriptor for that SKU in the gallery cache.
- **3. Class-specific margin / hard-negative routing:** If a product has a near-identical sister variant (for example, two variants that differ only in small text), we mark them as a linked sister pair in the catalog metadata so Stage 2A never auto-accepts them on embedding similarity alone and always routes them to Stage 2B zoom verification (or abstains if the text band is unreadable).


*Figure 4: Production topology connecting mobile uploads, Cloud Pub/Sub buffering, stateless Cloud Run CPU workers, Private Service Connect Vertex AI endpoints, and automated timeout quarantine.*

## **Appendix**

### **Key Parameters**

| **Stage** | **Component** | **Key Parameters** |
| --- | --- | --- |
| **Stage 1** | Dual-buffer lighting & cylindrical unwarp | Shades-of-Gray `p = 6`; `L*`-channel CLAHE (`8x8` tiles, clip limit `2.0`); `1.15x` horizontal unwarp when width/height `< 0.48`. |
| **Stage 1** | 960px Distilled YOLO11n & row twin grouping | `960x960` FP32 ONNX (`80 ms` on 4 vCPU); RANSAC horizontal plank filter; twin grouping at 3-zone CIELAB `>= 0.985`, `dHash <= 6`, `16x16` layout cosine `>= 0.965` (`35` facings -> `~11` unique crops). |
| **Stage 2A** | In-RAM gallery & confidence split | `2,659` Unilever SKUs + `874` rival brands (`10,082` vectors, `768-D`, `< 25 MB` RAM); `K = 3` angle centroids; 2.4 MB Residual InfoNCE adapter; CSLS hubness masking (`k = 10`); direct accept at `top1 >= 0.78` and `margin >= 0.04` (`68%` of groups). |
| **Stage 2B** | Contact sheet tie-breaker & competitor guard | 4-panel strip (`[Full | 2x Bottom Weight | 2.5x Saliency-Diff | Top Cap Crown]`); `K = 5` candidates (`88.4%` prefix cache hit); open-set guard commits SKU at `s1 >= 0.72`, rival at `0.45 <= s1 < 0.72`, background at `< 0.45`. |
| **Stage 3** | SoS prevalence calibration & session deduplication | Rogan-Gladen sensitivity/specificity adjustment; multi-photo session overlap suppression (`2–5` photos/visit); timeout quarantine at `180 s` (`RETRY_PENDING`). |

### **FAQ**

1. **Why use a two-stage hybrid instead of a fine-tuned DINOv3 or MaxViT classifier?** Fine-tuned classifiers work well on head SKUs (`90.9%` on `16+` training crops) but score `27.9%` on rare SKUs (`1–2` crops) and `0.0%` on the 1,121 catalog SKUs with zero training crops, and require full retraining for every new product launch.
2. **Why is 1,000-crop accuracy 95.2% (84.0% single-pass) while 5,000-crop accuracy is 82.3%?** The 5,000-crop benchmark spans a broader long tail of 892 stores and includes more low-resolution Kirana cutouts (`< 48 px`), sideways soap bars, and rival packs (`75.7%` competitor SKU vs. `88.8%` Unilever SKU).
3. **Why did looking up pack type in the catalog add +29.60 pp to the existing chain?** In the existing chain, `29.60%` of crops got Brand, Category, and Variant right and then failed on the standalone 46-folder `<Category>__<Pack>` classifier, which conflated store aisles with container shapes.
4. **How are overlapping aisle photos in a single store visit deduplicated?** Stage 3 aligns adjacent photos in the same session using box geometry and embedding similarity along each shelf row so overlapping facings in the seam region are counted once (`1.00%–2.25%` visit SoS error).
5. **Who owns the code and catalog assets?** All pipeline modules and deployment templates are delivered under the Apache-2.0 license (see [Model Whitelisting Report](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.m11c1bw64077)), while Unilever retains exclusive ownership of its 2,659-SKU catalog, benchmark annotations, and P2P scoring rules.


---
[^c1]: Anil Sener: @jjuneja@google.com What is the best link to describe this algorithm.
    - Jigyasu Juneja: Linked directly in the table. It combines two papers: (1) Residual Feature Adapter: Gao et al., 'CLIP-Adapter: Better Vision-Language Models with Feature Adapters' (https://arxiv.org/abs/2110.04544), and (2) InfoNCE Contrastive Loss: Oord et al., 'Representation Learning with Contrastive Predictive Coding' (https://arxiv.org/abs/1807.03748) / Supervised Contrastive Learning (https://arxiv.org/abs/2004.11362). It is a 2.4 MB matrix applying v' = Norm(v + alpha * W2 * GELU(W1 * v)) on 768-D gemini-embedding-2 vectors.
