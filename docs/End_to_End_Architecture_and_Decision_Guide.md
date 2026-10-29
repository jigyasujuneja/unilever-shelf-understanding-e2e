# **End-to-End Architecture Guide**

## **Context**

This guide summarizes the target architecture across the six tabs of this design document, sized for 500,000 store shelf photos per day (15.0M per month).

Field uploads arrive from the Sales EDGE (Modern Trade) and Shikhar (General Trade) apps during the 10.5-hour daytime store window (07:30 to 18:00 IST).

These photos drive two daily business workflows: [Market Share Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.bpg2xuy2veu) (Share of Shelf) and [Merchandising Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.b5xo3ahiv7af) (retailer payouts).

In the existing system (LLD001 v0.1), both workflows run across 14 separate model weights hosted on dedicated GPU endpoints.

As detailed in [Existing Model Analysis](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.ipskbbbvu50h), chaining separate classifiers compounds errors at every step and caps five-label SKU accuracy at 42.32% on 5,000 crops.

- **Goals:** Replace the existing 14-model GPU stack on Google Cloud to lift 5,000-crop SKU accuracy from 42.32% to 82.30% to 84.00% (95.20% on 1,000 crops), onboard new SKUs in under 60 seconds without retraining, process daytime and post-18:00 IST uploads 24x7 without drops, and lower AI inference cost to $0.00160 (₹0.15) per shelf (see [Cost Breakdown & Sizing](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.rnj5urhvxc2j)).
- **Non-Goals:** Changing mobile capture apps, network perimeters, or downstream database tables. Load balancers, Cloud Armor WAF, the Shared VPC perimeter, Cloud Composer, and Cloud SQL MySQL 8.4 schemas stay unchanged.

## **Key Findings**

Testing on the leak-free single-crop (1,000 and 5,000 crops), full-shelf, and display-bay benchmarks built in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0) established six findings:

- **Replacing the classifier chain with a four-stage funnel doubles accuracy:** Matching clear products in memory and reading fine print on lookalikes lifts five-label SKU accuracy from 42.32% to 82.30% to 84.00% on 5,000 crops (95.20% on 1,000 crops).
- **Looking up pack type in the catalog removes the single largest error source:** In [Existing Model Analysis](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.ipskbbbvu50h), 29.60% of crops got brand, category, and variant right and then failed on the 46-folder pack classifier. Looking up container shape in the 2,659-SKU catalog lifts existing accuracy from 42.32% to 71.92% (+29.60 pp) before changing any vision model.
- **Two-thirds of crops need no LLM call:** Comparing each crop against a < 25 MB in-memory photo gallery (`gemini-embedding-2`) resolves 68% of product groups in under 1 millisecond on CPU. Only the remaining 32% of lookalike ties (such as 50g vs. 100g sister packs) are sent to `gemini-3.5-flash-lite`.
- **New launches work on Day 1 without retraining:** In [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0), 1,121 of 2,659 catalog SKUs (42.2%) had zero training crops, where closed-set classifiers score 0.0%. Adding 3 to 5 reference photos to the in-memory gallery takes under 60 seconds and reaches 94.4% (17/18) on unseen launches.
- **Filtering out background store clutter saves 80% of merchandising API calls:** In [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0), 61.0% to 92.1% of Kirana store boxes sat outside the Unilever display bay. Keeping only in-bay products (IoA >= 0.50) prunes 61.0% of counter clutter before recognition begins.
- **Queuing and CPU workers eliminate post-18:00 IST drops at 500,000 shelves/day:** Buffering uploads in Cloud Pub/Sub and running stateless Python 3.12 Cloud Run CPU workers (4 vCPU / 8 GiB; 10 to 12 active containers during 07:30 to 18:00 IST, max 80) handles bursts up to 80 workers with 0.00% HTTP errors (2.69s median latency) and replaces false zero-payout timeouts with `RETRY_PENDING`.


*Figure 1: Four-stage pipeline flow combining a 960px Distilled YOLO11n CPU ONNX detector (Stage 1), in-memory gemini-embedding-2 vector retrieval (Stage 2A, 68% direct match), batched gemini-3.5-flash-lite visual tie-breaking (Stage 2B, 32% lookalike ties), and business use-case solvers (Stage 3).*

## **How the New Pipeline Works**

Every shelf photo passes through four stages inside a single stateless Cloud Run CPU worker.

Stages 0 through 2B identify every product on the shelf, and Stage 3 runs the domain solvers detailed in [Market Share Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.bpg2xuy2veu) and [Merchandising Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.b5xo3ahiv7af).


*Figure 2: Production cloud topology connecting mobile uploads, Cloud Pub/Sub buffering, stateless Cloud Run CPU workers, Private Service Connect Vertex AI endpoints, and Cloud SQL and BigQuery storage.*


*Figure 3: End-to-end request lifecycle showing how shared Stage 1 detections and Stage 2 identities feed the 8 Market Share tasks (Row A) and 4 Merchandising tasks (Row B).*

### **Stage 0 and Stage 1: Image Cleanup and Shelf Detection (92 ms CPU)**

**Data insight driving design:** Auditing raw store photos in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0) showed 35.5% motion blur, 66.1% sub-64px cutouts, and partial shelf annotations (~40% of visible packs boxed).

Brightening dark shelves helped box detection but shifted dark blue sachets toward black (-6.0 pp), while fine-tuning YOLO on partially boxed shelves taught it to suppress unboxed products.

- **Dual-buffer image preparation (12 ms):** The worker keeps two copies in memory: a contrast-boosted copy (LAB L*-channel CLAHE) for box detection, and a white-balanced, colour-true RGB copy for SKU recognition, plus a 1.15x cylindrical unwarp on narrow tubes so edge text stays readable.
- **Finding bays, products, and talkers via Distilled YOLO11n and YOLO11s ONNX (80 ms):** A 960px Distilled YOLO11n CPU ONNX detector (2.6M parameters) locates Class 0 display bays (99.0% mAP50) and Class 1 product packs (96.2% mAP50) in one pass, while a 960px YOLO11s ONNX detector finds 3-class promotional shelf talkers (98.2% mAP50).
- **Why we distill on SKU-110K instead of fine-tuning on partially boxed store shelves:** Distilling YOLO11n from a 1280px YOLOv12s teacher pre-trained on exhaustively boxed SKU-110K (11,762 shelves, 1.73M boxes) and checkout layouts (800 images) avoids false-negative penalties on unboxed store pockets.
- **In-bay filtering and adjacent twin grouping:** Class-aware Soft-NMS keeps display bays from erasing products inside them, a Display Boundary Filter (IoA >= 0.50) drops 61.0% of counter clutter outside the bay, and perceptual hash grouping collapses adjacent identical packs into ~11 unique crops per 35-facing shelf (3.2x fewer API calls).

### **Stage 2A: Matching Clear Products in Memory (< 1 ms CPU)**

**Data insight driving design:** Stratifying test crops by training sample count in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0) revealed that 42.2% of catalog SKUs (1,121 of 2,659) have zero training crops and 356 rare SKUs have only 1 to 2 crops.

Closed-set classifiers score 0.0% on unseen SKUs and 27.9% on 1 to 2 crop SKUs, motivating a training-free reference gallery that embeds all ~11 unique shelf crops in one batched `gemini-embedding-2` call (180 ms):

- **Offline gallery build with Multi-Prototype K=3 K-Means and Residual InfoNCE[^c1] Adapter:** We pool 10,082 clean reference crops covering all 2,659 Unilever SKUs and 874 competitor brands, encode each into a 768-D vector with `gemini-embedding-2`, and store up to K = 3 angle centroids per SKU (< 25 MB RAM) plus a 2.4 MB Residual InfoNCE Adapter that separates sister variants.
- **Runtime matching in worker RAM (68% Direct Match, < 1 ms):** A single matrix multiply (`[11, 768] @ [768, 10082]`) with aspect-ratio filtering scores all 11 crops against all 2,659 SKUs in 0.12 ms inside worker RAM without an external vector database. When the top-1 score is >= 0.78 and the margin over the runner-up SKU is >= 0.04 (68% of groups), the worker commits all five SKU labels immediately with zero LLM tokens.

### **Stage 2B: Reading Fine Print on Lookalikes (32% Ties)**

**Data insight driving design:** Error trace analysis in [Existing Model Analysis](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.ipskbbbvu50h) and [Market Share Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.bpg2xuy2veu) showed that `gemini-embedding-2` already places the true SKU in its Top-5 shortlist on 93.3% of crops.

Over 63% of remaining Stage 2A misses are lookalike sister variants (SPF 30 vs. SPF 50), gram/ml size twins (50g vs. 100g), or coupled `<Category>__<Pack>` errors (29.60% failure rate):

- **Zoomed four-panel contact sheet:** For the remaining 32% of groups (~3.5 ties per shelf), the worker stitches `[Full Crop | 2x Bottom Net-Weight Band | 2.5x Saliency-Diff Band | Top Cap Crown]` alongside the top 5 candidate SKU names and pairwise diff tokens from Stage 2A (88.4% context prefix cache hit rate).
- **Pre-choice text reading and catalog pack lookup:** One batched call to `gemini-3.5-flash-lite` transcribes visible brand, variant words, and gram/ml weight inside isolated `<untrusted_pack_text>` tags before picking the winning SKU via Bayesian Product-of-Experts fusion, and maps container shape deterministically from the 2,659-SKU catalog (+29.60 pp lift).

### **Stage 3: Scoring the Shelf for Business Use Cases (< 5 ms CPU)**

Once every box on the shelf has a verified SKU identity and `(x, y)` coordinate, Stage 3 executes the deterministic business rules detailed in [Market Share Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.bpg2xuy2veu) and [Merchandising Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.b5xo3ahiv7af):

- **Market Share and P2P Basepack Weight ([Market Share Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.bpg2xuy2veu)):** Counts Unilever vs. 874 named competitor facings per category, adjusts for classifier sensitivity and specificity via Rogan-Gladen prevalence calibration and multi-shot panorama deduplication (0.93% Modern Trade SoS error; 1.00% multi-photo visit SoS error), and computes P2P Basepack Weight (`W1 = 73.68%`).
- **Merchandising Across Three Pillars ([Merchandising Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.b5xo3ahiv7af)):** Executes **(1) Share of Shelf (SoS) & In-Bay Layout** (63 `AWP_Code` templates, `2.38%` strip SoS error, `94.4%–95.1%` brand-block purity, `92.6%` OOS gap F1), **(2) Promotions & Price Tags** (`94.1%` `"50% Free"`, `95.6%` `"Price Off"`, `90.02%` 3-class talker F2), and **(3) Bay Display & Clutter Filtering** (`98.8%` `Class 0` Bay F2, `61.0%` out-of-bay counter clutter pruned in `< 1 ms` CPU).

## **How We Evaluated & Results**

Every benchmark dataset in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0) is partitioned by store outlet and MD5 image digest so zero store fixtures or duplicate photos leak between train, val, and test splits across all 2,659 SKUs and 892 stores.

### **1. Executive Map Across Document Tabs**

The six tabs of this document follow a direct sequence from raw data forensics and baseline root-cause analysis to pipeline design, capacity sizing, and governance:


| **Document Tab** | **How It Connects to Other Tabs** | **Headline Validated Outcome** |
| --- | --- | --- |
| **1. [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0)** | Cleans the 7 raw GCS silos into the 2,659-SKU catalog and 11 leak-free benchmarks used by every downstream tab. | Unified **2,659 Unilever SKUs + 874 competitor brands**, separated 18 canonical pack shapes from store aisles, and eliminated 399,769 cross-silo duplicates and burned-in QA boxes. |
| **2. [Existing Model Analysis](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.ipskbbbvu50h)** | Benchmarks the existing 14-model GPU stack on the clean splits from Tab 1 to isolate the exact failure modes fixed in Tabs 3 and 4. | Chaining 9 classifiers compounds errors to **42.32% 5-attribute accuracy** on 5,000 crops (52.00% even with 100% Oracle Brand), scores **0.0% on 1,121 unseen SKUs**, and drops **51.1%** of merchandising bays. |
| **3. [Market Share Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.bpg2xuy2veu)** | Uses the shared Stage 1 detector and Stage 2A/2B retrieval funnel to solve all **8 Market Share tasks** and feed P2P weights. | Lifts 5,000-crop accuracy to **82.3% to 84.0% (95.2% on 1,000 crops)**, **Modern Trade SoS error to 0.93%**, **multi-photo visit SoS error to 1.00%**, and **P2P Basepack W1 from 34.82% to 73.68%**. |
| **4. [Merchandising Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.b5xo3ahiv7af)** | Structures Merchandising into **3 Pillars** (1. SoS & In-Bay Layout, 2. Promotions & Price Tags, 3. Bay Display & Clutter Filtering) on top of the shared funnel. | Prunes **61.0% of out-of-bay counter clutter**, reaching **2.38% strip SoS error**, **98.2% in-bay Brand/Cat/Pack** (`64.3%` strict SKU), **94.4% to 95.1% brand-block purity**, **94.1% to 95.6% Promo Offer**, and **90.02% 3-class Talker F2**. |
| **5. [Cost Breakdown & Sizing](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.rnj5urhvxc2j)** | Sizes daytime Cloud Run containers and Vertex AI throughput for the funnel in Tabs 3 and 4 at **500,000 shelves/day (15M/month)**. | Prices all Google Cloud services bottom-up across daytime store hours (07:30 to 18:00 IST) at **$0.00160 (₹0.15) per shelf** for AI inference and retires all 14 GPU endpoints and Document AI. |
| **6. [Model Whitelisting Report](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.m11c1bw64077)** | Governs the production models selected in Tabs 3, 4, and 5 under LLD001 Section 5.5 AI Foundations compliance. | Whitelists **3 Tier-1 production assets** (Distilled YOLO11n/s CPU ONNX detectors, `gemini-embedding-2`, and `gemini-3.5-flash-lite`) under Zero Data Retention, CMEK, and Google IP Indemnity. |

### **2. Use-Case & Task Summary (8 Market Share + 3 Merchandising Pillars)**

The table below summarizes all production tasks evaluated in [Market Share Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.bpg2xuy2veu) and [Merchandising Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.b5xo3ahiv7af):


| **Domain & Solution Task** | **Evaluation Dataset & Scope** | **Existing Baseline** | **New Pipeline Result & Business Impact** |
| --- | --- | --- | --- |
| **MS 1: Full-Shelf Market Share (Modern Trade & Kirana)** | 200 Modern Trade & 151 General Trade full shelves | 34.82% P2P W1; competitors pooled in "Others" | **0.93% Modern Trade SoS error** (2.28% facing error); **68.3% Kirana row F2**; upgrades 66.4% of "Others" to exact SKUs; **P2P Basepack W1 = 73.68% (2.12x)**. |
| **MS 2: Multi-Photo Store Visit Deduplication** | 100 to 150 store visits (2 to 5 overlapping aisle photos/visit) | Double-counts overlapping aisle photos | **1.00% to 2.25% visit SoS error** across overlapping photos (96.5% client-parity F2, 0.0% False Unilever attribution). |
| **MS 3: Single-Crop SKU Classification at Scale** | 5,000 held-out product crops across 892 stores | 42.32% 5-attribute SKU (87.1% Brand, 54.9% Pack) | **82.30% exact 5-attribute SKU** (96.1% Brand, 91.7% Pack) in 0.82s (+40.0 pp, p < 10^-6). |
| **MS 4: High-Precision Cap & Fine-Print Zoom** | 1,000 held-out product crops (4-panel zoom sheet) | 44.50% (52.00% with Oracle Brand) | **83.5% Val / 84.0% Test** single-pass zoom (94.0% Pack, 96.5% Brand) and **95.20%** on the full hybrid production lock. |
| **MS 5: Zero-LLM & Fast-Skip Embedding Retrieval** | 5,000 crops & 18 unseen Day-0 product launches | 0.0% on 1,121 unseen catalog SKUs | **81.2% with 50% LLM skip** and **78.9% to 80.9% with zero LLM calls**; **94.4% (17/18) on Day-0 launches in < 60s**. |
| **MS 6: Lookalike Sister Variant Disambiguation** | 1,000 lookalike crops across 1,922 sister variants | 44.50% on 9-model classifier chain | **85.0% Val / 83.5% Test** across lookalike sister variants and size twins in 0.56s. |
| **MS 7: Brand & Competitor Attribution** | 1,000 crops spanning Unilever & 874 rival brands | 87.1% Brand; rivals grouped into "Others" | **95.9% exact Brand, 98.4% ownership accuracy, 0.4% False Unilever** (94.4% Brand in zero-LLM mode). |
| **MS 8: 3D Packaging Shape Classification** | 1,000 & 5,000 crops across 18 canonical container shapes | 54.86% on standalone pack-type model | **93.3% to 94.0% (1K) and 91.7% (5K)** combining catalog lookup with top-cap crown zoom. |
| **Merch Pillar 1: Share of Shelf (SoS) & In-Bay Layout** | 100 strips (`gt_hul_sessions`) & 25 test bays (`gtmerch_golden`) | 51.1% Bay F2; 53.2% E2E strict basepack; 58.0% Planogram | **2.38% strip SoS error**; **98.2% avg. Brand/Cat/Pack** (`64.3%` strict Exact-SKU on test bays; `574` `"Others"` boxes named); **94.4% to 95.1% purity**; **92.6% OOS F1**. |
| **Merch Pillar 2: Promotions and Price Tags** | 1,000 promo crops (`hul_promos`) & 50 test shelves (`hul_talkers_golden`) | 75.0% fixed ViT (23.5% `"50% Free"`); 49.2% Talker F2 | **84.0% Brand+Cat+Pack** (`93.7%` Brand, `94.1%` `"50% Free"`, `95.6%` `"Price Off"`); **90.02% annotated-zone Talker F2** (`98.2%` mAP50) in 3.05s. |
| **Merch Pillar 3: Bay Display and Clutter Filtering** | 100 full store shelves (`gtmerch_golden` test) | 0% filtered (61.0% counter clutter sent to classifier) | **98.8% `Class 0` Bay F2**; prunes **61.0% out-of-bay counter clutter (`< 1 ms` CPU)**; **77.3% full-shelf box F2** (`94.7%` pre-cut block purity) in 3.7s. |

### **3. Architecture Options Overview**

The table below compares the four evaluated architecture options at 500,000 shelves/day (see [Cost Breakdown & Sizing](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.rnj5urhvxc2j) for complete infrastructure and SKU pricing tables):

*Figure 4: Monthly cloud spend at 500,000 shelves/day across architecture options (Panel A) alongside accuracy across Market Share tasks (Panel B1) and Merchandising tasks (Panel B2).*


| **Architecture Option** | **How It Works** | **5-Attribute SKU Accuracy (1K / 5K)** | **AI Inference Cost / Shelf** | **Operational Trade-Off** |
| --- | --- | --- | --- | --- |
| **Option 3: Hybrid + Local CPU ONNX Embedder** | Distilled INT8 CPU ONNX embedder ($0 Embedding API) + `gemini-3.5-flash-lite` on 32% ties | 94.8% / 82.1% to 83.8% | $0.00058 (₹0.06) | Lowest variable cost; requires maintaining local ONNX embedder alongside gallery. |
| **Option 1: Embedding-Only** | 960px Distilled YOLO11n ONNX + `gemini-embedding-2` (0 LLM tokens, 0.90s median latency) | 91.6% / 78.9% to 80.9% | $0.00105 (₹0.10) | Fastest turnaround; lower accuracy on fine-print gram/ml size twins. |
| **Option 2: Hybrid (Recommended)** | **Option 1 (68% direct match) + `gemini-3.5-flash-lite` on 32% lookalike ties (2.69s median latency)** | **95.2% / 82.3% to 84.0%** | **$0.00160 (₹0.15)** | **Best accuracy on lookalike variants and promo bands; Day-1 SKU onboarding in < 60s.** |
| **Existing GPU Baseline (v0.1)** | 14 separate model weights on dedicated L4 GPU endpoints + Document AI | 44.5% / 42.32% | $0.00200 to $0.00267 | Saturates above 150,000 shelves/day and requires retraining for every new launch. |

### **4. Error Taxonomy, Confidence Calibration & Operational Limitations**

To prevent downstream cross-sell algorithms from consuming low-evidence guesses on blurred or occluded crops, the pipeline logs calibrated per-attribute confidences (`confidence["brand"]`, `confidence["base_variant"]`, `confidence["packaging_type"]`, `confidence["category"]`), `joint_confidence` in `[0.02, 0.985]` (eliminating `1.0` confidence saturation from `84.1%` to `0.0%`), and `abstain_recommended`:

- **82.2% of SKU errors still get `Brand` right (`148 / 180` errors on 1,000 crops):** Out of 180 crops with any attribute error, **53.3% (`96`)** have `Brand`, `Category`, and `Packaging Type` 100% correct and differ only on micro-text sibling variants (`margin < 0.025`, e.g., *Lakme Sunscreen SPF 50* vs. *SPF 24*), **22.2% (`40`)** are same-brand container shifts (*Comfort Bottle* vs. *Pouch*), **6.7% (`12`)** are multi-category brands (*Dove*), **9.4% (`17`)** are within-category cross-brand errors, and only **8.3% (`15` crops, `1.5%` of total)** are severe blur/occlusion misses.
- **Stage-B reject abstention & hierarchical fallback:** Whenever the Stage-B contact-sheet verifier returns `choice = 0` ("none match / unreadable"), `joint_confidence` is capped at `<= 0.36` (`abstain_recommended = True`) and the endpoint returns `None` when abstention is enabled. For sibling-variant ties (`margin < 0.025`), downstream callers can either apply a global/category confidence threshold (`88.0%–90.7%` variant precision and `98.8%–99.4%` brand precision at $\tau \in [0.78, 0.85]$) or use **per-field abstention** (`abstain_per_field = True`) to keep high-confidence `Brand + Category + Pack` predictions while abstaining only on `base_variant`.

## **Path to Production & Rollout**

Production rollout, daytime capacity sizing, and governance are consolidated here and in [Cost Breakdown & Sizing](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.rnj5urhvxc2j) and [Model Whitelisting Report](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.m11c1bw64077) (replacing the standalone deprecated rollout tab).

### **1. Specification Updates (LLD001 v0.1 to v0.2) & Daytime Sizing**

Updating LLD001 from `v0.1` to `v0.2` keeps external load balancers, Cloud Armor WAF, mobile capture apps, and Cloud SQL MySQL 8.4 schemas unchanged while making six targeted infrastructure improvements:

1. **Runtime Security (Closes IST0001092):** Upgrades worker containers to `python:3.12-slim` with `onnxruntime` and `google-genai`, removing end-of-life Python 3.8 to 3.10 packages, existing classifier weights, and static credentials.
2. **Serverless Inference Consolidation:** Retires the 14 dedicated GPU endpoints and Document AI in favor of in-process 960px Distilled YOLO11n/s CPU ONNX detection plus Private Service Connect calls to `gemini-embedding-2` and `gemini-3.5-flash-lite`.
3. **Cloud Pub/Sub Surge Buffering:** Places a Cloud Pub/Sub queue and dead-letter topic between the ingestion router and Cloud Run workers to absorb daytime bursts and post-18:00 IST uploads without dropped connections (`< 150 ms` HTTP 202 acknowledgment).
4. **Timeout Quarantine (`RETRY_PENDING`):** Replaces false `Earning = "0"` payout rejections on 180-second timeouts with `contract_status = "RETRY_PENDING"` for automatic background retry.
5. **AI Foundations Governance ([Model Whitelisting Report](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.m11c1bw64077)):** Covers all Vertex AI calls under Zero Data Retention, CMEK encryption, and Google Cloud IP Indemnity, with zero open-weight R&D models deployed in production.
6. **Daytime Worker Sizing & Load Validation ([Cost Breakdown & Sizing](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.rnj5urhvxc2j)):** Grouping adjacent identical facings (`35` to `~11` unique crops) and batching API calls reduces per-shelf latency from `94.6s` (sequential script) to **`2.69s` P50**. Across the 10.5-hour daytime window (`07:30 to 18:00 IST`), 500,000 shelves/day averages `13.2 shelves/sec`, requiring **10 to 12 active `4 vCPU / 8 GiB` Cloud Run containers** (`25 to 40` at a `2.5x` midday peak, max 80), verified across 570 load-test requests with **0.00% HTTP failures** and P95 latency under 5.0 seconds.

*Figure 5: Concurrency load testing across 10, 40, and 80 workers with zero HTTP failures (Panel A) and per-shelf latency reduction from sequential execution to batched Cloud Run workers (Panel B).*

### **2. Four-Stage Rollout & Automated Rollback**

Because monthly P2P payouts aggregate store visits across a 30-day billing cycle, splitting traffic randomly by photo would mix `v0.1` and `v0.2` predictions inside the same store's monthly scorecard.

Instead, the ingestion router hashes each store cluster (`CRC32(cluster:ol_segment) % 100`) so every store stays on a single pipeline version for the full 30-day billing cycle, while Rogan-Gladen prevalence calibration keeps category Share-of-Shelf trendlines continuous across versions.

If HTTP 5xx errors exceed 0.5%, P95 latency exceeds 15.0s, or `RETRY_PENDING` callbacks exceed 3.0%, automated circuit breakers route live traffic back to the control subscription in under 60 seconds with zero database schema changes.


*Figure 6: Store-cluster hash router locked for the 30-day billing cycle across control and treatment subscriptions with automated rollback under 60 seconds.*


| **Rollout Stage (500,000 Shelves/Day Scale)** | **Traffic Split** | **Downstream Database & Callback Writes** | **Go / No-Go Exit Criteria** |
| --- | --- | --- | --- |
| **Stage 1: Paired Dark Shadow** | 100% live to `v0.1`; 100% shadow copy to `v0.2` | `v0.2` writes only to BigQuery audit tables (zero live callbacks or Cloud SQL writes). | >= 99.9% completion on 1M+ shelves; P95 < 12s; blinded 500-crop SME Paired McNemar >= 75% win share (p < 0.01). |
| **Stage 2: 10% Cluster Canary** | 10% of store clusters (~50,000 shelves/day) routed to `v0.2` | `v0.2` writes live callbacks and Cloud SQL for 10% of clusters; 20% BigQuery audit continues. | 0 false zero-payout timeouts; False Unilever rate <= 1.5%; 24h re-shoot rate <= Control; SoS MAE <= 2.5%. |
| **Stage 3: 50/50 Stratified A/B** | 50% of store clusters (~250,000 shelves/day) + 100% post-18:00 IST uploads routed to `v0.2` for 30 days | Full P2P monthly scoring and Share-of-Shelf continuity splice across control and treatment arms. | Treatment P2P Basepack W1 >= 70.0% vs. Control ~34.82%; AI inference unit cost <= $0.00160 (₹0.15) / shelf. |
| **Stage 4: 100% Production Cutover** | 100% of 500,000 shelves/day served by Cloud Run CPU workers (`v0.2`) | Drain `v0.1` subscription, hold warm standby during burn-in, then retire all 14 GPU endpoints and Document AI. | Close security exception `IST0001092` in ServiceNow and decommission existing GPU endpoints. |

## **Appendix & Stakeholder Sign-Off**

### **1. Verification Suite & Stakeholder Approvals**

All 159 unit and integration tests pass (`159/159`), covering store-disjoint benchmark splits, 2,659-SKU catalog completeness, P2P Basepack Weight (`W1..W6`) math, Rogan-Gladen prevalence adjustment, Paired McNemar significance testing, 63-template planogram alignment, `<untrusted_pack_text>` prompt isolation, and `RETRY_PENDING` timeout handling.


| **Stakeholder Group** | **Primary Document Tab** | **Scope of Approval** |
| --- | --- | --- |
| **Unilever Data Engineering** | [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0) | 2,659-SKU catalog linkage rules, 46-to-18 pack mapping, and store-disjoint benchmark splits. |
| **AI Foundations & Legal** | [Model Whitelisting Report](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.m11c1bw64077) | Section 5.5 Tier-1 whitelisting (`gemini-embedding-2`, `gemini-3.5-flash-lite`) and Distilled YOLO11n/s CPU ONNX deployment. |
| **Cloud Infra & Security** | [End-to-End Architecture Guide](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.erkk4n45aac9) & [Cost Breakdown & Sizing](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.rnj5urhvxc2j) | Cloud Pub/Sub queue + dead-letter topic, `RETRY_PENDING` callback handling, and Python 3.12 worker sizing (`IST0001092`). |
| **Brand Analytics & Commercial Ops** | [Market Share Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.bpg2xuy2veu) & [Merchandising Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.b5xo3ahiv7af) | Option 2 production architecture at 500,000 shelves/day and 30-day store-cluster A/B rollout. |

### **2. FAQ**

- **Why not use a Vision-Language Model directly to detect all boxes on the shelf?** A store shelf contains 35 to 120 tightly packed items. Asking a Vision-Language Model to emit 50+ bounding-box coordinate strings takes 4 to 6 seconds and misses small hanging sachets (< 65% recall). Our 960px Distilled YOLO11n CPU ONNX detector finds 96.2% to 99.0% of boxes in 80 ms on CPU at zero API cost.
- **Why not fine-tune a closed-set classifier as the default?** As shown in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0), supervised fine-tuning drops to 27.9% on rare SKUs (1 to 2 training crops) and 0.0% on new launches (1,121 catalog SKUs without training crops). Keeping `gemini-embedding-2` and `gemini-3.5-flash-lite` frozen and storing SKU knowledge in the < 25 MB reference gallery reaches 94.4% (17/18) on unseen launches in under 60 seconds with zero retraining.
- **What happens if a request takes longer than the 180-second SLA?** Instead of returning `AWP_Presence: "N", Earning: "0"` (which wrongly denies the retailer's payout), the worker returns `contract_status = "RETRY_PENDING"` and routes the message to the dead-letter queue for automatic background retry.


---
[^c1]: Anil Sener: @jjuneja@google.com What is the best link to describe this algorithm.
    - Jigyasu Juneja: Linked directly in the table. It combines two papers: (1) Residual Feature Adapter: Gao et al., 'CLIP-Adapter: Better Vision-Language Models with Feature Adapters' (https://arxiv.org/abs/2110.04544), and (2) InfoNCE Contrastive Loss: Oord et al., 'Representation Learning with Contrastive Predictive Coding' (https://arxiv.org/abs/1807.03748) / Supervised Contrastive Learning (https://arxiv.org/abs/2004.11362). It is a 2.4 MB matrix applying v' = Norm(v + alpha * W2 * GELU(W1 * v)) on 768-D gemini-embedding-2 vectors.
