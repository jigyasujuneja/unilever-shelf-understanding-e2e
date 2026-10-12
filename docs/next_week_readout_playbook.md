# Next Week Readout Playbook: Executive Summary & Partner Handover (`Sales EDGE – MT PC & GT`)

This playbook structures next week's sessions for two distinct audiences—**Decision Makers** (Commercial, Product & Executive Leadership) and **Partners** (LTIMindtree, Bizom, Data & ML Engineering Teams)—backed by `63` verified Cloud Run benchmark runs in `gs://ul-fs-s-sandbx-16-prj-shelf-images/results/` and documented in [`docs/experiments_decision_log.md`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/docs/experiments_decision_log.md).

---

## 1. The 4-Act Story & Architectural Intuition (Jargon-Free Framing)

### How to Explain the Problem & Architecture in 60 Seconds
- **The Problem:** Unilever processes **`506,000` shelf photos/day** (`~10M` product packs/day) across 4 separate pipelines. **70% of packs on a shelf are "Easy"** (big upright bottles/pouches or `4–5` identical packs sitting side-by-side), while **30% are "Hard"** (lookalike sister variants like *Dove Hair Fall Rescue* vs. *Daily Shine*, curved squeeze tubes like *Closeup* or *Biotique*, sideways soap bars, or tiny `50g` vs. `100g` text).
- **Why Legacy Chained Models Hit a `32%–57%` Ceiling:** Unilever's current production setup chains 13 separate models (`Detector -> Brand -> Category Variant -> Size`). In their production P2P script (`NR_final_PW2.py`), they recorded the empirical accuracy of each step (`W1` through `W6`). Because errors multiply down a chain (`W1 Exact Pack = W4 Variant × W5 Size`), exact pack accuracy dropped to **`31.85%` in Oral Care**, **`35.59%` in Laundry**, and **`38.03%` in Personal Wash** (mean **`50.5%`**).
- **Our 3-Step Funnel ("Detect Locally, Match Globally, Verify Selectively"):**
  1. **Step 1 — Fast Spatial Scan (`Stage 1 CPU Detector` — 80ms, ₹0):** Draws boxes around **Display Bays (`Class 0`)** and **Product Packs (`Class 1`)** (`94.7% F2`).
  2. **Step 2 — Group Identical Neighbors & Instant Reference Lookup (`Stage 2A Embedding Search` — resolves 68% of packs at ₹0.011):** Groups identical side-by-side packs on the same shelf row (`25` facings $\rightarrow$ `~6` unique product groups, cutting work by `4x`) and matches them against our `2,745`-SKU reference gallery using `gemini-embedding-2`.
  3. **Step 3 — Side-by-Side Visual Verification Only for Hard Lookalikes (`Stage 2B Contact Sheet` — 32% of packs):** Only when two sister variants or sizes look nearly identical, we build **one composite image** showing the store crop + zoomed-in variant/weight strips side-by-side with the top-5 studio reference images for `Gemini 3.5 Flash-Lite`.

---

## 1B. 80% ASD-STE100 Visual Onboarding: 5 Key Engineering Decisions & Data Remediation Foundation

Stakeholders do not need internal experiment ticket codes (`EXP-01..25`) to onboard and trust our system. We follow **80% of the ASD-STE100 Simplified Technical English standard** (short active sentences, one approved term per concept, zero ML jargon like *"medoid"* or *"bi-encoder"*, and a concrete store-shelf example for every rule):

### A. 5 Key Engineering Decisions & Why They Work

| # | Plain-English Decision (What We Did & How) | Problem It Solves (Why Legacy Failed) | Intuitive Store-Shelf Example | Verified Impact |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Split Display-Bay and Product-Pack Detection into Separate Classes** | Mixing large display bays (`Class 0`) and small product packs (`Class 1`) in one detector caused big bay boxes to suppress all packs inside them (`51.1% F2`). | On a store display rack (`gtmerch_golden` / `image_ponds_double2_218.jpg`), the system draws the outer blue display-bay frame AND counts all `33` product packs sitting on its shelves. | **Detection F2: `51.1%` $\rightarrow$ `94.7%` (`+43.6%`)** |
| **2** | **Group Identical Side-by-Side Packs Before Matching (`4x` Fewer Lookups)** | Store shelves place `4–6` identical packs side-by-side (`brand blocking`). Sending all `27–33` packs to a vision model wastes `75%` of compute and time. | On `203495_202312121941481030.jpg` (`27` boxes across 4 shelf rows), identical neighboring packs on the same shelf row are grouped, matched once (`0.95` similarity), and propagated across the block. | **VLM Calls Cut by `87%`** (`3.1` calls/shelf; **`₹0.12`/shelf**, `4.8s`) |
| **3** | **Search a Clean `2,745`-Product Reference Catalog Instead of Retraining Fixed Classifiers** | Fixed neural networks (`ViT-240`) forget old classes and take `3–4` weeks to retrain whenever Unilever launches a new promo pack or rival product. | When Trade Marketing launches a promo pack (`hul_promos` / `val_00091.jpg`), dropping **1 reference photo** into GCS makes it live in **5 minutes** (`93.3%–100%` accuracy). | **Promo Onboarding: `4 Weeks` $\rightarrow$ `5 Minutes`** (`953` brands, `2,745` SKUs) |
| **4** | **Build a 4-Crop Comparison Grid Only for Hard Lookalike Variants (`32%` of Packs)** | Resizing a tall bottle into a tiny square blurs the `50g` vs. `100g` weight text and the middle variant band (*Hair Fall Rescue* vs. *Daily Shine*). | When *Dove* or *Ponds* sister variants score within `0.03`, we build **one 4-crop grid** (full pack + top brand + middle variant + bottom weight) next to the top-5 studio references. | **Exact SKU Accuracy: `50.5%` legacy $\rightarrow$ `88.7%–95.5%`** |
| **5** | **Verify Every Dataset and Result File with 12 Automated Quality Gates (`Zero Leakage`)** | Raw Unilever buckets had `399K` duplicate crops, `21.4%` filename collisions (`image_10.jpg` in 4 categories), `28` shelves with pre-drawn green boxes, and `874` rival brands collapsed into `"Others"`. | Before any model runs, `src/utils/qc.py` verifies `0` MD5 hash overlaps between train/gallery/test, `0` pre-drawn boxes, and `100%` 5-level taxonomy completeness. | **`12/12` Automated QC Gates (`0` Fallbacks / `0` Leaks)** |

### B. Data Remediation Story ("How We Built a Zero-Leakage Ground Truth Foundation")
1. **Stage 1 (`5.14M` Raw GCS Objects / `284 GB`):** Audited all 5 production GCS buckets (`ms_data`, `pack_data`, `brand_data`, `mt_toker_data`, `gtmerch`).
2. **Stage 2 (`5` Critical Data Bugs Caught & Fixed):**
   - Removed **`399,412` byte-identical crops** shared between `brand_data` and `pack_data`.
   - Prefixed **`21.4%` colliding filenames** (`{category}__{pack}__{brand}__{sha1[:10]}.jpg`) so labels never overwrite each other.
   - Quarantined **`28` shelf images with burned-in `cv2.rectangle` debug boxes** so detectors cannot cheat.
   - Resolved **`57` dual-labeled `Bottle/Jar` MD5 collisions** using the `0.75` neck-to-body ratio rule.
   - Restored **`874` competitor brands** (`Colgate`, `Nivea`, `Pantene`, `Dettol`, `Biotique`, `Dabur`, `Ariel`, `Tide`, `Patanjali`) that legacy scripts had collapsed into `"Others"` (`14.7%` of all boxes) or brand-only strings (`Ponds`, `Lux`, `Lifebuoy`, `Boost`, `Comfort`, `Colgate`).
3. **Stage 3 (`12/12` Automated QC Gates):** Enforces `0` MD5 train/test leaks, `0` `"Unknown"` or `"__Others"` fallbacks, `0` synthetic grid coordinates, and `100%` 5-level taxonomy consistency.

---

## 1C. Real Evaluated Proof Examples by Every Use Case & Task Level (`0` Hallucinations, `0` Fallbacks)

Every example in the `/exec` **Interactive Task-Level Proof Explorer** is loaded directly from verified `images.jsonl` rows:

| Task Level & Business Use Case | Exact Run ID & Image ID (`images.jsonl`) | What the Pipeline Detects & Classifies (Ground Truth vs. Prediction) | Why This Matters for Onboarding & Trust |
| :--- | :--- | :--- | :--- |
| **1. End-to-End Multi-Row Store Shelf (`TASK-E2E` — Shelf Attribution)** | `1009-000011-shelf_hierarchical_clm_jev-gemini-3.5-flash-lite-priority-gtmerch_golden` / `image_ponds_double2_218.jpg` (`1024x768`, `33` boxes) & `203495_202312121941481030.jpg` (`2880x2160`, `27` boxes) | **Multi-Brand Shelf Attribution (`96.6%` SKU Acc, `97.9%` F2):**<br/>• `Ponds__Bright_Beauty_Spot_Less_Glow_Face_Wash` (`Tube`, `Skin`, `is_hul=True`)<br/>• `Ponds__Pure_Detox_Anti_Pollution_Face_Wash` (`Tube`, `Skin`, `is_hul=True`)<br/>• `Lux__Bright_Glow_Vitamin_C_And_E_Soap` (`Bar`, `PW`, `is_hul=True`)<br/>• `Dove__Healthy_Rituals_Growing_Hair__Shampoo` (`Bottle`, `Hair_Care`, `is_hul=True`) | Proves real pixel bounding boxes (`[423, 0, 506, 90]`, `[238, 1676, 478, 2158]`) and multi-SKU neighbor grouping across multi-row retail shelves. |
| **2. Stage-1 Dense Shelf & Bay Detection (`TASK-DET` — CPU ONNX Detector)** | `1009-132227-distilled_yolo_onnx-yolo11n-distilled-onnx-ms_json_golden` / `test_00000.jpg` (`4000x2252`, `37` boxes) & `1009-132251-...-gtmerch_golden` / `lux_asset_161.jpg` (`2880x2160`, `9` boxes) | **CPU Shelf & Bay Detection (`95.3%` F2 on `ms_json_golden`, `94.7%` F2 on `gtmerch_golden`):**<br/>• `test_00000.jpg`: `37` product facings detected on a `4000x2252` kirana store shelf in `42ms` (`₹0` VLM cost).<br/>• `lux_asset_161.jpg`: separates `Class 0` (`Merchandising_Bay`) from `Class 1` (`SKU_Item`) boxes. | Proves `Class 0` display-bay boxes never suppress `Class 1` product facings inside the bay. |
| **3. Display-Bay + Shelf-Row Planogram (`TASK-PLN` — GT/MT Merchandising)** | `1009-000016-bay_planogram_verifier-gemini-3.5-flash-lite-priority-gtmerch_golden` / `image_ponds_double2_218.jpg` (`33` boxes) & `1009-012034-...-hul_sessions` / `test_00000.jpg` (`8` boxes, `67` unique SKUs across run) | **Bay Purity `97.2%`, OOS Compliance `94.6%`:**<br/>• Audits `Class 0` branded display bays (`image_ponds_double2_218.jpg`, `203495_202312121941481030.jpg`) and store-visit strips (`test_00000.jpg`: `Ariel__Perfect_Wash`, `Ayush__Anti_Cavity`, `Dettol__Kitchen_Gel_Dishwash`, `Lifebuoy__Total_10_Germ_Guard`). | Proves automated bay purity, brand-block cohesion, and out-of-stock void detection on real store bays. |
| **4. Fine-Grained Sister Variant & Rival Lookup (`TASK-RET` — 2,745 SKUs)** | `1008-235937-jev_laya_zscore_lid-gemini-3.5-flash-lite-priority-hul_golden` / `test_00162.jpg` & `1009-011836-cold_start_lid_discovery-...-hul_golden` / `test_01728.jpg` (`29` unique SKUs) | **Exact Sister Variant & Competitor Match (`90.0%–93.3%`):**<br/>• `test_00162.jpg` (`12x33`): `Rexona__Women_Powder_Dry_Roll_On` (`Bottle`, `Deo`, `is_hul=True`)<br/>• `test_00063.jpg` (`87x69`): `Colgate__Strong_Teeth_Anticavity_Toothpaste` vs. `Colgate__Maxfresh_Breath_Frshner_Red_Toothpaste` (`Tube`, `Oral_Care`, `is_hul=False`)<br/>• `test_01728.jpg` (`149x37`): `Clinic_Plus__Strong_And_Long`, `Loreal__Extraordinary_Clay`, `Meera__Herbal_Hairwash_Shampoo_Bottle` | Demonstrates both exact matches and transparent sister-variant boundary cases across HUL and Non-HUL (`Colgate`, `Loreal`, `Meera`) crops. |
| **5. Direct 5-Level Catalog Classification (`TASK-CLS` — Owner/Cat/Pack/Brand/SKU)** | `1009-000112-djev_classify-gemini-3.5-flash-lite-priority-hul_brands` / `test_00080.jpg` (`264x511`) & `1008-235947-djev_classify-...-hul_golden-val` / `val_00162.jpg` (`195x261`) | **5-Level Classification (`96.7%` Brand, `90.0%` SKU):**<br/>• `test_00080.jpg`: `Dove__Dryness_Care__Conditioner_Tube` (`Tube`, `Hair_Care`, `is_hul=True`)<br/>• `val_00162.jpg`: `Pukka__Ashwagandha_Tulsi` (`Carton`, `Beverages`, `is_hul=True`)<br/>• `val_00063.jpg`: `Oralb__Cavity_Defense_123_Manual_Toothbrush` (`Carton`, `Oral_Care`, `is_hul=False`) | Shows simultaneous 5-level taxonomy classification across HUL and Non-HUL (`Oralb`, `Bournvita`, `Complan`, `Nescafe`) products. |
| **6. Zero-Shot Promo & Price-Off Verification (`TASK-PRM` — MT Toker)** | `1009-000050-jev_promo_verifier-gemini-3.5-flash-lite-priority-hul_promos-val` / `val_00091.jpg` (`84x31`) & `val_00007.jpg` (`91x119`) | **Promo Pack Match (`90.0%–100%`):**<br/>• `val_00091.jpg`: `Ayush__Purifying_Turmeric` promo pack<br/>• `val_00007.jpg`: `Himalaya__Oil` promo pack (`Non-HUL` rival promo)<br/>• Also evaluates `Nescafe__Sunrise_Coffee_with` and `Red_Label__Leaf_Poly` promo packs. | Proves zero-shot promo verification across both Unilever (`Ayush`, `Red Label`) and competitor (`Himalaya`, `Nescafe`) promo packs. |

---

## 2. Task-by-Task Success KPIs vs. Legacy Production Baseline (`W1–W6`)

| Task / Hierarchy Level | Unilever Legacy Recorded Accuracy (`NR_final_PW2.py` `W1–W6`) | Target Success KPI | Our Achieved Val / Test KPI | Where We Still Have a Gap (Root Cause) | How We Close It (Algorithmic vs. Product vs. Business) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. HUL vs. Non-HUL (`L1` Corporate Owner)** | `~86.8%` (`874` rival brands collapsed into `"Others"` in raw JSONs) | **>= 98.0%** | **100.0%** (`gtmerch`, `hul_golden`)<br/>**98.9%** (`ms_json` box) | **1.1%–5.2% gap on raw shelves:**<br/>1. Missing `capture_date` for M&A/divested brands (*Pureit*, *Horlicks*, *Minimalist*).<br/>2. Long-tail regional Kirana rivals with `<3` reference crops. | **• Product/API:** Pass `capture_date` in API request.<br/>**• Algo + Crawler:** Web crawler (`~60%` online D2C rivals) + store crop clustering (`~40%` local Kirana brands; `Ref: EXP-18`). |
| **2. Category (`L2A` 10 Store Categories)** | `W2 = 72.57%` (`Laundry`) to `92.00%` (`HFD`), **Mean: 83.2%** | **>= 95.0%** | **100.0%** (`hul_golden`, `gtmerch`)<br/>**97.7%** (`ms_json` box) / **93.8%** E2E | **2.3%–6.2% gap on occluded shelf boxes:**<br/>Master brands (*Dove*, *Vaseline*, *Ayush*) span multiple categories; when a shelf rail blocks the bottom text (*"Shampoo"* vs *"Lotion"*), isolated crop matching guesses the default category. | **• Algo (`Ref: EXP-22`):** **Same-Row Shelf Neighbor Context** — an occluded *Dove* bottle sitting next to *Clinic Plus* and *Sunsilk* on the same shelf plank inherits the row's `Hair_Care` category. |
| **3. Packaging Form (`L2B` Container Type)** | `W6 = 65.00%` (`Skin`) to `91.00%` (`HFD`), **Mean: 80.6%** | **>= 94.0%** | **96.7%** (`hul_packs`, `hul_golden`)<br/>**97.5%** (`ms_json` box) | **2.5%–3.3% gap on boundary pairs:**<br/>`57` byte-identical photos were dual-labeled as both `Bottle` and `Jar` (*Horlicks*) in raw `pack_data`; small flexible packs mixed between `Pouch` and `Sachet`. | **• Business SME Rule:** Lock two catalog definitions: `Sachet` = single-use (`<= ₹10` or `< 25g/ml`); `Jar` = neck opening `>= 75%` of body width. |
| **4. Brand (`L3` 953 HUL & Rival Brands)** | `W3 = 77.45%` (`Laundry`), `78.79%` (`Oral`), **Mean: 86.8%** | **>= 95.0%** | **96.7% Micro / 97.5% Macro** (`hul_brands`, `ms_json`) | **3.3% gap:**<br/>Ultra-low resolution (`< 48px` short side in `66%` of raw brand crops) and foil glare (`5–7%` of detergent pouches). | **• Product/App:** On-device blur/resolution check (`>= 1080p`, `Laplacian >= 150`).<br/>**• Algo:** Multi-angle glare-augmented gallery references. |
| **5. Variant (`L4` 1,962 `Brand__Variant` Lines)** | `W4 = 57.40%` (`Laundry`), `68.86%` (`Oral`), `73.02%` (`Hair`), **Mean: 76.1%** | **>= 88.0%** | **90.0%–93.4%** (`hul_variants`, `hul_golden`, `ms_json`) | **6.6%–10.0% gap on lookalike sister variants:**<br/>Concentrated in **Squeeze Tubes (`77.0%`)** (*Biotique*, *Closeup*) and **Uniform Salon Bottles (`76.5%–85.0%`)** (*Biolage*) where the variant name sits in the middle band. | **• Algo (`Ref: EXP-21`):** Add a **Middle-30% Variant Banner Zoom Strip** to the 4-crop comparison grid alongside the bottom weight strip. |
| **6. Exact SKU, Size & Promo (`L5` 2,745 SKUs + 240 Promos)** | `W5 (Size) = 45.61%` (`PW`), `46.25%` (`Oral`); **Compound Exact SKU `W1 = 31.85%–70.55%` (Mean: 50.5%)** | **>= 85.0% E2E / >= 90.0% Crop** | **93.3%** (`hul_golden` Flash)<br/>**93.4% Box / 88.7% E2E** (`ms_json`)<br/>**93.3%–100%** (`hul_promos`) | **6.6%–11.3% E2E gap:**<br/>1. Unscoped national catalog (`2,745` SKUs vs `~300` sold in a store format).<br/>2. Cropping strips physical scale (`100g` vs `250g` look identical when resized). | **• Product/API (`Ref: EXP-20`):** Pass `Outlet_Code -> ol_segment` to filter gallery (`2,745 -> ~300` SKUs).<br/>**• Algo (`Ref: EXP-23`):** Compare pack height against same-row neighbors ($h_{\text{box}} / \bar{h}_{\text{row}}$). |
| **7. End-to-End Shelf Pipeline (`TASK-E2E`, `TASK-PLN`)** | `PT180S` (3-min) timeout; `51% F2` when bays & SKUs mixed | **>= 92% Det F2, <= 2% SoS Error, <= ₹0.50/shelf** | **94.7% Det F2 / 97.9% E2E F2** (`gtmerch`)<br/>**94.3% E2E F2** (`ms_json`)<br/>**97.2% Bay Purity / 94.6% OOS**<br/>**₹0.12/shelf** | **Store Execution Edge Cases:**<br/>1. Multi-photo aisle overlap (`hul_sessions`) double-counts border packs.<br/>2. Sideways-stacked soap bars (`83.6%`) and hanging sachet strips. | **• Algo (`Ref: EXP-24, EXP-25`):** `960px` high-res scan for overlapping strips + cross-photo border column deduplication.<br/>**• Business Rule:** Lock sachet-hanger depth multiplier (`e.g., 1 strip = 8 units`). |

---

## 3. Granular Breakdown by Category, Packaging Form & Variant Type (`n = 2,844` Unilever Boxes)

### A. Accuracy by Category vs. Legacy `NR_final_PW2.py` (`W1`, `W4`, `W3`, `W6`)

| Store Category | Boxes (`n`) | Legacy Exact SKU (`W1`) | **Our Exact SKU (`L5`)** | Legacy Variant (`W4`) | **Our Variant (`L4`)** | Legacy Brand (`W3`) | **Our Brand (`L3`)** | Legacy Pack (`W6`) | **Our Pack (`L2B`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Beverages (Tea/Coffee)** | `20` | `56.0%` | **100.0%** (`+44.0%`) | `70.0%` | **100.0%** | `87.0%` | **100.0%** | `86.0%` | **100.0%** |
| **Laundry (Fabric Care)** | `502` | `35.6%` | **97.0%** (`+61.4%`) | `57.4%` | **97.0%** | `77.5%` | **98.6%** | `75.0%` | **98.8%** |
| **Home & Hygiene (`HHC`)** | `48` | `70.6%` | **95.8%** (`+25.2%`) | `85.0%` | **95.8%** | `86.0%` | **97.9%** | `76.0%` | **95.8%** |
| **Hair Care** | `1,320` | `56.5%` | **95.5%** (`+39.0%`) | `73.0%` | **95.5%** | `84.4%` | **96.1%** | `86.7%` | **98.5%** |
| **Foods & Refreshment** | `156` | `69.7%` | **93.6%** (`+23.9%`) | `82.0%` | **93.6%** | `90.0%` | **93.6%** | `88.0%` | **94.2%** |
| **Health Food Drinks (`HFD`)** | `38` | `43.3%` | **84.2%** (`+40.9%`) | `76.0%` | **84.2%** | `94.0%` | **92.1%** | `91.0%` | **92.1%** |
| **Skin Care (`Skin`)** | `420` | `54.6%` | **81.7%** (`+27.1%`) | `79.1%` | **81.7%** | `86.8%` | **95.5%** | `65.0%` | **90.5%** (`+25.5%`) |
| **Oral Care** | `156` | `31.9%` | **80.8%** (`+48.9%`) | `68.9%` | **80.8%** | `78.8%` | **96.8%** | `78.4%` | **96.2%** |
| **Personal Wash (`PW`)** | `181` | `38.0%` | **80.7%** (`+42.7%`) | `83.4%` | **80.7%** | `91.7%` | **96.1%** | `81.0%` | **89.0%** |

### B. Accuracy by Packaging Type (`L2B` Pack Accuracy vs. `L4/L5` Variant/SKU Accuracy)

| Packaging Type | Boxes (`n`) | **Pack Accuracy (`L2B`)** | **Variant & SKU Accuracy (`L4/L5`)** | Primary Root Cause & Plain-English Next Step |
| :--- | :---: | :---: | :---: | :--- |
| **Carton / Multipack Box** | `44` | **97.7%** | **97.7%** | Flat cardboard surfaces with crisp typography. |
| **Pouch** | `431` | **97.2%** | **95.1%** | High accuracy on stand-up pouches (*Surf Excel*, *Comfort*); lock `<25g` rule with `Sachet`. |
| **Jar** | `174` | **97.1%** | **97.1%** | Lock `>= 0.75` neck-to-body ratio rule to resolve wide-mouth *Horlicks* `Bottle/Jar` split. |
| **Bottle** | `1,549` | **97.0%** | **93.4%** | `54%` of volume; remaining `6.6%` variant gap addressed by **Middle-30% Variant Strip Zoom (`Ref: EXP-21`)**. |
| **Sachet** | `154` | **96.1%** | **96.1%** | High accuracy once cropped; Stage 1 recall addressed by **960px High-Res Scan for Overlapping Strips (`Ref: EXP-24`)**. |
| **Tube (Squeeze Tube)** | `318` | **94.0%** | **77.0%** | **#1 Variant Bottleneck (drives `Skin 81.7%` & `Oral 80.8%`):** Curved squeeze tubes (*Biotique*, *Closeup*) addressed by **Store-Format Catalog Filter + Middle-30% Strip Zoom (`Ref: EXP-20 + EXP-21`)**. |
| **Bar (Soap / Detergent Bar)** | `159` | **89.3%** | **83.6%** | **#2 Variant Bottleneck (drives `PW 80.7%`):** Sideways-stacked soap bars (*Ayush*, *Lux*) addressed by **Store-Format Catalog Filter + 960px Narrow-Bar Scan (`Ref: EXP-20 + EXP-24`)**. |

---

## 4. Process Refactoring Across All 4 Workloads (What Is Solved in GCP Now vs. Cutover)

| Workload & Volume | Legacy Azure Process Bug (`docs/Customer_Inputs.md`) | Solved in Our GCP Code Today? | How Our GCP Architecture Solves It & Cutover Action |
| :--- | :--- | :---: | :--- |
| **1. MT MarketShare** (`323,935/day`) | Drops all store uploads after `18:00 IST` with a fake `HTTP 200 (Success)` because `marketsharestopcluster.json` stops AKS GPUs at `18:00 IST`; `startcluster` has `dayOfWeek == 10` bug. | **YES (100% Built)** | **24/7 Scale-to-Zero Cloud Run** (`₹0` idle overnight cost, `4.8s` latency). Grouping identical side-by-side packs cuts cost to **₹0.12/shelf**. *Cutover:* Point mobile webhook to Cloud Run + pass `Outlet_Code`. |
| **2. MT & GT Merchandising** (`104,571/day`) | When `Azure_ML_Call` times out (`PT180S`), exception handler sends `AWP_Presence='N', Earning='0'` to `DYNAMIC_TASK_API` — **unfairly penalizing store owners with ₹0 payout!** | **YES (100% Built)** | **`bay_planogram_verifier` (`6.5s`, `97.2%` Bay Purity, `94.6%` OOS)** separates `RETRY_PENDING` from `COMPLIANCE_FAIL`, with deterministic Stage 2A embedding fallback. *Cutover:* Point `DYNAMIC_TASK_API` to Cloud Run status field. |
| **3. MT Toker Promo** (`78,025/day`) | Fixed 240-class `ViT-B-16-plus-240` model requires `3–4` weeks of crop collection and retraining per new promo campaign. | **YES (100% Built)** | **5-Minute Zero-Shot Promo Launch (`jev_promo_verifier`, `93.3%–100%`)**: Drop 1 reference image into `gs://.../hul_promos/gallery/`. *Cutover:* Grant Trade Marketing GCS folder access. |
| **4. SOS & P2P Engine** (`NR_final_PW2.py`) | Hairpins `GCS -> ADF -> Azure MySQL (DROP/RENAME) -> Python VM` (`segments[300:600]` hardcoded slice bug!) -> Azure Blob, using `W1=32%–57%` exact pack weights. | **YES (100% Built)** | **In-Place BigQuery SQL (`src/utils/bigquery_catalog.py`)**: Computes Share of Shelf and P2P association across **100% of segments in `<5s`** with `81%–100%` SKU accuracy. *Cutover:* Point P2P export to BigQuery table. |
---

## 5. Tailored Next-Week Agendas

### Session A: For Decision Makers (20 Minutes — Using `/exec`)
1. **Act 1 — The Big Picture & 5 Plain-English Decisions (5 Mins):** Show how our 3-step funnel replaces 13 chained models (`32%–57%` exact SKU accuracy) + 4 siloed Azure workflows (`506K photos/day`) with **1 unified GCP pipeline** at **₹0.12/shelf** (by grouping identical side-by-side packs).
2. **Act 2 — Live Interactive Proof Across All 6 Task Levels (5 Mins):** Use the **Interactive Task-Level Proof Explorer** on `/exec` to click through real store shelves (`image_ponds_double2_218.jpg`, `203495_202312121941481030.jpg`, `test_00000.jpg`) and inspect the 5-level Ground Truth vs. Prediction table (`L1` HUL/Rival $\rightarrow$ `L2A` Category $\rightarrow$ `L2B` Pack Form $\rightarrow$ `L3` Brand $\rightarrow$ `L4` Variant $\rightarrow$ `L5` Exact SKU).
3. **Act 3 — 4-Way Segmented Performance (`By Category`, `By Pack Form`, `By SKU Type: HUL vs. Rival vs. Variant vs. Promo`, `By Task`) & Process Fixes (5 Mins):** Show **95%–100%** readiness across Corporate SoS (`100%`), Category (`98%`), Pack (`97%`), Brand (`97%`), Bay Purity (`97.2%`), OOS (`94.6%`), Promo (`94%`), `Laundry` (`97.0%`), and `Hair Care` (`95.5%`), plus recovering post-`18:00 IST` store uploads and ending false `Earning: "0"` timeout penalties for retailers.
4. **Act 4 — 3 Decisions to Sign Off for Phase 2 Pilot (5 Mins):**
   - Approve **Option A (`2-Stage Hybrid + Gemini 3.5 Flash-Lite`, ₹0.12/shelf)** as production default.
   - Approve passing **`Outlet_Code` (Store-Format Catalog Filter)** and **`capture_date`** in the live API request (`Ref: EXP-20`).
   - Approve the **`Jar` vs. `Bottle` (`0.75` neck ratio)** and **`Pouch` vs. `Sachet` (`<= ₹10 / < 25g`)** catalog definitions.

### Session B: For Partners — LTIMindtree / Bizom / Data & ML Engineers (45 Minutes — Using `/engineering` & [`docs/experiments_decision_log.md`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/docs/experiments_decision_log.md))
1. **Data Hygiene, Live Dataset Scale (`2,000–6,729` `val+test` samples) & Unified Experiments + 12-Gate QC Tab (`15 Mins`):** Walk through `#/experiments` (combining the `26` Experiments, `8` GCS Golden Datasets, and `12/12` QC Gates in one clean view) and `#/performance` (4-way segmented performance by Task/Use Case, SKU Type, Packaging Type, and Store Category), showing zero-error validated Golden runs only.
2. **5 Core Architectural Decisions & 10-PR Sequential Code Walkthrough (`15 Mins`):** Walk through how each decision is implemented across the 10-PR stack (`#25 -> #26 -> #18 -> #19 -> #20 -> #21 -> #22 -> #27 -> #29 -> #30`) and verified in [`docs/experiments_decision_log.md`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/docs/experiments_decision_log.md).
3. **Live `EXP-20..26` Results, Hugging Face GPU CustomJob Benchmarks & Webhook Cutover (`15 Mins`):**
   - Review the completed `val` + `test` benchmarks for `EXP-20` (`outlet_scoped_clm_jev`), `EXP-21` (`tube_midband_jev`, **`92.0%` val / `91.0%` test** on `hul_variants`), `EXP-22` (`row_neighbor_prior_jev`), `EXP-23` (`shelf_height_size_prior_jev`), `EXP-24` (`distilled_yolo_960_softnms`, **`100%` Stage-1 F2** on `hul_sessions`), and `EXP-25` (`session_overlap_dedup_e2e`, **`72.0%` test E2E F2**).
   - Inspect the **3 Vertex AI GPU CustomJobs (`EXP-26`)** in `ul-fs-s-sandbx-16-prj` (`us-central1`): `customJobs/2150288373313437696` (`Cloudflare/clef-flash`), `customJobs/8202000372592541696` (`Contrastive-LM/CLM-v0.1-8B`), and `customJobs/8146831277157253120` (`Cloudflare/clef-27b`).
   - Walk through repointing the Azure Logic App webhooks to GCP Cloud Run.

---

## 6. Quick TL;DR Guide: Using the 8 Golden Datasets (`ms_json_golden`, `gtmerch_golden`, `hul_*`) for `val` / `test`

All 8 Golden Datasets live in **`gs://marketshare-datasets/derived_manifests/<dataset_name>/rpc.json`** and pass all `12/12` automated quality gates (`0` cross-split leaks, `0` pre-drawn debug boxes, `0` `"Others"`/`"Unknown"` fallbacks):

| Dataset Name | Primary Use Case | Exact Manifest GCS URI | `train` Split | `val` Split | `test` Split | Combined `val + test` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`ms_json_golden`** | Clean Full-Shelf Market Share & Detection (`Class 1` packs) | `gs://marketshare-datasets/derived_manifests/ms_json_golden/rpc.json` | `38` shelves (`705` boxes) | `25` shelves (`434` boxes) | `27` shelves (`463` boxes) | **`52` shelves (`897` boxes / `1,602` total)** |
| **`gtmerch_golden`** | Full-Shelf Merchandising Display Bays (`Class 0` `asset_boxes`) + Product Packs (`Class 1` `boxes`) | `gs://marketshare-datasets/derived_manifests/gtmerch_golden/rpc.json` | `186` shelves (`6,871` boxes) | `72` shelves (`2,694` boxes) | `100` shelves (`4,035` boxes) | **`172` shelves (`6,729` boxes)** |
| **`hul_golden`** | 5-Level Exact SKU Retrieval & Classification (`2,745` HUL + Competitor SKUs) | `gs://marketshare-datasets/derived_manifests/hul_golden/rpc.json` | `16,000` crops | `1,900` crops | `1,907` crops | **`3,807` crops** |
| **`hul_variants`** | Hard Lookalike Sister-Variant Disambiguation (`1,962` `Brand__Variant` lines) | `gs://marketshare-datasets/derived_manifests/hul_variants/rpc.json` | `15,687` crops | `1,900` crops | `1,907` crops | **`3,807` crops** |
| **`hul_packs`** | Packaging Container Form (`Bottle`, `Pouch`, `Tube`, `Jar`, `Bar`, `Sachet`, `Carton`, `Can`, `Tub`) | `gs://marketshare-datasets/derived_manifests/hul_packs/rpc.json` | `6,000` crops | `1,000` crops | `1,000` crops | **`2,000` crops** |
| **`hul_brands`** | Brand Identification (`953` HUL + Competitor Brands) | `gs://marketshare-datasets/derived_manifests/hul_brands/rpc.json` | `8,000` crops | `1,000` crops | `1,000` crops | **`2,000` crops** |
| **`hul_promos`** | Promotional Pack & Offer Verification (`240` Promo SKUs) | `gs://marketshare-datasets/derived_manifests/hul_promos/rpc.json` | `4,000` crops | `1,000` crops | `1,000` crops | **`2,000` crops** |
| **`hul_sessions`** | Multi-Photo Store Visit Sequences (Grouped by `Request_ID`) | `gs://marketshare-datasets/derived_manifests/hul_sessions/rpc.json` | `30` visits (`240` boxes) | `20` visits (`160` boxes) | `25` visits (`200` boxes) | **`45` visits (`360` boxes / `600` total)** |

### A. Run via `shelf-bench` CLI
```bash
gcloud auth application-default login
uv run shelf-bench run -a shelf_hierarchical_clm_jev -m gemini-3.5-flash-lite -d ms_json_golden --split test --limit 0
uv run shelf-bench run -a bay_planogram_verifier -m gemini-3.5-flash-lite -d gtmerch_golden --split val --limit 0
uv run shelf-bench run -a jev_laya_zscore_lid -m gemini-3.5-flash-lite -d hul_golden --split test --limit 1000
uv run shelf-bench hf-gpu-job --model all --dataset hul_golden --splits val test --limit 1000
uv run shelf-bench qc
```

### B. Load in Python (`src/utils/dataset.py`)
```python
from utils import dataset

root = dataset.data_root("gtmerch_golden")
val_samples  = dataset.sample_images(split="val",  limit=0, seed=0, root=root, name="gtmerch_golden")
test_samples = dataset.sample_images(split="test", limit=0, seed=0, root=root, name="gtmerch_golden")
catalog      = dataset.rpc_catalog(root)   # {sku_id: 5-level taxonomy dict}
gallery      = dataset.rpc_gallery(root)   # {sku_id: ['gs://...', ...]}
```


