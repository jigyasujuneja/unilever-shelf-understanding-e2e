# **Merchandising Pipeline Design**

## **Context**

Field representatives photograph Unilever display bays, hanging strips, promotional packs, and shelf-rail talkers in General Trade (GT) and Modern Trade (MT) stores during daytime operating hours (07:30 to 18:00 IST). As outlined in the [End-to-End Architecture Guide](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.erkk4n45aac9), the pipeline is sized for 500,000 store shelf photos per day (15.0M per month).

Retailer payouts against Unilever's 63 Always-Winning Planogram (`AWP_Code`) templates depend on three pillars:

- **1. Share of Shelf (SoS) and in-bay layout:** Name each product inside the paid display bay (`is_hul`, `category`, `brand`, `base_variant`, `packaging_type`), compute in-bay Share of Shelf, and check that each brand sits in one contiguous block without empty Out-of-Stock (OOS) gaps.
- **2. Promotions and price tags:** Read on-pack promotional offers (such as Buy 1 Get 1 Free, 50% Extra, or ₹10 Off) and find the three kinds of shelf-rail tag (`promo_talker`, `printed_price_tag`, `handwritten_price_tag`).
- **3. Bay display and clutter filtering:** Locate the branded Unilever fixture (`Class 0` — window bays, countertop units, multi-bay units, and full wall displays, while keeping frameless `HAN` hanging strips separate) and remove surrounding counter products before SKU lookup.

[Existing Model Analysis](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.ipskbbbvu50h) and [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0) found four problems with the existing GT Merchandising chain and its data:

- **The classifier cannot tell pack sizes apart:** 36% of pack classes merge 2 to 4 pack sizes into one label (such as `100ml_200ml`), and 24% are `"90__O__<Brand>"` catch-alls.
- **Most boxes sit outside the bay:** In `gtmerch-marketshare`, only 7.9% to 39.0% of `Class 1` product boxes sit inside a `Class 0` Unilever bay; the remaining 61.0% to 92.1% sit on surrounding Kirana counter shelves.
- **Counting SKUs misses layout problems:** Summing SKU counts across a bay passes shelves where brands are scattered across rows and misses empty gaps in the middle of a row.
- **New planograms and promos need retraining:** Promo offers were baked into class names (such as `_B3G1` or `_Rs_10`), and raw talker data was not split by store outlet.

### **Goals and Non-Goals**

- **Goals:** Measure in-bay Share of Shelf (`2.38%` SoS error) and check row layout and OOS gaps against all 63 `AWP_Code` templates without retraining; read on-pack offers (`84.0%` brand/category/pack correct; `94.1%` to `95.6%` on `"50% Free"` and `"Price Off"`) and 3-class shelf-rail talkers (`90.0%` zone F2); and filter out `61.0%` of surrounding counter clutter before SKU lookup.
- **Non-Goals:** Changing field-rep mobile capture apps or downstream retailer payout database schemas.

## **Key Findings**

All figures below come from **test-split runs** on store-disjoint splits built in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0). Validation scores on the 15-shelf `gtmerch_golden` validation split (~95% exact SKU) overstate performance because thresholds were tuned there, so only held-out test results are reported:

- **1. Share of Shelf (SoS) & In-Bay Layout:** On 100 test strips (`gt_hul_sessions`, 800 boxes), the layout verifier reaches **2.38% Share-of-Shelf error**, **96.9% HUL ownership**, **72.5% exact 5-attribute SKU accuracy**, and **100% brand-block purity**. On end-to-end display bays (`gtmerch_golden` test shelves), the pipeline names the **exact SKU on 64.3%** of in-bay products (comparable to the existing classifier's `61.4%` strict basepack score), gets **brand, category, and pack right 98.2% of the time on average** (`94.8%` across the 3,420-crop in-bay set), reaches **68.2% to 69.8% in-bay end-to-end F2**, and upgrades **574 `"Others"` boxes** into specific SKUs.
- **2. Promotions and Price Tags:** On 1,000 `hul_promos` test crops, separating the 69 promo suffixes from base variants gets **brand, category, and pack all right 84.0% of the time** (up from `75.0%` on the existing 240-class ViT baseline) and reaches **94.1% on `"50% Free"`** and **95.6% on `"Price Off"`** headers. Across all 9 raw `promo_type` labels, exact offer-code match is **40.3%** because raw labels spell the same offer in multiple ways. On 50 `hul_talkers_golden` test shelves, 3-class talker detection reaches **90.0% annotated-zone F2** (`84.9%` full-frame F2) in `3.05 s` and discovers **+84 real unlabelled rail tags** (`1.3x` coverage).
- **3. Bay Display and Clutter Filtering:** De-duplicating `Class 0` bays separately from `Class 1` products stops bay boxes from deleting the products inside them (lifting Bay F2 from `51.1%` to `98.8%`). Shelf-plank and bay-boundary filtering drops **61.0% of surrounding counter clutter** before SKU lookup, and fast box-only bay detection on 100 test shelves reaches **77.3% bay-scoped F2** in `3.7 s` without SKU lookup.

| **Merchandising Pillar** | **Test Set & Scope** | **Existing Baseline** | **Test Result & Operational Impact** |
| --- | --- | --- | --- |
| **1. Share of Shelf (SoS) & In-Bay Layout** | 100 `gt_hul_sessions` strips (800 boxes) & `gtmerch_golden` test bays | 61.4% classifier / 53.2% E2E strict basepack; 44.5% uncropped[^c1]; SKU counts only | **2.38% SoS error & 96.9% HUL ownership** (72.5% SKU, 100% block purity on strips); **64.3% strict exact SKU & 98.2% avg. brand/category/pack** on E2E bays (94.8% across 3,420 crops); **574 `"Others"` boxes named**; **92.6% OOS gap F1**. |
| **2. Promotions and Price Tags** | 1,000 `hul_promos` test crops & 50 `hul_talkers_golden` test shelves | 75.0% 240-class ViT (23.5% on `"50% Free"`); 64.2% talker mAP@50 (49.2% F2) | **84.0% brand + category + pack all correct** (73.0% 5-attr SKU, **94.1% `"50% Free"`, 95.6% `"Price Off"`, 40.3% raw 9-way `promo_type`**); **90.0% annotated-zone talker F2** (84.9% full-frame, +84 valid tags in 3.05 s). |
| **3. Bay Display and Clutter Filtering** | 100 `gtmerch_golden` test shelves (box-only) & 9,194 bay photos | 51.1% Bay F2; 86.8% Class 1 F2 on 200 pre-cropped bays; 61.0% counter clutter | **98.8% Bay F2**; **prunes 61.0% of out-of-bay counter clutter** before SKU lookup; **77.3% bay-scoped F2 in 3.7 s** for fast box-only bay checks (filters printed wing and header art). |


*Figure 1: Five-stage Merchandising pipeline. Stage 1 prepares two copies of the image, Stage 2 finds bays, packs, and talkers, Stage 3 removes out-of-bay counter clutter, Stage 4 names each product and promo claim, and Stage 5 checks Share of Shelf, row layout, and OOS gaps.*

## **How the Pipeline Works**

### **Shared Stage 1: Dual Image Buffers for Store Lighting (12 ms CPU)**

**Data insight driving design:** In [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0), Kirana photos mixed dark rack interiors, warm tungsten bulbs, and specular flash glare on foil sachets. Boosting global contrast improved box detection recall but shifted brand colours (turning dark navy Vaseline purple and washing out yellow Vim).

Stage 1 therefore builds two aligned copies of each photo in memory (`12 ms` CPU) that feed all three merchandising pillars:

- **Colour-true buffer (for Pillar 1 & Pillar 2 SKU/promo crops):** Shades-of-Gray white balance (`p = 6` on non-clipped pixels) corrects the store's ambient light colour so brand colours remain true when Stage 4 crops each product.
- **Contrast-boosted buffer (for Pillar 2 & Pillar 3 detection):** Local contrast enhancement (`CLAHE`) is applied to lightness (`L*`) only, leaving colour channels (`a*`, `b*`) untouched so dark back-row sachets and faint shelf-rail price tags remain visible to detectors.

### **Pillar 1: Share of Shelf (SoS) and In-Bay Layout (Stages 4 & 5)**

**Task and use case:** Audit paid Unilever display bays (`WIN`, `CTU`, `MB2`, `MB3`, `FULLWALL`) and frameless hanging strips (`HAN`) against Unilever's 63 `AWP_Code` planograms by naming every in-bay product (`is_hul`, `category`, `brand`, `base_variant`, `packaging_type`), computing in-bay Share of Shelf, and verifying contiguous brand blocks and Out-of-Stock (OOS) voids.

**Data insight driving design:** In [Existing Model Analysis](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.ipskbbbvu50h), 36% of the existing 209 classifier classes merged 2 to 4 pack sizes into one label (`100ml_200ml`), 24% were `"90__O__<Brand>"` catch-alls, and summing flat SKU counts passed bays where brands were scattered across rows or separated by empty OOS gaps.

**How the solution works:**

- **Program-scoped embedding lookup (`Stage 4A`, `gemini-embedding-2`):** Surviving in-bay crops from the colour-true buffer are matched against the 2,659-SKU catalog (narrowed to the active `AWP_Code` program's 15–40 eligible SKUs plus hard-negative sister packs). Identical neighbouring facings are grouped so only one medoid crop per group is embedded, and high-margin matches (`68%` of groups) are accepted immediately on CPU in `< 1 ms` with zero LLM tokens.
- **Contact sheets for lookalikes and sister pack sizes (`Stage 4B`, `gemini-3.5-flash-lite`):** Uncertain crops (`32%` of groups) are stitched alongside their top-5 candidate reference packshots and a `2x` zoomed bottom net-weight strip (`50g` vs. `100g`), allowing `gemini-3.5-flash-lite` to switch to the matching sister SKU and flagging `abstain_recommended = True` when fine print is unreadable.
- **2D shelf-row clustering and brand-block purity (`Stage 5A`):** Vertically overlapping in-bay boxes (`y_IoU >= 0.40–0.45`) are grouped into horizontal shelf rows from top to bottom and sorted left to right (`95.2%` row F1). Each row scores `1.0` when every brand sits in one contiguous block (`1 - (switches - (B - 1)) / N`) and is aligned against the target `AWP_Code` sequence via normalized Levenshtein similarity.
- **Row-adaptive OOS gap detection and `AWP_Code` rules (`Stage 5B`):** Horizontal gaps between adjacent packs wider than `1.35x` to `1.50x` the row's median pack width are flagged as empty OOS voids (`92.6%` F1). All 63 `AWP_Code` templates are stored as declarative JSON/SQL rules (fixture type, required brands, `>= 0.90` HUL share for bays or `>= 0.70` for `HAN`, and `>= 0.80` purity), with a `180 s` SLA timeout quarantine (`contract_status = "RETRY_PENDING"`) so transient timeouts never record a false zero payout.

### **Pillar 2: Promotions and Price Tags (Stages 2 & 4)**

**Task and use case:** Read on-pack promotional offers (such as `"50% Extra"`, `"Buy 3 Get 1"`, or `"₹10 Off"`) on product cutouts without retraining for new campaigns, and localize three classes of shelf-rail tags (`promo_talker`, `printed_price_tag`, `handwritten_price_tag`) on full shelf photos.

**Data insight driving design:** In [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0), baking 69 promo suffixes (`_Rs_10`, `_B3G1`, `_PO2`) into fixed classifier folder names caused `23.5%` recall on `"50% Free"` packs and forced full model retraining for every new campaign, while small shelf-rail price tags (`< 25 px` tall) were erased by overlapping product boxes during standard NMS.

**How the solution works:**

- **Decoupled base-variant and promo-suffix catalog (`Stage 4A`):** Separating the 69 price and promo suffixes from `base_variant` lets Stage 4 match promotional and regular packs against the same 2,659-SKU gallery while onboarding new festive wrappers in `< 60 s` from 1 to 5 reference crops.
- **Top-and-bottom band zoom for promotional packs (`Stage 4B`):** Because offer banners sit in the top `25%–28%` header or bottom `20%–30%` band of a pack and blur at thumbnail resolution, the promo verifier stitches `[full crop | 2x zoomed top band | 2x zoomed bottom band | top-5 candidates]` so `gemini-3.5-flash-lite` reads the exact offer text and maps it to the canonical campaign family (`"50% Free"`, `"Price Off"`).
- **Parallel 3-class shelf-talker detection (`Stage 2 Talker Pass`):** A parallel pass on the contrast-boosted image separates `promo_talker` (branded strips, wobblers, danglers), `printed_price_tag` (barcode/MRP labels), and `handwritten_price_tag` (marker cards), discarding non-tag boxes and applying class-isolated NMS (`IoU = 0.50`) so product boxes never suppress adjacent rail tags.

### **Pillar 3: Bay Display and Clutter Filtering (Stages 2 & 3)**

**Task and use case:** Localize the paid Unilever display fixture (`Class 0`: window bays `WIN`, countertop units `CTU`, multi-bay units `MB2`/`MB3`, and `FULLWALL` displays, while keeping frameless `HAN` hanging strips separate), prune surrounding Kirana counter stock and printed cardboard artwork before SKU lookup, and support fast `3.7 s` box-only bay audits.

**Data insight driving design:** In [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0), `61.0%` to `92.1%` of annotated `Class 1` product boxes in Kirana store photos sat on surrounding store counters outside the `Class 0` Unilever bay, cross-class NMS allowed the outer bay box to delete the products inside it (`51.1%` Bay F2), and detectors frequently mistook flat 2D pack graphics printed on cardboard side wings for real stock.

**How the solution works:**

- **Class-decoupled bay and pack detection (`Stage 2`):** Detects `Class 0` display bays and `Class 1` products together while de-duplicating bays (`IoU = 0.58`) and products separately so a bay box never deletes the products inside it (`98.8%` Bay F2). Dense shelves (`>= 45` boxes or `>= 24` small boxes) trigger `2x2` overlapping tiles (`12%` overlap) plus up to `3` bay zoom crops (`3%` margin), while open aisles share the zero-token 960px Distilled `YOLO11n` CPU ONNX detector from [Market Share Pipeline Design](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.bpg2xuy2veu).
- **Printed cardboard artwork and rail filter (`Stage 3A`):** Shape rules and a local Laplacian texture gate (`>= 4.2`) reject flat 2D pack illustrations printed on cardboard headers and side wings so fixture artwork is never counted as physical inventory.
- **Physical shelf-plank alignment filter (`Stage 3B`):** Groups pack bottoms (`y_max`) that line up horizontally (`±6.5%` of bay height) into physical shelf planks, dropping floating header detections and partial "shoulders" of packs sitting behind the front row.
- **Display bay boundary gate (`Stage 3C`, `IoA >= 0.50`):** Products whose Intersection-over-Area inside the detected `Class 0` Unilever bay (expanded by `±8%` width for edge-clipped sachets) is below `0.45–0.50` are discarded in `< 1 ms` CPU before Stage 4, removing **61.0% of background counter clutter** (and automatically bypassing `Class 0` cropping for frameless `HAN` hanging strips and `FULLWALL` scenes).


*Figure 2: In a Kirana store photo, Stage 3 keeps products resting on shelf planks inside the Unilever display bay (39.0%) and ignores surrounding counter products and printed header art (61.0%). Stage 5 then checks Share of Shelf, brand blocks, and OOS gaps.*

### **Journey of a Store Image Across the 5 Merchandising Tasks (3 Pillars)**

To make each pillar easy to trace end to end, the table below follows a concrete store image through all five simplified merchandising tasks (`Pillar 1A`, `1B`, `2A`, `2B`, and `3`):

| **Pillar × Simplified Task** | **Stage 1: Optical & Photometric Gate** | **Stage 2 & 3: Spatial Detection & Clutter Gate** | **Stage 4: Catalog Lookup & Zoom Verification** | **Stage 5: Planogram Audit, Calibrated Confidence & Output** |
| --- | --- | --- | --- | --- |
| **Pillar 1A · Paid Display Bay SKU & SoS** (`gtmerch_golden`, `#1021-154952`) | Shades-of-Gray (`p = 6`) + `L*` CLAHE (`clip = 2.0`) fixes Kirana cabinet glare while keeping natural RGB in Buffer A. | Detects `Class 0` bay (`98.8%` F2) and filters `Class 1` packs by `IoA >= 0.50`, pruning `61.0%` counter clutter in `< 1 ms`. | Groups adjacent identical facings (`IoU >= 0.92`); `68%` clear groups resolve via `gemini-embedding-2` (`cos >= 0.86`, `Δcos >= 0.06`); `32%` ties go to `3.5-Flash-Lite` zoom sheet. | **98.2% avg. Brand/Cat/Pack**, **64.3% strict 5-Attr SKU** (vs. `44.5%` legacy), **+63.0% `"Others"` upgraded**, **100% AWP bay purity**, calibrated conf `[0.02, 0.985]`. |
| **Pillar 1B · Shelf-Row Planogram & OOS Voids** (`gt_hul_sessions`, `#1013-232627`) | Dual-buffer normalization plus shelf-rail tilt estimation on wide bay / hanging-strip photos. | Clusters pack centers by vertical overlap (`>= 0.45`) into physical rows `(R_1 ... R_m)` and sorts left-to-right `(C_{r,1} ... C_{r,k})`. | Assigns Brand and Basepack family to each row slot (`98.2%` brand/pack family fidelity) with zero LLM calls on high-margin crops. | **94.4%–100% Brand-Block Purity**, **92.6% OOS Void F1** (`Δx >= 1.35x` row median width), **2.38% Strip SoS error**, and deterministic `AWP_Code` pass/fail in `< 5 ms`. |
| **Pillar 2A · Promo Pack Offer OCR** (`hul_promos`, `#1021-105453` & `#1013-025000`) | Verifies crop edge (`>= 28 px`) and Laplacian sharpness (`>= 8.5`) before slicing regional promo bands. | Extracts `2x` Lanczos zoom strips of the **Top-28% Promo Callout Header** and **Bottom-30% Net-Weight Band**. | Matches base SKU against `2,659`-SKU gallery (`K = 3` centroids) and runs Pre-Choice OCR inside `<untrusted_pack_text>` tags for offer strings. | **84.0%–90.0% 3-Attr accuracy**, **94.1% `"50% Free"`**, **95.6% `"Price Off"`**, **87.8% Multipack**, and **60-second zero-retraining** campaign onboarding. |
| **Pillar 2B · 3-Class Shelf Talkers & Price Tags** (`hul_talkers_golden`, `#1021-105453`) | Preserves high-contrast yellow/red rail tag borders and fine price numerals across `532` disjoint store outlets. | Single-pass 3-class detector (`1: shelf_talker`, `2: price_tag`, `3: wobbler_strip`) with per-class NMS (`IoU = 0.50`) and shelf-plank Y-band snapping. | Verifies rail-tag geometry (`aspect 1.1–6.5`) and filters printed pack graphics without requiring a separate SKU classification call. | **90.0% Annotated-Zone F2** (`93.4%` Recall, `98.2%` mAP@50) in **3.05 s/shelf** (`$0.0022`), plus **+84 real unlabelled rail tags** (`1.3x` human recall). |
| **Pillar 3 · Bay Boundary Gate & Clutter Filter** (`gtmerch_golden`, `#1017-152407`) | Evaluates local Laplacian texture variance (`>= 4.2`) to separate 3D physical packs from flat 2D printed cardboard headers. | Runs full-frame + `2x2` overlapping tiles (`20%` overlap) with decoupled `Class 0` bay vs. `Class 1` product NMS (`98.8%` Bay F2). | Applies RANSAC shelf-plank support (`±6.5%` bay height) and `IoA >= 0.50` bay containment (`±8%` width margin; bypassed for `HAN` / `FULLWALL`). | **-61.0% counter clutter pruned in `< 1 ms` CPU**, **0 printed wing false positives**, and **77.3% full-shelf box F2 in 3.7 s** (`₹0` VLM cost). |

## **How We Evaluated & Results**

Every pillar was evaluated on store-disjoint Golden test splits (partitioned by `outlet_id` / `outlet_code` and MD5-deduplicated in [Data Analysis and Enrichment](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.0)). Detection is scored with **F2** (weighting recall twice as heavily as precision because a missed facing cannot be recovered downstream).

Because raw `gtmerch-marketshare` boxes carried no SKU names, clear in-bay HUL crops were linked to catalog SKUs during enrichment while ambiguous rival packs stayed `"Others"`. To avoid penalizing the model for naming a real product inside an `"Others"` box, we report both **strict Exact-SKU accuracy on named boxes** and **average brand/category/pack accuracy**, plus the count of **`"Others"` boxes upgraded to specific SKUs**.

### **Pillar 1: Share of Shelf (SoS) and In-Bay Layout**

| **Evaluation Slice** | **Existing Baseline** | **New Pipeline (Test)** | **Change & Operational Notes** |
| --- | --- | --- | --- |
| **Strip Share-of-Shelf (SoS) error (`gt_hul_sessions`, 100 strips)** | Count-only (no row or strip grouping) | **2.38% SoS error** (96.9% HUL ownership) | Evaluates 800 labelled crops across 100 store strips with **100% bay-block purity** and **72.5% 5-attribute SKU accuracy**. |
| **Strict Exact-SKU on named in-bay boxes (`gtmerch_golden`, 25 test shelves)** | 61.4% classifier / 53.2% E2E strict basepack (200 bays) | **64.3%** (68.2% to 69.8% in-bay E2E F2) | Matches the strict 5-attribute SKU (including sister pack size); uncropped single-call baseline scored 44.5% (51.2% VLM[^c1]). |
| **Brand, category, and pack accuracy (in-bay)** | Not reported separately | **98.2% avg.** (25 shelves) / **94.8%** (3,420 crops) | High brand/category/pack accuracy shows remaining errors are almost entirely between sibling variants and pack sizes. |
| **`"Others"` boxes upgraded to specific SKUs** | 50 catch-all classes | **574 boxes named** (63.0% of `"Others"`) | Catalog lookup names real products that raw annotations left as `"Others"`. |
| **Row alignment, brand-block purity, and OOS gaps** | Not supported (58.0% flat count F1) | **94.4% to 95.1% purity · 95.2% row F1 · 92.6% OOS F1** | Rule-based row reconstruction and `AWP_Code` template checking in `< 5 ms` CPU. |

### **Pillar 2: Promotions and Price Tags**

| **Evaluation Slice** | **Existing Baseline** | **New Pipeline (Test)** | **Change & Operational Notes** |
| --- | --- | --- | --- |
| **Promo pack: brand + category + pack all correct (`hul_promos`, 1,000 crops)** | 75.0% fixed 240-class ViT | **84.0%** (93.7% Brand · 91.3% Category · 88.3% Pack) | **+9.0 pp** over the fixed ViT baseline (`73.0%` exact 5-attribute SKU) with zero retraining for new campaigns. |
| **Promo offer header verification (`"50% Free"` & `"Price Off"`)** | 23.5% on `"50% Free"`; 83.5% on `"Price Off"` | **94.1% on `"50% Free"` · 95.6% on `"Price Off"`** | **40.3%** across all 9 raw `promo_type` strings because human annotations spell the same offer in multiple ways (`"Rs 10 Off"`, `"_Rs_10"`, `"10/- Off"`). |
| **3-class shelf-talker detection (`hul_talkers_golden`, 50 test shelves)** | 64.2% mAP@50 (49.2% Talker F2) | **90.0% annotated-zone F2** (84.9% full-frame F2; 98.2% mAP@50) | **90.5% precision, 89.9% recall**, and **+84 real unlabelled rail tags** (`1.3x` coverage) in `3.05 s` per photo. |

### **Pillar 3: Bay Display and Clutter Filtering**

| **Evaluation Slice** | **Existing Baseline** | **New Pipeline (Test)** | **Change & Operational Notes** |
| --- | --- | --- | --- |
| **Display bay detection (`Class 0` Bay F2)** | 51.1% Bay F2 (57.1% recall) | **98.8% Bay F2** (99.6% recall) | Separating `Class 0` and `Class 1` NMS prevents bay boxes from erasing the products inside them. |
| **Out-of-bay counter clutter removed** | 0% filtered (61.0% counter clutter sent to classifier) | **61.0% clutter pruned** (`< 1 ms` CPU) | Plank support and bay boundary gating drop surrounding Kirana counter items and cut downstream API calls by 80%. |
| **Fast box-only bay detection (100 `gtmerch_golden` test shelves)** | 86.8% `Class 1` F2 (scored on 200 pre-cropped bays) | **77.3% bay-scoped F2** in `3.7 s` (`94.7%` pre-cut block purity) | Scored on 100 full store shelves (vs. 200 pre-cropped bays) and actively filters out printed wing and header pack artwork. |

### **Residual Forced-Choice Benchmark Gaps & Implemented Production Fixes Across the Three Pillars**

In standard benchmark scoring (`shelf-bench`), the evaluator forces the model to emit a Top-1 guess on every box (`0%` abstention allowed), whereas in production (`v0.2`, already live in `main`), our calibration and abstention guardrails automatically intercept physically unreadable crops and annotation noise:

| **Merchandising Pillar & Benchmark Slice** | **Residual Benchmark Gap (Forced Top-1 Scoring)** | **Physical Store-Image or Annotation Root Cause** | **Implemented Pipeline Fix & Production Behavior (Live in v0.2)** |
| --- | --- | --- | --- |
| **Pillar 1 (In-Bay SoS & Exact SKU: `98.2%` Brand/Cat/Pack vs. `64.3%` Exact-SKU)** | Same-brand sibling pack-size & sub-variant swaps (`33.9%` of in-bay boxes get Brand, Category, and Pack right when forced to guess a variant, but swap `50g` vs. `100g`). | Small in-bay crop resolution (`32–48 px` width on full-bay Kirana photos) blurs 4–6 pt gram/ml text below 2 pixels per character on curved jars and sachets. | **Already Live in `v0.2`:** `AWP_Code` compliance and Brand-Block Purity (`94.4%–95.1%`) score at the Brand/Basepack-family level (`98.2%` accurate), while [`apply_hierarchical_abstention_gate`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/approaches/market_share/retrieval/jev_laya_hybrid.py#L205-L246) keeps `Brand + Pack` and sets `abstain_recommended = True` (`UNVERIFIED_LOW_CONFIDENCE`) on unreadable variants. |
| **Pillar 1 (OOS Gap Detection: `92.6%` F1)** | `7.4%` residual OOS precision/recall gap between recessed dark products and true empty shelf voids. | Deep back-of-shelf shadows in unlit Kirana cabinets resemble empty gaps, while angled side-wall perspective foreshortens true gaps near bay edges. | **Already Live in `v0.2`:** Stage 1 `L*`-channel CLAHE shadow recovery paired with row-adaptive median pack width (`gap >= 1.35x–1.50x` local row median width) in `bay_planogram_verifier.py`. |
| **Pillar 2 (Promotions & Talkers: `94.1%–95.6%` Key Offers vs. `40.3%` 9-Class Raw String; `90.0%` Talker F2)** | String-label fragmentation across 9 raw `promo_type` strings (`"Rs 10 Off"`, `"_Rs_10"`, `"10/- Off"`) and `9.5%` apparent talker false positives on unannotated shelf rails. | Human ground truth uses multiple spelling variants for identical price-off promotions and leaves ~23% of real bottom-row price tags unlabelled. | **Already Live in `v0.2`:** `promo_ocr_zoom_verifier.py` maps promo strings to canonical campaign families (`94.1%` `"50% Free"`, `95.6%` `"Price Off"`), and the talker evaluator scores precision/recall inside annotated shelf bands (`90.02%` F2, `98.2%` mAP@50). |
| **Pillar 3 (Bay Display & Clutter Filtering: `77.3%` Full-Shelf Box F2, `61.0%` Clutter Pruned)** | Borderline hanging sachets clipped near bay frame edges (`IoA` near `0.45–0.50`) vs. printed 2D product illustrations on cardboard side wings. | Kirana merchants frequently clip sachet strips onto the outer metal rim of a bay right next to printed 2D pack graphics on the bay's cardboard header. | **Already Live in `v0.2`:** Stage 3 expands `Class 0` bay bounds by `±8%` width, enforces local Laplacian edge variance (`>= 4.2`) to reject flat printed wing artwork, and bypasses `Class 0` cropping automatically on frameless `HAN` strips. |

### **5. Production Confidence Contract & Client Cutover Readiness (What We Have vs. What Is Needed)**

#### **A. Default Confidence & Abstention Contract (Unified $\tau = 0.78$ — Emitted on 100% of Predictions)**
Because Stage-A cosine similarity sits in a uniform `0.922–0.941` band across all 10 business categories and remaining errors are driven by **3D container geometry** (`Jar` / `Bar` curvature) and **same-brand sibling margin ($\Delta_{\text{sib}}$)** rather than category, our pipeline uses a **single unified threshold ($\tau = 0.78$, `use_category_thresholds = False` by default)**—which outperforms separate per-category thresholds (`88.0%` vs. `87.8%` Variant precision) without maintaining 10–13 separate category cutoffs. Every prediction emits these calibrated fields by default while still returning the Top-1 label:

| **Output Field (Emitted by Default)** | **Condition / Range** | **Production Meaning & Downstream Action (`DYNAMIC_TASK_API` / Analytics)** |
| --- | --- | --- |
| **`confidence["brand" \| "category" \| "packaging_type"]`** | `[0.02, 0.985]` (`0.0%` saturation at `1.0`) | Used for **Share of Shelf (`SoS`)**, **Brand-Block Purity**, and **Bay (`AWP_Code`) Presence** whenever `>= 0.78`. |
| **`confidence["base_variant" \| "product"]`** | `[0.02, 0.985]` (discounted by sibling margin $\Delta_{\text{sib}}$) | Reflects fine-print variant readability. Discounted when two sister variants of the same brand are nearly tied ($\Delta_{\text{sib}} < 0.025$). |
| **`category_threshold`** | **`0.78` (Unified Global Default)** (`use_category_thresholds = False`) | Single self-calibrating cutoff across all categories (optional per-category override `0.74–0.82` available via config, proven unnecessary by ablation). |
| **`variant_abstain_recommended`** | `True` when `confidence["base_variant"] < 0.78` | **Keep Brand + Category + Pack** for Share of Shelf and Bay Purity (`98.8%–99.3%` Brand precision), but **suppress variant-level cross-sell or strict basepack failure** (`88.0%–90.7%` Variant precision). |
| **`abstain_recommended`** | `True` when Stage-B rejects (`choice == 0`), `joint_confidence < 0.38`, or `energy < 0.52` | **Full crop abstention (`UNVERIFIED_LOW_CONFIDENCE`):** Crop is severely blurred, glared, or out-of-catalog; exclude from SKU compliance and flag for store re-shoot if bay coverage is insufficient. |

#### **B. Like-for-Like Legacy Model Comparison Readiness Across All 3 Pillars**

| **Merchandising Pillar / Task** | **Legacy Weights Available in Repo / GCS?** | **Current Comparison Mode** | **Client Item Needed for Live Image-by-Image Legacy Replay** |
| --- | --- | --- | --- |
| **Pillar 1A & Pillar 3: Branded Bay (`Class 0`), In-Bay Box (`Class 1`), & OOS Void** | **YES** (`gs://sos-datasets/legacy-models/gtmerch/gtmerch_asset_sku_intermediary_17092026.onnx`) | **100% Live Weight-to-Weight Execution** on identical `gtmerch_golden` & `gt_hul_sessions` store photos (`51.1%` Legacy vs. **`98.8%`** Proposed Bay F2; `86.8%` vs. **`95.2%`** pre-cropped Box F2). | **None — Complete.** |
| **Pillar 1B: In-Bay GT Basepack / SKU Classification** | **YES** (`17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5` + 9-model `.h5` suite) | **100% Live Weight-to-Weight Execution** on identical `gtmerch_golden` bays (`53.2%` Legacy E2E / `61.4%` Classifier vs. **`98.2%`** Brand/Cat/Pack & **`64.3%`** Strict 5-Attr SKU). | **None for classifier weights** (see Table C below for the outer `AWP_Sim_Score` payout script). |
| **Pillar 2A: Promo Pack Classification (`hul_promos`)** | **NO** (Historical evaluation table in `docs/Customer_Inputs.md`) | **Metric-to-Metric Benchmark Comparison** against Unilever's 240-class ViT baseline (`75.0%` 3-Attr, `23.5%` `"50% Free"` vs. **`96.2%`** 3-Attr, **`92.9%`** `"50% Free"`). | **Legacy 240-class Promo ViT checkpoint + class index JSON** (if live per-crop legacy drawer replay is desired). |
| **Pillar 2B: Shelf-Rail Talker & Price-Tag Detection (`hul_talkers_golden`)** | **NO** (Historical evaluation metric in `docs/Customer_Inputs.md`) | **Metric-to-Metric Benchmark Comparison** against Unilever's Talker detector (`64.2%` mAP@50 / `49.2%` F2 vs. **`98.2%`** mAP@50 / **`90.0%`** Talker F2). | **Legacy Talker YOLO `.pt` / `.onnx` checkpoint** (if live legacy bounding-box overlay is desired). |

#### **C. End-to-End Production Cutover Checklist (What We Have vs. 4 Client Artifacts Needed)**

| **Production Cutover Component** | **What Is Already Built & Verified in Repo (`v0.2`)** | **What Is Still Needed from Client / Unilever Team for 100% Live Cutover** |
| --- | --- | --- |
| **1. Contractual `AWP_Sim_Score`, `Envision_Sim_Score`, & Rupee `Earning` Formula** | We have the outer Azure Logic App spec (`gt-merchandising-endpoint-prod.json`) and our spatial-graph planogram scorer (`bay_planogram_verifier.py`). | **Inside-container Python scoring script (`score.py`)** from `gt-merchandising-endpoint-prod.eastus2.inference.ml.azure.com/score` so our rupee `Earning` slab calculation matches legacy contractual payouts 1-for-1. |
| **2. Authoritative 63-`AWP_Code` × `Outlet_PLG` × `Tier` Planogram Rulebook** | Reconstructed 63 `AWP_Code` definitions (`data/awp_code_templates`) from dataset labels and class maps. | **Official master rule table (CSV/JSON)** mapping every `(AWP_Code, Outlet_PLG, Channel, Tier, Asset_Type)` to required basepack facings, shelf-row order, and minimum SoS thresholds. |
| **3. Distributor (`RSCode`) Cross-Sell Assortment Eligibility Table** | Pipeline detects missing mandatory sister basepacks and gates recommendations using `variant_abstain_recommended`. | **`Outlet_PLG × RSCode` active assortment lookup table** so missing-SKU alerts only recommend basepacks actively stocked by that store's distributor. |
| **4. Shelf-Rail Price-Tag MRP Ground Truth (Optional — Pillar 2B)** | 3-class detector (`promo_talker`, `printed_price_tag`, `handwritten_price_tag`) verified at **`90.0%` F2** on `hul_talkers_golden`. | If Unilever requires **written rupee price vs. official MRP verification**, provide `~200–300` price-tag crops annotated with ground-truth MRP/price strings. |

## **Appendix**

### **Key Parameters**

| **Stage** | **Component** | **Key Parameters** |
| --- | --- | --- |
| **Stage 1** | Shades-of-Gray white balance & `L*` CLAHE | Minkowski norm `p = 6` on non-clipped pixels `(12, 245)` at `512 px`; channel gains clamped to `[0.72, 1.38]`; CLAHE `8x8` tiles, clip limit `2.0` on `L*` only. |
| **Stage 2** | Bay, product, and talker detectors | Bay NMS IoU `0.58`; up to `3` bay zoom crops (`3%` margin); `2x2` tiling (`12%` overlap) when `>= 45` boxes or `>= 24` small boxes; CPU ONNX `960 px` effective resolution; talker per-class NMS IoU `0.50`. |
| **Stage 3** | Artwork, shelf-plank, and bay boundary filter | Printed wing texture threshold `< 4.2`; shelf-plank alignment band `±6.5%` of bay height; bay gate fires at `>= 22%` bay area (`±8%` margin, `IoA >= 0.45–0.50`) and skips `FULLWALL` and `HAN`. |
| **Stage 4** | Catalog embedding lookup & promo band zoom | Top-3 brand window `0.06`; `K = 5` candidates; direct accept without LLM when margin and confidence gates pass (`68%` of groups); promo header/footer zoom bands `25%–28%` top and `20%–30%` bottom. |
| **Stage 5** | Row clustering, OOS voids, and payout gate | Row vertical overlap `>= 0.40–0.45`; OOS gap threshold `>= 1.35x–1.50x` row median pack width; `AWP_Code` min HUL share `0.90` (`0.70` for `HAN`), min purity `0.80`; timeout quarantine at `180 s`. |

### **FAQ**

1. **Why are only test-split results reported?** The `gtmerch_golden` validation split has only 15 shelves (384 boxes) and was used to tune pipeline thresholds (`~95%` exact SKU on validation vs. `60%–64.3%` on test), so reporting validation scores would overstate accuracy.
2. **Why is average brand/category/pack accuracy 98.2% while strict Exact-SKU is 64.3%?** Most remaining errors are between sibling variants and pack sizes (`50g` vs. `100g`) of the right brand, and the `98.2%` figure averages three per-attribute scores rather than requiring all five attributes to match simultaneously.
3. **Why does fast box-only detection score 77.3% F2 vs. 86.8% for the existing YOLO `Class 1` model?** The two numbers come from different evaluation setups: the existing YOLO score was measured on 200 pre-cropped bays, whereas `77.3%` is measured on 100 full store shelves and intentionally drops printed pack illustrations on cardboard headers and side wings.
4. **How are frameless hanging strips (`HAN`) handled?** `HAN` strips have no outer cabinet frame, so Stage 2 excludes them from `Class 0`, Stage 3 skips bay cropping so neighbouring strips stay intact, and Stage 5 applies the `HAN` template rule (`>= 0.70` HUL share).
5. **How do we onboard a new `AWP_Code` planogram or festive promo pack?** Adding a new planogram requires only a template rule entry in the configuration index with zero model weight changes. Adding a new promo pack requires 1 to 5 reference crops and a catalog row, refreshing Cloud Run workers in under 60 seconds.
6. **How do single-shelf experimental run costs compare with production sizing?** Unbatched experimental runs that send multiple full-frame and multi-crop VLM passes take `52–53 s` (`₹1.79–₹1.88` per photo). In production ([Cost Breakdown & Sizing](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.rnj5urhvxc2j)), using the in-process CPU ONNX detector, grouping adjacent identical facings, and sending only the 32% lookalike ties to `gemini-3.5-flash-lite` reduces latency to `2.69 s` P50 at `$0.00160` (`₹0.15`) per shelf.
7. **Who owns the code and data assets?** Reference pipeline code is delivered under the Apache-2.0 license (see [Model Whitelisting Report](https://docs.google.com/document/d/1rzwQ5RTFXgOUjpkvybYoNdFvjWbtrx74vE79S7p4o_A/edit?tab=t.m11c1bw64077)), while Unilever retains exclusive ownership of its 63 `AWP_Code` templates, benchmark datasets, and brand catalog assets.


---
[^c1]: Anil Sener: @rgavigan@google.com not sure it is was a VLM
    - Jigyasu Juneja: Clarified in both the Benchmark Summary table and prose: the existing Unilever production baseline is the 640px YOLOv8n + 209-class classifier chain (51.1% Bay F2 / 44.5% in-bay crop SKU accuracy), whereas 51.2% was an unscoped single-call VLM exploratory baseline tested before introducing Stage 1.5 geometric bay gating.
