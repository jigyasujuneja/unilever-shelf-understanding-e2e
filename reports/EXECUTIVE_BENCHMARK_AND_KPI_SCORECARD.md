# Unilever Perfect Store AI: End-to-End Gondola Intelligence & Neural Routing Control Plane

**Owner:** Jigyasu Juneja (`jjuneja@google.com`)  
**Repository:** `https://github.com/jigyasujuneja/unilever-shelf-understanding-e2e`  
**Core Documentation:**
* [Google-Style Technical Design Document (`TDD_GOOGLE_DESIGN_DOC.md`)](TDD_GOOGLE_DESIGN_DOC.md)
* [Team Demo Presenter Playbook & Tab-by-Tab Runbook (`DEMO_PLAYBOOK.md`)](DEMO_PLAYBOOK.md)
* [System Architecture & 10-Edge-Case Taxonomy (`specs/DESIGN.md`)](specs/DESIGN.md)
* [Full Benchmark & KPI Scorecard (`reports/EXECUTIVE_BENCHMARK_AND_KPI_SCORECARD.md`)](reports/EXECUTIVE_BENCHMARK_AND_KPI_SCORECARD.md)

## 1. Executive TL;DR and Production Architecture Recommendation

Hindustan Unilever Limited (HUL) processes **500,000 retail shelf photographs per day** across Modern Trade hypermarkets and General Trade outlets under two strict operational SLAs and FinOps invariants (`SPEC-003`):

* **Use Case 1 (`Marketshare` Workflow: `5 to 7` overlapping aisle images per request):** Must return On-Shelf Availability (`OSA`), closed-catalog HUL SKU identification, open-set Non-HUL competitor identification, and a 4-factor replenishment order within **`<= 30.0s` E2E** (`<= 20.0s` server `P95`).
* **Use Case 2 (`Merchandizing` Workflow: `1` bay image per request):** Must return planogram sequence compliance, eye-level golden-zone share, brand-block contiguity, and promotional header (`Toker`) audit within **`<= 10.0s` E2E**.
* **FinOps & Scale Ceiling:** Blended unit cost **`<= INR 0.22` (`$0.00262 USD`)** per image across **`525` parallel worker threads** with **`>= 95.0%` weighted `F2`**.

### Final Recommendation: `Track D3` (`Unified Tiered Hybrid + Stage 4.5 + /v1/systemone dJev`)

1. **Why Monolithic 1-Pass VLMs (`Track B1`) Fail:** Passing a full `4000x3000` shelf photo into a single VLM prompt downscales the image to `2048px` (turning `8px` shade labels into `1.3px` blur) and emits `3,009` sequential output tokens. Recall collapses to **`24.9%` (`15.1% F2`)** on 221-SKU bays, latency hits **`31.68s`** (violating the `30s` SLA), and unit cost reaches **`INR 0.520/img`** (`2.4x` over the FinOps ceiling).
2. **Why Legacy 8-Model Supervised CNN Cascades (`Track A`) Fail:** Requires `3 to 4 weeks` of retraining per artwork update, is blind to unseen Non-HUL competitors (`UNKNOWN_OTHER`), and suffers catastrophic **Sister-Shade Collapse** (`Lakme 9to5 CC Almond = 3.6% F2`, `CC Honey = 0.0% F2`, collapsing into `314` false-positive `CC Bronze` predictions).
3. **How Our Recommended `Track D3` Architecture Wins Across Every Invariant:**
   * **Fast Visual Memory (`89%` of Shelf Volume at `0` LLM Tokens):** `Stage 3 RT-DETR-L` (`38.4 ms`) + `Stage 3.5 ORB/RANSAC Panorama Seam Deduplicator` (`24.5 ms`) + `Stage 4 I-JEPA 512-D De-Glare + DINOv2-reg4 + Vertex AI ScaNN` (`0.8 ms`) resolve `89%` of unobstructed core SKUs (`sim >= 0.82` and same-brand cosine margin `>= 0.045`) at **`98.6%` accuracy and `INR 0.004/img`**.
   * **Sister-Shade & Glare Disambiguation (`11%` Hard Tail via `Stage 4.5` + `/v1/systemone dJev`):** Ambiguous same-brand crops (`Margin < 0.045`) trigger **Stage 4.5** (`3x Sub-ROI Zoom [0.62H:0.88H]` + glare-damped `CIELAB Delta-E00` + `Menon Post-Hoc Logit Adjustment` + `Cap-Down vs Cap-Up Silhouette Taper`) and **Stage 5 `dJev` (`/v1/systemone`)**, which packs the crop into an **`8x8 = 64` Visual Token Canvas** and denoises all 64 slots in **3 parallel Jacobi steps (`8.9 ms`)** with **`vllm#58216` pinning `88.5%` (`56.6 / 64`) of tokens** to valid HUL ERP Base Packs (`0.0%` hallucination). This lifts the 44-variant failure cohort from **`38.4%` to `94.2% F2`** (`96.42%` overall macro `F2`).
   * **Surgical Open-Set Competitor Denominator (`sim < 0.82`):** Unseen competitor bottles (`Pantene`, `Head & Shoulders`, `Ariel`) route to the Open-Set Attribute Classifier (`Track F LoRA / Gemini Flash-Lite`) to emit `NON-HUL-<CAT>-<BRAND>-<SIZE>` (`95.8%` accuracy), providing the exact category denominator for Linear and Area Share of Shelf (`SOS %`).

## 2. Performance Breakdown 1: By Model Architecture (`All 8 Neural Tracks`)

Evaluated across the 25-image golden shelf benchmark (`3,649` human-labeled boxes) and HUL's 245-variant Skin and Personal Care production scorecard (`17,118` ground-truth facings).

| Track ID | Architecture Pipeline | Top-1 Acc | Recall | Weighted `F2` | Sister-Shade Tail `F2` (`44` SKUs) | 1-Img `P95` (`ms`) | 6-Img `P95` (`ms`) | Cost / Img (`INR`) | Annual Savings vs `Track B1` | `SPEC-003` Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Track A** | Legacy 8-Model Supervised CNN/ViT Cascade | `88.4%` | `78.1%` | `79.2%` | `18.4%` | `185 ms` | `920 ms` | `INR 0.185` | `$362,000` | **FAIL** (`F2 < 95%`, 3-wk retrain) |
| **Track B1** | 1-Pass Full-Shelf `Gemini 2.5 Flash` VLM | `50.1%` | `24.9%` | `15.1%` | `12.0%` | `5,840 ms` | `31,680 ms` | `INR 0.520` | Baseline (`$0`) | **FAIL** (`Cost > 0.22`, `Latency > 20s`) |
| **Track B2** | 2-Stage High-Res Crop + `Gemini 2.5 Flash-Lite` | `94.2%` | `91.8%` | `92.1%` | `78.5%` | `2,650 ms` | `14,800 ms` | `INR 0.390` | `$281,000` | **FAIL** (`Cost > 0.22`, `4.2%` ERP hall.) |
| **Track C** | `RT-DETR` + `DINOv2-reg4` + `Vertex AI ScaNN` | `93.1%` | `89.5%` | `88.4%` | `38.4%` | `142 ms` | `680 ms` | `INR 0.012` | `$1,098,000` | **PARTIAL** (Weak on `8px` sister shades) |
| **Track D1** | `ScaNN` + Deterministic `Jev` Rule State Machine | `94.0%` | `90.2%` | `89.8%` | `52.1%` | `158 ms` | `740 ms` | `INR 0.014` | `$1,094,000` | **PARTIAL** (Fails on curved/glared `8px` text) |
| **Track E** | `I-JEPA` `512-D` World Model + `DINOv2` + `ScaNN` | `94.8%` | `93.1%` | `93.4%` | `68.2%` | `168 ms` | `790 ms` | `INR 0.016` | `$1,090,000` | **PARTIAL** (Solves glare; needs `dJev` on shades) |
| **Track F** | `PaliGemma-2-3B-LoRA` Specialist (`L4 GPU`) | `95.4%` | `94.0%` | `94.2%` | `86.4%` | `295 ms` | `1,490 ms` | `INR 0.192` | `$709,000` | **PASS** (Adopted for Open-Set `<0.82` head) |
| **Track D2 / D3** | **Production Winner (`RT-DETR` + `ORB Dedup` + `I-JEPA` + `ScaNN` + `Stage 4.5` + `/v1/systemone dJev`)** | **`96.8%`** | **`96.2%`** | **`96.4%`** | **`94.2%`** | **`218 ms`** | **`948 ms`** | **`INR 0.0199` (1-img) / `INR 0.148` (blended)** | **`$642,000 / yr`** | **PASSES ALL INVARIANTS** |

## 3. Performance Breakdown 2: By HUL Product Dimension (`All 7 Dimensions: Classify vs. Derive`)

Shows how each of HUL's 7 taxonomy attributes is resolved across architectures and why `Track D3` orchestrates a dedicated specialist solver per dimension.

| Dimension # & Name | Pipeline Stage | Legacy ViT (`Track A`) | 1-Pass VLM (`Track B1`) | `ScaNN` Only (`Track C`) | `Track D3` Production Specialist Solver | `Track D3` Accuracy / `F2` | `Track D3` Latency & Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Category**<br>(`Personal Care`, `Home Care`, `Foods`) | `Stage 4: Classify` | `94.2%` (Closed-set only) | `98.5%` | `99.4%` | `DINOv2-reg4` + `ScaNN` Coarse Category Centroids (`0` tokens) | **`99.6%`** | `0.4 ms` / `INR 0.000` |
| **2. Subcategory**<br>(`Skin Cleansing`, `Hair Conditioning`, `Oral`) | `Stage 4: Classify` | `89.8%` | `91.4%` | `98.2%` | `DINOv2-reg4` + Hierarchical `ScaNN` Sub-Tree (`0` tokens) | **`99.1%`** | `0.4 ms` / `INR 0.000` |
| **3. Brand**<br>(`HUL` Closed-Catalog vs `Non-HUL` Competitor) | `Stage 4: Classify` | `88.5%` (Blind to Non-HUL) | `95.1%` | `95.8%` | `I-JEPA 512-D` Latent De-Glare + Dual-Threshold `ScaNN` Router (`tau = 0.82`) | **`98.7%`** | `1.1 ms` / `INR 0.000` |
| **4. Variant (Fine-Grained)**<br>(`Lakme CC Almond` vs `Honey` vs `Bronze`, `SPF`) | `Stage 4 + Stage 4.5` | `68.1%` (`0%` on `CC Honey`) | `42.0%` | `78.9%` (`38.4%` on tail) | **Stage 4.5 `3x Sub-ROI Zoom [0.62H:0.88H]` + `CIELAB Delta-E00` + `Menon Logit` + `/v1/systemone`** | **`96.4% F2`** (`94.2%` on sister shades) | `+1.8 ms` (on `11%`) / `INR 0.004` |
| **5. Packaging Type**<br>(`Bottle`, `Cap-Down Tube`, `Jar`, `Carton`, `Sachet`) | `Stage 4 + Stage 4.5` | `82.4%` (`66.3%` on tubes) | `84.0%` | `91.2%` | **Stage 4.5 Silhouette Taper Ratio ($W_{\text{top }15\%} / W_{\text{bottom }15\%} > 1.22$)** | **`98.9%`** | `0.2 ms` / `INR 0.000` |
| **6. Pack Type**<br>(`Single Unit` vs `Multipack / Banded / Hanging Strip`) | `Stage 3 + Stage 5 Derive` | `74.0%` (Drops `50%` sachets) | `71.2%` | `92.5%` | **Stage 3 Vertical-Chain `DIoU-NMS` + Stage 5 Horizontal Periodicity Shrink-Band Detector** | **`98.2%`** | `0.4 ms` / `INR 0.000` |
| **7. Size + `ERP Base Pack Code`**<br>(`180ml` vs `340ml` vs `650ml` -> `BP-HUL-...`) | `Stage 5: Derive` | `76.4%` (Perspective errors) | `64.8%` (`4.2%` invalid ERP) | `88.6%` | **Stage 5 Local Shelf-Rail Gap Normalization ($H_{\text{box}} / H_{\text{shelf\_gap}}$) + `/v1/systemone` (`vllm#58216`)** | **`97.4%` (`0.0%` ERP Hallucination)** | `8.9 ms` / `INR 0.008` |

## 4. Performance Breakdown 3: By Commercial Use Case (`Marketshare` vs. `Merchandizing`)

| Use Case & Operational Scope | Request Payload | Target SLA | Measured `Track D3` Server `P95` | Measured `Track D3` Total E2E (incl. 4G Upload) | Key Sub-Analyses & Outputs Delivered | Business Uplift / Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Use Case 1: `Marketshare` Workflow**<br>(Full 24ft Gondola Aisle Walk) | `5 to 7` overlapping 4K photos (`~22%` horizontal boundary overlap) | **`<= 30.0 s`**<br>(`<= 20.0s` Server `P95`) | **`948 ms`** (`21x` faster than budget) | **`1.35 s`** (`22x` faster than `30s` SLA) | 1. **Cross-Frame Seam Dedup:** `1,104` raw boxes $\to$ `902` unique facings (`202` duplicates suppressed).<br>2. **OSA & Red-Line Voids:** `97.8%` recall on front-row stockouts.<br>3. **HUL (`Closed-Set`) + Non-HUL (`Open-Set`) SOS %:** `96.8%` / `95.8%` attribution.<br>4. **Stage 6 4-Factor Recommend Cart:** Ranked distributor replenishment order. | **`+INR 10,260 to +INR 14,200 / week`** incremental revenue per store |
| **Use Case 2: `Merchandizing` Workflow**<br>(Single Bay Planogram & Promo Audit) | `1` 4K bay photo + Store Planogram ID | **`<= 10.0 s`** | **`218 ms`** (`45x` faster than `10s` SLA) | **`480 ms`** | 1. **Planogram Sequence Compliance:** Levenshtein row alignment (`98.4%`).<br>2. **Eye-Level Golden Zone (`1.2m-1.5m`):** Area & Linear share.<br>3. **Brand-Block Purity:** Detects competitor intrusions inside HUL blocks.<br>4. **Promotional `Toker` Banner OCR:** `97.6% F1` via cropped header check. | **Instant in-aisle rep coaching** before leaving the bay |

## 5. Performance Breakdown 4: By Pipeline Task & Real HUL 245-Variant Cohorts (`17,118` Facings)

Breaks down `Track D3` performance across every functional computer-vision task and across HUL's real `245`-variant Skin and Personal Care scorecard (`17,118` ground-truth facings).

| Task / Variant Cohort | Ground-Truth Volume | Baseline Pipeline Score | `Track D3` Production Score | Absolute Gain | Primary Engineering Mechanism Responsible |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1: Dense Shelf Object Localization (`Stage 3`)** | `3,649` boxes (`25` 4K bays) | `89.2% mAP@50` | **`97.8% mAP@50`** | **`+8.6%`** | `RT-DETR-L FP16` + Vertical-Chain `DIoU-NMS` (preserves `60%` overlapping hanging sachets) + Depth-Ghost Luminance Filter |
| **Task 2: 6-Frame Panorama Seam Deduplication (`Stage 3.5`)** | `1,104` raw boxes (`202` seam overlaps) | `0%` dedup (`+22.4%` overcount error) | **`99.1%` Seam Dedup Precision (`902` unique)** | **`-22.4%` overcount error** | Pairwise `ORB` keypoint matching in `25%` boundary strip + `RANSAC` homography projection ($H_{t \to t+1}$) |
| **Task 3: HUL Core Anchor Variants (`201 / 245` SKUs with `>= 50` GT facings)** | `15,490` facings (`e.g. Vaseline Deep Moisture 99.6%`, `Pond's Super Light Gel 97.6%`) | `94.1% F2` | **`97.9% F2`** | **`+3.8% F2`** | `DINOv2-reg4` + `Vertex AI ScaNN` (`0.8 ms`, `0` LLM tokens) + `I-JEPA` `512-D` specular glare masking |
| **Task 4: Sister-Shade Colliding Variants (`Lakme 9to5 CC Almond / Honey / Beige / Bronze`, `SPF 24/30/50`)** | `892` facings (`24` sister-shade variants) | **`24.6% F2`** (`CC Honey = 0.0%`, `CC Almond = 3.6%`) | **`94.2% F2`** | **`+69.6% F2`** | **Stage 4.5 `3x Sub-ROI Zoom [0.62H:0.88H]` + `CIELAB Delta-E00` + `Menon Logit Adjustment` + `/v1/systemone` (`vllm#58216`)** |
| **Task 5: Low-Shot / Day-0 Clinical Launches (`Novology Serums`, `Simple Smoothing Gel`, `Hydra Glow`)** | `736` facings (`20` low-shot variants with `11 to 26` GT facings) | **`19.8% F2`** | **`92.4% F2`** | **`+72.6% F2`** | Zero-retraining `ScaNN` studio-spin prototypes + Menon prior penalty ($-\gamma \log \pi_y$) + `F2`-optimal threshold ($\tau = 0.28$) |
| **Task 6: Inverted Form-Factor Disambiguation (`Dove Conditioner Tube` vs. `Shampoo Bottle`)** | `412` facings | `66.3% F2` | **`95.6% F2`** | **`+29.3% F2`** | Silhouette top-15% vs. bottom-15% contour taper ratio ($W_{\text{top }15\%} / W_{\text{bottom }15\%} > 1.22$) |
| **Task 7: All `245` HUL Skin & Personal Care Variants Combined** | **`17,118` GT Facings** | **`86.80%` Macro `F2`** | **`96.42%` Macro `F2`** | **`+9.62% F2`** | Unified `Track D3` 2-tier routing (`89% ScaNN` + `11% Stage 4.5 & dJev /v1/systemone`) |

## 6. Performance Breakdown 5: Master KPI Scorecard (`SPEC-003` Invariants & 8 Modern Trade Gondola KPIs)

| KPI Category | Metric Name | Mathematical Definition / Formula | `SPEC-003` Target Threshold | Measured `Track D3` Value | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FinOps & Scale** | **Blended Cost per Image** | `(GPU_Compute + ScaNN + SystemOne_Tokens + GCS) / N_imgs` (`1 USD = 84 INR`) | **`<= INR 0.220`** | **`INR 0.0199` (`1-img`) / `INR 0.148` (`blended`)** | **PASS (`33%–91%` under cap)** |
| **FinOps & Scale** | **Concurrent Worker Capacity** | Sustained parallel load without HTTP `429` or `503` errors | **`>= 525` Workers** | **`525` Workers (`0` HTTP 429s)** | **PASS** |
| **Latency SLA** | **Server `P95` Latency (`Marketshare` 6-Img / `Merchandizing` 1-Img)** | 95th percentile server execution time (`Capture` to `Respond`) | **`<= 20,000 ms` / `<= 5,000 ms`** | **`948 ms` / `218 ms`** | **PASS (`21x` faster)** |
| **Model Quality** | **Top-1 SKU Identification Accuracy** | `Correct_7Dim_BasePack_Preds / Total_GT_Facings` | **`>= 92.0%`** | **`96.8%`** | **PASS (`+4.8%`)** |
| **Model Quality** | **Weighted `F2` Score ($\beta = 2$)** | $(1 + 2^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$ (weights Recall `4x` over Precision) | **`>= 95.0%`** | **`96.42%`** | **PASS (`+1.42%`)** |
| **Gondola KPI 1** | **Facing-Count Share of Shelf (`Count SOS %`)** | $100 \times N_{\text{HUL}} / (N_{\text{HUL}} + N_{\text{Non-HUL}})$ | **`<= 1.5%` MAE** | **`0.42%` MAE** | **PASS** |
| **Gondola KPI 2** | **Linear Horizontal Width Share of Shelf (`Linear SOS %`)** | $100 \times \sum_{i \in \text{HUL}} W_i / \sum_{j \in \text{All}} W_j$ along shelf rail | **`<= 1.5%` MAE** | **`0.38%` MAE** | **PASS** |
| **Gondola KPI 3** | **2D Billboard Area Share of Shelf (`Area SOS %`)** | $100 \times \sum_{i \in \text{HUL}} (W_i \cdot H_i) / \sum_{j \in \text{All}} (W_j \cdot H_j)$ | **`<= 1.5%` MAE** | **`0.45%` MAE** | **PASS** |
| **Gondola KPI 4** | **Empty-Shelf Out-of-Stock (`OOS`) Void-Gap Recall** | Horizontal rail gaps where $\Delta x_{\text{gap}} \ge 1.25 \cdot W_{\text{median}}$ and no front facing exists | **`>= 95.0%` Recall** | **`97.8%` Recall** | **PASS** |
| **Gondola KPI 5** | **Planogram Sequence Compliance (`Levenshtein %`)** | $100 \times (1 - d_{\text{Lev}}(S_{\text{obs}}, S_{\text{target}}) / \max(|S_{\text{obs}}|, |S_{\text{target}}|))$ | **`>= 94.0%` Acc** | **`98.4%` Acc** | **PASS** |
| **Gondola KPI 6** | **Brand-Block Contiguity & Purity (`%`)** | $100 \times (1 - N_{\text{competitor\_intrusions}} / N_{\text{HUL\_facings}})$ | **`>= 95.0%` Acc** | **`98.9%` Acc** | **PASS** |
| **Gondola KPI 7** | **Eye-Level Golden-Zone Share (`1.2m to 1.5m`)** | Share of Shelves 3 and 4 occupied by HUL priority SKUs | **`<= 1.5%` MAE** | **`0.31%` MAE** | **PASS** |
| **Gondola KPI 8** | **Promotional Header (`Toker`) Compliance `F1`** | OCR and layout verification of shelf-strip promo banners | **`>= 95.0%` F1** | **`97.6%` F1** | **PASS** |

## 7. Quickstart: Run the Live Control Plane UI & Automated Verification Suite

### Launch the Interactive 8-Tab Control Plane (`http://127.0.0.1:8765`)
```bash
PYTHONPATH=src:. python3 src/shelf_e2e/platform/server.py --host 0.0.0.0 --port 8765
```

### Run the Automated Unit & Integration Benchmark Suite (`9/9` Tests Passing)
```bash
PYTHONPATH=src:. python3 -m unittest -v \
  tests/test_shelfbench_arena_platform.py \
  tests/test_spec006_djev_ijepa_and_mt_kpis.py
```
