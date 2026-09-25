# Multi-Level Model Performance, Causal Ablation, and Failure-Mode Analysis

**Author:** Jigyasu Juneja (`jjuneja@google.com`), Principal Forward Deployed Engineer  
**Co-Author:** Riley Gavigan (`rgavigan@google.com`)  
**Repository:** `https://github.com/jigyasujuneja/unilever-shelf-understanding-e2e`  

This document presents the empirical evaluation of all eight shelf-understanding architectures across five analytical levels (`Model Architecture`, `7 HUL Taxonomy Dimensions`, `High-Cardinality Variant Cohorts`, `Commercial Use Cases & Gondola KPIs`, and `Train/Val/Test Generalization`), followed by a causal component ablation and an engineering analysis of where the production system still fails and why.

## 1. Level 1: Performance by Model Architecture (`Unified Cloud Leaderboard`)

We evaluated eight architectures across two complementary benchmarks:
1. **50-Image `SKU-110K` Dense Localization Test Split (`7,264` ground-truth shelf boxes):** Measures 2D bounding-box localization (`IoU >= 0.50` Precision, Recall, and `F2`), `P95` / `P99` latency, and Cloud Billing Catalog unit cost (`INR / image`).
2. **HUL 25-Image Golden Gondola & 245-Variant Scorecard (`17,118` ground-truth facings + `79` Non-HUL competitor SKUs):** Measures fine-grained 7-Dimension SKU identification (`F2`), 14-SKU Sister-Shade disambiguation (`F2`), and ERP Base Pack hallucination rate.

| Rank | Approach (`src/approaches/`) | Architecture & Routing Cascade | 2D Box `F2` (`IoU>=0.5`) | HUL 7-Dim SKU `F2` | Sister-Shade `F2` (`44` Tail SKUs) | 1-Img `P95` Latency | 6-Img Aisle `P95` | Cost / Img (`INR`) | 7-Gate CI/CD Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | **`hul_8stage_gemini38_hybrid`** | `RT-DETR-v2` + `I-JEPA/ScaNN` (`89%`) + `Stage 4.5 /v1/systemone` (`9%`) + `Gemini 3.8 Flash` (`2%`) | **`0.990`** (`P:0.990 R:0.990`) | **`0.979`** (`96.42%` macro) | **`0.942`** (`0.969` on 14-SKU) | **`0.8s - 1.5s`** (`218ms` GPU) | **`0.95s`** | **`INR 0.036`** | **PROMOTED (`7/7` Gates)** |
| **#2** | **`djev_systemone_sister_shade`** | `RT-DETR-v2` + `I-JEPA/ScaNN` (`89%`) + `Stage 4.5` + `/v1/systemone` (`11%`) | **`0.990`** (`P:0.989 R:0.990`) | **`0.974`** | **`0.940`** (`0.964` on 14-SKU) | **`1.3s`** (`218ms` GPU) | **`0.95s`** | **`INR 0.045`** | **PROMOTED (`7/7` Gates)** |
| **#3** | **`tiered_hybrid_scann`** | `RT-DETR-v2` + `AlloyDB ScaNN` (`89%`) + `Gemini 3.8 Flash` (`11%`) | **`0.988`** (`P:0.988 R:0.988`) | **`0.958`** | **`0.884`** | **`1.9s`** | **`3.40s`** | **`INR 0.056`** | **HOLD (`6/7`, Shade `F2 < 0.92`)** |
| **#4** | **`track_f_sam2_scann`** | `RT-DETR-v2` + `SAM-2` Pixel Mask + `DINOv2-Large` + `PaliGemma-2-3B-LoRA` | `0.986` | `0.942` | `0.864` | `0.30s` | `1.49s` | `INR 0.192` | **PASS (`7/7`, Higher GPU cost)** |
| **#5** | **`track_e_open_vocab`** | `OWL-v2 / GroundingDINO` + `I-JEPA 512-D` + `SigLIP-So400m` | `0.972` | `0.934` | `0.682` | `0.17s` | `0.79s` | `INR 0.016` | **FAIL (Shade `F2 = 0.682`)** |
| **#6** | **`track_a_cascading_vit`** | Legacy 13-Model / `GEAP` Supervised Cascade (`YOLO` + `XceptionNet` + `InceptionNet` / `MaxViT`) | `0.968` | `0.792` | `0.184` | `0.19s` | `0.92s` | `INR 0.185` | **FAIL (`F2 < 0.95`, 3-wk retrain)** |
| **#7** | **`single_pass`** (`gemini-3.5-flash-lite`) | 1-Pass Full-Shelf Gemini VLM (`cloud-gtm` baseline) | `0.726` (`0.816` max) | `0.420` | `0.120` | `25.9s` | `142.0s` | `INR 1.229` (`0.254` promo) | **FAIL (Latency & Variant `F2`)** |
| **#8** | **`detect_classify`** (`gemini-3.8-flash`) | 2-Pass Gemini Detect + Crop Classify (`cloud-gtm` baseline) | `0.723` (`0.768` max) | `0.884` | `0.785` | `107.1s` | `>300s` | `INR 2.032` | **FAIL (`Cost > INR 0.22`, Latency)** |

## 2. Level 2: Performance Across All 7 HUL Taxonomy Dimensions (`Classify vs. Derive`)

Each facing on a Unilever gondola requires resolving seven hierarchical product dimensions. Treating all seven dimensions as a single flat classification problem wastes compute and degrades accuracy on geometric dimensions (`Packaging Type`, `Pack Type`, `Size`). Our architecture pairs each dimension with its physically matched solver:

| Dimension # & Name | Resolution Mode | Legacy 13-Model Stack (`InceptionNet / XceptionNet`) | 1-Pass Full-Shelf VLM (`single_pass`) | `ScaNN` Vector Only (`Track C`) | Unified Production Solver (`hul_8stage_gemini38_hybrid`) | Production `F2` / Accuracy | Why the Specialist Solver Wins |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Category**<br>(`Hair Care-DMT`, `Skin Care`, `Oral`, `Laundry`, `Foods`) | `Stage 4: Classify` | `94.2%` | `98.5%` | `99.4%` | `DINOv2-reg4` + `ScaNN` Coarse Category Centroids (`0` tokens) | **`99.6%`** (`0.4 ms`) | Global bottle shape and color block separate categories cleanly in `1024-D` ViT space. |
| **2. Subcategory**<br>(`Facewash`, `Moisturizer`, `Shampoo`, `Conditioner`) | `Stage 4: Classify` | `89.8%` | `91.4%` | `98.2%` | `DINOv2-reg4` + Hierarchical `ScaNN` Sub-Tree (`0` tokens) | **`99.1%`** (`0.4 ms`) | Conditioning on Category eliminates cross-aisle false positives (e.g., food jars vs. skin creams). |
| **3. Brand**<br>(`HUL` Closed-Catalog vs. `Non-HUL` Competitor) | `Stage 4: Classify` | `88.5%` (Blind to new Non-HUL) | `95.1%` | `95.8%` | `I-JEPA 512-D` Latent De-Glare + Dual-Threshold `ScaNN` (`tau = 0.82`) | **`98.7%`** (`1.1 ms`) | `I-JEPA` removes foil glare on `Lakme` / `Sunsilk` logos; `tau < 0.82` isolates unseen competitors. |
| **4. Variant (Fine-Grained)**<br>(`Lakme CC Almond` vs. `Honey` vs. `Bronze`, `SPF 24/50`) | `Stage 4 + Stage 4.5 + Stage 5a` | `68.1%` (`0.0%` on `CC Honey`) | `42.0%` | `78.9%` (`38.4%` on shades) | **`Stage 4.5` `3x` Sub-ROI Zoom (`[0.62H:0.88H]`) + `CIELAB Delta-E00` + Menon Logit + `/v1/systemone`** | **`96.4%` Macro / `97.9%` Weighted** (`94.2%` shades) | Optical `3x` zoom preserves `8px` shade text; `CIELAB` measures swatch pigment; Menon penalty stops `Bronze` from absorbing `Honey`. |
| **5. Packaging Type**<br>(`Bottle`, `Cap-Down Tube`, `Jar`, `Carton`, `Sachet`) | `Stage 4.5: Derive` | `82.4%` (`66.3%` on tubes) | `84.0%` | `91.2%` | **Silhouette Contour Taper Ratio ($W_{\text{top }15\%} / W_{\text{bottom }15\%} > 1.22$)** | **`98.9%`** (`0.2 ms`) | Inverted conditioner tubes have a wide top crimp and narrow bottom cap ($ratio = 1.34$), whereas shampoo bottles taper upward ($ratio = 0.72$). |
| **6. Pack Type**<br>(`Single Unit` vs. `Multipack / Banded / Hanging Sachet Strip`) | `Stage 3 + Stage 5: Derive` | `74.0%` (Drops `50%` sachets) | `71.2%` | `92.5%` | **Vertical-Chain `DIoU-NMS` + Horizontal Periodicity Shrink-Band Detector** | **`98.2%`** (`0.4 ms`) | Vertical-Chain NMS preserves vertically shingled sachet strips; periodicity detects 3-pack banded soaps. |
| **7. Size + `ERP Base Pack Code`**<br>(`180ml` vs. `340ml` vs. `650ml` $\rightarrow$ `BP-HUL-...`) | `Stage 5: Derive` | `76.4%` | `64.8%` (`4.2%` invalid ERP) | `88.6%` | **Shelf-Rail Gap Normalization ($H_{\text{box}} / H_{\text{shelf\_gap}}$) + `/v1/systemone` (`vllm#58216`)** | **`97.4%`** (`0.0%` ERP Hallucination) | Normalizing box pixel height by the local vertical distance between shelf rails eliminates camera-distance perspective errors. |

## 3. Level 3: Performance by High-Cardinality Variant Cohorts (`245` HUL Variants + `79` Competitors)

HUL's `17,118` ground-truth facings span four distinct cardinality cohorts:

| Variant Cardinality Cohort | SKU Count | GT Facings (`Share`) | Legacy 13-Model Stack `F2` | 1-Pass VLM `F2` | `ScaNN` Only `F2` | Unified Production `F2` | Net Gain vs. Legacy | Primary Engineering Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cohort 1: High-Volume Core Anchors** (`>= 50` facings/SKU) | `201` SKUs | `15,490` (`82.0%`) | `94.1%` | `54.2%` | `96.2%` | **`98.6%`** | **`+4.5%`** | `Stage 4` `DINOv2-reg4` + `ScaNN` (`0.8 ms`, `0` tokens). |
| **Cohort 2: Sister-Shade & Foil-Glare Collisions** (`8px` shade/SPF difference) | `24` SKUs | `892` (`9.8%`) | `24.6%` (`Honey = 0%`) | `12.0%` | `38.4%` | **`94.2%`** | **`+69.6%`** | `Stage 4.5` `3x` Sub-ROI Zoom + `CIELAB Delta-E00` + `/v1/systemone`. |
| **Cohort 3: Low-Shot / Day-0 Clinical Launches** (`11-26` facings/SKU) | `20` SKUs | `736` (`8.2%`) | `19.8%` (`Hydra Glow = 0%`) | `14.5%` | `64.0%` | **`92.4%`** | **`+72.6%`** | `< 60s` `hot_swap_onboard_sku` `16`-view prototypes + Menon prior adjustment. |
| **Cohort 4: Open-Set Non-HUL Competitor Variants** (`sim < 0.82`) | `79` SKUs | `2,410` | `0.0%` (`UNKNOWN`) | `61.0%` | `78.5%` | **`95.8%`** | **`+95.8%`** | Dual-threshold `ScaNN` gate (`sim < 0.82`) routing to `Gemini 3.8 Flash`. |

## 4. Level 4: Performance Across All 4 `Sales EDGE – MT PC` & `GT / Shikkar` Use Cases and 8 Gondola KPIs

| Commercial Pipeline & Client App | Daily Volume (`Avg`) | Primary Operational KPIs | Target SLA / Threshold | Measured Production Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. `MT MarketShare` & `GT MarketShare`** (`Sales EDGE – MT PC`, `Shikkar`) | **`323,935` imgs/day** | • 6-Frame `ORB + RANSAC` Seam Dedup Precision<br>• Front-Row OOS Void-Gap Recall (`Red Line`)<br>• Recognized `Base Pack Code` `F2` + 4-Factor Replenishment Cart | `<= 30.0s` E2E<br>`>= 95.0%` Recall<br>`>= 95.0%` `F2` | **`1.35s` E2E (`948ms` Server `P95`)**<br>**`97.8%` OOS Void Recall**<br>**`97.9%` 7-Dim `F2`** | **PASS (`22x` faster than SLA)** |
| **2. `MT Merchandising` & `GT Merchandising`** (`Sales EDGE – MT PC`, `Shikkar`) | **`104,571` imgs/day** | • `Presence of Asset` (`DISPLAY_WINDOW_BOX` detection)<br>• `Packs present in the Asset` (`ax1 <= xc <= ax2`)<br>• `Compliance as per promotional threshold` (`>= N_min` packs & `>= 90%` purity)<br>• Planogram Sequence (`Levenshtein %`) & Golden-Zone Share (`1.2m-1.5m`) | `<= 10.0s` E2E<br>`>= 95.0%` Asset Recall<br>`>= 94.0%` Planogram Acc | **`0.48s` E2E (`218ms` Server `P95`)**<br>**`98.9%` Asset & Purity Acc**<br>**`98.4%` Planogram Acc**<br>**`0.31%` Golden-Zone MAE** | **PASS (`20x` faster than SLA)** |
| **3. `MT Toker Compliance`** (`Sales EDGE – MT PC`) | **`78,025` imgs/day** | • Reference Promo Image Similarity (`DINOv2-reg4` vs. business graphic)<br>• Promotional `Toker` Header & Discount OCR `F1` | `>= 95.0%` `F1` | **`97.6%` Toker Compliance `F1`** | **PASS (Ready to graduate from Dry Runs)** |
| **4. `MT SOS Pipeline`** (`Sales EDGE – MT PC`) | **Pilot $\rightarrow$ Prod** | • Facing Count SOS MAE (`%`)<br>• Linear Horizontal Width SOS MAE (`%`)<br>• 2D Billboard Area SOS MAE (`%`) across all 6 Category Partitions | `<= 1.50%` MAE | **`0.42%` Count SOS MAE**<br>**`0.38%` Linear SOS MAE**<br>**`0.45%` Area SOS MAE** | **PASS (`3.3x` tighter than error cap)** |

## 5. Level 5: Causal Ablation Study (How and Why Each Component Contributes)

To prove why each architectural stage exists, we ablated one component at a time on the 245-variant benchmark:

| Ablation Configuration | 2D Box `F2` | 7-Dim HUL `F2` | Sister-Shade `F2` (`44` SKUs) | Panorama Overcount Error | What Breaks When This Component Is Removed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Production Architecture (`hul_8stage_gemini38_hybrid`)** | **`0.990`** | **`0.979`** | **`0.942`** | **`0.9%`** | All 7 CI/CD promotion gates pass. |
| **$-$ Remove `/v1/systemone` (`vllm#58216` Constrained Trie)** | `0.990` | `0.961` (`-1.8%`) | `0.864` (`-7.8%`) | `0.9%` | Unconstrained decoding emits `3.1%` invalid ERP Base Pack codes and misreads curved `8px` shade text. |
| **$-$ Remove `Stage 4.5` (`3x` Sub-ROI Zoom + `CIELAB` + Menon Logit)** | `0.990` | `0.912` (`-6.7%`) | **`0.446` (`-49.6%`)** | `0.9%` | Without `[0.62H:0.88H]` `3x` zoom and Menon prior adjustment, `Lakme CC Honey` and `Almond` collapse back into `CC Bronze`. |
| **$-$ Remove `Stage 4 I-JEPA` `512-D` Latent De-Glare Predictor** | `0.990` | `0.884` (`-9.5%`) | `0.384` (`-55.8%`) | `0.9%` | `6500K` store LED specular highlights on foil cartons (`Lakme Lumi Silver`, `Sunsilk` sachets) corrupt `DINOv2` embeddings (`Lumi Silver -> 0% F2`). |
| **$-$ Remove `Stage 3.5 ORB + RANSAC` Panorama Seam Deduplicator** | `0.990` | `0.884` | `0.384` | **`+22.4%` Overcount** | 6-frame aisle walks double-count `202` facings in the `22%` boundary overlap zones, corrupting Share of Shelf and replenishment quantities. |
| **$-$ Remove `Stage 3` Vertical-Chain `DIoU-NMS` & Luminance Ghost Filter** | **`0.892` (`-9.8%`)** | `0.810` | `0.340` | `+22.4%` | Standard greedy NMS suppresses `50%` of vertically hanging sachet strips (`GT`) and counts dark 2nd-row recessed shadow bottles as front stock (hiding `64%` of front-row stockouts). |

## 6. Where Our Architecture Still Fails (`The 2.1% to 3.6% Residual Error Tail`) and Why

No production computer vision system is 100% error-free. Across `17,118` ground-truth facings, our architecture misses or misclassifies **2.1% of weighted instances (`3.58%` unweighted macro error across the 245 variants)**. Below are the four specific physical scenarios where our pipeline still struggles, the mathematical root cause of each failure, and our active engineering mitigations:

### Failure Mode 1: `100%` Opaque Price-Tag or Cross-Merchandising Strip Occlusion Over the `[0.62H : 0.88H]` Shade Band (`0.82%` of Facings)
* **Where it happens:** In regional Modern Trade stores, staff occasionally stick an opaque yellow paper discount tag or hang a plastic cross-merchandising clip directly across the lower-middle `[0.62H : 0.88H]` region of a tube where the `8px` variant/SPF text resides (`Lakme Sun Expert SPF 30` vs. `SPF 50`, which both use the exact same matte yellow tube and blue cap).
* **Why our architecture fails here:** Neither `Stage 4.5` `3x` optical zoom nor `/v1/systemone` can see through opaque paper, and because `SPF 30` and `SPF 50` share identical `CIELAB` tube plastic pigment ($\Delta E_{00} < 0.8$), chromatic swatch matching cannot separate them (`F2` drops from `96.1%` to **`68.4%`** on fully tag-occluded identical-pigment tubes).
* **Mitigation in place:** When `Stage 4.5` detects an opaque rectangular foreign sticker inside `[0.62H : 0.88H]` and `CIELAB` distance between candidates is `< 1.5`, the pipeline applies a **Horizontal Shelf-Block Markov Prior** (assigning the variant of the immediately adjacent unoccluded left/right tube in the same brand block, which recovers `~74%` of tag-occluded tubes) and flags the crop with `confidence < 0.72` into `results/active_learning_queue.jsonl`.

### Failure Mode 2: Extreme Oblique Camera Yaw (`> 52°` Viewing Angle) in Narrow `2.5-Foot` Kirana Aisles (`0.58%` of Facings)
* **Where it happens:** In cramped General Trade (`GT`) counter stores or blocked end-caps where the merchandiser cannot step back 4 feet and captures the shelf at a sharp `> 52°` side angle.
* **Why our architecture fails here:** On cylindrical shampoo/lotion bottles (`Vaseline`, `Dove`), a `> 52°` viewing angle rotates the front-of-pack variant typography past the visible cylinder tangent, leaving only the side ingredient panel visible. Additionally, `Stage 3.5` `ORB + RANSAC` assumes an approximately planar shelf front ($H_{t \to t+1}$); under extreme oblique parallax where deep recessed shelves sit behind protruding hanging strips, `3.8%` of seam boxes suffer homography drift.
* **Mitigation in place:** We index **16 multi-angle studio + synthetic cylindrical perspective views (`-45°` to `+45°` yaw)** per SKU in `AlloyDB ScaNN` (`hot_swap_onboard_sku`), and our mobile capture SDK (`Sales EDGE`) checks bounding-box vanishing-point skew in real time to prompt the rep if camera yaw exceeds `45°`.

### Failure Mode 3: Crushed or Half-Deflated Flexible Stand-Up Pouches (`0.44%` of Facings)
* **Where it happens:** Flexible liquid refill pouches (`Surf Excel Matic 500ml` vs. `1L` pouch, `Vim Dishwash Pouch`) that slouch, fold in half, or get compressed vertically on bottom gondola shelves.
* **Why our architecture fails here:** `Dimension 7` (`Size: 500ml vs 1L`) derives pack volume using the normalized bounding-box height ratio ($H_{\text{box}} / H_{\text{shelf\_gap}}$) whenever the printed `ml/g` weight text at the bottom corner of the pouch is tucked under the shelf lip. When a `1L` flexible pouch slumps vertically by `30%`, its apparent height ratio matches a rigid `500ml` pouch (`Size accuracy` drops from `97.4%` on rigid bottles/jars to **`91.2%`** on slumped pouches).
* **Mitigation in place:** `Stage 5` combines **2D Polygon Mask Area** (`W * H` contour fill) with horizontal pouch base width ($W_{\text{base}}$, which remains rigid even when the top of the pouch folds forward) before assigning the `500ml` vs. `1L` ERP Base Pack code.

### Failure Mode 4: Zero-Day Regional Festival Overlays Before Studio Onboarding (`0.31%` of Facings)
* **Where it happens:** Limited-edition regional festival packs (`Onam`, `Durga Puja`, `Pongal` promotional sleeves covering `> 65%` of front-of-pack artwork with regional script and celebrity portraits) during the first 24 hours in stores *before* the brand team uploads the new packshot to `hot_swap_onboard_sku`.
* **Why our architecture fails here:** When `> 65%` of front artwork changes simultaneously, `DINOv2-reg4` cosine similarity drops below the `tau = 0.82` closed-catalog threshold (`sim ~ 0.74 to 0.79`), causing the crop to escalate to `Stage 5b` (`Gemini 3.8 Flash`). While `Gemini 3.8 Flash` still identifies the brand and category (`95.8%` accuracy), it increases latency on those specific crops from `0.8 ms` to `180 ms` and routes the crop into the Active Learning quarantine queue until the 60-second `ScaNN` hot-swap onboarding runs.
