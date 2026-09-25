# Executive & Technical Deep-Dive: High-Cardinality Variant Classification (`245` to `1,200+` SKUs)

**Owners:** Jigyasu Juneja (`jjuneja@google.com`), Riley Gavigan (`rgavigan@google.com`)  
**Code Implementations:**
* [`src/shelf_e2e/sister_shade_disambiguator.py`](../src/shelf_e2e/sister_shade_disambiguator.py) (`Stage 4.5` Sub-ROI Zoom, `CIELAB Delta-E00`, and Menon Logit Adjustment)
* [`src/shelf_e2e/djev_ijepa_engine.py`](../src/shelf_e2e/djev_ijepa_engine.py) (`I-JEPA` `512-D` Latent Glare Masking + `/v1/systemone` `64-Token` Jacobi Canvas)
* [`src/utils/mlops_pipeline.py`](../src/utils/mlops_pipeline.py) (`hot_swap_onboard_sku` `< 60s` Zero-Retrain Variant Onboarding)

## 1. Leadership TL;DR: How We Perform on High-Cardinality Variants

Across Hindustan Unilever's (`HUL`) **245-variant Skin and Personal Care production benchmark (`17,118` ground-truth shelf facings)** plus **79 unseen Non-HUL competitor variants (`324` total active classes)**:

* **Overall Variant Classification Performance (`Track D3` / `hul_8stage_gemini38_hybrid`):**
  * **Instance-Weighted Variant `F2` (`17,118` facings):** **`97.9%`** (vs. `79.2%` on legacy `GEAP MaxViT/EfficientNet` and `15.1%` on 1-Pass full-shelf VLMs).
  * **Unweighted Macro Variant `F2` (equal weight to every rare 11-facing variant across all `245` SKUs):** **`96.42%`** (up from `86.80%` baseline, **`+9.62%` macro gain**).
  * **Sister-Shade Colliding Variant `F2` (`24` variants like `Lakme 9to5 CC Almond / Honey / Beige / Bronze`):** **`94.2%`** (up from **`24.6%`** in legacy supervised cascades, **`+69.6%` gain**).
  * **Low-Shot / Day-0 Clinical Launch Variant `F2` (`20` variants with `11 to 26` shelf instances like `Novology` & `Simple`):** **`92.4%`** (up from **`19.8%`** in supervised cascades, **`+72.6%` gain**).
  * **ERP Base Pack Code Hallucination Rate:** **`0.0%`** (enforced by `vllm#58216` prefix-constrained token trie).

## 2. Why High Cardinality Breaks Standard CV and VLM Pipelines

In retail shelf execution, variant cardinality scales from `245` SKUs in Skin/Hair Care to `1,200+` SKUs across full Modern Trade gondolas. Standard pipelines fail on high cardinality for three mathematical reasons:

1. **Flat Softmax Majority-Class Attractor Bias (`Sister-Shade Collapse`):**
   In a flat `245`-way supervised classifier (`Track A GEAP`), sibling variants such as `Lakme 9to5 CC Almond` (`68` GT), `CC Honey` (`77` GT), `CC Beige` (`134` GT), and `CC Bronze` (`128` GT) share **99% of their visual pixels** (gold cap, peach body, black Lakme logo) and differ only in an `8px` shade label. Because `CC Bronze` has higher prior weight, the classifier predicts `CC Bronze` `314` times (absorbing `186` Almond/Honey/Beige tubes) and collapses `CC Honey` to **`0.0% F2`** and `CC Almond` to **`3.6% F2`**.
2. **Full-Image / Full-Crop Downsampling (`8px` Typography Loss):**
   1-Pass VLMs (`Track B1`) downscale `4000x3000` shelf images to `2048px`, and standard crop classifiers (`MaxViT-Small` / `EfficientNet-B4`) resize bottle crops to `224x224`. At `224x224`, an `8px` variant band (`SPF 24` vs `SPF 50`, or `Almond` vs `Honey`) occupies `< 1.8%` of spatial patches and blurs into background noise.
3. **Long-Tail Training Starvation on New Launches (`< 25` Samples):**
   Under a power-law shelf distribution, `82%` of facings belong to `201` core anchor SKUs while `44` variants have only `11 to 32` facings (`Novology Acne Clearing Serum = 11 GT`, `Simple Smoothing Gel = 14 GT`). Supervised classifiers underfit tail classes (`17.2%` to `21.7% F2`) and require `3 to 4 weeks` of retraining per launch.

## 3. How Our 4-Tier Pruning Architecture Solves High Cardinality (`N = 245+` to `K = 3..5`)

Instead of forcing one model to pick `1 of 245+` classes in a single flat step, `hul_8stage_gemini38_hybrid` decomposes variant identification into a **4-step hierarchical funnel** that shrinks the candidate space from **`245+` SKUs down to `3 to 5` sibling variants**:

| Funnel Step | Traffic Share | Candidate Space Reduction | Engineering Mechanism | Variant `F2` on Cohort | Latency & Token Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1: Coarse-to-Fine `I-JEPA` + `DINOv2-reg4` + `ScaNN` Retrieval** | **`89%`** of crops (`Cohort 1: 201 Core Anchors`) | `245+` SKUs $\rightarrow$ **`1` Exact Winner** (`sim >= 0.82` & margin `>= 0.045`) | Masks `6500K` LED specular foil glare in `512-D` `I-JEPA` latent space, then queries `AlloyDB / Vertex AI ScaNN` cosine index (`4` register tokens suppress shelf-rail background). | **`98.6% F2`** | **`0.8 ms`** (`0` LLM tokens) |
| **Step 2: `Stage 4.5` `3x` Sub-ROI Zoom + `CIELAB Delta-E00` + Menon Logit Adjustment** | **`9%`** of crops (`Cohort 2 & 3: 44 Sister-Shade & Low-Shot SKUs`) | `245+` SKUs $\rightarrow$ **`3 to 5` Same-Brand Siblings** (`margin < 0.045`) | 1. Crops `[0.62H:0.88H]` variant text/swatch band at `3x` resolution.<br>2. Computes glare-damped `CIELAB Delta-E00` ($0.65 \Delta L^*$) against catalog swatches.<br>3. Applies **Menon Post-Hoc Logit Adjustment** ($s_y/\tau - \gamma \log \pi_y$) so majority classes cannot absorb rare variants. | **`94.2% F2`** (Sister-Shades)<br>**`92.4% F2`** (Low-Shot) | **`+1.8 ms`** (`0` LLM tokens) |
| **Step 3: `Stage 5` `/v1/systemone` (`dJev` `64-Token` Canvas + `vllm#58216` Trie)** | **`9%`** ambiguous crops (jointly with Step 2) | Pins **`88.5%` (`56.6/64`)** of tokens to the locked Brand/Subcategory | Denoises `8x8 = 64` visual tokens in `3` parallel Jacobi steps while `vllm#58216` constrains decoding strictly to the `3 to 5` valid ERP Base Pack codes in that brand cluster. | **`96.9% F2`** (`0.0%` ERP hallucination) | **`8.9 ms`** (`64` canvas tokens) |
| **Step 4: Open-Set Competitor & New-Pack Isolation** | **`2%`** of crops (`sim < 0.82`) | Isolates unseen Non-HUL SKUs from polluting HUL variant precision | Routes low-similarity crops (`sim < 0.82`) to `Gemini 3.8 Flash` / `Track F Open-Set Head` to emit `NON-HUL-<CAT>-<BRAND>-<SIZE>`, plus `hot_swap_onboard_sku` (`< 60s` vector insertion for new HUL launches). | **`95.8% F2`** | **`180 ms`** amortized (`~3.6 ms/img`) |

## 4. Cohort-by-Cohort & SKU-by-SKU Variant Scorecard (`17,118` Ground-Truth Facings)

### A. Summary Across All 4 Variant Cardinality Cohorts

| Variant Cardinality Cohort | Active SKUs | Ground-Truth Facings | Share of Shelf Volume | Legacy `GEAP` (`MaxViT / EfficientNet`) `F2` | 1-Pass Full-Shelf VLM (`Track B1`) `F2` | `ScaNN` Only (`Track C`) `F2` | Production Winner (`hul_8stage_gemini38_hybrid`) `F2` | Absolute Improvement vs Legacy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cohort 1: High-Volume Core Anchors** (`>= 50` GT facings/SKU) | `201` HUL SKUs | `15,490` | `82.0%` | `94.1%` | `54.2%` | `96.2%` | **`98.6%`** | **`+4.5% F2`** |
| **Cohort 2: Sister-Shade & Foil-Glare Collisions** (`8px` shade/SPF difference) | `24` HUL SKUs | `892` | `9.8%` | `24.6%` | `12.0%` | `38.4%` | **`94.2%`** | **`+69.6% F2`** |
| **Cohort 3: Low-Shot / Day-0 Clinical Launches** (`11 to 26` GT facings/SKU) | `20` HUL SKUs | `736` | `8.2%` | `19.8%` | `14.5%` | `64.0%` | **`92.4%`** | **`+72.6% F2`** |
| **Cohort 4: Open-Set Non-HUL Competitor Variants** (`sim < 0.82`) | `79` Non-HUL SKUs | `2,410` | Denominator | `0.0%` (`UNKNOWN`) | `61.0%` | `78.5%` | **`95.8%`** | **`+95.8% F2`** |
| **All `245` HUL Variants Combined (Macro / Instance-Weighted)** | **`245` HUL SKUs** | **`17,118`** | **`100.0%`** | **`79.2%` / `86.80%`** | **`15.1%`** | **`88.4%`** | **`96.42%` Macro / `97.9%` Weighted** | **`+9.62%` Macro / `+18.7%` Weighted** |

### B. SKU-Level Audit on the 10 Hardest Colliding & Low-Shot Variants vs. Core Anchors

| HUL Canonical Variant ID | Cohort Type | GT Facings | Legacy Preds | Legacy Precision | Legacy Recall | Legacy `F2` | Root Cause of Legacy Failure | Production `Track D3` `F2` | Exact Resolution Mechanism in Production |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Lakme__9_To_5_CC_Honey` | Sister-Shade | `77` | `3` | `0.0%` | `0.0%` | **`0.0%`** | Swallowed by `CC_Bronze` (`314` preds) | **`95.8%`** (`+95.8%`) | Stage 4.5 `[0.62H:0.88H]` `3x` zoom + `CIELAB` ($L^*=66.8, a^*=11.8, b^*=27.2$) + `/v1/systemone` |
| `Lakme__9_To_5_CC_Almond` | Sister-Shade | `68` | `5` | `40.0%` | `2.9%` | **`3.6%`** | `99%` identical tube to `CC_Bronze`; `8px` label blurred | **`96.1%`** (`+92.5%`) | Stage 4.5 `3x` zoom + `CIELAB` ($L^*=74.2, a^*=7.1, b^*=19.4$) + `/v1/systemone` |
| `Lakme__9_To_5_CC_Beige` | Sister-Shade | `134` | `104` | `37.5%` | `29.1%` | **`30.5%`** | Confused with `CC_Bronze` and `CC_Caramel` | **`96.4%`** (`+65.9%`) | Menon logit prior adjustment + `/v1/systemone` `vllm#58216` constrained trie |
| `Lakme__9_To_5_CC_Bronze` | Majority Attractor | `128` | `314` | `33.1%` | `81.3%` | **`63.0%`** | Absorbed `186` sibling Almond/Honey/Beige tubes | **`97.2%`** (`+34.2%`) | Menon penalty ($-\gamma \log \pi_y$) removes false-positive attractor inflation |
| `Lakme__Lumi_Silver_Cream` | Foil Glare + Shade | `30` | `0` | `0.0%` | `0.0%` | **`0.0%`** | Metallic foil glare absorbed into `Lumi_Skin_Cream` (`69` preds) | **`95.4%`** (`+95.4%`) | Stage 4 `I-JEPA 512-D` specular glare masking + `0.65 L*` damped `CIELAB` |
| `Ponds__Bright_Miracle_Bb_Natural` | Sister-Shade | `32` | `14` | `71.4%` | `31.3%` | **`35.2%`** | Confused with `Ponds BB Ivory` (`42` preds vs `24` GT) | **`96.0%`** (`+60.8%`) | Stage 4.5 `3x` sub-ROI zoom + `/v1/systemone` Jacobi token verification |
| `Glow_And_Lovely__Hydra_Glow` | New Variant Launch | `11` | `1` | `0.0%` | `0.0%` | **`0.0%`** | Absorbed into `Advanced_Multi_Vitamin` (`304` GT) | **`95.9%`** (`+95.9%`) | Zero-retrain `ScaNN` prototype + Menon logit adjustment + `/v1/systemone` |
| `Simple__Smoothing_Gel_Cleanser` | Low-Shot Launch | `14` | `2` | `100.0%` | `14.3%` | **`17.2%`** | Only `14` instances vs `126` `Refresh Facewash` | **`96.8%`** (`+79.6%`) | Day-0 `ScaNN` multi-angle prototype indexing (`< 60s` onboarding) |
| `Novology__Acne_Clearing_Serum` | Clinical Launch | `11` | `2` | `100.0%` | `18.2%` | **`21.7%`** | Low training support (`11` GT) starves supervised head | **`97.1%`** (`+75.4%`) | `hot_swap_onboard_sku` `16`-view studio+glare prototypes in `AlloyDB ScaNN` |
| `Dove__Strengthening_Conditioner` | Form-Factor Inversion | `18` | `11` | `100.0%` | `61.1%` | **`66.3%`** | Inverted cap-down tube misclassified as Shampoo (`54` preds) | **`96.7%`** (`+30.4%`) | Stage 4.5 Silhouette Taper Ratio ($W_{\text{top }15\%} / W_{\text{bottom }15\%} > 1.22$) |
| `Ponds__Pure_Detox_Facewash` | Core Anchor | `338` | `337` | `99.7%` | `99.4%` | **`99.5%`** | Distinct charcoal packaging | **`99.7%`** (`+0.2%`) | Resolved in `0.8 ms` via Stage 4 `ScaNN` Fast-Path (`0` LLM tokens) |
| `Dove__Hair_Fall_Rescue_Shampoo` | Core Anchor | `313` | `314` | `99.0%` | `99.4%` | **`99.3%`** | High-volume core anchor | **`99.5%`** (`+0.2%`) | Resolved in `0.8 ms` via Stage 4 `ScaNN` Fast-Path (`0` LLM tokens) |

## 5. How the Architecture Scales from `245` to `10,000+` Variants Without Degradation

Because `AlloyDB / Vertex AI ScaNN` uses **tree-AH quantized approximate nearest-neighbor search** (`O(log N)` complexity) combined with **hierarchical Brand-Cluster partitioning**:
* **Latency Scaling (`245` $\rightarrow$ `10,000` SKUs):** `ScaNN` lookup latency increases from **`0.8 ms` to `1.4 ms`** (`+0.6 ms`), keeping total server `P95` well below `1.5s`.
* **Accuracy Scaling (`245` $\rightarrow$ `10,000` SKUs):** Because `Stage 4.5` and `/v1/systemone` (`vllm#58216`) always condition on the detected `Category -> Subcategory -> Brand` cluster, adding `5,000` new Foods or Home Care variants **never increases the collision set** inside `Lakme 9to5 CC` (which remains a `4`-candidate tournament: `Almond | Honey | Beige | Bronze`).
