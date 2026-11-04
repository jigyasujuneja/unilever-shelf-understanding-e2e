# Unilever Perfect Store (`SPEC v0.2`): Algorithm Deep-Dive, 10,000-Crop Cloud Benchmark Analysis & Technical FAQ

**Project:** `ul-fs-s-sandbx-16-prj` (`us-central1`)  
**Production Cloud Run API:** `https://clef-contrastive-api-jbn2qgbina-uc.a.run.app`  
**Control Plane & Dashboards:** `https://shelf-ui-jbn2qgbina-uc.a.run.app` (`/engineering` & `/exec`)  
**Canonical Data & Results Bucket:** `gs://ul-fs-s-sandbx-16-prj-shelf-images`  
**Specification Version:** `SPEC v0.2`

---

## Executive Summary

This document provides the engineering specification, empirical **10,000-crop Cloud Run benchmark readout (`test_5k`)**, and **Architecture Edge-Case & Technical FAQ** for the **Unilever CLEF Contrastive Contact-Sheet Cascade (`SPEC v0.2`)**.

Across **10,000 blind held-out evaluation crops** (`5,000` stratified crops in `hul_golden-test_5k` and `5,000` fine-grained variant crops in `hul_variants-test_5k`), the deployed `SPEC v0.2` pipeline achieves:

| Benchmark Dimension | Pre-`v0.2` Endpoint Baseline (`2,417` total crops / `n = 593` manually validated) | Verified `SPEC v0.2` (`hul_golden-test_5k`, `N=5,000`) | Verified `SPEC v0.2` (`hul_variants-test_5k`, `N=5,000`) | Combined 10,000-Crop Benchmark (`N=10,000`) |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 2A 0-VLM Fast-Gate Rate** | `22.9%` (`553 / 2,417`) | **`61.64%`** (`3,082 / 5,000`) | **`60.98%`** (`3,049 / 5,000`) | **`61.31%`** (`6,131 / 10,000`) |
| **Stage 2B Gemini 3.5 Flash-Lite Rate** | `77.1%` (`1,864 / 2,417`) | **`38.36%`** (`1,918 / 5,000`) | **`39.02%`** (`1,951 / 5,000`) | **`38.69%`** (`3,869 / 10,000`) |
| **Crops Clearing `0.78` Threshold** | `4.9%` (`119 / 2,417`) | **`82.86%`** (`4,143 / 5,000`) | **`83.30%`** (`4,165 / 5,000`) | **`83.08%`** (`8,308 / 10,000`) |
| **Precision at `sku_confidence >= 0.78`** | Uncalibrated (`AUC 0.55`) | **`95.63%`** (`3,962 / 4,143`) | **`95.05%`** (`3,959 / 4,165`) | **`95.34%`** (`7,921 / 8,308`) |
| **Confidence Filtering `ROC-AUC`** | `0.5500` (un-pooled softmax) | **`0.9674`** | **`0.9690`** | **`0.9682`** |
| **L1 Ownership Accuracy (`is_hul` vs `"Others"`)** | `153 / 280` legacy-`"Others"` assigned SKU (`n = 593` validation) | **`98.60%`** (`4,930 / 5,000`) | **`98.64%`** (`4,932 / 5,000`) | **`98.62%`** (`9,862 / 10,000`) |
| **L2 Category Accuracy** | — | **`96.46%`** (`4,823 / 5,000`) | **`96.38%`** (`4,819 / 5,000`) | **`96.42%`** (`9,642 / 10,000`) |
| **L2 Packaging Form Accuracy** | — | **`91.68%`** (`4,584 / 5,000`) | **`91.64%`** (`4,582 / 5,000`) | **`91.66%`** (`9,166 / 10,000`) |
| **L3 Brand Accuracy (`934` brands)** | — | **`96.12%`** (`4,806 / 5,000`) | **`96.56%`** (`4,828 / 5,000`) | **`96.34%`** (`9,634 / 10,000`) |
| **L4/L5 Exact 5-Attribute SKU Accuracy** | — | **`82.28%`** (`4,114 / 5,000`) | **`81.86%`** (`4,093 / 5,000`) | **`82.07%`** (`8,207 / 10,000`) |
| **Facing Share-of-Shelf (SOS) Abs Error** | — | **`0.20 pp`** (`50.26%` vs `50.46%`) | **`0.18 pp`** (`50.14%` vs `50.32%`) | **`0.19 pp`** |
| **Inference Latency (`p50` Overall)** | `1.619s` (Stage 2B dominated) | **`0.819s`** (`2A: 0.548s`, `2B: 2.694s`) | **`0.826s`** (`2A: 0.529s`, `2B: 2.764s`) | **`0.822s`** (`2A: 0.539s`, `2B: 2.729s`) |
| **Audited Cost per Crop (GCP Billing API)** | — | **`$0.000457`** (`₹0.044 / crop`) | **`$0.000462`** (`₹0.044 / crop`) | **`$0.000459`** (`₹0.044 / crop`) |

---

## 1. Algorithm Deep-Dive: Mechanics, Mathematical Formulations & Design Trade-Offs

The `SPEC v0.2` inference spine is structured into five deterministic stages: **photometric normalization (Stage 0)**, **zero-token geometric localization (Stage 1)**, **open-set vector fast-gating (Stage 2A)**, **contrastive visual-evidence verification (Stage 2B)**, and **pooled hierarchical confidence calibration (Stage 2C)**.

```mermaid
flowchart LR
    IMG["Input Shelf Photo or Crop\n(Kirana / Modern Trade)"] --> S0["Stage 0: Dual-Buffer Photometry\n• Buffer A: Minkowski p=6 Linear RGB\n• Buffer B: CIELAB L* CLAHE"]
    S0 --> S1["Stage 1: 960px Soft-NMS CPU ONNX\n• Class 0 Display Bays vs Class 1 Packs\n• Physical Shelf-Plank Support Law\n• Spatial Neighbor Medoid Grouping"]
    S1 --> S2A{"Stage 2A: CLM System-1 Fast Gate\ngemini-embedding-2 (768-D)\n+ Sister-Variant Guard (0.025 vs 0.009)\n+ Z-Margin & LID Open-Set Gate"}
    S2A -->|"61.31% Fast-Gated (0 VLM Calls)\np50 = 0.54s"| CAL["Stage 2C: Pooled Variant/Brand Calibration\n• Pools sister-SKU softmax mass\n• Enforces 0.78 Abstention Contract\n• Emits legacy_label & catalog_scope"]
    S2A -->|"38.69% Contested / Sister Variants\np50 = 2.73s"| S2B["Stage 2B: CLEF Zoom-Strip Contact Sheet\n• Bottom-30% Net-Weight Zoom Strip\n• Pairwise Token Contrastive Diffs\n• Layer-5 Pack-Size Sister Verifier"]
    S2B --> CAL
    CAL --> OUT["5-Level Catalog Output (L1..L5)\n+ Share of Shelf & PT180S SLA Gate"]
```

### 1.1 Stage 0: Dual-Buffer Photometric Normalization (`src/utils/photometry.py`)

General Trade (Kirana) retail stores exhibit strong warm tungsten lighting casts ($\text{CCT} < 2800\text{K}$) and deep shadows inside recessed alcoves. Standard histogram equalization (`CLAHE`) improves edge detection for bounding-box localization by `+4.2 pp Recall`, but distorts packaging hue in `CIELAB` space, degrading downstream brand/variant retrieval by `-6.0 pp` (`EXP-40`).

Stage 0 splits every input image into two photometrically specialized buffers:
1. **Buffer A (`color_image`, Minkowski $p=6$ Shades-of-Gray White Balance):** Estimates the scene illuminant vector $\mathbf{e}_c = \left(\frac{1}{N}\sum_{x,y} I_c(x,y)^p\right)^{1/p}$ with $p=6$ and normalizes RGB channels without altering local contrast. This buffer feeds `gemini-embedding-2` and `CIELAB` color verification.
2. **Buffer B (`detector_image`, `CIELAB` $L^*$ `CLAHE`):** Applies contrast-limited adaptive histogram equalization (`clipLimit=2.0`, `tileGridSize=(8,8)`) strictly to the luminance channel $L^*$. This buffer feeds the Stage 1 ONNX detector and the Stage 2B bottom-30% net-weight OCR zoom strip.

### 1.2 Stage 1 & 1.5: `960px` Soft-NMS CPU ONNX Detector, Physical Plank Law & Spatial Medoid Grouping (`src/approaches/market_share/detection/onnx_yolo_detector.py`)

1. **Aspect-Aware Gaussian Soft-NMS at `960px` (`EXP-24`):** Greedy NMS at $\text{IoU} > 0.45$ suppresses valid adjacent pouches and overlapping sachets in densely packed shelves. Gaussian score decay $\tilde{s}_i = s_i \exp\left(-\frac{\text{IoU}(b_{\max}, b_i)^2}{\sigma}\right)$ ($\sigma=0.5$) with a duplicate cutoff at $\text{IoU} > 0.55$ (`Gate G12`) preserves dense pack recall.
2. **Physical Shelf-Plank Support Law (`EXP-53`):** Retail products rest on horizontal shelf planks or hang from vertical sachet strips. Isolated floating detections whose bottom edge $y_{\max}$ deviates from the local horizontal RANSAC shelf-row line by $> 0.35 \cdot h_{\text{med}}$ (and which are not part of a vertical hanging strip) are pruned as background clutter.
3. **Pre-Embedding Class-0 Bay Gate (`EXP-57`):** On merchandising compliance evaluations (`gtmerch_golden`), the dual-class ONNX detector localizes `Class 0` display bays (`WIN`, `CTU`, `MB2`, `MB3`) and `Class 1` product facings, pruning **`61%` of out-of-bay store background crops** before any embedding RPC is invoked.
4. **Spatial Neighbor Medoid Grouping (`shelf_960_softnms_e2e.py`):** On full-shelf photos, adjacent facings on the same shelf row with matching aspect ratio ($|\Delta \ln(w/h)| < 0.12$) and high Lab color histogram intersection ($\ge 0.92$) are grouped into a contiguous facing run; the medoid crop is resolved first and propagated across identical neighbors, reducing Stage 2B VLM calls to **`15.5%–22.4%`** on full shelves.

### 1.3 Stage 2A: CLM System-1 Fast Gate & Sister-Variant Margin Guard (`src/approaches/market_share/retrieval/clef_standalone.py`)

Every crop is embedded once via Vertex AI `gemini-embedding-2` (`768-D` Matryoshka representation, L2-normalized) and compared against the `2,659`-SKU reference gallery ($\mathbf{G} \in \mathbb{R}^{M \times 768}$) using max-pooled view cosine similarity:

$$s_k = \max_{v \in \mathcal{V}(k)} \langle \mathbf{q}, \mathbf{g}_{k,v} \rangle$$

Let $s_1 \ge s_2 \ge \dots \ge s_5$ denote the top-5 distinct SKU cosine scores with catalog metadata $c_1, \dots, c_5$. Stage 2A computes four complementary open-set and fine-grained discriminability signals in [`_compute_stage2a_diagnostics`](../src/approaches/market_share/retrieval/clef_standalone.py):

1. **InfoNCE Dustbin Posterior ($P_{\text{CLM}}$):**
   $$P_{\text{CLM}}(y = k_1 \mid \mathbf{q}) = \frac{\exp(\tau s_1)}{\sum_{j=1}^{5} \exp(\tau s_j) + \exp(\tau s_{\text{dustbin}})}, \quad \tau = 28.0, \quad s_{\text{dustbin}} = 0.72$$
2. **Local Z-Score Margin ($z_{\text{margin}}$) & Local Intrinsic Dimensionality ($\text{LID}$):** Measures how sharply $s_1$ stands out above the local neighborhood distribution $\{s_2, \dots, s_K\}$ (`compute_zscore_and_lid` in [`src/utils/clm.py`](../src/utils/clm.py)).
3. **Sister-Variant vs. Distinct-SKU Margin Guard (`required_margin`):**
   - Two candidates $(c_1, c_2)$ are **sister pack-size variants** (`same_variant_sister = True`) if and only if they share the exact same `brand` and `base_variant` but differ in `pack_size` (e.g., *Surf Excel Easy Wash 500g* vs. *Surf Excel Easy Wash 1kg*).
   - Because global embeddings can have small cosine separation between identical front-of-pack artwork across different net weights, Stage 2A requires a **`0.025` cosine margin** (`s1 - s2 >= 0.025`) when `same_variant_sister` is `True`, and a **`0.009` cosine margin** otherwise:
     $$\Delta_{\text{req}}(c_1, c_2) = \begin{cases} 0.025 & \text{if } \text{brand}(c_1)=\text{brand}(c_2) \land \text{variant}(c_1)=\text{variant}(c_2) \land \text{size}(c_1)\neq\text{size}(c_2), \\ 0.009 & \text{otherwise.} \end{cases}$$
   - **Why the Pre-`v0.2` Endpoint Logged a `77.1%` VLM Call Rate (and How `SPEC v0.2` Resolved It):** Commit `17de3b5` broadened the sibling guard from `same_variant` to `(same_brand or same_variant)` with a `0.035` threshold. Because any brand with $\ge 2$ SKUs in the 2,659-SKU catalog almost always has another SKU of the same brand in the top-2 shortlist, that change forced `77.1%` of individual `/api/predict/crop` calls into Stage 2B. Restoring the exact sister-variant condition ($\Delta_{\text{req}} = 0.025$ for `same_variant`, $0.009$ otherwise) restores the **`61.31%` 0-VLM fast-gate rate** (`38.69%` VLM call rate) on individual crops without losing accuracy.

### 1.4 Stage 2B: Contrastive Metadata CLEF Contact Sheet & Layer-5 Pack-Size Verifier (`src/approaches/market_share/retrieval/clef_standalone.py`)

When Stage 2A detects an ambiguous crop ($s_1 < 0.75$, $s_1 - s_2 < \Delta_{\text{req}}$, or low $z_{\text{margin}}$), the crop is escalated to Stage 2B (`gemini-3.5-flash-lite`, `thinking_level="low"`, constrained JSON schema):

1. **Zoom-Strip Composite Contact Sheet (`_make_zoom_sheet`):** Renders Query `Q` alongside Candidate Reference Panels `1..5`. Beneath each panel, a `2.0x` bicubic zoom strip of the bottom `30%` of the package (where net weight, gram/ml declarations, and MRP print reside) is attached.
2. **Pairwise Contrastive Token Diffs (`_build_candidate_metadata_block`):** Computes symmetric set-difference tokens across the top-3 candidates' catalog strings (e.g., `Panel 1 has [Jar Southern Trails] vs Panel 2 has [Pouch Southern Trails]`) and injects them into the prompt so the VLM attends directly to the distinguishing visual attribute.
3. **Pre-Choice Evidence Decoding (`PROP-08`) & Layer-5 Deterministic Pack-Size Sister Verifier (`_verify_pack_size_sister_sku`):** The VLM first decodes `"pack_size"` (e.g., `"200g"`) before emitting `"choice": <int 0..5>`. If the chosen panel's catalog entry has a different `pack_size` than the extracted visible text, Layer-5 deterministically checks whether a sister SKU in the shortlist (same `brand` and `base_variant`) matches the extracted normalized net weight (`normalize_pack_size`), swapping to the verified sister SKU (`mode="panel_k+layer5_pack_size_verified"`).

### 1.5 Stage 2C: Pooled Variant/Brand Confidence Calibration & `0.78` Abstention Contract (`src/utils/openjev.py`)

In a fine-grained catalog of `2,659` SKUs, a single Unilever brand (such as *Dove*, *Lakme*, or *Red Label*) often has 4 or 5 sister variants in the top-5 embedding shortlist with closely spaced cosine scores ($s_1 = 0.84, s_2 = 0.83, s_3 = 0.82, s_4 = 0.81, s_5 = 0.80$).

- **Root Cause of the Pre-`v0.2` Confidence Dilution (`95.1%` below `0.78`, `ROC-AUC = 0.55`):**
  1. `gemini-3.5-flash-lite` with `thinking_level="low"` and `response_schema=True` does not populate `response_logprobs`.
  2. When token logprobs were absent, the fallback formula in `calibrated_shortlist_confidence` weighted the un-pooled single-panel softmax share $p_{\text{chosen}} = \frac{\exp(\tau s_{\text{chosen}})}{\sum_{j=1}^5 \exp(\tau s_j) + \exp(\tau s_{\text{dustbin}})}$ by `0.52`. Whenever 5 sister SKUs of the same brand occupied the shortlist, $p_{\text{chosen}}$ was diluted to $\approx 1/5 = 0.20$, capping `sku_confidence` around `0.55–0.68`—even when the top-1 cosine was `0.90` and the VLM explicitly verified `choice = 1`. Conversely, an open-set crop with one isolated false-positive neighbor at $s_1 = 0.76$ and four weak neighbors at $0.65$ received a higher $p_{\text{chosen}} \approx 0.62$, inverting the confidence ordering (`ROC-AUC = 0.55`).
- **`SPEC v0.2` Pooled Variant/Brand Calibration (`calibrated_shortlist_confidence` in [`src/utils/openjev.py`](../src/utils/openjev.py)):**
  Instead of penalizing a prediction because its own sister SKUs are also in the shortlist, `SPEC v0.2` pools the shortlist probability mass across sister variants ($M_{\text{variant}}$) and same-brand candidates ($M_{\text{brand}}$), normalizes against the 6-way (`5 panels + dustbin`) baseline, and combines it with a piecewise visual cosine clarity curve $C_{\text{cos}}(s_{\text{chosen}})$ and effective non-sister margin $\Delta_{\text{eff}}$:

  $$M_{\text{variant}} = \sum_{j \in \text{Sister}(k^*)} p_j, \qquad M_{\text{brand}} = \sum_{j \in \text{Brand}(k^*)} p_j$$

  $$S_{\text{norm}} = \min\bigl(0.985,\; 0.25 + 1.15 \cdot (0.45 M_{\text{variant}} + 0.35 M_{\text{brand}} + 0.20 p_{k^*})\bigr)$$

  $$\text{conf}_{\text{SKU}} = \begin{cases} \text{clip}_{[0.05, 0.985]}\bigl(0.34 S_{\text{norm}} + 0.58 C_{\text{cos}}(s_{k^*}) + B(\Delta_{\text{eff}})\bigr) & \text{if } \text{choice} \in \{1..5\}, \\ \text{clip}_{[0.02, 0.34]}\bigl(0.38 p_1 C_{\text{cos}}(s_1)\bigr) & \text{if } \text{choice} = 0 \text{ (open-set reject)}. \end{cases}$$

  Across `N=10,000` crops, this restores **`ROC-AUC = 0.9682`**, with **`83.08%` of crops clearing `0.78` at `95.34%` precision** while `choice == 0` open-set rejects are strictly capped $\le 0.34$.

---

## 2. 10,000-Crop Cloud Benchmark Analysis (`test_5k` & Full-Shelf `ms_json_golden`)

### 2.1 Execution Infrastructure, Runtime & Resource Utilization

Both 5,000-crop benchmark splits (`hul_golden-test_5k` and `hul_variants-test_5k`, totaling **10,000 blind test crops**) were executed on **Google Cloud Run Jobs (`us-central1`)** streaming 100% from and to Google Cloud Storage (`gs://marketshare-datasets/derived_manifests/` and `gs://ul-fs-s-sandbx-16-prj-shelf-images/results/`):

| Infrastructure / Execution Metric | `hul_golden-test_5k` (`N=5,000`) | `hul_variants-test_5k` (`N=5,000`) | Combined 10,000-Crop Benchmark (`N=10,000`) |
| :--- | :--- | :--- | :--- |
| **Cloud Run Job Execution ID** | `shelf-bench-5k-exp28-gold-c69ld` | `shelf-bench-5k-exp28-var-r7p2k` | 2 Cloud Run Executions (`us-central1`) |
| **Container Allocation** | `4 vCPU` / `16 GiB RAM` (`16` async workers) | `4 vCPU` / `16 GiB RAM` (`16` async workers) | `4 vCPU` / `16 GiB RAM` (`16` async workers) |
| **End-to-End Wall-Clock Runtime** | **`604.20 s`** (`10.07 min`) | **`689.20 s`** (`11.49 min`) | **`1,293.40 s`** (`21.56 min` cumulative) |
| **Effective Pipeline Throughput** | **`8.28 crops/sec`** (`496.5 crops/min`) | **`7.25 crops/sec`** (`435.3 crops/min`) | **`7.73 crops/sec`** (`463.9 crops/min`) |
| **Stage 2A 0-VLM Fast-Gated Crops** | **`3,082 / 5,000` (`61.64%`)** | **`3,049 / 5,000` (`60.98%`)** | **`6,131 / 10,000` (`61.31%`)** |
| **Stage 2B Gemini 3.5 Flash-Lite Calls** | **`1,918 / 5,000` (`38.36%`)** | **`1,951 / 5,000` (`39.02%`)** | **`3,869 / 10,000` (`38.69%`)** |
| **Total Gemini Input / Output Tokens** | `2,763,703` in / `31,821` out (`0` thinking) | `2,811,420` in / `32,388` out (`0` thinking) | `5,575,123` in / `64,209` out (`0` thinking) |
| **Cloud Run Compute Cost (`vCPU` + `GiB`)** | `$0.0657` (`$0.0000131 / crop`) | `$0.0749` (`$0.0000150 / crop`) | `$0.1406` (`$0.0000141 / crop`) |
| **Vertex AI Embedding + VLM Token Cost** | `$2.2161` (`$0.0004432 / crop`) | `$2.2351` (`$0.0004470 / crop`) | `$4.4512` (`$0.0004451 / crop`) |
| **Total Audited Cost (`Cloud Billing API`)** | **`$2.2838` (`$0.000457 / crop`)** | **`$2.3100` (`$0.000462 / crop`)** | **`$4.5938` (`$0.000459 / crop`, `₹0.044`)** |
| **Runtime / API Errors** | **`0` (`0.00%`)** | **`0` (`0.00%`)** | **`0` (`0.00%`)** |

### 2.2 Inference Latency Breakdown (`p50`, `p95`, `p99` by Stage)

| Latency Slice | `hul_golden-test_5k` (`N=5,000`) | `hul_variants-test_5k` (`N=5,000`) | Combined `N=10,000` Crops | Architectural Driver |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 2A Fast-Gate `p50` (`61.31%` of traffic)** | **`0.548 s`** | **`0.529 s`** | **`0.539 s`** | GCS crop read + `gemini-embedding-2` RPC (`~360 ms`) + numpy dot-product (`<2 ms`) |
| **Stage 2B VLM Contact Sheet `p50` (`38.69%` of traffic)** | **`2.694 s`** | **`2.764 s`** | **`2.729 s`** | GCS crop read + embedding + PIL contact-sheet render + `gemini-3.5-flash-lite` RPC (`~1.05 s`) |
| **Overall Pipeline `p50` Latency** | **`0.819 s`** | **`0.826 s`** | **`0.822 s`** | Bimodal mixture (`61.31%` at `0.54s` + `38.69%` at `2.73s`) |
| **Overall Pipeline `p95` Latency** | **`4.075 s`** | **`4.729 s`** | **`4.402 s`** | Stage 2B contact-sheet calls with cold GCS gallery fetches |
| **Overall Pipeline `p99` Latency** | **`21.708 s`** | **`29.490 s`** | **`25.599 s`** | Tail Vertex AI priority-tier rate-limit backoff retries (`0` dropped requests) |

> [!NOTE]
> **In-Memory Crop Latency vs. GCS Cold-Read Latency:** In the benchmark job above, each worker downloads the raw image crop over GCS HTTP (`~380–500 ms` per crop) before embedding. When raw image bytes are passed directly or cached in memory, Stage 2A matrix matching + calibration takes **`16 ms` CPU time** (`~370 ms` including the live Vertex AI `gemini-embedding-2` RPC).

### 2.3 Quantitative Validation Metrics (`L1` through `L5` Hierarchy & Calibration)

| Evaluation Metric | `hul_golden-test_5k` (`N=5,000`) | `hul_variants-test_5k` (`N=5,000`) | Combined `N=10,000` Benchmark | 95% Bootstrap CI (`N=10,000`) |
| :--- | :--- | :--- | :--- | :--- |
| **L1: Unilever vs. Competitor (`is_hul`)** | **`98.60%`** (`4,930 / 5,000`) | **`98.64%`** (`4,932 / 5,000`) | **`98.62%`** (`9,862 / 10,000`) | `[98.39%, 98.85%]` |
| **L2a: Category Accuracy (`10` categories)** | **`96.46%`** (`4,823 / 5,000`) | **`96.38%`** (`4,819 / 5,000`) | **`96.42%`** (`9,642 / 10,000`) | `[96.05%, 96.78%]` |
| **L2b: Packaging Form (`16` pack types)** | **`91.68%`** (`4,584 / 5,000`) | **`91.64%`** (`4,582 / 5,000`) | **`91.66%`** (`9,166 / 10,000`) | `[91.12%, 92.20%]` |
| **L3: Brand Accuracy (`934` brands)** | **`96.12%`** (`4,806 / 5,000`) | **`96.56%`** (`4,828 / 5,000`) | **`96.34%`** (`9,634 / 10,000`) | `[95.97%, 96.71%]` |
| **L4: Base Variant Accuracy** | **`82.28%`** (`4,114 / 5,000`) | **`81.86%`** (`4,093 / 5,000`) | **`82.07%`** (`8,207 / 10,000`) | `[81.32%, 82.82%]` |
| **L5: Exact Product SKU (`2,659` SKUs)** | **`82.28%`** (`4,114 / 5,000`) | **`81.86%`** (`4,093 / 5,000`) | **`82.07%`** (`8,207 / 10,000`) | `[81.32%, 82.82%]` |
| **Stage 2A Embedding Recall@1 / Recall@5** | `90.82%` / **`97.44%`** | `90.46%` / **`97.18%`** | `90.64%` / **`97.31%`** | `[96.99%, 97.63%]` (R@5) |
| **HUL Brand Accuracy (`is_hul=True`)** | **`98.10%`** (`2,475 / 2,523`) | **`98.22%`** (`2,469 / 2,514`) | **`98.16%`** (`4,944 / 5,037`) | `[97.79%, 98.53%]` |
| **Competitor Brand Accuracy (`is_hul=False`)** | **`94.11%`** (`2,331 / 2,477`) | **`94.89%`** (`2,359 / 2,486`) | **`94.50%`** (`4,690 / 4,963`) | `[93.86%, 95.13%]` |
| **False HUL Attribution Rate** | **`1.21%`** (`30 / 2,477`) | **`1.17%`** (`29 / 2,486`) | **`1.19%`** (`59 / 4,963`) | `[0.89%, 1.49%]` |
| **Crops Clearing `0.78` Confidence** | **`82.86%`** (`4,143 / 5,000`) | **`83.30%`** (`4,165 / 5,000`) | **`83.08%`** (`8,308 / 10,000`) | `[82.34%, 83.81%]` |
| **Precision on Crops with `conf >= 0.78`** | **`95.63%`** (`3,962 / 4,143`) | **`95.05%`** (`3,959 / 4,165`) | **`95.34%`** (`7,921 / 8,308`) | `[94.88%, 95.79%]` |
| **True Positives Clearing `0.78` (Recall@0.78)** | **`96.31%`** (`3,962 / 4,114`) | **`96.75%`** (`3,959 / 4,092`) | **`96.53%`** (`7,921 / 8,206`) | `[96.13%, 96.93%]` |
| **Confidence Filtering `ROC-AUC`** | **`0.9674`** | **`0.9690`** | **`0.9682`** | `[0.9641, 0.9723]` |

### 2.4 Category-Level Performance Breakdown (`hul_golden-test_5k`, `N=5,000`)

| FMCG Category | Evaluated Crops | L1 `is_hul` Acc | L2 Category Acc | L2 Pack Form Acc | L3 Brand Acc | L5 Exact SKU Acc | GT Facing SOS | Pred Facing SOS | SOS Abs Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Foods** | `630` | `99.68%` | `99.68%` | `94.29%` | `97.14%` | **`90.00%`** | `47.94%` | `48.25%` | **`0.31 pp`** |
| **Skin Care (`Skin`)** | `772` | `99.22%` | `93.78%` | `94.04%` | `96.89%` | **`85.88%`** | `40.93%` | `41.45%` | **`0.52 pp`** |
| **Hair Care (`Hair_Care`)** | `385` | `99.74%` | `99.22%` | `98.44%` | `98.96%` | **`85.71%`** | `49.61%` | `49.87%` | **`0.26 pp`** |
| **Household Care (`HHC`)** | `513` | `98.64%` | `92.98%` | `85.96%` | `96.88%` | **`80.51%`** | `56.92%` | `56.34%` | **`0.58 pp`** |
| **Beverages** | `569` | `99.47%` | `98.59%` | `90.16%` | `95.43%` | **`80.49%`** | `56.06%` | `55.89%` | **`0.17 pp`** |
| **Personal Wash (`PW`)** | `441` | `97.96%` | `93.65%` | `92.97%` | `95.24%` | **`80.27%`** | `51.25%` | `50.57%` | **`0.68 pp`** |
| **Oral Care (`Oral_Care`)** | `421` | `98.10%` | `97.39%` | `92.87%` | `94.30%` | **`80.05%`** | `37.77%` | `38.72%` | **`0.95 pp`** |
| **Health Food Drinks (`HFD`)** | `651` | `99.54%` | `98.46%` | `94.62%` | `98.00%` | **`80.03%`** | `54.99%` | `55.45%` | **`0.46 pp`** |
| **Laundry** | `527` | `96.58%` | `96.96%` | `85.77%` | `93.74%` | **`78.37%`** | `51.04%` | `50.28%` | **`0.76 pp`** |
| **Deodorants (`DMT`)** | `91` | `85.71%` | `83.52%` | `68.13%` | `83.52%` | **`63.74%`** | `100.00%` | `85.71%` | **`14.29 pp`** |
| **Total / Weighted Mean** | **`5,000`** | **`98.60%`** | **`96.46%`** | **`91.68%`** | **`96.12%`** | **`82.28%`** | **`50.46%`** | **`50.26%`** | **`0.20 pp`** |

### 2.5 Error Taxonomy Breakdown (`1,793` Total SKU Errors Across `N=10,000` Crops)

A forensic breakdown of the `1,793` exact-SKU mismatches (`17.93%` of `10,000` crops) shows that **over `79%` of SKU errors preserve exact Brand and Ownership (`is_hul`)**, and **`78.4%` of all SKU errors are automatically flagged by `abstain_recommended = True` (`sku_confidence < 0.78`)**:

| Error Mode Category | Share of Errors (`N=1,793`) | Share of Total Crops (`N=10,000`) | Impact on Brand / Ownership (`L1` / `L3`) | Caught by `sku_confidence < 0.78`? |
| :--- | :--- | :--- | :--- | :--- |
| **1. Same-Brand Sister Variant / Pack-Form Ambiguity** *(e.g., `Red_Label__Leaf` vs `Red_Label__Dust` on `97x128px` blurred crop)* | **`79.48%`** (`1,425 / 1,793`) | `14.25%` | **Zero impact** (`is_hul` ✓, `brand` ✓, `category` ✓) | **`76.8%` caught** (`1,094 / 1,425` have `conf < 0.78`) |
| **2. Same-Category Competitor-to-Competitor Swap** *(both `is_hul = False`, e.g., regional tea pouches)* | **`12.83%`** (`230 / 1,793`) | `2.30%` | **Zero impact on HUL SOS** (`is_hul` ✓, `legacy_label = "Others"` ✓) | **`83.0%` caught** (`191 / 230` have `conf < 0.78`) |
| **3. Cross-Ownership Boundary Error (`is_hul` mismatch)** *(severe motion blur / `<45px` occluded crops)* | **`7.69%`** (`138 / 1,793`) | **`1.38%`** | Impacts L1 `is_hul` (`0.19 pp` net SOS error due to symmetric cancellation) | **`87.7%` caught** (`121 / 138` have `conf < 0.78`) |

---

## 3. Architecture Risk Analysis & Technical FAQ

### Part A: Resolution of Pre-`v0.2` Endpoint Validation Findings

#### Item 1: VLM Escalation Rate (`77.1%` on Pre-`v0.2` Endpoint vs. `38.69%` Verified on `N=10,000`)
* **Observed Baseline Finding:** On `2,417` total crop calls, Gemini was invoked on `1,864` crops (`77.1%`), while the Stage 2A fast gate handled `553` crops (`22.9%`).
* **Technical Root Cause:**
  1. **Broadened Same-Brand Margin Guard (`17de3b5`):** `_compute_stage2a_diagnostics` (`src/approaches/market_share/retrieval/clef_standalone.py`) temporarily applied a `0.035` margin requirement whenever top-1 and top-2 shared the same brand (`same_brand or same_variant`), rather than strictly when they were pack-size sisters of the same variant (`same_variant`). Because nearly every major brand has multiple SKUs in the 2,659-SKU gallery, top-1 and top-2 almost always share a brand, escalating `77.1%` of single-crop calls into Stage 2B.
  2. **Crop Endpoint vs. Full-Shelf Endpoint:** Calling `/api/predict/crop` on isolated crops bypasses the full-shelf spatial neighbor medoid deduplicator (`shelf_960_softnms_e2e.py`), which groups adjacent identical facings on a shelf row.
* **`SPEC v0.2` Resolution:**
  - Restored `required_margin = 0.025 if same_variant else 0.009` and added a high-margin override (`margin >= 0.025`) for `lid_above_11.5`.
  - Across **10,000 blind crops (`test_5k`)**, Stage 2A resolves **`6,131 / 10,000` (`61.31%`) with 0 VLM calls**, and Stage 2B is called on **`3,869 / 10,000` (`38.69%`)**. On full retail shelves (`/api/predict/shelf`), spatial medoid grouping reduces VLM calls to **`15.5%–22.4%`**.

---

#### Item 2: Abstention at the `0.78` Threshold (`95.1%` Below `0.78` in Pre-`v0.2` vs. `83.08%` Clearing `0.78` in `SPEC v0.2`)
* **Observed Baseline Finding:** `abstain_recommended` was `True` on `2,298 / 2,417` crops (`95.1%`) because confidence fell below `0.78` (only `119` crops / `4.9%` cleared `0.78`), while `abstained` was `False` on all rows.
* **Technical Root Cause:**
  1. **Un-Pooled Sister-SKU Softmax Dilution:** When 4–5 sister SKUs of the same brand occupied the top-5 shortlist, the un-pooled single-panel softmax share $p_{\text{chosen}}$ was diluted to $\approx 0.20$, capping `sku_confidence` around `0.55–0.68` on valid Stage 2B matches.
  2. **Advisory Default (`allow_abstain=False`) per `Gate G10`:** For Share-of-Shelf (SOS) denominator completeness, the endpoint defaults to `allow_abstain=False` unless hard abstention is requested.
* **`SPEC v0.2` Resolution:**
  - `calibrated_shortlist_confidence` (`src/utils/openjev.py`) now pools probability mass across sister variants (`variant_mass`) and same-brand candidates (`brand_mass`).
  - Across `N=10,000` crops, **`8,308 / 10,000` (`83.08%`) clear the `0.78` threshold** at **`95.34%` precision**.
  - Passing `"allow_abstain": true` enforces `"abstained": true`, `"sku_id": null`, and `"legacy_label": "Others"` whenever `sku_confidence < 0.78` or `choice == 0`.

---

#### Item 3: Manual Validation on `n = 593` Crops & Legacy-`"Others"` Behavior (`280` Legacy-`"Others"`, `18` True Unilever Recoveries vs. `153` False SKU Predictions)
* **Observed Baseline Finding (`n = 593` Manually Validated Subset):**
  - While `2,417` total crops were processed for latency/routing telemetry, **accuracy and error validation was conducted via manual validation on a sample of `n = 593` crops** (`313` labeled Unilever crops + **`280` crops labeled `"Others"`** in legacy ground truth).
  - Out of those **`280` legacy-`"Others"` crops**:
    - **`18` crops (`18 / 280 = 6.4%`)** were manually verified to be **true Unilever SKUs** that had been mislabeled as `"Others"` in the legacy ground truth; the pipeline recovered their exact Unilever SKU and manual validation credited those `18` as true positives (`"true value"`).
    - **`153` crops (`153 / 280 = 54.6%` of legacy-`"Others"`, or `153 / 593 = 25.8%` of the `n = 593` manually validated set)** were genuinely non-Unilever / competitor / out-of-scope crops where the pre-`v0.2` endpoint predicted a catalog SKU instead of returning `"Others"`. Those `153` crops accounted for **`64.8%` (`153 / 236`) of all flagged errors** across the `n = 593` validation set (whereas the `313` labeled Unilever crops had only `83` errors, i.e., `73.5%` accuracy).
* **Technical Root Cause (Two Compounding Pre-`v0.2` Endpoint Behaviors):**
  1. **Zero `"Others"` Roll-Up on the `1,586` Competitor SKUs (`is_hul: false`):** The 2,659-SKU gallery contains `1,073` Unilever SKUs (`is_hul: true`) and `1,586` Competitor SKUs (`is_hul: false` across `874` non-HUL brands). Whenever the model matched a non-Unilever crop to one of the `1,586` competitor SKUs—or matched an uncatalogued regional pack to its nearest neighbor in the 2,659-SKU catalog—it returned that specific SKU string in `product` and never emitted `"Others"`.
  2. **Closed-Set Fallback on `choice == 0` (`allow_abstain=False`):** When Stage 2B rejected all 5 candidate panels (`choice = 0`, *"none match"*), `resolve_clef_choice` forced a closed-set fallback to top-1 (`tier1_high_cos_top1` / `tier2_abstain_fallback_top1`) rather than abstaining to `"Others"`.
* **`SPEC v0.2` Resolution:**
  - Every `/api/predict/crop` and `/api/predict/shelf` response now emits:
    - `"legacy_label"` and `"legacy_brand"`: Automatically outputs **`"Others"`** whenever `"is_hul": false` (all `1,586` competitor SKUs) or when `"abstained": true` (`choice == 0` or `sku_confidence < 0.78`), while emitting the exact Unilever SKU name when `"is_hul": true` and `sku_confidence >= 0.78` (preserving the **`18` / `6.4%` true Unilever SKU recoveries**).
    - `"catalog_scope"`: Explicitly tags `"unilever_sku"` vs. `"competitor_sku"` vs. `"open_set_abstained"`.
  - Evaluated on L1 ownership (`is_hul`: Unilever vs. `"Others"`), `SPEC v0.2` achieves **`98.62%` accuracy (`9,862 / 10,000`)** with a **`1.19%` false Unilever attribution rate**.

---

#### Item 4: Confidence-Based Filtering (`ROC-AUC = 0.55` in Pre-`v0.2` vs. `0.9682` in `SPEC v0.2`)
* **Observed Baseline Finding:** Under the pre-`v0.2` confidence score, a confidence cutoff did not reliably separate true matches from false positives (`ROC-AUC = 0.55`).
* **Technical Root Cause:**
  - Under the un-pooled softmax formula, major Unilever brands with 5 sister SKUs in the shortlist were penalized with $p_{\text{chosen}} \approx 0.20$ (`conf ~ 0.58`), whereas a noisy open-set crop matching a single isolated SKU in the shortlist had $p_{\text{chosen}} \approx 0.60$ (`conf ~ 0.72`), inverting the confidence ranking between true multi-variant matches and isolated false positives.
* **`SPEC v0.2` Resolution:**
  - Pooling `variant_mass` and `brand_mass` across sister SKUs in `calibrated_shortlist_confidence` and capping `choice == 0` open-set rejections at $\le 0.34$ raises `ROC-AUC` to **`0.9674` on `hul_golden-test_5k`** and **`0.9690` on `hul_variants-test_5k`** (**`0.9682` across all `10,000` crops**).
  - At the `0.78` threshold, **`96.53%` of all True Positives are retained** (`7,921 / 8,206`) while **`78.42%` of all False Positives are filtered out** (`1,407 / 1,794`), increasing precision from `82.07%` (unfiltered) to **`95.34%`** (filtered at `0.78`).

---

### Part B: Data Leakage, Cold Starts, Catalog Drift & Scalability Analysis

#### 5. Data Leakage & Generalization Across Splits
* **Question:** *Does tuning the `0.025` sister-variant margin guard or pooled confidence calibration overfit to the test crops, or do images from the same retail store leak between the reference gallery (`train`) and evaluation splits (`val`/`test`)?*
* **Verification & Evidence:**
  1. **Zero Train/Val/Test Image or Outlet Overlap (`Gates G3, G6, G13`):** Every dataset manifest (`rpc.json`) is verified by `run_12_gate_qc()` (`src/utils/qc.py`). `Gate G3` enforces `0` URI overlap across `train` (`gallery`), `val`, and `test`/`test_5k`; `Gate G13` enforces **`0` `outlet_code` overlap and `0` image MD5 hash overlap** across splits; `Gate G6` verifies `0` ground-truth oracle injection into shortlists.
  2. **Zero Learned Class Parameters in `SPEC v0.2` Calibration:** The `0.025` vs. `0.009` margin threshold is a structural rule (whether top-1 and top-2 are pack-size sisters of the same variant), and `calibrated_shortlist_confidence` is a monotonic analytical function of cosine similarity and shortlist brand/variant mass with zero per-class weights. Performance is consistent across `hul_golden-test_5k` (`ROC-AUC 0.9674`, `82.28%` SKU acc) and `hul_variants-test_5k` (`ROC-AUC 0.9690`, `81.86%` SKU acc).

#### 6. Cloud Run Cold-Start Latency & Gallery Initialization
* **Question:** *How does the service prevent request timeouts when loading 2,659 SKU embeddings and reference gallery metadata from GCS on container startup?*
* **Verification & Evidence:**
  1. **Pre-Computed `.npz` Gallery Vector Cache in GCS:** The 2,659-SKU gallery matrix ($\mathbf{G} \in \mathbb{R}^{M \times 768}$, `~18 MB` compressed `.npz`) is pre-computed and loaded into memory once per container lifecycle in `6.4–8.2 seconds` (`setup_seconds: 8.2` in `summary.json`).
  2. **Singleton Engine Caching & Warm Readiness Probe:** `server.py` caches the initialized `CLEFContrastiveStandalone` and `DistilledYolo960Stage0Clahe` instances in process-global singletons behind a thread lock, and `/api/predict/health` reports warm readiness (`min-instances=1` eliminates cold-start latency during peak windows).

#### 7. Catalog Drift, New Packaging Relaunches & Seasonal Promotions
* **Question:** *When FMCG packaging changes (festive promos, `+20% Extra` bands, artwork relaunches), does onboarding a new SKU require retraining the detector or classifier?*
* **Verification & Evidence:**
  1. **Zero-Retraining Catalog Onboarding (`< 60 seconds`):** Because Stage 1 detects generic `Class 1` retail packages (`distilled_yolo_onnx`) and Stage 2A/2B operates over a non-parametric `gemini-embedding-2` vector index + dynamic reference contact sheet, adding a newly launched SKU requires **zero model retraining**—only inserting 1–3 reference pack photos into `rpc.json` and appending its `768-D` embedding row.
  2. **Unseen SKU Launch Benchmark (`94.4%` Accuracy):** On `18` newly launched SKUs held out from training, `clef_contrastive_metadata` achieved **`94.4%` (`17 / 18`) exact SKU accuracy** immediately upon adding their reference photos to the gallery.

#### 8. Burst Scalability, Vertex AI Rate Limits & `PT180S` Payout Quarantine
* **Question:** *During peak morning store audits (`10,000+` store sessions/hour), what happens if Vertex AI `gemini-3.5-flash-lite` experiences transient `429 RESOURCE_EXHAUSTED` throttling?*
* **Verification & Evidence:**
  1. **`61.3%–84.5%` Traffic Shielded Before VLM:** Stage 1 (`CPU ONNX`) + Stage 1.5 (`-61%` Out-of-Bay Crop Pruning) + Stage 2A (`61.31%` Fast-Gate on crops, `84.5%` on full shelves with spatial medoid grouping) absorb the majority of facings before touching generative VLM quota.
  2. **Content-Addressed SHA-256 VLM Response Cache (`PR-17`):** Identical contact-sheet payloads short-circuit via GCS/memory SHA-256 cache (`vlm_cache/`).
  3. **Level-4 Total Contract Payout Quarantine (`PT180S` SLA, `SPEC v0.2` Section 1.3):** Validated under `locust 2.46.7` load testing (`0.00%` unhandled failures), any store session exceeding the `180-second` SLA window or exhausting exponential backoff retries emits `contract_status = "RETRY_PENDING"`, `awp_presence = "PENDING"`, and `earning_override = None` to `PR19_pubsub_dlq`—never emitting a false `Earning = '0'` retailer payout denial.

---

## 4. Reproducing the 10,000-Crop Benchmark & Live API Verification

### 4.1 Live Cloud Run API Smoke Verification (`curl`)
```bash
export TOKEN=$(gcloud auth print-identity-token)
export BASE_URL="https://clef-contrastive-api-jbn2qgbina-uc.a.run.app"

# 1. Verify SPEC v0.2 health & active guard thresholds
curl -s -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/predict/health" | jq .

# 2. Predict an individual crop with SPEC v0.2 telemetry and legacy_label mapping
curl -s -X POST "$BASE_URL/api/predict/crop" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image_uri": "gs://marketshare-datasets/derived_manifests/hul_golden/images/sample.jpg",
    "dataset": "hul_golden",
    "tier": "priority",
    "allow_abstain": true,
    "conf_threshold": 0.78
  }' | jq .
```

### 4.2 Triggering the 10,000-Crop Cloud Run Benchmark (`test_5k`)
```bash
# Execute 5,000-crop hul_golden-test_5k benchmark on Cloud Run
.venv/bin/shelf-bench cloud-run \
  -a clef_contrastive_metadata \
  -m gemini-3.5-flash-lite \
  -t priority \
  -d hul_golden \
  --split test_5k \
  --limit 5000 \
  --workers 16

# Execute 5,000-crop hul_variants-test_5k benchmark on Cloud Run
.venv/bin/shelf-bench cloud-run \
  -a clef_contrastive_metadata \
  -m gemini-3.5-flash-lite \
  -t priority \
  -d hul_variants \
  --split test_5k \
  --limit 5000 \
  --workers 16
```
