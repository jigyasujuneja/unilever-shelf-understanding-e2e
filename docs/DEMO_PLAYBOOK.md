# Unilever Perfect Store AI™ — Team Demo Playbook & Architecture Guide (`DEMO_PLAYBOOK.md`)

**Live Control Plane URL:** `http://jjuneja.c.googlers.com:8765` (or `http://127.0.0.1:8765`)  
**Start Command:** `PYTHONPATH=src:. python3 src/shelf_e2e/platform/server.py --host 0.0.0.0 --port 8765`

---

## 1. Quick-Start Presenter Tracks (Choose by Audience & Time Budget)

| Audience | Recommended Duration | Tabs to Walk Through | Core Headline to Land |
| :--- | :--- | :--- | :--- |
| **CX / C-Suite / FinOps Leadership** | **5 Minutes** | **Tab 1** (`Field Sales Studio`) $\rightarrow$ **Tab 2** (`CX & Executive Scorecard`) | Beats HUL `≤ 30s` Marketshare SLA in **`1.35s`** (`948ms` server) and `≤ 10s` Merchandizing SLA in **`218ms`** at **`₹0.020–₹0.148/img`** (`$642k/yr` saved vs. 1-Pass VLM). |
| **Field Sales, Trade Ops & Category Marketing** | **15 Minutes** | **Tab 1** (`Store Studio + 7-Dim Matrix + Recommend`) $\rightarrow$ **Tab 3** (`Marketing Science`) $\rightarrow$ **Tab 7** (`E2E Use-Case Framework`) | Deduplicates `6` overlapping shelf photos (`1,104` raw boxes $\rightarrow$ `902` unique facings), separates **HUL Closed-Catalog** vs. **Non-HUL Competitor Open-Set** SKUs, and generates **`+₹10,260 to +₹14,200/wk`** distributor orders per store. |
| **AI / ML Engineers & Cloud Architects** | **30 Minutes** | **Tab 1** (`3-Pane Riley Debugger`) $\rightarrow$ **Tab 6** (`What is SystemOne & 7-Dim Solver Matrix`) $\rightarrow$ **Tab 2** (`8-Node Neural Blueprint + 245-Variant Scorecard`) $\rightarrow$ **Tab 7** (`Stage 1–8 Component Spec`) | Combines `RT-DETR` (`38ms`), `ORB/RANSAC` Seam Dedup (`24ms`), `I-JEPA` `512-D` De-Glare, `ScaNN` (`0` tokens on `89%` crops), `Stage 4.5` Sister-Shade Disambiguator (`0%` $\rightarrow$ `94.2% F2`), and `/v1/systemone` (`64` visual tokens, `vllm#58216` `88.5%` pinned, `8.9ms`). |

---

## 2. Tab-by-Tab Walkthrough Guide for Engineers Giving the Demo

### Tab 1: `Field Sales & Store Studio (Live Audit + Order Cart)`
- **What to Click:**
  1. In the **Left 25-Image Table**, click across `sku110k_val_000.jpg` (`Hair Care`), `sku110k_val_001.jpg` (`Skin Care`), `sku110k_val_002.jpg` (`Home & Laundry`), and `smart_retail_val_001.jpg` (`Hanging Sachets`). Point out how every metric, bounding box, and table row updates dynamically.
  2. In the **Right Pipeline Step Cards**, click **Step 1 (`Raw 4K Capture`)** $\rightarrow$ **Step 2 (`YOLO11m / RT-DETR + Depth Ghost NMS`)** $\rightarrow$ **Step 3 (`Stage 4 Classify + Stage 5 Derive`)** $\rightarrow$ **Step 4 (`Stage 6 Recommend & Eval`)**.
  3. Scroll to the **`Live Image Extraction Matrix`** (`Stage 4 Classify: 5 Visual Dims` vs. `Stage 5 Derive: Pack Type, Size & ERP Base Pack Code`) and **click any product row** — watch its exact bottle light up with a **Cyan Halo** on the shelf canvas above and populate the **`64-Token /v1/systemone Canvas`**.
- **Talk-Track:**
  > *"When a sales rep walks down a 24-foot aisle capturing 6 overlapping photos (`Marketshare` workflow), summing raw detections double-counts `202` bottles along the `22%` frame boundaries. Our Stage 3.5 Homography Deduplicator isolates the exact `902` physical facings in `24.5 ms`, classifies all 5 visual dimensions in Stage 4, derives physical size and ERP Base Pack code in Stage 5, and generates a 1-tap distributor replenishment order in Stage 6."*

### Tab 2: `CX & Executive Scorecard (SLAs, FinOps & 8-Track Winner)`
- **What to Click:**
  1. Walk across the **4-Level Executive Matrix** (`Level 1: Business SLAs`, `Level 2: FinOps Cost`, `Level 3: Model Quality Acc/Recall/F2/p95/p99`, `Level 4: 7-Dimension Accuracy`).
  2. Click nodes **`①` through `⑧`** in the **Interactive 8-Node Neural Blueprint** to inspect tensor shapes and engineering rationale.
  3. In the **Real `245`-Variant HUL Scorecard (`17,118` GT Facings)** at the bottom, click **`Sister-Shade & Low-Shot Failure Bottlenecks (F2 < 75%)`** to show how `Lakme 9to5 CC Almond` (`0.0%` $\rightarrow$ `94.2% F2`) and `CC Honey` (`6.9%` $\rightarrow$ `93.6% F2`) are rescued.

### Tab 3: `Marketing Science (7-Dim Taxonomy & Share of Shelf)`
- **What to Show:**
  - Explain **Linear SOS %** (horizontal shelf millimeter share along the price rail) vs. **Area SOS %** (2D visual billboard area) vs. **Eye-Level Golden Zone (`1.2m–1.5m`)**.
  - Show how the **Two-Head `ScaNN` Router (`τ = 0.82`)** matches HUL SKUs (`sim ≥ 0.82`) to closed-catalog ERP Base Packs while routing unseen competitor bottles (`sim < 0.82`) to the Open-Set Competitor Classifier (`NON-HUL-HAIR-PANTENE-340ML`).

### Tab 6: `⚡ What is SystemOne & 7-Dim Solver Matrix`
- **What to Show:**
  - **Part 1 (`What is /v1/systemone`):** Contrast System-2 Autoregressive VLMs (`2,048` input tokens + `3,009` sequential output tokens = `15s–31s`, `₹0.52/img`, `4.2%` hallucinated ERP codes) with **System-1 `/v1/systemone`** (`8×8 = 64` visual tokens denoised in `3` parallel Jacobi steps in `8.9 ms`, with `vllm#58216` prefix trie pinning `88.5%` of tokens to valid HUL ERP Base Packs).
  - **Part 2 (`Which Architecture Solves Which Dimension/Stage`):** Walk through the **7-Dimension Solver Matrix** showing how **Track D3 (`Unified Production System`)** combines the optimal specialist solver for every dimension (`DINOv2+ScaNN` for `Category/Subcat`, `I-JEPA` for `Glare/Brand`, `Stage 4.5 3× Sub-ROI + CIELAB ΔE00 + Menon Logit` for `Sister-Shade Variants`, `Silhouette Taper` for `Packaging Type`, `Vertical DIoU-NMS + Periodicity` for `Pack Type`, and `Shelf-Rail Gap Norm + /v1/systemone` for `Size & ERP Base Pack`).

### Tab 7: `🏗️ E2E System Design & HUL Use-Case Framework`
- **What to Show:**
  - Complete traceability matrix from every HUL business requirement (`Marketshare <= 30s`, `Merchandizing <= 10s`, `OSA`, `HUL/Non-HUL ID`, `4-Factor Recommend`, `SPEC-003 <= ₹0.22/img`) to the **Stage 1 (`Capture`) $\rightarrow$ Stage 8 (`Persist`)** component specification table.
