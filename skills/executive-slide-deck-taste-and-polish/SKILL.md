---
name: executive-slide-deck-taste-and-polish
description: >-
  Executive slide deck taste, plain-English storytelling, visual-journey layout,
  and deck-wide elevation skill for technical architecture, ML benchmark, and
  cloud FinOps presentations. Enforces 12 deck-wide principles: collaborative
  "proposed alternative" framing, solution- and mechanism-first narrative ("what
  we solved and how we did it"), plain English free of ML/ticket jargon,
  single-currency USD standardization, one-line pillar definitions with
  asterisked validation footnotes, full-width legible hero diagrams and real-image
  journeys, dedicated decision-model and smart edge-case slides, step-by-step
  FinOps cost bridges, and priority-driven next-steps tables without artificial
  calendar windows.
---

# Executive Slide Deck Taste, Visual Journey & Polish Skill

This skill codifies the narrative, visual, and editorial principles required to elevate dense engineering, machine learning, and cloud architecture decks into clear, self-explanatory executive presentations. Apply these rules across **every slide** in a presentation—never treat slide feedback as isolated point fixes.

---

## 1. Twelve Deck-Wide Elevation Principles

### P1. Collaborative, Non-Displacement Framing ("Proposed Alternative")
* Frame new architectures as a **proposed unified alternative** rather than attacking or "replacing" existing systems.
* **Avoid:** *"4-Stage Hybrid Funnel Replaces 14 Existing Models"* or *"Legacy Stack Fails on Size Heads"*.
* **Prefer:** *"Proposed 4-Stage Hybrid Alternative: Doubles 5K SKU Accuracy at 58% Lower Cloud Spend"* and *"How the Existing 14-Model Pipeline Works & Where Resolution Limits Accuracy"*.

### P2. Solution-First & Mechanism-First Narrative ("What We Solved & How We Did It")
* Lead with **what business/technical problem we solved** and **the physical mechanism of how we solved it**, rather than dwelling on existing system flaws.
* Structure executive summaries around:
  1. **What We Solved:** Accurately reading fine-print pack sizes (`50g` vs. `100g`), eliminating photo timeout drops (`0%` dropped shelves), and onboarding new product launches in `<60 seconds` without GPU retraining.
  2. **How We Did It:** (a) High-resolution detection + physical shelf-row grouping, (b) Fast vector matching against multi-angle reference packshots for `61%–68%` of clear packs in `<1 ms`, and (c) Side-by-side visual tie-breaking + zoomed pack-weight verification for the hardest `32%–39%` lookalike crops.

### P3. Plain English Over ML Jargon & Internal Ticket/Spec Codes
* **Never put internal ticket IDs, spec codes, or raw ML/math jargon in slide titles, KPI cards, or executive bullets.** Every headline and card must be immediately understood by a business or engineering leader reading the slide cold.
* **Jargon-to-Plain-English Translation Table:**

| Cryptic Jargon / Internal Code | Plain-English Executive Replacement |
| :--- | :--- |
| `Closes Timeout Defect IST0001092` | `Async Queue Eliminates Photo Timeouts (0% Dropped Shelves)` |
| `LLD001 v0.2` / `PR19 Pub/Sub` | `Updated Production Service` / `Asynchronous Request Queue` |
| `15-Gate Audit Purges 408K+ Leaks` | `4-Step Data Cleaning Removes 408K Duplicate & Flipped Test Photos` |
| `Size-Head Collapse (-29.6 pp)` | `224x224 Image Resize Blurs Small Pack-Weight Text (-29.6 pp on Size)` |
| `Dual-Buffer CLAHE / Shades-of-Gray` | `Lighting Cleanup: De-glares Dark Shelves While Keeping True Pack Colors` |
| `Residual InfoNCE / K=3..8 Medoids` | `Multi-Angle Reference Gallery (3–8 Views per SKU) with Lookalike Separation` |
| `Saliency-Diff / 2x2 Crown Sheet` | `Side-by-Side Visual Comparison with 2x Zoom on Cap & Weight Band` |
| `Open-Set Dustbin (τ0=0.62)` | `Competitor Filter Routes Non-Catalog Brands Cleanly to "Others"` |
| `y_plank Physical Plank Law` | `Shelf-Row Grouping Verifies Products Sit on Real Shelf Rows` |
| `pHash ≤ 6b / MD5 Cross-Split Dupes` | `Byte-Identical & Visually Identical Photos Separated Across Train and Test` |

### P4. Single-Currency Standardization (`USD $` Only)
* **Never mix currencies** (e.g., mixing `₹ Lakhs/Crores` with `$ USD`) across slides, KPI cards, tables, or embedded charts.
* Standardize **100% of financial metrics, unit receipts, monthly portfolio budgets, and annual savings to `USD ($)`** (e.g., `$23.5K/mo` vs. `$56.5K/mo` existing, saving `$33.0K/mo` or `$396K/year` at `-58.4%`; `$0.0013–$0.0016` per shelf photo).
* Even when citing product packaging examples in prose, prefer gram/ml or percentage examples (`50g vs. 100g`, `50% Extra`) over local currency symbols so financial vs. packaging numbers are never confused.

### P5. One-Line Task/Pillar Definitions + Asterisked Validation Footnotes
* In scorecard and benchmark tables:
  1. **Add a short plain-English description** on each row explaining what the task or pillar measures (e.g., `Pillar 1 — Display Stand Detection: Verify branded display rack & filter out counter clutter*`).
  2. **Remove parenthetical sample counts** (`(60,308 crops)`, `(412 bays)`, `(17.8K)`) from task names inside table cells; mark tasks with an asterisk (`*`) and state the validation dataset sample sizes cleanly in the bottom slide footnote (`*Validated on: 60,308 5K crops, 17,842 1K active crops, 10,000 blind Cloud Run crops...`).

### P6. Step-by-Step Visual Enrichment & Diagnostic Journeys
* When presenting **Data Cleaning / Enrichment** or **Existing Baseline Diagnostics**, never list disconnected bullets.
* Present a **connected left-to-right visual journey (`Step 1 → Step 2 → Step 3 → Step 4`)** with explicit arrows showing:
  * **What raw state started at the beginning** (`5.14M raw files across 7 disconnected folders`),
  * **What concrete transformation happened at each step** (merged folders → standardized 18 container shapes → removed 408K duplicate/flipped test images → separated stores between train and test), and
  * **What measurable evidence proves the improvement** (e.g., before/after record trace and `+170%` usable shelf facings).

### P7. Large, Legible, Self-Explanatory Hero Diagrams Linked to Improvements
* **Never squeeze an architecture or retrieval diagram into a tiny corner box (`<250 pt` wide) next to disconnected text cards.**
* Make the architecture diagram a **large, high-DPI, full-width or half-canvas hero visual** with `>= 11–14pt` equivalent text inside nodes.
* **Link every stage box in the diagram directly to the engineering improvements (`Stage 0`, `Stage 1`, `Stage 2A`, `Stage 2B`, `Stage 3`) and quantitative lift**, so a viewer can understand the entire pipeline by reading the diagram alone.

### P8. Full-Canvas Visual Journey Storyboards on Real Store Images
* For **End-to-End Image Journey slides** (Market Share Shelf Journey & Merchandising 3-Pillar Journey):
  * Make the **visual trace of real store photos (`Step 1 → Step 5`) the primary full-width centerpiece** of the slide.
  * Place numbered narrative callout cards directly above or below the corresponding visual stage in the image journey so the boxes and images tell **one unified end-to-end story**.

### P9. Dedicated Decision-Model, Tie-Breaker & Smart Edge-Case Slide
* Any multi-stage AI funnel deck must include a dedicated slide answering three questions in plain English with **3 real crop visual examples**:
  1. **When does the Fast Vector Path decide immediately (`~61%–68%` of crops, `<1 ms`)?** High visual similarity (`>= 0.75–0.82`) and clear margin (`>= 0.025–0.060`) with no sister-size conflict.
  2. **What triggers the Visual Tie-Breaker for sure (`~32%–39%` of crops sent to VLM)?** Triggered deterministically whenever: (a) top candidates are **same-brand sister sizes or variants** (`50g vs. 100g`), (b) top-1 vs. top-2 vector margin is tight (`< 0.025`), or (c) similarity is in the grey zone (`< 0.75`).
  3. **What smart calls does the model make on edge cases?**
     * **Edge Case 1 (Same-Brand Size Twins):** Zooms `2x` onto the bottom weight strip, reads printed `g/ml` weight, and overrides vector #1 if the printed weight matches candidate #2.
     * **Edge Case 2 (Blurry/Tilted Pack — Two-Level Abstention):** Locks in Brand & Category (`99.3%` precision) so Share-of-Shelf stays accurate (`1.25%` error) while flagging `variant_abstain_recommended = true` below `0.78` confidence.
     * **Edge Case 3 (Lookalike Competitor Brand — Open-Set `"Others"` Roll-Up):** Matches against `874` rival brand profiles + open-set rejection and outputs `"Others"` (`98.6%` ownership accuracy) instead of guessing a false catalog SKU.

### P10. Clear FinOps Cost-Reduction Bridge (Zero Irrelevant Operational Trivia)
* Structure Cloud FinOps slides around a **clear 4-step cost bridge in USD (`$`)** explaining *why* cost drops (`$56.5K/mo → $23.5K/mo`, `-58.4%`):
  1. Retire 14 always-on GPU endpoints (`-$31.2K/mo`) in favor of a single CPU detector (`$0.00018/photo`).
  2. Resolve `61%–68%` of packs via fast vector lookup (`$0` VLM cost), invoking Gemini Flash-Lite on only `32%–39%` of crops (`$0.0013–$0.0016/photo` total AI).
  3. Eliminate standalone Document AI OCR (`-$3.2K/mo`).
  4. Apply contracted GCP Enterprise Agreement discounts (`-20%` to `-25%`).
* **Cut operational trivia** (such as daytime IST store upload hours or RPS micro-sizing) that distracts from the financial narrative.

### P11. Plain-English Production Safeguards & Load Verification
* Frame production readiness around concrete engineering outcomes:
  * **Zero Timed-Out Photos:** Asynchronous queue decouples burst uploads from inference so `100%` of store photos complete (`2.69s` median latency at `80` parallel workers, `0.00%` errors).
  * **Content-Addressed Caching (`-22%` Calls):** 24-hour SHA-256 image cache returns instant results on duplicate store uploads.
  * **Calibrated `0.78` Confidence Gate (`ROC-AUC = 0.968`):** Separates high-confidence SKU matches (`95.3%` precision) from blurry crops.

### P12. Prioritized Next Steps & Asks Table (Zero Artificial Calendar Windows)
* On the closing Next Steps & Asks slide, **never invent `Week 1 (Days 1–7)` calendar timelines**.
* Present a clean **Prioritized Asks Table** with columns:
  * **`Priority` (`P0 — Immediate`, `P1 — High`, `P2 — Optional`)**
  * **`Specific Ask / Action Item`**
  * **`What Is Needed`**
  * **`Why It Matters & Quantitative Impact`**
  * **`Readiness Status`**

---

## 2. Pre-Delivery Visual & Editorial Audit Checklist

Before concluding any slide deck build or polish pass, export PNG thumbnails of every slide (`gslides readonly export-thumbnail`) and visually verify all 8 gates:

| Gate | Audit Check | Pass Criterion |
| :--- | :--- | :--- |
| **G1. Plain-English Titles** | Read all slide titles in sequence | Zero ticket IDs (`IST...`), spec codes (`LLD001`), or unexplained ML acronyms; titles form a connected executive story. |
| **G2. Collaborative Tone** | Check all comparisons to existing stack | Uses `"existing"` and `"proposed alternative"`; zero `"legacy"` or `"replaces"`. |
| **G3. USD Currency Purity** | Scan all text boxes, tables, and charts | `100%` of cost figures in `USD ($)`; zero `₹`, `Lakhs`, or `Crores` anywhere on slides or inside chart PNGs. |
| **G4. Hero Visual Legibility** | Inspect all embedded diagrams & journeys | Diagrams occupy `45%–65%` of canvas width/height with large, crisp labels (`>= 1800px` source PNGs) directly linked to Stage numbers. |
| **G5. Scorecard Clarity** | Inspect all benchmark tables | Every task/pillar has a 1-line plain-English explanation; validation sample sizes sit in an `*` footnote. |
| **G6. Decision Model & Edge Cases** | Verify dedicated Decision Model slide | Explains Fast Path vs. Mandatory VLM Tie-Breaker with 3 real crop examples + 3 smart edge-case calls. |
| **G7. Zero Overlap / Clipping** | Inspect exported PNG thumbnails | Zero text overflow, overlapping shapes, or clipped table cells. Bottom-half tables (`y >= 200pt`) contain at most `5–6` single-line data rows (`~22.5pt` native Google Slides row height) so they end well above the `y=391pt` footer. |
| **G8. Comment Resolution** | Check open slide comments (`list-comments`) | Every reviewer comment is addressed on the canvas, replied to with a summary of changes, and resolved. |

