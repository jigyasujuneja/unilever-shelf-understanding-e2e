---
name: spatial-detection-and-planogram-physics
description: >-
  Playbook for dense small-object and structural-zone detection, physical shelf-plank
  support filtering, Pre-Embedding Class-0 Bay Gating (-61% background crop pruning),
  spatial medoid deduplication, and 2D spatial-graph planogram, out-of-stock (OOS),
  and promotional talker compliance auditing.
---

# Spatial Detection, Physical Support Laws & Planogram Compliance Playbook

## 1. Dual-Class CPU ONNX Distillation Over Single-Pass VLM Boxes (`EXP-24`, `EXP-55`)

1. **Why Single-Pass VLM Bounding Boxes Fail on Dense Scenes**:
   - Direct VLM bounding-box generation drifts on scenes with `30-189` items (`~51.2%` F2, `15.4s` latency).
2. **Distilled Dual-Class CPU ONNX (`yolo11n-onnx`, `<100 ms`, `$0` Token Cost)**:
   - Train a dual-class detector separating structural containers (`Class 0`: branded display bays, coolers, hanger strips) from individual items (`Class 1`: product facings).
   - Run `960px` high-resolution sub-window inference with aspect-aware Gaussian Soft-NMS ($\sigma = 0.55$) so vertically hanging sachets and tightly packed bottles are not suppressed.
3. **Density-Triggered $2\times 2$ Quadrant Tiling (`EXP-53`)**:
   - When initial full-scene detection count exceeds the dense-wall threshold (`>= 65` boxes), trigger overlapping $2\times 2$ quadrant tiling with Soft-NMS seam deduplication to capture `100-189` pack walls.

## 2. Physical Shelf-Plank Support Law & Pre-Embedding Bay Gate (`EXP-50`, `EXP-53`, `EXP-57`)

1. **Physical Shelf-Plank Support Law (`EXP-53`)**:
   - Cluster item bottom edges ($y_{\max}$) into horizontal physical shelf planks ($\text{IoU}_y \ge 0.45$).
   - Reject unplanked 2D printed product illustrations on display headers/wings and filter back-row depth shoulders peeking behind front-row facings.
2. **Stage 1.5 Pre-Embedding Class-0 Display-Bay Gate (`EXP-57`, `-61%` Compute Pruning)**:
   - On fixture-scoped merchandising audits (`WIN`, `CTU`, `MB2`, `MB3`), filter `Class 1` product detections to those geometrically contained within `Class 0` display bays **before** calling Stage 2 embeddings or VLM contact sheets, eliminating **61% of background shelf crop embeddings**.
3. **Aspect-Preserving `+12%` Cap/Rim Padding & Projective Scale Priors (`EXP-50`)**:
   - Expand top crop boundaries by `+12%` to preserve bottle caps, spray nozzles, and tube crimps.
   - Compute projective scale priors combining box aspect ratio ($h/w$), top-25% crown taper, and plank-relative height ($h / h_{\text{plank}}$) to separate `50g` vs. `100g` vs. `200g` vs. `500g` packs (`EXP-23`).
4. **Spatial Neighbor Medoid Deduplication (`3.5x` Call Compression)**:
   - Within each horizontal shelf row, group adjacent facings with color/embedding cosine $\ge 0.92$ into a single medoid query and propagate the verified SKU label across all cluster members.

## 3. 2D Spatial-Graph Planogram, OOS Gap & Talker/Promo Engine (`EXP-42`, `EXP-46`, `EXP-47`, `EXP-48`)

1. **Fixture Archetype Conditioning (`AWP_Code`, `63` Studio Templates)**:
   - Match detected `Class 0` bays against the `63`-template `AWP_Code` studio reference index (`WIN`, `CTU`, `MB2`, `MB3`, `FULLWALL`, `HAN` hanger fallback) to load expected shelf count, brand-block sequence, and mandatory SKU assortment.
2. **Brand-Block Purity ($W_5$), Assortment ($W_4$), and Out-of-Stock (`OOS`) Gaps**:
   - Measure horizontal empty shelf gaps between adjacent `Class 1` boxes on each plank ($\Delta x > 1.25 \times \overline{w}_{\text{row}}$) to flag Out-of-Stock facings.
   - Compute contractual store payout weight $W_1 = \frac{W_4 \times W_5}{100}$ alongside area-weighted and count-weighted Share of Shelf (`SOS`).
3. **3-Class Shelf Talker, Price Tag & Promo Callout Verification (`EXP-47`, `EXP-48`)**:
   - Detect shelf talkers, price rail tags, and on-pack promo bands (`Class 0/1/2` on `hul_talkers_golden`), then apply a targeted top/bottom 25% zoom strip to extract discount percentages, price-off callouts, and multipack offers.
