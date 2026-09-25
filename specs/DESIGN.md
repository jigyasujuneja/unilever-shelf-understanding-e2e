# SPEC-001: Retail Shelf Understanding End-to-End Prototype & Benchmark (`DESIGN.md`)

## 1. System Intent & Constraints
- **Goal:** Ingest 4K retail shelf images, localize SKUs, resolve Base Pack codes, and audit promotional compliance across 4 Modern Trade use cases (`MarketShare`, `Merchandising`, `Toker Compliance`, `SOS`).
- **Hard Cost Ceiling:** $\le$ ₹0.22 (~$0.0026 USD at 1 USD = 84 INR) per image at 500k/day scale.
- **Hard Latency SLA:** P95 $\le$ 20.0s (`20000.0 ms`, leaving 10s buffer for mobile network uplink).
- **Concurrency Target:** 525 parallel worker threads without rate-limit saturation.

## 2. Public Benchmark Datasets
- **Dense Shelf Imagery:** `SKU-110k` (11,762 images, avg 147 SKUs/image) for spatial detection and Share-of-Shelf (SOS) linear calculation.
- **Product Catalog Reference:** `Retail Product Checkout (RPC)` dataset (200 categories, 83k images) for catalog vector search and zero-shot SKU onboarding.

## 3. Universal I/O Contract (JSON Schema)
Every pipeline candidate MUST accept and emit the identical JSON payload:

### Input Contract
```json
{
  "image_path": "str",
  "store_metadata": {
    "store_id": "str",
    "channel": "MODERN_TRADE",
    "planogram_id": "str"
  },
  "planogram_contract": {
    "target_skus": ["list[str]"],
    "promo_rules": {
      "toker_text": "str",
      "min_display_count": "int"
    }
  }
}
```

### Output Contract
```json
{
  "metrics": {
    "total_detected": "int",
    "share_of_shelf_pct": "float",
    "latency_ms": {
      "tier1_detection": "float",
      "tier2_catalog_match": "float",
      "tier3_compliance": "float",
      "total_e2e": "float"
    },
    "estimated_cost_inr": "float"
  },
  "marketshare": {
    "resolved_skus": [
      {
        "box_xyxy": ["float", "float", "float", "float"],
        "base_pack_id": "str",
        "confidence": "float"
      }
    ],
    "red_line_gaps": ["list[str]"],
    "width_pack_gaps": ["list[str]"]
  },
  "compliance": {
    "toker_status": "COMPLIANT | NON_COMPLIANT",
    "display_status": "COMPLIANT | NON_COMPLIANT",
    "coaching_message": "str"
  }
}
```

## 4. Competing Architecture Candidates (The 4 Tracks)
1. **Track A (`track_a_cascading_vit` — Cascading ViT Baseline):** `RT-DETR` Detection $\rightarrow$ `ViT-B-16` Brand Classifier $\rightarrow$ `InceptionNet` Variant Classifier.
2. **Track B (`track_b_e2e_vlm` — End-to-End VLM):** Direct prompt to `Gemini 2.5 Flash Lite` passing raw image + planogram rules.
3. **Track C (`track_c_tiered_hybrid` — Tiered Hybrid Engine, FDE Recommended):** `RT-DETR` (Cloud Run GPU) $\rightarrow$ Vertex AI Multimodal Embeddings + `ScaNN` Vector Search $\rightarrow$ `Gemini 2.5 Flash Lite` for promo crops only.
4. **Track D (`track_d_jev_routing` — `Jev` & Vector Routing + `GeminiDiffusion-as-Jev`):** Tier 1 `RT-DETR` $\rightarrow$ Tier 2 `ScaNN` Vector Search $\rightarrow$ `Jev` Deterministic State Machine (or `GeminiDiffusion-as-Jev` latent de-glaring & spatial refinement) for Base Pack schema resolution $\rightarrow$ `Gemini 2.5 Flash Lite` for promo reasoning.

---

## 5. Stage 4.5: 5-Stage Sister-Shade & Low-`F2` Disambiguation Engine (`sister_shade_disambiguator.py`)
Global `224x224` ViT/ScaNN embeddings average `99%` of the shared brand bottle/tube body (e.g., `Lakme 9to5 CC Cream` gold cap + peach body) and drown out the `8px` shade/SPF badge (`Almond`, `Honey`, `Beige`, `Bronze`), causing majority-class attractor collapse (`CC_Almond = 0.0% F2`, `CC_Honey = 6.9% F2`). Whenever `ScaNN` Top-1 vs Top-2 cosine margin $< 0.045$ within the same `(Category, Subcategory, Brand)` cluster, **Stage 4.5** executes 5 deterministic micro-interventions:
1. **3× Discriminative Sub-ROI Spatial Zoom (`[0.62H : 0.88H]`):** Crops the lower-middle shade/SPF micro-band and upscales `3×` via Lanczos/super-resolution (`60px -> 180px`) so `8px` typography becomes `24px`.
2. **Specular-Invariant CIELAB Chromaticity (`ΔE00`):** Separates `Almond` (`L*=74.2, a*=7.1, b*=19.4`) from `Honey` (`L*=66.8, a*=11.8, b*=27.2`) and `Bronze` (`L*=51.5, a*=16.9, b*=29.8`) while down-weighting `L*` by `0.65` to reject supermarket LED glare.
3. **Menon Post-Hoc Logit Adjustment for Tail Classes:** Applies $-\tau \log(\pi_y)$ prior correction (`τ=0.18`) to prevent `139`-facing majority anchors from swallowing `11–26` facing tail launches (`Novology`, `Simple`).
4. **Cap-Down vs. Cap-Up Geometry:** Uses contour width ratio ($W_{\text{top }15\%} / W_{\text{bottom }15\%} > 1.22$) to separate inverted `Conditioner` tubes (`66.3% -> 94.8% F2`) from upright `Shampoo` bottles.
5. **Constrained `/v1/systemone` (`vllm#58216`) Token Trie:** Restricts decoding strictly to the `(2..5)` sister variants in that brand cluster (`+1.8 ms`, `64` visual tokens).

---

## 6. Production Retail Vision Taxonomy: 10 Real-World Shelf Edge Cases & Architectural Solutions

| # | Shelf Edge Case | Real HUL / FMCG Example | Root Failure Mode in Naive Vision/VLM | Production Architectural Fix (`Track D3 / Stage 3–6`) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Sister-Shade & `8px` Text/SPF Collapse** | `Lakme 9to5 CC` (`Almond`/`Honey`/`Beige`/`Bronze`), `SPF 24` vs `30` vs `50` | `224x224` crop downsamples `8px` word to `1.3px` blur; `99%` identical tube triggers majority collapse | **Stage 4.5 `3× Sub-ROI Zoom ([0.62H:0.88H])` + `CIELAB ΔE00` + `/v1/systemone` constrained trie** |
| **2** | **Metallic Foil & Specular Cello-Wrap Glare** | `Lakme Lumi Lit Cream` (`67.9% F2`), silver cartons, glossy sachets | Overhead LED blows out RGB (`255,255,255`) across brand logo | **`I-JEPA` `512-D` Latent Predictive Masking** (predicts semantic features rather than specular pixels) + **$0.65 L^*$ CIELAB** |
| **3** | **Low-Shot / Day-0 Clinical Launch Starvation** | `Novology Barrier Serum` (`11 GT`), `Simple Smoothing Gel` (`14 GT`) | Softmax decision boundary shifts toward `130+` facing core SKUs | **Zero-Retraining `ScaNN` Prototype Insertion** (5 studio spins + 3 shelf crops) + **Menon Logit Adjustment** |
| **4** | **Inverted Form-Factor Confusion (`Conditioner` vs `Shampoo`)** | `Dove Strengthening Conditioner` (`66.3% F2`) vs `Dove Hair Fall Shampoo` | Identical white + gold graphics; differs only by inverted base-down tube vs cap-up bottle | **Stage 4.5 Contour Taper Ratio ($W_{\text{top }15\%} / W_{\text{bottom }15\%} > 1.22$)** + aspect-ratio prior |
| **5** | **$90^\circ / 180^\circ$ Rotated, Toppled & Side-Spine Facings** | `Dove` / `Lux` soap cartons stacked sideways showing only narrow spine | Upright-only `ScaNN` embeddings drop below `0.82` threshold on rotated/sideways boxes | **4-Orientation (`0°, 90°, 270°`) + Side-Spine `ScaNN` Prototypes** + automatic aspect-ratio orientation rectifier |
| **6** | **Price-Rail, Promo Wobbler & `"Buy-2-Get-1"` Shrink-Band Occlusion** | Lower `35%` of jar blocked by yellow price tag or bundled by red promo tape | Bottom text (`Size` / `Variant`) occluded; multipack miscounted as 3 single units | **Upper-Crown Fallback Embedding (`[0.05H:0.55H]`)** + **Stage 5 Horizontal Periodicity Multipack Detector** |
| **7** | **Wide-Angle Perspective Foreshortening (`Size` Confusion)** | `Sunsilk 180ml` vs `340ml` vs `650ml` on top/bottom shelves (`45°` camera tilt) | `650ml` bottle on top shelf appears shorter in raw pixels than `340ml` at eye level | **Stage 5 Local Shelf-Rail Gap Normalization ($\tilde{H}_{\text{norm}} = H_{\text{box}} / H_{\text{shelf\_gap}}(y)$)** |
| **8** | **Hanging Sachet Strip Vertical Shingling (`60%` Overlap)** | `Sunsilk` / `Clinic Plus` / `Bru` 12-sachet hanging strips in GT/MT | Standard `IoU > 0.50` NMS suppresses every odd sachet in the vertical cascade | **Stage 3 Vertical-Chain `DIoU-NMS`** (preserves vertically stepped centers with $\Delta y_{\text{center}} \ge 0.28 H_{\text{box}}$) |
| **9** | **2nd-Row Recessed Shadow Ghosts Behind Front Stockouts (`OOS`)** | Front facing sold out; dark recessed bottle `15cm` back in shadow | Detector fires on rear shadow bottle, hiding a true front-row `Red-Line OOS Gap` | **Stage 3 Depth-Aware Luminance & Baseline-Offset Ghost Filter** ($\Delta y_{\text{base}} > 0.12 H_{\text{shelf}}$ & $\Delta L^* < -28 \rightarrow$ `RECESSED_BACK_ROW`) |
| **10** | **Cross-Frame Panorama Seam Duplication (`5–7` Image `Marketshare`)** | Sales rep walks down `24ft` aisle taking `6` overlapping photos (`~22%` seam overlap) | Naive summation double-counts `220+` boundary bottles, inflating `SOS %` and masking `OOS` | **Stage 3.5 Pairwise `ORB/RANSAC Homography Seam Deduplicator` ($H_{t, t+1}$)** before `Stage 6 Recommend` |

