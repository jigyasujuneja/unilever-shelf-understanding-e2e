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
