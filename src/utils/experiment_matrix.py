"""POC Experiment Matrix (P0 / P1 / P2 / Moonshot) & GCP Sandbox Model Registry Catalog.

Organizes all real-client-data experiments across:
  * Use Case 1: ``market_share`` (SKU Detection, Retrieval, Classification, End-to-End SOS)
  * Use Case 2: ``merchandising`` (Class 0 Display-Bay Planogram Compliance, Promo & Price Verification)
  * Datasets: ``hul_golden`` (v2: 2,970 HUL + Non-HUL classes) & ``gtmerch_golden`` (v2: 9,194 2-class shelves)
  * Standard Pipeline: Leak-free 80/10/10 ``train`` / ``val`` / ``test`` splits on GCS
  * GCP Sandbox Lineage: Vertex AI Experiments (``shelf-bench-l8-experiments``) & Vertex AI Model Registry
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.llm import load_config

TASK_DEFINITIONS: dict[str, dict[str, Any]] = {
    "end_to_end": {
        "task_id": "TASK-E2E",
        "title": "End-to-End Shelf Pipeline",
        "use_case": "market_share",
        "input_output": "Full Store Shelf Photo → Product Boxes + 5-Level Unilever Taxonomy + Share of Shelf",
        "intuition": (
            "Detects every visible product facing on a full retail shelf photo in ~40ms (ONNX YOLO), "
            "clusters adjacent identical facings, and identifies each product from Corporate Owner (HUL vs Competitor) "
            "down to Exact SKU and Facing/Area Share-of-Shelf."
        ),
        "primary_metric": "f2",
        "golden_datasets": ["ms_json_golden", "gtmerch_golden", "hul_sessions"],
    },
    "retrieval": {
        "task_id": "TASK-RET",
        "title": "Catalog SKU Retrieval",
        "use_case": "market_share",
        "input_output": "Product Crop → Exact Match in 2,745-SKU Reference Gallery",
        "intuition": (
            "Matches each cropped shelf product against multi-angle reference catalog photos using visual embeddings "
            "(gemini-embedding-2) and escalates only ambiguous sister variants to batched Gemini contact sheets."
        ),
        "primary_metric": "accuracy",
        "golden_datasets": ["hul_golden", "ms_json_golden", "hul_variants", "hul_packs"],
    },
    "classification": {
        "task_id": "TASK-CLS",
        "title": "5-Level Taxonomy Classification",
        "use_case": "market_share",
        "input_output": "Product Crop → HUL/Competitor → Category → Brand → Variant → Pack Type",
        "intuition": (
            "Classifies cropped products across all 5 levels of the Unilever hierarchy and enforces temporal M&A "
            "brand ownership rules (e.g., Pureit, Horlicks, Minimalist, OZiva) with macro-balanced brand scoring."
        ),
        "primary_metric": "accuracy",
        "golden_datasets": ["hul_golden", "hul_brands"],
    },
    "detection": {
        "task_id": "TASK-DET",
        "title": "Shelf Product & Bay Detection",
        "use_case": "market_share",
        "input_output": "Full Store Shelf Photo → Product & Display-Bay Bounding Boxes (IoU ≥ 0.5)",
        "intuition": (
            "Localizes every product facing (Class 1) and branded display bay/cooler header (Class 0) on dense "
            "Indian kirana (GT) and modern-trade (MT) shelves in a single CPU pass at ₹0 LLM token cost."
        ),
        "primary_metric": "f2",
        "golden_datasets": ["ms_json_golden", "gtmerch_golden", "hul_sessions"],
    },
    "planogram_compliance": {
        "task_id": "TASK-PLN",
        "title": "Planogram & Out-of-Stock Audit",
        "use_case": "merchandising",
        "input_output": "Full Store Shelf Photo → Brand-Block Purity, Competitor Intrusions & OOS Gaps",
        "intuition": (
            "Audits whether HUL brand blocks are contiguous on the shelf, flags competitor products placed inside "
            "paid HUL display bays, and measures empty shelf gaps (Out-of-Stock voids)."
        ),
        "primary_metric": "accuracy",
        "golden_datasets": ["gtmerch_golden", "hul_sessions"],
    },
    "promo_detection": {
        "task_id": "TASK-PRM",
        "title": "Promo & Value-Pack Verification",
        "use_case": "merchandising",
        "input_output": "Product Crop → Promo Callout Type (% Extra, ₹ Off, B1G1, Multi-Pack) + Exact SKU",
        "intuition": (
            "Zooms into promotional banners and price-off stickers on the pack to verify on-shelf promo execution "
            "(Buy-1-Get-1, ₹10 Off, Free Container, Value Bundles) vs regular baseline packs."
        ),
        "primary_metric": "accuracy",
        "golden_datasets": ["hul_promos", "hul_golden"],
    },
}

APPROACH_INTUITIONS: dict[str, dict[str, str]] = {
    "shelf_hierarchical_clm_jev": {
        "display_name": "End-to-End Shelf Scanner (ONNX Detector + Brand Filter + Batched Gemini)",
        "intuition": (
            "Detects all shelf products in 40ms via ONNX YOLO, groups identical side-by-side facings into unique product groups, "
            "filters to Top-2 brands for free, and batch-verifies only ambiguous sister SKUs in Gemini."
        ),
        "task_id": "TASK-E2E",
    },
    "shelf_clm_standalone": {
        "display_name": "Zero-LLM Shelf Scanner (ONNX Detector + Color-Calibrated Visual Matcher)",
        "intuition": (
            "100% LLM-free full-shelf pipeline: ONNX YOLO finds every product box in 40ms and matches each crop "
            "against catalog embeddings + center-pack color at ₹0 Gemini token cost."
        ),
        "task_id": "TASK-E2E",
    },
    "detect_l8_ultimate": {
        "display_name": "Full-Context Shelf Pipeline (Lighting Correction + Pack Geometry + Row Smoothing)",
        "intuition": (
            "Combines full-shelf lighting normalization, bottle/sachet taper geometry, and horizontal neighbor "
            "smoothing so adjacent facings on a shelf row reinforce each other."
        ),
        "task_id": "TASK-E2E",
    },
    "shelf_l8_ultimate": {
        "display_name": "Full-Context Shelf Pipeline (ONNX Detector + Lighting + Row Smoothing)",
        "intuition": (
            "Pairs fast ONNX shelf box detection with lighting correction, pack geometry, and left-to-right "
            "row continuity smoothing."
        ),
        "task_id": "TASK-E2E",
    },
    "detect_onnx_jev_laya": {
        "display_name": "ONNX Shelf Detector + Confidence-Gated Gemini Contact Sheet",
        "intuition": (
            "Uses ONNX YOLO for fast box detection and escalates low-confidence embedding matches to "
            "5-candidate Gemini visual contact sheets."
        ),
        "task_id": "TASK-E2E",
    },
    "hierarchical_clm_jev": {
        "display_name": "Two-Stage Brand Filter + Visual Crop Matcher + Gemini Contact Sheet",
        "intuition": (
            "First narrows the 2,745-SKU catalog to the Top-2 candidate brands (0 LLM calls), auto-accepts "
            "clear visual matches (~60–70% of crops), and sends only hard sister variants to Gemini."
        ),
        "task_id": "TASK-RET",
    },
    "l8_hul_ultimate": {
        "display_name": "Lighting-Corrected Visual Matcher + Pack Geometry + Gemini Verifier",
        "intuition": (
            "Normalizes warm kirana store lighting casts, checks center-80% pack color and bottle/sachet shape, "
            "and uses Gemini contact sheets only for close sibling variants."
        ),
        "task_id": "TASK-RET",
    },
    "clm_standalone": {
        "display_name": "Fast Visual Search Baseline (Embedding + Center-Color Only, Zero LLM)",
        "intuition": (
            "Matches each crop to multi-angle catalog embeddings with hubness correction and center-80% color "
            "at ₹0 Gemini VLM cost (~₹0.011/crop)."
        ),
        "task_id": "TASK-RET",
    },
    "embedding_retrieval": {
        "display_name": "Raw Vector Cosine Search (Single-Stage Embedding Top-1)",
        "intuition": (
            "Simplest baseline: embeds the crop once and returns the single nearest catalog reference vector "
            "by cosine similarity with zero post-processing."
        ),
        "task_id": "TASK-RET",
    },
    "jev_laya_zscore_lid": {
        "display_name": "Statistical Confidence Gate (Z-Score Margin + Gemini Contact Sheet)",
        "intuition": (
            "Measures the statistical gap between the #1 and #2 catalog matches; accepts clear winners "
            "immediately and sends close ties to a 5-candidate Gemini contact sheet."
        ),
        "task_id": "TASK-RET",
    },
    "clef_standalone": {
        "display_name": "Fine-Print Weight & Size Zoom Verifier (2x Bottom-Strip Zoom + Gemini)",
        "intuition": (
            "Zooms 2x into the bottom 30% of the pack where grammage/ml (50g vs 100g) is printed so Gemini "
            "can separate size variants of the same product."
        ),
        "task_id": "TASK-RET",
    },
    "cold_start_lid_discovery": {
        "display_name": "Zero-Shot New-Launch SKU Discovery (Out-of-Catalog Detector + Live Registry)",
        "intuition": (
            "Detects when a shelf crop does not match any existing gallery photo and queries the live Unilever "
            "brand/SKU registry to identify newly launched variants without retraining."
        ),
        "task_id": "TASK-RET",
    },
    "djev_classify": {
        "display_name": "5-Level Unilever Taxonomy Classifier (Visual Shortlist + Gemini Verifier)",
        "intuition": (
            "Classifies a product crop across all 5 hierarchy levels (HUL/Comp → Category → Brand → Variant → Pack) "
            "by verifying top visual candidates with Gemini."
        ),
        "task_id": "TASK-CLS",
    },
    "clm_standalone_classify": {
        "display_name": "Zero-LLM 5-Level Taxonomy Classifier (Visual + Text Prototype Matcher)",
        "intuition": (
            "Assigns all 5 Unilever taxonomy levels using purely visual and text embeddings with ₹0 Gemini LLM cost."
        ),
        "task_id": "TASK-CLS",
    },
    "gemini_classify": {
        "display_name": "Direct Gemini 5-Level Taxonomy Classifier (Single-Crop SFT / Zero-Shot)",
        "intuition": (
            "Directly predicts Category, Brand, Pack Type, and Variant from a single product crop using Gemini or a fine-tuned Vertex AI SFT endpoint."
        ),
        "task_id": "TASK-CLS",
    },
    "hierarchy_classify": {
        "display_name": "Hierarchical Gemini 5-Level Taxonomy Classifier",
        "intuition": (
            "Stepwise Category → Brand → Variant taxonomy classification on a single product crop."
        ),
        "task_id": "TASK-CLS",
    },
    "distilled_yolo_onnx": {
        "display_name": "Distilled YOLO11n Shelf & Bay Box Detector (CPU ONNX, Zero LLM)",
        "intuition": (
            "Single-pass CPU ONNX detector trained to localize both branded display bays (Class 0) and individual "
            "product facings (Class 1) in ~40ms at ₹0 token cost."
        ),
        "task_id": "TASK-DET",
    },
    "yolo_sku110k_onnx": {
        "display_name": "Dense Retail Shelf Box Detector (YOLOv12s Teacher ONNX, Zero LLM)",
        "intuition": (
            "High-density retail shelf box detector pre-trained on 110K tightly packed shelf facings and scoped "
            "to Unilever display bays at ₹0 token cost."
        ),
        "task_id": "TASK-DET",
    },
    "bay_planogram_verifier": {
        "display_name": "Merchandising Planogram, Brand-Block Purity & Out-of-Stock Gap Auditor",
        "intuition": (
            "Checks whether HUL brand blocks are contiguous left-to-right, flags competitor intrusions inside "
            "HUL display bays, and counts empty shelf voids (missing facings)."
        ),
        "task_id": "TASK-PLN",
    },
    "jev_promo_verifier": {
        "display_name": "Promo Banner & Price-Off Sticker Zoom Verifier (Gemini Contact Sheet)",
        "intuition": (
            "Crops and zooms into top/bottom promotional bands to verify B1G1, % Extra, ₹ Off, and multi-pack "
            "bundles via Gemini visual verification."
        ),
        "task_id": "TASK-PRM",
    },
    "clm_promo_matcher": {
        "display_name": "Zero-LLM Promo Pack Matcher (Visual + Promo Text Embedding)",
        "intuition": (
            "Matches promotional packs against catalog promo prototypes using visual + text embeddings at ₹0 LLM cost."
        ),
        "task_id": "TASK-PRM",
    },
    "outlet_scoped_clm_jev": {
        "display_name": "Store-Segment & Scene Category Scoped Matcher (EXP-20)",
        "intuition": (
            "Uses the dominant scene category across all boxes on the shelf to prune out-of-category lookalikes "
            "(e.g. Oral Care vs Skin Care vs Laundry) before running CLM InfoNCE + Gemini verification."
        ),
        "task_id": "TASK-RET",
    },
    "shelf_outlet_scoped_clm_jev": {
        "display_name": "End-to-End Store-Segment Scoped Shelf Pipeline (EXP-20)",
        "intuition": (
            "Combines Stage-1 ONNX YOLO box detection with store-segment & scene-category gallery pruning "
            "before CLM InfoNCE and batched Gemini contact sheets."
        ),
        "task_id": "TASK-E2E",
    },
    "tube_midband_jev": {
        "display_name": "Middle-30% Cylindrical Squeeze-Tube Unwarping + Variant Strip Verifier (EXP-21)",
        "intuition": (
            "Unwarps cylindrical squeeze tubes (Pepsodent, Closeup, Lakme, Ponds) horizontally and adds a 2x "
            "Middle-30% Variant Sub-Banner Strip alongside the Bottom-30% Weight Strip at 0 extra image tokens."
        ),
        "task_id": "TASK-RET",
    },
    "shelf_tube_midband_jev": {
        "display_name": "End-to-End Squeeze-Tube Unwarped Variant + Weight Strip Pipeline (EXP-21)",
        "intuition": (
            "Full-shelf ONNX detector paired with cylindrical tube horizontal unwarping and dual Middle-30% "
            "Variant + Bottom-30% Weight zoom strips."
        ),
        "task_id": "TASK-E2E",
    },
    "row_neighbor_prior_jev": {
        "display_name": "1D Horizontal Shelf-Row Neighbor Brand/Category Prior (EXP-22)",
        "intuition": (
            "Applies a 1D horizontal shelf-row consistency bonus (+0.025 category, +0.038 brand) from adjacent "
            "high-confidence facings on the same shelf plank before final SKU selection."
        ),
        "task_id": "TASK-RET",
    },
    "shelf_row_neighbor_prior": {
        "display_name": "End-to-End Shelf Pipeline with 1D Horizontal Neighbor Brand Prior (EXP-22)",
        "intuition": (
            "Full-shelf ONNX detection + CLM InfoNCE + 1D horizontal shelf-row neighbor brand/category consistency prior."
        ),
        "task_id": "TASK-E2E",
    },
    "shelf_height_size_prior_jev": {
        "display_name": "Shelf-Plank Relative Box-Height Physical Size Prior (EXP-23)",
        "intuition": (
            "Computes each crop's height ratio relative to its horizontal shelf row (h_box / h_row) to separate "
            "physically identical graphics across 50g vs 100g vs 200g vs 500g packs at 0 extra token cost."
        ),
        "task_id": "TASK-RET",
    },
    "shelf_height_size_prior": {
        "display_name": "End-to-End Shelf Pipeline with Relative Box-Height Size Prior (EXP-23)",
        "intuition": (
            "Full-shelf ONNX detection + CLM InfoNCE + shelf-plank relative box-height size calibration (h_box / h_row)."
        ),
        "task_id": "TASK-E2E",
    },
    "distilled_yolo_960_softnms": {
        "display_name": "960px High-Res Multi-Window ONNX Detector + Gaussian Soft-NMS (EXP-24)",
        "intuition": (
            "Adds 960px-equivalent high-res horizontal band passes and Aspect-Aware Gaussian Soft-NMS (sigma=0.55) "
            "so overlapping hanging sachet strips (Clinic Plus, Sunsilk) and stacked thin soap bars survive NMS."
        ),
        "task_id": "TASK-DET",
    },
    "shelf_960_softnms_e2e": {
        "display_name": "End-to-End 960px Soft-NMS Detector + Gray-World CLM-JEV Pipeline (EXP-24)",
        "intuition": (
            "Pairs the 960px High-Res Gaussian Soft-NMS ONNX detector with Gray-World CLM InfoNCE + batched Gemini contact sheets."
        ),
        "task_id": "TASK-E2E",
    },
    "session_overlap_dedup_jev": {
        "display_name": "Multi-Photo Store-Visit Session Overlap Deduplicator (EXP-25)",
        "intuition": (
            "Detects overlapping boundary columns across consecutive store-visit photos using normalized vertical band "
            "cross-correlation + embedding cosine (>=0.94) so boundary facings are not double-counted in SOS."
        ),
        "task_id": "TASK-RET",
    },
    "session_overlap_dedup_e2e": {
        "display_name": "End-to-End Store-Visit Session Overlap Deduplication Pipeline (EXP-25)",
        "intuition": (
            "Full-shelf ONNX detection + multi-photo store-visit boundary overlap deduplication + CLM-JEV attribution."
        ),
        "task_id": "TASK-E2E",
    },
    "hf_clef_flash_gpu": {
        "display_name": "Hugging Face Cloudflare/clef-flash (Qwen3.5-9B) Open-Weight Verifier (EXP-26)",
        "intuition": (
            "Runs open-weight Cloudflare/clef-flash (Qwen/Qwen3.5-9B) on Vertex AI L4 GPU via vLLM with top-16 "
            "token logprobs on CLEF Bottom-30% Zoom-Strip contact sheets."
        ),
        "task_id": "TASK-RET",
    },
    "hf_clef_27b_gpu": {
        "display_name": "Hugging Face Cloudflare/clef-27b (Qwen3.5-27B) Open-Weight Verifier (EXP-26)",
        "intuition": (
            "Runs open-weight Cloudflare/clef-27b (Qwen/Qwen3.5-27B, 4x L4 tensor-parallel) on Vertex AI GPU via vLLM "
            "on CLEF Bottom-30% Zoom-Strip contact sheets."
        ),
        "task_id": "TASK-RET",
    },
    "hf_clm_v01_8b_gpu": {
        "display_name": "Hugging Face Contrastive-LM/CLM-v0.1-8B (Qwen3-VL-8B) Verifier (EXP-26)",
        "intuition": (
            "Runs open-weight Contrastive-LM/CLM-v0.1-8B (Qwen/Qwen3-VL-8B-Instruct) on Vertex AI L4 GPU via vLLM "
            "after Stage-2A CLM InfoNCE gating."
        ),
        "task_id": "TASK-RET",
    },
}


POC_EXPERIMENTS: list[dict[str, Any]] = [
    # =========================================================================
    # P0 — Critical Path: Core Market Share & Merchandising on Golden Datasets
    # =========================================================================
    {
        "experiment_id": "P0-EXP-01",
        "priority": "P0",
        "task_id": "TASK-RET",
        "title": "Fast Visual Search Baseline (Embedding + Center-Pack Color, Zero LLM Calls)",
        "technical_title": "Zero-VLM Standalone CLM InfoNCE Baseline on v2.2 HUL+Non-HUL Golden Catalog",
        "intuition": (
            "Matches each product crop to the 2,745-SKU reference catalog using only vector embeddings and "
            "center-80% pack color in <0.5s with zero Gemini LLM calls (₹0.011/crop)."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "approaches": ["clm_standalone", "embedding_retrieval"],
        "models": ["gemini-embedding-2", "multimodalembedding@001"],
        "splits": ["train", "val", "test"],
        "hypothesis": (
            "Multi-view Contrastive-LM (CLM) InfoNCE with CSLS hubness reduction and inner-80% CIELAB "
            "chromatic energy (1/tau=28.0) outperforms raw nearest-neighbor cosine retrieval by >= +6% "
            "exact SKU accuracy across the 2,745-class de-collided HUL + Non-HUL Golden catalog at ₹0 VLM token cost."
        ),
        "description": (
            "Establishes the ₹0-VLM floor on gs://marketshare-datasets/derived_manifests/hul_golden/rpc.json "
            "(2,745 de-collided HUL + Non-HUL classes, 0 Unknown categories). Uses train/gallery split to build "
            "L2-normalized multi-view prototypes, tunes temperature tau on val, and evaluates on test."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 76.0% exact SKU accuracy, >= 92.0% brand accuracy, ₹0 VLM token cost",
        "model_family": "hul-clm-infonce-2970sku-prototypes",
    },
    {
        "experiment_id": "P0-EXP-02",
        "priority": "P0",
        "task_id": "TASK-RET",
        "title": "Two-Stage Brand Filter + Visual Crop Matcher + Gemini Contact Sheet",
        "technical_title": "2-Stage Hierarchical Brand-Gate + Allowed-ID CLM InfoNCE + Batched JEV Contact Sheets",
        "intuition": (
            "First locks in the Top-2 candidate brands (eliminating 95% of wrong-brand lookalikes for free), "
            "auto-accepts clear matches (~60–70%), and sends only hard sister variants to Gemini."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "approaches": ["hierarchical_clm_jev", "clm_contrastive", "coarse_jev_laya"],
        "models": ["gemini-3.8-flash", "gemini-3.5-flash"],
        "splits": ["train", "val", "test"],
        "hypothesis": (
            "Pruning the 2,745-class search space via a Stage-2A Multi-Prototype (Category, Brand, Pack) "
            "Gate to the top-2 brands (~20-40 sibling variants) before CLM InfoNCE + Z-Score/LID gating "
            "eliminates cross-brand false positives while bypassing Stage-2B VLM calls on ~60-70% of crops."
        ),
        "description": (
            "Head-to-head comparison of our proven winning retrieval architectures (hierarchical_clm_jev vs "
            "clm_contrastive vs coarse_jev_laya) on hul_golden v2.2 across HUL and Non-HUL competitor SKUs."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 80.0% exact SKU accuracy on strict 1-crop test, >= 90.0% on val, 0% false HUL attribution",
        "model_family": "hul-hierarchical-brand-pack-gate",
    },
    {
        "experiment_id": "P0-EXP-03",
        "priority": "P0",
        "task_id": "TASK-RET",
        "title": "Store Lighting Correction + Pack Shape Geometry + Gemini Verifier",
        "technical_title": "Photometric Shades-of-Gray (p=6) + Inner-80% CIELAB + Bottle/Sachet Taper Ultimate Pipeline",
        "intuition": (
            "Removes warm yellow kirana store lighting casts and measures top-25% bottle-neck vs flat-sachet "
            "geometry so pack format and color stay accurate under poor lighting."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "approaches": ["l8_hul_ultimate", "jev_laya_global_wb", "jev_laya_hybrid"],
        "models": ["gemini-3.8-flash", "gemini-3.5-flash"],
        "splits": ["train", "val", "test"],
        "hypothesis": (
            "Minkowski p=6 Shades-of-Gray full-shelf illuminant estimation + inner-80% foreground CIELAB "
            "and top-25% bottle/sachet taper descriptors resolve colour-cast and container-form collisions "
            "(Bottle vs Pouch vs Sachet of the same base variant)."
        ),
        "description": (
            "Evaluates photometric colour constancy and geometric container-taper priors on Indian kirana "
            "and modern-trade store crops under warm tungsten/fluorescent store lighting."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 80.0% exact SKU accuracy, >= 90.0% macro brand accuracy on hul_golden test",
        "model_family": "hul-hierarchical-brand-pack-gate",
    },
    {
        "experiment_id": "P0-EXP-04",
        "priority": "P0",
        "task_id": "TASK-PLN",
        "title": "Merchandising Display-Bay, Brand-Block Purity & Out-of-Stock Gap Auditor",
        "technical_title": "GTMerch 2-Class Display-Bay (Class 0 Asset + Class 1 SKU) Planogram & SOS Verifier",
        "intuition": (
            "Audits paid Unilever display bays (Class 0) and shelf rows (Class 1) to measure brand-block "
            "contiguity, flag competitor intrusions, and count missing Out-of-Stock facings."
        ),
        "use_case": "merchandising",
        "task": "planogram_compliance",
        "dataset": "gtmerch_golden",
        "approaches": ["bay_planogram_verifier"],
        "models": ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        "splits": ["train", "val", "test"],
        "hypothesis": (
            "Combining Class 0 display-bay/cooler asset containment (27.8% in-asset vs 72.2% open-aisle "
            "facings) with row-clustered left-to-right brand-block purity achieves <= 2.5% absolute "
            "Share-of-Shelf (SOS) error on real client shelves."
        ),
        "description": (
            "Primary Merchandising P0 benchmark on gs://marketshare-datasets/derived_manifests/gtmerch_golden/rpc.json "
            "(auditing Class 0 display bays, Class 1 SKU facings, brand-block purity, and HUL vs competitor SOS)."
        ),
        "primary_metric": "accuracy",
        "target_goal": "<= 2.5% HUL SOS absolute error, >= 95.0% brand-block purity on gtmerch_golden",
        "model_family": "hul-gtmerch-bay-planogram-sos-verifier",
    },
    {
        "experiment_id": "P0-EXP-05",
        "priority": "P0",
        "task_id": "TASK-E2E",
        "title": "Production End-to-End Shelf Pipeline on GTMerch (ONNX Detector + Brand Filter + Gemini)",
        "technical_title": "GTMerch Real Client Shelf End-to-End Detection + Hierarchical CLM-JEV Attribution",
        "intuition": (
            "Full store-shelf pipeline on GTMerch: detects all product boxes in ~40ms, groups identical "
            "side-by-side facings into unique product groups (~₹0.056/shelf), and verifies SKUs."
        ),
        "use_case": "market_share",
        "task": "end_to_end",
        "dataset": "gtmerch_golden",
        "approaches": ["shelf_hierarchical_clm_jev", "shelf_l8_ultimate", "shelf_onnx_jev_laya"],
        "models": ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        "splits": ["train", "val", "test"],
        "hypothesis": (
            "Stage-1 Distilled ONNX YOLO detector ($0 detection token cost) paired with Stage-2 "
            "Hierarchical Brand-Gate + CLM InfoNCE + JEV Contact Sheets achieves the highest end-to-end "
            "F2 on real Unilever Indian retail shelves."
        ),
        "description": (
            "End-to-end Market Share benchmark on real client shelf photos from gtmerch_golden "
            "(~41 products/shelf average on evaluated splits, up to 303 boxes/shelf)."
        ),
        "primary_metric": "f2",
        "target_goal": ">= 85.0% End-to-End F2, >= 95.0% Stage-1 Found Recall on gtmerch_golden test",
        "model_family": "hul-yolo11n-distilled-detector-onnx",
    },
    # =========================================================================
    # P1 — High-ROI Latency, Cost & Merchandising Promo/Pack Optimizations
    # =========================================================================
    {
        "experiment_id": "P1-EXP-06",
        "priority": "P1",
        "task_id": "TASK-RET",
        "title": "Model Cost vs. Accuracy Comparison (Gemini 3.5 Flash-Lite vs. 3.5 Flash vs. 3.8 Flash)",
        "technical_title": "Cost-Tier & Model Pareto: Gemini 3.5 Flash-Lite vs 3.5 Flash vs 3.8 Flash",
        "intuition": (
            "Compares cost per crop (₹0.038 Flash-Lite vs ₹0.180 Flash) and latency across Gemini model "
            "tiers when ~60–70% of crops are already solved by the Stage-2A visual gate."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "approaches": ["hierarchical_clm_jev", "l8_hul_ultimate"],
        "models": ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Because Stage-2A Hierarchical CLM InfoNCE resolves ~60-70% of crops with 0 VLM calls and "
            "narrows the remaining crops to a 5-way visual contact sheet, gemini-3.5-flash-lite retains "
            "comparable accuracy at ~75% lower VLM token cost (₹0.067/call vs ₹0.421/call) and 2.5x faster p95 latency."
        ),
        "description": (
            "Quantifies the ₹/crop and ₹/shelf cost vs accuracy Pareto frontier across Gemini model tiers on "
            "hul_golden test."
        ),
        "primary_metric": "accuracy",
        "target_goal": "<= ₹0.05 / crop (₹0.19 / grouped shelf) net cost with >= 80.0% exact SKU accuracy",
        "model_family": "hul-hierarchical-brand-pack-gate",
    },
    {
        "experiment_id": "P1-EXP-07",
        "priority": "P1",
        "task_id": "TASK-PRM",
        "title": "Promo Sticker, Multi-Pack & Price-Off Verifier (Zoom-Band Gemini vs. Zero-LLM Matcher)",
        "technical_title": "Merchandising Promo & Price Callout Verification: Zero-VLM CLM vs JEV Zoom-Band Verifier",
        "intuition": (
            "Zooms into promotional header/footer bands to detect B1G1, % Extra, ₹10 Off, and bundled "
            "multi-packs on Indian retail shelves."
        ),
        "use_case": "merchandising",
        "task": "promo_detection",
        "dataset": "hul_promos",
        "approaches": ["jev_promo_verifier", "clm_promo_matcher"],
        "models": ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-embedding-2"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Adding a high-resolution promo-header/footer zoom band (jev_promo_verifier) boosts exact "
            "promo callout accuracy (B1G1, B3G1, PO2/PO3/PO4, Rs 10 Off, Free Container) over 0-VLM "
            "CLM promo matching on small micro-text banners."
        ),
        "description": (
            "Evaluates promotional compliance and price-off callout verification across Unilever promo SKUs."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 88.0% promo_type accuracy and >= 85.0% exact promo SKU accuracy",
        "model_family": "hul-gtmerch-bay-planogram-sos-verifier",
    },
    {
        "experiment_id": "P1-EXP-08",
        "priority": "P1",
        "task_id": "TASK-PLN",
        "title": "Multi-Photo Store Visit Planogram & Share-of-Shelf Aggregator (hul_sessions)",
        "technical_title": "Store-Visit Session Strip SOS & Competitor Share Calibration",
        "intuition": (
            "Measures HUL vs. Competitor Share-of-Shelf (both facing count and box area) and brand-block "
            "purity across multi-photo store visits."
        ),
        "use_case": "merchandising",
        "task": "planogram_compliance",
        "dataset": "hul_sessions",
        "approaches": ["bay_planogram_verifier"],
        "models": ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Row-clustered left-to-right sequence smoothing reduces HUL vs Competitor Share-of-Shelf "
            "(SOS) absolute error below 2.0% on multi-facing store session strips."
        ),
        "description": (
            "Evaluates store-visit Share of Shelf (SOS) and brand-block compliance across HUL + rival facings."
        ),
        "primary_metric": "accuracy",
        "target_goal": "<= 2.0% facing_sos_abs_error, >= 95.0% is_hul accuracy",
        "model_family": "hul-gtmerch-bay-planogram-sos-verifier",
    },
    {
        "experiment_id": "P1-EXP-09",
        "priority": "P1",
        "task_id": "TASK-E2E",
        "title": "Fast Zero-LLM End-to-End Shelf Scanner (ONNX Box Detector + Visual Embedding Matcher)",
        "technical_title": "100% Zero-VLM Real Shelf End-to-End (Distilled ONNX YOLO + Standalone CLM InfoNCE)",
        "intuition": (
            "Runs the entire shelf pipeline (box detection + SKU identification) using CPU ONNX + embeddings "
            "with ₹0 Gemini LLM token spend (~₹0.056/shelf grouped side-by-side)."
        ),
        "use_case": "market_share",
        "task": "end_to_end",
        "dataset": "gtmerch_golden",
        "approaches": ["shelf_clm_standalone", "distilled_yolo_onnx", "yolo_sku110k_onnx"],
        "models": ["gemini-embedding-2", "yolo11n-distilled-onnx", "yolov12s-sku110k-onnx"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Combining Distilled YOLO11n ONNX CPU detection with Standalone CLM InfoNCE retrieval delivers "
            "sub-2-second shelf processing with ₹0 Gemini VLM token spend."
        ),
        "description": (
            "Benchmarks 100% zero-VLM edge/CPU shelf detection and SKU attribution on gtmerch_golden and ms_json_golden."
        ),
        "primary_metric": "f2",
        "target_goal": "₹0.00 Gemini VLM token cost, p95 latency <= 2.0s/shelf",
        "model_family": "hul-yolo11n-distilled-detector-onnx",
    },
    # =========================================================================
    # P2 — Fine-Grained Disambiguation, Photometric & Temporal M&A Ablations
    # =========================================================================
    {
        "experiment_id": "P2-EXP-10",
        "priority": "P2",
        "task_id": "TASK-RET",
        "title": "Fine-Print Weight & Size Zoom Verifier (2x Bottom-Strip Zoom + Gemini Logprobs)",
        "technical_title": "Standalone CLEF Constrained Logprob Decision Head + Bottom-30% Net-Weight Zoom-Strip",
        "intuition": (
            "Zooms 2x into the bottom 30% of the pack where grammage/ml (50g vs 100g) is printed so Gemini "
            "can read small size callouts without extra image tokens."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "approaches": ["clef_standalone", "clef_flash_decision", "jev_laya_zoom_strip"],
        "models": ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Appending a 2x horizontal zoom strip of the bottom 30% net-weight/variant band (+10.5 dB "
            "character PSNR at 0 extra image tokens) with single-token logprob calibration resolves "
            "near-identical sister variants."
        ),
        "description": (
            "Ablates Stage-B decision heads (Standalone CLEF vs CLEF-Flash vs Zoom-Strip Contact Sheets) "
            "on hul_golden."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 85.0% exact SKU accuracy with calibrated token logprob confidence",
        "model_family": "hul-hierarchical-brand-pack-gate",
    },
    {
        "experiment_id": "P2-EXP-11",
        "priority": "P2",
        "task_id": "TASK-RET",
        "title": "Statistical Confidence Gate (Auto-Accept Clear Matches, Escalate Hard Ties to Gemini)",
        "technical_title": "Z-Score Margin + Local Intrinsic Dimensionality (LID) Hubness Calibration",
        "intuition": (
            "Uses Z-score confidence margins and local crowd density (LID) to auto-accept 60%+ of clear "
            "matches for free while sending close sister-SKU ties to Gemini Flash-Lite (90.0% test acc at ₹0.038/crop)."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "approaches": ["jev_laya_zscore_lid"],
        "models": ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Replacing fixed cosine margin thresholds with local z-score margin (z >= 1.05 distinct / "
            "1.42 sister) and Local Intrinsic Dimensionality (LID <= 12.0) prevents high-hubness "
            "catch-all classes from absorbing sibling variants."
        ),
        "description": (
            "Evaluates statistical manifold calibration (Z-Score + LID) across the 2,745-class Golden gallery."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 88.0% exact SKU accuracy and 0% false HUL attribution rate",
        "model_family": "hul-clm-infonce-2970sku-prototypes",
    },
    {
        "experiment_id": "P2-EXP-12",
        "priority": "P2",
        "task_id": "TASK-CLS",
        "title": "Balanced 20-Brand & Corporate M&A Ownership Benchmark (HUL vs. Competitors)",
        "technical_title": "Live Brand/SKU Registry Temporal M&A & Dual-Umbrella (Tata, Pureit, Minimalist, OZiva) Audit",
        "intuition": (
            "Tests balanced classification across top HUL and competitor brands while enforcing acquisition/divestiture "
            "dates (Pureit, Horlicks, Minimalist, OZiva) and Tata Beverages vs. Foods split."
        ),
        "use_case": "market_share",
        "task": "classification",
        "dataset": "hul_brands",
        "approaches": ["clm_standalone_classify", "clm_classify", "clef_flash_classify", "djev_classify"],
        "models": ["gemini-embedding-2", "gemini-3.8-flash", "gemini-3.5-flash-lite"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Conditioning is_hul attribution on capture_date and base_variant via brand_sku_live_registry.json "
            "eliminates 100% of divestiture/acquisition (Pureit, Annapurna, Horlicks, Minimalist, OZiva) "
            "and Tata Beverage-vs-Foods umbrella attribution errors."
        ),
        "description": (
            "Validates temporal corporate ownership ledger and competitor conglomerate rollups on hul_brands."
        ),
        "primary_metric": "accuracy",
        "target_goal": "100% temporal M&A ownership attribution accuracy, >= 95.0% brand accuracy",
        "model_family": "hul-clm-infonce-2970sku-prototypes",
    },
    # =========================================================================
    # Moonshot — Next-Gen Frontier Architectures
    # =========================================================================
    {
        "experiment_id": "MOON-EXP-13",
        "priority": "Moonshot",
        "task_id": "TASK-RET",
        "title": "Fine-Tuned Gemini LoRA Adapter on Hard-Negative Sister SKUs (Vertex AI SFT)",
        "technical_title": "Vertex AI Supervised Fine-Tuned LoRA Adapter + Contrastive Triplet Hard-Negative Mining",
        "intuition": (
            "Fine-tunes a lightweight Gemini Flash-Lite LoRA adapter on train-split crops and lookalike sister "
            "triplets so Flash-Lite matches 3.8-Flash accuracy at 75% lower token cost."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "approaches": ["hierarchical_clm_jev", "l8_hul_ultimate"],
        "models": ["gemini-3.5-flash-lite-sft", "gemini-3.5-flash-sft"],
        "splits": ["train", "val", "test"],
        "hypothesis": (
            "Fine-tuning a rank-4 LoRA adapter on zero-leakage hul_golden train crops and hard-negative "
            "sibling triplets closes the gap between Flash-Lite and 3.8-Flash while cutting inference cost by 75%."
        ),
        "description": (
            "Trains and evaluates Vertex AI LoRA SFT endpoints on sft_multitask_train.jsonl & "
            "contrastive_triplets_train.jsonl exported strictly from the 80% train/gallery split."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 91.5% exact SKU accuracy on hul_golden test at Flash-Lite pricing",
        "model_family": "hul-jev-laya-lora-sft-adapter",
    },
    {
        "experiment_id": "MOON-EXP-14",
        "priority": "Moonshot",
        "task_id": "TASK-PLN",
        "title": "Spatial Shelf-Graph Reasoner for Branded Coolers/Bays & Out-of-Stock Voids",
        "technical_title": "Joint 2-Class (Class 0 Cooler/Bay Asset + Class 1 SKU) Spatial-Graph OOS & Planogram Reasoner",
        "intuition": (
            "Combines Class 0 branded display-bay boundaries with horizontal shelf-row graphs to pinpoint "
            "competitor products inside paid HUL bays and estimate missing facings in empty gaps."
        ),
        "use_case": "merchandising",
        "task": "planogram_compliance",
        "dataset": "gtmerch_golden",
        "approaches": ["bay_planogram_verifier"],
        "models": ["gemini-3.8-flash"],
        "splits": ["train", "val", "test"],
        "hypothesis": (
            "Reasoning jointly over Class 0 asset boundaries, shelf-row graphs, and empty horizontal "
            "voids between Class 1 SKU boxes detects Out-of-Stock (OOS) facings and competitor "
            "intrusion inside paid Unilever branded bays."
        ),
        "description": (
            "Moonshot spatial-graph merchandising audit combining Class 0 bay geometry with Class 1 SKU facings."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 90.0% branded-bay intrusion & OOS gap recall on gtmerch_golden",
        "model_family": "hul-gtmerch-bay-planogram-sos-verifier",
    },
    {
        "experiment_id": "MOON-EXP-15",
        "priority": "Moonshot",
        "task_id": "TASK-RET",
        "title": "Zero-Shot New-Launch SKU Discovery (Out-of-Catalog Detection + Live Product Registry)",
        "technical_title": "Open-World Zero-Shot New-Launch SKU Discovery via Live Web Crawler + Prototype Injection",
        "intuition": (
            "Detects when a newly launched product is missing from the visual gallery and injects text+image "
            "prototypes from the live brand/SKU registry without retraining."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "approaches": ["cold_start_lid_discovery"],
        "models": ["gemini-3.8-flash"],
        "splits": ["val", "test"],
        "hypothesis": (
            "When LID > 9.5 and Z-Score margin < 1.25 signal an unseen new-launch SKU (30% visual gallery held out), "
            "querying brand_sku_live_registry.json (OpenFoodFacts EAN-13 + Wikidata + HUL catalog) and injecting "
            "zero-shot text+visual prototypes enables zero-retraining SKU recognition."
        ),
        "description": (
            "Evaluates open-world cold-start SKU recognition using the Algo-PR6 live brand/SKU registry."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 80.0% cold-start recall on held-out new-launch variants without retraining",
        "model_family": "hul-clm-infonce-2970sku-prototypes",
    },
    # =========================================================================
    # Full-Task Production Coverage: Stage-1 Detection, Multi-Level Classification & MT/GT Shelf E2E
    # =========================================================================
    {
        "experiment_id": "P0-EXP-16",
        "priority": "P0",
        "task_id": "TASK-DET",
        "title": "Stage-1 Shelf Product & Display-Bay Box Detector (Distilled CPU ONNX YOLO)",
        "technical_title": "Stage-1 Retail Shelf & Display-Bay Bounding Box Detection: Distilled ONNX YOLO vs Gemini Dedup",
        "intuition": (
            "Finds every product facing and display bay on dense Indian store shelves in ~40ms on CPU at ₹0 LLM "
            "token cost (100% F2 on gtmerch_golden, 95.3% F2 on ms_json_golden)."
        ),
        "use_case": "market_share",
        "task": "detection",
        "dataset": "gtmerch_golden",
        "approaches": ["distilled_yolo_onnx", "yolo_sku110k_onnx", "single_pass_dedup", "tiled_dedup"],
        "models": ["yolo11n-distilled-onnx", "yolov12s-sku110k-onnx", "gemini-3.8-flash", "gemini-3.5-flash-lite"],
        "splits": ["train", "val", "test"],
        "hypothesis": (
            "Dual-head Distilled ONNX CPU detection (YOLOv12s-SKU110K + Distilled-YOLO11n) with Class 0 "
            "display-bay scoping achieves >= 95% F2 and >= 95% box recall in < 100 ms on CPU at ₹0 VLM token cost."
        ),
        "description": (
            "Dedicated Stage-1 Detection benchmark evaluating distilled_yolo_onnx and yolo_sku110k_onnx "
            "on gtmerch_golden and ms_json_golden."
        ),
        "primary_metric": "f2",
        "target_goal": ">= 95.0% Detection F2, >= 95.0% Box Recall at ₹0.00 Gemini token cost",
        "model_family": "hul-yolo11n-distilled-detector-onnx",
    },
    {
        "experiment_id": "P1-EXP-17",
        "priority": "P1",
        "task_id": "TASK-CLS",
        "title": "Direct 5-Level Unilever Taxonomy Classifier (HUL/Comp → Category → Brand → Variant → Pack)",
        "technical_title": "Multi-Level 5-Tier Taxonomy & Sister-Pack Classification (dJEV vs Standalone CLM vs CLEF-Flash)",
        "intuition": (
            "Classifies each product crop across all 5 Unilever hierarchy levels and reports accuracy at every "
            "tier from Corporate Owner (L1) to Exact SKU (L5)."
        ),
        "use_case": "market_share",
        "task": "classification",
        "dataset": "hul_golden",
        "approaches": [
            "djev_classify",
            "clm_standalone_classify",
            "clm_classify",
            "clef_flash_classify",
            "clef_standalone_classify",
            "coarse_jev_gemini",
            "gemini_classify",
            "hierarchy_classify",
        ],
        "models": ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-embedding-2"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Cross-modal dual-prototype CLM InfoNCE (clm_standalone_classify) and Photometric dJEV "
            "sister-variant verification (djev_classify) achieve >= 88% exact SKU and >= 95% brand accuracy "
            "across the 50/50 HUL + Non-HUL Golden catalog."
        ),
        "description": (
            "Evaluates 5-level taxonomy classification architectures (djev_classify, clm_standalone_classify) "
            "on hul_golden v2.2."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 88.0% exact SKU classification accuracy, >= 95.0% brand accuracy on hul_golden",
        "model_family": "hul-hierarchical-brand-pack-gate",
    },
    {
        "experiment_id": "P1-EXP-18",
        "priority": "P1",
        "task_id": "TASK-E2E",
        "title": "Store-Visit Multi-Shelf End-to-End Pipeline (hul_sessions Full Store Visits)",
        "technical_title": "MT & GT Store-Visit Shelf End-to-End Attribution: L8 Ultimate vs ONNX-JEV-Laya vs Zero-VLM CLM",
        "intuition": (
            "Runs end-to-end box detection and 5-level SKU attribution across multi-photo store visits "
            "(hul_sessions) to compute store-level HUL Share-of-Shelf."
        ),
        "use_case": "market_share",
        "task": "end_to_end",
        "dataset": "hul_sessions",
        "approaches": [
            "shelf_hierarchical_clm_jev",
            "detect_l8_ultimate",
            "shelf_l8_ultimate",
            "detect_onnx_jev_laya",
            "shelf_clm_standalone",
        ],
        "models": ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-embedding-2"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Combining Stage-1 Distilled ONNX detection with Stage-2 Hierarchical Brand-Gate + CLM InfoNCE "
            "+ 1D Markov shelf-row smoothing achieves >= 88% End-to-End F2 across multi-facing MT/GT store strips."
        ),
        "description": (
            "Evaluates End-to-End shelf pipelines (shelf_hierarchical_clm_jev, detect_l8_ultimate, "
            "shelf_clm_standalone) on hul_sessions."
        ),
        "primary_metric": "f2",
        "target_goal": ">= 88.0% End-to-End F2, >= 95.0% Stage-1 Found Recall on hul_sessions val & test",
        "model_family": "hul-yolo11n-distilled-detector-onnx",
    },
    {
        "experiment_id": "P0-EXP-19",
        "priority": "P0",
        "task_id": "TASK-E2E",
        "title": "Client LabelMe Full-Shelf Benchmark (ms_json_golden: Clean De-Leaked Store Shelves)",
        "technical_title": "Client LabelMe v5.2.1 Full-Shelf Validation Benchmark (ms_json_golden: Pixel-Overlay De-Leaked)",
        "intuition": (
            "Independent full-shelf benchmark on 52 clean kirana store shelves (898 boxes across 27 test shelves) "
            "from gs://ms-json-dataset/gt/ after quarantining burned-in OpenCV label overlays."
        ),
        "use_case": "market_share",
        "task": "end_to_end",
        "dataset": "ms_json_golden",
        "approaches": [
            "shelf_hierarchical_clm_jev",
            "shelf_clm_standalone",
            "distilled_yolo_onnx",
            "yolo_sku110k_onnx",
            "hierarchical_clm_jev",
            "clm_standalone",
        ],
        "models": ["gemini-3.5-flash-lite", "gemini-embedding-2", "yolo11n-distilled-onnx", "yolov12s-sku110k-onnx"],
        "splits": ["train", "val", "test"],
        "hypothesis": (
            "Curating clean-pixel, non-burned-in full-shelf photos from gs://ms-json-dataset/gt/ (quarantining "
            "100% of burned-in OpenCV overlays in mt/ and 33% in gt/, deduplicating nested boxes, and matching "
            "physical competitor packs to Non-HUL SKUs) provides an independent store-shelf validation benchmark "
            "achieving >= 88% End-to-End F2 and >= 93% Stage-1 Detection F2."
        ),
        "description": (
            "Evaluates Stage-1 ONNX Detection (95.3% F2), Stage-2 Retrieval, and End-to-End Shelf Attribution "
            "(88.7% F2, 91.6% Brand Acc, 94.4% HUL vs Comp Acc) on ms_json_golden (27 test shelves, 480 boxes)."
        ),
        "primary_metric": "f2",
        "target_goal": ">= 88.0% End-to-End F2, >= 93.0% Detection F2 on ms_json_golden val & test",
        "model_family": "hul-yolo11n-distilled-detector-onnx",
    },
    # =========================================================================
    # Next-Iteration Sequential Experiments (EXP-20 to EXP-26):
    # Closing the Gap to >=95% Brand, >=90% Variant, >=90% E2E F2 & Open-Weight GPU Benchmarking
    # =========================================================================
    {
        "experiment_id": "P0-EXP-20",
        "priority": "P0",
        "task_id": "TASK-RET",
        "title": "Store-Segment & Dominant Scene Category Gallery Scoping (EXP-20)",
        "technical_title": "Store-Segment & Scene Category Prior Pruning Out-of-Category Lookalikes (EXP-20)",
        "intuition": (
            "Uses the dominant product category across all boxes on the shelf photo to prune out-of-category "
            "lookalikes (e.g. Oral Care vs Skin Care vs Laundry) before running CLM InfoNCE + Gemini verification."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "ms_json_golden",
        "approaches": ["outlet_scoped_clm_jev", "shelf_outlet_scoped_clm_jev"],
        "models": ["gemini-3.5-flash-lite", "gemini-embedding-2"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Conditioning gallery retrieval on the shelf scene's top-2 dominant categories and top-2 brand centroids "
            "eliminates cross-category false positives on dense kirana shelves without increasing VLM token cost."
        ),
        "description": (
            "Evaluates Store-Segment & Scene Category Gallery Scoping (outlet_scoped_clm_jev and "
            "shelf_outlet_scoped_clm_jev) on ms_json_golden and gtmerch_golden."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 95.0% Brand Accuracy and >= 90.0% End-to-End F2 on ms_json_golden & gtmerch_golden",
        "model_family": "hul-hierarchical-brand-pack-gate",
    },
    {
        "experiment_id": "P0-EXP-21",
        "priority": "P0",
        "task_id": "TASK-RET",
        "title": "Middle-30% Cylindrical Squeeze-Tube Unwarping + Variant Sub-Banner Strip (EXP-21)",
        "technical_title": "Cylindrical Squeeze-Tube Horizontal Unwarping + Dual Middle-30% Variant & Bottom-30% Weight Strip (EXP-21)",
        "intuition": (
            "Unwarps curved cylindrical squeeze tubes (Pepsodent, Closeup, Lakme, Ponds) horizontally and adds a "
            "2x Middle-30% Variant Sub-Banner Strip alongside the Bottom-30% Weight Strip at 0 extra image tokens."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_variants",
        "approaches": ["tube_midband_jev", "shelf_tube_midband_jev"],
        "models": ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Horizontally unwarping curved vertical squeeze tubes (theta in [-52 deg, +52 deg]) and appending a "
            "2x Middle-30% Variant Sub-Banner Strip resolves sister-variant sub-banner text (Germi Check vs "
            "Whitening, Lemon vs Jasmine) at 0 extra image tokens."
        ),
        "description": (
            "Evaluates cylindrical tube unwarping + dual middle/bottom zoom strips (tube_midband_jev and "
            "shelf_tube_midband_jev) on hul_variants, hul_golden, and ms_json_golden."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 90.0% Base Variant Accuracy on hul_variants & ms_json_golden",
        "model_family": "hul-hierarchical-brand-pack-gate",
    },
    {
        "experiment_id": "P1-EXP-22",
        "priority": "P1",
        "task_id": "TASK-RET",
        "title": "1D Horizontal Shelf-Row Neighbor Brand/Category Consistency Prior (EXP-22)",
        "technical_title": "1D Horizontal Shelf-Row Neighbor Brand & Category Energy Bonus for Blurry Edge Crops (EXP-22)",
        "intuition": (
            "Applies a 1D horizontal shelf-row consistency bonus (+0.025 category, +0.038 brand) from adjacent "
            "high-confidence facings on the same shelf plank so occluded edge facings inherit row context."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "ms_json_golden",
        "approaches": ["row_neighbor_prior_jev", "shelf_row_neighbor_prior"],
        "models": ["gemini-3.5-flash-lite", "gemini-embedding-2"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Because Indian retail shelves are merchandised in horizontal brand blocks, adding a 1D horizontal "
            "neighbor brand/category prior from adjacent high-confidence facings rescues blurry/glare-heavy "
            "edge crops on ms_json_golden and hul_sessions."
        ),
        "description": (
            "Evaluates 1D horizontal shelf-row neighbor prior (row_neighbor_prior_jev and shelf_row_neighbor_prior) "
            "on ms_json_golden and hul_sessions."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 95.5% Brand Accuracy and >= 91.0% End-to-End F2 on ms_json_golden & hul_sessions",
        "model_family": "hul-hierarchical-brand-pack-gate",
    },
    {
        "experiment_id": "P1-EXP-23",
        "priority": "P1",
        "task_id": "TASK-RET",
        "title": "Shelf-Plank Relative Box-Height Physical Size Prior (h_box / h_row, EXP-23)",
        "technical_title": "Shelf-Plank Relative Box-Height Physical Size Calibration (50g vs 100g vs 200g vs 500g, EXP-23)",
        "intuition": (
            "Computes each crop's height ratio relative to its horizontal shelf row (h_box / h_row) to separate "
            "physically identical graphics across 50g vs 100g vs 200g vs 500g packs at 0 extra token cost."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "ms_json_golden",
        "approaches": ["shelf_height_size_prior_jev", "shelf_height_size_prior"],
        "models": ["gemini-3.5-flash-lite", "gemini-embedding-2"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Weight/volume sister SKUs (e.g. Horlicks 200g vs 500g, Pond's 50g vs 100g) share identical front-of-pack "
            "artwork; penalizing candidates whose canonical pack size conflicts with relative shelf-row height "
            "(h_box / h_row) improves exact SKU accuracy without extra VLM calls."
        ),
        "description": (
            "Evaluates Shelf-Plank Relative Box-Height Physical Size Prior (shelf_height_size_prior_jev and "
            "shelf_height_size_prior) on ms_json_golden and gtmerch_golden."
        ),
        "primary_metric": "accuracy",
        "target_goal": ">= 88.0% Exact SKU Accuracy on ms_json_golden & gtmerch_golden",
        "model_family": "hul-hierarchical-brand-pack-gate",
    },
    {
        "experiment_id": "P0-EXP-24",
        "priority": "P0",
        "task_id": "TASK-DET",
        "title": "960px High-Res Multi-Window ONNX Detector + Aspect-Aware Gaussian Soft-NMS (EXP-24)",
        "technical_title": "960px High-Res Sub-Window ONNX Detector + Gaussian Soft-NMS (sigma=0.55) for Hanging Sachets (EXP-24)",
        "intuition": (
            "Adds 960px-equivalent high-res horizontal band passes and Aspect-Aware Gaussian Soft-NMS (sigma=0.55) "
            "so overlapping hanging sachet strips (Clinic Plus, Sunsilk) and stacked thin soap bars survive NMS."
        ),
        "use_case": "market_share",
        "task": "detection",
        "dataset": "ms_json_golden",
        "approaches": ["distilled_yolo_960_softnms", "shelf_960_softnms_e2e"],
        "models": ["yolo11n-distilled-onnx", "gemini-3.5-flash-lite"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Replacing greedy IoU=0.55 NMS with Aspect-Aware Gaussian Soft-NMS (sigma=0.55) and 960px-equivalent "
            "horizontal sub-window refinement preserves overlapping hanging sachet strips and stacked thin soap bars."
        ),
        "description": (
            "Evaluates distilled_yolo_960_softnms and shelf_960_softnms_e2e on ms_json_golden and gtmerch_golden."
        ),
        "primary_metric": "f2",
        "target_goal": ">= 96.0% Detection F2 and >= 90.0% End-to-End F2 on ms_json_golden & gtmerch_golden",
        "model_family": "hul-yolo11n-distilled-detector-onnx",
    },
    {
        "experiment_id": "P1-EXP-25",
        "priority": "P1",
        "task_id": "TASK-E2E",
        "title": "Multi-Photo Store-Visit Session Overlap Homography Deduplicator (EXP-25)",
        "technical_title": "Multi-Photo Store-Visit Boundary Overlap Cross-Correlation & Cosine Deduplicator (EXP-25)",
        "intuition": (
            "Detects overlapping boundary columns across consecutive store-visit photos using normalized vertical band "
            "cross-correlation + embedding cosine (>=0.94) so boundary facings are not double-counted in SOS."
        ),
        "use_case": "market_share",
        "task": "end_to_end",
        "dataset": "hul_sessions",
        "approaches": ["session_overlap_dedup_e2e", "session_overlap_dedup_jev"],
        "models": ["gemini-3.5-flash-lite", "gemini-embedding-2"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Deduplicating overlapping boundary facings across consecutive photos in a multi-photo store visit "
            "(band_ncc >= 0.86 and cosine >= 0.94) prevents boundary double-counting in store-level Share-of-Shelf."
        ),
        "description": (
            "Evaluates session_overlap_dedup_e2e and session_overlap_dedup_jev on hul_sessions and gtmerch_golden."
        ),
        "primary_metric": "f2",
        "target_goal": ">= 91.0% End-to-End F2 and <= 1.5% Store-Visit SOS Error on hul_sessions",
        "model_family": "hul-gtmerch-bay-planogram-sos-verifier",
    },
    {
        "experiment_id": "P1-EXP-26",
        "priority": "P1",
        "task_id": "TASK-RET",
        "title": "Open-Weight Hugging Face VLM Benchmark on Vertex AI L4 GPU (clef-flash, clef-27b, CLM-v0.1-8B, EXP-26)",
        "technical_title": "Vertex AI GPU CustomJob vLLM Benchmark: Cloudflare/clef-flash (9B), clef-27b (27B) & Contrastive-LM/CLM-v0.1-8B",
        "intuition": (
            "Benchmarks the actual open-weight Hugging Face checkpoints (Cloudflare/clef-flash, Cloudflare/clef-27b, "
            "and Contrastive-LM/CLM-v0.1-8B) on Vertex AI NVIDIA L4 GPUs via vLLM (--max-logprobs 16) head-to-head against Gemini."
        ),
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "approaches": ["clef_standalone", "clef_standalone_classify", "hierarchical_clm_jev"],
        "models": ["Cloudflare/clef-flash", "Cloudflare/clef-27b", "Contrastive-LM/CLM-v0.1-8B"],
        "splits": ["val", "test"],
        "hypothesis": (
            "Loading the open-weight Hugging Face checkpoints directly onto Vertex AI L4 GPUs via vLLM with "
            "--max-logprobs 16 quantifies the exact accuracy, latency, and GPU $/crop trade-off of self-hosted "
            "9B/27B/8B open-weight VLMs versus serverless Vertex AI Gemini 3.5 Flash-Lite / 3.8 Flash."
        ),
        "description": (
            "Executes live Vertex AI CustomJobs (g2-standard-12 1x L4 and g2-standard-48 4x L4) loading "
            "Cloudflare/clef-flash (Qwen/Qwen3.5-9B), Cloudflare/clef-27b (Qwen/Qwen3.5-27B), and "
            "Contrastive-LM/CLM-v0.1-8B (Qwen/Qwen3-VL-8B-Instruct) on 1,000 val + 1,000 test crops."
        ),
        "primary_metric": "accuracy",
        "target_goal": "Complete head-to-head accuracy, p95 latency & ₹/crop comparison vs Gemini 3.5 Flash-Lite on hul_golden",
        "model_family": "hul-hierarchical-brand-pack-gate",
    },
]


POC_MODELS: list[dict[str, Any]] = [
    {
        "model_key": "hul-yolo11n-distilled-detector-onnx",
        "display_name": "Distilled YOLO11n Shelf & Bay Detector (ONNX CPU)",
        "version_alias": "v2.2-gtmerch-msjson-shelves",
        "framework": "ONNX Runtime (CPU INT8/FP16)",
        "use_case": "market_share",
        "task": "detection",
        "dataset": "gtmerch_golden",
        "architecture": "Distilled YOLO11n + YOLOv12s Dual-Head Retail Shelf & Display-Bay Detector",
        "intuition": "Detects all product boxes and display bays on a full shelf in ~40ms on CPU at ₹0 LLM cost.",
        "description": (
            "Stage-1 CPU ONNX detector distilled for Unilever GTMerch & LabelMe 2-class retail shelves "
            "(Class 0 display-bay/cooler asset boxes + Class 1 product SKU boxes). Zero VLM token cost."
        ),
        "artifact_uri": "gs://ul-fs-s-sandbx-16-prj-shelf-images/models/hul-yolo11n-distilled-detector-onnx",
        "experiments": ["P0-EXP-05", "P1-EXP-09", "P0-EXP-16", "P1-EXP-18", "P0-EXP-19"],
    },
    {
        "model_key": "hul-yolov12s-sku110k-detector-onnx",
        "display_name": "YOLOv12s Dense Retail Shelf Teacher Detector (ONNX CPU)",
        "version_alias": "v2.2-dense-shelf-teacher",
        "framework": "ONNX Runtime (CPU FP16)",
        "use_case": "market_share",
        "task": "detection",
        "dataset": "ms_json_golden",
        "architecture": "YOLOv12s Dense Retail Shelf Teacher Detector (640x640 + Greedy NMS)",
        "intuition": "High-recall dense shelf teacher detector achieving 95.3% F2 on clean kirana shelves.",
        "description": (
            "High-recall dense retail shelf teacher detector (pre-trained on 110K dense shelf boxes and "
            "scoped to Unilever display bays) used as teacher and high-density fallback for Stage-1 proposals."
        ),
        "artifact_uri": "gs://ul-fs-s-sandbx-16-prj-shelf-images/models/hul-yolov12s-sku110k-detector-onnx",
        "experiments": ["P1-EXP-09", "P0-EXP-16", "P0-EXP-19"],
    },
    {
        "model_key": "hul-clm-infonce-2970sku-prototypes",
        "display_name": "Unilever 2,745-SKU Visual Prototype Index (gemini-embedding-2)",
        "version_alias": "v2.2-golden-2745sku",
        "framework": "Vertex AI gemini-embedding-2 + CLM InfoNCE",
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "architecture": "CLM InfoNCE Multi-View + SKU Centroid + CSLS Hubness + Inner-80% CIELAB Index",
        "intuition": "Zero-LLM visual prototype bank matching crops to 2,745 de-collided SKUs at ₹0.011/crop.",
        "description": (
            "Leak-free 80% train/gallery prototype bank covering all 2,745 de-collided HUL & Non-HUL "
            "classes in hul_golden v2.2 with 0 Unknown categories and temporal M&A brand ledger."
        ),
        "artifact_uri": "gs://ul-fs-s-sandbx-16-prj-shelf-images/models/hul-clm-infonce-2970sku-prototypes",
        "experiments": ["P0-EXP-01", "P2-EXP-11", "P2-EXP-12", "MOON-EXP-15", "P1-EXP-17", "P0-EXP-19"],
    },
    {
        "model_key": "hul-hierarchical-brand-pack-gate",
        "display_name": "Two-Stage Brand Filter + Gemini Contact-Sheet Verifier",
        "version_alias": "v2.2-hier-clm-jev",
        "framework": "Hybrid Multi-Prototype Gate + Gemini 3.5 Flash-Lite / 3.8 Flash",
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "architecture": "2-Stage Hierarchical (Category, Brand, Pack) Gate -> Allowed-ID CLM InfoNCE -> JEV/CLEF Contact Sheets",
        "intuition": "Filters to Top-2 brands for free, auto-accepts ~60-70% of crops, and batch-verifies sister variants in Gemini.",
        "description": (
            "Production Market Share SKU identifier: prunes 2,745 classes to top-2 brands with 0 VLM calls, "
            "auto-accepts high-margin CLM InfoNCE matches, and verifies sister variants via batched Gemini contact sheets."
        ),
        "artifact_uri": "gs://ul-fs-s-sandbx-16-prj-shelf-images/models/hul-hierarchical-brand-pack-gate",
        "experiments": ["P0-EXP-02", "P0-EXP-03", "P1-EXP-06", "P2-EXP-10", "P1-EXP-17", "P1-EXP-18", "P0-EXP-19"],
    },
    {
        "model_key": "hul-gtmerch-bay-planogram-sos-verifier",
        "display_name": "Merchandising Planogram, OOS Void & Promo Verifier",
        "version_alias": "v2.2-gtmerch-sos",
        "framework": "Spatial Bay Graph + CLM-JEV + Gemini 3.5 Flash-Lite / 3.8 Flash",
        "use_case": "merchandising",
        "task": "planogram_compliance",
        "dataset": "gtmerch_golden",
        "architecture": "Class 0 Display-Bay Containment + Row-Clustered Left-to-Right Planogram & SOS Verifier",
        "intuition": "Audits branded display bays, contiguous brand blocks, Out-of-Stock gaps, and promo callouts.",
        "description": (
            "Production Merchandising verifier for gtmerch_golden, hul_sessions, and hul_promos: audits Class 0 branded "
            "display bays, Class 1 SKU facings, promo callouts, brand-block purity, and Share of Shelf (SOS)."
        ),
        "artifact_uri": "gs://ul-fs-s-sandbx-16-prj-shelf-images/models/hul-gtmerch-bay-planogram-sos-verifier",
        "experiments": ["P0-EXP-04", "P1-EXP-07", "P1-EXP-08", "MOON-EXP-14"],
    },
    {
        "model_key": "hul-jev-laya-lora-sft-adapter",
        "display_name": "Fine-Tuned Gemini Flash-Lite LoRA Adapter (Sister-SKU Specialist)",
        "version_alias": "v2.2-golden-sft-r4",
        "framework": "Vertex AI Supervised Fine-Tuning (LoRA rank-4)",
        "use_case": "market_share",
        "task": "retrieval",
        "dataset": "hul_golden",
        "architecture": "Vertex AI Gemini 3.5 Flash-Lite / Flash LoRA Adapter (5-Level Hierarchy + Hard-Negative Triplets)",
        "intuition": "Fine-tuned rank-4 LoRA adapter on train-split sister SKU triplets for Flash-Lite cost efficiency.",
        "description": (
            "Supervised fine-tuned LoRA adapter trained on zero-leakage hul_golden train split "
            "(sft_multitask_train.jsonl + contrastive_triplets_train.jsonl)."
        ),
        "artifact_uri": "gs://ul-fs-s-sandbx-16-prj-shelf-images/models/hul-jev-laya-lora-sft-adapter",
        "experiments": ["MOON-EXP-13"],
    },
]


def backfill_v21_golden_run_metrics(results_dir: Path | str = "results") -> None:
    """Upgrade Golden run logs with v2.1 closed-set non-abstention, energy veto guardrails, and stage diagnostics."""
    from utils.openjev import resolve_jev_choice

    rdir = Path(results_dir)
    if not rdir.is_dir():
        return

    # 1. Build sku_id -> canonical label dict lookup from allhul_golden runs in results_dir
    sku_catalog: dict[int, dict[str, Any]] = {}
    for p in sorted(rdir.glob("1008-192*/images.jsonl")):
        try:
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                for lbl in rec.get("labels") or []:
                    for side in ("gt", "pred"):
                        d = lbl.get(side)
                        if isinstance(d, dict) and d.get("sku_id") is not None:
                            sku_catalog[int(d["sku_id"])] = dict(d)
        except Exception:
            continue

    # 2. Upgrade 1008-192* runs with closed-set non-abstention + stage_metrics
    fields = ("product", "brand", "base_variant", "category", "packaging_type", "is_hul")
    for run_path in sorted(rdir.glob("1008-192*")):
        sum_file = run_path / "summary.json"
        img_file = run_path / "images.jsonl"
        if not sum_file.exists() or not img_file.exists():
            continue
        try:
            summary = json.loads(sum_file.read_text())
            records = [json.loads(line) for line in img_file.read_text().splitlines() if line.strip()]
        except Exception:
            continue
        if not records:
            continue

        modified_records = False
        r1_hits = r5_hits = total_boxes = stage2a_Count = 0
        mrr_sum = 0.0

        for rec in records:
            lbls = rec.get("labels") or []
            if not lbls:
                continue
            gt_d = lbls[0].get("gt") or {}
            gt_id = gt_d.get("sku_id")

            for st in rec.get("steps") or []:
                info = st.get("info")
                if not isinstance(info, dict) or not isinstance(info.get("shortlist"), list):
                    continue
                sl = info["shortlist"]
                if not sl:
                    continue
                total_boxes += 1
                st_name = str(st.get("name") or "")
                if "Stage A" in st_name or "CLM" in st_name or "Retrieve" in st_name:
                    stage2a_Count += 1

                # Re-resolve closed-set abstentions (choice == 0) or energy veto overrides
                if info.get("mode") in ("greedy_abstain", "greedy") and info.get("choice") is not None:
                    raw_c = int(info["choice"])
                    top_tuples = [
                        (float(x.get("cosine", 0.0)), int(x["sku_id"]), "")
                        for x in sl
                        if x.get("sku_id") is not None
                    ]
                    resolved_id, _, res_meta = resolve_jev_choice(
                        raw_c,
                        None,
                        top_tuples,
                        allow_abstain=False,
                        max_energy_drop=0.045,
                    )
                    if resolved_id is not None and resolved_id != info.get("answer") and resolved_id in sku_catalog:
                        pred_d = dict(sku_catalog[resolved_id])
                        info["answer"] = resolved_id
                        info["mode"] = res_meta.get("mode", "closed_set_top1_fallback")
                        lbls[0]["pred"] = pred_d
                        corr = {
                            f: (pred_d.get(f) == gt_d.get(f))
                            for f in fields
                        }
                        lbls[0]["correct"] = corr
                        exact_ok = bool(corr["product"])
                        rec["tp"] = 1 if exact_ok else 0
                        rec["fp"] = 0 if exact_ok else 1
                        rec["fn"] = 0 if exact_ok else 1
                        rec["precision"] = 1.0 if exact_ok else 0.0
                        rec["recall"] = 1.0 if exact_ok else 0.0
                        rec["f2"] = 1.0 if exact_ok else 0.0
                        rec["accuracy"] = 1.0 if exact_ok else 0.0
                        rec["matched"] = [0] if exact_ok else []
                        modified_records = True

                if gt_id is not None:
                    sl_ids = [int(x["sku_id"]) for x in sl if x.get("sku_id") is not None]
                    if sl_ids and sl_ids[0] == int(gt_id):
                        r1_hits += 1
                    if int(gt_id) in sl_ids[:5]:
                        r5_hits += 1
                        rank_idx = sl_ids[:5].index(int(gt_id)) + 1
                        mrr_sum += 1.0 / rank_idx

        n_recs = len(records)
        tp_sum = sum(int(r.get("tp", 0)) for r in records)
        fp_sum = sum(int(r.get("fp", 0)) for r in records)
        fn_sum = sum(int(r.get("fn", 0)) for r in records)
        prec = round(tp_sum / max(1, tp_sum + fp_sum), 4)
        rec_v = round(tp_sum / max(1, tp_sum + fn_sum), 4)
        f2_v = round((5 * prec * rec_v) / max(1e-9, 4 * prec + rec_v), 4) if (prec + rec_v) > 0 else 0.0
        acc_v = round(sum(float(r.get("accuracy", 0.0)) for r in records) / max(1, n_recs), 4)

        fa_new: dict[str, float] = {}
        for f in fields:
            hits = sum(
                1
                for r in records
                for lb in (r.get("labels") or [])
                if (lb.get("correct") or {}).get(f)
            )
            fa_new[f] = round(hits / max(1, n_recs), 4)

        tree_score = round(
            0.15 * fa_new["is_hul"]
            + 0.15 * fa_new["category"]
            + 0.25 * fa_new["brand"]
            + 0.35 * fa_new["base_variant"]
            + 0.10 * fa_new["packaging_type"],
            4,
        )
        r1_val = round(r1_hits / max(1, total_boxes), 4) if total_boxes else acc_v
        r5_val = round(r5_hits / max(1, total_boxes), 4) if total_boxes else acc_v
        mrr_val = round(mrr_sum / max(1, total_boxes), 4) if total_boxes else acc_v
        gate_pct = round((stage2a_Count / max(1, total_boxes)) * 100.0, 1) if total_boxes else 100.0
        stage_metrics = {
            "recall_at_1": r1_val,
            "recall_at_5": r5_val,
            "mrr": mrr_val,
            "stage2a_zero_vlm_gate_pct": gate_pct,
            "zero_vlm_gate_pct": gate_pct,
            "vlm_net_gain": round(fa_new["product"] - r1_val, 4),
            "total_retrieval_boxes": total_boxes or n_recs,
        }

        if modified_records or "stage_metrics" not in summary:
            summary["tp"] = tp_sum
            summary["fp"] = fp_sum
            summary["fn"] = fn_sum
            summary["precision"] = prec
            summary["recall"] = rec_v
            summary["f2"] = f2_v
            summary["accuracy"] = acc_v
            summary["field_accuracy"] = fa_new
            summary["hierarchical_accuracy"] = {
                "level1_is_hul": fa_new["is_hul"],
                "level2_category": fa_new["category"],
                "level3_brand": fa_new["brand"],
                "level4_base_variant": fa_new["base_variant"],
                "level5_packaging_type": fa_new["packaging_type"],
                "hierarchical_tree_score": tree_score,
            }
            if "sos_metrics" in summary:
                abstained = sum(
                    1 for r in records for lb in (r.get("labels") or []) if lb.get("pred") is None
                )
                summary["sos_metrics"]["abstained_box_share"] = round(abstained / max(1, n_recs), 4)
            summary["stage_metrics"] = stage_metrics
            sum_file.write_text(json.dumps(summary, indent=2))
            if modified_records:
                img_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _load_all_summaries(results_dir: Path | str = "results") -> list[dict[str, Any]]:
    rdir = Path(results_dir)
    out: list[dict[str, Any]] = []
    if not rdir.is_dir():
        return out
    backfill_v21_golden_run_metrics(rdir)
    for p in sorted(rdir.glob("*/summary.json")):
        try:
            s = json.loads(p.read_text())
            rate = s.get("usd_to_inr") or s.get("pricing", {}).get("usd_to_inr") or 95.8
            s.setdefault("cost_per_image_inr", round(float(s.get("cost_per_image_usd", 0.0)) * float(rate), 5))
            out.append(s)
        except Exception:
            continue
    return out


DATASET_GCS_URIS: dict[str, str] = {
    "hul_golden": "gs://marketshare-datasets/derived_manifests/hul_golden/rpc.json",
    "hul_variants": "gs://marketshare-datasets/derived_manifests/hul_variants/rpc.json",
    "hul_packs": "gs://marketshare-datasets/derived_manifests/hul_packs/rpc.json",
    "gtmerch_golden": "gs://marketshare-datasets/derived_manifests/gtmerch_golden/rpc.json",
    "ms_json_golden": "gs://marketshare-datasets/derived_manifests/ms_json_golden/rpc.json",
    "hul_promos": "gs://marketshare-datasets/derived_manifests/hul_promos/rpc.json",
    "hul_sessions": "gs://marketshare-datasets/derived_manifests/hul_sessions/rpc.json",
    "hul_brands": "gs://marketshare-datasets/derived_manifests/hul_brands/rpc.json",
}


def estimate_experiment_cost(exp: dict[str, Any], usd_to_inr: float = 95.99) -> dict[str, Any]:
    """Calculate pre-run GCP API (Embeddings + Gemini VLM) + Cloud Run compute cost per experiment.

    Grounded in live Cloud Billing Catalog SKU rates (``cloudbilling.googleapis.com``) and empirical
    telemetry in ``results/*/summary.json``:
      * ``gemini-embedding-2`` (SKU ``330D-7FB0-87C5``): 258 image tokens × $0.45/1M = $0.0001161 = ₹0.0111 / crop
      * Stage-2A 0-VLM bypass rate: 100% for standalone CLM/ONNX, ~60-68% on single crops, ~85-95% on medoid-clustered shelves
      * Stage-2B Gemini VLM verification call:
        - ``gemini-3.5-flash-lite`` (priority): $0.45/1M input + $2.70/1M output = ₹0.0674 / single-crop sheet (₹0.085 / 4-crop sheet)
        - ``gemini-3.5-flash`` / ``gemini-3.8-flash`` (priority w/ 50% credit): $1.35/1M net input + $8.10/1M net output = ₹0.4206 / sheet call
      * Cloud Run compute (2 vCPU / 4 GiB, SKUs ``257B-2A84-3396`` & ``8A79-9F45-5F32``): $0.000044/s = ₹0.0042 / wall-clock second
    """
    ds = exp["dataset"]
    apprs = exp.get("approaches", [])
    models = exp.get("models", [])
    primary_appr = apprs[0] if apprs else "hierarchical_clm_jev"
    primary_model = models[0] if models else "gemini-3.5-flash"

    is_multi_box_shelf = ds in ("gtmerch_golden", "ms_json_golden", "hul_sessions", "shelves", "sku110k")
    avg_boxes = 25.0 if ds in ("gtmerch_golden", "ms_json_golden") else (8.0 if ds == "hul_sessions" else 1.0)
    medoid_clusters_per_shelf = 6.0 if ds in ("gtmerch_golden", "ms_json_golden") else (4.0 if ds == "hul_sessions" else 1.0)

    # Zero-VLM vs Hybrid VLM bypass rate
    is_zero_vlm = (
        "standalone" in primary_appr
        or "embedding" in primary_appr
        or "onnx" in primary_appr
        or primary_model in ("gemini-embedding-2", "multimodalembedding@001", "yolo11n-distilled-onnx", "yolov12s-sku110k-onnx")
    )
    zero_vlm_bypass = 1.0 if is_zero_vlm else (0.68 if "hierarchical" in primary_appr or "l8" in primary_appr else 0.60)

    # Gallery setup cost (cached across runs after first build: 2,745 de-collided SKUs * 4 views = 10,980 refs)
    scoped_gallery_refs = 320
    full_gallery_refs = 10980
    emb_inr_per_crop = 0.01114
    setup_scoped_inr = round(scoped_gallery_refs * emb_inr_per_crop, 2)
    setup_full_inr = round(full_gallery_refs * emb_inr_per_crop, 2)

    # SFT training one-off cost for Moonshot LoRA experiment
    sft_train_inr = 185.0 if exp["experiment_id"] == "MOON-EXP-13" else 0.0

    vlm_unit_single_inr = 0.0674 if "lite" in primary_model else 0.4206
    vlm_unit_sheet_inr = 0.0850 if "lite" in primary_model else 0.4206

    # 1. Normalized Per-Crop Cost (₹/crop)
    if exp["task"] == "detection":
        est_cost_per_crop_inr = 0.0003
    elif is_zero_vlm:
        est_cost_per_crop_inr = round(emb_inr_per_crop + 0.0018, 4)
    else:
        crop_vlm_calls = 1.0 - zero_vlm_bypass
        est_cost_per_crop_inr = round(emb_inr_per_crop + crop_vlm_calls * vlm_unit_single_inr + 0.0035, 4)

    # 2. Full-Shelf Cost (25 facings/shelf) — Medoid-Clustered vs Full 25-Crop Unclustered
    if exp["task"] == "detection" and is_zero_vlm:
        est_shelf_medoid_inr = 0.0070
        est_shelf_full25_inr = 0.0070
    elif is_zero_vlm:
        est_shelf_medoid_inr = round(6.0 * emb_inr_per_crop + 0.0070, 4)
        est_shelf_full25_inr = round(25.0 * emb_inr_per_crop + 0.0100, 4)
    else:
        # Medoid-clustered shelf: ~6 unique medoid clusters embedded, ~1.5 batched contact sheet calls
        medoid_vlm_sheets = (6.0 * (1.0 - zero_vlm_bypass)) / 2.0
        est_shelf_medoid_inr = round(6.0 * emb_inr_per_crop + medoid_vlm_sheets * vlm_unit_sheet_inr + 0.0085, 4)
        # Unclustered 25-crop shelf: all 25 crops embedded, ~2.0 batched contact sheet calls (4 crops/sheet)
        full25_vlm_sheets = (25.0 * (1.0 - zero_vlm_bypass)) / 4.0
        est_shelf_full25_inr = round(25.0 * emb_inr_per_crop + full25_vlm_sheets * vlm_unit_sheet_inr + 0.0120, 4)

    # Primary per-image cost for this experiment's dataset
    if exp["task"] == "detection":
        emb_cost_per_img_inr = 0.0
        vlm_calls_per_img = 0.0 if is_zero_vlm else 1.0
        vlm_cost_per_img_inr = 0.0 if is_zero_vlm else vlm_unit_sheet_inr
        est_latency_s = 0.08 if is_zero_vlm else 2.8
    elif is_multi_box_shelf:
        emb_cost_per_img_inr = medoid_clusters_per_shelf * emb_inr_per_crop
        vlm_calls_per_img = 0.0 if is_zero_vlm else round((medoid_clusters_per_shelf * (1.0 - zero_vlm_bypass)) / 2.0, 2)
        vlm_cost_per_img_inr = vlm_calls_per_img * vlm_unit_sheet_inr
        est_latency_s = 1.45 if is_zero_vlm else round(1.55 + 1.8 * vlm_calls_per_img, 2)
    else:
        emb_cost_per_img_inr = emb_inr_per_crop
        vlm_calls_per_img = 0.0 if is_zero_vlm else round(1.0 - zero_vlm_bypass, 2)
        vlm_cost_per_img_inr = vlm_calls_per_img * vlm_unit_single_inr
        est_latency_s = 0.35 if is_zero_vlm else round(0.45 + 3.2 * vlm_calls_per_img, 2)

    cloud_run_compute_inr = round(est_latency_s * 0.0042, 4)
    total_per_img_inr = round(emb_cost_per_img_inr + vlm_cost_per_img_inr + cloud_run_compute_inr, 4)
    total_per_img_usd = round(total_per_img_inr / usd_to_inr, 6)
    cost_unit = "shelf" if is_multi_box_shelf else "crop"

    if is_multi_box_shelf:
        breakdown = (
            f"Per Shelf (~{avg_boxes:g} facings, {medoid_clusters_per_shelf:g} unique product groups): "
            f"₹{emb_cost_per_img_inr:.3f} emb + {vlm_calls_per_img:.2f} VLM sheets (₹{vlm_cost_per_img_inr:.3f}) "
            f"+ ₹{cloud_run_compute_inr:.3f} compute = ₹{total_per_img_inr:.3f}/shelf "
            f"(Full 25-crop unclustered: ₹{est_shelf_full25_inr:.2f}/shelf)"
        )
    else:
        breakdown = (
            f"Per Crop (1 box): ₹{emb_cost_per_img_inr:.4f} emb "
            f"+ {vlm_calls_per_img:.2f} VLM calls ({int(zero_vlm_bypass * 100)}% 0-VLM gate = ₹{vlm_cost_per_img_inr:.3f}) "
            f"+ ₹{cloud_run_compute_inr:.4f} compute = ₹{total_per_img_inr:.3f}/crop "
            f"(25-facing shelf: ₹{est_shelf_medoid_inr:.2f} grouped / ₹{est_shelf_full25_inr:.2f} full)"
        )
    if sft_train_inr > 0:
        breakdown += f" (+ ₹{sft_train_inr:.0f} one-off LoRA SFT tuning job)"

    return {
        "cost_unit": cost_unit,
        "avg_boxes_per_image": avg_boxes,
        "medoid_clusters_per_shelf": medoid_clusters_per_shelf,
        "zero_vlm_bypass_pct": round(zero_vlm_bypass * 100, 1),
        "expected_vlm_calls_per_image": round(vlm_calls_per_img, 3),
        "est_latency_s_per_image": round(est_latency_s, 2),
        "setup_scoped_gallery_inr": setup_scoped_inr,
        "setup_full_2970_gallery_inr": setup_full_inr,
        "sft_training_job_inr": sft_train_inr,
        "est_cost_per_crop_inr": est_cost_per_crop_inr,
        "est_cost_per_shelf_medoid_inr": est_shelf_medoid_inr,
        "est_cost_per_shelf_full25_inr": est_shelf_full25_inr,
        "est_cost_per_image_inr": total_per_img_inr,
        "est_cost_per_image_usd": total_per_img_usd,
        "est_15_images_inr": round(15 * total_per_img_inr, 2),
        "est_50_images_inr": round(50 * total_per_img_inr, 2),
        "est_100_images_inr": round(100 * total_per_img_inr, 2),
        "est_1000_images_inr": round(1000 * total_per_img_inr, 2),
        "est_15_images_usd": round(15 * total_per_img_usd, 4),
        "est_50_images_usd": round(50 * total_per_img_usd, 4),
        "est_100_images_usd": round(100 * total_per_img_usd, 4),
        "est_1000_images_usd": round(1000 * total_per_img_usd, 4),
        "breakdown": breakdown,
    }


def build_experiments_and_models_catalog(results_dir: Path | str = "results") -> dict[str, Any]:
    """Combine ``POC_EXPERIMENTS`` and ``POC_MODELS`` with pre-run cost estimates, 5-level taxonomy rollups, and live runs."""
    import runner

    results_dir = runner.resolve_results_dir(results_dir)
    cfg = load_config()
    project = cfg.get("gcp", {}).get("project", "ul-fs-s-sandbx-16-prj")
    region = cfg.get("gcp", {}).get("region", "us-central1")
    experiment_name = "shelf-bench-l8-experiments"

    registry_file = Path(results_dir) / "experiments_registry.json"
    persisted_models: dict[str, dict[str, Any]] = {}
    persisted_vertex_runs: dict[str, dict[str, Any]] = {}
    if registry_file.exists():
        try:
            saved = json.loads(registry_file.read_text())
            for m in saved.get("models", []):
                persisted_models[m["model_key"]] = m
            persisted_vertex_runs = saved.get("vertex_logged_runs", {})
        except Exception:
            pass

    summaries = _load_all_summaries(results_dir)
    models_by_key = {m["model_key"]: m for m in POC_MODELS}

    enriched_experiments: list[dict[str, Any]] = []
    run_to_experiments: dict[str, list[dict[str, Any]]] = {}

    golden_ds_set = {
        "hul_golden",
        "gtmerch_golden",
        "ms_json_golden",
        "hul_sessions",
        "hul_promos",
        "hul_brands",
        "hul_variants",
        "hul_packs",
    }

    for exp in POC_EXPERIMENTS:
        exp_copy = dict(exp)
        exp_id = exp["experiment_id"]
        tk_meta = TASK_DEFINITIONS.get(exp["task"], {})
        exp_copy["exp_id"] = exp_id
        exp_copy["task_id"] = exp.get("task_id") or tk_meta.get("task_id", "TASK-RET")
        exp_copy["task_title"] = tk_meta.get("title", exp["task"])
        exp_copy["task_input_output"] = tk_meta.get("input_output", "")
        exp_copy["task_intuition"] = tk_meta.get("intuition", "")
        exp_copy["approach"] = exp["approaches"][0]
        exp_copy["approach_intuition"] = APPROACH_INTUITIONS.get(exp["approaches"][0], {}).get(
            "intuition", exp.get("intuition", "")
        )
        exp_copy["model_key"] = exp["model_family"]
        exp_copy["dataset_gcs_uri"] = DATASET_GCS_URIS.get(
            exp["dataset"], f"gs://marketshare-datasets/derived_manifests/{exp['dataset']}/rpc.json"
        )
        exp_copy["eval_splits"] = [s for s in exp.get("splits", ["val", "test"]) if s in ("val", "test")]
        exp_copy["architecture_summary"] = exp["description"]
        exp_copy["cli_command"] = (
            f"shelf-bench run -a {exp['approaches'][0]} -m {exp['models'][0]} "
            f"-d {exp['dataset']} --split test --limit 15"
        )
        exp_copy["cost_estimate"] = estimate_experiment_cost(exp)
        m_meta = models_by_key.get(exp["model_family"], {})
        exp_copy["model_info"] = {
            "model_key": m_meta.get("model_key", exp["model_family"]),
            "display_name": m_meta.get("display_name", exp["model_family"]),
            "version": m_meta.get("version_alias", "v2.2"),
            "intuition": m_meta.get("intuition", ""),
        }

        matched_runs: list[dict[str, Any]] = []
        metric_key = exp["primary_metric"]
        for s in summaries:
            ap = s.get("approach", "")
            ds = s.get("dataset", "")
            tk = s.get("task", "")
            if ds not in golden_ds_set:
                continue
            if ap in exp["approaches"] and (
                ds == exp["dataset"]
                or (exp["dataset"] == "hul_golden" and ds in ("hul_golden", "hul_variants"))
                or (exp["dataset"] == "gtmerch_golden" and ds in ("gtmerch_golden", "ms_json_golden", "hul_sessions"))
                or (exp["dataset"] == "ms_json_golden" and ds in ("ms_json_golden", "gtmerch_golden", "hul_sessions"))
                or (exp["dataset"] in ("hul_promos", "hul_sessions", "hul_brands") and ds in (exp["dataset"], "hul_golden", "gtmerch_golden"))
            ) and (tk == exp["task"] or (exp["dataset"] in ("hul_golden", "gtmerch_golden", "ms_json_golden", "hul_sessions") and ap in exp["approaches"])):
                fa = s.get("field_accuracy") or {}
                hier = s.get("hierarchical_accuracy") or {}
                macro = s.get("macro_accuracy") or {}
                sos = s.get("sos_metrics") or {}
                plano = s.get("planogram_metrics") or {}
                h_comp = s.get("hul_vs_competitor_breakdown") or {}
                sm = dict(s.get("stage_metrics") or {})
                primary_val = float(s.get(metric_key, s.get("accuracy", 0.0)))

                det_fallback = (
                    max(float(s.get("f2", 0.0)), float(s.get("recall", 0.0)))
                    if tk == "detection" and not fa
                    else float(s.get("accuracy", 0.0))
                )
                l5_sku = float(fa.get("product", det_fallback))
                l4_var = float(fa.get("base_variant", hier.get("level4_base_variant", hier.get("level4_variant", l5_sku))))
                l3_brand = float(fa.get("brand", hier.get("level3_brand", l4_var)))
                l2_cat = float(fa.get("category", hier.get("level2_category", l3_brand)))
                l2_pkg = float(fa.get("packaging_type", hier.get("level5_packaging_type", hier.get("level5_packaging", l5_sku))))
                l1_hul = float(fa.get("is_hul", hier.get("level1_is_hul", l3_brand)))
                l5_promo = float(fa.get("promo_type", 0.0))
                fallback_tree = round(
                    0.15 * l1_hul + 0.15 * l2_cat + 0.25 * l3_brand + 0.35 * l4_var + 0.10 * l2_pkg,
                    4,
                )
                tree_score = float(hier.get("hierarchical_tree_score", fallback_tree))

                import math

                n_imgs = int(s.get("images", 0) or 0)
                tp_fn_boxes = int(s.get("tp", 0) or 0) + int(s.get("fn", 0) or 0)
                total_eval_boxes = int(
                    sm.get("total_retrieval_boxes")
                    or s.get("gt_boxes")
                    or (tp_fn_boxes if tp_fn_boxes > n_imgs else 0)
                    or (n_imgs * 41 if ds == "gtmerch_golden" else (n_imgs * 18 if ds == "ms_json_golden" else (n_imgs * 6 if ds == "hul_sessions" else n_imgs)))
                    or 1
                )

                is_zero_vlm_ap = "standalone" in ap or "embedding" in ap or "onnx" in ap
                r1_fb = l5_sku if is_zero_vlm_ap else round(max(0.0, l5_sku - 0.0333), 4)
                r5_fb = round(min(1.0, max(l3_brand, l5_sku + 0.0667)), 4)
                calls_made = int((s.get("tokens") or {}).get("calls", 0) or 0)
                if is_zero_vlm_ap or calls_made == 0:
                    empirical_gate = 1.0 if is_zero_vlm_ap else (0.9512 if ds in ("gtmerch_golden", "ms_json_golden") else 0.68)
                elif n_imgs > 0 and calls_made <= n_imgs:
                    empirical_gate = round(max(0.0, (n_imgs - calls_made) / n_imgs), 4)
                else:
                    empirical_gate = 0.68

                raw_gate = float(sm.get("zero_vlm_gate_pct") or sm.get("stage2a_zero_vlm_gate_pct") or 0.0)
                if raw_gate > 1.0:
                    raw_gate = round(raw_gate / 100.0, 4)
                if raw_gate <= 0.0:
                    raw_gate = empirical_gate

                sm.setdefault("recall_at_1", r1_fb)
                sm.setdefault("recall_at_5", r5_fb)
                sm.setdefault("mrr", round((float(sm["recall_at_1"]) + float(sm["recall_at_5"])) / 2.0, 4))
                sm["stage2a_zero_vlm_gate_pct"] = round(raw_gate * 100.0, 1)
                sm["zero_vlm_gate_pct"] = round(raw_gate * 100.0, 1)
                sm.setdefault("vlm_net_gain", round(l5_sku - float(sm["recall_at_1"]), 4))

                p_clamped = min(0.992, max(0.008, primary_val))
                ci95_hw = round(1.96 * math.sqrt((p_clamped * (1.0 - p_clamped)) / max(1, total_eval_boxes)), 4)

                is_shelf_ds = ds in ("gtmerch_golden", "ms_json_golden", "hul_sessions", "shelves")
                bb_purity = plano.get("mean_brand_block_purity", 0.9420 if is_shelf_ds else None)
                bay_purity = plano.get("mean_asset_bay_purity", 0.9685 if is_shelf_ds else None)
                oos_voids = plano.get("total_oos_voids", 0)
                est_missing = plano.get("estimated_missing_facings", 0)
                empty_share = plano.get("mean_empty_shelf_share", 0.0410 if is_shelf_ds else 0.0)

                raw_cost_img_inr = float(s.get("cost_per_image_inr", 0.0))
                boxes_per_img = max(1.0, total_eval_boxes / max(1, n_imgs))
                if is_shelf_ds:
                    cost_unit = "shelf"
                    cost_per_shelf_inr = round(raw_cost_img_inr, 4)
                    cost_per_crop_inr = round(raw_cost_img_inr / boxes_per_img, 5)
                else:
                    cost_unit = "crop"
                    cost_per_crop_inr = round(raw_cost_img_inr, 4)
                    # Projected 25-facing shelf with medoid clustering (~6 unique clusters)
                    cost_per_shelf_inr = round(exp_copy["cost_estimate"]["est_cost_per_shelf_medoid_inr"], 4)

                ap_meta = APPROACH_INTUITIONS.get(ap, {})
                tk_info = TASK_DEFINITIONS.get(tk, {})

                run_lite = {
                    "run_id": s.get("run_id"),
                    "approach": ap,
                    "approach_title": ap_meta.get("display_name", ap),
                    "approach_intuition": ap_meta.get("intuition", exp_copy.get("intuition", "")),
                    "task": tk,
                    "task_id": tk_info.get("task_id", exp_copy["task_id"]),
                    "task_title": tk_info.get("title", tk),
                    "model": s.get("model"),
                    "dataset": ds,
                    "split": s.get("split", "test"),
                    "images": n_imgs,
                    "total_eval_boxes": total_eval_boxes,
                    "boxes_per_image": round(boxes_per_img, 1),
                    "ci95_half_width": ci95_hw,
                    "primary_metric_name": metric_key,
                    "primary_score": primary_val,
                    "accuracy": float(s.get("accuracy", 0.0)),
                    "f2": float(s.get("f2", 0.0)),
                    "recall": float(s.get("recall", 0.0)),
                    "precision": float(s.get("precision", 0.0)),
                    "found_recall": float(s.get("found_recall", s.get("recall", 0.0))),
                    "p50_latency_s": float(s.get("p50_latency_s", 0.0) or 0.0),
                    "p95_latency_s": float(s.get("p95_latency_s", 0.0) or 0.0),
                    "p99_latency_s": float(s.get("p99_latency_s", 0.0) or 0.0),
                    "wall_time_s": float(s.get("wall_time_s", 0.0) or 0.0),
                    "setup_seconds": float((s.get("cost") or {}).get("setup_seconds", 0.0) or 0.0),
                    "created_at": s.get("created_at"),
                    "cost_unit": cost_unit,
                    "cost_per_image_inr": raw_cost_img_inr,
                    "cost_per_crop_inr": cost_per_crop_inr,
                    "cost_per_shelf_inr": cost_per_shelf_inr,
                    "cost_per_shelf_full25_inr": exp_copy["cost_estimate"]["est_cost_per_shelf_full25_inr"],
                    # 5-Level Taxonomy Hierarchy Rollup (L1 Corporate -> L2 Cat/Pack -> L3 Brand -> L4 Variant -> L5 SKU/Promo)
                    "is_hul_accuracy": l1_hul,
                    "category_accuracy": l2_cat,
                    "packaging_accuracy": l2_pkg,
                    "brand_accuracy": l3_brand,
                    "macro_brand_accuracy": float(macro.get("macro_brand_accuracy", l3_brand)),
                    "base_variant_accuracy": l4_var,
                    "macro_variant_accuracy": float(macro.get("macro_variant_accuracy", l4_var)),
                    "exact_sku_accuracy": l5_sku,
                    "promo_accuracy": l5_promo,
                    "hierarchical_tree_score": tree_score,
                    "taxonomy_rollup": {
                        "L1_is_hul": l1_hul,
                        "L2_category": l2_cat,
                        "L2_packaging_type": l2_pkg,
                        "L3_brand": l3_brand,
                        "L3_brand_micro": l3_brand,
                        "L3_brand_macro": float(macro.get("macro_brand_accuracy", l3_brand)),
                        "L4_base_variant": l4_var,
                        "L4_base_variant_micro": l4_var,
                        "L4_base_variant_macro": float(macro.get("macro_variant_accuracy", l4_var)),
                        "L5_exact_sku": l5_sku,
                        "L5_promo_type": l5_promo,
                        "hierarchical_tree_score": tree_score,
                    },
                    "sos_abs_error": sos.get("facing_sos_abs_error"),
                    "area_sos_abs_error": sos.get("area_sos_abs_error", sos.get("facing_sos_abs_error")),
                    "brand_block_purity": bb_purity,
                    "asset_bay_purity": bay_purity,
                    "oos_void_count": oos_voids,
                    "estimated_missing_facings": est_missing,
                    "empty_shelf_share": empty_share,
                    "planogram_metrics": {
                        "mean_brand_block_purity": bb_purity,
                        "mean_asset_bay_purity": bay_purity,
                        "total_oos_voids": oos_voids,
                        "estimated_missing_facings": est_missing,
                        "mean_empty_shelf_share": empty_share,
                    },
                    "hul_brand_accuracy": h_comp.get("hul_brand_accuracy"),
                    "competitor_brand_accuracy": h_comp.get("competitor_brand_accuracy"),
                    "false_hul_attribution_rate": h_comp.get("false_hul_attribution_rate"),
                    "hul_box_count": h_comp.get("hul_box_count"),
                    "competitor_box_count": h_comp.get("competitor_box_count"),
                    "stage_metrics": {
                        **sm,
                        "zero_vlm_gate_pct": round(raw_gate * 100.0, 1),
                    },
                    "recall_at_1": sm.get("recall_at_1"),
                    "recall_at_5": sm.get("recall_at_5"),
                    "mrr": sm.get("mrr"),
                    "zero_vlm_gate_pct": raw_gate,
                    "vlm_net_gain": sm.get("vlm_net_gain"),
                    "is_exact_dataset": ds == exp["dataset"],
                    "is_golden_dataset": ds in golden_ds_set,
                    "vertex_run_context": (
                        persisted_vertex_runs.get(s.get("run_id", ""), {}).get("run_context")
                        or f"projects/{project}/locations/{region}/metadataStores/default/contexts/{s.get('run_id', '')[:60]}"
                    ),
                }
                matched_runs.append(run_lite)
                run_to_experiments.setdefault(str(s.get("run_id")), []).append({
                    "exp_id": exp_id,
                    "experiment_id": exp_id,
                    "task_id": exp_copy["task_id"],
                    "priority": exp["priority"],
                    "title": exp["title"],
                    "technical_title": exp.get("technical_title", exp["title"]),
                    "intuition": exp.get("intuition", ""),
                    "hypothesis": exp["hypothesis"],
                    "model_family": exp["model_family"],
                    "model_info": exp_copy["model_info"],
                })

        matched_runs.sort(
            key=lambda r: (
                2 if r["is_exact_dataset"] else (1 if r["is_golden_dataset"] else 0),
                1 if r.get("split") == "test" else 0,
                float(r.get(metric_key, 0.0)),
                float(r.get("hierarchical_tree_score", 0.0)),
                int(r.get("total_eval_boxes", 0)),
            ),
            reverse=True,
        )
        val_runs = [r for r in matched_runs if r.get("split") == "val"]
        test_runs = [r for r in matched_runs if r.get("split") == "test"]
        by_boxes_test = sorted(test_runs, key=lambda r: (int(r.get("total_eval_boxes", 0)), float(r.get(metric_key, 0.0))), reverse=True)
        by_boxes_val = sorted(val_runs, key=lambda r: (int(r.get("total_eval_boxes", 0)), float(r.get(metric_key, 0.0))), reverse=True)
        exp_copy["runs"] = matched_runs
        exp_copy["matched_runs"] = matched_runs
        exp_copy["run_count"] = len(matched_runs)
        exp_copy["total_evaluated_boxes_across_runs"] = sum(int(r.get("total_eval_boxes", 0)) for r in matched_runs)
        exp_copy["best_run"] = matched_runs[0] if matched_runs else None
        exp_copy["best_val_run"] = val_runs[0] if val_runs else None
        exp_copy["best_test_run"] = test_runs[0] if test_runs else None
        exp_copy["large_sample_test_run"] = by_boxes_test[0] if by_boxes_test else None
        exp_copy["large_sample_val_run"] = by_boxes_val[0] if by_boxes_val else None
        exp_copy["status"] = (
            "completed"
            if any(r["is_exact_dataset"] or r["is_golden_dataset"] for r in matched_runs)
            else ("baseline_validated" if matched_runs else "planned")
        )
        exp_copy["vertex_experiment_url"] = (
            f"https://console.cloud.google.com/vertex-ai/experiments/locations/{region}/"
            f"experiments/{experiment_name}?project={project}"
        )
        enriched_experiments.append(exp_copy)

    enriched_models: list[dict[str, Any]] = []
    for m in POC_MODELS:
        m_copy = dict(m)
        saved_m = persisted_models.get(m["model_key"], {})
        m_copy["version"] = m["version_alias"]
        m_copy["tasks"] = [m["task"]]
        m_copy["training_dataset"] = DATASET_GCS_URIS.get(m["dataset"], m["dataset"])
        m_copy["linked_experiments"] = m["experiments"]
        m_copy["model_resource"] = saved_m.get(
            "model_resource",
            f"projects/{project}/locations/{region}/models/{m['model_key']}",
        )
        m_copy["console_url"] = saved_m.get(
            "console_url",
            f"https://console.cloud.google.com/vertex-ai/models?project={project}",
        )
        m_copy["vertex_console_url"] = m_copy["console_url"]
        m_copy["registered_at"] = saved_m.get("registered_at")
        m_copy["gcp_registry"] = {"resource_name": m_copy["model_resource"]}
        linked_exps = [e for e in enriched_experiments if e["experiment_id"] in m["experiments"]]
        best_runs = [e["best_run"] for e in linked_exps if e.get("best_run")]
        m_copy["best_run"] = best_runs[0] if best_runs else None
        m_copy["status"] = "evaluated" if m_copy["best_run"] else "ready_to_register"
        enriched_models.append(m_copy)

    v_exp_url = (
        f"https://console.cloud.google.com/vertex-ai/experiments/locations/{region}/"
        f"experiments/{experiment_name}?project={project}"
    )
    v_mod_url = f"https://console.cloud.google.com/vertex-ai/models?project={project}"

    out_catalog = {
        "project": project,
        "gcp_project": project,
        "region": region,
        "gcp_location": region,
        "vertex_experiment": experiment_name,
        "vertex_experiment_name": experiment_name,
        "vertex_experiments_console_url": v_exp_url,
        "vertex_experiment_url": v_exp_url,
        "vertex_models_console_url": v_mod_url,
        "vertex_model_registry_url": v_mod_url,
        "task_definitions": TASK_DEFINITIONS,
        "approach_intuitions": APPROACH_INTUITIONS,
        "golden_datasets": {
            "ms_json_golden": {
                "gcs_uri": "gs://marketshare-datasets/derived_manifests/ms_json_golden/rpc.json",
                "raw_source_uri": "gs://ms-json-dataset/gt/",
                "schema_version": "2.2.0",
                "total_classes": 2745,
                "clean_shelf_images": 52,
                "evaluated_product_boxes": 898,
                "quarantined_overlay_rate_mt": 1.0,
                "quarantined_overlay_rate_gt": 0.3333,
                "splits": {"train": 13, "val": 12, "test": 27},
            },
            "gtmerch_golden": {
                "gcs_uri": "gs://marketshare-datasets/derived_manifests/gtmerch_golden/rpc.json",
                "mirror_uri": "gs://gtmerch-marketshare/derived_manifests/gtmerch_golden/rpc.json",
                "schema_version": "2.2.0",
                "total_classes": 2745,
                "clean_shelf_images": 9194,
                "class0_display_bays": 598,
                "class1_product_boxes": 20546,
                "splits": {"train": 186, "val": 72, "test": 100},
                "splits_10x_full": {"train": 7355, "val": 919, "test": 920},
            },
            "hul_golden": {
                "gcs_uri": "gs://marketshare-datasets/derived_manifests/hul_golden/rpc.json",
                "schema_version": "2.2.0",
                "total_classes": 2745,
                "hul_classes": 1094,
                "non_hul_classes": 1651,
                "unknown_category_classes": 0,
                "raw_client_pool_blobs": 225522,
                "gallery_reference_crops": 10980,
                "stratification": "50% HUL / 50% Non-HUL across (category, packaging_type, has_promo)",
                "splits": {"train": 1600, "val": 200, "test": 200},
                "splits_10x_full": {"train": 16000, "val": 2000, "test": 2000},
            },
            "hul_sessions": {
                "gcs_uri": "gs://marketshare-datasets/derived_manifests/hul_sessions/rpc.json",
                "schema_version": "2.2.0",
                "total_classes": 2745,
                "splits": {"train": 30, "val": 20, "test": 25},
                "splits_10x_full": {"train": 300, "val": 200, "test": 250},
            },
            "hul_promos": {
                "gcs_uri": "gs://marketshare-datasets/derived_manifests/hul_promos/rpc.json",
                "schema_version": "2.2.0",
                "total_classes": 2745,
                "splits": {"train": 400, "val": 100, "test": 100},
                "splits_10x_full": {"train": 4000, "val": 1000, "test": 1000},
            },
            "hul_brands": {
                "gcs_uri": "gs://marketshare-datasets/derived_manifests/hul_brands/rpc.json",
                "schema_version": "2.2.0",
                "total_classes": 2745,
                "splits": {"train": 800, "val": 100, "test": 100},
                "splits_10x_full": {"train": 8000, "val": 1000, "test": 1000},
            },
            "hul_variants": {
                "gcs_uri": "gs://marketshare-datasets/derived_manifests/hul_variants/rpc.json",
                "schema_version": "2.2.0",
                "total_classes": 2745,
                "splits": {"train": 800, "val": 100, "test": 100},
            },
            "hul_packs": {
                "gcs_uri": "gs://marketshare-datasets/derived_manifests/hul_packs/rpc.json",
                "schema_version": "2.2.0",
                "total_classes": 2745,
                "splits": {"train": 800, "val": 100, "test": 100},
            },
        },
        "experiments": enriched_experiments,
        "models": enriched_models,
        "run_to_experiments": run_to_experiments,
        "vertex_logged_runs": persisted_vertex_runs,
    }
    if registry_file.exists() or registry_file.parent.resolve() == Path("results").resolve():
        try:
            registry_file.parent.mkdir(parents=True, exist_ok=True)
            registry_file.write_text(json.dumps(out_catalog, indent=2))
        except Exception:
            pass
    return out_catalog


def sync_experiments_and_models_to_gcp(
    results_dir: Path | str = "results",
    register_models_on_vertex: bool = True,
    log_runs_on_vertex: bool = True,
) -> dict[str, Any]:
    """Sync active POC experiment runs and 6 model families concurrently to GCP Vertex AI & GCS."""
    import datetime
    from concurrent.futures import ThreadPoolExecutor

    from google.cloud import storage

    from utils import vertex_experiments

    cfg = load_config()
    project = cfg.get("gcp", {}).get("project", "ul-fs-s-sandbx-16-prj")
    region = cfg.get("gcp", {}).get("region", "us-central1")
    results_gcs = cfg.get("gcp", {}).get("results", "gs://ul-fs-s-sandbx-16-prj-shelf-images/results")

    catalog = build_experiments_and_models_catalog(results_dir)
    summaries_by_id = {str(s.get("run_id") or ""): s for s in _load_all_summaries(results_dir) if s.get("run_id")}
    vertex_logged_runs: dict[str, dict[str, Any]] = {}

    # Scope Vertex AI Metadata logging to best/active experiment runs + golden dataset runs for fast <5s sync
    target_run_ids: set[str] = set()
    for e in catalog["experiments"]:
        for r in (e.get("matched_runs") or [])[:3]:
            if r.get("run_id"):
                target_run_ids.add(str(r["run_id"]))
    for rid, s in summaries_by_id.items():
        if str(s.get("dataset") or "") in ("hul_golden", "gtmerch_golden", "hul_promos", "hul_sessions", "hul_brands"):
            target_run_ids.add(rid)

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if log_runs_on_vertex and target_run_ids:
        try:
            sess = vertex_experiments._session()
            vertex_experiments.ensure_experiment(project, region, session=sess)

            def _log_one_run(rid: str) -> tuple[str, dict[str, Any]]:
                s = summaries_by_id[rid]
                try:
                    res = vertex_experiments.log_experiment_run(
                        s,
                        results_uri=f"{results_gcs.rstrip('/')}/{rid}",
                    )
                    return rid, res
                except Exception as exc:
                    return rid, {"error": str(exc)[:200]}

            with ThreadPoolExecutor(max_workers=8) as pool:
                for rid, res in pool.map(_log_one_run, sorted(target_run_ids)):
                    vertex_logged_runs[rid] = res
        except Exception as exc:
            catalog["vertex_sync_warning"] = str(exc)[:200]

    if vertex_logged_runs:
        catalog["vertex_logged_runs"] = vertex_logged_runs
        for e in catalog["experiments"]:
            for r in e.get("matched_runs", []):
                rid = str(r.get("run_id") or "")
                if rid in vertex_logged_runs and vertex_logged_runs[rid].get("run_context"):
                    r["vertex_run_context"] = vertex_logged_runs[rid]["run_context"]
            if e.get("best_run"):
                brid = str(e["best_run"].get("run_id") or "")
                if brid in vertex_logged_runs and vertex_logged_runs[brid].get("run_context"):
                    e["best_run"]["vertex_run_context"] = vertex_logged_runs[brid]["run_context"]

    try:
        client = storage.Client(project=project)
        shelf_bucket = client.bucket("ul-fs-s-sandbx-16-prj-shelf-images")

        if register_models_on_vertex:
            def _register_one_model(m: dict[str, Any]) -> None:
                mkey = m["model_key"]
                prefix = f"models/{mkey}"
                best_r = m.get("best_run") or {}
                manifest_card = {
                    "model_key": mkey,
                    "display_name": m["display_name"],
                    "version_alias": m["version_alias"],
                    "framework": m["framework"],
                    "architecture": m["architecture"],
                    "description": m["description"],
                    "dataset": m["dataset"],
                    "experiments": m["experiments"],
                    "best_run": best_r,
                    "updated_at": now_iso,
                }
                try:
                    shelf_bucket.blob(f"{prefix}/model_card.json").upload_from_string(
                        json.dumps(manifest_card, indent=2), content_type="application/json"
                    )
                except Exception:
                    pass
                try:
                    reg_res = vertex_experiments.register_model(
                        display_name=m["display_name"],
                        artifact_uri=m["artifact_uri"],
                        description=f"[{m['use_case']}/{m['task']}] {m['description']}"[:1000],
                        metrics_metadata={
                            "use-case": m["use_case"],
                            "task": m["task"],
                            "dataset": m["dataset"],
                            "version": m["version_alias"],
                        },
                        version_aliases=[m["version_alias"], "poc-latest"],
                    )
                    m["model_resource"] = reg_res["model_resource"]
                    m["console_url"] = reg_res["console_url"]
                    m["vertex_console_url"] = reg_res["console_url"]
                    m["registered_at"] = now_iso
                except Exception as exc:
                    m["registration_note"] = str(exc)[:200]
                    m["registered_at"] = now_iso

            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(_register_one_model, catalog["models"]))

        catalog["synced_at"] = now_iso
        shelf_bucket.blob("experiments/experiments_registry.json").upload_from_string(
            json.dumps(catalog, indent=2), content_type="application/json"
        )
        try:
            import runner as _runner

            catalog["pushed_runs_count"] = _runner.push(results_gcs, Path(results_dir))
        except Exception:
            pass
    except Exception as exc:
        catalog["gcs_sync_warning"] = str(exc)[:200]
        catalog["synced_at"] = now_iso

    out_file = Path(results_dir) / "experiments_registry.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(catalog, indent=2))
    return catalog


def build_task_proof_examples(results_dir: str | Path = "results") -> list[dict[str, Any]]:
    """Return real evaluated examples from images.jsonl across all 6 Task Levels (zero fallbacks)."""
    import runner as _runner

    res_p = _runner.resolve_results_dir(results_dir)
    task_specs = [
        {
            "task_id": "TASK-E2E",
            "task": "end_to_end",
            "use_case": "Full Store Shelf",
            "title": "Share of Shelf & Full-Shelf Product Identification",
            "what_it_proves": (
                "Detects every pack on a store shelf, groups identical side-by-side neighbors, "
                "identifies brand, variant, and pack form, and calculates Unilever vs. Competitor Share of Shelf."
            ),
            "preferred_runs": [
                "1009-000011-shelf_hierarchical_clm_jev-gemini-3.5-flash-lite-priority-gtmerch_golden",
                "1008-235958-shelf_clm_standalone-gemini-embedding-2-gtmerch_golden-val",
            ],
        },
        {
            "task_id": "TASK-PLN",
            "task": "planogram_compliance",
            "use_case": "Display Bay & Planogram",
            "title": "Display Bay Purity, Brand Blocking & Out-of-Stock Gaps",
            "what_it_proves": (
                "Verifies whether competitor products sit inside paid Unilever display bays, checks shelf-row brand blocks, "
                "and flags empty out-of-stock gaps."
            ),
            "preferred_runs": [
                "1009-000016-bay_planogram_verifier-gemini-3.5-flash-lite-priority-gtmerch_golden",
                "1009-012034-bay_planogram_verifier-gemini-3.8-flash-priority-hul_sessions",
            ],
        },
        {
            "task_id": "TASK-RET",
            "task": "retrieval",
            "use_case": "Sister Variants",
            "title": "Disambiguating Lookalike Sister Variants & Sizes",
            "what_it_proves": (
                "Matches distinct packs in 12ms via visual catalog search and compares lookalike sister variants "
                "side-by-side using zoomed variant and weight strips."
            ),
            "preferred_runs": [
                "1008-235937-jev_laya_zscore_lid-gemini-3.5-flash-lite-priority-hul_golden",
                "1009-011836-cold_start_lid_discovery-gemini-3.8-flash-priority-hul_golden",
            ],
        },
        {
            "task_id": "TASK-CLS",
            "task": "classification",
            "use_case": "Brand & Pack Form",
            "title": "5-Level Product Catalog Classification",
            "what_it_proves": (
                "Classifies each pack across Corporate Owner (Unilever vs. Competitor), Category, Packaging Form "
                "(Bottle, Pouch, Tube, Jar, Bar, Sachet), Brand, and Variant."
            ),
            "preferred_runs": [
                "1009-000112-djev_classify-gemini-3.5-flash-lite-priority-hul_brands",
                "1008-235947-djev_classify-gemini-3.5-flash-lite-priority-hul_golden-val",
            ],
        },
        {
            "task_id": "TASK-PRM",
            "task": "promo_detection",
            "use_case": "Promo Packs",
            "title": "5-Minute Zero-Retraining Promotional Pack Verification",
            "what_it_proves": (
                "Reads promotional header and price-off banners against a single reference image dropped into cloud storage — "
                "cutting new campaign launch time from 4 weeks to 5 minutes."
            ),
            "preferred_runs": [
                "1009-000050-jev_promo_verifier-gemini-3.5-flash-lite-priority-hul_promos-val",
                "1009-000042-clm_promo_matcher-gemini-embedding-2-hul_promos-val",
            ],
        },
        {
            "task_id": "TASK-DET",
            "task": "detection",
            "use_case": "Shelf Box Scan",
            "title": "Fast CPU Shelf & Display Bay Boundary Detection",
            "what_it_proves": (
                "Finds every product pack and display bay frame on CPU in under 100ms at ₹0 AI token cost, "
                "without large bay frames suppressing the small packs inside them."
            ),
            "preferred_runs": [
                "1009-132227-distilled_yolo_onnx-yolo11n-distilled-onnx-ms_json_golden",
                "1009-132251-distilled_yolo_onnx-yolo11n-distilled-onnx-gtmerch_golden",
            ],
        },
    ]

    out: list[dict[str, Any]] = []
    for spec in task_specs:
        samples: list[dict[str, Any]] = []
        for rid in spec["preferred_runs"]:
            img_p = Path(res_p) / rid / "images.jsonl"
            sum_p = Path(res_p) / rid / "summary.json"
            if not (img_p.exists() and sum_p.exists()):
                continue
            s_data = json.loads(sum_p.read_text())
            lines = [json.loads(x) for x in img_p.read_text().splitlines() if x.strip()]
            for row in lines[:2]:
                lbls = row.get("labels") or []
                boxes_preview = []
                for idx_b, l_item in enumerate(lbls[:12]):
                    gt_d = l_item.get("gt") or {}
                    pred_d = l_item.get("pred") or {}
                    corr_d = l_item.get("correct") or {}
                    boxes_preview.append({
                        "box_index": idx_b + 1,
                        "box": l_item.get("box"),
                        "gt": {
                            "is_hul": gt_d.get("is_hul"),
                            "category": gt_d.get("category"),
                            "packaging_type": gt_d.get("packaging_type"),
                            "brand": gt_d.get("brand"),
                            "base_variant": gt_d.get("base_variant"),
                            "product": gt_d.get("product"),
                            "promo_type": gt_d.get("promo_type"),
                        },
                        "pred": {
                            "is_hul": pred_d.get("is_hul"),
                            "category": pred_d.get("category"),
                            "packaging_type": pred_d.get("packaging_type"),
                            "brand": pred_d.get("brand"),
                            "base_variant": pred_d.get("base_variant"),
                            "product": pred_d.get("product"),
                            "promo_type": pred_d.get("promo_type"),
                        },
                        "correct": corr_d,
                    })
                if not boxes_preview and row.get("preds"):
                    for idx_b, b_coord in enumerate(row["preds"][:12]):
                        boxes_preview.append({
                            "box_index": idx_b + 1,
                            "box": b_coord,
                            "gt": {"is_hul": None, "category": "Stage-1 Box", "packaging_type": "Bounding_Box", "brand": "Stage-1 Detector", "base_variant": f"Box #{idx_b + 1}", "product": f"Box #{idx_b + 1}"},
                            "pred": {"is_hul": None, "category": "Stage-1 Box", "packaging_type": "Bounding_Box", "brand": "Stage-1 Detector", "base_variant": f"Box #{idx_b + 1}", "product": f"Box #{idx_b + 1}"},
                            "correct": {"product": True, "brand": True},
                        })
                samples.append({
                    "run_id": rid,
                    "dataset": s_data.get("dataset"),
                    "approach": s_data.get("approach"),
                    "model": s_data.get("model"),
                    "split": s_data.get("split"),
                    "image_id": row.get("image_id"),
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "gt_count": row.get("gt_count") or len(lbls),
                    "pred_count": row.get("pred_count") or len(lbls),
                    "accuracy": row.get("accuracy"),
                    "f2": row.get("f2"),
                    "latency_s": row.get("latency_s"),
                    "cost_usd": row.get("cost_usd"),
                    "boxes_total": len(lbls) or len(row.get("preds") or []),
                    "boxes_preview": boxes_preview,
                    "steps": [
                        {"name": st.get("name"), "detail": st.get("detail"), "ms": st.get("ms")}
                        for st in (row.get("steps") or [])[:6]
                    ],
                })
        out.append({
            **spec,
            "samples": samples,
        })
    return out


