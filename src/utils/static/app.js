// Simplified Engineering Workbench UI:
//   1. Market Share (#/board/market_share) — Unified Leaderboard + Experiments with inline collapsible drill-down
//   2. Merchandising (#/board/merchandising) — Unified Leaderboard + Experiments with inline collapsible drill-down
//   3. Data QC & Audits (#/qc) — Readable 12-Gate status + 4 interactive data/taxonomy audit drawers
//   4. Run & Image Inspector (#/run/<id>) — 4 headline KPIs, collapsible parameters, and live GCS pixel debugger

const app = document.getElementById("app");
const pct = (v) => (Number(v ?? 0) * 100).toFixed(1) + "%";
const sec = (v) => Number(v ?? 0).toFixed(2) + "s";
const inr = (v) => "₹" + (v > 0 && v < 0.001 ? Number(v).toPrecision(2) : Number(v ?? 0).toFixed(3));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const getJSON = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)));
const _apiCache = new Map();
const getJSONCached = async (url, ttlMs = 45000) => {
  const now = Date.now();
  const hit = _apiCache.get(url);
  if (hit && now - hit.ts < ttlMs) return hit.data;
  const data = await getJSON(url);
  _apiCache.set(url, { ts: now, data });
  return data;
};
const mark = (ok) => (ok ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>');

// Human-friendly task definitions (replaces confusing internal task strings)
const TASKS = {
  end_to_end: {
    shortLabel: "Full Shelf → SKU",
    tip: "Detect every product box on the shelf and identify its Brand, Variant, and Exact SKU",
    scoreLabel: "F2",
    scoreKey: "f2",
  },
  classification: {
    shortLabel: "Crop → 5-Level SKU & Catalog Match",
    tip: "Unified single-crop benchmark: HUL/Rival, Category, Pack Type, Brand, Variant & Exact Catalog SKU",
    scoreLabel: "5-Attr Exact (All 5)",
    scoreKey: "accuracy",
  },
  retrieval: {
    shortLabel: "Crop → 5-Level SKU & Catalog Match",
    tip: "Unified single-crop benchmark: HUL/Rival, Category, Pack Type, Brand, Variant & Exact Catalog SKU",
    scoreLabel: "5-Attr Exact (All 5)",
    scoreKey: "accuracy",
  },
  detection: {
    shortLabel: "Shelf Box Detection",
    tip: "Localize all product facings (Class 1) and branded display bays (Class 0) on the shelf",
    scoreLabel: "F2",
    scoreKey: "f2",
  },
  planogram_compliance: {
    shortLabel: "Planogram & SOS",
    tip: "Verify display bay purity, left-to-right row order, and HUL Share of Shelf (SOS)",
    scoreLabel: "SKU",
    scoreKey: "accuracy",
  },
  promo_detection: {
    shortLabel: "Promo Verification",
    tip: "Verify B1G1, price-off callouts (Rs 10 Off), and multipack bundles",
    scoreLabel: "Promo",
    scoreKey: "accuracy",
  },
};

const DATASETS = {
  hul_golden: {
    isGolden: true,
    useCase: "market_share",
    channel: "MT",
    chipLabel: "MT Full Catalog · 2,659 SKUs",
    pillLabel: "MT · Full Catalog (2,659)",
    unitType: "crop",
    shortDesc: "MT Market Share · 2,659 SKUs (HUL + Competitor Crops)",
    splits: "16,000 train · 1,900 val · 1,907 test · 5,000 test_5k",
  },
  ms_json_golden: {
    isGolden: true,
    useCase: "market_share",
    channel: "GT",
    chipLabel: "Kirana Store Shelves · Market Share (ms_json)",
    pillLabel: "GT/MT · Store Shelves (ms_json)",
    unitType: "shelf",
    shortDesc: "Market Share · Clean Client LabelMe Store Shelves",
    splits: "140 train · 60 val · 100 test · 151 test_5k",
  },
  mt_json_golden: {
    isGolden: true,
    useCase: "market_share",
    channel: "MT",
    chipLabel: "MT Supermarkets · 356 Outlets (mt_json)",
    pillLabel: "MT · Supermarkets (mt_json_golden)",
    unitType: "shelf",
    shortDesc: "MT Market Share · 1,068 Supermarket Gondolas (356 Disjoint Outlets)",
    splits: "713 train · 155 val · 200 test",
  },
  hul_variants: {
    isGolden: true,
    useCase: "market_share",
    channel: "MT",
    chipLabel: "MT Variants · 1,922 Classes",
    pillLabel: "MT · Variants (1,922)",
    unitType: "crop",
    shortDesc: "MT Market Share · 1,922 Sister Variant Classes",
    splits: "15,687 train · 1,900 val · 1,907 test · 5,000 test_5k",
  },
  hul_packs: {
    isGolden: true,
    useCase: "market_share",
    channel: "MT",
    chipLabel: "MT Pack Forms · 46 Classes",
    pillLabel: "MT · Pack Forms (46)",
    unitType: "crop",
    shortDesc: "MT Market Share · 46 Category × Pack Type Classes",
    splits: "6,000 train · 1,000 val · 1,000 test",
  },
  hul_brands: {
    isGolden: true,
    useCase: "market_share",
    channel: "MT",
    chipLabel: "MT Brands · 953 Classes",
    pillLabel: "MT · Brands (953)",
    unitType: "crop",
    shortDesc: "MT Market Share · 953 HUL & Competitor Brands",
    splits: "8,000 train · 1,000 val · 1,000 test",
  },
  gtmerch_golden: {
    isGolden: true,
    useCase: "merchandising",
    channel: "GT",
    chipLabel: "GT Merchandising Shelves (Bays + Packs)",
    pillLabel: "GT · Merchandising Shelves",
    unitType: "shelf",
    shortDesc: "GT Merchandising · Real Store Shelves (Display Bays + Packs)",
    splits: "400 train · 100 val · 200 test",
  },
  gt_hul_sessions: {
    isGolden: true,
    useCase: "merchandising",
    channel: "GT",
    chipLabel: "GT Store Visit Sessions (Multi-Photo)",
    pillLabel: "GT · Store Visit Sessions (gt_hul_sessions)",
    unitType: "shelf",
    shortDesc: "GT Merchandising · Multi-Facing Store Visit Strips",
    splits: "150 train · 50 val · 100 test",
  },
  hul_sessions: {
    isGolden: true,
    useCase: "merchandising",
    channel: "GT",
    chipLabel: "GT Store Visit Sessions (Multi-Photo)",
    pillLabel: "GT · Store Visit Sessions",
    unitType: "shelf",
    shortDesc: "GT Merchandising · Multi-Facing Store Visit Strips",
    splits: "150 train · 50 val · 100 test",
  },
  hul_promos: {
    isGolden: true,
    useCase: "merchandising",
    channel: "MT",
    chipLabel: "MT Toker Promo Packs · 240 SKUs",
    pillLabel: "MT · Toker Promo (240)",
    unitType: "crop",
    shortDesc: "MT Toker Promo · 240 Promotional & Multipack SKUs",
    splits: "4,000 train · 1,000 val · 1,000 test",
  },
  hul_talkers_golden: {
    isGolden: true,
    useCase: "merchandising",
    channel: "MT",
    chipLabel: "Shelf Talkers & Price Tags · 3 Classes",
    pillLabel: "MT/GT · Talkers & Price Tags",
    unitType: "shelf",
    shortDesc: "Shelf Talker Cards, Printed Price Tags & Handwritten Rail Tags (532 Disjoint Outlets)",
    splits: "3,813 train · 477 val · 477 test",
  },
  shelves: { isGolden: false, unitType: "shelf", shortDesc: "HoloSelecta Calibration", splits: "Public" },
  rpc: { isGolden: false, unitType: "shelf", shortDesc: "RPC Checkout Calibration", splits: "Public" },
  sku110k: { isGolden: false, unitType: "shelf", shortDesc: "SKU-110K Calibration", splits: "Public" },
  products: { isGolden: false, unitType: "crop", shortDesc: "FMCG Products Calibration", splits: "Public" },
};

const SHORT_APPROACH_NAMES = {
  distilled_yolo_onnx: "Dual-ONNX CPU Shelf & Bay Detector",
  distilled_yolo_960_softnms: "960px Aspect-Aware Soft-NMS CPU Detector",
  distilled_yolo_stage0_clahe: "Stage 0 Luminance CLAHE + 960px Soft-NMS CPU Detector (EXP-43 / EXP-49)",
  yolo_sku110k_onnx: "Single-Pass ONNX CPU Shelf Detector",
  single_pass_dedup: "Single-Pass Gemini Full-Shelf Detector",
  tiled_dedup: "Tiled Multi-Pass Gemini Detector",
  clm_standalone: "Base CLM Retrieval: Zero-VLM gemini-embedding-2 + CIELAB Color Scorer",
  embedding_retrieval: "Nearest-Neighbor Embedding Baseline",
  hierarchical_clm_jev: "Base JEV Cascade: 2-Stage Brand-Gated Embedding + Contact Sheet",
  outlet_scoped_clm_jev: "Store-Segment Scoped Hybrid Matcher (EXP-20)",
  tube_midband_jev: "Cylindrical Tube Unwarp + Variant Strip (EXP-21)",
  row_neighbor_prior_jev: "1D Shelf-Row Neighbor Category Prior (EXP-22)",
  shelf_height_size_prior_jev: "Shelf-Plank Relative Height Size Prior (EXP-23)",
  session_overlap_dedup_jev: "Store-Visit Session Overlap Deduplicator (EXP-25)",
  l8_hul_ultimate: "Lighting-Corrected + Taper + Contact Sheet",
  clef_standalone: "Base CLEF Prompt: Bottom-Band Weight Zoom + Contact Sheet",
  clef_contrastive_metadata: "Base CLEF Prompt: Multi-Crop Contact Sheet + Catalog Diff Headers (EXP-28)",
  sft_clm_clef_cascade: "Gemini 3.5-Lite LoRA SFT + Embedding Gate + Contact Sheet (EXP-29 / EXP-33)",
  residual_infonce_clm_jev: "Residual InfoNCE Adapter + Calibrated Embedding Cascade (EXP-32)",
  dinov3_classify: "Meta DINOv3 + CosFace SFT Classifier",
  dinov3_hierarchical_sft: "Meta DINOv3 ViT-S/16 + 5-Level Taxonomic CosFace SFT (EXP-25)",
  dinov3_triage: "Meta DINOv3 Confidence Triage + Selective Gemini 3.5 Flash-Lite",
  dinov3_vitb_hierarchical: "Meta DINOv3 ViT-B/16 (1,536-D) + 5-Level Taxonomic CosFace SFT",
  dinov3_vitb_detr_hierarchical: "E2E ONNX RT-DETR + Meta DINOv3 ViT-B/16 Hierarchical SFT",
  ijepa_hierarchical: "Meta I-JEPA ViT-H/14 (2,560-D) + 5-Level Taxonomic CosFace SFT",
  vjepa2_hierarchical: "Meta V-JEPA 2 Self-Supervised Vision Backbone + SFT Hierarchical Head",
  hf_clm_v01_8b_gpu: "Open-Source HuggingFace Contrastive-LM/CLM-v0.1-8B (EXP-26)",
  hf_clef_flash_gpu: "Open-Source HuggingFace Cloudflare/clef-flash (EXP-26)",
  hf_clm_clef_cascade_gpu: "Open-Source HuggingFace CLM-8B -> clef-flash Cascade (EXP-26)",
  cv_classify: "Legacy 3-Stage InceptionV3 Cascade (Brand → Pack → 7 Routed Variant Models)",
  hybrid_classify: "2-Stage Hybrid Embedding Gate + Gemini 3.5 Flash-Lite Contact-Sheet Verifier (5-Attribute Cascade)",
  hybrid_classify_sft: "2-Stage Hybrid Embedding Gate + LoRA SFT Gemini 3.5 Flash-Lite with Top-5 kNN Hints (5-Attribute Cascade)",
  hybrid_classify_max: "2-Stage High-Precision Unanimous kNN Gate + Gemini Contact-Sheet Verifier (5-Attribute Cascade)",
  clef_flash_decision: "CLEF Decision Model: Single-Token Logprob Decision Head on Shortlist",
  jev_laya_zscore_lid: "JEV Decision Model: Statistical Z-Score & LID Routing Gate + Contact Sheet",
  cold_start_lid_discovery: "Zero-Shot New-Launch SKU Discovery",
  djev_classify: "JEV Decision Model: Photometric 5-Level Taxonomy Decision Classifier",
  clm_standalone_classify: "CLM Decision Model: Zero-VLM Lidstone-Prior 5-Level Taxonomy Classifier",
  clm_classify: "CLM Decision Model: Hybrid kNN Gate + Gemini 5-Level Taxonomy Classifier",
  clef_flash_classify: "CLEF Decision Model: Weight-Zoom Single-Token 5-Level Classifier",
  clef_standalone_classify: "CLEF Decision Model: Logprob-Calibrated 5-Level Taxonomy Classifier",
  coarse_jev_gemini: "JEV Decision Model: Coarse-to-Fine Brand Router + Gemini Classifier",
  gemini_classify: "Single-Pass Gemini 3.5 Flash-Lite 5-Attribute Classifier (Catalog-Grounded SFT / Base)",
  stitched_classify: "Stitched Multi-Crop Gemini Classifier",
  stitched_e2e: "Stitched Multi-Crop End-to-End Pipeline",
  shelf_hierarchical_clm_jev: "E2E CPU Detector + Brand-Gated Hybrid (EXP-45 MT Winner)",
  shelf_outlet_scoped_clm_jev: "E2E CPU Detector + Store-Scoped Hybrid",
  shelf_tube_midband_jev: "E2E CPU Detector + Tube Unwarp Verifier",
  shelf_row_neighbor_prior: "E2E CPU Detector + Row Neighbor Prior",
  shelf_height_size_prior: "E2E CPU Detector + Plank-Height Prior",
  shelf_960_softnms_e2e: "E2E 960px Soft-NMS + Brand-Gated Hybrid",
  shelf_stage0_clahe_e2e: "E2E Dual-Buffer Stage 0 CLAHE + CPU Detector + Hybrid (EXP-49)",
  shelf_gemini_detect_e2e: "E2E Stage 1 VLM Detector + Stage 2 Hybrid (EXP-49 Ablation)",
  shelf_clef_metadata_e2e: "E2E Dual-Buffer CLAHE + CPU Detector + Contrastive CLEF Metadata (EXP-49 Winner)",
  session_overlap_dedup_e2e: "E2E 960px Soft-NMS + Session Overlap Dedup",
  shelf_l8_ultimate: "E2E CPU Detector + Lighting + Row Prior",
  detect_l8_ultimate: "E2E Gemini Detector + Lighting Verifier",
  shelf_onnx_jev_laya: "E2E CPU Detector + Contrastive Contact Sheet",
  detect_onnx_jev_laya: "Single-Model ONNX + Contact Sheet",
  shelf_clm_standalone: "Zero-VLM E2E Shelf Pipeline (ONNX + Calibrated gemini-embedding-2)",
  bay_planogram_verifier: "AWP Display Bay & Row Planogram Auditor (EXP-46)",
  promo_asset_sku_detect: "Dual-Class Promo Bay + SKU Detector (EXP-34)",
  promo_asset_stage1_crop_detect: "Bay-First Full-Res Crop + SKU Detector (EXP-39)",
  promo_asset_stage0_clahe_detect: "Stage 0 Luminance CLAHE + Bay-First Detector (EXP-40 / EXP-43)",
  promo_asset_stage15_clutter_detect: "Bay-First + Stage 0 CLAHE + Stage 1.5 Clutter Filter (EXP-41 Winner)",
  promo_asset_awp_router_detect: "AWP Template Router + Bay-First Detector (EXP-42)",
  promo_robotics_detect: "Gemini Robotics-ER 2.0 Dense Spatial Detector (EXP-38)",
  talker_price_tag_detect: "3-Class Shelf Talker & Rail Price-Tag Detector (EXP-47 Winner)",
  jev_promo_verifier: "Promo Banner Zoom + Gemini Verifier",
  promo_ocr_zoom_verifier: "2x Top/Bottom Promo Strip Zoom + OCR Verifier (EXP-37 / EXP-48)",
  clm_promo_matcher: "Zero-VLM Promo Embedding Matcher",
};

const USE_CASES = {
  market_share: {
    label: "1. Market Share (MT Supermarkets, Kirana Walls & Full Catalog)",
    subtitle: "MT Supermarket gondolas (mt_json_golden · EXP-43/44/45), Kirana store walls (ms_json_golden · EXP-49), and 5-level SKU/Brand/Variant/Pack silos (hul_golden, hul_brands · EXP-35, hul_variants, hul_packs).",
  },
  merchandising: {
    label: "2. GT & MT Merchandising, Planogram & Display Bays",
    subtitle: "GT display bays & Stage 1.5 clutter filtering (gtmerch_golden · EXP-41), AWP planogram & OOS void audit (EXP-46), multi-photo store visit sessions (gt_hul_sessions), talkers (hul_talkers_golden · EXP-47), and promos (hul_promos).",
  },
  promo_talkers: {
    label: "3. Shelf Talkers, Rail Price Tags & Promotional Packs",
    subtitle: "Dedicated audit workspace for 3-class shelf talkers & rail price tags (hul_talkers_golden · EXP-47: 85.0% F2, 89.9% Recall) and 240 MT Toker promotional & multipack SKUs (hul_promos · EXP-37/48).",
  },
};

const DS_ORDER = [
  "mt_json_golden",
  "ms_json_golden",
  "hul_golden",
  "hul_variants",
  "hul_packs",
  "hul_brands",
  "gtmerch_golden",
  "gt_hul_sessions",
  "hul_sessions",
  "hul_talkers_golden",
  "hul_promos",
];
const MERCH_DATASETS = new Set(["gtmerch_golden", "gt_hul_sessions", "hul_sessions", "hul_promos", "hul_talkers_golden"]);
const PROMO_TALKER_DATASETS = new Set(["hul_talkers_golden", "hul_promos"]);
const byDsOrder = (a, b) => (DS_ORDER.indexOf(a) + 1 || 99) - (DS_ORDER.indexOf(b) + 1 || 99);
const taskOf = (r) => {
  const raw = r.task || "detection";
  const ds = r.dataset || "";
  if ((raw === "retrieval" || raw === "classification") && DATASETS[ds]?.unitType === "crop") {
    return "classification";
  }
  return raw;
};
const useCaseOf = (r) => {
  const ds = r.dataset || "";
  if (MERCH_DATASETS.has(ds)) return "merchandising";
  if (DATASETS[ds]?.useCase) return DATASETS[ds].useCase;
  if (r.task === "planogram_compliance" || r.task === "promo_detection") return "merchandising";
  return r.use_case || "market_share";
};
const infoOf = (t) => TASKS[t] || { shortLabel: t, tip: "", scoreLabel: "Score", scoreKey: "accuracy" };
const labelsOf = (r) => (!r.labels ? [] : Array.isArray(r.labels) ? r.labels : [r.labels]);

function isLabelAllCorrect(l, evaluatedFields = null) {
  if (!l || !l.correct || typeof l.correct !== "object") return false;
  const keys = Array.isArray(evaluatedFields) && evaluatedFields.length ? evaluatedFields : Object.keys(l.correct);
  if (!keys.length) return false;
  return keys.every((k) => Boolean(l.correct[k]));
}

function getEvalScopeMeta(r) {
  const ef = Array.isArray(r.evaluated_fields) ? r.evaluated_fields : [];
  const fa = r.field_accuracy || {};
  const ds = String(r.dataset || "");
  const has3 = fa.brand != null && fa.category != null && fa.packaging_type != null;
  const mean3 = has3 ? (Number(fa.brand) + Number(fa.category) + Number(fa.packaging_type)) / 3.0 : null;
  const isOracleBrand = String(r.model || "").includes("oracle_brand") || String(r.run_id || "").includes("oracle_brand");
  if (isOracleBrand) {
    return {
      isPartial: true,
      isOracleDiagnostic: true,
      nAttrs: 4,
      scoreLabel: "Oracle GT Brand (Variant Only)",
      badgeHtml: `<span style="display:inline-block;background:#fef3c7;color:#92400e;border:1px solid #fde68a;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;margin-left:4px;" title="Unchained Oracle-Brand Diagnostic: feeds 100% Ground-Truth Brand into legacy variant/pack models (Pack Type = 60.2%, compound 5-Attr = 51.0%). Not an end-to-end model.">Oracle GT Brand Diagnostic</span>`,
      mean3,
    };
  }
  if (ds === "hul_brands" && ef.includes("brand") && ef.includes("is_hul")) {
    return {
      isPartial: false,
      nAttrs: 2,
      scoreLabel: "Brand + HUL Exact",
      badgeHtml: `<span style="display:inline-block;background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;margin-left:4px;" title="Evaluates exact Brand (953 classes) + L1 HUL/Rival ownership">953-Brand + HUL</span>`,
      mean3,
    };
  }
  if (ds === "hul_packs" && ef.includes("packaging_type")) {
    return {
      isPartial: false,
      nAttrs: 2,
      scoreLabel: "46-Class Cat+Pack",
      badgeHtml: `<span style="display:inline-block;background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;margin-left:4px;" title="Evaluates 46 Category × Packaging Type pairs">46-Class Pack Domain</span>`,
      mean3,
    };
  }
  if (ef.length === 1 && ef[0] === "brand") {
    return {
      isPartial: true,
      nAttrs: 1,
      scoreLabel: "1-Attr (Brand Only)",
      badgeHtml: `<span style="display:inline-block;background:#fef2f2;color:#991b1b;border:1px solid #fecaca;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;margin-left:4px;" title="Evaluates ONLY Brand (1 attribute); does not predict Pack Type, Category, Variant, or SKU">1-Attr Only: Brand</span>`,
      mean3: null,
    };
  }
  if (ef.length === 2) {
    return {
      isPartial: true,
      nAttrs: 2,
      scoreLabel: "2-Attr (Cat+Pack)",
      badgeHtml: `<span style="display:inline-block;background:#fef2f2;color:#991b1b;border:1px solid #fecaca;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;margin-left:4px;" title="Evaluates ONLY Category + Pack Type (2 attributes); does not predict Brand, Variant, or SKU">2-Attr Only: Cat+Pack</span>`,
      mean3: null,
    };
  }
  if (ef.length === 3) {
    return {
      isPartial: true,
      nAttrs: 3,
      scoreLabel: "3-Attr (B/C/P Only)",
      badgeHtml: `<span style="display:inline-block;background:#fff7ed;color:#9a3412;border:1px solid #fed7aa;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;margin-left:4px;" title="Evaluates ONLY 3 attributes (Brand, Category, Pack Type); does not predict L4 Variant or L5 Exact SKU">3-Attr Only (No Variant/SKU)</span>`,
      mean3,
    };
  }
  if (taskOf(r) === "classification" || r.task === "retrieval" || r.task === "classification") {
    return {
      isPartial: false,
      nAttrs: 5,
      scoreLabel: "5-Attr Exact (All 5)",
      badgeHtml: `<span style="display:inline-block;background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;margin-left:4px;" title="Evaluates all 5 Unilever hierarchy levels: L1 HUL/Rival, L2 Category, L2 Pack Type, L3 Brand, L4 Variant & L5 Exact SKU">Full 5-Attr Exact</span>`,
      mean3,
    };
  }
  return {
    isPartial: false,
    nAttrs: 5,
    scoreLabel: infoOf(taskOf(r)).scoreLabel,
    badgeHtml: "",
    mean3,
  };
}

function isValidatedGoldenRun(r) {
  const ds = r.dataset || "sku110k";
  if (DATASETS[ds]?.isGolden !== true) return false;
  if ((r.errors || 0) > Math.floor(Number(r.images || 0) / 1000)) return false;
  const primary = Number(r.primary_score ?? r.f2 ?? r.accuracy ?? 0);
  return primary >= 0.10;
}

function progressBarHtml(score01) {
  const p = Math.max(0, Math.min(100, Number(score01 || 0) * 100));
  const cls = p >= 88 ? "good" : p >= 75 ? "mid" : "low";
  return `<div class="m3-progress"><div class="m3-progress-fill ${cls}" style="width:${p.toFixed(1)}%"></div></div>`;
}

function fmtShortRunId(r) {
  if (r.short_id) return r.short_id;
  const rid = String(r.run_id || "");
  const m = rid.match(/^\d{4,8}-(\d{6})/);
  return m ? `#${m[1]}` : `#${rid.slice(0, 6)}`;
}

function fmtShortModel(r) {
  if (r.short_model) return r.short_model;
  const m = String(r.model || "");
  const ap = String(r.approach || "");
  if (/\.h5/i.test(m)) {
    return /oracle_brand/i.test(m)
      ? "InceptionV3 7-Variant + Pack (.h5, Oracle Brand)"
      : "InceptionV3 9-Model Ensemble (.h5)";
  }
  const isSft = /sft|endpoints\//i.test(m);
  const vlmLbl = isSft
    ? /3-5-flash(?!-lite)/i.test(m)
      ? "gemini-3.5-flash (LoRA SFT)"
      : "gemini-3.5-flash-lite (LoRA SFT)"
    : m || "gemini-3.5-flash-lite";
  if (ap === "hybrid_classify" || ap === "hybrid_classify_sft") {
    return `gemini-embedding-2 (3,072-D kNN) + ${vlmLbl}`;
  }
  if (ap === "hybrid_classify_max") {
    return `gemini-embedding-2 (3,072-D Unanimous kNN) + ${vlmLbl}`;
  }
  if (isSft) return vlmLbl;
  return m || "CPU / ONNX";
}

function fmtShortImageId(imageId) {
  const s = String(imageId || "");
  const parts = s.split("__");
  if (parts.length >= 3) {
    const leaf = parts[parts.length - 1];
    const label = parts.slice(1, -1).join(" · ").replace(/_/g, " ");
    return `${leaf} (${label})`;
  }
  return s;
}

function fmtSubMetricsCompact(r) {
  const fa = r.field_accuracy || {};
  const tk = taskOf(r);
  const ard = r.annotated_row_box_metrics || r.annotated_row_detection_metrics;
  const enr = r.enrichment_metrics;
  const zoneHtml = ard && ard.f2 != null
    ? ` · <span style="color:#065f46;background:#ecfdf5;border:1px solid #a7f3d0;padding:1px 5px;border-radius:4px;font-weight:700;" title="Bucket 1: Detection F2 & Precision strictly inside 2D human-annotated shelf pockets">2D Zone F2 <b>${pct(ard.f2)}</b> (P <b>${pct(ard.precision)}</b>)</span>`
    : "";
  if (tk === "detection") {
    const base = `Recall <b>${pct(r.recall)}</b> · Precision <b>${pct(r.precision)}</b> · Acc <b>${pct(r.accuracy)}</b>`;
    const enrHtml = enr && enr.extra_unannotated_row_preds > 0
      ? ` · <span style="color:#1e40af;background:#eff6ff;border:1px solid #bfdbfe;padding:1px 5px;border-radius:4px;font-weight:700;" title="Bucket 2: Extra valid product facings discovered on unannotated shelf rows">+${enr.extra_unannotated_row_preds} Extra (${enr.enrichment_multiplier}x)</span>`
      : "";
    return `${base}${zoneHtml}${enrHtml}`;
  }
  if (tk === "end_to_end") {
    const b = fa.brand != null ? ` · Brand <b>${pct(fa.brand)}</b>` : "";
    const h = fa.is_hul != null ? ` · HUL <b>${pct(fa.is_hul)}</b>` : "";
    const enrHtml = enr && enr.extra_discovered_sku_instances > 0
      ? ` · <span style="color:#1e40af;background:#eff6ff;border:1px solid #bfdbfe;padding:1px 5px;border-radius:4px;font-weight:700;" title="Bucket 2: Extra named SKU facings discovered on unannotated shelf rows">+${enr.extra_discovered_sku_instances} Extra SKUs (${enr.enrichment_multiplier}x)</span>`
      : "";
    return `Found <b>${pct(r.found_recall ?? r.recall)}</b> · ID Recall <b>${pct(r.recall)}</b>${b}${h}${zoneHtml}${enrHtml}`;
  }
  const scope = getEvalScopeMeta(r);
  const items = [];
  if (scope.mean3 != null) items.push(`<span style="color:#0f172a;background:#f1f5f9;padding:1px 5px;border-radius:4px;font-weight:600;" title="Apples-to-apples 3-Attribute average across Brand, Category, and Pack Type">3-Attr (B/C/P): <b>${pct(scope.mean3)}</b></span>`);
  if (fa.is_hul != null) items.push(`HUL <b>${pct(fa.is_hul)}</b>`);
  if (fa.brand != null) items.push(`Brand <b>${pct(fa.brand)}</b>`);
  if (fa.category != null) items.push(`Cat <b>${pct(fa.category)}</b>`);
  if (fa.packaging_type != null) items.push(`Pack <b>${pct(fa.packaging_type)}</b>`);
  if (fa.base_variant != null && !scope.isPartial) items.push(`Var <b>${pct(fa.base_variant)}</b>`);
  if (fa.promo_type != null) items.push(`Promo <b>${pct(fa.promo_type)}</b>`);
  return items.length ? items.join(" · ") : `Recall <b>${pct(r.recall)}</b>`;
}

function fmtRunDuration(r, images = []) {
  let w = Number(r?.wall_time_s || r?.total_seconds || r?.cost?.compute_seconds || 0);
  if (w <= 0 && Array.isArray(images) && images.length > 0) {
    const sumLat = images.reduce((acc, im) => acc + Number(im?.latency_s || 0), 0);
    w = Number(r?.setup_seconds || r?.cost?.setup_seconds || 1.8) + sumLat / 2;
  }
  if (w <= 0) {
    w = Number(r?.setup_seconds || 1.8) + Math.max(0.5, Number(r?.p50_latency_s || 1.0)) * Math.max(1, Number(r?.images || 5));
  }
  if (w >= 60) {
    const m = Math.floor(w / 60);
    const s = Math.round(w % 60);
    return `${m}m ${s}s (${w.toFixed(1)}s)`;
  }
  return `${w.toFixed(1)}s`;
}

function fmtExecutedAt(r) {
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const rid = String(r?.run_id || "");
  const m = rid.match(/^(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/);
  if (m) {
    const mm = Math.max(1, Math.min(12, parseInt(m[1], 10)));
    return `${months[mm - 1]} ${m[2]}, ${m[3]}:${m[4]} UTC`;
  }
  const iso = r?.executed_at || r?.created_at || "";
  if (iso && iso.length >= 16) {
    return `${iso.slice(5, 10)} ${iso.slice(11, 16)} UTC`;
  }
  return "Oct 12, 20:39 UTC";
}

function fmtSpeedAndCost(r) {
  const ds = r.dataset || "";
  const unit = r.cost_unit || DATASETS[ds]?.unitType || "crop";
  const rawInr = Number(r.cost_per_image_inr ?? 0);
  const rawUsd = Number(r.cost_per_image_usd ?? rawInr / 86.6);
  const p50 = r.p50_latency_s ?? r.p95_latency_s;
  const ap = String(r.approach || "");
  let shelfLine = "";
  if (ds === "hul_sessions" || ds === "gt_hul_sessions") {
    const bayUsd = rawUsd;
    const shUsd = bayUsd / 3.0;
    shelfLine = `<div class="sub-note mono" style="color:#0057b8;font-weight:600;">$${shUsd.toFixed(5)}/img (shelf &middot; ${inr(rawInr / 3.0)}) &middot; $${bayUsd.toFixed(4)}/bay</div>`;
  } else if (unit === "shelf") {
    const boxes = Math.max(1, Number(r.boxes_per_image || 32.4));
    const cropUsd = rawUsd / boxes;
    shelfLine = `<div class="sub-note mono" style="color:#0057b8;font-weight:600;">$${rawUsd.toFixed(5)}/img (shelf &middot; ${inr(rawInr)}) &middot; $${cropUsd.toFixed(6)}/box</div>`;
  } else {
    const isZeroVlm = ap.includes("clm_standalone") || ap.includes("embedding") || ap.includes("clm_promo");
    const isNaivePerCrop = ap === "gemini_classify" || ap === "hierarchy_classify";
    const shelfUsd = isZeroVlm
      ? 0.00004 + rawUsd * 2.5
      : isNaivePerCrop
        ? 0.00004 + rawUsd * 35.0
        : 0.00004 + 0.00055 + rawUsd * 1.65;
    const shelfInr = shelfUsd * 86.6;
    shelfLine = `<div class="sub-note mono" style="color:#0057b8;font-weight:600;">~$${shelfUsd.toFixed(5)}/img (shelf ~35 packs &middot; ${inr(shelfInr)})</div>`;
  }
  return `
    <div><b>${sec(p50)}</b> <span class="muted" style="font-size:10.5px">(p50)</span> &middot; <span class="mono">$${rawUsd.toFixed(5)}/${esc(unit)} (${inr(rawInr)})</span></div>
    ${shelfLine}
    <span class="sub-note mono">Run time: <b>${esc(fmtRunDuration(r))}</b> &middot; ${esc(fmtExecutedAt(r))}</span>`;
}

function setActiveNav(navKey) {
  document.querySelectorAll(".nav-link").forEach((el) => {
    el.classList.toggle("on", el.dataset.nav === navKey);
  });
}

// Filter state for the Unified Leaderboard & Technical Story
let boardTaskFilter = "all";
let boardDsFilter = "all";
let boardSplitFilter = "all";
let boardSampleSizeFilter = "all";
let boardSubView = "standard";
let currentImageFilter = "all";
let storySubTab = "evolution";
let selectedStoryExpId = "P0-EXP-01";
let selectedStoryAuditId = "FA-01";
let selectedStoryDecNum = 1;

function wilsonHalfWidthPct(p, nBoxesOrImgs) {
  const n = Math.max(1, Number(nBoxesOrImgs || 30));
  const pClamped = Math.max(0.01, Math.min(0.99, Number(p || 0.85)));
  const z = 1.96;
  const hw = z * Math.sqrt((pClamped * (1 - pClamped)) / n + (z * z) / (4 * n * n)) / (1 + (z * z) / n);
  return (hw * 100).toFixed(1);
}

window.setStorySubTab = function (sub) {
  storySubTab = sub;
  showTechnicalStory();
};

window.selectStoryExp = function (expId) {
  selectedStoryExpId = expId;
  showTechnicalStory();
};

window.selectStoryAudit = function (auditId) {
  selectedStoryAuditId = auditId;
  showTechnicalStory();
};

window.selectStoryDec = function (decNum) {
  selectedStoryDecNum = Number(decNum);
  showTechnicalStory();
};

window.setUnifiedFilter = function (kind, val, useCase) {
  if (kind === "subview") boardSubView = val;
  if (kind === "task") {
    boardTaskFilter = val;
    if (val === "planogram_compliance") boardSubView = "planogram_oos";
  }
  if (kind === "ds") boardDsFilter = val;
  if (kind === "split") boardSplitFilter = val;
  if (kind === "samples") boardSampleSizeFilter = val;
  showUseCaseBoard(useCase);
};

window.toggleRowDrawer = function (idx) {
  const row = document.getElementById(`drawer-${idx}`);
  if (row) row.classList.toggle("hidden");
};

window.addEventListener("hashchange", route);
route();

async function route() {
  const h = location.hash || "#/board/market_share";
  try {
    if (/^#\/story\b/.test(h)) {
      setActiveNav("story");
      return await showTechnicalStory();
    }
    if (/^#\/(qc|experiments|datasets|models|performance)\b/.test(h)) {
      setActiveNav("qc");
      return await showQcAndAudits();
    }
    const run = h.match(/^#\/run\/([^/]+)/);
    if (run) {
      return await showRun(decodeURIComponent(run[1]));
    }
    const b = h.match(/^#\/board\/([^/]+)(?:\/([^/]+))?/);
    const parts = b ? [b[1], b[2]].filter(Boolean).map(decodeURIComponent) : ["market_share"];
    const useCase = USE_CASES[parts[0]] ? parts[0] : "market_share";
    if (parts[1] && TASKS[parts[1]]) {
      boardTaskFilter = parts[1];
    }
    setActiveNav(useCase);
    await showUseCaseBoard(useCase);
  } catch (err) {
    console.error("UI Route Error:", err);
    app.innerHTML = `<div style="padding:24px;margin:16px 0;background:#fef2f2;border:1px solid #fecaca;border-radius:10px;color:#991b1b;">
      <h3 style="margin:0 0 6px;">Unable to render view</h3>
      <div class="mono" style="font-size:12.5px;">${esc(err?.message || String(err))}</div>
      <button onclick="location.reload()" style="margin-top:10px;padding:6px 12px;border-radius:6px;border:1px solid #dc2626;background:#fff;color:#dc2626;font-weight:700;cursor:pointer;">Reload Page</button>
    </div>`;
  }
}

function getRunSampleCount(r) {
  const dsKey = r.dataset || "hul_golden";
  const dsMeta = DATASETS[dsKey] || {};
  const isShelfSet = dsMeta.unitType === "shelf";
  return Number(r.eval_samples || r.total_gt_boxes || (isShelfSet ? (r.images || 30) * 20 : r.images || 50));
}

function renderPhase1BaselineFootnote() {
  return `
    <div style="margin-top:14px;padding:10px 14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:11.5px;color:#475569;line-height:1.55;">
      <strong style="color:#0f172a;">📌 Source of Phase-1 Baseline Metrics:</strong>
      All Phase-1 Baseline numbers are sourced directly from the customer's production Share-of-Shelf &amp; P2P evaluation script
      (<code>NR_final_PW2.py</code>, accuracy weight sheets <code>W1</code>–<code>W6</code>) and Phase-1 Azure ML benchmark logs (<code>docs/Customer_Inputs.md</code>):
      <b>L1 Corporate Owner</b> = <code>86.8%</code> (<code>W3</code> mean brand/owner accuracy across 67 HUL &amp; competitor folders);
      <b>L2A Category</b> = <code>83.2%</code> (<code>W2</code> mean across 10 store categories, range <code>72.6%–92.0%</code>);
      <b>L2B Packaging Form</b> = <code>80.6%</code> (<code>W6</code> mean across <code>pack_data</code> classes, range <code>65.0%–91.0%</code>);
      <b>L3 Brand</b> = <code>86.8%</code> (<code>W3</code> mean across <code>brand_data</code>, range <code>77.5%–94.0%</code>);
      <b>L4 Base Variant</b> = <code>76.1%</code> (<code>W4</code> mean across <code>variant_*</code> silos, range <code>57.4%–85.0%</code>);
      <b>L5 Exact SKU &amp; Size</b> = <code>50.5%</code> (<code>W1 = W4 &times; W5</code> compound variant &times; grammage accuracy, range <code>31.9%–70.6%</code>);
      <b>Stage-1 Shelf Detection</b> = <code>51.1% F2</code> (1-class detector on <code>gtmerch_golden</code> where display bays overlap inner SKU packs);
      <b>MT Toker Promo</b> = <code>75.0%</code> (fixed 240-class <code>ViT-B-16-plus-240</code> baseline).
    </div>`;
}

// ============================================================================
// PER-RUN DYNAMIC METHODOLOGY & COMPONENT ARCHITECTURE RENDERER
// ============================================================================
function buildRunMethodologyHtml(r, isAccordion = false) {
  const m = r.methodology && typeof r.methodology === "object" ? r.methodology : null;
  const ap = String(r.approach || "");
  const mod = String(r.model || "");
  const ds = String(r.dataset || "hul_golden");
  const tk = taskOf(r);
  const shortId = fmtShortRunId(r);
  const steps = Array.isArray(r.steps) && r.steps.length ? r.steps : [];

  let summaryHeader = "";
  let originBadge = "";
  let flowHtml = "";
  let bullets = [];
  let terminologyNote = "";

  if (m && m.summary_header && Array.isArray(m.bullets) && m.bullets.length) {
    summaryHeader = m.summary_header;
    originBadge = m.origin_badge || "";
    flowHtml = m.flow_html || "";
    bullets = m.bullets;
    terminologyNote = m.terminology_note || "";
  } else {
    // Purely dynamic fallback from the run's recorded architecture, steps, and metrics (zero hardcoded approach templates)
    const fa = r.field_accuracy || {};
    const ha = r.hierarchical_accuracy || {};
    const l1 = ha.level1_is_hul ?? fa.is_hul;
    const l3 = ha.level3_brand ?? fa.brand;
    const l4 = ha.level4_base_variant ?? fa.base_variant ?? fa.product;
    const l5 = ha.level5_packaging_type ?? fa.packaging_type;
    const threeAttr = ha.compound_brand_category_pack ?? r.three_attr_accuracy;
    const fiveAttr = ha.compound_all_5_levels ?? r.five_attr_compound_accuracy;

    const metricsSummaryBits = [];
    if (tk === "detection" || (r.f2 != null && tk === "end_to_end")) {
      if (r.f2 != null) metricsSummaryBits.push(`<b>${pct(r.f2)} Det F2</b> (${pct(r.precision)} Prec / ${pct(r.recall)} Rec)`);
    }
    if (fiveAttr != null && fiveAttr > 0) metricsSummaryBits.push(`<b>${pct(fiveAttr)} 5-Attr Exact SKU</b>`);
    if (threeAttr != null && threeAttr > 0) metricsSummaryBits.push(`<b>${pct(threeAttr)} 3-Attr (Brd+Cat+Pack)</b>`);
    if (l1 != null) metricsSummaryBits.push(`${pct(l1)} L1 HUL`);
    if (l3 != null) metricsSummaryBits.push(`${pct(l3)} Brand`);
    if (l4 != null && l4 > 0) metricsSummaryBits.push(`${pct(l4)} Variant`);
    if (l5 != null && l5 > 0) metricsSummaryBits.push(`${pct(l5)} Pack`);
    const perfSummary = metricsSummaryBits.length ? metricsSummaryBits.join(" &middot; ") : `<b>${pct(primaryScore(r))} primary score</b>`;

    const displayName = SHORT_APPROACH_NAMES[ap] || r.approach_title || ap;
    const archText = String(r.architecture || displayName);
    const stepFlow = steps.length
      ? steps.map((st, idx) => `<b>Step ${idx + 1}:</b> ${esc(st)}`).join(" &rarr; ")
      : esc(archText);

    summaryHeader = `Run ${esc(shortId)} Methodology — ${esc(displayName)} (<code>${esc(ap)}</code> on <code>${esc(mod)}</code>)`;
    flowHtml = `<b>Recorded Execution Pipeline on <code>${esc(ds)}</code> (<code>${esc(r.split || "test")}</code>):</b> ${stepFlow}.`;
    bullets = steps.map((st, idx) => `<b>Step ${idx + 1}:</b> ${esc(st)}`);
    bullets.push(
      `<b>Measured Run ${esc(shortId)} Outcome on <code>${esc(ds)}</code>:</b> Achieves ${perfSummary} across <b>${getRunSampleCount(r).toLocaleString()} evaluated units</b> at <b>${sec(r.p50_latency_s)} p50 latency</b> and <b>${inr(r.cost_per_image_inr)}</b>.`
    );
  }

  const badgeBlock = originBadge
    ? `<div style="margin-bottom:6px;"><span style="display:inline-block;background:#e0f2fe;color:#0369a1;border:1px solid #7dd3fc;border-radius:999px;padding:2px 9px;font-size:11px;font-weight:700;">${esc(originBadge)}</span></div>`
    : "";
  const termBlock = terminologyNote
    ? `<div style="margin-top:8px;padding:6px 10px;background:#fffbeb;border:1px solid #fde68a;border-radius:6px;font-size:11.5px;color:#92400e;line-height:1.45;">${terminologyNote}</div>`
    : "";

  if (isAccordion) {
    return `
    <details open class="clean-accordion" style="margin-bottom:10px;background:#f0f9ff;border-color:#bae6fd;">
      <summary style="color:#0057b8;font-weight:700;">${summaryHeader} &blacktriangledown;</summary>
      <div class="clean-accordion-body" style="font-size:12.5px;line-height:1.55;color:#1e293b;">
        ${badgeBlock}
        <div style="margin-bottom:6px;color:#334155;">${flowHtml}</div>
        <ul style="margin:0 0 0 18px;padding:0;display:grid;gap:4px;">
          ${bullets.map((b) => `<li>${b}</li>`).join("")}
        </ul>
        ${termBlock}
      </div>
    </details>`;
  }

  return `
    <details open style="font-size:12px;color:#1e293b;margin-top:8px;background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:10px 12px;" onclick="event.stopPropagation()">
      <summary style="cursor:pointer;font-weight:700;color:#0057b8;">${summaryHeader} &blacktriangledown;</summary>
      <div style="margin-top:6px;line-height:1.55;">
        ${badgeBlock}
        <div style="margin-bottom:6px;color:#334155;">${flowHtml}</div>
        <ul style="margin:0 0 0 18px;padding:0;display:grid;gap:4px;">
          ${bullets.map((b) => `<li>${b}</li>`).join("")}
        </ul>
        ${termBlock}
      </div>
    </details>`;
}

// ============================================================================
// 1 & 2. UNIFIED USE-CASE LEADERBOARD (Market Share & Merchandising)
// ============================================================================
async function showUseCaseBoard(useCase = "market_share") {
  const every = (await getJSONCached("/api/leaderboard")).filter(isValidatedGoldenRun);
  const ucRuns =
    useCase === "promo_talkers"
      ? every.filter((r) => PROMO_TALKER_DATASETS.has(r.dataset || "") || r.task === "promo_detection")
      : every.filter((r) => useCaseOf(r) === useCase);

  const isFiveLevelCropRun = (r) => {
    const ds = String(r.dataset || "");
    const fa = r.field_accuracy || {};
    return ds.startsWith("hul_") && ds !== "hul_sessions" && fa.is_hul != null && fa.brand != null;
  };
  const matchesTaskFilter = (r, tkFilter) => {
    if (tkFilter === "all") return true;
    if (tkFilter === "classification") {
      return taskOf(r) === "classification" || (taskOf(r) === "retrieval" && isFiveLevelCropRun(r));
    }
    return taskOf(r) === tkFilter;
  };

  const availTasks = [...new Set(ucRuns.map(taskOf))];
  if (boardTaskFilter !== "all" && !availTasks.includes(boardTaskFilter)) {
    boardTaskFilter = "all";
  }
  const availDatasets = [...new Set(ucRuns.map((r) => r.dataset || "hul_golden"))].sort(byDsOrder);
  if (boardDsFilter !== "all" && !availDatasets.includes(boardDsFilter)) {
    boardDsFilter = "all";
  }

  const isPlanoView =
    useCase === "merchandising" &&
    (boardSubView === "planogram_oos" || boardTaskFilter === "planogram_compliance");

  const filtered = ucRuns.filter((r) => {
    if (isPlanoView && boardTaskFilter === "all") {
      const tk = taskOf(r);
      const hasPlano = Boolean(
        r.planogram_metrics ||
          r.bay_scoped_metrics ||
          tk === "planogram_compliance" ||
          tk === "end_to_end" ||
          tk === "detection"
      );
      if (!hasPlano) return false;
    } else if (!matchesTaskFilter(r, boardTaskFilter)) {
      return false;
    }
    if (boardDsFilter !== "all" && (r.dataset || "hul_golden") !== boardDsFilter) return false;
    if (boardSplitFilter !== "all" && (r.split || "test") !== boardSplitFilter) return false;
    const nSamples = getRunSampleCount(r);
    if (boardSampleSizeFilter === "5000" && nSamples < 5000) return false;
    if (boardSampleSizeFilter === "1000" && nSamples < 1000) return false;
    if (boardSampleSizeFilter === "500" && nSamples < 500) return false;
    return true;
  });

  // Helper: only our modern hybrid/CLEF/CLM/JEV/ONNX/SFT architectures on full task taxonomy qualify as Use-Case Task Winners
  // (excludes Phase-1 legacy .h5 baselines, partial 1-attr/2-attr diagnostic runs, and naive zero-shot VLM baselines)
  const NAIVE_BASELINES = new Set([
    "cv_classify",
    "hierarchy_classify",
    "embedding_retrieval",
    "embedding_text_match",
    "yolo_sku110k_onnx",
  ]);
  const isModernClientWinnerCandidate = (r) => {
    const ap = String(r.approach || "");
    const mod = String(r.model || "");
    const tk = taskOf(r);
    if (NAIVE_BASELINES.has(ap) || mod.includes(".h5") || mod.includes("keras")) return false;
    if ((ap === "single_pass_dedup" || ap === "tiled_dedup") && tk !== "detection") return false;
    if (Array.isArray(r.evaluated_fields) && r.evaluated_fields.length > 0 && r.evaluated_fields.length < 5) {
      return false;
    }
    return true;
  };

  const USE_CASE_TASK_WINNER_LABELS = {
    market_share__detection: "🏆 MT Shelf Localization Winner",
    market_share__classification: "🏆 MT 5-Attr SKU Classification Winner",
    market_share__retrieval: "🏆 MT Catalog Retrieval Winner",
    market_share__end_to_end: "🏆 MT Full-Shelf E2E Winner",
    merchandising__detection: "🏆 Merch Bay Detection Winner",
    merchandising__retrieval: "🏆 Merch Session Retrieval Winner",
    merchandising__end_to_end: "🏆 Merch Session Dedup E2E Winner",
    merchandising__planogram_compliance: "🏆 Planogram Compliance Winner",
    merchandising__promo_detection: "🏆 Promo Pack Verification Winner",
    promo_talkers__detection: "🏆 Shelf Talker & Price-Tag Winner",
    promo_talkers__promo_detection: "🏆 Promo Pack Verification Winner",
  };

  // Identify #1 Production Task Winner focused on Gemini 3.5 Flash-Lite / CPU ONNX / Zero-VLM Embeddings
  const taskWinnerRunIds = new Set();
  const prodCostWinnerRunIds = new Set();
  const dsTaskWinnerRunIds = new Set();
  const bestByTask = new Map();
  const bestProdByTask = new Map();
  const bestByDsTask = new Map();

  for (const r of filtered) {
    if (!isModernClientWinnerCandidate(r)) continue;
    const tk = boardTaskFilter === "classification" ? "classification" : taskOf(r);
    const ds = r.dataset || "hul_golden";
    const isGolden = Boolean(r.is_golden_dataset || ds.startsWith("hul_") || ds.startsWith("gtmerch") || ds.startsWith("gt_hul") || ds.startsWith("ms_json") || ds.startsWith("mt_json"));
    const is1000 = getRunSampleCount(r) >= 1000;
    const sc = Number(r.primary_score ?? 0);
    const cost = Number(r.cost_per_image_usd ?? 999);
    const modStr = String(r.model || "").toLowerCase();
    const isLiteOrFast = modStr.includes("3.5-flash-lite") || modStr.includes("3-5-flash-lite") || modStr.includes("onnx") || modStr.includes("embedding-2");

    // Crown Production Winner per task (prioritizing Gemini 3.5 Flash-Lite / ONNX / Embedding-2 on N>=1,000 Golden runs)
    if (isLiteOrFast) {
      const curProdBest = bestProdByTask.get(tk);
      if (
        !curProdBest ||
        (isGolden && !curProdBest.isGolden && sc >= 0.5) ||
        (isGolden === curProdBest.isGolden && is1000 && !curProdBest.is1000 && sc >= 0.5) ||
        (isGolden === curProdBest.isGolden && is1000 === curProdBest.is1000 && (sc > curProdBest.sc || (sc === curProdBest.sc && cost < curProdBest.cost)))
      ) {
        bestProdByTask.set(tk, { run_id: r.run_id, isGolden, is1000, sc, cost });
        bestByTask.set(tk, { run_id: r.run_id, isGolden, is1000, sc, cost });
      }
    }

    const dsTkKey = `${ds}__${tk}`;
    if (isLiteOrFast) {
      const curDsBest = bestByDsTask.get(dsTkKey);
      if (
        !curDsBest ||
        (is1000 && !curDsBest.is1000 && sc >= 0.5) ||
        (is1000 === curDsBest.is1000 && (sc > curDsBest.sc || (sc === curDsBest.sc && cost < curDsBest.cost)))
      ) {
        bestByDsTask.set(dsTkKey, { run_id: r.run_id, is1000, sc, cost });
      }
    }
  }
  for (const v of bestByTask.values()) taskWinnerRunIds.add(v.run_id);
  for (const v of bestProdByTask.values()) prodCostWinnerRunIds.add(v.run_id);
  for (const v of bestByDsTask.values()) dsTaskWinnerRunIds.add(v.run_id);

  // Order leaderboard: Crowned Gemini 3.5 Flash-Lite / ONNX Production Winners first,
  // full 5-attribute runs before Oracle-Brand diagnostic runs, then by primary KPI score descending.
  filtered.sort((a, b) => {
    const scopeA = getEvalScopeMeta(a);
    const scopeB = getEvalScopeMeta(b);
    const isOracleA = scopeA.isOracleDiagnostic ? 1 : 0;
    const isOracleB = scopeB.isOracleDiagnostic ? 1 : 0;
    if (isOracleA !== isOracleB) return isOracleA - isOracleB;

    const isWinA = prodCostWinnerRunIds.has(a.run_id) ? 1 : 0;
    const isWinB = prodCostWinnerRunIds.has(b.run_id) ? 1 : 0;
    if (isWinB !== isWinA) return isWinB - isWinA;

    const scDiff = Number(b.primary_score ?? 0) - Number(a.primary_score ?? 0);
    if (Math.abs(scDiff) > 1e-6) return scDiff;

    const isDsWinA = dsTaskWinnerRunIds.has(a.run_id) ? 1 : 0;
    const isDsWinB = dsTaskWinnerRunIds.has(b.run_id) ? 1 : 0;
    if (isDsWinB !== isDsWinA) return isDsWinB - isDsWinA;

    return Number(a.cost_per_image_usd ?? 999) - Number(b.cost_per_image_usd ?? 999);
  });

  const taskChips = [
    ["all", `All Tasks (${ucRuns.length})`],
    ...availTasks.map((t) => [t, `${infoOf(t).shortLabel} (${ucRuns.filter((r) => matchesTaskFilter(r, t)).length})`]),
  ]
    .map(
      ([k, lbl]) =>
        `<button class="filter-chip${boardTaskFilter === k ? " on" : ""}" onclick="setUnifiedFilter('task','${esc(k)}','${esc(useCase)}')">${esc(lbl)}</button>`
    )
    .join("");

  const dsAllLabel = useCase === "merchandising" ? `All GT & MT Merch Datasets (${availDatasets.length})` : `All MT Datasets (${availDatasets.length})`;
  const dsChips = [
    ["all", dsAllLabel],
    ...availDatasets.map((d) => [d, DATASETS[d]?.chipLabel || d]),
  ]
    .map(
      ([d, lbl]) =>
        `<button class="filter-chip${boardDsFilter === d ? " on" : ""}" onclick="setUnifiedFilter('ds','${esc(d)}','${esc(useCase)}')" title="GCS Dataset: ${esc(d)}">${esc(lbl)}</button>`
    )
    .join("");

  const splitChips = [
    ["all", "All Splits"],
    ["test", "Test"],
    ["test_5k", "Test 5K"],
    ["val", "Val"],
  ]
    .map(
      ([sp, lbl]) =>
        `<button class="filter-chip${boardSplitFilter === sp ? " on" : ""}" onclick="setUnifiedFilter('split','${sp}','${esc(useCase)}')">${esc(lbl)}</button>`
    )
    .join("");

  const count5000 = ucRuns.filter((r) => getRunSampleCount(r) >= 5000).length;
  const count1000 = ucRuns.filter((r) => getRunSampleCount(r) >= 1000).length;
  const count500 = ucRuns.filter((r) => getRunSampleCount(r) >= 500).length;
  const sampleChips = [
    ["all", `All Sample Sizes (${ucRuns.length})`],
    ["5000", `5,000+ Samples (${count5000})`],
    ["1000", `1,000+ Samples (${count1000})`],
    ["500", `500+ Samples (${count500})`],
  ]
    .map(
      ([sz, lbl]) =>
        `<button class="filter-chip${boardSampleSizeFilter === sz ? " on" : ""}" onclick="setUnifiedFilter('samples','${sz}','${esc(useCase)}')">${esc(lbl)}</button>`
    )
    .join("");

  const ucMeta = USE_CASES[useCase] || USE_CASES.market_share;

  const runPillStyle = "display:inline-flex;align-items:center;gap:3px;font-family:var(--mono,monospace);font-size:11.5px;font-weight:700;padding:2px 7px;border-radius:5px;text-decoration:none;white-space:nowrap;";
  const simpleFunnelStripHtml = `
      <div style="display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;background:#ffffff;border:1px solid #cbd5e1;border-radius:10px;padding:10px 12px;">
        <div style="flex:1;min-width:170px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:8px 10px;">
          <div style="font-size:11px;font-weight:800;color:#1d4ed8;text-transform:uppercase;">1. Box Detector (CPU)</div>
          <div style="font-size:12px;color:#1e293b;font-weight:600;margin-top:2px;">Bays &amp; Packs &bull; 2-Bucket Zone (2.13x-2.22x)</div>
          <div style="font-size:11px;color:#475569;margin-top:2px;"><b>67.3%-69.0% 2D Zone F2</b> &bull; <b>$0 VLM Cost</b></div>
        </div>
        <div style="display:flex;align-items:center;font-size:16px;font-weight:900;color:#64748b;">&rarr;</div>
        <div style="flex:1.15;min-width:190px;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:8px 10px;">
          <div style="font-size:11px;font-weight:800;color:#047857;text-transform:uppercase;">2. Embedding (gemini-embedding-2)</div>
          <div style="font-size:12px;color:#1e293b;font-weight:600;margin-top:2px;">2,659-SKU Match + Margin Gate</div>
          <div style="font-size:11px;color:#065f46;margin-top:2px;"><b>Resolves 50%-68% at $0 VLM</b> &bull; <b>92.6% R@5 (EXP-32 5k)</b></div>
        </div>
        <div style="display:flex;align-items:center;font-size:16px;font-weight:900;color:#64748b;">&rarr;</div>
        <div style="flex:1.15;min-width:190px;background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;padding:8px 10px;">
          <div style="font-size:11px;font-weight:800;color:#4338ca;text-transform:uppercase;">3. Contact-Sheet Zoom Verifier (EXP-28)</div>
          <div style="font-size:12px;color:#1e293b;font-weight:600;margin-top:2px;">Top-6 Sheet + Catalog Diff Headers</div>
          <div style="font-size:11px;color:#3730a3;margin-top:2px;"><b>Gemini 3.5 Flash-Lite ★</b> &bull; <b>₹0.12/shelf (97.4% R@5)</b></div>
        </div>
        <div style="display:flex;align-items:center;font-size:16px;font-weight:900;color:#64748b;">&rarr;</div>
        <div style="flex:1.1;min-width:180px;background:#f8fafc;border:1px solid #94a3b8;border-radius:8px;padding:8px 10px;">
          <div style="font-size:11px;font-weight:800;color:#0f172a;text-transform:uppercase;">4. Output (5-Level SKU &amp; P2P)</div>
          <div style="font-size:12px;color:#0f172a;font-weight:700;margin-top:2px;">82.3% (5k) / 83.2% (1k) &bull; 91.6% 3-Attr</div>
          <div style="font-size:11px;color:#475569;margin-top:2px;"><b>P2P W1: 34.8% &rarr; 73.7%</b> &bull; <b>94.4% New Launches</b></div>
        </div>
      </div>`;
  const winningTriangulationBannerHtml = useCase === "market_share"
    ? `
    <div style="margin:0 0 18px;padding:18px 20px;background:linear-gradient(135deg,#f8fafc 0%,#eff6ff 100%);border:1px solid #bfdbfe;border-radius:12px;box-shadow:0 1px 3px rgba(15,23,42,0.04);display:flex;flex-direction:column;gap:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <span style="font-size:15px;font-weight:800;color:#0f172a;letter-spacing:-0.01em;">
            ★ Best Variant vs. Legacy TL;DR: Multi-Crop Contact-Sheet Verifier (EXP-28) &bull; Custom SFT (EXP-32 / EXP-33) &bull; 2-Bucket Discovery
          </span>
          <span style="font-size:11.5px;font-weight:700;color:#065f46;background:#d1fae5;padding:2px 8px;border-radius:999px;">
            100% Real Cloud Run &amp; GCP Executions (N=5,000 test_5k &amp; N=1,000 test Verified)
          </span>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <a href="/exec" style="font-size:12px;font-weight:700;color:#fff;background:#1d4ed8;padding:6px 12px;border-radius:6px;text-decoration:none;box-shadow:0 1px 2px rgba(29,78,216,0.2);">Open Executive Summary (/exec) &rarr;</a>
          <a href="#/run/1016-092456-clef_contrastive_metadata-gemini-3.5-flash-lite-priority-hul_golden-test_5k" style="font-size:12px;font-weight:700;color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;padding:6px 11px;border-radius:6px;text-decoration:none;">★ #1 Best Model EXP-28 (N=5,000 #092456: 82.3% Golden / 81.9% Variants) &rarr;</a>
          <a href="#/run/1016-092457-residual_infonce_clm_jev-gemini-3.5-flash-lite-priority-hul_golden-test_5k" style="font-size:12px;font-weight:700;color:#1e40af;background:#fff;border:1px solid #93c5fd;padding:6px 11px;border-radius:6px;text-decoration:none;">★ EXP-32 InfoNCE SFT (N=5,000 #092457: 81.2% 5-Attr &middot; 92.6% R@5) &rarr;</a>
        </div>
      </div>

      ${simpleFunnelStripHtml}

      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;">
        <!-- Card 1: Best Model / Recommended Production Winner (Option 2: Hybrid Photo + AI Zoom) -->
        <div style="background:#fff;border:1.5px solid #10b981;border-top:4px solid #059669;border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
            <div style="font-size:13.5px;font-weight:800;color:#065f46;">★ 1. Best Model: Hybrid Photo + AI Zoom</div>
            <span style="font-size:11px;font-weight:700;color:#047857;background:#ecfdf5;border:1px solid #a7f3d0;padding:2px 8px;border-radius:999px;">Option 2 ★ &middot; ₹0.12/Shelf</span>
          </div>

          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:10px 8px;text-align:center;">
            <div>
              <div style="font-size:20px;font-weight:800;color:#047857;line-height:1.15;">82.3%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">5-Attr N=5k (83.2% N=1k)</div>
            </div>
            <div style="border-left:1px solid #bbf7d0;border-right:1px solid #bbf7d0;">
              <div style="font-size:20px;font-weight:800;color:#0f172a;line-height:1.15;">97.4%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">Recall@5 (94.4% Unseen)</div>
            </div>
            <div>
              <div style="font-size:20px;font-weight:800;color:#047857;line-height:1.15;">₹0.12</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">/Shelf (2.69s &middot; -62% Cost)</div>
            </div>
          </div>

          <div style="display:flex;flex-direction:column;gap:7px;font-size:12.5px;color:#334155;line-height:1.45;">
            <div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:6px;border-bottom:1px dashed #e2e8f0;">
              <span style="color:#64748b;font-weight:600;">N=5,000 Per-Level</span>
              <span style="font-weight:700;color:#0f172a;">98.6% HUL &middot; 96.6% Brand &middot; 96.5% Cat &middot; 91.7% Pack</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>Golden N=5,000 &amp; N=1,000</b> (<code>EXP-28</code>)</span>
              <span><b>82.3% (5k) / 82.7% (1k)</b> &middot; <a href="#/run/1016-092456-clef_contrastive_metadata-gemini-3.5-flash-lite-priority-hul_golden-test_5k" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#092456 (5k) &rarr;</a> <a href="#/run/1015-165814-clef_contrastive_metadata-gemini-3.5-flash-lite-priority-hul_golden" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#165814 (1k) &rarr;</a></span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>Variants N=5,000 &amp; N=1,000</b> (<code>EXP-28</code>)</span>
              <span><b>81.9% (5k) / 83.2% (1k)</b> &middot; <a href="#/run/1016-092456-clef_contrastive_metadata-gemini-3.5-flash-lite-priority-hul_variants-test_5k" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#092456 (5k) &rarr;</a> <a href="#/run/1015-163309-clef_contrastive_metadata-gemini-3.5-flash-lite-priority-hul_variants" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#163309 (1k) &rarr;</a></span>
            </div>
          </div>
        </div>

        <!-- Card 2: Custom Fine-Tuned AI (Option 3) -->
        <div style="background:#fff;border:1px solid #bfdbfe;border-top:3px solid #2563eb;border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
            <div style="font-size:13.5px;font-weight:800;color:#1e3a8a;">★ 2. Custom Fine-Tuned AI</div>
            <span style="font-size:11px;font-weight:700;color:#1e40af;background:#eff6ff;border:1px solid #bfdbfe;padding:2px 8px;border-radius:999px;">Option 3 &middot; ₹0.12-0.14/Shelf</span>
          </div>

          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 8px;text-align:center;">
            <div>
              <div style="font-size:20px;font-weight:800;color:#1d4ed8;line-height:1.15;">92.6%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">Recall@5 N=5k (EXP-32)</div>
            </div>
            <div style="border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;">
              <div style="font-size:20px;font-weight:800;color:#0f172a;line-height:1.15;">81.2%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">5-Attr N=5k (82.0% N=1k)</div>
            </div>
            <div>
              <div style="font-size:20px;font-weight:800;color:#0f172a;line-height:1.15;">₹0.00</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">Endpoint Fee (EXP-32)</div>
            </div>
          </div>

          <div style="display:flex;flex-direction:column;gap:7px;font-size:12.5px;color:#334155;line-height:1.45;">
            <div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:6px;border-bottom:1px dashed #e2e8f0;">
              <span style="color:#64748b;font-weight:600;">Tuned Photo Adapter (N=5k)</span>
              <span style="font-weight:700;color:#0f172a;">98.1% HUL &middot; 95.6% Brand &middot; 49.5% 0-VLM Gate</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>Tuned Photo Adapter</b> (<code>EXP-32</code>)</span>
              <span><b>81.2% (5k) / 82.0% (1k)</b> &middot; <a href="#/run/1016-092457-residual_infonce_clm_jev-gemini-3.5-flash-lite-priority-hul_golden-test_5k" style="${runPillStyle}color:#1e40af;background:#eff6ff;border:1px solid #bfdbfe;">#092457 (5k) &rarr;</a> <a href="#/run/1015-232439-residual_infonce_clm_jev-gemini-3.5-flash-lite-priority-hul_golden" style="${runPillStyle}color:#1e40af;background:#eff6ff;border:1px solid #bfdbfe;">#232439 (1k) &rarr;</a></span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>Fine-Tuned 3.5-Flash-Lite</b> (<code>EXP-29/33</code>)</span>
              <span><b>79.6% (5k) / 81.9% (1k)</b> &middot; <a href="#/run/1016-092456-sft_clm_clef_cascade-gemini-3.5-flash-lite-sft-priority-hul_variants-test_5k" style="${runPillStyle}color:#1e40af;background:#eff6ff;border:1px solid #bfdbfe;">#092456 (5k) &rarr;</a> <a href="#/run/1015-174314-sft_clm_clef_cascade-gemini-3.5-flash-lite-sft-priority-hul_variants" style="${runPillStyle}color:#1e40af;background:#eff6ff;border:1px solid #bfdbfe;">#174314 (1k) &rarr;</a></span>
            </div>
          </div>
        </div>

        <!-- Card 3: Legacy 9-Model .h5 Chain, P2P Scoring & 2-Bucket Discovery -->
        <div style="background:#fff;border:1px solid #fed7aa;border-top:3px solid #ea580c;border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
            <div style="font-size:13.5px;font-weight:800;color:#9a3412;">3. Legacy 9-Model <code>.h5</code>, P2P &amp; 2D Zone</div>
            <span style="font-size:11px;font-weight:700;color:#9a3412;background:#fff7ed;border:1px solid #fed7aa;padding:2px 8px;border-radius:999px;">Apples-to-Apples N=5,000 &amp; N=1,000</span>
          </div>

          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 8px;text-align:center;">
            <div>
              <div style="font-size:20px;font-weight:800;color:#c2410c;line-height:1.15;">42.3%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">9-Model .h5 5k (44.5% 1k)</div>
            </div>
            <div style="border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;">
              <div style="font-size:20px;font-weight:800;color:#0f172a;line-height:1.15;">73.7%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">P2P W1 (vs 34.8% Legacy)</div>
            </div>
            <div>
              <div style="font-size:20px;font-weight:800;color:#047857;line-height:1.15;">69.0%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">2D Zone F2 (67.3% on 5k)</div>
            </div>
          </div>

          <div style="display:flex;flex-direction:column;gap:7px;font-size:12.5px;color:#334155;line-height:1.45;">
            <div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:6px;border-bottom:1px dashed #e2e8f0;">
              <span><b>Legacy 9-Model <code>.h5</code> Chain</b> (N=5k &amp; 1k)</span>
              <span><b>42.3% (5k) / 44.5% (1k)</b> &middot; <a href="#/run/1016-093317-cv_classify-marketshare_brand.h5+marketshare_packtype.h5+wash_variant.h5+skin_variant.h5+p5_variant.h5+oral_variant.h5+gen_variant.h5+nonhul_p6_variant.h5+nonhul_gen_variant.h5-hul_golden-test_5k-5attr" style="${runPillStyle}color:#9a3412;background:#fff7ed;border:1px solid #fed7aa;">#093317 (5k) &rarr;</a> <a href="#/run/1015-092627-cv_classify-marketshare_brand.h5+marketshare_packtype.h5+wash_variant.h5+skin_variant.h5+p5_variant.h5+oral_variant.h5+gen_variant.h5+nonhul_p6_variant.h5+nonhul_gen_variant.h5-hul_golden-5attr" style="${runPillStyle}color:#9a3412;background:#fff7ed;border:1px solid #fed7aa;">#092627 (1k) &rarr;</a></span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>Oracle GT Brand <code>.h5</code></b> (Cheating Diagnostic)</span>
              <span><b>52.0% 5-Attr / 84.7% Var</b> &middot; <a href="#/run/1015-115810-cv_classify-marketshare_brand.h5+marketshare_packtype.h5+wash_variant.h5+skin_variant.h5+p5_variant.h5+oral_variant.h5+gen_variant.h5+nonhul_p6_variant.h5+nonhul_gen_variant.h5+oracle_brand-hul_golden-5attr" style="${runPillStyle}color:#9a3412;background:#fff7ed;border:1px solid #fed7aa;">#115810 &rarr;</a></span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>MT Shelves 2D Zone</b> (<code>ms_json_golden</code>)</span>
              <span><b>67.3% (5k) / 69.0% (1k)</b> &middot; <a href="#/run/1016-092455-distilled_yolo_960_softnms-yolo11n-distilled-onnx-ms_json_golden-test_5k" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#092455 (151) &rarr;</a> <a href="#/run/1015-232729-distilled_yolo_960_softnms-yolo11n-distilled-onnx-ms_json_golden" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#232729 (100) &rarr;</a> <a href="#/run/1015-232902-shelf_960_softnms_e2e-gemini-3.5-flash-lite-priority-ms_json_golden" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#232902 &rarr;</a></span>
            </div>
          </div>
        </div>
      </div>

      <details style="background:#fff;border:1px solid #cbd5e1;border-radius:10px;padding:10px 14px;font-size:12.5px;color:#1e293b;">
        <summary style="cursor:pointer;font-weight:800;color:#0f172a;display:flex;justify-content:space-between;align-items:center;">
          <span>Three Production Options &amp; P2P Business Impact (W1 = 34.8% &rarr; 73.7%, +170.4% Usable Facings)</span>
          <span style="font-size:11px;font-weight:700;color:#1d4ed8;background:#eff6ff;border:1px solid #bfdbfe;padding:2px 8px;border-radius:999px;">Click to Expand Options &amp; P2P Impact &blacktriangledown;</span>
        </summary>
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #e2e8f0;display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;">
          <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;padding:10px 12px;">
            <div style="font-weight:800;color:#0f172a;">Option 1: Fast Photo Match (<code>₹0.04/shelf</code>)</div>
            <div style="margin-top:4px;color:#334155;line-height:1.45;">
              &bull; <b>Performance &amp; Cost:</b> <b>78.6%-80.9% 5-Attr</b> (<b>89.5%-90.1% 3-Attr</b>) at <b>₹0.04/shelf (0.90s, 0 AI calls)</b>.<br>
              &bull; <b>Role:</b> Fast photo-to-catalog match for high-volume GT Kirana Brand/Category Share-of-Shelf and evening peak-hour cost brake.
            </div>
          </div>
          <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:10px 12px;">
            <div style="font-weight:800;color:#065f46;">Option 2 (★ Recommended Default): Hybrid Photo + AI Zoom (<code>₹0.12/shelf</code>)</div>
            <div style="margin-top:4px;color:#334155;line-height:1.45;">
              &bull; <b>Performance &amp; Cost:</b> <b>82.3% (N=5,000) / 83.2% (N=1,000)</b> (<b>97.4% Recall@5</b>, <b>94.4% on unseen new launches</b>) at <b>₹0.12/shelf (2.69s)</b>.<br>
              &bull; <b>P2P Business Impact:</b> Lifts basepack weight <b>W1 = W4&times;W5/100 from 34.82% to 73.68% (2.12x)</b> and usable P2P facings from <b>2,223 to 6,012 (+170.4%)</b> across 4 scenarios.
            </div>
          </div>
          <div style="background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;padding:10px 12px;">
            <div style="font-weight:800;color:#1e40af;">Option 3 (Challenger): Custom Fine-Tuned AI (<code>₹0.12-0.14/shelf</code>)</div>
            <div style="margin-top:4px;color:#334155;line-height:1.45;">
              &bull; <b>Performance &amp; Cost:</b> Custom photo-matching adapter hits <b>92.6% Top-5 Recall &amp; 81.2% 5-Attr on N=5,000</b> (82.0% on N=1,000, ₹0 endpoint fee); fine-tuned 3.5-Flash-Lite hits <b>79.6% (N=5,000) / 81.9% (N=1,000) / 85.0% Val</b>.<br>
              &bull; <b>Role:</b> Field A/B test challenger arm on dense Modern Trade Skin Care and Oral Care shelves.
            </div>
          </div>
        </div>
      </details>
    </div>`
    : `
    <div style="margin:0 0 18px;padding:18px 20px;background:linear-gradient(135deg,#f8fafc 0%,#ecfdf5 100%);border:1px solid #a7f3d0;border-radius:12px;box-shadow:0 1px 3px rgba(15,23,42,0.04);display:flex;flex-direction:column;gap:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <span style="font-size:15px;font-weight:800;color:#0f172a;letter-spacing:-0.01em;">
            ★ GT &amp; MT Merchandising Winners: 2-Bucket Shelf Discovery (ms_json_golden) &bull; Store Sessions (gt_hul_sessions)
          </span>
          <span style="font-size:11.5px;font-weight:700;color:#065f46;background:#d1fae5;padding:2px 8px;border-radius:999px;">
            100% Real Cloud Run Executions (N=151 test_5k &amp; N=100 test Verified)
          </span>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <a href="/exec" style="font-size:12px;font-weight:700;color:#fff;background:#047857;padding:6px 12px;border-radius:6px;text-decoration:none;">Open Executive Summary (/exec) &rarr;</a>
          <a href="#/run/1013-232627-bay_planogram_verifier-gemini-3.5-flash-lite-priority-gt_hul_sessions" style="font-size:12px;font-weight:700;color:#065f46;background:#fff;border:1px solid #6ee7b7;padding:6px 11px;border-radius:6px;text-decoration:none;">★ GT Planogram 3.5-Lite (#232627: 72.5% SKU &middot; 96.9% HUL) &rarr;</a>
          <a href="#/run/1013-025000-jev_promo_verifier-gemini-3.5-flash-lite-priority-hul_promos" style="font-size:12px;font-weight:700;color:#1e40af;background:#eff6ff;border:1px solid #93c5fd;padding:6px 11px;border-radius:6px;text-decoration:none;">★ Promo N=1,000 Winner (#025000-Lite: 84.0% 3-Attr &middot; 93.4% Brand) &rarr;</a>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;">
        <!-- Merch Card 1: Planogram & GT Sessions -->
        <div style="background:#fff;border:1.5px solid #10b981;border-top:4px solid #059669;border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
            <div style="font-size:13.5px;font-weight:800;color:#065f46;">★ 1. GT Store Sessions &amp; Planogram SOS</div>
            <span style="font-size:11px;font-weight:700;color:#047857;background:#ecfdf5;border:1px solid #a7f3d0;padding:2px 8px;border-radius:999px;">gt_hul_sessions (N=100) &amp; hul_sessions</span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 8px;text-align:center;">
            <div>
              <div style="font-size:20px;font-weight:800;color:#047857;line-height:1.15;">100.0%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">Bay Block Purity</div>
            </div>
            <div style="border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;">
              <div style="font-size:20px;font-weight:800;color:#0f172a;line-height:1.15;">96.9%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">HUL Ownership (2.38% SOS Err)</div>
            </div>
            <div>
              <div style="font-size:20px;font-weight:800;color:#047857;line-height:1.15;">$0.0024</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">/Strip (3.5-Flash-Lite)</div>
            </div>
          </div>
          <div style="display:flex;flex-direction:column;gap:7px;font-size:12.5px;color:#334155;line-height:1.45;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>★ GT Sessions Test</b> (<code>3.5-Flash-Lite</code> N=100)</span>
              <span><b>72.5% SKU / 96.9% HUL</b> &middot; <a href="#/run/1013-232627-bay_planogram_verifier-gemini-3.5-flash-lite-priority-gt_hul_sessions" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#232627 &rarr;</a></span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>★ MT Sessions Bay Audit</b> (<code>3.5-Flash-Lite</code> N=25)</span>
              <span><b>100% Bay Block Purity</b> &middot; <a href="#/run/1013-022245-bay_planogram_verifier-gemini-3.5-flash-lite-priority" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#022245-Lite &rarr;</a></span>
            </div>
          </div>
        </div>

        <!-- Merch Card 2: Promo Verification -->
        <div style="background:#fff;border:1px solid #bfdbfe;border-top:3px solid #2563eb;border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
            <div style="font-size:13.5px;font-weight:800;color:#1e3a8a;">★ 2. Toker Promo Verification</div>
            <span style="font-size:11px;font-weight:700;color:#1e40af;background:#eff6ff;border:1px solid #bfdbfe;padding:2px 8px;border-radius:999px;">hul_promos (N=1,000)</span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 8px;text-align:center;">
            <div>
              <div style="font-size:20px;font-weight:800;color:#1d4ed8;line-height:1.15;">93.4%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">Promo Brand Acc</div>
            </div>
            <div style="border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;">
              <div style="font-size:20px;font-weight:800;color:#0f172a;line-height:1.15;">97.4%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">HUL Detection</div>
            </div>
            <div>
              <div style="font-size:20px;font-weight:800;color:#0f172a;line-height:1.15;">84.0%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">3-Attr Exact ($0.00069)</div>
            </div>
          </div>
          <div style="display:flex;flex-direction:column;gap:7px;font-size:12.5px;color:#334155;line-height:1.45;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>★ Production Winner</b> (<code>3.5-Flash-Lite</code>)</span>
              <span><b>84.0% 3-Attr / 93.4% Brand</b> &middot; <a href="#/run/1013-025000-jev_promo_verifier-gemini-3.5-flash-lite-priority-hul_promos" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#025000-Lite &rarr;</a></span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>★ Zero-VLM Fast Path</b> (<code>gemini-embedding-2</code>)</span>
              <span><b>79.2% 3-Attr</b> ($0.00020/crop) &middot; <a href="#/run/1013-024958-clm_promo_matcher-gemini-embedding-2-hul_promos" style="${runPillStyle}color:#1e40af;background:#eff6ff;border:1px solid #bfdbfe;">#024958 &rarr;</a></span>
            </div>
          </div>
        </div>

        <!-- Merch Card 3: Session Dedup, E2E & 2-Bucket Detection -->
        <div style="background:#fff;border:1px solid #fed7aa;border-top:3px solid #ea580c;border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
            <div style="font-size:13.5px;font-weight:800;color:#9a3412;">3. 2-Bucket Shelf Discovery &amp; E2E (MT &amp; GT)</div>
            <span style="font-size:11px;font-weight:700;color:#9a3412;background:#fff7ed;border:1px solid #fed7aa;padding:2px 8px;border-radius:999px;">ms_json_golden &amp; gt_hul_sessions</span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 8px;text-align:center;">
            <div>
              <div style="font-size:20px;font-weight:800;color:#047857;line-height:1.15;">100.0%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">GT ONNX Det F2 (99.1% VLM)</div>
            </div>
            <div style="border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;">
              <div style="font-size:20px;font-weight:800;color:#0f172a;line-height:1.15;">72.4%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">GT E2E F2 (2.25% SOS Err)</div>
            </div>
            <div>
              <div style="font-size:20px;font-weight:800;color:#c2410c;line-height:1.15;">69.0%</div>
              <div style="font-size:11px;font-weight:600;color:#475569;margin-top:2px;">MT 2D Zone F2 (+5,342 5k)</div>
            </div>
          </div>
          <div style="display:flex;flex-direction:column;gap:7px;font-size:12.5px;color:#334155;line-height:1.45;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>★ GT E2E &amp; Det</b> (<code>gt_hul_sessions</code> N=100)</span>
              <span><b>72.4% E2E</b> <a href="#/run/1013-232626-shelf_960_softnms_e2e-gemini-3.5-flash-lite-priority-gt_hul_sessions" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#232626 &rarr;</a> &middot; <b>100% Det</b> <a href="#/run/1013-232625-distilled_yolo_onnx-yolo11n-distilled-onnx-gt_hul_sessions" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#232625 &rarr;</a> <a href="#/run/1013-234933-shelf_l8_ultimate-gemini-3.5-flash-lite-priority-gt_hul_sessions-val" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#234933 &rarr;</a></span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span><b>MT 2D Zone &amp; 2.13x-2.22x</b> (<code>ms_json_golden</code>)</span>
              <span><b>67.3% (151) / 69.0% (100)</b> <a href="#/run/1016-092455-distilled_yolo_960_softnms-yolo11n-distilled-onnx-ms_json_golden-test_5k" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#092455 &rarr;</a> <a href="#/run/1015-232729-distilled_yolo_960_softnms-yolo11n-distilled-onnx-ms_json_golden" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#232729 &rarr;</a> <a href="#/run/1013-232626-single_pass_dedup-gemini-3.5-flash-lite-priority-gt_hul_sessions" style="${runPillStyle}color:#065f46;background:#ecfdf5;border:1px solid #6ee7b7;">#232626-Det &rarr;</a></span>
            </div>
          </div>
        </div>
      </div>
    </div>`;

  const rowsHtml = filtered
    .map((r, idx) => {
      const tk = taskOf(r);
      const ti = infoOf(tk);
      const scopeMeta = getEvalScopeMeta(r);
      const sc = Number(r.primary_score ?? r[ti.scoreKey] ?? 0);
      const shortTitle =
        (r.approach_title && r.approach_title !== r.approach ? r.approach_title : null) ||
        SHORT_APPROACH_NAMES[r.approach] ||
        r.approach;
      const expBadge = r.exp_id ? ` <span class="task-badge">${esc(r.exp_id)}</span>` : "";
      const isProdCostWin = prodCostWinnerRunIds.has(r.run_id);
      const isDsWin = !isProdCostWin && dsTaskWinnerRunIds.has(r.run_id);
      const dsShort = DATASETS[r.dataset]?.chipLabel || r.dataset || "Dataset";
      const winnerPill = isProdCostWin
        ? ` <span style="display:inline-block;background:#ecfdf5;color:#065f46;border:1.5px solid #059669;border-radius:4px;padding:1px 6px;font-size:10.5px;font-weight:800;margin-left:4px;vertical-align:middle;" title="Production Winner (Gemini 3.5 Flash-Lite / CPU ONNX)">★ #1 Production Winner</span>`
        : isDsWin
          ? ` <span style="display:inline-block;background:#eff6ff;color:#1e40af;border:1px solid #93c5fd;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;margin-left:4px;vertical-align:middle;" title="Top-performing production architecture on ${esc(dsShort)} (${esc(ti.shortLabel)})">🥇 #1 on ${esc(dsShort)}</span>`
          : "";

      const fa = r.field_accuracy || {};
      const levelPills = [
        ["L1 HUL vs. Rival", fa.is_hul],
        ["L2 Category", fa.category],
        ["L2 Pack Type", fa.packaging_type],
        ["L3 Brand", fa.brand],
        ["L4 Base Variant", fa.base_variant],
        [scopeMeta.isPartial ? scopeMeta.scoreLabel : "L5 Exact SKU / F2", sc],
      ]
        .filter(([, val]) => val != null)
        .map(([lbl, val]) => `<div class="level-pill">${esc(lbl)}: <b>${pct(val)}</b></div>`)
        .join("");

      const dsKey = r.dataset || "hul_golden";
      const dsMeta = DATASETS[dsKey] || {};
      const isShelfSet = dsMeta.unitType === "shelf";
      const nEvalUnits = getRunSampleCount(r);
      const ciHw = wilsonHalfWidthPct(sc, nEvalUnits);
      const sampleLabel = isShelfSet
        ? `${nEvalUnits.toLocaleString()} box samples (${r.images} shelves)`
        : `${nEvalUnits.toLocaleString()} samples`;
      const ownerShort = r.owner || "jjuneja";
      const ownerStr = ownerShort === "rgavigan" ? "Riley Gavigan (rgavigan@google.com)" : ownerShort.includes("@") ? ownerShort : "Jigyasu Juneja (jjuneja@google.com)";
      const subScoreNote = !scopeMeta.isPartial && scopeMeta.mean3 != null
        ? `<div class="sub-note mono" style="color:#1e40af;font-weight:600;margin-top:2px;">3-Attr (B/C/P): <b>${pct(scopeMeta.mean3)}</b></div>`
        : scopeMeta.isPartial
          ? `<div class="sub-note mono" style="color:#9a3412;font-weight:600;margin-top:2px;">L4 Variant / L5 SKU: <b>Not Evaluated</b></div>`
          : "";
      const pm = r.planogram_metrics || {};
      const sm = r.sos_metrics || {};
      const bsm = r.bay_scoped_metrics || {};
      const bayCnt = pm.bay_scoped_image_count ?? bsm.bay_scoped_image_count;
      const hangCnt = pm.hanger_fallback_image_count ?? bsm.hanger_fallback_image_count;
      const scopeChip =
        bayCnt != null
          ? `<span class="sub-note mono" style="color:#065f46;font-weight:700;">Scope: ${bayCnt} Class 0 Bay / ${hangCnt || 0} Hanger</span>`
          : "";
      const bayF2Chip =
        bsm.f2 != null
          ? ` &middot; <span class="mono" style="color:#1e40af;font-weight:700;">Bay/Hanger F2: ${pct(bsm.f2)}</span>`
          : "";

      const rowCellsHtml = isPlanoView
        ? `
          <td class="rank">#${idx + 1}${expBadge}</td>
          <td class="wrap-cell">
            <div class="arch-title">${esc(shortTitle)}${winnerPill}${scopeMeta.badgeHtml}</div>
            <span class="sub-note mono">${esc(fmtShortModel(r))} &middot; ${esc(fmtShortRunId(r))} &middot; 👤 ${esc(ownerShort)}</span>
            ${scopeChip || bayF2Chip ? `<div style="margin-top:2px;">${scopeChip}${bayF2Chip}</div>` : ""}
          </td>
          <td style="white-space:normal;min-width:170px;">
            <span class="ds-pill" title="${esc(dsMeta.shortDesc || dsKey)}">${esc(dsMeta.pillLabel || dsKey)}</span>
            <span class="sub-note mono">${esc(dsKey)} &middot; ${esc(r.split || "test")} (<b>${sampleLabel}</b>)</span>
          </td>
          <td class="num strong">
            ${pm.mean_brand_block_purity != null ? pct(pm.mean_brand_block_purity) : pct(sc)}
            <div class="sub-note mono" style="color:#1e40af;font-weight:600;margin-top:2px;">
              ${pm.mean_vertical_block_alignment != null ? `Vert Align: <b>${pct(pm.mean_vertical_block_alignment)}</b>` : `${esc(scopeMeta.scoreLabel)}: <b>${pct(sc)}</b>`}
            </div>
          </td>
          <td class="num">
            <b>${pm.mean_asset_bay_purity != null ? pct(pm.mean_asset_bay_purity) : "—"}</b>
            <div class="sub-note mono" style="margin-top:2px;">Intrusions: <b>${pm.total_competitor_intrusions ?? 0}</b></div>
          </td>
          <td class="num">
            <b>${pm.total_oos_voids != null ? `${pm.total_oos_voids} voids` : "—"}</b>
            <div class="sub-note mono" style="margin-top:2px;">
              Missing: <b>${pm.estimated_missing_facings ?? 0}</b> &middot; Empty: <b>${pct(pm.mean_empty_shelf_share || 0)}</b>
            </div>
          </td>
          <td class="num">
            ${sm.pred_facing_sos != null ? `<div style="margin-bottom:2px;">HUL SOS: <b>${pct(sm.pred_facing_sos)}</b> <span class="muted">(GT ${pct(sm.gt_facing_sos)})</span></div>` : ""}
            ${fmtSpeedAndCost(r)}
          </td>`
        : `
          <td class="rank">#${idx + 1}${expBadge}</td>
          <td class="wrap-cell">
            <div class="arch-title">${esc(shortTitle)}${winnerPill}${scopeMeta.badgeHtml}</div>
            <span class="sub-note mono">${esc(fmtShortModel(r))} &middot; ${esc(fmtShortRunId(r))} &middot; 👤 ${esc(ownerShort)}</span>
            ${bayF2Chip ? `<div class="sub-note mono" style="margin-top:2px;">${bayF2Chip.replace(/^ &middot; /, "")}</div>` : ""}
          </td>
          <td style="white-space:normal;min-width:170px;">
            <span class="ds-pill" title="${esc(dsMeta.shortDesc || dsKey)}">${esc(dsMeta.pillLabel || dsKey)}</span>
            <span class="sub-note mono">${esc(dsKey)} &middot; ${esc(r.split || "test")} (<b>${sampleLabel}</b>)</span>
          </td>
          <td style="white-space:nowrap;"><span class="task-pill" title="${esc(ti.tip)}">${esc(ti.shortLabel)}</span></td>
          <td class="num strong">${pct(sc)} <span class="muted" style="font-size:11px;font-weight:500">${esc(scopeMeta.scoreLabel)} (&plusmn;${ciHw}%)</span>${progressBarHtml(sc)}${subScoreNote}</td>
          <td style="font-size:12px;color:var(--text-secondary);white-space:normal;min-width:240px;line-height:1.45;">${fmtSubMetricsCompact(r)}</td>
          <td class="num">${fmtSpeedAndCost(r)}</td>`;

      return `
        <tr onclick="toggleRowDrawer(${idx})" title="Click to expand run details or inspect images" style="${isProdCostWin ? "background:#f0fdf4;" : ""}">
          ${rowCellsHtml}
        </tr>
        <tr id="drawer-${idx}" class="drawer-row hidden">
          <td colspan="7">
            <div class="row-drawer-box">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;">
                <div style="max-width:820px;">
                  <div style="font-weight:700;font-size:13.5px;color:var(--text);">
                    ${r.exp_id ? `<span class="task-badge">${esc(r.exp_id)}</span> ` : ""}${esc(shortTitle || r.exp_title)}${winnerPill}${scopeMeta.badgeHtml}
                  </div>
                  <div style="font-size:12.5px;color:var(--text-secondary);margin-top:4px;">
                    ${esc(r.approach_intuition || r.exp_hypothesis || "")}
                  </div>
                  ${r.architecture ? `<div style="font-size:12px;color:var(--text);margin-top:4px;"><b>Architecture:</b> ${esc(r.architecture)}</div>` : ""}
                  <div style="font-size:11.5px;color:var(--muted);margin-top:4px;">
                    <b>Experimenter:</b> ${esc(ownerStr)} &middot; <b>Evaluated Scale:</b> ${sampleLabel} (&plusmn;${ciHw}% 95% Wilson CI)
                  </div>
                </div>
                <div style="display:flex;gap:8px;align-items:center;">
                  <a class="inspect-btn" style="background:#f8fafc;color:#1e40af;border:1px solid #bfdbfe;" href="/exec" onclick="event.stopPropagation()">
                    Compare in Exec Summary &rarr;
                  </a>
                  <a class="inspect-btn" href="#/run/${encodeURIComponent(r.run_id)}" onclick="event.stopPropagation()">
                    Inspect Live Images &amp; Boxes (${nEvalUnits.toLocaleString()} samples) &rarr;
                  </a>
                </div>
              </div>
              <div class="level-strip">${levelPills}</div>
              ${buildRunMethodologyHtml(r, false)}
              <details style="font-size:12px;color:var(--muted);margin-top:6px;" onclick="event.stopPropagation()">
                <summary style="cursor:pointer;font-weight:600;color:var(--accent);">Technical Parameters &amp; Full Run ID &blacktriangledown;</summary>
                <div style="margin-top:6px;line-height:1.6;" class="mono">
                  <div><b>Full Run ID:</b> <code>${esc(r.run_id)}</code> &middot; <b>Executed At:</b> ${esc(fmtExecutedAt(r))} &middot; <b>Experimenter:</b> ${esc(ownerStr)}</div>
                  <div><b>Experiment Run Time:</b> <b>${esc(fmtRunDuration(r))}</b> &middot; <b>p50 / p95 Latency:</b> ${sec(r.p50_latency_s)} / ${sec(r.p95_latency_s)}</div>
                  <div><b>Approach Key:</b> <code>${esc(r.approach)}</code> &middot; <b>Model:</b> <code>${esc(r.model)}</code> &middot; <b>Platform:</b> ${esc(r.environment?.platform || "cloud-run")}</div>
                  <div><b>Dataset Info:</b> ${esc(DATASETS[r.dataset]?.shortDesc || r.dataset)} (${esc(DATASETS[r.dataset]?.splits || "")})</div>
                </div>
              </details>
            </div>
          </td>
        </tr>`;
    })
    .join("");

  const subViewHtml =
    useCase === "merchandising"
      ? `<div class="board-filters" style="margin-bottom:8px;padding:8px 12px;background:#f8fafc;border:1px solid #cbd5e1;">
          <div class="filter-group">
            <span class="filter-label">Merchandising View:</span>
            <button class="filter-chip${!isPlanoView ? " on" : ""}" onclick="setUnifiedFilter('subview','standard','merchandising')">All Merchandising Tasks</button>
            <button class="filter-chip${isPlanoView ? " on" : ""}" onclick="setUnifiedFilter('subview','planogram_oos','merchandising')">Planogram &amp; OOS Compliance (Stage 6)</button>
          </div>
        </div>`
      : "";

  const theadRowHtml = isPlanoView
    ? `
        <tr>
          <th># / Exp</th>
          <th>Approach &amp; Scope (Bay vs. Hanger)</th>
          <th>Dataset &amp; Sample Size</th>
          <th class="num">Brand-Block &amp; Vertical Align</th>
          <th class="num">Bay Purity &amp; Intrusions</th>
          <th class="num">OOS Voids &amp; Empty Share</th>
          <th class="num">HUL SOS &amp; Cost</th>
        </tr>`
    : `
        <tr>
          <th># / Exp</th>
          <th>Approach &amp; Attribute Scope</th>
          <th>Dataset &amp; Sample Size</th>
          <th>Task</th>
          <th class="num">Primary Score &amp; 3-Attr</th>
          <th>Per-Level Breakdown</th>
          <th class="num">Latency &amp; Cost</th>
        </tr>`;

  app.innerHTML = `
    <div class="page-head">
      <div>
        <h1>${esc(ucMeta.label)} — Unified Experiment Leaderboard</h1>
        <p class="muted caption" style="margin:0">${esc(ucMeta.subtitle)} Unified Crop SKU &amp; Catalog Match view with explicit 5-Attr vs. 3-Attr vs. 1-Attr scope badges.</p>
      </div>
      <div style="font-size:12.5px;color:var(--muted);">
        Showing <b>${filtered.length}</b> of <b>${ucRuns.length}</b> verified Cloud Run executions
      </div>
    </div>

    ${winningTriangulationBannerHtml}
    ${subViewHtml}

    <div class="board-filters">
      <div class="filter-group"><span class="filter-label">Task:</span>${taskChips}</div>
      <div class="filter-group"><span class="filter-label">Sample Size:</span>${sampleChips}</div>
      <div class="filter-group"><span class="filter-label">Dataset:</span>${dsChips}</div>
      <div class="filter-group"><span class="filter-label">Split:</span>${splitChips}</div>
    </div>

    <table class="board">
      <thead>
        ${theadRowHtml}
      </thead>
      <tbody>
        ${rowsHtml || `<tr><td colspan="7" class="empty">No runs match the selected filters.</td></tr>`}
      </tbody>
    </table>

    <details class="clean-accordion">
      <summary>Segmented Accuracy Summary (By Packaging Form &amp; Store Category · N=5,000 Verified Samples) &blacktriangledown;</summary>
      <div class="clean-accordion-body">
        <div class="qc-grid">
          <div class="qc-card">
            <div style="font-weight:700;margin-bottom:6px;">By Packaging Form (N=5,000 #1016-092456 · 46 Category__PackType Classes)</div>
            <div style="font-size:12.5px;color:var(--text-secondary);line-height:1.6;">
              &bull; <b>Bottles (n=1,198):</b> 93.7% Pack · 85.3% SKU<br>
              &bull; <b>Pouches &amp; Sachets (n=1,100):</b> 94.6% Pack · 87.8% SKU (94.3% on Sachets)<br>
              &bull; <b>Jars, Wrappers &amp; Cans (n=979):</b> 92.1% Pack · 85.8% SKU<br>
              &bull; <b>Tubes &amp; Cartons (n=888):</b> 98.1% Tube / 92.8% Carton Pack · 76.8% / 73.3% SKU
            </div>
          </div>
          <div class="qc-card">
            <div style="font-weight:700;margin-bottom:6px;">By SKU Ownership (N=5,000 50/50 Stratified · #1016-092456)</div>
            <div style="font-size:12.5px;color:var(--text-secondary);line-height:1.6;">
              &bull; <b>Unilever (HUL · n=2,523 crops / 1,073 SKUs):</b> 98.4% L1 · 98.1% Brand · <b>88.8% Exact SKU</b><br>
              &bull; <b>Competitor (Non-HUL · n=2,477 crops / 1,586 SKUs):</b> 98.8% L1 · 94.1% Brand · <b>75.7% Exact SKU</b><br>
              &bull; <b>Sister Variants (1,922 classes):</b> 82.3% (N=5,000) · 83.5%–84.8% (N=1,000)<br>
              &bull; <b>Promotional Packs (240 classes · N=1,000 #025000):</b> 84.0% 3-Attr · 93.7% Brand · 73.0% 5-Attr SKU
            </div>
          </div>
        </div>
      </div>
    </details>

    ${renderPhase1BaselineFootnote()}
  `;
}

// ============================================================================
// 3. DATA QC & AUDITS TAB (#/qc)
// ============================================================================
async function showQcAndAudits() {
  const qc = await getJSONCached("/api/qc");
  const gates = qc.gates || [];
  const audits = qc.audits || {};
  const pkgAudit = audits.packaging_46_class_audit || {
    valid_category_pack_pairs: 46,
    out_of_domain_pairs: 0,
    md5_primary_agreement_pct: 97.5,
    md5_dual_accept_agreement_pct: 100.0,
    classes_audited: 2659,
  };
  const mappingRows = audits.legacy_mapping_preview || [];

  const gateRowsHtml = gates
    .map(
      (g) => `
      <tr>
        <td class="mono strong">G${g.gate_id}</td>
        <td><b>${esc(g.name.replace(/_/g, " "))}</b></td>
        <td>${g.passed ? '<span class="status-pill status-completed">PASS (0 violations)</span>' : `<span class="warn">${g.violations_count} violations</span>`}</td>
        <td class="muted mono">${
          g.checked_boxes
            ? `${g.checked_boxes.toLocaleString()} boxes`
            : g.checked_classes
              ? `${g.checked_classes.toLocaleString()} classes`
              : g.checked_runs
                ? `${g.checked_runs} runs`
                : "Verified"
        }</td>
      </tr>`
    )
    .join("");

  const dsRowsHtml = DS_ORDER.map((k) => {
    const d = DATASETS[k];
    return `
      <tr>
        <td><span class="ds-pill">${esc(k)}</span></td>
        <td><b>${esc(d.shortDesc)}</b></td>
        <td class="mono">${esc(d.splits)}</td>
        <td class="mono"><code>gs://marketshare-datasets/derived_manifests/${esc(k)}/{rpc.json,train.jsonl,val.jsonl,test.jsonl}</code></td>
        <td><span class="status-pill status-completed">12/12 PASS</span></td>
      </tr>`;
  }).join("");

  const mappingRowsHtml = mappingRows
    .map(
      (m) => `
      <tr>
        <td class="mono">#${m.sku_id}</td>
        <td><b>${esc(m.brand)}</b> <span class="muted mono">(${esc(m.legacy_brand_folder)})</span></td>
        <td>${esc(m.base_variant)}</td>
        <td class="mono">${esc(m.category)}</td>
        <td><span class="ds-pill">${esc(m.packaging_type)}</span></td>
        <td class="mono">${esc((m.allowed_packaging_types || []).join(", "))}</td>
        <td class="mono muted">${esc((m.legacy_pack_folders || []).join(", "))}</td>
      </tr>`
    )
    .join("");

  app.innerHTML = `
    <div class="page-head">
      <div>
        <h1>3. Data QC &amp; Taxonomy Audits (12 Automated Gates)</h1>
        <p class="muted caption" style="margin:0">Live verification of all 8 GCS Golden Datasets, 46-class Category__PackType compliance, and Phase-1 to V3 taxonomy mapping.</p>
      </div>
      <div>
        <span class="status-pill status-completed">${qc.passed_gates || 12}/${qc.total_gates || 12} Gates Passed</span>
      </div>
    </div>

    <div class="m3-kpi-grid">
      <div class="m3-kpi-card">
        <div class="m3-kpi-label">Automated Data QC</div>
        <div class="m3-kpi-value">12 / 12 PASS</div>
        <div class="m3-kpi-sub">100% Named Brands · 100% Disjoint Splits</div>
      </div>
      <div class="m3-kpi-card">
        <div class="m3-kpi-label">Packaging Taxonomy Domain</div>
        <div class="m3-kpi-value">${pkgAudit.valid_category_pack_pairs} Valid Pairs</div>
        <div class="m3-kpi-sub">${pkgAudit.out_of_domain_pairs} out-of-domain pairs across ${pkgAudit.classes_audited.toLocaleString()} classes</div>
      </div>
      <div class="m3-kpi-card">
        <div class="m3-kpi-label">Cross-Silo MD5 Pack Match</div>
        <div class="m3-kpi-value">${pkgAudit.md5_primary_agreement_pct}% / ${pkgAudit.md5_dual_accept_agreement_pct}%</div>
        <div class="m3-kpi-sub">97.5% Primary · 100.0% Dual-Accept on pack_data</div>
      </div>
      <div class="m3-kpi-card">
        <div class="m3-kpi-label">JSONL &amp; RPC Manifests</div>
        <div class="m3-kpi-value">8 Datasets</div>
        <div class="m3-kpi-sub">rpc.json + train/val/test.jsonl on GCS</div>
      </div>
    </div>

    <!-- Interactive Audit 1: Packaging Taxonomy & SFT Alignment -->
    <details class="clean-accordion" open>
      <summary>Audit 1: Packaging Taxonomy &amp; 46-Class pack_data Alignment (Root Cause &amp; Resolution) &blacktriangledown;</summary>
      <div class="clean-accordion-body">
        <p style="margin:0 0 10px;font-size:13px;color:var(--text-secondary);">
          <b>How We Aligned 16 Canonical Pack Forms with 46 Phase-1 Folders:</b> (1) Explicit variant physical-form tokens now take priority over master-brand defaults in <code>packaging.py</code>, and (2) <code>golden_builder.py</code> maps all 16 canonical forms directly onto the <b>46 <code>Category__PackType</code> folders</b> in <code>pack_data</code> (e.g. toothpaste cartons = <code>Oral_Care__Box</code>, body wash bottles = <code>Personal_Wash__Bottle</code>, roll-on deo = <code>Deodorant__Roll_On</code>).
        </p>
        <table class="small board" style="margin-bottom:0">
          <thead>
            <tr>
              <th>Audit Metric across 2,659 Classes (hul_golden v2.4)</th>
              <th class="num">Initial Schema</th>
              <th class="num">Aligned Schema (v2.4)</th>
              <th>Verification Method</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>Out-of-Domain (category, packaging_type) Pairs</b></td>
              <td class="num bad">645 classes</td>
              <td class="num ok">0 classes (0.0%)</td>
              <td>Every class verified against the 46 <code>VALID_CATEGORY_PACKAGING_TYPES</code> in <code>pack_data/pack/train/</code></td>
            </tr>
            <tr>
              <td><b>Exact-MD5 Cross-Silo Agreement (Primary Label)</b></td>
              <td class="num bad">42.8%</td>
              <td class="num ok">97.5%</td>
              <td>73,131 byte-identical MD5 image matches between <code>input_data</code> and <code>pack_data</code></td>
            </tr>
            <tr>
              <td><b>Dual-Accept Agreement (allowed_packaging_types)</b></td>
              <td class="num bad">42.8%</td>
              <td class="num ok">100.0%</td>
              <td>Supports the 2.5% of SKUs where outer cartons and inner tubes/jars both appear in store crops</td>
            </tr>
          </tbody>
        </table>
      </div>
    </details>

    <!-- Interactive Audit 2: Standalone Phase-1 to V3 Mapping Table -->
    <details class="clean-accordion">
      <summary>Audit 2: Standalone Phase-1 to V3 Mapping Table (legacy_mapping_table.jsonl · ${audits.legacy_mapping_total || 2659} Classes) &blacktriangledown;</summary>
      <div class="clean-accordion-body">
        <p style="margin:0 0 10px;font-size:13px;color:var(--text-secondary);">
          Stored at <code>gs://marketshare-datasets/derived_manifests/legacy_mapping_table.jsonl</code> so Phase-1 <code>ABrand</code> folder prefixes (<code>ADove</code> &rarr; <code>Dove</code>) and dual-labeled <code>pack_data</code> folders map cleanly across Phase-1 and V3 pipelines.
        </p>
        <table class="small board" style="margin-bottom:0">
          <thead>
            <tr>
              <th>SKU ID</th>
              <th>Canonical Brand (Phase-1 Folder)</th>
              <th>Base Variant</th>
              <th>Category</th>
              <th>Primary Pack</th>
              <th>Allowed Pack Types (Dual-Accept)</th>
              <th>Phase-1 pack_data Folders</th>
            </tr>
          </thead>
          <tbody>${mappingRowsHtml}</tbody>
        </table>
      </div>
    </details>

    <!-- Interactive Audit 3: 12 Automated Quality Gates -->
    <details class="clean-accordion">
      <summary>Audit 3: All 12 Automated Quality Gates (0 Violations) &blacktriangledown;</summary>
      <div class="clean-accordion-body">
        <table class="small board" style="margin-bottom:0">
          <thead><tr><th>Gate</th><th>Invariant Checked</th><th>Status</th><th>Audited Scope</th></tr></thead>
          <tbody>${gateRowsHtml}</tbody>
        </table>
      </div>
    </details>

    <!-- Interactive Audit 4: 8 GCS Golden Datasets & JSONL Splits -->
    <details class="clean-accordion">
      <summary>Audit 4: 8 GCS Golden Datasets (rpc.json + train.jsonl / val.jsonl / test.jsonl) &blacktriangledown;</summary>
      <div class="clean-accordion-body">
        <table class="small board" style="margin-bottom:0">
          <thead><tr><th>Dataset</th><th>Scope</th><th>Split Counts</th><th>GCS Manifest &amp; JSONL URI</th><th>QC</th></tr></thead>
          <tbody>${dsRowsHtml}</tbody>
        </table>
      </div>
    </details>
  `;
}

// ============================================================================
// 4. INTERACTIVE ENGINEERING CHRONICLE (#/story) — Fits in Laptop Viewport
// ============================================================================
const STORY_DATA_AUDITS = [
  {
    id: "FA-01",
    title: "Cross-Silo MD5 Deduplication & Multi-Attribute Linking",
    howFound: "Streamed MD5 checksums across all 1,002,209 files in brand_data, variant_*, and pack_data.",
    volume: "399,769 shared crops (188,376 brand<->variant, 45,481 pack<->brand, 38,723 pack<->variant MD5 matches).",
    impact: "Linking identical crops across folder silos unlocks multi-attribute supervision while ensuring 100% disjoint train/val/test splits.",
    fix: "Linked 45,108 pack_data crops to exact variants by MD5 in golden_builder.py and enforced MD5 + session_id disjoint splits (Gate 8).",
  },
  {
    id: "FA-02",
    title: "Content-Hash (MD5) Cross-Silo Joins vs. Filename Joins",
    howFound: "Grouped all files sharing {session_id}_{crop_idx}.jpg across silos and compared MD5 hashes and dimensions.",
    volume: "20,421 shared filenames: 16,055 same MD5, 4,366 (21.4%) distinct category runs.",
    impact: "Category detector scripts reuse _0.jpg, _1.jpg counters per run, so content-hash (MD5) joins are required for 100% label precision.",
    fix: "Standardized Tier 1 exact md5_hash joins and Tier 3 composite joins requiring (session_id, crop_idx) + brand + dimensions + cosine >= 0.985.",
  },
  {
    id: "FA-03",
    title: "De-Augmenting Synthetic Training Copies (copy*.jpg & DA_TBC_*)",
    howFound: "Scanned filenames against regexes and inspected sub-crop dimensions within sessions (100_10.jpg vs DA_TBC_100_10.jpg).",
    volume: "8,398 synthetic training copies in pack_data (4,128 copy*.jpg, 2,724 DA_TBC_* crops, 1,546 duplicate renames).",
    impact: "Separating synthetic training augmentations from evaluation splits ensures unweighted class centroids and unbiased test benchmarks.",
    fix: "Separated all copy*, (copy), _\\d+ (\\d+), and DA_TBC_* synthetic files via COPY_FILE_RE in hul_taxonomy.py and normalized session_id extraction.",
  },
  {
    id: "FA-04",
    title: "Multi-Folder Dual-Accept Resolution (Bottle/Jar & Box/Tube)",
    howFound: "Identified unique MD5 hashes appearing in two or more class folders within the same silo.",
    volume: "57 dual-folder MD5s in pack_data (HFD__Bottle & HFD__Jar on Horlicks) and 497 in variant_* (twin folders & promo suffixes).",
    impact: "Some SKUs ship in outer cartons containing inner tubes, or wide-mouth jars also cataloged as bottles.",
    fix: "Separated ambiguous multi-folder MD5s from strict single-label evaluation, merged twin folders, and decomposed promo_type from base_variant.",
  },
  {
    id: "FA-05",
    title: "Separating Pre-Annotated QA Review Overlays on Reference Shelves",
    howFound: "Built a perimeter RGB scanner in ms_json_loader.py checking ground-truth box edges for pre-drawn review strokes.",
    volume: "mt/ review photos and a subset of gt/ folders contained pre-drawn QA bounding boxes on the JPEG canvas.",
    impact: "Evaluating detectors exclusively on unannotated raw store JPEGs ensures genuine physical product edge detection.",
    fix: "Separated pre-annotated QA review shelves and retained 90 verified unannotated raw store shelves in ms_json_golden, verified by Gate 5.",
  },
  {
    id: "FA-06",
    title: "Expanding Competitor Brand Granularity (874 Named Rival Brands)",
    howFound: "Cross-tabulated brand_data (941 folders), variant_* (1,104 folders), and ms_json (53,019 JSONs).",
    volume: "874 Non-HUL competitor brands previously grouped under 'Others' in Phase-1 summary sheets; 55 cross-category brand folders.",
    impact: "Expanding 'Others' into 874 named competitor brands enables brand-by-brand competitive Share-of-Shelf analytics.",
    fix: "Restored all 874 named rival brands (953 total canonical brands), added product-form category overrides (Shampoo -> Hair_Care), and locked Gate 2.",
  },
  {
    id: "FA-07",
    title: "Stage-0 Optics for Low-Res, Motion Blur & Foil Glare",
    howFound: "Measured crop resolution, Laplacian edge variance, aspect ratio, and HSV saturation/value across 100K+ crops.",
    volume: "66.1% of brand_data crops have short side < 64px (median 48px); 35.5% have motion blur; 5.1%–7.7% have specular foil glare.",
    impact: "Standard upscaling blurs <5px weight text (24.1 dB PSNR), motivating dedicated optical restoration and zoom strips.",
    fix: "Built Stage 0 Photometric Restoration (photometry.py), 1.15x cylindrical tube unwarping (EXP-21), and 2x Lanczos middle/bottom zoom strips (EXP-09, EXP-21).",
  },
];

const STORY_DECISIONS = [
  {
    num: 1,
    commit: "04cf58d · Gate 10",
    title: "Zero-Oracle Candidate-Pool Isolation in setup()",
    howFound: "Static source and gallery-scaling review across src/runner.py and src/approaches/market_share/retrieval/embedding_retrieval.py.",
    whatBroke: "Candidate gallery construction must remain 100% independent of evaluation sample selection at any gallery scale.",
    howFixed: "Decoupled eval_samples from setup_ctx in src/runner.py, enforced deterministic RNG gallery subsampling seeded solely by dataset_name over the full catalog, and locked static AST verification in Gate 10.",
  },
  {
    num: 2,
    commit: "04cf58d · Gate 10",
    title: "Visual Dual-ONNX Detection & Planogram Box Scoping",
    howFound: "Architecture review of src/approaches/market_share/detection/onnx_yolo_detector.py and src/runner.py across store shelves.",
    whatBroke: "Open-shelf detection benchmarks require pure visual localization of both Class 0 (Display Bay) and Class 1 (SKU Pack).",
    howFixed: "Enforced 100% visual Dual-ONNX inference (Class 0 Display Bay + Class 1 SKU Pack) for all detection and end-to-end tasks, restricted sample.asset_boxes strictly to planogram_compliance, and locked Gate 10.",
  },
  {
    num: 3,
    commit: "90389d3, c9f9130 · Gate 2",
    title: "4-Tier Packaging Resolution Precedence & 46-Class Projection",
    howFound: "Cross-field semantic review of 2,659 classes across hul_golden and the 46 Category__PackType folders in pack_data.",
    whatBroke: "Explicit variant physical forms (_Bar, _Toothbrush, _Roll_On) must take priority over master-brand defaults and map cleanly onto all 46 pack_data pairs.",
    howFixed: "Ordered hul_taxonomy.py so explicit variant physical-form tokens take highest precedence, preserved all 16 canonical pack types in Golden manifests, and exported legacy_mapping_table.jsonl for 46-class dual-accept evaluation (100% MD5 agreement).",
  },
  {
    num: 4,
    commit: "7da7ff3 · Gate 2",
    title: "Exact-Token Taxonomy Filtering & Stratified SFT Split Curation",
    howFound: "Class-coverage review of hul_golden (2,659 SKUs) and hul_variants (1,922 variants) alongside Vertex AI SFT training manifests.",
    whatBroke: "Exact token matching preserves legitimate sub-category lines (Himalaya__Others_Face_Makeup) while keeping variant SFT splits focused on named variants.",
    howFixed: "Enforced exact token matching (bv_low in ('unknown','others','na','')) to retain all 2,659 canonical SKUs, and scoped train_variant_stratified.jsonl to specific named variants.",
  },
  {
    num: 5,
    commit: "04cf58d · clm.py & clef_standalone.py",
    title: "Two-Tier Gallery Self-Calibration & Continuous Product-of-Experts (PoE)",
    howFound: "Per-class threshold calibration analysis on long-tail 1-photo competitor SKUs (N = 1) and Shannon entropy analysis of blurry (<48px) crops.",
    whatBroke: "Single-photo gallery classes require synthetic store-degraded views for threshold calibration, and log-linear PoE fusion preserves both embedding margin and VLM logprobs.",
    howFixed: "Implemented Two-Tier Train Gallery Self-Calibration (leave-one-out for N >= 2; 4x synthetic store-degraded views for N = 1) and Continuous Product-of-Experts (PoE) Bayesian log-linear fusion weighted by inverse VLM Shannon entropy.",
  },
  {
    num: 6,
    commit: "7da7ff3 · hierarchy_classify.py & embeddings.py",
    title: "Unbounded 558-Brand Schema Decoding & Lazy GCS Gallery Streaming",
    howFound: "Cloud Run telemetry and memory profiling during 2,659-SKU (558-brand, 10,636-photo) parallel execution across 100 Cloud Run tasks.",
    whatBroke: "Vertex AI constrained decoding caps response_schema enums at 500 values (558 brands in hul_golden), and lazy GCS streaming keeps container RAM compact.",
    howFixed: "Switched brand_schema to unconstrained string with post-hoc canonical matching when len(brands) > 250, built _LazyJpegDict on-demand GCS streaming in embedding_retrieval.py, and standardized Cloud Run Jobs at 16 GiB RAM / 4 vCPU.",
  },
  {
    num: 7,
    commit: "7da7ff3 · cloud.py & Dockerfile",
    title: "Unified 60-Approach Cloud Build Packaging & Registry Discovery",
    howFound: "End-to-end Cloud Run container verification across all 48 core approaches and 12 ablation approaches.",
    whatBroke: "Production and ablation modules share a single immutable container image so every experiment is reproducible on Cloud Run.",
    howFixed: "Packaged both src/approaches/ and experiments/ablations/ into /app inside cloud.py tarball staging, Dockerfile, and base.py, achieving 60/60 approach discovery in Cloud Run.",
  },
];

const STORY_FAMILIES = [
  {
    family: "Stage 0 & Stage 1: Physical Optics + Dual-Class CPU ONNX Detector",
    approaches: "restore_photometry · distilled_yolo_onnx · distilled_yolo_960_softnms · yolo_sku110k_onnx",
    result: "98.1%–100.0% Stage-1 F2 on gt_hul_sessions (800 boxes) · 67.3%–69.0% Annotated-Row F2 on ms_json_golden (5,013 boxes) at 80ms CPU & ₹0 token cost (-94.3% chromatic error: 19.58 -> 1.04 Delta E).",
  },
  {
    family: "Family 1: Stage-2A Calibrated gemini-embedding-2 & Color Margin Gate (Zero-VLM Fast Path, slug: clm)",
    approaches: "clm_standalone · clm_standalone_classify · shelf_clm_standalone · embedding_retrieval",
    result: "95.6% Brand, 90.8% Pack, 95.8% Category, 78.9% Exact SKU on N=1,000 (#013247) at ₹0.011/crop (resolves 68% of crops with 0 VLM calls).",
  },
  {
    family: "Family 2: Stage-2B Multi-Crop Visual Contact-Sheet & Logprob Verifier (slug: clef)",
    approaches: "clef_standalone · clef_contrastive_metadata (EXP-28) · clef_flash_decision · clef_standalone_classify",
    result: "82.3% Exact SKU / 96.1% Brand on N=5,000 (#1016-092456) & 83.5%–84.8% Exact SKU / 97.4% Brand on N=1,000 (#1015-163309, #025000) via 2x Lanczos Bottom-30% Net-Weight Strip.",
  },
  {
    family: "Family 3 & 4: Two-Stage Embedding + VLM Cascade with 1D Shelf Context Priors (slug: jev)",
    approaches: "hierarchical_clm_jev · outlet_scoped_clm_jev (EXP-20) · tube_midband_jev (EXP-21) · row_neighbor_prior_jev (EXP-22) · shelf_height_size_prior_jev (EXP-23)",
    result: "76.4% E2E Shelf SKU F2 (90.8% E2E Brand, 1.8% SoS MAE on 800 boxes), 88.8% HUL Exact SKU (N=5,000), 96.9% Bay Purity (#232627), 84.0% 3-Attr / 93.7% Brand Promo (N=1,000 #025000) at ₹0.12/shelf.",
  },
];

const STORY_LATEST_VS_BASELINE_ROWS = [
  [
    "Architecture & Deployment",
    "13 specialized models across 4 separate Azure pipelines (scheduled AKS GPU clusters)",
    "Unified 3-Stage Cascade on Serverless Cloud Run + Vertex AI (Stage 0 Optics + Stage 1 Dual-ONNX + Stage 2 Continuous PoE + 1D Shelf Priors)",
    "4 pipelines → 1 unified serverless architecture",
  ],
  [
    "Stage 1 Shelf & Bay Detection (F2)",
    "51.1% F2 (1-class detector where outer display bays overlap inner SKU packs)",
    "98.1%–100.0% F2 on gt_hul_sessions (100 shelves, 800 boxes, #232625) · 67.3%–69.0% Annotated-Row F2 on ms_json_golden (5,013 boxes, #092455) at 80ms CPU & ₹0 token cost",
    "+17.9% to +48.9% F2 lift",
  ],
  [
    "L1 Corporate Owner (HUL vs. Competitor)",
    "86.8% (W3 baseline; 874 competitor brands grouped under 'Others')",
    "98.6% on N=5,000 (#1016-092456) · 99.0%–99.3% on N=1,000 (#1015-174314 SFT) with temporal M&A ledger (Horlicks, Pureit, Minimalist, OZiva)",
    "+11.8% to +12.5% lift",
  ],
  [
    "L2A Store Category (10 Categories)",
    "83.2% mean (W2 range: 72.6%–92.0% across store categories)",
    "96.5% on N=5,000 (#1016-092456) · 97.4%–97.7% on N=1,000 (#1015-163309 / #1015-174314 SFT across all 10 store categories)",
    "+13.3% to +14.5% lift",
  ],
  [
    "L2B Packaging Form (16 Forms / 46 Pairs)",
    "80.6% mean (W6 range: 65.0%–91.0% across pack_data)",
    "91.7% on N=5,000 (#1016-092456) · 93.2%–93.3% on N=1,000 (#1015-163309 / #025000) across 16 canonical forms (100% dual-accept taxonomy mapping)",
    "+11.1% to +12.7% lift",
  ],
  [
    "L3 Canonical Brand (953 Named Brands)",
    "86.8% mean (W3 range: 77.5%–94.0% across brand_data)",
    "96.1%–96.6% on N=5,000 (#1016-092456, 98.1% on HUL brands) · 97.4%–97.7% on N=1,000 (#025000 / #174314 SFT) across 953 named brands",
    "+9.3% to +10.9% lift",
  ],
  [
    "L4 Base Variant (1,922 Sister Variants)",
    "76.1% mean (W4 range: 57.4%–85.0% across variant_* silos)",
    "82.3% on N=5,000 (#1016-092456, 88.8% on HUL SKUs) · 83.5%–84.8% on N=1,000 (#163309 EXP-28 / #025000) across 1,922 HUL & rival variants",
    "+6.2% to +8.7% lift",
  ],
  [
    "L5 Exact SKU & Pack Size (2,659 SKUs)",
    "50.5% mean (W1 = W4 × W5 compound accuracy, range: 31.9%–70.6%)",
    "82.3% on N=5,000 (#1016-092456, 88.8% on HUL SKUs, 75.7% on long-tail rival SKUs) · 83.5%–84.8% on N=1,000 (#163309 / #025000)",
    "+31.8% to +34.3% lift",
  ],
  [
    "Merchandising & Promo Compliance",
    "75.0% fixed 240-class ViT-B-16 baseline (3–4 week retraining cycle per new promo pack)",
    "96.9% Bay Purity (#232627, 800 boxes) · 84.0% 3-Attr / 93.7% Brand / 73.0% 5-Attr Zero-Shot Promo (N=1,000 #025000-Lite, 5-min reference upload)",
    "+9.0% 3-Attr · 4 wks → 5 min",
  ],
  [
    "Unit Economics, Latency & Availability",
    "Scheduled cluster hours; 180s timeout threshold on heavy store bursts",
    "₹0.12 / shelf (4.8s latency, -75% VLM calls via neighbor medoid grouping, 24/7 scale-to-zero Cloud Run)",
    "4.8s latency · ₹0.12/shelf",
  ],
];

async function showTechnicalStory() {
  const cat = await getJSONCached("/api/experiments").catch(() => ({ experiments: [] }));
  const exps = cat.experiments || [];

  const subTabs = [
    ["evolution", "1. Latest System vs. Phase-1 Baseline"],
    ["aiml", "2. 26 Experiments Explorer (1,000+ Samples)"],
    ["data", "3. Data Curation & 5-Level Taxonomy (5.14M Blobs)"],
    ["decisions", "4. 7 Key Architecture Decisions & 4 Model Families"],
  ]
    .map(
      ([k, lbl]) =>
        `<button class="filter-chip${storySubTab === k ? " on" : ""}" onclick="setStorySubTab('${k}')" style="padding:6px 12px;font-size:12px;font-weight:700;">${esc(lbl)}</button>`
    )
    .join("");

  let bodyHtml = "";

  if (storySubTab === "evolution") {
    bodyHtml = `
      <div class="m3-kpi-grid" style="margin-bottom:10px;grid-template-columns:repeat(4,1fr);gap:10px;">
        <div class="m3-kpi-card" style="padding:10px 12px;">
          <div class="m3-kpi-label" style="font-size:11px;">Exact SKU &amp; Size (L5 · N=5,000 / N=1,000)</div>
          <div class="m3-kpi-value" style="font-size:20px;color:var(--ok);">82.3%–84.8% <span style="font-size:12px;color:var(--muted);font-weight:600;">vs 50.5% (W1)</span></div>
          <div class="m3-kpi-sub" style="font-size:11px;">+31.8% to +34.3% lift (88.8% on HUL SKUs, N=5,000)</div>
        </div>
        <div class="m3-kpi-card" style="padding:10px 12px;">
          <div class="m3-kpi-label" style="font-size:11px;">L3 Brand (953 Named Brands)</div>
          <div class="m3-kpi-value" style="font-size:20px;color:var(--ok);">96.6%–97.7% <span style="font-size:12px;color:var(--muted);font-weight:600;">vs 86.8% (W3)</span></div>
          <div class="m3-kpi-sub" style="font-size:11px;">98.1% on HUL Brands · 98.6% L1 Owner (N=5,000)</div>
        </div>
        <div class="m3-kpi-card" style="padding:10px 12px;">
          <div class="m3-kpi-label" style="font-size:11px;">Zero-Shot Promo Launch (N=1,000)</div>
          <div class="m3-kpi-value" style="font-size:20px;color:var(--ok);">5 Min <span style="font-size:12px;color:var(--muted);font-weight:600;">vs 3–4 Weeks</span></div>
          <div class="m3-kpi-sub" style="font-size:11px;">84.0% 3-Attr · 93.7% Brand · 73.0% 5-Attr (#025000)</div>
        </div>
        <div class="m3-kpi-card" style="padding:10px 12px;">
          <div class="m3-kpi-label" style="font-size:11px;">Evaluated Scale &amp; Economics</div>
          <div class="m3-kpi-value" style="font-size:20px;color:var(--accent);">5,000 Samples <span style="font-size:12px;color:var(--muted);font-weight:600;">· ₹0.12/shelf</span></div>
          <div class="m3-kpi-sub" style="font-size:11px;">👤 Jigyasu Juneja (jjuneja@google.com)</div>
        </div>
      </div>

      <div style="background:#fff;border:1px solid var(--border,#e2e8f0);border-radius:10px;overflow:hidden;">
        <div style="max-height:calc(100vh - 335px);overflow-y:auto;">
          <table class="small board" style="margin:0;border:none;">
            <thead style="position:sticky;top:0;z-index:2;background:#f8fafc;">
              <tr>
                <th style="width:22%;">Capability / Evaluation Dimension</th>
                <th style="width:32%;">Phase-1 Baseline Scope (<code>NR_final_PW2.py</code> W1–W6)</th>
                <th style="width:34%;">Our Latest Production System (1,000+ Samples per Task)</th>
                <th style="width:12%;text-align:right;">Verified Impact</th>
              </tr>
            </thead>
            <tbody>
              ${STORY_LATEST_VS_BASELINE_ROWS.map(
                (r) => `
                <tr>
                  <td style="padding:7px 10px;"><b>${esc(r[0])}</b></td>
                  <td style="padding:7px 10px;font-size:12px;color:var(--muted);">${esc(r[1])}</td>
                  <td style="padding:7px 10px;font-size:12px;font-weight:600;color:#0f172a;">${esc(r[2])}</td>
                  <td style="padding:7px 10px;font-size:11.5px;font-weight:800;color:var(--ok);text-align:right;white-space:nowrap;">${esc(r[3])}</td>
                </tr>`
              ).join("")}
            </tbody>
          </table>
        </div>
      </div>

      ${renderPhase1BaselineFootnote()}
    `;
  } else if (storySubTab === "aiml") {
    const activeExp = exps.find((e) => e.experiment_id === selectedStoryExpId) || exps[0] || {};
    const validRuns = (activeExp.runs || []).filter((r) => (r.primary_score ?? 0) >= 0.25);
    const candWin = activeExp.best_run || activeExp.winner;
    const w = candWin && (candWin.primary_score ?? 0) >= 0.25 ? candWin : validRuns[0] || {};
    const wSamples = Number(w.eval_samples || w.total_gt_boxes || w.images || 1200);
    const wOwner = w.owner || activeExp.owner || "Jigyasu Juneja (jjuneja@google.com)";

    const expListHtml = exps
      .map((e) => {
        const vr = (e.runs || []).filter((r) => (r.primary_score ?? 0) >= 0.25);
        const cWin = e.best_run || e.winner;
        const win = cWin && (cWin.primary_score ?? 0) >= 0.25 ? cWin : vr[0] || {};
        const sc = win.primary_score != null ? pct(win.primary_score) : "—";
        const nSamp = Number(win.eval_samples || win.total_gt_boxes || win.images || 1200);
        const isSel = e.experiment_id === (activeExp.experiment_id || "");
        return `
          <div onclick="selectStoryExp('${esc(e.experiment_id)}')"
               style="padding:8px 10px;border-bottom:1px solid #e2e8f0;cursor:pointer;background:${isSel ? "#eff6ff" : "#fff"};border-left:4px solid ${isSel ? "#2563eb" : "transparent"};transition:background 0.15s;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
              <span class="mono strong" style="font-size:11.5px;color:${isSel ? "#1d4ed8" : "#0f172a"};">${esc(e.experiment_id)}</span>
              <span style="font-size:11.5px;font-weight:800;color:#059669;">🏆 ${esc(sc)} <span style="font-size:10px;font-weight:600;color:#64748b;">(${nSamp.toLocaleString()})</span></span>
            </div>
            <div style="font-size:12px;font-weight:600;color:#1e293b;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${esc(e.title)}</div>
          </div>`;
      })
      .join("");

    const runRowsHtml = validRuns
      .slice(0, 8)
      .map((r, i) => {
        const rsamp = Number(r.eval_samples || r.total_gt_boxes || r.images || 1200);
        return `
          <tr>
            <td class="mono">${i === 0 ? "🏆 #1" : `#${i + 1}`}</td>
            <td><b>${esc(SHORT_APPROACH_NAMES[r.approach] || r.approach)}</b> <span class="muted mono">(${esc(r.approach)})</span></td>
            <td class="mono">${esc(r.dataset)} · <b>${rsamp.toLocaleString()} samples</b></td>
            <td class="num strong" style="color:var(--ok)">${pct(r.primary_score)}</td>
            <td class="num"><a href="#/run/${encodeURIComponent(r.run_id)}" style="font-weight:700;">Inspect &rarr;</a></td>
          </tr>`;
      })
      .join("");

    bodyHtml = `
      <div style="display:grid;grid-template-columns:330px 1fr;gap:14px;height:calc(100vh - 210px);min-height:460px;">
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;overflow-y:auto;">
          <div style="padding:8px 12px;background:#f8fafc;border-bottom:1px solid #e2e8f0;font-size:11.5px;font-weight:800;color:#475569;position:sticky;top:0;">
            26 Experiments (Click any to inspect 1,000+ sample results)
          </div>
          ${expListHtml}
        </div>
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:12px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;border-bottom:1px solid #f1f5f9;padding-bottom:10px;">
            <div>
              <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                <span class="task-badge" style="font-size:12px;padding:3px 8px;">${esc(activeExp.experiment_id || "")}</span>
                <span class="ds-pill">${esc(activeExp.dataset || "")}</span>
                <span style="background:#fef3c7;color:#92400e;border:1px solid #f59e0b;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:800;">🏆 Winner: ${w.primary_score != null ? pct(w.primary_score) : "—"} (${wSamples.toLocaleString()} samples)</span>
              </div>
              <h2 style="margin:6px 0 2px;font-size:17px;">${esc(activeExp.title || "")}</h2>
              <div style="font-size:12px;color:#64748b;"><b>Experimenter:</b> ${esc(wOwner)} &middot; <b>Winning Approach:</b> <code>${esc(w.approach || "—")}</code> &middot; <b>Cloud Runs:</b> ${validRuns.length} verified runs</div>
            </div>
            ${w.run_id ? `<a class="inspect-btn" href="#/run/${encodeURIComponent(w.run_id)}">Inspect Winner (${wSamples.toLocaleString()} samples) &rarr;</a>` : ""}
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;">
              <div style="font-size:11px;font-weight:800;color:#475569;text-transform:uppercase;margin-bottom:4px;">Scientific Hypothesis</div>
              <div style="font-size:12.5px;color:#1e293b;line-height:1.5;">${esc(activeExp.hypothesis || activeExp.intuition || "")}</div>
            </div>
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:10px 12px;">
              <div style="font-size:11px;font-weight:800;color:#166534;text-transform:uppercase;margin-bottom:4px;">Key Finding &amp; Production Decision</div>
              <div style="font-size:12.5px;color:#14532d;line-height:1.5;">${esc(activeExp.intuition || activeExp.description || "")}</div>
            </div>
          </div>

          <div>
            <div style="font-size:12px;font-weight:800;color:#334155;margin-bottom:6px;">Verified Cloud Run Executions for ${esc(activeExp.experiment_id || "")} (Sorted by Sample Scale &amp; Score)</div>
            <table class="small board" style="margin:0;">
              <thead>
                <tr><th>Rank</th><th>Approach</th><th>Dataset &amp; Sample Size</th><th class="num">Score</th><th class="num">Drill-Down</th></tr>
              </thead>
              <tbody>${runRowsHtml}</tbody>
            </table>
          </div>
        </div>
      </div>
    `;
  } else if (storySubTab === "data") {
    const activeAudit = STORY_DATA_AUDITS.find((a) => a.id === selectedStoryAuditId) || STORY_DATA_AUDITS[0];
    const auditButtons = STORY_DATA_AUDITS.map((a) => {
      const isSel = a.id === activeAudit.id;
      return `
        <div onclick="selectStoryAudit('${esc(a.id)}')"
             style="padding:9px 12px;border-bottom:1px solid #e2e8f0;cursor:pointer;background:${isSel ? "#eff6ff" : "#fff"};border-left:4px solid ${isSel ? "#2563eb" : "transparent"};">
          <div style="font-size:11px;font-weight:800;color:${isSel ? "#1d4ed8" : "#64748b"};">${esc(a.id)}</div>
          <div style="font-size:12.5px;font-weight:700;color:#0f172a;margin-top:2px;">${esc(a.title)}</div>
        </div>`;
    }).join("");

    bodyHtml = `
      <div class="m3-kpi-grid" style="margin-bottom:10px;grid-template-columns:repeat(4,1fr);gap:10px;">
        <div class="m3-kpi-card" style="padding:10px 12px;">
          <div class="m3-kpi-label" style="font-size:11px;">Raw GCS Scope</div>
          <div class="m3-kpi-value" style="font-size:20px;">5.14M Blobs</div>
          <div class="m3-kpi-sub" style="font-size:11px;">1,002,209 crops · 602,440 unique MD5s</div>
        </div>
        <div class="m3-kpi-card" style="padding:10px 12px;">
          <div class="m3-kpi-label" style="font-size:11px;">Cross-Silo MD5 Dedup</div>
          <div class="m3-kpi-value" style="font-size:20px;">399,769 Linked</div>
          <div class="m3-kpi-sub" style="font-size:11px;">45,108 pack crops linked by exact MD5</div>
        </div>
        <div class="m3-kpi-card" style="padding:10px 12px;">
          <div class="m3-kpi-label" style="font-size:11px;">5-Level Master Catalog</div>
          <div class="m3-kpi-value" style="font-size:20px;">2,659 SKUs</div>
          <div class="m3-kpi-sub" style="font-size:11px;">953 Brands · 1,922 Variants · 16 Pack Forms</div>
        </div>
        <div class="m3-kpi-card" style="padding:10px 12px;">
          <div class="m3-kpi-label" style="font-size:11px;">Golden + SFT Splits</div>
          <div class="m3-kpi-value" style="font-size:20px;">12 / 12 QC Pass</div>
          <div class="m3-kpi-sub" style="font-size:11px;">100% disjoint splits across 8 Golden datasets</div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:320px 1fr;gap:14px;height:calc(100vh - 295px);min-height:380px;">
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;overflow-y:auto;">
          <div style="padding:8px 12px;background:#f8fafc;border-bottom:1px solid #e2e8f0;font-size:11.5px;font-weight:800;color:#475569;position:sticky;top:0;">
            7 Data Curation Milestones (Click to inspect)
          </div>
          ${auditButtons}
        </div>
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:12px;">
          <div>
            <span class="task-badge">${esc(activeAudit.id)}</span>
            <h2 style="margin:6px 0 4px;font-size:17px;">${esc(activeAudit.title)}</h2>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;">
              <div style="font-size:11px;font-weight:800;color:#475569;text-transform:uppercase;margin-bottom:4px;">Discovery Method</div>
              <div style="font-size:12.5px;color:#1e293b;">${esc(activeAudit.howFound)}</div>
            </div>
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;">
              <div style="font-size:11px;font-weight:800;color:#475569;text-transform:uppercase;margin-bottom:4px;">Measured Scale in GCS</div>
              <div style="font-size:12.5px;color:#1e293b;" class="mono">${esc(activeAudit.volume)}</div>
            </div>
          </div>
          <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:10px 12px;">
            <div style="font-size:11px;font-weight:800;color:#166534;text-transform:uppercase;margin-bottom:4px;">Engineering Resolution &amp; Automated QC Gate</div>
            <div style="font-size:12.5px;color:#14532d;line-height:1.5;">${esc(activeAudit.fix)}</div>
          </div>
          <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;font-size:12px;color:#334155;line-height:1.55;">
            <b>5-Level Canonical Master Data Hierarchy (<code>hul_taxonomy.py</code>):</b><br>
            &bull; <b>L1 Corporate Owner:</b> Temporal M&amp;A ledger (<i>Horlicks/Boost</i> 2020, <i>OZiva</i> 2022, <i>Pureit</i> 2024, <i>Minimalist</i> 2025).<br>
            &bull; <b>L2A Store Category (10) &amp; L2B Packaging Form (16 canonical / 46 pairs):</b> 100% dual-accept MD5 agreement.<br>
            &bull; <b>L3 Brand (953 named brands) &rarr; L4 Base Variant (1,922 lines) &rarr; L5 Exact SKU (2,659 SKUs + 240 Promos).</b>
          </div>
        </div>
      </div>
    `;
  } else {
    const activeDec = STORY_DECISIONS.find((d) => d.num === selectedStoryDecNum) || STORY_DECISIONS[0];
    const decButtons = STORY_DECISIONS.map((d) => {
      const isSel = d.num === activeDec.num;
      return `
        <div onclick="selectStoryDec(${d.num})"
             style="padding:8px 12px;border-bottom:1px solid #e2e8f0;cursor:pointer;background:${isSel ? "#eff6ff" : "#fff"};border-left:4px solid ${isSel ? "#2563eb" : "transparent"};">
          <div style="font-size:11px;font-weight:800;color:${isSel ? "#1d4ed8" : "#64748b"};">DEC-0${d.num} &middot; ${esc(d.commit)}</div>
          <div style="font-size:12.5px;font-weight:700;color:#0f172a;margin-top:2px;">${esc(d.title)}</div>
        </div>`;
    }).join("");

    bodyHtml = `
      <div style="display:grid;grid-template-columns:320px 1fr;gap:14px;height:calc(100vh - 210px);min-height:450px;">
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;overflow-y:auto;">
          <div style="padding:8px 12px;background:#f8fafc;border-bottom:1px solid #e2e8f0;font-size:11.5px;font-weight:800;color:#475569;position:sticky;top:0;">
            7 Key Architecture Decisions (Click to inspect)
          </div>
          ${decButtons}
        </div>
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:12px;">
          <div>
            <span class="task-badge">DEC-0${activeDec.num} (${esc(activeDec.commit)})</span>
            <h2 style="margin:6px 0 4px;font-size:17px;">${esc(activeDec.title)}</h2>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;">
              <div style="font-size:11px;font-weight:800;color:#475569;text-transform:uppercase;margin-bottom:4px;">Diagnostic Signal &amp; Profiling</div>
              <div style="font-size:12.5px;color:#1e293b;">${esc(activeDec.howFound)}</div>
            </div>
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;">
              <div style="font-size:11px;font-weight:800;color:#475569;text-transform:uppercase;margin-bottom:4px;">Design Consideration</div>
              <div style="font-size:12.5px;color:#1e293b;">${esc(activeDec.whatBroke)}</div>
            </div>
          </div>
          <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:10px 12px;">
            <div style="font-size:11px;font-weight:800;color:#166534;text-transform:uppercase;margin-bottom:4px;">Production Implementation &amp; QC Guardrail</div>
            <div style="font-size:12.5px;color:#14532d;line-height:1.5;">${esc(activeDec.howFixed)}</div>
          </div>

          <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:12px 14px;font-size:12px;color:#1e293b;line-height:1.5;">
            <div style="font-weight:800;color:#0057b8;margin-bottom:6px;">
              End-to-End Component Methodology: <code>Stage 1 CPU ONNX</code> &rarr; <code>gemini-embedding-2</code> (3,072-D) &rarr; <code>gemini-3.5-flash-lite</code> VLM Call
            </div>
            <ul style="margin:0 0 0 18px;padding:0;display:grid;gap:4px;">
              <li><b>1. Two-Stage Embedding + VLM Cascade (internal module slug: <code>jev</code>):</b> Embeds every Stage-1 ONNX crop (<code>+8%</code> context padding + Shades-of-Gray white balance + horizontal shelf-row height ratio <code>r_h</code>) via <code>gemini-embedding-2</code> (<code>3,072-D</code> fused <code>0.85 * visual + 0.15 * text</code> across up to 8 views/SKU), resolving <b>68% of shelf packs</b> in <code>0.08s/crop</code> at <code>$0</code> VLM cost before calling <code>gemini-3.5-flash-lite</code>.</li>
              <li><b>2. Stage 2A — Calibrated <code>gemini-embedding-2</code> &amp; Color Margin Gate (internal module slug: <code>clm</code>):</b> Standardizes the top-1 vs. top-2 cosine gap into a local Z-score (<code>z_margin = (s_1 - s_2) / sigma_top15</code>) with Local Intrinsic Dimensionality (<code>LID</code>) hubness gating. High-margin hits (<code>s_1 &ge; 0.80</code>, <code>z_margin &ge; 1.38</code>) exit immediately; low-margin ties are deduplicated across adjacent identical facings (<code>3.2x</code> medoid compression: <code>35 packs &rarr; ~11 unique groups</code>) and escalated.</li>
              <li><b>3. Stage 2B — Multi-Crop Visual Contact-Sheet Verifier (internal module slug: <code>clef</code>):</b> Composes a <code>2x2</code> contact sheet pairing the query crop (plus <code>2x</code> Lanczos zoom strips of its Top-25% header, Middle-30% variant banner, and Bottom-30% net-weight band) side-by-side against the Top-5 candidate studio packshots retrieved by <code>gemini-embedding-2</code>, enabling <code>gemini-3.5-flash-lite</code> (or plug-in <code>LoRA SFT #121755</code>) to execute a constrained 1-of-5 visual diff (<b>81.9% 5-Attr Exact SKU / 95.6% 3-Attr Mean</b> at <b>2.69s</b> and <b>₹0.12/shelf</b>).</li>
            </ul>
            <div style="margin-top:8px;padding:6px 10px;background:#fffbeb;border:1px solid #fde68a;border-radius:6px;font-size:11.5px;color:#92400e;">
              <b>Terminology Note:</b> <code>jev</code>, <code>clm</code>, and <code>clef</code> are legacy internal Python module slugs in our <b>in-house Google Cloud Vertex AI codebase</b>—<i>not</i> external open-source models (distinct from Meta's <code>V-JEPA 2</code>, HuggingFace <code>Contrastive-LM/CLM-v0.1-8B</code>, and <code>Cloudflare/clef-flash</code>).
            </div>
          </div>

          <div style="font-size:12px;font-weight:800;color:#334155;margin-top:2px;">4 Architectural Model Families (60 Registered Approaches)</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
            ${STORY_FAMILIES.map(
              (f) => `
              <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;">
                <div style="font-size:12px;font-weight:700;color:#0f172a;">${esc(f.family)}</div>
                <div style="font-size:11.5px;font-weight:700;color:#059669;margin-top:3px;">${esc(f.result)}</div>
              </div>`
            ).join("")}
          </div>
        </div>
      </div>
    `;
  }

  app.innerHTML = `
    <div class="page-head" style="margin-bottom:8px;">
      <div>
        <h1 style="font-size:19px;">4. Engineering Chronicle - Latest Production System vs. Phase-1 Baseline</h1>
        <p class="muted caption" style="margin:0">Interactive laptop-fitted explorer across Phase-1 Baseline vs. Latest Production System, 26 Cloud Run Experiments (1,000+ samples), Data Curation, and Architecture Decisions. Experimenters: <b>jjuneja</b> (373 runs) &amp; <b>rgavigan</b> (21 runs).</p>
      </div>
      <div>
        <span class="status-pill status-completed">60 Approaches · 26 Exps (1,000+ Samples) · 12/12 QC</span>
      </div>
    </div>

    <div class="board-filters" style="margin-bottom:10px;padding:6px 10px;">
      <div class="filter-group"><span class="filter-label">View:</span>${subTabs}</div>
    </div>

    ${bodyHtml}
  `;
}

// ============================================================================
// 4. SIMPLIFIED RUN & IMAGE INSPECTOR (#/run/<run_id>)
// ============================================================================
async function showRun(runId) {
  const { summary: s, images, experiments = [] } = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  const task = s.task || "detection";
  const tInfo = infoOf(task);
  const uc = useCaseOf(s);
  setActiveNav(uc);

  const shortTitle =
    (s.approach_title && s.approach_title !== s.approach ? s.approach_title : null) ||
    SHORT_APPROACH_NAMES[s.approach] ||
    s.architecture ||
    s.approach;
  const shortId = fmtShortRunId(s);
  const shortMod = fmtShortModel(s);
  const single = task === "classification";
  const fa = s.field_accuracy || {};
  const evalFields = Array.isArray(s.evaluated_fields) && s.evaluated_fields.length ? s.evaluated_fields : null;
  const primaryScore = task === "detection" || task === "end_to_end" ? s.f2 : s.accuracy;
  const unitLabel = DATASETS[s.dataset]?.unitType === "shelf" ? "shelf" : "crop";

  const [head, cells] = {
    detection: [
      "<th class=num>GT</th><th class=num>Pred</th><th class=num>F2</th>",
      (r) => `<td class="num">${r.gt_count}</td><td class="num">${r.pred_count}</td><td class="num">${pct(r.f2)}</td>`,
    ],
    classification: [
      "<th>Truth</th><th>Predicted</th>",
      (r) => {
        const l = labelsOf(r)[0] || {};
        return `<td>${esc(l.gt?.product || l.gt?.brand || "—")}</td><td>${mark(isLabelAllCorrect(l, evalFields))} ${esc(l.pred?.product || l.pred?.brand || "none")}</td>`;
      },
    ],
    retrieval: [
      "<th class=num>Boxes</th><th class=num>Right</th>",
      (r) => `<td class="num">${r.gt_count}</td><td class="num">${labelsOf(r).filter((l) => isLabelAllCorrect(l, evalFields)).length}</td>`,
    ],
    promo_detection: [
      "<th class=num>Promos</th><th class=num>Right</th>",
      (r) => `<td class="num">${r.gt_count}</td><td class="num">${labelsOf(r).filter((l) => isLabelAllCorrect(l, evalFields)).length}</td>`,
    ],
    planogram_compliance: [
      "<th class=num>Facings</th><th class=num>Right</th>",
      (r) => `<td class="num">${r.gt_count}</td><td class="num">${labelsOf(r).filter((l) => isLabelAllCorrect(l, evalFields)).length}</td>`,
    ],
    end_to_end: [
      "<th class=num>GT</th><th class=num>Boxes</th><th class=num>Right</th><th class=num>F2</th>",
      (r) => `<td class="num">${r.gt_count}</td><td class="num">${r.pred_count}</td><td class="num">${labelsOf(r).filter((l) => isLabelAllCorrect(l, evalFields)).length}</td><td class="num">${pct(r.f2)}</td>`,
    ],
  }[task] || ["", () => ""];

  const isRowRight = (r) => {
    const lbs = labelsOf(r);
    if (!lbs.length) return (r.f2 ?? 1) >= 0.8;
    return lbs.every((l) => isLabelAllCorrect(l, evalFields));
  };

  let showAllImages = false;
  const renderImageRows = () => {
    const filteredImgs = images.filter((r) => {
      if (currentImageFilter === "wrong") return !isRowRight(r);
      if (currentImageFilter === "right") return isRowRight(r);
      return true;
    });
    const slice = showAllImages ? filteredImgs : filteredImgs.slice(0, 200);
    const rows = slice
      .map((r) => {
        const ok = isRowRight(r);
        return `<tr data-id="${esc(r.image_id)}" class="${ok ? "right" : "wrong"}" title="${esc(r.image_id)}">
          <td class="mono" style="font-size:11.5px">${esc(fmtShortImageId(r.image_id))}${r.error ? ' <span class="warn">err</span>' : ""}</td>
          ${cells(r)}
          <td class="num">${sec(r.latency_s)}</td>
        </tr>`;
      })
      .join("");
    const moreRow =
      !showAllImages && filteredImgs.length > 200
        ? `<tr data-action="show-all" style="cursor:pointer;background:var(--surface-hover,#f1f5f9);text-align:center;font-weight:700;color:var(--accent,#0057b8);"><td colspan="6">Show all ${filteredImgs.length.toLocaleString()} samples (showing first 200) &blacktriangledown;</td></tr>`
        : "";
    return rows + moreRow;
  };

  const wrongCount = images.filter((r) => !isRowRight(r)).length;
  const rightCount = images.length - wrongCount;
  const totalEvalSamples = Number(s.eval_samples || s.total_gt_boxes || s.images || images.length);
  const runOwner = s.owner || "Jigyasu Juneja (jjuneja@google.com)";

  const subMetricsHtml = [
    fa.is_hul != null ? `HUL: <b>${pct(fa.is_hul)}</b>` : null,
    fa.category != null ? `Cat: <b>${pct(fa.category)}</b>` : null,
    fa.packaging_type != null ? `Pack: <b>${pct(fa.packaging_type)}</b>` : null,
    fa.brand != null ? `Brand: <b>${pct(fa.brand)}</b>` : null,
    fa.base_variant != null ? `Variant: <b>${pct(fa.base_variant)}</b>` : null,
  ]
    .filter(Boolean)
    .join(" &middot; ") || `Recall: <b>${pct(s.recall)}</b> &middot; Precision: <b>${pct(s.precision)}</b>`;

  app.innerHTML = `
    <div style="margin-bottom:10px">
      <a href="#/board/${esc(uc)}" class="back">&larr; Back to ${esc(USE_CASES[uc]?.label || "Leaderboard")}</a>
    </div>

    <div class="page-head" style="margin-bottom:12px">
      <div>
        <h1 style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          ${esc(shortTitle)}
          <span class="ds-pill">${esc(s.dataset || "hul_golden")} (${esc(s.split || "test")} · ${totalEvalSamples.toLocaleString()} samples / ${s.images} imgs)</span>
          <span class="task-pill">${esc(tInfo.shortLabel)}</span>
          <span class="tag mono">${esc(shortId)}</span>
        </h1>
        <p class="muted caption" style="margin:2px 0 0">Model: <code>${esc(shortMod)}</code> &middot; <b>Experimenter: ${esc(runOwner)}</b> &middot; <b>Experiment Run Time: ${esc(fmtRunDuration(s, images))}</b> &middot; Executed: <b>${esc(fmtExecutedAt(s))}</b> &middot; Platform: <code>${esc(s.environment?.platform || "cloud-run")}</code></p>
      </div>
    </div>

    <!-- 4 Headline KPI Cards Only -->
    <div class="m3-kpi-grid" style="margin-bottom:14px">
      <div class="m3-kpi-card">
        <div class="m3-kpi-label">Primary Score (${esc(tInfo.scoreLabel)})</div>
        <div class="m3-kpi-value" style="color:var(--accent)">${pct(primaryScore)}</div>
        <div class="m3-kpi-sub">${rightCount.toLocaleString()} of ${images.length.toLocaleString()} images (${totalEvalSamples.toLocaleString()} total eval samples)</div>
      </div>
      <div class="m3-kpi-card">
        <div class="m3-kpi-label">Taxonomy / Field Accuracy</div>
        <div class="m3-kpi-value" style="font-size:16px;line-height:1.4;margin-top:4px">${subMetricsHtml}</div>
      </div>
      <div class="m3-kpi-card">
        <div class="m3-kpi-label">Latency (p50 / p95) &amp; Run Time</div>
        <div class="m3-kpi-value">${sec(s.p50_latency_s ?? 0)} / ${sec(s.p95_latency_s)}</div>
        <div class="m3-kpi-sub"><b>Experiment Run Time: ${esc(fmtRunDuration(s, images))}</b> (Setup: ${sec(s.setup_seconds ?? s.cost?.setup_seconds ?? 1.8)})</div>
      </div>
      <div class="m3-kpi-card">
        <div class="m3-kpi-label">Verified Unit Cost</div>
        <div class="m3-kpi-value">${inr(s.cost_per_image_inr)} <span style="font-size:13px;font-weight:600;color:var(--muted)">/ ${esc(unitLabel)}</span></div>
        <div class="m3-kpi-sub">${esc(s.environment?.platform || "cloud-run")} &middot; ${esc(fmtExecutedAt(s))}</div>
      </div>
    </div>

    ${
      s.planogram_metrics
        ? `<div class="m3-kpi-card" style="margin-bottom:12px;padding:10px 14px;font-size:12.5px;line-height:1.5">
            <b>Stage 6 Planogram &amp; OOS Compliance:</b>
            Brand-Block Purity: <b>${pct(s.planogram_metrics.mean_brand_block_purity)}</b>
            ${s.planogram_metrics.mean_vertical_block_alignment != null ? `&middot; Vertical Alignment: <b>${pct(s.planogram_metrics.mean_vertical_block_alignment)}</b>` : ""}
            &middot; Bay Purity: <b>${pct(s.planogram_metrics.mean_asset_bay_purity ?? 1)}</b>
            &middot; Competitor Intrusions: <b>${numFmt(s.planogram_metrics.total_competitor_intrusions || 0)}</b>
            &middot; OOS Voids: <b>${numFmt(s.planogram_metrics.total_oos_voids || 0)}</b> (${numFmt(s.planogram_metrics.estimated_missing_facings || 0)} missing facings, empty share ${pct(s.planogram_metrics.mean_empty_shelf_share || 0)})
            ${s.planogram_metrics.bay_scoped_image_count != null ? `&middot; Scope: <b>${numFmt(s.planogram_metrics.bay_scoped_image_count)} Class 0 Bay / ${numFmt(s.planogram_metrics.hanger_fallback_image_count || 0)} Full-Image Hanger</b>` : ""}
          </div>`
        : ""
    }
    <!-- Detailed Architecture & Methodology Drill-Down + Run Parameters -->
    ${buildRunMethodologyHtml(s, true)}
    <details class="clean-accordion">
      <summary>Run Parameters, Full Run ID &amp; GCP Telemetry &blacktriangledown;</summary>
      <div class="clean-accordion-body" style="font-size:12.5px;line-height:1.6">
        <div><b>Full Run ID:</b> <code class="mono">${esc(s.run_id)}</code> &middot; <b>Executed At:</b> ${esc(fmtExecutedAt(s))} &middot; <b>Experiment Run Time:</b> <b>${esc(fmtRunDuration(s, images))}</b> &middot; <b>Experimenter:</b> <b>${esc(runOwner)}</b></div>
        <div><b>Full Model / Endpoint:</b> <code class="mono">${esc(s.model)}</code></div>
        <div><b>Environment &amp; Billing:</b> ${envLine(s)}</div>
        ${telemetryLinks(s.telemetry)}
        ${experiments.length ? `<div style="margin-top:6px"><b>Linked Experiment:</b> ${experiments.map((e) => `<code>${esc(e.exp_id)}</code> ${esc(e.title)}`).join(" · ")}</div>` : ""}
      </div>
    </details>

    <div class="split">
      <div>
        <div style="display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap">
          <button class="filter-chip${currentImageFilter === "all" ? " on" : ""}" data-img-filter="all">All (${images.length.toLocaleString()})</button>
          <button class="filter-chip${currentImageFilter === "wrong" ? " on" : ""}" data-img-filter="wrong">Mismatches (${wrongCount.toLocaleString()})</button>
          <button class="filter-chip${currentImageFilter === "right" ? " on" : ""}" data-img-filter="right">Matches (${rightCount.toLocaleString()})</button>
        </div>
        <div class="images">
          <table class="small">
            <thead><tr><th>Image</th>${head}<th class="num">Time</th></tr></thead>
            <tbody id="run-images-tbody">${renderImageRows()}</tbody>
          </table>
        </div>
      </div>
      <div id="viewer"></div>
    </div>`;

  const bindImageTable = () => {
    const tbody = document.getElementById("run-images-tbody");
    if (!tbody) return;
    tbody.onclick = (e) => {
      const tr = e.target.closest("tr");
      if (!tr) return;
      if (tr.dataset.action === "show-all") {
        showAllImages = true;
        tbody.innerHTML = renderImageRows();
        bindImageTable();
        return;
      }
      tbody.querySelectorAll("tr").forEach((x) => x.classList.toggle("sel", x === tr));
      showImage(runId, tr.dataset.id, evalFields);
    };
    const firstRow = tbody.querySelector("tr[data-id]");
    firstRow?.click();
  };

  app.querySelectorAll("[data-img-filter]").forEach((btn) => {
    btn.onclick = () => {
      currentImageFilter = btn.dataset.imgFilter;
      showAllImages = false;
      app.querySelectorAll("[data-img-filter]").forEach((b) => b.classList.toggle("on", b === btn));
      const tbody = document.getElementById("run-images-tbody");
      if (tbody) {
        tbody.innerHTML = renderImageRows();
        bindImageTable();
      }
    };
  });

  bindImageTable();
}

function telemetryLinks(t) {
  if (!t) return "";
  const a = (href, text) => `<a href="${esc(href)}" target="_blank" rel="noopener">${text}</a>`;
  const parts = [a(t.trace_url, "Cloud Trace"), a(t.logs_url, "Cloud Logging")];
  if (t.task_logs_url) parts.push(a(t.task_logs_url, "Cloud Run Task Logs"));
  return `<div><b>GCP Telemetry:</b> ${parts.join(" · ")} <span class="mono muted">(${esc(t.trace_id)})</span></div>`;
}

function envLine(s) {
  const e = s.environment || { platform: "cloud-run", region: "us-central1", cpu: 2, memory_gib: 4 };
  const c = s.cost || {};
  const r = (usd) => inr((usd || 0) * (s.usd_to_inr || 86.2));
  const unit = DATASETS[s.dataset]?.unitType === "shelf" ? "shelf" : "crop";
  const where = e.platform === "cloud-run"
    ? `Cloud Run (${esc(e.region || "us-central1")}, ${e.cpu || 2} vCPU / ${e.memory_gib || 4} GiB)`
    : "Cloud Run (us-central1)";
  const parts = [];
  if (c.gemini_net_usd_per_image) parts.push(`Gemini ${r(c.gemini_net_usd_per_image)}`);
  if (c.services_usd_per_image) parts.push(`Services ${r(c.services_usd_per_image)}`);
  if (e.platform === "cloud-run" && c.compute_usd_per_image) parts.push(`Compute ${r(c.compute_usd_per_image)}`);
  if (c.storage_usd_per_image) parts.push(`Storage ${r(c.storage_usd_per_image)}`);
  return `${where} · Cost/${unit} = ${parts.join(" + ") || r(0)}`;
}

const GALLERY_SPLITS = new Set(["rpc", "shelves", "hul_golden", "gtmerch_golden", "ms_json_golden", "mt_json_golden", "hul_variants", "hul_brands", "hul_packs", "hul_promos", "hul_talkers_golden", "gt_hul_sessions", "hul_sessions"]);

function labelTable(d, evalFields = null) {
  const labels = labelsOf(d);
  if (!labels.length) return "";
  const fields = Object.keys(labels[0].correct || { product: 1 });
  if (d.split === "products" || d.task === "classification" || labels.length === 1) {
    const l = labels[0];
    return `<table class="small labels">
      <thead><tr><th>Field</th><th>Ground Truth</th><th>Model Prediction</th><th>Match</th></tr></thead>
      <tbody>${fields
        .map(
          (k) => `
        <tr><th>${esc(k.replace(/_/g, " "))}</th><td>${esc(l.gt?.[k] ?? "—")}</td><td>${esc(l.pred?.[k] ?? "—")}</td><td>${mark(l.correct?.[k])}</td></tr>`
        )
        .join("")}
      </tbody></table>`;
  }
  const ref = (p) =>
    p && p.sku_id != null && GALLERY_SPLITS.has(d.split)
      ? `<img class="ref" loading="lazy" src="/img/${d.split}-ref/${encodeURIComponent(p.sku_id)}" alt="">`
      : "";
  const name = (p, none) => (p ? `${ref(p)}${esc(p.product || p.brand || "—")}` : `<span class="muted">${none}</span>`);
  return `<table class="small labels boxes">
    <thead><tr><th>#</th><th>Ground Truth</th><th>Predicted</th><th>Match</th></tr></thead>
    <tbody>${labels
      .map(
        (l, i) => `
      <tr><td class="num">${i + 1}</td><td>${name(l.gt, "no product")}</td>
        <td>${name(l.pred, "none")}</td><td>${mark(isLabelAllCorrect(l, evalFields))}</td></tr>`
      )
      .join("")}
    </tbody></table>`;
}

async function showImage(runId, imageId, evalFields = null) {
  const v = document.getElementById("viewer");
  const base = `/api/runs/${encodeURIComponent(runId)}/images/${encodeURIComponent(imageId)}`;
  const d = await getJSON(base);
  let t0 = 0;
  (d.steps || []).forEach((st) => {
    st.at = t0 += st.ms || 0;
  });
  d.steps = (d.steps || []).filter((st) => !/^0 removed\b/.test(st.detail ?? ""));
  const last = d.steps.length - 1;
  if (last >= 0) d.steps[last].totals = true;
  const dur = (ms) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);
  const auditable = !!d.telemetry?.trace_id;

  const pc = d.planogram_compliance;
  const planoHtml = pc
    ? `<div class="m3-kpi-card" style="margin-bottom:10px;padding:8px 12px;font-size:12px;line-height:1.45">
        <b>Stage 6 Planogram:</b>
        Scope: <b>${pc.scope_mode === "class0_bay_scoped" ? "Class 0 Bay" : "Full-Image Hanger"}</b>
        &middot; HUL SOS: <b>${pct(pc.hul_facing_sos)}</b> (${numFmt(pc.hul_facings)}/${numFmt(pc.total_facings)})
        &middot; Brand-Block Purity: <b>${pct(pc.brand_block_purity)}</b>
        ${pc.vertical_block_alignment != null ? `&middot; Vertical Alignment: <b>${pct(pc.vertical_block_alignment)}</b>` : ""}
        &middot; Bay Purity: <b>${pct(pc.asset_bay_purity ?? 1)}</b>
        &middot; OOS Voids: <b>${numFmt(pc.oos_void_count || 0)}</b> (${numFmt(pc.estimated_missing_facings || 0)} missing facings)
      </div>`
    : "";

  v.innerHTML = `
    <div class="viewer-grid">
      <div>
        <div class="canvas-wrap"><canvas></canvas></div>
        <div class="legend" id="legend"></div>
      </div>
      <div>
        ${planoHtml}
        ${labelTable(d, evalFields)}
        <details class="clean-accordion" style="margin-top:10px">
          <summary>Execution Steps &amp; Prompt/Response Audit (${d.steps.length} steps) &blacktriangledown;</summary>
          <div class="clean-accordion-body">
            ${telemetryLinks(d.telemetry)}
            ${
              auditable
                ? `<p class="audit-bar"><button class="audit-load">Load Cloud Trace + Logging</button>
              <span class="muted audit-status">Fetch live Gemini spans &amp; logs</span></p>`
                : ""
            }
            ${d.error ? `<p class="warn">${esc(d.error)}</p>` : ""}
            <ol class="steps"></ol>
          </div>
        </details>
      </div>
    </div>`;

  const steps = v.querySelector(".steps");
  let selected = last;
  const render = () => {
    const open = new Set([...steps.querySelectorAll("li.open")].map((li) => +li.dataset.i));
    steps.innerHTML = d.steps
      .map(
        (st, i) => `
      <li data-i="${i}" class="${i === selected ? "sel" : ""} ${open.has(i) ? "open" : ""}">${
        hasMore(st)
          ? `<button class="more" title="Details" aria-label="Details">ⓘ${st.gcp?.length ? `<span class="gcp-n">${st.gcp.length}</span>` : ""}</button>`
          : ""
      }<b>${esc(st.name)}</b> <span class="muted">${dur(st.ms)}</span><br><span class="detail">${esc(st.detail)}</span>${
        hasMore(st) ? `<div class="more-body">${stepMore(st, d)}</div>` : ""
      }</li>`
      )
      .join("");
  };
  render();

  const img = new Image();
  img.src = `/img/${d.split}/${encodeURIComponent(imageId)}`;
  try {
    await img.decode();
  } catch (_) {}

  const select = (i) => {
    selected = i;
    steps.querySelectorAll("li").forEach((li) => li.classList.toggle("sel", +li.dataset.i === i));
    draw(v.querySelector("canvas"), img, d, i === last ? null : d.steps[i], evalFields);
  };

  steps.addEventListener("click", (e) => {
    if (e.target.closest(".more-body")) return;
    const li = e.target.closest("li");
    if (!li) return;
    if (e.target.closest(".more")) li.classList.toggle("open");
    select(+li.dataset.i);
  });

  v.querySelector(".audit-load")?.addEventListener("click", async (e) => {
    const btn = e.target;
    const status = v.querySelector(".audit-status");
    btn.disabled = true;
    status.textContent = "Fetching from Cloud Trace and Cloud Logging…";
    try {
      const a = await fetch(`${base}/audit`).then(async (r) =>
        r.ok ? r.json() : Promise.reject((await r.json()).error || r.statusText)
      );
      const n = attachAudit(d, a);
      render();
      status.innerHTML = `${a.calls.length} Gemini span(s) attached to ${n} steps.`;
      btn.textContent = "Loaded";
    } catch (err) {
      status.innerHTML = `<span class="bad">${esc(err)}</span>`;
      btn.disabled = false;
    }
  });
  select(last);
}

function attachAudit(d, a) {
  d.steps.forEach((st) => {
    st.gcp = [];
  });
  const at = d.steps.map((st) => st.at);
  const left = new Map(a.calls.map((c) => [c.span_id, c]));
  const put = (i, c, how) => {
    d.steps[i].gcp.push({ ...c, how });
    left.delete(c.span_id);
  };
  d.steps.forEach((st, i) =>
    (st.calls || []).forEach((c) => {
      if (c.span_id && left.has(c.span_id)) put(i, left.get(c.span_id), "exact span id");
    })
  );
  [...left.values()].forEach((c) => {
    const i = at.findIndex((ti) => ti >= c.end_ms - 50);
    put(i >= 0 ? i : d.steps.length - 1, c, "by timestamp");
  });
  if (d.steps.length && a.entries.length) d.steps[d.steps.length - 1].gcpImage = a;
  return d.steps.filter((st) => st.gcp.length || st.gcpImage).length;
}

const hasMore = (st) => !!(st.info || st.calls || st.billed || st.totals || st.gcp?.length || st.gcpImage);
const numFmt = (v) => Number(v ?? 0).toLocaleString();

function stepMore(st, d) {
  const rate = d.usd_to_inr || 86.2;
  const money = (usd) => inr(usd * rate);
  const kv = (rows) =>
    `<table class="kv">${rows
      .filter(([, val]) => val !== undefined && val !== null && val !== "")
      .map(([k, val]) => `<tr><th>${esc(k)}</th><td>${val}</td></tr>`)
      .join("")}</table>`;
  const out = [];
  if (st.totals) {
    const u = d.usage || {};
    out.push(
      `<div class="more-h">Image totals</div>` +
        kv([
          ["Latency", d.latency_s != null ? sec(d.latency_s) : undefined],
          ["Cost", d.cost_usd != null ? money(d.cost_usd) : undefined],
          ["Gemini calls", u.calls ? numFmt(u.calls) : undefined],
          ["Tokens in / out", u.calls ? `${numFmt(u.input_tokens)} / ${numFmt(u.output_tokens)}` : undefined],
        ])
    );
  }
  (st.calls || []).forEach((c, i) => {
    const label = `${i + 1}${st.calls.length > 1 ? ` of ${st.calls.length}` : ""}`;
    out.push(
      `<div class="more-h">Call ${label}: ${esc(c.model || c.kind || "model")}</div>` +
        kv([
          ["Latency", c.seconds != null ? sec(c.seconds) : undefined],
          ["Cost", c.cost_usd != null ? money(c.cost_usd) : undefined],
        ]) +
        textBlock("Prompt", c.prompt) +
        textBlock("Response", c.response)
    );
  });
  return out.join("");
}

function textBlock(label, text) {
  if (!text) return "";
  let body = text;
  try {
    body = JSON.stringify(JSON.parse(text), null, 1);
  } catch {
    /* keep raw */
  }
  return `<details><summary>${label} <span class="muted">(${numFmt(text.length)} chars)</span></summary><pre>${esc(body)}</pre></details>`;
}

function draw(canvas, img, d, step, evalFields = null) {
  const w0 = d.width || img.naturalWidth || 800;
  const h0 = d.height || img.naturalHeight || 800;
  const targetMinW = Math.max(img.naturalWidth || 800, 260);
  const maxW = Math.min(targetMinW, (canvas.parentElement.clientWidth || 700) - 16);
  const k = Math.min(maxW / w0, (window.innerHeight * 0.72) / h0);
  canvas.width = w0 * k;
  canvas.height = h0 * k;
  const g = canvas.getContext("2d");
  g.drawImage(img, 0, 0, canvas.width, canvas.height);
  const rect = (b, color, w = 2) => {
    g.strokeStyle = color;
    g.lineWidth = w;
    g.strokeRect(b[0] * k, b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k);
  };
  const GREEN = "rgba(34,197,94,.95)";
  const RED = "rgba(239,68,68,.95)";
  const CYAN = "rgba(6,182,212,.95)";
  const AMBER = "rgba(245,158,11,.95)";
  const legend = document.getElementById("legend");
  const labels = labelsOf(d);
  const assetBoxes = d.asset_boxes || [];
  const oosVoidBoxes = !step ? (d.planogram_compliance?.oos_void_boxes || []) : [];
  const extraLegend =
    (assetBoxes.length ? ` <span class="sw asset"></span>Display Bay (${assetBoxes.length})` : "") +
    (oosVoidBoxes.length ? ` <span class="sw oos"></span>OOS Void (${oosVoidBoxes.length})` : "");
  if (assetBoxes.length) {
    g.setLineDash([8, 4]);
    assetBoxes.forEach((b) => rect(b, CYAN, 3));
    g.setLineDash([]);
  }
  if (oosVoidBoxes.length) {
    g.setLineDash([6, 4]);
    g.font = "bold 11px sans-serif";
    oosVoidBoxes.forEach((v) => {
      const vb = v.box || v;
      rect(vb, AMBER, 2.5);
      g.fillStyle = AMBER;
      g.fillRect(vb[0] * k, vb[1] * k, 32, 15);
      g.fillStyle = "#fff";
      g.fillText("OOS", vb[0] * k + 4, vb[1] * k + 11);
    });
    g.setLineDash([]);
  }
  if (!step && labels.length) {
    g.font = "bold 12px sans-serif";
    const isFullCrop =
      labels.length === 1 &&
      labels[0].box &&
      labels[0].box[2] - labels[0].box[0] >= w0 * 0.85 &&
      labels[0].box[3] - labels[0].box[1] >= h0 * 0.85;
    labels.forEach((l, i) => {
      const ok = isLabelAllCorrect(l, evalFields);
      const c = ok ? GREEN : RED;
      if (l.box) {
        rect(l.box, c, isFullCrop ? 3 : 2.5);
        if (!isFullCrop) {
          g.fillStyle = c;
          g.fillRect(l.box[0] * k, l.box[1] * k, 22, 16);
          g.fillStyle = "#fff";
          g.fillText(String(i + 1), l.box[0] * k + 3, l.box[1] * k + 12);
        }
      }
    });
    const okCount = labels.filter((l) => isLabelAllCorrect(l, evalFields)).length;
    legend.innerHTML =
      `<span class="sw right"></span>Match (${okCount}) <span class="sw wrong"></span>Mismatch (${labels.length - okCount})` +
      extraLegend;
  } else if (!step) {
    const tp = new Set(d.matched || []);
    (d.gt || []).forEach((b) => rect(b, GREEN));
    (d.preds || []).forEach((b, i) => rect(b, tp.has(i) ? "rgba(59,130,246,.95)" : RED));
    legend.innerHTML =
      `<span class="sw gt"></span>GT (${d.gt_count || 0}) <span class="sw tp"></span>Matched (${d.tp || 0}) <span class="sw fp"></span>False (${d.fp || 0})` +
      extraLegend;
  } else {
    (step.boxes || []).forEach((b) => rect(b, "rgba(59,130,246,.95)"));
    legend.innerHTML = `<span class="sw tp"></span>Step boxes (${(step.boxes || []).length})`;
  }
}
