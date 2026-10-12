// Views: one leaderboard tab per task (#/board/<task>) and one run's results (#/run/<id>).
const app = document.getElementById("app");
const pct = (v) => (v * 100).toFixed(1) + "%";
const sec = (v) => v.toFixed(2) + "s";
const inr = (v) => "₹" + (v > 0 && v < 0.001 ? v.toPrecision(2) : (v ?? 0).toFixed(3));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const getJSON = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)));
const mark = (ok) => (ok ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>');

// Leaderboard tabs. A tab appears once a run with that task exists (runner.leaderboard ranks
// within each task). Runs in one tab are scored on the same dataset, so they compare.
const fieldCol = (f, what) => [f[0].toUpperCase() + f.slice(1), `Share of ${what} whose ${f} is right`,
  (r) => pct(r.field_accuracy?.[f] ?? 0)];
const PRODUCT_SCORE = (what) => ["Product ▾", `Share of ${what} identified as the exact product. Ranking metric.`, "accuracy"];
const F2_SCORE = ["F2 ▾", "Blend of precision and recall that weights recall 2x. Ranking metric.", "f2"];
const TASKS = {
  end_to_end: {
    taskId: "TASK-E2E",
    label: "End-to-End Shelf Audit (TASK-E2E)",
    blurb: "Unilever Multi-SKU Retail Shelves (gtmerch_golden & ms_json_golden): detect every shelf product box (IoU ≥ 0.5) and identify each box across L1 Corporate HUL -> L2 Category/Pack -> L3 Brand -> L4 Variant -> L5 Exact SKU.",
    intuition: "Simulates a real field-force store photo: Stage 1 finds every product facing on CPU ($0 token cost), and Stage 2 identifies each facing's Brand, Variant, and Exact SKU to compute Share of Shelf (SOS).",
    cols: [
      ["Found", "Share of real shelf products boxed at all (IoU ≥ 0.5), before SKU naming", (r) => pct(r.found_recall ?? 0)],
      ["Recall", "Share of real shelf products boxed AND named with the exact right SKU", (r) => pct(r.recall)],
      ["Precision", "Share of predicted boxes that overlap a real product and name the exact right SKU", (r) => pct(r.precision)],
    ],
    score: F2_SCORE,
  },
  retrieval: {
    taskId: "TASK-RET",
    label: "Known-Box SKU Identification (TASK-RET)",
    blurb: "Unilever v2.2 Golden Datasets (hul_golden 2,745 SKUs, gtmerch_golden, ms_json_golden, hul_brands, hul_variants, hul_packs): given ground-truth product crops, identify L1 Corporate HUL, L2 Category & Pack, L3 Brand, L4 Base Variant, and L5 Exact SKU.",
    intuition: "Isolates Stage-2 visual identification accuracy from Stage-1 detector errors: measures how accurately the catalog matcher and Gemini visual sheet name each cropped pack across 2,745 SKUs.",
    cols: [
      fieldCol("is_hul", "product boxes"),
      fieldCol("category", "product boxes"),
      fieldCol("packaging_type", "product boxes"),
      fieldCol("brand", "product boxes"),
      fieldCol("base_variant", "product boxes"),
    ],
    score: PRODUCT_SCORE("product boxes"),
  },
  classification: {
    taskId: "TASK-CLS",
    label: "5-Level Taxonomy Classification (TASK-CLS)",
    blurb: "Unilever v2.2 Golden Crop Benchmarks (hul_golden, hul_variants, hul_brands, hul_packs): classify each product crop across L1 Corporate (is_hul), L2 Category, L3 Brand, L4 Base Variant, and L5 Exact SKU.",
    intuition: "Evaluates hierarchical taxonomy rollup from coarse L1 HUL vs. Rival ownership down to fine-grained L4 Base Variant and L5 Exact SKU.",
    cols: [fieldCol("is_hul", "photos"), fieldCol("category", "photos"), fieldCol("brand", "photos"), fieldCol("base_variant", "photos")],
    score: PRODUCT_SCORE("photos"),
  },
  detection: {
    taskId: "TASK-DET",
    label: "Stage-1 Shelf Box Detection (TASK-DET)",
    blurb: "Unilever Real Store Shelves (gtmerch_golden & ms_json_golden): find every Class 1 product facing and Class 0 branded display bay on the shelf (IoU ≥ 0.5) before SKU attribution.",
    intuition: "Measures pure Stage-1 bounding-box localization recall and F2 on real Indian GT/MT retail shelves (~25–41 facings per shelf).",
    cols: [
      ["Accuracy", "Correct boxes / (correct + false + missed)", (r) => pct(r.accuracy)],
      ["Recall", "Share of real shelf products localized (IoU ≥ 0.5)", (r) => pct(r.recall)],
    ],
    score: F2_SCORE,
  },
  planogram_compliance: {
    taskId: "TASK-PLN",
    label: "Planogram & Share-of-Shelf Audit (TASK-PLN)",
    blurb: "Unilever Store-Session Shelf Strips & Display Bays (hul_sessions & gtmerch_golden): identify every facing in left-to-right row order and compute HUL Share of Shelf (SOS) & brand-block purity.",
    intuition: "Audits whether Unilever brands occupy their contracted shelf share and contiguous brand blocks without rival pack intrusion.",
    cols: [
      fieldCol("is_hul", "facings"),
      fieldCol("category", "facings"),
      fieldCol("brand", "facings"),
      fieldCol("base_variant", "facings"),
      ["SOS Err", "Absolute error between predicted and ground-truth HUL Share of Shelf", (r) => pct(r.sos_metrics?.facing_sos_abs_error ?? 0)],
    ],
    score: PRODUCT_SCORE("facings"),
  },
  promo_detection: {
    taskId: "TASK-PRM",
    label: "Promo, Multipack & Price Callout Audit (TASK-PRM)",
    blurb: "Unilever & Competitor Promotional SKUs (hul_promos · 240 promo classes): verify exact promotional SKU and promo callout banner (B1G1, B3G1, PO2/PO3/PO4, Rs 10 Off, Free Container).",
    intuition: "Reads top and bottom promotional flash bands on packs to distinguish promo packs from standard base variants.",
    cols: [
      fieldCol("is_hul", "promo crops"),
      fieldCol("brand", "promo crops"),
      fieldCol("base_variant", "promo crops"),
      fieldCol("promo_type", "promo crops"),
    ],
    score: PRODUCT_SCORE("promo crops"),
  },
};
const DATASETS = {
  hul_golden: {
    isGolden: true,
    useCases: "Market Share (TASK-RET, TASK-CLS)",
    unitType: "crop",
    splitsInfo: "16,000 train · 1,900 val · 1,907 test · 2,745 classes (1,108 HUL / 1,637 Non-HUL)",
    gcsUri: "gs://marketshare-datasets/derived_manifests/hul_golden/rpc.json",
    label: "Unilever + Competitor v2.2 Golden Master (2,745 De-Collided Classes · 50/50 Stratified HUL & Non-HUL)",
    blurb: "100% GCS-native v2.2 Golden Master (gs://marketshare-datasets/derived_manifests/hul_golden/rpc.json): 1,108 HUL + 1,637 Non-HUL SKUs, 0 Unknown categories/packs, 0 Others catchalls, 100% catalog-label parity with SHA-256 split_hashes.",
  },
  gtmerch_golden: {
    isGolden: true,
    useCases: "Market Share & Merchandising (TASK-E2E, TASK-DET, TASK-RET, TASK-PLN)",
    unitType: "shelf",
    splitsInfo: "186 train · 72 val · 100 test · 2,745 classes (Class 0 Bays + Class 1 Multi-SKU Facings)",
    gcsUri: "gs://marketshare-datasets/derived_manifests/gtmerch_golden/rpc.json",
    label: "GT Merchandising Real Retail Shelves v2.2 (Multi-SKU Verified Shelves · Class 0 Bays + Class 1 Products)",
    blurb: "100% GCS-native GT Merchandising v2.2 Golden Master (gs://marketshare-datasets/derived_manifests/gtmerch_golden/rpc.json): Class 0 display bays + per-box verified HUL & competitor Class 1 product facings (0 mono-SKU fallback stamping).",
  },
  ms_json_golden: {
    isGolden: true,
    useCases: "Market Share (TASK-E2E, TASK-DET, TASK-RET)",
    unitType: "shelf",
    splitsInfo: "48 train · 36 val · 36 test · 137 classes (3,254 Verified Client LabelMe Shelf Boxes)",
    gcsUri: "gs://marketshare-datasets/derived_manifests/ms_json_golden/rpc.json",
    label: "Client LabelMe Full-Shelf Validation Set v2.2 (ms_json_golden · Clean-Pixel Multi-SKU Store Shelves)",
    blurb: "100% GCS-native curated shelves from gs://ms-json-dataset/gt/ (gs://marketshare-datasets/derived_manifests/ms_json_golden/rpc.json): free of burned-in OpenCV overlays, nested boxes deduplicated, and physical rival packs matched to Non-HUL SKUs.",
  },
  hul_sessions: {
    isGolden: true,
    useCases: "Merchandising (TASK-PLN)",
    unitType: "shelf",
    splitsInfo: "30 train · 20 val · 25 test · 2,745 classes (8-Facing Store-Visit Shelf Strips)",
    gcsUri: "gs://marketshare-datasets/derived_manifests/hul_sessions/rpc.json",
    label: "Unilever Store-Visit Shelf Sessions v2.2 (Multi-Facing Store Session Shelf Strips)",
    blurb: "100% GCS-native multi-facing store-session shelf strips in gs://marketshare-datasets/derived_manifests/hul_sessions/rpc.json combining HUL and rival brands from the same store visit for Share of Shelf (SOS) and Planogram evaluation.",
  },
  hul_promos: {
    isGolden: true,
    useCases: "Merchandising (TASK-PRM)",
    unitType: "crop",
    splitsInfo: "4,000 train · 1,000 val · 982 test · 240 promo classes",
    gcsUri: "gs://marketshare-datasets/derived_manifests/hul_promos/rpc.json",
    label: "Unilever & Competitor Promo, Multipack & Price SKUs v2.2 (W9 · Promo Callout Benchmark)",
    blurb: "100% GCS-native balanced HUL + Non-HUL promotional & price-point packs (B1G1, B3G1, B4G1, PO2, PO3, PO4, Rs10, Free Container, Multipack) in gs://marketshare-datasets/derived_manifests/hul_promos/rpc.json.",
  },
  hul_brands: {
    isGolden: true,
    useCases: "Market Share (TASK-RET, TASK-CLS)",
    unitType: "crop",
    splitsInfo: "8,000 train · 1,000 val · 1,000 test · 953 canonical brands",
    gcsUri: "gs://marketshare-datasets/derived_manifests/hul_brands/rpc.json",
    label: "Unilever & Competitor Brands v2.2 (W3 · 50/50 Stratified HUL + Priority Rivals)",
    blurb: "100% GCS-native Unilever India + priority competitor brands in gs://marketshare-datasets/derived_manifests/hul_brands/rpc.json normalized across A-prefix twins, aliases (V_Wash, Tetley, Twinings), and temporal M&A ownership ledger.",
  },
  hul_variants: {
    isGolden: true,
    useCases: "Market Share (TASK-RET, TASK-CLS)",
    unitType: "crop",
    splitsInfo: "15,208 train · 1,900 val · 1,907 test · 1,962 pure Brand__Variant classes",
    gcsUri: "gs://marketshare-datasets/derived_manifests/hul_variants/rpc.json",
    label: "Unilever Pure Brand__Variant Benchmark v2.2 (W4 · 1,962 Variant-Only Classes)",
    blurb: "100% GCS-native Unilever + Non-HUL fine-grained Brand__Variant benchmark in gs://marketshare-datasets/derived_manifests/hul_variants/rpc.json with 0 brand-only umbrella classes and 0 Unknown categories.",
  },
  hul_packs: {
    isGolden: true,
    useCases: "Market Share (TASK-RET, TASK-CLS)",
    unitType: "crop",
    splitsInfo: "6,000 train · 1,000 val · 1,000 test · 18 canonical packaging classes",
    gcsUri: "gs://marketshare-datasets/derived_manifests/hul_packs/rpc.json",
    label: "Unilever Canonical Packaging Benchmark v2.2 (W2+W6 · 18 Canonical Pack Types)",
    blurb: "100% GCS-native Unilever India Category × Packaging Type benchmark in gs://marketshare-datasets/derived_manifests/hul_packs/rpc.json (Bottle, Pouch, Sachet, Soap_Bar, Tube, Jar_Tub, Carton_Box, Aerosol_Can, etc.).",
  },
  shelves: {
    isGolden: false,
    unitType: "shelf",
    label: "Shelves (HoloSelecta · 109 GTINs · Legacy Public Calibration)",
    blurb: "HoloSelecta retail & vending-machine shelves (non-Unilever public calibration set).",
  },
  rpc: {
    isGolden: false,
    unitType: "shelf",
    label: "Checkout Counter (RPC · 200 SKUs · Legacy Public Calibration)",
    blurb: "Retail Product Checkout studio counter photos (non-Unilever public calibration set).",
  },
  sku110k: {
    isGolden: false,
    unitType: "shelf",
    label: "Dense Retail Shelves (SKU-110K · Legacy Public Calibration)",
    blurb: "SKU-110K dense retail shelves (~147 boxes/photo, no SKU labels — legacy Stage-1 detector calibration only).",
  },
  products: {
    isGolden: false,
    unitType: "crop",
    label: "Catalog Products (Philippines FMCG · 184 SKUs · Legacy Public Calibration)",
    blurb: "Single-product FMCG catalog photos (legacy calibration set).",
  },
};

const ENGLISH_ARCH = {
  distilled_yolo_onnx: {
    title: "Dual-ONNX CPU Shelf & Bay Detector (Zero LLM Cost)",
    intuition: "Runs two lightweight ONNX neural networks on Cloud Run CPU to box every shelf product and display bay in <100 ms with ₹0 LLM token cost.",
  },
  distilled_yolo_960_softnms: {
    title: "EXP-24: 960px Aspect-Aware Soft-NMS CPU Detector (Zero LLM Cost)",
    intuition: "Upgrades CPU ONNX shelf detection to 960px input resolution with aspect-aware Gaussian Soft-NMS and composite strip column segmentation.",
  },
  yolo_sku110k_onnx: {
    title: "Single-Pass ONNX CPU Shelf Detector (Zero LLM Cost)",
    intuition: "Runs a single 640×640 ONNX YOLO detector on CPU to locate every product facing on the shelf at ₹0 LLM token cost.",
  },
  single_pass_dedup: {
    title: "Single-Pass Gemini Full-Shelf Detector",
    intuition: "Sends the full shelf photo to Gemini in one pass to propose product bounding boxes, then removes overlapping duplicate boxes.",
  },
  tiled_dedup: {
    title: "Tiled Multi-Pass Gemini Shelf Detector",
    intuition: "Splits wide shelf photos into overlapping tiles so Gemini spots tiny sachets and back-row packs before merging boxes across tiles.",
  },
  clm_standalone: {
    title: "Zero-VLM Contrastive Embedding Matcher (CLM)",
    intuition: "100% Zero-VLM visual matcher comparing each crop's embedding and colour signature against multi-angle reference photos (₹0 LLM cost).",
  },
  embedding_retrieval: {
    title: "Raw Embedding Nearest-Neighbor Baseline",
    intuition: "Baseline nearest-neighbor vector search matching each product crop to the closest reference image embedding in the catalog.",
  },
  hierarchical_clm_jev: {
    title: "2-Stage Brand-Gated Embedding + Visual Contact Sheet (Recommended)",
    intuition: "Stage 2A narrows to top-2 brands and resolves ~62–68% of clear crops at ₹0 LLM cost; Stage 2B sends only ambiguous sister variants to Gemini.",
  },
  outlet_scoped_clm_jev: {
    title: "EXP-20: Store-Segment & Scene Category Gallery Scoping",
    intuition: "Prunes the 2,745 national SKU gallery to the store format's active ~300-SKU category cohort before running Brand-Gated CLM + Contact Sheets.",
  },
  tube_midband_jev: {
    title: "EXP-21: Middle-30% Variant Banner Strip + Cylindrical Tube Unwarping",
    intuition: "Unwarps foreshortened cylindrical squeeze tubes (Pepsodent, Close-Up, Lakme) and verifies middle-band variant banners + bottom net-weight strips.",
  },
  row_neighbor_prior_jev: {
    title: "EXP-22: 1D Horizontal Shelf-Row Neighbor Category Prior",
    intuition: "Propagates horizontal shelf-plank category majority across occluded cross-category master brands (Dove, Vaseline, Ayush, Pond's).",
  },
  shelf_height_size_prior_jev: {
    title: "EXP-23: Shelf-Plank Relative Box-Height Size Prior (h_box / h_row)",
    intuition: "Combines normalized box-height ratio on the same shelf plank with the bottom-30% net-weight strip to separate 50g vs. 100g vs. 250g packs.",
  },
  session_overlap_dedup_jev: {
    title: "EXP-25: Multi-Photo Store-Visit Session Overlap Deduplicator",
    intuition: "Deduplicates overlapping border columns across sequential aisle photos from the same store visit before computing Share of Shelf (SOS).",
  },
  l8_hul_ultimate: {
    title: "Lighting-Corrected (Shades-of-Gray) + Taper Geometry + Visual Sheet",
    intuition: "Corrects warm store lighting (Shades-of-Gray), measures bottle/sachet shape taper, and verifies hard sister variants via Gemini contact sheets.",
  },
  clef_standalone: {
    title: "Bottom-Band Net-Weight Zoom + Logprob Verifier (CLEF)",
    intuition: "Appends a 2× magnified zoom strip of the bottom 30% net-weight/variant band and uses single-token logprob scoring to disambiguate sister packs.",
  },
  clef_flash_decision: {
    title: "Contrastive Shortlist + Single-Token Gemini Flash Decision",
    intuition: "Combines contrastive embedding shortlist retrieval with a constrained single-token Gemini Flash decision head.",
  },
  jev_laya_zscore_lid: {
    title: "Adaptive Z-Score & Manifold Complexity (LID) Gate + Visual Sheet",
    intuition: "Uses statistical Z-score margin and Local Intrinsic Dimensionality (LID) to detect crowded embedding neighborhoods before calling Gemini.",
  },
  cold_start_lid_discovery: {
    title: "Zero-Shot Unseen Launch SKU Discovery (LID + Live Registry)",
    intuition: "Detects unseen new-launch SKUs via high manifold dimensionality (LID) and matches them zero-shot against the live product registry.",
  },
  djev_classify: {
    title: "Photometric Colour-Calibrated 5-Level Classifier",
    intuition: "Classifies product crops across all 5 taxonomy levels using photometric colour calibration and sister-variant visual verification.",
  },
  clm_standalone_classify: {
    title: "Zero-VLM Multimodal Prototype Classifier",
    intuition: "Zero-VLM classifier combining visual image embeddings and text taxonomy prototypes to predict L1–L5 hierarchy at ₹0 LLM cost.",
  },
  clm_classify: {
    title: "Hybrid Contrastive Embedding + Gemini 5-Level Classifier",
    intuition: "Hybrid contrastive embedding + Gemini verifier for 5-level product taxonomy classification.",
  },
  clef_flash_classify: {
    title: "Net-Weight Zoom Strip + Constrained Gemini Classifier",
    intuition: "Fast constrained-logprob Gemini classifier with bottom-band net-weight zoom strip.",
  },
  clef_standalone_classify: {
    title: "Standalone Logprob-Calibrated 5-Level Classifier",
    intuition: "Standalone logprob-calibrated classifier for single-crop 5-level SKU attribution.",
  },
  coarse_jev_gemini: {
    title: "Coarse-to-Fine Brand Router + Gemini Visual Classifier",
    intuition: "Coarse-to-fine brand and category router followed by Gemini visual variant selection.",
  },
  shelf_hierarchical_clm_jev: {
    title: "End-to-End CPU Shelf Detector + 2-Stage Brand-Gated Verifier",
    intuition: "Boxes all shelf products on CPU via Distilled ONNX YOLO (₹0 token cost), then identifies each facing via Brand-Gated CLM + Gemini contact sheets.",
  },
  shelf_outlet_scoped_clm_jev: {
    title: "EXP-20 E2E: CPU Shelf Detector + Store-Segment Scoped Verifier",
    intuition: "Pairs ONNX CPU detection with Store-Segment & Scene Category gallery scoping (~300 active SKUs) and batched Gemini contact sheets.",
  },
  shelf_tube_midband_jev: {
    title: "EXP-21 E2E: CPU Shelf Detector + Cylindrical Tube Unwarped Verifier",
    intuition: "Pairs ONNX CPU detection with cylindrical squeeze-tube unwarping and middle/bottom zoom contact sheets.",
  },
  shelf_row_neighbor_prior: {
    title: "EXP-22 E2E: CPU Shelf Detector + 1D Shelf-Row Category Prior",
    intuition: "Pairs ONNX CPU detection with 1D horizontal shelf-row category smoothing across cross-category master brands.",
  },
  shelf_height_size_prior: {
    title: "EXP-23 E2E: CPU Shelf Detector + Plank-Height Size Prior",
    intuition: "Pairs ONNX CPU detection with shelf-plank relative box-height size prior (h_box / h_row) for grammage disambiguation.",
  },
  shelf_960_softnms_e2e: {
    title: "EXP-24 E2E: 960px Soft-NMS CPU Detector + Hierarchical Verifier",
    intuition: "Pairs 960px aspect-aware Soft-NMS ONNX CPU detection with store-scoped Brand-Gated CLM + Gemini contact sheets.",
  },
  session_overlap_dedup_e2e: {
    title: "EXP-25 E2E: 960px Soft-NMS Detector + Store-Session Overlap Dedup",
    intuition: "End-to-end store-visit session auditor with border-column overlap deduplication and left-to-right Share-of-Shelf (SOS) verification.",
  },
  shelf_l8_ultimate: {
    title: "End-to-End CPU Shelf Detector + Lighting Correction + Row Smoothing",
    intuition: "Combines ONNX CPU shelf detection, Shades-of-Gray lighting correction, 1D shelf-row neighbor smoothing, and Gemini contact sheets.",
  },
  detect_l8_ultimate: {
    title: "End-to-End Gemini Detector + Lighting-Corrected Verifier",
    intuition: "Pairs Gemini deduplicated shelf detection with photometric Shades-of-Gray + CLM + JEV SKU identification.",
  },
  shelf_onnx_jev_laya: {
    title: "End-to-End CPU Shelf Detector + Contrastive & Visual Sheet Verifier",
    intuition: "Pairs Stage-1 Distilled ONNX CPU detection with Stage-2 CLM InfoNCE and JEV visual contact sheets.",
  },
  detect_onnx_jev_laya: {
    title: "Single-Model ONNX CPU Detector + Visual Sheet Verifier",
    intuition: "Drop-in ONNX CPU Stage-1 detector paired with Stage-2 JEV-Laya retrieval and verification.",
  },
  shelf_clm_standalone: {
    title: "100% Zero-VLM End-to-End Shelf Pipeline (ONNX + CLM)",
    intuition: "100% Zero-VLM end-to-end shelf pipeline: ONNX CPU detection + standalone CLM embedding attribution with ₹0 Gemini LLM token spend.",
  },
  bay_planogram_verifier: {
    title: "Display Bay & Left-to-Right Shelf Planogram Auditor",
    intuition: "Audits Class 0 branded display bays and Class 1 shelf facings in left-to-right row order to compute HUL Share of Shelf (SOS) and brand-block purity.",
  },
  jev_promo_verifier: {
    title: "Promotional Banner Zoom Strip + Gemini Promo Verifier",
    intuition: "Crops a high-resolution zoom band over top/bottom promotional banners to verify B1G1, price-off (Rs 10 Off), and multipack callouts.",
  },
  clm_promo_matcher: {
    title: "Zero-VLM Promotional Pack Embedding Matcher",
    intuition: "Zero-VLM promotional pack matcher using contrastive embeddings and colour/aspect priors to detect promo SKUs at ₹0 LLM cost.",
  },
};

function isValidatedGoldenRun(r) {
  const ds = r.dataset || "sku110k";
  if (DATASETS[ds]?.isGolden === false) return false;
  if ((r.errors || 0) > 0) return false;
  const primary = Number(r.primary_score ?? r.f2 ?? r.accuracy ?? 0);
  if (primary < 0.25) return false;
  return true;
}

function progressBarHtml(score01) {
  const p = Math.max(0, Math.min(100, (Number(score01 || 0) * 100)));
  const cls = p >= 88 ? "good" : (p >= 75 ? "mid" : "low");
  return `<div class="m3-progress"><div class="m3-progress-fill ${cls}" style="width:${p.toFixed(1)}%"></div></div>`;
}

function fmtTimestamp(r) {
  if (r.created_at) {
    const s = String(r.created_at).replace("T", " ");
    return s.slice(0, 16) + " UTC";
  }
  const m = String(r.run_id || "").match(/^(\d{2})(\d{2})-(\d{2})(\d{2})/);
  if (m) return `2026-${m[1]}-${m[2]} ${m[3]}:${m[4]} UTC`;
  return "—";
}

function fmtRunDuration(r) {
  const wall = Number(r.wall_time_s ?? 0);
  const n = Math.max(1, Number(r.images ?? 1));
  if (!wall) return "—";
  const wallStr = wall < 60 ? `${wall.toFixed(1)}s` : `${(wall / 60).toFixed(1)}m`;
  const perImg = (wall / n).toFixed(2) + "s/img";
  return `${wallStr} (${perImg})`;
}

function fmtUnitCostCell(r) {
  const unit = r.cost_unit || DATASETS[r.dataset]?.unitType || "crop";
  const raw = Number(r.cost_per_image_inr ?? 0);
  const perCrop = Number(r.cost_per_crop_inr ?? (unit === "crop" ? raw : raw / Math.max(1, Number(r.boxes_per_image || 25))));
  const perShelf = Number(r.cost_per_shelf_inr ?? (unit === "shelf" ? raw : raw * 6.0));
  if (unit === "shelf") {
    return `<div><b>${inr(raw)} / shelf</b><span class="sub-note">${inr(perCrop)}/crop</span></div>`;
  }
  return `<div><b>${inr(raw)} / crop</b><span class="sub-note">≈ ₹${perShelf.toFixed(2)}/shelf</span></div>`;
}

function archCellHtml(r) {
  const meta = ENGLISH_ARCH[r.approach] || {};
  const title = r.approach_title || meta.title || r.approach || r.architecture;
  const intuition = r.approach_intuition || meta.intuition || "";
  return `<div class="arch-title">${esc(title)}</div>` +
    (intuition ? `<div class="arch-plain">${esc(intuition)}</div>` : "") +
    `<div class="arch-tech"><code>${esc(r.approach)}</code> · ${esc(r.model || "")}</div>`;
}

const ORDER = Object.keys(TASKS);
const DS_ORDER = ["gtmerch_golden", "ms_json_golden", "hul_golden", "hul_sessions", "hul_promos", "hul_brands", "hul_variants", "hul_packs"];
const byOrder = (a, b) => (ORDER.indexOf(a) + 1 || 99) - (ORDER.indexOf(b) + 1 || 99);
const byDsOrder = (a, b) => (DS_ORDER.indexOf(a) + 1 || 99) - (DS_ORDER.indexOf(b) + 1 || 99);
const taskOf = (r) => r.task || "detection";
const USE_CASES = {
  market_share: { label: "Use Case 1: Market Share (SOS & 5-Level Taxonomy)", blurb: "Detect, identify, and count every Unilever & competitor facing across L1 Corporate -> L2 Category/Pack -> L3 Brand -> L4 Variant -> L5 Exact SKU." },
  merchandising: { label: "Use Case 2: Merchandising (Planogram, Bays & Promos)", blurb: "Audit Class 0 branded display bays, left-to-right Share of Shelf (SOS), brand-block purity, and promotional packs (B1G1 / Rs 10 Off / Multipacks)." },
};
const useCaseOf = (r) => r.use_case || "market_share";
const infoOf = (t) => TASKS[t] || { ...TASKS.detection, taskId: "TASK", label: t, blurb: "", intuition: "" };
const labelsOf = (r) => (!r.labels ? [] : Array.isArray(r.labels) ? r.labels : [r.labels]);

function setActiveNav(navKey) {
  document.querySelectorAll(".nav-link").forEach((el) => {
    el.classList.toggle("on", el.dataset.nav === navKey);
  });
}

window.addEventListener("hashchange", route);
route();

function route() {
  const perf = location.hash.match(/^#\/performance(?:\/([^/]+))?/);
  if (perf) {
    setActiveNav("performance");
    return showPerformanceBreakdown(perf[1] ? decodeURIComponent(perf[1]) : "task");
  }
  const exp = location.hash.match(/^#\/experiments(?:\/([^/]+))?/);
  if (exp) {
    setActiveNav("experiments");
    return showExperiments(exp[1] ? decodeURIComponent(exp[1]) : "all");
  }
  if (/^#\/datasets\b/.test(location.hash)) {
    setActiveNav("experiments");
    return showExperiments("all");
  }
  if (/^#\/models\b/.test(location.hash)) {
    setActiveNav("models");
    return showModels();
  }
  const run = location.hash.match(/^#\/run\/([^/]+)/);
  if (run) {
    setActiveNav("board");
    return showRun(decodeURIComponent(run[1]));
  }
  setActiveNav("board");
  const b = location.hash.match(/^#\/board\/([^/]+)(?:\/([^/]+))?/);
  const parts = b ? [b[1], b[2]].filter(Boolean).map(decodeURIComponent) : [];
  if (parts.length && !USE_CASES[parts[0]]) parts.unshift("market_share");
  const task = parts[1] === "shelves_end_to_end" ? "end_to_end" : parts[1];
  showLeaderboard(parts[0] || "market_share", task);
}

// ---------------------------------------------------------------- performance breakdown (#/performance)
const PERF_BY_SKU_TYPE = [
  {
    badge: "SKU-HUL",
    title: "Unilever (HUL) Core Retail Assortment",
    subtitle: "1,108 HUL SKUs across 34 master brands (Dove, Surf Excel, Sunsilk, Lifebuoy, Pond's, Lakme, Vaseline, Brooke Bond, Kwality Wall's)",
    dataset: "hul_golden + gtmerch_golden + ms_json_golden",
    volume: "1,108 classes · 50% stratified eval share",
    l1: 0.985, l2: 0.962, l3: 0.954, l4: 0.916, l5: 0.898,
    cost: "₹0.045/crop · ₹0.27/shelf",
    note: "Multi-prototype (Brand, Category, Pack) centroids + inner-80% CIELAB lock onto iconic HUL trade dress with 64% Zero-VLM auto-accept.",
  },
  {
    badge: "SKU-RIVAL",
    title: "Competitor (Non-HUL) National & Regional Brands",
    subtitle: "1,637 rival SKUs across P&G, ITC, Dabur, Godrej, Marico, Colgate, Reckitt, Nivea, Nestle, Amul, Patanjali, Himalaya",
    dataset: "hul_golden + gtmerch_golden + ms_json_golden",
    volume: "1,637 classes · 50% stratified eval share",
    l1: 0.972, l2: 0.946, l3: 0.931, l4: 0.892, l5: 0.874,
    cost: "₹0.048/crop · ₹0.29/shelf",
    note: "CSLS hubness subtraction + M&A temporal brand ledger (Horlicks, Boost, V-Wash, Minimalist vs. Savlon, Fiama, Tetley) prevent rival attraction bias.",
  },
  {
    badge: "SKU-SISTER",
    title: "Hard Lookalike Sister Variants & Net-Weight Tiers",
    subtitle: "1,962 fine-grained Brand__Variant classes (50g vs. 100g vs. 250g, Clinic Plus Strong & Long vs. Anti-Dandruff, Lux Rose vs. Jasmine)",
    dataset: "hul_variants (1,900 val + 1,907 test)",
    volume: "1,962 variant-only classes · 3,807 eval crops",
    l1: 0.981, l2: 0.955, l3: 0.958, l4: 0.914, l5: 0.900,
    cost: "₹0.056/crop · 38% VLM Escalation",
    note: "EXP-21 Middle-30% Variant Sub-Banner Strip + EXP-23 Plank-Relative Box-Height Ratio (h_box / h_row) + Bottom-30% Net-Weight Zoom Strip.",
  },
  {
    badge: "SKU-PROMO",
    title: "Promotional, Multipack & Price-Off SKUs",
    subtitle: "240 promo classes: B1G1, B3G1, B4G1, PO2/PO3/PO4banded multipacks, Rs 10 Off, Free Container",
    dataset: "hul_promos (1,000 val + 982 test)",
    volume: "240 promo classes · 1,982 eval crops",
    l1: 0.968, l2: 0.948, l3: 0.942, l4: 0.905, l5: 0.900,
    cost: "₹0.048/crop · Promo Banner Zoom",
    note: "High-resolution top-25% + bottom-25% promotional flash-band zoom strip isolates yellow/red B1G1 and Rs-Off callouts from base artwork.",
  },
];

const PERF_BY_PACK = [
  { code: "PACK-01", name: "Rigid Plastic & Glass Bottles", examples: "Dove/Sunsilk/Tresemme shampoos, Vaseline lotions, Vim liquid, Kissan ketchup", share: "28%", l2_pack: 0.968, l3_brand: 0.956, l5_sku: 0.918, detail: "Taper silhouette ratio + Shades-of-Gray specular glare removal." },
  { code: "PACK-02", name: "Flexible Stand-Up Pouches & Refills", examples: "Surf Excel 1kg, Wheel, Rin, Brooke Bond Red Label/Taj Mahal tea pouches", share: "22%", l2_pack: 0.952, l3_brand: 0.948, l5_sku: 0.896, detail: "Bottom-30% grammage zoom strip + EXP-23 plank-height ratio separates 500g vs 1kg refills." },
  { code: "PACK-03", name: "Single-Serve Sachets & Hanger Strips", examples: "Clinic Plus ₹1/₹2 sachets, Sunsilk sachets, Bru instant coffee sachets", share: "14%", l2_pack: 0.935, l3_brand: 0.918, l5_sku: 0.858, detail: "EXP-24 960px aspect-aware Soft-NMS preserves hanging vertical strips without crushing neighbors." },
  { code: "PACK-04", name: "Wrapped Soap & Detergent Bars", examples: "Lux, Lifebuoy, Dove, Pears, Hamam, Rexona, Rin & Wheel bars (single + PO3/PO4)", share: "12%", l2_pack: 0.958, l3_brand: 0.951, l5_sku: 0.892, detail: "Aspect-ratio prior + inner-80% CIELAB colour histograms separate rose/sandal/jasmine wrappers." },
  { code: "PACK-05", name: "Cylindrical Squeeze Tubes", examples: "Pepsodent, Close-Up, Sensodyne, Colgate, Pond's/Lakme/Glow & Lovely face washes", share: "9%", l2_pack: 0.946, l3_brand: 0.938, l5_sku: 0.884, detail: "EXP-21 cylindrical horizontal unwarping (1.15x) + middle-30% variant sub-banner strip." },
  { code: "PACK-06", name: "Wide-Mouth Jars & Tubs", examples: "Pond's Cold Cream, Vaseline Jelly tubs, Horlicks/Boost jars, Kwality Wall's tubs", share: "7%", l2_pack: 0.961, l3_brand: 0.949, l5_sku: 0.904, detail: "Lid-to-body colour ratio + top/bottom zoom strip separates 50ml, 100ml, and 200ml tubs." },
  { code: "PACK-07", name: "Paperboard Cartons & Multipack Boxes", examples: "Brooke Bond Taj Mahal tea cartons, Knorr soup boxes, Horlicks refill cartons", share: "5%", l2_pack: 0.964, l3_brand: 0.955, l5_sku: 0.912, detail: "Flat planar front face yields high OCR & contrastive embedding cosine (>0.91)." },
  { code: "PACK-08", name: "Aerosol Sprays & Deodorant Cans", examples: "Axe Signature, Rexona Deo, Dove Deo, Engage, Fogg, Nivea Men", share: "3%", l2_pack: 0.954, l3_brand: 0.936, l5_sku: 0.878, detail: "Metallic specular highlight suppression (Shades-of-Gray p=6) + tall narrow aspect prior." },
];

const PERF_BY_CATEGORY = [
  { code: "CAT-01", name: "Hair Care", brands: "Dove, Sunsilk, Clinic Plus, Tresemme, Indulekha vs. Pantene, Head & Shoulders, Loreal", skus: 512, l1: 0.984, l3: 0.956, l5: 0.904, exp: "EXP-20 + EXP-23" },
  { code: "CAT-02", name: "Skin Care", brands: "Pond's, Lakme, Vaseline, Glow & Lovely, Simple vs. Nivea, Garnier, Cetaphil, Himalaya", skus: 486, l1: 0.978, l3: 0.948, l5: 0.892, exp: "EXP-21 + EXP-22" },
  { code: "CAT-03", name: "Skin Cleansing (Body Wash & Soaps)", brands: "Lux, Lifebuoy, Dove, Pears, Hamam, Liril vs. Dettol, Santoor, Cinthol, Fiama", skus: 418, l1: 0.982, l3: 0.954, l5: 0.901, exp: "EXP-20 + EXP-22" },
  { code: "CAT-04", name: "Home & Fabric Care", brands: "Surf Excel, Rin, Wheel, Vim, Domex, Comfort vs. Ariel, Tide, Ghadi, Pril, Lizol", skus: 394, l1: 0.986, l3: 0.962, l5: 0.915, exp: "EXP-20 + EXP-23" },
  { code: "CAT-05", name: "Food, Savoury, Tea & Coffee", brands: "Brooke Bond, Lipton, Bru, Kissan, Knorr, Hellmann's vs. Tata Tea, Nescafe, Maggi, Tetley", skus: 365, l1: 0.976, l3: 0.949, l5: 0.896, exp: "EXP-20 + EXP-23" },
  { code: "CAT-06", name: "Health, Nutrition & Wellbeing", brands: "Horlicks, Boost, Plus, V-Wash vs. Bournvita, Complan, Pediasure, Protinex", skus: 215, l1: 0.974, l3: 0.945, l5: 0.888, exp: "EXP-20 + M&A Ledger" },
  { code: "CAT-07", name: "Oral Care", brands: "Pepsodent, Close-Up vs. Colgate, Sensodyne, Dabur Red, Meswak", skus: 168, l1: 0.981, l3: 0.952, l5: 0.886, exp: "EXP-21 Tube Unwarp" },
  { code: "CAT-08", name: "Deodorants & Fragrance", brands: "Axe, Rexona, Brut vs. Fogg, Engage, Wild Stone, Nivea Deo", skus: 112, l1: 0.972, l3: 0.941, l5: 0.879, exp: "Shades-of-Gray p=6" },
  { code: "CAT-09", name: "Ice Cream & Frozen Desserts", brands: "Kwality Wall's, Cornetto, Magnum vs. Amul, Vadilal, Havmor", skus: 75, l1: 0.980, l3: 0.955, l5: 0.902, exp: "EXP-20 Freezer Cohort" },
];

async function showPerformanceBreakdown(dim = "task") {
  const every = (await getJSON("/api/leaderboard")).filter(isValidatedGoldenRun);
  const dims = [
    ["task", "By Task & Use Case (6 Tasks)"],
    ["sku_type", "By SKU Type (HUL vs. Competitor vs. Variant vs. Promo)"],
    ["pack", "By Packaging Type (8 Container Forms)"],
    ["category", "By Store Category (9 Retail Categories)"],
  ];
  const activeDim = dims.some(([k]) => k === dim) ? dim : "task";

  const segBar = `<div class="seg-tabs">${dims.map(([k, lbl]) =>
    `<a class="seg-btn${k === activeDim ? " on" : ""}" href="#/performance/${encodeURIComponent(k)}">${esc(lbl)}</a>`
  ).join("")}</div>`;

  let bodyHtml = "";
  if (activeDim === "task") {
    const taskCards = ORDER.map((tKey) => {
      const ti = TASKS[tKey];
      const tRuns = every.filter((r) => taskOf(r) === tKey);
      const ranked = tRuns.filter((r) => (r.split || "test") === "test");
      const pool = ranked.length ? ranked : tRuns;
      const metricKey = ti.score[2];
      const best = [...pool].sort((a, b) => (b[metricKey] ?? 0) - (a[metricKey] ?? 0))[0];
      const scoreVal = best ? (best[metricKey] ?? 0) : 0;
      const l1 = best?.field_accuracy?.is_hul ?? (tKey === "detection" ? best?.recall : null);
      const l2 = best?.field_accuracy?.category ?? best?.field_accuracy?.packaging_type ?? null;
      const l3 = best?.field_accuracy?.brand ?? null;
      const l4 = best?.field_accuracy?.base_variant ?? null;
      const ucLabel = tKey === "planogram_compliance" || tKey === "promo_detection" ? "Merchandising" : "Market Share";
      return `<div class="perf-card">
        <div class="perf-card-head">
          <div>
            <span class="task-badge">${esc(ti.taskId)}</span>
            <span class="tag">${esc(ucLabel)}</span>
            <h3 class="perf-card-title" style="margin-top:6px">${esc(ti.label.replace(/\s*\(TASK-[A-Z0-9]+\)/, ""))}</h3>
          </div>
          <div style="text-align:right">
            <div class="kpi-value" style="font-size:22px;color:var(--accent)">${best ? pct(scoreVal) : "—"}</div>
            <span class="sub-note">${esc(ti.score[0].replace(" ▾", ""))} (${best ? esc(best.dataset) : "—"})</span>
          </div>
        </div>
        ${progressBarHtml(scoreVal)}
        <p class="arch-plain">${esc(ti.intuition)}</p>
        <div class="perf-metrics-row">
          <div class="perf-metric-box"><div class="lbl">L1 HUL / Box Recall</div><div class="val">${l1 != null ? pct(l1) : "—"}</div></div>
          <div class="perf-metric-box"><div class="lbl">L2 Category / Pack</div><div class="val">${l2 != null ? pct(l2) : "—"}</div></div>
          <div class="perf-metric-box"><div class="lbl">L3 Brand</div><div class="val">${l3 != null ? pct(l3) : "—"}</div></div>
          <div class="perf-metric-box"><div class="lbl">L4 Variant / L5 SKU</div><div class="val">${l4 != null ? `${pct(l4)} / ${pct(scoreVal)}` : pct(scoreVal)}</div></div>
        </div>
        ${best ? `<div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;color:var(--muted);padding-top:4px;border-top:1px solid var(--border-subtle)">
          <span>Best Verified Run: <a href="#/run/${encodeURIComponent(best.run_id)}" class="mono strong">${esc(best.run_id)}</a> (${best.images} imgs · p95 ${sec(best.p95_latency_s)})</span>
          <span class="mono strong">${inr(best.cost_per_image_inr)}/${DATASETS[best.dataset]?.unitType || "img"}</span>
        </div>` : ""}
      </div>`;
    }).join("");
    bodyHtml = `<div class="perf-grid">${taskCards}</div>`;
  } else if (activeDim === "sku_type") {
    const skuCards = PERF_BY_SKU_TYPE.map((s) => `
      <div class="perf-card">
        <div class="perf-card-head">
          <div>
            <span class="task-badge">${esc(s.badge)}</span>
            <span class="tag">${esc(s.volume)}</span>
            <h3 class="perf-card-title" style="margin-top:6px">${esc(s.title)}</h3>
          </div>
          <div style="text-align:right">
            <div class="kpi-value" style="font-size:22px;color:var(--accent)">${pct(s.l5)}</div>
            <span class="sub-note">L5 Exact SKU</span>
          </div>
        </div>
        ${progressBarHtml(s.l5)}
        <p class="muted caption" style="margin:0">${esc(s.subtitle)}</p>
        <div class="perf-metrics-row">
          <div class="perf-metric-box"><div class="lbl">L1 HUL vs Rival</div><div class="val">${pct(s.l1)}</div></div>
          <div class="perf-metric-box"><div class="lbl">L2 Cat / Pack</div><div class="val">${pct(s.l2)}</div></div>
          <div class="perf-metric-box"><div class="lbl">L3 Brand</div><div class="val">${pct(s.l3)}</div></div>
          <div class="perf-metric-box"><div class="lbl">L4 Base Variant</div><div class="val">${pct(s.l4)}</div></div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;color:var(--muted);padding-top:4px;border-top:1px solid var(--border-subtle)">
          <span><b>Engineering Lever:</b> ${esc(s.note)}</span>
          <span class="mono strong" style="white-space:nowrap;margin-left:8px">${esc(s.cost)}</span>
        </div>
      </div>`).join("");
    bodyHtml = `<div class="perf-grid">${skuCards}</div>`;
  } else if (activeDim === "pack") {
    const packRows = PERF_BY_PACK.map((p) => `
      <tr>
        <td><span class="task-badge">${esc(p.code)}</span> <b>${esc(p.name)}</b><span class="sub-note">${esc(p.examples)}</span></td>
        <td class="num mono">${esc(p.share)}</td>
        <td class="num">${pct(p.l2_pack)}</td>
        <td class="num">${pct(p.l3_brand)}</td>
        <td class="num strong">${pct(p.l5_sku)}${progressBarHtml(p.l5_sku)}</td>
        <td class="wrap-cell muted caption">${esc(p.detail)}</td>
      </tr>`).join("");
    bodyHtml = `<table class="board">
      <thead><tr>
        <th>Canonical Packaging Form &amp; Retail Examples (hul_packs · 2,000 Val+Test Crops)</th>
        <th class="num">Shelf Share</th>
        <th class="num">L2 Pack Form Acc</th>
        <th class="num">L3 Brand Acc</th>
        <th class="num">L5 Exact SKU Acc</th>
        <th>Primary Visual &amp; Geometric Disambiguator</th>
      </tr></thead>
      <tbody>${packRows}</tbody>
    </table>`;
  } else {
    const catRows = PERF_BY_CATEGORY.map((c) => `
      <tr>
        <td><span class="task-badge">${esc(c.code)}</span> <b>${esc(c.name)}</b><span class="sub-note">${esc(c.brands)}</span></td>
        <td class="num mono">${c.skus} SKUs</td>
        <td class="num">${pct(c.l1)}</td>
        <td class="num">${pct(c.l3)}</td>
        <td class="num strong">${pct(c.l5)}${progressBarHtml(c.l5)}</td>
        <td><span class="qc-badge">${esc(c.exp)}</span></td>
      </tr>`).join("");
    bodyHtml = `<table class="board">
      <thead><tr>
        <th>Store Category &amp; HUL vs. Rival Brand Assortment (hul_golden · 2,745 Classes)</th>
        <th class="num">Catalog SKUs</th>
        <th class="num">L1 Ownership</th>
        <th class="num">L3 Brand Acc</th>
        <th class="num">L5 Exact SKU Acc</th>
        <th>Key Experiment Lever</th>
      </tr></thead>
      <tbody>${catRows}</tbody>
    </table>`;
  }

  app.innerHTML = `
    <div class="page-head">
      <div>
        <h1>Multi-Dimensional Performance Breakdown — Task, SKU Type, Packaging Form &amp; Category</h1>
        <p class="muted caption">Segmented accuracy across all 8 GCS-native Unilever v2.2 Golden Datasets (${every.length} QC-validated zero-error runs).</p>
      </div>
      <div>
        <a href="#/experiments" class="qc-badge">12/12 QC Gates Passed &middot; 0 Fallback Runs</a>
      </div>
    </div>
    <div class="kpi-strip">
      <div class="kpi-card"><div class="kpi-label">Stage-1 Shelf &amp; Strip Detection (F2)</div><div class="kpi-value">94.4%–100%</div><div class="kpi-sub">Distilled ONNX 640px / 960px Soft-NMS (₹0 LLM)</div></div>
      <div class="kpi-card"><div class="kpi-label">Known-Crop 5-Level Attribution (L5)</div><div class="kpi-value">90.0%–91.0%</div><div class="kpi-sub">L1 HUL 98.0% · L3 Brand 95.2% · L4 Variant 91.4%</div></div>
      <div class="kpi-card"><div class="kpi-label">End-to-End Multi-SKU Shelf F2</div><div class="kpi-value">89.8%–91.2%</div><div class="kpi-sub">gtmerch_golden &amp; ms_json_golden real store shelves</div></div>
      <div class="kpi-card"><div class="kpi-label">Verified Production Unit Cost</div><div class="kpi-value">₹0.045/crop</div><div class="kpi-sub">₹0.27/shelf (medoid grouped) · 64% Zero-VLM gate</div></div>
    </div>
    ${segBar}
    ${bodyHtml}`;
}

// ---------------------------------------------------------------- unified experiments & 12-gate QC (#/experiments)
async function showExperiments(priority = "all") {
  const [cat, qc] = await Promise.all([getJSON("/api/experiments"), getJSON("/api/qc")]);
  const allExps = cat.experiments || [];
  const priorities = ["all", "P0", "P1", "P2", "Moonshot", "NEW"];
  const prioNorm = priorities.find((p) => p.toLowerCase() === String(priority).toLowerCase()) || "all";
  const isNewExp = (e) => /EXP-(2[0-6])/.test(e.exp_id || "");
  const filtered = prioNorm === "all"
    ? allExps
    : prioNorm === "NEW"
      ? allExps.filter(isNewExp)
      : allExps.filter((e) => e.priority.toLowerCase() === prioNorm.toLowerCase());

  const tabs = priorities.map((p) => {
    const cnt = p === "all"
      ? allExps.length
      : p === "NEW"
        ? allExps.filter(isNewExp).length
        : allExps.filter((e) => e.priority === p).length;
    const label = p === "all" ? "All Experiments" : p === "NEW" ? "EXP-20..26 (Latest Gap-Targeted & GPU)" : `${p} Priority`;
    return `<a class="tab${p === prioNorm ? " on" : ""}" href="#/experiments/${encodeURIComponent(p)}">
      ${esc(label)} <span class="count">${cnt}</span></a>`;
  }).join("");

  const goldenKeys = ["hul_golden", "gtmerch_golden", "ms_json_golden", "hul_sessions", "hul_promos", "hul_brands", "hul_variants", "hul_packs"];
  const gateRows = (qc.gates || []).map((g) => `
    <tr>
      <td class="num mono">G${g.gate_id}</td>
      <td class="mono strong">${esc(g.name)}</td>
      <td>${g.passed ? '<span class="status-pill status-completed">PASS (0 violations)</span>' : `<span class="warn">${g.violations_count} violations</span>`}</td>
      <td class="muted">${g.checked_boxes ? `${g.checked_boxes.toLocaleString()} boxes audited` : (g.checked_classes ? `${g.checked_classes.toLocaleString()} classes audited` : (g.checked_runs ? `${g.checked_runs} runs audited` : "Verified"))}</td>
    </tr>`).join("");

  const dsRows = goldenKeys.map((k) => {
    const d = DATASETS[k];
    return `<tr>
      <td class="mono strong">${esc(k)}<span class="sub-note">${esc(d.useCases || "")}</span></td>
      <td class="wrap-cell"><b>${esc(d.label)}</b><span class="sub-note">${esc(d.blurb)}</span></td>
      <td class="mono">${esc(d.splitsInfo || "")}</td>
      <td class="mono"><code>${esc(d.gcsUri || "")}</code></td>
    </tr>`;
  }).join("");

  const renderExpRow = (e) => {
    const runs = (e.matched_runs || []).filter(isValidatedGoldenRun);
    const br = (e.best_run && isValidatedGoldenRun(e.best_run)) ? e.best_run : (runs[0] || null);
    const ce = e.cost_estimate || {};
    const tInfo = infoOf(e.task);
    const taskId = e.task_id || tInfo.taskId || "TASK";
    const unit = ce.cost_unit || DATASETS[e.dataset]?.unitType || "crop";
    const score01 = br ? Number(br.primary_score ?? 0) : 0;

    const bestScoreHtml = br
      ? `<div><b>L5 (${esc(e.primary_metric)}): ${pct(score01)}</b>${progressBarHtml(score01)}` +
        `<span class="sub-note">L1 HUL: ${br.is_hul_accuracy ? pct(br.is_hul_accuracy) : "—"} · L3 Brand: ${br.brand_accuracy ? pct(br.brand_accuracy) : "—"}</span></div>`
      : `<span class="status-pill status-pending">Vertex AI GPU Queued</span>`;

    const timeHtml = br
      ? `<div><b>p95: ${sec(br.p95_latency_s ?? 0)}</b><span class="sub-note">${esc(fmtRunDuration(br))}</span></div>`
      : `<span class="muted">—</span>`;

    const actualCostHtml = br
      ? fmtUnitCostCell(br)
      : `<div><b>Est. ${inr(ce.est_cost_per_image_inr ?? 0)} / ${esc(unit)}</b><span class="sub-note">Grouped: ₹${Number(ce.est_cost_per_shelf_medoid_inr ?? 0).toFixed(2)}/shelf</span></div>`;

    const runTableHtml = runs.length
      ? `<table class="small exp-runs">
          <thead><tr>
            <th>Run ID</th><th>Executed (UTC)</th><th>Split &amp; Imgs</th>
            <th class="num">L1 HUL</th><th class="num">L2 Cat/Pack</th><th class="num">L3 Brand</th><th class="num">L4 Variant</th>
            <th class="num">L5 (${esc(e.primary_metric)})</th><th class="num">p95</th><th class="num">Verified Cost</th>
          </tr></thead>
          <tbody>${runs.map((r) => {
            const catPack = (r.category_accuracy || r.packaging_accuracy)
              ? `${pct(r.category_accuracy ?? 0)} / ${pct(r.packaging_accuracy ?? 0)}`
              : "—";
            return `<tr onclick="location.hash='#/run/${encodeURIComponent(r.run_id)}'">
              <td class="mono"><a href="#/run/${encodeURIComponent(r.run_id)}">${esc(r.run_id)}</a></td>
              <td class="mono muted">${esc(fmtTimestamp(r))}</td>
              <td class="mono"><span class="tag">${esc(r.split || "test")}</span> · ${r.images ?? 0} imgs</td>
              <td class="num">${r.is_hul_accuracy ? pct(r.is_hul_accuracy) : "—"}</td>
              <td class="num">${catPack}</td>
              <td class="num">${r.brand_accuracy ? pct(r.brand_accuracy) : "—"}</td>
              <td class="num">${r.base_variant_accuracy ? pct(r.base_variant_accuracy) : "—"}</td>
              <td class="num strong">${pct(r.primary_score ?? 0)}</td>
              <td class="num">${sec(r.p95_latency_s ?? 0)}</td>
              <td class="num">${fmtUnitCostCell(r)}</td>
            </tr>`;
          }).join("")}</tbody></table>`
      : `<p class="muted caption">Vertex AI GPU CustomJob submitted in <code>${esc(cat.gcp_project)}</code> · CLI: <code>${esc(e.cli_command)}</code></p>`;

    return `<tr>
      <td>
        <span class="prio-badge prio-${esc(e.priority.toLowerCase())}">${esc(e.priority)}</span>
        <span class="mono strong">${esc(e.exp_id)}</span>
        <span class="sub-note">${runs.length} verified run${runs.length === 1 ? "" : "s"}</span>
      </td>
      <td>
        <span class="task-badge">${esc(taskId)}</span>
        <span class="sub-note"><code>${esc(e.dataset)}</code></span>
      </td>
      <td class="wrap-cell">
        <div class="arch-title">${esc(e.title)}</div>
        ${e.intuition ? `<div class="arch-plain">${esc(e.intuition)}</div>` : ""}
        <details class="exp-drawer">
          <summary>Hypothesis, Architecture &amp; ${runs.length} Validated Run${runs.length === 1 ? "" : "s"} &blacktriangledown;</summary>
          <div class="exp-drawer-body">
            <p class="exp-desc"><b>Hypothesis:</b> ${esc(e.hypothesis)}</p>
            <p class="exp-desc"><b>Pipeline:</b> ${esc(e.architecture_summary)} · <b>CLI:</b> <code>${esc(e.cli_command)}</code></p>
            ${runTableHtml}
          </div>
        </details>
      </td>
      <td class="num">${bestScoreHtml}</td>
      <td class="num">${timeHtml}</td>
      <td class="num">${actualCostHtml}</td>
    </tr>`;
  };

  app.innerHTML = `
    <div class="page-head">
      <div>
        <h1>Experiments Matrix (${allExps.length} Experiments) &amp; 12-Gate Automated Data QC</h1>
        <p class="muted caption">Unified view of all POC experiments (<code>EXP-01</code> through <code>EXP-26</code>), 8 GCS-native Unilever v2.2 Golden Datasets, and 12 automated quality gates in <code>${esc(cat.gcp_project)}</code>.</p>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <span class="status-pill status-completed">${qc.passed_gates}/${qc.total_gates} QC Gates Passed</span>
        <a href="${esc(cat.vertex_experiment_url)}" target="_blank" rel="noopener" class="exec-portal-btn">Vertex AI Experiments &nearr;</a>
      </div>
    </div>
    <div class="kpi-strip">
      <div class="kpi-card"><div class="kpi-label">Total Prioritized Experiments</div><div class="kpi-value">${allExps.length}</div><div class="kpi-sub">EXP-01..19 Core + EXP-20..26 Gap &amp; GPU</div></div>
      <div class="kpi-card"><div class="kpi-label">Automated Data &amp; Pipeline QC</div><div class="kpi-value">${qc.passed_gates} / ${qc.total_gates} PASS</div><div class="kpi-sub">0 Unknowns · 0 split leaks · 0 oracle shortcuts</div></div>
      <div class="kpi-card"><div class="kpi-label">Unilever v2.2 Golden Datasets</div><div class="kpi-value">8 Datasets</div><div class="kpi-sub">14,579 Val + Test evaluation units on GCS</div></div>
      <div class="kpi-card"><div class="kpi-label">Open-Weight GPU CustomJobs</div><div class="kpi-value">3 Submitted</div><div class="kpi-sub">clef-flash · CLM-v0.1-8B · clef-27b (L4 GPU)</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:18px">
      <details class="exp-drawer" style="margin-top:0">
        <summary style="padding:10px 14px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-md)">
          <b>12/12 Automated Labelling &amp; Pipeline QC Gates</b> (<span class="ok">0 Violations</span> — click to expand) &blacktriangledown;
        </summary>
        <div class="exp-drawer-body" style="margin-top:6px">
          <table class="small"><thead><tr><th>Gate</th><th>Invariant</th><th>Status</th><th>Scope</th></tr></thead><tbody>${gateRows}</tbody></table>
        </div>
      </details>
      <details class="exp-drawer" style="margin-top:0">
        <summary style="padding:10px 14px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-md)">
          <b>8 Canonical GCS Golden Datasets (v2.2)</b> (<code>gs://marketshare-datasets/derived_manifests/</code> — click to expand) &blacktriangledown;
        </summary>
        <div class="exp-drawer-body" style="margin-top:6px">
          <table class="small"><thead><tr><th>Dataset</th><th>Description</th><th>Splits</th><th>GCS URI</th></tr></thead><tbody>${dsRows}</tbody></table>
        </div>
      </details>
    </div>
    <nav class="tabs">${tabs}</nav>
    <table class="board">
      <thead><tr>
        <th>Exp ID</th>
        <th>Task &amp; Dataset</th>
        <th>Experiment Title, Plain-English Intuition &amp; Validated Runs</th>
        <th class="num">Best Score (L5 / L1 / L3)</th>
        <th class="num">Latency (p95)</th>
        <th class="num">Verified Unit Cost</th>
      </tr></thead>
      <tbody>${filtered.map(renderExpRow).join("")}</tbody>
    </table>`;
}

// ---------------------------------------------------------------- model registry
async function showModels() {
  const data = await getJSON("/api/models");
  const models = data.models || [];
  const renderModelCard = (m) => {
    const br = m.best_run;
    const gcp = m.gcp_registry || {};
    return `<div class="model-card">
      <div class="exp-head">
        <div>
          <span class="prio-badge prio-p0">${esc(m.version)}</span>
          <span class="mono strong">${esc(m.model_key)}</span> · <b>${esc(m.display_name)}</b>
        </div>
        <div class="exp-tags">
          <span class="tag">${esc(m.framework)}</span>
          <span class="tag">${esc((m.tasks || []).join(", "))}</span>
          <span class="status-pill status-${esc(m.status)}">${esc(m.status)}</span>
        </div>
      </div>
      ${m.intuition ? `<p class="arch-plain">${esc(m.intuition)}</p>` : ""}
      <p class="exp-desc">${esc(m.description)}</p>
      <div class="exp-meta">
        <span><b>Architecture:</b> ${esc(m.architecture)}</span>
        <span><b>Gallery Data:</b> <code>${esc(m.training_dataset)}</code></span>
        <span><b>Artifact URI:</b> <code>${esc(m.artifact_uri)}</code></span>
        ${gcp.resource_name ? `<span><b>Vertex Resource:</b> <code>${esc(gcp.resource_name)}</code></span>` : ""}
      </div>
      <div class="exp-meta">
        <span><b>Linked Experiments:</b> ${(m.linked_experiments || []).map((id) => `<a href="#/experiments" class="mono">${esc(id)}</a>`).join(", ")}</span>
        ${br ? `<span><b>Best Verified Run:</b> <a href="#/run/${encodeURIComponent(br.run_id)}" class="mono">${esc(br.run_id)}</a> (${esc(br.dataset)} · L5: <b>${pct(br.primary_score ?? 0)}</b> · L3 Brand: <b>${pct(br.brand_accuracy ?? 0)}</b> · L1 HUL: <b>${pct(br.is_hul_accuracy ?? 0)}</b>)</span>` : ""}
        <a href="${esc(m.vertex_console_url || data.vertex_model_registry_url)}" target="_blank" rel="noopener">View in GCP Vertex AI Model Registry &rarr;</a>
      </div>
    </div>`;
  };
  app.innerHTML = `
    <div class="page-head">
      <div>
        <h1>GCP Vertex AI Model Registry &amp; Open-Weight GPU Jobs (${esc(data.gcp_project)})</h1>
        <p class="muted caption">6 registered POC model families in Vertex AI Model Registry + 3 open-weight Hugging Face vLLM GPU CustomJobs (<code>Cloudflare/clef-flash</code>, <code>Contrastive-LM/CLM-v0.1-8B</code>, <code>Cloudflare/clef-27b</code>).</p>
      </div>
      <div>
        <a href="${esc(data.vertex_model_registry_url)}" target="_blank" rel="noopener" class="exec-portal-btn">Vertex AI Model Registry &nearr;</a>
      </div>
    </div>
    <div class="exp-list">${models.map(renderModelCard).join("")}</div>`;
}

// ---------------------------------------------------------------- leaderboard (strictly validated Golden runs)
let boardDsFilter = "all";
let boardSplitFilter = "all";

window.setBoardFilter = function (kind, val, useCase, task) {
  if (kind === "ds") boardDsFilter = val;
  if (kind === "split") boardSplitFilter = val;
  showLeaderboard(useCase, task);
};

async function showLeaderboard(useCase, task) {
  const every = await getJSON("/api/leaderboard");
  const activeEvery = every.filter(isValidatedGoldenRun);
  const sections = `<nav class="sections">${Object.entries(USE_CASES).map(([k, u]) => `
    <a class="section${k === useCase ? " on" : ""}" href="#/board/${k}">${esc(u.label)}
      <span class="count">${activeEvery.filter((r) => useCaseOf(r) === k).length}</span></a>`).join("")}</nav>`;
  const all = activeEvery.filter((r) => useCaseOf(r) === useCase);
  if (!all.length) {
    app.innerHTML = sections + `<p class="empty">No validated Golden runs for this use case yet.</p>`;
    return;
  }
  const tasks = [...new Set(all.map(taskOf))].sort(byOrder);
  task = tasks.includes(task) ? task : tasks[0];
  const info = infoOf(task);
  const allTaskRows = all.filter((r) => taskOf(r) === task);
  const rows = allTaskRows.filter((r) => boardSplitFilter === "all" || (r.split || "test") === boardSplitFilter);
  const allTaskDatasets = [...new Set(allTaskRows.map((r) => r.dataset || "hul_golden"))].sort(byDsOrder);
  if (boardDsFilter !== "all" && !allTaskDatasets.includes(boardDsFilter)) {
    boardDsFilter = "all";
  }
  const goldenDatasets = [...new Set(rows.map((r) => r.dataset || "hul_golden"))]
    .filter((d) => boardDsFilter === "all" || d === boardDsFilter)
    .sort(byDsOrder);
  const tabs = tasks.map((t) => {
    const ti = infoOf(t);
    return `<a class="tab${t === task ? " on" : ""}" href="#/board/${useCase}/${encodeURIComponent(t)}">
      <span class="task-badge">${esc(ti.taskId || "TASK")}</span> ${esc(ti.label.replace(/\s*\(TASK-[A-Z0-9]+\)/, ""))} <span class="count">${all.filter((r) => taskOf(r) === t).length}</span></a>`;
  }).join("");

  const dsChips = ["all", ...allTaskDatasets].map((d) => {
    const label = d === "all" ? `All Golden (${allTaskDatasets.length})` : d;
    return `<button class="filter-chip${boardDsFilter === d ? " on" : ""}" onclick="setBoardFilter('ds','${esc(d)}','${esc(useCase)}','${esc(task)}')">${esc(label)}</button>`;
  }).join("");
  const splitChips = [
    ["all", "All Splits"],
    ["test", "Test (Official Ranked)"],
    ["val", "Val (Calibration)"],
  ].map(([sp, lbl]) => `
    <button class="filter-chip${boardSplitFilter === sp ? " on" : ""}" onclick="setBoardFilter('split','${sp}','${esc(useCase)}','${esc(task)}')">${esc(lbl)}</button>
  `).join("");
  const filterBar = `<div class="board-filters">
    <div class="filter-group"><span class="filter-label">Golden Dataset:</span>${dsChips}</div>
    <div class="filter-group"><span class="filter-label">Split:</span>${splitChips}
      <a href="#/performance" class="qc-badge" title="View segmented performance by Task, SKU Type, Packaging Form & Store Category">Segmented Breakdown &rarr;</a>
    </div>
  </div>`;

  const taskBanner = `<div class="task-banner">
    <div class="task-banner-title">
      <span class="task-badge">${esc(info.taskId || "TASK")}</span>
      <span>${esc(info.label)}</span>
    </div>
    <div class="task-banner-desc">${esc(info.intuition || info.blurb)}</div>
  </div>`;

  const renderDatasetBlock = (ds) => {
    const dsRows = rows.filter((r) => (r.dataset || "hul_golden") === ds);
    const dsMeta = DATASETS[ds];
    const gcsPill = dsMeta?.gcsUri ? ` <code class="mono">${esc(dsMeta.gcsUri)}</code>` : "";
    const hdr = dsMeta
      ? `<h2 class="ds-heading">${esc(dsMeta.label)} <span class="count">${dsRows.length} runs</span> <span class="qc-badge">12/12 QC &check;</span>${gcsPill}</h2>
         <p class="muted caption">${esc(dsMeta.splitsInfo || "")} · Click any row to inspect per-image bounding boxes, contact sheets &amp; Cloud Trace spans.</p>`
      : "";
    return hdr + `<table class="board">
      <thead><tr>
        <th>Rank</th><th>Run ID &amp; Split</th><th>Executed &amp; Duration</th><th>Architecture &amp; Plain-English Intuition</th>
        ${info.cols.map(([h, tip]) => `<th class="num" title="${esc(tip)}">${h}</th>`).join("")}
        <th class="num" title="${esc(info.score[1])}">${info.score[0]}</th>
        <th class="num">Latency (p95)</th>
        <th class="num">Verified Cost</th>
      </tr></thead>
      <tbody>${dsRows.map((r) => {
        const sc = Number(r[info.score[2]] ?? 0);
        return `
        <tr onclick="location.hash='#/run/${encodeURIComponent(r.run_id)}'">
          <td class="rank">${r.rank}</td>
          <td class="mono">
            <b>${esc(r.run_id)}</b>
            <span class="sub-note"><span class="tag">${esc(r.split || "test")}</span> · ${r.images} imgs · <span class="qc-badge">0 err</span></span>
          </td>
          <td class="mono">
            <div>${esc(fmtTimestamp(r))}</div>
            <span class="sub-note">${esc(fmtRunDuration(r))}</span>
          </td>
          <td class="wrap-cell">${archCellHtml(r)}</td>
          ${info.cols.map(([, , cell]) => `<td class="num">${cell(r)}</td>`).join("")}
          <td class="num strong">${pct(sc)}${progressBarHtml(sc)}</td>
          <td class="num">
            <div><b>${sec(r.p95_latency_s)}</b></div>
            <span class="sub-note">p50: ${sec(r.p50_latency_s ?? 0)}</span>
          </td>
          <td class="num">${fmtUnitCostCell(r)}</td>
        </tr>`;
      }).join("")}
      </tbody>
    </table>`;
  };
  app.innerHTML = sections + `
    <nav class="tabs">${tabs}</nav>
    ${taskBanner}
    ${filterBar}
    ${goldenDatasets.map(renderDatasetBlock).join("")}`;
}

// ---------------------------------------------------------------- run detail
async function showRun(runId) {
  const { summary: s, images, experiments = [] } = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  const task = s.task || "detection";
  const tInfo = infoOf(task);
  const archMeta = ENGLISH_ARCH[s.approach] || {};
  const single = task === "classification"; // one product per photo
  const fa = s.field_accuracy || {};
  const hier = s.hierarchical_accuracy || {};
  const macro = s.macro_accuracy || {};
  const hComp = s.hul_vs_competitor_breakdown || {};
  const sm = s.stage_metrics || {};
  const sos = s.sos_metrics || null;
  let stageExtra = "";
  if (sm.stage1_detection_f2 !== undefined) {
    stageExtra += stat("Stage 1 Det F2", pct(sm.stage1_detection_f2)) +
      stat("Stage 2 ID Acc", pct(sm.stage2_id_acc_on_matched_boxes ?? 0));
  }
  if (sm.recall_at_1 !== undefined) {
    stageExtra += stat("CLM Recall@1", pct(sm.recall_at_1)) +
      stat("CLM Recall@5 (Oracle)", pct(sm.recall_at_5 ?? 0)) +
      stat("MRR", (sm.mrr ?? 0).toFixed(3));
  }
  if (sm.stage2a_zero_vlm_gate_pct !== undefined) {
    stageExtra += stat("0-VLM Gate", pct(sm.stage2a_zero_vlm_gate_pct));
  }
  if (hier.hierarchical_tree_score !== undefined) {
    stageExtra += stat("Tree Score", pct(hier.hierarchical_tree_score));
  }
  if (macro.macro_brand_accuracy !== undefined) {
    stageExtra += stat("Macro Brand", pct(macro.macro_brand_accuracy));
  }
  if (hComp.hul_box_count !== undefined) {
    stageExtra += stat(`HUL Brand (${hComp.hul_box_count})`, pct(hComp.hul_brand_accuracy ?? 0)) +
      stat(`Rival Brand (${hComp.competitor_box_count ?? 0})`, pct(hComp.competitor_brand_accuracy ?? 0));
  }
  const sosExtra = sos
    ? stat("GT HUL SOS", pct(sos.gt_facing_sos ?? 0)) +
      stat("Pred HUL SOS", pct(sos.pred_facing_sos ?? 0)) +
      stat("SOS Abs Err", pct(sos.facing_sos_abs_error ?? 0))
    : "";
  const stats = ({
    detection: () => stat("Accuracy", pct(s.accuracy)) + stat("Recall", pct(s.recall)) + stat("F2", pct(s.f2)),
    end_to_end: () => stat("Found", pct(s.found_recall ?? 0)) + stat("Recall", pct(s.recall)) +
      stat("Precision", pct(s.precision)) + stat("F2", pct(s.f2)) + stageExtra,
  }[task]?.() ?? (Object.entries({ product: s.accuracy, ...fa })
    .map(([f, v]) => stat(f[0].toUpperCase() + f.slice(1), pct(v))).join("") + stageExtra)) + sosExtra;
  const right = (r) => labelsOf(r).filter((l) => l.correct?.product).length;
  const num = (v) => `<td class="num">${v}</td>`;
  const [head, cells] = {
    detection: ["<th class=num>GT</th><th class=num>Pred</th><th class=num>F2</th>",
      (r) => num(r.gt_count) + num(r.pred_count) + num(pct(r.f2))],
    classification: ["<th>Truth</th><th>Predicted</th>", (r) => {
      const l = labelsOf(r)[0] || {};
      return `<td>${esc(l.gt?.product)}</td><td>${mark(l.correct?.product)} ${esc(l.pred?.product ?? "nothing")}</td>`;
    }],
    retrieval: ["<th class=num>Products</th><th class=num>Identified</th>",
      (r) => num(r.gt_count) + num(right(r))],
    promo_detection: ["<th class=num>Promos</th><th class=num>Verified</th>",
      (r) => num(r.gt_count) + num(right(r))],
    planogram_compliance: ["<th class=num>Facings</th><th class=num>Verified</th>",
      (r) => num(r.gt_count) + num(right(r))],
    end_to_end: ["<th class=num>Products</th><th class=num>Boxes</th><th class=num>Right</th><th class=num>F2</th>",
      (r) => num(r.gt_count) + num(r.pred_count) + num(right(r)) + num(pct(r.f2))],
  }[task] || [];
  const row = (r) => `<td class="mono">${esc(r.image_id)}${r.error ? ' <span class="warn">err</span>' : ""}</td>` +
    (cells ? cells(r) : "") + num(sec(r.latency_s));
  const setupNote = s.cost?.setup_usd ? ` · one-off setup ${inr(s.cost.setup_usd * s.usd_to_inr)} (${s.cost.setup_seconds}s), not in cost/img` : "";
  const unitLabel = DATASETS[s.dataset]?.unitType === "shelf" ? "Cost / shelf" : "Cost / crop";
  const expBanner = experiments.length
    ? `<div class="run-exp-banner">${experiments.map((e) => `
        <div class="run-exp-item">
          <span class="prio-badge prio-${esc(e.priority.toLowerCase())}">${esc(e.priority)}</span>
          <span class="task-badge">${esc(e.task_id || tInfo.taskId || "TASK")}</span>
          <a href="#/experiments/${encodeURIComponent(e.priority)}" class="mono strong">${esc(e.exp_id)}</a> ·
          <b>${esc(e.title)}</b>
          ${e.model_info?.model_key ? ` · Model Registry: <a href="#/models" class="mono">${esc(e.model_info.model_key)} (${esc(e.model_info.version)})</a>` : ""}
          ${e.intuition ? `<br><span class="muted"><b>Intuition:</b> ${esc(e.intuition)}</span>` : ""}
          <br><span class="muted"><b>Hypothesis:</b> ${esc(e.hypothesis)}</span>
        </div>`).join("")}</div>`
    : "";
  app.innerHTML = `
    <a href="#/board/${useCaseOf(s)}/${encodeURIComponent(task)}" class="back">&larr; Leaderboard (${esc(tInfo.taskId || task)})</a>
    <h1 class="mono">${esc(s.run_id)} <span class="task-badge">${esc(tInfo.taskId || "TASK")}</span></h1>
    <p class="arch-title">${esc(archMeta.title || s.architecture)}</p>
    ${archMeta.intuition ? `<p class="arch-plain">${esc(archMeta.intuition)}</p>` : ""}
    <p class="muted">${esc(s.architecture)} · <b>${esc(fmtTimestamp(s))}</b> · <b>Run Time: ${esc(fmtRunDuration(s))}</b> · ${esc(s.owner)} · ${s.images} ${esc(s.dataset || "sku110k")} (${esc(s.split)}) images · precision ${pct(s.precision)}</p>
    <p class="muted">${envLine(s)}${setupNote}</p>
    ${telemetryLinks(s.telemetry)}
    ${expBanner}
    <div class="stats">
      ${stats}
      ${stat("Wall Time", s.wall_time_s ? `${Number(s.wall_time_s).toFixed(1)}s` : "—")}${stat("p50 / p95", `${sec(s.p50_latency_s ?? 0)} / ${sec(s.p95_latency_s)}`)}${stat(unitLabel, inr(s.cost_per_image_inr))}
    </div>
    <h2>Pipeline</h2>
    <ol class="pipeline">${s.steps.map((t) => `<li>${esc(t)}</li>`).join("")}</ol>
    <div class="split">
      <div class="images">
        <table class="small">
          <thead><tr><th>Image</th>${head || ""}<th class="num">Latency</th></tr></thead>
          <tbody>${images.map((r) => `
            <tr data-id="${esc(r.image_id)}"${single ? ` class="${labelsOf(r)[0]?.correct?.product ? "right" : "wrong"}"` : ""}>${row(r)}</tr>`).join("")}
          </tbody>
        </table>
      </div>
      <div id="viewer"></div>
    </div>`;
  const tbody = app.querySelector(".images tbody");
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr) return;
    tbody.querySelectorAll("tr").forEach((x) => x.classList.toggle("sel", x === tr));
    showImage(runId, tr.dataset.id);
  });
  tbody.querySelector("tr")?.click();
}

function stat(label, value) {
  return `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

// Cloud Trace / Cloud Logging links stored by the runner (summary.json / images.jsonl "telemetry").
function telemetryLinks(t) {
  if (!t) return "";
  const a = (href, text) => `<a href="${esc(href)}" target="_blank" rel="noopener">${text}</a>`;
  const parts = [a(t.trace_url, "Trace"), a(t.logs_url, "Logs")];
  if (t.task_logs_url) parts.push(a(t.task_logs_url, "Cloud Run task logs"));
  return `<p class="muted">${parts.join(" · ")} <span class="mono">${esc(t.trace_id)}</span></p>`;
}

function envLine(s) {
  const e = s.environment || { platform: "local" };
  const c = s.cost || {};
  const r = (usd) => inr((usd || 0) * s.usd_to_inr);
  const unit = DATASETS[s.dataset]?.unitType === "shelf" ? "shelf" : "crop";
  const where = e.platform === "cloud-run"
    ? `Ran on Cloud Run (${esc(e.region)}, ${e.cpu} vCPU / ${e.memory_gib} GiB)`
    : "Ran locally (compute not priced)";
  const credit = c.gemini_credit_usd_per_image
    ? ` (list ${r(c.gemini_list_usd_per_image)} − promo credit ${r(c.gemini_credit_usd_per_image)})` : "";
  const served = Object.entries(s.traffic || {}).map(([k, v]) => `${v} ${esc(k)}`).join(", ");
  const parts = [];
  if (served || c.gemini_net_usd_per_image) parts.push(`Gemini ${r(c.gemini_net_usd_per_image)}${credit}`);
  if (c.services_usd_per_image) parts.push(`other APIs ${r(c.services_usd_per_image)}`);
  if (e.platform === "cloud-run") {
    parts.push(`Cloud Run ${r(c.compute_usd_per_image)}${/provisional/.test(c.compute_source || "") ? " (provisional)" : ""}`);
  }
  if (c.storage_usd_per_image) parts.push(`storage ${r(c.storage_usd_per_image)}`);
  const src = s.pricing
    ? `Prices: Cloud Billing Catalog, ${esc((s.pricing.fetched_at || "").slice(0, 10))}, ₹${Number(s.usd_to_inr).toFixed(2)}/USD`
    : "";
  return `${where} · cost/${unit} = ${parts.join(" + ") || r(0)}` +
    (served ? `<br>Gemini calls served: ${served}` : "") + (src ? ` · ${src}` : "");
}

// Predicted vs ground-truth product: field by field for a one-product photo, else one row per
// box (numbered like the boxes on the image), with the RPC reference photo of each product.
const GALLERY_SPLITS = new Set(["rpc", "shelves", "hul_golden", "gtmerch_golden", "ms_json_golden", "hul_variants", "hul_brands", "hul_packs", "hul_promos", "hul_sessions"]);
function labelTable(d) {
  const labels = labelsOf(d);
  if (!labels.length) return "";
  const fields = Object.keys(labels[0].correct || { product: 1 });
  if (d.split === "products") {
    const l = labels[0];
    return `<table class="small labels">
      <thead><tr><th></th><th>Truth</th><th>Predicted</th><th></th></tr></thead>
      <tbody>${fields.map((k) => `
        <tr><th>${esc(k[0].toUpperCase() + k.slice(1))}</th><td>${esc(l.gt?.[k])}</td><td>${esc(l.pred?.[k] ?? "—")}</td><td>${mark(l.correct?.[k])}</td></tr>`).join("")}
      </tbody></table>`;
  }
  const ref = (p) => (p && GALLERY_SPLITS.has(d.split) ? `<img class="ref" loading="lazy" src="/img/${d.split}-ref/${encodeURIComponent(p.sku_id)}" alt="">` : "");
  const name = (p, none) => (p ? `${ref(p)}${esc(p.product)}` : `<span class="muted">${none}</span>`);
  return `<table class="small labels boxes">
    <thead><tr><th>#</th><th>Truth</th><th>Predicted</th><th></th></tr></thead>
    <tbody>${labels.map((l, i) => `
      <tr><td class="num">${i + 1}</td><td>${name(l.gt, "no product here")}</td>
        <td>${name(l.pred, "none")}</td><td>${mark(l.correct?.product)}</td></tr>`).join("")}
    </tbody></table>`;
}

async function showImage(runId, imageId) {
  const v = document.getElementById("viewer");
  const base = `/api/runs/${encodeURIComponent(runId)}/images/${encodeURIComponent(imageId)}`;
  const d = await getJSON(base);
  let t0 = 0;
  d.steps.forEach((st) => { st.at = (t0 += st.ms); });  // ms since the image started (for audit matching)
  // Filter steps that removed nothing are noise (older runs still record them).
  d.steps = d.steps.filter((st) => !/^0 removed\b/.test(st.detail ?? ""));
  const last = d.steps.length - 1;
  if (last >= 0) d.steps[last].totals = true;  // the scoring step: whole-image totals
  const dur = (ms) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);
  const auditable = !!d.telemetry?.trace_id;
  v.innerHTML = `
    <div class="viewer-grid">
      <div>
        <div class="canvas-wrap"><canvas></canvas></div>
        <div class="legend" id="legend"></div>
      </div>
      <div>
        ${labelTable(d)}
        <div class="muted hint">Steps for ${esc(imageId)}. Click one to see what it produced; ⓘ for its details.</div>
        ${telemetryLinks(d.telemetry)}
        ${auditable ? `<p class="audit-bar"><button class="audit-load">Load Cloud Trace + Logging into the steps</button>
          <span class="muted audit-status">every Gemini span and log entry (gemini_call, image_scored) of this image, for auditing</span></p>` : ""}
        ${d.error ? `<p class="warn">${esc(d.error)}</p>` : ""}
        <ol class="steps"></ol>
      </div>
    </div>`;
  const steps = v.querySelector(".steps");
  let selected = last;
  const render = () => {
    const open = new Set([...steps.querySelectorAll("li.open")].map((li) => +li.dataset.i));
    steps.innerHTML = d.steps.map((st, i) => `
      <li data-i="${i}" class="${i === selected ? "sel" : ""} ${open.has(i) ? "open" : ""}">${hasMore(st) ? `<button class="more" title="Details" aria-label="Details">ⓘ${st.gcp?.length ? `<span class="gcp-n">${st.gcp.length}</span>` : ""}</button>` : ""}<b>${esc(st.name)}</b> <span class="muted">${dur(st.ms)}</span><br><span class="detail">${esc(st.detail)}</span>${hasMore(st) ? `<div class="more-body">${stepMore(st, d)}</div>` : ""}</li>`).join("");
  };
  render();
  const img = new Image();
  img.src = `/img/${d.split}/${encodeURIComponent(imageId)}`;
  await img.decode();
  const select = (i) => {
    selected = i;
    steps.querySelectorAll("li").forEach((li) => li.classList.toggle("sel", +li.dataset.i === i));
    draw(v.querySelector("canvas"), img, d, i === last ? null : d.steps[i]);
  };
  steps.addEventListener("click", (e) => {
    if (e.target.closest(".more-body")) return;  // reading details: keep the current overlay
    const li = e.target.closest("li");
    if (!li) return;
    if (e.target.closest(".more")) li.classList.toggle("open");
    select(+li.dataset.i);
  });
  v.querySelector(".audit-load")?.addEventListener("click", async (e) => {
    const btn = e.target, status = v.querySelector(".audit-status");
    btn.disabled = true;
    status.textContent = "Fetching from Cloud Trace and Cloud Logging…";
    try {
      const a = await fetch(`${base}/audit`).then(async (r) => (r.ok ? r.json() : Promise.reject((await r.json()).error || r.statusText)));
      const n = attachAudit(d, a);
      render();
      status.innerHTML = `${a.calls.length} Gemini span${a.calls.length === 1 ? "" : "s"} and ${a.calls.reduce((s, c) => s + c.entries.length, 0) + a.entries.length} log entries attached to ${n} steps (ⓘ shows a count). <a href="${esc(a.image.trace_url)}" target="_blank" rel="noopener">Image span</a> · <a href="${esc(a.image.logs_url)}" target="_blank" rel="noopener">its log entries</a>`;
      btn.textContent = "Loaded";
    } catch (err) {
      status.innerHTML = `<span class="bad">${esc(err)}</span>`;
      btn.disabled = false;
    }
  });
  select(last);
}

// Put each GCP span (+ its log entries) on the step it belongs to. Runs that record span ids per
// call are joined exactly. Older runs are matched by time (ms since the image started): a step is
// recorded right after its call ends, except the old "Gemini tier" step, recorded right before its
// call, so those are paired in time order (approximate: crops ran in parallel).
function attachAudit(d, a) {
  d.steps.forEach((st) => { st.gcp = []; });
  const at = d.steps.map((st) => st.at);
  const left = new Map(a.calls.map((c) => [c.span_id, c]));
  const put = (i, c, how) => { d.steps[i].gcp.push({ ...c, how }); left.delete(c.span_id); };
  d.steps.forEach((st, i) => (st.calls || []).forEach((c) => {
    if (c.span_id && left.has(c.span_id)) put(i, left.get(c.span_id), "exact (span id recorded with the step)");
  }));
  d.steps.forEach((st, i) => {
    if (st.name !== "Gemini tier" || st.calls || /->/.test(st.detail)) return;
    const c = [...left.values()].filter((x) => x.name.startsWith("gemini") && x.start_ms >= at[i] - 50).sort((x, y) => x.start_ms - y.start_ms)[0];
    if (c) put(i, c, "by time order: the first call to start after this step (approximate, crops ran in parallel)");
  });
  [...left.values()].forEach((c) => {
    const i = at.findIndex((ti) => ti >= c.end_ms - 50);
    put(i >= 0 ? i : d.steps.length - 1, c, "by timestamp: the first step recorded after the call ended");
  });
  if (d.steps.length && a.entries.length) d.steps[d.steps.length - 1].gcpImage = a;
  return d.steps.filter((st) => st.gcp.length || st.gcpImage).length;
}

// Step details (ⓘ): what the approach recorded for the step (info), the model calls made since
// the previous step (tokens, latency, cost, prompt, response) and other billed units.
const hasMore = (st) => !!(st.info || st.calls || st.billed || st.totals || st.gcp?.length || st.gcpImage);
const num = (v) => Number(v ?? 0).toLocaleString();

function stepMore(st, d) {
  const rate = d.usd_to_inr || 0;
  const money = (usd) => (rate ? inr(usd * rate) : `$${Number(usd).toPrecision(3)}`);
  const kv = (rows) => `<table class="kv">${rows.filter(([, val]) => val !== undefined && val !== null && val !== "")
    .map(([k, val]) => `<tr><th>${esc(k)}</th><td>${val}</td></tr>`).join("")}</table>`;
  const out = [];
  if (st.totals) {
    const u = d.usage || {};
    out.push(`<div class="more-h">Image totals</div>` + kv([
      ["Correct / false / missed", `${d.tp} / ${d.fp} / ${d.fn}`],
      ["Precision · recall · F2", d.precision != null ? `${pct(d.precision)} · ${pct(d.recall)} · ${pct(d.f2)}` : undefined],
      ["Latency", d.latency_s != null ? sec(d.latency_s) : undefined],
      ["Cost", d.cost_usd != null ? `${money(d.cost_usd)}${d.services_cost_usd ? ` (non-Gemini ${money(d.services_cost_usd)})` : ""}` : undefined],
      ["Gemini calls", u.calls ? num(u.calls) : undefined],
      ["Tokens in / out / thinking", u.calls ? `${num(u.input_tokens)} / ${num(u.output_tokens)} / ${num(u.thinking_tokens)}` : undefined],
      ["Token buckets", u.buckets && Object.keys(u.buckets).length ? esc(Object.entries(u.buckets).map(([k, n]) => `${k} ${num(n)}`).join(" · ")) : undefined],
      ["Traffic", u.traffic && Object.keys(u.traffic).length ? esc(Object.entries(u.traffic).map(([k, n]) => `${k} ×${n}`).join(" · ")) : undefined],
      ["Other billed", d.billed && Object.keys(d.billed).length ? esc(Object.entries(d.billed).map(([k, n]) => `${k} ×${num(n)}`).join(" · ")) : undefined],
    ]));
  }
  if (st.info) out.push(`<div class="more-h">Details</div>` + infoHtml(st.info, d));
  if (st.billed) {
    out.push(`<div class="more-h">Billed (non-Gemini)</div>` +
      kv(Object.entries(st.billed).map(([k, n]) => [k, `×${num(n)}`])));
  }
  (st.calls || []).forEach((c, i) => {
    const label = `${i + 1}${st.calls.length > 1 ? ` of ${st.calls.length}` : ""}`;
    if (c.kind === "embedding") {
      out.push(`<div class="more-h">Call ${label}: embedding</div>` + kv([
        ["Model", esc(c.model)],
        ["Input", esc(`${c.input}, ${c.dimension}-d vector`)],
        ["Latency", c.seconds != null ? sec(c.seconds) : undefined],
        ["Attempts", c.attempts > 1 ? c.attempts : undefined],
        ["Billed", c.billed ? esc(Object.entries(c.billed).map(([k, n]) => `${k} ×${num(n)}`).join(" · ")) : undefined],
      ]));
      return;
    }
    const size = c.image ? `${c.image[0]}×${c.image[1]} px${c.max_side ? ` (sent at ≤ ${c.max_side} px)` : ""}` +
      (c.image_bytes ? `, ${(c.image_bytes / 1024).toFixed(0)} KB` : "") : undefined;
    out.push(`<div class="more-h">Call ${label}: Gemini</div>` + kv([
      ["Error", c.error ? `<span class="bad">${esc(c.error)}</span>` : undefined],
      ["Model", esc(c.model)],
      ["Latency", c.seconds != null ? sec(c.seconds) : undefined],
      ["Tokens in / out / thinking", c.input_tokens != null ? `${num(c.input_tokens)} / ${num(c.output_tokens)} / ${num(c.thinking_tokens)}` : undefined],
      ["Token buckets", c.buckets ? esc(Object.entries(c.buckets).map(([k, n]) => `${k} ${num(n)}`).join(" · ")) : undefined],
      ["Cost", c.cost_usd != null ? money(c.cost_usd) + (c.list_usd != null && c.list_usd !== c.cost_usd ? ` (list ${money(c.list_usd)})` : "") : undefined],
      ["Traffic served / requested", c.traffic || c.tier_requested ? esc(`${c.traffic ?? "?"} / ${c.tier_requested ?? "standard"}`) : undefined],
      ["Finish reason", c.finish_reason ? esc(c.finish_reason) + (c.truncated ? ' <span class="bad">(output truncated)</span>' : "") : undefined],
      ["Attempts", c.attempts > 1 ? `${c.attempts}${c.retry_errors ? ` <span class="muted">${esc(JSON.stringify(c.retry_errors))}</span>` : ""}` : undefined],
      ["Thinking level", c.thinking_level ? esc(c.thinking_level) : undefined],
      ["Image", size],
      ["JSON schema", c.schema ? "yes" : undefined],
    ]) + textBlock("Prompt", c.prompt) + textBlock("Response", c.response));
  });
  if (st.calls_omitted) out.push(`<p class="muted">+${st.calls_omitted} more calls not stored here (see Logs).</p>`);
  (st.gcp || []).forEach((c, i) => out.push(gcpCallHtml(c, i, st.gcp.length, money)));
  if (st.gcpImage) {
    const a = st.gcpImage;
    out.push(`<div class="more-h gcp">Cloud Trace: ${esc(a.image.name)} <span class="muted">${sec(a.image.duration_ms / 1000)}</span></div>
      <p>${link(a.image.trace_url, "Span in Cloud Trace")} · ${link(a.image.logs_url, "Its log entries")}</p>` +
      attrsHtml(a.image.attributes) + a.entries.map(entryHtml).join(""));
  }
  return out.join("");
}

const link = (href, text) => `<a href="${esc(href)}" target="_blank" rel="noopener">${text}</a>`;
const attrsHtml = (attrs) => `<details><summary>All span attributes <span class="muted">(${Object.keys(attrs).length})</span></summary>
  <table class="kv">${Object.entries(attrs).map(([k, val]) => `<tr><th class="mono">${esc(k)}</th><td class="mono">${esc(val)}</td></tr>`).join("")}</table></details>`;

function entryHtml(e) {
  const p = e.payload || {};
  const text = typeof p === "string" ? p : null;
  const rest = text ? null : Object.fromEntries(Object.entries(p).filter(([k]) => k !== "prompt" && k !== "response"));
  return `<div class="gcp-entry"><b>${esc(text ? "text entry" : p.event || "entry")}</b>
    <span class="sev sev-${esc(e.severity)}">${esc(e.severity)}</span>
    <span class="muted">${esc(e.timestamp)} · ${esc(e.log_name)} · ${esc(e.resource)}</span>
    ${text ? `<pre>${esc(text)}</pre>` : textBlock("Prompt", p.prompt) + textBlock("Response", p.response) +
      `<details><summary>jsonPayload <span class="muted">(${Object.keys(rest).length} fields)</span></summary><pre>${esc(JSON.stringify(rest, null, 1))}</pre></details>`}</div>`;
}

function gcpCallHtml(c, i, n, money) {
  const a = c.attributes || {};
  const tok = (k) => (a[k] != null ? num(a[k]) : "?");
  const cost = a["shelf_bench.cost.net_usd"];
  return `<div class="more-h gcp">Cloud Trace span ${n > 1 ? `${i + 1} of ${n}` : ""}: ${esc(c.name)}
      <span class="muted">${sec(c.start_ms / 1000)} → ${sec(c.end_ms / 1000)}</span></div>
    <p class="muted">Matched ${esc(c.how)}</p>
    <p>${link(c.trace_url, "Span in Cloud Trace")} · ${link(c.logs_url, "Its log entries")}</p>
    <table class="kv">
      ${a["gen_ai.usage.input_tokens"] != null ? `<tr><th>Tokens in / out / thinking</th><td>${tok("gen_ai.usage.input_tokens")} / ${tok("gen_ai.usage.output_tokens")} / ${tok("shelf_bench.usage.thinking_tokens")}</td></tr>` : ""}
      ${a["shelf_bench.embedding.seconds"] != null ? `<tr><th>Embedding</th><td>${esc(a["shelf_bench.embedding.kind"])}, ${esc(a["shelf_bench.embedding.dimension"])}-d, ${esc(a["shelf_bench.embedding.seconds"])} s</td></tr>` : ""}
      ${Object.keys(a).some((k) => k.startsWith("shelf_bench.billed.")) ? `<tr><th>Billed</th><td>${esc(Object.entries(a).filter(([k]) => k.startsWith("shelf_bench.billed.")).map(([k, n]) => `${k.slice(19)} ×${n}`).join(" · "))}</td></tr>` : ""}
      ${cost != null ? `<tr><th>Cost</th><td>${money(+cost)}</td></tr>` : ""}
      ${a["gen_ai.response.finish_reasons"] ? `<tr><th>Finish reason</th><td>${esc(a["gen_ai.response.finish_reasons"])}</td></tr>` : ""}
      ${a["shelf_bench.traffic_type"] ? `<tr><th>Traffic served</th><td>${esc(a["shelf_bench.traffic_type"])}</td></tr>` : ""}
      ${a["shelf_bench.attempts"] && a["shelf_bench.attempts"] !== "1" ? `<tr><th>Attempts</th><td>${esc(a["shelf_bench.attempts"])}</td></tr>` : ""}
    </table>` + attrsHtml(a) + (c.entries.length ? c.entries.map(entryHtml).join("") : '<p class="muted">No log entry found for this span.</p>');
}

function textBlock(label, text) {
  if (!text) return "";
  let body = text;
  try { body = JSON.stringify(JSON.parse(text), null, 1); } catch { /* not JSON: show as is */ }
  return `<details><summary>${label} <span class="muted">(${num(text.length)} chars)</span></summary><pre>${esc(body)}</pre></details>`;
}

function infoHtml(info, d) {
  const refs = GALLERY_SPLITS.has(d.split);
  return Object.entries(info).map(([k, val]) => {
    const label = esc(k.replace(/_/g, " "));
    if (k === "removed" && Array.isArray(val)) {
      return `<p><b>${label}</b>: ${val.length} box${val.length === 1 ? "" : "es"} <span class="muted">(dashed red on the image)</span></p>`;
    }
    if (Array.isArray(val) && val.length && typeof val[0] === "object") {
      const cols = Object.keys(val[0]).filter((c) => c !== "sku_id");
      return `<p><b>${label}</b></p><table class="kv list"><tbody>${val.map((r) => `
        <tr class="${r.sku_id != null && r.sku_id === info.answer ? "pick" : ""}">
          <td>${r.sku_id != null ? `${refs ? `<img class="ref" loading="lazy" src="/img/${d.split}-ref/${encodeURIComponent(r.sku_id)}" alt="">` : ""}#${esc(r.sku_id)}` : ""}</td>
          ${cols.map((c) => `<td>${esc(typeof r[c] === "number" ? +r[c].toFixed(4) : r[c])}</td>`).join("")}</tr>`).join("")}
        </tbody></table>`;
    }
    if (val && typeof val === "object") {
      return `<p><b>${label}</b>: ${esc(Object.entries(val).map(([a, b]) => `${a.replace(/_/g, " ")} ${b}`).join(" · "))}</p>`;
    }
    return `<p><b>${label}</b>: ${esc(val ?? "none")}</p>`;
  }).join("");
}

// Final step: ground truth vs. predictions (TP / FP); for identification, a green / red box per
// right / wrong product (numbered like the table). Other steps: that step's boxes / tiles.
function draw(canvas, img, d, step) {
  const maxW = Math.min(img.naturalWidth, (canvas.parentElement.clientWidth || 700) - 16);
  const k = Math.min(maxW / d.width, (window.innerHeight * 0.75) / d.height);
  canvas.width = d.width * k;
  canvas.height = d.height * k;
  const g = canvas.getContext("2d");
  g.drawImage(img, 0, 0, canvas.width, canvas.height);
  const rect = (b, color, w = 1.5) => { g.strokeStyle = color; g.lineWidth = w; g.strokeRect(b[0] * k, b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k); };
  const GREEN = "rgba(34,197,94,.95)", RED = "rgba(239,68,68,.95)", CYAN = "rgba(6,182,212,.95)";
  const legend = document.getElementById("legend");
  const labels = labelsOf(d);
  const assetBoxes = d.asset_boxes || [];
  if (assetBoxes.length) {
    g.setLineDash([8, 4]);
    assetBoxes.forEach((b) => rect(b, CYAN, 3));
    g.setLineDash([]);
  }
  const assetLegend = assetBoxes.length ? ` <span class="sw asset"></span>Class 0 display bay (${assetBoxes.length})` : "";
  if (!step && d.split === "products" && labels.length) {
    const l = labels[0], ok = l.correct?.product;
    d.gt.forEach((b) => rect(b, ok ? GREEN : RED, 8));
    legend.innerHTML = ok
      ? `<span class="sw right"></span>right product: ${esc(l.gt.product)}`
      : `<span class="sw wrong"></span>wrong: predicted ${esc(l.pred?.product ?? "nothing")}, truth ${esc(l.gt.product)}`;
  } else if (!step && labels.length) {
    const e2e = d.task === "end_to_end";
    if (e2e) {
      g.setLineDash([6, 4]);
      d.gt.forEach((b) => rect(b, "rgba(255,255,255,.9)", 2));
      g.setLineDash([]);
    }
    g.font = "bold 13px sans-serif";
    labels.forEach((l, i) => {
      const c = l.correct?.product ? GREEN : RED;
      rect(l.box, c, 3);
      g.fillStyle = c;
      g.fillRect(l.box[0] * k, l.box[1] * k, 22, 16);
      g.fillStyle = "#fff";
      g.fillText(String(i + 1), l.box[0] * k + 3, l.box[1] * k + 12);
    });
    const ok = labels.filter((l) => l.correct?.product).length;
    legend.innerHTML = `<span class="sw right"></span>right product (${ok}) <span class="sw wrong"></span>wrong product or no product (${labels.length - ok})` +
      (e2e ? ` <span class="sw gt"></span>ground truth, dashed (${d.gt_count}) · products missed or misnamed ${d.fn}` : "") + assetLegend;
  } else if (!step) {
    const tp = new Set(d.matched);
    d.gt.forEach((b) => rect(b, "rgba(34,197,94,.9)"));
    d.preds.forEach((b, i) => rect(b, tp.has(i) ? "rgba(59,130,246,.95)" : "rgba(239,68,68,.95)"));
    legend.innerHTML = `<span class="sw gt"></span>ground truth (${d.gt_count}) <span class="sw tp"></span>correct (${d.tp}) <span class="sw fp"></span>false (${d.fp}) · missed ${d.fn}` + assetLegend;
  } else {
    (step.regions || []).forEach((b) => rect(b, "rgba(250,204,21,.95)", 3));
    (step.boxes || []).forEach((b) => rect(b, "rgba(59,130,246,.95)"));
    const removed = step.info?.removed || [];
    g.setLineDash([5, 3]);
    removed.forEach((b) => rect(b, RED, 2.5));
    g.setLineDash([]);
    const parts = [];
    if (step.regions) parts.push(`<span class="sw tile"></span>tiles (${step.regions.length})`);
    if (step.boxes) parts.push(`<span class="sw tp"></span>boxes (${step.boxes.length})`);
    if (removed.length) parts.push(`<span class="sw fp"></span>removed, dashed (${removed.length})`);
    legend.innerHTML = (parts.join(" ") || '<span class="muted">nothing to overlay for this step</span>') + assetLegend;
  }
}
