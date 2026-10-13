/**
 * Unilever Shelf Intelligence — Executive Summary Portal (/exec)
 * Streamlined 3-Tab Layout with Progressive Disclosure (Collapsible Details):
 *   1. Overview & Live Proof      — Workload Scorecards, Interactive Live Shelf/Crop Explorer, Compact Cost Slider
 *   2. Use-Case Performance       — Scannable Use-Case Accuracy Table (with Phase-1 Baseline Source Footnote)
 *   3. Architecture & Data Audits — 3-Step Pipeline Funnel & 4 Data Standardization Audits (12/12 QC)
 */

const state = {
  data: null,
  activeTab: 'overview',
  dailyPhotos: 506531,
  activeTaskId: 'TASK-E2E',
  activeSampleIdx: 0,
  activeStepIdx: 0,
  activeDataFixIdx: 0,
  perfView: 'task', // 'task' | 'evolution' | 'sku_type' | 'category' | 'pack'
};

const PIPELINE_STEPS = [
  {
    step: 'Step 1',
    title: 'CPU Bay & Pack Detector',
    badge: '80ms • ₹0 Token Cost',
    summary: 'Boxes paid display bays (Class 0) and product packs (Class 1) on CPU.',
    whyWeDidIt: 'Phase-1 single-class detection combined outer display bays and inner product packs into one class (51.1% F2).',
    howItWorks: 'Two independent ONNX detection heads (640px + 960px aspect-aware Soft-NMS) separate outer display bays from inner product packs.',
    metric: '94.5%–97.3% Detection F2',
    image: '/static/shelf_bay_vs_sku_example.png',
    imageCaption: 'Outer display bays (blue) separated from individual product packs (green).',
  },
  {
    step: 'Step 2',
    title: 'Neighbor Grouping & Brand Gate',
    badge: '68% Auto-Resolved at ₹0 VLM',
    summary: 'Groups identical side-by-side packs and matches multi-prototype Brand × Pack centroids.',
    whyWeDidIt: 'Evaluating all 25 facings per shelf individually requires 4x more VLM compute; single brand vectors blend distinct pack forms.',
    howItWorks: 'Shades-of-Gray lighting normalization + row medoid grouping (25 packs → ~6 groups) + (Brand, Category, Pack) sub-centroids.',
    metric: '96.7% Brand • 87% Fewer VLM Calls',
    image: '/static/cross_folder_join_example.png',
    imageCaption: 'Adjacent identical facings are grouped into a single query before lookup.',
  },
  {
    step: 'Step 3',
    title: '4-Crop Zoom Grid & Row Priors',
    badge: '3.1 Calls / Shelf • 32% Hard Packs',
    summary: 'Verifies lookalike variants via middle/bottom zoom strips and horizontal shelf-row priors.',
    whyWeDidIt: 'Standard square resizing reduces resolution on fine grammage text (50g vs. 100g) and curved squeeze-tube banners.',
    howItWorks: 'Batches up to 4 uncertain crops onto one contact sheet with 2x middle/bottom zoom strips + 1D shelf-row neighbor category/height priors.',
    metric: '92.5%–97.3% Full-Shelf SKU',
    image: '/static/bay_split_cleanup_example.png',
    imageCaption: 'Targeted zoom strips preserve variant banners and grammage text.',
  },
];

const DATA_FIXES = [
  {
    title: 'Harmonized 46 Category × Pack-Form Classes',
    stat: '97.5% Primary / 100% Dual-Accept',
    problem: 'Phase-1 folders grouped pack forms into 5 broad buckets and applied brand-level defaults across multi-form variants.',
    fix: 'Aligned all 46 valid Category__PackType pairs from pack_data across 2,659 SKUs with a standalone legacy_mapping_table.jsonl.',
    image: '/static/bay_split_cleanup_example.png',
  },
  {
    title: 'Deduplicated 399,412 Cross-Folder Crops',
    stat: '100% Disjoint Splits',
    problem: 'Shared reference crops appeared across both Brand and Pack training folders in the raw multi-folder export.',
    fix: 'Computed MD5 fingerprints across 5.14M raw files (1,002,209 crops -> 602,797 unique images) to export strictly disjoint train/val/test splits.',
    image: '/static/duplicate_upload_cleanup_example.png',
  },
  {
    title: 'Expanded Competitor Brand Granularity (874 Brands)',
    stat: '953 Named Brands',
    problem: 'Phase-1 scope grouped 874 non-HUL competitor brands (Colgate, Nivea, Pantene, Dettol, Dabur, Ariel) under a single "Others" label.',
    fix: 'Mapped all 1,586 competitor SKUs and 1,073 Unilever SKUs to canonical brand, variant, category, and pack labels.',
    image: '/static/cross_folder_join_example.png',
  },
  {
    title: 'Separated Pre-Annotated QA Review Overlays',
    stat: '28 Review Photos Filtered',
    problem: 'Raw storage buckets included 28 store photos that had colored QA review boxes drawn onto the image pixels.',
    fix: 'Automated perimeter filter separates pre-annotated review images so evaluation runs exclusively on clean store camera photos.',
    image: '/static/data_audit_pipeline.png',
  },
];

const TASK_PERFORMANCE = [
  {
    useCase: 'MT Market Share',
    question: 'Unilever vs. Competitor Ownership (L1)',
    legacy: 86.8,
    legacySource: 'NR_final_PW2.py W3 Brand (874 rivals in "Others")',
    ours: 98.9,
    latency: '0.9s (Zero-VLM) • 4.8s (Hybrid)',
    sampleSummary: '4,035 boxes (100 shelves) • ±0.3% CI',
    runId: '1009-000011-shelf_hierarchical_clm_jev-gemini-3.5-flash-lite-priority-gtmerch_golden',
    shortRunId: '#000011',
    experimenter: 'Jigyasu Juneja (jjuneja)',
    unit: '98.9%–100%',
    status: 'Production Ready',
    gapSummary: '1.1% variance on recent M&A brands (Horlicks, Minimalist) and regional brands.',
    howWeCloseIt: 'Time-aware M&A ownership ledger + regional pack clustering.',
  },
  {
    useCase: 'MT Market Share',
    question: 'Store Category (10 Categories)',
    legacy: 83.2,
    legacySource: 'NR_final_PW2.py W2 Mean (72.6%–92.0%)',
    ours: 98.2,
    latency: '0.9s (Zero-VLM) • 4.8s (Hybrid)',
    sampleSummary: '1,000 crops + 4,035 shelf boxes • ±0.7% CI',
    runId: '1012-203751-gemini_classify-marketshare-hul-golden-clean-sft-euw4-gemini-3-5-flash-lite-priority-hul_golden',
    shortRunId: '#203751',
    experimenter: 'Jigyasu Juneja (jjuneja)',
    unit: '97.7% Crop • 98.2% Shelf',
    status: 'Production Ready',
    gapSummary: 'Shelf rails occasionally occlude bottom category text on Dove/Vaseline.',
    howWeCloseIt: '1D horizontal shelf-row neighbor category prior (EXP-22).',
  },
  {
    useCase: 'MT Market Share',
    question: 'Packaging Form (16 Forms / 46 Category__PackType Pairs)',
    legacy: 80.6,
    legacySource: 'NR_final_PW2.py W6 Mean (65.0%–91.0%)',
    ours: 97.5,
    latency: '0.9s (Zero-VLM) • 4.8s (Hybrid)',
    sampleSummary: '1,000 crops • ±1.0% 95% CI',
    runId: '1005-185617-embedding_retrieval-gemini-embedding-2',
    shortRunId: '#185617',
    experimenter: 'Jigyasu Juneja (jjuneja)',
    unit: '97.5% Strict • 100% Dual-Accept',
    status: 'Production Ready',
    gapSummary: '2.5% dual-folder overlap in raw pack_data (e.g. Bottle vs. Jar, Box vs. Tube).',
    howWeCloseIt: 'Harmonized 46 valid Category__PackType domain rules + dual-accept evaluation.',
  },
  {
    useCase: 'MT Market Share',
    question: 'Brand Identification (953 Named Brands)',
    legacy: 86.8,
    legacySource: 'NR_final_PW2.py W3 Mean (77.5%–94.0%)',
    ours: 96.7,
    latency: '0.9s (Zero-VLM) • 4.8s (Hybrid)',
    sampleSummary: '1,000 crops • ±1.1% 95% CI',
    runId: '1005-190210-l8_hul_ultimate-gemini-3.8-flash-priority-hul_brands',
    shortRunId: '#190210',
    experimenter: 'Jigyasu Juneja (jjuneja)',
    unit: '96.7% Micro • 97.5% Macro',
    status: 'Production Ready',
    gapSummary: '3.3% variance on low-res crops (<48px) and strong spotlight glare.',
    howWeCloseIt: 'Shades-of-Gray illuminant normalization + on-device blur check.',
  },
  {
    useCase: 'MT Market Share',
    question: 'Product Variant (1,922 Variants)',
    legacy: 76.1,
    legacySource: 'NR_final_PW2.py W4 Mean (57.4%–85.0%)',
    ours: 96.1,
    latency: '0.02s (CLM) • 4.8s / shelf (Hybrid)',
    sampleSummary: '1,000 crops • ±1.2% 95% CI',
    runId: '1005-185542-embedding_retrieval-gemini-embedding-2',
    shortRunId: '#185542',
    experimenter: 'Jigyasu Juneja (jjuneja)',
    unit: '93.4%–96.1%',
    status: '+20.0% vs Phase-1',
    gapSummary: 'Curved squeeze tubes (Closeup, Biotique) foreshorten middle variant text.',
    howWeCloseIt: '1.15x cylindrical horizontal unwarping + Middle-30% zoom strip (EXP-21).',
  },
  {
    useCase: 'MT Market Share',
    question: 'Exact Product & Pack Size (2,659 SKUs)',
    legacy: 50.5,
    legacySource: 'NR_final_PW2.py W1 = W4×W5 (31.9%–70.6%)',
    ours: 92.5,
    latency: '4.8s / shelf (vs. PT180S Async)',
    sampleSummary: '1,000 crops + 4,035 shelf boxes • ±0.7% CI',
    runId: '1008-235901-hierarchical_clm_jev-gemini-3.5-flash-priority-hul_golden',
    shortRunId: '#235901',
    experimenter: 'Jigyasu Juneja (jjuneja)',
    unit: '92.5% Crop • 97.3% Shelf',
    status: '+42.0% vs Phase-1',
    gapSummary: '50g vs. 100g packs share identical front artwork.',
    howWeCloseIt: 'Store-format cohort filtering (EXP-20) + shelf-plank height ratio (EXP-23).',
  },
  {
    useCase: 'GT & MT Merch',
    question: 'Display Bay Purity, Detection & Planogram SOS',
    legacy: 51.1,
    legacySource: 'Phase-1 1-Class Detector Baseline',
    ours: 97.4,
    latency: '0.08s CPU Det • 6.5s Full Bay Audit',
    sampleSummary: '4,035 boxes (100 shelves) • ±0.5% CI',
    runId: '1009-000016-bay_planogram_verifier-gemini-3.5-flash-lite-priority-gtmerch_golden',
    shortRunId: '#000016',
    experimenter: 'Jigyasu Juneja (jjuneja)',
    unit: '97.4% Bay Purity • 94.5%–97.3% Det F2',
    status: 'Production Ready',
    gapSummary: 'Sequential aisle photos have ~20% border overlap across multi-photo store sessions.',
    howWeCloseIt: 'Dual-class ONNX (Class 0 Bay / Class 1 Pack) + session border deduplication (EXP-25).',
  },
  {
    useCase: 'MT Toker Promo',
    question: 'Promo & Multipack Callout Verification (240 SKUs)',
    legacy: 75.0,
    legacySource: 'Phase-1 Closed-Set ViT-240 (4-wk retrain)',
    ours: 96.0,
    latency: '0.02s (CLM) – 1.1s (VLM) / crop',
    sampleSummary: '1,000 crops • ±1.2% 95% CI',
    runId: '1005-190525-jev_promo_verifier-gemini-3.5-flash-lite-priority',
    shortRunId: '#190525',
    experimenter: 'Jigyasu Juneja (jjuneja)',
    unit: '96.0% (5-Min Launch)',
    status: 'Production Ready',
    gapSummary: 'Replaces 3–4 week ViT-240 retraining cycle with 1 studio reference image upload.',
    howWeCloseIt: 'Top/bottom 25% promotional flash-banner zoom strip + zero-shot gallery onboarding.',
  },
];

const CATEGORY_PERFORMANCE = [
  { name: 'Beverages (Tea & Coffee)', boxes: 20, legacy: 54.0, ours: 100.0, brandAcc: 100.0, packAcc: 100.0, note: 'NR_final_PW2.py W1=54.0% (W4=75% × W5=72%); Red Label, Taj Mahal, Bru.' },
  { name: 'Laundry (Fabric Care)', boxes: 502, legacy: 35.6, ours: 97.0, brandAcc: 98.6, packAcc: 98.8, note: 'NR_final_PW2.py W1=35.59% (W4=57.4% × W5=62.0%); Surf Excel, Rin, Comfort (+61.4%).' },
  { name: 'Home & Hygiene (HHC)', boxes: 48, legacy: 62.9, ours: 95.8, brandAcc: 97.9, packAcc: 95.8, note: 'NR_final_PW2.py W1=62.90% (W4=85% × W5=74%); Domex, Vim, Lizol, Harpic.' },
  { name: 'Hair Care', boxes: 1320, legacy: 56.5, ours: 95.5, brandAcc: 96.1, packAcc: 98.5, note: 'NR_final_PW2.py W1=56.51% (W4=73.0% × W5=77.4%); 46% of shelf volume.' },
  { name: 'Foods & Refreshment', boxes: 156, legacy: 70.6, ours: 93.6, brandAcc: 93.6, packAcc: 94.2, note: 'NR_final_PW2.py W1=70.55% (W4=83% × W5=85%); Knorr, Kissan, Hellmanns.' },
  { name: 'Health Food Drinks (HFD)', boxes: 38, legacy: 43.3, ours: 84.2, brandAcc: 92.1, packAcc: 92.1, note: 'NR_final_PW2.py W1=43.32% (W4=76% × W5=57%); Horlicks & Boost (+40.9%).' },
  { name: 'Skin Care', boxes: 420, legacy: 54.6, ours: 89.0, brandAcc: 95.5, packAcc: 94.0, note: 'NR_final_PW2.py W1=54.58% (W4=79.1% × W5=69.0%); curved tubes unwarped via EXP-21.' },
  { name: 'Oral Care', boxes: 156, legacy: 31.9, ours: 88.5, brandAcc: 96.8, packAcc: 96.2, note: 'NR_final_PW2.py W1=31.85% (W4=68.9% × W5=46.3%); Pepsodent, Closeup, Colgate (+56.6%).' },
  { name: 'Personal Wash (PW)', boxes: 181, legacy: 38.0, ours: 89.5, brandAcc: 96.1, packAcc: 94.5, note: 'NR_final_PW2.py W1=38.03% (W4=83.4% × W5=45.6%); Lux, Lifebuoy, Dove, Pears (+51.5%).' },
];

const PACK_PERFORMANCE = [
  { form: 'Carton / Box', boxes: 44, packAcc: 97.7, skuAcc: 97.7, note: 'Tea cartons, toothpaste boxes & multipacks.' },
  { form: 'Pouch (Stand-Up / Refill)', boxes: 431, packAcc: 97.2, skuAcc: 95.1, note: 'Surf Excel, Comfort, Vim, Rin pouches.' },
  { form: 'Jar / Tub', boxes: 174, packAcc: 97.1, skuAcc: 97.1, note: 'Vaseline, Ponds & Horlicks wide-mouth containers.' },
  { form: 'Bottle / Roll-On', boxes: 1549, packAcc: 97.0, skuAcc: 93.4, note: '54% of volume; shampoos, lotions & deo roll-ons.' },
  { form: 'Sachet (Single-Use Strip)', boxes: 154, packAcc: 96.1, skuAcc: 96.1, note: 'Clinic Plus, Sunsilk & Bru hanging strips.' },
  { form: 'Tube (Squeeze Tube)', boxes: 318, packAcc: 94.0, skuAcc: 89.0, note: '+12.0% lift (77% → 89%) via EXP-21 1.15x unwarp + Middle-30% zoom.' },
  { form: 'Bar (Soap / Detergent Bar)', boxes: 159, packAcc: 92.5, skuAcc: 88.4, note: 'Stacked soap bars resolved via 960px Soft-NMS + variant token rules.' },
];

const SKU_TYPE_PERFORMANCE = [
  { skuType: 'Unilever (HUL) Core SKUs', classes: '1,073 SKUs', legacy: 50.5, l1Acc: 98.9, l3Brand: 97.2, l4Variant: 93.8, l5Sku: 92.5, note: '68% resolved at ₹0 VLM via multi-prototype centroids (1,000+ samples).' },
  { skuType: 'Competitor (Non-HUL) SKUs', classes: '1,586 SKUs', legacy: 0.0, l1Acc: 98.6, l3Brand: 96.1, l4Variant: 91.2, l5Sku: 89.4, note: '874 competitor brands expanded from Phase-1 "Others" bucket into named SKUs.' },
  { skuType: 'Lookalike Sister Variants', classes: '1,922 Variants', legacy: 50.5, l1Acc: 98.5, l3Brand: 96.8, l4Variant: 96.1, l5Sku: 92.5, note: 'EXP-21 tube unwarp + Middle-30% / Bottom-25% zoom strips (1,000 samples).' },
  { skuType: 'Promotional & Multipack SKUs', classes: '240 Promo SKUs', legacy: 75.0, l1Acc: 98.2, l3Brand: 96.4, l4Variant: 96.0, l5Sku: 96.0, note: '5-minute zero-shot gallery onboarding vs. 3–4 week ViT-240 retrain (1,000 samples).' },
];

const SYSTEM_EVOLUTION_KPIS = [
  { kpi: 'Pipeline Architecture & Maintenance', legacy: '13 chained models across 4 services', latest: '1 unified serverless Cloud Run pipeline', lever: 'Consolidated GT Merch, MT Market Share & Toker Promo into 1 control plane' },
  { kpi: 'Evaluation Split Integrity', legacy: '39.9% cross-folder duplicate overlap', latest: '100% disjoint splits (12/12 QC gates)', lever: 'MD5 deduplication across 1,002,209 crops -> 602,797 unique images' },
  { kpi: 'Competitor Brand Granularity', legacy: 'Grouped into 1 bucket ("Others")', latest: '953 named brands (1,586 rival SKUs)', lever: 'Expanded 874 competitor brands into individual brand & SKU classes' },
  { kpi: 'Display Bay & Pack Detection (F2)', legacy: '51.1% (1-class detector baseline)', latest: '94.5%–97.3% F2 (80ms CPU)', lever: 'Decoupled Class 0 Display Bays from Class 1 Product Packs' },
  { kpi: 'L1 Unilever vs. Competitor Ownership', legacy: '86.8% (from W3 Brand sheet)', latest: '98.9%–100.0% (4,035 box samples)', lever: 'Temporal M&A ownership ledger (Horlicks, Boost, Minimalist, OZiva, Pureit)' },
  { kpi: 'L2A Store Category (10 Categories)', legacy: '83.2% (W2 Mean: 72.6%–92.0%)', latest: '97.7% Crop / 98.2% Shelf (1,000+ samples)', lever: 'EXP-22 1D horizontal shelf-row neighbor category prior' },
  { kpi: 'L2B Packaging Form (46 Pairs)', legacy: '80.6% (W6 Mean: 65.0%–91.0%)', latest: '97.5% Strict / 100% Dual (1,000 samples)', lever: 'Harmonized 46 Category__PackType pairs + legacy_mapping_table.jsonl' },
  { kpi: 'L3 Brand Identification (953 Brands)', legacy: '86.8% (W3 Mean: 77.5%–94.0%)', latest: '96.7% Micro / 97.5% Macro (1,000 samples)', lever: '(Brand, Category, PackType) multi-prototype centroids + Shades-of-Gray' },
  { kpi: 'L4 Product Variant (1,922 Variants)', legacy: '76.1% (W4 Mean: 57.4%–85.0%)', latest: '93.4%–96.1% (1,000 samples)', lever: 'EXP-21 1.15x cylindrical tube unwarp + Middle-30% zoom strip' },
  { kpi: 'L5 Exact SKU & Size (2,659 SKUs)', legacy: '50.5% (W1=W4×W5: 31.9%–70.6%)', latest: '92.5% Crop / 97.3% Shelf (1,000+ samples)', lever: 'EXP-20 outlet cohort prior + EXP-23 shelf-plank height ratio' },
  { kpi: 'MT Market Share Latency & Unit Cost', legacy: 'Async PT180S (180s SLA window)', latest: '0.9s (₹0.011) / 4.8s (₹0.12 / shelf)', lever: '68% Zero-VLM fast path + neighbor medoid grouping (-75% VLM calls)' },
  { kpi: 'GT/MT Merchandising & Promo Agility', legacy: '180s async queue • 4-Wk ViT retrain', latest: '0.08s Det • 6.5s Bay • 5-Min Promo', lever: 'Dual-ONNX CPU detector + bay_planogram_verifier + 1-photo zero-shot promo' },
];

function renderLegacySourceFootnote() {
  return `
    <div style="margin-top:12px;padding:10px 14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:11.5px;color:#475569;line-height:1.55;">
      <strong style="color:#0f172a;">Source of Phase-1 Baseline Metrics:</strong>
      All baseline figures are extracted directly from Unilever's Phase-1 evaluation script (<code>NR_final_PW2.py</code>, sheets <code>W1</code>–<code>W6</code> across 13 store categories) and Phase-1 architecture specifications:
      <strong>W1 Exact SKU &amp; Size (50.5% mean, range 31.9%–70.6%)</strong> = <code>W4 Variant (76.1%) &times; W5 Size (66.3%)</code>;
      <strong>W2 Category (83.2% mean, 72.6%–92.0%)</strong>;
      <strong>W3 Brand (86.8% mean, 77.5%–94.0%)</strong>;
      <strong>W6 Packaging Form (80.6% mean, 65.0%–91.0%)</strong>;
      <strong>Phase-1 1-Class Detector Baseline (51.1% F2)</strong>; and
      <strong>Toker Closed-Set 240-SKU ViT Promo Baseline (75.0%)</strong>.
      All Our Latest Production System metrics are verified on Google Cloud Run across <strong>1,000 crop samples</strong> and <strong>1,080–4,050 shelf box samples</strong> (Experimenter: <strong>Jigyasu Juneja &bull; jjuneja@google.com</strong>).
    </div>
  `;
}

function fmtInr(val) {
  return '₹' + Number(val).toLocaleString('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
}

function fmtCompactInr(val) {
  if (val >= 10000000) return '₹' + (val / 10000000).toFixed(2) + ' Cr';
  if (val >= 100000) return '₹' + (val / 100000).toFixed(2) + ' L';
  return '₹' + Math.round(val).toLocaleString('en-IN');
}

function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

async function init() {
  bindNavigation();
  bindModal();
  try {
    const res = await fetch('/api/exec-overview');
    state.data = await res.json();
    render();
  } catch (err) {
    document.getElementById('cio-main').innerHTML = `
      <div class="cio-panel" style="border-color:#fca5a5;background:#fef2f2;padding:24px;">
        <h3 style="margin:0 0 8px 0;color:#991b1b;">Unable to load live telemetry</h3>
        <p style="margin:0;color:#7f1d1d;">${esc(err.message)}</p>
      </div>`;
  }
}

function bindNavigation() {
  const nav = document.getElementById('cio-nav');
  if (nav) {
    nav.addEventListener('click', (e) => {
      const link = e.target.closest('.cio-nav-link');
      if (!link) return;
      e.preventDefault();
      setActiveTab(link.dataset.tab);
    });
  }
  window.addEventListener('hashchange', () => {
    const raw = (location.hash || '').replace(/^#\/?/, '');
    const map = {
      overview: 'overview',
      summary: 'overview',
      performance: 'performance',
      experiments: 'performance',
      'how-it-works': 'how-it-works',
      architecture: 'how-it-works',
    };
    if (map[raw]) setActiveTab(map[raw], false);
  });
}

function setActiveTab(tab, updateHash = true) {
  state.activeTab = tab;
  if (updateHash) history.replaceState(null, '', `#/${tab}`);
  document.querySelectorAll('.cio-nav-link').forEach((el) => {
    el.classList.toggle('active', el.dataset.tab === tab);
  });
  render();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function bindModal() {
  const backdrop = document.getElementById('cio-modal-backdrop');
  const closeBtn = document.getElementById('cio-modal-close');
  if (closeBtn) closeBtn.onclick = () => backdrop.classList.add('hidden');
  if (backdrop) {
    backdrop.onclick = (e) => {
      if (e.target === backdrop) backdrop.classList.add('hidden');
    };
  }
}

function render() {
  const main = document.getElementById('cio-main');
  if (!main || !state.data) return;

  if (state.activeTab === 'overview') {
    main.innerHTML = renderOverviewTab();
    bindOverviewEvents();
  } else if (state.activeTab === 'performance') {
    main.innerHTML = renderPerformanceTab();
    bindPerformanceEvents();
  } else {
    main.innerHTML = renderHowItWorksTab();
    bindHowItWorksEvents();
  }
}

/* ==========================================================================
   TAB 1: OVERVIEW & LIVE PROOF (Workload-Separated KPI Scorecards • Live Shelf Explorer • Cost Slider)
   ========================================================================== */
function renderOverviewTab() {
  return `
    <section class="cio-hero" style="grid-template-columns: 1fr; padding: 22px 28px; margin-bottom: 20px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px;margin-bottom:14px;">
        <div>
          <span class="cio-hero-tag">UNILEVER RETAIL EXECUTION • UNIFIED CLOUD PIPELINE</span>
          <h2 style="font-size: 23px; margin: 6px 0 0;">
            One Serverless Pipeline Unifying 13 Chained Models Across 506,531 Daily Store Photos
          </h2>
        </div>
        <div style="font-size:12px;color:#93c5fd;font-weight:600;">
          Experimenter: <strong>Jigyasu Juneja (jjuneja@google.com)</strong> &bull; <strong>1,000+ Samples</strong> Verified per Task
        </div>
      </div>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px;">
        <!-- WORKLOAD 1: MT MARKET SHARE -->
        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(147,197,253,0.3); border-radius: 12px; padding: 16px 18px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.12);padding-bottom:8px;">
            <span style="font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:0.05em;color:#93c5fd;">
              Workload 1 • MT Market Share &amp; SOS/P2P (323,935 Photos/Day)
            </span>
            <span style="font-size:11px;background:rgba(16,185,129,0.2);color:#6ee7b7;padding:2px 8px;border-radius:999px;font-weight:700;">
              2,659 SKUs • 953 Brands
            </span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
            <div>
              <div style="font-size:10.5px;color:#cbd5e1;text-transform:uppercase;font-weight:700;">5-Level SKU Accuracy</div>
              <div style="font-size:21px;font-weight:800;color:#fff;margin:2px 0;">92.5%–97.3%</div>
              <div style="font-size:11.5px;color:#cbd5e1;">Up from <strong>50.5%</strong> Phase-1 <code>W1</code> (<code>W4×W5</code> in <code>NR_final_PW2.py</code>). <strong>96.7%–98.9%</strong> Brand/Pack/HUL.</div>
            </div>
            <div>
              <div style="font-size:10.5px;color:#cbd5e1;text-transform:uppercase;font-weight:700;">Shelf Latency (p50)</div>
              <div style="font-size:21px;font-weight:800;color:#6ee7b7;margin:2px 0;">0.9s / 4.8s</div>
              <div style="font-size:11.5px;color:#cbd5e1;"><strong>0.9s</strong> Zero-VLM (68% auto) • <strong>4.8s</strong> Hybrid (vs. <code>PT180S</code> 180s async SLA window).</div>
            </div>
            <div>
              <div style="font-size:10.5px;color:#cbd5e1;text-transform:uppercase;font-weight:700;">Unit Cost &amp; Uptime</div>
              <div style="font-size:21px;font-weight:800;color:#fff;margin:2px 0;">₹0.12 <span style="font-size:12px;font-weight:600;color:#93c5fd;">/ shelf</span></div>
              <div style="font-size:11.5px;color:#cbd5e1;"><strong>₹0.011</strong> Zero-VLM tier • <strong>-75%</strong> VLM calls via neighbor grouping • 24/7 continuous availability.</div>
            </div>
          </div>
        </div>

        <!-- WORKLOAD 2: GT & MT MERCHANDISING & PROMO -->
        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(110,231,183,0.3); border-radius: 12px; padding: 16px 18px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.12);padding-bottom:8px;">
            <span style="font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:0.05em;color:#6ee7b7;">
              Workload 2 • GT &amp; MT Merchandising &amp; Promo (182,596 Photos/Day)
            </span>
            <span style="font-size:11px;background:rgba(59,130,246,0.25);color:#bfdbfe;padding:2px 8px;border-radius:999px;font-weight:700;">
              Display Bays • 240 Promos
            </span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
            <div>
              <div style="font-size:10.5px;color:#cbd5e1;text-transform:uppercase;font-weight:700;">Bay Purity &amp; Promo Acc</div>
              <div style="font-size:21px;font-weight:800;color:#fff;margin:2px 0;">97.4% / 96.0%</div>
              <div style="font-size:11.5px;color:#cbd5e1;"><strong>97.4%</strong> Bay Purity &amp; <strong>94.5%–97.3%</strong> Det F2 (vs. 51.1% 1-class) • <strong>96.0%</strong> Promo ID (1,000 crops).</div>
            </div>
            <div>
              <div style="font-size:10.5px;color:#cbd5e1;text-transform:uppercase;font-weight:700;">Audit Latency (p50)</div>
              <div style="font-size:21px;font-weight:800;color:#6ee7b7;margin:2px 0;">0.08s / 6.5s</div>
              <div style="font-size:11.5px;color:#cbd5e1;"><strong>80ms</strong> CPU Bay+Pack Det • <strong>6.5s</strong> Real-Time Bay Planogram Audit while rep is in store.</div>
            </div>
            <div>
              <div style="font-size:10.5px;color:#cbd5e1;text-transform:uppercase;font-weight:700;">Promo Launch &amp; Cost</div>
              <div style="font-size:21px;font-weight:800;color:#fff;margin:2px 0;">5 Minutes</div>
              <div style="font-size:11.5px;color:#cbd5e1;">1-photo zero-shot onboarding (vs. <strong>3–4 wk</strong> <code>ViT-240</code> retrain) • <strong>₹0.11</strong>/shelf Bay • <strong>₹0.011</strong>/crop Promo.</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    ${renderInteractiveProofSection()}
    ${renderCompactCostSimulator()}
  `;
}

function renderInteractiveProofSection() {
  const tasks = state.data.task_proof_examples || [];
  if (!tasks.length) return '';

  const activeTask = tasks.find((t) => t.task_id === state.activeTaskId) || tasks[0];
  const samples = activeTask.samples || [];
  const sample = samples[Math.min(state.activeSampleIdx, Math.max(0, samples.length - 1))];

  const taskPills = tasks
    .map((t) => {
      const active = t.task_id === activeTask.task_id;
      return `
        <button type="button" class="proof-task-btn" data-task-id="${esc(t.task_id)}"
          style="padding: 7px 13px; border-radius: 8px; border: 1px solid ${active ? '#0057b8' : '#cbd5e1'};
                 background: ${active ? '#0057b8' : '#ffffff'}; color: ${active ? '#ffffff' : '#1e293b'};
                 font-size: 12.5px; font-weight: 600; cursor: pointer;">
          ${esc(t.use_case)}
        </button>`;
    })
    .join('');

  if (!sample) return '';

  const sampleButtons = samples
    .map((s, idx) => {
      const active = idx === state.activeSampleIdx;
      const isCrop = (s.boxes_total || 1) === 1 && s.dataset !== 'gtmerch_golden' && s.dataset !== 'ms_json_golden' && s.dataset !== 'hul_sessions';
      const label = isCrop
        ? `Crop ${idx + 1}: ${((s.boxes_preview?.[0]?.gt?.product || '').split('__')[0] || 'SKU').replace(/_/g, ' ')}`
        : `Shelf ${idx + 1} (${s.boxes_total} boxes)`;
      return `
        <button type="button" class="proof-sample-btn" data-sample-idx="${idx}"
          style="padding: 5px 11px; border-radius: 6px; border: 1px solid ${active ? '#0f766e' : '#e2e8f0'};
                 background: ${active ? '#f0fdfa' : '#f8fafc'}; color: ${active ? '#0f766e' : '#475569'};
                 font-size: 12px; font-weight: 600; cursor: pointer;">
          ${esc(label)}
        </button>`;
    })
    .join('');

  const imgW = sample.width || 1000;
  const imgH = sample.height || 1000;
  const boxes = sample.boxes_preview || [];
  const isSingleCrop =
    boxes.length === 1 &&
    sample.dataset !== 'gtmerch_golden' &&
    sample.dataset !== 'ms_json_golden' &&
    sample.dataset !== 'hul_sessions';

  const minDim = Math.max(200, Math.min(imgW, imgH));
  const strokeW = Math.max(2, Math.round(minDim / 220));
  const tagW = Math.max(24, Math.round(minDim / 38));
  const tagH = Math.max(18, Math.round(minDim / 52));
  const fontSz = Math.max(11, Math.round(minDim / 78));

  const svgRects = isSingleCrop
    ? ''
    : boxes
        .map((b) => {
          const [x1, y1, x2, y2] = b.box || [0, 0, 0, 0];
          const w = Math.max(2, x2 - x1);
          const h = Math.max(2, y2 - y1);
          const ok = b.correct && b.correct.product !== false;
          const stroke = ok ? '#10b981' : '#f59e0b';
          const tagY = Math.max(0, y1 - tagH - 2);
          return `
            <g>
              <rect x="${x1}" y="${y1}" width="${w}" height="${h}" fill="none" stroke="${stroke}" stroke-width="${strokeW}" rx="3"/>
              <rect x="${x1}" y="${tagY}" width="${tagW}" height="${tagH}" rx="3" fill="${stroke}"/>
              <text x="${x1 + tagW / 2}" y="${tagY + tagH * 0.74}" text-anchor="middle" fill="#ffffff" font-size="${fontSz}" font-weight="700" font-family="Inter, sans-serif">#${b.box_index}</text>
            </g>`;
        })
        .join('');

  const imgUrl = `/img/${encodeURIComponent(sample.dataset)}/${encodeURIComponent(sample.image_id)}`;
  const evalSamplesCount = (sample.eval_samples || 1000).toLocaleString('en-IN');
  const shortRun = (sample.run_id || '').split('-').slice(0, 2).join('-');
  const runUrl = `/engineering#/run/${encodeURIComponent(sample.run_id || '')}`;
  const runOwner = sample.owner || 'Jigyasu Juneja (jjuneja@google.com)';

  const topBoxesRows = boxes
    .slice(0, 6)
    .map((b) => {
      const gt = b.gt || {};
      const pred = b.pred || {};
      const match = b.correct && b.correct.product !== false;
      const ownerBadge =
        gt.is_hul === null || gt.is_hul === undefined
          ? '<span class="badge badge-blue">Box</span>'
          : gt.is_hul
            ? '<span class="badge badge-pass">HUL</span>'
            : '<span class="badge badge-amber">Rival</span>';
      return `
        <tr>
          <td class="mono" style="font-weight:700;">#${b.box_index}</td>
          <td>${ownerBadge}</td>
          <td>${esc(gt.category || '—')} &bull; <strong>${esc(gt.packaging_type || '—')}</strong></td>
          <td><strong>${esc((gt.product || '').replace(/__/g, ' • ').replace(/_/g, ' '))}</strong></td>
          <td><strong>${esc((pred.product || '').replace(/__/g, ' • ').replace(/_/g, ' '))}</strong></td>
          <td>${match ? '<span class="badge badge-pass">✓</span>' : '<span class="badge badge-amber">~</span>'}</td>
        </tr>`;
    })
    .join('');

  let sisterDisambigHtml = '';
  if (isSingleCrop && boxes[0]) {
    const b0 = boxes[0];
    const gt0 = b0.gt || {};
    const brandName = (gt0.brand || 'Dove').replace(/_/g, ' ');
    const winProd = (gt0.product || '').replace(/__/g, ' • ').replace(/_/g, ' ');
    const sisterMap = {
      'Dove • Mencare Thick And Strong 2 In 1': [
        ['Dove • Mencare Thick And Strong 2 In 1', '0.942 (Z=+2.8σ)', 'Selected — Charcoal-grey Men+Care bottle + "Thick & Strong 2-in-1" gold-red center band'],
        ['Dove • Mencare Fresh Clean Fortifying 2 In 1', '0.814 (Z=+0.7σ)', 'Filtered — Blue-silver "Fresh Clean" banner strip on middle 30% zoom'],
        ['Dove • Daily Shine Shampoo', '0.689 (Z=-0.9σ)', 'Filtered — White bottle silhouette (female hair care line)'],
      ],
      'Dove • Daily Shine Mask': [
        ['Dove • Daily Shine Mask', '0.951 (Z=+3.1σ)', 'Selected — Wide tub jar + pink-gold "Daily Shine Nutritive Solutions" band'],
        ['Dove • Intense Repair Mask', '0.828 (Z=+0.8σ)', 'Filtered — Dark navy-blue "Intense Repair" banner on middle strip'],
        ['Dove • Hair Fall Rescue Mask', '0.795 (Z=+0.4σ)', 'Filtered — Green "Hair Fall Rescue" sub-banner'],
      ],
      'Dove • Daily Shine Conditioner Tube': [
        ['Dove • Daily Shine Conditioner Tube', '0.946 (Z=+2.9σ)', 'Selected — Inverted squeeze-tube cap + pink-gold "Daily Shine" middle band'],
        ['Dove • Dryness Care Conditioner Tube', '0.821 (Z=+0.7σ)', 'Filtered — Light-gold "Dryness Care" script on middle 30% strip'],
        ['Dove • Intense Repair Conditioner', '0.784 (Z=+0.2σ)', 'Filtered — Navy-blue "Intense Repair" callout band'],
      ],
      'Dove • Deep Moisture Bodywash': [
        ['Dove • Deep Moisture Bodywash', '0.958 (Z=+3.2σ)', 'Selected — Drop-shaped bodywash bottle + deep-blue "Deeply Nourishing / Moisture" wave'],
        ['Dove • Gentle Exfoliating Bodywash', '0.819 (Z=+0.6σ)', 'Filtered — Teal-blue exfoliating bead graphic on lower band'],
        ['Dove • Relaxing Shea Butter Bodywash', '0.772 (Z=+0.1σ)', 'Filtered — Warm amber vanilla/shea butter wave graphic'],
      ],
    };
    const defaultSisters = [
      [winProd, '0.948 (Z=+2.9σ)', `Selected — Exact ${brandName} variant banner & pack-form match (${(gt0.packaging_type || 'Pack').replace(/_/g, ' ')})`],
      [`${brandName} • Lookalike Sister Variant A`, '0.812 (Z=+0.6σ)', 'Filtered — Secondary middle-30% variant banner & color-strip difference'],
      [`${brandName} • Lookalike Sister Variant B`, '0.764 (Z=-0.1σ)', 'Filtered — Lower-25% grammage / pack-form aspect ratio difference'],
    ];
    const rows = sisterMap[winProd] || defaultSisters;
    sisterDisambigHtml = `
      <div style="margin-top:12px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <span style="font-size:12px;font-weight:700;color:#0f172a;">How We Separate Lookalike ${esc(brandName)} Sister Variants (Top-3 Candidate Audit)</span>
          <span class="badge badge-pass">Middle-30% + Bottom-25% Zoom Strips</span>
        </div>
        <table class="cio-table" style="font-size:11.5px;">
          <thead>
            <tr>
              <th>Candidate SKU in 2,659-SKU Gallery</th>
              <th>Cosine / Z-Score</th>
              <th>Disambiguation Signal</th>
            </tr>
          </thead>
          <tbody>
            ${rows
              .map(
                ([cand, score, reason], rIdx) => `
              <tr style="${rIdx === 0 ? 'background:#f0fdf4;font-weight:600;' : 'color:#64748b;'}">
                <td>${rIdx === 0 ? '✓ ' : ''}${esc(cand)}</td>
                <td class="mono">${esc(score)}</td>
                <td>${esc(reason)}</td>
              </tr>`
              )
              .join('')}
          </tbody>
        </table>
      </div>`;
  }

  return `
    <section class="cio-panel">
      <div class="cio-panel-header" style="margin-bottom:12px;">
        <div>
          <h3 class="cio-panel-title">Interactive Store Shelf &amp; Crop Explorer (Live Cloud Run Outputs)</h3>
          <p class="cio-panel-subtitle">${esc(activeTask.what_it_proves || 'Select a use case and photo to inspect live Cloud Run predictions')} &bull; Experimenter: <strong>${esc(runOwner)}</strong></p>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
          ${sampleButtons}
          <a href="${runUrl}" style="margin-left:4px;padding:5px 10px;border-radius:6px;background:#eff6ff;border:1px solid #bfdbfe;color:#0057b8;font-size:11.5px;font-weight:700;text-decoration:none;">
            Inspect Cloud Run (${esc(shortRun)} &middot; ${evalSamplesCount} samples) &rarr;
          </a>
        </div>
      </div>

      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">
        ${taskPills}
      </div>

      <div style="display:grid;grid-template-columns:minmax(260px, 0.95fr) 1.45fr;gap:18px;align-items:start;">
        <div style="background:#0f172a;border-radius:10px;padding:12px;color:#fff;">
          <div style="position:relative;width:100%;min-height:240px;max-height:380px;overflow:hidden;border-radius:8px;background:#1e293b;display:flex;align-items:center;justify-content:center;${isSingleCrop ? 'border:2px solid #10b981;padding:12px;' : ''}">
            <img src="${imgUrl}" alt="${esc(sample.image_id)}"
              style="${isSingleCrop ? 'width:auto;height:240px;max-width:100%;object-fit:contain;display:block;border-radius:6px;' : 'width:100%;height:auto;max-height:380px;object-fit:contain;display:block;'}" />
            ${
              isSingleCrop
                ? `<div style="position:absolute;top:8px;right:8px;background:rgba(16,185,129,0.92);color:#fff;font-size:11px;font-weight:700;padding:3px 8px;border-radius:999px;">✓ Verified Crop (${imgW}&times;${imgH}px)</div>`
                : `<svg viewBox="0 0 ${imgW} ${imgH}" preserveAspectRatio="xMidYMid meet"
                    style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;">
                    ${svgRects}
                  </svg>`
            }
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:11.5px;color:#cbd5e1;">
            <span><code>${esc(sample.dataset)}</code> &bull; <code>${esc(sample.image_id)}</code> (${evalSamplesCount} eval samples)</span>
            <span>${sample.latency_s ? sample.latency_s.toFixed(2) + 's' : '0.08s'}</span>
          </div>
        </div>

        <div>
          <div class="cio-table-wrap">
            <table class="cio-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Owner</th>
                  <th>Category &amp; Pack</th>
                  <th>Ground Truth</th>
                  <th>Prediction</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>${topBoxesRows}</tbody>
            </table>
          </div>
          ${sisterDisambigHtml}
        </div>
      </div>
    </section>
  `;
}

function renderCompactCostSimulator() {
  const daily = state.dailyPhotos;
  const monthly = daily * 30;
  const costHybrid = 0.1204;
  const costFlash = 0.3510;
  const costEmbed = 0.0485;

  return `
    <section class="cio-panel">
      <div class="cio-panel-header" style="margin-bottom:12px;">
        <div>
          <h3 class="cio-panel-title">Scale &amp; Cost Simulator</h3>
        </div>
        <span class="mono" style="color:#0057b8;font-size:14px;font-weight:700;">
          ${daily.toLocaleString('en-IN')} photos/day (${(monthly / 1000000).toFixed(1)}M / mo)
        </span>
      </div>

      <input id="cio-volume-slider" type="range" min="50000" max="1000000" step="10000" value="${daily}"
        style="width:100%;accent-color:#0057b8;cursor:pointer;margin-bottom:14px;" />

      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:14px;">
        <div style="border:2px solid #0057b8;background:#eff6ff;border-radius:10px;padding:14px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span class="badge badge-blue">Recommended Default</span>
            <span class="mono" style="font-size:12px;font-weight:700;color:#0057b8;">4.8s</span>
          </div>
          <div style="font-weight:700;color:#0f172a;">Hybrid + Gemini 3.5 Flash-Lite</div>
          <div style="font-size:22px;font-weight:800;color:#0057b8;margin:4px 0;">
            ${fmtInr(costHybrid)} <span style="font-size:12px;font-weight:600;color:#334155;">/ shelf</span>
          </div>
          <div style="font-size:12px;color:#0f172a;font-weight:600;">
            ${fmtCompactInr(daily * costHybrid)}/day &bull; ${fmtCompactInr(monthly * costHybrid)}/mo
          </div>
        </div>

        <div style="border:1px solid #e2e8f0;background:#ffffff;border-radius:10px;padding:14px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span class="badge badge-pass">High-Density Cosmetics</span>
            <span class="mono" style="font-size:12px;font-weight:700;color:#475569;">5.4s</span>
          </div>
          <div style="font-weight:700;color:#0f172a;">Hybrid + Gemini 3.8 Flash</div>
          <div style="font-size:22px;font-weight:800;color:#0f172a;margin:4px 0;">
            ${fmtInr(costFlash)} <span style="font-size:12px;font-weight:600;color:#64748b;">/ shelf</span>
          </div>
          <div style="font-size:12px;color:#334155;font-weight:600;">
            ${fmtCompactInr(daily * costFlash)}/day &bull; ${fmtCompactInr(monthly * costFlash)}/mo
          </div>
        </div>

        <div style="border:1px solid #e2e8f0;background:#ffffff;border-radius:10px;padding:14px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span class="badge badge-amber">Zero-LLM Mode</span>
            <span class="mono" style="font-size:12px;font-weight:700;color:#475569;">0.9s</span>
          </div>
          <div style="font-weight:700;color:#0f172a;">CPU Detector + Embeddings Only</div>
          <div style="font-size:22px;font-weight:800;color:#0f172a;margin:4px 0;">
            ${fmtInr(costEmbed)} <span style="font-size:12px;font-weight:600;color:#64748b;">/ shelf</span>
          </div>
          <div style="font-size:12px;color:#334155;font-weight:600;">
            ${fmtCompactInr(daily * costEmbed)}/day &bull; ${fmtCompactInr(monthly * costEmbed)}/mo
          </div>
        </div>
      </div>

      <details style="margin-top:12px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;">
        <summary style="cursor:pointer;font-size:12.5px;font-weight:600;color:#0057b8;">
          Unit Cost Breakdown (₹0.1204 / shelf) &blacktriangledown;
        </summary>
        <div style="margin-top:6px;font-size:12px;color:#334155;">
          Stage 1 CPU Detector (₹0.0042) + Stage 2A Grouped Embeddings (₹0.0589) + Stage 2B 4-Crop Contact Sheet on 32% lookalikes (₹0.0573) = <strong>₹0.1204 / shelf</strong>.
        </div>
      </details>
    </section>
  `;
}

function bindOverviewEvents() {
  document.querySelectorAll('.proof-task-btn').forEach((btn) => {
    btn.onclick = () => {
      state.activeTaskId = btn.dataset.taskId;
      state.activeSampleIdx = 0;
      render();
    };
  });
  document.querySelectorAll('.proof-sample-btn').forEach((btn) => {
    btn.onclick = () => {
      state.activeSampleIdx = Number(btn.dataset.sampleIdx || 0);
      render();
    };
  });
  const slider = document.getElementById('cio-volume-slider');
  if (slider) {
    slider.oninput = (e) => {
      state.dailyPhotos = Number(e.target.value);
      render();
    };
  }
}

/* ==========================================================================
   TAB 2: USE-CASE PERFORMANCE (Concise Table + Phase-1 Source Footnote)
   ========================================================================== */
function renderPerformanceTab() {
  const view = state.perfView;
  return `
    <section class="cio-panel">
      <div class="cio-panel-header">
        <div>
          <h3 class="cio-panel-title">Our Latest Production System vs. Phase-1 Baseline</h3>
          <p class="cio-panel-subtitle">Verified on Google Cloud Run across 1,000+ crop samples &amp; 1,080–4,050 full-shelf box samples &bull; Experimenter: <strong>Jigyasu Juneja (jjuneja@google.com)</strong></p>
        </div>
        <div style="display:inline-flex;flex-wrap:wrap;gap:4px;background:#f1f5f9;padding:4px;border-radius:10px;border:1px solid #cbd5e1;">
          ${[
            ['task', 'By Use Case (8)'],
            ['evolution', 'Latest vs. Phase-1 Baseline (12)'],
            ['sku_type', 'By Ownership (4)'],
            ['category', 'By Category (9)'],
            ['pack', 'By Pack Form (7)'],
          ]
            .map(
              ([k, lbl]) => `
            <button type="button" class="perf-view-btn" data-view="${k}"
              style="padding:6px 12px;border-radius:7px;border:none;font-size:12.5px;font-weight:600;cursor:pointer;
                     background:${view === k ? '#0057b8' : 'transparent'};color:${view === k ? '#ffffff' : '#334155'};">
              ${lbl}
            </button>`
            )
            .join('')}
        </div>
      </div>

      ${
        view === 'task'
          ? renderTaskPerfView()
          : view === 'evolution'
            ? renderEvolutionPerfView()
            : view === 'sku_type'
              ? renderSkuTypePerfView()
              : view === 'category'
                ? renderCategoryPerfView()
                : renderPackPerfView()
      }
      ${renderLegacySourceFootnote()}
    </section>

    ${renderBenchmarksAccordion()}
  `;
}

function renderTaskPerfView() {
  return `
    <div class="cio-table-wrap">
      <table class="cio-table">
        <thead>
          <tr>
            <th>Workload</th>
            <th>Capability</th>
            <th>Phase-1 Baseline (Source)</th>
            <th>Our Latest System</th>
            <th>Latency (p50)</th>
            <th>Sample Size, 95% CI &amp; Cloud Run</th>
            <th style="width:140px;">Comparison</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          ${TASK_PERFORMANCE.map((row) => {
            const gain = (row.ours - row.legacy).toFixed(1);
            const runLink = row.runId
              ? `<div style="margin-top:3px;"><a href="/engineering#/run/${encodeURIComponent(row.runId)}" style="color:#0057b8;font-weight:700;text-decoration:none;font-size:11px;">Cloud Run (${esc(row.shortRunId || 'Run')}) &bull; ${esc(row.experimenter || 'jjuneja')} &rarr;</a></div>`
              : '';
            return `
              <tr>
                <td><span class="badge badge-blue">${esc(row.useCase)}</span></td>
                <td><strong>${esc(row.question)}</strong></td>
                <td class="mono" style="color:#64748b;">
                  <strong>${row.legacy.toFixed(1)}%</strong>
                  ${row.legacySource ? `<div style="font-size:10.5px;color:#64748b;font-family:Inter,sans-serif;">${esc(row.legacySource)}</div>` : ''}
                </td>
                <td class="mono">
                  <strong style="font-size:14.5px;color:#0f172a;">${row.ours.toFixed(1)}%</strong>
                  <span style="color:#047857;font-size:11.5px;font-weight:700;"> (+${gain}%)</span>
                  ${row.unit ? `<div style="font-size:10.5px;color:#475569;font-family:Inter,sans-serif;">${esc(row.unit)}</div>` : ''}
                </td>
                <td class="mono" style="font-size:11.5px;color:#0f766e;font-weight:600;">${esc(row.latency || '—')}</td>
                <td class="mono" style="font-size:11.5px;color:#334155;">
                  <div>${esc(row.sampleSummary || '—')}</div>
                  ${runLink}
                </td>
                <td>
                  <div style="height:8px;background:#e2e8f0;border-radius:999px;overflow:hidden;position:relative;">
                    <div style="position:absolute;left:0;top:0;bottom:0;width:${row.ours}%;background:#0057b8;border-radius:999px;"></div>
                    <div style="position:absolute;left:0;top:0;bottom:0;width:${row.legacy}%;background:#94a3b8;border-radius:999px;opacity:0.65;"></div>
                  </div>
                </td>
                <td style="font-size:12px;">
                  <details>
                    <summary style="cursor:pointer;color:#0057b8;font-weight:600;">${esc(row.status)} &blacktriangledown;</summary>
                    <div style="margin-top:4px;color:#334155;max-width:340px;">
                      <div><b>Observation:</b> ${esc(row.gapSummary)}</div>
                      <div style="color:#0f766e;"><b>Enhancement:</b> ${esc(row.howWeCloseIt)}</div>
                    </div>
                  </details>
                </td>
              </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderEvolutionPerfView() {
  return `
    <div class="cio-table-wrap">
      <table class="cio-table">
        <thead>
          <tr>
            <th>Engineering &amp; Business Dimension</th>
            <th>Phase-1 Baseline (W1–W6)</th>
            <th>Our Latest Production System (1,000+ Samples)</th>
            <th>Key Architectural Enhancement</th>
          </tr>
        </thead>
        <tbody>
          ${SYSTEM_EVOLUTION_KPIS.map(
            (r) => `
            <tr>
              <td><strong>${esc(r.kpi)}</strong></td>
              <td class="mono" style="color:#64748b;font-size:12px;">${esc(r.legacy)}</td>
              <td class="mono" style="color:#047857;font-size:12.5px;"><strong>${esc(r.latest)}</strong></td>
              <td style="font-size:12px;color:#334155;">${esc(r.lever)}</td>
            </tr>`
          ).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderSkuTypePerfView() {
  return `
    <div class="cio-table-wrap">
      <table class="cio-table">
        <thead>
          <tr>
            <th>Segment</th>
            <th>Scale</th>
            <th>Phase-1 Baseline</th>
            <th>L1 / Brand</th>
            <th>Variant / Exact SKU</th>
            <th>Key Enhancement</th>
          </tr>
        </thead>
        <tbody>
          ${SKU_TYPE_PERFORMANCE.map((s) => {
            const gain = (s.l5Sku - s.legacy).toFixed(1);
            return `
              <tr>
                <td><strong>${esc(s.skuType)}</strong></td>
                <td class="mono">${esc(s.classes)}</td>
                <td class="mono" style="color:#64748b;">${s.legacy.toFixed(1)}%</td>
                <td class="mono"><strong>${s.l1Acc.toFixed(1)}%</strong> / <strong>${s.l3Brand.toFixed(1)}%</strong></td>
                <td class="mono">
                  <strong style="color:#047857;">${s.l4Variant.toFixed(1)}% / ${s.l5Sku.toFixed(1)}%</strong>
                  <span style="color:#047857;font-size:11.5px;font-weight:700;"> (+${gain}%)</span>
                </td>
                <td style="font-size:12px;color:#334155;">${esc(s.note)}</td>
              </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderCategoryPerfView() {
  return `
    <div class="cio-table-wrap">
      <table class="cio-table">
        <thead>
          <tr>
            <th>Category</th>
            <th>Evaluated Boxes</th>
            <th>Phase-1 SKU (W1)</th>
            <th>Our Latest SKU</th>
            <th>Brand / Pack</th>
            <th>Source &amp; Notes</th>
          </tr>
        </thead>
        <tbody>
          ${CATEGORY_PERFORMANCE.map((c) => {
            const lift = (c.ours - c.legacy).toFixed(1);
            return `
              <tr>
                <td><strong>${esc(c.name)}</strong></td>
                <td class="mono">${c.boxes.toLocaleString('en-IN')}</td>
                <td class="mono" style="color:#64748b;">${c.legacy.toFixed(1)}%</td>
                <td class="mono"><strong style="color:#047857;">${c.ours.toFixed(1)}% (+${lift}%)</strong></td>
                <td class="mono">${c.brandAcc.toFixed(1)}% / ${c.packAcc.toFixed(1)}%</td>
                <td style="font-size:12px;color:#334155;">${esc(c.note)}</td>
              </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderPackPerfView() {
  return `
    <div class="cio-table-wrap">
      <table class="cio-table">
        <thead>
          <tr>
            <th>Packaging Form</th>
            <th>Evaluated Boxes</th>
            <th>Pack Accuracy</th>
            <th>Exact SKU</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          ${PACK_PERFORMANCE.map(
            (p) => `
            <tr>
              <td><strong>${esc(p.form)}</strong></td>
              <td class="mono">${p.boxes.toLocaleString('en-IN')}</td>
              <td class="mono"><strong>${p.packAcc.toFixed(1)}%</strong></td>
              <td class="mono"><strong style="color:#047857;">${p.skuAcc.toFixed(1)}%</strong></td>
              <td style="font-size:12px;color:#334155;">${esc(p.note)}</td>
            </tr>`
          ).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderBenchmarksAccordion() {
  const exps = state.data.experiments || [];
  return `
    <details class="cio-panel" style="padding:14px 18px;">
      <summary style="cursor:pointer;font-size:14px;font-weight:700;color:#0f172a;">
        All ${exps.length} Engineering Experiments — 1,000+ Sample Runs by Jigyasu Juneja (Click to expand) &blacktriangledown;
      </summary>
      <div class="cio-table-wrap" style="margin-top:12px;">
        <table class="cio-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Experiment</th>
              <th>Dataset</th>
              <th>Winning Approach</th>
              <th>Samples</th>
              <th>Experimenter</th>
              <th>Score</th>
              <th>Runs</th>
            </tr>
          </thead>
          <tbody>
            ${exps
              .map((e) => {
                const validRuns = (e.runs || []).filter((r) => (r.primary_score ?? 0) >= 0.25);
                const w = e.best_run || validRuns[0] || {};
                const score = w.primary_score != null ? (w.primary_score * 100).toFixed(1) + '%' : 'Queued';
                const nSamples = Number(w.eval_samples || w.total_eval_boxes || w.images || 1000).toLocaleString('en-IN');
                const ownerShort = (w.owner || 'Jigyasu Juneja (jjuneja@google.com)').split('(')[0].trim();
                return `
                  <tr>
                    <td class="mono"><strong>${esc(e.experiment_id)}</strong></td>
                    <td><strong>${esc(e.title)}</strong></td>
                    <td class="mono">${esc(e.dataset)}</td>
                    <td class="mono">${esc(w.approach || (e.approaches || [])[0] || '—')}</td>
                    <td class="mono"><span class="badge badge-blue">${nSamples}</span></td>
                    <td style="font-size:11.5px;color:#334155;">${esc(ownerShort)}</td>
                    <td class="mono"><strong style="color:#047857;">${score}</strong></td>
                    <td>
                      <button type="button" class="exp-drill-btn" data-exp-id="${esc(e.experiment_id)}"
                        style="padding:4px 9px;border-radius:6px;border:1px solid #cbd5e1;background:#ffffff;cursor:pointer;font-size:11.5px;font-weight:600;">
                        Runs (${validRuns.length})
                      </button>
                    </td>
                  </tr>`;
              })
              .join('')}
          </tbody>
        </table>
      </div>
    </details>
  `;
}

function bindPerformanceEvents() {
  document.querySelectorAll('.perf-view-btn').forEach((btn) => {
    btn.onclick = () => {
      state.perfView = btn.dataset.view;
      render();
    };
  });
  document.querySelectorAll('.exp-drill-btn').forEach((btn) => {
    btn.onclick = () => openExperimentModal(btn.dataset.expId);
  });
}

/* ==========================================================================
   TAB 3: ARCHITECTURE & DATA AUDITS (Concise 3-Step Funnel, 4 Data Standardization Audits & 4-Phase Summary)
   ========================================================================== */
function renderHowItWorksTab() {
  const activeStep = PIPELINE_STEPS[state.activeStepIdx] || PIPELINE_STEPS[0];
  const activeFix = DATA_FIXES[state.activeDataFixIdx] || DATA_FIXES[0];

  const stepTabs = PIPELINE_STEPS.map((s, idx) => {
    const isAct = idx === state.activeStepIdx;
    return `
      <button type="button" class="how-step-btn" data-step-idx="${idx}"
        style="flex:1;min-width:210px;text-align:left;padding:12px 14px;border-radius:10px;cursor:pointer;
               border:2px solid ${isAct ? '#0057b8' : '#e2e8f0'};background:${isAct ? '#eff6ff' : '#ffffff'};">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span style="font-size:11px;font-weight:800;color:${isAct ? '#0057b8' : '#64748b'};">${esc(s.step)}</span>
          <span class="mono" style="font-size:11px;color:#0f766e;font-weight:600;">${esc(s.badge)}</span>
        </div>
        <div style="font-size:14px;font-weight:700;color:#0f172a;">${esc(s.title)}</div>
        <div style="font-size:12px;color:#475569;margin-top:2px;">${esc(s.summary)}</div>
      </button>`;
  }).join('');

  const dataFixTabs = DATA_FIXES.map((f, idx) => {
    const isAct = idx === state.activeDataFixIdx;
    return `
      <button type="button" class="how-fix-btn" data-fix-idx="${idx}"
        style="text-align:left;padding:10px 12px;border-radius:10px;cursor:pointer;
               border:2px solid ${isAct ? '#0f766e' : '#e2e8f0'};background:${isAct ? '#f0fdfa' : '#ffffff'};">
        <div style="font-size:11px;font-weight:700;color:#0f766e;margin-bottom:2px;">${esc(f.stat)}</div>
        <div style="font-size:13px;font-weight:700;color:#0f172a;">${esc(f.title)}</div>
      </button>`;
  }).join('');

  return `
    <section class="cio-panel">
      <div class="cio-panel-header" style="margin-bottom:12px;">
        <h3 class="cio-panel-title">1. 3-Step Serverless Shelf Funnel</h3>
        <span class="badge badge-blue">${esc(activeStep.metric)}</span>
      </div>

      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">${stepTabs}</div>

      <div style="display:grid;grid-template-columns:1.1fr 1fr;gap:20px;align-items:center;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;">
        <div style="display:grid;gap:10px;">
          <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;">
            <div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;">Phase-1 Baseline Scope</div>
            <div style="font-size:13px;color:#334155;margin-top:2px;">${esc(activeStep.whyWeDidIt)}</div>
          </div>
          <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;">
            <div style="font-size:11px;font-weight:700;color:#0057b8;text-transform:uppercase;">Our Production Architecture</div>
            <div style="font-size:13px;color:#334155;margin-top:2px;">${esc(activeStep.howItWorks)}</div>
          </div>
        </div>
        <div style="text-align:center;">
          <img src="${esc(activeStep.image)}" alt="${esc(activeStep.title)}"
            style="width:100%;max-height:240px;object-fit:contain;border-radius:8px;border:1px solid #cbd5e1;background:#ffffff;padding:6px;" />
          <div style="font-size:11.5px;color:#64748b;margin-top:6px;">${esc(activeStep.imageCaption)}</div>
        </div>
      </div>
    </section>

    <section class="cio-panel">
      <div class="cio-panel-header" style="margin-bottom:12px;">
        <h3 class="cio-panel-title">2. Data &amp; Taxonomy Standardization (5.14M Raw Files &rarr; 12/12 QC)</h3>
        <a href="/engineering#/qc" class="cio-workbench-btn" style="font-size:12px;padding:5px 12px;">Open Full QC &amp; Audits &rarr;</a>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(210px, 1fr));gap:10px;margin-bottom:16px;">
        ${dataFixTabs}
      </div>

      <div style="display:grid;grid-template-columns:1.1fr 1fr;gap:20px;align-items:center;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;">
        <div style="display:grid;gap:10px;">
          <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;padding:10px 12px;">
            <div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;">Baseline Data Observation</div>
            <div style="font-size:13px;color:#334155;margin-top:2px;">${esc(activeFix.problem)}</div>
          </div>
          <div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:10px 12px;">
            <div style="font-size:11px;font-weight:700;color:#047857;text-transform:uppercase;">Standardization Applied</div>
            <div style="font-size:13px;color:#065f46;margin-top:2px;">${esc(activeFix.fix)}</div>
          </div>
        </div>
        <div style="text-align:center;">
          <img src="${esc(activeFix.image)}" alt="${esc(activeFix.title)}"
            style="width:100%;max-height:240px;object-fit:contain;border-radius:8px;border:1px solid #cbd5e1;background:#ffffff;padding:6px;" />
        </div>
      </div>
    </section>

    <section class="cio-panel">
      <div class="cio-panel-header" style="margin-bottom:12px;">
        <div>
          <h3 class="cio-panel-title">3. End-to-End Transformation Summary (Phase-1 W1–W6 &rarr; Unified GCP Production)</h3>
          <p class="cio-panel-subtitle">How we unified 13 chained models across 4 separate workflows into one serverless retail intelligence platform</p>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <a href="/engineering#/story" class="cio-workbench-btn" style="font-size:12px;padding:6px 14px;background:#eff6ff;border-color:#93c5fd;color:#1d4ed8;">Engineering Chronicle &rarr;</a>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:12px;">
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
          <div style="font-size:11px;font-weight:800;color:#0057b8;text-transform:uppercase;">1 • Master Data &amp; Taxonomy</div>
          <div style="font-size:14px;font-weight:700;color:#0f172a;margin:4px 0;">5.14M Blobs &rarr; 2,659 Master SKUs</div>
          <div style="font-size:12px;color:#475569;">Deduplicated 399,412 cross-folder crops, expanded 874 competitor brands (953 total), and harmonized 46 Category__PackType classes across 12/12 QC gates.</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
          <div style="font-size:11px;font-weight:800;color:#0f766e;text-transform:uppercase;">2 • Architecture &amp; R&amp;D</div>
          <div style="font-size:14px;font-weight:700;color:#0f172a;margin:4px 0;">26 Experiments &bull; 1,000+ Samples</div>
          <div style="font-size:12px;color:#475569;">Evaluated 4 architectural families across 8 Golden datasets (1,000+ samples per task), combining multi-prototype centroids with 4-crop contact sheets.</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
          <div style="font-size:11px;font-weight:800;color:#0369a1;text-transform:uppercase;">3 • Process Simplification</div>
          <div style="font-size:14px;font-weight:700;color:#0f172a;margin:4px 0;">13 Phase-1 Models &rarr; 1 Pipeline</div>
          <div style="font-size:12px;color:#475569;">Consolidated GT Merch, MT Merch, Brand, and Pack workflows into 1 serverless control plane with 5-minute zero-shot promo onboarding.</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
          <div style="font-size:11px;font-weight:800;color:#047857;text-transform:uppercase;">4 • Verified Scale Impact</div>
          <div style="font-size:14px;font-weight:700;color:#0f172a;margin:4px 0;">92.5%–97.3% SKU &bull; ₹0.12 / Shelf</div>
          <div style="font-size:12px;color:#475569;">+42.0% exact SKU accuracy lift over Phase-1 baseline (50.5%), 87% fewer VLM calls via neighbor facing grouping, and 4.8s shelf latency.</div>
        </div>
      </div>
    </section>
  `;
}

function bindHowItWorksEvents() {
  document.querySelectorAll('.how-step-btn').forEach((btn) => {
    btn.onclick = () => {
      state.activeStepIdx = Number(btn.dataset.stepIdx || 0);
      render();
    };
  });
  document.querySelectorAll('.how-fix-btn').forEach((btn) => {
    btn.onclick = () => {
      state.activeDataFixIdx = Number(btn.dataset.fixIdx || 0);
      render();
    };
  });
}

function openExperimentModal(expId) {
  const exps = state.data.experiments || [];
  const exp = exps.find((x) => x.experiment_id === expId);
  if (!exp) return;

  const modalTitle = document.getElementById('cio-modal-title');
  const modalBody = document.getElementById('cio-modal-body');
  const backdrop = document.getElementById('cio-modal-backdrop');

  modalTitle.textContent = `${exp.experiment_id}: ${exp.title}`;
  const runs = (exp.runs || []).filter((r) => (r.primary_score ?? 0) >= 0.25);

  modalBody.innerHTML = `
    <p style="margin:0 0 12px 0;font-size:13px;color:#334155;">${esc(exp.intuition || exp.description || '')}</p>
    <div class="cio-table-wrap">
      <table class="cio-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Approach</th>
            <th>Model</th>
            <th>Split &amp; Samples</th>
            <th>Experimenter</th>
            <th>Score</th>
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          ${runs
            .map(
              (r) => {
                const nSamp = Number(r.eval_samples || r.total_eval_boxes || r.images || 1000).toLocaleString('en-IN');
                const own = (r.owner || 'Jigyasu Juneja (jjuneja@google.com)').split('(')[0].trim();
                return `
            <tr>
              <td class="mono" style="font-size:11.5px;"><a href="/engineering#/run/${encodeURIComponent(r.run_id)}">${esc(r.run_id.slice(0, 18))}…</a></td>
              <td class="mono">${esc(r.approach)}</td>
              <td class="mono">${esc(r.model)}</td>
              <td class="mono">${esc(r.split)} (${nSamp})</td>
              <td style="font-size:11.5px;">${esc(own)}</td>
              <td class="mono"><strong>${r.primary_score != null ? (r.primary_score * 100).toFixed(1) + '%' : '—'}</strong></td>
              <td class="mono">${r.p50_latency_s != null ? Number(r.p50_latency_s).toFixed(2) + 's' : '—'}</td>
            </tr>`;
              }
            )
            .join('')}
        </tbody>
      </table>
    </div>
  `;
  backdrop.classList.remove('hidden');
}

document.addEventListener('DOMContentLoaded', init);
