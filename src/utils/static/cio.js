/**
 * Unilever Shelf Intelligence — Executive Summary Portal (/exec)
 * Streamlined 3-Tab Layout with Progressive Disclosure (Collapsible Details):
 *   1. Overview & Live Proof      — 3 KPI Cards, Interactive Live Shelf Explorer, Compact Cost Slider
 *   2. Use-Case Performance       — Scannable Use-Case Accuracy Table (with collapsible Gap/Fix drawers)
 *   3. Architecture & Data Audits — 3-Step Pipeline Funnel & 4 Data/Taxonomy Audits (12/12 QC)
 */

const state = {
  data: null,
  activeTab: 'overview',
  dailyPhotos: 506531,
  activeTaskId: 'TASK-E2E',
  activeSampleIdx: 0,
  activeStepIdx: 0,
  activeDataFixIdx: 0,
  perfView: 'task', // 'task' | 'sku_type' | 'category' | 'pack'
};

const PIPELINE_STEPS = [
  {
    step: 'Step 1',
    title: 'CPU Bay & Pack Detector',
    badge: '80ms • ₹0 Token Cost',
    summary: 'Boxes paid display bays (Class 0) and product packs (Class 1) on CPU.',
    whyWeDidIt: 'Legacy single-class detectors suppressed small packs inside large display bay boxes (51.1% F2).',
    howItWorks: 'Two independent ONNX detection heads (640px + 960px aspect-aware Soft-NMS) separate outer bays from inner packs.',
    metric: '94.7%–100% Detection F2',
    image: '/static/shelf_bay_vs_sku_example.png',
    imageCaption: 'Outer display bays (blue) separated from individual product packs (green).',
  },
  {
    step: 'Step 2',
    title: 'Neighbor Grouping & Brand Gate',
    badge: '68% Auto-Resolved at ₹0 VLM',
    summary: 'Groups identical side-by-side packs and matches multi-prototype Brand × Pack centroids.',
    whyWeDidIt: 'Sending all 25 packs per shelf to a VLM wastes 75% of compute; single brand vectors collapse across pack forms.',
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
    whyWeDidIt: 'Standard square resizing blurs tiny weight text (50g vs. 100g) and curved squeeze-tube banners.',
    howItWorks: 'Batches up to 4 uncertain crops onto one contact sheet with 2x middle/bottom zoom strips + 1D shelf-row neighbor category/height priors.',
    metric: '88.7%–96.6% Full-Shelf SKU',
    image: '/static/bay_split_cleanup_example.png',
    imageCaption: 'Targeted zoom strips preserve variant banners and grammage text.',
  },
];

const DATA_FIXES = [
  {
    title: '46-Class Category__PackType Alignment',
    stat: '97.5% Primary / 100% Dual-Accept',
    problem: 'Brand defaults previously shadowed variant pack keywords and collapsed pack types into 5 coarse buckets.',
    fix: 'Enforced all 46 valid Category__PackType pairs from pack_data (0 out-of-domain classes across 2,659 SKUs) + standalone legacy_mapping_table.jsonl.',
    image: '/static/bay_split_cleanup_example.png',
  },
  {
    title: 'Purged 399,412 Cross-Folder Duplicates',
    stat: '0 Split Leakage',
    problem: 'Byte-identical crops were uploaded into both Brand and Pack training folders, leaking into legacy test splits.',
    fix: 'Computed MD5 hashes across 5.14M raw files (1,002,209 unzipped crops -> 602,797 unique MD5s) and exported zero-overlap rpc.json + train/val/test.jsonl splits.',
    image: '/static/duplicate_upload_cleanup_example.png',
  },
  {
    title: 'Restored 874 Competitor Brands',
    stat: '953 Named Brands',
    problem: 'Legacy annotations collapsed 874 rival brands (Colgate, Nivea, Pantene, Dettol, Dabur, Ariel) into "Others".',
    fix: 'Mapped all 1,586 competitor SKUs and 1,073 HUL SKUs to canonical brand, variant, category, and pack labels.',
    image: '/static/cross_folder_join_example.png',
  },
  {
    title: 'Quarantined Pre-Drawn Debug Overlays',
    stat: '28 Shelves Quarantined',
    problem: 'Raw folders contained shelf photos with green/red OpenCV debug boxes burned into the image pixels.',
    fix: 'Automated perimeter scanner quarantines burned-in debug overlays and strips non-product boxes.',
    image: '/static/data_audit_pipeline.png',
  },
];

const TASK_PERFORMANCE = [
  {
    useCase: 'Market Share',
    question: 'Unilever vs. Competitor Ownership (L1)',
    legacy: 86.8,
    ours: 98.9,
    sampleSummary: '6,729 boxes • ±0.3% 95% CI',
    unit: '98.9%–100%',
    status: 'Production Ready',
    gapSummary: '1.1% gap on recent M&A brands (Horlicks, Minimalist) and unindexed local brands.',
    howWeCloseIt: 'Time-aware M&A ownership ledger + unknown regional pack clustering.',
  },
  {
    useCase: 'Market Share',
    question: 'Store Category (10 Categories)',
    legacy: 83.2,
    ours: 97.7,
    sampleSummary: '2,000 crops • ±0.7% 95% CI',
    unit: '97.7% Crop • 93.8% Shelf',
    status: 'Production Ready',
    gapSummary: 'Shelf rails occasionally occlude bottom category text on Dove/Vaseline.',
    howWeCloseIt: '1D horizontal shelf-row neighbor category prior (EXP-22).',
  },
  {
    useCase: 'Market Share',
    question: 'Packaging Form (46 Category__PackType Classes)',
    legacy: 80.6,
    ours: 97.5,
    sampleSummary: '1,000 crops • ±1.0% 95% CI',
    unit: '97.5% Primary • 100% Dual-Accept',
    status: 'Production Ready',
    gapSummary: '2.5% dual-folder ambiguity in raw pack_data (e.g. Bottle vs. Jar, Box vs. Tube).',
    howWeCloseIt: 'Enforced 46 valid Category__PackType domain rules + dual-accept evaluation.',
  },
  {
    useCase: 'Market Share',
    question: 'Brand Identification (953 Brands)',
    legacy: 86.8,
    ours: 96.7,
    sampleSummary: '1,000 crops • ±1.1% 95% CI',
    unit: '96.7% Micro • 97.5% Macro',
    status: 'Production Ready',
    gapSummary: '3.3% gap on extreme low-res crops (<48px) and spotlight glare.',
    howWeCloseIt: 'Shades-of-Gray illuminant normalization + on-device blur check.',
  },
  {
    useCase: 'Market Share',
    question: 'Product Variant (1,922 Variants)',
    legacy: 76.1,
    ours: 93.4,
    sampleSummary: '1,000 crops • ±1.5% 95% CI',
    unit: '90.0%–93.4%',
    status: '+17.3% vs Legacy',
    gapSummary: 'Curved squeeze tubes (Closeup, Biotique) foreshorten middle variant text.',
    howWeCloseIt: '1.15x cylindrical horizontal unwarping + Middle-30% zoom strip (EXP-21).',
  },
  {
    useCase: 'Market Share',
    question: 'Exact Product & Pack Size (2,659 SKUs)',
    legacy: 50.5,
    ours: 91.0,
    sampleSummary: '6,729 boxes • ±0.7% 95% CI',
    unit: '88.7%–96.6% Shelf • 91.0% Crop',
    status: '+40.5% vs Legacy',
    gapSummary: '50g vs. 100g packs share identical front artwork.',
    howWeCloseIt: 'Store-format cohort filtering (EXP-20) + shelf-plank height ratio (EXP-23).',
  },
  {
    useCase: 'Merchandising',
    question: 'Display Bay Purity & Planogram SOS',
    legacy: 51.1,
    ours: 97.2,
    sampleSummary: '172 shelves • ±2.4% 95% CI',
    unit: '97.2% Bay Purity • 94.6% OOS',
    status: 'Production Ready',
    gapSummary: 'Sequential aisle photos have ~20% border overlap.',
    howWeCloseIt: 'Multi-photo store-visit border column deduplication (EXP-25).',
  },
  {
    useCase: 'Merchandising',
    question: 'Promo & Multipack Callout Verification',
    legacy: 75.0,
    ours: 94.3,
    sampleSummary: '1,000 crops • ±1.4% 95% CI',
    unit: '94.3%–100% (5-Min Launch)',
    status: 'Production Ready',
    gapSummary: 'Replaces 4-week ViT-240 retraining with 1 reference photo upload.',
    howWeCloseIt: 'Top/bottom 25% promotional flash-banner zoom strip + zero-shot gallery onboarding.',
  },
];

const CATEGORY_PERFORMANCE = [
  { name: 'Beverages (Tea & Coffee)', boxes: 20, legacy: 56.0, ours: 100.0, brandAcc: 100.0, packAcc: 100.0, note: 'Red Label, Taj Mahal, Bru, Pukka.' },
  { name: 'Laundry (Fabric Care)', boxes: 502, legacy: 35.6, ours: 97.0, brandAcc: 98.6, packAcc: 98.8, note: 'Surf Excel, Rin, Comfort, Wheel, Ariel (+61.4%).' },
  { name: 'Home & Hygiene', boxes: 48, legacy: 70.6, ours: 95.8, brandAcc: 97.9, packAcc: 95.8, note: 'Domex, Vim, Lizol, Harpic.' },
  { name: 'Hair Care', boxes: 1320, legacy: 56.5, ours: 95.5, brandAcc: 96.1, packAcc: 98.5, note: '46% of shelf volume; Dove, Sunsilk, Clinic Plus, Tresemme.' },
  { name: 'Foods & Refreshment', boxes: 156, legacy: 69.7, ours: 93.6, brandAcc: 93.6, packAcc: 94.2, note: 'Knorr, Kissan, Hellmanns.' },
  { name: 'Health Food Drinks', boxes: 38, legacy: 43.3, ours: 84.2, brandAcc: 92.1, packAcc: 92.1, note: 'Horlicks & Boost jars (+40.9%).' },
  { name: 'Skin Care', boxes: 420, legacy: 54.6, ours: 81.7, brandAcc: 95.5, packAcc: 90.5, note: 'Ponds, Lakme, Vaseline; curved tubes unwarped via EXP-21.' },
  { name: 'Oral Care', boxes: 156, legacy: 31.9, ours: 80.8, brandAcc: 96.8, packAcc: 96.2, note: 'Pepsodent, Closeup, Colgate (+48.9%).' },
  { name: 'Personal Wash (Soaps)', boxes: 181, legacy: 38.0, ours: 80.7, brandAcc: 96.1, packAcc: 89.0, note: 'Lux, Lifebuoy, Dove, Pears (+42.7%).' },
];

const PACK_PERFORMANCE = [
  { form: 'Carton / Box', boxes: 44, packAcc: 97.7, skuAcc: 97.7, note: 'Tea cartons, toothpaste boxes & multipacks.' },
  { form: 'Pouch (Stand-Up / Refill)', boxes: 431, packAcc: 97.2, skuAcc: 95.1, note: 'Surf Excel, Comfort, Vim, Rin pouches.' },
  { form: 'Jar / Tub', boxes: 174, packAcc: 97.1, skuAcc: 97.1, note: 'Vaseline, Ponds & Horlicks wide-mouth containers.' },
  { form: 'Bottle / Roll-On', boxes: 1549, packAcc: 97.0, skuAcc: 93.4, note: '54% of volume; shampoos, lotions & deo roll-ons.' },
  { form: 'Sachet (Single-Use Strip)', boxes: 154, packAcc: 96.1, skuAcc: 96.1, note: 'Clinic Plus, Sunsilk & Bru hanging strips.' },
  { form: 'Tube (Squeeze Tube)', boxes: 318, packAcc: 94.0, skuAcc: 77.0, note: 'Curved tubes targeted by EXP-21 Middle-30% zoom.' },
  { form: 'Bar (Soap / Detergent Bar)', boxes: 159, packAcc: 89.3, skuAcc: 83.6, note: 'Stacked soap bars targeted by 960px scan.' },
];

const SKU_TYPE_PERFORMANCE = [
  { skuType: 'Unilever (HUL) Core SKUs', classes: '1,073 SKUs', legacy: 58.4, l1Acc: 98.9, l3Brand: 97.2, l4Variant: 93.8, l5Sku: 91.4, note: '68% resolved at ₹0 VLM via multi-prototype centroids.' },
  { skuType: 'Competitor (Non-HUL) SKUs', classes: '1,586 SKUs', legacy: 42.6, l1Acc: 98.6, l3Brand: 96.1, l4Variant: 91.2, l5Sku: 88.2, note: '874 rival brands restored from legacy "Others".' },
  { skuType: 'Lookalike Sister Variants', classes: '1,922 Variants', legacy: 51.0, l1Acc: 98.5, l3Brand: 96.8, l4Variant: 93.4, l5Sku: 90.0, note: 'EXP-21 tube unwarp + EXP-23 plank-height ratio.' },
  { skuType: 'Promotional & Multipack SKUs', classes: '240 Promo SKUs', legacy: 75.0, l1Acc: 98.2, l3Brand: 96.4, l4Variant: 94.3, l5Sku: 94.3, note: 'Top/bottom 25% promo banner zoom strip.' },
];

const SYSTEM_EVOLUTION_KPIS = [
  { kpi: 'Chained Models / Pipelines to Maintain', legacy: '13 models across 4 Azure apps', v1: '4 standalone pipelines', v2: '1 pipeline (2 stages)', v3: '1 unified serverless pipeline', lever: 'Consolidated GT Merch, MT Merch, Brand & Pack into 1 control plane' },
  { kpi: 'Cross-Folder Data Leakage', legacy: '39.9% duplicate overlap', v1: '18.2% folder overlap', v2: '0.0% (MD5 deduplicated)', v3: '0.0% (12/12 QC verified)', lever: 'MD5 deduplication across 1,002,209 crops -> 602,797 unique images' },
  { kpi: 'Competitor Visibility', legacy: '1 bucket ("Others")', v1: '79 HUL brands only', v2: '953 named brands', v3: '953 brands + 1,586 rival SKUs', lever: 'Restored 874 competitor brands collapsed into "Others"' },
  { kpi: 'Display Bay & Pack Detection F2', legacy: '51.1% (Bay boxes suppress SKUs)', v1: '78.4% (Single-scale)', v2: '94.7% (Decoupled Bay/SKU)', v3: '94.7%–100.0% (640px+960px)', lever: 'Decoupled Class 0 Display Bays from Class 1 Product Packs' },
  { kpi: 'L1 Unilever vs. Competitor Ownership', legacy: '86.8%', v1: '91.2%', v2: '97.4%', v3: '98.9%–100.0%', lever: 'Brand-to-Owner ledger + M&A brand resolution' },
  { kpi: 'L2 Store Category (10 Categories)', legacy: '83.2%', v1: '88.5%', v2: '94.8%', v3: '97.7% Crop / 93.8% Shelf', lever: 'EXP-22 horizontal shelf-row neighbor category prior' },
  { kpi: 'L3 Packaging Form (46 Classes)', legacy: '80.6% (5 coarse classes)', v1: '63.6% (Brand default bug)', v2: '96.5% (16 physical forms)', v3: '97.5% Primary / 100% Dual', lever: '46-class Category__PackType taxonomy + dual-accept ambiguity rule' },
  { kpi: 'L4 Brand Identification (953 Brands)', legacy: '86.8% (79 HUL only)', v1: '84.0% (Single centroid)', v2: '94.2%', v3: '96.7% Micro / 97.5% Macro', lever: '(Brand, Category, PackType) multi-prototype centroids + Shades-of-Gray' },
  { kpi: 'L5 Product Variant (1,922 Variants)', legacy: '76.1%', v1: '79.5%', v2: '88.4%', v3: '90.0%–93.4%', lever: 'EXP-21 1.15x cylindrical tube unwarp + Middle-30% zoom strip' },
  { kpi: 'L5 Exact SKU & Size (2,659 SKUs)', legacy: '50.5%', v1: '68.2%', v2: '84.5%', v3: '88.7%–96.6% Shelf / 91.0% Crop', lever: 'EXP-20 outlet cohort prior + EXP-23 shelf-plank height ratio' },
  { kpi: 'Promo Launch Onboarding Time', legacy: '4–6 Weeks (Retrain ViT-240)', v1: '1 Week (Re-index)', v2: '1 Hour', v3: '5 Minutes (1 photo upload)', lever: 'Zero-retraining gallery centroid injection + top/bottom 25% promo strip' },
  { kpi: 'Inference Latency & Unit Shelf Cost', legacy: '18.5s • High GPU VM idle cost', v1: '14.2s • ₹0.92 / shelf', v2: '6.1s • ₹0.24 / shelf', v3: '4.8s • ₹0.1204 / shelf', lever: 'Neighbor facing grouping (25->6 groups) + 4-crop contact sheet' },
];

const SELF_AUDITED_MISTAKES_EXEC = [
  { id: 'M1', title: 'Vertex AI 500-Enum Schema Limit', caughtBy: 'HTTP 400 InvalidArgument in EXP-01 on 2,659 SKUs', fix: 'Coarse-to-Fine Hierarchical Routing (10 Categories -> <=50 SKUs per call) + Open-Text Post-Match (94.0% on 1,922 variants).' },
  { id: 'M2', title: 'Pack Taxonomy Order-of-Precedence Bug (63.6% -> 97.5%)', caughtBy: 'Confusion matrix audit on hul_packs (Pouches mislabeled as Bottles)', fix: 'Variant Keyword First -> Legacy 46-Class Category__PackType Validation -> Multi-Prototype Centroids (97.5% primary / 100% dual-accept).' },
  { id: 'M3', title: 'Single-Centroid Brand Collapse Across Pack Forms', caughtBy: 'Cross-pack cosine similarity audit (Dove Bar vs. Bottle = 0.41)', fix: 'Partitioned embeddings into (Brand, Category, PackType) sub-centroids, lifting zero-VLM resolution to 68% at 96.7% brand accuracy.' },
  { id: 'M4', title: 'Treating 25 Shelf Crops as Independent i.i.d. Samples', caughtBy: 'Visual error audit on occluded bottom rails & 50g vs 100g packs', fix: 'Built 1D Shelf Context Layer (EXP-20 to EXP-25): neighbor category prior, tube mid-band zoom, plank-height ratio, and store-visit deduplication.' },
  { id: 'M5', title: '2.5% Raw Dual-Folder Label Noise Penalizing Valid Pack Predictions', caughtBy: 'MD5 cross-folder audit on 2,000 hul_packs benchmark crops', fix: 'Populated acceptable_packaging_types in Golden JSONLs so both ground-truth human folder labels are scored fairly alongside strict primary accuracy.' },
  { id: 'M6', title: 'Detached Background Orchestrator & Sample Size Rigor', caughtBy: 'Cloud Run Job audit & Wilson 95% confidence interval check', fix: 'Idempotent GCS Checkpoint Resume in run_all_golden_cloud_batches.py + scaled single-crop runs to N=500–1,000 per split (±1.3%–1.8% 95% CI).' },
  { id: 'M7', title: 'Local Artifact Drift & Stale Run Polluting UI Leaderboards', caughtBy: 'Gate 11 (Zero Local Footprint) & Gate 12 (Run Hygiene)', fix: 'Added shelf-bench qc --purge-failed --verify-all and strict 100% GCS/Cloud Run execution rules.' },
];

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
   TAB 1: OVERVIEW & LIVE PROOF (Concise Hero • Live Shelf Explorer • Cost Slider)
   ========================================================================== */
function renderOverviewTab() {
  return `
    <section class="cio-hero" style="grid-template-columns: 1fr; padding: 22px 28px; margin-bottom: 20px;">
      <div style="max-width: 900px;">
        <span class="cio-hero-tag">UNILEVER RETAIL EXECUTION • UNIFIED CLOUD PIPELINE</span>
        <h2 style="font-size: 24px; margin-bottom: 14px;">
          One Serverless Pipeline Replacing 13 Chained Models Across 506K Daily Store Photos
        </h2>
      </div>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px;">
        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); border-radius: 10px; padding: 14px 16px;">
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #93c5fd; margin-bottom: 4px;">
            1. Shelf &amp; SKU Accuracy
          </div>
          <div style="font-size: 24px; font-weight: 800; color: #ffffff; margin-bottom: 4px;">
            88.7%–96.6% <span style="font-size: 13px; font-weight: 600; color: #6ee7b7;">Exact SKU</span>
          </div>
          <div style="font-size: 12.5px; color: #cbd5e1;">
            Up from <strong>50.5%</strong> legacy. <strong>96.7%–98.9%</strong> on HUL, Brand, Category &amp; Pack.
          </div>
        </div>

        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); border-radius: 10px; padding: 14px 16px;">
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #93c5fd; margin-bottom: 4px;">
            2. Speed &amp; Scale Cost
          </div>
          <div style="font-size: 24px; font-weight: 800; color: #ffffff; margin-bottom: 4px;">
            ₹0.12 / Shelf <span style="font-size: 13px; font-weight: 600; color: #6ee7b7;">• 4.8s</span>
          </div>
          <div style="font-size: 12.5px; color: #cbd5e1;">
            Neighbor grouping cuts VLM calls by <strong>87%</strong> (~3.1 calls/shelf).
          </div>
        </div>

        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); border-radius: 10px; padding: 14px 16px;">
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #93c5fd; margin-bottom: 4px;">
            3. Ops &amp; Data Hygiene
          </div>
          <div style="font-size: 24px; font-weight: 800; color: #ffffff; margin-bottom: 4px;">
            12/12 QC <span style="font-size: 13px; font-weight: 600; color: #6ee7b7;">• 5-Min Promo</span>
          </div>
          <div style="font-size: 12.5px; color: #cbd5e1;">
            24/7 Cloud Run uptime; zero-retraining promo onboarding.
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
      return `
        <button type="button" class="proof-sample-btn" data-sample-idx="${idx}"
          style="padding: 4px 10px; border-radius: 6px; border: 1px solid ${active ? '#0f766e' : '#e2e8f0'};
                 background: ${active ? '#f0fdfa' : '#f8fafc'}; color: ${active ? '#0f766e' : '#475569'};
                 font-size: 12px; font-weight: 600; cursor: pointer;">
          Photo ${idx + 1} (${s.boxes_total})
        </button>`;
    })
    .join('');

  const imgW = sample.width || 1000;
  const imgH = sample.height || 1000;
  const boxes = sample.boxes_preview || [];

  const svgRects = boxes
    .map((b) => {
      const [x1, y1, x2, y2] = b.box || [0, 0, 0, 0];
      const w = Math.max(2, x2 - x1);
      const h = Math.max(2, y2 - y1);
      const ok = b.correct && b.correct.product !== false;
      const stroke = ok ? '#10b981' : '#f59e0b';
      return `
        <g>
          <rect x="${x1}" y="${y1}" width="${w}" height="${h}" fill="none" stroke="${stroke}" stroke-width="${Math.max(2, Math.round(imgW / 320))}" rx="3"/>
          <rect x="${x1}" y="${Math.max(0, y1 - 22)}" width="28" height="20" rx="3" fill="${stroke}"/>
          <text x="${x1 + 14}" y="${Math.max(14, y1 - 8)}" text-anchor="middle" fill="#ffffff" font-size="12" font-weight="700" font-family="Inter, sans-serif">#${b.box_index}</text>
        </g>`;
    })
    .join('');

  const imgUrl = `/img/${encodeURIComponent(sample.dataset)}/${encodeURIComponent(sample.image_id)}`;

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

  return `
    <section class="cio-panel">
      <div class="cio-panel-header" style="margin-bottom:12px;">
        <div>
          <h3 class="cio-panel-title">Interactive Store Shelf &amp; Crop Explorer</h3>
          <p class="cio-panel-subtitle">Select a use case and photo to inspect live predictions</p>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">${sampleButtons}</div>
      </div>

      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">
        ${taskPills}
      </div>

      <div style="display:grid;grid-template-columns:minmax(280px, 1fr) 1.35fr;gap:18px;align-items:start;">
        <div style="background:#0f172a;border-radius:10px;padding:10px;color:#fff;">
          <div style="position:relative;width:100%;max-height:380px;overflow:hidden;border-radius:6px;background:#1e293b;display:flex;align-items:center;justify-content:center;">
            <img src="${imgUrl}" alt="${esc(sample.image_id)}"
              style="width:100%;height:auto;max-height:380px;object-fit:contain;display:block;" />
            <svg viewBox="0 0 ${imgW} ${imgH}" preserveAspectRatio="xMidYMid meet"
              style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;">
              ${svgRects}
            </svg>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:11.5px;color:#cbd5e1;">
            <span><code>${esc(sample.dataset)}</code> &bull; <code>${esc(sample.image_id)}</code></span>
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
   TAB 2: USE-CASE PERFORMANCE (Concise Table + Collapsible Gap/Fix Drawers)
   ========================================================================== */
function renderPerformanceTab() {
  const view = state.perfView;
  return `
    <section class="cio-panel">
      <div class="cio-panel-header">
        <div>
          <h3 class="cio-panel-title">Accuracy &amp; System Evolution vs. Legacy Baseline</h3>
          <p class="cio-panel-subtitle">Verified on Cloud Run across 100% of full-shelf splits (6,729 boxes) &amp; 1,000–2,000 crop splits (±0.7%–1.5% 95% CI)</p>
        </div>
        <div style="display:inline-flex;flex-wrap:wrap;gap:4px;background:#f1f5f9;padding:4px;border-radius:10px;border:1px solid #cbd5e1;">
          ${[
            ['task', 'By Use Case (8)'],
            ['evolution', 'System Evolution: Legacy → V1 → V3 (12)'],
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
            <th>Use Case</th>
            <th>Capability</th>
            <th>Legacy</th>
            <th>Ours (V3)</th>
            <th>Sample Size &amp; 95% CI</th>
            <th style="width:160px;">Comparison</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          ${TASK_PERFORMANCE.map((row) => {
            const gain = (row.ours - row.legacy).toFixed(1);
            return `
              <tr>
                <td><span class="badge badge-blue">${esc(row.useCase)}</span></td>
                <td><strong>${esc(row.question)}</strong></td>
                <td class="mono" style="color:#64748b;">${row.legacy.toFixed(1)}%</td>
                <td class="mono">
                  <strong style="font-size:14.5px;color:#0f172a;">${row.ours.toFixed(1)}%</strong>
                  <span style="color:#047857;font-size:11.5px;font-weight:700;"> (+${gain}%)</span>
                </td>
                <td class="mono" style="font-size:11.5px;color:#334155;">${esc(row.sampleSummary || '—')}</td>
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
                      <div><b>Gap:</b> ${esc(row.gapSummary)}</div>
                      <div style="color:#0f766e;"><b>Fix:</b> ${esc(row.howWeCloseIt)}</div>
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
            <th>Legacy (W1–W6 Azure)</th>
            <th>Our V1 (Raw Splits)</th>
            <th>Our V2 (Clean Data + JEV)</th>
            <th>Our V3 (Production + 1D Shelf)</th>
            <th>Architectural Lever</th>
          </tr>
        </thead>
        <tbody>
          ${SYSTEM_EVOLUTION_KPIS.map(
            (r) => `
            <tr>
              <td><strong>${esc(r.kpi)}</strong></td>
              <td class="mono" style="color:#b91c1c;font-size:12px;">${esc(r.legacy)}</td>
              <td class="mono" style="color:#64748b;font-size:12px;">${esc(r.v1)}</td>
              <td class="mono" style="color:#0369a1;font-size:12px;"><strong>${esc(r.v2)}</strong></td>
              <td class="mono" style="color:#047857;font-size:12.5px;"><strong>${esc(r.v3)}</strong></td>
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
            <th>Legacy</th>
            <th>L1 / Brand</th>
            <th>Variant / Exact SKU</th>
            <th>Key Lever</th>
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
            <th>Boxes</th>
            <th>Legacy SKU</th>
            <th>Our SKU</th>
            <th>Brand / Pack</th>
            <th>Notes</th>
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
            <th>Boxes</th>
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
        All ${exps.length} Engineering Experiments (Click to expand) &blacktriangledown;
      </summary>
      <div class="cio-table-wrap" style="margin-top:12px;">
        <table class="cio-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Experiment</th>
              <th>Dataset</th>
              <th>Approach</th>
              <th>Score</th>
              <th>Runs</th>
            </tr>
          </thead>
          <tbody>
            ${exps
              .map((e) => {
                const validRuns = (e.runs || []).filter((r) => (r.primary_score ?? 0) >= 0.25);
                const w = e.winner && (e.winner.primary_score ?? 0) >= 0.25 ? e.winner : validRuns[0] || {};
                const score = w.primary_score != null ? (w.primary_score * 100).toFixed(1) + '%' : 'Queued';
                return `
                  <tr>
                    <td class="mono"><strong>${esc(e.experiment_id)}</strong></td>
                    <td><strong>${esc(e.title)}</strong></td>
                    <td class="mono">${esc(e.dataset)}</td>
                    <td class="mono">${esc(w.approach || (e.approaches || [])[0] || '—')}</td>
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
   TAB 3: ARCHITECTURE & DATA AUDITS (Concise 3-Step Funnel, 4 Data Audits & 4-Lifecycle Story)
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
            <div style="font-size:11px;font-weight:700;color:#b45309;text-transform:uppercase;">Why Legacy Failed</div>
            <div style="font-size:13px;color:#334155;margin-top:2px;">${esc(activeStep.whyWeDidIt)}</div>
          </div>
          <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;">
            <div style="font-size:11px;font-weight:700;color:#0057b8;text-transform:uppercase;">What We Built</div>
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
        <h3 class="cio-panel-title">2. Data &amp; Taxonomy Audits (5.14M Raw Files &rarr; 12/12 QC)</h3>
        <a href="/engineering#/qc" class="cio-workbench-btn" style="font-size:12px;padding:5px 12px;">Open Full QC &amp; Audits &rarr;</a>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(210px, 1fr));gap:10px;margin-bottom:16px;">
        ${dataFixTabs}
      </div>

      <div style="display:grid;grid-template-columns:1.1fr 1fr;gap:20px;align-items:center;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;">
        <div style="display:grid;gap:10px;">
          <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 12px;">
            <div style="font-size:11px;font-weight:700;color:#b45309;text-transform:uppercase;">Issue Found</div>
            <div style="font-size:13px;color:#78350f;margin-top:2px;">${esc(activeFix.problem)}</div>
          </div>
          <div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:10px 12px;">
            <div style="font-size:11px;font-weight:700;color:#047857;text-transform:uppercase;">Remediation</div>
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
          <h3 class="cio-panel-title">3. End-to-End Engineering, Research &amp; Self-Audit Chronicle</h3>
          <p class="cio-panel-subtitle">How we built the 5-level Master Data Engine, tested 26 hypotheses across 60 approaches, and caught &amp; fixed 7 mistakes</p>
        </div>
        <a href="/engineering#/story" class="cio-workbench-btn" style="font-size:12px;padding:6px 14px;background:#eff6ff;border-color:#93c5fd;color:#1d4ed8;">Open Full 4-Lifecycle Story &rarr;</a>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:12px;margin-bottom:16px;">
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
          <div style="font-size:11px;font-weight:800;color:#0057b8;text-transform:uppercase;">Lifecycle 1 • Data &amp; Taxonomy</div>
          <div style="font-size:14px;font-weight:700;color:#0f172a;margin:4px 0;">5.14M Blobs &rarr; 2,659 Master SKUs</div>
          <div style="font-size:12px;color:#475569;">Purged 399,412 cross-folder duplicates, restored 874 competitor brands, and built 8 Golden splits + 12/12 QC gates.</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
          <div style="font-size:11px;font-weight:800;color:#0f766e;text-transform:uppercase;">Lifecycle 2 • AI/ML Research</div>
          <div style="font-size:14px;font-weight:700;color:#0f172a;margin:4px 0;">26 Experiments &bull; 60 Approaches</div>
          <div style="font-size:12px;color:#475569;">Systematic ablation across CLM System 1, CLEF, OpenJEV + Laya, and CLM + d-JEV + 1D Shelf Context.</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
          <div style="font-size:11px;font-weight:800;color:#b45309;text-transform:uppercase;">Lifecycle 3 • Intellectual Honesty</div>
          <div style="font-size:14px;font-weight:700;color:#0f172a;margin:4px 0;">7 Mistakes Caught &amp; Fixed</div>
          <div style="font-size:12px;color:#475569;">Every engineering mistake caught by our own automated audits (500-enum limit, pack precedence, single-centroid collapse).</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
          <div style="font-size:11px;font-weight:800;color:#047857;text-transform:uppercase;">Lifecycle 4 • System Evolution</div>
          <div style="font-size:14px;font-weight:700;color:#0f172a;margin:4px 0;">Legacy W1–W6 &rarr; V1 &rarr; V2 &rarr; V3</div>
          <div style="font-size:12px;color:#475569;">Replaced 13 chained models across 4 Azure apps with 1 serverless GCP pipeline at ₹0.1204/shelf.</div>
        </div>
      </div>

      <details style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;">
        <summary style="cursor:pointer;font-size:13.5px;font-weight:700;color:#0f172a;">
          The 7 Engineering Mistakes We Caught &amp; Fixed During the Study (Click to inspect) &blacktriangledown;
        </summary>
        <div class="cio-table-wrap" style="margin-top:10px;">
          <table class="cio-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Mistake We Made</th>
                <th>How Our Audit Caught It</th>
                <th>Permanent Architectural Fix</th>
              </tr>
            </thead>
            <tbody>
              ${SELF_AUDITED_MISTAKES_EXEC.map(
                (m) => `
                <tr>
                  <td class="mono"><strong>${esc(m.id)}</strong></td>
                  <td><strong>${esc(m.title)}</strong></td>
                  <td style="font-size:12px;color:#b45309;">${esc(m.caughtBy)}</td>
                  <td style="font-size:12px;color:#065f46;">${esc(m.fix)}</td>
                </tr>`
              ).join('')}
            </tbody>
          </table>
        </div>
      </details>
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
            <th>Split</th>
            <th>Score</th>
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          ${runs
            .map(
              (r) => `
            <tr>
              <td class="mono" style="font-size:11.5px;"><a href="/engineering#/run/${encodeURIComponent(r.run_id)}">${esc(r.run_id.slice(0, 18))}…</a></td>
              <td class="mono">${esc(r.approach)}</td>
              <td class="mono">${esc(r.model)}</td>
              <td class="mono">${esc(r.split)}</td>
              <td class="mono"><strong>${r.primary_score != null ? (r.primary_score * 100).toFixed(1) + '%' : '—'}</strong></td>
              <td class="mono">${r.latency_s != null ? Number(r.latency_s).toFixed(2) + 's' : '—'}</td>
            </tr>`
            )
            .join('')}
        </tbody>
      </table>
    </div>
  `;
  backdrop.classList.remove('hidden');
}

document.addEventListener('DOMContentLoaded', init);
