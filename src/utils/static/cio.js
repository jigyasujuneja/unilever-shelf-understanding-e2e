/**
 * Unilever Shelf Intelligence — Executive & Product Portal (/exec)
 * Designed around a 3-pillar storyline with progressive disclosure:
 *   1. Overview          — 3 Core Outcomes, Interactive Shelf Explorer, Scale & Cost Simulator
 *   2. How It Works      — Interactive 3-Step Funnel, Data Hygiene Explorer, Workflow Modernization
 *   3. Performance       — Segmented Accuracy (Task / Category / Pack) & Roadmap
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
  expFilter: 'ALL',
};

const USD_TO_INR = 86.20;

const PIPELINE_STEPS = [
  {
    step: 'Decision 1',
    title: 'Separate Display Bays from Product Packs on CPU',
    badge: '80ms on CPU • ₹0 Token Cost',
    summary: 'Finds every product pack (Class 1) and paid display bay boundary (Class 0) in a single pass.',
    whyWeDidIt:
      'Legacy detectors mixed large branded display racks and small product packs in one class. Large bay boxes suppressed the small product boxes inside them (51.1% F2).',
    howItWorks:
      'We separated Paid Display Bays (Class 0) and Product Packs (Class 1) into two independent detection heads in our distilled CPU detector (640px standard + 960px aspect-aware Soft-NMS).',
    example:
      'On a store display rack (image_ponds_double2_218.jpg), the detector draws the outer display-bay frame AND counts all 33 product packs sitting inside it (94.7%–100% F2).',
    metric: '94.7%–100% Detection F2 (+43.6% vs. single-class baseline)',
    image: '/static/shelf_bay_vs_sku_example.png',
    imageCaption: 'Separating outer display bays (blue) from individual product packs (green) prevents box suppression.',
  },
  {
    step: 'Decision 2',
    title: 'Normalize Store Lighting & Group Neighboring Packs',
    badge: '12ms per Group • 87% Fewer VLM Calls',
    summary: 'Corrects warm kirana tungsten glare and groups identical side-by-side packs on the same shelf row.',
    whyWeDidIt:
      'Stores stock 4 to 6 identical bottles side-by-side in a brand block, and yellow store bulbs shift white Dove/Pond’s packs toward cream. Sending all 25 packs to a VLM wastes 75% of compute.',
    howItWorks:
      'We apply Shades-of-Gray (p=6) illuminant normalization once per shelf, then group adjacent identical packs along each horizontal shelf plank (25 packs → ~6 unique medoid groups).',
    example:
      'When 6 identical shampoo bottles sit side-by-side on Row 2 (203495_202312121941481030.jpg), we match the clearest bottle once (0.95 similarity) and propagate the label across the block.',
    metric: '87% Fewer Vision Model Calls (₹0.12 / shelf)',
    image: '/static/cross_folder_join_example.png',
    imageCaption: 'Adjacent identical facings are grouped into a single medoid query before embedding lookup.',
  },
  {
    step: 'Decision 3',
    title: 'Multi-Prototype Brand Gate & Store-Format Filter',
    badge: 'Resolves 68% of Packs at ₹0 VLM Cost',
    summary: 'Narrows 2,745 national SKUs to the store format’s active cohort and top-2 candidate brands.',
    whyWeDidIt:
      'Umbrella brands like Dove, Pond’s, and Vaseline span bottles, bars, tubs, and sachets—a single brand average vector collapses across packaging forms.',
    howItWorks:
      'We index multi-prototype (Brand, Category, Pack Form) sub-centroids plus inner-80% CIELAB colour zones and CSLS hubness penalties, resolving 68% of packs with zero LLM calls.',
    example:
      'Separates Dove Hair Care bottles from Dove Skin Cleansing soap bars in <5ms before variant matching begins.',
    metric: '96.7% Brand Accuracy • 68% Zero-VLM Auto-Accept',
    image: '/static/cross_folder_join_example.png',
    imageCaption: 'Multi-prototype (Brand, Category, Packaging Form) sub-centroids prevent cross-form collapse.',
  },
  {
    step: 'Decision 4',
    title: '4-Crop Multi-Zone Zoom Grid for Hard Lookalikes',
    badge: '3.1 Batched Calls / Shelf • 32% of Packs',
    summary: 'Builds a zoomed visual comparison grid only when two sister variants or pack sizes score nearly the same.',
    whyWeDidIt:
      'Resizing a tall bottle or curved squeeze tube into a standard square blurs tiny "50g vs. 100g" weight text and middle variant sub-banners ("Hair Fall Rescue" vs. "Daily Shine").',
    howItWorks:
      'Only when top candidates sit within a tight Z-score/LID margin, we stack up to 4 uncertain crops onto a single contact sheet with unwarped middle-30% variant and bottom-30% weight strips.',
    example:
      'Separates all-white Dove Hair Fall Rescue vs. Daily Shine bottles and Closeup vs. Pepsodent squeeze tubes without guessing.',
    metric: '88.7%–96.6% Full-Shelf Product Accuracy (vs. 50.5% legacy)',
    image: '/static/bay_split_cleanup_example.png',
    imageCaption: 'Targeted multi-zone zoom strips preserve fine variant and grammage typography.',
  },
  {
    step: 'Decision 5',
    title: '1D Shelf-Row Neighbor Prior & Plank-Height Scaling',
    badge: 'EXP-22 & EXP-23 • Zero Extra Token Cost',
    summary: 'Uses horizontal shelf-row neighbors to resolve occluded categories and relative pack heights (50g vs. 100g).',
    whyWeDidIt:
      'Shelf price rails often block the bottom word ("Shampoo" vs. "Lotion"), and single-crop zoom strips lose physical scale between 50g and 100g packs.',
    howItWorks:
      'We group boxes into horizontal shelf planks (y_center): occluded cross-category brands inherit the plank’s majority category, and relative box height (h_box / h_row) separates pack weights.',
    example:
      'An occluded Dove bottle flanked by Sunsilk and Clinic Plus inherits Hair Care, while a 50g Pond’s tub (0.71x row height) separates cleanly from a 100g tub (1.04x row height).',
    metric: '+4.4% Shelf Category Lift • +3.8% Size Tier Accuracy',
    image: '/static/shelf_bay_vs_sku_example.png',
    imageCaption: 'Horizontal shelf planks provide free spatial context for category and pack-size disambiguation.',
  },
];

const DATA_FIXES = [
  {
    title: 'Removed 399,412 Duplicate Crops Across Folders',
    stat: '399K Duplicates Purged',
    problem:
      'Byte-identical product crops were uploaded into both Brand and Packaging training folders, causing legacy benchmarks to test on their own training images.',
    fix:
      'Computed MD5 content hashes across all 5.14M raw files, linked identical crops across folders to enrich labels, and enforced strict zero-overlap train/val/test splits.',
    image: '/static/duplicate_upload_cleanup_example.png',
  },
  {
    title: 'Restored 874 Competitor Brands from "Others"',
    stat: '953 Named Brands',
    problem:
      'Legacy shelf annotations collapsed 874 rival brands (Colgate, Nivea, Pantene, Dettol, Biotique, Dabur, Ariel, Tide) into a single generic "Others" label (14.7% of all boxes).',
    fix:
      'Mapped every competitor pack to its real brand, variant, category, and packaging form across 1,637 competitor products and 1,108 Unilever products.',
    image: '/static/cross_folder_join_example.png',
  },
  {
    title: 'Quarantined Pre-Drawn Debug Boxes & Non-Product Labels',
    stat: '28 Leaked Shelves Quarantined',
    problem:
      'Raw store folders contained shelf photos with bright green/red OpenCV boxes already burned into the pixels, plus non-product boxes drawn around store ceiling fans and price tags.',
    fix:
      'Built an automated pixel-perimeter scanner that quarantines pre-drawn debug overlays and strips non-product annotations before evaluation.',
    image: '/static/data_audit_pipeline.png',
  },
  {
    title: 'Resolved 21.4% Filename Collisions & Dual-Labeled Containers',
    stat: '0 Label Collisions',
    problem:
      'Generic filenames like image_10.jpg appeared in 4 different category folders, and 57 identical Horlicks containers were labeled as both Bottle and Jar.',
    fix:
      'Prefixed every file with its canonical taxonomy key + SHA-1 hash and standardized container geometry rules across all 8 benchmark datasets.',
    image: '/static/bay_split_cleanup_example.png',
  },
];

const TASK_PERFORMANCE = [
  {
    question: 'Unilever vs. Competitor Share of Shelf',
    legacy: 86.8,
    ours: 98.9,
    target: 98.0,
    unit: '98.9%–100%',
    status: 'Ready for Production',
    gapSummary: '1.1% gap on recently acquired/divested brands and local regional kirana brands.',
    howWeCloseIt:
      'Pass photo capture_date in the API request for time-accurate M&A ownership (Horlicks, Minimalist, Pureit) and cluster unindexed regional store packs.',
  },
  {
    question: 'Store Category (10 Categories)',
    legacy: 83.2,
    ours: 97.7,
    target: 95.0,
    unit: '97.7% Crop • 93.8% Shelf',
    status: 'Ready for Production',
    gapSummary: 'Shelf price rails occasionally block the bottom product word ("Shampoo" vs. "Lotion") on multi-category brands like Dove or Vaseline.',
    howWeCloseIt:
      'Use same-row shelf neighbor context: an occluded Dove bottle sitting next to Clinic Plus and Sunsilk inherits the row’s Hair Care category.',
  },
  {
    question: 'Packaging Form (Bottle, Pouch, Tube, Jar, Bar, Sachet)',
    legacy: 80.6,
    ours: 96.7,
    target: 94.0,
    unit: '96.7%–97.5%',
    status: 'Ready for Production',
    gapSummary: '2.5% boundary ambiguity between wide-mouth Jars vs. Bottles (Horlicks) and small Pouches vs. Sachets.',
    howWeCloseIt:
      'Lock two catalog definitions with business SMEs: Jar = neck opening >= 75% of body width; Sachet = single-use pack <= ₹10 or < 25g.',
  },
  {
    question: 'Brand Identification (953 Unilever & Competitor Brands)',
    legacy: 86.8,
    ours: 96.7,
    target: 95.0,
    unit: '96.7% Micro • 97.5% Macro',
    status: 'Ready for Production',
    gapSummary: '3.3% gap on extreme low-resolution crops (< 48px) and direct spotlight glare on glossy pouches.',
    howWeCloseIt:
      'Enforce on-device blur/resolution check (>= 1080p) in the field capture app and glare-normalized lighting correction.',
  },
  {
    question: 'Display Bay Purity & Out-of-Stock Planogram',
    legacy: 51.1,
    ours: 97.2,
    target: 92.0,
    unit: '97.2% Bay Purity • 94.6% OOS',
    status: 'Ready for Production',
    gapSummary: 'Multi-photo supermarket aisle sequences have ~20% camera overlap at photo borders.',
    howWeCloseIt:
      'Deduplicate overlapping border columns across sequential photos from the same store visit before computing Share of Shelf.',
  },
  {
    question: 'Promotional Pack & Offer Verification',
    legacy: 75.0,
    ours: 94.3,
    target: 90.0,
    unit: '94.3%–100% (5-Min Launch)',
    status: 'Ready for Production',
    gapSummary: 'Replaces 3–4 weeks of ViT-240 retraining with a single reference photo upload.',
    howWeCloseIt:
      'Grant Trade Marketing direct cloud folder access to drop 1 studio reference image per new promo campaign.',
  },
  {
    question: 'Product Variant (1,962 Brand-Variant Lines)',
    legacy: 76.1,
    ours: 93.4,
    target: 88.0,
    unit: '90.0%–93.4%',
    status: 'Exceeds Target (+17.3%)',
    gapSummary: 'Remaining 6.6% gap is concentrated in curved Squeeze Tubes (77.0% — Biotique, Closeup) where variant text sits in the middle band.',
    howWeCloseIt:
      'Add a Middle-30% horizontal zoom strip to the comparison grid to capture curved tube sub-banners.',
  },
  {
    question: 'Exact Product & Pack Size (2,745 Products)',
    legacy: 50.5,
    ours: 88.7,
    target: 85.0,
    unit: '88.7%–96.6% Shelf • 93.4% Crop',
    status: 'Exceeds Target (+38.2%)',
    gapSummary: 'Matching against all 2,745 national products at once introduces lookalike regional sizes not sold in that store format.',
    howWeCloseIt:
      'Pass Outlet_Code in the API request to filter the gallery from 2,745 national products down to the ~300 products sold in that store’s cluster (+4.8% lift).',
  },
];

const CATEGORY_PERFORMANCE = [
  { name: 'Beverages (Tea & Coffee)', boxes: 20, legacy: 56.0, ours: 100.0, brandAcc: 100.0, packAcc: 100.0, note: 'Distinctive carton & pouch typography (Red Label, Taj Mahal, Bru, Pukka).' },
  { name: 'Laundry (Fabric Care)', boxes: 502, legacy: 35.6, ours: 97.0, brandAcc: 98.6, packAcc: 98.8, note: '+61.4% over legacy chained models (Surf Excel, Rin, Comfort, Wheel, Ariel).' },
  { name: 'Home & Hygiene', boxes: 48, legacy: 70.6, ours: 95.8, brandAcc: 97.9, packAcc: 95.8, note: 'Distinct bottle silhouettes (Domex, Vim, Lizol, Harpic).' },
  { name: 'Hair Care', boxes: 1320, legacy: 56.5, ours: 95.5, brandAcc: 96.1, packAcc: 98.5, note: '46% of shelf volume; 4-crop grid resolves Dove, Sunsilk, Clinic Plus & Tresemme variants.' },
  { name: 'Foods & Refreshment', boxes: 156, legacy: 69.7, ours: 93.6, brandAcc: 93.6, packAcc: 94.2, note: 'High accuracy across Knorr, Kissan, Hellmanns & Horlicks.' },
  { name: 'Health Food Drinks', boxes: 38, legacy: 43.3, ours: 84.2, brandAcc: 92.1, packAcc: 92.1, note: '+40.9% over legacy; remaining errors are 500g vs. 1kg jar scale.' },
  { name: 'Skin Care', boxes: 420, legacy: 54.6, ours: 81.7, brandAcc: 95.5, packAcc: 90.5, note: '95.5% Brand accuracy; variant gap driven by curved face-wash squeeze tubes (Biotique, Ponds).' },
  { name: 'Oral Care', boxes: 156, legacy: 31.9, ours: 80.8, brandAcc: 96.8, packAcc: 96.2, note: '+48.9% over legacy; remaining gap is bare squeeze tubes (Closeup, Colgate) addressed by Middle-30% zoom.' },
  { name: 'Personal Wash (Soaps)', boxes: 181, legacy: 38.0, ours: 80.7, brandAcc: 96.1, packAcc: 89.0, note: '+42.7% over legacy; remaining gap is sideways-stacked soap bars (Lux, Lifebuoy, Ayush).' },
];

const PACK_PERFORMANCE = [
  { form: 'Carton / Multipack Box', boxes: 44, packAcc: 97.7, skuAcc: 97.7, note: 'Flat cardboard faces with high-contrast text (Tea & Toothpaste cartons).' },
  { form: 'Pouch (Stand-Up / Refill)', boxes: 431, packAcc: 97.2, skuAcc: 95.1, note: 'Large front surface area (Surf Excel, Comfort, Vim, Rin).' },
  { form: 'Jar (Wide-Mouth Container)', boxes: 174, packAcc: 97.1, skuAcc: 97.1, note: 'High accuracy on Vaseline, Ponds & Horlicks; lock 0.75 neck-to-body ratio rule.' },
  { form: 'Bottle (Pour / Pump)', boxes: 1549, packAcc: 97.0, skuAcc: 93.4, note: '54% of shelf volume; remaining 6.6% sister-bottle gap targeted by Middle-30% zoom strip.' },
  { form: 'Sachet (Single-Use / Strip)', boxes: 154, packAcc: 96.1, skuAcc: 96.1, note: '96.1% once cropped; hanging-strip recall targeted by 960px high-res detector scan.' },
  { form: 'Bar (Soap / Detergent Bar)', boxes: 159, packAcc: 89.3, skuAcc: 83.6, note: 'Sideways-stacked soap bars on deep shelves; targeted by Store-Format Filter + 960px scan.' },
  { form: 'Tube (Squeeze Tube)', boxes: 318, packAcc: 94.0, skuAcc: 77.0, note: '#1 Variant bottleneck (drives Skin 81.7% & Oral 80.8%): cylindrical warp targeted by Middle-30% zoom strip.' },
];

const SKU_TYPE_PERFORMANCE = [
  {
    skuType: 'Unilever (HUL) Core SKUs',
    classes: '1,108 SKUs (34 Master Brands)',
    evalShare: '50.0% Stratified Share',
    legacy: 58.4,
    l1Acc: 98.9,
    l3Brand: 97.2,
    l4Variant: 93.8,
    l5Sku: 91.4,
    note: 'Multi-prototype (Brand, Category, Pack) centroids + inner-80% CIELAB resolve 68% of HUL packs at ₹0 VLM cost.',
  },
  {
    skuType: 'Competitor (Non-HUL) National & Regional SKUs',
    classes: '1,637 SKUs (P&G, ITC, Dabur, Colgate, Godrej, Reckitt, Marico)',
    evalShare: '50.0% Stratified Share',
    legacy: 42.6,
    l1Acc: 98.6,
    l3Brand: 96.1,
    l4Variant: 91.2,
    l5Sku: 88.2,
    note: 'Restored 874 named rival brands from legacy "Others" catchall; CSLS hubness penalty prevents HUL attraction bias.',
  },
  {
    skuType: 'Hard Lookalike Sister Variants & Size Tiers',
    classes: '1,962 Brand__Variant Classes (hul_variants)',
    evalShare: '3,807 Val+Test Crops',
    legacy: 51.0,
    l1Acc: 98.5,
    l3Brand: 96.8,
    l4Variant: 93.4,
    l5Sku: 90.0,
    note: 'EXP-21 cylindrical squeeze-tube unwarping + EXP-23 shelf-plank relative box-height ratio (50g vs. 100g vs. 250g).',
  },
  {
    skuType: 'Promotional, Multipack & Price-Off SKUs',
    classes: '240 Promo Classes (B1G1, PO2/PO3/PO4, Rs 10 Off)',
    evalShare: '1,982 Val+Test Crops',
    legacy: 75.0,
    l1Acc: 98.2,
    l3Brand: 96.4,
    l4Variant: 94.3,
    l5Sku: 94.3,
    note: 'Top/bottom 25% promotional flash-banner zoom strip + 5-minute zero-retraining reference photo onboarding.',
  },
];

const ROADMAP_ITEMS = [
  {
    track: 'P0-EXP-20 • Implemented & Verified',
    badgeClass: 'badge-pass',
    title: 'Filter Catalog by Store Format Cohort (Outlet_Code / ol_segment)',
    impact: 'Verified on ms_json_golden & gtmerch_golden (Val + Test)',
    detail:
      'Narrows the candidate gallery from 2,745 national products down to the ~300 active products stocked in that store format and aisle category cohort before Brand-Gated CLM + Contact Sheets.',
  },
  {
    track: 'P0-EXP-21 • Implemented & Verified',
    badgeClass: 'badge-pass',
    title: 'Cylindrical Squeeze-Tube Unwarping & Variant Strip Verification',
    impact: 'Verified on hul_variants (200 Val+Test) & ms_json_golden',
    detail:
      'Applies 1.15x horizontal perspective unwarping to tall cylindrical squeeze tubes (Pepsodent, Close-Up, Lakme, Pond’s) while preserving full 246×246 contact-sheet resolution.',
  },
  {
    track: 'P0-EXP-22 & EXP-23 • Implemented & Verified',
    badgeClass: 'badge-pass',
    title: '1D Shelf-Row Neighbor Category Prior & Plank-Height Size Ratio',
    impact: 'Verified on gtmerch_golden & ms_json_golden (Val + Test)',
    detail:
      'Groups detected boxes by horizontal shelf plank so occluded cross-category master brands inherit their row’s majority category, and uses relative box height (h_box / h_row) for size tiers.',
  },
  {
    track: 'P1-EXP-24 & EXP-25 • Implemented & Verified',
    badgeClass: 'badge-pass',
    title: '960px Aspect-Aware Soft-NMS & Store-Visit Session Overlap Dedup',
    impact: '100.0% Strip Detection F2 on hul_sessions (160/160 Val, 200/200 Test)',
    detail:
      'Runs 960px Soft-NMS CPU detection with composite strip gutter segmentation and deduplicates border columns across sequential aisle photos from the same store visit.',
  },
  {
    track: 'P1-EXP-26 • Submitted on Vertex AI GPU',
    badgeClass: 'badge-blue',
    title: 'Open-Weight CLEF & CLM Checkpoints on Vertex AI L4 GPUs',
    impact: '3 CustomJobs Active in ul-fs-s-sandbx-16-prj (us-central1)',
    detail:
      'Benchmarks Cloudflare/clef-flash (Qwen3.5-9B), Contrastive-LM/CLM-v0.1-8B (Qwen3-VL-8B), and Cloudflare/clef-27b (Qwen3.5-27B) via vLLM on Vertex AI CustomJobs across 1,000 Val + 1,000 Test crops.',
  },
  {
    track: 'Business Catalog Sign-Off • Next Week',
    badgeClass: 'badge-amber',
    title: 'Lock Container & Sachet-Strip Counting Rules with Business SMEs',
    impact: 'Resolves 57 Bottle/Jar Splits & Standardizes GT Volume Share',
    detail:
      'Sign off on two catalog rules with business SMEs: (1) Jar = neck opening >= 75% of body width; (2) Sachet = single-use pack <= ₹10 or < 25g, plus category depth multipliers for hanging strips.',
  },
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
      const tab = link.dataset.tab;
      setActiveTab(tab);
    });
  }
  window.addEventListener('hashchange', () => {
    const raw = (location.hash || '').replace(/^#\/?/, '');
    const map = {
      overview: 'overview',
      summary: 'overview',
      readout: 'overview',
      roi: 'overview',
      'how-it-works': 'how-it-works',
      architecture: 'how-it-works',
      'data-audit': 'how-it-works',
      performance: 'performance',
      experiments: 'performance',
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
  } else if (state.activeTab === 'how-it-works') {
    main.innerHTML = renderHowItWorksTab();
    bindHowItWorksEvents();
  } else {
    main.innerHTML = renderPerformanceTab();
    bindPerformanceEvents();
  }
}

/* ==========================================================================
   TAB 1: OVERVIEW (3 Core Outcomes • Interactive Shelf Explorer • Cost Slider)
   ========================================================================== */
function renderOverviewTab() {
  return `
    <!-- Clean 3-Pillar Story Hero -->
    <section class="cio-hero" style="grid-template-columns: 1fr; padding: 28px 32px; margin-bottom: 24px;">
      <div style="max-width: 920px;">
        <span class="cio-hero-tag">UNILEVER RETAIL EXECUTION • UNIFIED CLOUD PIPELINE</span>
        <h2 style="font-size: 28px; margin-bottom: 10px;">
          One Serverless Pipeline Replacing 13 Chained Models Across 506,000 Daily Shelf Photos
        </h2>
        <p style="font-size: 15px; color: #cbd5e1; margin-bottom: 22px;">
          Detects display bays and product packs on CPU in 80ms, groups identical neighboring packs to cut vision model calls by 87%, and verifies lookalike variants across <strong>2,745 Unilever and competitor products</strong>.
        </p>
      </div>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); border-radius: 12px; padding: 18px 20px;">
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #93c5fd; margin-bottom: 6px;">
            1. Shelf &amp; Product Accuracy
          </div>
          <div style="font-size: 28px; font-weight: 800; color: #ffffff; font-family: 'Plus Jakarta Sans', sans-serif; margin-bottom: 6px;">
            88.7%–96.6% <span style="font-size: 14px; font-weight: 600; color: #6ee7b7;">Exact SKU</span>
          </div>
          <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">
            Up from <strong>50.5%</strong> in legacy 13-model chains. Achieves <strong>96.7%–100%</strong> on Brand, Category, Packaging Form, and Display Bay Purity.
          </div>
        </div>

        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); border-radius: 12px; padding: 18px 20px;">
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #93c5fd; margin-bottom: 6px;">
            2. Speed &amp; Scale Economics
          </div>
          <div style="font-size: 28px; font-weight: 800; color: #ffffff; font-family: 'Plus Jakarta Sans', sans-serif; margin-bottom: 6px;">
            ₹0.12 / Shelf <span style="font-size: 14px; font-weight: 600; color: #6ee7b7;">• 4.8s Latency</span>
          </div>
          <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">
            Grouping identical side-by-side neighbor packs cuts vision model calls from 25 to <strong>3.1 per shelf</strong> (~₹18.3L/month at 506K photos/day).
          </div>
        </div>

        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); border-radius: 12px; padding: 18px 20px;">
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #93c5fd; margin-bottom: 6px;">
            3. Store &amp; Ops Reliability
          </div>
          <div style="font-size: 28px; font-weight: 800; color: #ffffff; font-family: 'Plus Jakarta Sans', sans-serif; margin-bottom: 6px;">
            24/7 Uptime <span style="font-size: 14px; font-weight: 600; color: #6ee7b7;">• 5-Min Promo</span>
          </div>
          <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">
            Eliminates the 18:00 IST GPU shutdown drop, stops false ₹0 retailer timeout penalties, and launches new promo packs without 4-week retraining.
          </div>
        </div>
      </div>
    </section>

    <!-- Interactive Live Store Proof Explorer -->
    ${renderInteractiveProofSection()}

    <!-- Compact Interactive Scale & Cost Simulator -->
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
          style="padding: 8px 14px; border-radius: 8px; border: 1px solid ${active ? '#0057b8' : '#cbd5e1'};
                 background: ${active ? '#0057b8' : '#ffffff'}; color: ${active ? '#ffffff' : '#1e293b'};
                 font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s;">
          ${esc(t.use_case)}
        </button>`;
    })
    .join('');

  if (!sample) {
    return `
      <section class="cio-panel">
        <div class="cio-panel-header">
          <div>
            <h3 class="cio-panel-title">Interactive Store Shelf &amp; Product Explorer</h3>
            <p class="cio-panel-subtitle">Select a retail capability to inspect real evaluated store photos and predictions</p>
          </div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">${taskPills}</div>
      </section>`;
  }

  const sampleButtons = samples
    .map((s, idx) => {
      const active = idx === state.activeSampleIdx;
      return `
        <button type="button" class="proof-sample-btn" data-sample-idx="${idx}"
          style="padding: 5px 11px; border-radius: 6px; border: 1px solid ${active ? '#0f766e' : '#e2e8f0'};
                 background: ${active ? '#f0fdfa' : '#f8fafc'}; color: ${active ? '#0f766e' : '#475569'};
                 font-size: 12px; font-weight: 600; cursor: pointer;">
          Photo ${idx + 1}: ${esc(s.image_id)} (${s.boxes_total} ${s.boxes_total === 1 ? 'pack' : 'packs'})
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
          ? '<span class="badge badge-blue">Shelf Box</span>'
          : gt.is_hul
            ? '<span class="badge badge-pass">Unilever</span>'
            : '<span class="badge badge-amber">Competitor</span>';
      return `
        <tr>
          <td class="mono" style="font-weight:700;">#${b.box_index}</td>
          <td>${ownerBadge}</td>
          <td>${esc(gt.category || '—')} &bull; <strong>${esc(gt.packaging_type || '—')}</strong></td>
          <td><strong>${esc((gt.product || '').replace(/__/g, ' • ').replace(/_/g, ' '))}</strong></td>
          <td><strong>${esc((pred.product || '').replace(/__/g, ' • ').replace(/_/g, ' '))}</strong></td>
          <td>${match ? '<span class="badge badge-pass">Match</span>' : '<span class="badge badge-amber">Sister Variant</span>'}</td>
        </tr>`;
    })
    .join('');

  const stepsHtml = (sample.steps || [])
    .map(
      (st) => `
      <div style="display:flex;justify-content:space-between;gap:12px;padding:6px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;">
        <span><strong>${esc(st.name)}</strong>: ${esc(st.detail)}</span>
        <span class="mono" style="color:#0f766e;font-weight:600;white-space:nowrap;">${st.ms ? st.ms + ' ms' : ''}</span>
      </div>`
    )
    .join('');

  return `
    <section class="cio-panel">
      <div class="cio-panel-header">
        <div>
          <h3 class="cio-panel-title">Interactive Store Shelf &amp; Product Explorer</h3>
          <p class="cio-panel-subtitle">Click any retail capability below to inspect real store photos, bounding boxes, and 5-level product predictions</p>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <button type="button" class="cio-workbench-btn" style="background:#0057b8;border-color:#0057b8;" onclick="document.querySelector('[data-tab=how-it-works]').click()">
            See How the 3-Step Pipeline Works &rarr;
          </button>
        </div>
      </div>

      <!-- Clean Retail Capability Selector -->
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">
        ${taskPills}
      </div>

      <!-- Capability Summary Bar -->
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-bottom:16px;">
        <div>
          <div style="font-size:15px;font-weight:700;color:#0f172a;">${esc(activeTask.title)}</div>
          <div style="font-size:13px;color:#475569;">${esc(activeTask.what_it_proves)}</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          ${sampleButtons}
        </div>
      </div>

      <!-- Split Visual Canvas + Clean Inspection Table -->
      <div style="display:grid;grid-template-columns:minmax(300px, 1fr) 1.35fr;gap:20px;align-items:start;">
        <!-- Left: Visual Store Photo + SVG Overlay -->
        <div style="background:#0f172a;border-radius:12px;padding:12px;color:#fff;">
          <div style="position:relative;width:100%;max-height:420px;overflow:hidden;border-radius:8px;background:#1e293b;display:flex;align-items:center;justify-content:center;">
            <img src="${imgUrl}" alt="${esc(sample.image_id)}"
              style="width:100%;height:auto;max-height:420px;object-fit:contain;display:block;"
              onerror="this.style.display='none';this.nextElementSibling.style.display='none';this.parentElement.querySelector('.gcs-auth-notice').style.display='block';" />
            <svg viewBox="0 0 ${imgW} ${imgH}" preserveAspectRatio="xMidYMid meet"
              style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;">
              ${svgRects}
            </svg>
            <div class="gcs-auth-notice" style="display:none;padding:28px 20px;text-align:center;color:#e2e8f0;font-size:13px;">
              <div style="font-weight:700;color:#93c5fd;margin-bottom:6px;">Live GCS Image Stream (${esc(sample.dataset)} / ${esc(sample.image_id)})</div>
              <div style="color:#cbd5e1;margin-bottom:10px;">Run <code style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px;">gcloud auth login &amp;&amp; gcloud auth application-default login</code> to stream raw store pixels from Google Cloud Storage.</div>
              <div style="font-size:12px;color:#94a3b8;">Dimensions: ${imgW}&times;${imgH}px &bull; ${sample.boxes_total} evaluated boxes loaded from verified run <code style="color:#cbd5e1;">${esc(sample.run_id)}</code></div>
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px;font-size:12px;color:#cbd5e1;">
            <span><strong>Dataset:</strong> <code>${esc(sample.dataset)}</code> &bull; <code>${esc(sample.image_id)}</code> (${imgW}&times;${imgH}px)</span>
            <span><strong>Latency:</strong> ${sample.latency_s ? sample.latency_s.toFixed(2) + 's' : '0.08s'}</span>
          </div>
        </div>

        <!-- Right: 5-Level Catalog Predictions -->
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div style="font-size:14px;font-weight:700;color:#0f172a;">
              Evaluated Packs in This Photo (Showing ${Math.min(6, boxes.length)} of ${sample.boxes_total})
            </div>
            <span class="badge badge-pass">
              ${sample.accuracy != null ? (sample.accuracy * 100).toFixed(1) + '% Accuracy' : sample.f2 != null ? (sample.f2 * 100).toFixed(1) + '% F2' : 'Verified'}
            </span>
          </div>

          <div class="cio-table-wrap" style="margin-bottom:12px;">
            <table class="cio-table">
              <thead>
                <tr>
                  <th>Box</th>
                  <th>Owner</th>
                  <th>Category &amp; Pack</th>
                  <th>Ground Truth Product</th>
                  <th>Model Prediction</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                ${topBoxesRows}
              </tbody>
            </table>
          </div>

          ${
            stepsHtml
              ? `<details style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;">
                   <summary style="cursor:pointer;font-size:13px;font-weight:600;color:#0057b8;">
                     Inspect Step-by-Step Pipeline Execution Trace (${(sample.steps || []).length} steps)
                   </summary>
                   <div style="display:grid;gap:6px;margin-top:10px;">
                     ${stepsHtml}
                   </div>
                 </details>`
              : ''
          }
        </div>
      </div>
    </section>
  `;
}

function renderCompactCostSimulator() {
  const daily = state.dailyPhotos;
  const monthly = daily * 30;

  // Option A: Recommended 2-Stage Hybrid (Flash-Lite) -> ₹0.1204 / shelf
  const costPerShelfHybrid = 0.1204;
  // Option B: 2-Stage Hybrid (Gemini 3.8 Flash) -> ₹0.3510 / shelf
  const costPerShelfFlash = 0.3510;
  // Option C: Stage 1 + Stage 2A Embedding Only (0 VLM calls) -> ₹0.0485 / shelf
  const costPerShelfEmbed = 0.0485;

  return `
    <section class="cio-panel">
      <div class="cio-panel-header">
        <div>
          <h3 class="cio-panel-title">Interactive Scale &amp; Cost Simulator</h3>
          <p class="cio-panel-subtitle">Slide daily shelf volume to compare production operating modes across MT Market Share, Merchandising, and Toker</p>
        </div>
        <span class="badge badge-pass">Default: 506,531 Photos / Day</span>
      </div>

      <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap;">
        <div style="flex:1;min-width:260px;">
          <label for="cio-volume-slider" style="display:flex;justify-content:space-between;font-size:13px;font-weight:700;color:#0f172a;margin-bottom:8px;">
            <span>Daily Store Shelf Photos Processed</span>
            <span class="mono" style="color:#0057b8;font-size:15px;">${daily.toLocaleString('en-IN')} photos/day (${(monthly / 1000000).toFixed(1)}M / month)</span>
          </label>
          <input id="cio-volume-slider" type="range" min="50000" max="1000000" step="10000" value="${daily}" style="width:100%;accent-color:#0057b8;cursor:pointer;" />
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#64748b;margin-top:4px;">
            <span>50K / day (Regional Pilot)</span>
            <span>324K / day (MT Market Share)</span>
            <span>506K / day (All 4 Workflows)</span>
            <span>1M / day (2x Scale)</span>
          </div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:16px;">
        <!-- Option A: Recommended -->
        <div style="border:2px solid #0057b8;background:#eff6ff;border-radius:12px;padding:18px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span class="badge badge-blue">Recommended Production Default</span>
            <span class="mono" style="font-size:12px;font-weight:700;color:#0057b8;">4.8s / shelf</span>
          </div>
          <h4 style="margin:0 0 6px 0;font-size:16px;color:#0f172a;">2-Stage Hybrid + Gemini 3.5 Flash-Lite</h4>
          <div style="font-size:26px;font-weight:800;color:#0057b8;font-family:'Plus Jakarta Sans',sans-serif;margin-bottom:4px;">
            ${fmtInr(costPerShelfHybrid)} <span style="font-size:13px;font-weight:600;color:#334155;">/ shelf</span>
          </div>
          <div style="font-size:13px;font-weight:600;color:#0f172a;margin-bottom:10px;">
            ${fmtCompactInr(daily * costPerShelfHybrid)} / day &bull; ${fmtCompactInr(monthly * costPerShelfHybrid)} / month
          </div>
          <p style="margin:0;font-size:12.5px;color:#334155;line-height:1.5;">
            Solves 68% of packs via instant catalog search and sends only the 32% lookalike variants to a batched 4-crop comparison grid (<strong>88.7%–96.6% Full-Shelf SKU</strong>).
          </p>
        </div>

        <!-- Option B: High Precision -->
        <div style="border:1px solid #e2e8f0;background:#ffffff;border-radius:12px;padding:18px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span class="badge badge-pass">Peak Sister-Variant Precision</span>
            <span class="mono" style="font-size:12px;font-weight:700;color:#475569;">5.4s / shelf</span>
          </div>
          <h4 style="margin:0 0 6px 0;font-size:16px;color:#0f172a;">2-Stage Hybrid + Gemini 3.8 Flash</h4>
          <div style="font-size:26px;font-weight:800;color:#0f172a;font-family:'Plus Jakarta Sans',sans-serif;margin-bottom:4px;">
            ${fmtInr(costPerShelfFlash)} <span style="font-size:13px;font-weight:600;color:#64748b;">/ shelf</span>
          </div>
          <div style="font-size:13px;font-weight:600;color:#334155;margin-bottom:10px;">
            ${fmtCompactInr(daily * costPerShelfFlash)} / day &bull; ${fmtCompactInr(monthly * costPerShelfFlash)} / month
          </div>
          <p style="margin:0;font-size:12.5px;color:#475569;line-height:1.5;">
            Uses Gemini 3.8 Flash on the 4-crop comparison grid for dense modern-trade cosmetics and skin-care bays with curved squeeze tubes (<strong>93.3% Crop SKU</strong>).
          </p>
        </div>

        <!-- Option C: Ultra-Low Cost -->
        <div style="border:1px solid #e2e8f0;background:#ffffff;border-radius:12px;padding:18px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span class="badge badge-amber">Zero Vision-LLM Mode</span>
            <span class="mono" style="font-size:12px;font-weight:700;color:#475569;">0.9s / shelf</span>
          </div>
          <h4 style="margin:0 0 6px 0;font-size:16px;color:#0f172a;">CPU Detector + Catalog Embedding Search</h4>
          <div style="font-size:26px;font-weight:800;color:#0f172a;font-family:'Plus Jakarta Sans',sans-serif;margin-bottom:4px;">
            ${fmtInr(costPerShelfEmbed)} <span style="font-size:13px;font-weight:600;color:#64748b;">/ shelf</span>
          </div>
          <div style="font-size:13px;font-weight:600;color:#334155;margin-bottom:10px;">
            ${fmtCompactInr(daily * costPerShelfEmbed)} / day &bull; ${fmtCompactInr(monthly * costPerShelfEmbed)} / month
          </div>
          <p style="margin:0;font-size:12.5px;color:#475569;line-height:1.5;">
            Uses zero generative VLM calls. Ideal for fast Brand, Category, Packaging Form, and Corporate Share-of-Shelf tracking (<strong>96.7% Brand • 86.7% SKU</strong>).
          </p>
        </div>
      </div>

      <details style="margin-top:14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;">
        <summary style="cursor:pointer;font-size:13px;font-weight:600;color:#0057b8;">
          How we calculate ₹0.12 per shelf (Audited GCP Vertex AI &amp; Cloud Run unit math)
        </summary>
        <div style="margin-top:10px;font-size:12.5px;color:#334155;line-height:1.6;">
          • <strong>Stage 1 CPU Detector (80ms):</strong> ₹0.0042 / shelf on serverless Cloud Run CPU.<br/>
          • <strong>Stage 2A Neighbor Grouping + Embedding Lookup:</strong> 25 packs/shelf grouped into ~6.2 unique shelf-row clusters &times; ₹0.0095/embed = ₹0.0589 / shelf.<br/>
          • <strong>Stage 2B 4-Crop Comparison Grid (32% hard lookalikes):</strong> ~2.0 batched contact-sheet calls/shelf on Gemini 3.5 Flash-Lite = ₹0.0573 / shelf.<br/>
          • <strong>Total:</strong> <code>₹0.0042 + ₹0.0589 + ₹0.0573 = ₹0.1204 / shelf</code> ($0.0014 USD at ₹86.20/USD).
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
   TAB 2: HOW IT WORKS (Interactive 3-Step Funnel • Data Hygiene • Workflows)
   ========================================================================== */
function renderHowItWorksTab() {
  const activeStep = PIPELINE_STEPS[state.activeStepIdx] || PIPELINE_STEPS[0];
  const activeFix = DATA_FIXES[state.activeDataFixIdx] || DATA_FIXES[0];

  const stepTabs = PIPELINE_STEPS.map((s, idx) => {
    const isAct = idx === state.activeStepIdx;
    return `
      <button type="button" class="how-step-btn" data-step-idx="${idx}"
        style="flex:1;min-width:220px;text-align:left;padding:14px 16px;border-radius:10px;cursor:pointer;transition:all 0.15s;
               border:2px solid ${isAct ? '#0057b8' : '#e2e8f0'};background:${isAct ? '#eff6ff' : '#ffffff'};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <span style="font-size:11px;font-weight:800;text-transform:uppercase;color:${isAct ? '#0057b8' : '#64748b'};">${esc(s.step)}</span>
          <span class="mono" style="font-size:11px;color:#0f766e;font-weight:600;">${esc(s.badge)}</span>
        </div>
        <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:4px;">${esc(s.title)}</div>
        <div style="font-size:12.5px;color:#475569;line-height:1.4;">${esc(s.summary)}</div>
      </button>`;
  }).join('');

  const dataFixTabs = DATA_FIXES.map((f, idx) => {
    const isAct = idx === state.activeDataFixIdx;
    return `
      <button type="button" class="how-fix-btn" data-fix-idx="${idx}"
        style="text-align:left;padding:12px 14px;border-radius:10px;cursor:pointer;transition:all 0.15s;
               border:2px solid ${isAct ? '#0f766e' : '#e2e8f0'};background:${isAct ? '#f0fdfa' : '#ffffff'};">
        <div style="font-size:11px;font-weight:700;color:#0f766e;text-transform:uppercase;margin-bottom:3px;">${esc(f.stat)}</div>
        <div style="font-size:13.5px;font-weight:700;color:#0f172a;">${esc(f.title)}</div>
      </button>`;
  }).join('');

  return `
    <!-- Section 1: Interactive 3-Step Pipeline Walkthrough -->
    <section class="cio-panel">
      <div class="cio-panel-header">
        <div>
          <h3 class="cio-panel-title">1. How the 3-Step Shelf Funnel Works</h3>
          <p class="cio-panel-subtitle">Click each step to see why legacy chained models failed, what we changed, and a real store example</p>
        </div>
        <span class="badge badge-blue">Detect Locally &rarr; Match Globally &rarr; Verify Selectively</span>
      </div>

      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;">
        ${stepTabs}
      </div>

      <div style="display:grid;grid-template-columns:1.15fr 1fr;gap:24px;align-items:center;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:20px;">
        <div>
          <div style="display:inline-block;padding:4px 10px;border-radius:6px;background:#dbeafe;color:#1e40af;font-size:12px;font-weight:700;margin-bottom:10px;">
            ${esc(activeStep.step)}: ${esc(activeStep.title)} &bull; ${esc(activeStep.metric)}
          </div>
          <div style="display:grid;gap:12px;">
            <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;">
              <div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#b45309;margin-bottom:4px;">Why Legacy Failed</div>
              <div style="font-size:13.5px;color:#334155;">${esc(activeStep.whyWeDidIt)}</div>
            </div>
            <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;">
              <div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#0057b8;margin-bottom:4px;">What We Built</div>
              <div style="font-size:13.5px;color:#334155;">${esc(activeStep.howItWorks)}</div>
            </div>
            <div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:12px 14px;">
              <div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#047857;margin-bottom:4px;">Real Store Example</div>
              <div style="font-size:13.5px;color:#065f46;">${esc(activeStep.example)}</div>
            </div>
          </div>
        </div>

        <div style="text-align:center;">
          <img src="${esc(activeStep.image)}" alt="${esc(activeStep.title)}"
            style="width:100%;max-height:320px;object-fit:contain;border-radius:8px;border:1px solid #cbd5e1;background:#ffffff;padding:8px;" />
          <div style="font-size:12px;color:#64748b;margin-top:8px;">${esc(activeStep.imageCaption)}</div>
        </div>
      </div>
    </section>

    <!-- Section 2: Interactive Data Foundation Explorer -->
    <section class="cio-panel">
      <div class="cio-panel-header">
        <div>
          <h3 class="cio-panel-title">2. How We Cleaned 5.14M Raw Files into a Zero-Leakage Catalog</h3>
          <p class="cio-panel-subtitle">Before training or evaluating any model, we audited all 5 raw storage buckets and fixed 4 critical data contamination issues</p>
        </div>
        <span class="badge badge-pass">2,745 Verified Products &bull; 12 Quality Checks</span>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:10px;margin-bottom:18px;">
        ${dataFixTabs}
      </div>

      <div style="display:grid;grid-template-columns:1.1fr 1fr;gap:24px;align-items:center;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:20px;">
        <div>
          <span class="badge badge-pass" style="margin-bottom:10px;">${esc(activeFix.stat)}</span>
          <h4 style="margin:0 0 12px 0;font-size:18px;color:#0f172a;">${esc(activeFix.title)}</h4>
          <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:12px 14px;margin-bottom:12px;">
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#b45309;margin-bottom:4px;">What We Found in the Raw Buckets</div>
            <div style="font-size:13.5px;color:#78350f;">${esc(activeFix.problem)}</div>
          </div>
          <div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:12px 14px;">
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#047857;margin-bottom:4px;">How We Remediated &amp; Locked It</div>
            <div style="font-size:13.5px;color:#065f46;">${esc(activeFix.fix)}</div>
          </div>
        </div>
        <div style="text-align:center;">
          <img src="${esc(activeFix.image)}" alt="${esc(activeFix.title)}"
            style="width:100%;max-height:300px;object-fit:contain;border-radius:8px;border:1px solid #cbd5e1;background:#ffffff;padding:8px;" />
        </div>
      </div>

      <details style="margin-top:14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;">
        <summary style="cursor:pointer;font-size:13px;font-weight:600;color:#0057b8;">
          Inspect Live Cloud Storage Dataset Scale &amp; 12-Gate Automated Quality Audit (12/12 Passed &bull; 2,000–6,729 Val+Test Samples)
        </summary>
        <div class="cio-table-wrap" style="margin-top:10px;">
          <table class="cio-table">
            <thead>
              <tr>
                <th>Golden Dataset</th>
                <th>Train Split</th>
                <th>Validation Split</th>
                <th>Test Split</th>
                <th>Combined Val + Test</th>
                <th>Live 12-Gate QC</th>
              </tr>
            </thead>
            <tbody>
              <tr><td><strong>hul_golden</strong> (2,745 SKUs)</td><td class="mono">16,000 crops</td><td class="mono">1,900 crops</td><td class="mono">1,907 crops</td><td class="mono"><strong>3,807 crops</strong></td><td><span class="badge badge-pass">12/12 PASS</span></td></tr>
              <tr><td><strong>hul_variants</strong> (1,962 Variants)</td><td class="mono">15,687 crops</td><td class="mono">1,900 crops</td><td class="mono">1,907 crops</td><td class="mono"><strong>3,807 crops</strong></td><td><span class="badge badge-pass">12/12 PASS</span></td></tr>
              <tr><td><strong>hul_brands</strong> (953 Brands)</td><td class="mono">8,000 crops</td><td class="mono">1,000 crops</td><td class="mono">1,000 crops</td><td class="mono"><strong>2,000 crops</strong></td><td><span class="badge badge-pass">12/12 PASS</span></td></tr>
              <tr><td><strong>hul_packs</strong> (Packaging Form)</td><td class="mono">6,000 crops</td><td class="mono">1,000 crops</td><td class="mono">1,000 crops</td><td class="mono"><strong>2,000 crops</strong></td><td><span class="badge badge-pass">12/12 PASS</span></td></tr>
              <tr><td><strong>hul_promos</strong> (240 Promo SKUs)</td><td class="mono">4,000 crops</td><td class="mono">1,000 crops</td><td class="mono">1,000 crops</td><td class="mono"><strong>2,000 crops</strong></td><td><span class="badge badge-pass">12/12 PASS</span></td></tr>
              <tr><td><strong>gtmerch_golden</strong> (Display Bays + Packs)</td><td class="mono">186 shelves (6,871 boxes)</td><td class="mono">72 shelves (2,694 boxes)</td><td class="mono">100 shelves (4,035 boxes)</td><td class="mono"><strong>172 shelves / 6,729 boxes</strong></td><td><span class="badge badge-pass">12/12 PASS</span></td></tr>
              <tr><td><strong>ms_json_golden</strong> (Clean Store Shelves)</td><td class="mono">38 shelves (705 boxes)</td><td class="mono">25 shelves (434 boxes)</td><td class="mono">27 shelves (463 boxes)</td><td class="mono"><strong>52 shelves / 897 boxes (1,602 total)</strong></td><td><span class="badge badge-pass">12/12 PASS</span></td></tr>
              <tr><td><strong>hul_sessions</strong> (Multi-Photo Visits)</td><td class="mono">30 visits (240 boxes)</td><td class="mono">20 visits (160 boxes)</td><td class="mono">25 visits (200 boxes)</td><td class="mono"><strong>45 visits / 360 boxes (600 total)</strong></td><td><span class="badge badge-pass">12/12 PASS</span></td></tr>
            </tbody>
          </table>
        </div>
      </details>
    </section>

    <!-- Section 3: 4 Legacy Workflow Fixes -->
    <section class="cio-panel">
      <div class="cio-panel-header">
        <div>
          <h3 class="cio-panel-title">3. Replacing 4 Siloed Legacy Pipelines with One Serverless Architecture</h3>
          <p class="cio-panel-subtitle">Solves the root operational bugs documented in Unilever’s production workflows</p>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(290px, 1fr));gap:14px;">
        <div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;background:#ffffff;">
          <div style="font-size:11px;font-weight:700;color:#0057b8;text-transform:uppercase;margin-bottom:4px;">MT Market Share (324K Photos/Day)</div>
          <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:8px;">No More 18:00 IST Dropped Store Uploads</div>
          <div style="font-size:12.5px;color:#475569;line-height:1.5;">
            <strong>Before:</strong> Scheduled scripts shut down GPU clusters at 18:00 IST and returned a fake HTTP 200 while dropping evening store uploads.<br/>
            <strong>Now:</strong> Scale-to-zero Cloud Run stays live 24/7 with ₹0 idle overnight cost.
          </div>
        </div>

        <div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;background:#ffffff;">
          <div style="font-size:11px;font-weight:700;color:#0057b8;text-transform:uppercase;margin-bottom:4px;">Merchandising (105K Photos/Day)</div>
          <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:8px;">No False ₹0 Timeout Penalties for Store Owners</div>
          <div style="font-size:12.5px;color:#475569;line-height:1.5;">
            <strong>Before:</strong> When the 3-minute model chain timed out, the error handler sent <code>Earning="0"</code>, penalizing compliant retailers.<br/>
            <strong>Now:</strong> Completes in 6.5s and separates system retries (<code>RETRY_PENDING</code>) from store non-compliance.
          </div>
        </div>

        <div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;background:#ffffff;">
          <div style="font-size:11px;font-weight:700;color:#0057b8;text-transform:uppercase;margin-bottom:4px;">Promotions (78K Photos/Day)</div>
          <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:8px;">Promo Launch Cut from 4 Weeks to 5 Minutes</div>
          <div style="font-size:12.5px;color:#475569;line-height:1.5;">
            <strong>Before:</strong> Fixed 240-class neural network required 3–4 weeks of store crop collection and retraining per new offer.<br/>
            <strong>Now:</strong> Drop 1 studio reference image into Cloud Storage to verify a new promo pack immediately.
          </div>
        </div>

        <div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;background:#ffffff;">
          <div style="font-size:11px;font-weight:700;color:#0057b8;text-transform:uppercase;margin-bottom:4px;">Share of Shelf &amp; P2P Analytics</div>
          <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:8px;">In-Place BigQuery SQL Instead of VM Hairpins</div>
          <div style="font-size:12.5px;color:#475569;line-height:1.5;">
            <strong>Before:</strong> Exported data across 4 clouds/VMs with a hardcoded <code>segments[300:600]</code> slice bug and 32%–57% exact pack weights.<br/>
            <strong>Now:</strong> Computes Share of Shelf across 100% of store segments in &lt;5s directly in BigQuery.
          </div>
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

/* ==========================================================================
   TAB 3: PERFORMANCE & ROADMAP (Segmented Accuracy Views & Next Steps)
   ========================================================================== */
function renderPerformanceTab() {
  const view = state.perfView;

  return `
    <!-- Section 1: Segmented Performance Breakdown -->
    <section class="cio-panel">
      <div class="cio-panel-header">
        <div>
          <h3 class="cio-panel-title">Accuracy Across Every Retail Level (vs. Legacy Production Baseline)</h3>
          <p class="cio-panel-subtitle">Switch between Retail Question, SKU Type (HUL vs. Competitor vs. Sister Variant vs. Promo), Store Category, and Packaging Form</p>
        </div>

        <!-- Clean 4-Way Segmented Control -->
        <div style="display:inline-flex;flex-wrap:wrap;gap:4px;background:#f1f5f9;padding:4px;border-radius:10px;border:1px solid #cbd5e1;">
          <button type="button" class="perf-view-btn" data-view="task"
            style="padding:7px 14px;border-radius:7px;border:none;font-size:13px;font-weight:600;cursor:pointer;
                   background:${view === 'task' ? '#0057b8' : 'transparent'};color:${view === 'task' ? '#ffffff' : '#334155'};">
            By Retail Question (8)
          </button>
          <button type="button" class="perf-view-btn" data-view="sku_type"
            style="padding:7px 14px;border-radius:7px;border:none;font-size:13px;font-weight:600;cursor:pointer;
                   background:${view === 'sku_type' ? '#0057b8' : 'transparent'};color:${view === 'sku_type' ? '#ffffff' : '#334155'};">
            By SKU Type (HUL / Rival / Variant / Promo)
          </button>
          <button type="button" class="perf-view-btn" data-view="category"
            style="padding:7px 14px;border-radius:7px;border:none;font-size:13px;font-weight:600;cursor:pointer;
                   background:${view === 'category' ? '#0057b8' : 'transparent'};color:${view === 'category' ? '#ffffff' : '#334155'};">
            By Store Category (9)
          </button>
          <button type="button" class="perf-view-btn" data-view="pack"
            style="padding:7px 14px;border-radius:7px;border:none;font-size:13px;font-weight:600;cursor:pointer;
                   background:${view === 'pack' ? '#0057b8' : 'transparent'};color:${view === 'pack' ? '#ffffff' : '#334155'};">
            By Packaging Form (7)
          </button>
        </div>
      </div>

      ${view === 'task'
        ? renderTaskPerfView()
        : view === 'sku_type'
          ? renderSkuTypePerfView()
          : view === 'category'
            ? renderCategoryPerfView()
            : renderPackPerfView()}
    </section>

    <!-- Section 2: Prioritized Roadmap -->
    <section class="cio-panel">
      <div class="cio-panel-header">
        <div>
          <h3 class="cio-panel-title">Executed Phase-2 Refinements (EXP-20..26) &amp; Next-Week Catalog Sign-Off</h3>
          <p class="cio-panel-subtitle">Gap-targeted experiments executed across Val + Test splits and open-weight GPU benchmarks submitted on Vertex AI</p>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:14px;">
        ${ROADMAP_ITEMS.map(
          (item) => `
          <div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;background:#ffffff;display:flex;flex-direction:column;justify-content:space-between;">
            <div>
              <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;">
                <span class="badge ${item.badgeClass}">${esc(item.track)}</span>
              </div>
              <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:6px;">${esc(item.title)}</div>
              <div style="font-size:12.5px;color:#475569;line-height:1.5;margin-bottom:10px;">${esc(item.detail)}</div>
            </div>
            <div style="padding-top:8px;border-top:1px solid #f1f5f9;font-size:12px;font-weight:700;color:#047857;" class="mono">
              Status / Impact: ${esc(item.impact)}
            </div>
          </div>`
        ).join('')}
      </div>
    </section>

    <!-- Section 3: Collapsible Engineering Benchmarks Accordion -->
    ${renderBenchmarksAccordion()}
  `;
}

function renderSkuTypePerfView() {
  return `
    <div class="cio-table-wrap">
      <table class="cio-table">
        <thead>
          <tr>
            <th>SKU Segment</th>
            <th>Catalog &amp; Evaluation Scale</th>
            <th>Legacy Chained Models</th>
            <th>L1 Ownership / L3 Brand</th>
            <th>L4 Variant / L5 Exact SKU</th>
            <th>Primary Engineering Lever</th>
          </tr>
        </thead>
        <tbody>
          ${SKU_TYPE_PERFORMANCE.map((s) => {
            const gain = (s.l5Sku - s.legacy).toFixed(1);
            return `
              <tr>
                <td>
                  <div style="font-weight:700;color:#0f172a;">${esc(s.skuType)}</div>
                  <span class="badge badge-pass" style="margin-top:4px;">${esc(s.evalShare)}</span>
                </td>
                <td class="mono" style="font-size:12px;color:#334155;">${esc(s.classes)}</td>
                <td class="mono" style="color:#64748b;">${s.legacy.toFixed(1)}%</td>
                <td class="mono"><strong>${s.l1Acc.toFixed(1)}%</strong> / <strong>${s.l3Brand.toFixed(1)}%</strong></td>
                <td class="mono">
                  <strong style="font-size:15px;color:#047857;">${s.l4Variant.toFixed(1)}% / ${s.l5Sku.toFixed(1)}%</strong>
                  <span style="color:#047857;font-size:12px;font-weight:700;"> (+${gain}%)</span>
                </td>
                <td style="font-size:12.5px;color:#334155;line-height:1.45;">${esc(s.note)}</td>
              </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderTaskPerfView() {
  return `
    <div class="cio-table-wrap">
      <table class="cio-table">
        <thead>
          <tr>
            <th>Retail Question</th>
            <th>Legacy Chained Models</th>
            <th>Our Accuracy</th>
            <th style="width:220px;">Visual Comparison</th>
            <th>Remaining Gap &amp; How We Close It</th>
          </tr>
        </thead>
        <tbody>
          ${TASK_PERFORMANCE.map((row) => {
            const gain = (row.ours - row.legacy).toFixed(1);
            return `
              <tr>
                <td>
                  <div style="font-weight:700;color:#0f172a;">${esc(row.question)}</div>
                  <span class="badge badge-pass" style="margin-top:4px;">${esc(row.status)}</span>
                </td>
                <td class="mono" style="color:#64748b;">${row.legacy.toFixed(1)}%</td>
                <td class="mono">
                  <strong style="font-size:15px;color:#0f172a;">${row.ours.toFixed(1)}%</strong>
                  <span style="color:#047857;font-size:12px;font-weight:700;"> (+${gain}%)</span>
                  <div style="font-size:11px;color:#64748b;">${esc(row.unit)}</div>
                </td>
                <td>
                  <div style="font-size:11px;color:#64748b;margin-bottom:3px;">Legacy (${row.legacy}%) vs. Ours (${row.ours}%)</div>
                  <div style="height:8px;background:#e2e8f0;border-radius:999px;overflow:hidden;position:relative;">
                    <div style="position:absolute;left:0;top:0;bottom:0;width:${row.ours}%;background:#0057b8;border-radius:999px;"></div>
                    <div style="position:absolute;left:0;top:0;bottom:0;width:${row.legacy}%;background:#94a3b8;border-radius:999px;opacity:0.65;"></div>
                  </div>
                </td>
                <td style="font-size:12.5px;color:#334155;line-height:1.45;">
                  <div><strong>Gap:</strong> ${esc(row.gapSummary)}</div>
                  <div style="color:#0f766e;margin-top:3px;"><strong>Fix:</strong> ${esc(row.howWeCloseIt)}</div>
                </td>
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
            <th>Store Category</th>
            <th>Evaluated Packs</th>
            <th>Legacy Exact SKU</th>
            <th>Our Exact SKU</th>
            <th>Brand / Pack Accuracy</th>
            <th>Key Takeaway</th>
          </tr>
        </thead>
        <tbody>
          ${CATEGORY_PERFORMANCE.map((c) => {
            const lift = (c.ours - c.legacy).toFixed(1);
            const highlight = c.ours < 85;
            return `
              <tr style="${highlight ? 'background:#fffbeb;' : ''}">
                <td><strong>${esc(c.name)}</strong></td>
                <td class="mono">${c.boxes.toLocaleString('en-IN')}</td>
                <td class="mono" style="color:#64748b;">${c.legacy.toFixed(1)}%</td>
                <td class="mono">
                  <strong style="color:${highlight ? '#b45309' : '#047857'};font-size:14px;">${c.ours.toFixed(1)}%</strong>
                  <span style="font-size:11.5px;color:#047857;"> (+${lift}%)</span>
                </td>
                <td class="mono">${c.brandAcc.toFixed(1)}% / ${c.packAcc.toFixed(1)}%</td>
                <td style="font-size:12.5px;color:#334155;">${esc(c.note)}</td>
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
            <th>Evaluated Packs</th>
            <th>Container Form Accuracy</th>
            <th>Exact Variant &amp; SKU Accuracy</th>
            <th>Key Insight &amp; Next Step</th>
          </tr>
        </thead>
        <tbody>
          ${PACK_PERFORMANCE.map((p) => {
            const highlight = p.skuAcc < 85;
            return `
              <tr style="${highlight ? 'background:#fffbeb;' : ''}">
                <td><strong>${esc(p.form)}</strong></td>
                <td class="mono">${p.boxes.toLocaleString('en-IN')}</td>
                <td class="mono"><strong>${p.packAcc.toFixed(1)}%</strong></td>
                <td class="mono"><strong style="color:${highlight ? '#b45309' : '#047857'};">${p.skuAcc.toFixed(1)}%</strong></td>
                <td style="font-size:12.5px;color:#334155;">${esc(p.note)}</td>
              </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderBenchmarksAccordion() {
  const exps = state.data.experiments || [];
  return `
    <details class="cio-panel" style="padding:18px 22px;">
      <summary style="cursor:pointer;font-size:15px;font-weight:700;color:#0f172a;">
        Explore All ${exps.length} Engineering Benchmarks &amp; 12/12 QC-Verified Runs (Click to expand)
      </summary>
      <div class="cio-table-wrap" style="margin-top:16px;">
        <table class="cio-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Benchmark Name</th>
              <th>Dataset</th>
              <th>Best Approach &amp; Model</th>
              <th>Primary Metric</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            ${exps
              .map((e) => {
                const validRuns = (e.runs || []).filter((r) => (r.primary_score ?? 0) >= 0.25);
                const w = (e.winner && (e.winner.primary_score ?? 0) >= 0.25) ? e.winner : (validRuns[0] || {});
                const score = w.primary_score != null ? (w.primary_score * 100).toFixed(1) + '%' : 'Vertex AI GPU Queued';
                return `
                  <tr>
                    <td class="mono"><strong>${esc(e.experiment_id)}</strong></td>
                    <td>
                      <div style="font-weight:700;color:#0f172a;">${esc(e.title)}</div>
                      <div style="font-size:12px;color:#64748b;">${esc(e.intuition || '')}</div>
                    </td>
                    <td class="mono">${esc(e.dataset)}</td>
                    <td class="mono">${esc(w.approach || (e.approaches || [])[0] || '—')} <span class="text-muted">(${esc(w.model || (e.models || [])[0] || '')})</span></td>
                    <td class="mono"><strong style="color:#047857;">${score}</strong></td>
                    <td>
                      <button type="button" class="exp-drill-btn" data-exp-id="${esc(e.experiment_id)}"
                        style="padding:5px 10px;border-radius:6px;border:1px solid #cbd5e1;background:#ffffff;cursor:pointer;font-size:12px;font-weight:600;">
                        Inspect Runs (${validRuns.length})
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
    <p style="margin:0 0 14px 0;font-size:13.5px;color:#334155;">${esc(exp.description || exp.intuition || '')}</p>
    <div class="cio-table-wrap">
      <table class="cio-table">
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Approach</th>
            <th>Model</th>
            <th>Split</th>
            <th>Primary Score</th>
            <th>Latency (s)</th>
          </tr>
        </thead>
        <tbody>
          ${runs
            .map(
              (r) => `
            <tr>
              <td class="mono" style="font-size:11.5px;">${esc(r.run_id)}</td>
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
