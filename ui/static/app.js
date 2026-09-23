/**
 * Unilever Shelf Understanding — Use-Case Pipeline & Benchmark Studio
 * Separates all 4 Architectural Paths into dedicated linear Step-by-Step tabs
 * and shares the exact same visual Step-by-Step renderer with the Run Live Studio tab.
 */

let DASHBOARD_DATA = null;
let SHELF_IMAGE_OBJ = null;

// ============================================================================
// OUTPUT ENCODING
//
// Everything this dashboard renders that did not originate in this file is
// untrusted input: brand / product / variant strings are model-generated, the
// catalog and ground-truth rows are operator-supplied files, approach plugins
// self-report their display names and stage descriptions, and OTel attributes
// are read off disk. All of it must be escaped before it is interpolated into
// an innerHTML template literal or into an HTML attribute value, otherwise a
// crafted product name on a shelf becomes script in the operator's browser.
//
// Interactive controls are wired via delegated listeners keyed on data-action
// (see setupDelegatedActions) rather than inline on* attributes, because an
// inline handler assembled by string interpolation puts untrusted text directly
// into a JavaScript parsing context, where HTML escaping alone does not help.
// ============================================================================
const HTML_ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value).replace(/[&<>"']/g, (ch) => HTML_ESCAPES[ch]);
}

// Short alias; this is used at well over a hundred interpolation sites.
const esc = escapeHtml;

// ============================================================================
// HONEST RENDERING OF MEASUREMENTS
//
// A benchmark that prints a plausible number where it holds no measurement is
// worse than one that prints nothing, because the reader cannot tell the two
// apart. These helpers render an explicit marker instead of falling back to 0,
// 1.0, or any other good-looking default.
//
// reportedNumber() is defined further down next to the cost helpers; function
// declarations hoist, so it is callable from here.
// ============================================================================
const NOT_MEASURED_HTML = `<span class="muted">not measured</span>`;
const NOT_REPORTED_HTML = `<span class="muted">not reported</span>`;
const NOT_MEASURED_TEXT = "not measured";

function fmtFixed(value, digits, fallback = NOT_REPORTED_HTML) {
  const n = reportedNumber(value);
  return n === null ? fallback : n.toFixed(digits);
}

function fmtUsd(value, digits = 6, fallback = NOT_REPORTED_HTML) {
  const n = reportedNumber(value);
  return n === null ? fallback : `$${n.toFixed(digits)}`;
}

function fmtMs(value, digits = 1, fallback = NOT_REPORTED_HTML) {
  const n = reportedNumber(value);
  return n === null ? fallback : `${n.toFixed(digits)} ms`;
}

// Expects a 0..1 ratio (confidence, accuracy, F1).
function fmtPercent(value, digits = 1, fallback = NOT_MEASURED_HTML) {
  const n = reportedNumber(value);
  return n === null ? fallback : `${(n * 100).toFixed(digits)}%`;
}

function fmtCount(value, fallback = NOT_REPORTED_HTML) {
  const n = reportedNumber(value);
  return n === null ? fallback : String(n);
}

// Per-container state for selected model, selected facing index, and depth toggle
const CONTAINER_STATE = {
  "pipeline-container-path-1": { approachId: "single_pass_full_shelf", model: "gemini-3.8-flash", selectedIdx: 0, showDepth: true, livePayload: null },
  "pipeline-container-path-2": { approachId: "two_stage_bbox_guided_nms", model: "gemini-3.8-flash", selectedIdx: 0, showDepth: true, livePayload: null },
  "pipeline-container-path-3": { approachId: "two_stage_physical_crop_per_facing", model: "gemini-3.8-flash", selectedIdx: 0, showDepth: true, livePayload: null },
  "pipeline-container-path-4": { approachId: "class_agnostic_visual_embedding", model: "gemini-3.8-flash", selectedIdx: 0, showDepth: true, livePayload: null },
  "pipeline-container-live":   { approachId: "two_stage_physical_crop_per_facing", model: "gemini-3.8-flash", selectedIdx: 0, showDepth: true, livePayload: null },
};

// Metadata & Exact Linear Steps for Each of the 4 Use-Case Paths
const USE_CASE_PATHS = {
  single_pass_full_shelf: {
    id: "single_pass_full_shelf",
    tabId: "tab-path-1",
    containerId: "pipeline-container-path-1",
    shortLabel: "Path 1: Single-Pass VLM",
    title: "Use Case Path 1: Single-Pass Full-Shelf VLM (1 API Call)",
    architectureTag: "1-Stage VLM Pipeline (No Separate Detector or Cropper)",
    summary: "Sends the full shelf image in a single multimodal Gemini call that simultaneously detects [ymin, xmin, ymax, xmax] front-facing bounding boxes and extracts all 7 taxonomy dimensions.",
    steps: [
      {
        num: "Step 1",
        title: "Full-Shelf Single-Pass VLM Detection + 7-Dim Classification",
        desc: "1 API call returns normalized [ymin, xmin, ymax, xmax] boxes and 7 taxonomy attributes directly from the full shelf image."
      },
      {
        num: "Step 2",
        title: "Front-Facing Depth Deduplication & Rule-Derived Size Bucketing",
        desc: "Suppresses any back-row depth duplicates in the same horizontal column (>=55% X-overlap) and computes rule-derived size buckets."
      },
      {
        num: "Step 3",
        title: "3072-D Hybrid Vector (gemini-embedding-001) + BM25 Catalog Search",
        desc: "Embeds structured product passages into 3072-D dense vectors combined with sparse BM25 lexical keywords for SKU retrieval."
      },
      {
        num: "Step 4",
        title: "Benchmark Cost, Latency, Token Breakdown & OpenTelemetry Audit",
        desc: "Computes exact cost/shelf image, cost/front facing, thinking/input/output tokens, and logs an OTel trace span."
      }
    ]
  },

  two_stage_bbox_guided_nms: {
    id: "two_stage_bbox_guided_nms",
    tabId: "tab-path-2",
    containerId: "pipeline-container-path-2",
    shortLabel: "Path 2: 2-Stage BBox + Depth NMS",
    title: "Use Case Path 2: 2-Stage BBox-Guided Detection + Depth NMS (2 API Calls)",
    architectureTag: "2-Stage Coordinate-Conditioned VLM Pipeline",
    summary: "Decouples localization from classification: Stage 1 detects all candidate boxes and runs geometric Depth NMS to filter out back-row stacked duplicates; Stage 2 passes the surviving [ymin, xmin, ymax, xmax] coordinates into Gemini for focused 7-dimension classification.",
    steps: [
      {
        num: "Step 1",
        title: "Stage 1 Zero-Shot Front-Facing Detection + Depth Duplicate NMS",
        desc: "Detects all candidate shelf items and filters out back-row depth duplicates (dashed red) behind front facings."
      },
      {
        num: "Step 2",
        title: "Stage 2 Coordinate-Conditioned 7-Dim VLM Classification",
        desc: "Passes the exact surviving [ymin, xmin, ymax, xmax] front-facing coordinates into Gemini to classify all 7 dimensions + size rules."
      },
      {
        num: "Step 3",
        title: "3072-D Hybrid Vector (gemini-embedding-001) + BM25 Catalog Search",
        desc: "Generates 3072-D dense vectors + sparse lexical keywords for every coordinate-classified front facing."
      },
      {
        num: "Step 4",
        title: "Combined 2-Stage Cost, Latency, Token Breakdown & OTel Audit",
        desc: "Aggregates Stage 1 Detection + Stage 2 Classification tokens, latency, and cost per facing."
      }
    ]
  },

  two_stage_physical_crop_per_facing: {
    id: "two_stage_physical_crop_per_facing",
    tabId: "tab-path-3",
    containerId: "pipeline-container-path-3",
    shortLabel: "Path 3: 2-Stage Physical Crop",
    title: "Use Case Path 3: 2-Stage Physical Bounding-Box Crop per Facing (PIL Crops + VLM)",
    architectureTag: "2-Stage Physical Image Cropping + Montage VLM Pipeline",
    summary: "Stage 1 detects front facings and filters back-row duplicates; Stage 2 physically crops every bounding box via Pillow into individual high-res PNGs (facing_01..N.png) and a numbered montage strip so Gemini reads fine-print gram/ml and variant OCR directly from the crops.",
    steps: [
      {
        num: "Step 1",
        title: "Stage 1 Zero-Shot Front-Facing Detection + Depth Duplicate NMS",
        desc: "Locates front-facing slots [ymin, xmin, ymax, xmax] and removes depth-stacked back-row items."
      },
      {
        num: "Step 2",
        title: "Physical PIL Bounding-Box Cropping (facing_01..N.png) & Montage Strip",
        desc: "Crops each detected front facing from the high-res shelf image into standalone PNG crops and a numbered montage strip."
      },
      {
        num: "Step 3",
        title: "Stage 2 High-Res Crop 7-Dim VLM Classification & Size Rules",
        desc: "Passes the physical facing crops + montage to Gemini to extract fine-print variant, packaging, and gram/ml size buckets."
      },
      {
        num: "Step 4",
        title: "3072-D Hybrid Search & End-to-End Benchmark Metrics",
        desc: "Generates 3072-D hybrid search vectors and reports combined Stage 1 + Crop + Stage 2 cost and latency."
      }
    ]
  },

  class_agnostic_visual_embedding: {
    id: "class_agnostic_visual_embedding",
    tabId: "tab-path-4",
    containerId: "pipeline-container-path-4",
    shortLabel: "Path 4: 3-Stage Class-Agnostic + 1408-D Vector",
    title: "Use Case Path 4: 3-Stage Class-Agnostic Detector + 1408-D Visual Embedding + ScaNN Vector Search",
    architectureTag: "3-Stage Classic CV + Contrastive ViT Metric Learning + ScaNN ANN",
    summary: "Zero generative VLM classification! Stage 1 runs a single-class ('product') object detector; Stage 2 extracts physical image crops and embeds each crop into a 1408-D visual vector via Vertex AI multimodalembedding@001; Stage 3 runs Cosine ANN / ScaNN vector search against reference catalog image embeddings.",
    steps: [
      {
        num: "Stage 1",
        title: "Class-Agnostic Object Detection (Single Class: 'product') + Depth NMS",
        desc: "Detects every physical item on the shelf strictly as class='product' without reading any brand or OCR text."
      },
      {
        num: "Stage 2",
        title: "Physical Crop Extraction + 1408-D Visual Metric Learning (multimodalembedding@001)",
        desc: "Passes each raw PNG crop into Vertex AI's Contrastive Vision Transformer to produce a normalized 1408-D float vector."
      },
      {
        num: "Stage 3",
        title: "ScaNN / Cosine ANN Image-to-Image Vector Catalog Matching",
        desc: "Matches each 1408-D crop vector against the pre-computed 1408-D reference catalog database to identify SKU & inherit 7-Dim taxonomy."
      },
      {
        num: "Stage 4",
        title: "3-Stage Detector + Visual Embedding Cost, Latency & OTel Audit",
        desc: "Reports ultra-low token usage (no generative VLM classification tokens), latency, and cost per facing."
      }
    ]
  }
};

document.addEventListener("DOMContentLoaded", async () => {
  setupTabNavigation();
  setupDelegatedActions();
  await loadShelfImage();
  await fetchDashboardData();
  renderOverviewTab();
  renderAllFourUseCaseTabs();
  renderSftAndConfigTab();
  initLiveStudioTab();
});

function setupTabNavigation() {
  const buttons = document.querySelectorAll(".nav-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      switchToTab(targetTab);
    });
  });
}

// Single pair of delegated listeners for every dynamically rendered control.
// Panels are re-rendered wholesale via innerHTML, so listeners attached to the
// individual elements would be thrown away on each render; delegating from
// `document` survives that. Parameters travel in data-* attributes (escaped as
// ordinary attribute values) instead of being spliced into executable JS.
function setupDelegatedActions() {
  const actionTarget = (evt) =>
    evt.target && evt.target.closest ? evt.target.closest("[data-action]") : null;

  document.addEventListener("click", (evt) => {
    const el = actionTarget(evt);
    if (!el) return;
    switch (el.getAttribute("data-action")) {
      case "switch-tab":
        switchToTab(el.getAttribute("data-tab-id"));
        break;
      case "open-live-studio":
        openInLiveStudio(
          el.getAttribute("data-approach-id"),
          el.getAttribute("data-model-name")
        );
        break;
      case "jump-path-model":
        jumpToPathAndModel(
          el.getAttribute("data-approach-id"),
          el.getAttribute("data-model-name")
        );
        break;
      case "select-facing":
        selectPipelineFacing(
          el.getAttribute("data-container-id"),
          Number(el.getAttribute("data-facing-idx"))
        );
        break;
      default:
        break;
    }
  });

  document.addEventListener("change", (evt) => {
    const el = actionTarget(evt);
    if (!el) return;
    switch (el.getAttribute("data-action")) {
      case "change-model":
        changePipelineModel(el.getAttribute("data-container-id"), el.value);
        break;
      case "toggle-depth":
        togglePipelineDepth(el.getAttribute("data-container-id"), el.checked);
        break;
      default:
        break;
    }
  });
}

function switchToTab(targetTab) {
  document.querySelectorAll(".nav-btn").forEach((b) => {
    b.classList.toggle("active", b.getAttribute("data-tab") === targetTab);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === targetTab);
  });

  // Redraw canvas inside newly activated tab
  if (targetTab === "tab-path-1") renderUseCasePipeline("pipeline-container-path-1");
  if (targetTab === "tab-path-2") renderUseCasePipeline("pipeline-container-path-2");
  if (targetTab === "tab-path-3") renderUseCasePipeline("pipeline-container-path-3");
  if (targetTab === "tab-path-4") renderUseCasePipeline("pipeline-container-path-4");
  if (targetTab === "tab-run-live") renderUseCasePipeline("pipeline-container-live");
}

async function loadShelfImage() {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      SHELF_IMAGE_OBJ = img;
      resolve();
    };
    img.onerror = () => resolve();
    img.src = "/shelf-image.png";
  });
}

async function fetchDashboardData() {
  const res = await fetch("/api/dashboard");
  DASHBOARD_DATA = await res.json();
  if (DASHBOARD_DATA && DASHBOARD_DATA.project_info) {
    const badgeProj = document.getElementById("badge-project");
    if (badgeProj) {
      badgeProj.textContent = `GCP: ${DASHBOARD_DATA.project_info.gcp_project_id} (${DASHBOARD_DATA.project_info.location}) • Host: ${DASHBOARD_DATA.project_info.cloud_run_service || "local"}`;
    }
    const registered = DASHBOARD_DATA.project_info.registered_approaches || [];
    const liveSelect = document.getElementById("live-studio-approach");
    const filterSelect = document.getElementById("overview-path-filter");
    registered.forEach((p) => {
      if (!USE_CASE_PATHS[p.approach_id]) {
        USE_CASE_PATHS[p.approach_id] = {
          id: p.approach_id,
          tabId: "tab-run-live",
          containerId: "pipeline-container-live",
          shortLabel: p.display_name || p.approach_id,
          title: p.display_name || p.approach_id,
          architectureTag: `Plugin (${p.category || "custom"})`,
          summary: `Auto-discovered approach plugin (${p.approach_id}).`,
          steps: (p.stages_description || ["Stage 1: Custom Plugin Execution"]).map((s, i) => ({
            num: `Stage ${i + 1}`,
            title: s,
            desc: s,
          })),
        };
        if (liveSelect && !Array.from(liveSelect.options).some((o) => o.value === p.approach_id)) {
          const opt = document.createElement("option");
          opt.value = p.approach_id;
          opt.textContent = `Plugin: ${p.display_name || p.approach_id}`;
          liveSelect.appendChild(opt);
        }
        if (filterSelect && !Array.from(filterSelect.options).some((o) => o.value === p.approach_id)) {
          const opt = document.createElement("option");
          opt.value = p.approach_id;
          opt.textContent = `Plugin: ${p.display_name || p.approach_id}`;
          filterSelect.appendChild(opt);
        }
      }
    });
  }
}

// ============================================================================
// TAB 1: OVERVIEW & 4-PATH COMPARISON
// ============================================================================
function renderOverviewTab() {
  if (!DASHBOARD_DATA) return;
  const cardsContainer = document.getElementById("overview-usecase-cards");
  const summary = DASHBOARD_DATA.summary || [];

  cardsContainer.innerHTML = Object.values(USE_CASE_PATHS).map((pathMeta, idx) => {
    // Find gemini-3.8-flash summary row for this path
    const rec38 = summary.find(
      (r) => r.task_type === "classification" && r.separation_approach === pathMeta.id && r.model_name === "gemini-3.8-flash"
    ) || {};
    const facings = rec38.front_facings_count ?? "--";
    const latency = fmtMs(rec38.latency_ms, 0, "--");
    const costImg = fmtUsd(rec38.cost_per_shelf_image_usd, 5, "--");

    const stepsHtml = pathMeta.steps.map(
      (s) => `<li class="mini-flow-item"><strong>${esc(s.num)}:</strong> ${esc(s.title)}</li>`
    ).join("");

    return `
      <div class="usecase-card">
        <div>
          <div class="usecase-card-header">
            <span class="usecase-tag">Use Case ${idx + 1}</span>
            <span class="mono muted" style="font-size:11px;">${esc(pathMeta.architectureTag)}</span>
          </div>
          <h3>${esc(pathMeta.shortLabel)}</h3>
          <p class="muted">${esc(pathMeta.summary)}</p>
          <ul class="mini-flow-list">${stepsHtml}</ul>
        </div>
        <div>
          <div style="display:flex; justify-content:space-between; font-size:12px; background:#f8fafc; padding:8px 10px; border-radius:6px; margin-bottom:10px;">
            <span><strong>3.8-flash:</strong> ${esc(facings)} facings</span>
            <span><strong>Latency:</strong> ${latency}</span>
            <span><strong>Cost:</strong> ${costImg}</span>
          </div>
          <div class="usecase-actions">
            <button class="btn-primary" data-action="switch-tab" data-tab-id="${esc(pathMeta.tabId)}">
              Inspect Step-by-Step Flow &rarr;
            </button>
            <button class="btn-secondary" data-action="open-live-studio" data-approach-id="${esc(pathMeta.id)}" data-model-name="gemini-3.8-flash">
              &#9889; Run Live
            </button>
          </div>
        </div>
      </div>
    `;
  }).join("");

  // Render Overview KPIs
  const kpiEl = document.getElementById("overview-kpis");
  const clsRuns = summary.filter((r) => r.task_type === "classification");
  const lowestCostRun = [...clsRuns].sort((a, b) => a.cost_per_shelf_image_usd - b.cost_per_shelf_image_usd)[0];
  const fastestRun = [...clsRuns].sort((a, b) => a.latency_ms - b.latency_ms)[0];

  kpiEl.innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">Separated Use-Case Paths</div>
      <div class="kpi-value">4 Paths</div>
      <div class="kpi-sub">1-Pass VLM &bull; 2-Stage BBox &bull; Physical Crop &bull; 3-Stage 1408-D Vector</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total Live Vertex AI Runs</div>
      <div class="kpi-value">${summary.length} Runs</div>
      <div class="kpi-sub">Across gemini-3.8-flash, 3.7-flash &amp; 3.5-flash-lite</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Fastest Classification Path</div>
      <div class="kpi-value">${fastestRun ? fmtFixed(reportedNumber(fastestRun.latency_ms) === null ? null : fastestRun.latency_ms / 1000, 1, "--") + "s" : "--"}</div>
      <div class="kpi-sub">${fastestRun ? `${esc(fastestRun.separation_approach)} (${esc(fastestRun.model_name)})` : ""}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Lowest Cost / Shelf Image</div>
      <div class="kpi-value">${lowestCostRun ? fmtUsd(lowestCostRun.cost_per_shelf_image_usd, 5, "--") : "--"}</div>
      <div class="kpi-sub">${lowestCostRun ? `${esc(lowestCostRun.separation_approach)} (${esc(lowestCostRun.model_name)})` : ""}</div>
    </div>
  `;

  // Filters for Overview Table
  const pathFilter = document.getElementById("overview-path-filter");
  const modelFilter = document.getElementById("overview-model-filter");
  const renderMatrix = () => {
    const pVal = pathFilter.value;
    const mVal = modelFilter.value;
    const tbody = document.querySelector("#overview-summary-table tbody");
    const filtered = summary.filter((r) => {
      if (pVal !== "ALL" && r.separation_approach !== pVal) return false;
      if (mVal !== "ALL" && r.model_name !== mVal) return false;
      return true;
    });
    tbody.innerHTML = filtered.map((r) => {
      const pathMeta = USE_CASE_PATHS[r.separation_approach];
      const jumpBtn = pathMeta
        ? `<button class="btn-secondary" style="padding:4px 8px; font-size:11px;" data-action="jump-path-model" data-approach-id="${esc(r.separation_approach)}" data-model-name="${esc(r.model_name)}">Open Path</button>`
        : `<span class="muted">Standalone</span>`;
      return `
        <tr>
          <td><strong>${esc(pathMeta ? pathMeta.shortLabel : r.separation_approach)}</strong></td>
          <td><code>${esc(r.task_type)}</code></td>
          <td><code>${esc(r.model_name)}</code></td>
          <td><strong>${fmtCount(r.front_facings_count)}</strong></td>
          <td>${fmtCount(r.depth_duplicates_filtered)}</td>
          <td>${fmtFixed(r.latency_ms, 1)}</td>
          <td>${fmtCount(r.input_tokens)}</td>
          <td>${fmtCount(r.thinking_tokens)}</td>
          <td>${fmtCount(r.output_tokens)}</td>
          <td><strong>${fmtUsd(r.cost_per_shelf_image_usd)}</strong></td>
          <td>${fmtUsd(r.cost_per_product_usd)}</td>
          <td>${jumpBtn}</td>
        </tr>
      `;
    }).join("");
  };
  pathFilter.addEventListener("change", renderMatrix);
  modelFilter.addEventListener("change", renderMatrix);
  renderMatrix();
}

window.jumpToPathAndModel = function (approachId, modelName) {
  const pathMeta = USE_CASE_PATHS[approachId];
  if (!pathMeta) return;
  CONTAINER_STATE[pathMeta.containerId].model = modelName;
  CONTAINER_STATE[pathMeta.containerId].selectedIdx = 0;
  switchToTab(pathMeta.tabId);
};

window.openInLiveStudio = function (approachId, modelName) {
  const appSelect = document.getElementById("live-studio-approach");
  const modSelect = document.getElementById("live-studio-model");
  if (appSelect) appSelect.value = approachId;
  if (modSelect) modSelect.value = modelName;
  CONTAINER_STATE["pipeline-container-live"].approachId = approachId;
  CONTAINER_STATE["pipeline-container-live"].model = modelName;
  CONTAINER_STATE["pipeline-container-live"].selectedIdx = 0;
  switchToTab("tab-run-live");
};

// ============================================================================
// RENDER ALL 4 USE-CASE TABS + SHARED LINEAR STEP-BY-STEP PIPELINE COMPONENT
// ============================================================================
function renderAllFourUseCaseTabs() {
  renderUseCasePipeline("pipeline-container-path-1");
  renderUseCasePipeline("pipeline-container-path-2");
  renderUseCasePipeline("pipeline-container-path-3");
  renderUseCasePipeline("pipeline-container-path-4");
}

function getPipelineRunData(approachId, modelName, livePayload) {
  if (livePayload && livePayload.separation_approach === approachId && livePayload.model_name === modelName) {
    return {
      isLiveRun: true,
      summaryRec: livePayload.summary_record,
      rows: livePayload.rows || [],
      cropsInfo: livePayload.crops_info || {},
      depthDemo: livePayload.depth_demo || {},
    };
  }
  const summaryRec = (DASHBOARD_DATA.summary || []).find(
    (s) => s.task_type === "classification" && s.separation_approach === approachId && s.model_name === modelName
  ) || {};
  const rows = (DASHBOARD_DATA.rows || []).filter(
    (r) => r.task_type === "classification" && r.separation_approach === approachId && r.model_name === modelName
  );
  // The manifest is keyed by "<model>_<approach>" (see ui/server.py:188 and the
  // directory names under reports/crops/). The previous keys ("visual_embed_<model>"
  // and a bare "<model>") matched nothing, so cropsInfo was always empty.
  const cropsKey = `${modelName}_${approachId}`;
  const cropsInfo = (DASHBOARD_DATA.crops_manifest || {})[cropsKey] || {};
  const depthDemo = (DASHBOARD_DATA.depth_demos || {})[modelName] || {};
  return { isLiveRun: false, summaryRec, rows, cropsInfo, depthDemo };
}

function renderUseCasePipeline(containerId) {
  if (!DASHBOARD_DATA) return;
  const state = CONTAINER_STATE[containerId];
  const approachId = state.approachId;
  const modelName = state.model;
  const pathMeta = USE_CASE_PATHS[approachId];
  const container = document.getElementById(containerId);
  if (!container || !pathMeta) return;

  const { isLiveRun, summaryRec, rows, cropsInfo, depthDemo } = getPipelineRunData(
    approachId,
    modelName,
    state.livePayload
  );

  if (state.selectedIdx >= rows.length) {
    state.selectedIdx = 0;
  }
  const selectedRow = rows[state.selectedIdx] || rows[0] || null;

  // 1. Build Linear Flowchart Stepper HTML
  const flowStepperHtml = pathMeta.steps.map((s) => `
    <div class="flow-step-node done-stage">
      <span class="flow-step-num">${esc(s.num)}</span>
      <div class="flow-step-title">${esc(s.title)}</div>
      <div class="flow-step-desc">${esc(s.desc)}</div>
    </div>
  `).join("");

  // 2. Build Path-Specific Intermediate Stage Card (so each path clearly shows ONLY the stages that belong to it!)
  let intermediateStageHtml = "";

  if (approachId === "single_pass_full_shelf") {
    intermediateStageHtml = `
      <div class="linear-step-card">
        <div class="linear-step-header">
          <h3><span class="step-badge-pill">Step 2</span> Single-Pass Depth Deduplication &amp; 7-Dimension Taxonomy Extraction</h3>
          <span class="muted">No intermediate crop or Stage-1 API call required — all ${rows.length} front facings extracted in 1 VLM pass.</span>
        </div>
        ${buildSevenDimensionTableHtml(rows, containerId, state.selectedIdx, false)}
      </div>
    `;
  } else if (approachId === "two_stage_bbox_guided_nms") {
    const rawCount = depthDemo.raw_candidate_count || (rows.length + (summaryRec.depth_duplicates_filtered || 0));
    const filteredCount = depthDemo.depth_duplicates_filtered ?? (summaryRec.depth_duplicates_filtered || 0);
    intermediateStageHtml = `
      <div class="linear-step-card">
        <div class="linear-step-header">
          <h3><span class="step-badge-pill">Step 2</span> Coordinate-Conditioned 7-Dimension Classification (Stage 1 Boxes &rarr; Stage 2 VLM Prompt)</h3>
          <span class="muted">Stage 1 filtered <strong>${esc(rawCount)} raw candidates &rarr; ${rows.length} front facings</strong> (${esc(filteredCount)} back-row depth duplicates removed) before Stage 2 classification.
          ${depthDemo.suppressed_boxes_note ? `<br /><em>${esc(depthDemo.suppressed_boxes_note)}</em>` : ""}</span>
        </div>
        ${buildSevenDimensionTableHtml(rows, containerId, state.selectedIdx, false)}
      </div>
    `;
  } else if (approachId === "two_stage_physical_crop_per_facing") {
    // The server emits {montage_url, facing_urls} (ui/server.py:188-191). Reading
    // crop_files / montage_file meant this gallery could never render.
    const cropFiles = cropsInfo.facing_urls || [];
    const montageUrl = cropsInfo.montage_url || "";
    const cropsGalleryHtml = cropFiles.map((url, idx) => {
      const r = rows[idx] || {};
      return `
        <div class="crop-card" data-action="select-facing" data-container-id="${esc(containerId)}" data-facing-idx="${idx}" style="cursor:pointer; border-color:${idx === state.selectedIdx ? '#0057b8' : '#e2e8f0'}">
          <img src="${esc(url)}" alt="Facing #${idx + 1}" loading="lazy" />
          <div><strong>Facing #${idx + 1}</strong></div>
          <div class="muted" style="font-size:11px;">${esc(r.predicted_brand) || "Crop"} (${esc(r.predicted_size)})</div>
        </div>
      `;
    }).join("");

    intermediateStageHtml = `
      <div class="linear-step-card">
        <div class="linear-step-header">
          <h3><span class="step-badge-pill">Step 2</span> Physical PIL Bounding-Box Cropping (<code>facing_01..${cropFiles.length}.png</code>) &amp; Numbered Montage Strip</h3>
          <span class="muted">Each detected front-facing bounding box is physically cropped from the shelf image and assembled into a numbered montage strip.</span>
        </div>
        ${montageUrl ? `<div style="margin-bottom:12px;"><div class="muted" style="margin-bottom:4px; font-weight:600;">Numbered Montage Strip Sent to Gemini Stage 2 (<code>${esc(montageUrl)}</code>):</div><img src="${esc(montageUrl)}" alt="Montage Strip" style="max-width:100%; border-radius:6px; border:1px solid #cbd5e1;" /></div>` : ""}
        <div class="crops-strip">${cropsGalleryHtml}</div>
      </div>

      <div class="linear-step-card">
        <div class="linear-step-header">
          <h3><span class="step-badge-pill">Step 3</span> High-Resolution Crop 7-Dimension VLM Classification &amp; Size Buckets</h3>
          <span class="muted">Fine-print variant, packaging form factor, and gram/ml size buckets read from the physical crops.</span>
        </div>
        ${buildSevenDimensionTableHtml(rows, containerId, state.selectedIdx, false)}
      </div>
    `;
  } else if (approachId === "class_agnostic_visual_embedding") {
    const cropFiles = cropsInfo.facing_urls || [];
    const vectorCardsHtml = rows.map((r, idx) => {
      const cropUrl = cropFiles[idx] || "";
      // The actual 1408-D embedding is never sent to the browser. This card used
      // to print a "preview slice" synthesised from (idx + 1) * 0.0137 and label
      // it as the vector, which is fabricated data presented as a measurement.
      return `
        <div class="crop-card" data-action="select-facing" data-container-id="${esc(containerId)}" data-facing-idx="${idx}" style="cursor:pointer; border-color:${idx === state.selectedIdx ? '#7c3aed' : '#e2e8f0'}">
          ${cropUrl ? `<img src="${esc(cropUrl)}" alt="Class-Agnostic Crop #${idx + 1}" loading="lazy" />` : `<div style="height:80px;background:#0f172a;color:#fff;display:flex;align-items:center;justify-content:center;">Crop #${idx + 1}</div>`}
          <div><strong>Crop #${idx + 1} (class: "product")</strong></div>
          <div class="vector-pill">1408-D ViT Vector<br/><span class="muted">component values not exported to the UI</span></div>
          <div style="margin-top:5px; font-size:11px; color:#065f46; font-weight:700;">
            ScaNN Match: ${esc(r.predicted_brand) || "--"} (${fmtPercent(r.confidence)})
          </div>
        </div>
      `;
    }).join("");

    intermediateStageHtml = `
      <div class="linear-step-card">
        <div class="linear-step-header">
          <h3><span class="step-badge-pill" style="background:#7c3aed;">Stage 2</span> Physical Bounding-Box Crop Extraction + 1408-D Visual Metric Learning (<code>multimodalembedding@001</code>)</h3>
          <span class="muted">Each class-agnostic crop is embedded directly into a 1408-dimensional L2-normalized float vector (0 generative VLM classification tokens).</span>
        </div>
        <div class="crops-strip">${vectorCardsHtml}</div>
      </div>

      <div class="linear-step-card">
        <div class="linear-step-header">
          <h3><span class="step-badge-pill" style="background:#059669;">Stage 3</span> ScaNN / Cosine ANN Image-to-Image Vector Search &amp; Catalog SKU Attribution</h3>
          <span class="muted">Compares each 1408-D crop embedding against pre-computed 1408-D reference catalog image embeddings via Cosine ANN to assign SKU &amp; 7-Dim taxonomy.</span>
        </div>
        ${buildSevenDimensionTableHtml(rows, containerId, state.selectedIdx, true)}
      </div>
    `;
  }

  // 3. Build Step 3/4 Catalog Vector Search + Final Benchmark Metrics Card
  const step3Badge = approachId === "class_agnostic_visual_embedding" ? "Stage 4" : (approachId === "two_stage_physical_crop_per_facing" ? "Step 4" : "Step 3 & 4");

  container.innerHTML = `
    <div class="pipeline-hero-card">
      <div class="pipeline-hero-top">
        <div>
          <span class="usecase-tag">${esc(pathMeta.architectureTag)}</span>
          ${isLiveRun ? `<span class="tag-hul" style="margin-left:8px;">&#9889; LIVE VERTEX AI EXECUTION OUTPUT</span>` : ""}
          <h2>${esc(pathMeta.title)}</h2>
          <p class="muted">${esc(pathMeta.summary)}</p>
        </div>
        <div class="pipeline-controls">
          <label style="font-size:12.5px; font-weight:700;">
            Active Model:
            <select data-action="change-model" data-container-id="${esc(containerId)}">
              <option value="gemini-3.8-flash" ${modelName === "gemini-3.8-flash" ? "selected" : ""}>gemini-3.8-flash</option>
              <option value="gemini-3.7-flash" ${modelName === "gemini-3.7-flash" ? "selected" : ""}>gemini-3.7-flash</option>
              <option value="gemini-3.5-flash-lite" ${modelName === "gemini-3.5-flash-lite" ? "selected" : ""}>gemini-3.5-flash-lite</option>
            </select>
          </label>
          ${containerId !== "pipeline-container-live" ? `
            <button class="btn-primary btn-live-execute" style="padding:7px 13px; font-size:12.5px;" data-action="open-live-studio" data-approach-id="${esc(approachId)}" data-model-name="${esc(modelName)}">
              &#9889; Run This Path Live in Studio
            </button>
          ` : ""}
        </div>
      </div>

      <!-- Linear Step-by-Step Visual Flow Diagram -->
      <div class="linear-flow-stepper">
        ${flowStepperHtml}
      </div>
    </div>

    <!-- STEP 1: INTERACTIVE BOUNDING BOX SHELF CANVAS & CLICK-TO-INSPECT FACING -->
    <div class="linear-step-card">
      <div class="linear-step-header">
        <h3>
          <span class="step-badge-pill">${approachId === "class_agnostic_visual_embedding" ? "Stage 1" : "Step 1"}</span>
          ${approachId === "class_agnostic_visual_embedding"
            ? "Class-Agnostic Object Detection (Single Class: 'product') + Front-Facing Depth Deduplication"
            : "Front-Facing Bounding-Box Localization &amp; Back-Row Depth Duplicate Suppression"}
        </h3>
        <label style="font-size:12.5px; cursor:pointer;">
          <input type="checkbox" ${state.showDepth ? "checked" : ""} data-action="toggle-depth" data-container-id="${esc(containerId)}" />
          Show Filtered Back-Row Depth Duplicates (Dashed Red)
        </label>
      </div>

      <div class="split-canvas-inspector">
        <div>
          <div class="canvas-wrapper">
            <canvas id="canvas-${esc(containerId)}" class="shelf-canvas" width="980" height="620"></canvas>
          </div>
          <div class="canvas-legend">
            <span><i class="legend-dot" style="background:#10b981;"></i>HUL Brand Facing</span>
            <span><i class="legend-dot" style="background:#f59e0b;"></i>Non-HUL Brand Facing</span>
            <span><i class="legend-dot" style="background:#ef4444;"></i>Back-Row Depth Duplicate (Filtered Out)</span>
            <span class="muted">(Click any bounding box on the shelf image to inspect that facing)</span>
          </div>
        </div>

        <div class="inspector-box">
          ${buildInspectorHtml(selectedRow, approachId)}
        </div>
      </div>
    </div>

    <!-- PATH-SPECIFIC INTERMEDIATE STEPS (STEPS 2 & 3) -->
    ${intermediateStageHtml}

    <!-- FINAL STEP: HYBRID / VECTOR SEARCH + COST, LATENCY, TOKENS & OPENTELEMETRY AUDIT -->
    <div class="linear-step-card">
      <div class="linear-step-header">
        <h3><span class="step-badge-pill">${step3Badge}</span> End-to-End Benchmark Metrics, Vector Catalog Retrieval &amp; OpenTelemetry Audit</h3>
        <span class="mono muted">Run ID: ${esc(summaryRec.run_id || "N/A")} &bull; Trace ID: ${esc(summaryRec.trace_id || "N/A")}</span>
      </div>

      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Front Facings Detected</div>
          <div class="kpi-value">${esc(summaryRec.front_facings_count ?? rows.length)}</div>
          <div class="kpi-sub">Depth Duplicates Filtered: ${fmtCount(summaryRec.depth_duplicates_filtered)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">End-to-End Latency</div>
          <div class="kpi-value">${fmtMs(summaryRec.latency_ms, 1, "--")}</div>
          <div class="kpi-sub">${reportedNumber(summaryRec.latency_per_facing_ms) === null ? "" : fmtMs(summaryRec.latency_per_facing_ms) + " / facing"}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Cost / Shelf Image</div>
          <div class="kpi-value">${fmtUsd(summaryRec.cost_per_shelf_image_usd, 6, "--")}</div>
          <div class="kpi-sub">Cost / Facing: ${fmtUsd(summaryRec.cost_per_product_usd, 6, "--")}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Token Breakdown</div>
          <div class="kpi-value">${fmtCount(summaryRec.total_tokens)} tok</div>
          <div class="kpi-sub">In: ${fmtCount(summaryRec.input_tokens)} &bull; Think: ${fmtCount(summaryRec.thinking_tokens)} &bull; Out: ${fmtCount(summaryRec.output_tokens)}</div>
        </div>
      </div>

      ${buildTraceWaterfallAndCostHtml(summaryRec, approachId, rows)}
      ${buildOnboardingTraceAndObservabilityHtml(summaryRec, approachId)}
    </div>
  `;

  drawPipelineCanvas(containerId, rows, depthDemo, state.selectedIdx, state.showDepth, approachId);
}

function buildOnboardingTraceAndObservabilityHtml(summaryRec, approachId) {
  const trace = summaryRec.execution_trace || {};
  const modelsInvoked = trace.models_invoked || [
    { stage: "Primary Pipeline", model: summaryRec.model_name || "gemini-3.8-flash", type: "Vertex AI Model" },
  ];
  const dataUsed = trace.data_used || {};
  const acc = trace.accuracy_summary || summaryRec;
  const otel = trace.opentelemetry || {};
  const gtAvailable = Boolean(summaryRec.ground_truth_available ?? acc.ground_truth_available);
  const traceId = otel.trace_id || summaryRec.trace_id || "N/A";
  const spanId = otel.span_id || summaryRec.span_id || "N/A";
  const otelPath = otel.otel_log_path || "reports/otel_logs.jsonl";
  const jqCmd = otel.local_jq_command || `jq 'select(.TraceId == "${traceId}")' ${otelPath}`;
  const gcpQuery = otel.gcp_cloud_logging_query || `logName="projects/unilever-shelf-understanding/logs/unilever-shelf-benchmark-otel" AND trace="projects/unilever-shelf-understanding/traces/${traceId}"`;
  // The IoU threshold a run was actually scored at. Falling back to 0.5 told the
  // reader a threshold had been applied when none was recorded.
  const iouThreshold = reportedNumber(summaryRec.iou_threshold ?? acc.iou_threshold);

  const fmtMetric = (v) => (v === null || v === undefined ? `<span class="muted">None (awaiting GT)</span>` : Number(v).toFixed(4));

  return `
    <div style="margin-top:20px; border-top:1px solid rgba(148,163,184,0.22); padding-top:18px;">
      <h4 style="margin:0 0 10px 0;">Engineering &amp; Onboarding Traceability Inspector (Calls, Models, Accuracy, Data &amp; OpenTelemetry)</h4>
      <div class="split-2col">
        <div class="inspector-box" style="padding:14px;">
          <h5 style="margin:0 0 8px 0;">1. Implementation &amp; Call Topology</h5>
          <div class="inspector-grid">
            <div class="inspector-item">
              <span>Call Pattern</span>
              <strong>${esc(trace.call_topology || approachId)}</strong>
            </div>
            <div class="inspector-item">
              <span>API Calls / Image</span>
              <strong>${trace.api_calls_count == null
                ? NOT_REPORTED_HTML
                : `${esc(trace.api_calls_count)} call(s)`}</strong>
            </div>
            <div class="inspector-item" style="grid-column: span 2;">
              <span>How Detection &amp; Classification Execute</span>
              <strong>${esc(trace.detect_and_classify_mode || "See pipeline stages above")}</strong>
            </div>
            <div class="inspector-item" style="grid-column: span 2;">
              <span>Models &amp; Services Invoked per Stage</span>
              <div>
                ${modelsInvoked.map((m) => `<div style="margin-top:3px;"><code>${esc(m.stage)}</code> &rarr; <strong><code>${esc(m.model)}</code></strong> <span class="muted">(${esc(m.type)})</span></div>`).join("")}
              </div>
            </div>
            <div class="inspector-item">
              <span>Shelf Image Input</span>
              <strong><code>${esc(String(dataUsed.shelf_image_uri || summaryRec.shelf_image_uri || "shelf-image.png").split("/").slice(-2).join("/"))}</code></strong>
            </div>
            <div class="inspector-item">
              <span>Taxonomy Config</span>
              <strong><code>${esc(dataUsed.taxonomy_source || "(unreported)")}</code></strong>
            </div>
          </div>
        </div>

        <div class="inspector-box" style="padding:14px;">
          <h5 style="margin:0 0 8px 0;">2. Ground Truth Accuracy &amp; OpenTelemetry Lookup</h5>
          <div class="inspector-grid">
            <div class="inspector-item">
              <span>Accuracy Status</span>
              <strong><code>${esc(summaryRec.accuracy_status || acc.accuracy_status || "PLACEHOLDER_AWAITING_GROUND_TRUTH")}</code></strong>
            </div>
            <div class="inspector-item">
              <span>GT Version &amp; IoU Threshold</span>
              <strong><code>${esc(summaryRec.gt_version || acc.gt_version || "unversioned")}</code> (IoU &ge; ${iouThreshold === null ? NOT_REPORTED_HTML : esc(iouThreshold)})</strong>
            </div>
            <div class="inspector-item">
              <span>Detection Precision / Recall / F1</span>
              <strong>${fmtMetric(summaryRec.detection_precision)} / ${fmtMetric(summaryRec.detection_recall)} / ${fmtMetric(summaryRec.detection_f1)}</strong>
            </div>
            <div class="inspector-item">
              <span>Brand / Product / Count Accuracy</span>
              <strong>${fmtMetric(summaryRec.brand_classification_accuracy)} / ${fmtMetric(summaryRec.product_classification_accuracy)} / ${fmtMetric(summaryRec.count_accuracy)}</strong>
            </div>
            <div class="inspector-item" style="grid-column: span 2;">
              <span>Why is Accuracy <code>${gtAvailable ? "Evaluated" : "None (not 0.0)"}</code>?</span>
              <div class="muted" style="font-size:12px;">
                ${gtAvailable
                  ? `Scored against connected ground truth (${esc(summaryRec.gt_version || acc.gt_version || "unversioned")}). Geometry paired first at IoU &ge; ${iouThreshold === null ? NOT_REPORTED_HTML : esc(iouThreshold)}; brand &amp; product scored on matched pairs.`
                  : `Ground truth is not connected yet, so metrics report <code>None</code> rather than <code>0.0</code>. In Run Live Studio above, select <strong>Connect Sample GT</strong> or run <code>shelf-benchmark score</code> when annotations arrive.`}
              </div>
            </div>
            <div class="inspector-item" style="grid-column: span 2;">
              <span>OpenTelemetry Trace Lookup (Local JSONL &amp; GCP Cloud Logging)</span>
              <pre class="code-box" style="margin:4px 0 0 0; font-size:11px; padding:8px;"># 1. Inspect span locally in ${esc(otelPath)}:
${esc(jqCmd)}

# 2. Query in GCP Cloud Logging (Cloud Run / Vertex AI):
${esc(gcpQuery)}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

// Returns a finite Number, or null when the backend did not report the value.
// Deliberately NOT a `|| fallback`: a fabricated dollar figure is indistinguishable
// from a metered one once it is rendered, which is how this panel used to display
// invented costs for runs that never incurred them.
function reportedNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function usd(value, digits = 6) {
  return value === null ? `<span class="muted">not reported</span>` : `$${value.toFixed(digits)}`;
}

function usdPer(value, divisor, digits = 6) {
  if (value === null || !divisor) return `<span class="muted">not reported</span>`;
  return `$${(value / divisor).toFixed(digits)}`;
}

function sumReported(values) {
  const present = values.filter((v) => v !== null);
  return present.length ? present.reduce((a, b) => a + b, 0) : null;
}

// Returns the OpenTelemetry records actually present for this trace id.
// Nothing is synthesised: if the JSONL has no span, the caller renders an
// explicit "unavailable" state.
function realSpansForTrace(traceId) {
  if (!traceId || traceId === "N/A") return [];
  return ((DASHBOARD_DATA && DASHBOARD_DATA.otel_spans) || []).filter(
    (s) => s && s.TraceId === traceId
  );
}

// Renders one row per real span. Bar offset/width come from the recorded
// start/end nanosecond timestamps; when those are missing the bar is omitted
// and said to be omitted, rather than being positioned by guesswork.
function buildRealSpanRowsHtml(spans) {
  const starts = [];
  const ends = [];
  spans.forEach((s) => {
    const attr = s.Attributes || {};
    const st = reportedNumber(attr.start_time_unix_nano);
    const en = reportedNumber(attr.end_time_unix_nano);
    if (st !== null) starts.push(st);
    if (en !== null) ends.push(en);
  });
  const t0 = starts.length ? Math.min(...starts) : null;
  const t1 = ends.length ? Math.max(...ends) : null;
  const windowNs = t0 !== null && t1 !== null && t1 > t0 ? t1 - t0 : null;

  return spans.map((s) => {
    const attr = s.Attributes || {};
    const op = attr["gen_ai.operation.name"] || attr["shelf_benchmark.task_type"] || "span";
    const model = attr["gen_ai.request.model"] || "";
    const durMs = reportedNumber(attr.latency_ms);
    const st = reportedNumber(attr.start_time_unix_nano);
    const en = reportedNumber(attr.end_time_unix_nano);
    const hasBar = windowNs !== null && st !== null && en !== null;
    const offsetPct = hasBar ? ((st - t0) / windowNs) * 100 : 0;
    const widthPct = hasBar ? Math.max(1, ((en - st) / windowNs) * 100) : 0;
    const tokens = reportedNumber(attr["gen_ai.usage.total_tokens"]);
    return `
      <div style="display:grid; grid-template-columns: 320px 1fr 95px; gap:12px; align-items:center; padding:7px 10px; border-bottom:1px solid #e2e8f0; font-size:12px;">
        <div>
          <div class="mono" style="font-weight:700; color:#0f172a;">gen_ai.${esc(op)}</div>
          <div class="muted" style="font-size:11px;">span_id: <code>${esc(s.SpanId) || "unreported"}</code></div>
        </div>
        <div>
          <div style="background:#f1f5f9; height:18px; border-radius:4px; position:relative; overflow:hidden;">
            ${hasBar ? `<div style="position:absolute; left:${offsetPct}%; width:${widthPct}%; height:100%; background:#0284c7; border-radius:4px;"></div>` : ""}
          </div>
          <div class="muted" style="font-size:11px; margin-top:2px;">
            ${esc(model) || "model unreported"}${tokens === null ? "" : ` &bull; ${tokens} tokens`}${hasBar ? "" : " &bull; <em>no start/end timestamps recorded; bar omitted</em>"}
          </div>
        </div>
        <div class="mono" style="text-align:right; font-weight:700; color:#0f172a;">${durMs === null ? NOT_REPORTED_HTML : durMs.toFixed(1) + " ms"}</div>
      </div>
    `;
  }).join("");
}

function buildTraceWaterfallAndCostHtml(summaryRec, approachId, rows) {
  const reportedLat = reportedNumber(summaryRec.latency_ms);
  const traceId = summaryRec.trace_id || "N/A";
  const runId = summaryRec.run_id || "N/A";
  const facings = Number(summaryRec.front_facings_count || rows.length || 0);

  const paygUsd = reportedNumber(summaryRec.vertex_ai_payg_tokens_usd);
  // Provisioned throughput only accrues when traffic actually ran as PROVISIONED_THROUGHPUT.
  // On-demand runs must show nothing here, not a prorated slot rental.
  const ptGsuUsd = reportedNumber(summaryRec.vertex_ai_provisioned_throughput_usd);
  const embedUsd = reportedNumber(summaryRec.vertex_ai_embeddings_and_vision_usd);
  const cloudRunUsd = reportedNumber(summaryRec.cloud_run_compute_usd);
  const gcsObsUsd = reportedNumber(summaryRec.gcs_and_observability_usd);
  const allInPaygUsd = reportedNumber(summaryRec.cost_per_shelf_image_usd);
  // Null unless the run actually served on reserved GSUs. Summing the other buckets
  // when ptGsuUsd is null would present an on-demand run as having a GSU-mode cost.
  const allInPtGsuUsd = ptGsuUsd === null
    ? null
    : (reportedNumber(summaryRec.all_in_pt_gsu_total_usd) ??
       sumReported([ptGsuUsd, embedUsd, cloudRunUsd, gcsObsUsd]));

  const liveRates = summaryRec.rates_from_live_catalog === true;
  const billingSource = summaryRec.billing_source || (liveRates ? "gcp_billing_catalog_api" : "yaml_rate_table");
  const infraIncluded = summaryRec.includes_modelled_infrastructure === true;
  const rateBadge = liveRates
    ? `<span class="tag-hul" style="font-size:11px;">Live Cloud Billing SKU rates</span>`
    : `<span class="tag-non-hul" style="font-size:11px;">Configured rate table (${billingSource})</span>`;
  const infraNote = infraIncluded
    ? "Modelled infrastructure (rows 4 and 5) IS included in the all-in totals."
    : "Modelled infrastructure (rows 4 and 5) is shown for reference only and is EXCLUDED from the all-in totals. Enable billing.include_infrastructure_costs to fold it in.";

  // ---------------------------------------------------------------------------
  // OpenTelemetry waterfall.
  //
  // The benchmark emits ONE OTel record per task execution (see
  // reports/otel_logs.jsonl); it does not emit per-stage child spans. This panel
  // previously manufactured four child spans by multiplying the total latency by
  // hardcoded fractions (0.34 / 0.22 / 0.31 / remainder) and derived their
  // span_ids by slicing substrings out of the trace id. None of those spans ever
  // existed, and they were rendered identically to real telemetry. Only spans
  // actually present in the JSONL are shown now; when there are none, the
  // breakdown is reported as unavailable rather than estimated.
  // ---------------------------------------------------------------------------
  const realSpans = realSpansForTrace(summaryRec.trace_id);
  const waterfallRowsHtml = realSpans.length
    ? buildRealSpanRowsHtml(realSpans)
    : `<div class="muted" style="padding:12px; font-size:12px;">
         Per-stage span breakdown <strong>unavailable</strong> &mdash; no OpenTelemetry spans matching
         <code>trace_id=${esc(traceId)}</code> were found in <code>reports/otel_logs.jsonl</code>.
         The benchmark records one span per task execution and does not emit per-stage child
         spans, so no stage timings are shown here. They are not estimated.
       </div>`;


  return `
    <div style="margin-top:18px; display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
      <!-- 100% NON-ZERO 5-BUCKET SEPARATED GCP COST TABLE -->
      <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <h4 style="margin:0; font-size:14px;">Separated GCP Cost Breakdown</h4>
          ${rateBadge}
        </div>
        <table class="data-table" style="font-size:12px; margin:0;">
          <thead>
            <tr>
              <th>GCP Billing Component</th>
              <th>GCP Service / SKU</th>
              <th style="text-align:right;">Cost / Image ($)</th>
              <th style="text-align:right;">Cost / Facing ($)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>1. Vertex AI PAYG Tokens</strong> <span class="muted">(metered)</span></td>
              <td><code>gemini (${fmtCount(summaryRec.input_tokens)} in / ${fmtCount(summaryRec.thinking_tokens)} think / ${fmtCount(summaryRec.output_tokens)} out)</code></td>
              <td class="mono" style="text-align:right; font-weight:700; color:#0284c7;">${usd(paygUsd)}</td>
              <td class="mono" style="text-align:right;">${usdPer(paygUsd, facings)}</td>
            </tr>
            <tr>
              <td><strong>2. Vertex AI Provisioned GSU</strong> <span class="muted">(metered)</span></td>
              <td><code>${ptGsuUsd === null ? "Traffic did not run as PROVISIONED_THROUGHPUT" : `Reserved GSU slot occupancy (${reportedLat === null ? "duration not reported" : (reportedLat / 1000).toFixed(1) + "s"})`}</code></td>
              <td class="mono" style="text-align:right; font-weight:700; color:#7c3aed;">${usd(ptGsuUsd)}</td>
              <td class="mono" style="text-align:right;">${usdPer(ptGsuUsd, facings)}</td>
            </tr>
            <tr>
              <td><strong>3. Embeddings &amp; Vision API</strong> <span class="muted">(modelled per facing)</span></td>
              <td><code>multimodalembedding@001 (1408-D) + gemini-embedding-001 (3072-D)</code></td>
              <td class="mono" style="text-align:right; font-weight:700; color:#059669;">${usd(embedUsd)}</td>
              <td class="mono" style="text-align:right;">${usdPer(embedUsd, facings)}</td>
            </tr>
            <tr>
              <td><strong>4. Cloud Run Worker Compute</strong> <span class="muted">(modelled)</span></td>
              <td><code>vCPU-seconds + GiB-seconds derived from measured latency</code></td>
              <td class="mono" style="text-align:right; font-weight:700; color:#d97706;">${usd(cloudRunUsd)}</td>
              <td class="mono" style="text-align:right;">${usdPer(cloudRunUsd, facings)}</td>
            </tr>
            <tr>
              <td><strong>5. GCS + Cloud Logging / Trace</strong> <span class="muted">(modelled)</span></td>
              <td><code>GCS Class A/B ops + Cloud Logging OTel sink</code></td>
              <td class="mono" style="text-align:right; font-weight:700; color:#475569;">${usd(gcsObsUsd)}</td>
              <td class="mono" style="text-align:right;">${usdPer(gcsObsUsd, facings)}</td>
            </tr>
            <tr style="background:#e0f2fe; font-weight:700;">
              <td colspan="2"><strong>ALL-IN TOTAL (On-Demand PAYG mode)</strong></td>
              <td class="mono" style="text-align:right; color:#0369a1;">${usd(allInPaygUsd)}</td>
              <td class="mono" style="text-align:right; color:#0369a1;">${usdPer(allInPaygUsd, facings)}</td>
            </tr>
            <tr style="background:#f3e8ff; font-weight:700;">
              <td colspan="2"><strong>ALL-IN TOTAL (Provisioned GSU mode)</strong></td>
              <td class="mono" style="text-align:right; color:#6d28d9;">${usd(allInPtGsuUsd)}</td>
              <td class="mono" style="text-align:right; color:#6d28d9;">${usdPer(allInPtGsuUsd, facings)}</td>
            </tr>
          </tbody>
        </table>
        <div class="muted" style="font-size:11px; margin-top:8px;">
          <strong>billing_source:</strong> <code>${esc(billingSource)}</code> &bull; ${infraNote}
          Blank cells mean the backend did not report that component for this run. They are not zeros.
        </div>
      </div>

      <!-- DISTRIBUTED OPENTELEMETRY TRACE WATERFALL -->
      <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <h4 style="margin:0; font-size:14px;">OpenTelemetry Distributed Trace Waterfall (Cloud Run Worker &rarr; Vertex AI)</h4>
          <span class="mono" style="font-size:11px; background:#0f172a; color:#38bdf8; padding:3px 7px; border-radius:4px;">trace_id: ${esc(traceId)}</span>
        </div>
        <div style="background:#fff; border:1px solid #e2e8f0; border-radius:6px;">
          ${waterfallRowsHtml}
        </div>
        <div class="muted" style="font-size:11px; margin-top:8px;">
          <strong>GCP Sinks:</strong> Cloud Logging <code>projects/unilever-shelf-understanding/logs/unilever-shelf-benchmark-otel</code> &bull;
          Cloud Trace <code>projects/unilever-shelf-understanding/traces/${esc(traceId)}</code> &bull;
          GCS <code>gs://unilever-shelf-understanding-shelf-images/otel/otel_logs.jsonl</code>
        </div>
      </div>
    </div>
  `;
}

function buildInspectorHtml(row, approachId) {
  if (!row) {
    return `<p class="muted">No product facings found for this selection.</p>`;
  }
  const isAgnostic = approachId === "class_agnostic_visual_embedding";
  const badgeHtml = row.is_hul_brand
    ? `<span class="tag-hul">HUL Portfolio Brand</span>`
    : `<span class="tag-non-hul">Non-HUL Brand</span>`;

  return `
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <h4 style="margin:0;">Facing #${esc(row.product_index)} Inspector</h4>
      ${badgeHtml}
    </div>
    <p class="muted" style="margin-top:4px;">
      BBox <code>[${esc(row.bbox_ymin)}, ${esc(row.bbox_xmin)}, ${esc(row.bbox_ymax)}, ${esc(row.bbox_xmax)}]</code> &bull; Row: <code>${esc(row.shelf_row)}</code>
    </p>
    <div class="inspector-grid">
      <div class="inspector-item"><span>1. Category</span><strong>${esc(row.predicted_category) || "--"}</strong></div>
      <div class="inspector-item"><span>2. Subcategory</span><strong>${esc(row.predicted_subcategory) || "--"}</strong></div>
      <div class="inspector-item"><span>3. Brand</span><strong>${esc(row.predicted_brand) || "--"}</strong></div>
      <div class="inspector-item"><span>4. Variant</span><strong>${esc(row.predicted_variant) || "--"}</strong></div>
      <div class="inspector-item"><span>5. Packaging Type</span><strong>${esc(row.predicted_packaging) || "--"}</strong></div>
      <div class="inspector-item"><span>6. Pack Type</span><strong>${esc(row.predicted_pack_type) || "--"}</strong></div>
      <div class="inspector-item" style="grid-column: span 2;">
        <span>7. Rule-Derived Size Bucket</span>
        <strong>${esc(row.rule_derived_size_bucket || row.predicted_size) || "--"} (OCR hint: ${esc(row.predicted_size) || "none"})</strong>
      </div>
    </div>
    <div style="margin-top:10px; padding:9px; background:#fff; border:1px solid #e2e8f0; border-radius:6px; font-size:12px;">
      <div><strong>${isAgnostic ? "Stage 3 ScaNN Catalog SKU Match:" : "Hybrid Search Keywords &amp; Vector Passage:"}</strong></div>
      <div class="mono muted" style="margin-top:3px;">
        ${isAgnostic
          ? `Matched Catalog SKU: <strong>${esc(row.matched_sku_id) || "N/A"}</strong> &bull; Cosine ANN Similarity: <strong>${fmtPercent(row.confidence, 2)}</strong>`
          : `Keywords: ${esc(row.lexical_search_keywords || row.predicted_product_name)}<br/>Vector Dim: ${row.embedding_dimensions ? esc(row.embedding_dimensions) + "-D" : NOT_REPORTED_HTML}`}
      </div>
    </div>
  `;
}

function buildSevenDimensionTableHtml(rows, containerId, selectedIdx, isScannMode) {
  return `
    <div class="table-scroll">
      <table class="data-table">
        <thead>
          <tr>
            <th>Facing #</th>
            <th>BBox <code>[ymin,xmin,ymax,xmax]</code></th>
            <th>1. Category</th>
            <th>2. Subcategory</th>
            <th>3. Brand</th>
            <th>4. Variant</th>
            <th>5. Packaging</th>
            <th>6. Pack Type</th>
            <th>7. Rule-Derived Size</th>
            <th>${isScannMode ? "ScaNN Catalog SKU (Cosine Sim)" : "Hybrid Vector Dim &amp; Cost"}</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((r, idx) => `
            <tr data-action="select-facing" data-container-id="${esc(containerId)}" data-facing-idx="${idx}" style="cursor:pointer; background:${idx === selectedIdx ? '#eff6ff' : 'transparent'};">
              <td><strong>#${esc(r.product_index)}</strong></td>
              <td><code>[${esc(r.bbox_ymin)},${esc(r.bbox_xmin)},${esc(r.bbox_ymax)},${esc(r.bbox_xmax)}]</code></td>
              <td>${esc(r.predicted_category)}</td>
              <td>${esc(r.predicted_subcategory)}</td>
              <td>
                <strong>${esc(r.predicted_brand)}</strong>
                ${r.is_hul_brand ? `<span class="tag-hul">HUL</span>` : `<span class="tag-non-hul">Non-HUL</span>`}
              </td>
              <td>${esc(r.predicted_variant)}</td>
              <td><code>${esc(r.predicted_packaging)}</code></td>
              <td>${esc(r.predicted_pack_type)}</td>
              <td><strong>${esc(r.rule_derived_size_bucket)}</strong></td>
              <td>
                ${isScannMode
                  ? `<code>${r.matched_sku_id ? esc(r.matched_sku_id) : NOT_REPORTED_HTML}</code> (<strong>${fmtPercent(r.confidence)}</strong>)`
                  : `<code>${r.embedding_dimensions ? esc(r.embedding_dimensions) + "-D" : NOT_REPORTED_HTML}</code> &bull; ${fmtUsd(r.cost_per_product_usd, 5)}`}
              </td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

window.changePipelineModel = function (containerId, newModel) {
  CONTAINER_STATE[containerId].model = newModel;
  CONTAINER_STATE[containerId].selectedIdx = 0;
  renderUseCasePipeline(containerId);
};

window.selectPipelineFacing = function (containerId, idx) {
  CONTAINER_STATE[containerId].selectedIdx = idx;
  renderUseCasePipeline(containerId);
};

window.togglePipelineDepth = function (containerId, checked) {
  CONTAINER_STATE[containerId].showDepth = checked;
  renderUseCasePipeline(containerId);
};

function drawPipelineCanvas(containerId, rows, depthDemo, selectedIdx, showDepth, approachId) {
  const canvas = document.getElementById(`canvas-${containerId}`);
  if (!canvas || !SHELF_IMAGE_OBJ) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(SHELF_IMAGE_OBJ, 0, 0, W, H);

  // Draw filtered back-row depth duplicates in dashed red if enabled
  if (showDepth && depthDemo && Array.isArray(depthDemo.raw_candidates)) {
    const keptSet = new Set(
      (depthDemo.kept_front_facings || []).map((k) => (k.bbox_2d || []).join(","))
    );
    depthDemo.raw_candidates.forEach((cand) => {
      const key = (cand.bbox_2d || []).join(",");
      if (!keptSet.has(key) && Array.isArray(cand.bbox_2d) && cand.bbox_2d.length === 4) {
        const [ymin, xmin, ymax, xmax] = cand.bbox_2d;
        const x = (xmin / 1000) * W;
        const y = (ymin / 1000) * H;
        const w = ((xmax - xmin) / 1000) * W;
        const h = ((ymax - ymin) / 1000) * H;
        ctx.save();
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = "rgba(239, 68, 68, 0.85)";
        ctx.fillRect(x, Math.max(0, y - 18), 118, 18);
        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px sans-serif";
        ctx.fillText("DEPTH DUPLICATE", x + 4, Math.max(12, y - 5));
        ctx.restore();
      }
    });
  }

  // Draw surviving front-facing bounding boxes
  const hitRegions = [];
  rows.forEach((r, idx) => {
    const x = (r.bbox_xmin / 1000) * W;
    const y = (r.bbox_ymin / 1000) * H;
    const w = ((r.bbox_xmax - r.bbox_xmin) / 1000) * W;
    const h = ((r.bbox_ymax - r.bbox_ymin) / 1000) * H;
    hitRegions.push({ idx, x, y, w, h });

    const isSelected = idx === selectedIdx;
    const color = r.is_hul_brand ? "#10b981" : "#f59e0b";

    ctx.save();
    ctx.strokeStyle = isSelected ? "#38bdf8" : color;
    ctx.lineWidth = isSelected ? 4 : 2.5;
    ctx.strokeRect(x, y, w, h);
    if (isSelected) {
      ctx.fillStyle = "rgba(56, 189, 248, 0.18)";
      ctx.fillRect(x, y, w, h);
    }

    const label = approachId === "class_agnostic_visual_embedding"
      ? `#${r.product_index} product -> ${r.predicted_brand}`
      : `#${r.product_index} ${r.predicted_brand}`;
    ctx.font = "bold 11px sans-serif";
    const textW = Math.min(w + 20, ctx.measureText(label).width + 10);
    ctx.fillStyle = isSelected ? "#0284c7" : color;
    ctx.fillRect(x, Math.max(0, y - 19), textW, 19);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label.slice(0, 18), x + 4, Math.max(13, y - 5));
    ctx.restore();
  });

  canvas.onclick = (evt) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clickX = (evt.clientX - rect.left) * scaleX;
    const clickY = (evt.clientY - rect.top) * scaleY;
    const hit = hitRegions.find(
      (b) => clickX >= b.x && clickX <= b.x + b.w && clickY >= b.y && clickY <= b.y + b.h
    );
    if (hit) {
      selectPipelineFacing(containerId, hit.idx);
    }
  };
}

// ============================================================================
// TAB 6: RUN LIVE STUDIO (REAL-TIME VERTEX AI EXECUTION + FULL PIPELINE UI)
// ============================================================================
function initLiveStudioTab() {
  const approachSelect = document.getElementById("live-studio-approach");
  const modelSelect = document.getElementById("live-studio-model");
  const runBtn = document.getElementById("btn-execute-live-studio");

  const updateLivePreviewSelection = () => {
    const appId = approachSelect.value;
    const modName = modelSelect.value;
    CONTAINER_STATE["pipeline-container-live"].approachId = appId;
    CONTAINER_STATE["pipeline-container-live"].model = modName;
    CONTAINER_STATE["pipeline-container-live"].selectedIdx = 0;
    renderLiveProgressStepper(appId, -1, true);
    renderUseCasePipeline("pipeline-container-live");
  };

  approachSelect.addEventListener("change", updateLivePreviewSelection);
  modelSelect.addEventListener("change", updateLivePreviewSelection);

  runBtn.addEventListener("click", async () => {
    const appId = approachSelect.value;
    const modName = modelSelect.value;
    const modeSelect = document.getElementById("live-studio-mode");
    const gtSelect = document.getElementById("live-studio-gt");
    const brandModeSelect = document.getElementById("live-studio-brand-mode");
    const attrModeSelect = document.getElementById("live-studio-attr-mode");
    const accelSelect = document.getElementById("live-studio-accelerator");
    const execMode = modeSelect ? modeSelect.value : "offline";
    const connectGt = gtSelect ? gtSelect.value === "sample" : false;
    const brandMode = brandModeSelect ? brandModeSelect.value : "open_vocabulary_generative";
    const attrMode = attrModeSelect ? attrModeSelect.value : "single_call";
    const accelType = accelSelect ? accelSelect.value : "none";
    const statusPill = document.getElementById("live-status-pill");
    const timerDisplay = document.getElementById("live-timer-display");

    runBtn.disabled = true;
    runBtn.textContent = execMode === "offline" ? "Running Offline Local Test..." : "Executing Live on Vertex AI...";
    statusPill.className = "status-pill status-running";
    statusPill.textContent = execMode === "offline"
      ? `RUNNING OFFLINE LOCAL FIXTURE (${modName} • ${appId} • ${accelType.toUpperCase()})...`
      : `RUNNING LIVE ON VERTEX AI (${modName} • ${appId} • ${accelType.toUpperCase()})...`;

    let activeStageIdx = 0;
    const tStart = performance.now();
    renderLiveProgressStepper(appId, activeStageIdx, false);

    const timerInterval = setInterval(() => {
      const elapsedSec = ((performance.now() - tStart) / 1000).toFixed(1);
      timerDisplay.textContent = `Elapsed: ${elapsedSec}s`;
      const pathSteps = (USE_CASE_PATHS[appId] && USE_CASE_PATHS[appId].steps.length) || 3;
      activeStageIdx = Math.min(pathSteps - 1, Math.floor(Number(elapsedSec) / 2.8));
      renderLiveProgressStepper(appId, activeStageIdx, false);
    }, 300);

    try {
      const response = await fetch("/api/run-live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_type: "classification",
          model_name: modName,
          separation_approach: appId,
          mode: execMode,
          connect_sample_gt: connectGt,
          brand_mode: brandMode,
          attribute_call_mode: attrMode,
          accelerator: accelType,
        }),
      });
      const liveResult = await response.json();
      clearInterval(timerInterval);

      if (liveResult.error) {
        statusPill.className = "status-pill status-idle";
        statusPill.textContent = `ERROR: ${liveResult.error}`;
      } else {
        const modeTag = execMode === "offline" ? "OFFLINE LOCAL RUN COMPLETE" : "LIVE VERTEX AI RUN COMPLETE";
        // A ground-truth score that the backend did not return is NOT a perfect
        // score. Defaulting to 1.0 here reported flawless detection for runs
        // that were never scored at all.
        const f1 = reportedNumber(liveResult.accuracy?.detection_f1);
        const gtTag = connectGt
          ? ` • GT F1=${f1 === null ? NOT_MEASURED_TEXT : f1.toFixed(2)}`
          : " • GT=Placeholder (None)";
        const latencyMs = reportedNumber(liveResult.latency_ms);
        const facingCount = reportedNumber(liveResult.front_facings_count);
        const latencyTxt = latencyMs === null
          ? "latency not reported"
          : `${(latencyMs / 1000).toFixed(2)}s`;
        const facingTxt = facingCount === null
          ? "facing count not reported"
          : `${facingCount} Front Facings`;
        statusPill.className = "status-pill status-done";
        statusPill.textContent = `${modeTag} (${latencyTxt} • ${facingTxt}${gtTag})`;

        renderLiveProgressStepper(appId, (USE_CASE_PATHS[appId] && USE_CASE_PATHS[appId].steps.length) || 3, true);

        CONTAINER_STATE["pipeline-container-live"].approachId = appId;
        CONTAINER_STATE["pipeline-container-live"].model = modName;
        CONTAINER_STATE["pipeline-container-live"].selectedIdx = 0;
        CONTAINER_STATE["pipeline-container-live"].livePayload = liveResult;
        renderUseCasePipeline("pipeline-container-live");
      }
    } catch (err) {
      clearInterval(timerInterval);
      statusPill.className = "status-pill status-idle";
      statusPill.textContent = `Execution Error: ${err.message}`;
    } finally {
      runBtn.disabled = false;
      runBtn.innerHTML = "&#9654; Execute Selected Approach &amp; Trace";
    }
  });

  updateLivePreviewSelection();
}

function renderLiveProgressStepper(approachId, activeIdx, allDone) {
  const stepper = document.getElementById("live-stage-stepper");
  const pathMeta = USE_CASE_PATHS[approachId];
  if (!stepper || !pathMeta) return;

  stepper.innerHTML = pathMeta.steps.map((s, idx) => {
    let cls = "flow-step-node";
    let statusBadge = "Queued";
    if (allDone || idx < activeIdx) {
      cls += " done-stage";
      statusBadge = "✓ Completed";
    } else if (idx === activeIdx) {
      cls += " active-stage";
      statusBadge = "⟳ Running on Vertex AI...";
    }
    return `
      <div class="${cls}">
        <span class="flow-step-num">${esc(s.num)} &bull; ${statusBadge}</span>
        <div class="flow-step-title">${esc(s.title)}</div>
        <div class="flow-step-desc">${esc(s.desc)}</div>
      </div>
    `;
  }).join("");
}

// ============================================================================
// TAB 7: FINE-TUNING (SFT) & CENTRALIZED TAXONOMY CONFIG
// ============================================================================
function renderSftAndConfigTab() {
  if (!DASHBOARD_DATA) return;
  const tax = DASHBOARD_DATA.taxonomy_reference || {};
  const viewer = document.getElementById("taxonomy-config-viewer");
  if (viewer) {
    const escList = (items, sep) => (items || []).map(esc).join(sep);
    viewer.innerHTML = `
      <div class="inspector-grid">
        <div class="inspector-item">
          <span>Config File Source</span>
          <strong><code>${esc(tax.config_source || "(unreported)")}</code></strong>
        </div>
        <div class="inspector-item">
          <span>Pack Types</span>
          <strong>${escList(tax.pack_types, ", ")}</strong>
        </div>
        <div class="inspector-item" style="grid-column: span 2;">
          <span>Categories (Configurable)</span>
          <strong>${escList(tax.categories, " • ")}</strong>
        </div>
        <div class="inspector-item" style="grid-column: span 2;">
          <span>Subcategories (Configurable)</span>
          <strong>${escList(tax.subcategories, " • ")}</strong>
        </div>
        <div class="inspector-item" style="grid-column: span 2;">
          <span>Packaging Types (Configurable)</span>
          <strong>${escList(tax.packaging_types, " • ")}</strong>
        </div>
        <div class="inspector-item" style="grid-column: span 2;">
          <span>Rule-Derived Size Buckets (Configurable)</span>
          <strong>${escList(tax.size_buckets, " • ")}</strong>
        </div>
        <div class="inspector-item" style="grid-column: span 2;">
          <span>Brand Attribution Mode (HUL vs. Non-HUL)</span>
          <strong>${
            (tax.hul_brands || []).length > 0
              ? escList((tax.hul_brands || []).slice(0, 20), ", ")
              : "Open-Vocabulary VLM &amp; Catalog Attribution (No predefined brand definitions required in shelf_benchmark/_resources/taxonomy.yaml)"
          }</strong>
        </div>
      </div>
    `;
  }

  // OTel Spans
  const otelBody = document.querySelector("#otel-audit-table tbody");
  if (otelBody) {
    otelBody.innerHTML = (DASHBOARD_DATA.otel_spans || []).slice(-12).reverse().map((s) => {
      const attr = s.Attributes || {};
      return `
        <tr>
          <td><code>${esc(String(s.TraceId || "").slice(0, 12))}...</code></td>
          <td><code>${esc(attr["shelf_benchmark.task_type"])}</code> (${esc(attr["shelf_benchmark.separation_approach"])})</td>
          <td><code>${esc(attr["gen_ai.request.model"])}</code></td>
          <td>${esc(attr["shelf_benchmark.latency_ms"])}</td>
          <td>${esc(attr["gen_ai.usage.total_tokens"])}</td>
        </tr>
      `;
    }).join("");
  }

  // SFT Summary
  const sftBody = document.querySelector("#sft-summary-table tbody");
  if (sftBody) {
    const sftRuns = (DASHBOARD_DATA.summary || []).filter((r) => r.task_type === "fine_tuning");
    sftBody.innerHTML = sftRuns.map((r) => `
      <tr>
        <td><code>${esc(r.model_name)}</code></td>
        <td><code>gs://unilever-shelf-understanding-shelf-images/sft/sft_training_examples.jsonl</code></td>
        <td><span class="tag-hul">${esc(r.status)}</span></td>
        <td><strong>${fmtCount(r.front_facings_count)}</strong></td>
        <td>${fmtMs(r.latency_ms)}</td>
        <td>${fmtUsd(r.cost_per_shelf_image_usd)}</td>
        <td>${fmtUsd(r.cost_per_product_usd)}</td>
      </tr>
    `).join("");
  }

  const sftPre = document.getElementById("sft-jsonl-preview");
  if (sftPre && DASHBOARD_DATA.sft_examples && DASHBOARD_DATA.sft_examples.length > 0) {
    sftPre.textContent = JSON.stringify(DASHBOARD_DATA.sft_examples[0], null, 2);
  }
}
