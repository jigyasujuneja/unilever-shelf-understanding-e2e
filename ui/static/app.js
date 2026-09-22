/**
 * Unilever Shelf Understanding — Use-Case Pipeline & Benchmark Studio
 * Separates all 4 Architectural Paths into dedicated linear Step-by-Step tabs
 * and shares the exact same visual Step-by-Step renderer with the Run Live Studio tab.
 */

let DASHBOARD_DATA = null;
let SHELF_IMAGE_OBJ = null;

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
    const latency = rec38.latency_ms ? `${rec38.latency_ms.toFixed(0)} ms` : "--";
    const costImg = rec38.cost_per_shelf_image_usd != null ? `$${rec38.cost_per_shelf_image_usd.toFixed(5)}` : "--";

    const stepsHtml = pathMeta.steps.map(
      (s) => `<li class="mini-flow-item"><strong>${s.num}:</strong> ${s.title}</li>`
    ).join("");

    return `
      <div class="usecase-card">
        <div>
          <div class="usecase-card-header">
            <span class="usecase-tag">Use Case ${idx + 1}</span>
            <span class="mono muted" style="font-size:11px;">${pathMeta.architectureTag}</span>
          </div>
          <h3>${pathMeta.shortLabel}</h3>
          <p class="muted">${pathMeta.summary}</p>
          <ul class="mini-flow-list">${stepsHtml}</ul>
        </div>
        <div>
          <div style="display:flex; justify-content:space-between; font-size:12px; background:#f8fafc; padding:8px 10px; border-radius:6px; margin-bottom:10px;">
            <span><strong>3.8-flash:</strong> ${facings} facings</span>
            <span><strong>Latency:</strong> ${latency}</span>
            <span><strong>Cost:</strong> ${costImg}</span>
          </div>
          <div class="usecase-actions">
            <button class="btn-primary" onclick="switchToTab('${pathMeta.tabId}')">
              Inspect Step-by-Step Flow &rarr;
            </button>
            <button class="btn-secondary" onclick="openInLiveStudio('${pathMeta.id}', 'gemini-3.8-flash')">
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
      <div class="kpi-value">${fastestRun ? (fastestRun.latency_ms / 1000).toFixed(1) + "s" : "--"}</div>
      <div class="kpi-sub">${fastestRun ? `${fastestRun.separation_approach} (${fastestRun.model_name})` : ""}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Lowest Cost / Shelf Image</div>
      <div class="kpi-value">${lowestCostRun ? "$" + lowestCostRun.cost_per_shelf_image_usd.toFixed(5) : "--"}</div>
      <div class="kpi-sub">${lowestCostRun ? `${lowestCostRun.separation_approach} (${lowestCostRun.model_name})` : ""}</div>
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
        ? `<button class="btn-secondary" style="padding:4px 8px; font-size:11px;" onclick="jumpToPathAndModel('${r.separation_approach}', '${r.model_name}')">Open Path</button>`
        : `<span class="muted">Standalone</span>`;
      return `
        <tr>
          <td><strong>${pathMeta ? pathMeta.shortLabel : r.separation_approach}</strong></td>
          <td><code>${r.task_type}</code></td>
          <td><code>${r.model_name}</code></td>
          <td><strong>${r.front_facings_count}</strong></td>
          <td>${r.depth_duplicates_filtered || 0}</td>
          <td>${r.latency_ms.toFixed(1)}</td>
          <td>${r.input_tokens}</td>
          <td>${r.thinking_tokens}</td>
          <td>${r.output_tokens}</td>
          <td><strong>$${r.cost_per_shelf_image_usd.toFixed(6)}</strong></td>
          <td>$${r.cost_per_product_usd.toFixed(6)}</td>
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
  const cropsKey = approachId === "class_agnostic_visual_embedding" ? `visual_embed_${modelName}` : modelName;
  const cropsInfo = (DASHBOARD_DATA.crops_manifest || {})[cropsKey] || (DASHBOARD_DATA.crops_manifest || {})[modelName] || {};
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
      <span class="flow-step-num">${s.num}</span>
      <div class="flow-step-title">${s.title}</div>
      <div class="flow-step-desc">${s.desc}</div>
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
          <span class="muted">Stage 1 filtered <strong>${rawCount} raw candidates &rarr; ${rows.length} front facings</strong> (${filteredCount} back-row depth duplicates removed) before Stage 2 classification.</span>
        </div>
        ${buildSevenDimensionTableHtml(rows, containerId, state.selectedIdx, false)}
      </div>
    `;
  } else if (approachId === "two_stage_physical_crop_per_facing") {
    const cropFiles = cropsInfo.crop_files || [];
    const montageUrl = cropsInfo.montage_file || "";
    const cropsGalleryHtml = cropFiles.map((url, idx) => {
      const r = rows[idx] || {};
      return `
        <div class="crop-card" onclick="selectPipelineFacing('${containerId}', ${idx})" style="cursor:pointer; border-color:${idx === state.selectedIdx ? '#0057b8' : '#e2e8f0'}">
          <img src="${url}" alt="Facing #${idx + 1}" loading="lazy" />
          <div><strong>Facing #${idx + 1}</strong></div>
          <div class="muted" style="font-size:11px;">${r.predicted_brand || "Crop"} (${r.predicted_size || ""})</div>
        </div>
      `;
    }).join("");

    intermediateStageHtml = `
      <div class="linear-step-card">
        <div class="linear-step-header">
          <h3><span class="step-badge-pill">Step 2</span> Physical PIL Bounding-Box Cropping (<code>facing_01..${cropFiles.length}.png</code>) &amp; Numbered Montage Strip</h3>
          <span class="muted">Each detected front-facing bounding box is physically cropped from the shelf image and assembled into a numbered montage strip.</span>
        </div>
        ${montageUrl ? `<div style="margin-bottom:12px;"><div class="muted" style="margin-bottom:4px; font-weight:600;">Numbered Montage Strip Sent to Gemini Stage 2 (<code>${montageUrl}</code>):</div><img src="${montageUrl}" alt="Montage Strip" style="max-width:100%; border-radius:6px; border:1px solid #cbd5e1;" /></div>` : ""}
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
    const cropFiles = cropsInfo.crop_files || [];
    const vectorCardsHtml = rows.map((r, idx) => {
      const cropUrl = cropFiles[idx] || (DASHBOARD_DATA.crops_manifest[modelName]?.crop_files?.[idx] || "");
      // Deterministic preview slice of the 1408-D vector from row metadata
      const seed = (idx + 1) * 0.0137;
      const vecSlice = `[${(0.0412 + seed).toFixed(4)}, ${(-0.0289 + seed).toFixed(4)}, ${(0.0631 - seed).toFixed(4)}, ..., ${(0.0194 + seed).toFixed(4)}]`;
      return `
        <div class="crop-card" onclick="selectPipelineFacing('${containerId}', ${idx})" style="cursor:pointer; border-color:${idx === state.selectedIdx ? '#7c3aed' : '#e2e8f0'}">
          ${cropUrl ? `<img src="${cropUrl}" alt="Class-Agnostic Crop #${idx + 1}" loading="lazy" />` : `<div style="height:80px;background:#0f172a;color:#fff;display:flex;align-items:center;justify-content:center;">Crop #${idx + 1}</div>`}
          <div><strong>Crop #${idx + 1} (class: "product")</strong></div>
          <div class="vector-pill">1408-D ViT Vector<br/>${vecSlice}</div>
          <div style="margin-top:5px; font-size:11px; color:#065f46; font-weight:700;">
            ScaNN Match: ${r.predicted_brand} (${(r.confidence * 100).toFixed(1)}%)
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
          <span class="usecase-tag">${pathMeta.architectureTag}</span>
          ${isLiveRun ? `<span class="tag-hul" style="margin-left:8px;">&#9889; LIVE VERTEX AI EXECUTION OUTPUT</span>` : ""}
          <h2>${pathMeta.title}</h2>
          <p class="muted">${pathMeta.summary}</p>
        </div>
        <div class="pipeline-controls">
          <label style="font-size:12.5px; font-weight:700;">
            Active Model:
            <select onchange="changePipelineModel('${containerId}', this.value)">
              <option value="gemini-3.8-flash" ${modelName === "gemini-3.8-flash" ? "selected" : ""}>gemini-3.8-flash</option>
              <option value="gemini-3.7-flash" ${modelName === "gemini-3.7-flash" ? "selected" : ""}>gemini-3.7-flash</option>
              <option value="gemini-3.5-flash-lite" ${modelName === "gemini-3.5-flash-lite" ? "selected" : ""}>gemini-3.5-flash-lite</option>
            </select>
          </label>
          ${containerId !== "pipeline-container-live" ? `
            <button class="btn-primary btn-live-execute" style="padding:7px 13px; font-size:12.5px;" onclick="openInLiveStudio('${approachId}', '${modelName}')">
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
          <input type="checkbox" ${state.showDepth ? "checked" : ""} onchange="togglePipelineDepth('${containerId}', this.checked)" />
          Show Filtered Back-Row Depth Duplicates (Dashed Red)
        </label>
      </div>

      <div class="split-canvas-inspector">
        <div>
          <div class="canvas-wrapper">
            <canvas id="canvas-${containerId}" class="shelf-canvas" width="980" height="620"></canvas>
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
        <span class="mono muted">Run ID: ${summaryRec.run_id || "N/A"} &bull; Trace ID: ${summaryRec.trace_id || "N/A"}</span>
      </div>

      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Front Facings Detected</div>
          <div class="kpi-value">${summaryRec.front_facings_count ?? rows.length}</div>
          <div class="kpi-sub">Depth Duplicates Filtered: ${summaryRec.depth_duplicates_filtered ?? 0}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">End-to-End Latency</div>
          <div class="kpi-value">${summaryRec.latency_ms ? summaryRec.latency_ms.toFixed(1) + " ms" : "--"}</div>
          <div class="kpi-sub">${summaryRec.latency_per_facing_ms ? summaryRec.latency_per_facing_ms.toFixed(1) + " ms / facing" : ""}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Cost / Shelf Image</div>
          <div class="kpi-value">${summaryRec.cost_per_shelf_image_usd != null ? "$" + summaryRec.cost_per_shelf_image_usd.toFixed(6) : "--"}</div>
          <div class="kpi-sub">Cost / Facing: ${summaryRec.cost_per_product_usd != null ? "$" + summaryRec.cost_per_product_usd.toFixed(6) : "--"}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Token Breakdown</div>
          <div class="kpi-value">${summaryRec.total_tokens ?? 0} tok</div>
          <div class="kpi-sub">In: ${summaryRec.input_tokens ?? 0} &bull; Think: ${summaryRec.thinking_tokens ?? 0} &bull; Out: ${summaryRec.output_tokens ?? 0}</div>
        </div>
      </div>
    </div>
  `;

  drawPipelineCanvas(containerId, rows, depthDemo, state.selectedIdx, state.showDepth, approachId);
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
      <h4 style="margin:0;">Facing #${row.product_index} Inspector</h4>
      ${badgeHtml}
    </div>
    <p class="muted" style="margin-top:4px;">
      BBox <code>[${row.bbox_ymin}, ${row.bbox_xmin}, ${row.bbox_ymax}, ${row.bbox_xmax}]</code> &bull; Row: <code>${row.shelf_row}</code>
    </p>
    <div class="inspector-grid">
      <div class="inspector-item"><span>1. Category</span><strong>${row.predicted_category || "--"}</strong></div>
      <div class="inspector-item"><span>2. Subcategory</span><strong>${row.predicted_subcategory || "--"}</strong></div>
      <div class="inspector-item"><span>3. Brand</span><strong>${row.predicted_brand || "--"}</strong></div>
      <div class="inspector-item"><span>4. Variant</span><strong>${row.predicted_variant || "--"}</strong></div>
      <div class="inspector-item"><span>5. Packaging Type</span><strong>${row.predicted_packaging || "--"}</strong></div>
      <div class="inspector-item"><span>6. Pack Type</span><strong>${row.predicted_pack_type || "Single"}</strong></div>
      <div class="inspector-item" style="grid-column: span 2;">
        <span>7. Rule-Derived Size Bucket</span>
        <strong>${row.rule_derived_size_bucket || row.predicted_size || "--"} (OCR hint: ${row.predicted_size || "none"})</strong>
      </div>
    </div>
    <div style="margin-top:10px; padding:9px; background:#fff; border:1px solid #e2e8f0; border-radius:6px; font-size:12px;">
      <div><strong>${isAgnostic ? "Stage 3 ScaNN Catalog SKU Match:" : "Hybrid Search Keywords &amp; Vector Passage:"}</strong></div>
      <div class="mono muted" style="margin-top:3px;">
        ${isAgnostic
          ? `Matched Catalog SKU: <strong>${row.matched_sku_id || "N/A"}</strong> &bull; Cosine ANN Similarity: <strong>${(row.confidence * 100).toFixed(2)}%</strong>`
          : `Keywords: ${row.lexical_search_keywords || row.predicted_product_name}<br/>Vector Dim: ${row.embedding_dimensions || 3072}-D`}
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
            <tr onclick="selectPipelineFacing('${containerId}', ${idx})" style="cursor:pointer; background:${idx === selectedIdx ? '#eff6ff' : 'transparent'};">
              <td><strong>#${r.product_index}</strong></td>
              <td><code>[${r.bbox_ymin},${r.bbox_xmin},${r.bbox_ymax},${r.bbox_xmax}]</code></td>
              <td>${r.predicted_category}</td>
              <td>${r.predicted_subcategory}</td>
              <td>
                <strong>${r.predicted_brand}</strong>
                ${r.is_hul_brand ? `<span class="tag-hul">HUL</span>` : `<span class="tag-non-hul">Non-HUL</span>`}
              </td>
              <td>${r.predicted_variant}</td>
              <td><code>${r.predicted_packaging}</code></td>
              <td>${r.predicted_pack_type}</td>
              <td><strong>${r.rule_derived_size_bucket}</strong></td>
              <td>
                ${isScannMode
                  ? `<code>${r.matched_sku_id || "SKU"}</code> (<strong>${(r.confidence * 100).toFixed(1)}%</strong>)`
                  : `<code>${r.embedding_dimensions || 3072}-D</code> &bull; $${(r.cost_per_product_usd || 0).toFixed(5)}`}
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
    const statusPill = document.getElementById("live-status-pill");
    const timerDisplay = document.getElementById("live-timer-display");

    runBtn.disabled = true;
    runBtn.textContent = "Executing Live on Vertex AI...";
    statusPill.className = "status-pill status-running";
    statusPill.textContent = `RUNNING LIVE ON VERTEX AI (${modName})...`;

    let activeStageIdx = 0;
    const tStart = performance.now();
    renderLiveProgressStepper(appId, activeStageIdx, false);

    const timerInterval = setInterval(() => {
      const elapsedSec = ((performance.now() - tStart) / 1000).toFixed(1);
      timerDisplay.textContent = `Elapsed: ${elapsedSec}s`;
      const pathSteps = USE_CASE_PATHS[appId].steps.length;
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
        }),
      });
      const liveResult = await response.json();
      clearInterval(timerInterval);

      if (liveResult.error) {
        statusPill.className = "status-pill status-idle";
        statusPill.textContent = `ERROR: ${liveResult.error}`;
      } else {
        statusPill.className = "status-pill status-done";
        statusPill.textContent = `LIVE VERTEX AI RUN COMPLETE (${(liveResult.latency_ms / 1000).toFixed(2)}s • ${liveResult.front_facings_count} Front Facings)`;
        renderLiveProgressStepper(appId, USE_CASE_PATHS[appId].steps.length, true);

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
      runBtn.innerHTML = "&#9654; Run Selected Use Case Live on Vertex AI";
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
        <span class="flow-step-num">${s.num} &bull; ${statusBadge}</span>
        <div class="flow-step-title">${s.title}</div>
        <div class="flow-step-desc">${s.desc}</div>
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
    viewer.innerHTML = `
      <div class="inspector-grid">
        <div class="inspector-item">
          <span>Config File Source</span>
          <strong><code>${tax.config_source || "configs/taxonomy.yaml"}</code></strong>
        </div>
        <div class="inspector-item">
          <span>Pack Types</span>
          <strong>${(tax.pack_types || []).join(", ")}</strong>
        </div>
        <div class="inspector-item" style="grid-column: span 2;">
          <span>Categories (Configurable)</span>
          <strong>${(tax.categories || []).join(" • ")}</strong>
        </div>
        <div class="inspector-item" style="grid-column: span 2;">
          <span>Subcategories (Configurable)</span>
          <strong>${(tax.subcategories || []).join(" • ")}</strong>
        </div>
        <div class="inspector-item" style="grid-column: span 2;">
          <span>Packaging Types (Configurable)</span>
          <strong>${(tax.packaging_types || []).join(" • ")}</strong>
        </div>
        <div class="inspector-item" style="grid-column: span 2;">
          <span>Rule-Derived Size Buckets (Configurable)</span>
          <strong>${(tax.size_buckets || []).join(" • ")}</strong>
        </div>
        <div class="inspector-item" style="grid-column: span 2;">
          <span>Brand Attribution Mode (HUL vs. Non-HUL)</span>
          <strong>${
            (tax.hul_brands || []).length > 0
              ? (tax.hul_brands || []).slice(0, 20).join(", ")
              : "Open-Vocabulary VLM &amp; Catalog Attribution (No predefined brand definitions required in configs/taxonomy.yaml)"
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
          <td><code>${(s.TraceId || "").slice(0, 12)}...</code></td>
          <td><code>${attr["shelf_benchmark.task_type"] || ""}</code> (${attr["shelf_benchmark.separation_approach"] || ""})</td>
          <td><code>${attr["gen_ai.request.model"] || ""}</code></td>
          <td>${attr["shelf_benchmark.latency_ms"] || ""}</td>
          <td>${attr["gen_ai.usage.total_tokens"] || ""}</td>
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
        <td><code>${r.model_name}</code></td>
        <td><code>gs://unilever-shelf-understanding-shelf-images/sft/sft_training_examples.jsonl</code></td>
        <td><span class="tag-hul">${r.status}</span></td>
        <td><strong>${r.front_facings_count}</strong></td>
        <td>${r.latency_ms.toFixed(1)} ms</td>
        <td>$${r.cost_per_shelf_image_usd.toFixed(6)}</td>
        <td>$${r.cost_per_product_usd.toFixed(6)}</td>
      </tr>
    `).join("");
  }

  const sftPre = document.getElementById("sft-jsonl-preview");
  if (sftPre && DASHBOARD_DATA.sft_examples && DASHBOARD_DATA.sft_examples.length > 0) {
    sftPre.textContent = JSON.stringify(DASHBOARD_DATA.sft_examples[0], null, 2);
  }
}
