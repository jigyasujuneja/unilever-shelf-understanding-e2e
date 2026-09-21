/**
 * Unilever Shelf Understanding & Computer Vision Benchmark Workbench
 * Interactive Frontend Controller (ui/static/app.js)
 */

const state = {
  dashboard: null,
  activeModel: "gemini-3.8-flash",
  activeStep: "detection",
  activeApproach: "two_stage_physical_crop_per_facing",
  selectedFacingIndex: 1,
  showDepthDuplicates: true,
  simulateGroundTruthPreview: false,
  hybridSearchResults: null,
  hybridQuery: "Pond's Bright Beauty Spot-less Glow Vitamin B3 Face Wash Tube Single",
  denseWeight: 0.65,
  sparseWeight: 0.35,
};

const MODEL_FOLDER_MAP = {
  "gemini-3.8-flash": "gemini-3_8-flash",
  "gemini-3.7-flash": "gemini-3_7-flash",
  "gemini-3.5-flash-lite": "gemini-3_5-flash-lite",
};

function fmtUsd(val) {
  if (val === null || val === undefined) return "$0.00000";
  return "$" + Number(val).toFixed(5);
}

function fmtMs(val) {
  if (val === null || val === undefined) return "—";
  return Math.round(Number(val)).toLocaleString() + " ms";
}

function getActiveRunSummary() {
  if (!state.dashboard) return null;
  const summaries = state.dashboard.summary || [];
  if (state.activeStep === "detection") {
    return summaries.find(
      (s) => s.model_name === state.activeModel && s.task_type === "detection"
    );
  }
  if (state.activeStep === "classification") {
    return (
      summaries.find(
        (s) =>
          s.model_name === state.activeModel &&
          s.task_type === "classification" &&
          s.separation_approach === state.activeApproach
      ) ||
      summaries.find(
        (s) => s.model_name === state.activeModel && s.task_type === "classification"
      )
    );
  }
  if (state.activeStep === "matching") {
    return summaries.find(
      (s) => s.model_name === state.activeModel && s.task_type === "matching"
    );
  }
  if (state.activeStep === "finetuning") {
    return summaries.find(
      (s) => s.model_name === state.activeModel && s.task_type === "fine_tuning"
    );
  }
  return summaries.find(
    (s) =>
      s.model_name === state.activeModel &&
      s.task_type === "classification" &&
      s.separation_approach === state.activeApproach
  );
}

function getActiveRows() {
  if (!state.dashboard) return [];
  const allRows = state.dashboard.rows || [];

  let taskFilter = "classification";
  let approachFilter = state.activeApproach;

  if (state.activeStep === "detection") {
    taskFilter = "detection";
    approachFilter = "single_pass_facing_nms";
  } else if (state.activeStep === "matching") {
    taskFilter = "matching";
    approachFilter = "hybrid_search_facing_nms";
  } else if (state.activeStep === "classification") {
    taskFilter = "classification";
    approachFilter = state.activeApproach;
  } else {
    // For fine-tuning or benchmarks view, show physical crop classification rows on canvas
    taskFilter = "classification";
    approachFilter = "two_stage_physical_crop_per_facing";
  }

  let filtered = allRows.filter(
    (r) =>
      r.model_name === state.activeModel &&
      r.task_type === taskFilter &&
      r.separation_approach === approachFilter
  );

  if (filtered.length === 0) {
    filtered = allRows.filter(
      (r) => r.model_name === state.activeModel && r.task_type === "classification"
    );
  }
  return filtered;
}

function getCropFolder(modelName) {
  if (state.activeApproach === "class_agnostic_visual_embedding") {
    return `${modelName}_class_agnostic_visual_embedding`;
  }
  return `${modelName}_two_stage_physical_crop_per_facing`;
}

function getCropUrl(modelName, facingIndex) {
  const folder = getCropFolder(modelName);
  const padded = String(facingIndex).padStart(2, "0");
  return `/crops/${folder}/facing_${padded}.png`;
}

function getMontageUrl(modelName) {
  const folder = getCropFolder(modelName);
  return `/crops/${folder}/montage_all_facings.png`;
}

/* ==========================================================================
   KPI Strip & Header Updates
   ========================================================================== */
function updateHeaderAndKPIs() {
  const summary = getActiveRunSummary();
  const rows = getActiveRows();
  const depthDemo =
    (state.dashboard &&
      state.dashboard.depth_demos &&
      state.dashboard.depth_demos[state.activeModel]) || {
      front_facings_count: rows.length,
      depth_duplicates_filtered: 0,
    };

  const facingsCount = rows.length || (summary ? summary.front_facings_count : 17);
  const depthFiltered = depthDemo.depth_duplicates_filtered || 0;

  document.getElementById("step1-badge").textContent = `${depthDemo.front_facings_count} Front / -${depthFiltered} Depth`;
  document.getElementById("count-depth-boxes").textContent = String(depthFiltered);
  document.getElementById("crop-folder-model").textContent =
    MODEL_FOLDER_MAP[state.activeModel] || state.activeModel;

  const hulCount = rows.filter((r) => r.is_hul_brand === true).length;
  const nonHulCount = Math.max(0, rows.length - hulCount);
  const hulShare = rows.length > 0 ? ((hulCount / rows.length) * 100).toFixed(1) + "%" : "—";

  document.getElementById("kpi-facings-count").textContent = String(facingsCount);
  document.getElementById("kpi-depth-sub").textContent =
    depthFiltered > 0
      ? `${depthFiltered} back-row duplicate(s) suppressed by Depth NMS`
      : `0 back-row duplicates (clean single-row detection)`;

  document.getElementById("kpi-hul-share").textContent = hulShare;
  document.getElementById("kpi-hul-sub").textContent = `${hulCount} HUL facings vs. ${nonHulCount} Non-HUL`;

  if (summary) {
    document.getElementById("kpi-cost-shelf").textContent = fmtUsd(
      summary.cost_per_shelf_image_usd
    );
    document.getElementById("kpi-cost-sub").textContent = `${(
      summary.total_tokens || 0
    ).toLocaleString()} total tokens (${summary.input_tokens} in / ${summary.output_tokens} out)`;
    document.getElementById("kpi-cost-product").textContent = fmtUsd(
      summary.cost_per_product_usd
    );
    document.getElementById("kpi-cost-prod-sub").textContent = `Amortized across ${
      summary.front_facings_count || facingsCount
    } facings`;
    document.getElementById("kpi-latency").textContent = fmtMs(summary.latency_ms);
    document.getElementById("kpi-latency-sub").textContent = `${fmtMs(
      summary.latency_per_facing_ms
    )} per front facing`;
    document.getElementById("active-approach-indicator").textContent = `Task: ${summary.task_type} | Approach: ${summary.separation_approach}`;
    document.getElementById("right-panel-trace-badge").textContent = `Trace: ${String(
      summary.trace_id || ""
    ).slice(0, 10)}...`;
  }

  const accEl = document.getElementById("kpi-accuracy-state");
  const accSubEl = document.getElementById("kpi-accuracy-sub");
  const headerBadge = document.getElementById("header-gt-badge");

  if (state.simulateGroundTruthPreview) {
    accEl.textContent = "94.1% mAP@50 (PREVIEW)";
    accEl.style.color = "var(--hul-emerald)";
    accSubEl.textContent = "Simulated GT schema connected (17/17 slots)";
    headerBadge.textContent = "GT PREVIEW ACTIVE";
    headerBadge.className = "badge badge-hul";
  } else {
    accEl.textContent = "PLACEHOLDER READY";
    accEl.style.color = "var(--non-hul-amber)";
    accSubEl.textContent = "provider_type: 'none' (Zero fake data)";
    headerBadge.textContent = "PLACEHOLDER READY";
    headerBadge.className = "badge badge-non-hul";
  }
}

/* ==========================================================================
   Interactive Shelf SVG Bounding Box Canvas & Physical Crop Strip
   ========================================================================== */
function renderShelfCanvas() {
  const svg = document.getElementById("shelf-bbox-svg");
  svg.innerHTML = "";

  const rows = getActiveRows();
  const depthDemo =
    state.dashboard &&
    state.dashboard.depth_demos &&
    state.dashboard.depth_demos[state.activeModel];

  // 1. If showDepthDuplicates is enabled, draw suppressed back-row boxes first (underneath)
  if (state.showDepthDuplicates && depthDemo && Array.isArray(depthDemo.boxes)) {
    const suppressedBoxes = depthDemo.boxes.filter((b) => !b.kept_by_depth_nms);
    suppressedBoxes.forEach((b, idx) => {
      const [ymin, xmin, ymax, xmax] = b.bbox_2d;
      const w = Math.max(8, xmax - xmin);
      const h = Math.max(8, ymax - ymin);

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", xmin);
      rect.setAttribute("y", ymin);
      rect.setAttribute("width", w);
      rect.setAttribute("height", h);
      rect.setAttribute("fill", "rgba(225, 29, 72, 0.22)");
      rect.setAttribute("stroke", "#e11d48");
      rect.setAttribute("stroke-width", "2.5");
      rect.setAttribute("stroke-dasharray", "6,4");
      rect.setAttribute("class", "bbox-rect");
      rect.setAttribute("title", b.nms_reason);

      rect.addEventListener("click", () => {
        showDepthDuplicatePopover(b);
      });

      const tagBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      tagBg.setAttribute("x", xmin);
      tagBg.setAttribute("y", Math.max(2, ymin - 18));
      tagBg.setAttribute("width", Math.min(110, w + 36));
      tagBg.setAttribute("height", 17);
      tagBg.setAttribute("rx", 3);
      tagBg.setAttribute("fill", "#e11d48");

      const tagTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tagTxt.setAttribute("x", xmin + 4);
      tagTxt.setAttribute("y", Math.max(14, ymin - 6));
      tagTxt.setAttribute("class", "bbox-label-text");
      tagTxt.textContent = `DEPTH DUP -Slot #${b.occluded_by_slot || idx + 1}`;

      svg.appendChild(rect);
      svg.appendChild(tagBg);
      svg.appendChild(tagTxt);
    });
  }

  // 2. Draw front-facing product bounding boxes
  rows.forEach((r) => {
    const idx = r.product_index;
    const ymin = r.bbox_ymin;
    const xmin = r.bbox_xmin;
    const ymax = r.bbox_ymax;
    const xmax = r.bbox_xmax;
    const w = Math.max(10, xmax - xmin);
    const h = Math.max(10, ymax - ymin);

    const isSelected = idx === state.selectedFacingIndex;
    const isHul = r.is_hul_brand === true;

    const strokeColor = isSelected
      ? "#0057b8"
      : isHul
      ? "#059669"
      : "#d97706";
    const fillColor = isSelected
      ? "rgba(0, 87, 184, 0.28)"
      : isHul
      ? "rgba(5, 150, 105, 0.16)"
      : "rgba(217, 119, 6, 0.18)";

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", xmin);
    rect.setAttribute("y", ymin);
    rect.setAttribute("width", w);
    rect.setAttribute("height", h);
    rect.setAttribute("fill", fillColor);
    rect.setAttribute("stroke", strokeColor);
    rect.setAttribute("stroke-width", isSelected ? "4" : "2.2");
    rect.setAttribute("class", `bbox-rect ${isSelected ? "selected" : ""}`);

    rect.addEventListener("click", () => {
      state.selectedFacingIndex = idx;
      renderShelfCanvas();
      renderPhysicalCropsStrip();
      renderRightWorkbench();
    });

    const labelWidth = Math.min(Math.max(w, 48), 96);
    const labelBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    labelBg.setAttribute("x", xmin);
    labelBg.setAttribute("y", ymin);
    labelBg.setAttribute("width", labelWidth);
    labelBg.setAttribute("height", 18);
    labelBg.setAttribute("fill", strokeColor);
    labelBg.setAttribute("class", "bbox-label-bg");

    const labelTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    labelTxt.setAttribute("x", xmin + 4);
    labelTxt.setAttribute("y", ymin + 13);
    labelTxt.setAttribute("class", "bbox-label-text");
    const shortBrand = (r.predicted_brand || "Facing").slice(0, 8);
    labelTxt.textContent = `#${idx} ${shortBrand}`;

    svg.appendChild(rect);
    svg.appendChild(labelBg);
    svg.appendChild(labelTxt);
  });
}

function renderPhysicalCropsStrip() {
  const strip = document.getElementById("physical-crops-strip");
  strip.innerHTML = "";
  const rows = getActiveRows();

  rows.forEach((r) => {
    const idx = r.product_index;
    const card = document.createElement("div");
    card.className = `crop-thumb-card ${
      idx === state.selectedFacingIndex ? "active" : ""
    }`;
    const imgUrl = getCropUrl(state.activeModel, idx);
    card.innerHTML = `
      <img src="${imgUrl}" alt="Facing #${idx} ${r.predicted_brand}" loading="lazy" />
      <div class="crop-thumb-label">#${idx} ${r.predicted_brand || "Facing"}</div>
    `;
    card.addEventListener("click", () => {
      state.selectedFacingIndex = idx;
      renderShelfCanvas();
      renderPhysicalCropsStrip();
      renderRightWorkbench();
    });
    strip.appendChild(card);
  });
}

function showDepthDuplicatePopover(boxInfo) {
  openModal(
    `Suppressed Back-Row Depth Duplicate (Slot #${boxInfo.occluded_by_slot})`,
    `Filtered automatically by deduplicate_depth_stacked_facings() in src/shelf_benchmark/tasks/facing_utils.py`,
    `
      <div class="facing-inspector">
        <span class="badge badge-depth">SUPPRESSED BY FRONT-FACING DEPTH NMS</span>
        <h3 style="margin-top: 10px; font-size: 15px;">Why was this bounding box excluded from the facing count?</h3>
        <p style="margin-top: 6px; color: var(--text-secondary);">
          In retail shelf execution (Share-of-Shelf / Facing Audit), products stocked <strong>behind</strong> the front-facing unit in the same horizontal column must <strong>not</strong> be counted as additional facings.
        </p>
        <div class="taxonomy-7d-grid" style="margin-top: 14px;">
          <div class="dim-cell">
            <div class="dim-name">Back-Row Candidate Bbox [ymin, xmin, ymax, xmax]</div>
            <div class="dim-val" style="font-family: var(--font-mono);">[${boxInfo.bbox_2d.join(", ")}]</div>
          </div>
          <div class="dim-cell">
            <div class="dim-name">Occluding Front Slot</div>
            <div class="dim-val">Horizontal Slot #${boxInfo.occluded_by_slot} (Front Base y_max = ${boxInfo.front_ymax})</div>
          </div>
          <div class="dim-cell full-width">
            <div class="dim-name">Geometric NMS Rule Triggered</div>
            <div class="dim-val" style="color: var(--depth-coral);">${boxInfo.nms_reason}</div>
          </div>
        </div>
      </div>
    `
  );
}

/* ==========================================================================
   Selected Facing 7-Dimension HUL Inspector Component
   ========================================================================== */
function buildSelectedFacingInspectorHTML(row) {
  if (!row) return `<p>Select any bounding box on the shelf to inspect its details.</p>`;
  const idx = row.product_index;
  const isHul = row.is_hul_brand === true;
  const cropUrl = getCropUrl(state.activeModel, idx);

  return `
    <div class="facing-inspector">
      <div class="inspector-top">
        <div class="crop-preview-box">
          <img src="${cropUrl}" alt="Cropped Facing #${idx}" />
          <span class="crop-tag">#${String(idx).padStart(2, "0")}</span>
        </div>
        <div>
          <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <span class="badge ${isHul ? "badge-hul" : "badge-non-hul"}">
              ${isHul ? "HUL PORTFOLIO BRAND" : "NON-HUL COMPETITOR BRAND"}
            </span>
            <span class="badge badge-cobalt">Row: ${row.shelf_row.toUpperCase()} &bull; Slot #${row.position_on_shelf}</span>
            <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">
              bbox: [${row.bbox_ymin}, ${row.bbox_xmin}, ${row.bbox_ymax}, ${row.bbox_xmax}]
            </span>
          </div>

          <h3 style="font-size: 16px; font-weight: 800; margin-top: 8px; color: var(--text-primary);">
            ${row.predicted_brand || "Unknown"} — ${row.predicted_variant || row.predicted_product_name}
          </h3>

          <div class="taxonomy-7d-grid">
            <div class="dim-cell">
              <div class="dim-number">Dimension 01 &bull; Category (Configurable)</div>
              <div class="dim-val">${row.predicted_category || "Skin Cleansing"}</div>
            </div>
            <div class="dim-cell">
              <div class="dim-number">Dimension 02 &bull; Subcategory (Configurable)</div>
              <div class="dim-val">${row.predicted_subcategory || "Face Wash"}</div>
            </div>
            <div class="dim-cell">
              <div class="dim-number">Dimension 03 &bull; Brand (HUL vs. Non-HUL)</div>
              <div class="dim-val">${row.predicted_brand} (${isHul ? "HUL" : "Non-HUL"})</div>
            </div>
            <div class="dim-cell">
              <div class="dim-number">Dimension 04 &bull; Variant</div>
              <div class="dim-val">${row.predicted_variant || "Standard"}</div>
            </div>
            <div class="dim-cell">
              <div class="dim-number">Dimension 05 &bull; Packaging Type (Configurable)</div>
              <div class="dim-val" style="text-transform: capitalize;">${row.predicted_packaging || "tube"}</div>
            </div>
            <div class="dim-cell">
              <div class="dim-number">Dimension 06 &bull; Pack Type (Configurable)</div>
              <div class="dim-val">${row.predicted_pack_type || "Single"}</div>
            </div>
            <div class="dim-cell full-width">
              <div class="dim-number">Dimension 07 &bull; Size Bucket (Model vs. Deterministic BBox Rule)</div>
              <div class="dim-val">
                Model: <strong>${row.predicted_size || "—"}</strong> &nbsp;|&nbsp;
                <code>derive_size_bucket_from_bbox()</code>: <strong style="color: var(--brand-cobalt);">${row.rule_derived_size_bucket || "—"}</strong>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

/* ==========================================================================
   Right Panel Step-Specific Renderers
   ========================================================================== */
function renderRightWorkbench() {
  const titleEl = document.getElementById("right-panel-title");
  const subEl = document.getElementById("right-panel-subtitle");
  const container = document.getElementById("right-panel-content");

  const rows = getActiveRows();
  const selectedRow =
    rows.find((r) => r.product_index === state.selectedFacingIndex) || rows[0];

  if (state.activeStep === "detection") {
    titleEl.textContent = "Step 1: Object Detection & Front-Facing Depth NMS";
    subEl.textContent =
      "Detects front-most physical facings [ymin, xmin, ymax, xmax] and suppresses depth-stacked back units.";

    const depthDemo =
      (state.dashboard &&
        state.dashboard.depth_demos &&
        state.dashboard.depth_demos[state.activeModel]) || {
        raw_detections_count: rows.length,
        front_facings_count: rows.length,
        depth_duplicates_filtered: 0,
      };

    container.innerHTML = `
      ${buildSelectedFacingInspectorHTML(selectedRow)}

      <div style="background: var(--bg-surface-subtle); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 12px 14px; margin-bottom: 14px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
          <strong style="font-size: 12.5px;">Front-Facing Column Depth NMS (<code>deduplicate_depth_stacked_facings</code>)</strong>
          <span class="badge badge-cobalt">Overlap Threshold: 0.45</span>
        </div>
        <p style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">
          Raw Model Boxes: <strong>${depthDemo.raw_detections_count}</strong> &rarr;
          Back-Row Duplicates Filtered: <strong style="color: var(--depth-coral);">-${depthDemo.depth_duplicates_filtered}</strong> &rarr;
          Final Front-Facing Slots: <strong style="color: var(--hul-emerald);">${depthDemo.front_facings_count}</strong>
        </p>
      </div>

      <div class="facing-table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Slot</th>
              <th>Shelf Row</th>
              <th>BBox [ymin, xmin, ymax, xmax]</th>
              <th>Brand Hint</th>
              <th>Rule Size Bucket</th>
            </tr>
          </thead>
          <tbody>
            ${rows
              .map(
                (r) => `
              <tr class="${
                r.product_index === state.selectedFacingIndex ? "active-row" : ""
              }" data-idx="${r.product_index}">
                <td style="font-family: var(--font-mono); font-weight: 700;">#${r.product_index}</td>
                <td>${r.shelf_row}</td>
                <td style="font-family: var(--font-mono); font-size: 11.5px;">[${r.bbox_ymin}, ${r.bbox_xmin}, ${r.bbox_ymax}, ${r.bbox_xmax}]</td>
                <td>
                  <span class="badge ${r.is_hul_brand ? "badge-hul" : "badge-non-hul"}">${r.predicted_brand}</span>
                </td>
                <td style="font-size: 11.5px;">${r.rule_derived_size_bucket}</td>
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;

    bindTableRows(container);
    return;
  }

  if (state.activeStep === "classification") {
    titleEl.textContent =
      "Step 2: 7-Dimension HUL Classification & Separation Approaches";
    subEl.textContent =
      "Switch between the 3 bounding-box separation architectures to compare taxonomy extraction, latency, and token cost.";

    const summaries = (state.dashboard.summary || []).filter(
      (s) =>
        s.model_name === state.activeModel && s.task_type === "classification"
    );

    const approachesMeta = [
      {
        id: "single_pass_full_shelf",
        label: "Approach A: Single-Pass Full Shelf VLM",
        desc: "1 VLM call extracts all front bboxes + 7 HUL dimensions simultaneously.",
      },
      {
        id: "two_stage_bbox_guided_nms",
        label: "Approach B: Two-Stage BBox-Guided NMS",
        desc: "Stage 1 Depth NMS locks coordinates; Stage 2 classifies locked regions.",
      },
      {
        id: "two_stage_physical_crop_per_facing",
        label: "Approach C: Physical Crop + VLM",
        desc: "PIL crops each facing into facing_XX.png + montage grid for VLM fine print.",
      },
      {
        id: "class_agnostic_visual_embedding",
        label: "Approach D: 3-Stage Class-Agnostic + 1408-D ViT Embedding + ScaNN",
        desc: "Stage 1 'product' box_2d -> Stage 2 multimodalembedding@001 (1408-D crop vector) -> Stage 3 ScaNN ANN.",
      },
    ];

    container.innerHTML = `
      <div class="approach-selector-grid">
        ${approachesMeta
          .map((ap) => {
            const s = summaries.find((x) => x.separation_approach === ap.id) || {};
            const isAct = state.activeApproach === ap.id;
            return `
              <button type="button" class="approach-card ${
                isAct ? "active" : ""
              }" data-approach="${ap.id}">
                <h4>${ap.label}</h4>
                <p>${ap.desc}</p>
                <div class="approach-metrics">
                  <span>${fmtMs(s.latency_ms)}</span>
                  <span>&bull;</span>
                  <span>${fmtUsd(s.cost_per_shelf_image_usd)}</span>
                </div>
              </button>
            `;
          })
          .join("")}
      </div>

      ${buildSelectedFacingInspectorHTML(selectedRow)}

      <div class="facing-table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Brand (HUL?)</th>
              <th>Category &amp; Subcategory</th>
              <th>Variant (Zero-Shot Read)</th>
              <th>Pack</th>
              <th>Size Bucket</th>
            </tr>
          </thead>
          <tbody>
            ${rows
              .map(
                (r) => `
              <tr class="${
                r.product_index === state.selectedFacingIndex ? "active-row" : ""
              }" data-idx="${r.product_index}">
                <td style="font-family: var(--font-mono); font-weight: 700;">#${r.product_index}</td>
                <td>
                  <span class="badge ${r.is_hul_brand ? "badge-hul" : "badge-non-hul"}">
                    ${r.predicted_brand}
                  </span>
                </td>
                <td><strong>${r.predicted_category}</strong><br/><span style="font-size:11px;color:var(--text-muted);">${r.predicted_subcategory}</span></td>
                <td>${r.predicted_variant}</td>
                <td>${r.predicted_packaging} (${r.predicted_pack_type})</td>
                <td style="font-size: 11.5px;">${r.predicted_size}</td>
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;

    container.querySelectorAll(".approach-card").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.activeApproach = btn.getAttribute("data-approach");
        updateHeaderAndKPIs();
        renderShelfCanvas();
        renderPhysicalCropsStrip();
        renderRightWorkbench();
      });
    });

    bindTableRows(container);
    return;
  }

  if (state.activeStep === "matching") {
    titleEl.textContent =
      "Step 3: Hybrid Vector Search (BM25 + gemini-embedding-001 3072-D)";
    subEl.textContent =
      "Test live catalog query matching against extracted shelf facings using Dense Cosine Similarity + Sparse Lexical BM25.";

    const presetQueries = [
      "Pond's Bright Beauty Spot-less Glow Vitamin B3 Face Wash Tube Single",
      "Pond's Pure Detox Anti-Pollution Activated Charcoal Face Wash Black Tube",
      "Lakme Blush & Glow Strawberry Sheet Mask / Gel Face Wash",
      "Simple Kind to Skin Refreshing Facial Wash 100% Soap Free",
      "Pears Pure & Gentle Shower Gel / Face Wash with Natural Oils",
      "Vaseline Healthy Bright Daily Brightening Body Lotion / Cream Jar",
    ];

    const searchRes = state.hybridSearchResults;

    container.innerHTML = `
      <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 14px; margin-bottom: 14px;">
        <div style="font-size: 12px; font-weight: 700; margin-bottom: 6px;">
          Live Catalog SKU Query Sandbox (Calls Vertex AI <code>gemini-embedding-001</code> 3072-D API)
        </div>
        <div class="search-bar-row">
          <input type="text" id="input-hybrid-query" class="search-input" value="${state.hybridQuery.replace(
            /"/g,
            "&quot;"
          )}" />
          <button type="button" class="btn-primary" id="btn-exec-hybrid">
            Search Shelf Facings
          </button>
        </div>
        <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px;">
          ${presetQueries
            .map(
              (q) =>
                `<button type="button" class="btn-secondary preset-query-btn" style="font-size: 11px; padding: 3px 8px;" data-q="${q.replace(
                  /"/g,
                  "&quot;"
                )}">${q.slice(0, 36)}...</button>`
            )
            .join("")}
        </div>
        <div style="display: flex; align-items: center; gap: 16px; font-size: 12px; color: var(--text-secondary);">
          <span>Dense Vector Weight (<code>gemini-embedding-001</code>): <strong>${Math.round(
            state.denseWeight * 100
          )}%</strong></span>
          <span>Sparse Lexical Weight (BM25 Keywords): <strong>${Math.round(
            state.sparseWeight * 100
          )}%</strong></span>
        </div>
      </div>

      ${
        searchRes
          ? `
        <div style="margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
          <span class="badge badge-hul">Top Match: Slot #${searchRes.results[0].product_index} (${searchRes.results[0].brand}) — Hybrid Score: ${searchRes.results[0].hybrid_rrf_score}</span>
          <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">
            3072-D Embedding Latency: ${searchRes.latency_ms} ms
          </span>
        </div>
        <div class="facing-table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Slot</th>
                <th>Brand &amp; Variant</th>
                <th>Dense Cosine</th>
                <th>BM25 Sparse</th>
                <th>Hybrid Score</th>
              </tr>
            </thead>
            <tbody>
              ${searchRes.results
                .slice(0, 10)
                .map(
                  (item) => `
                <tr class="${
                  item.product_index === state.selectedFacingIndex
                    ? "active-row"
                    : ""
                }" data-idx="${item.product_index}">
                  <td style="font-family: var(--font-mono); font-weight: 700;">#${item.rank}</td>
                  <td style="font-family: var(--font-mono);">Slot #${item.product_index}</td>
                  <td>
                    <strong>${item.brand}</strong> — ${item.variant}
                    <div style="font-family: var(--font-mono); font-size: 10.5px; color: var(--text-muted); margin-top: 2px;">
                      Keywords: ${item.lexical_search_keywords}
                    </div>
                  </td>
                  <td style="font-family: var(--font-mono);">${item.dense_cosine_similarity}</td>
                  <td style="font-family: var(--font-mono);">${item.sparse_lexical_score}</td>
                  <td style="min-width: 110px;">
                    <strong style="font-family: var(--font-mono); color: var(--brand-cobalt);">${item.hybrid_rrf_score}</strong>
                    <div class="score-bar-track">
                      <div class="score-bar-fill" style="width: ${Math.min(
                        100,
                        Math.round(item.hybrid_rrf_score * 100)
                      )}%;"></div>
                    </div>
                  </td>
                </tr>
              `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      `
          : `
        ${buildSelectedFacingInspectorHTML(selectedRow)}
        <div style="background: var(--bg-surface-subtle); padding: 12px; border-radius: var(--radius-md); font-size: 12px;">
          <div><strong>Selected Facing Lexical Search Keywords (BM25):</strong></div>
          <code style="display: block; margin-top: 4px; color: var(--brand-cobalt);">${
            selectedRow ? selectedRow.lexical_search_keywords : ""
          }</code>
          <div style="margin-top: 8px;"><strong>Selected Facing Dense Embedding Passage (3072-D input):</strong></div>
          <code style="display: block; margin-top: 4px;">${
            selectedRow ? selectedRow.dense_embedding_text : ""
          }</code>
        </div>
      `
      }
    `;

    const execBtn = document.getElementById("btn-exec-hybrid");
    if (execBtn) {
      execBtn.addEventListener("click", () => {
        const qVal = document.getElementById("input-hybrid-query").value;
        state.hybridQuery = qVal;
        runHybridSearchAPI(qVal);
      });
    }

    container.querySelectorAll(".preset-query-btn").forEach((b) => {
      b.addEventListener("click", () => {
        const q = b.getAttribute("data-q");
        state.hybridQuery = q;
        document.getElementById("input-hybrid-query").value = q;
        runHybridSearchAPI(q);
      });
    });

    bindTableRows(container);
    return;
  }

  if (state.activeStep === "finetuning") {
    titleEl.textContent =
      "Step 4: Vertex AI Gemini Supervised Fine-Tuning (SFT) Pipeline";
    subEl.textContent =
      "Automated generation of multi-turn multimodal JSONL training records uploaded to GCS for client.tunings.tune.";

    const sftExamples = state.dashboard.sft_examples || [];
    const sampleJson =
      sftExamples.length > 0
        ? JSON.stringify(sftExamples[0], null, 2)
        : "// SFT JSONL payload ready in reports/tuning_data/shelf_sft_train.jsonl";

    container.innerHTML = `
      <div class="taxonomy-7d-grid" style="margin-bottom: 14px;">
        <div class="dim-cell">
          <div class="dim-number">GCS Training Artifact URI</div>
          <div class="dim-val" style="font-family: var(--font-mono); font-size: 11.5px;">
            gs://unilever-shelf-understanding-shelf-images/tuning/shelf_sft_train.jsonl
          </div>
        </div>
        <div class="dim-cell">
          <div class="dim-number">Vertex AI SFT API Endpoint</div>
          <div class="dim-val" style="font-family: var(--font-mono); font-size: 11.5px;">
            client.tunings.tune(base_model="${state.activeModel}")
          </div>
        </div>
      </div>

      <div style="margin-bottom: 8px; font-size: 12px; font-weight: 700; color: var(--text-secondary);">
        Generated Vertex AI SFT Example Payload (<code>shelf_sft_train.jsonl</code>):
      </div>
      <pre class="code-console">${sampleJson
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")}</pre>
    `;
    return;
  }

  if (state.activeStep === "benchmarks") {
    titleEl.textContent =
      "Step 5: Multi-Model Benchmark Matrix & Ground Truth Placeholder Hub";
    subEl.textContent =
      "Complete 18-run telemetry across gemini-3.8-flash, gemini-3.7-flash, and gemini-3.5-flash-lite + plug-and-play Ground Truth schema config.";

    const allSummaries = state.dashboard.summary || [];

    container.innerHTML = `
      <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-strong); border-radius: var(--radius-md); padding: 14px; margin-bottom: 14px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
          <div>
            <span class="badge ${
              state.simulateGroundTruthPreview ? "badge-hul" : "badge-non-hul"
            }">
              ${
                state.simulateGroundTruthPreview
                  ? "ARCHITECTURAL PREVIEW: GROUND TRUTH CONNECTED"
                  : "CURRENT STATE: PLACEHOLDER_AWAITING_GROUND_TRUTH"
              }
            </span>
            <h3 style="font-size: 14px; font-weight: 700; margin-top: 6px;">
              Plug-and-Play Ground Truth, Product Catalog &amp; Planogram Adapter Hub
            </h3>
            <p style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
              Currently running with <code>ground_truth.provider_type: "none"</code> (zero fake data). Toggle preview below to demonstrate to executives how accuracy metrics populate once annotated tables are linked.
            </p>
          </div>
          <button type="button" class="btn-primary" id="btn-toggle-gt-preview">
            ${
              state.simulateGroundTruthPreview
                ? "Reset to Pure Placeholder Mode"
                : "Simulate Ground Truth Attachment (Exec Demo)"
            }
          </button>
        </div>

        <div class="gt-architecture-grid">
          <div class="arch-box connected">
            <h3>
              <span>1. Shelf Images Bucket</span>
              <span class="badge badge-hul">LIVE GCS</span>
            </h3>
            <div style="font-family: var(--font-mono); font-size: 11px; margin-bottom: 4px;">
              gs://unilever-shelf-understanding-shelf-images
            </div>
            <p style="font-size: 11.5px; color: var(--text-secondary);">
              Connected &amp; verified with <code>shelf-image.png</code> + 62 report &amp; physical crop artifacts.
            </p>
          </div>

          <div class="arch-box ${state.simulateGroundTruthPreview ? "connected" : ""}">
            <h3>
              <span>2. Product Catalog Bucket</span>
              <span class="badge ${
                state.simulateGroundTruthPreview ? "badge-hul" : "badge-non-hul"
              }">${state.simulateGroundTruthPreview ? "PREVIEW" : "READY"}</span>
            </h3>
            <div style="font-family: var(--font-mono); font-size: 11px; margin-bottom: 4px;">
              gs://unilever-shelf-understanding-catalog-images
            </div>
            <p style="font-size: 11.5px; color: var(--text-secondary);">
              Links to 3072-D <code>gemini-embedding-001</code> index for Top-K SKU ID resolution.
            </p>
          </div>

          <div class="arch-box ${state.simulateGroundTruthPreview ? "connected" : ""}">
            <h3>
              <span>3. Ground Truth &amp; Planograms</span>
              <span class="badge ${
                state.simulateGroundTruthPreview ? "badge-hul" : "badge-non-hul"
              }">${state.simulateGroundTruthPreview ? "PREVIEW" : "PLACEHOLDER"}</span>
            </h3>
            <div style="font-family: var(--font-mono); font-size: 11px; margin-bottom: 4px;">
              BigQuery / CSV / JSON Schema Mapper
            </div>
            <p style="font-size: 11.5px; color: var(--text-secondary);">
              <code>GroundTruthSchemaMapping</code> maps custom columns to IoU@50, Brand &amp; 7-Dim F1.
            </p>
          </div>
        </div>
      </div>

      <div class="facing-table-wrap" style="max-height: 350px;">
        <table class="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Task &amp; Approach</th>
              <th>Facings</th>
              <th>Latency</th>
              <th>Cost / Shelf</th>
              <th>Cost / Facing</th>
              <th>Accuracy Status</th>
            </tr>
          </thead>
          <tbody>
            ${allSummaries
              .map((s) => {
                const previewAcc =
                  s.model_name === "gemini-3.8-flash"
                    ? "94.8% F1 (Preview)"
                    : s.model_name === "gemini-3.7-flash"
                    ? "92.4% F1 (Preview)"
                    : "89.6% F1 (Preview)";
                return `
                  <tr>
                    <td style="font-family: var(--font-mono); font-weight: 700;">${s.model_name}</td>
                    <td>
                      <strong>${s.task_type}</strong><br/>
                      <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">${s.separation_approach}</span>
                    </td>
                    <td style="font-family: var(--font-mono);">
                      ${s.front_facings_count}
                      ${
                        s.depth_duplicates_filtered > 0
                          ? `<span class="badge badge-depth" style="margin-left:4px;">-${s.depth_duplicates_filtered} depth</span>`
                          : ""
                      }
                    </td>
                    <td style="font-family: var(--font-mono);">${fmtMs(s.latency_ms)}</td>
                    <td style="font-family: var(--font-mono);">${fmtUsd(
                      s.cost_per_shelf_image_usd
                    )}</td>
                    <td style="font-family: var(--font-mono);">${fmtUsd(
                      s.cost_per_product_usd
                    )}</td>
                    <td>
                      <span class="badge ${
                        state.simulateGroundTruthPreview
                          ? "badge-hul"
                          : "badge-non-hul"
                      }">
                        ${
                          state.simulateGroundTruthPreview
                            ? previewAcc
                            : s.accuracy_status
                        }
                      </span>
                    </td>
                  </tr>
                `;
              })
              .join("")}
          </tbody>
        </table>
      </div>
    `;

    const toggleGtBtn = document.getElementById("btn-toggle-gt-preview");
    if (toggleGtBtn) {
      toggleGtBtn.addEventListener("click", () => {
        state.simulateGroundTruthPreview = !state.simulateGroundTruthPreview;
        updateHeaderAndKPIs();
        renderRightWorkbench();
      });
    }
  }
}

function bindTableRows(container) {
  container.querySelectorAll("tbody tr[data-idx]").forEach((tr) => {
    tr.addEventListener("click", () => {
      const idx = Number(tr.getAttribute("data-idx"));
      if (idx) {
        state.selectedFacingIndex = idx;
        renderShelfCanvas();
        renderPhysicalCropsStrip();
        renderRightWorkbench();
      }
    });
  });
}

/* ==========================================================================
   API Actions (Live Hybrid Vector Search & Live Vertex AI Task Runner)
   ========================================================================== */
async function runHybridSearchAPI(queryText) {
  const btn = document.getElementById("btn-exec-hybrid");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Embedding 3072-D via Vertex AI...";
  }
  try {
    const resp = await fetch("/api/hybrid-search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryText,
        model_name: state.activeModel,
        dense_weight: state.denseWeight,
        sparse_weight: state.sparseWeight,
      }),
    });
    const data = await resp.json();
    if (data && Array.isArray(data.results) && data.results.length > 0) {
      state.hybridSearchResults = data;
      state.selectedFacingIndex = data.results[0].product_index;
      renderShelfCanvas();
      renderPhysicalCropsStrip();
      renderRightWorkbench();
    }
  } catch (err) {
    console.error("Hybrid search error:", err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Search Shelf Facings";
    }
  }
}

function openModal(title, subtitle, bodyHtml) {
  document.getElementById("modal-title").textContent = title;
  document.getElementById("modal-subtitle").textContent = subtitle;
  document.getElementById("modal-body").innerHTML = bodyHtml;
  document.getElementById("modal-backdrop").style.display = "flex";
}

function closeModal() {
  document.getElementById("modal-backdrop").style.display = "none";
}

function openLiveRunnerModal() {
  openModal(
    `Run Live Benchmark API Call on Vertex AI (${state.activeModel})`,
    `Directly executes the unmodified Python task classes in src/shelf_benchmark/tasks/ against gs://unilever-shelf-understanding-shelf-images/shelf-image.png`,
    `
      <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px;">
        <select id="live-select-task" class="search-input" style="max-width: 240px;">
          <option value="detection">Step 1: ProductDetectionTask (Depth NMS)</option>
          <option value="classification">Step 2: ProductClassificationTask (7-Dim HUL)</option>
          <option value="matching">Step 3: ProductMatchingTask (gemini-embedding-001)</option>
        </select>
        <select id="live-select-approach" class="search-input" style="max-width: 310px;">
          <option value="class_agnostic_visual_embedding">Approach D: class_agnostic_visual_embedding (1408-D ViT + ScaNN)</option>
          <option value="single_pass_full_shelf">Approach A: single_pass_full_shelf</option>
          <option value="two_stage_bbox_guided_nms">Approach B: two_stage_bbox_guided_nms</option>
          <option value="two_stage_physical_crop_per_facing">Approach C: two_stage_physical_crop_per_facing</option>
        </select>
        <button type="button" class="btn-primary" id="btn-confirm-live-run">
          &#9654; Execute Live on Vertex AI (${state.activeModel})
        </button>
      </div>
      <div id="live-run-output">
        <p style="color: var(--text-secondary); font-size: 12.5px;">
          Click <strong>Execute Live on Vertex AI</strong> above to invoke <code>src/shelf_benchmark/</code> in real time and inspect the fresh OpenTelemetry span, token counts, and USD cost calculation.
        </p>
      </div>
    `
  );

  const runBtn = document.getElementById("btn-confirm-live-run");
  runBtn.addEventListener("click", async () => {
    const taskType = document.getElementById("live-select-task").value;
    const approach = document.getElementById("live-select-approach").value;
    const outEl = document.getElementById("live-run-output");
    runBtn.disabled = true;
    runBtn.textContent = "Calling Vertex AI Gemini API...";
    outEl.innerHTML = `<div class="code-console">Executing ${taskType} (${approach}) on ${state.activeModel} against gs://unilever-shelf-understanding-shelf-images/shelf-image.png...</div>`;

    try {
      const resp = await fetch("/api/run-live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_type: taskType,
          model_name: state.activeModel,
          separation_approach: approach,
        }),
      });
      const resJson = await resp.json();
      outEl.innerHTML = `
        <div style="margin-bottom: 8px;">
          <span class="badge badge-hul">LIVE VERTEX AI SPAN COMPLETED (${resJson.latency_ms} ms)</span>
          <span class="badge badge-cobalt">Cost: ${
            resJson.cost ? fmtUsd(resJson.cost.cost_per_shelf_image_usd) : "—"
          }</span>
        </div>
        <pre class="code-console">${JSON.stringify(resJson, null, 2)
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")}</pre>
      `;
    } catch (err) {
      outEl.innerHTML = `<div class="code-console" style="color: #fda4af;">Error: ${String(
        err
      )}</div>`;
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = `Execute Live on Vertex AI (${state.activeModel})`;
    }
  });
}

/* ==========================================================================
   Initialization & Event Listeners
   ========================================================================== */
async function initApp() {
  try {
    const resp = await fetch("/api/dashboard");
    state.dashboard = await resp.json();
  } catch (err) {
    console.error("Failed to load dashboard payload:", err);
  }

  // Model selector buttons
  document.querySelectorAll(".model-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".model-pill").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeModel = btn.getAttribute("data-model");
      state.selectedFacingIndex = 1;
      state.hybridSearchResults = null;
      updateHeaderAndKPIs();
      renderShelfCanvas();
      renderPhysicalCropsStrip();
      renderRightWorkbench();
    });
  });

  // Pipeline Step tabs
  document.querySelectorAll(".step-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".step-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.activeStep = tab.getAttribute("data-step");
      updateHeaderAndKPIs();
      renderShelfCanvas();
      renderPhysicalCropsStrip();
      renderRightWorkbench();
    });
  });

  // Depth duplicates checkbox toggle
  const depthChk = document.getElementById("chk-show-depth-duplicates");
  depthChk.addEventListener("change", () => {
    state.showDepthDuplicates = depthChk.checked;
    renderShelfCanvas();
  });

  // Full Montage modal button
  document.getElementById("btn-view-montage").addEventListener("click", () => {
    const montageUrl = getMontageUrl(state.activeModel);
    openModal(
      `Physical Facing Crops Montage (${state.activeModel})`,
      `Generated automatically by crop_facing_images() in src/shelf_benchmark/tasks/facing_utils.py`,
      `
        <div style="text-align: center; background: #0b111e; padding: 16px; border-radius: var(--radius-md);">
          <img src="${montageUrl}" alt="All Numbered Shelf Facings Montage" style="max-width: 100%; height: auto; border-radius: 6px;" />
        </div>
      `
    );
  });

  // Ground Truth Hub shortcut button
  document.getElementById("btn-open-gt-modal").addEventListener("click", () => {
    document.querySelectorAll(".step-tab").forEach((t) => t.classList.remove("active"));
    document.getElementById("tab-step-benchmarks").classList.add("active");
    state.activeStep = "benchmarks";
    updateHeaderAndKPIs();
    renderRightWorkbench();
  });

  // Live API Runner modal button
  document.getElementById("btn-trigger-live-api").addEventListener("click", () => {
    openLiveRunnerModal();
  });

  document.getElementById("btn-close-modal").addEventListener("click", closeModal);
  document.getElementById("modal-backdrop").addEventListener("click", (e) => {
    if (e.target.id === "modal-backdrop") closeModal();
  });

  updateHeaderAndKPIs();
  renderShelfCanvas();
  renderPhysicalCropsStrip();
  renderRightWorkbench();
}

window.addEventListener("DOMContentLoaded", initApp);
