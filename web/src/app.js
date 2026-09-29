// Perfect Store Control Plane — Unified EPIC Decision-First Validation UI & Benchmark Leaderboard Arena
const app = document.getElementById("app");
const pct = (v) => ((v ?? 0) * 100).toFixed(1) + "%";
const sec = (v) => (v ?? 0).toFixed(2) + "s";
const inr = (v) => "₹" + (v > 0 && v < 0.001 ? v.toPrecision(2) : (v ?? 0).toFixed(3));
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const getJSON = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)));

// Sister-shade quicklist for AMBIGUOUS box disambiguation
const SISTER_SHADE_QUICKLIST = [
  "Lakme 9to5 CC Cream 01 Beige",
  "Lakme 9to5 CC Cream 02 Honey",
  "Lakme 9to5 CC Cream 03 Bronze",
  "Lakme 9to5 CC Cream 04 Almond",
  "Dove Hair Therapy Daily Shine Shampoo",
  "Dove Intense Repair Conditioner Tube",
  "Vaseline Intensive Care Deep Moisture",
  "Ponds Super Light Gel Oil Free Moisturizer",
  "Sunsilk Stunning Black Shine Shampoo",
  "Clinic Plus Strong & Long Health Shampoo",
  "Surf Excel Matic Top Load Liquid",
  "Non-HUL Competitor SKU",
];

// Built-in offline fallback scenarios covering both Modern Trade (MT) and General Trade (GT)
const FALLBACK_AUDITS = [
  {
    audit_id: "AUD-MT-2026-0929-01",
    store_metadata: {
      store_id: "HUL-MT-MUM-4021 (Reliance Smart Bazaar)",
      channel_type: "MT",
      endpoint_source: "Sales EDGE",
    },
    media: {
      raw_input_url: "/img/test/test_661.jpg",
      processed_canvas_url: "/img/test/test_661.jpg",
    },
    identification_detections: [
      {
        box_id: "BOX-101",
        sku_name: "Vaseline Intensive Care Deep Moisture",
        confidence: 0.96,
        status: "CONFIDENT",
        coordinates: { x: 80, y: 140, w: 150, h: 310 },
      },
      {
        box_id: "BOX-102",
        sku_name: "Dove Hair Therapy Daily Shine Shampoo",
        confidence: 0.94,
        status: "CONFIDENT",
        coordinates: { x: 255, y: 145, w: 145, h: 305 },
      },
      {
        box_id: "BOX-103",
        sku_name: "Lakme 9to5 CC Cream 02 Honey",
        confidence: 0.78,
        status: "AMBIGUOUS",
        coordinates: { x: 430, y: 170, w: 135, h: 280 },
      },
      {
        box_id: "BOX-104",
        sku_name: "Ponds Super Light Gel Oil Free Moisturizer",
        confidence: 0.93,
        status: "CONFIDENT",
        coordinates: { x: 590, y: 160, w: 155, h: 290 },
      },
      {
        box_id: "BOX-105",
        sku_name: "Unrecognized Competitor Body Lotion",
        confidence: 0.44,
        status: "UNRECOGNIZED",
        coordinates: { x: 775, y: 150, w: 140, h: 300 },
      },
    ],
    compliance_scorecard: {
      share_of_shelf_pct: { target: 58.5, detected: 54.2, status: "FAILED" },
      toker_compliance: { expected_promos: 2, detected_promos: 2, status: "PASSED" },
      red_line_alignment: { status: "PASSED" },
    },
    pipeline_trace: {
      active_models: ["RT-DETR-v2", "gemini-embedding-2-preview (ScaNN)", "Gemini 3.8 Flash"],
      latency_ms: 1460,
      cost_saved_inr: 1.12,
    },
    review_status: "PENDING",
    review_notes: "",
  },
  {
    audit_id: "AUD-GT-2026-0929-02",
    store_metadata: {
      store_id: "HUL-GT-DEL-1189 (Kirana Shikhar Outlet)",
      channel_type: "GT",
      endpoint_source: "Shikhar",
    },
    media: {
      raw_input_url: "/img/test/test_1956.jpg",
      processed_canvas_url: "/img/test/test_1956.jpg",
    },
    identification_detections: [
      {
        box_id: "BOX-201",
        sku_name: "Clinic Plus Strong & Long Health Shampoo",
        confidence: 0.95,
        status: "CONFIDENT",
        coordinates: { x: 110, y: 180, w: 140, h: 290 },
      },
      {
        box_id: "BOX-202",
        sku_name: "Sunsilk Stunning Black Shine Shampoo",
        confidence: 0.92,
        status: "CONFIDENT",
        coordinates: { x: 280, y: 185, w: 145, h: 285 },
      },
      {
        box_id: "BOX-203",
        sku_name: "Lakme 9to5 CC Cream 04 Almond",
        confidence: 0.74,
        status: "AMBIGUOUS",
        coordinates: { x: 460, y: 210, w: 130, h: 260 },
      },
      {
        box_id: "BOX-204",
        sku_name: "Surf Excel Matic Top Load Liquid",
        confidence: 0.97,
        status: "CONFIDENT",
        coordinates: { x: 620, y: 175, w: 165, h: 310 },
      },
    ],
    compliance_scorecard: {
      share_of_shelf_pct: { target: 50.0, detected: 61.4, status: "PASSED" },
      toker_compliance: { expected_promos: 2, detected_promos: 1, status: "FAILED" },
      red_line_alignment: { status: "FAILED" },
    },
    pipeline_trace: {
      active_models: ["YoloN26", "gemini-embedding-2-preview (ScaNN)", "Gemini 3.5 Flash Lite"],
      latency_ms: 920,
      cost_saved_inr: 0.94,
    },
    review_status: "PENDING",
    review_notes: "",
  },
];

// Reactive State Store
const store = {
  audits: [...FALLBACK_AUDITS],
  activeAuditIndex: 0,
  selectedBoxId: null,
  drawerOpen: false,
  listeners: new Set(),
  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  },
  notify() {
    this.listeners.forEach((fn) => fn(this));
  },
  get currentAudit() {
    return this.audits[this.activeAuditIndex] || this.audits[0];
  },
};

function updateNavHighlight() {
  const isAudit = location.hash.startsWith("#/audit");
  const navLb = document.getElementById("nav-leaderboard");
  const navAud = document.getElementById("nav-audit");
  if (navLb && navAud) {
    navLb.className = isAudit
      ? "px-3 py-1.5 rounded-lg font-medium text-slate-400 hover:text-white hover:bg-slate-800"
      : "px-3 py-1.5 rounded-lg font-medium bg-blue-600 text-white";
    navAud.className = isAudit
      ? "px-3 py-1.5 rounded-lg font-medium bg-blue-600 text-white"
      : "px-3 py-1.5 rounded-lg font-medium text-slate-400 hover:text-white hover:bg-slate-800";
  }
}

window.addEventListener("hashchange", route);
route();

function route() {
  updateNavHighlight();
  const runMatch = location.hash.match(/^#\/run\/([^/]+)/);
  if (runMatch) {
    showRun(decodeURIComponent(runMatch[1]));
  } else if (location.hash.startsWith("#/audit")) {
    loadAndShowAuditWorkspace();
  } else {
    showLeaderboard();
  }
}

// ============================================================================
// VIEW 1: DECISION-FIRST EPIC AUDIT WORKSPACE (#/audit)
// ============================================================================
async function loadAndShowAuditWorkspace() {
  try {
    const remote = await getJSON("/api/v1/audits");
    if (Array.isArray(remote) && remote.length > 0) {
      store.audits = remote;
      if (store.activeAuditIndex >= store.audits.length) store.activeAuditIndex = 0;
    }
  } catch (_) {
    // Offline fallback remains active
  }
  renderAuditWorkspace();
}

function renderAuditWorkspace() {
  const audit = store.currentAudit;
  if (!audit) return;

  const sc = audit.compliance_scorecard;
  const sosFailed = sc.share_of_shelf_pct.status === "FAILED";
  const tokerFailed = sc.toker_compliance.status === "FAILED";
  const redLineFailed = sc.red_line_alignment.status === "FAILED";

  const statusBadgeColors = {
    PENDING: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    APPROVED: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
    FLAGGED_FOR_AUDIT: "bg-orange-500/20 text-orange-300 border-orange-500/40",
  };

  const selectedBox = audit.identification_detections.find((b) => b.box_id === store.selectedBoxId);

  app.innerHTML = `
    <!-- Top Audit Selector & Store Metadata Bar -->
    <div class="flex flex-wrap items-center justify-between gap-4 mb-5 bg-slate-900 border border-slate-800 rounded-xl p-4">
      <div class="flex flex-wrap items-center gap-3">
        <label class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Scenario</label>
        <select id="audit-scenario-select" class="bg-slate-950 border border-slate-700 text-slate-100 text-sm rounded-lg px-3 py-1.5 font-mono">
          ${store.audits
            .map(
              (a, idx) => `
            <option value="${idx}" ${idx === store.activeAuditIndex ? "selected" : ""}>
              [${esc(a.store_metadata.channel_type)}] ${esc(a.audit_id)} · ${esc(a.store_metadata.endpoint_source)}
            </option>`
            )
            .join("")}
        </select>
        <span class="text-xs px-2.5 py-1 rounded-md bg-slate-800 text-slate-200 font-mono">
          Store: ${esc(audit.store_metadata.store_id)}
        </span>
        <span class="text-xs px-2.5 py-1 rounded-md bg-blue-500/15 text-blue-300 border border-blue-500/30 font-semibold">
          Channel: ${esc(audit.store_metadata.channel_type)} (${audit.store_metadata.channel_type === "MT" ? "Modern Trade" : "General Trade"})
        </span>
        <span class="text-xs px-2.5 py-1 rounded-md bg-purple-500/15 text-purple-300 border border-purple-500/30 font-semibold">
          Endpoint [E]: ${esc(audit.store_metadata.endpoint_source)}
        </span>
      </div>
      <div class="flex items-center gap-3">
        <span class="text-xs uppercase tracking-wider px-3 py-1 rounded-full border font-semibold ${
          statusBadgeColors[audit.review_status] || statusBadgeColors.PENDING
        }">
          Status: ${esc(audit.review_status)}
        </span>
        <button id="toggle-drawer-btn" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-mono transition">
          [P] Pipeline Telemetry (${audit.pipeline_trace.latency_ms}ms · ₹${Number(audit.pipeline_trace.cost_saved_inr).toFixed(2)} saved)
        </button>
      </div>
    </div>

    <!-- B. Compliance Scorecard (Center Workspace) -->
    <div class="mb-5">
      <div class="flex items-center justify-between mb-2">
        <h2 class="!m-0 text-xs uppercase tracking-wider text-slate-400 font-semibold">
          [C] Business Compliance Scorecard · Decision-First Audit Rules
        </h2>
        <span class="text-xs text-slate-400">Orange border indicates failed SLA threshold requiring reviewer attention</span>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <!-- Card 1: Share of Shelf -->
        <div class="kpi-card ${sosFailed ? "failed" : ""}">
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Share of Shelf (SOS %)</span>
            <span class="text-xs font-bold px-2 py-0.5 rounded ${
              sosFailed ? "bg-orange-500/20 text-orange-400" : "bg-emerald-500/20 text-emerald-400"
            }">${esc(sc.share_of_shelf_pct.status)}</span>
          </div>
          <div class="flex items-baseline gap-3 mt-1">
            <span class="text-2xl font-bold tabular-nums text-white">${Number(sc.share_of_shelf_pct.detected).toFixed(1)}%</span>
            <span class="text-xs text-slate-400">Detected vs Target <strong class="text-slate-200">${Number(sc.share_of_shelf_pct.target).toFixed(1)}%</strong></span>
          </div>
        </div>

        <!-- Card 2: Promo Asset (Toker) Presence -->
        <div class="kpi-card ${tokerFailed ? "failed" : ""}">
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Promo Asset (Toker) Presence</span>
            <span class="text-xs font-bold px-2 py-0.5 rounded ${
              tokerFailed ? "bg-orange-500/20 text-orange-400" : "bg-emerald-500/20 text-emerald-400"
            }">${esc(sc.toker_compliance.status)}</span>
          </div>
          <div class="flex items-baseline gap-3 mt-1">
            <span class="text-2xl font-bold tabular-nums text-white">${sc.toker_compliance.detected_promos} / ${sc.toker_compliance.expected_promos}</span>
            <span class="text-xs text-slate-400">Detected vs Expected Promo Headers</span>
          </div>
        </div>

        <!-- Card 3: Eye-Level (Red Line) Alignment -->
        <div class="kpi-card ${redLineFailed ? "failed" : ""}">
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Eye-Level (Red Line) Alignment</span>
            <span class="text-xs font-bold px-2 py-0.5 rounded ${
              redLineFailed ? "bg-orange-500/20 text-orange-400" : "bg-emerald-500/20 text-emerald-400"
            }">${esc(sc.red_line_alignment.status)}</span>
          </div>
          <div class="flex items-baseline gap-3 mt-1">
            <span class="text-2xl font-bold text-white">${redLineFailed ? "Misaligned" : "Aligned"}</span>
            <span class="text-xs text-slate-400">Golden Zone (1.2m–1.5m) Brand-Block Contiguity</span>
          </div>
        </div>
      </div>
    </div>

    <!-- A. Dual-Viewport Split Screen -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
      <!-- Left Panel: Endpoint Viewport [E] -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
        <div class="flex items-center justify-between mb-3">
          <div>
            <span class="text-xs font-semibold uppercase tracking-wider text-blue-400">[E] Endpoint Viewport</span>
            <h3 class="text-sm font-semibold text-white">Raw Mobile Capture (${esc(audit.store_metadata.endpoint_source)})</h3>
          </div>
          <span class="text-xs font-mono text-slate-400">${esc(audit.media.raw_input_url)}</span>
        </div>
        <div class="relative w-full overflow-hidden rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-center min-h-[360px]">
          <img id="raw-viewport-img" src="${esc(audit.media.raw_input_url)}" alt="Raw Shelf Capture" class="w-full h-auto block object-contain max-h-[540px]" />
        </div>
      </div>

      <!-- Right Panel: Identification Canvas [I] -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <span class="text-xs font-semibold uppercase tracking-wider text-emerald-400">[I] Identification Canvas</span>
            <h3 class="text-sm font-semibold text-white">Interactive SVG Overlay (Click Yellow AMBIGUOUS box to resolve shade)</h3>
          </div>
          <div class="flex items-center gap-3 text-xs">
            <span class="inline-flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block"></span> CONFIDENT</span>
            <span class="inline-flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-sm bg-yellow-400 inline-block"></span> AMBIGUOUS</span>
            <span class="inline-flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-sm bg-red-500 inline-block"></span> UNRECOGNIZED</span>
          </div>
        </div>

        <div id="interactive-canvas-container" class="relative w-full overflow-hidden rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-center min-h-[360px]">
          <div class="relative inline-block w-full">
            <img id="processed-viewport-img" src="${esc(audit.media.processed_canvas_url)}" alt="Processed Shelf Canvas" class="w-full h-auto block max-h-[540px] object-contain" />
            <svg id="bbox-svg-overlay" viewBox="0 0 1000 750" preserveAspectRatio="none" class="absolute inset-0 w-full h-full pointer-events-auto">
              ${audit.identification_detections
                .map((det) => {
                  const { x, y, w, h } = det.coordinates;
                  const cls =
                    det.status === "CONFIDENT"
                      ? "bbox-confident"
                      : det.status === "AMBIGUOUS"
                      ? "bbox-ambiguous"
                      : "bbox-unrecognized";
                  const strokeColor =
                    det.status === "CONFIDENT" ? "#22c55e" : det.status === "AMBIGUOUS" ? "#facc15" : "#ef4444";
                  return `
                    <g data-box-id="${esc(det.box_id)}" class="bbox-group">
                      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" stroke-width="4" class="${cls}" />
                      <rect x="${x}" y="${Math.max(0, y - 28)}" width="${Math.min(280, Math.max(120, w))}" height="24" rx="4" fill="${strokeColor}" />
                      <text x="${x + 6}" y="${Math.max(16, y - 11)}" fill="#020617" font-size="13" font-weight="700" font-family="monospace">
                        ${esc(det.box_id)} · ${Math.round(det.confidence * 100)}%
                      </text>
                    </g>`;
                })
                .join("")}
            </svg>
          </div>
        </div>

        <!-- Ambiguous Sister-Shade Disambiguation Dropdown Bar -->
        ${
          selectedBox
            ? `
          <div class="mt-3 p-3 rounded-lg bg-slate-950 border border-yellow-500/50 flex flex-wrap items-center justify-between gap-3">
            <div>
              <span class="text-xs font-mono text-yellow-400 font-semibold">${esc(selectedBox.box_id)} (${esc(selectedBox.status)} · ${(selectedBox.confidence * 100).toFixed(1)}%)</span>
              <p class="text-xs text-slate-300 mt-0.5">Current SKU: <strong>${esc(selectedBox.sku_name)}</strong></p>
            </div>
            <div class="flex items-center gap-2">
              <select id="shade-quicklist-select" class="bg-slate-900 border border-slate-700 text-slate-100 text-xs rounded-lg px-2.5 py-1.5">
                ${SISTER_SHADE_QUICKLIST.map(
                  (sku) => `<option value="${esc(sku)}" ${sku === selectedBox.sku_name ? "selected" : ""}>${esc(sku)}</option>`
                ).join("")}
              </select>
              <button id="apply-shade-btn" class="text-xs px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold">
                Confirm Shade
              </button>
            </div>
          </div>`
            : `<div class="mt-2 text-xs text-slate-400">Tip: Click any bounding box above (especially yellow <span class="text-yellow-400 font-semibold">AMBIGUOUS</span> sister shades) to reassign or confirm its SKU.</div>`
        }
      </div>
    </div>

    <!-- C. Review Action Footer -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
      <div class="flex-1">
        <label for="audit-feedback-input" class="block text-xs uppercase tracking-wider text-slate-400 font-semibold mb-1">
          Reviewer Audit Feedback / Rejection Notes
        </label>
        <input
          id="audit-feedback-input"
          type="text"
          value="${esc(audit.review_notes || "")}"
          placeholder="Enter store coaching notes, Toker header discrepancy, or sister-shade override reason..."
          class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500"
        />
      </div>
      <div class="flex items-center gap-3 self-end md:self-center">
        <button
          id="btn-flag-audit"
          class="px-4 py-2.5 rounded-lg bg-orange-600 hover:bg-orange-500 text-white font-semibold text-xs tracking-wider uppercase shadow transition"
        >
          [ FLAGGED FOR AUDIT ]
        </button>
        <button
          id="btn-approve-audit"
          class="px-4 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs tracking-wider uppercase shadow transition"
        >
          [ APPROVE AUDIT ]
        </button>
      </div>
    </div>

    <!-- D. Collapsible Pipeline Drawer (Bottom Right) -->
    ${
      store.drawerOpen
        ? `
      <div class="fixed bottom-4 right-4 z-40 w-full max-w-xl bg-slate-900 border border-slate-700 rounded-xl shadow-2xl p-4 max-h-[75vh] flex flex-col">
        <div class="flex items-center justify-between pb-3 border-b border-slate-800">
          <div>
            <span class="text-xs uppercase tracking-wider text-blue-400 font-semibold">[P] Pipeline Telemetry Drawer</span>
            <h4 class="text-sm font-semibold text-white">Model Execution Cascade & FinOps Ledger</h4>
          </div>
          <button id="close-drawer-btn" class="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300">Close ✕</button>
        </div>
        <div class="py-3 grid grid-cols-3 gap-3 border-b border-slate-800 text-xs">
          <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <div class="text-slate-400">Active Cascade</div>
            <div class="font-semibold text-slate-100 mt-0.5">${esc(audit.pipeline_trace.active_models.join(" → "))}</div>
          </div>
          <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <div class="text-slate-400">End-to-End Latency</div>
            <div class="font-bold text-emerald-400 text-base mt-0.5">${audit.pipeline_trace.latency_ms} ms</div>
          </div>
          <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <div class="text-slate-400">Cost Saved vs 1-Pass VLM</div>
            <div class="font-bold text-blue-400 text-base mt-0.5">₹${Number(audit.pipeline_trace.cost_saved_inr).toFixed(3)}</div>
          </div>
        </div>
        <div class="mt-3 flex-1 overflow-auto">
          <div class="text-xs text-slate-400 mb-1 font-mono">Raw EPIC State Payload Tree:</div>
          <pre class="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-300 overflow-x-auto">${esc(
            JSON.stringify(audit, null, 2)
          )}</pre>
        </div>
      </div>`
        : ""
    }
  `;

  // Wire Event Listeners
  document.getElementById("audit-scenario-select")?.addEventListener("change", (e) => {
    store.activeAuditIndex = Number(e.target.value);
    store.selectedBoxId = null;
    renderAuditWorkspace();
  });

  document.getElementById("toggle-drawer-btn")?.addEventListener("click", () => {
    store.drawerOpen = !store.drawerOpen;
    renderAuditWorkspace();
  });

  document.getElementById("close-drawer-btn")?.addEventListener("click", () => {
    store.drawerOpen = false;
    renderAuditWorkspace();
  });

  document.querySelectorAll(".bbox-group").forEach((g) => {
    g.addEventListener("click", () => {
      store.selectedBoxId = g.getAttribute("data-box-id");
      renderAuditWorkspace();
    });
  });

  document.getElementById("apply-shade-btn")?.addEventListener("click", () => {
    const sel = document.getElementById("shade-quicklist-select");
    if (sel && selectedBox) {
      selectedBox.sku_name = sel.value;
      selectedBox.status = "CONFIDENT";
      selectedBox.confidence = 0.99;
      renderAuditWorkspace();
    }
  });

  const submitReview = async (newStatus) => {
    const notesInput = document.getElementById("audit-feedback-input");
    if (notesInput) audit.review_notes = notesInput.value;
    audit.review_status = newStatus;
    try {
      await fetch(`/api/v1/audits/${encodeURIComponent(audit.audit_id)}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          review_status: newStatus,
          review_notes: audit.review_notes,
          identification_detections: audit.identification_detections,
        }),
      });
    } catch (_) {
      // Offline state updated in memory
    }
    renderAuditWorkspace();
  };

  document.getElementById("btn-flag-audit")?.addEventListener("click", () => submitReview("FLAGGED_FOR_AUDIT"));
  document.getElementById("btn-approve-audit")?.addEventListener("click", () => submitReview("APPROVED"));
}

// ============================================================================
// VIEW 2: BENCHMARK LEADERBOARD ARENA (#/)
// ============================================================================
async function showLeaderboard() {
  const rows = await getJSON("/api/leaderboard");
  if (!rows.length) {
    app.innerHTML = `<p class="empty">No runs yet.<br><code>shelf-bench run -a single_pass -m gemini-3.8-flash</code></p>`;
    return;
  }
  const sets = new Set(rows.map((r) => `${r.split} · ${r.images} images · seed ${r.seed}`));
  const caption =
    sets.size === 1
      ? `Eval set: ${[...sets][0]}`
      : `Runs across eval sets (${[...sets].join(" / ")}) · click any row for step-by-step trace or open the Decision-First Reviewer above`;
  app.innerHTML = `
    <div class="flex flex-wrap items-center justify-between gap-4 mb-3">
      <p class="muted caption !m-0">${esc(caption)}</p>
      <a href="#/audit" class="text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold">
        Open Decision-First Reviewer &rarr;
      </a>
    </div>
    <table class="board">
      <thead><tr>
        <th>Rank</th><th>Task</th><th>EPIC</th><th>Run ID</th><th>Architecture</th><th>Owner</th>
        <th class="num" title="Correct boxes / (correct + false + missed)">Accuracy</th>
        <th class="num" title="Share of real products found">Recall</th>
        <th class="num" title="Blend of precision and recall that weights recall 2x. Ranking metric.">F2 ▾</th>
        <th class="num" title="95% of images finished within this time">p95</th>
        <th class="num" title="Gemini + Cloud Run cost per image">Cost / img</th>
      </tr></thead>
      <tbody>${rows
        .map(
          (r) => `
        <tr onclick="location.hash='#/run/${encodeURIComponent(r.run_id)}'">
          <td class="rank">${r.rank}</td>
          <td class="mono">${esc(r.task || "detection")}</td>
          <td>${esc(r.epic || "")}</td>
          <td class="mono">${esc(r.run_id)}${
            r.errors ? ` <span class="warn" title="images that errored">${r.errors} err</span>` : ""
          }</td>
          <td>${esc(r.architecture)}</td>
          <td>${esc(r.owner)}</td>
          <td class="num">${pct(r.accuracy)}</td>
          <td class="num">${pct(r.recall)}</td>
          <td class="num strong">${pct(r.f2)}</td>
          <td class="num">${sec(r.p95_latency_s)}</td>
          <td class="num">${inr(r.cost_per_image_inr)}</td>
        </tr>`
        )
        .join("")}
      </tbody>
    </table>`;
}

// ============================================================================
// VIEW 3: RUN STEP-BY-STEP TRACE (#/run/<id>)
// ============================================================================
async function showRun(runId) {
  const { summary: s, images } = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  app.innerHTML = `
    <a href="#/" class="back">&larr; Leaderboard</a>
    <h1 class="mono">${esc(s.run_id)}</h1>
    <p class="muted">${esc(s.architecture)} · ${esc(s.owner)} · ${s.images} ${esc(s.split)} images · precision ${pct(s.precision)}</p>
    <p class="muted">${envLine(s)}</p>
    ${telemetryLinks(s.telemetry)}
    <div class="stats">
      ${stat("Accuracy", pct(s.accuracy))}${stat("Recall", pct(s.recall))}${stat("F2", pct(s.f2))}
      ${stat("p95", sec(s.p95_latency_s))}${stat("p99", sec(s.p99_latency_s))}${stat("Cost / img", inr(s.cost_per_image_inr))}
    </div>
    <h2>Pipeline</h2>
    <ol class="pipeline">${(s.steps || []).map((t) => `<li>${esc(t)}</li>`).join("")}</ol>
    <div class="split">
      <div class="images">
        <table class="small">
          <thead><tr><th>Image</th><th class="num">GT</th><th class="num">Pred</th><th class="num">F2</th><th class="num">Latency</th></tr></thead>
          <tbody>${images
            .map(
              (r) => `
            <tr data-id="${esc(r.image_id)}">
              <td class="mono">${esc(r.image_id)}${r.error ? ' <span class="warn">err</span>' : ""}</td>
              <td class="num">${r.gt_count}</td><td class="num">${r.pred_count}</td>
              <td class="num">${pct(r.f2)}</td><td class="num">${sec(r.latency_s)}</td>
            </tr>`
            )
            .join("")}
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

function telemetryLinks(t) {
  if (!t) return "";
  const a = (href, text) => `<a href="${esc(href)}" target="_blank" rel="noopener" class="text-blue-400 underline">${text}</a>`;
  const parts = [a(t.trace_url, "Trace"), a(t.logs_url, "Logs")];
  if (t.task_logs_url) parts.push(a(t.task_logs_url, "Cloud Run task logs"));
  return `<p class="muted">${parts.join(" · ")} <span class="mono">${esc(t.trace_id)}</span></p>`;
}

function envLine(s) {
  const e = s.environment || { platform: "local" };
  const c = s.cost || {};
  const r = (usd) => inr((usd || 0) * s.usd_to_inr);
  const where =
    e.platform === "cloud-run"
      ? `Ran on Cloud Run (${esc(e.region)}, ${e.cpu} vCPU / ${e.memory_gib} GiB)`
      : "Ran locally (compute not priced)";
  const credit = c.gemini_credit_usd_per_image
    ? ` (list ${r(c.gemini_list_usd_per_image)} - promo credit ${r(c.gemini_credit_usd_per_image)})`
    : "";
  const compute =
    e.platform === "cloud-run"
      ? ` + Cloud Run ${r(c.compute_usd_per_image)}${/provisional/.test(c.compute_source || "") ? " (provisional)" : ""}`
      : "";
  const storage = c.storage_usd_per_image ? ` + storage ${r(c.storage_usd_per_image)}` : "";
  const services = c.services_usd_per_image ? ` + other APIs ${r(c.services_usd_per_image)}` : "";
  const served = Object.entries(s.traffic || {})
    .map(([k, v]) => `${v} ${esc(k)}`)
    .join(", ");
  const src = s.pricing
    ? `Prices: Cloud Billing Catalog, ${esc((s.pricing.fetched_at || "").slice(0, 10))}, ₹${Number(s.usd_to_inr).toFixed(2)}/USD`
    : "";
  return (
    `${where} · cost/img = Gemini ${r(c.gemini_net_usd_per_image)}${credit}${compute}${storage}${services}` +
    (served ? `<br>Gemini calls served: ${served}` : "") +
    (src ? ` · ${src}` : "")
  );
}

async function showImage(runId, imageId) {
  const v = document.getElementById("viewer");
  const d = await getJSON(`/api/runs/${encodeURIComponent(runId)}/images/${encodeURIComponent(imageId)}`);
  const last = d.steps.length - 1;
  const dur = (ms) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);
  v.innerHTML = `
    <div class="viewer-grid">
      <div>
        <div class="canvas-wrap"><canvas></canvas></div>
        <div class="legend" id="legend"></div>
      </div>
      <div>
        <div class="muted hint">Steps for ${esc(imageId)}. Click one to see what it produced.</div>
        ${telemetryLinks(d.telemetry)}
        ${d.error ? `<p class="warn">${esc(d.error)}</p>` : ""}
        <ol class="steps">${d.steps
          .map(
            (st, i) => `
          <li data-i="${i}"><b>${esc(st.name)}</b> <span class="muted">${dur(st.ms)}</span><br><span class="detail">${esc(
              st.detail
            )}</span></li>`
          )
          .join("")}
        </ol>
      </div>
    </div>`;
  const img = new Image();
  img.src = `/img/${d.split}/${encodeURIComponent(imageId)}`;
  await img.decode();
  const steps = v.querySelector(".steps");
  const select = (i) => {
    steps.querySelectorAll("li").forEach((li) => li.classList.toggle("sel", +li.dataset.i === i));
    draw(v.querySelector("canvas"), img, d, i === last ? null : d.steps[i]);
  };
  steps.addEventListener("click", (e) => {
    const li = e.target.closest("li");
    if (li) select(+li.dataset.i);
  });
  select(last);
}

function draw(canvas, img, d, step) {
  const maxW = Math.min(img.naturalWidth, (canvas.parentElement.clientWidth || 700) - 16);
  const k = Math.min(maxW / d.width, (window.innerHeight * 0.75) / d.height);
  canvas.width = d.width * k;
  canvas.height = d.height * k;
  const g = canvas.getContext("2d");
  g.drawImage(img, 0, 0, canvas.width, canvas.height);
  const rect = (b, color, w = 1.5) => {
    g.strokeStyle = color;
    g.lineWidth = w;
    g.strokeRect(b[0] * k, b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k);
  };
  const legend = document.getElementById("legend");
  if (!step) {
    const tp = new Set(d.matched);
    d.gt.forEach((b) => rect(b, "rgba(34,197,94,.9)"));
    d.preds.forEach((b, i) => rect(b, tp.has(i) ? "rgba(59,130,246,.95)" : "rgba(239,68,68,.95)"));
    legend.innerHTML = `<span class="sw gt"></span>ground truth (${d.gt_count}) <span class="sw tp"></span>correct (${d.tp}) <span class="sw fp"></span>false (${d.fp}) · missed ${d.fn}`;
  } else {
    (step.regions || []).forEach((b) => rect(b, "rgba(250,204,21,.95)", 3));
    (step.boxes || []).forEach((b) => rect(b, "rgba(59,130,246,.95)"));
    const parts = [];
    if (step.regions) parts.push(`<span class="sw tile"></span>tiles (${step.regions.length})`);
    if (step.boxes) parts.push(`<span class="sw tp"></span>boxes (${step.boxes.length})`);
    legend.innerHTML = parts.join(" ") || '<span class="muted">nothing to overlay for this step</span>';
  }
}
