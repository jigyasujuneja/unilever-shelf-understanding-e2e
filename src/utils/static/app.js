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
const RPC = "RPC checkout photos (Retail Product Checkout: 200 products in 17 categories, 4 reference photos each).";
const TASKS = {
  detection: {
    label: "Detection",
    blurb: "SKU-110K shelf photos: find every product box (IoU ≥ 0.5). SKU-110K has no product labels.",
    cols: [
      ["Accuracy", "Correct boxes / (correct + false + missed)", (r) => pct(r.accuracy)],
      ["Recall", "Share of real products found", (r) => pct(r.recall)],
    ],
    score: F2_SCORE,
  },
  classification: {
    label: "Classification",
    blurb: "Labelled product photos (kierth/retail-products-philippines): each photo shows one product, to be picked from a 184-product catalog.",
    cols: [fieldCol("brand", "photos"), fieldCol("category", "photos")],
    score: PRODUCT_SCORE("photos"),
  },
  retrieval: {
    label: "Retrieval",
    blurb: RPC + " Every ground-truth product box is cut out of the photo and must be matched to the right product's reference photos (image to image).",
    cols: [fieldCol("category", "product boxes")],
    score: PRODUCT_SCORE("product boxes"),
  },
  end_to_end: {
    label: "End-to-end",
    blurb: "Find every product and name it: a box counts only if it overlaps a real product (IoU ≥ 0.5) and names that product.",
    cols: [
      ["Found", "Share of real products boxed at all (IoU ≥ 0.5), whatever the name", (r) => pct(r.found_recall ?? 0)],
      ["Recall", "Share of real products boxed and named correctly", (r) => pct(r.recall)],
      ["Precision", "Share of predicted boxes that are a real product, named correctly", (r) => pct(r.precision)],
    ],
    score: F2_SCORE,
  },
};
const DATASETS = {
  shelves: {
    label: "Shelves (HoloSelecta)",
    blurb: "HoloSelecta retail & vending-machine shelves (115 products with GTINs; reference crops from other sessions' photos).",
  },
  rpc: {
    label: "Checkout counter (RPC)",
    blurb: RPC,
  },
};
const ORDER = Object.keys(TASKS);
const DS_ORDER = ["shelves", "rpc", "sku110k", "products"];
const byOrder = (a, b) => (ORDER.indexOf(a) + 1 || 99) - (ORDER.indexOf(b) + 1 || 99);
const byDsOrder = (a, b) => (DS_ORDER.indexOf(a) + 1 || 99) - (DS_ORDER.indexOf(b) + 1 || 99);
const taskOf = (r) => r.task || "detection";
// Leaderboard sections (Approach.use_case); runs from before sections existed are market share.
const USE_CASES = {
  market_share: { label: "Market share", blurb: "Which products are on the shelf and how many: find, identify and count them." },
  merchandising: { label: "Merchandising", blurb: "How products are displayed: planogram compliance, shelf position, promo material, price tags." },
};
const useCaseOf = (r) => r.use_case || "market_share";
const infoOf = (t) => TASKS[t] || { ...TASKS.detection, label: t, blurb: "" };
// Per-box identifications: [{box, pred, gt, correct}] (older product runs stored one dict).
const labelsOf = (r) => (!r.labels ? [] : Array.isArray(r.labels) ? r.labels : [r.labels]);

window.addEventListener("hashchange", route);
route();

function route() {
  const run = location.hash.match(/^#\/run\/([^/]+)/);
  if (run) return showRun(decodeURIComponent(run[1]));
  // #/board/<use case>/<task>; #/board/<task> (older links) means market share.
  const b = location.hash.match(/^#\/board\/([^/]+)(?:\/([^/]+))?/);
  const parts = b ? [b[1], b[2]].filter(Boolean).map(decodeURIComponent) : [];
  if (parts.length && !USE_CASES[parts[0]]) parts.unshift("market_share");
  const task = parts[1] === "shelves_end_to_end" ? "end_to_end" : parts[1];
  showLeaderboard(parts[0] || "market_share", task);
}

// ---------------------------------------------------------------- leaderboard
async function showLeaderboard(useCase, task) {
  const every = await getJSON("/api/leaderboard");
  const sections = `<nav class="sections">${Object.entries(USE_CASES).map(([k, u]) => `
    <a class="section${k === useCase ? " on" : ""}" href="#/board/${k}">${esc(u.label)}
      <span class="count">${every.filter((r) => useCaseOf(r) === k).length}</span></a>`).join("")}</nav>
    <p class="muted caption">${esc(USE_CASES[useCase].blurb)}</p>`;
  const all = every.filter((r) => useCaseOf(r) === useCase);
  if (!all.length) {
    app.innerHTML = sections + (useCase === "merchandising"
      ? `<p class="empty">No merchandising benchmarks yet.<br>Approaches with <code>use_case = "merchandising"</code> will be ranked here.</p>`
      : `<p class="empty">No runs yet.<br><code>shelf-bench run -a single_pass -m gemini-3.8-flash</code></p>`);
    return;
  }
  const tasks = [...new Set(all.map(taskOf))].sort(byOrder);
  task = tasks.includes(task) ? task : tasks[0];
  const info = infoOf(task);
  const rows = all.filter((r) => taskOf(r) === task);
  const datasets = [...new Set(rows.map((r) => r.dataset || "sku110k"))].sort(byDsOrder);
  const tabs = tasks.map((t) => `
    <a class="tab${t === task ? " on" : ""}" href="#/board/${useCase}/${encodeURIComponent(t)}">
      ${esc(infoOf(t).label)} <span class="count">${all.filter((r) => taskOf(r) === t).length}</span></a>`).join("");
  const renderDatasetBlock = (ds) => {
    const dsRows = rows.filter((r) => (r.dataset || "sku110k") === ds);
    const ranked = dsRows.filter((r) => r.rank !== "dev");
    const sets = new Set(ranked.map((r) => `${r.split} · ${r.images} images · seed ${r.seed}`));
    const caption = !ranked.length
      ? "No ranked runs yet (only Cloud Run runs on the leaderboard image set are ranked)"
      : sets.size === 1
        ? `Eval set: ${[...sets][0]}`
        : `⚠ Runs use different eval sets (${[...sets].join(" / ")}) - compare with care.`;
    const dsMeta = DATASETS[ds];
    const hdr = datasets.length > 1 && dsMeta
      ? `<h2 class="ds-heading">${esc(dsMeta.label)} <span class="count">${dsRows.length}</span></h2>
         <p class="muted caption">${esc(dsMeta.blurb)}<br>${esc(caption)} · <i>dev</i> = local run or other image set, not ranked · click a row for per-image results</p>`
      : `<p class="muted caption">${esc(info.blurb)}<br>${esc(caption)} · <i>dev</i> = local run or other image set, not ranked · click a row for per-image results</p>`;
    return hdr + `<table class="board">
      <thead><tr>
        <th>Rank</th><th>Run ID</th><th>Architecture</th><th>Owner</th>
        ${info.cols.map(([h, tip]) => `<th class="num" title="${esc(tip)}">${h}</th>`).join("")}
        <th class="num" title="${esc(info.score[1])}">${info.score[0]}</th>
        <th class="num" title="95% of images finished within this time">p95</th>
        <th class="num" title="99% of images finished within this time">p99</th>
        <th class="num" title="Model + other API + Cloud Run + storage cost per image">Cost / img</th>
      </tr></thead>
      <tbody>${dsRows.map((r) => `
        <tr onclick="location.hash='#/run/${encodeURIComponent(r.run_id)}'">
          <td class="rank">${r.rank}</td>
          <td class="mono">${esc(r.run_id)}${r.errors ? ` <span class="warn" title="images that errored">${r.errors} err</span>` : ""}</td>
          <td>${esc(r.architecture)}</td>
          <td>${esc(r.owner)}</td>
          ${info.cols.map(([, , cell]) => `<td class="num">${cell(r)}</td>`).join("")}
          <td class="num strong">${pct(r[info.score[2]])}</td>
          <td class="num">${sec(r.p95_latency_s)}</td>
          <td class="num">${sec(r.p99_latency_s)}</td>
          <td class="num">${inr(r.cost_per_image_inr)}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;
  };
  app.innerHTML = sections + `
    <nav class="tabs">${tabs}</nav>
    ${datasets.length > 1 ? `<p class="muted caption">${esc(info.blurb)}</p>` : ""}
    ${datasets.map(renderDatasetBlock).join("")}`;
}

// ---------------------------------------------------------------- run detail
async function showRun(runId) {
  const { summary: s, images } = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  const task = s.task || "detection";
  const single = task === "classification"; // one product per photo
  const fa = s.field_accuracy || {};
  const stats = {
    detection: () => stat("Accuracy", pct(s.accuracy)) + stat("Recall", pct(s.recall)) + stat("F2", pct(s.f2)),
    end_to_end: () => stat("Found", pct(s.found_recall ?? 0)) + stat("Recall", pct(s.recall)) +
      stat("Precision", pct(s.precision)) + stat("F2", pct(s.f2)),
  }[task]?.() ?? Object.entries({ product: s.accuracy, ...fa })
    .map(([f, v]) => stat(f[0].toUpperCase() + f.slice(1), pct(v))).join("");
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
    end_to_end: ["<th class=num>Products</th><th class=num>Boxes</th><th class=num>Right</th><th class=num>F2</th>",
      (r) => num(r.gt_count) + num(r.pred_count) + num(right(r)) + num(pct(r.f2))],
  }[task] || [];
  const row = (r) => `<td class="mono">${esc(r.image_id)}${r.error ? ' <span class="warn">err</span>' : ""}</td>` +
    (cells ? cells(r) : "") + num(sec(r.latency_s));
  const setupNote = s.cost?.setup_usd ? ` · one-off setup ${inr(s.cost.setup_usd * s.usd_to_inr)} (${s.cost.setup_seconds}s), not in cost/img` : "";
  app.innerHTML = `
    <a href="#/board/${useCaseOf(s)}/${encodeURIComponent(task)}" class="back">&larr; Leaderboard</a>
    <h1 class="mono">${esc(s.run_id)}</h1>
    <p class="muted">${esc(s.architecture)} · ${esc(s.owner)} · ${s.images} ${esc(s.dataset || "sku110k")} ${esc(s.split)} images · precision ${pct(s.precision)}</p>
    <p class="muted">${envLine(s)}${setupNote}</p>
    ${telemetryLinks(s.telemetry)}
    <div class="stats">
      ${stats}
      ${stat("p95", sec(s.p95_latency_s))}${stat("p99", sec(s.p99_latency_s))}${stat("Cost / img", inr(s.cost_per_image_inr))}
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
  return `${where} · cost/img = ${parts.join(" + ") || r(0)}` +
    (served ? `<br>Gemini calls served: ${served}` : "") + (src ? ` · ${src}` : "");
}

// Predicted vs ground-truth product: field by field for a one-product photo, else one row per
// box (numbered like the boxes on the image), with the RPC reference photo of each product.
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
  const ref = (p) => (p && (d.split === "rpc" || d.split === "shelves") ? `<img class="ref" loading="lazy" src="/img/${d.split}-ref/${encodeURIComponent(p.sku_id)}" alt="">` : "");
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
  const refs = d.split === "rpc" || d.split === "shelves";
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
  const GREEN = "rgba(34,197,94,.95)", RED = "rgba(239,68,68,.95)";
  const legend = document.getElementById("legend");
  const labels = labelsOf(d);
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
      (e2e ? ` <span class="sw gt"></span>ground truth, dashed (${d.gt_count}) · products missed or misnamed ${d.fn}` : "");
  } else if (!step) {
    const tp = new Set(d.matched);
    d.gt.forEach((b) => rect(b, "rgba(34,197,94,.9)"));
    d.preds.forEach((b, i) => rect(b, tp.has(i) ? "rgba(59,130,246,.95)" : "rgba(239,68,68,.95)"));
    legend.innerHTML = `<span class="sw gt"></span>ground truth (${d.gt_count}) <span class="sw tp"></span>correct (${d.tp}) <span class="sw fp"></span>false (${d.fp}) · missed ${d.fn}`;
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
    legend.innerHTML = parts.join(" ") || '<span class="muted">nothing to overlay for this step</span>';
  }
}
