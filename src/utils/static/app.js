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
    blurb: RPC + " Find every product and name it: a box counts only if it overlaps a real product (IoU ≥ 0.5) and names that product.",
    cols: [
      ["Found", "Share of real products boxed at all (IoU ≥ 0.5), whatever the name", (r) => pct(r.found_recall ?? 0)],
      ["Recall", "Share of real products boxed and named correctly", (r) => pct(r.recall)],
      ["Precision", "Share of predicted boxes that are a real product, named correctly", (r) => pct(r.precision)],
    ],
    score: F2_SCORE,
  },
};
TASKS.shelves_end_to_end = {
  ...TASKS.end_to_end,
  label: "Shelf end-to-end",
  blurb: "HoloSelecta vending-machine shelves (115 products with GTINs; reference crops from other sessions' photos). Find every product and name it: a box counts only if it overlaps a real product (IoU ≥ 0.5) and names that product.",
};
const ORDER = Object.keys(TASKS);
const byOrder = (a, b) => (ORDER.indexOf(a) + 1 || 99) - (ORDER.indexOf(b) + 1 || 99);
// One tab per task and dataset (runner.leaderboard ranks within each).
const taskOf = (r) => (r.dataset === "shelves" ? "shelves_" : "") + (r.task || "detection");
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
  showLeaderboard(parts[0] || "market_share", parts[1]);
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
  const ranked = rows.filter((r) => r.rank !== "dev");
  const sets = new Set(ranked.map((r) => `${r.split} · ${r.images} images · seed ${r.seed}`));
  const caption = !ranked.length
    ? "No ranked runs yet (only Cloud Run runs on the leaderboard image set are ranked)"
    : sets.size === 1
      ? `Eval set: ${[...sets][0]}`
      : `⚠ Runs use different eval sets (${[...sets].join(" / ")}) - compare with care.`;
  const tabs = tasks.map((t) => `
    <a class="tab${t === task ? " on" : ""}" href="#/board/${useCase}/${encodeURIComponent(t)}">
      ${esc(infoOf(t).label)} <span class="count">${all.filter((r) => taskOf(r) === t).length}</span></a>`).join("");
  app.innerHTML = sections + `
    <nav class="tabs">${tabs}</nav>
    <p class="muted caption">${esc(info.blurb)}<br>${esc(caption)} · <i>dev</i> = local run or other image set, not ranked · click a row for per-image results</p>
    <table class="board">
      <thead><tr>
        <th>Rank</th><th>Run ID</th><th>Architecture</th><th>Owner</th>
        ${info.cols.map(([h, tip]) => `<th class="num" title="${esc(tip)}">${h}</th>`).join("")}
        <th class="num" title="${esc(info.score[1])}">${info.score[0]}</th>
        <th class="num" title="95% of images finished within this time">p95</th>
        <th class="num" title="99% of images finished within this time">p99</th>
        <th class="num" title="Model + other API + Cloud Run + storage cost per image">Cost / img</th>
      </tr></thead>
      <tbody>${rows.map((r) => `
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
}

// ---------------------------------------------------------------- run detail
async function showRun(runId) {
  const { summary: s, images } = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  const tab = taskOf(s);
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
    <a href="#/board/${useCaseOf(s)}/${encodeURIComponent(tab)}" class="back">&larr; Leaderboard</a>
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
        ${labelTable(d)}
        <div class="muted hint">Steps for ${esc(imageId)}. Click one to see what it produced.</div>
        ${telemetryLinks(d.telemetry)}
        ${d.error ? `<p class="warn">${esc(d.error)}</p>` : ""}
        <ol class="steps">${d.steps.map((st, i) => `
          <li data-i="${i}"><b>${esc(st.name)}</b> <span class="muted">${dur(st.ms)}</span><br><span class="detail">${esc(st.detail)}</span></li>`).join("")}
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
  steps.addEventListener("click", (e) => { const li = e.target.closest("li"); if (li) select(+li.dataset.i); });
  select(last);
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
    const parts = [];
    if (step.regions) parts.push(`<span class="sw tile"></span>tiles (${step.regions.length})`);
    if (step.boxes) parts.push(`<span class="sw tp"></span>boxes (${step.boxes.length})`);
    legend.innerHTML = parts.join(" ") || '<span class="muted">nothing to overlay for this step</span>';
  }
}
