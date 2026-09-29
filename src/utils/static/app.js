// Two views: the leaderboard (#/) and one run's step-by-step results (#/run/<id>).
const app = document.getElementById("app");
const pct = (v) => (v * 100).toFixed(1) + "%";
const sec = (v) => v.toFixed(2) + "s";
const inr = (v) => "₹" + (v > 0 && v < 0.001 ? v.toPrecision(2) : (v ?? 0).toFixed(3));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const getJSON = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)));

window.addEventListener("hashchange", route);
route();

function route() {
  const m = location.hash.match(/^#\/run\/([^/]+)/);
  m ? showRun(decodeURIComponent(m[1])) : showLeaderboard();
}

// ---------------------------------------------------------------- leaderboard
async function showLeaderboard() {
  const rows = await getJSON("/api/leaderboard");
  if (!rows.length) {
    app.innerHTML = `<p class="empty">No runs yet.<br><code>shelf-bench run -a single_pass -m gemini-3.8-flash</code></p>`;
    return;
  }
  const sets = new Set(rows.map((r) => `${r.split} · ${r.images} images · seed ${r.seed}`));
  const caption = sets.size === 1
    ? `Eval set: ${[...sets][0]}`
    : `⚠ Runs use different eval sets (${[...sets].join(" / ")}) - compare with care.`;
  app.innerHTML = `
    <p class="muted caption">${esc(caption)} · click a row for step-by-step results</p>
    <table class="board">
      <thead><tr>
        <th>Rank</th><th>Run ID</th><th>Architecture</th><th>Owner</th>
        <th class="num" title="Correct boxes / (correct + false + missed)">Accuracy</th>
        <th class="num" title="Share of real products found">Recall</th>
        <th class="num" title="Blend of precision and recall that weights recall 2x. Ranking metric.">F2 ▾</th>
        <th class="num" title="95% of images finished within this time">p95</th>
        <th class="num" title="99% of images finished within this time">p99</th>
        <th class="num" title="Gemini token cost per image">Cost / img</th>
      </tr></thead>
      <tbody>${rows.map((r) => `
        <tr onclick="location.hash='#/run/${encodeURIComponent(r.run_id)}'">
          <td class="rank">${r.rank}</td>
          <td class="mono">${esc(r.run_id)}${r.errors ? ` <span class="warn" title="images that errored">${r.errors} err</span>` : ""}</td>
          <td>${esc(r.architecture)}</td>
          <td>${esc(r.owner)}</td>
          <td class="num">${pct(r.accuracy)}</td>
          <td class="num">${pct(r.recall)}</td>
          <td class="num strong">${pct(r.f2)}</td>
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
    <ol class="pipeline">${s.steps.map((t) => `<li>${esc(t)}</li>`).join("")}</ol>
    <div class="split">
      <div class="images">
        <table class="small">
          <thead><tr><th>Image</th><th class="num">GT</th><th class="num">Pred</th><th class="num">F2</th><th class="num">Latency</th></tr></thead>
          <tbody>${images.map((r) => `
            <tr data-id="${esc(r.image_id)}">
              <td class="mono">${esc(r.image_id)}${r.error ? ' <span class="warn">err</span>' : ""}</td>
              <td class="num">${r.gt_count}</td><td class="num">${r.pred_count}</td>
              <td class="num">${pct(r.f2)}</td><td class="num">${sec(r.latency_s)}</td>
            </tr>`).join("")}
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
  const compute = e.platform === "cloud-run"
    ? ` + Cloud Run ${r(c.compute_usd_per_image)}${/provisional/.test(c.compute_source || "") ? " (provisional)" : ""}` : "";
  const storage = c.storage_usd_per_image ? ` + storage ${r(c.storage_usd_per_image)}` : "";
  const services = c.services_usd_per_image ? ` + other APIs ${r(c.services_usd_per_image)}` : "";
  const served = Object.entries(s.traffic || {}).map(([k, v]) => `${v} ${esc(k)}`).join(", ");
  const src = s.pricing
    ? `Prices: Cloud Billing Catalog, ${esc((s.pricing.fetched_at || "").slice(0, 10))}, ₹${Number(s.usd_to_inr).toFixed(2)}/USD`
    : "";
  return `${where} · cost/img = Gemini ${r(c.gemini_net_usd_per_image)}${credit}${compute}${storage}${services}` +
    (served ? `<br>Gemini calls served: ${served}` : "") + (src ? ` · ${src}` : "");
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

// Final step: ground truth vs. predictions (TP / FP). Other steps: that step's boxes / tiles.
function draw(canvas, img, d, step) {
  const maxW = Math.min(img.naturalWidth, (canvas.parentElement.clientWidth || 700) - 16);
  const k = Math.min(maxW / d.width, (window.innerHeight * 0.75) / d.height);
  canvas.width = d.width * k;
  canvas.height = d.height * k;
  const g = canvas.getContext("2d");
  g.drawImage(img, 0, 0, canvas.width, canvas.height);
  const rect = (b, color, w = 1.5) => { g.strokeStyle = color; g.lineWidth = w; g.strokeRect(b[0] * k, b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k); };
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
