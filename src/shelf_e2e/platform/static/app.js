/* ShelfBench Arena Frontend Logic (100% Safe DOM Construction — Zero innerHTML/alert) */

(function () {
  'use strict';

  const state = {
    csrfToken: '',
    runs: [],
    groundTruthImages: {},
    imageDimensions: {},
    taxonomy7Dim: {},
    spec005Summary: {},
    selectedRunId: '',
    selectedImage: 'sku110k_val_000.jpg',
  };

  function makeCell(text, className) {
    const td = document.createElement('td');
    td.textContent = String(text);
    if (className) {
      td.className = className;
    }
    return td;
  }

  function makeBadge(text, badgeClass) {
    const span = document.createElement('span');
    span.className = 'badge ' + badgeClass;
    span.textContent = String(text);
    return span;
  }

  function showToast(message) {
    const toast = document.getElementById('status-toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.remove('hidden');
    window.setTimeout(function () {
      toast.classList.add('hidden');
    }, 4000);
  }

  function renderLeaderboard() {
    const tbody = document.getElementById('leaderboard-tbody');
    if (!tbody) return;
    tbody.replaceChildren();

    state.runs.forEach(function (run, idx) {
      const tr = document.createElement('tr');
      tr.appendChild(makeCell('#' + (idx + 1), 'mono-cell'));

      const medalTd = document.createElement('td');
      const medalClass =
        run.medal_tier === 'GOLD'
          ? 'badge-gold'
          : run.medal_tier === 'SILVER'
          ? 'badge-silver'
          : 'badge-fail';
      medalTd.appendChild(makeBadge(run.medal_tier, medalClass));
      tr.appendChild(medalTd);

      tr.appendChild(makeCell(run.run_id, 'mono-cell'));
      tr.appendChild(makeCell(run.track_name));
      tr.appendChild(makeCell(run.engineer_ldap, 'mono-cell'));
      tr.appendChild(makeCell(run.pareto_score.toFixed(2), 'mono-cell'));
      tr.appendChild(
        makeCell((run.public_metrics.top1_acc * 100).toFixed(1) + '%', 'mono-cell')
      );
      tr.appendChild(
        makeCell((run.private_metrics.top1_acc * 100).toFixed(1) + '%', 'mono-cell')
      );
      tr.appendChild(makeCell(run.public_metrics.map_50_95.toFixed(4), 'mono-cell'));
      tr.appendChild(makeCell(run.public_metrics.mrr.toFixed(2), 'mono-cell'));
      tr.appendChild(
        makeCell(run.latency_tiers_ms.p95_525_workers_ms.toFixed(1) + ' ms', 'mono-cell')
      );
      tr.appendChild(
        makeCell('₹' + run.cost_breakdown.cost_per_image_inr.toFixed(4), 'mono-cell')
      );

      const slaTd = document.createElement('td');
      slaTd.appendChild(
        makeBadge(
          run.meets_all_slas ? 'PASS (<= ₹0.22)' : 'SLA VIOLATION',
          run.meets_all_slas ? 'badge-pass' : 'badge-fail'
        )
      );
      tr.appendChild(slaTd);

      tbody.appendChild(tr);
    });
  }

  function renderMLflowTable() {
    const tbody = document.getElementById('mlflow-tbody');
    if (!tbody) return;
    tbody.replaceChildren();

    state.runs.forEach(function (run) {
      const tr = document.createElement('tr');
      tr.appendChild(makeCell(run.run_id, 'mono-cell'));
      tr.appendChild(makeCell(run.timestamp_utc, 'mono-cell'));
      tr.appendChild(makeCell(run.track_id, 'mono-cell'));
      tr.appendChild(makeCell(run.hyperparameters.glare_solver || 'Standard'));
      tr.appendChild(makeCell(run.latency_tiers_ms.tier1_detection.toFixed(1), 'mono-cell'));
      tr.appendChild(makeCell(run.latency_tiers_ms.tier2_catalog_match.toFixed(1), 'mono-cell'));
      tr.appendChild(makeCell(run.latency_tiers_ms.tier3_compliance.toFixed(1), 'mono-cell'));
      tr.appendChild(makeCell(run.latency_tiers_ms.total_e2e.toFixed(1), 'mono-cell'));
      tr.appendChild(
        makeCell('₹' + run.cost_breakdown.daily_500k_cost_inr.toLocaleString(), 'mono-cell')
      );
      tr.appendChild(
        makeCell((run.public_metrics.hallucination_rate * 100).toFixed(1) + '%', 'mono-cell')
      );
      tbody.appendChild(tr);
    });
  }

  function renderParetoSVG() {
    const holder = document.getElementById('pareto-svg-holder');
    if (!holder) return;
    holder.replaceChildren();

    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 760 340');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '340');

    // Axes
    const xAxis = document.createElementNS(ns, 'line');
    xAxis.setAttribute('x1', '70');
    xAxis.setAttribute('y1', '290');
    xAxis.setAttribute('x2', '710');
    xAxis.setAttribute('y2', '290');
    xAxis.setAttribute('stroke', '#94a3b8');
    xAxis.setAttribute('stroke-width', '2');
    svg.appendChild(xAxis);

    const yAxis = document.createElementNS(ns, 'line');
    yAxis.setAttribute('x1', '70');
    yAxis.setAttribute('y1', '30');
    yAxis.setAttribute('x2', '70');
    yAxis.setAttribute('y2', '290');
    yAxis.setAttribute('stroke', '#94a3b8');
    yAxis.setAttribute('stroke-width', '2');
    svg.appendChild(yAxis);

    // Hard Cost Ceiling Vertical Wall at ₹0.22 (max x = ₹0.30)
    const slaX = 70 + (0.22 / 0.30) * 620;
    const slaLine = document.createElementNS(ns, 'line');
    slaLine.setAttribute('x1', String(slaX));
    slaLine.setAttribute('y1', '30');
    slaLine.setAttribute('x2', String(slaX));
    slaLine.setAttribute('y2', '290');
    slaLine.setAttribute('stroke', '#dc2626');
    slaLine.setAttribute('stroke-dasharray', '6,4');
    slaLine.setAttribute('stroke-width', '2');
    svg.appendChild(slaLine);

    const slaLabel = document.createElementNS(ns, 'text');
    slaLabel.setAttribute('x', String(slaX - 135));
    slaLabel.setAttribute('y', '48');
    slaLabel.setAttribute('fill', '#dc2626');
    slaLabel.setAttribute('font-size', '12');
    slaLabel.setAttribute('font-weight', '700');
    slaLabel.textContent = 'Hard SLA Ceiling (₹0.22 / img)';
    svg.appendChild(slaLabel);

    state.runs.forEach(function (run) {
      const cost = Math.min(0.30, run.cost_breakdown.cost_per_image_inr);
      const acc = run.public_metrics.top1_acc;
      const cx = 70 + (cost / 0.30) * 620;
      const cy = 290 - acc * 230;

      const circle = document.createElementNS(ns, 'circle');
      circle.setAttribute('cx', String(cx));
      circle.setAttribute('cy', String(cy));
      circle.setAttribute('r', '8');
      circle.setAttribute('fill', run.meets_all_slas ? '#0d9488' : '#e11d48');
      svg.appendChild(circle);

      const label = document.createElementNS(ns, 'text');
      label.setAttribute('x', String(cx + 12));
      label.setAttribute('y', String(cy + 4));
      label.setAttribute('fill', '#1e293b');
      label.setAttribute('font-size', '11');
      label.setAttribute('font-weight', '600');
      label.textContent =
        run.track_id + ' (₹' + run.cost_breakdown.cost_per_image_inr.toFixed(3) + ', ' + (acc * 100).toFixed(0) + '%)';
      svg.appendChild(label);
    });

    holder.appendChild(svg);
  }

  function populateInspectorRunSelect() {
    const select = document.getElementById('inspector-run-select');
    if (!select) return;
    select.replaceChildren();
    state.runs.forEach(function (run) {
      const opt = document.createElement('option');
      opt.value = run.run_id;
      opt.textContent = run.run_id + ' — ' + run.track_id;
      select.appendChild(opt);
    });
    if (!state.selectedRunId && state.runs.length > 0) {
      state.selectedRunId = state.runs[0].run_id;
    }
    select.value = state.selectedRunId;
  }

  function renderInspector() {
    const imgEl = document.getElementById('inspector-shelf-img');
    const svgOverlay = document.getElementById('inspector-bbox-svg');
    const tbody = document.getElementById('inspector-predictions-tbody');
    if (!imgEl || !svgOverlay || !tbody) return;

    imgEl.src = '/images/' + state.selectedImage;
    const dims = (state.imageDimensions || {})[state.selectedImage] || [2336, 4160];
    svgOverlay.setAttribute('viewBox', '0 0 ' + dims[0] + ' ' + dims[1]);
    svgOverlay.replaceChildren();
    tbody.replaceChildren();

    const activeRun =
      state.runs.find(function (r) {
        return r.run_id === state.selectedRunId;
      }) || state.runs[0];
    if (!activeRun) return;

    const preds = (activeRun.predictions_by_image || {})[state.selectedImage] || [];
    const gts = (state.groundTruthImages || {})[state.selectedImage] || [];
    const ns = 'http://www.w3.org/2000/svg';
    const strokeWidth = dims[0] > 1200 ? '8' : '3';

    preds.forEach(function (p, idx) {
      const gtItem = gts[idx] || {};
      const gtSku = gtItem.gt_base_pack_id || p.base_pack_id;
      const isMatch = p.base_pack_id === gtSku;
      const tax = (state.taxonomy7Dim || {})[p.base_pack_id] || {};
      const isHul = tax.is_hul_brand !== false;

      const rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', String(p.box_xyxy[0]));
      rect.setAttribute('y', String(p.box_xyxy[1]));
      rect.setAttribute('width', String(p.box_xyxy[2] - p.box_xyxy[0]));
      rect.setAttribute('height', String(p.box_xyxy[3] - p.box_xyxy[1]));
      rect.setAttribute('fill', 'none');
      rect.setAttribute('stroke', !isMatch ? '#f43f5e' : isHul ? '#10b981' : '#3b82f6');
      rect.setAttribute('stroke-width', strokeWidth);
      svgOverlay.appendChild(rect);

      const tr = document.createElement('tr');
      tr.appendChild(makeCell('#' + (idx + 1), 'mono-cell'));
      tr.appendChild(makeCell('[' + p.box_xyxy.map(Math.round).join(', ') + ']', 'mono-cell'));
      tr.appendChild(makeCell(p.base_pack_id, 'mono-cell'));
      tr.appendChild(makeCell((tax.brand || 'Dove') + (isHul ? ' (HUL)' : ' (Comp)')));
      tr.appendChild(makeCell(tax.rule_derived_size_bucket || 'Large (>110g/ml)', 'mono-cell'));
      tr.appendChild(makeCell((p.confidence * 100).toFixed(1) + '%', 'mono-cell'));
      const statusTd = document.createElement('td');
      statusTd.appendChild(
        makeBadge(isMatch ? 'MATCH' : 'MISMATCH', isMatch ? 'badge-pass' : 'badge-fail')
      );
      tr.appendChild(statusTd);
      tbody.appendChild(tr);
    });
  }

  function renderSpec005() {
    const tbody = document.getElementById('spec005-taxonomy-tbody');
    if (!tbody) return;
    tbody.replaceChildren();
    Object.keys(state.taxonomy7Dim || {}).forEach(function (skuId) {
      const t = state.taxonomy7Dim[skuId];
      const tr = document.createElement('tr');
      tr.appendChild(makeCell(skuId, 'mono-cell'));
      tr.appendChild(makeCell(t.category || 'Personal Care'));
      tr.appendChild(makeCell(t.subcategory || 'General'));
      tr.appendChild(makeCell(t.brand || 'Dove'));
      const hulTd = document.createElement('td');
      hulTd.appendChild(
        makeBadge(t.is_hul_brand ? 'UNILEVER (HUL)' : 'COMPETITOR', t.is_hul_brand ? 'badge-pass' : 'badge-silver')
      );
      tr.appendChild(hulTd);
      tr.appendChild(makeCell(t.variant || 'Standard'));
      tr.appendChild(makeCell(t.packaging_type || 'bottle', 'mono-cell'));
      tr.appendChild(makeCell(t.pack_type || 'Single', 'mono-cell'));
      tr.appendChild(makeCell(t.rule_derived_size_bucket || 'Large / Family (>110g/ml)', 'mono-cell'));
      tbody.appendChild(tr);
    });
  }

  function refreshAllViews() {
    renderLeaderboard();
    renderMLflowTable();
    renderParetoSVG();
    populateInspectorRunSelect();
    renderInspector();
    renderSpec005();
  }

  function fetchArenaState() {
    fetch('/api/state')
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        state.csrfToken = data.csrf_token || '';
        state.runs = data.runs || [];
        state.groundTruthImages = data.ground_truth_images || {};
        state.imageDimensions = data.image_dimensions || {};
        state.taxonomy7Dim = data.taxonomy_7dim || {};
        state.spec005Summary = data.spec005_summary || {};
        refreshAllViews();
      });
  }

  function wireEvents() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        tabBtns.forEach(function (b) {
          b.classList.remove('active');
        });
        btn.classList.add('active');
        const targetId = btn.getAttribute('data-tab');
        document.querySelectorAll('.tab-panel').forEach(function (panel) {
          if (panel.id === targetId) {
            panel.classList.remove('hidden');
          } else {
            panel.classList.add('hidden');
          }
        });
      });
    });

    const triggerBtn = document.getElementById('trigger-run-btn');
    if (triggerBtn) {
      triggerBtn.addEventListener('click', function () {
        const trackSelect = document.getElementById('track-selector');
        const ldapInput = document.getElementById('engineer-ldap-input');
        fetch('/api/run-track', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': state.csrfToken,
          },
          body: JSON.stringify({
            track_id: trackSelect ? trackSelect.value : 'track_d_jev_routing',
            engineer_ldap: ldapInput ? ldapInput.value : 'jjuneja',
          }),
        })
          .then(function (r) {
            return r.json();
          })
          .then(function (res) {
            if (res.runs) {
              state.runs = res.runs;
              state.selectedRunId = res.new_run.run_id;
              refreshAllViews();
              showToast(
                'Logged MLflow Run ' +
                  res.new_run.run_id +
                  ' (' +
                  res.new_run.medal_tier +
                  ', Pareto Score: ' +
                  res.new_run.pareto_score +
                  ')'
              );
            }
          });
      });
    }

    const runSelect = document.getElementById('inspector-run-select');
    if (runSelect) {
      runSelect.addEventListener('change', function () {
        state.selectedRunId = runSelect.value;
        renderInspector();
      });
    }

    const imgSelect = document.getElementById('inspector-image-select');
    if (imgSelect) {
      imgSelect.addEventListener('change', function () {
        state.selectedImage = imgSelect.value;
        renderInspector();
      });
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    wireEvents();
    fetchArenaState();
  });
})();
