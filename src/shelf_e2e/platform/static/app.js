/* Unilever Perfect Store AI — Enterprise Gondola Intelligence & Neural Architecture Control Plane */

(function () {
  'use strict';

  const state = {
    csrfToken: '',
    runs: [],
    groundTruthImages: {},
    imageDimensions: {},
    taxonomy7Dim: {},
    spec005Summary: {},
    spec006DjevCanvas: null,
    selectedRunId: '',
    selectedImage: 'sku110k_val_000.jpg',
    demoImage: 'sku110k_val_000.jpg',
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
    }, 4500);
  }

  function renderDjev64TokenCanvas() {
    const grid = document.getElementById('djev-canvas-grid');
    if (!grid || !state.spec006DjevCanvas) return;
    grid.replaceChildren();

    const tokens = state.spec006DjevCanvas.diffusion_seed_canvas || [];
    const pinned = state.spec006DjevCanvas.diffusion_pinned || [];
    const denoisedMap = {
      10: 'bottle',
      12: '750ml',
      13: 'Large (650-750ml)',
      15: 'BP-DOVE-BW-750',
      16: '0.9820',
    };

    tokens.slice(0, 24).forEach(function (tok, idx) {
      const chip = document.createElement('span');
      const isPinned = pinned[idx] !== false;
      chip.className = isPinned ? 'djev-tok djev-tok-pinned' : 'djev-tok djev-tok-denoised';
      chip.textContent = isPinned ? tok : denoisedMap[idx] || tok;
      chip.title = isPinned
        ? 'Token #' + idx + ': Pinned Context (diffusion_pinned=true)'
        : 'Token #' + idx + ': Denoised in 1 Parallel Step (/v1/systemone)';
      grid.appendChild(chip);
    });

    const tailChip = document.createElement('span');
    tailChip.className = 'djev-tok djev-tok-pinned';
    tailChip.textContent = '+ 40 Pinned <|eos|> Slots (Total = 64 Tokens)';
    grid.appendChild(tailChip);
  }

  function renderExecutiveDemoCanvas() {
    const imgEl = document.getElementById('demo-shelf-img');
    const svgOverlay = document.getElementById('demo-bbox-svg');
    if (!imgEl || !svgOverlay) return;

    imgEl.src = '/images/' + state.demoImage;
    const dims = (state.imageDimensions || {})[state.demoImage] || [2336, 4160];
    svgOverlay.setAttribute('viewBox', '0 0 ' + dims[0] + ' ' + dims[1]);
    svgOverlay.replaceChildren();

    const activeRun =
      state.runs.find(function (r) {
        return r.track_id === 'track_d_gemini_diffusion_as_jev';
      }) || state.runs[0];
    if (!activeRun) return;

    const preds = (activeRun.predictions_by_image || {})[state.demoImage] || [];
    const ns = 'http://www.w3.org/2000/svg';
    const strokeWidth = dims[0] > 1200 ? '9' : '3';

    preds.forEach(function (p, idx) {
      const tax = (state.taxonomy7Dim || {})[p.base_pack_id] || {};
      const isHul = tax.is_hul_brand !== false;
      const rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', String(p.box_xyxy[0]));
      rect.setAttribute('y', String(p.box_xyxy[1]));
      rect.setAttribute('width', String(p.box_xyxy[2] - p.box_xyxy[0]));
      rect.setAttribute('height', String(p.box_xyxy[3] - p.box_xyxy[1]));
      rect.setAttribute('fill', 'none');
      rect.setAttribute('stroke', idx % 19 === 0 ? '#f43f5e' : isHul ? '#10b981' : '#3b82f6');
      rect.setAttribute('stroke-width', strokeWidth);
      svgOverlay.appendChild(rect);
    });
  }

  function renderHUL7DimAndRecommendations() {
    const wfSelect = document.getElementById('demo-workflow-select');
    const wfKey = wfSelect ? wfSelect.value : 'MARKETSHARE';
    const wfData = (state.hulWorkflows || {})[wfKey] || (state.hulWorkflows || {}).MARKETSHARE;
    if (!wfData) return;

    const dedupBadge = document.getElementById('demo-dedup-badge');
    if (dedupBadge) {
      dedupBadge.textContent =
        wfData.image_count +
        '-Image Request (' +
        wfData.workflow_name +
        '): ' +
        wfData.raw_rois_across_images +
        ' Raw ROIs → ' +
        wfData.deduplicated_unique_facings +
        ' Unique Gondola Facings (' +
        wfData.overlap_duplicates_suppressed +
        ' Overlap Duplicates Suppressed) • Total E2E: ' +
        wfData.actual_total_ms +
        ' ms (SLA <= ' +
        wfData.sla_limit_ms +
        ' ms)';
    }

    const extTbody = document.getElementById('demo-7dim-extraction-tbody');
    if (extTbody) {
      extTbody.replaceChildren();
      (wfData.sample_resolved_rois || []).slice(0, 12).forEach(function (roi) {
        const tr = document.createElement('tr');
        tr.appendChild(
          makeCell(
            '[' + (roi.roi_box_xyxy || [0, 0, 0, 0]).map(Math.round).join(', ') + ']',
            'mono-cell'
          )
        );
        const ownTd = document.createElement('td');
        ownTd.appendChild(
          makeBadge(
            roi.is_hul_sku ? 'HUL SKU' : 'NON-HUL COMPETITOR',
            roi.is_hul_sku ? 'badge-pass' : 'badge-silver'
          )
        );
        tr.appendChild(ownTd);
        tr.appendChild(makeCell(roi.category));
        tr.appendChild(makeCell(roi.subcategory));
        tr.appendChild(makeCell(roi.brand));
        tr.appendChild(makeCell(roi.variant));
        tr.appendChild(makeCell(roi.packaging_type, 'mono-cell'));
        tr.appendChild(makeCell(roi.pack_type, 'mono-cell'));
        tr.appendChild(makeCell(roi.size, 'mono-cell'));
        tr.appendChild(makeCell(roi.base_pack_code, 'mono-cell'));
        const branchTd = document.createElement('td');
        branchTd.appendChild(
          makeBadge(
            roi.routing_branch === 'CLOSED_SET_HUL_PRODUCT_MASTER'
              ? 'ScaNN + /v1/systemone Master'
              : 'Open-Set 5-Dim Classifier',
            roi.routing_branch === 'CLOSED_SET_HUL_PRODUCT_MASTER' ? 'badge-gold' : 'badge-silver'
          )
        );
        tr.appendChild(branchTd);
        extTbody.appendChild(tr);
      });
    }

    const recTbody = document.getElementById('demo-recommend-tbody');
    if (recTbody) {
      recTbody.replaceChildren();
      (wfData.recommendations || []).forEach(function (rec) {
        const tr = document.createElement('tr');
        tr.appendChild(makeCell(rec.recommended_base_pack_code, 'mono-cell'));
        tr.appendChild(makeCell(rec.product_name));
        const typeTd = document.createElement('td');
        typeTd.appendChild(makeBadge(rec.recommendation_type, 'badge-gold'));
        tr.appendChild(typeTd);
        tr.appendChild(makeCell(rec.sales_velocity_percentile.toFixed(1) + 'th %ile', 'mono-cell'));
        tr.appendChild(makeCell(rec.association_score.toFixed(2), 'mono-cell'));
        tr.appendChild(makeCell(rec.region_match));
        const exTd = document.createElement('td');
        exTd.appendChild(makeBadge(rec.exclusion_check, 'badge-pass'));
        tr.appendChild(exTd);
        tr.appendChild(makeCell(rec.composite_recommendation_score.toFixed(4), 'mono-cell'));
        tr.appendChild(makeCell('+₹' + rec.expected_weekly_uplift_inr.toLocaleString() + '/wk', 'mono-cell'));
        recTbody.appendChild(tr);
      });
    }
  }

  function renderLeaderboard() {
    const tbody = document.getElementById('leaderboard-tbody');
    if (!tbody) return;
    tbody.replaceChildren();

    state.runs.forEach(function (run, idx) {
      const tr = document.createElement('tr');
      tr.appendChild(makeCell('#' + (idx + 1), 'mono-cell'));

      const medalTd = document.createElement('td');
      const tierText =
        run.medal_tier === 'GOLD'
          ? 'TIER-1 PRODUCTION READY'
          : run.medal_tier === 'SILVER'
          ? 'TIER-2 CANDIDATE'
          : 'SLA EXCEEDED';
      const medalClass =
        run.medal_tier === 'GOLD'
          ? 'badge-gold'
          : run.medal_tier === 'SILVER'
          ? 'badge-silver'
          : 'badge-fail';
      medalTd.appendChild(makeBadge(tierText, medalClass));
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
          run.meets_all_slas ? 'APPROVED (<= ₹0.22)' : 'COST SLA EXCEEDED',
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
    slaLabel.setAttribute('x', String(slaX - 155));
    slaLabel.setAttribute('y', '48');
    slaLabel.setAttribute('fill', '#dc2626');
    slaLabel.setAttribute('font-size', '12');
    slaLabel.setAttribute('font-weight', '700');
    slaLabel.textContent = 'Enterprise Cost Ceiling (₹0.22 / Audit)';
    svg.appendChild(slaLabel);

    state.runs.forEach(function (run, i) {
      const cost = Math.min(0.30, run.cost_breakdown.cost_per_image_inr);
      const acc = run.public_metrics.top1_acc;
      const cx = 70 + (cost / 0.30) * 620;
      const cy = 290 - acc * 220 + (i % 3) * 12;

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
      tr.appendChild(makeCell((tax.brand || 'Dove') + (isHul ? ' (Unilever)' : ' (Competitor)')));
      tr.appendChild(makeCell(tax.rule_derived_size_bucket || 'Large (>110g/ml)', 'mono-cell'));
      tr.appendChild(makeCell((p.confidence * 100).toFixed(1) + '%', 'mono-cell'));
      const statusTd = document.createElement('td');
      statusTd.appendChild(
        makeBadge(isMatch ? 'VERIFIED' : 'DISCREPANCY', isMatch ? 'badge-pass' : 'badge-fail')
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
    renderExecutiveDemoCanvas();
    renderDjev64TokenCanvas();
    renderHUL7DimAndRecommendations();
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
        state.spec006DjevCanvas = data.spec006_djev_systemone || null;
        state.hulWorkflows = data.hul_workflows || {};
        refreshAllViews();
      });
  }

  function triggerLiveAudit() {
    const trackSelect = document.getElementById('track-selector');
    const ldapInput = document.getElementById('engineer-ldap-input');
    fetch('/api/run-track', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': state.csrfToken,
      },
      body: JSON.stringify({
        track_id: trackSelect ? trackSelect.value : 'track_d_gemini_diffusion_as_jev',
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
          const latEl = document.getElementById('demo-live-latency');
          if (latEl && res.new_run.latency_tiers_ms) {
            latEl.textContent = res.new_run.latency_tiers_ms.total_e2e.toFixed(0) + ' ms';
          }
          refreshAllViews();
          showToast(
            'Completed Live Gondola Audit (' +
              res.new_run.track_id +
              ') • Unit Cost: ₹' +
              res.new_run.cost_breakdown.cost_per_image_inr.toFixed(4) +
              ' • Composite Score: ' +
              res.new_run.pareto_score
          );
        }
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
      triggerBtn.addEventListener('click', triggerLiveAudit);
    }

    const demoSimBtn = document.getElementById('demo-simulate-btn');
    if (demoSimBtn) {
      demoSimBtn.addEventListener('click', triggerLiveAudit);
    }

    const demoStoreSelect = document.getElementById('demo-store-select');
    if (demoStoreSelect) {
      demoStoreSelect.addEventListener('change', function () {
        state.demoImage = demoStoreSelect.value;
        state.selectedImage = demoStoreSelect.value;
        renderExecutiveDemoCanvas();
        renderInspector();
      });
    }

    const demoWorkflowSelect = document.getElementById('demo-workflow-select');
    if (demoWorkflowSelect) {
      demoWorkflowSelect.addEventListener('change', function () {
        renderHUL7DimAndRecommendations();
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
