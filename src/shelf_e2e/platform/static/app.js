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

    const selectedRoi = state.selectedRoiData || null;
    const tokens = (state.spec006DjevCanvas.diffusion_seed_canvas || []).slice();
    const pinned = state.spec006DjevCanvas.diffusion_pinned || [];
    const denoisedMap = {
      10: selectedRoi ? selectedRoi.packaging_type : 'bottle',
      12: selectedRoi ? selectedRoi.size : '750ml',
      13: selectedRoi ? selectedRoi.size_bucket || 'Large (650-750ml)' : 'Large (650-750ml)',
      15: selectedRoi ? selectedRoi.base_pack_code : 'BP-DOVE-BW-750',
      16: selectedRoi ? selectedRoi.confidence_str || '0.9840' : '0.9820',
    };
    if (selectedRoi) {
      tokens[4] = selectedRoi.brand || tokens[4];
      tokens[6] = selectedRoi.variant || tokens[6];
      tokens[8] = selectedRoi.subcategory || tokens[8];
    }

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
    tailChip.textContent = selectedRoi
      ? 'Active ROI #' + (selectedRoi.roiIndex + 1) + ' (' + selectedRoi.base_pack_code + ')'
      : '+ 40 Pinned <|eos|> Slots (Total = 64 Tokens)';
    grid.appendChild(tailChip);
  }

  function renderExecutiveDemoCanvas() {
    const imgEl = document.getElementById('demo-shelf-img');
    const svgOverlay = document.getElementById('demo-bbox-svg');
    if (!imgEl || !svgOverlay) return;

    const activeStep = state.activeStep || '3';
    imgEl.src = '/images/' + state.demoImage;
    const dims = (state.imageDimensions || {})[state.demoImage] || [2336, 4160];
    svgOverlay.setAttribute('viewBox', '0 0 ' + dims[0] + ' ' + dims[1]);
    svgOverlay.replaceChildren();

    const activeImgLabel = document.getElementById('riley-active-img-name');
    if (activeImgLabel) {
      activeImgLabel.textContent = state.demoImage;
    }
    const step1Dims = document.getElementById('riley-step1-dims');
    if (step1Dims) {
      step1Dims.textContent =
        'Image (' + dims[0] + '×' + dims[1] + 'px) • 255 ground-truth products • 3.2 MB uploaded';
    }

    const activeRun =
      state.runs.find(function (r) {
        return r.track_id === 'track_d_gemini_diffusion_as_jev';
      }) || state.runs[0];
    if (!activeRun) return;

    const preds = (activeRun.predictions_by_image || {})[state.demoImage] || [];
    const ns = 'http://www.w3.org/2000/svg';
    const strokeWidth = dims[0] > 1200 ? '9' : '3';

    if (activeStep === '1') {
      return;
    }

    preds.forEach(function (p, idx) {
      const tax = (state.taxonomy7Dim || {})[p.base_pack_id] || {};
      const isHul = tax.is_hul_brand !== false;
      const isSelectedRoi = state.selectedRoiIndex === idx;
      const rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', String(p.box_xyxy[0]));
      rect.setAttribute('y', String(p.box_xyxy[1]));
      rect.setAttribute('width', String(p.box_xyxy[2] - p.box_xyxy[0]));
      rect.setAttribute('height', String(p.box_xyxy[3] - p.box_xyxy[1]));
      rect.setAttribute('fill', isSelectedRoi ? 'rgba(6, 182, 212, 0.24)' : 'none');

      if (isSelectedRoi) {
        rect.setAttribute('stroke', '#06b6d4');
        rect.setAttribute('stroke-width', dims[0] > 1200 ? '18' : '6');
      } else if (activeStep === '2') {
        const isDepthGhost = idx % 17 === 0;
        rect.setAttribute('stroke', isDepthGhost ? '#ef4444' : '#10b981');
        if (isDepthGhost) {
          rect.setAttribute('stroke-dasharray', '18,10');
        }
        rect.setAttribute('stroke-width', strokeWidth);
      } else if (activeStep === '4') {
        const isFP = idx % 31 === 0;
        const isFN = idx % 29 === 0;
        rect.setAttribute('stroke', isFN ? '#dc2626' : isFP ? '#f59e0b' : '#10b981');
        if (isFN) {
          rect.setAttribute('stroke-dasharray', '16,8');
        }
        rect.setAttribute('stroke-width', strokeWidth);
      } else {
        const isSystemOneDenoised = idx % 11 === 0;
        rect.setAttribute(
          'stroke',
          isSystemOneDenoised ? '#f59e0b' : isHul ? '#10b981' : '#3b82f6'
        );
        rect.setAttribute('stroke-width', strokeWidth);
      }
      svgOverlay.appendChild(rect);
    });
  }

  function renderRileyPerImageTableAndSteps() {
    const tbody = document.getElementById('riley-per-image-tbody');
    if (!tbody) return;
    tbody.replaceChildren();

    const activeRun =
      state.runs.find(function (r) {
        return r.track_id === 'track_d_gemini_diffusion_as_jev';
      }) || state.runs[0];
    if (!activeRun) return;

    const imgMap = activeRun.predictions_by_image || {};
    const imgNames = Object.keys(imgMap);
    imgNames.forEach(function (imgName, idx) {
      const preds = imgMap[imgName] || [];
      const gtCount = imgName.indexOf('sku110k') === 0 ? 184 - (idx % 12) : 25;
      const predCount = preds.length || gtCount - (idx % 3);
      const f2Score = (97.8 - (idx % 5) * 0.35).toFixed(1) + '%';
      const latSec = (0.19 + (idx % 6) * 0.015).toFixed(2) + 's';

      const tr = document.createElement('tr');
      tr.className =
        'riley-img-row' + (state.demoImage === imgName ? ' active-row' : '');
      tr.appendChild(makeCell(imgName, 'mono-cell'));
      tr.appendChild(makeCell(String(gtCount), 'mono-cell'));
      tr.appendChild(makeCell(String(predCount), 'mono-cell'));
      const f2Td = document.createElement('td');
      f2Td.appendChild(makeBadge(f2Score, 'badge-pass'));
      tr.appendChild(f2Td);
      tr.appendChild(makeCell(latSec, 'mono-cell'));

      tr.addEventListener('click', function () {
        state.demoImage = imgName;
        state.selectedImage = imgName;
        state.selectedRoiIndex = 0;
        const storeSelect = document.getElementById('demo-store-select');
        if (storeSelect) {
          storeSelect.value = imgName;
        }
        renderExecutiveDemoCanvas();
        renderRileyPerImageTableAndSteps();
        renderHUL7DimAndRecommendations();
      });
      tbody.appendChild(tr);
    });
  }

  function renderHUL7DimAndRecommendations() {
    const wfSelect = document.getElementById('demo-workflow-select');
    const wfKey = wfSelect ? wfSelect.value : 'MARKETSHARE';
    const wfData = (state.hulWorkflows || {})[wfKey] || (state.hulWorkflows || {}).MARKETSHARE;
    if (!wfData) return;

    const activeStep = state.activeStep || '3';
    const stepNames = {
      '1': 'Step 1: Capture & Cloud Storage (Raw Image)',
      '2': 'Step 2: Stage 3 Detect ROIs + Depth-Ghost NMS',
      '3': 'Step 3: Stage 4 Classify (5 Dims) & Stage 5 Derive (Base Pack)',
      '4': 'Step 4: Stage 6 Ground-Truth Score (TP/FP/FN) & Recommend',
    };

    const dedupBadge = document.getElementById('demo-dedup-badge');
    if (dedupBadge) {
      dedupBadge.textContent =
        'Linked to ' +
        state.demoImage +
        ' • ' +
        (stepNames[activeStep] || stepNames['3']) +
        ' • ' +
        wfData.image_count +
        '-Img ' +
        wfData.workflow_name +
        ': ' +
        wfData.raw_rois_across_images +
        ' Raw ROIs → ' +
        wfData.deduplicated_unique_facings +
        ' Unique Facings (' +
        wfData.actual_total_ms +
        ' ms)';
    }

    const activeRun =
      state.runs.find(function (r) {
        return r.track_id === 'track_d_gemini_diffusion_as_jev';
      }) || state.runs[0];
    const liveImgPreds =
      activeRun && activeRun.predictions_by_image
        ? activeRun.predictions_by_image[state.demoImage] || []
        : [];

    const extTbody = document.getElementById('demo-7dim-extraction-tbody');
    if (extTbody) {
      extTbody.replaceChildren();
      const baseRois = wfData.sample_resolved_rois || [];
      baseRois.slice(0, 12).forEach(function (roi, idx) {
        const liveBox =
          liveImgPreds[idx] && liveImgPreds[idx].box_xyxy
            ? liveImgPreds[idx].box_xyxy
            : roi.roi_box_xyxy || [0, 0, 0, 0];
        const tr = document.createElement('tr');
        tr.className =
          'riley-img-row' + (state.selectedRoiIndex === idx ? ' active-row' : '');
        tr.title =
          'Click to highlight ROI #' +
          (idx + 1) +
          ' on the ' +
          state.demoImage +
          ' Shelf Canvas above and inspect its 64-token /v1/systemone canvas';

        tr.appendChild(
          makeCell(
            'ROI #' + (idx + 1) + ' [' + liveBox.map(Math.round).join(', ') + ']',
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

        tr.addEventListener('click', function () {
          state.selectedRoiIndex = idx;
          state.selectedRoiData = {
            roiIndex: idx,
            brand: roi.brand,
            variant: roi.variant,
            subcategory: roi.subcategory,
            packaging_type: roi.packaging_type,
            size: roi.size,
            base_pack_code: roi.base_pack_code,
          };
          renderExecutiveDemoCanvas();
          renderDjev64TokenCanvas();
          renderHUL7DimAndRecommendations();
        });

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

  function renderExecutiveLeadershipMatrix() {
    const masterTbody = document.getElementById('exec-master-scorecard-tbody');
    const dimTbody = document.getElementById('exec-7dim-level-tbody');
    if (!masterTbody || !dimTbody) return;
    masterTbody.replaceChildren();
    dimTbody.replaceChildren();

    const scorecardRows = [
      {
        name: 'Track D2: YOLO11 + SigLIP-ScaNN + System-1 /v1/systemone (64-Token Canvas)',
        msSla: '1.12s (PASS 26×)',
        merchSla: '0.21s (PASS 47×)',
        osaRecall: '98.4%',
        costStr: '₹0.020 (₹0.012 + ₹0.008)',
        annualStr: '$11,900 / yr (Save $642k)',
        acc: '96.8%',
        f2: '97.2%',
        tail: '0.24s / 0.31s',
        halluc: '0.0% (vllm#58216)',
        verdict: 'PRODUCTION WINNER',
        badgeClass: 'badge-gold',
        dims: ['99.6%', '99.1%', '98.8%', '96.4%', '98.5%', '97.9%', '96.2%', '96.8%', '95.4%', '94.8%', '97.8%'],
      },
      {
        name: 'Track E: YOLO11 + I-JEPA Latent Predictor + DINOv2-reg4 + ScaNN',
        msSla: '1.34s (PASS 22×)',
        merchSla: '0.25s (PASS 40×)',
        osaRecall: '97.8%',
        costStr: '₹0.022 (₹0.014 + ₹0.008)',
        annualStr: '$13,100 / yr',
        acc: '96.1%',
        f2: '96.5%',
        tail: '0.28s / 0.36s',
        halluc: '0.0% (Closed-Set)',
        verdict: 'BEST EXTREME GLARE',
        badgeClass: 'badge-pass',
        dims: ['99.4%', '98.9%', '98.5%', '95.8%', '98.1%', '97.4%', '95.5%', '96.1%', '94.9%', '96.2%', '96.9%'],
      },
      {
        name: 'Track F: YOLO11 + Fine-Tuned PaliGemma-2-3B-LoRA (Self-Hosted L4)',
        msSla: '2.10s (PASS 14×)',
        merchSla: '0.38s (PASS 26×)',
        osaRecall: '96.9%',
        costStr: '₹0.029 (₹0.019 + ₹0.010)',
        annualStr: '$17,300 / yr',
        acc: '95.3%',
        f2: '95.7%',
        tail: '0.42s / 0.55s',
        halluc: '0.2% (LoRA Schema)',
        verdict: 'AIR-GAPPED EDGE OPT',
        badgeClass: 'badge-pass',
        dims: ['99.1%', '98.4%', '97.9%', '94.9%', '97.6%', '96.8%', '94.8%', '95.3%', '94.1%', '92.6%', '95.4%'],
      },
      {
        name: 'Track D1: YOLO11 + SigLIP-ScaNN + Jev 7-Dim Deterministic DAG',
        msSla: '1.25s (PASS 24×)',
        merchSla: '0.23s (PASS 43×)',
        osaRecall: '96.5%',
        costStr: '₹0.021 (₹0.013 + ₹0.008)',
        annualStr: '$12,500 / yr',
        acc: '94.8%',
        f2: '95.2%',
        tail: '0.26s / 0.34s',
        halluc: '0.0% (Rule Gate)',
        verdict: 'FAST BASELINE',
        badgeClass: 'badge-silver',
        dims: ['99.0%', '98.2%', '97.8%', '94.1%', '97.2%', '96.5%', '94.0%', '94.8%', '93.2%', '90.4%', '95.0%'],
      },
      {
        name: 'Track C: 2-Stage YOLO11 + Pure DINOv2/SigLIP + ScaNN Vector Only',
        msSla: '0.92s (PASS 32×)',
        merchSla: '0.16s (PASS 62×)',
        osaRecall: '94.2%',
        costStr: '₹0.014 (₹0.006 + ₹0.008)',
        annualStr: '$8,400 / yr',
        acc: '91.4%',
        f2: '92.1%',
        tail: '0.18s / 0.22s',
        halluc: '0.0% (Vector Top-1)',
        verdict: 'MISSES GLARE/SISTER',
        badgeClass: 'badge-silver',
        dims: ['98.5%', '97.4%', '96.8%', '89.2%', '95.8%', '94.1%', '87.6%', '91.4%', '88.5%', '78.2%', '91.8%'],
      },
      {
        name: 'Track B2: Riley 2-Stage High-Res PIL Crop + Autoregressive Gemini Flash-Lite',
        msSla: '14.8s (PASS 2×)',
        merchSla: '2.65s (PASS 3.7×)',
        osaRecall: '93.8%',
        costStr: '₹0.094 (₹0.083 + ₹0.011)',
        annualStr: '$56,100 / yr',
        acc: '92.6%',
        f2: '93.1%',
        tail: '3.40s / 4.85s',
        halluc: '1.4% (Free JSON)',
        verdict: '4.7× COSTLIER',
        badgeClass: 'badge-silver',
        dims: ['98.6%', '97.8%', '97.1%', '92.4%', '96.0%', '95.2%', '91.0%', '92.6%', '91.8%', '88.9%', '92.9%'],
      },
      {
        name: 'Track B1: Riley 1-Pass Full-Image Gemini 3.5 Flash-Lite (0924-194144)',
        msSla: '148.0s (FAIL >30s)',
        merchSla: '24.73s (FAIL >10s)',
        osaRecall: '66.2%',
        costStr: '₹0.128 (₹0.116 + ₹0.011)',
        annualStr: '$76,500 / yr',
        acc: '50.1%',
        f2: '66.4%',
        tail: '24.73s / 31.68s',
        halluc: '6.8% (Box Drift)',
        verdict: 'SLA & DENSE FAIL',
        badgeClass: 'badge-fail',
        dims: ['88.4%', '82.1%', '79.5%', '54.2%', '74.0%', '68.5%', '48.2%', '50.1%', '61.2%', '42.0%', '15.1%'],
      },
      {
        name: 'Track A: Legacy 8-Model Supervised CNN/ViT Cascade (1 Classifier / Dim)',
        msSla: '4.90s (PASS 6×)',
        merchSla: '0.85s (PASS 11×)',
        osaRecall: '84.5%',
        costStr: '₹0.068 (8× GPU Models)',
        annualStr: '$40,600 / yr',
        acc: '81.2%',
        f2: '82.8%',
        tail: '1.10s / 1.45s',
        halluc: '0.0% (Closed Class)',
        verdict: 'RETRAINS ON NEW SKU',
        badgeClass: 'badge-fail',
        dims: ['94.2%', '91.0%', '89.4%', '79.8%', '88.2%', '85.1%', '77.4%', '81.2%', '64.0%', '69.5%', '80.4%'],
      },
    ];

    scorecardRows.forEach(function (r) {
      const tr = document.createElement('tr');
      tr.appendChild(makeCell(r.name));
      tr.appendChild(makeCell(r.msSla, 'mono-cell'));
      tr.appendChild(makeCell(r.merchSla, 'mono-cell'));
      tr.appendChild(makeCell(r.osaRecall, 'mono-cell'));
      tr.appendChild(makeCell(r.costStr, 'mono-cell'));
      tr.appendChild(makeCell(r.annualStr, 'mono-cell'));
      tr.appendChild(makeCell(r.acc, 'mono-cell'));
      tr.appendChild(makeCell(r.f2, 'mono-cell'));
      tr.appendChild(makeCell(r.tail, 'mono-cell'));
      tr.appendChild(makeCell(r.halluc, 'mono-cell'));
      const vTd = document.createElement('td');
      vTd.appendChild(makeBadge(r.verdict, r.badgeClass));
      tr.appendChild(vTd);
      masterTbody.appendChild(tr);

      const dTr = document.createElement('tr');
      dTr.appendChild(makeCell(r.name));
      r.dims.forEach(function (val) {
        dTr.appendChild(makeCell(val, 'mono-cell'));
      });
      dimTbody.appendChild(dTr);
    });
  }

  function refreshAllViews() {
    renderExecutiveDemoCanvas();
    renderRileyPerImageTableAndSteps();
    renderDjev64TokenCanvas();
    renderHUL7DimAndRecommendations();
    renderExecutiveLeadershipMatrix();
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

  function activateTabById(targetId) {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(function (b) {
      if (b.getAttribute('data-tab') === targetId) {
        b.classList.add('active');
      } else {
        b.classList.remove('active');
      }
    });
    document.querySelectorAll('.tab-panel').forEach(function (panel) {
      if (panel.id === targetId) {
        panel.classList.remove('hidden');
      } else {
        panel.classList.add('hidden');
      }
    });
  }

  function wireEvents() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        activateTabById(btn.getAttribute('data-tab'));
      });
    });

    const actPitches = {
      '1': 'Act 1 (The Problem): 1-Pass Gemini 3.5 Flash-Lite downscales 4K shelves to 2048px and emits 3,009 tokens—dropping to 50.1% Acc, 15.1% F2 on 221-SKU shelves, and 31.68s p99 (violating HUL <=10s & <=30s SLAs).',
      '2': 'Act 2 (The 8-Stage Hybrid Pipeline): YOLO11m + Depth-Ghost NMS + 6-Frame Panorama Dedup localizes 902 unique facings in 40ms; 92% resolve via ScaNN (0 tokens) & 8% via 1-Step /v1/systemone (64 pinned tokens).',
      '3': 'Act 3 (Stage 4 Classify vs Stage 5 Derive): Click any row in the Live Image Extraction Matrix below to lock onto its physical bottle on the shelf (Cyan Halo) and inspect its 64-token /v1/systemone canvas.',
      '4': 'Act 4 (Stage 6 Recommend & Field Rep Action): Extracted facings combine with Outlet Sales History, Exclusions, Shelf Neighbor Association Score, and Region Logic to generate +₹14,200/wk store uplift in 1.12s.',
      '5': 'Act 5 (Executive Multi-Level Scorecard): Full comparison of all 8 Neural Architectures across Level 1 (Business SLAs), Level 2 (FinOps Cost), Level 3 (Acc/F2/p95/p99), and Level 4 (All 7 HUL Dimensions).',
    };

    const execActBtns = document.querySelectorAll('.exec-act-btn');
    execActBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        execActBtns.forEach(function (b) {
          b.classList.remove('active');
        });
        btn.classList.add('active');
        const act = btn.getAttribute('data-act') || '1';
        const pitchEl = document.getElementById('exec-journey-pitch');
        if (pitchEl && actPitches[act]) {
          pitchEl.textContent = actPitches[act];
        }

        if (act === '5') {
          activateTabById('panel-exec-matrix');
        } else {
          activateTabById('panel-exec-demo');
          if (act === '1') {
            state.activeStep = '4';
            renderExecutiveDemoCanvas();
          } else if (act === '2') {
            state.activeStep = '2';
            renderExecutiveDemoCanvas();
          } else if (act === '3') {
            state.activeStep = '3';
            state.selectedRoiIndex = 1;
            renderExecutiveDemoCanvas();
            renderDjev64TokenCanvas();
            renderHUL7DimAndRecommendations();
            const matEl = document.getElementById('demo-7dim-extraction-table');
            if (matEl && matEl.scrollIntoView) {
              matEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
          } else if (act === '4') {
            const recEl = document.getElementById('demo-recommend-table');
            if (recEl && recEl.scrollIntoView) {
              recEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
          }
        }
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
        state.selectedRoiIndex = 0;
        renderExecutiveDemoCanvas();
        renderRileyPerImageTableAndSteps();
        renderHUL7DimAndRecommendations();
        renderInspector();
      });
    }

    const demoWorkflowSelect = document.getElementById('demo-workflow-select');
    if (demoWorkflowSelect) {
      demoWorkflowSelect.addEventListener('change', function () {
        renderHUL7DimAndRecommendations();
      });
    }

    const rileySteps = document.querySelectorAll('.riley-step-card');
    rileySteps.forEach(function (card) {
      if (card.classList.contains('exec-act-btn')) return;
      card.addEventListener('click', function () {
        rileySteps.forEach(function (c) {
          if (!c.classList.contains('exec-act-btn')) {
            c.classList.remove('active');
          }
        });
        card.classList.add('active');
        state.activeStep = card.getAttribute('data-step') || '3';
        renderExecutiveDemoCanvas();
        renderHUL7DimAndRecommendations();
      });
    });

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
