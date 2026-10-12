"""Exploratory L8 Ablation Studies (Archived Outside Core ``src/approaches/``).

Empirical Root-Cause Postmortem (from 26 Cloud Run runs in ``results_experiments/*/images.jsonl``):
  1. ``jev_laya_zoom_strip`` (0.9000 / 0.9078 F2 vs. 0.9571 for ``jev_laya_zscore_lid``):
     - Shrinking the main contact-sheet panel height from 246px to 176px to fit a 56px bottom
       zoom strip reduced 2D pixel area of the primary brand logo/silhouette by 52%, and blindly
       cropping y in [0.65H, 0.96H] captured table shadows on tilted items (test_00033.jpg, test_00061.jpg).
  2. ``clef_flash_decision`` (0.8714 / 0.9222 F2, 36.53s p95 on Flash) & ``detect_clef_flash``:
     - Inherited zoom_multi_sheet's 52% main-panel pixel loss and forced bare integer arrays
       [c1..cm] without JSON object keys, causing row-index drift on 9-item sheets (test_00065.jpg,
       test_00097.jpg) on flash-lite and 26.1s-36.5s p95 latency spikes on flash.
  3. ``clef_flash_classify`` (0.8400 Acc on flash-lite vs. 0.8800 for ``djev_classify``):
     - Removing ``djev_classify``'s ``"pack_size"`` grounding field in favor of a bare integer
       decision caused flash-lite to misclassify ``labeled_sku_0183_starfile.jpg`` (Short vs Long).
  4. ``clm_classify`` & ``detect_clm_contrastive`` (0.9402 vs. 0.9687 E2E F2 for ``detect_l8_ultimate``):
     - Standalone temperature-scaled InfoNCE (1/tau=28.0) without Global Shelf White-Balance and
       without the local manifold Z-Score gate (z_margin >= req_z) falsely accepted low-cosine
       crops on ``test_00049.jpg`` and ``test_00051.jpg`` (3 FPs). Intersecting Z-Score + LID +
       InfoNCE + Global WB in ``l8_hul_ultimate`` / ``detect_l8_ultimate`` eliminated those errors.
"""
