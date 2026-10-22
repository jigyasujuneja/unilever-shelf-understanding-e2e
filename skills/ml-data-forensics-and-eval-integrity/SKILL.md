---
name: ml-data-forensics-and-eval-integrity
description: >-
  End-to-end playbook for auditing raw multimodal/CV datasets across 15 automated
  quality gates, building canonical 5-level master data taxonomies, enforcing
  outlet-code and MD5 split disjointness, and running Paired McNemar discordant
  significance tests with Rogan-Gladen prevalence calibration.
---

# ML Data Forensics, 15-Gate Hygiene & Statistical Evaluation Playbook

## 1. 15-Gate Automated Dataset & Leaderboard Hygiene Checklist (`G01-G15`)

Before training or evaluating any model on multi-folder or multi-silo datasets, enforce all 15 deterministic gates:

1. **G01 (`catalog_label_parity`)**: 100% of split box labels match canonical catalog metadata (`0` schema drift).
2. **G02 (`zero_burned_in_overlays`)**: Scan bounding-box perimeters in HSV/RGB space for burned-in debug rectangles (`(0,255,0)`, `(255,0,0)`, `(0,255,255)`). Quarantine any image containing pre-drawn QA wireframes.
3. **G03 (`split_hash_disjointness`)**: Compute streaming SHA-256/MD5 hashes across all image bytes; enforce zero URI or content-hash overlap across `train` (`gallery`), `val`, and `test`. Never join crops to annotations by recycled filenames (`image_001.jpg`).
4. **G04 (`stratified_owner_balance`)**: Verify balanced primary-owner vs. competitor distribution on golden evaluation splits.
5. **G05 (`zero_unknown_taxonomy`)**: Reject any `"Unknown"`, `"Unbranded"`, or `"Others"` catch-all strings in canonical category or packaging fields; unpack rival brands into named entities.
6. **G06 (`shortlist_oracle_deleakage`)**: `setup()` and candidate retrieval must search the full catalog (`2,659-2,745` SKUs) and never restrict the candidate pool to ground-truth test labels.
7. **G07 (`bay_and_asset_box_integrity`)**: Validate structural display bay (`Class 0` `asset_boxes`) vs. product facing (`Class 1`) annotations on merchandising splits.
8. **G08 (`results_taxonomy_parity`)**: Verify that all persisted `summary.json` and `images.jsonl` artifacts match canonical 5-level taxonomy keys.
9. **G09 (`five_attribute_crop_coverage`)**: Require complete 5-level evaluation (`L1` Owner, `L2` Category/Pack, `L3` Brand, `L4` Variant, `L5` SKU) on every crop run.
10. **G10 (`closed_set_non_abstention`)**: Ensure closed-set benchmarks resolve `choice=0` via deterministic Tier-1/Tier-2 fallback guards rather than silently dropping items.
11. **G11 (`e2e_dual_lens_eval_parity`)**: Report both Full-Catalog (`2,659+` SKUs) and Annotated-Zone box metrics on end-to-end shelf evaluations.
12. **G12 (`detector_dedup_iou_audit`)**: Verify Stage 1 NMS suppresses duplicate overlapping boxes (`IoU > 0.55`).
13. **G13 (`outlet_and_md5_split_deleakage`)**: Group store photos by physical `outlet_code` and `300s` capture session bursts so photos from the same store or visit never leak across `train`, `val`, and `test`.
14. **G14 (`dual_buffer_photometry_integrity`)**: Verify Stage 0 routes linear RGB (`color_image`) to Stage 2 CIELAB color scoring while routing `L* CLAHE` (`detector_image`) only to Stage 1 detection and Stage 2B OCR crops.
15. **G15 (`zero_synthetic_fallbacks_and_zero_official_errors`)**: Prohibit synthetic or dummy fallback responses in production/evaluation runners; require `errors == 0` for official leaderboard ranking.

## 2. Canonical 5-Level Taxonomy Resolution Rules

1. **Specific-Over-General Attribute Precedence**:
   - Resolve fine-grained variant physical-form tokens (`toothbrush -> Carton`, `roll_on -> Roll_On`, `bathing_bar -> Bar`, `dishwash_gel -> Bottle`, `cup_noodles -> Tub`) before applying brand-level or category-level default packaging.
2. **Exact Token Quarantine Over Substring Filtering**:
   - Match exact normalized tokens (`token in {"others", "unknown", "na"}`) rather than substring checks (`"others" in name`) so valid named categories (such as `Others_Face_Makeup`) are preserved.
3. **Orthogonal Attribute Decomposition**:
   - Separate physical container identity (`base_variant` + `packaging_type`) from transient promotional state (`promo_type`) to eliminate multi-folder MD5 label collisions.

## 3. Statistical Power, Paired McNemar & Rogan-Gladen Calibration

1. **Paired McNemar Test on Escalated Cascade Slices (`PR-15`, `EXP-58`)**:
   - In a gated cascade where Stage 2A fast-exits `60%-70%` of items untouched (`a + d` concordant pairs), unpaired Z-tests dilute the statistical signal of Stage 2B.
   - Always evaluate Stage 2B using the continuity-corrected Paired McNemar statistic on the discordant pairs $(b, c)$:
     $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
     where $c$ counts Stage 2A errors fixed by Stage 2B and $b$ counts Stage 2A correct items regressed by Stage 2B.
2. **Inter-Rater Agreement ($\kappa$) & Rogan-Gladen Prevalence Correction (`PR-15`, `EXP-59`)**:
   - Measure human/judge annotation agreement using Cohen's $\kappa = \frac{p_o - p_e}{1 - p_e}$ and explicitly quantify the physical occlusion ceiling.
   - Correct raw production judge pass rates $\hat{p}_{\text{obs}}$ for calibrated judge sensitivity $p$ and specificity $q$ using the Rogan-Gladen estimator:
     $$\hat{\theta} = \text{clip}_{[0,1]}\left(\frac{\hat{p}_{\text{obs}} + q - 1}{p + q - 1}\right)$$
