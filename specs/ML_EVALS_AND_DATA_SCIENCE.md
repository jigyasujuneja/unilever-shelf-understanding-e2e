# SPEC-004: Data Science, Statistical Rigor & Metamorphic ML Evals (`ML_EVALS_AND_DATA_SCIENCE.md`)

Synthesizes best practices from Marketplace Data Science & ML Skills (`ml-best-practices`, `data-autocleaning`, `schema-mapping`, and `Design is the New Code`):

## 1. The 5 Pillars of Production ML Evaluation

### Pillar 1: Bootstrap 95% Confidence Intervals & Paired Significance Testing
- Single point estimates (`Top-1 = 1.00 vs 0.75`) are insufficient for production sign-off.
- Every core KPI (`Top-1 Accuracy`, `mAP@50:95`, `MRR`) MUST report a non-parametric **Bootstrap 95% Confidence Interval (`[ci_lower, ci_upper]`)** ($B = 500$ resamples) and a **Paired Bootstrap $p$-value** testing $H_0: \text{Metric}(\text{Track D}) \le \text{Metric}(\text{Baseline})$.

### Pillar 2: Slice-Based Subpopulation Error Analysis
- Aggregate accuracy hides catastrophic failures on specific optical or packaging subpopulations.
- Every candidate pipeline MUST be disaggregated across **4 mandatory domain slices**:
  1. **Optical Glare Slice:** `high_glare` (`glare_intensity >= 0.35`) vs. `low_glare` (`glare_intensity < 0.35`).
  2. **Pack Geometry Slice:** `small_narrow_packs` (`width_px < 65.0`, e.g., `100g` face wash tubes) vs. `large_bottles` (`width_px >= 65.0`, e.g., `750ml` shampoo/body wash pumps).
  3. **Shelf Density Slice:** `dense_bay` (`>= 15 facings/image`) vs. `standard_bay` (`< 15 facings/image`).
  4. **Brand Ownership Slice:** `hul_brands` (`is_hul == True`) vs. `competitor_brands` (`is_hul == False`).

### Pillar 3: Confidence Calibration (`ECE`, `Brier Score`) & SKU Confusion Matrix
- Downstream store-rep coaching depends on well-calibrated probabilities (`confidence`).
- Calculate **Expected Calibration Error (`ECE`, $M=10$ bins)**, **Multi-class Brier Score**, and the **Pairwise SKU Confusion Matrix** (`(true_sku, predicted_sku) -> count`) to surface fine-grained variant confusion (such as `BP-DOVE-BW-750` vs. `BP-DOVE-BW-500`).

### Pillar 4: Automated Data Quality Profiling & Split Leakage Prevention (`data-autocleaning`)
- Before scoring any benchmark split, run automated data profiling (`profile_dataset_and_check_leakage`) to verify:
  - `0` null/empty `base_pack_id`s or degenerate bounding boxes (`x2 <= x1` or `y2 <= y1`).
  - `0.0%` image filename or SHA-256 content leakage between **Public Split** (`sku110k_val_001/002`) and **Private Holdout Split** (`sku110k_val_003_dense147`).
  - `100%` foreign-key referential integrity between ground-truth `gt_base_pack_id`s and the `RPC` Master Catalog.

### Pillar 5: "Design is the New Code" Metamorphic & Property-Based Invariants
- **Invariant 1 (Rank-Order Monotonicity):** `Top-5 Recall >= Top-1 Accuracy` and `mAP@50 >= mAP@50:95` for every run.
- **Invariant 2 (Spatial Jitter Robustness):** Perturbing bounding box coordinates by $\pm 2.0\text{px}$ MUST preserve `JevDeterministicStateMachine` `750ml` vs. `500ml` resolution.
- **Invariant 3 (Zero-Catalog Hallucination Gate):** Any predicted `base_pack_id` outside `RPCCatalogAdapter.valid_base_pack_ids()` MUST increment `hallucination_rate` and fail the production SLA gate.
