# P2P (Plan-to-Purchase) Impact Simulation Readout

## 1. Purpose & Downstream Pipeline Alignment
- Mirrors Unilever's exact `ROI_Pipeline.json` (`12-column Past_ROI_Data_*.csv`), Stage A (`NR_Brand_Hair_Beverages.py`), and Stage B (`NR_final_PW2.py`).
- Formula: `Association_Score = (W1*S_pack + W2*S_subcat + W3*S_brand + W4*S_variant + W5*S_size + W6*S_pkg) / sum(W_k), where W1 = (W4 * W5) / 100`.
- Compares 4 concrete scenarios on `gt_hul_sessions` (100 store visit strips) and `ms_json_golden` (100 LabelMe store shelves).

## 2. Scenario Comparison (`gt_hul_sessions` — 100 Store Visits)

| Scenario | Usable P2P Facings | Usable Share (%) | Unique Brands | Unique Basepacks | Mean $W_1$ Weight (%) | Mean Association Score |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. Legacy `.h5` Baseline | 673 | 84.1% | 102 | 165 | 34.82% | 0.2577 |
| 2. Ours (Audited Crops Drop-in) | 800 | 100.0% | 108 | 338 | 73.68% | 0.2683 |
| 3. Ours (+ Competitor `'Others'` Recovery) | 800 | 100.0% | 108 | 338 | 73.68% | 0.2683 |
| 4. Ours (+ Full-Shelf Unannotated Enrichment) | 800 | 100.0% | 108 | 338 | 73.68% | 0.2683 |

## 3. Scenario Comparison (`ms_json_golden` — 100 Modern Trade Store Shelves)

| Scenario | Usable P2P Facings | Usable Share (%) | Unique Brands | Unique Basepacks | Mean $W_1$ Weight (%) | Mean Association Score |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. Legacy `.h5` Baseline | 2,223 | 59.9% | 144 | 364 | 34.82% | 0.6167 |
| 2. Ours (Audited Crops Drop-in) | 2,718 | 73.3% | 147 | 606 | 73.68% | 0.6293 |
| 3. Ours (+ Competitor `'Others'` Recovery) | 3,709 | 100.0% | 149 | 614 | 73.68% | 0.6555 |
| 4. Ours (+ Full-Shelf Unannotated Enrichment) | 6,012 | 100.0% | 149 | 614 | 73.68% | 0.6581 |

## 4. Key Incremental Business Numbers
- **Basepack Trust Weight ($W_1 = W_4 \times W_5 / 100$)**: Rises by **+38.86 pts** (**2.12x** higher trust in exact SKU adjacency).
- **Model Upgrade on Audited Boxes (`gt_hul_sessions`)**: **+18.9%** more usable P2P facings (eliminating `Model_Output` drop-offs).
- **Competitor `'Others'` Recovery (`ms_json_golden`)**: **+36.5%** more usable facings by resolving `Others____Others____NA` into named competitor SKUs instead of dropping them in `NR_Brand_Hair_Beverages.py`.
- **Full-Shelf Enrichment (`ms_json_golden`)**: **+170.4%** total usable facings vs legacy `.h5` (adding **+5** brands and **+250** basepacks across unannotated shelf rows).
