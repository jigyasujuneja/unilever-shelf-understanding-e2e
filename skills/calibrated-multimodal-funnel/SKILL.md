---
name: calibrated-multimodal-funnel
description: >-
  Architecture and calibration playbook for large-catalog (1,000 to 10,000+ classes)
  visual recognition using the Multimodal JEV (CLM + CLEF) funnel: Stage 0 Dual-Buffer
  Photometry, Stage 2A Multi-Prototype InfoNCE + Residual Adapter fast-gating, Stage 2B
  Self-Supervised Saliency-Diff and Crown Contact Sheets, Pre-Choice Evidence Decoding
  (PROP-08), Bayesian PoE logprob fusion (PROP-09), and Layer-5 deterministic pack-size
  post-verification.
---

# Calibrated Multimodal Funnel Playbook (`Multimodal JEV`: `CLM` + `CLEF`)

## 1. When to Use This Architecture

Use this 4-stage visual decision funnel whenever classifying or retrieving fine-grained visual entities across `1,000-10,000+` catalog classes under strict latency (`<3s` p95) and unit-cost (`<$0.002/scene`) budgets, where:
- Closed-set classifiers fail on zero-shot product launches (`0%` unseen accuracy without retraining).
- Single-pass full-scene VLMs suffer coordinate drift (`~51%` F2 on `30+` items) and high token cost.
- Zero-token VLM choice heads collapse to 50/50 guessing on lookalike sister variants differing only in `<10px` weight or pack-type typography.

## 2. Stage 0: Dual-Buffer Photometry (`EXP-40`)

Never feed a single contrast-stretched image buffer to both an embedding model and an OCR/VLM verifier:
1. **Illuminant Normalization (Minkowski $p=6$ Shades-of-Gray)**:
   - Estimate channel illuminant $\mu_c = \left(\frac{1}{HW}\sum_{x,y} I_c(x,y)^p\right)^{1/p}$ with $p=6$ so specular shows set the white reference without washing out mid-tones.
2. **Dual Image Buffers**:
   - **Buffer A (`color_image`, Unsharpened Linear RGB)**: Route to Stage 2A embedding retrieval and inner-80% CIELAB color scoring. Applying `CLAHE` before Stage 2A warps brand color ratios and drops retrieval accuracy by `-6.0 pp`.
   - **Buffer B (`detector_image`, CIELAB `L* CLAHE` + Unwarp)**: Route to Stage 1 edge detection and Stage 2B contact sheets (`+0.8 pp` micro-text OCR gain).

## 3. Stage 2A: Multi-Prototype InfoNCE, Residual Adapter & Fast-Gate (`CLM`)

1. **Residual InfoNCE Adapter (`EXP-32`)**:
   - Project frozen `gemini-embedding-2` vectors via $e' = \text{norm}(q + \alpha W_2 \text{GELU}(W_1 q))$ ($\alpha = 0.35$) to separate sister variants while preserving zero-shot brand geometry.
2. **Unstarved Multi-Prototype Brand-Gate (`EXP-54`)**:
   - Combine $k=4$ medoid prototypes per SKU with a global Top-2K floor so tail brands are never starved by head-brand hubness (`CSLS` local intrinsic dimensionality correction).
3. **Lidstone-Smoothed 3D Container Prior (`EXP-52`)**:
   - Score 16 physical container forms using aspect ratio $h/w$, top-crown taper, and shelf-plank height ratio $h/h_{\text{plank}}$ smoothed with Lidstone $\epsilon = 0.01$: $-\lambda \ln P_{\text{Lidstone}}(\text{pack})$.
4. **Adaptive Fast-Exit Gate (`60%-68%` Embedding Match Without Generative VLM Decoding)**:
   - Auto-accept Stage 2A Rank-1 when top-1/top-2 probability margin $P_1 - P_2 \ge 0.14$ and $Z \ge 1.65$ (and top candidates do not collide on sister pack sizes).

## 4. Stage 2B: Saliency-Diff & Crown Contact Sheets + Calibrated Readout (`CLEF`)

For the `32%-40%` ambiguous crops escalated from Stage 2A, build a single composite visual comparison sheet:
1. **3 Targeted High-Resolution Zoom Strips**:
   - **Bottom-30% Net-Weight Strip (`EXP-09`)**: $2\times$ Lanczos zoom on the bottom 30% of the crop where `ml`/`g` text is printed.
   - **Self-Supervised Saliency-Diff Strip (`EXP-51`)**: Stack aligned Top-$K$ studio references $R_1,\dots,R_K$ and compute pixel variance $V(y) = \text{Mean}_{x,c}(\text{Var}_k(R_k(x,y,c)))$ to automatically zoom $2.5\times$ into the exact horizontal band where sibling candidates differ most.
   - **Top-22% Crown/Nozzle Strip (`EXP-52`)**: Include whenever top candidates disagree on `packaging_type` (`Bottle` vs. `Jar` vs. `Spray` vs. `Tube`).
   - **Cylindrical Tube Unwarping (`EXP-21`)**: For squeeze tubes, apply inverse-sine horizontal unwarping $x_{\text{src}}(\theta) = \frac{W-1}{2}\left(\frac{\sin\theta}{\sin(\pi/3)} + 1\right)$ (`1.15x` stretch) so curved edge letters unroll flat.
2. **Causal Attention Evidence-First Schema (`PROP-08`, `+8.9 pp`)**:
   - Require the JSON schema to emit `"pack_size_read"` (3-5 tokens, e.g. `"340ml"`) **before** `"choice"` (integer `1..K` or `0`). Under a lower-triangular causal mask, `"choice"` attends directly to the decoded weight string in the KV cache.
   - Pin static rubric instructions and schema before dynamic crop images (`88.4%` KV-cache prefix hit rate).
3. **Shannon Entropy Confidence $C_2$ & Bayesian PoE Fusion (`PROP-09`)**:
   - Compute $C_2 = 1 - \frac{H(p)}{\ln(K+1)}$. When $C_2 < 0.35$, fuse VLM token probabilities with Stage 2A visual priors via $P_{\text{final}}(k) \propto P_{\text{token}}(k)^{0.62} \cdot P_{\text{vis}}(k)^{0.38}$.
4. **Three Deterministic Post-Generation Guards (`PR-14`, `EXP-58`)**:
   - **Precedence 1 (Layer-5 Pack-Size Sister Switch)**: If `pack_size_read` (`"340ml"`) contradicts the chosen panel (`180ml`) and a sister panel in the shortlist has `"340ml"`, deterministically switch to the matching sister panel.
   - **Precedence 2 (`0.045` Visual Energy Veto)**: If Stage 2B flips Stage 2A Rank-1 (`k* != 1`) with low token confidence ($C_2 < 0.55$) and visual energy drop $\Delta E = E(1) - E(k^*) > 0.045$, veto the flip and retain Rank-1.
   - **Precedence 3 (`choice=0` Competitor Guard)**: If `choice == 0`, retain Rank-1 if visual similarity $\ge 0.72$; otherwise classify as an unlisted competitor.
