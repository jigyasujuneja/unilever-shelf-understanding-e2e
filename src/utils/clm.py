"""Pure Mathematical Primitives for Z-Score + LID Gating & CLM InfoNCE Energy Scoring.

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Input: Top-K candidate cosines [s_1 >= s_2 >= ... >= s_K], query vector   │
  │         q, inner-80% CIELAB descriptor q_lab, and SKU centroid vectors.    │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  1. compute_zscore_and_lid(scores)                                         │
  │     • Scale-invariant Z-margin:  z_margin = (s_1 - s_2) / (sigma_topK+eps) │
  │     • Local Intrinsic Dim (LID): 1 / mean(|ln((1 - s_k) / (1 - s_1))|)     │
  │  2. infonce_rerank(...) & infonce_posterior(ranked, inv_temp)              │
  │     • Joint energy: E_k = 0.78*cos_k + 0.22*cent_cos_k + w_lab*(s_lab-0.5) │
  │     • Softmax posterior: P_CLM(y_k|x) = exp((s_k - s_1)/tau) / Z           │
  │  3. zoom_sheet / zoom_multi_sheet                                          │
  │     • Contact-sheet builder with optional Bottom-30% pack-size zoom strip. │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.openjev import jev_entropy_confidence
from utils.photometry import lab_similarity

CELL = 256
DEFAULT_K = 5


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=False)))


def compute_zscore_and_lid(scores: list[float]) -> dict[str, float]:
    """Compute scale-invariant Z-score margin, Z-score top-1, and Amsaleg MLE Local Intrinsic Dimensionality (LID).

    Regularizes the top-K score standard deviation with a minimum cosine noise floor
    ``sigma_eff = sqrt(var(scores) + 1e-4)`` (>= 0.01 cosine units) so tied runner-ups
    (e.g. [0.751, 0.750, 0.750, 0.750, 0.750]) cannot artificially inflate ``z_margin``.
    Computes Amsaleg et al.'s Maximum Likelihood Estimator for LID anchored at the K-th
    neighbor distance ``d_K = max(1e-5, 1.0 - s_K)``.
    """
    if not scores:
        return {"z_margin": 0.0, "z_top1": 0.0, "lid": 10.0}
    s1 = scores[0]
    s2 = scores[1] if len(scores) > 1 else s1 - 0.1
    margin = s1 - s2
    mu = float(np.mean(scores))
    sd = math.sqrt(float(np.var(scores)) + 1e-4)
    z_margin = margin / sd
    z_top1 = (s1 - mu) / sd
    if len(scores) >= 2:
        d_all = [max(1e-5, 1.0 - float(s)) for s in scores]
        d_k = max(d_all[-1], d_all[0] + 1e-6)
        log_ratios = [max(0.0, math.log(max(d_k / d, 1.0))) for d in d_all[:-1]]
        mean_log_ratio = float(np.mean(log_ratios)) if log_ratios else 0.0
        lid = float(min(50.0, 1.0 / (mean_log_ratio + 1e-4)))
    else:
        lid = 1.0
    return {
        "z_margin": round(z_margin, 4),
        "z_top1": round(z_top1, 4),
        "lid": round(lid, 4),
    }


def compute_csls_hubness(index_matrix: np.ndarray, k_nn: int = 10) -> np.ndarray:
    """Compute Cross-Domain Similarity Local Scaling (CSLS) mean k-NN similarity per gallery vector.

    Penalizes high-dimension 'universal hub' reference images that appear in the top-K shortlist
    of unrelated shelf crops (Conneau et al., ICLR 2018).
    """
    n = index_matrix.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    k_eff = min(k_nn, n - 1)
    sim = index_matrix @ index_matrix.T
    np.fill_diagonal(sim, -np.inf)
    part = np.partition(sim, -k_eff, axis=1)[:, -k_eff:]
    r_t = part.mean(axis=1).astype(np.float32)
    return r_t - float(r_t.mean())


def infonce_rerank(
    coarse: list[tuple[float, int, str]],
    q: list[float],
    q_lab: list[float],
    lab_index: dict[str, list[float]],
    sku_centroids: dict[int, list[float]],
    lab_weight: float = 0.035,
    k: int = DEFAULT_K,
    csls_penalties: dict[str, float] | None = None,
) -> list[tuple[float, int, str]]:
    """Rerank top-2K candidates using multi-view cosine + SKU centroid cosine + inner-80% CIELAB + optional CSLS."""
    if not coarse:
        return []
    head = []
    for cos, pid, path in coarse[: 2 * k]:
        ref_lab = lab_index.get(path)
        s_lab = lab_similarity(q_lab, ref_lab) if ref_lab else 0.5
        cent = sku_centroids.get(pid)
        cent_cos = _dot(q, cent) if cent is not None else cos
        hub_pen = 0.5 * float(csls_penalties.get(path, 0.0)) if csls_penalties else 0.0
        energy = 0.78 * (cos - hub_pen) + 0.22 * cent_cos + lab_weight * (s_lab - 0.5)
        head.append((energy, cos, pid, path))
    head.sort(reverse=True)
    return [(energy, pid, path) for energy, _, pid, path in head] + coarse[2 * k :]


def infonce_posterior(
    ranked: list[tuple[float, int, str]],
    inv_temp: float = 28.0,
    k: int = DEFAULT_K,
    open_set_threshold: float | None = 0.70,
) -> tuple[float, float]:
    """Return ``(top1_prob, entropy_confidence)`` under temperature-scaled InfoNCE softmax with open-set dustbin.

    Includes an open-set dustbin logit ``exp(inv_temp * (open_set_threshold - s_1))`` in the partition
    function so low-similarity out-of-distribution crops cannot produce high posterior confidence
    solely from a relative gap over weaker runner-ups.
    """
    if not ranked:
        return 0.0, 0.0
    top_k = ranked[:k]
    s1 = top_k[0][0]
    logits = [inv_temp * (s - s1) for s, _, _ in top_k]
    exps = [math.exp(max(-30.0, min(30.0, x))) for x in logits]
    dustbin = (
        math.exp(max(-30.0, min(30.0, inv_temp * (open_set_threshold - s1))))
        if open_set_threshold is not None
        else 0.0
    )
    z_sum = (sum(exps) + dustbin) or 1.0
    probs = [e / z_sum for e in exps]
    return round(probs[0], 4), jev_entropy_confidence(probs, n_options=len(top_k))


def zoom_sheet(query: Image.Image, refs: list[Image.Image]) -> Image.Image:
    """Single-row contact sheet where each cell includes a Bottom-30% Pack-Size Zoom Strip."""
    return zoom_multi_sheet([query], [refs])


def zoom_multi_sheet(
    queries: list[Image.Image], refs_per_query: list[list[Image.Image]], k: int = DEFAULT_K
) -> Image.Image:
    """Stack ``m`` query rows (Q1..Qm); each cell has upper full-pack view + bottom-30% zoom strip."""
    cols = max((len(r) for r in refs_per_query), default=k) + 1
    main_h, zoom_h = 176, 56
    row_h = 28 + main_h + zoom_h
    out = Image.new("RGB", (CELL * cols, row_h * len(queries)), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()

    for r_idx, (q, refs) in enumerate(zip(queries, refs_per_query, strict=True)):
        y0 = r_idx * row_h
        for c_idx, orig_im in enumerate([q, *refs]):
            x0 = c_idx * CELL
            im = orig_im.copy()
            im.thumbnail((CELL - 10, main_h - 6))
            out.paste(im, (x0 + (CELL - im.width) // 2, y0 + 26 + (main_h - im.height) // 2))

            ow, oh = max(1, orig_im.width), max(1, orig_im.height)
            band = orig_im.crop((int(ow * 0.10), int(oh * 0.65), int(ow * 0.90), int(oh * 0.96)))
            band.thumbnail((CELL - 8, zoom_h - 4))
            zy = y0 + 26 + main_h + (zoom_h - band.height) // 2
            out.paste(band, (x0 + (CELL - band.width) // 2, zy))

            draw.line((x0 + 4, y0 + 26 + main_h - 1, x0 + CELL - 4, y0 + 26 + main_h - 1), fill="#ddd")
            draw.rectangle((x0, y0, x0 + CELL - 1, y0 + row_h - 1), outline="#999")
            label = ("Q" if len(queries) == 1 else f"Q{r_idx + 1}") if c_idx == 0 else str(c_idx)
            draw.text((x0 + 6, y0 + 2), label, fill="red" if c_idx == 0 else "black", font=font)
    return out


def zoom_sheet_midband(query: Image.Image, refs: list[Image.Image]) -> Image.Image:
    """Single-row contact sheet with Full Pack + Middle-30% Variant Banner Strip + Bottom-30% Pack-Size Strip."""
    return zoom_multi_sheet_midband([query], [refs])


def zoom_multi_sheet_midband(
    queries: list[Image.Image], refs_per_query: list[list[Image.Image]], k: int = DEFAULT_K
) -> Image.Image:
    """Stack ``m`` query rows (Q1..Qm); each cell has Full Pack + Middle-30% Variant Strip (with cylindrical tube unwarping) + Bottom-30% Weight Strip."""
    cols = max((len(r) for r in refs_per_query), default=k) + 1
    main_h, mid_h, bot_h = 148, 52, 48
    row_h = 28 + main_h + mid_h + bot_h
    out = Image.new("RGB", (CELL * cols, row_h * len(queries)), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()

    for r_idx, (q, refs) in enumerate(zip(queries, refs_per_query, strict=True)):
        y0 = r_idx * row_h
        for c_idx, orig_im in enumerate([q, *refs]):
            x0 = c_idx * CELL
            im = orig_im.copy()
            im.thumbnail((CELL - 10, main_h - 6))
            out.paste(im, (x0 + (CELL - im.width) // 2, y0 + 26 + (main_h - im.height) // 2))

            ow, oh = max(1, orig_im.width), max(1, orig_im.height)
            # Middle-30% variant sub-banner strip (0.33..0.65 height); unwarp cylindrical squeeze tubes
            mid_band = orig_im.crop((int(ow * 0.06), int(oh * 0.33), int(ow * 0.94), int(oh * 0.65)))
            if oh / max(1.0, float(ow)) >= 1.55:
                mid_band = mid_band.resize(
                    (max(16, int(mid_band.width * 1.25)), max(12, mid_band.height)),
                    Image.Resampling.BICUBIC,
                )
            mid_band.thumbnail((CELL - 8, mid_h - 4))
            my = y0 + 26 + main_h + (mid_h - mid_band.height) // 2
            out.paste(mid_band, (x0 + (CELL - mid_band.width) // 2, my))

            # Bottom-30% net-weight / grammage strip (0.65..0.96 height)
            bot_band = orig_im.crop((int(ow * 0.10), int(oh * 0.65), int(ow * 0.90), int(oh * 0.96)))
            bot_band.thumbnail((CELL - 8, bot_h - 4))
            by = y0 + 26 + main_h + mid_h + (bot_h - bot_band.height) // 2
            out.paste(bot_band, (x0 + (CELL - bot_band.width) // 2, by))

            draw.line((x0 + 4, y0 + 26 + main_h - 1, x0 + CELL - 4, y0 + 26 + main_h - 1), fill="#ddd")
            draw.line((x0 + 4, y0 + 26 + main_h + mid_h - 1, x0 + CELL - 4, y0 + 26 + main_h + mid_h - 1), fill="#ddd")
            draw.rectangle((x0, y0, x0 + CELL - 1, y0 + row_h - 1), outline="#999")
            label = ("Q" if len(queries) == 1 else f"Q{r_idx + 1}") if c_idx == 0 else str(c_idx)
            draw.text((x0 + 6, y0 + 2), label, fill="red" if c_idx == 0 else "black", font=font)
    return out


def calibrate_gallery_geometry(
    index: list[tuple[int, str, list[float]]],
    sku_centroids: dict[int, list[float]],
    *,
    target_system1_precision: float = 0.96,
    split: str = "train",
) -> dict[str, object]:
    """Two-Tier Self-Calibration of System 1 -> System 2 gates strictly from ``train`` gallery geometry.

    Tier A (Intrinsic ``train`` Gallery Geometry at ``setup()``):
      - Computes per-SKU closest-sibling centroid similarity ``s_sib(i) = max_{j != i} <c_i, c_j>``.
      - Derives scale-free InfoNCE inverse temperature ``inv_temp = clip(1.40 / median_i(1 - s_sib(i)), 18.0, 44.0)``.
      - Derives per-SKU local margin requirement ``req_margin(i) = clip(0.35 * (1 - s_sib(i)), 0.015, 0.055)``.
      - Computes Leave-One-View-Out (LOVO) intra-SKU cosine floor for 1D shelf-row spatial context ``[P, x_i, P]``.

    Tier B (Leave-One-View-Out Conformal Risk Quantile controlled by ``target_system1_precision``):
      - Evaluates LOVO non-conformity on multi-view ``train`` gallery references to calibrate ``conformal_min_prob``.
      - Strictly forbids ``split == 'test'`` to guarantee zero hyperparameter leakage.
    """
    if split == "test":
        raise ValueError("Zero-leakage invariant violated: calibrate_gallery_geometry cannot be fit on 'test' split")

    pids = sorted(sku_centroids.keys())
    if len(pids) < 4:
        return {
            "source_split": split,
            "target_system1_precision": target_system1_precision,
            "inv_temp": 28.0,
            "conformal_min_prob": 0.56,
            "intra_sku_cosine_floor": 0.86,
            "median_sibling_cosine": 0.80,
            "sku_sibling_cosine": {pid: 0.80 for pid in pids},
            "sku_required_margin": {pid: 0.026 for pid in pids},
        }

    mat = np.asarray([sku_centroids[pid] for pid in pids], dtype=np.float32)
    sim = mat @ mat.T
    np.fill_diagonal(sim, -np.inf)
    max_sib = sim.max(axis=1)

    sku_sib_cos: dict[int, float] = {}
    sku_req_margin: dict[int, float] = {}
    gaps: list[float] = []
    for idx_p, pid in enumerate(pids):
        s_sib = float(max(-1.0, min(0.999, float(max_sib[idx_p]))))
        gap = max(1e-3, 1.0 - s_sib)
        gaps.append(gap)
        sku_sib_cos[pid] = round(s_sib, 4)
        sku_req_margin[pid] = round(float(max(0.015, min(0.055, 0.35 * gap))), 4)

    med_gap = float(np.median(gaps)) if gaps else 0.05
    inv_temp = round(float(max(18.0, min(44.0, 1.40 / max(1e-3, med_gap)))), 2)
    med_sib_cos = round(1.0 - med_gap, 4)

    # Leave-One-View-Out (LOVO) intra-SKU cosine floor & Conformal posterior quantile
    by_sku: dict[int, list[np.ndarray]] = {}
    for pid, _, vec in index:
        by_sku.setdefault(pid, []).append(np.asarray(vec, dtype=np.float32))

    lovo_intra_cos: list[float] = []
    lovo_posteriors: list[float] = []
    for pid, v_list in by_sku.items():
        if len(v_list) < 2:
            continue
        v_sum = np.sum(v_list, axis=0)
        s_sib = sku_sib_cos.get(pid, med_sib_cos)
        for v in v_list:
            rest_mean = v_sum - v
            norm = float(np.linalg.norm(rest_mean))
            if norm <= 1e-8:
                continue
            rest_unit = rest_mean / norm
            pos_cos = float(np.dot(v, rest_unit))
            lovo_intra_cos.append(pos_cos)
            # 2-class LOVO conformal proxy against closest sibling centroid
            logit_diff = inv_temp * (pos_cos - s_sib)
            p_lovo = 1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, logit_diff))))
            lovo_posteriors.append(p_lovo)

    if lovo_intra_cos:
        intra_floor = round(float(max(0.82, min(0.92, float(np.percentile(lovo_intra_cos, 15))))), 4)
    else:
        intra_floor = 0.86

    alpha = max(0.01, min(0.25, 1.0 - float(target_system1_precision)))
    if lovo_posteriors:
        q_prob = float(np.quantile(lovo_posteriors, alpha))
        conformal_min_prob = round(float(max(0.50, min(0.72, q_prob))), 4)
    else:
        conformal_min_prob = 0.56

    return {
        "source_split": split,
        "target_system1_precision": target_system1_precision,
        "inv_temp": inv_temp,
        "conformal_min_prob": conformal_min_prob,
        "intra_sku_cosine_floor": intra_floor,
        "median_sibling_cosine": med_sib_cos,
        "sku_sibling_cosine": sku_sib_cos,
        "sku_required_margin": sku_req_margin,
    }


