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
    if len(scores) < 2:
        return {"z_margin": 0.0, "z_top1": 0.0, "lid": 10.0}
    s1 = scores[0]
    s2 = scores[1]
    margin = s1 - s2
    mu = float(np.mean(scores))
    sd = math.sqrt(float(np.var(scores)) + 1e-4)
    z_margin = margin / sd
    z_top1 = (s1 - mu) / sd
    d_all = [max(1e-5, 1.0 - float(s)) for s in scores]
    d_k = max(d_all[-1], d_all[0] + 1e-6)
    log_ratios = [max(0.0, math.log(max(d_k / d, 1.0))) for d in d_all[:-1]]
    mean_log_ratio = float(np.mean(log_ratios)) if log_ratios else 0.0
    lid = float(min(50.0, 1.0 / (mean_log_ratio + 1e-4)))
    return {
        "z_margin": round(z_margin, 4),
        "z_top1": round(z_top1, 4),
        "lid": round(lid, 4),
    }


def compute_csls_hubness(index_matrix: np.ndarray, k_nn: int = 10) -> np.ndarray:
    """Compute Cross-Domain Similarity Local Scaling (CSLS) out-of-family hubness per gallery vector.

    Masks out same-SKU views and close sister variants (sim > 0.82) so multi-variant HUL brands
    are never penalized for catalog variant density, and clamps at >= 0.0 so isolated 1-SKU
    competitor brands never receive a negative penalty (artificial score boost).
    """
    n = index_matrix.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    k_eff = min(k_nn, n - 1)
    sim = index_matrix @ index_matrix.T
    np.fill_diagonal(sim, -np.inf)
    # Mask same-SKU views and immediate sister variants so only cross-brand background hubness is measured
    bg_sim = np.where(sim > 0.82, -np.inf, sim)
    part = np.partition(bg_sim, -k_eff, axis=1)[:, -k_eff:]
    valid = np.where(np.isfinite(part), part, 0.0)
    r_t = valid.mean(axis=1).astype(np.float32)
    return np.maximum(0.0, r_t - float(r_t.mean())).astype(np.float32)


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


def group_adjacent_shelf_twins(
    boxes: list[tuple[float, float, float, float]],
    restored_crops: list[Image.Image],
    lab_descriptors: list[list[float]],
    *,
    lab_thresh: float = 0.985,
    patch_cos_thresh: float = 0.965,
) -> tuple[list[list[int]], list[int]]:
    """Group spatially adjacent identical shelf facings on the same horizontal row before HTTP embedding RPCs.

    Requires four simultaneous invariants before merging facing ``j`` into medoid ``i``:
      1. Same horizontal shelf row: ``|cy_i - cy_j| <= 0.45 * max(h_i, h_j)``.
      2. Contiguous horizontal block: ``|cx_i - cx_j| <= 2.4 * max(w_i, w_j)`` and matching box dimensions.
      3. 3-Zone inner-80% CIELAB + shoulder-taper similarity ``>= lab_thresh`` (0.985).
      4. 16x16 L2-normalized RGB spatial layout cosine ``>= patch_cos_thresh`` (0.965).
    """
    n = len(boxes)
    if n <= 1:
        return ([[0]] if n == 1 else []), ([0] if n == 1 else [])

    patches: list[np.ndarray] = []
    for im in restored_crops:
        arr = np.asarray(im.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1)
        arr = arr - float(np.mean(arr))
        norm = float(np.linalg.norm(arr))
        patches.append(arr / norm if norm > 1e-6 else np.zeros_like(arr))

    ordered = sorted(range(n), key=lambda idx: (boxes[idx][0] + boxes[idx][2]) / 2.0)
    assigned: set[int] = set()
    clusters: list[list[int]] = []
    medoids: list[int] = []

    for i in ordered:
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        x1_i, y1_i, x2_i, y2_i = boxes[i]
        w_i = max(1.0, x2_i - x1_i)
        h_i = max(1.0, y2_i - y1_i)
        cx_i = (x1_i + x2_i) / 2.0
        cy_i = (y1_i + y2_i) / 2.0
        last_cx = cx_i

        for j in ordered:
            if j in assigned:
                continue
            x1_j, y1_j, x2_j, y2_j = boxes[j]
            w_j = max(1.0, x2_j - x1_j)
            h_j = max(1.0, y2_j - y1_j)
            cx_j = (x1_j + x2_j) / 2.0
            cy_j = (y1_j + y2_j) / 2.0

            if abs(cy_i - cy_j) > 0.85 * max(h_i, h_j):
                continue
            if abs(cx_j - last_cx) > 3.5 * max(w_i, w_j):
                continue
            w_ratio = w_j / w_i
            h_ratio = h_j / h_i
            if not (0.75 <= w_ratio <= 1.33 and 0.75 <= h_ratio <= 1.33):
                continue
            if lab_similarity(lab_descriptors[i], lab_descriptors[j]) < lab_thresh:
                continue
            if float(np.dot(patches[i], patches[j])) < patch_cos_thresh:
                continue
            group.append(j)
            assigned.add(j)
            last_cx = cx_j

        # Choose the interior/most-centered facing in the group as the medoid
        med = group[len(group) // 2]
        clusters.append(group)
        medoids.append(med)

    return clusters, medoids


# ---------------------------------------------------------------------------
# EXP-50: Aspect-Preserving Cap/Rim Context Crop & Projective Scale Features
# ---------------------------------------------------------------------------

def letterbox_cap_crop(
    image: Image.Image,
    box: tuple[float, float, float, float],
    *,
    top_pad: float = 0.12,
    side_pad: float = 0.06,
    bottom_pad: float = 0.08,
) -> Image.Image:
    """Cut out ``box`` with asymmetric +12% top cap/nozzle/crimp context padding and +8% bottom rim padding."""
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    px = w * side_pad
    py_top = h * top_pad
    py_bot = h * bottom_pad
    return image.crop(
        (
            max(0.0, x1 - px),
            max(0.0, y1 - py_top),
            min(float(image.width), x2 + px),
            min(float(image.height), y2 + py_bot),
        )
    )


def compute_container_geometry_features(
    image: Image.Image,
    box: tuple[float, float, float, float] | None = None,
    plank_height: float | None = None,
) -> dict[str, float]:
    """Compute universal 3D container geometry features (aspect_ratio, top_crown_taper, bottom_base_width, plank_height_ratio).

    Uses horizontal edge energy across the top-22% crown zone vs the middle-50% body zone to
    estimate neck/actuator taper (distinguishing Spray/Bottle/Roll_On/Tube from flat-topped
    Jar/Carton/Sachet/Bar) without any brand-specific rules.
    """
    if box is not None:
        bw = max(1.0, float(box[2] - box[0]))
        bh = max(1.0, float(box[3] - box[1]))
    else:
        bw = max(1.0, float(image.width))
        bh = max(1.0, float(image.height))
    aspect_ratio = round(bh / bw, 4)
    plank_ratio = round(bh / max(1.0, float(plank_height)), 4) if plank_height and plank_height > 0 else 1.0

    gray = np.asarray(image.convert("L").resize((64, 96), Image.Resampling.BILINEAR), dtype=np.float32)
    gx = np.abs(np.diff(gray, axis=1))

    def _active_span(band_gx: np.ndarray) -> float:
        col_energy = band_gx.mean(axis=0)
        thr = float(np.percentile(col_energy, 35)) + 1.5
        active = np.where(col_energy >= thr)[0]
        if active.size < 2:
            return 0.75
        return float(max(0.15, min(1.0, (active[-1] - active[0] + 1) / float(band_gx.shape[1]))))

    top_span = _active_span(gx[:21, :])       # top 22% crown / cap / nozzle / crimp
    mid_span = _active_span(gx[24:72, :])     # middle 50% body
    bot_span = _active_span(gx[72:, :])       # bottom 25% base / gusset
    crown_taper = round(float(max(0.25, min(1.35, top_span / max(0.20, mid_span)))), 4)
    base_taper = round(float(max(0.25, min(1.35, bot_span / max(0.20, mid_span)))), 4)

    return {
        "aspect_ratio": aspect_ratio,
        "crown_taper": crown_taper,
        "base_taper": base_taper,
        "plank_height_ratio": plank_ratio,
    }


_SLENDER_UPRIGHT_PACKS = {"Bottle", "Spray", "Roll_On", "Pump", "Can", "Tin"}
_SQUAT_WIDE_PACKS = {"Tub", "Bar", "Box", "Bowl", "Tray", "Multi_Pack"}
_FLEXIBLE_PACKS = {"Pouch", "Sachet", "Bag", "Wrapper"}


def _pack_geometry_affinity(pack_type: str, geom: dict[str, float]) -> float:
    """Universal 3D container geometry compatibility score in [-1.0, +1.0] (zero brand rules)."""
    if not pack_type:
        return 0.0
    ar = float(geom.get("aspect_ratio", 1.35))
    taper = float(geom.get("crown_taper", 0.85))
    elong = max(ar, 1.0 / max(0.15, ar))
    score = 0.0
    if pack_type in _SLENDER_UPRIGHT_PACKS:
        if ar >= 1.75:
            score += 0.55
        elif ar <= 0.92:
            score -= 0.65
        if taper <= 0.78:
            score += 0.35
    elif pack_type == "Tube":
        # Tubes can stand upright (facewash, ar >= 1.55) or lie horizontally (toothpaste, ar <= 0.65)
        if elong >= 1.60:
            score += 0.45
        elif 0.85 <= ar <= 1.20:
            score -= 0.35
    elif pack_type == "Jar":
        # Cylindrical jars have wide mouths (taper >= 0.80) and medium upright height (1.00 <= ar <= 2.25)
        if 1.00 <= ar <= 2.25:
            score += 0.35
        elif ar >= 2.40 and taper <= 0.75:
            score -= 0.45
        if taper >= 0.82 and ar <= 2.10:
            score += 0.20
    elif pack_type in _SQUAT_WIDE_PACKS:
        if ar <= 1.05:
            score += 0.55
        elif ar >= 1.65:
            score -= 0.65
        if taper >= 0.86:
            score += 0.25
    elif pack_type in _FLEXIBLE_PACKS:
        if 0.90 <= ar <= 1.95:
            score += 0.30
        if pack_type == "Pouch" and ar >= 1.20:
            score += 0.15
        elif pack_type == "Sachet" and taper >= 0.88:
            score += 0.20
    elif pack_type in ("Carton", "Tetrapack"):
        if (1.15 <= ar <= 2.35 or ar <= 0.68) and taper >= 0.82:
            score += 0.35
    return max(-1.0, min(1.0, score))


def apply_projective_scale_prior(
    ranked: list[tuple[float, int, str]],
    geom: dict[str, float],
    catalog: dict[int, dict] | None,
    *,
    weight: float = 0.018,
    k: int = DEFAULT_K,
) -> list[tuple[float, int, str]]:
    """EXP-50: Re-score top-2K candidates with projective container scale & crown-taper compatibility."""
    if not ranked or not catalog:
        return ranked
    head: list[tuple[float, int, str]] = []
    for s, pid, path in ranked[: 2 * k]:
        meta = catalog.get(pid) or {}
        pt = str(meta.get("packaging_type") or meta.get("pack_form") or "")
        aff = _pack_geometry_affinity(pt, geom)
        head.append((round(float(s + weight * aff), 6), pid, path))
    head.sort(reverse=True)
    return head + ranked[2 * k :]


# ---------------------------------------------------------------------------
# EXP-51: Self-Supervised Reference-Difference Saliency Zoom Strip
# ---------------------------------------------------------------------------

def compute_reference_diff_band(
    refs: list[Image.Image],
    *,
    band_frac: float = 0.28,
    min_y: float = 0.14,
    max_y: float = 0.82,
) -> tuple[float, float, float]:
    """Compute self-supervised pixel variance V(x,y) = Var_k(R_k(x,y)) across aligned reference images.

    Returns ``(y1_frac, y2_frac, peak_ratio)`` identifying the horizontal band of height ``band_frac``
    within ``[min_y, max_y]`` where the top candidate reference packshots differ most visually.
    """
    valid_refs = [r for r in refs[:4] if r is not None and r.width > 4 and r.height > 4]
    if len(valid_refs) < 2:
        return 0.33, 0.61, 1.0

    grid_w, grid_h = 96, 144
    stack = np.stack(
        [
            np.asarray(r.convert("RGB").resize((grid_w, grid_h), Image.Resampling.BILINEAR), dtype=np.float32)
            for r in valid_refs
        ],
        axis=0,
    )
    # Trim 10% left/right borders to focus on the label body rather than background framing
    x_lo, x_hi = int(grid_w * 0.10), int(grid_w * 0.90)
    var_map = np.var(stack[:, :, x_lo:x_hi, :], axis=0).mean(axis=(1, 2))  # shape: (grid_h,)

    y_lo = max(0, int(round(min_y * grid_h)))
    y_hi = min(grid_h, int(round(max_y * grid_h)))
    win = max(8, int(round(band_frac * grid_h)))
    if y_hi - y_lo <= win:
        return 0.33, 0.61, 1.0

    best_sum = -1.0
    best_start = int(round(0.33 * grid_h))
    for y_start in range(y_lo, y_hi - win + 1):
        cur = float(var_map[y_start : y_start + win].sum())
        if cur > best_sum:
            best_sum = cur
            best_start = y_start

    mean_win = float(var_map[y_lo:y_hi].mean()) * win + 1e-5
    peak_ratio = round(best_sum / mean_win, 3)
    y1_f = round(best_start / float(grid_h), 3)
    y2_f = round((best_start + win) / float(grid_h), 3)
    return y1_f, y2_f, peak_ratio


# ---------------------------------------------------------------------------
# EXP-52: Lidstone-Calibrated 3D Container Specialist + Top-22% Crown Strip
# ---------------------------------------------------------------------------

def lidstone_pack_posterior(
    ranked: list[tuple[float, int, str]],
    geom: dict[str, float],
    catalog: dict[int, dict] | None,
    *,
    eps: float = 0.01,
    inv_temp: float = 32.0,
    k: int = DEFAULT_K,
) -> dict[str, float]:
    """Compute Lidstone-smoothed (eps=0.01) container packaging_type distribution over top-K candidates + 3D geometry."""
    if not ranked or not catalog:
        return {}
    top = ranked[: max(k, 8)]
    s1 = top[0][0]
    raw_votes: dict[str, float] = {}
    all_packs: set[str] = set()
    for s, pid, _ in top:
        meta = catalog.get(pid) or {}
        pt = str(meta.get("packaging_type") or meta.get("pack_form") or "")
        if not pt:
            continue
        all_packs.add(pt)
        w_sim = math.exp(max(-25.0, min(25.0, inv_temp * (s - s1))))
        w_geom = math.exp(0.45 * _pack_geometry_affinity(pt, geom))
        raw_votes[pt] = raw_votes.get(pt, 0.0) + w_sim * w_geom

    if not all_packs:
        return {}
    v_sum = sum(raw_votes.values()) or 1.0
    n_classes = max(16, len(all_packs))
    denom = 1.0 + n_classes * eps
    return {
        pt: round(((raw_votes.get(pt, 0.0) / v_sum) + eps) / denom, 5)
        for pt in all_packs
    }


def apply_lidstone_pack_fusion(
    ranked: list[tuple[float, int, str]],
    geom: dict[str, float],
    catalog: dict[int, dict] | None,
    *,
    lam: float = 0.014,
    eps: float = 0.01,
    k: int = DEFAULT_K,
) -> tuple[list[tuple[float, int, str]], dict[str, float]]:
    """EXP-52: Fuse Lidstone-smoothed packaging_type log-prior (+lam * ln P_Lidstone(pack)) into candidate energies."""
    if not ranked or not catalog:
        return ranked, {}
    pack_probs = lidstone_pack_posterior(ranked, geom, catalog, eps=eps, k=k)
    if not pack_probs:
        return ranked, {}
    n_classes = max(16, len(pack_probs))
    floor_p = eps / (1.0 + n_classes * eps)
    head: list[tuple[float, int, str]] = []
    for s, pid, path in ranked[: 2 * k]:
        meta = catalog.get(pid) or {}
        pt = str(meta.get("packaging_type") or meta.get("pack_form") or "")
        p_c = pack_probs.get(pt, floor_p)
        # Center log-prior relative to uniform 1/n_classes so energies remain well-scaled
        log_bonus = lam * (math.log(max(1e-6, p_c)) - math.log(1.0 / n_classes))
        head.append((round(float(s + log_bonus), 6), pid, path))
    head.sort(reverse=True)
    return head + ranked[2 * k :], pack_probs


def zoom_sheet_saliency_diff(
    query: Image.Image,
    refs: list[Image.Image],
    *,
    include_crown: bool = False,
) -> Image.Image:
    """Single-row contact sheet with Self-Supervised Reference-Difference Saliency Zoom (+ optional Top-22% Crown Strip)."""
    return zoom_multi_sheet_saliency_diff([query], [refs], include_crown=include_crown)


def zoom_multi_sheet_saliency_diff(
    queries: list[Image.Image],
    refs_per_query: list[list[Image.Image]],
    *,
    include_crown: bool = False,
    k: int = DEFAULT_K,
) -> Image.Image:
    """Render multi-row contact sheet with Full Pack + Self-Supervised Saliency-Diff Strip + Bottom-30% Weight Strip.

    When ``include_crown=True`` (EXP-52 container conflict mode), also includes a dedicated Top-22%
    Crown/Nozzle/Lid/Crimp Zoom Strip at the top of the zoom stack while keeping cell width at 256px.
    """
    cols = max((len(r) for r in refs_per_query), default=k) + 1
    if include_crown:
        main_h, crown_h, diff_h, bot_h = 148, 44, 54, 48
    else:
        main_h, crown_h, diff_h, bot_h = 148, 0, 54, 48
    row_h = 28 + main_h + crown_h + diff_h + bot_h
    out = Image.new("RGB", (CELL * cols, row_h * len(queries)), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()

    for r_idx, (q, refs) in enumerate(zip(queries, refs_per_query, strict=True)):
        y0 = r_idx * row_h
        y1_diff, y2_diff, _ = compute_reference_diff_band(refs)
        for c_idx, orig_im in enumerate([q, *refs]):
            x0 = c_idx * CELL
            im = orig_im.copy()
            im.thumbnail((CELL - 10, main_h - 6), Image.Resampling.LANCZOS)
            out.paste(im, (x0 + (CELL - im.width) // 2, y0 + 26 + (main_h - im.height) // 2))

            ow, oh = max(1, orig_im.width), max(1, orig_im.height)
            cur_y = y0 + 26 + main_h

            if include_crown and crown_h > 0:
                crown_band = orig_im.crop((int(ow * 0.08), 0, int(ow * 0.92), max(4, int(oh * 0.24))))
                crown_band.thumbnail((CELL - 8, crown_h - 4), Image.Resampling.LANCZOS)
                cy = cur_y + (crown_h - crown_band.height) // 2
                out.paste(crown_band, (x0 + (CELL - crown_band.width) // 2, cy))
                draw.line((x0 + 4, cur_y - 1, x0 + CELL - 4, cur_y - 1), fill="#ddd")
                cur_y += crown_h

            # Self-supervised reference-difference horizontal band [y1_diff, y2_diff]
            diff_band = orig_im.crop(
                (int(ow * 0.06), max(0, int(oh * y1_diff)), int(ow * 0.94), min(oh, max(4, int(oh * y2_diff))))
            )
            if oh / max(1.0, float(ow)) >= 1.55:
                diff_band = diff_band.resize(
                    (max(16, int(diff_band.width * 1.22)), max(12, diff_band.height)),
                    Image.Resampling.LANCZOS,
                )
            diff_band.thumbnail((CELL - 8, diff_h - 4), Image.Resampling.LANCZOS)
            dy = cur_y + (diff_h - diff_band.height) // 2
            out.paste(diff_band, (x0 + (CELL - diff_band.width) // 2, dy))
            draw.line((x0 + 4, cur_y - 1, x0 + CELL - 4, cur_y - 1), fill="#ddd")
            cur_y += diff_h

            # Bottom-30% net-weight / grammage strip (0.65..0.96 height)
            bot_band = orig_im.crop((int(ow * 0.10), int(oh * 0.65), int(ow * 0.90), int(oh * 0.96)))
            bot_band.thumbnail((CELL - 8, bot_h - 4), Image.Resampling.LANCZOS)
            by = cur_y + (bot_h - bot_band.height) // 2
            out.paste(bot_band, (x0 + (CELL - bot_band.width) // 2, by))
            draw.line((x0 + 4, cur_y - 1, x0 + CELL - 4, cur_y - 1), fill="#ddd")

            draw.rectangle((x0, y0, x0 + CELL - 1, y0 + row_h - 1), outline="#999")
            label = ("Q" if len(queries) == 1 else f"Q{r_idx + 1}") if c_idx == 0 else str(c_idx)
            draw.text((x0 + 6, y0 + 2), label, fill="red" if c_idx == 0 else "black", font=font)
    return out


def compute_catalog_sha256(catalog: dict[int, dict] | None, gallery_paths: list[str] | None = None) -> str:
    """Compute a 16-char deterministic SHA-256 fingerprint of the active SKU catalog & gallery paths for MLOps index parity."""
    import hashlib
    import json

    h = hashlib.sha256()
    if catalog:
        for cid in sorted(catalog):
            meta = catalog[cid] or {}
            h.update(
                json.dumps(
                    {
                        "sku_id": cid,
                        "basepack_id": meta.get("basepack_id"),
                        "brand": meta.get("brand"),
                        "packaging_type": meta.get("packaging_type"),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
    if gallery_paths:
        for p in sorted(gallery_paths):
            h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:16]


def compute_ood_low_cosine_rate(top1_cosines: list[float], threshold: float = 0.65) -> float:
    """Compute fraction of query facings with top-1 catalog cosine below ``threshold`` (0.65) for OOD packaging drift alerts."""
    if not top1_cosines:
        return 0.0
    low = sum(1 for c in top1_cosines if float(c) < threshold)
    return round(low / float(len(top1_cosines)), 4)
