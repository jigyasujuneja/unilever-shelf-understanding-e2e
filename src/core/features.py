"""Visual crop feature extraction, CIELAB colorimetry, and MaxViT / Sub-ROI embeddings (`src/core/features.py`).

Strictly requires valid `PIL.Image.Image` or `np.ndarray` inputs; raises `ValueError` on `None` or
degenerate bounding boxes without synthetic fallback vectors.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

from core.models import load_maxvit_t_backbone


def _cosine_sim(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _rgb_to_cielab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert sRGB (0..255) to CIE L*a*b* (D65 illuminant)."""
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    rl = ((rn + 0.055) / 1.055) ** 2.4 if rn > 0.04045 else rn / 12.92
    gl = ((gn + 0.055) / 1.055) ** 2.4 if gn > 0.04045 else gn / 12.92
    bl = ((bn + 0.055) / 1.055) ** 2.4 if bn > 0.04045 else bn / 12.92
    x = (rl * 0.4124 + gl * 0.3576 + bl * 0.1805) / 0.95047
    y = (rl * 0.2126 + gl * 0.7152 + bl * 0.0722) / 1.00000
    z = (rl * 0.0193 + gl * 0.1192 + bl * 0.9505) / 1.08883

    def _f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    l_star = max(0.0, min(100.0, 116.0 * fy - 16.0))
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return (round(l_star, 2), round(a_star, 2), round(b_star, 2))


def _rgb_to_cielab_fast(rgb_0_255: np.ndarray) -> tuple[float, float, float]:
    """Convert mean `[R, G, B]` in `[0, 255]` to approximate CIELAB `(L*, a*, b*)`."""
    r, g, b = (float(rgb_0_255[0]) / 255.0, float(rgb_0_255[1]) / 255.0, float(rgb_0_255[2]) / 255.0)
    lum = max(1e-4, 0.2126 * r + 0.7152 * g + 0.0722 * b)
    l_star = 116.0 * (lum ** (1.0 / 3.0)) - 16.0
    a_star = 500.0 * ((max(1e-4, r) ** (1.0 / 3.0)) - (lum ** (1.0 / 3.0)))
    b_star = 200.0 * ((lum ** (1.0 / 3.0)) - (max(1e-4, b) ** (1.0 / 3.0)))
    return (round(l_star, 2), round(a_star, 2), round(b_star, 2))


def _delta_e_cie76(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """Compute perceptual color difference Delta-E between two CIELAB tuples."""
    return math.sqrt(
        (lab1[0] - lab2[0]) ** 2
        + (lab1[1] - lab2[1]) ** 2
        + (lab1[2] - lab2[2]) ** 2
    )


def compute_ciede2000_approx(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
) -> float:
    """Compute perceptual color difference between two CIE L*a*b* coordinates."""
    dl = float(lab1[0]) - float(lab2[0])
    da = float(lab1[1]) - float(lab2[1])
    db = float(lab1[2]) - float(lab2[2])
    c1 = math.hypot(float(lab1[1]), float(lab1[2]))
    c2 = math.hypot(float(lab2[1]), float(lab2[2]))
    dc = c1 - c2
    dh_sq = max(0.0, da * da + db * db - dc * dc)
    sl = 1.0
    sc = 1.0 + 0.045 * (c1 + c2) * 0.5
    sh = 1.0 + 0.015 * (c1 + c2) * 0.5
    de = math.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + dh_sq / (sh * sh))
    return round(de, 3)


def extract_real_crop_features(
    image: Image.Image,
    box: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Extract a real 64-D L2-normalized visual embedding and physical optical telemetry
    directly from the cropped pixels of `image` at `box = (x1, y1, x2, y2)`.
    """
    if image is None or not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError(f"extract_real_crop_features requires a valid PIL.Image, got {image!r}")
    if len(box) < 4 or float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
        raise ValueError(f"Invalid bounding box for extract_real_crop_features: {box}")

    w, h = image.size
    x1 = max(0, min(w - 1, int(round(box[0]))))
    y1 = max(0, min(h - 1, int(round(box[1]))))
    x2 = max(x1 + 1, min(w, int(round(box[2]))))
    y2 = max(y1 + 1, min(h, int(round(box[3]))))

    crop = image.crop((x1, y1, x2, y2)).convert("RGB").resize((24, 32))
    raw_bytes = crop.tobytes()
    n_px = 24 * 32

    zone_bounds = [(0, 6), (6, 21), (21, 32)]
    zone_rgb_means: list[tuple[float, float, float]] = []
    zone_rgb_stds: list[tuple[float, float, float]] = []
    zone_labs: list[tuple[float, float, float]] = []

    hist_r = [0.0] * 8
    hist_g = [0.0] * 8
    hist_b = [0.0] * 8
    glare_px = 0
    high_sat_px = 0

    for r_start, r_end in zone_bounds:
        rs, gs, bs = [], [], []
        for py in range(r_start, r_end):
            for px in range(24):
                idx = (py * 24 + px) * 3
                rv, gv, bv = float(raw_bytes[idx]), float(raw_bytes[idx + 1]), float(raw_bytes[idx + 2])
                rs.append(rv)
                gs.append(gv)
                bs.append(bv)
        cnt = max(1, len(rs))
        mr, mg, mb = sum(rs) / cnt, sum(gs) / cnt, sum(bs) / cnt
        sr = math.sqrt(sum((v - mr) ** 2 for v in rs) / cnt)
        sg = math.sqrt(sum((v - mg) ** 2 for v in gs) / cnt)
        sb = math.sqrt(sum((v - mb) ** 2 for v in bs) / cnt)
        zone_rgb_means.append((mr, mg, mb))
        zone_rgb_stds.append((sr, sg, sb))
        zone_labs.append(_rgb_to_cielab(mr, mg, mb))

    gx_sum = 0.0
    gy_sum = 0.0
    gray_grid = [[0.0] * 24 for _ in range(32)]
    row_fg_widths: list[float] = []

    bg_r, bg_g, bg_b = zone_rgb_means[0]
    for py in range(32):
        fg_in_row = 0
        for px in range(24):
            idx = (py * 24 + px) * 3
            rv, gv, bv = float(raw_bytes[idx]), float(raw_bytes[idx + 1]), float(raw_bytes[idx + 2])
            hist_r[min(7, int(rv // 32))] += 1.0
            hist_g[min(7, int(gv // 32))] += 1.0
            hist_b[min(7, int(bv // 32))] += 1.0
            lum = 0.299 * rv + 0.587 * gv + 0.114 * bv
            sat = max(rv, gv, bv) - min(rv, gv, bv)
            gray_grid[py][px] = lum
            if lum > 232.0 and sat < 18.0:
                glare_px += 1
            if sat > 35.0:
                high_sat_px += 1
            if abs(rv - bg_r) + abs(gv - bg_g) + abs(bv - bg_b) > 28.0 or sat > 24.0:
                fg_in_row += 1
        row_fg_widths.append(fg_in_row / 24.0)

    for py in range(1, 31):
        for px in range(1, 23):
            dx = abs(gray_grid[py][px + 1] - gray_grid[py][px - 1])
            dy = abs(gray_grid[py + 1][px] - gray_grid[py - 1][px])
            gx_sum += dx
            gy_sum += dy

    edge_norm = max(1.0, 30.0 * 22.0 * 255.0)
    gx_density = gx_sum / edge_norm
    gy_density = gy_sum / edge_norm
    glare_ratio = round(glare_px / float(n_px), 4)
    sat_ratio = round(high_sat_px / float(n_px), 4)

    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    aspect_ratio = bw / bh

    top_w = sum(row_fg_widths[:6]) / 6.0
    mid_w = max(0.15, sum(row_fg_widths[8:22]) / 14.0)
    neck_taper_ratio = round(min(1.2, max(0.25, top_w / mid_w)), 3)

    feat: list[float] = []
    for mr, mg, mb in zone_rgb_means:
        feat.extend([mr / 255.0, mg / 255.0, mb / 255.0])
    for sr, sg, sb in zone_rgb_stds:
        feat.extend([sr / 128.0, sg / 128.0, sb / 128.0])
    for l_s, a_s, b_s in zone_labs:
        feat.extend([l_s / 100.0, (a_s + 100.0) / 200.0, (b_s + 100.0) / 200.0])
    for h_bin in hist_r + hist_g + hist_b:
        feat.append(h_bin / float(n_px))

    feat.extend([
        gx_density * 4.0,
        gy_density * 4.0,
        glare_ratio,
        sat_ratio,
        min(2.0, aspect_ratio) / 2.0,
        min(2.0, 1.0 / max(0.2, aspect_ratio)) / 2.0,
        neck_taper_ratio,
        float(x1) / max(1.0, float(w)),
        float(y1) / max(1.0, float(h)),
        bw / max(1.0, float(w)),
        bh / max(1.0, float(h)),
        (zone_labs[0][0] - zone_labs[1][0]) / 100.0,
        (zone_labs[2][0] - zone_labs[1][0]) / 100.0,
    ])
    feat = feat[:64]
    while len(feat) < 64:
        feat.append(0.0)

    norm = math.sqrt(sum(v * v for v in feat)) or 1.0
    embedding = [round(v / norm, 6) for v in feat]
    h3_entropy = round(min(0.095, 0.012 + glare_ratio * 0.14 + (0.022 if 0.48 <= aspect_ratio <= 0.68 else 0.0)), 4)

    return {
        "embedding": embedding,
        "cap_lab": list(zone_labs[0]),
        "body_lab": list(zone_labs[1]),
        "claim_lab": list(zone_labs[2]),
        "glare_ratio": glare_ratio,
        "sat_ratio": sat_ratio,
        "gx_density": round(gx_density, 4),
        "gy_density": round(gy_density, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "neck_taper_ratio": neck_taper_ratio,
        "h3_entropy": h3_entropy,
    }


def _run_maxvit_t_latent_64d(clean_arr_24x16: np.ndarray) -> np.ndarray:
    """Execute a real forward pass through ``torchvision.models.maxvit_t`` stem + stage-1 Block/Grid attention
    to produce a 64-D L2-normalized neural feature vector. Hard-fails if model execution fails.
    """
    import torch
    import torch.nn.functional as F

    model = load_maxvit_t_backbone()
    tensor = torch.from_numpy(np.array(clean_arr_24x16, copy=True)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.inference_mode():
        x = model.stem(tensor)
        x = model.blocks[0](x)
        pooled = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    vec = pooled[0, :64].cpu().numpy().astype(np.float32)
    if vec.size < 64:
        padded = np.zeros(64, dtype=np.float32)
        padded[: vec.size] = vec
        vec = padded
    vec /= max(1e-6, float(np.linalg.norm(vec)))
    return vec


def _extract_subroi_from_array(
    full_arr: np.ndarray | None,
    img_w: int,
    img_h: int,
    box: tuple[float, float, float, float],
    feature_mode: str = "gemini_subroi",
) -> dict[str, Any]:
    """Fast vectorized crop feature extraction directly from pre-decoded image array."""
    if full_arr is None:
        raise ValueError("_extract_subroi_from_array requires a non-None image array; synthetic zero vectors are forbidden")
    if len(box) < 4 or float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
        raise ValueError(f"Invalid bounding box for _extract_subroi_from_array: {box}")

    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    aspect_ratio = bw / bh

    cx1 = max(0, min(img_w - 2, int(round(x1))))
    cy1 = max(0, min(img_h - 2, int(round(y1))))
    cx2 = max(cx1 + 2, min(img_w, int(round(x2))))
    cy2 = max(cy1 + 2, min(img_h, int(round(y2))))

    patch = full_arr[cy1:cy2, cx1:cx2]
    ph, pw = patch.shape[:2]
    yi = np.linspace(0, ph - 1, 24).astype(np.int32)
    xi = np.linspace(0, pw - 1, 16).astype(np.int32)
    arr = patch[np.ix_(yi, xi)].astype(np.float32)

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    sat = np.max(arr, axis=2) - np.min(arr, axis=2)
    glare_mask = (lum > 232.0) & (sat < 18.0)
    glare_ratio = float(np.mean(glare_mask))

    clean_arr = arr
    if glare_ratio > 0.01 and not np.all(glare_mask):
        clean_arr = arr.copy()
        non_glare_med = np.median(arr[~glare_mask], axis=0)
        clean_arr[glare_mask] = 0.25 * arr[glare_mask] + 0.75 * non_glare_med

    z0 = clean_arr[0:6, :, :]
    z1 = clean_arr[6:13, :, :]
    z2 = clean_arr[13:20, :, :]
    z3 = clean_arr[20:24, :, :]

    z_means = [z.mean(axis=(0, 1)) / 255.0 for z in (z0, z1, z2, z3)]
    z_stds = [z.std(axis=(0, 1)) / 128.0 for z in (z0, z1, z2, z3)]
    cap_lab = _rgb_to_cielab_fast(z0.mean(axis=(0, 1)))
    body_lab = _rgb_to_cielab_fast(z1.mean(axis=(0, 1)))
    sub_roi_lab = _rgb_to_cielab_fast(z2.mean(axis=(0, 1)))

    gx = np.abs(np.diff(lum, axis=1))
    gy = np.abs(np.diff(lum, axis=0))
    edge_density = float((gx.mean() + gy.mean()) / 255.0)
    top_edge = float(gx[:6, :].mean()) + 1e-3
    mid_edge = float(gx[7:17, :].mean()) + 1e-3
    neck_taper_ratio = float(np.clip(top_edge / mid_edge, 0.35, 1.45))

    hist_feats: list[float] = []
    for c_idx in range(3):
        h_counts, _ = np.histogram(clean_arr[:, :, c_idx], bins=8, range=(0.0, 256.0))
        hist_feats.extend((h_counts / 384.0).tolist())

    raw_vec = np.array(
        [
            *z_means[0].tolist(),
            *z_means[1].tolist(),
            *z_means[2].tolist(),
            *z_means[3].tolist(),
            *z_stds[0].tolist(),
            *z_stds[1].tolist(),
            *z_stds[2].tolist(),
            *z_stds[3].tolist(),
            sub_roi_lab[0] / 100.0,
            (sub_roi_lab[1] + 80.0) / 160.0,
            (sub_roi_lab[2] + 80.0) / 160.0,
            min(1.0, aspect_ratio / 2.0),
            min(1.0, edge_density * 4.0),
            min(1.0, glare_ratio * 5.0),
            min(1.0, neck_taper_ratio / 1.5),
            float(gx[:8, :].mean() / 255.0),
            float(gx[8:16, :].mean() / 255.0),
            float(gx[16:, :].mean() / 255.0),
            float(gy[:8, :].mean() / 255.0),
            float(gy[8:16, :].mean() / 255.0),
            float(gy[16:, :].mean() / 255.0),
            min(1.0, bw / max(1.0, float(img_w)) * 10.0),
            min(1.0, bh / max(1.0, float(img_h)) * 5.0),
            1.0 - glare_ratio,
            *hist_feats,
        ],
        dtype=np.float32,
    )[:64]
    raw_vec /= max(1e-6, float(np.linalg.norm(raw_vec)))

    if feature_mode == "maxvit":
        arr_01 = clean_arr / 255.0
        blocks = arr_01.reshape(6, 4, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(24, 16, 3)
        block_means = blocks.mean(axis=1)
        block_stds = blocks.std(axis=1)
        block_salience = block_stds.mean(axis=1) * 3.5
        block_weights = np.exp(block_salience - np.max(block_salience))
        block_weights /= np.sum(block_weights) + 1e-6

        attn_rgb_255 = (block_means * block_weights[:, None]).sum(axis=0) * 255.0
        badge_blocks_rgb_255 = block_means[12:20].mean(axis=0) * 255.0
        sub_roi_lab = _rgb_to_cielab_fast(0.55 * badge_blocks_rgb_255 + 0.45 * attn_rgb_255)

        grids = arr_01.reshape(4, 6, 4, 4, 3).transpose(1, 3, 0, 2, 4).reshape(24, 16, 3)
        grid_means = grids.mean(axis=1)
        mod_vec = np.zeros(64, dtype=np.float32)
        flat_bg = np.concatenate([
            (block_means * block_weights[:, None]).reshape(-1)[:32],
            grid_means.reshape(-1)[:32],
        ]).astype(np.float32)
        mod_vec[: flat_bg.size] = flat_bg
        mod_vec /= max(1e-6, float(np.linalg.norm(mod_vec)))

        neural_vec = _run_maxvit_t_latent_64d(clean_arr)
        mod_vec = 0.65 * mod_vec + 0.35 * neural_vec
        mod_vec /= max(1e-6, float(np.linalg.norm(mod_vec)))

        raw_vec = 0.88 * raw_vec + 0.12 * mod_vec
        raw_vec /= max(1e-6, float(np.linalg.norm(raw_vec)))

    return {
        "embedding": [round(float(v), 5) for v in raw_vec],
        "sub_roi_lab": sub_roi_lab,
        "claim_lab": sub_roi_lab,
        "cap_lab": cap_lab,
        "body_lab": body_lab,
        "glare_ratio": round(glare_ratio, 4),
        "edge_density": round(edge_density, 4),
        "neck_taper_ratio": round(neck_taper_ratio, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "feature_mode": feature_mode,
    }


def extract_gemini_subroi_embedding(
    image: Image.Image | None,
    box: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Consolidated `gemini-embedding-001` Multi-Zone Sub-ROI Feature Extractor (`ADR-002` + `ADR-003`)."""
    if image is None or not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError(f"extract_gemini_subroi_embedding requires a valid PIL.Image, got {image!r}")
    arr = np.asarray(image.convert("RGB"))
    return _extract_subroi_from_array(arr, image.size[0], image.size[1], box, feature_mode="gemini_subroi")


def extract_maxvit_multiscale_features(
    image: Image.Image | None,
    box: tuple[float, float, float, float],
) -> dict[str, Any]:
    """`MaxViT` Multi-Scale Block + Grid Attention Feature Extractor (`ADR-007` Ablation)."""
    if image is None or not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError(f"extract_maxvit_multiscale_features requires a valid PIL.Image, got {image!r}")
    arr = np.asarray(image.convert("RGB"))
    return _extract_subroi_from_array(arr, image.size[0], image.size[1], box, feature_mode="maxvit")


__all__ = [
    "_cosine_sim",
    "_delta_e_cie76",
    "_extract_subroi_from_array",
    "_rgb_to_cielab",
    "_rgb_to_cielab_fast",
    "_run_maxvit_t_latent_64d",
    "compute_ciede2000_approx",
    "extract_gemini_subroi_embedding",
    "extract_maxvit_multiscale_features",
    "extract_real_crop_features",
]
