"""Photometric specular de-glare restoration and CIELAB sister-shade chromatic descriptors.

Used by ``djev_classify`` (Classification) and ``jev_laya_hybrid`` (Retrieval / End-to-end):

1. ``restore_photometry(image)``:
   - Detects specular glare highlights on foil/plastic packaging (pixels with high luminance
     ``V >= 242`` and low saturation ``S <= 28`` in HSV space).
   - Replaces specular glare regions via local Gaussian/median neighborhood inpainting when glare
     covers between 0.2% and 35% of the crop (skipping pure-white backgrounds).
   - Applies mild luminance contrast stretch + unsharp masking so fine pack-size text (e.g.
     ``10g`` vs ``100g``) and variant sub-brands are legible.
2. ``lab_zones(image, zones=3)`` & ``lab_similarity(z1, z2)``:
   - Converts an RGB crop into CIE 1976 ``(L*, a*, b*)`` space and extracts mean + std chromatic
     descriptors across ``zones`` horizontal bands (top, middle, bottom) plus aspect ratio.
   - Computes a bounded ``[0, 1]`` perceptual similarity score from CIE76 / CIELAB ``Delta E``
     to break ties between sister variants (same bottle/pouch shape, different flavour colour band).
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter


def specular_glare_fraction(image: Image.Image) -> float:
    """Share of pixels in ``image`` that exhibit saturated low-saturation specular glare."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    cmax = arr.max(axis=-1)
    cmin = arr.min(axis=-1)
    sat = np.where(cmax > 0, (cmax - cmin) / np.maximum(cmax, 1.0), 0.0)
    glare = (cmax >= 242.0) & (sat <= 0.11)
    return float(glare.mean()) if glare.size else 0.0


def restore_photometry(image: Image.Image) -> tuple[Image.Image, dict]:
    """Inpaint localized specular glare highlights and sharpen pack typography.

    Returns ``(restored_image, stats_dict)``. Pure-colour or white-background studio images
    (where 'glare' is either 0% or > 35% background) are returned without destructive inpainting.
    """
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.shape[0] < 8 or arr.shape[1] < 8:
        return rgb, {"glare_fraction": 0.0, "inpainted": False, "contrast_stretched": False}

    cmax = arr.max(axis=-1)
    cmin = arr.min(axis=-1)
    sat = np.where(cmax > 0, (cmax - cmin) / np.maximum(cmax, 1.0), 0.0)
    glare_mask = (cmax >= 242.0) & (sat <= 0.11)
    glare_frac = float(glare_mask.mean())

    if not (0.002 <= glare_frac <= 0.35):
        return rgb, {
            "glare_fraction": round(glare_frac, 4),
            "inpainted": False,
            "contrast_stretched": False,
        }

    mask_img = Image.fromarray((glare_mask.astype(np.uint8) * 255), mode="L").filter(
        ImageFilter.MaxFilter(3)
    )
    dilated = (np.asarray(mask_img) > 127)[..., None]
    smooth = np.asarray(rgb.filter(ImageFilter.BoxBlur(3)), dtype=np.float32)
    arr = np.clip(np.where(dilated, smooth * 0.88, arr), 0, 255)
    inpainted = True

    # Mild luminance-preserving contrast normalization when dynamic range is compressed
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    p5, p95 = float(np.percentile(lum, 5)), float(np.percentile(lum, 95))
    contrast_stretched = False
    if 35.0 < (p95 - p5) < 155.0:
        new_lum = np.clip((lum - p5) * (210.0 / max(p95 - p5, 1.0)) + 18.0, 0.0, 255.0)
        scale = (new_lum / np.maximum(lum, 1.0))[..., None]
        arr = np.clip(arr * scale, 0, 255)
        contrast_stretched = True

    restored = Image.fromarray(arr.astype(np.uint8), mode="RGB").filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=3)
    )
    return restored, {
        "glare_fraction": round(glare_frac, 4),
        "inpainted": inpainted,
        "contrast_stretched": contrast_stretched,
    }


def _rgb_to_lab(arr: np.ndarray) -> np.ndarray:
    """Vectorized sRGB [0..255] -> CIE L*a*b* (D65 reference white)."""
    srgb = np.clip(arr / 255.0, 0.0, 1.0)
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    # sRGB to XYZ (D65)
    x = lin[..., 0] * 0.4124564 + lin[..., 1] * 0.3575761 + lin[..., 2] * 0.1804375
    y = lin[..., 0] * 0.2126729 + lin[..., 1] * 0.7151522 + lin[..., 2] * 0.0721750
    z = lin[..., 0] * 0.0193339 + lin[..., 1] * 0.1191920 + lin[..., 2] * 0.9503041
    # Normalize by D65 white point
    xr, yr, zr = x / 0.95047, y / 1.00000, z / 1.08883
    eps, kappa = 0.008856, 903.3
    fx = np.where(xr > eps, np.cbrt(xr), (kappa * xr + 16.0) / 116.0)
    fy = np.where(yr > eps, np.cbrt(yr), (kappa * yr + 16.0) / 116.0)
    fz = np.where(zr > eps, np.cbrt(zr), (kappa * zr + 16.0) / 116.0)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_zones(image: Image.Image, zones: int = 3) -> list[float]:
    """3-zone horizontal CIELAB descriptor: per band (0.35*L*, a*, b*) + log aspect ratio.

    Downweights L* by 0.35 relative to a* and b* so shadow/lighting shifts between shelf rows
    do not dominate sister-shade colour comparisons.
    """
    w, h = max(1, image.width), max(1, image.height)
    small = image.convert("RGB").resize((24, max(zones * 4, 24)), Image.Resampling.BILINEAR)
    lab = _rgb_to_lab(np.asarray(small, dtype=np.float32))
    band_h = lab.shape[0] // zones
    desc: list[float] = []
    for z in range(zones):
        sl = lab[z * band_h : (z + 1) * band_h if z < zones - 1 else lab.shape[0]]
        mean = sl.mean(axis=(0, 1))
        desc.extend([float(mean[0]) * 0.35, float(mean[1]), float(mean[2])])
    desc.append(math.log(w / h) * 15.0)
    return desc


def lab_similarity(z1: list[float], z2: list[float]) -> float:
    """Bounded [0, 1] similarity from Euclidean distance in weighted multi-zone CIELAB space."""
    if not z1 or not z2 or len(z1) != len(z2):
        return 0.5
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(z1, z2, strict=True)) / (len(z1) / 3))
    # Delta E ~ 0 -> 1.0; Delta E ~ 25 -> 0.50; Delta E >= 75 -> ~0.12
    return math.exp(-dist / 36.0)


def estimate_shelf_illuminant(shelf_image: Image.Image, p: float = 6.0) -> list[float]:
    """Estimate ambient store illuminant once on the full shelf image via Shades-of-Gray (Minkowski p=6).

    Across a full shelf photo (30-140 diverse products + metal shelf rails), the Shades-of-Gray
    color-constancy assumption holds, whereas on an individual red/blue product crop it fails.
    Excludes saturated specular pixels (>= 245) and clamps channel gains to [0.72, 1.38].
    """
    small = shelf_image.convert("RGB")
    small.thumbnail((512, 512))
    arr = np.asarray(small, dtype=np.float32)
    valid = (arr.max(axis=-1) < 245.0) & (arr.min(axis=-1) > 12.0)
    pixels = arr[valid] if int(valid.sum()) > 100 else arr.reshape(-1, 3)
    norm_p = np.power(np.mean(np.power(pixels, p), axis=0), 1.0 / p) + 1e-5
    gains = np.clip(float(norm_p.mean()) / norm_p, 0.72, 1.38)
    return [round(float(g), 4) for g in gains]


def apply_illuminant(image: Image.Image, gains: list[float] | tuple[float, ...] | None) -> Image.Image:
    """Apply pre-estimated full-shelf RGB channel gains [r_gain, g_gain, b_gain] to a product crop."""
    if not gains or len(gains) != 3:
        return image.convert("RGB")
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    g = np.array(gains, dtype=np.float32).reshape((1, 1, 3))
    return Image.fromarray(np.clip(arr * g, 0.0, 255.0).astype(np.uint8), mode="RGB")


def inner_lab_zones(image: Image.Image, zones: int = 3, trim_x: float = 0.10, trim_y: float = 0.10) -> list[float]:
    """Foreground-centered 3-zone CIELAB + silhouette/promo descriptor.

    Trims the outer ``trim_x`` / ``trim_y`` border of a shelf crop before computing CIELAB bands so
    dark shelf backgrounds, adjacent product bleed, and bottom metal price rails do not corrupt
    chromatic matching against clean studio packshots. Also appends shoulder-taper (bottle vs
    pouch/sachet) and promo-header chroma cues.
    """
    rgb = image.convert("RGB")
    w, h = max(1, rgb.width), max(1, rgb.height)
    if w >= 16 and h >= 16:
        x1, x2 = int(w * trim_x), max(int(w * trim_x) + 1, int(w * (1.0 - trim_x)))
        y1, y2 = int(h * trim_y * 0.8), max(int(h * trim_y) + 1, int(h * (1.0 - trim_y * 1.2)))
        core = rgb.crop((x1, y1, x2, y2))
    else:
        core = rgb
    desc = lab_zones(core, zones=zones)
    # Replace aspect ratio with full-crop aspect ratio + shoulder taper (bottle neck vs flat-top sachet/pouch)
    desc[-1] = math.log(w / h) * 15.0
    small_full = rgb.resize((24, 24), Image.Resampling.BILINEAR)
    lab_full = _rgb_to_lab(np.asarray(small_full, dtype=np.float32))
    # Top-25% corner-vs-center chromatic/luminance contrast (bottles have background corners at top neck)
    top_band = lab_full[:6]
    corners = np.concatenate([top_band[:, :6], top_band[:, 18:]], axis=1)
    center = top_band[:, 6:18]
    neck_taper = float(np.linalg.norm(corners.mean(axis=(0, 1)) - center.mean(axis=(0, 1)))) * 0.25
    top_a_std = float(top_band[..., 1].std()) * 0.5
    top_b_std = float(top_band[..., 2].std()) * 0.5
    desc.extend([neck_taper, top_a_std, top_b_std])
    return desc


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Vectorized CIE L*a*b* (D65 reference white) -> sRGB uint8 [0..255]."""
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps, kappa = 0.008856, 903.3
    fx3 = fx * fx * fx
    fz3 = fz * fz * fz
    xr = np.where(fx3 > eps, fx3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, fy * fy * fy, L / kappa)
    zr = np.where(fz3 > eps, fz3, (116.0 * fz - 16.0) / kappa)
    x = xr * 0.95047
    y = yr * 1.00000
    z = zr * 1.08883
    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    lin = np.clip(np.stack([r_lin, g_lin, b_lin], axis=-1), 0.0, 1.0)
    srgb = np.where(
        lin <= 0.0031308,
        12.92 * lin,
        1.055 * (np.maximum(lin, 1e-12) ** (1.0 / 2.4)) - 0.055,
    )
    return np.clip(srgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _clahe_l_channel(
    L: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply bilinearly interpolated tile-based CLAHE to CIELAB L* channel in [0, 100]."""
    h, w = L.shape
    ty = max(1, min(int(tile_grid[0]), max(1, h // 4)))
    tx = max(1, min(int(tile_grid[1]), max(1, w // 4)))
    q = np.clip(np.round(L * 2.55), 0, 255).astype(np.int32)
    lut = np.zeros((ty, tx, 256), dtype=np.float32)

    for iy in range(ty):
        y0 = (iy * h) // ty
        y1 = ((iy + 1) * h) // ty if iy < ty - 1 else h
        for ix in range(tx):
            x0 = (ix * w) // tx
            x1 = ((ix + 1) * w) // tx if ix < tx - 1 else w
            tile_q = q[y0:y1, x0:x1]
            tile_pixels = max(1, tile_q.size)
            hist = np.bincount(tile_q.ravel(), minlength=256).astype(np.int64)
            if clip_limit > 0.0:
                limit = max(1, int(round(clip_limit * tile_pixels / 256.0)))
                excess = int(np.maximum(hist - limit, 0).sum())
                hist = np.minimum(hist, limit) + (excess // 256)
                rem = excess % 256
                if rem > 0:
                    hist[:rem] += 1
            cdf = np.cumsum(hist).astype(np.float32)
            nonzero = cdf[cdf > 0]
            cdf_min = float(nonzero[0]) if nonzero.size else 0.0
            denom = max(1.0, float(tile_pixels) - cdf_min)
            lut[iy, ix] = np.clip((cdf - cdf_min) / denom * 100.0, 0.0, 100.0)

    gy = np.clip((np.arange(h, dtype=np.float32) + 0.5) * ty / max(1, h) - 0.5, 0.0, ty - 1.0)
    gx = np.clip((np.arange(w, dtype=np.float32) + 0.5) * tx / max(1, w) - 0.5, 0.0, tx - 1.0)
    iy0 = np.floor(gy).astype(np.int32)
    iy1 = np.minimum(iy0 + 1, ty - 1)
    wy = (gy - iy0)[:, None]
    ix0 = np.floor(gx).astype(np.int32)
    ix1 = np.minimum(ix0 + 1, tx - 1)
    wx = (gx - ix0)[None, :]

    v00 = lut[iy0[:, None], ix0[None, :], q]
    v01 = lut[iy0[:, None], ix1[None, :], q]
    v10 = lut[iy1[:, None], ix0[None, :], q]
    v11 = lut[iy1[:, None], ix1[None, :], q]
    return (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)


def enhance_shelf_for_detection(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> Image.Image:
    """Enhance a full shelf image strictly for Stage 1 box detection (Shades-of-Gray + CIELAB L* CLAHE).

    WARNING: This applies non-linear local contrast enhancement (CLAHE) to lift dark interior
    shelf shadows and compress acrylic glare. Never pass the returned image into Stage 2 CIELAB
    color matching (``lab_zones`` / ``inner_lab_zones``); use :func:`dual_buffer_shelf_photometry`
    so Stage 2 crops come from the linear Shades-of-Gray ``color_image``.
    """
    detector_im, _, _ = dual_buffer_shelf_photometry(
        image, clip_limit=clip_limit, tile_grid=tile_grid
    )
    return detector_im


def dual_buffer_shelf_photometry(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> tuple[Image.Image, Image.Image, dict]:
    """Build Stage 0 dual photometric buffers: ``(detector_image, color_image, stats)``.

    - ``color_image``: Linear Shades-of-Gray illuminant-corrected shelf image. Safe for Stage 2
      product crop embedding and CIELAB sister-shade matching (preserves pastel hues).
    - ``detector_image``: Non-linear CIELAB ``L*``-channel CLAHE applied on top of ``color_image``.
      Lifts dark back-row shelf shadows and compresses overhead bulb glare strictly for Stage 1
      box detection.
    """
    rgb = image.convert("RGB")
    gains = estimate_shelf_illuminant(rgb)
    color_image = apply_illuminant(rgb, gains)
    arr = np.asarray(color_image, dtype=np.float32)
    if arr.shape[0] < 8 or arr.shape[1] < 8:
        return color_image, color_image, {
            "gains": gains,
            "clip_limit": clip_limit,
            "tile_grid": list(tile_grid),
            "l_std_before": 0.0,
            "l_std_after": 0.0,
        }
    lab = _rgb_to_lab(arr)
    l_before = lab[..., 0]
    l_after = _clahe_l_channel(l_before, clip_limit=clip_limit, tile_grid=tile_grid)
    lab_enhanced = np.stack([l_after, lab[..., 1], lab[..., 2]], axis=-1)
    detector_arr = _lab_to_rgb(lab_enhanced)
    detector_image = Image.fromarray(detector_arr, mode="RGB")
    return detector_image, color_image, {
        "gains": gains,
        "clip_limit": float(clip_limit),
        "tile_grid": [int(tile_grid[0]), int(tile_grid[1])],
        "l_mean_before": round(float(l_before.mean()), 2),
        "l_mean_after": round(float(l_after.mean()), 2),
        "l_std_before": round(float(l_before.std()), 2),
        "l_std_after": round(float(l_after.std()), 2),
    }


