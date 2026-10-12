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

