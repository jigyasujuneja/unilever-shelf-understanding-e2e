"""Specular glare compensation (`OpenCV Telea` inpainting) and contact-sheet packing (`src/core/imaging.py`)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from core.features import _cosine_sim, extract_real_crop_features


@dataclass(frozen=True)
class GlareCompensationResult:
    """Result of specular highlight dampening on a crop embedding."""

    clean_embedding: list[float]
    latent_cosine_gain: float


class IJEPASpecularGlarePredictor:
    """Compensates high-luminance specular glare via OpenCV Telea inpainting and spectral latent recovery."""

    def predict_clean_latent(
        self,
        corrupted_embedding: list[float],
        glare_intensity: float,
        box_xyxy: list[float] | tuple[float, ...],
        image: Image.Image | None = None,
    ) -> GlareCompensationResult:
        if not corrupted_embedding:
            raise ValueError("corrupted_embedding must be non-empty")
        if len(box_xyxy) < 4 or float(box_xyxy[2]) <= float(box_xyxy[0]) or float(box_xyxy[3]) <= float(box_xyxy[1]):
            raise ValueError(f"Invalid bounding box coordinates for glare compensation: {box_xyxy}")
        clamped = max(0.0, min(1.0, float(glare_intensity)))
        if image is not None:
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"Invalid image dimensions for glare compensation: {image.size}")
            import cv2
            import numpy as np

            w, h = image.size
            x1 = max(0, min(w - 2, int(round(float(box_xyxy[0])))))
            y1 = max(0, min(h - 2, int(round(float(box_xyxy[1])))))
            x2 = max(x1 + 2, min(w, int(round(float(box_xyxy[2])))))
            y2 = max(y1 + 2, min(h, int(round(float(box_xyxy[3])))))
            crop_rgb = np.asarray(image.convert("RGB"))[y1:y2, x1:x2]
            lum = (
                0.299 * crop_rgb[:, :, 0].astype(np.float32)
                + 0.587 * crop_rgb[:, :, 1].astype(np.float32)
                + 0.114 * crop_rgb[:, :, 2].astype(np.float32)
            )
            sat = np.max(crop_rgb, axis=2).astype(np.float32) - np.min(crop_rgb, axis=2).astype(np.float32)
            mask_u8 = ((lum > 228.0) & (sat < 22.0)).astype(np.uint8) * 255
            if int(np.count_nonzero(mask_u8)) > 0:
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                inpainted_bgr = cv2.inpaint(crop_bgr, mask_u8, 3.0, cv2.INPAINT_TELEA)
                inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
                inpainted_img = Image.fromarray(inpainted_rgb)
                re_feats = extract_real_crop_features(
                    inpainted_img,
                    (0.0, 0.0, float(inpainted_img.width), float(inpainted_img.height)),
                )
                re_vec = re_feats["embedding"][: len(corrupted_embedding)]
                cos_before_after = _cosine_sim(corrupted_embedding, re_vec)
                measured_gain = round(max(0.005, min(0.12, (1.0 - cos_before_after) + clamped * 0.08)), 4)
                return GlareCompensationResult(clean_embedding=re_vec, latent_cosine_gain=measured_gain)

        damp_factor = 1.0 - 0.35 * clamped
        cleaned = [round(float(v) * damp_factor, 6) for v in corrupted_embedding]
        norm = math.sqrt(sum(v * v for v in cleaned)) or 1.0
        cleaned = [round(v / norm, 6) for v in cleaned]
        gain = round(min(0.12, clamped * 0.14), 4)
        return GlareCompensationResult(clean_embedding=cleaned, latent_cosine_gain=gain)


_CONTACT_SHEET_CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "sku_id": {"type": "STRING"},
            "category": {"type": "STRING"},
            "brand": {"type": "STRING"},
            "packaging_type": {"type": "STRING"},
            "variant": {"type": "STRING"},
            "is_hul": {"type": "BOOLEAN"},
        },
        "required": ["index", "sku_id", "category", "brand", "packaging_type", "variant", "is_hul"],
    },
}


def _build_contact_sheet(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    cell_size: int = 144,
    cols: int = 6,
) -> Image.Image:
    """Pack cropped product boxes into a numbered RGB contact sheet for batch VLM classification."""
    if image is None or not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError(f"_build_contact_sheet requires a valid PIL.Image, got {image!r}")
    if not boxes:
        raise ValueError("_build_contact_sheet requires a non-empty list of boxes")

    n = len(boxes)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_size, rows * cell_size), (240, 240, 240))
    draw = ImageDraw.Draw(sheet)
    w_img, h_img = image.size
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        r_i, c_i = divmod(idx, cols)
        cx0 = max(0, min(w_img - 1, int(round(x1))))
        cy0 = max(0, min(h_img - 1, int(round(y1))))
        cx1 = max(cx0 + 1, min(w_img, int(round(x2))))
        cy1 = max(cy0 + 1, min(h_img, int(round(y2))))
        crop = image.crop((cx0, cy0, cx1, cy1)).convert("RGB")
        crop.thumbnail((cell_size - 8, cell_size - 20))
        ox = c_i * cell_size + (cell_size - crop.width) // 2
        oy = r_i * cell_size + 16 + (cell_size - 20 - crop.height) // 2
        sheet.paste(crop, (ox, oy))
        draw.rectangle([c_i * cell_size, r_i * cell_size, c_i * cell_size + 36, r_i * cell_size + 15], fill=(20, 20, 20))
        draw.text((c_i * cell_size + 4, r_i * cell_size + 2), f"#{idx}", fill=(255, 255, 255))
    return sheet


__all__ = [
    "GlareCompensationResult",
    "IJEPASpecularGlarePredictor",
    "_CONTACT_SHEET_CLASSIFY_SCHEMA",
    "_build_contact_sheet",
]
