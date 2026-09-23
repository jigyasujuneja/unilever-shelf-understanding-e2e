"""PIL shelf image loading, per-facing physical crop extraction, and numbered montage generation."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from shelf_benchmark.data.storage import StorageManager

logger = logging.getLogger(__name__)

__all__ = ["crop_detected_facings", "load_pil_image"]


def load_pil_image(
    storage: StorageManager,
    shelf_image_uri: str,
    local_fallback: Optional[str] = None,
) -> Image.Image:
    """Load the shelf image named by `shelf_image_uri` (local path or `gs://` URI)."""
    if not shelf_image_uri:
        raise ValueError("shelf_image_uri is required to load a shelf image.")

    try:
        if shelf_image_uri.startswith("gs://"):
            bucket_name, blob_name = storage.parse_gcs_uri(shelf_image_uri)
            data = storage.client.bucket(bucket_name).blob(blob_name).download_as_bytes()
            return Image.open(io.BytesIO(data)).convert("RGB")
        path = Path(shelf_image_uri)
        if path.exists():
            return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"Shelf image not found: {shelf_image_uri}")
    except Exception as primary_error:
        if not local_fallback:
            raise
        fallback_path = Path(local_fallback)
        if not fallback_path.exists():
            raise
        logger.warning(
            "Could not read shelf image '%s' (%s). Falling back to local file '%s'. "
            "Crops, embeddings and metrics for this run describe the FALLBACK image, not '%s'.",
            shelf_image_uri,
            primary_error,
            local_fallback,
            shelf_image_uri,
        )
        return Image.open(fallback_path).convert("RGB")


def crop_detected_facings(
    storage: StorageManager,
    shelf_image_uri: str,
    detected_items: List[Dict[str, Any]],
    output_crop_dir: str | Path = "reports/crops",
    model_tag: str = "model",
    local_fallback: Optional[str] = None,
) -> Tuple[List[str], bytes]:
    """Physically separate/crop each detected front-facing bounding box into individual image files and a numbered montage strip."""
    img = load_pil_image(storage, shelf_image_uri, local_fallback=local_fallback)
    width, height = img.size
    crop_dir = Path(output_crop_dir) / model_tag.replace("/", "_")
    crop_dir.mkdir(parents=True, exist_ok=True)

    crop_paths: List[str] = []
    pil_crops: List[Image.Image] = []

    for idx, item in enumerate(detected_items, start=1):
        box = item.get("bbox_2d") or [0, 0, 1000, 1000]
        ymin, xmin, ymax, xmax = box[:4]
        left = max(0, min(width - 1, int((xmin / 1000.0) * width)))
        top = max(0, min(height - 1, int((ymin / 1000.0) * height)))
        right = max(left + 4, min(width, int((xmax / 1000.0) * width)))
        bottom = max(top + 4, min(height, int((ymax / 1000.0) * height)))

        crop_img = img.crop((left, top, right, bottom))
        crop_file = crop_dir / f"facing_{idx:02d}.png"
        crop_img.save(crop_file, format="PNG")
        crop_paths.append(str(crop_file))

        target_h = 260
        aspect = crop_img.width / max(crop_img.height, 1)
        target_w = max(60, min(200, int(target_h * aspect)))
        resized = crop_img.resize((target_w, target_h))
        draw = ImageDraw.Draw(resized)
        draw.rectangle([0, 0, 38, 24], fill=(0, 0, 0))
        draw.text((6, 5), f"#{idx}", fill=(255, 255, 0))
        pil_crops.append(resized)

    if not pil_crops:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return [], buf.getvalue()

    padding = 8
    total_w = sum(c.width for c in pil_crops) + padding * (len(pil_crops) + 1)
    montage = Image.new("RGB", (total_w, 280), color=(245, 245, 245))
    x_offset = padding
    for c in pil_crops:
        montage.paste(c, (x_offset, 10))
        x_offset += c.width + padding

    montage_path = crop_dir / "montage_all_facings.png"
    montage.save(montage_path, format="PNG")
    buf = io.BytesIO()
    montage.save(buf, format="PNG")
    return crop_paths, buf.getvalue()
