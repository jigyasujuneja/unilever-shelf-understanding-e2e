"""Utilities for Front-Facing-Only Depth De-duplication, Rule-Derived Size Bucketing, and Physical Bounding-Box Cropping."""

from __future__ import annotations

import io
from pathlib import Path
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from shelf_benchmark.config import TaxonomyConfig
from shelf_benchmark.data.storage import StorageManager

_DEFAULT_TAXONOMY = TaxonomyConfig.from_yaml_or_defaults()

HUL_PORTFOLIO_BRANDS = {
    re.sub(r"[^a-z0-9\s&']", "", b.lower()).strip()
    for b in _DEFAULT_TAXONOMY.hul_brands
}


def check_is_hul_brand(
    brand_name: str,
    taxonomy: Optional[TaxonomyConfig] = None,
    model_predicted: Optional[bool] = None,
) -> bool:
    """Determine whether a predicted brand belongs to HUL without requiring predefined brand lists in YAML.

    - If the user has explicitly configured `hul_brands` (or `non_hul_brands`) in `TaxonomyConfig`, those lists take precedence.
    - Otherwise (when `hul_brands` is empty `[]` by default), relies on `model_predicted` (`is_hul_brand` returned by the VLM or Catalog join).
    """
    norm = re.sub(r"[^a-z0-9\s&']", "", (brand_name or "").lower()).strip()
    if not norm:
        return False

    if taxonomy is not None and taxonomy.non_hul_brands:
        non_hul_set = {re.sub(r"[^a-z0-9\s&']", "", b.lower()).strip() for b in taxonomy.non_hul_brands}
        if any(nh in norm or norm in nh for nh in non_hul_set if nh):
            return False

    if taxonomy is not None and taxonomy.hul_brands:
        hul_set = {re.sub(r"[^a-z0-9\s&']", "", b.lower()).strip() for b in taxonomy.hul_brands}
        return any(h in norm or norm in h for h in hul_set if h)

    if model_predicted is not None:
        return bool(model_predicted)

    return False


def horizontal_overlap_ratio(box_a: List[int], box_b: List[int]) -> float:
    """Compute 1D horizontal overlap divided by the narrower box's width."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    xa1, xa2 = box_a[1], box_a[3]
    xb1, xb2 = box_b[1], box_b[3]
    w_a = max(1, xa2 - xa1)
    w_b = max(1, xb2 - xb1)
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    return float(inter_w) / float(min(w_a, w_b))


def deduplicate_depth_stacked_facings(
    items: List[Dict[str, Any]],
    x_overlap_threshold: float = 0.55,
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter out products stacked behind the front facing in the same horizontal column on the same shelf row.

    On a retail shelf, products in the same facing column share horizontal `[xmin, xmax]` coordinates,
    while units stacked in depth behind the front facing have a higher `ymax` (occluded bottom) or smaller visible height.
    We keep only the front-most unit per horizontal facing column (the one resting lowest on the shelf ledge, i.e., largest `ymax` and height).
    """
    if not items:
        return [], 0

    # Filter out any items explicitly marked is_front_facing=False
    candidates = [it for it in items if it.get("is_front_facing", True) is not False]
    removed_count = len(items) - len(candidates)

    # Sort by front-most dominance: largest ymax (sits on the front shelf lip) + largest height
    def front_dominance_key(it: Dict[str, Any]) -> Tuple[int, int]:
        b = it.get("bbox_2d") or [0, 0, 0, 0]
        if len(b) < 4:
            return (0, 0)
        ymin, xmin, ymax, xmax = b[:4]
        height = max(0, ymax - ymin)
        return (ymax, height)

    sorted_by_front = sorted(candidates, key=front_dominance_key, reverse=True)
    kept: List[Dict[str, Any]] = []

    for cand in sorted_by_front:
        c_box = cand.get("bbox_2d") or [0, 0, 0, 0]
        c_row = str(cand.get("shelf_row") or "middle").lower()
        is_depth_duplicate = False
        for existing in kept:
            e_box = existing.get("bbox_2d") or [0, 0, 0, 0]
            e_row = str(existing.get("shelf_row") or "middle").lower()
            if c_row == e_row and horizontal_overlap_ratio(c_box, e_box) >= x_overlap_threshold:
                is_depth_duplicate = True
                break
        if is_depth_duplicate:
            removed_count += 1
        else:
            kept.append(cand)

    # Sort kept front facings strictly left-to-right by xmin
    kept.sort(key=lambda it: (it.get("bbox_2d", [0, 0, 0, 0])[1] if len(it.get("bbox_2d", [])) >= 2 else 0))
    for idx, it in enumerate(kept, start=1):
        it["product_index"] = idx
        it["position_on_shelf"] = idx
    return kept, removed_count


def derive_size_bucket_from_bbox(
    bbox_2d: List[int],
    all_bboxes_on_row: List[List[int]],
    packaging_type: str = "tube",
    model_size_hint: str = "",
    taxonomy: TaxonomyConfig | None = None,
) -> str:
    """Rule-derived Size Bucket combining packaging form factor, OCR weight/volume cues, and relative bounding-box geometry configured in TaxonomyConfig."""
    tax = taxonomy if taxonomy is not None else _DEFAULT_TAXONOMY
    sb = tax.size_buckets
    pkg = (packaging_type or "").lower()
    hint = (model_size_hint or "").lower()

    if "sachet" in pkg or "sachet" in hint or "pouch" in pkg:
        return sb.sachet_label

    # Check if explicit gram/ml number is present in model_size_hint
    match = re.search(r"(\d+)\s*(g|gm|ml)\b", hint)
    if match:
        val = int(match.group(1))
        if val < sb.sachet_max_grams:
            return sb.sachet_label
        if val <= sb.small_max_grams:
            return sb.small_label
        if val <= sb.medium_max_grams:
            return sb.medium_label
        return sb.large_label

    # Geometric rule based on bounding-box height relative to shelf row median height
    if len(bbox_2d) < 4:
        return model_size_hint or sb.medium_label

    height = max(1, bbox_2d[2] - bbox_2d[0])
    valid_heights = [max(1, b[2] - b[0]) for b in all_bboxes_on_row if len(b) >= 4 and (b[2] > b[0])]
    median_h = statistics.median(valid_heights) if valid_heights else height

    ratio = float(height) / float(max(median_h, 1))
    if ratio < sb.small_bbox_height_ratio:
        return sb.small_label
    if ratio > sb.large_bbox_height_ratio:
        return sb.large_label
    return sb.medium_label


def load_pil_image(storage: StorageManager, shelf_image_uri: str, local_fallback: str = "shelf-image.png") -> Image.Image:
    """Load a PIL Image from local path or GCS URI."""
    if not shelf_image_uri.startswith("gs://") and Path(shelf_image_uri).exists():
        return Image.open(shelf_image_uri).convert("RGB")
    if Path(local_fallback).exists():
        return Image.open(local_fallback).convert("RGB")
    bucket_name, blob_name = storage.parse_gcs_uri(shelf_image_uri)
    data = storage.client.bucket(bucket_name).blob(blob_name).download_as_bytes()
    return Image.open(io.BytesIO(data)).convert("RGB")


def crop_detected_facings(
    storage: StorageManager,
    shelf_image_uri: str,
    detected_items: List[Dict[str, Any]],
    output_crop_dir: str | Path = "reports/crops",
    model_tag: str = "model",
) -> Tuple[List[str], bytes]:
    """Physically separate/crop each detected front-facing bounding box into individual image files and a numbered montage strip."""
    img = load_pil_image(storage, shelf_image_uri)
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

        # Resize crop to uniform height (260px) for numbered visual montage
        target_h = 260
        aspect = crop_img.width / max(crop_img.height, 1)
        target_w = max(60, min(200, int(target_h * aspect)))
        resized = crop_img.resize((target_w, target_h))
        draw = ImageDraw.Draw(resized)
        draw.rectangle([0, 0, 38, 24], fill=(0, 0, 0))
        draw.text((6, 5), f"#{idx}", fill=(255, 255, 0))
        pil_crops.append(resized)

    # Assemble horizontal montage strip of all separated facing crops
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
