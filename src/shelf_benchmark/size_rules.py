"""Rule-derived size bucket computation from packaging type, OCR size hints, and relative row height."""

from __future__ import annotations

import re
import statistics
from functools import lru_cache
from typing import List, Optional

from shelf_benchmark.config import TaxonomyConfig

__all__ = ["derive_size_bucket_from_bbox"]


@lru_cache(maxsize=1)
def _default_taxonomy() -> TaxonomyConfig:
    return TaxonomyConfig.from_yaml_or_defaults()


def derive_size_bucket_from_bbox(
    bbox_2d: List[int],
    all_bboxes_on_row: List[List[int]],
    packaging_type: str = "tube",
    model_size_hint: str = "",
    taxonomy: Optional[TaxonomyConfig] = None,
) -> str:
    """Rule-derived Size Bucket combining packaging form factor, OCR weight/volume cues, and relative bounding-box geometry."""
    tax = taxonomy if taxonomy is not None else _default_taxonomy()
    sb = tax.size_buckets
    pkg = (packaging_type or "").lower()
    hint = (model_size_hint or "").lower()

    if "sachet" in pkg or "sachet" in hint or "pouch" in pkg:
        return sb.sachet_label

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
