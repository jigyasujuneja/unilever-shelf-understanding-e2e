"""2D shelf geometry and depth-stacked column NMS (`BBox`, `horizontal_overlap_ratio`, `deduplicate_depth_stacked_facings`).

Extracted from `tasks/facing_utils.py` so that `approaches/base.py` and the evaluation layer can
use shelf geometry without importing `shelf_benchmark.tasks` (which previously pulled in all four
task modules and `google.genai` at import time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

__all__ = [
    "BBox",
    "deduplicate_depth_stacked_facings",
    "horizontal_overlap_ratio",
]


@dataclass(frozen=True)
class BBox:
    """Canonical `[ymin, xmin, ymax, xmax]` bounding box normalized to 0..1000."""

    ymin: int
    xmin: int
    ymax: int
    xmax: int

    @classmethod
    def from_sequence(cls, coords: Sequence[int | float]) -> BBox:
        if len(coords) < 4:
            return cls(0, 0, 0, 0)
        return cls(int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))

    def as_list(self) -> List[int]:
        return [self.ymin, self.xmin, self.ymax, self.xmax]

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)


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
    """Filter out products stacked behind the front facing in the same horizontal column on the same shelf row."""
    if not items:
        return [], 0

    candidates = [it for it in items if it.get("is_front_facing", True) is not False]
    removed_count = len(items) - len(candidates)

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

    kept.sort(key=lambda it: (it.get("bbox_2d", [0, 0, 0, 0])[1] if len(it.get("bbox_2d", [])) >= 2 else 0))
    for idx, it in enumerate(kept, start=1):
        it["product_index"] = idx
        it["position_on_shelf"] = idx
    return kept, removed_count
