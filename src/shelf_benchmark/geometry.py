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
    "infer_shelf_row_from_bbox",
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


def infer_shelf_row_from_bbox(box: Sequence[int | float]) -> str:
    """Infer a canonical shelf row band ('top', 'middle', 'bottom') from 0..1000 vertical coordinates."""
    if len(box) < 4:
        return "middle"
    ymin, _, ymax, _ = box[:4]
    if ymax <= ymin:
        return "middle"
    y_center = (float(ymin) + float(ymax)) / 2.0
    if y_center < 334.0:
        return "top"
    if y_center <= 666.0:
        return "middle"
    return "bottom"


def _vertical_depth_overlap(cand_box: List[int], front_box: List[int]) -> bool:
    """True when `cand_box` vertically overlaps or sits immediately behind the top of `front_box`.

    Prevents a product on an upper shelf (e.g. y=[100..300]) from being suppressed as a
    depth-stacked duplicate of a product in the same X column on a lower shelf (e.g. y=[400..600])
    when a detector omits `shelf_row` or labels both with the default 'middle'.
    """
    if len(cand_box) < 4 or len(front_box) < 4:
        return True
    c_ymin, _, c_ymax, _ = cand_box[:4]
    f_ymin, _, f_ymax, _ = front_box[:4]
    f_height = max(1, f_ymax - f_ymin)
    return c_ymax >= (f_ymin - int(0.15 * f_height)) and c_ymin <= f_ymax


def _same_shelf_elevation(
    cand_box: List[int], front_box: List[int], cand_row: str, front_row: str
) -> bool:
    """True when `cand_box` and `front_box` occupy the same shelf elevation in depth.

    Handles both matching `shelf_row` labels and cases where an LLM detector mislabels a back-row
    unit peeking behind a front unit with an adjacent row name (e.g. 'top' vs 'middle') despite
    heavy physical 2D vertical overlap (`>= 40%` of the smaller box height).
    """
    if cand_row == front_row:
        return _vertical_depth_overlap(cand_box, front_box)
    if len(cand_box) < 4 or len(front_box) < 4:
        return False
    c_ymin, _, c_ymax, _ = cand_box[:4]
    f_ymin, _, f_ymax, _ = front_box[:4]
    c_h = max(1, c_ymax - c_ymin)
    f_h = max(1, f_ymax - f_ymin)
    inter_h = max(0, min(c_ymax, f_ymax) - max(c_ymin, f_ymin))
    return (float(inter_h) / float(min(c_h, f_h))) >= 0.40


_ROW_ORDER = {"top": 0, "upper": 1, "middle": 2, "lower": 3, "bottom": 4}


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
        raw_c_row = str(cand.get("shelf_row") or "").strip().lower()
        c_row = raw_c_row or infer_shelf_row_from_bbox(c_box)
        is_depth_duplicate = False
        for existing in kept:
            e_box = existing.get("bbox_2d") or [0, 0, 0, 0]
            raw_e_row = str(existing.get("shelf_row") or "").strip().lower()
            e_row = raw_e_row or infer_shelf_row_from_bbox(e_box)
            if (
                horizontal_overlap_ratio(c_box, e_box) >= x_overlap_threshold
                and _same_shelf_elevation(c_box, e_box, c_row, e_row)
            ):
                is_depth_duplicate = True
                break
        if is_depth_duplicate:
            removed_count += 1
        else:
            kept.append(cand)

    def _shelf_sort_key(it: Dict[str, Any]) -> Tuple[int, int]:
        b = it.get("bbox_2d") or [0, 0, 0, 0]
        row_label = str(it.get("shelf_row") or "").strip().lower() or infer_shelf_row_from_bbox(b)
        row_rank = _ROW_ORDER.get(row_label, 2)
        xmin = int(b[1]) if len(b) >= 2 else 0
        return (row_rank, xmin)

    kept.sort(key=_shelf_sort_key)
    row_counters: Dict[str, int] = {}
    for idx, it in enumerate(kept, start=1):
        b = it.get("bbox_2d") or [0, 0, 0, 0]
        row_label = str(it.get("shelf_row") or "").strip().lower() or infer_shelf_row_from_bbox(b)
        if not it.get("shelf_row"):
            it["shelf_row"] = row_label
        row_counters[row_label] = row_counters.get(row_label, 0) + 1
        it["product_index"] = idx
        it["position_on_shelf"] = row_counters[row_label]
    return kept, removed_count
