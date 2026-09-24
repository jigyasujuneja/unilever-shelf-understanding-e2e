"""Spatial Geometry, 2nd-Row 'Depth Ghost' NMS Suppression, Shelf Row Clustering, and Bipartite IoU Matching.

Ported and unified from Riley's `facing_utils.py`, `geometry.py`, and `metrics.py` for SPEC-005.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def compute_box_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    """Compute 2D Intersection-over-Union (IoU) for [ymin, xmin, ymax, xmax] boxes."""
    inter_ymin = max(int(box_a[0]), int(box_b[0]))
    inter_xmin = max(int(box_a[1]), int(box_b[1]))
    inter_ymax = min(int(box_a[2]), int(box_b[2]))
    inter_xmax = min(int(box_a[3]), int(box_b[3]))
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_area = inter_h * inter_w
    if inter_area <= 0:
        return 0.0
    area_a = max(1, (int(box_a[2]) - int(box_a[0])) * (int(box_a[3]) - int(box_a[1])))
    area_b = max(1, (int(box_b[2]) - int(box_b[0])) * (int(box_b[3]) - int(box_b[1])))
    return float(inter_area) / float(area_a + area_b - inter_area)


def deduplicate_depth_stacked_facings(
    candidates: List[Dict[str, Any]],
    x_overlap_threshold: float = 0.55,
    max_vertical_recess_ratio: float = 0.65,
) -> Tuple[List[Dict[str, Any]], int]:
    """Suppress 2nd-row 'depth ghosts' (back-row stacked items peeking above/behind front-row facings).

    Returns:
        (kept_front_row_facings, depth_duplicates_filtered_count)
    """
    if not candidates:
        return ([], 0)

    def _sort_key(item: Dict[str, Any]) -> Tuple[int, int]:
        b = item.get("bbox_2d", [0, 0, 0, 0])
        ymax = int(b[2])
        area = max(1, (int(b[2]) - int(b[0])) * (int(b[3]) - int(b[1])))
        return (ymax, area)

    ordered = sorted(candidates, key=_sort_key, reverse=True)
    kept: List[Dict[str, Any]] = []
    filtered_count = 0

    for cand in ordered:
        c_box = cand.get("bbox_2d", [0, 0, 0, 0])
        c_ymin, c_xmin, c_ymax, c_xmax = int(c_box[0]), int(c_box[1]), int(c_box[2]), int(c_box[3])
        c_w = max(1, c_xmax - c_xmin)
        c_h = max(1, c_ymax - c_ymin)
        c_area = c_w * c_h
        c_ycenter = (c_ymin + c_ymax) / 2.0

        is_depth_ghost = False
        for front in kept:
            f_box = front.get("bbox_2d", [0, 0, 0, 0])
            f_ymin, f_xmin, f_ymax, f_xmax = int(f_box[0]), int(f_box[1]), int(f_box[2]), int(f_box[3])
            f_w = max(1, f_xmax - f_xmin)
            f_h = max(1, f_ymax - f_ymin)
            f_area = f_w * f_h
            f_ycenter = (f_ymin + f_ymax) / 2.0

            inter_x = max(0, min(c_xmax, f_xmax) - max(c_xmin, f_xmin))
            x_overlap = inter_x / float(min(c_w, f_w))
            vert_dist = abs(c_ycenter - f_ycenter)

            if (
                x_overlap >= x_overlap_threshold
                and vert_dist <= f_h * max_vertical_recess_ratio
                and c_area <= f_area * 0.92
            ):
                is_depth_ghost = True
                break

        if is_depth_ghost:
            filtered_count += 1
        else:
            kept.append(cand)

    # Restore natural left-to-right, top-to-bottom reading order
    kept.sort(key=lambda item: (int(item["bbox_2d"][0]) // 150, int(item["bbox_2d"][1])))
    return (kept, filtered_count)


def cluster_boxes_into_shelf_rows(
    boxes: Sequence[Sequence[int]],
    num_rows: int = 5,
) -> List[str]:
    """Cluster [ymin, xmin, ymax, xmax] boxes into named shelf levels ('top', 'upper_middle', 'middle', 'lower_middle', 'bottom')."""
    if not boxes:
        return []
    row_names = ["top", "upper_middle", "middle", "lower_middle", "bottom"]
    y_centers = [(int(b[0]) + int(b[2])) / 2.0 for b in boxes]
    y_min, y_max = min(y_centers), max(y_centers)
    span = max(1.0, y_max - y_min)
    assigned: List[str] = []
    for yc in y_centers:
        idx = int(((yc - y_min) / span) * len(row_names))
        idx = max(0, min(len(row_names) - 1, idx))
        assigned.append(row_names[idx])
    return assigned
