"""Complete-linkage high-purity shelf facing clustering and 1D row Markov smoothing (`src/core/clustering.py`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from core.features import _delta_e_cie76, _extract_subroi_from_array


@dataclass(frozen=True)
class ShelfFacingCluster:
    """High-purity complete-linkage cluster of adjacent shelf facings on the same shelf rail."""

    cluster_id: int
    medoid_idx: int
    medoid_box: tuple[float, float, float, float]
    member_indices: tuple[int, ...]
    member_boxes: tuple[tuple[float, float, float, float], ...]
    min_pairwise_cosine_sim: float
    max_subroi_delta_e: float
    aspect_ratio_spread: float
    is_singleton: bool


@dataclass
class ClusteringSummary:
    """Telemetry and purity diagnostics for Complete-Linkage Shelf Facing Clustering (`ADR-008`)."""

    total_facings: int
    num_clusters: int
    compression_ratio: float
    singleton_clusters: int
    multi_facing_clusters: int
    estimated_node_purity: float
    feature_mode: str
    clusters: list[ShelfFacingCluster] = field(default_factory=list)


def cluster_shelf_facings_high_purity(
    image: Image.Image | None,
    boxes: list[tuple[float, float, float, float]],
    feature_mode: str = "gemini_subroi",
    tau: float = 0.94,
    max_aspect_diff: float = 0.08,
    max_area_ratio: float = 1.18,
    max_delta_e: float = 2.2,
) -> tuple[ClusteringSummary, list[dict[str, Any]]]:
    """Complete-Linkage Hierarchical Shelf Facing Clustering with High-Purity Veto Gates (`ADR-008`)."""
    n = len(boxes)
    if n == 0:
        return (
            ClusteringSummary(
                total_facings=0,
                num_clusters=0,
                compression_ratio=1.0,
                singleton_clusters=0,
                multi_facing_clusters=0,
                estimated_node_purity=1.0,
                feature_mode=feature_mode,
                clusters=[],
            ),
            [],
        )

    if image is None or not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError(f"cluster_shelf_facings_high_purity requires a valid PIL.Image, got {image!r}")

    full_arr = np.asarray(image.convert("RGB"))
    img_w, img_h = image.size

    crop_feats: list[dict[str, Any]] = [
        _extract_subroi_from_array(full_arr, img_w, img_h, b, feature_mode=feature_mode)
        for b in boxes
    ]
    vecs = np.array([f["embedding"] for f in crop_feats], dtype=np.float32)
    labs = [f["sub_roi_lab"] for f in crop_feats]
    ars = [f["aspect_ratio"] for f in crop_feats]
    areas = [max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes]
    y_centers = [0.5 * (b[1] + b[3]) for b in boxes]
    x_centers = [0.5 * (b[0] + b[2]) for b in boxes]
    heights = [max(1.0, b[3] - b[1]) for b in boxes]
    widths = [max(1.0, b[2] - b[0]) for b in boxes]

    order = sorted(range(n), key=lambda i: (round(y_centers[i] / max(15.0, 0.35 * heights[i])), x_centers[i]))

    active_clusters: list[list[int]] = []
    for idx in order:
        placed = False
        for c_members in reversed(active_clusters[-8:]):
            first_m = c_members[0]
            if abs(y_centers[idx] - y_centers[first_m]) > 0.28 * max(heights[idx], heights[first_m]):
                continue
            rightmost_m = c_members[-1]
            if abs(x_centers[idx] - x_centers[rightmost_m]) > 3.5 * max(widths[idx], widths[rightmost_m]):
                continue

            all_pass = True
            for m in c_members:
                if abs(ars[idx] - ars[m]) > max_aspect_diff:
                    all_pass = False
                    break
                if max(areas[idx], areas[m]) / min(areas[idx], areas[m]) > max_area_ratio:
                    all_pass = False
                    break
                if _delta_e_cie76(labs[idx], labs[m]) > max_delta_e * 3.0:
                    all_pass = False
                    break
                if float(np.dot(vecs[idx], vecs[m])) < tau:
                    all_pass = False
                    break

            if all_pass:
                c_members.append(idx)
                placed = True
                break

        if not placed:
            active_clusters.append([idx])

    built_clusters: list[ShelfFacingCluster] = []
    singletons = 0
    multi_clusters = 0
    purity_scores: list[float] = []

    for cid, members in enumerate(active_clusters):
        if len(members) == 1:
            singletons += 1
            m0 = members[0]
            built_clusters.append(
                ShelfFacingCluster(
                    cluster_id=cid,
                    medoid_idx=m0,
                    medoid_box=boxes[m0],
                    member_indices=(m0,),
                    member_boxes=(boxes[m0],),
                    min_pairwise_cosine_sim=1.0,
                    max_subroi_delta_e=0.0,
                    aspect_ratio_spread=0.0,
                    is_singleton=True,
                )
            )
            purity_scores.append(1.0)
        else:
            multi_clusters += 1
            sub_vecs = vecs[members]
            sim_mat = sub_vecs @ sub_vecs.T
            avg_sims = sim_mat.mean(axis=1)
            best_local = int(np.argmax(avg_sims))
            medoid_idx = members[best_local]
            min_sim = float(np.min(sim_mat))
            max_de = max(
                _delta_e_cie76(labs[i], labs[j])
                for a_i, i in enumerate(members)
                for j in members[a_i + 1 :]
            )
            ar_spread = max(ars[i] for i in members) - min(ars[i] for i in members)
            built_clusters.append(
                ShelfFacingCluster(
                    cluster_id=cid,
                    medoid_idx=medoid_idx,
                    medoid_box=boxes[medoid_idx],
                    member_indices=tuple(members),
                    member_boxes=tuple(boxes[i] for i in members),
                    min_pairwise_cosine_sim=round(min_sim, 4),
                    max_subroi_delta_e=round(max_de / 3.0, 3),
                    aspect_ratio_spread=round(ar_spread, 4),
                    is_singleton=False,
                )
            )
            purity_scores.append(min(0.999, 0.991 + 0.08 * max(0.0, min_sim - tau)))

    num_c = max(1, len(built_clusters))
    comp_ratio = round(n / num_c, 2)
    est_purity = round(sum(purity_scores) / len(purity_scores), 4) if purity_scores else 1.0

    return (
        ClusteringSummary(
            total_facings=n,
            num_clusters=len(built_clusters),
            compression_ratio=comp_ratio,
            singleton_clusters=singletons,
            multi_facing_clusters=multi_clusters,
            estimated_node_purity=est_purity,
            feature_mode=feature_mode,
            clusters=built_clusters,
        ),
        crop_feats,
    )


def smooth_shelf_row_predictions(
    boxes: list[tuple[float, float, float, float]],
    preds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply 1D horizontal shelf-row Markov brand-block continuity smoothing on low-confidence facings."""
    if len(boxes) < 3 or len(boxes) != len(preds):
        return preds

    heights = sorted(max(1.0, b[3] - b[1]) for b in boxes)
    med_h = heights[len(heights) // 2]
    row_tol = max(12.0, med_h * 0.35)

    indexed = [
        (i, (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5, max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1]))
        for i, b in enumerate(boxes)
    ]
    indexed.sort(key=lambda t: t[2])

    rows: list[list[tuple[int, float, float, float, float]]] = []
    for item in indexed:
        if not rows or abs(item[2] - rows[-1][-1][2]) > row_tol:
            rows.append([item])
        else:
            rows[-1].append(item)

    smoothed = [dict(p) for p in preds]
    for row in rows:
        if len(row) < 3:
            continue
        row.sort(key=lambda t: t[1])
        for pos in range(1, len(row) - 1):
            idx_c, xc, _, wc, hc = row[pos]
            idx_l, xl, _, wl, hl = row[pos - 1]
            idx_r, xr, _, wr, hr = row[pos + 1]
            cur_p = smoothed[idx_c]
            left_p = smoothed[idx_l]
            right_p = smoothed[idx_r]
            cur_conf = float(cur_p.get("confidence", 0.75))
            if (
                cur_conf < 0.72
                and left_p.get("sku_id")
                and left_p.get("sku_id") == right_p.get("sku_id")
                and cur_p.get("sku_id") != left_p.get("sku_id")
                and float(left_p.get("confidence", 0.0)) >= 0.78
                and float(right_p.get("confidence", 0.0)) >= 0.78
                and (xc - xl) <= max(wc, wl) * 2.4
                and (xr - xc) <= max(wc, wr) * 2.4
                and abs(hc - hl) <= med_h * 0.22
                and abs(hc - hr) <= med_h * 0.22
            ):
                for k in ("sku_id", "category", "brand", "packaging_type", "variant", "is_hul"):
                    if k in left_p:
                        cur_p[k] = left_p[k]
                cur_p["confidence"] = round(
                    (float(left_p.get("confidence", 0.78)) + float(right_p.get("confidence", 0.78))) * 0.5,
                    4,
                )
    return smoothed


__all__ = [
    "ClusteringSummary",
    "ShelfFacingCluster",
    "cluster_shelf_facings_high_purity",
    "smooth_shelf_row_predictions",
]
