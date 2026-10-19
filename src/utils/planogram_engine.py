"""Stage 6 Out-of-Stock (OOS) Void Detection, Brand Block Purity, and Planogram Compliance Engine.

Implements Unilever's merchandising and shelf-compliance rules across both Modern Trade (MT)
and General Trade (GT) shelves:
  - Bay Scope Router: scores only SKUs inside Class 0 display bays (`asset_boxes`) when bays are
    present (`scope_mode="class0_bay_scoped"`), and falls back to full-image SKU scoring when no
    Class 0 bay exists (`scope_mode="full_image_hanger_fallback"`, e.g. hanging strips / hangers).
  - Shelf Plank Clustering: groups packs into physical horizontal shelf planks (`row_idx = 0..R-1`)
    using vertical band overlap so row transitions never corrupt horizontal brand-block purity.
  - Per-Row OOS Void Detector: computes median pack width strictly per shelf plank and detects
    inter-pack gaps plus left/right bay-edge voids (when inside a Class 0 bay), returning exact
    `oos_void_boxes` for UI visualization.
  - Brand Block Purity: computes weighted per-row horizontal brand-block purity and vertical
    brand-block alignment across adjacent shelf planks.
  - Planogram Spec Audit: checks optional `Outlet_PLG` / `AWP_Code` compliance rules.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from utils.hul_taxonomy import normalize_brand_folder

Box = tuple[float, float, float, float]


def _to_box(b: Sequence[float]) -> Box:
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def _median(vals: list[float]) -> float:
    if not vals:
        return 1.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _resolve_item(
    pid: int | dict | None,
    catalog: dict[int, dict] | None,
) -> dict | None:
    if pid is None:
        return None
    if isinstance(pid, dict):
        return pid
    if catalog is None:
        return None
    return catalog.get(pid)


def _item_brand_and_hul(item: dict) -> tuple[str, bool]:
    brand = str(item.get("brand") or item.get("product") or "Unknown")
    if "is_hul" in item:
        is_hul = bool(item["is_hul"])
    elif "is_unilever" in item:
        is_hul = bool(item["is_unilever"])
    else:
        nb = normalize_brand_folder(brand)
        is_hul = bool(nb and nb.get("is_hul", False))
    return brand, is_hul


def _box_in_bay(box: Box, bay: Box, min_containment: float = 0.45) -> bool:
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    if bay[0] <= cx <= bay[2] and bay[1] <= cy <= bay[3]:
        return True
    ix1 = max(box[0], bay[0])
    iy1 = max(box[1], bay[1])
    ix2 = min(box[2], bay[2])
    iy2 = min(box[3], bay[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return (inter / area) >= min_containment


def cluster_shelf_planks(
    boxes: Sequence[Sequence[float]],
    y_overlap_ratio: float = 0.60,
) -> list[list[int]]:
    """Group boxes top-to-bottom into shelf planks, each sorted left-to-right by x1.

    Two packs belong to the same shelf plank when their vertical spans overlap by at least
    `(1 - y_overlap_ratio)` of the shorter pack's height, or when their vertical centers
    lie within `0.50 * max(pack_height, plank_median_height)`.
    """
    if not boxes:
        return []
    norm_boxes = [_to_box(b) for b in boxes]
    order = sorted(
        range(len(norm_boxes)),
        key=lambda i: (0.5 * (norm_boxes[i][1] + norm_boxes[i][3]), norm_boxes[i][0]),
    )
    min_y_overlap = max(0.15, 1.0 - y_overlap_ratio)

    planks: list[dict] = []
    for idx in order:
        b = norm_boxes[idx]
        bh = max(1.0, b[3] - b[1])
        bcy = 0.5 * (b[1] + b[3])
        best_p: dict | None = None
        best_score = -1.0

        for p in planks:
            p_cy = p["med_cy"]
            p_h = p["med_h"]
            ov = max(0.0, min(b[3], p["max_y2"]) - max(b[1], p["min_y1"]))
            ov_ratio = ov / max(1.0, min(bh, p_h))
            cy_diff = abs(bcy - p_cy)
            if ov_ratio >= min_y_overlap or cy_diff <= 0.50 * max(bh, p_h):
                score = ov_ratio - (cy_diff / max(1.0, max(bh, p_h)))
                if score > best_score:
                    best_score = score
                    best_p = p

        if best_p is None:
            planks.append(
                {
                    "indices": [idx],
                    "cys": [bcy],
                    "hs": [bh],
                    "med_cy": bcy,
                    "med_h": bh,
                    "min_y1": b[1],
                    "max_y2": b[3],
                }
            )
        else:
            best_p["indices"].append(idx)
            best_p["cys"].append(bcy)
            best_p["hs"].append(bh)
            best_p["med_cy"] = _median(best_p["cys"])
            best_p["med_h"] = _median(best_p["hs"])
            best_p["min_y1"] = min(best_p["min_y1"], b[1])
            best_p["max_y2"] = max(best_p["max_y2"], b[3])

    planks.sort(key=lambda p: p["med_cy"])
    result: list[list[int]] = []
    for p in planks:
        row_sorted = sorted(p["indices"], key=lambda i: (norm_boxes[i][0], norm_boxes[i][1]))
        result.append(row_sorted)
    return result


def detect_row_oos_voids(
    row_boxes: Sequence[Sequence[float]],
    row_idx: int = 0,
    scope_bounds: Sequence[float] | None = None,
    gap_factor: float = 1.5,
) -> dict:
    """Detect horizontal Out-of-Stock (OOS) voids along a single shelf plank.

    Uses the median width of packs on this specific plank (`row_med_w`) so narrow sachets on top
    planks are not masked by wide 2L bottles on bottom planks. When `scope_bounds` (`[bx1, by1, bx2, by2]`)
    is provided from a Class 0 display bay, also detects left-edge and right-edge bay voids.
    """
    if not row_boxes:
        return {
            "row_idx": row_idx,
            "row_med_w": 0.0,
            "oos_void_count": 0,
            "estimated_missing_facings": 0,
            "total_void_px": 0.0,
            "total_row_span_px": 0.0,
            "oos_void_boxes": [],
        }

    r_sorted = sorted((_to_box(b) for b in row_boxes), key=lambda b: (b[0], b[1]))
    widths = [max(1.0, b[2] - b[0]) for b in r_sorted]
    row_med_w = _median(widths)
    thr_px = gap_factor * row_med_w
    vy1 = round(min(b[1] for b in r_sorted), 1)
    vy2 = round(max(b[3] for b in r_sorted), 1)

    void_boxes: list[dict] = []
    total_void_px = 0.0

    # 1. Left bay-edge void (only when physical Class 0 bay bounds are known)
    if scope_bounds is not None:
        sb = _to_box(scope_bounds)
        left_gap = r_sorted[0][0] - sb[0]
        if left_gap >= thr_px:
            missing = max(1, round(left_gap / row_med_w))
            void_boxes.append(
                {
                    "box": [round(sb[0], 1), vy1, round(r_sorted[0][0], 1), vy2],
                    "row_idx": row_idx,
                    "missing_facings": int(missing),
                    "kind": "bay_edge",
                }
            )
            total_void_px += left_gap

    # 2. Inter-pack voids along the plank
    for prev_b, next_b in zip(r_sorted, r_sorted[1:], strict=False):
        gap = next_b[0] - prev_b[2]
        if gap >= thr_px:
            missing = max(1, round(gap / row_med_w))
            void_boxes.append(
                {
                    "box": [round(prev_b[2], 1), vy1, round(next_b[0], 1), vy2],
                    "row_idx": row_idx,
                    "missing_facings": int(missing),
                    "kind": "inter_pack",
                }
            )
            total_void_px += gap

    # 3. Right bay-edge void (only when physical Class 0 bay bounds are known)
    if scope_bounds is not None:
        sb = _to_box(scope_bounds)
        right_gap = sb[2] - r_sorted[-1][2]
        if right_gap >= thr_px:
            missing = max(1, round(right_gap / row_med_w))
            void_boxes.append(
                {
                    "box": [round(r_sorted[-1][2], 1), vy1, round(sb[2], 1), vy2],
                    "row_idx": row_idx,
                    "missing_facings": int(missing),
                    "kind": "bay_edge",
                }
            )
            total_void_px += right_gap

    if scope_bounds is not None:
        sb = _to_box(scope_bounds)
        span_px = max(1.0, sb[2] - sb[0])
    elif len(r_sorted) >= 2:
        span_px = max(1.0, r_sorted[-1][2] - r_sorted[0][0])
    else:
        span_px = max(1.0, r_sorted[0][2] - r_sorted[0][0])

    return {
        "row_idx": row_idx,
        "row_med_w": round(row_med_w, 2),
        "oos_void_count": len(void_boxes),
        "estimated_missing_facings": sum(int(v["missing_facings"]) for v in void_boxes),
        "total_void_px": round(total_void_px, 2),
        "total_row_span_px": round(span_px, 2),
        "oos_void_boxes": void_boxes,
    }


def compute_row_and_vertical_brand_purity(
    plank_rows: Sequence[Sequence[int]],
    boxes: Sequence[Sequence[float]],
    box_brands: dict[int, str],
) -> dict:
    """Compute weighted per-row horizontal brand-block purity and vertical block alignment.

    Unidentified slots (`__unidentified_<idx>__`) always break brand blocks and never group
    into a single block.
    """
    if not plank_rows:
        return {
            "brand_block_purity": 0.0,
            "vertical_block_alignment": 0.0,
            "row_purities": [],
        }

    norm_boxes = [_to_box(b) for b in boxes]
    weighted_sum = 0.0
    total_facings = 0
    row_purities: list[dict] = []
    row_brand_spans: list[dict[str, tuple[float, float]]] = []

    for r_idx, r_indices in enumerate(plank_rows):
        if not r_indices:
            row_brand_spans.append({})
            continue
        r_brands = [box_brands.get(i, f"__unidentified_{i}__") for i in r_indices]
        r_known = [b for b in r_brands if not b.startswith("__unidentified_")]

        if not r_known:
            r_purity = 0.0
        elif len(r_brands) <= 1:
            r_purity = 1.0
        else:
            transitions = sum(
                1 for a, b in zip(r_brands, r_brands[1:], strict=False) if a != b
            )
            min_transitions = max(0, len(set(r_known)) - 1)
            extra_transitions = max(0, transitions - min_transitions)
            r_purity = max(0.0, 1.0 - (extra_transitions / max(1, len(r_brands) - 1)))

        weighted_sum += r_purity * len(r_indices)
        total_facings += len(r_indices)
        row_purities.append(
            {
                "row_idx": r_idx,
                "facings": len(r_indices),
                "distinct_brands": len(set(r_known)),
                "purity": round(r_purity, 4),
            }
        )

        spans: dict[str, tuple[float, float]] = {}
        for i in r_indices:
            br = box_brands.get(i, "")
            if not br or br.startswith("__unidentified_"):
                continue
            b = norm_boxes[i]
            if br not in spans:
                spans[br] = (b[0], b[2])
            else:
                spans[br] = (min(spans[br][0], b[0]), max(spans[br][1], b[2]))
        row_brand_spans.append(spans)

    brand_block_purity = round(weighted_sum / max(1, total_facings), 4) if total_facings else 0.0

    # Vertical brand-block alignment across adjacent planks
    adj_pairs = 0
    aligned_pairs = 0
    for r_a, r_b in zip(row_brand_spans, row_brand_spans[1:], strict=False):
        shared_brands = set(r_a.keys()) & set(r_b.keys())
        for br in shared_brands:
            adj_pairs += 1
            sa = r_a[br]
            sb = r_b[br]
            ov = max(0.0, min(sa[1], sb[1]) - max(sa[0], sb[0]))
            min_span = max(1.0, min(sa[1] - sa[0], sb[1] - sb[0]))
            if (ov / min_span) >= 0.30:
                aligned_pairs += 1

    if total_facings == 0 or all(
        b.startswith("__unidentified_") for b in box_brands.values()
    ):
        vertical_alignment = 0.0
    elif adj_pairs == 0:
        vertical_alignment = 1.0
    else:
        vertical_alignment = round(aligned_pairs / adj_pairs, 4)

    return {
        "brand_block_purity": brand_block_purity,
        "vertical_block_alignment": vertical_alignment,
        "row_purities": row_purities,
    }


def check_planogram_spec(
    metrics_dict: dict,
    planogram_spec: dict | None = None,
) -> dict:
    """Audit computed shelf/bay metrics against an optional Outlet_PLG / AWP_Code rule spec."""
    if not planogram_spec:
        return {
            "has_spec": False,
            "spec_id": None,
            "is_compliant": True,
            "violations": [],
        }

    violations: list[str] = []
    spec_id = planogram_spec.get("spec_id") or planogram_spec.get("outlet_plg") or planogram_spec.get("awp_code")

    present_brands = set((metrics_dict.get("brand_counts") or metrics_dict.get("top_brand_facings") or {}).keys())
    req_brands = planogram_spec.get("required_brands") or []
    for br in req_brands:
        if br not in present_brands:
            violations.append(f"missing_required_brand:{br}")

    min_sos = planogram_spec.get("min_hul_sos")
    if min_sos is not None and float(metrics_dict.get("hul_facing_sos", 0.0)) < float(min_sos):
        violations.append(
            f"hul_sos_below_target:{metrics_dict.get('hul_facing_sos', 0.0):.3f}<{float(min_sos):.3f}"
        )

    min_purity = planogram_spec.get("min_brand_block_purity")
    if min_purity is not None and float(metrics_dict.get("brand_block_purity", 0.0)) < float(min_purity):
        violations.append(
            f"brand_block_purity_below_target:{metrics_dict.get('brand_block_purity', 0.0):.3f}<{float(min_purity):.3f}"
        )

    if planogram_spec.get("forbid_competitors_in_bay") and int(metrics_dict.get("competitor_intrusion_count", 0)) > 0:
        violations.append(
            f"competitor_intrusion_in_bay:{int(metrics_dict.get('competitor_intrusion_count', 0))}"
        )

    max_voids = planogram_spec.get("max_oos_voids")
    if max_voids is not None and int(metrics_dict.get("oos_void_count", 0)) > int(max_voids):
        violations.append(
            f"oos_voids_exceed_limit:{int(metrics_dict.get('oos_void_count', 0))}>{int(max_voids)}"
        )

    req_row_brands = planogram_spec.get("required_row_brands") or {}
    row_brand_map = metrics_dict.get("row_brands_by_plank") or {}
    for rk, expected in req_row_brands.items():
        r_idx = int(rk)
        actual_on_row = set(row_brand_map.get(r_idx, []))
        for br in expected:
            if br not in actual_on_row:
                violations.append(f"row_{r_idx}_missing_brand:{br}")

    return {
        "has_spec": True,
        "spec_id": spec_id,
        "is_compliant": len(violations) == 0,
        "violations": violations,
    }


def _evaluate_scope(
    scope_indices: list[int],
    all_boxes: list[Box],
    box_brands: dict[int, str],
    box_is_hul: dict[int, bool],
    box_known: dict[int, bool],
    scope_bounds: Box | None = None,
    gap_factor: float = 1.5,
) -> dict:
    """Evaluate planks, OOS voids, brand purity, and HUL share on a subset of box indices."""
    if not scope_indices:
        return {
            "total_facings": 0,
            "hul_facings": 0,
            "rival_facings": 0,
            "unidentified_facings": 0,
            "hul_facing_sos": 0.0,
            "brand_block_purity": 0.0,
            "vertical_block_alignment": 0.0,
            "distinct_brands": 0,
            "brand_counts": {},
            "top_brand_facings": {},
            "shelf_plank_count": 0,
            "row_purities": [],
            "row_brands_by_plank": {},
            "oos_void_count": 0,
            "estimated_missing_facings": 0,
            "empty_shelf_share": 0.0,
            "oos_void_boxes": [],
        }

    sub_boxes = [all_boxes[i] for i in scope_indices]
    local_planks = cluster_shelf_planks(sub_boxes)
    global_planks = [[scope_indices[loc_i] for loc_i in row] for row in local_planks]

    purity_res = compute_row_and_vertical_brand_purity(global_planks, all_boxes, box_brands)

    oos_void_count = 0
    estimated_missing_facings = 0
    total_void_px = 0.0
    total_row_span_px = 0.0
    oos_void_boxes: list[dict] = []
    row_brands_by_plank: dict[int, list[str]] = {}

    for r_idx, row_global_idxs in enumerate(global_planks):
        r_boxes = [all_boxes[i] for i in row_global_idxs]
        v_res = detect_row_oos_voids(
            r_boxes, row_idx=r_idx, scope_bounds=scope_bounds, gap_factor=gap_factor
        )
        oos_void_count += v_res["oos_void_count"]
        estimated_missing_facings += v_res["estimated_missing_facings"]
        total_void_px += v_res["total_void_px"]
        total_row_span_px += v_res["total_row_span_px"]
        oos_void_boxes.extend(v_res["oos_void_boxes"])
        row_brands_by_plank[r_idx] = [
            box_brands[i] for i in row_global_idxs if box_known.get(i, False)
        ]

    empty_shelf_share = round(min(1.0, total_void_px / max(1.0, total_row_span_px)), 4)

    hul_count = sum(1 for i in scope_indices if box_known.get(i, False) and box_is_hul.get(i, False))
    rival_count = sum(
        1 for i in scope_indices if box_known.get(i, False) and not box_is_hul.get(i, False)
    )
    unidentified_count = sum(1 for i in scope_indices if not box_known.get(i, False))
    known_brands = [box_brands[i] for i in scope_indices if box_known.get(i, False)]
    brand_counter = Counter(known_brands)

    return {
        "total_facings": len(scope_indices),
        "hul_facings": hul_count,
        "rival_facings": rival_count,
        "unidentified_facings": unidentified_count,
        "hul_facing_sos": round(hul_count / max(1, len(scope_indices)), 4),
        "brand_block_purity": purity_res["brand_block_purity"],
        "vertical_block_alignment": purity_res["vertical_block_alignment"],
        "distinct_brands": len(brand_counter),
        "brand_counts": dict(brand_counter),
        "top_brand_facings": dict(brand_counter.most_common(5)),
        "shelf_plank_count": len(global_planks),
        "row_purities": purity_res["row_purities"],
        "row_brands_by_plank": row_brands_by_plank,
        "oos_void_count": oos_void_count,
        "estimated_missing_facings": estimated_missing_facings,
        "empty_shelf_share": empty_shelf_share,
        "oos_void_boxes": oos_void_boxes,
    }


def evaluate_shelf_planogram(
    boxes: Sequence[Sequence[float]],
    pred_ids: Sequence[int | dict | None],
    catalog: dict[int, dict] | None = None,
    asset_boxes: Sequence[Sequence[float]] | None = None,
    image_size: tuple[int, int] | None = None,
    planogram_spec: dict | None = None,
    gap_factor: float = 1.5,
) -> dict:
    """Stage 6 Planogram Compliance Engine: bay scope routing, per-plank OOS voids, and brand blocks.

    Scope routing rule (per Unilever merchandising protocol):
      - When `asset_boxes` (`Class 0` display bays) are present and contain SKUs:
        `scope_mode = "class0_bay_scoped"` and primary compliance metrics score only SKUs inside
        the Class 0 bay(s).
      - When `asset_boxes` is empty (e.g. hanging strips / hangers / open shelf):
        `scope_mode = "full_image_hanger_fallback"` and compliance scores all full-image SKUs.
    """
    norm_assets = [_to_box(ab) for ab in (asset_boxes or [])]
    if not boxes:
        base_empty = {
            "scope_mode": "class0_bay_scoped" if norm_assets else "full_image_hanger_fallback",
            "total_facings": 0,
            "full_shelf_facings": 0,
            "hul_facings": 0,
            "rival_facings": 0,
            "unidentified_facings": 0,
            "hul_facing_sos": 0.0,
            "brand_block_purity": 0.0,
            "vertical_block_alignment": 0.0,
            "distinct_brands": 0,
            "top_brand_facings": {},
            "shelf_plank_count": 0,
            "row_purities": [],
            "asset_bays": len(norm_assets),
            "competitor_intrusion_count": 0,
            "asset_bay_purity": 1.0,
            "oos_void_count": 0,
            "estimated_missing_facings": 0,
            "empty_shelf_share": 0.0,
            "oos_void_boxes": [],
            "per_bay": [],
        }
        if image_size is not None:
            base_empty["image_size"] = [int(image_size[0]), int(image_size[1])]
        base_empty["spec_compliance"] = check_planogram_spec(base_empty, planogram_spec)
        return base_empty

    norm_boxes = [_to_box(b) for b in boxes]
    box_brands: dict[int, str] = {}
    box_is_hul: dict[int, bool] = {}
    box_known: dict[int, bool] = {}

    for idx in range(len(norm_boxes)):
        pid = pred_ids[idx] if idx < len(pred_ids) else None
        item = _resolve_item(pid, catalog)
        if not item:
            box_brands[idx] = f"__unidentified_{idx}__"
            box_is_hul[idx] = False
            box_known[idx] = False
        else:
            brand, is_hul = _item_brand_and_hul(item)
            box_brands[idx] = brand
            box_is_hul[idx] = is_hul
            box_known[idx] = True

    full_indices = list(range(len(norm_boxes)))
    full_eval = _evaluate_scope(
        full_indices,
        norm_boxes,
        box_brands,
        box_is_hul,
        box_known,
        scope_bounds=None,
        gap_factor=gap_factor,
    )

    per_bay: list[dict] = []
    in_any_bay_indices: list[int] = []
    seen_in_bay: set[int] = set()
    intrusions = 0
    in_bay_total = 0

    for b_idx, ab in enumerate(norm_assets):
        bay_idxs = [i for i, b in enumerate(norm_boxes) if _box_in_bay(b, ab)]
        for i in bay_idxs:
            if i not in seen_in_bay:
                seen_in_bay.add(i)
                in_any_bay_indices.append(i)
        bay_intrusions = sum(1 for i in bay_idxs if not box_is_hul.get(i, False))
        bay_purity = round(1.0 - (bay_intrusions / max(1, len(bay_idxs))), 4) if bay_idxs else 1.0
        bay_eval = _evaluate_scope(
            bay_idxs,
            norm_boxes,
            box_brands,
            box_is_hul,
            box_known,
            scope_bounds=ab,
            gap_factor=gap_factor,
        )
        per_bay.append(
            {
                "bay_index": b_idx,
                "bay_box": [round(v, 1) for v in ab],
                "facings": bay_eval["total_facings"],
                "hul_facings": bay_eval["hul_facings"],
                "rival_facings": bay_eval["rival_facings"],
                "competitor_intrusion_count": bay_intrusions,
                "asset_bay_purity": bay_purity,
                "brand_block_purity": bay_eval["brand_block_purity"],
                "vertical_block_alignment": bay_eval["vertical_block_alignment"],
                "oos_void_count": bay_eval["oos_void_count"],
                "estimated_missing_facings": bay_eval["estimated_missing_facings"],
                "empty_shelf_share": bay_eval["empty_shelf_share"],
                "oos_void_boxes": bay_eval["oos_void_boxes"],
            }
        )

    if norm_assets:
        in_bay_total = len(in_any_bay_indices)
        intrusions = sum(1 for i in in_any_bay_indices if not box_is_hul.get(i, False))
        asset_bay_purity = round(1.0 - (intrusions / max(1, in_bay_total)), 4) if in_bay_total > 0 else 1.0
    else:
        asset_bay_purity = 1.0

    # Scope router: Class 0 bay scoped when bays exist and hold SKUs; else full-image hanger fallback
    if norm_assets and in_any_bay_indices:
        scope_mode = "class0_bay_scoped"
        # Aggregate per-bay OOS voids so each bay uses its own physical left/right bay walls
        if len(norm_assets) == 1:
            primary = _evaluate_scope(
                in_any_bay_indices,
                norm_boxes,
                box_brands,
                box_is_hul,
                box_known,
                scope_bounds=norm_assets[0],
                gap_factor=gap_factor,
            )
        else:
            primary = _evaluate_scope(
                in_any_bay_indices,
                norm_boxes,
                box_brands,
                box_is_hul,
                box_known,
                scope_bounds=None,
                gap_factor=gap_factor,
            )
            merged_voids = [vb for pb in per_bay for vb in pb["oos_void_boxes"]]
            primary["oos_void_boxes"] = merged_voids
            primary["oos_void_count"] = len(merged_voids)
            primary["estimated_missing_facings"] = sum(
                int(pb["estimated_missing_facings"]) for pb in per_bay
            )
            primary["empty_shelf_share"] = round(
                sum(float(pb["empty_shelf_share"]) for pb in per_bay) / max(1, len(per_bay)),
                4,
            )
    else:
        scope_mode = "full_image_hanger_fallback"
        primary = full_eval

    out = {
        "scope_mode": scope_mode,
        "total_facings": primary["total_facings"],
        "full_shelf_facings": len(norm_boxes),
        "hul_facings": primary["hul_facings"],
        "rival_facings": primary["rival_facings"],
        "unidentified_facings": primary["unidentified_facings"],
        "hul_facing_sos": primary["hul_facing_sos"],
        "brand_block_purity": primary["brand_block_purity"],
        "vertical_block_alignment": primary["vertical_block_alignment"],
        "distinct_brands": primary["distinct_brands"],
        "top_brand_facings": primary["top_brand_facings"],
        "shelf_plank_count": primary["shelf_plank_count"],
        "row_purities": primary["row_purities"],
        "asset_bays": len(norm_assets),
        "competitor_intrusion_count": intrusions,
        "asset_bay_purity": asset_bay_purity,
        "oos_void_count": primary["oos_void_count"],
        "estimated_missing_facings": primary["estimated_missing_facings"],
        "empty_shelf_share": primary["empty_shelf_share"],
        "oos_void_boxes": primary["oos_void_boxes"],
        "per_bay": per_bay,
    }
    if image_size is not None:
        out["image_size"] = [int(image_size[0]), int(image_size[1])]
    spec_input = {**primary, **out}
    out["spec_compliance"] = check_planogram_spec(spec_input, planogram_spec)
    return out
