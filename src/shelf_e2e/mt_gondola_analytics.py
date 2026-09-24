"""Unilever Modern Trade (`MT`) Complete 8-KPI Gondola Analytics Engine (`SPEC-006`).

Closes the 4 Gondola-Level MT KPI gaps identified in `SPEC-006` Section 4:
1. `compute_linear_and_area_sos`: Linear Horizontal Share of Shelf (`sum(w_u)/sum(w_all)`) + 2D Area Share of Shelf (`sum(w_u*h_u)/sum(w_all*h_all)`) alongside Facing Count SOS.
2. `detect_shelf_oos_void_gaps`: Horizontal shelf-row clustering (`y_center`) and empty-shelf void detection where `gap_px >= 1.25 * median_facing_width` -> `[x1, y1, x2, y2]` `OOS_VOID_GAP` + estimated missing facings.
3. `compute_planogram_sequence_compliance`: Normalized Levenshtein edit distance (`1.0 - edit_dist / max_len`) between observed left-to-right shelf row sequence and target planogram sequence.
4. `compute_brand_block_purity`: Contiguous brand-block purity (`1.0 - competitor_intrusions / total_brand_facings`) detecting competitor SKUs breaking a Unilever brand block (Dove / TRESemme / Vaseline / Sunsilk).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import statistics
from typing import Any, Dict, List, Optional

from shelf_e2e.schemas import ResolvedSKU


@dataclass
class OOSVoidGap:
    """Physical empty-shelf Out-Of-Stock (`OOS`) void gap localized on a shelf tier."""

    shelf_row: int
    void_box_xyxy: List[float]
    void_width_px: float
    estimated_missing_facings: int
    adjacent_left_sku: str
    adjacent_right_sku: str


@dataclass
class MTGondolaAuditReport:
    """Complete 8-KPI Unilever Modern Trade (`MT`) Gondola Execution Audit."""

    facing_count_sos_pct: float
    linear_width_sos_pct: float
    area_2d_sos_pct: float
    oos_void_gaps: List[OOSVoidGap] = field(default_factory=list)
    total_missing_facings_est: int = 0
    planogram_sequence_compliance_pct: float = 100.0
    brand_block_purity_pct: float = 100.0
    competitor_intrusions_count: int = 0
    shelf_tiers_detected: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_unilever_sku(base_pack_id: str) -> bool:
    u_prefixes = ("BP-DOVE", "BP-TRES", "BP-VAS", "BP-SUN", "BP-LUX", "BP-PONDS", "BP-LIFEBUOY", "BP-CLEAR")
    return any(base_pack_id.upper().startswith(p) for p in u_prefixes)


def _extract_brand(base_pack_id: str) -> str:
    parts = base_pack_id.upper().split("-")
    if len(parts) >= 2 and parts[0] == "BP":
        return parts[1]
    return parts[0]


def cluster_facings_into_shelf_rows(
    resolved_skus: List[ResolvedSKU], row_merge_tolerance_px: Optional[float] = None
) -> Dict[int, List[ResolvedSKU]]:
    """Group detected facings into horizontal shelf rows (1 = top shelf, increasing downward) sorted left-to-right."""
    if not resolved_skus:
        return {}

    heights = [max(1.0, s.box_xyxy[3] - s.box_xyxy[1]) for s in resolved_skus]
    median_h = statistics.median(heights) if heights else 60.0
    tol = row_merge_tolerance_px or (0.55 * median_h)

    sorted_by_y = sorted(resolved_skus, key=lambda s: 0.5 * (s.box_xyxy[1] + s.box_xyxy[3]))
    rows: List[List[ResolvedSKU]] = []
    row_centers: List[float] = []

    for sku in sorted_by_y:
        yc = 0.5 * (sku.box_xyxy[1] + sku.box_xyxy[3])
        placed = False
        for idx, rc in enumerate(row_centers):
            if abs(yc - rc) <= tol:
                rows[idx].append(sku)
                row_centers[idx] = sum(0.5 * (x.box_xyxy[1] + x.box_xyxy[3]) for x in rows[idx]) / len(rows[idx])
                placed = True
                break
        if not placed:
            rows.append([sku])
            row_centers.append(yc)

    paired = sorted(zip(row_centers, rows), key=lambda pair: pair[0])
    result: Dict[int, List[ResolvedSKU]] = {}
    for row_idx, (_, skus_in_row) in enumerate(paired, start=1):
        result[row_idx] = sorted(skus_in_row, key=lambda s: s.box_xyxy[0])
    return result


def compute_linear_and_area_sos(resolved_skus: List[ResolvedSKU]) -> Dict[str, float]:
    """Compute Facing-Count SOS %, Linear Horizontal Width SOS %, and 2D Area SOS %."""
    if not resolved_skus:
        return {"facing_count_sos_pct": 0.0, "linear_width_sos_pct": 0.0, "area_2d_sos_pct": 0.0}

    total_count = len(resolved_skus)
    u_count = 0
    total_w = 0.0
    u_w = 0.0
    total_area = 0.0
    u_area = 0.0

    for s in resolved_skus:
        w = max(1.0, s.box_xyxy[2] - s.box_xyxy[0])
        h = max(1.0, s.box_xyxy[3] - s.box_xyxy[1])
        area = w * h
        total_w += w
        total_area += area
        if _is_unilever_sku(s.base_pack_id):
            u_count += 1
            u_w += w
            u_area += area

    return {
        "facing_count_sos_pct": round(100.0 * u_count / max(1, total_count), 2),
        "linear_width_sos_pct": round(100.0 * u_w / max(1.0, total_w), 2),
        "area_2d_sos_pct": round(100.0 * u_area / max(1.0, total_area), 2),
    }


def detect_shelf_oos_void_gaps(
    resolved_skus: List[ResolvedSKU], gap_multiplier: float = 1.25
) -> List[OOSVoidGap]:
    """Scan each shelf row left-to-right and localize physical empty-shelf OOS void gaps."""
    rows = cluster_facings_into_shelf_rows(resolved_skus)
    if not rows:
        return []

    all_widths = [max(1.0, s.box_xyxy[2] - s.box_xyxy[0]) for s in resolved_skus]
    median_w = statistics.median(all_widths) if all_widths else 42.0
    threshold_px = gap_multiplier * median_w

    voids: List[OOSVoidGap] = []
    for row_num, row_skus in rows.items():
        if len(row_skus) < 2:
            continue
        for i in range(len(row_skus) - 1):
            left = row_skus[i]
            right = row_skus[i + 1]
            gap_px = right.box_xyxy[0] - left.box_xyxy[2]
            if gap_px >= threshold_px:
                est_missing = max(1, int(round(gap_px / median_w)))
                y1 = min(left.box_xyxy[1], right.box_xyxy[1])
                y2 = max(left.box_xyxy[3], right.box_xyxy[3])
                voids.append(
                    OOSVoidGap(
                        shelf_row=row_num,
                        void_box_xyxy=[
                            round(left.box_xyxy[2], 1),
                            round(y1, 1),
                            round(right.box_xyxy[0], 1),
                            round(y2, 1),
                        ],
                        void_width_px=round(gap_px, 1),
                        estimated_missing_facings=est_missing,
                        adjacent_left_sku=left.base_pack_id,
                        adjacent_right_sku=right.base_pack_id,
                    )
                )
    return voids


def _levenshtein_distance(seq_a: List[str], seq_b: List[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    dp = [[0] * (len(seq_b) + 1) for _ in range(len(seq_a) + 1)]
    for i in range(len(seq_a) + 1):
        dp[i][0] = i
    for j in range(len(seq_b) + 1):
        dp[0][j] = j
    for i in range(1, len(seq_a) + 1):
        for j in range(1, len(seq_b) + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(seq_a)][len(seq_b)]


def compute_planogram_sequence_compliance(
    resolved_skus: List[ResolvedSKU], target_planogram_skus: List[str]
) -> float:
    """Compute sequence alignment score [0..100%] between deduplicated row sequence and target planogram order."""
    if not resolved_skus or not target_planogram_skus:
        return 100.0

    rows = cluster_facings_into_shelf_rows(resolved_skus)
    observed_runs: List[str] = []
    for _, row_skus in sorted(rows.items()):
        for s in row_skus:
            if s.base_pack_id in target_planogram_skus:
                if not observed_runs or observed_runs[-1] != s.base_pack_id:
                    observed_runs.append(s.base_pack_id)

    if not observed_runs:
        return 0.0

    target_dedup: List[str] = []
    for t in target_planogram_skus:
        if not target_dedup or target_dedup[-1] != t:
            target_dedup.append(t)

    dist = _levenshtein_distance(observed_runs, target_dedup)
    max_len = max(len(observed_runs), len(target_dedup), 1)
    return round(max(0.0, 100.0 * (1.0 - dist / float(max_len))), 2)


def compute_brand_block_purity(resolved_skus: List[ResolvedSKU]) -> tuple[float, int]:
    """Compute Brand-Block Purity % and count competitor intrusions splitting contiguous Unilever brand blocks."""
    rows = cluster_facings_into_shelf_rows(resolved_skus)
    if not rows:
        return 100.0, 0

    intrusions = 0
    unilever_facings = sum(1 for s in resolved_skus if _is_unilever_sku(s.base_pack_id))
    if unilever_facings <= 1:
        return 100.0, 0

    for _, row_skus in rows.items():
        brands = [_extract_brand(s.base_pack_id) for s in row_skus]
        for i in range(1, len(brands) - 1):
            # If left and right belong to the same Unilever brand, but middle is a different/competitor brand
            if brands[i - 1] == brands[i + 1] and brands[i] != brands[i - 1]:
                if _is_unilever_sku(row_skus[i - 1].base_pack_id):
                    intrusions += 1

    purity = max(0.0, 100.0 * (1.0 - intrusions / float(max(1, unilever_facings))))
    return round(purity, 2), intrusions


def evaluate_full_mt_gondola_audit(
    resolved_skus: List[ResolvedSKU], target_planogram_skus: List[str]
) -> MTGondolaAuditReport:
    """Compute all 8 Modern Trade Gondola KPIs in a single unified audit pass."""
    sos = compute_linear_and_area_sos(resolved_skus)
    voids = detect_shelf_oos_void_gaps(resolved_skus)
    seq_pct = compute_planogram_sequence_compliance(resolved_skus, target_planogram_skus)
    purity_pct, intrusions = compute_brand_block_purity(resolved_skus)
    rows = cluster_facings_into_shelf_rows(resolved_skus)

    return MTGondolaAuditReport(
        facing_count_sos_pct=sos["facing_count_sos_pct"],
        linear_width_sos_pct=sos["linear_width_sos_pct"],
        area_2d_sos_pct=sos["area_2d_sos_pct"],
        oos_void_gaps=voids,
        total_missing_facings_est=sum(v.estimated_missing_facings for v in voids),
        planogram_sequence_compliance_pct=seq_pct,
        brand_block_purity_pct=purity_pct,
        competitor_intrusions_count=intrusions,
        shelf_tiers_detected=len(rows),
    )
