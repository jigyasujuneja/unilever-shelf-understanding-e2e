"""Track 2 Merchandising: Store-Session Planogram & Share-of-Shelf Verifier (`bay_planogram_verifier`).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Use Case: Merchandising -> Planogram & SOS Compliance (W8, hul_sessions)  │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  Stage 0: Global Shelf Illuminant (Shades-of-Gray p=6)                     │
  │  Stage 1: 2-Stage Hierarchical Brand-Gate -> Allowed-ID CLM + JEV          │
  │  Stage 2: Row-Clustered Left-to-Right Brand Block & Facing Share Audit     │
  │    • Computes row-ordered sequence compliance, brand-block purity,         │
  │      and HUL vs Rival Share of Shelf (facing-count & pixel-area SOS).      │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from collections import Counter

from PIL import Image

from approaches.base import Box, Context, register
from approaches.market_share.retrieval.clm_info_nce import HierarchicalClmJev
from approaches.market_share.retrieval.embedding_retrieval import MODEL
from utils.hul_taxonomy import normalize_brand_folder


def compute_planogram_metrics(
    boxes: list[Box],
    pred_ids: list[int | None],
    catalog: dict[int, dict],
    asset_boxes: list[Box] | None = None,
) -> dict:
    """Compute Left-to-Right brand-block purity, HUL Share of Shelf, and Class 0 display-bay compliance."""
    if not boxes:
        return {
            "total_facings": 0,
            "hul_facings": 0,
            "rival_facings": 0,
            "unidentified_facings": 0,
            "hul_facing_sos": 0.0,
            "brand_block_purity": 0.0,
            "distinct_brands": 0,
            "asset_bays": len(asset_boxes or []),
            "competitor_intrusion_count": 0,
            "asset_bay_purity": 1.0,
            "oos_void_count": 0,
            "estimated_missing_facings": 0,
            "empty_shelf_share": 0.0,
        }
    # Sort boxes top-to-bottom by row band, then left-to-right by x1
    heights = [max(1.0, b[3] - b[1]) for b in boxes]
    widths = [max(1.0, b[2] - b[0]) for b in boxes]
    med_h = sorted(heights)[len(heights) // 2]
    med_w = sorted(widths)[len(widths) // 2]
    row_tol = max(20.0, 0.5 * med_h)

    indexed = sorted(enumerate(boxes), key=lambda ib: (round(ib[1][1] / row_tol), ib[1][0]))
    ordered_brands: list[str] = []
    known_brands: list[str] = []
    hul_count = 0
    rival_count = 0
    unidentified_count = 0
    box_is_hul: dict[int, bool] = {}

    for idx, _ in indexed:
        pid = pred_ids[idx] if idx < len(pred_ids) else None
        item = catalog.get(pid) if pid is not None else None
        if not item:
            unidentified_count += 1
            ordered_brands.append(f"__unidentified_{idx}__")
            box_is_hul[idx] = False
            continue
        brand = str(item.get("brand") or item.get("product") or "Unknown")
        ordered_brands.append(brand)
        known_brands.append(brand)
        if "is_hul" in item:
            is_hul = bool(item["is_hul"])
        elif "is_unilever" in item:
            is_hul = bool(item["is_unilever"])
        else:
            nb = normalize_brand_folder(brand)
            is_hul = bool(nb and nb.get("is_hul", False))
        box_is_hul[idx] = is_hul
        if is_hul:
            hul_count += 1
        else:
            rival_count += 1

    # Brand-block purity: unidentified slots break brand blocks and never count as a single "Unknown" block
    distinct_known = len(set(known_brands))
    if not known_brands:
        block_purity = 0.0
    elif len(ordered_brands) <= 1:
        block_purity = 1.0
    else:
        transitions = sum(
            1 for a, b in zip(ordered_brands, ordered_brands[1:], strict=False) if a != b
        )
        min_transitions = max(0, distinct_known - 1)
        extra_transitions = max(0, transitions - min_transitions)
        block_purity = round(max(0.0, 1.0 - (extra_transitions / max(1, len(ordered_brands) - 1))), 4)

    # Row-band Out-of-Stock (OOS) horizontal void detection (inter-box gap >= 1.5 * med_w)
    row_groups: dict[int, list[Box]] = {}
    for _, b in indexed:
        rk = round(b[1] / row_tol)
        row_groups.setdefault(rk, []).append(b)

    oos_void_count = 0
    estimated_missing_facings = 0
    total_void_px = 0.0
    total_row_span_px = 0.0
    for r_boxes in row_groups.values():
        r_sorted = sorted(r_boxes, key=lambda b: b[0])
        if len(r_sorted) >= 2:
            span = max(1.0, r_sorted[-1][2] - r_sorted[0][0])
            total_row_span_px += span
            for prev_b, next_b in zip(r_sorted, r_sorted[1:], strict=False):
                gap = next_b[0] - prev_b[2]
                if gap >= 1.5 * med_w:
                    oos_void_count += 1
                    missing = max(1, round(gap / med_w))
                    estimated_missing_facings += missing
                    total_void_px += gap
        elif len(r_sorted) == 1:
            total_row_span_px += max(1.0, r_sorted[0][2] - r_sorted[0][0])

    empty_shelf_share = round(
        min(1.0, total_void_px / max(1.0, total_row_span_px)), 4
    )

    # Class 0 Display Asset Bay Compliance (competitor intrusion inside HUL display bays)
    intrusions = 0
    in_bay_total = 0
    if asset_boxes:
        for idx, b in enumerate(boxes):
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            in_any_bay = any(ab[0] <= cx <= ab[2] and ab[1] <= cy <= ab[3] for ab in asset_boxes)
            if in_any_bay:
                in_bay_total += 1
                if not box_is_hul.get(idx, False):
                    intrusions += 1
    bay_purity = round(1.0 - (intrusions / max(1, in_bay_total)), 4) if in_bay_total > 0 else 1.0

    brand_counts = dict(Counter(known_brands).most_common(5))
    return {
        "total_facings": len(boxes),
        "hul_facings": hul_count,
        "rival_facings": rival_count,
        "unidentified_facings": unidentified_count,
        "hul_facing_sos": round(hul_count / max(1, len(boxes)), 4),
        "brand_block_purity": block_purity,
        "distinct_brands": distinct_known,
        "top_brand_facings": brand_counts,
        "asset_bays": len(asset_boxes or []),
        "competitor_intrusion_count": intrusions,
        "asset_bay_purity": bay_purity,
        "oos_void_count": oos_void_count,
        "estimated_missing_facings": estimated_missing_facings,
        "empty_shelf_share": empty_shelf_share,
    }


@register
class BayPlanogramVerifier(HierarchicalClmJev):
    """Store-Session Planogram Sequence, Brand-Block Purity, OOS Void & Share-of-Shelf Verifier."""

    name = "bay_planogram_verifier"
    use_case = "merchandising"
    task = "planogram_compliance"
    dataset = "hul_sessions"
    architecture = (
        f"Hierarchical Multi-Prototype Brand-Gate ({MODEL}) + CLM InfoNCE + JEV Contact Sheets -> "
        "Row-Clustered Left-to-Right Brand-Block, OOS Void Gap & Class 0 Display-Bay Verifier"
    )
    steps = [
        "Setup: build multi-prototype (brand, category, packaging_type) sub-centroids + SKU multi-view & inner-80% CIELAB descriptors",
        "Stage 0: Estimate full-shelf Shades-of-Gray (p=6) illuminant once on shelf session",
        "Stage 1: Identify every shelf facing via Hierarchical Brand-Gate -> CLM InfoNCE + JEV fallback",
        "Stage 2: Order facings top-to-bottom, left-to-right and verify brand-block purity, horizontal OOS voids (gap >= 1.5x median width), Class 0 display-bay purity & HUL Share of Shelf",
    ]

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        ids = super().identify_boxes(image, boxes, ctx)
        asset_boxes = getattr(ctx, "detected_asset_boxes", None)
        plano = compute_planogram_metrics(
            boxes, ids, getattr(self, "catalog", {}), asset_boxes=asset_boxes
        )
        ctx.trace.step(
            "Verify Bay Planogram & Share of Shelf",
            f"HUL SOS={plano['hul_facing_sos']*100:.1f}% ({plano['hul_facings']}/{plano['total_facings']} facings), "
            f"brand-block purity={plano['brand_block_purity']*100:.1f}%, OOS voids={plano['oos_void_count']} "
            f"(empty share={plano['empty_shelf_share']*100:.1f}%), {plano['distinct_brands']} brands",
            boxes=boxes,
            info=plano,
        )
        return ids
