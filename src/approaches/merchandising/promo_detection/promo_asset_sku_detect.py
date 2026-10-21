"""Track 2 Merchandising: Dual-Class Promotional Asset & SKU Detector (`promo_asset_sku_detect`).

Detects both Class 0 promotional display bays (`promo_asset`) and Class 1 retail product packs
(`product_sku`) in Gemini (designed for `gemini-3.5-flash-lite` on `gtmerch_golden` and the 40
Miniature / Promo Asset shelves).

Unlike `single_pass_dedup`, which drops outer container boxes globally, this approach preserves
Class 0 `promo_asset` boxes on `ctx.detected_asset_boxes` for downstream Stage 6 bay-scoped
planogram compliance while deduplicating Class 1 `product_sku` boxes separately.

Also provides configurable Wave 2 upgrades (EXP-39 through EXP-42):
- ``PromoAssetBayFirstDetect`` (EXP-39): Two-stage Bay-First high-res crop detection.
- ``PromoAssetStage0ClaheDetect`` (EXP-40): Stage 0 Dual-Buffer CIELAB L* CLAHE photometry.
- ``PromoAssetStage15ClutterDetect`` (EXP-41): Stage 1.5 3D-pack vs 2D-cardboard/price-tag filter.
- ``PromoAssetAwpRouterDetect`` (EXP-42): Dedicated Class 0 AWP fixture archetype & hanger router.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from approaches.base import Approach, Box, Context, register, to_pixels
from approaches.market_share.detection.single_pass_dedup import (
    NMS_IOU,
    dedup_traced,
)
from utils import metrics
from utils.photometry import dual_buffer_shelf_photometry

PROMO_DET_CLASSES = ("promo_asset", "product_sku", "not_a_product")
FIXTURE_ARCHETYPES = ("WIN", "CTU", "MB2", "MB3", "HAN", "FULLWALL", "NONE")

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "box_2d": {"type": "array", "items": {"type": "integer"}},
            "label": {"type": "string", "enum": list(PROMO_DET_CLASSES)},
        },
        "required": ["box_2d", "label"],
    },
}

AWP_ROUTER_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "box_2d": {"type": "array", "items": {"type": "integer"}},
            "label": {"type": "string", "enum": list(PROMO_DET_CLASSES)},
            "fixture_type": {"type": "string", "enum": list(FIXTURE_ARCHETYPES)},
        },
        "required": ["box_2d", "label"],
    },
}

SKU_ONLY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "box_2d": {"type": "array", "items": {"type": "integer"}},
            "label": {"type": "string", "enum": ["product_sku", "not_a_product"]},
        },
        "required": ["box_2d", "label"],
    },
}

PROMPT = (
    "Detect every promotional display unit and every individual retail product visible in this "
    "store image. Use two distinct labels:\n"
    "1. 'promo_asset': branded promotional display bays, coolers, countertop display units, "
    "endcaps, or shelf display frames (Class 0 container assets).\n"
    "2. 'product_sku': each individual physical retail product package, bottle, box, jar, tube, "
    "pouch, or sachet (Class 1 SKUs). Give each physical item its own tight box and do not merge "
    "adjacent identical items.\n"
    "3. 'not_a_product': price tags, shelf strips, banners, or background clutter.\n"
    "List any 'promo_asset' container boxes first, followed by every 'product_sku' item. "
    'Return ONLY a JSON array like [{"box_2d": [ymin, xmin, ymax, xmax], "label": "product_sku"}] '
    "with coordinates normalized to 0-1000."
)

AWP_ROUTER_PROMPT = (
    "Detect every Unilever promotional display fixture (Class 0 'promo_asset') and every individual "
    "physical retail product package (Class 1 'product_sku') in this store photo.\n"
    "Unilever Class 0 fixture archetypes:\n"
    "- 'WIN': Single-window shelf display bay with top header board, printed side wings, and 3-6 shelf planks.\n"
    "- 'CTU': Countertop tabletop display unit (1-2 tiers sitting on a shop counter).\n"
    "- 'MB2' / 'MB3': Multi-bay 2-window or 3-window wall display frame.\n"
    "- 'FULLWALL': Full-height branded wall bay.\n"
    "- 'HAN': Hanging sachet strip or parasite hanger clip. Do NOT wrap a frameless hanging sachet strip "
    "as a 'promo_asset' bay; only label rigid framed display units (WIN, CTU, MB2, MB3, FULLWALL) as 'promo_asset'.\n"
    "For each 'promo_asset', set 'fixture_type' to one of WIN, CTU, MB2, MB3, FULLWALL (or HAN/NONE). "
    "For each physical bottle, carton, jar, tube, pouch, or sachet, set 'label' to 'product_sku' with its own "
    "tight box (do not merge touching items). Label flat price tags, shelf strips, and 2D printed cardboard "
    "graphics as 'not_a_product'.\n"
    "List all 'promo_asset' boxes first, then all 'product_sku' boxes. Return ONLY a JSON array with "
    "coordinates normalized to 0-1000."
)

BAY_CROP_PROMPT = (
    "This image is a close-up crop of a promotional display bay. Detect every individual physical "
    "retail product package ('product_sku') sitting on the shelves inside this display bay, including "
    "small sachets, dark back-row bottles, and top-row items sitting partly behind the top header sign. "
    "Give each physical item its own tight box and do not merge adjacent identical products. "
    "Label flat 2D printed graphics on the side wings, header board, and shelf-rail price tags as "
    "'not_a_product'. Return ONLY a JSON array like "
    '[{"box_2d": [ymin, xmin, ymax, xmax], "label": "product_sku"}] normalized to 0-1000.'
)

DETECTION_MAX_OUTPUT_TOKENS = 8192


def parse_promo_and_sku_boxes(
    raw: list | dict | None,
    w: float,
    h: float,
) -> tuple[list[Box], list[Box]]:
    """Split Gemini's [{box_2d, label}] response (or truncated [y1,x1,y2,x2] quads) into (promo_asset_boxes, product_sku_boxes)."""
    if isinstance(raw, dict):
        raw = next((v for v in raw.values() if isinstance(v, list)), [])
    asset_boxes: list[Box] = []
    sku_boxes: list[Box] = []
    for item in raw or []:
        if not isinstance(item, (dict, list, tuple)):
            continue
        px = to_pixels([item], 0, 0, w, h)
        if not px:
            continue
        lab = str(item.get("label") or "product_sku") if isinstance(item, dict) else "product_sku"
        if lab == "promo_asset":
            asset_boxes.extend(px)
        elif lab == "product_sku":
            sku_boxes.extend(px)
    return asset_boxes, sku_boxes


def parse_awp_router_boxes(
    raw: list | dict | None,
    w: float,
    h: float,
) -> tuple[list[Box], list[str], list[Box]]:
    """Parse AWP fixture router output into (asset_boxes, fixture_types, sku_boxes).

    Frameless hanging strips ('HAN') are automatically excluded from Class 0 asset_boxes so
    downstream scoring uses the full-image hanger fallback rule cleanly.
    """
    if isinstance(raw, dict):
        raw = next((v for v in raw.values() if isinstance(v, list)), [])
    asset_boxes: list[Box] = []
    fixture_types: list[str] = []
    sku_boxes: list[Box] = []
    for item in raw or []:
        if not isinstance(item, (dict, list, tuple)):
            continue
        px = to_pixels([item], 0, 0, w, h)
        if not px:
            continue
        lab = str(item.get("label") or "product_sku") if isinstance(item, dict) else "product_sku"
        ftype = str(item.get("fixture_type") or "WIN").upper() if isinstance(item, dict) else "WIN"
        if lab == "promo_asset":
            if ftype == "HAN":
                continue
            bx = px[0]
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            # Suppress narrow frameless hanger strips misclassified as Class 0 bays
            if bh > 0 and (bw / bh) < 0.24 and bw < 0.22 * w:
                continue
            asset_boxes.extend(px)
            fixture_types.append(ftype if ftype in FIXTURE_ARCHETYPES else "WIN")
        elif lab == "product_sku":
            sku_boxes.extend(px)
    return asset_boxes, fixture_types, sku_boxes


def filter_stage15_clutter(
    sku_boxes: list[Box],
    asset_boxes: list[Box],
    image: Image.Image,
) -> tuple[list[Box], list[dict]]:
    """Stage 1.5 open-set-safe filter: reject 2D structural clutter while keeping all 3D packs.

    Never uses closed-set SKU catalog cosine similarity (which would delete regional 'Others'
    competitors). Filters only:
      - Flat shelf-rail price tags (wide strip aspect ratio + small vertical height).
      - Vertical side-wing pillar banners along the outer 4% border of a Class 0 bay.
      - Wide top-header sign banners inside the top 14% band of a Class 0 bay.
      - Flat 2D printed graphics on outer side wings with near-zero horizontal 3D shading relief.
    """
    if not sku_boxes:
        return [], []
    w_img, h_img = float(image.width), float(image.height)
    img_area = max(1.0, w_img * h_img)
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    kept: list[Box] = []
    removed: list[dict] = []
    for b in sku_boxes:
        x1, y1, x2, y2 = b
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        aspect = bw / bh
        rel_area = (bw * bh) / img_area

        # 1. Tiny speck or flat shelf-rail price tag strip
        if rel_area < 0.00020 or (aspect >= 2.85 and bh < 0.042 * h_img):
            removed.append({"box": [round(v, 1) for v in b], "reason": "rail_price_tag"})
            continue

        reason: str | None = None
        for ab in asset_boxes:
            ax1, ay1, ax2, ay2 = ab
            aw = max(1.0, ax2 - ax1)
            ah = max(1.0, ay2 - ay1)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            if not (ax1 - 0.02 * aw <= cx <= ax2 + 0.02 * aw and ay1 - 0.02 * ah <= cy <= ay2 + 0.02 * ah):
                continue
            # 2. Top header sign board graphic inside top 14% of Class 0 bay
            if y2 <= ay1 + 0.16 * ah and aspect >= 2.3 and bw >= 0.42 * aw:
                reason = "header_board_graphic"
                break
            # 3. Tall thin side-wing structural pillar strip
            on_wing_edge = (cx <= ax1 + 0.045 * aw) or (cx >= ax2 - 0.045 * aw)
            if on_wing_edge and bh >= 0.55 * ah and aspect <= 0.17:
                reason = "wing_pillar_banner"
                break
            # 4. Flat 2D printed pack graphic on outer display wing (low horizontal gradient relief)
            if on_wing_edge and bw <= 0.10 * aw:
                ix1 = max(0, min(gray.shape[1] - 2, int(round(x1))))
                iy1 = max(0, min(gray.shape[0] - 2, int(round(y1))))
                ix2 = max(ix1 + 2, min(gray.shape[1], int(round(x2))))
                iy2 = max(iy1 + 2, min(gray.shape[0], int(round(y2))))
                patch = gray[iy1:iy2, ix1:ix2]
                if patch.size >= 36:
                    gx_mean = float(np.abs(np.diff(patch, axis=1)).mean())
                    if gx_mean < 4.2:
                        reason = "2d_wing_cardboard_print"
                        break
        if reason is not None:
            removed.append({"box": [round(v, 1) for v in b], "reason": reason})
        else:
            kept.append(b)
    return kept, removed


def _merge_bay_crop_skus(
    full_skus: list[Box],
    crop_skus: list[Box],
    bay_box: Box,
    iou_dup_thr: float = 0.45,
) -> tuple[list[Box], int]:
    """Merge high-res bay-crop SKU boxes with full-shelf SKU boxes without double-counting."""
    if not crop_skus:
        return full_skus, 0
    merged = list(full_skus)
    added = 0
    for cb in crop_skus:
        if not metrics._box_inside_any_asset_bay(cb, [bay_box], min_containment=0.45):
            continue
        if all(metrics.iou(cb, eb) < iou_dup_thr for eb in merged):
            merged.append(cb)
            added += 1
    return merged, added


def filter_plank_support_and_shoulders(
    sku_boxes: list[Box],
    asset_boxes: list[Box],
    fixture_types: list[str] | None = None,
) -> tuple[list[Box], list[dict]]:
    """EXP-53 Stage 1 Physical Plank-Support Law & Depth-Row Shoulder Suppression.

    Inside framed Class 0 display bays (WIN, CTU, MB2, MB3, FULLWALL):
      1. Clusters in-bay boxes by bottom base coordinate ``y_max`` (within +-6.5% of bay height)
         to discover physical horizontal shelf planks (planks with >= 2 supported items).
      2. Rejects isolated floating 2D graphics in the top 28% header zone or outer 7% wing border
         that do not rest on any multi-item horizontal shelf plank (``|y_max - y_plank| > 0.08 * h_bay``).
      3. Rejects back-row depth shoulders sticking up behind taller front-row packs on the same plank
         (horizontal overlap >= 0.62, ``h < 0.68 * h_front``, and ``y_max`` above front pack base).
    """
    if not sku_boxes or not asset_boxes:
        return list(sku_boxes), []

    dropped_indices: set[int] = set()
    removed: list[dict] = []

    for b_idx, ab in enumerate(asset_boxes):
        ftype = (fixture_types[b_idx] if fixture_types and b_idx < len(fixture_types) else "WIN").upper()
        if ftype == "HAN":
            continue
        ax1, ay1, ax2, ay2 = ab
        aw = max(1.0, ax2 - ax1)
        ah = max(1.0, ay2 - ay1)
        in_bay_idxs = [
            i
            for i, sb in enumerate(sku_boxes)
            if i not in dropped_indices
            and (ax1 - 0.02 * aw <= 0.5 * (sb[0] + sb[2]) <= ax2 + 0.02 * aw)
            and (ay1 - 0.02 * ah <= 0.5 * (sb[1] + sb[3]) <= ay2 + 0.02 * ah)
        ]
        if len(in_bay_idxs) < 4:
            continue

        # Cluster in-bay boxes into horizontal shelf planks by y_max (bottom resting line)
        sorted_idxs = sorted(in_bay_idxs, key=lambda idx: sku_boxes[idx][3])
        planks: list[list[int]] = []
        for idx in sorted_idxs:
            ym = sku_boxes[idx][3]
            placed = False
            for pl in planks:
                ref_ym = float(np.median([sku_boxes[k][3] for k in pl]))
                if abs(ym - ref_ym) <= 0.065 * ah:
                    pl.append(idx)
                    placed = True
                    break
            if not placed:
                planks.append([idx])

        multi_planks = [pl for pl in planks if len(pl) >= 2]
        if not multi_planks:
            continue
        plank_ys = [float(np.median([sku_boxes[k][3] for k in pl])) for pl in multi_planks]

        # 1. Floating header/wing graphic check (isolated 1-item cluster not resting on any multi-item plank)
        for idx in in_bay_idxs:
            if idx in dropped_indices:
                continue
            x1, y1, x2, y2 = sku_boxes[idx]
            cx = 0.5 * (x1 + x2)
            min_plank_dist = min(abs(y2 - py) for py in plank_ys)
            in_header_or_wing = (
                (y2 <= ay1 + 0.28 * ah)
                or (cx <= ax1 + 0.075 * aw)
                or (cx >= ax2 - 0.075 * aw)
            )
            if min_plank_dist > 0.085 * ah and in_header_or_wing:
                dropped_indices.add(idx)
                removed.append(
                    {
                        "box": [round(v, 1) for v in sku_boxes[idx]],
                        "reason": "floating_unplanked_bay_graphic",
                        "min_plank_dist_frac": round(min_plank_dist / ah, 3),
                    }
                )

        # 2. Back-row depth shoulder ghost suppression against front-row plank items
        multi_plank_members = {k for pl in multi_planks for k in pl if len(pl) >= 2}
        active_in_bay = [i for i in in_bay_idxs if i not in dropped_indices]
        for idx in active_in_bay:
            if idx in dropped_indices:
                continue
            x1, y1, x2, y2 = sku_boxes[idx]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            for front_idx in active_in_bay:
                if front_idx == idx or front_idx in dropped_indices:
                    continue
                fx1, fy1, fx2, fy2 = sku_boxes[front_idx]
                fw = max(1.0, fx2 - fx1)
                fh = max(1.0, fy2 - fy1)
                inter_x = max(0.0, min(x2, fx2) - max(x1, fx1))
                overlap_x = inter_x / max(1.0, min(bw, fw))
                # True back-row shoulder: horizontally behind front pack, shorter, and bottom edge y2
                # terminates inside the top shoulder band [fy1 - 0.04*fh, fy1 + 0.42*fh] of the front pack
                # (never an independent upper shelf tier where y2 <= fy1 or a multi-item plank).
                if (
                    overlap_x >= 0.65
                    and bh < 0.62 * fh
                    and y1 < fy1 - 0.02 * fh
                    and (fy1 - 0.04 * fh) <= y2 <= (fy1 + 0.42 * fh)
                    and (idx not in multi_plank_members or bh < 0.42 * fh)
                ):
                    dropped_indices.add(idx)
                    removed.append(
                        {
                            "box": [round(v, 1) for v in sku_boxes[idx]],
                            "reason": "depth_row_shoulder_ghost",
                        }
                    )
                    break

    kept = [b for i, b in enumerate(sku_boxes) if i not in dropped_indices]
    return kept, removed


@register
class PromoAssetSkuDetect(Approach):
    """Dual-class Gemini detector for Class 0 promotional display bays and Class 1 product SKUs."""

    name = "promo_asset_sku_detect"
    use_case = "merchandising"
    task = "detection"
    dataset = "gtmerch_golden"
    stage0_clahe: bool = False
    bay_first_crop: bool = False
    stage15_clutter_gate: bool = False
    awp_template_conditioning: bool = False
    plank_support_filter: bool = False
    density_quadrant_tiling: bool = False

    architecture = (
        "Gemini single-pass dual-class detector (Class 0 promo_asset bays + Class 1 product_sku "
        "packs, 8192 output tokens) -> separate bay NMS & SKU geometric dedup"
    )
    steps = [
        "Downscale image (longest side 2048 px) and prompt Gemini for Class 0 promo_asset bays first + Class 1 product_sku boxes (max_output_tokens=8192)",
        f"Deduplicate Class 0 promo_asset display bays via greedy NMS at IoU {NMS_IOU} and store on ctx.detected_asset_boxes",
        "Clean Class 1 product_sku boxes via container filter, NMS, and depth-ghost suppression",
    ]

    def _run_density_quadrant_tiling(
        self, det_image: Image.Image, raw_skus: list[Box], ctx: Context
    ) -> list[Box]:
        """EXP-53: Run 2x2 overlapping quadrant tiling when pass 1 indicates a dense shelf wall."""
        from concurrent.futures import ThreadPoolExecutor

        w, h = float(det_image.width), float(det_image.height)
        if w < 320 or h < 320:
            return raw_skus
        img_area = max(1.0, w * h)
        areas = [max(1.0, (b[2] - b[0]) * (b[3] - b[1])) / img_area for b in raw_skus]
        med_area = float(np.median(areas)) if areas else 1.0
        # Trigger on dense full-shelf walls (>=45 boxes, or >=24 small boxes with median area < 0.45% of frame)
        if not (len(raw_skus) >= 45 or (len(raw_skus) >= 24 and med_area < 0.0045)):
            return raw_skus

        quads = [
            (0.0, 0.0, 0.56 * w, 0.56 * h),
            (0.44 * w, 0.0, w, 0.56 * h),
            (0.0, 0.44 * h, 0.56 * w, h),
            (0.44 * w, 0.44 * h, w, h),
        ]

        def _detect_quad(qbox: tuple[float, float, float, float]) -> list[Box]:
            qx1, qy1, qx2, qy2 = qbox
            qw, qh = qx2 - qx1, qy2 - qy1
            q_im = det_image.crop((int(round(qx1)), int(round(qy1)), int(round(qx2)), int(round(qy2))))
            q_res = ctx.ask(
                q_im,
                BAY_CROP_PROMPT,
                schema=SKU_ONLY_SCHEMA,
                max_side=1536,
                max_output_tokens=DETECTION_MAX_OUTPUT_TOKENS,
            )
            _, local_boxes = parse_promo_and_sku_boxes(q_res.data, qw, qh)
            global_boxes: list[Box] = []
            for bx1, by1, bx2, by2 in local_boxes:
                # Skip partial edge fragments clipped right on an interior quadrant seam
                if qx1 > 0 and bx1 <= 0.015 * qw:
                    continue
                if qx2 < w and bx2 >= 0.985 * qw:
                    continue
                if qy1 > 0 and by1 <= 0.015 * qh:
                    continue
                if qy2 < h and by2 >= 0.985 * qh:
                    continue
                global_boxes.append((bx1 + qx1, by1 + qy1, bx2 + qx1, by2 + qy1))
            return global_boxes

        with ThreadPoolExecutor(max_workers=4) as pool:
            quad_lists = list(pool.map(_detect_quad, quads))

        merged = list(raw_skus)
        added = 0
        for q_boxes in quad_lists:
            for qb in q_boxes:
                if all(metrics.iou(qb, eb) < 0.45 for eb in merged):
                    merged.append(qb)
                    added += 1
        ctx.trace.step(
            "EXP-53 Density-Triggered 2x2 Quadrant Tiling",
            f"Triggered on dense shelf (initial={len(raw_skus)}, med_area={med_area:.4f}) -> +{added} small/wall SKU boxes",
            boxes=merged,
            regions=list(quads),
            info={"initial_boxes": len(raw_skus), "median_area_frac": round(med_area, 5), "added_boxes": added},
        )
        return merged

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        w, h = image.size
        det_image = image
        if self.stage0_clahe:
            det_image, color_image, photo_stats = dual_buffer_shelf_photometry(image)
            ctx.detector_image = det_image
            ctx.color_image = color_image
            ctx.trace.step(
                "Stage 0 Dual-Buffer Photometry",
                f"Shades-of-Gray gains={photo_stats['gains']}, CIELAB L* CLAHE "
                f"std {photo_stats['l_std_before']:.1f} -> {photo_stats['l_std_after']:.1f}",
                info=photo_stats,
            )

        prompt = AWP_ROUTER_PROMPT if self.awp_template_conditioning else PROMPT
        schema = AWP_ROUTER_SCHEMA if self.awp_template_conditioning else SCHEMA
        res = ctx.ask(
            det_image,
            prompt,
            schema=schema,
            max_side=2048,
            max_output_tokens=DETECTION_MAX_OUTPUT_TOKENS,
        )

        fixture_types: list[str] = []
        if self.awp_template_conditioning:
            raw_assets, fixture_types, raw_skus = parse_awp_router_boxes(res.data, w, h)
        else:
            raw_assets, raw_skus = parse_promo_and_sku_boxes(res.data, w, h)

        if raw_assets:
            kept_asset_idxs = metrics.nms(raw_assets, thr=NMS_IOU)
            ctx.detected_asset_boxes = [raw_assets[i] for i in kept_asset_idxs]
            kept_ftypes = [fixture_types[i] for i in kept_asset_idxs if i < len(fixture_types)]
        else:
            ctx.detected_asset_boxes = []
            kept_ftypes = []

        if self.awp_template_conditioning:
            scene_mode = kept_ftypes[0] if kept_ftypes else "NONE_HANGER_FALLBACK"
            ctx.detected_fixture_type = scene_mode
        else:
            scene_mode = "class0_bay" if ctx.detected_asset_boxes else "full_image_hanger_fallback"

        ctx.trace.step(
            "Gemini dual-class promo asset + SKU detection",
            f"1 call, {res.seconds:.1f}s -> {len(ctx.detected_asset_boxes)} Class 0 promo bays "
            f"({len(raw_assets)} raw, mode={scene_mode}), {len(raw_skus)} Class 1 SKU boxes",
            boxes=raw_skus,
            info={
                "promo_asset_count": len(ctx.detected_asset_boxes),
                "raw_promo_asset_count": len(raw_assets),
                "raw_sku_count": len(raw_skus),
                "scene_mode": scene_mode,
                "fixture_types": kept_ftypes,
                "detected_asset_boxes": [
                    [round(v, 1) for v in b] for b in ctx.detected_asset_boxes
                ],
            },
        )

        if self.bay_first_crop and ctx.detected_asset_boxes:
            total_added = 0
            crop_calls = 0
            for ab in ctx.detected_asset_boxes[:3]:
                ax1, ay1, ax2, ay2 = ab
                pad_x = 0.03 * (ax2 - ax1)
                pad_y = 0.03 * (ay2 - ay1)
                cx1 = max(0.0, ax1 - pad_x)
                cy1 = max(0.0, ay1 - pad_y)
                cx2 = min(float(w), ax2 + pad_x)
                cy2 = min(float(h), ay2 + pad_y)
                if (cx2 - cx1) < 64 or (cy2 - cy1) < 64:
                    continue
                bay_im = det_image.crop((int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))))
                crop_res = ctx.ask(
                    bay_im,
                    BAY_CROP_PROMPT,
                    schema=SKU_ONLY_SCHEMA,
                    max_side=2048,
                    max_output_tokens=DETECTION_MAX_OUTPUT_TOKENS,
                )
                crop_calls += 1
                _, crop_sku_local = parse_promo_and_sku_boxes(crop_res.data, cx2 - cx1, cy2 - cy1)
                crop_sku_global = [
                    (bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1)
                    for bx1, by1, bx2, by2 in crop_sku_local
                ]
                raw_skus, added = _merge_bay_crop_skus(raw_skus, crop_sku_global, ab)
                total_added += added
            if crop_calls > 0:
                ctx.trace.step(
                    "Bay-First high-res crop detection",
                    f"{crop_calls} bay crop call(s) -> +{total_added} small/back-row SKU boxes inside Class 0 bays",
                    boxes=raw_skus,
                    info={"bay_crop_calls": crop_calls, "added_sku_boxes": total_added},
                )

        if self.density_quadrant_tiling:
            raw_skus = self._run_density_quadrant_tiling(det_image, raw_skus, ctx)

        if self.stage15_clutter_gate and raw_skus:
            filtered_skus, removed_clutter = filter_stage15_clutter(
                raw_skus, ctx.detected_asset_boxes, image
            )
            if removed_clutter:
                ctx.trace.step(
                    "Stage 1.5 3D-pack vs 2D-clutter gate",
                    f"{len(removed_clutter)} 2D tag/wing graphic box(es) removed -> {len(filtered_skus)} 3D packs",
                    boxes=filtered_skus,
                    info={"removed_clutter": removed_clutter},
                )
            raw_skus = filtered_skus

        if self.plank_support_filter and raw_skus and ctx.detected_asset_boxes:
            plank_skus, removed_plank = filter_plank_support_and_shoulders(
                raw_skus, ctx.detected_asset_boxes, kept_ftypes
            )
            if removed_plank:
                ctx.trace.step(
                    "EXP-53 Physical Plank-Support & Shoulder Filter",
                    f"{len(removed_plank)} unplanked/shoulder box(es) removed -> {len(plank_skus)} plank-supported packs",
                    boxes=plank_skus,
                    info={"removed_plank_clutter": removed_plank},
                )
            raw_skus = plank_skus

        return dedup_traced(raw_skus, ctx)


@register
class PromoAssetBayFirstDetect(PromoAssetSkuDetect):
    """EXP-39: Bay-First High-Res Crop Detection + Bottom-Plank Containment Rule."""

    name = "promo_asset_bay_first_detect"
    bay_first_crop = True
    architecture = (
        "EXP-39: Gemini Class 0 bay detection -> native-res bay crop zoom pass for small/back-row "
        "Class 1 SKUs -> separate bay NMS & SKU geometric dedup"
    )
    steps = [
        "Stage 1a: Full-shelf pass detects Class 0 promo_asset bays and initial Class 1 SKUs",
        "Stage 1b: Crop each detected Class 0 bay at native camera resolution and detect small/stacked Class 1 SKUs",
        "Merge bay-crop SKUs with full-shelf SKUs (IoU < 0.45) and apply bottom-plank containment rule",
    ]


@register
class PromoAssetStage0ClaheDetect(PromoAssetSkuDetect):
    """EXP-40: Stage 0 Dual-Buffer Photometry (Shades-of-Gray + CIELAB L* CLAHE) for Stage 1 Detection."""

    name = "promo_asset_stage0_clahe_detect"
    stage0_clahe = True
    architecture = (
        "EXP-40: Stage 0 Dual-Buffer Photometry (Shades-of-Gray + CIELAB L* CLAHE on detector_image) "
        "-> Gemini dual-class Class 0 + Class 1 detector"
    )
    steps = [
        "Stage 0: Build dual photometric buffers (CIELAB L* CLAHE detector_image + linear Shades-of-Gray color_image)",
        "Stage 1: Run Gemini dual-class promo_asset + product_sku detection on detector_image",
        "Preserve linear color_image on ctx.color_image for Stage 2 chromatic fidelity",
    ]


@register
class PromoAssetStage15ClutterDetect(PromoAssetSkuDetect):
    """EXP-41: Stage 1.5 3D-Pack vs 2D-Cardboard/Price-Tag Clutter Gate."""

    name = "promo_asset_stage15_clutter_detect"
    stage0_clahe = True
    bay_first_crop = True
    stage15_clutter_gate = True
    architecture = (
        "EXP-41: Stage 0 CLAHE + Bay-First crop pass -> Stage 1.5 3D-pack vs 2D-cardboard/price-tag "
        "clutter filter (open-set competitor safe)"
    )
    steps = [
        "Stage 0 + 1: Run CLAHE-enhanced Bay-First Class 0 + Class 1 detection",
        "Stage 1.5: Filter 2D printed display-wing graphics, header banners, and shelf-rail price tags while keeping open-set Others packs",
        "Stage 1 post-filter: Container, NMS, and depth-ghost suppression",
    ]


@register
class PromoAssetAwpRouterDetect(PromoAssetSkuDetect):
    """EXP-42: Dedicated Class 0 AWP Fixture Archetype & Scene Router (WIN/CTU/MB2/MB3/HAN/FULLWALL)."""

    name = "promo_asset_awp_router_detect"
    stage0_clahe = True
    bay_first_crop = True
    stage15_clutter_gate = True
    awp_template_conditioning = True
    architecture = (
        "EXP-42: Stage 0 CLAHE -> AWP Fixture Archetype Router (WIN/CTU/MB2/MB3/FULLWALL vs HAN hanger fallback) "
        "-> Bay-First crop pass + Stage 1.5 clutter filter"
    )
    steps = [
        "Stage 0: Build dual photometric buffers (detector_image + color_image)",
        "Stage 1a: AWP-conditioned Gemini pass labels Class 0 fixture archetype (WIN/CTU/MB2/MB3/FULLWALL) and rejects frameless HAN hanger strips as bays",
        "Stage 1b + 1.5: High-res bay crop pass on confirmed Class 0 bays + 2D clutter filter",
    ]


@register
class PromoAssetPlankTilingDetect(PromoAssetAwpRouterDetect):
    """EXP-53: Physical Shelf-Plank Support Law + Depth-Row Shoulder Suppression + Density-Triggered 2x2 Quadrant Tiling."""

    name = "promo_asset_plank_tiling_detect"
    stage0_clahe = True
    bay_first_crop = True
    stage15_clutter_gate = True
    awp_template_conditioning = True
    plank_support_filter = True
    density_quadrant_tiling = True
    architecture = (
        "EXP-53: Stage 0 CLAHE -> AWP Fixture Router + Bay-First Zoom + Density-Triggered 2x2 Quadrant Tiling "
        "-> Physical Shelf-Plank Support Law & Depth-Row Shoulder Filter"
    )
    steps = [
        "Stage 0: Build dual photometric buffers (CIELAB L* CLAHE detector_image + linear Shades-of-Gray color_image)",
        "Stage 1a: AWP-conditioned pass detects Class 0 bays (WIN/CTU/MB2/MB3/FULLWALL) and initial Class 1 SKUs",
        "Stage 1b: High-res bay crop pass on Class 0 bays OR density-triggered 2x2 overlapping quadrant tiling on dense full-shelf walls",
        "Stage 1.5 (EXP-53 Plank Law): Enforce multi-item horizontal shelf-plank support (y_max alignment) and suppress back-row depth shoulders",
    ]

