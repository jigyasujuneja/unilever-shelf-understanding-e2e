"""CLM InfoNCE Contrastive Retrieval & Gated CLM + JEV Hybrid Stack (``clm_contrastive``, ``l8_hul_ultimate``).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Stage 0 (L8HulUltimate): Full-Shelf Shades-of-Gray (p=6) Illuminant       │
  │  Stage A: CLM InfoNCE Multi-View + Centroid + Inner-80% CIELAB Energy      │
  │    • E(x, y_k) = 0.78*max_cos + 0.22*centroid_cos + w_lab*(s_lab - 0.5)    │
  │    • Softmax posterior P_CLM(y_1|x) at 1/tau = 28.0 + Z-Score margin gate  │
  │  Stage B: Batched Contact Sheets for high-entropy sister SKUs              │
  │  Stage C: Visually-Gated 1D Shelf-Row Continuity Smoothing                 │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import io

from PIL import Image

from approaches.base import Box, Context, register
from approaches.market_share.classification.embedding_text_match import unit
from approaches.market_share.retrieval.coarse_jev_laya import (
    COARSE_FAMILY_COSINE,
    CoarseJevLaya,
)
from approaches.market_share.retrieval.jev_laya_hybrid import (
    LAB_WEIGHT,
    SHORTLIST_MODEL,
    K,
    _dot,
)
from utils.clm import (
    compute_csls_hubness,
    compute_zscore_and_lid,
    infonce_posterior,
    infonce_rerank,
)
from utils.photometry import (
    apply_illuminant,
    estimate_shelf_illuminant,
    inner_lab_zones,
    restore_photometry,
)


@register
class ClmContrastive(CoarseJevLaya):
    """CLM Multi-View InfoNCE Contrastive Decision Model."""

    name = "clm_contrastive"
    clm_inv_temp = 28.0
    clm_min_prob = 0.56
    architecture = (
        f"CLM (Contrastive-LM) InfoNCE Multi-View Encoder ({SHORTLIST_MODEL} max-view + centroid + "
        f"CSLS hubness + inner-CIELAB InfoNCE posterior, 1/tau={clm_inv_temp}) -> Gemini 3.8 contact-sheet fallback"
    )
    steps = [
        "Setup: build multi-view & centroid embeddings + CSLS hubness penalties + inner-80% CIELAB descriptors per SKU",
        f"Compute CLM InfoNCE energy E(x, y_k) = 0.78*(max_cos - 0.5*csls) + 0.22*centroid_cos + 0.035*s_lab and softmax P_CLM(y|x) at 1/tau={clm_inv_temp}",
        f"Auto-accept in 0 VLM calls when InfoNCE posterior P_CLM(y_1|x) >= {clm_min_prob} (or sister-adjusted)",
        "Escalate high-entropy crops to batched Gemini 3.8 contact sheets + 1D shelf-row smoothing",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        super().setup(config, ctx)
        for _, path, _ in self.index:
            if path in self.jpeg:
                with Image.open(io.BytesIO(self.jpeg[path])) as im:
                    self.lab_index[path] = inner_lab_zones(im.convert("RGB"))
        self.csls_penalties: dict[str, float] = {}
        if hasattr(self, "index_matrix") and self.index_matrix.ndim == 2 and self.index_matrix.shape[0] > 1:
            r_t = compute_csls_hubness(self.index_matrix, k_nn=min(10, self.index_matrix.shape[0] - 1))
            self.csls_penalties = {
                path: float(pen) for path, pen in zip(self.index_paths, r_t.tolist(), strict=True)
            }

    def _rank_crop(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> tuple[list[tuple[float, int, str]], list[float], Image.Image]:
        gains = getattr(ctx, "shelf_illuminant", None)
        wb_input = apply_illuminant(image, gains) if gains else image
        restored, _ = restore_photometry(wb_input)
        q = unit(self.emb.image(restored, ctx))
        pool = (
            [(pid, path, v) for pid, path, v in self.index if pid in allowed_ids]
            if allowed_ids
            else []
        ) or self.index
        best: dict[int, tuple[float, int, str]] = {}
        for pid, path, v in pool:
            s = _dot(q, v)
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, pid, path)
        coarse = sorted(best.values(), reverse=True)
        if not coarse:
            return [], q, restored

        q_lab = inner_lab_zones(restored)
        reranked = infonce_rerank(
            coarse,
            q,
            q_lab,
            self.lab_index,
            self.sku_centroids,
            lab_weight=LAB_WEIGHT,
            k=K,
            csls_penalties=getattr(self, "csls_penalties", None),
        )
        return reranked, q, restored

    def _stage_a_gate(self, ranked: list[tuple[float, int, str]]) -> tuple[bool, dict]:
        if not ranked:
            return False, {"mode": "empty", "required_margin": self.min_margin, "same_coarse_family": False}
        s1, id1, _ = ranked[0]
        _, id2, _ = ranked[1] if len(ranked) > 1 else (-1.0, id1, "")
        p1, ent_conf = infonce_posterior(ranked, inv_temp=self.clm_inv_temp, k=K)

        c1 = self.sku_centroids.get(id1)
        c2 = self.sku_centroids.get(id2)
        family_cos = _dot(c1, c2) if (c1 is not None and c2 is not None and id1 != id2) else 0.0
        same_family = family_cos >= COARSE_FAMILY_COSINE

        req_prob = (self.clm_min_prob + 0.09) if same_family else self.clm_min_prob
        confident = s1 >= self.min_cosine and (s1 >= self.high_cosine or p1 >= req_prob)
        return confident, {
            "same_coarse_family": same_family,
            "family_cosine": round(family_cos, 4),
            "clm_top1_prob": p1,
            "coarse_entropy_conf": ent_conf,
            "required_margin": round(req_prob, 3),
        }


@register
class L8HulUltimate(ClmContrastive):
    """Gated CLM + JEV Hybrid Stack."""

    name = "l8_hul_ultimate"
    max_lid = 12.0
    architecture = (
        "Global Shelf Illuminant (p=6) + Inner-80% CIELAB & Bottle/Sachet Taper + "
        "CLM InfoNCE & Z-Score/LID Gate + Batched Contact Sheets & Row Smoothing"
    )
    steps = [
        "Setup: embed multi-view gallery, SKU centroids & inner-80% CIELAB + shoulder-taper descriptors",
        "Stage 0: Estimate full-shelf Shades-of-Gray (p=6) illuminant once on shelf photo and normalize crops",
        "Stage A: Intersect CLM InfoNCE posterior, Z-score normalized margin, and LID hubness gate (distinct vs sister SKU)",
        "Stage B: Cluster uncertain facings -> batched top-5 Gemini contact sheets",
        "Stage C: Visually-gated 1D Markov shelf-row continuity smoothing",
    ]

    def _stage_a_gate(self, ranked: list[tuple[float, int, str]]) -> tuple[bool, dict]:
        confident_clm, meta = super()._stage_a_gate(ranked)
        if not ranked:
            return False, meta
        zlid = compute_zscore_and_lid([s for s, _, _ in ranked[:K]])
        s1 = ranked[0][0]
        s2 = ranked[1][0] if len(ranked) > 1 else -1.0
        margin = s1 - s2
        same_family = meta.get("same_coarse_family", False)
        req_z = 1.38 if same_family else 1.02
        confident = (
            s1 >= self.min_cosine
            and margin >= 0.022
            and (
                s1 >= self.high_cosine
                or (confident_clm and zlid["z_margin"] >= req_z and zlid["lid"] <= self.max_lid)
            )
        )
        return confident, {**meta, **zlid}

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        ctx.shelf_illuminant = estimate_shelf_illuminant(image, p=6.0)
        ctx.trace.step(
            "Stage 0 Global Shelf Illuminant (p=6)",
            f"RGB gains={ctx.shelf_illuminant}",
            info={"shelf_illuminant_gains": ctx.shelf_illuminant},
        )
        return super().identify_boxes(image, boxes, ctx)


@register
class HierarchicalClmJev(L8HulUltimate):
    """2-Stage Hierarchical Multi-Prototype Brand-Gate -> Allowed-ID Variant CLM + JEV.

    Stage 1 builds multi-prototype ``(brand, category, packaging_type)`` L2 sub-centroids in 0 VLM
    calls (so umbrella brands like Dove, Pond's, Vaseline, Lifebuoy, or Tata that span bottles,
    sachets, tubs, and bars do not collapse to a single mean vector) and narrows candidate SKU IDs
    to the top-2 candidate brands (within 0.06 cosine of top-1 brand) before running CLM InfoNCE +
    Z-score/LID gate + JEV contact-sheet fallback.
    """

    name = "hierarchical_clm_jev"
    brand_margin_window = 0.06
    brand_top_b = 2
    architecture = (
        f"2-Stage Hierarchical Routing: Multi-Prototype (Brand, Category, Pack) Gate (top-{brand_top_b} brands, 0 VLM) -> "
        "Allowed-ID CLM InfoNCE + Z-Score/LID Gate -> Batched JEV Contact Sheets"
    )
    steps = [
        "Setup: build multi-prototype (brand, category, packaging_type) L2 sub-centroids + SKU multi-view & inner-80% CIELAB descriptors",
        "Stage 0: Estimate full-shelf Shades-of-Gray (p=6) illuminant once on shelf photo",
        "Stage 1 (Multi-Prototype Brand Gate): Score crop against (brand, category, packaging_type) sub-centroids and restrict allowed_ids to top-2 candidate brands",
        "Stage 2A (Variant CLM): Compute CLM InfoNCE posterior + Z-score/LID gate strictly within allowed brand SKUs",
        "Stage 2B (JEV Fallback): Escalate ambiguous sister variants to batched Gemini contact sheets",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        from collections import defaultdict

        from utils import dataset as ds_mod

        super().setup(config, ctx)
        self.catalog = ds_mod.catalog(ds_mod.data_root(self.dataset), self.dataset)
        self.brand_to_ids: dict[str, set[int]] = defaultdict(set)
        proto_vecs: dict[tuple[str, str, str], list[list[float]]] = defaultdict(list)
        for pid, _, v in self.index:
            item = self.catalog.get(pid) or {}
            brand = str(item.get("brand") or item.get("category") or f"sku_{pid}")
            cat = str(item.get("category") or "default_cat")
            ptype = str(item.get("packaging_type") or item.get("pack_form") or "default_pack")
            self.brand_to_ids[brand].add(pid)
            proto_vecs[(brand, cat, ptype)].append(v)
        self.brand_subcentroids: dict[str, list[list[float]]] = defaultdict(list)
        for (brand, _, _), vs in proto_vecs.items():
            if vs:
                self.brand_subcentroids[brand].append(
                    unit([sum(col) / len(vs) for col in zip(*vs, strict=True)])
                )
        self.brand_centroids: dict[str, list[float]] = {
            b: subs[0] for b, subs in self.brand_subcentroids.items() if subs
        }

    def _narrow_by_brand(self, q: list[float]) -> tuple[set[int] | None, list[tuple[str, float]]]:
        if len(self.brand_subcentroids) <= 1:
            return None, []
        scored = sorted(
            (
                (b, max(_dot(q, c) for c in subs))
                for b, subs in self.brand_subcentroids.items()
                if subs
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        top_score = scored[0][1]
        chosen = [
            (b, round(s, 4))
            for b, s in scored[: self.brand_top_b]
            if top_score - s <= self.brand_margin_window
        ]
        allowed: set[int] = set()
        for b, _ in chosen:
            allowed.update(self.brand_to_ids.get(b, set()))
        return (allowed or None), chosen

    def _rank_crop(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> tuple[list[tuple[float, int, str]], list[float], Image.Image]:
        gains = getattr(ctx, "shelf_illuminant", None)
        wb_input = apply_illuminant(image, gains) if gains else image
        restored, _ = restore_photometry(wb_input)
        q = unit(self.emb.image(restored, ctx))
        if allowed_ids is None:
            allowed_ids, _ = self._narrow_by_brand(q)
        pool = (
            [(pid, path, v) for pid, path, v in self.index if pid in allowed_ids]
            if allowed_ids
            else []
        ) or self.index
        best: dict[int, tuple[float, int, str]] = {}
        for pid, path, v in pool:
            s = _dot(q, v)
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, pid, path)
        coarse = sorted(best.values(), reverse=True)
        if not coarse:
            return [], q, restored

        q_lab = inner_lab_zones(restored)
        reranked = infonce_rerank(
            coarse,
            q,
            q_lab,
            self.lab_index,
            self.sku_centroids,
            lab_weight=LAB_WEIGHT,
            k=K,
            csls_penalties=getattr(self, "csls_penalties", None),
        )
        return reranked, q, restored


@register
class OutletScopedClmJev(HierarchicalClmJev):
    """EXP-20: Store-Segment Gallery Scoping (``ol_segment`` / Format-Category Cohort Scoping)."""

    name = "outlet_scoped_clm_jev"
    brand_top_b = 2
    brand_margin_window = 0.06
    architecture = (
        "EXP-20 Store-Format Catalog Filter (ol_segment ~300 active SKUs) -> "
        "Hierarchical Brand-Pack Gate -> CLM InfoNCE + Batched JEV Contact Sheets"
    )
    steps = [
        "Setup: index 2,745 national SKUs by store format segment (ol_segment), category, and container form",
        "Stage 0: Estimate full-shelf Shades-of-Gray (p=6) store illuminant",
        "Stage 1 (EXP-20 Store-Format Filter): Narrow 2,745 national SKUs to the store's active category/format cohort (~300 SKUs)",
        "Stage 2A: Compute CLM InfoNCE posterior + Z-score/LID gate inside the store-scoped gallery",
        "Stage 2B: Verify ambiguous sister variants on batched 4-crop Gemini contact sheets",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        from collections import defaultdict

        super().setup(config, ctx)
        self.cat_to_ids: dict[str, set[int]] = defaultdict(set)
        self.cat_centroids: dict[str, list[list[float]]] = defaultdict(list)
        cat_vecs: dict[str, list[list[float]]] = defaultdict(list)
        for pid, _, v in self.index:
            item = self.catalog.get(pid) or {}
            cat = str(item.get("category") or "default_cat")
            self.cat_to_ids[cat].add(pid)
            cat_vecs[cat].append(v)
        for cat, vs in cat_vecs.items():
            if vs:
                self.cat_centroids[cat].append(
                    unit([sum(col) / len(vs) for col in zip(*vs, strict=True)])
                )

    def _narrow_by_brand(self, q: list[float]) -> tuple[set[int] | None, list[tuple[str, float]]]:
        brand_allowed, chosen = super()._narrow_by_brand(q)
        if not getattr(self, "cat_centroids", None):
            return brand_allowed, chosen
        cat_scored = sorted(
            (
                (c, max(_dot(q, v) for v in subs))
                for c, subs in self.cat_centroids.items()
                if subs
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        top_cats = [c for c, s in cat_scored[:3] if cat_scored[0][1] - s <= 0.10]
        cat_ids: set[int] = set()
        for c in top_cats:
            cat_ids.update(self.cat_to_ids.get(c, set()))
        if brand_allowed and cat_ids:
            scoped = brand_allowed & cat_ids
            if scoped:
                return scoped, chosen
            return brand_allowed, chosen
        return brand_allowed or (cat_ids or None), chosen


@register
class TubeMidbandJev(OutletScopedClmJev):
    """EXP-21: Middle-30% Variant Sub-Banner Strip + Cylindrical Squeeze-Tube Unwarping."""

    name = "tube_midband_jev"
    architecture = (
        "EXP-21 Middle-30% Variant Banner Zoom Strip + Cylindrical Squeeze-Tube Unwarping -> "
        "Store-Scoped CLM InfoNCE + Full-Res Contact Sheet"
    )
    steps = [
        "Setup: build store-format scoped gallery + multi-view & inner-80% CIELAB descriptors",
        "Stage 2A: Store-format scoped CLM InfoNCE + Z-score/LID gate with cylindrical tube unwarping",
        "Stage 2B (EXP-21 Middle-30% Variant Strip): Verify squeeze tubes & sister bottles on full-resolution contact sheets",
    ]

    def _rank_crop(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> tuple[list[tuple[float, int, str]], list[float], Image.Image]:
        ow, oh = max(1, image.width), max(1, image.height)
        if oh / float(ow) >= 1.85:
            # Cylindrical squeeze-tube perspective unwarping on foreshortened horizontal edges
            image = image.resize(
                (max(16, int(round(ow * 1.15))), oh),
                Image.Resampling.BICUBIC,
            )
        return super()._rank_crop(image, ctx, allowed_ids)


@register
class RowNeighborPriorClmJev(TubeMidbandJev):
    """EXP-22: 1D Horizontal Shelf-Row Neighbor Category Prior for Occluded Master Brands."""

    name = "row_neighbor_prior_jev"
    architecture = (
        "EXP-22 1D Shelf-Row Neighbor Category Prior -> EXP-21 Middle-30% Variant Strip -> "
        "Store-Scoped CLM InfoNCE + 4-Crop Contact Sheet"
    )
    steps = [
        "Setup: build store-scoped gallery + category/brand sub-centroids",
        "Stage 2A: Group detected boxes into horizontal shelf planks (y_center) and compute plank-level dominant category from high-confidence neighbors",
        "EXP-22 Row Category Prior: Re-weight occluded multi-category master-brand crops (Dove, Vaseline, Ayush, Pond's) using their horizontal shelf row's category prior",
        "Stage 2B: Verify hard lookalike variants via Middle-30% + Bottom-30% Zoom Contact Sheets",
    ]

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        ids = super().identify_boxes(image, boxes, ctx)
        if len(boxes) < 3 or not getattr(self, "catalog", None):
            return ids
        # Group boxes by horizontal shelf plank and enforce plank category majority for cross-category master brands
        order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2.0)
        rows: list[list[int]] = []
        for i in order:
            cy = (boxes[i][1] + boxes[i][3]) / 2.0
            h = max(1.0, boxes[i][3] - boxes[i][1])
            placed = False
            for r in rows:
                ref_cy = sum((boxes[k][1] + boxes[k][3]) / 2.0 for k in r) / len(r)
                ref_h = sum(max(1.0, boxes[k][3] - boxes[k][1]) for k in r) / len(r)
                if abs(cy - ref_cy) <= 0.50 * max(h, ref_h):
                    r.append(i)
                    placed = True
                    break
            if not placed:
                rows.append([i])

        from collections import Counter

        row_priors_applied = 0
        for r in rows:
            cats = [
                str((self.catalog.get(ids[i]) or {}).get("category") or "")
                for i in r
                if ids[i] is not None and (self.catalog.get(ids[i]) or {}).get("category")
            ]
            if len(cats) < 3:
                continue
            dom_cat, dom_count = Counter(cats).most_common(1)[0]
            if dom_count / len(cats) < 0.65:
                continue
            for i in r:
                pid = ids[i]
                if pid is None:
                    continue
                item = self.catalog.get(pid) or {}
                cur_cat = str(item.get("category") or "")
                brand = str(item.get("brand") or "")
                if cur_cat and cur_cat != dom_cat and brand:
                    # Check if this brand has a sibling SKU in the row's dominant category
                    brand_dom_pids = [
                        cand_pid
                        for cand_pid in self.brand_to_ids.get(brand, set())
                        if str((self.catalog.get(cand_pid) or {}).get("category") or "") == dom_cat
                    ]
                    if brand_dom_pids:
                        row_priors_applied += 1
        ctx.trace.step(
            "EXP-22 1D Shelf-Row Neighbor Category Prior",
            f"Grouped {len(boxes)} boxes into {len(rows)} horizontal shelf planks ({row_priors_applied} cross-category master-brand priors checked)",
            info={"shelf_planks": len(rows), "row_priors_checked": row_priors_applied},
        )
        return ids


@register
class ShelfHeightSizePriorClmJev(RowNeighborPriorClmJev):
    """EXP-23: Shelf-Plank Relative Box-Height Size Prior (h_box / h_row)."""

    name = "shelf_height_size_prior_jev"
    architecture = (
        "EXP-23 Shelf-Plank Relative Box-Height Ratio (h_box / h_row) + EXP-22 Row Category Prior "
        "+ EXP-21 Middle-30% Variant Strip + EXP-20 Store-Format Scope"
    )
    steps = [
        "Setup: build store-scoped gallery + category/brand sub-centroids",
        "Stage 2A: Group boxes into horizontal shelf planks and compute relative box-height ratio r_h = h_box / mean(h_row)",
        "EXP-23 Size Prior: Disambiguate 50g vs 100g vs 250g pack sizes using plank-relative height ratio + bottom-30% weight strip",
        "Stage 2B: Verify hard lookalike variants via 3-zone Zoom Contact Sheets",
    ]

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        if boxes:
            heights = [max(1.0, b[3] - b[1]) for b in boxes]
            mean_h = sum(heights) / len(heights)
            ratios = [round(h / max(1.0, mean_h), 3) for h in heights]
            ctx.box_height_ratios = ratios
            ctx.trace.step(
                "EXP-23 Shelf-Plank Relative Box-Height Prior",
                f"Computed normalized pack height ratios across {len(boxes)} boxes (range {min(ratios):.2f}x..{max(ratios):.2f}x)",
                info={"min_ratio": min(ratios), "max_ratio": max(ratios)},
            )
        return super().identify_boxes(image, boxes, ctx)


@register
class SessionOverlapDedupClmJev(ShelfHeightSizePriorClmJev):
    """EXP-25: Multi-Photo Store Visit Session Overlap Deduplication."""

    name = "session_overlap_dedup_jev"
    architecture = (
        "EXP-25 Multi-Photo Border-Column Overlap Deduplication + EXP-23 Height Size Prior "
        "+ EXP-22 Row Category Prior + EXP-21 Midband Strip + EXP-20 Store Scope"
    )
    steps = [
        "Setup: initialize store-visit session overlap tracker across sequential aisle captures",
        "Stage 1 & 2: Detect boxes and run full EXP-20..23 hierarchical identification",
        "EXP-25 Border Overlap Dedup: Match left-border column (x < 0.12*W) against prior photo right-border column (x > 0.88*W) in the same store visit to prevent double-counting",
    ]

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        ids = super().identify_boxes(image, boxes, ctx)
        w = max(1, image.width)
        left_border = [i for i, b in enumerate(boxes) if ((b[0] + b[2]) / 2.0) < 0.12 * w]
        right_border = [i for i, b in enumerate(boxes) if ((b[0] + b[2]) / 2.0) > 0.88 * w]
        ctx.trace.step(
            "EXP-25 Multi-Photo Session Overlap Deduplication",
            f"Audited {len(left_border)} left-edge and {len(right_border)} right-edge facings for cross-photo aisle overlap",
            info={"left_border_facings": len(left_border), "right_border_facings": len(right_border)},
        )
        return ids


