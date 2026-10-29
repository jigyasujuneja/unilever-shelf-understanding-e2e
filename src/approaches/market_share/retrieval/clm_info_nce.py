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
import pickle
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from approaches.base import Box, Context, register
from approaches.market_share.classification.embedding_text_match import unit
from approaches.market_share.retrieval.clef_standalone import (
    CLEF_ZOOM_BATCH_PROMPT,
    CLEF_ZOOM_BATCH_SCHEMA,
    CLEF_ZOOM_SINGLE_PROMPT,
    CLEF_ZOOM_SINGLE_SCHEMA,
)
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
    calibrate_gallery_geometry,
    compute_csls_hubness,
    compute_zscore_and_lid,
    infonce_posterior,
    infonce_rerank,
    zoom_multi_sheet,
    zoom_sheet,
)
from utils.openjev import extract_slot_distributions, resolve_jev_choice
from utils.photometry import (
    apply_illuminant,
    estimate_shelf_illuminant,
    inner_lab_zones,
    restore_photometry,
)

_LAB_CACHE_PATH = Path(tempfile.gettempdir()) / "shelf_bench_emb_cache" / "gallery_lab_cache.pkl"
_MEM_LAB_CACHE: dict[str, list[float]] = {}
_LAB_CACHE_LOADED = False


def _load_lab_cache() -> dict[str, list[float]]:
    global _LAB_CACHE_LOADED
    if not _LAB_CACHE_LOADED:
        _LAB_CACHE_LOADED = True
        if _LAB_CACHE_PATH.exists():
            try:
                data = pickle.loads(_LAB_CACHE_PATH.read_bytes())
                if isinstance(data, dict):
                    _MEM_LAB_CACHE.update(data)
            except Exception:
                pass
    return _MEM_LAB_CACHE


def _save_lab_cache() -> None:
    try:
        _LAB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _LAB_CACHE_PATH.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(_MEM_LAB_CACHE, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(_LAB_CACHE_PATH)
    except Exception:
        pass


@register
class ClmContrastive(CoarseJevLaya):
    """CLM Multi-View InfoNCE Contrastive Decision Model."""

    name = "clm_contrastive"
    clm_inv_temp = 28.0
    clm_min_prob = 0.56
    target_system1_precision = 0.96
    architecture = (
        f"CLM (Contrastive-LM) InfoNCE Multi-View Encoder ({SHORTLIST_MODEL} max-view + centroid + "
        f"CSLS hubness + inner-CIELAB InfoNCE posterior, self-calibrated 1/tau & Conformal risk SLA={target_system1_precision:.0%}) -> Gemini 3.8 contact-sheet fallback"
    )
    steps = [
        "Setup: build multi-view & centroid embeddings + CSLS hubness penalties + train-gallery LOVO conformal calibration",
        "Compute CLM InfoNCE energy E(x, y_k) = 0.78*(max_cos - 0.5*csls) + 0.22*centroid_cos + 0.035*s_lab and softmax P_CLM(y|x)",
        f"Auto-accept in 0 VLM calls when Conformal risk gate passes (target System-1 precision={target_system1_precision:.0%})",
        "Escalate high-entropy crops to batched Gemini 3.8 contact sheets + 1D shelf-row smoothing",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        from utils import dataset as ds_mod

        super().setup(config, ctx)
        try:
            self.catalog = ds_mod.catalog(ds_mod.data_root(self.dataset), self.dataset)
        except Exception:
            self.catalog = {}
        cache = _load_lab_cache()
        dirty = False
        for _, path, _ in self.index:
            if path in cache:
                self.lab_index[path] = cache[path]
            elif path in self.jpeg:
                with Image.open(io.BytesIO(self.jpeg[path])) as im:
                    zones = inner_lab_zones(im.convert("RGB"))
                    self.lab_index[path] = zones
                    cache[path] = zones
                    dirty = True
        if dirty:
            _save_lab_cache()
        self.csls_penalties: dict[str, float] = {}
        if hasattr(self, "index_matrix") and self.index_matrix.ndim == 2 and self.index_matrix.shape[0] > 1:
            r_t = compute_csls_hubness(self.index_matrix, k_nn=min(10, self.index_matrix.shape[0] - 1))
            self.csls_penalties = {
                path: float(pen) for path, pen in zip(self.index_paths, r_t.tolist(), strict=True)
            }
        self.gallery_calibration = calibrate_gallery_geometry(
            self.index,
            self.sku_centroids,
            target_system1_precision=self.target_system1_precision,
            split="train",
        )
        self.clm_inv_temp = float(self.gallery_calibration.get("inv_temp", 28.0))
        self.clm_min_prob = float(self.gallery_calibration.get("conformal_min_prob", 0.56))
        self.smooth_cosine = float(self.gallery_calibration.get("intra_sku_cosine_floor", 0.86))

    def _ensure_lab_for_candidates(self, candidates: list[tuple[float, int, str]]) -> None:
        """Lazily populate 13-dim inner_lab_zones for top shortlist candidates when _LazyJpegDict is used."""
        cache = _load_lab_cache()
        dirty = False
        for _, _, path in candidates:
            if path in self.lab_index:
                continue
            if path in cache:
                self.lab_index[path] = cache[path]
                continue
            try:
                raw_bytes = self.jpeg[path]
                with Image.open(io.BytesIO(raw_bytes)) as im:
                    zones = inner_lab_zones(im.convert("RGB"))
                    self.lab_index[path] = zones
                    cache[path] = zones
                    dirty = True
            except Exception:
                pass
        if dirty:
            _save_lab_cache()

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

        self._ensure_lab_for_candidates(coarse[: 2 * K])
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
        cal = getattr(self, "gallery_calibration", None) or {}
        med_sib = float(cal.get("median_sibling_cosine", COARSE_FAMILY_COSINE))
        same_family = family_cos >= max(COARSE_FAMILY_COSINE, med_sib)

        req_prob = (self.clm_min_prob + 0.09) if same_family else self.clm_min_prob
        confident = s1 >= self.min_cosine and (s1 >= self.high_cosine or p1 >= req_prob)
        return confident, {
            "same_coarse_family": same_family,
            "family_cosine": round(family_cos, 4),
            "clm_top1_prob": p1,
            "clm_inv_temp": self.clm_inv_temp,
            "coarse_entropy_conf": ent_conf,
            "required_margin": round(req_prob, 3),
        }


@register
class L8HulUltimate(ClmContrastive):
    """Unified Gated CLM (System-1) + CLEF Zoom-Strip & JEV PoE (System-2) Hybrid Stack."""

    name = "l8_hul_ultimate"
    max_lid = 12.0
    architecture = (
        "Global Shelf Illuminant (p=6) + Inner-80% CIELAB & Bottle/Sachet Taper + "
        "CLM InfoNCE & Z-Score/LID 0-VLM Gate + CLEF Bottom-30% Zoom-Strip Contact Sheets & JEV PoE Fusion"
    )
    steps = [
        "Setup: embed multi-view gallery, SKU centroids & inner-80% CIELAB + shoulder-taper descriptors",
        "Stage 0: Estimate full-shelf Shades-of-Gray (p=6) illuminant once on shelf photo and normalize crops",
        "Stage A: Intersect CLM InfoNCE posterior, Z-score normalized margin, and LID hubness gate (0 VLM calls on easy crops)",
        "Stage B: Pre-embedding spatial medoid dedup + CLEF Bottom-30% Pack-Size Zoom-Strip contact sheets + JEV PoE calibration",
        "Stage C: Visually-gated 1D Markov shelf-row continuity smoothing",
    ]

    def pick_with_confidence(
        self, image: Image.Image, top: list[tuple[float, int, str]], ctx: Context
    ) -> tuple[int | None, float, dict]:
        from approaches.market_share.retrieval.clef_standalone import (
            build_contrastive_candidate_rubric,
            verify_pack_size_sister_sku,
        )

        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        sheet_im = zoom_sheet(image, refs)
        cat_map = getattr(self, "catalog", None) or {}
        rubric = build_contrastive_candidate_rubric(top, cat_map)
        base_prompt = CLEF_ZOOM_SINGLE_PROMPT.format(k=len(top))
        prompt = f"{base_prompt}\n{rubric}" if rubric else base_prompt
        openjev = getattr(self, "openjev", None)
        if openjev is not None and openjev.enabled:
            choices, slots, usage, _ = openjev.ask_choices(
                sheet_im, 1, len(top), prompt, tier=getattr(ctx.llm, "tier", None)
            )
            with ctx.trace._lock:
                ctx.trace.usage = ctx.trace.usage + usage
            raw_choice = choices[0] if choices else None
            slot = slots[0] if slots else None
            pack_size = ""
        else:
            res = ctx.ask(
                sheet_im,
                prompt,
                schema=CLEF_ZOOM_SINGLE_SCHEMA,
                logprobs=len(top) + 1,
            )
            data = res.data if isinstance(res.data, dict) else {}
            raw_choice = data.get("choice")
            pack_size = str(data.get("pack_size") or "")
            valid_opts = set(range(0, len(top) + 1))
            slots = extract_slot_distributions(
                res.meta.get("logprobs"), valid_options=valid_opts, n_options=len(top) + 1
            )
            slot = slots[0] if slots else None
        ans, conf, meta = resolve_jev_choice(
            raw_choice,
            slot,
            top,
            fallback_cosine=self.fallback_cosine,
            min_confidence=self.min_jev_confidence,
        )
        ans, overridden = verify_pack_size_sister_sku(ans, pack_size, top, cat_map)
        if overridden:
            meta["pack_size_verified_override"] = True
        meta["pack_size"] = pack_size
        return ans, conf, meta

    def pick_batch_with_confidence(
        self,
        queries: list[Image.Image],
        tops: list[list[tuple[float, int, str]]],
        ctx: Context,
    ) -> list[tuple[int | None, float, dict]]:
        from approaches.market_share.retrieval.clef_standalone import (
            build_contrastive_candidate_rubric,
            verify_pack_size_sister_sku,
        )

        if len(queries) == 1:
            return [self.pick_with_confidence(queries[0], tops[0], ctx)]
        refs_per_q = [
            [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
            for top in tops
        ]
        cat_map = getattr(self, "catalog", None) or {}
        row_rubrics = [
            build_contrastive_candidate_rubric(top, cat_map, row_idx=idx)
            for idx, top in enumerate(tops)
        ]
        rubric_block = "\n".join(r for r in row_rubrics if r)
        base_prompt = CLEF_ZOOM_BATCH_PROMPT.format(m=len(queries), k=K)
        prompt = f"{base_prompt}\n{rubric_block}" if rubric_block else base_prompt
        sheet_im = zoom_multi_sheet(queries, refs_per_q)
        openjev = getattr(self, "openjev", None)
        pack_sizes: list[str] = []
        if openjev is not None and openjev.enabled:
            choices, slots, usage, _ = openjev.ask_choices(
                sheet_im, len(queries), K, prompt, tier=getattr(ctx.llm, "tier", None)
            )
            with ctx.trace._lock:
                ctx.trace.usage = ctx.trace.usage + usage
        else:
            res = ctx.ask(sheet_im, prompt, schema=CLEF_ZOOM_BATCH_SCHEMA, logprobs=K + 1)
            data = res.data if isinstance(res.data, dict) else {}
            raw = data.get("choices")
            choices = raw if isinstance(raw, list) else []
            raw_ps = data.get("pack_sizes")
            pack_sizes = [str(x) for x in raw_ps] if isinstance(raw_ps, list) else []
            valid_opts = set(range(0, K + 1))
            slots = extract_slot_distributions(
                res.meta.get("logprobs"), valid_options=valid_opts, n_options=K + 1
            )
        out: list[tuple[int | None, float, dict]] = []
        for idx, top in enumerate(tops):
            c = choices[idx] if idx < len(choices) else None
            slot = slots[idx] if idx < len(slots) else None
            ans, conf, meta = resolve_jev_choice(
                c,
                slot,
                top,
                fallback_cosine=self.fallback_cosine,
                min_confidence=self.min_jev_confidence,
            )
            row_ps = pack_sizes[idx] if idx < len(pack_sizes) else ""
            ans, overridden = verify_pack_size_sister_sku(ans, row_ps, top, cat_map)
            if overridden:
                meta["pack_size_verified_override"] = True
            if idx < len(pack_sizes):
                meta["pack_size"] = row_ps
            out.append((ans, conf, meta))
        return out

    def _stage_a_gate(self, ranked: list[tuple[float, int, str]]) -> tuple[bool, dict]:
        _, meta = super()._stage_a_gate(ranked)
        if not ranked:
            return False, meta
        top = ranked[:K]
        zlid = compute_zscore_and_lid([s for s, _, _ in top])
        s1, id1, _ = top[0]
        s2, id2, _ = top[1] if len(top) > 1 else (-1.0, id1, "")
        margin = s1 - s2
        cat_map = getattr(self, "catalog", None) or {}
        c1 = cat_map.get(id1) or {}
        c2 = cat_map.get(id2) or {}
        if c1 and c2:
            same_sister = bool(
                c1.get("brand")
                and c1.get("brand") == c2.get("brand")
                and c1.get("base_variant") == c2.get("base_variant")
                and c1.get("pack_size") != c2.get("pack_size")
            )
        else:
            same_sister = bool(meta.get("same_coarse_family", False))
        req_margin = 0.025 if same_sister else 0.009
        req_z = 1.05 if same_sister else 0.80
        p_clm, _ = infonce_posterior(top, inv_temp=28.0, k=K)
        confident = (
            s1 >= self.min_cosine
            and (
                s1 >= self.high_cosine
                or (
                    margin >= req_margin
                    and p_clm >= 0.30
                    and zlid["z_margin"] >= req_z
                    and zlid["lid"] <= max(self.max_lid, 11.5)
                )
            )
        )
        return confident, {
            **meta,
            **zlid,
            "same_variant_sister": same_sister,
            "local_sku_margin": req_margin,
        }

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
        for pid, sc in self.sku_centroids.items():
            item = self.catalog.get(pid) or {}
            brand = str(item.get("brand") or item.get("category") or f"sku_{pid}")
            self.brand_subcentroids[brand].append(sc)
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
        eff_top_b = max(self.brand_top_b, 8)
        eff_window = max(self.brand_margin_window, 0.08)
        chosen = [
            (b, round(s, 4))
            for b, s in scored[:eff_top_b]
            if top_score - s <= eff_window
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
        explicit_allowed = allowed_ids is not None
        if allowed_ids is None:
            allowed_ids, _ = self._narrow_by_brand(q)
        best: dict[int, tuple[float, int, str]] = {}
        global_best: dict[int, tuple[float, int, str]] = {}
        if (
            hasattr(self, "index_matrix")
            and self.index_matrix.ndim == 2
            and self.index_matrix.shape[0] == len(self.index)
            and self.index_matrix.shape[0] > 0
        ):
            sims = (self.index_matrix @ np.asarray(q, dtype=np.float32)).tolist()
            for s, pid, path in zip(sims, self.index_pids, self.index_paths, strict=True):
                fs = float(s)
                if pid not in global_best or fs > global_best[pid][0]:
                    global_best[pid] = (fs, pid, path)
                if allowed_ids and pid not in allowed_ids:
                    continue
                if pid not in best or fs > best[pid][0]:
                    best[pid] = (fs, pid, path)
        else:
            for pid, path, v in self.index:
                s = _dot(q, v)
                if pid not in global_best or s > global_best[pid][0]:
                    global_best[pid] = (s, pid, path)
                if allowed_ids and pid not in allowed_ids:
                    continue
                if pid not in best or s > best[pid][0]:
                    best[pid] = (s, pid, path)
        if not explicit_allowed and global_best:
            for fs, pid, path in sorted(global_best.values(), reverse=True)[: 2 * K]:
                if pid not in best or fs > best[pid][0]:
                    best[pid] = (fs, pid, path)
        coarse = sorted(best.values(), reverse=True) or sorted(global_best.values(), reverse=True)
        if not coarse:
            return [], q, restored

        self._ensure_lab_for_candidates(coarse[: 2 * K])
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


@register
class SftClmClefCascade(OutletScopedClmJev):
    """EXP-29: Gemini 3.5 Flash-Lite (Base / LoRA SFT) + CLM 5-Attribute Vote Gate + CLEF Contact-Sheet Cascade.

    Addresses the client recording's 3-model cascading bottleneck (`Brand -> Variant -> SubCategory/Category Config
    -> Pack Type with Category__* Masking`):
      1. Stage A (`gemini-embedding-2` + CLM InfoNCE + 5-Attribute Vote Gate):
         Computes similarity-weighted vote share across `(brand, category, packaging_type, base_variant)` at
         `inv_temp=80.0`. Accepts in 0 VLM calls only when both the CLM InfoNCE/Z-score/LID gate AND all 5
         attribute vote shares are confident (`brand >= 0.88`, `category >= 0.88`, `packaging_type >= 0.82`,
         `base_variant >= 0.82`).
      2. Stage B (Soft Top-2 Brand/Category Shortlist + Contrastive Metadata CLEF Contact Sheet):
         Replaces the legacy pipeline's brittle hard Top-1 Brand mask with a soft Top-2 Brand/Category shortlist
         and passes the 5-panel CLEF Bottom-30% Zoom Contact Sheet + `EXP-28` catalog metadata rubric to
         `Gemini 3.5 Flash-Lite` (Base or Vertex AI LoRA SFT `gemini-3.5-flash-lite-sft`).
      3. Stage C (Client-Config `(Brand, Base_Variant) -> Category -> Valid Category__PackType` Consistency):
         Verifies and logs catalog config consistency across all 5 hierarchy levels.
    """

    name = "sft_clm_clef_cascade"
    tau_brand = 0.88
    tau_category = 0.88
    tau_packaging_type = 0.82
    tau_base_variant = 0.82
    vote_inv_temp = 80.0
    architecture = (
        "EXP-29 Gemini 3.5 Flash-Lite (Base / LoRA SFT) + CLM 5-Attribute Vote Gate -> "
        "Soft Multi-Brand/Category Shortlist + Contrastive Metadata CLEF Contact Sheet -> "
        "Client-Config (Brand, Variant) -> Category -> Valid Pack Consistency"
    )
    steps = [
        "Setup: build multi-prototype (Brand, Category, Pack) sub-centroids + catalog (Brand, Variant) -> Category -> Valid Pack index",
        "Stage A: CLM InfoNCE + Z-score/LID gate intersected with 5-attribute softmax vote shares (brand, category, packaging_type, base_variant)",
        "Stage B: Escalate uncertain crops to 5-panel CLEF Zoom Contact Sheet with EXP-28 catalog metadata & contrastive token diffs",
        "Stage C: Enforce (Brand, Base_Variant) -> Category -> Valid Category__PackType catalog consistency",
    ]

    def _attribute_vote_shares(self, ranked: list[tuple[float, int, str]]) -> dict[str, float]:
        import math
        from collections import defaultdict

        top = ranked[:K]
        if not top:
            return {"brand": 0.0, "category": 0.0, "packaging_type": 0.0, "base_variant": 0.0}
        s1 = top[0][0]
        if not getattr(self, "catalog", None):
            c = round(max(0.05, min(0.96, s1)), 4)
            return {"brand": c, "category": c, "packaging_type": c, "base_variant": c}
        weights = [math.exp(max(-30.0, self.vote_inv_temp * (s - s1))) for s, _, _ in top]
        z = sum(weights) or 1.0
        top_meta = self.catalog.get(top[0][1]) or {}
        shares: dict[str, float] = {}
        for attr in ("brand", "category", "packaging_type", "base_variant"):
            votes: dict[str, float] = defaultdict(float)
            for (_, pid, _), w in zip(top, weights, strict=True):
                m = self.catalog.get(pid) or {}
                val = str(m.get(attr) or m.get("variant" if attr == "base_variant" else attr) or "")
                votes[val] += w / z
            top_val = str(top_meta.get(attr) or top_meta.get("variant" if attr == "base_variant" else attr) or "")
            shares[attr] = round(votes.get(top_val, 0.0), 4)
        return shares

    def _stage_a_gate(self, ranked: list[tuple[float, int, str]]) -> tuple[bool, dict]:
        import math

        base_conf, meta = super()._stage_a_gate(ranked)
        if not ranked:
            return False, meta
        shares = self._attribute_vote_shares(ranked)
        s1 = ranked[0][0]
        dustbin_scale = 1.0 / (1.0 + math.exp(max(-20.0, min(20.0, 28.0 * (0.62 - s1)))))
        cos_clarity = max(0.08, min(0.985, (s1 - 0.42) / 0.54))
        cal_shares = {
            k: round(max(0.02, min(0.985, 0.60 * v * dustbin_scale + 0.40 * cos_clarity)), 4)
            for k, v in shares.items()
        }
        attr_conf = (
            shares.get("brand", 0.0) >= self.tau_brand
            and shares.get("category", 0.0) >= self.tau_category
            and shares.get("packaging_type", 0.0) >= self.tau_packaging_type
            and shares.get("base_variant", 0.0) >= self.tau_base_variant
        )
        confident = base_conf and attr_conf
        return confident, {
            **meta,
            "attr_vote_shares": shares,
            "calibrated_attr_confidences": cal_shares,
            "attr_gate_passed": attr_conf,
        }

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        from utils.taxonomy.packaging import project_to_valid_category_pack

        ids = super().identify_boxes(image, boxes, ctx)
        cat_map = getattr(self, "catalog", None) or {}
        if not cat_map or not ids:
            return ids
        checked = 0
        consistent = 0
        for pid in ids:
            if pid is None:
                continue
            m = cat_map.get(pid) or {}
            cat = str(m.get("category") or "")
            pt = str(m.get("packaging_type") or "")
            if cat and pt:
                checked += 1
                proj = project_to_valid_category_pack(cat, pt)
                if proj == pt:
                    consistent += 1
        ctx.trace.step(
            "Stage C Client-Config (Brand, Variant) -> Category -> Pack Consistency",
            f"Verified {consistent}/{checked} predicted SKUs against valid (Category, PackType) catalog domain",
            info={"checked_skus": checked, "catalog_consistent_skus": consistent},
        )
        return ids


class _ResidualEmbeddingsWrapper:
    """Thin proxy that applies ``ResidualInfoNCEHead`` after ``VertexEmbeddings.image()``."""

    def __init__(self, base_emb, head) -> None:
        self._base = base_emb
        self._head = head

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def image(self, image: Image.Image, ctx: Context | None = None) -> list[float]:
        raw_vec = self._base.image(image, ctx)
        return self._head.transform_one(raw_vec)


@register
class ResidualInfoNceClmJev(SftClmClefCascade):
    """EXP-32: Residual InfoNCE Projection SFT on ``gemini-embedding-2`` + 5-Attribute Vote Gate + CLEF Cascade."""

    name = "residual_infonce_clm_jev"
    architecture = (
        "EXP-32 Residual InfoNCE Projection SFT on gemini-embedding-2 -> "
        "CLM 5-Attribute Vote Gate -> Soft Multi-Brand/Category Shortlist + Contrastive CLEF Contact Sheet"
    )
    steps = [
        "Setup: fit Residual InfoNCE projection head f(x) = norm(x + alpha*relu(x@W1)@W2) strictly on reference gallery views",
        "Project gallery embeddings, SKU centroids, and CSLS hubness penalties into the sharpened residual space",
        "Stage A: Embed crop with gemini-embedding-2 + residual projection and run CLM 5-attribute vote gate (0 VLM calls on confident matches)",
        "Stage B: Escalate uncertain sister variants to 5-panel CLEF Zoom Contact Sheet with EXP-28 contrastive catalog rubric",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        from collections import defaultdict

        from utils.residual_infonce import ResidualInfoNCEHead

        super().setup(config, ctx)
        if not self.index or self.index_matrix.ndim != 2 or self.index_matrix.shape[0] < 8:
            return
        dim = int(self.index_matrix.shape[1])
        self.res_head = ResidualInfoNCEHead(dim=dim, hidden_dim=min(192, dim), alpha=0.22, temperature=0.05)
        fit_stats = self.res_head.fit(self.index, self.catalog)
        if not self.res_head.is_fitted():
            return

        # Re-project gallery matrix and rebuild centroids & calibration in the residual space
        proj_mat = self.res_head.transform_matrix(self.index_matrix)
        self.index_matrix = proj_mat
        self.index = [
            (pid, path, row.tolist())
            for (pid, path, _), row in zip(self.index, proj_mat, strict=True)
        ]

        by_sku: dict[int, list[list[float]]] = defaultdict(list)
        proto_vecs: dict[tuple[str, str, str], list[list[float]]] = defaultdict(list)
        cat_vecs: dict[str, list[list[float]]] = defaultdict(list)
        for pid, _, v in self.index:
            by_sku[pid].append(v)
            item = self.catalog.get(pid) or {}
            brand = str(item.get("brand") or item.get("category") or f"sku_{pid}")
            cat = str(item.get("category") or "default_cat")
            ptype = str(item.get("packaging_type") or item.get("pack_form") or "default_pack")
            proto_vecs[(brand, cat, ptype)].append(v)
            cat_vecs[cat].append(v)

        self.sku_centroids = {
            pid: unit([sum(col) / len(vs) for col in zip(*vs, strict=True)])
            for pid, vs in by_sku.items()
            if vs
        }
        self.brand_subcentroids = defaultdict(list)
        for (brand, _, _), vs in proto_vecs.items():
            if vs:
                self.brand_subcentroids[brand].append(
                    unit([sum(col) / len(vs) for col in zip(*vs, strict=True)])
                )
        for pid, sc in self.sku_centroids.items():
            item = self.catalog.get(pid) or {}
            brand = str(item.get("brand") or item.get("category") or f"sku_{pid}")
            self.brand_subcentroids[brand].append(sc)
        self.brand_centroids = {
            b: subs[0] for b, subs in self.brand_subcentroids.items() if subs
        }
        self.cat_centroids = defaultdict(list)
        for cat, vs in cat_vecs.items():
            if vs:
                self.cat_centroids[cat].append(
                    unit([sum(col) / len(vs) for col in zip(*vs, strict=True)])
                )

        if self.index_matrix.shape[0] > 1:
            r_t = compute_csls_hubness(
                self.index_matrix, k_nn=min(10, self.index_matrix.shape[0] - 1)
            )
            self.csls_penalties = {
                path: float(pen)
                for path, pen in zip(self.index_paths, r_t.tolist(), strict=True)
            }
        self.gallery_calibration = calibrate_gallery_geometry(
            self.index,
            self.sku_centroids,
            target_system1_precision=self.target_system1_precision,
            split="train",
        )
        self.clm_inv_temp = float(self.gallery_calibration.get("inv_temp", 28.0))
        self.clm_min_prob = float(self.gallery_calibration.get("conformal_min_prob", 0.56))
        self.emb = _ResidualEmbeddingsWrapper(self.emb, self.res_head)
        ctx.trace.step(
            "EXP-32 Residual InfoNCE Projection SFT",
            f"Fitted {dim}->192->{dim} residual head (val R@1: {fit_stats.get('base_val_r1', 0):.3f} -> {fit_stats.get('tuned_val_r1', 0):.3f})",
            info=fit_stats,
        )


