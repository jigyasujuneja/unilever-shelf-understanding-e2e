"""Standalone End-to-End CLM and CLEF Pipelines (Zero JEV Coupling).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  1. detect_clm_standalone : GeminiBoxDetector -> ClmStandaloneRetrieval    │
  │     (Stage 1 Gemini Boxes + Stage 2 Pure 0-VLM CLM InfoNCE)                │
  │  2. shelf_clm_standalone  : DistilledYoloOnnx -> ClmStandaloneRetrieval    │
  │     (100% Zero-VLM E2E: Stage 1 ONNX YOLO + Stage 2 Pure CLM InfoNCE)      │
  │  3. detect_clef_standalone: GeminiBoxDetector -> ClefStandaloneRetrieval   │
  │     (Stage 1 Gemini Boxes + Stage 2 100% CLEF Zoom-Strip Decision Head)    │
  │  4. shelf_clef_standalone : DistilledYoloOnnx -> ClefStandaloneRetrieval   │
  │     (Stage 1 ONNX YOLO + Stage 2 100% CLEF Zoom-Strip Decision Head)       │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Box, Context, compose, register
from approaches.market_share.detection.onnx_yolo_detector import (
    DistilledYolo960SoftNms,
    DistilledYoloOnnx,
)
from approaches.market_share.detection.single_pass_dedup import SinglePassDedup
from approaches.market_share.end_to_end.detect_identify import GeminiBoxDetector
from approaches.market_share.retrieval.clef_standalone import (
    ClefContrastiveMetadata,
    ClefPackCrownFusion,
    ClefStandaloneRetrieval,
)
from approaches.merchandising.promo_detection.promo_asset_sku_detect import (
    PromoAssetPlankTilingDetect,
)
from approaches.market_share.retrieval.clm_info_nce import (
    HierarchicalClmJev,
    L8HulUltimate,
    OutletScopedClmJev,
    RowNeighborPriorClmJev,
    SessionOverlapDedupClmJev,
    ShelfHeightSizePriorClmJev,
    TubeMidbandJev,
)
from approaches.market_share.retrieval.clm_standalone import ClmStandaloneRetrieval
from approaches.market_share.retrieval.embedding_retrieval import MODEL
from utils.embeddings import MODELS
from utils.photometry import dual_buffer_shelf_photometry

ONNX_MODEL_ID = (DistilledYoloOnnx.models or ["distilled-yolo11n-onnx"])[0]

DetectClmStandalone = compose(
    "detect_clm_standalone",
    GeminiBoxDetector,
    ClmStandaloneRetrieval,
    dataset="rpc",
    architecture=(
        f"Gemini detects boxes -> Standalone CLM InfoNCE ({MODEL} multi-view + SKU centroid + "
        "inner-80% CIELAB energy, 0 Stage-2 VLM calls)"
    ),
    steps=[
        "Setup: embed reference gallery views, SKU centroids & inner-80% CIELAB descriptors",
        "Stage 1: One Gemini call on the photo returns every product box",
        "Stage 2: Standalone CLM InfoNCE scores every box in 0 VLM calls (zero JEV coupling)",
    ],
)

ShelfClmStandalone = compose(
    "shelf_clm_standalone",
    DistilledYoloOnnx,
    ClmStandaloneRetrieval,
    dataset="rpc",
    models=MODELS,
    also_calls=[ONNX_MODEL_ID],
    architecture=(
        f"100% Zero-VLM E2E: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> "
        "Stage 2 Standalone CLM InfoNCE (0 Gemini calls total)"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery views, SKU centroids & inner-80% CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: Standalone CLM InfoNCE identifies all boxes in 0 VLM calls (zero JEV coupling)",
    ],
)

DetectClefStandalone = compose(
    "detect_clef_standalone",
    GeminiBoxDetector,
    ClefStandaloneRetrieval,
    dataset="rpc",
    architecture=(
        f"Gemini detects boxes -> Standalone CLEF ({MODEL} shortlist -> 100% CLEF Zoom-Strip "
        "Constrained Logprob Decision Head, zero JEV coupling)"
    ),
    steps=[
        "Setup: embed reference gallery photos",
        "Stage 1: One Gemini call on the photo returns every product box",
        "Stage 2: 100% of detected crops evaluated on CLEF Bottom-30% Zoom-Strip contact sheets",
    ],
)

ShelfClefStandalone = compose(
    "shelf_clef_standalone",
    DistilledYoloOnnx,
    ClefStandaloneRetrieval,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> Stage 2 Standalone CLEF "
        "(100% CLEF Zoom-Strip Constrained Logprob Decision Head, zero JEV coupling)"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery photos",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: 100% of detected crops evaluated on CLEF Bottom-30% Zoom-Strip contact sheets",
    ],
)

ShelfHierarchicalClmJev = compose(
    "shelf_hierarchical_clm_jev",
    DistilledYoloOnnx,
    HierarchicalClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> Stage 2 Hierarchical Brand-Gate (0 VLM) -> "
        "Allowed-ID CLM InfoNCE + Z-Score Gate -> Batched JEV Contact Sheets"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + build brand centroids, SKU centroids & inner-80% CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: Hierarchical Brand-Gate restricts allowed_ids -> CLM InfoNCE + JEV contact-sheet fallback",
    ],
)

ShelfOutletScopedClmJev = compose(
    "shelf_outlet_scoped_clm_jev",
    DistilledYoloOnnx,
    OutletScopedClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-20: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> Store-Segment & Scene Category "
        "Gallery Scoping -> CLM InfoNCE + Batched JEV Contact Sheets"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + build category centroids, SKU centroids & inner-80% CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: Store-Segment & dominant scene category prior prunes gallery -> CLM InfoNCE + JEV fallback",
    ],
)

ShelfTubeMidbandJev = compose(
    "shelf_tube_midband_jev",
    DistilledYoloOnnx,
    TubeMidbandJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-21: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> CLM InfoNCE -> Middle-30% "
        "Cylindrical Squeeze-Tube Unwarped Variant Strip + Bottom-30% CLEF Weight Strip JEV"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery views, SKU centroids & CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: CLM InfoNCE + Middle-30% Unwarped Variant Strip + Bottom-30% CLEF Weight Strip JEV",
    ],
)

ShelfRowNeighborPrior = compose(
    "shelf_row_neighbor_prior",
    DistilledYoloOnnx,
    RowNeighborPriorClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-22: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> CLM InfoNCE + JEV -> "
        "1D Horizontal Shelf-Row Neighbor Brand/Category Consistency Prior"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery views, SKU centroids & CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: CLM InfoNCE + JEV + 1D Horizontal Shelf-Row Neighbor Category Prior",
    ],
)

ShelfHeightSizePrior = compose(
    "shelf_height_size_prior",
    DistilledYoloOnnx,
    ShelfHeightSizePriorClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-23: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> CLM InfoNCE + JEV -> "
        "Shelf-Plank Relative Box-Height Physical Size Prior (h_box / h_row)"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery views, SKU centroids & CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: CLM InfoNCE + JEV + Shelf-Plank Relative Box-Height Physical Size Prior",
    ],
)

Shelf960SoftNmsE2E = compose(
    "shelf_960_softnms_e2e",
    DistilledYolo960SoftNms,
    L8HulUltimate,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-24: Stage 1 960px High-Res + Gaussian Soft-NMS ONNX Detector ({ONNX_MODEL_ID}) -> "
        "Stage 2 Gray-World CLM InfoNCE + Batched JEV Contact Sheets"
    ),
    steps=[
        "Setup: load 960px Soft-NMS ONNX detector + embed reference gallery views, SKU centroids & CIELAB",
        "Stage 1: 960px High-Res + Aspect-Aware Gaussian Soft-NMS ONNX detection ($0 token cost)",
        "Stage 2: Gray-World CLM InfoNCE + Batched JEV Contact Sheets",
    ],
)

SessionOverlapDedupE2E = compose(
    "session_overlap_dedup_e2e",
    DistilledYoloOnnx,
    SessionOverlapDedupClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-25: Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> Stage 2 Multi-Photo Store-Visit "
        "Session Overlap Homography Deduplication + CLM InfoNCE + JEV"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery views, SKU centroids & CIELAB",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: Multi-Photo Store-Visit Session Overlap Deduplication + CLM InfoNCE + JEV",
    ],
)


@register
class DistilledYoloStage0Clahe(DistilledYoloOnnx):
    """EXP-49a Stage 0 + Stage 1: Dual-Buffer CLAHE Photometry -> Distilled ONNX YOLO."""

    name = "distilled_yolo_stage0_clahe"
    architecture = (
        f"EXP-49a: Stage 0 Dual-Buffer Photometry (CIELAB L* CLAHE) -> "
        f"Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID})"
    )
    steps = [
        "Setup: load Vertex AI Model Registry ONNX graphs into CPUExecutionProvider",
        "Stage 0: Build dual photometric buffers (CIELAB L* CLAHE detector_image + linear Shades-of-Gray color_image)",
        "Stage 1: Run dual-head ONNX CPU detection on detector_image while preserving linear color_image for Stage 2",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        det_im, color_im, stats = dual_buffer_shelf_photometry(image)
        ctx.detector_image = det_im
        ctx.color_image = color_im
        ctx.trace.step(
            "Stage 0 Dual-Buffer Photometry",
            f"Shades-of-Gray gains={stats['gains']}, CIELAB L* CLAHE std {stats['l_std_before']:.1f} -> {stats['l_std_after']:.1f}",
            info=stats,
        )
        return super().detect(det_im, ctx)


ShelfStage0ClaheE2E = compose(
    "shelf_stage0_clahe_e2e",
    DistilledYoloStage0Clahe,
    HierarchicalClmJev,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-49a (Stage 0 Swap): Stage 0 Dual-Buffer CLAHE -> Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) "
        "-> Stage 2 Hierarchical Brand-Gate CLM InfoNCE + Batched JEV Contact Sheets (linear color_image)"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + build brand centroids, SKU centroids & inner-80% CIELAB",
        "Stage 0: Build CIELAB L* CLAHE detector_image and linear Shades-of-Gray color_image",
        "Stage 1: Single-pass ONNX YOLO detects shelf boxes on detector_image (~40ms, 0 VLM calls)",
        "Stage 2: Hierarchical Brand-Gate + CLM InfoNCE + JEV contact-sheet fallback on linear color_image crops",
    ],
)

ShelfGeminiDetectE2E = compose(
    "shelf_gemini_detect_e2e",
    SinglePassDedup,
    HierarchicalClmJev,
    dataset="rpc",
    also_calls=[MODEL],
    architecture=(
        "EXP-49b (Stage 1 Swap): Stage 1 Gemini single_pass_dedup -> Stage 2 Hierarchical "
        "Brand-Gate CLM InfoNCE + Batched JEV Contact Sheets"
    ),
    steps=[
        "Setup: build brand centroids, SKU centroids & inner-80% CIELAB descriptors",
        "Stage 1: Gemini single_pass_dedup detects all shelf product boxes + container/NMS/ghost filters",
        "Stage 2: Hierarchical Brand-Gate restricts allowed_ids -> CLM InfoNCE + JEV contact-sheet fallback",
    ],
)

ShelfClefMetadataE2E = compose(
    "shelf_clef_metadata_e2e",
    DistilledYoloOnnx,
    ClefContrastiveMetadata,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-49c (Stage 2 Swap): Stage 1 Distilled ONNX YOLO ({ONNX_MODEL_ID}) -> "
        "Stage 2 Contrastive Metadata CLEF Zoom-Strip Contact Sheets + 2-Tier Open-Set Guard"
    ),
    steps=[
        "Setup: load ONNX YOLO detector + embed reference gallery photos and load SKU catalog metadata",
        "Stage 1: Single-pass ONNX YOLO detects all shelf boxes in ~40ms (0 VLM calls)",
        "Stage 2: CLM InfoNCE fast gate -> Contrastive Metadata CLEF Bottom-30% Zoom-Strip Contact Sheets",
    ],
)

ShelfPlankTilingE2E = compose(
    "shelf_plank_tiling_e2e",
    PromoAssetPlankTilingDetect,
    HierarchicalClmJev,
    dataset="gtmerch_golden",
    also_calls=[MODEL],
    architecture=(
        "EXP-53 E2E: Stage 0 Dual-Buffer CLAHE -> Stage 1 AWP Bay Router + Physical Plank-Support Law "
        "& Density 2x2 Tiling -> Stage 2 Hierarchical Brand-Gate CLM InfoNCE + JEV Contact Sheets"
    ),
    steps=[
        "Setup: build brand centroids, SKU centroids & inner-80% CIELAB descriptors",
        "Stage 0 + 1 (EXP-53): Dual-Buffer CLAHE + AWP bay router + physical shelf-plank support filter + density 2x2 tiling",
        "Stage 2: Hierarchical Brand-Gate + CLM InfoNCE + JEV contact sheets on linear color_image crops",
    ],
)

class _GeneralizableStage1Detector(PromoAssetPlankTilingDetect, DistilledYoloOnnx):
    """Unified Stage 0 + Stage 1 detector for ShelfGeneralizableSpineE2E across Track 1 and Track 2."""

    models = None

    def setup(self, config: dict, ctx: Context) -> None:
        if getattr(self, "dataset", "gtmerch_golden") != "gtmerch_golden":
            try:
                DistilledYoloOnnx.setup(self, config, ctx)
            except Exception:
                self.sku_session = None
                self.dist_session = None

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        if (
            getattr(self, "dataset", "gtmerch_golden") != "gtmerch_golden"
            and getattr(self, "sku_session", None) is not None
        ):
            return DistilledYoloOnnx.detect(self, image, ctx)
        return PromoAssetPlankTilingDetect.detect(self, image, ctx)


class _GeneralizableHierarchicalSpineJev(HierarchicalClmJev):
    """Stage-2 Retriever for ShelfGeneralizableSpineE2E combining strict 2-brand HierarchicalClmJev with EXP-50..52."""

    brand_margin_window = 0.06
    brand_top_b = 2

    def setup(self, config: dict, ctx: Context) -> None:
        from collections import defaultdict
        from approaches.market_share.classification.embedding_text_match import unit
        from utils import dataset as ds_mod

        super().setup(config, ctx)
        self.catalog = ds_mod.catalog(ds_mod.data_root(self.dataset), self.dataset)
        self.brand_to_ids = defaultdict(set)
        proto_vecs: dict[tuple[str, str, str], list[list[float]]] = defaultdict(list)
        for pid, _, v in self.index:
            item = self.catalog.get(pid) or {}
            brand = str(item.get("brand") or item.get("category") or f"sku_{pid}")
            cat = str(item.get("category") or "default_cat")
            ptype = str(item.get("packaging_type") or item.get("pack_form") or "default_pack")
            self.brand_to_ids[brand].add(pid)
            proto_vecs[(brand, cat, ptype)].append(v)
        # Keep pure (brand, category, packaging_type) sub-centroids without single-SKU dilution
        self.brand_subcentroids = defaultdict(list)
        for (brand, _, _), vs in proto_vecs.items():
            if vs:
                self.brand_subcentroids[brand].append(
                    unit([sum(col) / len(vs) for col in zip(*vs, strict=True)])
                )
        self.brand_centroids = {
            b: subs[0] for b, subs in self.brand_subcentroids.items() if subs
        }

    def _narrow_by_brand(self, q: list[float]) -> tuple[set[int] | None, list[tuple[str, float]]]:
        from approaches.market_share.retrieval.jev_laya_hybrid import _dot

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
        import numpy as np
        from approaches.market_share.classification.embedding_text_match import unit
        from approaches.market_share.retrieval.jev_laya_hybrid import LAB_WEIGHT, K, _dot
        from utils.clm import (
            apply_lidstone_pack_fusion,
            apply_projective_scale_prior,
            compute_container_geometry_features,
            infonce_rerank,
        )
        from utils.photometry import apply_illuminant, inner_lab_zones, restore_photometry

        gains = getattr(ctx, "shelf_illuminant", None)
        wb_input = apply_illuminant(image, gains) if gains else image
        restored, _ = restore_photometry(wb_input)
        q = unit(self.emb.image(restored, ctx))
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
        geom = compute_container_geometry_features(image)
        cat_map = getattr(self, "catalog", None)
        scaled = apply_projective_scale_prior(reranked, geom, cat_map, weight=0.004)
        fused, _ = apply_lidstone_pack_fusion(scaled, geom, cat_map, lam=0.003, eps=0.01)
        return fused, q, restored

    def _stage_a_gate(self, ranked: list[tuple[float, int, str]]) -> tuple[bool, dict]:
        from approaches.market_share.retrieval.clm_info_nce import ClmContrastive
        from approaches.market_share.retrieval.jev_laya_hybrid import K
        from utils.clm import compute_zscore_and_lid

        confident_clm, meta = ClmContrastive._stage_a_gate(self, ranked)
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

    def pick_with_confidence(
        self, image: Image.Image, top: list[tuple[float, int, str]], ctx: Context
    ) -> tuple[int | None, float, dict]:
        from approaches.market_share.retrieval.jev_laya_hybrid import JevLayaHybrid

        return JevLayaHybrid.pick_with_confidence(self, image, top, ctx)

    def pick_batch_with_confidence(
        self,
        queries: list[Image.Image],
        tops: list[list[tuple[float, int, str]]],
        ctx: Context,
    ) -> list[tuple[int | None, float, dict]]:
        from approaches.market_share.retrieval.jev_laya_hybrid import JevLayaHybrid

        return JevLayaHybrid.pick_batch_with_confidence(self, queries, tops, ctx)


ShelfGeneralizableSpineE2E = compose(
    "shelf_generalizable_spine_e2e",
    _GeneralizableStage1Detector,
    _GeneralizableHierarchicalSpineJev,
    dataset="gtmerch_golden",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        "Unified Non-Overfitting Production Spine (EXP-50..53): Stage 0 Dual-Buffer CLAHE -> "
        "Stage 1 Plank-Support Law & 2x2 Tiling -> Stage 2 Projective Scale + Saliency-Diff Zoom + Lidstone Pack Crown CLEF"
    ),
    steps=[
        "Setup: embed reference gallery photos and load SKU catalog metadata",
        "Stage 0 + 1 (EXP-53): Dual-Buffer CLAHE + AWP bay router / ONNX YOLO + physical shelf-plank support filter + density 2x2 tiling",
        "Stage 1.5 (EXP-50): Aspect-preserving cap-padded crop (+12% top pad) + projective scale features (h/w, crown_taper, h/h_plank)",
        "Stage 2A (EXP-50 + EXP-52): Hierarchical Brand-Gate + CLM InfoNCE + Lidstone-calibrated 16-class container specialist fusion",
        "Stage 2B (EXP-51 + EXP-52): Self-Supervised Reference-Diff Saliency Zoom + Top-22% Crown/Nozzle Strip + 1D Shelf-Row Smoothing",
    ],
)




