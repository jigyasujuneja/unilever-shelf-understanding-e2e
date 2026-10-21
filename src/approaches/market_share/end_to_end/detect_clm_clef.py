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
    SftClmClefCascade,
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

Shelf960SftClmClefE2E = compose(
    "shelf_960_sft_clm_clef_e2e",
    DistilledYolo960SoftNms,
    SftClmClefCascade,
    dataset="rpc",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        f"EXP-29 / EXP-33: Stage 1 960px High-Res + Gaussian Soft-NMS ONNX Detector ({ONNX_MODEL_ID}) -> "
        "Stage 2 Calibrated 5-Attribute CLM InfoNCE Vote Gate + Contrastive CLEF Contact Sheet (Gemini 3.5 Flash-Lite / LoRA SFT)"
    ),
    steps=[
        "Setup: load 960px Soft-NMS ONNX detector + build multi-prototype (Brand, Category, Pack) sub-centroids & catalog",
        "Stage 1: 960px High-Res + Aspect-Aware Gaussian Soft-NMS ONNX detection ($0 token cost)",
        "Stage 2A: CLM InfoNCE + Z-score/LID gate intersected with 5-attribute softmax vote shares (0 VLM calls on confident crops)",
        "Stage 2B: Escalate uncertain crops to 5-panel CLEF Zoom Contact Sheet with EXP-28 catalog metadata & contrastive token diffs",
        "Stage 2C: Enforce (Brand, Base_Variant) -> Category -> Valid Category__PackType catalog consistency",
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
        from approaches.market_share.detection.onnx_yolo_detector import (
            _segment_composite_strip_columns,
        )

        det_im, color_im, stats = dual_buffer_shelf_photometry(image)
        ctx.detector_image = det_im
        ctx.color_image = color_im
        ctx.trace.step(
            "Stage 0 Dual-Buffer Photometry",
            f"Shades-of-Gray gains={stats['gains']}, CIELAB L* CLAHE std {stats['l_std_before']:.1f} -> {stats['l_std_after']:.1f}",
            info=stats,
        )
        # Check composite strip gutters on raw image before tile-adaptive CLAHE alters uniform gutter std
        if _segment_composite_strip_columns(image) is not None:
            return super().detect(image, ctx)
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

@register
class DistilledYolo960Stage0Clahe(DistilledYolo960SoftNms):
    """EXP-55 Market-Share Stage 0 + Stage 1: Dual-Buffer CLAHE -> 960px High-Res Gaussian Soft-NMS ONNX."""

    name = "distilled_yolo_960_stage0_clahe"
    architecture = (
        f"EXP-55: Stage 0 Dual-Buffer Photometry (CIELAB L* CLAHE) -> "
        f"Stage 1 960px High-Res + Aspect-Aware Gaussian Soft-NMS ONNX ({ONNX_MODEL_ID}, $0 token cost)"
    )
    steps = [
        "Setup: load Vertex AI Model Registry ONNX graphs into CPUExecutionProvider",
        "Stage 0: Build dual photometric buffers (CIELAB L* CLAHE detector_image + linear Shades-of-Gray color_image)",
        "Stage 1 (EXP-55): 960px High-Res + Sachet-Gated Gaussian Soft-NMS ONNX detection on detector_image ($0 token cost)",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        from approaches.market_share.detection.onnx_yolo_detector import (
            _segment_composite_strip_columns,
        )

        det_im, color_im, stats = dual_buffer_shelf_photometry(image)
        ctx.detector_image = det_im
        ctx.color_image = color_im
        ctx.trace.step(
            "Stage 0 Dual-Buffer Photometry",
            f"Shades-of-Gray gains={stats['gains']}, CIELAB L* CLAHE std {stats['l_std_before']:.1f} -> {stats['l_std_after']:.1f}",
            info=stats,
        )
        # Preserve composite-strip gutter detection on raw image before tile-adaptive CLAHE
        if _segment_composite_strip_columns(image) is not None:
            return super().detect(image, ctx)
        return super().detect(det_im, ctx)


@register
class PromoAssetBayScopedPlankDetect(PromoAssetPlankTilingDetect):
    """EXP-55 + EXP-57 Merchandising Stage 1: AWP Bay Router + Plank-Support Law + Pre-Embedding Bay Gate."""

    name = "promo_asset_bay_scoped_plank_detect"
    use_case = "merchandising"
    dataset = "gtmerch_golden"
    density_quadrant_tiling = True
    architecture = (
        "EXP-55 + EXP-57: Stage 0 CLAHE -> AWP Bay Router + Primary-Bay High-Res Pass "
        "-> Physical Plank-Support Filter + Class-0 Bay-Scoped Pre-Embedding Gate"
    )
    steps = [
        "Stage 0: Build dual photometric buffers (CIELAB L* CLAHE detector_image + linear Shades-of-Gray color_image)",
        "Stage 1a: AWP-conditioned pass detects Class 0 display bays (WIN/CTU/MB2/MB3/FULLWALL) and Class 1 SKUs",
        "Stage 1b (EXP-53 + EXP-57): Enforce physical shelf-plank support (y_max alignment) and prune peripheral out-of-bay store-edge boxes before Stage 2 embedding",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        boxes = super().detect(image, ctx)
        asset_boxes = getattr(ctx, "detected_asset_boxes", None) or []
        scene_mode = getattr(ctx, "detected_fixture_type", "") or ""
        if not asset_boxes or not boxes or scene_mode in ("FULLWALL", "NONE_HANGER_FALLBACK", "HAN"):
            return boxes
        w, h = image.size
        img_area = float(max(1, w * h))
        bay_area = sum(max(0.0, ab[2] - ab[0]) * max(0.0, ab[3] - ab[1]) for ab in asset_boxes)
        # Only apply pre-embedding bay pruning when Class 0 bay covers a primary fixture (>=22% of frame)
        # and retains >=75% of plank-supported boxes (pruning only peripheral store-edge clutter).
        if bay_area < 0.22 * img_area:
            return boxes
        scoped: list[Box] = []
        for b in boxes:
            cx = 0.5 * (b[0] + b[2])
            cy = 0.5 * (b[1] + b[3])
            in_any_bay = False
            for ab in asset_boxes:
                aw = max(1.0, ab[2] - ab[0])
                ah = max(1.0, ab[3] - ab[1])
                if (
                    ab[0] - 0.08 * aw <= cx <= ab[2] + 0.08 * aw
                    and ab[1] - 0.08 * ah <= cy <= ab[3] + 0.08 * ah
                ):
                    in_any_bay = True
                    break
            if in_any_bay:
                scoped.append(b)
        if scoped and len(scoped) < len(boxes) and len(scoped) >= max(8, int(0.75 * len(boxes))):
            dropped = len(boxes) - len(scoped)
            ctx.trace.step(
                "EXP-57 Class-0 Bay-Scoped Pre-Embedding Gate",
                f"Pruned {dropped} peripheral out-of-bay shelf box(es) -> {len(scoped)} in-bay SKUs sent to Stage 2",
                boxes=scoped,
                info={"pruned_out_of_bay": dropped, "kept_in_bay": len(scoped)},
            )
            return scoped
        return boxes


class _GeneralizableStage1Detector(PromoAssetBayScopedPlankDetect, DistilledYolo960Stage0Clahe):
    """Unified Stage 0 + Stage 1 detector for ShelfGeneralizableSpineE2E across Market Share & Merchandising."""

    models = None

    def setup(self, config: dict, ctx: Context) -> None:
        if getattr(self, "dataset", "gtmerch_golden") != "gtmerch_golden":
            try:
                DistilledYolo960Stage0Clahe.setup(self, config, ctx)
            except Exception:
                self.sku_session = None
                self.dist_session = None

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        if (
            getattr(self, "dataset", "gtmerch_golden") != "gtmerch_golden"
            and getattr(self, "sku_session", None) is not None
        ):
            return DistilledYolo960Stage0Clahe.detect(self, image, ctx)
        return PromoAssetBayScopedPlankDetect.detect(self, image, ctx)


class _GeneralizableHierarchicalSpineJev(HierarchicalClmJev):
    """Stage-2 Retriever for ShelfGeneralizableSpineE2E combining EXP-50..52 with EXP-54..57.

    Fixes:
      - EXP-54: Includes SKU-level centroids for multi-SKU variant brands in brand_subcentroids
        while suppressing 1-SKU `/brand_data/` hubness noise folders; enforces 1D horizontal
        shelf-plank category/brand consensus across detected shelf rows.
      - EXP-56: Inherits L8HulUltimate's same-sister Stage A fast gate and Contrastive Metadata
        Zoom-Strip contact sheets (`zoom_multi_sheet` + `build_contrastive_candidate_rubric`),
        routing high-res crops (>=64px) on single-item queries through Saliency-Diff + Crown Strip.
    """

    brand_margin_window = 0.06
    brand_top_b = 3

    def setup(self, config: dict, ctx: Context) -> None:
        from collections import defaultdict
        from approaches.market_share.classification.embedding_text_match import unit
        from utils import dataset as ds_mod

        super().setup(config, ctx)
        self.catalog = ds_mod.catalog(ds_mod.data_root(self.dataset), self.dataset)
        self.brand_to_ids = defaultdict(set)
        self.brand_only_pids: set[int] = set()
        proto_vecs: dict[tuple[str, str, str], list[list[float]]] = defaultdict(list)
        for pid, path, v in self.index:
            item = self.catalog.get(pid) or {}
            brand = str(item.get("brand") or item.get("category") or f"sku_{pid}")
            cat = str(item.get("category") or "default_cat")
            ptype = str(item.get("packaging_type") or item.get("pack_form") or "default_pack")
            self.brand_to_ids[brand].add(pid)
            if "/brand_data/" in str(path) and not item.get("is_hul"):
                self.brand_only_pids.add(pid)
            proto_vecs[(brand, cat, ptype)].append(v)
        # EXP-54: Include (brand, category, packaging_type) sub-centroids AND individual SKU centroids
        # for multi-SKU variant brands (>=2 SKUs) so 35-SKU brands are never starved while 1-SKU
        # `/brand_data/` low-res hubness folders do not flood the brand gate.
        self.brand_subcentroids = defaultdict(list)
        for (brand, _, _), vs in proto_vecs.items():
            if vs:
                self.brand_subcentroids[brand].append(
                    unit([sum(col) / len(vs) for col in zip(*vs, strict=True)])
                )
        for pid, sc in self.sku_centroids.items():
            if pid in self.brand_only_pids:
                continue
            item = self.catalog.get(pid) or {}
            brand = str(item.get("brand") or item.get("category") or f"sku_{pid}")
            if len(self.brand_to_ids.get(brand, ())) >= 2:
                self.brand_subcentroids[brand].append(sc)
        self.brand_centroids = {
            b: subs[0] for b, subs in self.brand_subcentroids.items() if subs
        }

    def _narrow_by_brand(self, q: list[float]) -> tuple[set[int] | None, list[tuple[str, float]]]:
        from approaches.market_share.retrieval.jev_laya_hybrid import _dot

        if len(self.brand_subcentroids) <= 1:
            return None, []
        scored = sorted(
            (
                (
                    b,
                    max(_dot(q, c) for c in subs)
                    - (0.035 if all(pid in self.brand_only_pids for pid in self.brand_to_ids.get(b, ())) else 0.0),
                )
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
        explicit_allowed = allowed_ids is not None
        if allowed_ids is None:
            allowed_ids, _ = self._narrow_by_brand(q)
        best: dict[int, tuple[float, int, str]] = {}
        global_best: dict[int, tuple[float, int, str]] = {}
        brand_only = getattr(self, "brand_only_pids", set())
        if (
            hasattr(self, "index_matrix")
            and self.index_matrix.ndim == 2
            and self.index_matrix.shape[0] == len(self.index)
            and self.index_matrix.shape[0] > 0
        ):
            sims = (self.index_matrix @ np.asarray(q, dtype=np.float32)).tolist()
            for s, pid, path in zip(sims, self.index_pids, self.index_paths, strict=True):
                fs = float(s) - (0.035 if pid in brand_only else 0.0)
                if pid not in global_best or fs > global_best[pid][0]:
                    global_best[pid] = (fs, pid, path)
                if allowed_ids and pid not in allowed_ids:
                    continue
                if pid not in best or fs > best[pid][0]:
                    best[pid] = (fs, pid, path)
        else:
            for pid, path, v in self.index:
                s = _dot(q, v) - (0.035 if pid in brand_only else 0.0)
                if pid not in global_best or s > global_best[pid][0]:
                    global_best[pid] = (s, pid, path)
                if allowed_ids and pid not in allowed_ids:
                    continue
                if pid not in best or s > best[pid][0]:
                    best[pid] = (s, pid, path)
        # EXP-54: Only backfill from global_best when brand gate produced fewer than 6 candidates
        if not explicit_allowed and len(best) < 6 and global_best:
            for fs, pid, path in sorted(global_best.values(), reverse=True)[:K]:
                if pid not in best or fs > best[pid][0]:
                    best[pid] = (fs, pid, path)
                if len(best) >= K:
                    break
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

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        ids = super().identify_boxes(image, boxes, ctx)
        if len(boxes) < 3 or not getattr(self, "catalog", None):
            return ids
        # EXP-54: 1D Horizontal Shelf-Plank Category & Brand Consensus Prior
        from collections import Counter

        order = sorted(range(len(boxes)), key=lambda i: boxes[i][3])
        planks: list[list[int]] = []
        for i in order:
            y2 = boxes[i][3]
            h = max(1.0, boxes[i][3] - boxes[i][1])
            placed = False
            for r in planks:
                ref_y2 = sum(boxes[k][3] for k in r) / len(r)
                ref_h = sum(max(1.0, boxes[k][3] - boxes[k][1]) for k in r) / len(r)
                if abs(y2 - ref_y2) <= 0.30 * max(h, ref_h):
                    r.append(i)
                    placed = True
                    break
            if not placed:
                planks.append([i])

        plank_checks = 0
        brand_only = getattr(self, "brand_only_pids", set())
        for r in planks:
            cats = []
            for i in r:
                pid_r = ids[i]
                if pid_r is not None and pid_r not in brand_only:
                    c_val = str((self.catalog.get(pid_r) or {}).get("category") or "")
                    if c_val:
                        cats.append(c_val)
            if len(cats) < 2:
                continue
            dom_cat, dom_count = Counter(cats).most_common(1)[0]
            if dom_count / len(cats) < 0.60:
                continue
            # Sort plank left-to-right by x_center
            r_lr = sorted(r, key=lambda idx: 0.5 * (boxes[idx][0] + boxes[idx][2]))
            for pos, i in enumerate(r_lr):
                pid = ids[i]
                if pid is None:
                    continue
                item = self.catalog.get(pid) or {}
                cur_cat = str(item.get("category") or "")
                # If a box on a homogeneous plank was assigned a 1-SKU `/brand_data/` hubness noise ID
                # or an out-of-category outlier flanked by two identical-brand neighbors on the same plank,
                # snap to the flanking neighbor SKU.
                if pid in brand_only or (cur_cat and cur_cat != dom_cat):
                    neighbors = []
                    if pos > 0 and ids[r_lr[pos - 1]] is not None and ids[r_lr[pos - 1]] not in brand_only:
                        neighbors.append(ids[r_lr[pos - 1]])
                    if pos + 1 < len(r_lr) and ids[r_lr[pos + 1]] is not None and ids[r_lr[pos + 1]] not in brand_only:
                        neighbors.append(ids[r_lr[pos + 1]])
                    if len(neighbors) == 2 and neighbors[0] == neighbors[1]:
                        ids[i] = neighbors[0]
                        plank_checks += 1
                    elif pid in brand_only and neighbors:
                        n_cat = str((self.catalog.get(neighbors[0]) or {}).get("category") or "")
                        if n_cat == dom_cat:
                            ids[i] = neighbors[0]
                            plank_checks += 1
        ctx.trace.step(
            "EXP-54 1D Shelf-Plank Consensus Prior",
            f"Grouped {len(boxes)} boxes into {len(planks)} shelf planks ({plank_checks} plank consensus corrections)",
            info={"shelf_planks": len(planks), "plank_priors_checked": plank_checks},
        )
        return ids


ShelfGeneralizableSpineE2E = compose(
    "shelf_generalizable_spine_e2e",
    _GeneralizableStage1Detector,
    _GeneralizableHierarchicalSpineJev,
    dataset="gtmerch_golden",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        "Unified Non-Overfitting Production Spine (EXP-50..57): Stage 0 Dual-Buffer CLAHE -> "
        "Scene-Adaptive Stage 1 (960px Soft-NMS ONNX / Bay-Scoped Plank Law) -> "
        "Stage 2 Unstarved Brand-Gate + Projective Scale + Lidstone Pack + Resolution-Adaptive Saliency Zoom"
    ),
    steps=[
        "Setup: build multi-prototype brand sub-centroids + SKU centroids + 16-class Lidstone container profiles",
        "Stage 0 + 1 (EXP-53 + EXP-55 + EXP-57): Dual-Buffer CLAHE + 960px Soft-NMS ONNX (Open-Shelf) or AWP Bay-Scoped Plank Law (Promo-Bay)",
        "Stage 1.5 (EXP-50): Aspect-preserving cap-padded crop (+12% top pad) + projective scale features (h/w, crown_taper, h/h_plank)",
        "Stage 2A (EXP-50 + EXP-52 + EXP-54): Unstarved Top-8 Brand-Gate + Global Floor + CLM InfoNCE + Lidstone 16-class pack fusion",
        "Stage 2B (EXP-51 + EXP-54 + EXP-56): Resolution-Adaptive Saliency-Diff Zoom / Multi-Sheet + 1D Shelf-Plank Consensus",
    ],
)

ShelfMarketshareGeneralizableE2E = compose(
    "shelf_marketshare_generalizable_e2e",
    DistilledYolo960Stage0Clahe,
    _GeneralizableHierarchicalSpineJev,
    dataset="ms_json_golden",
    use_case="market_share",
    also_calls=[ONNX_MODEL_ID, MODEL],
    architecture=(
        "EXP-54..56 Open-Shelf Market Share Spine: Stage 0 Dual-Buffer CLAHE -> "
        f"Stage 1 960px Soft-NMS ONNX ({ONNX_MODEL_ID}, $0 token cost) -> "
        "Stage 2 Unstarved Brand-Gate + Projective Scale + Lidstone Pack + 1D Plank Consensus"
    ),
    steps=[
        "Setup: load 960px Soft-NMS ONNX detector + build multi-prototype brand & SKU centroids",
        "Stage 0 + 1 (EXP-55): Dual-Buffer CLAHE + 960px High-Res Gaussian Soft-NMS ONNX detection ($0 token cost)",
        "Stage 2A (EXP-50 + EXP-52 + EXP-54): Unstarved Brand-Gate + Global Top-2K Floor + CLM InfoNCE + Projective Scale & Lidstone Pack",
        "Stage 2B (EXP-54 + EXP-56): Resolution-Adaptive Saliency Zoom / Multi-Sheet + 1D Shelf-Plank Consensus",
    ],
)

ShelfMerchandisingBayE2E = compose(
    "shelf_merchandising_bay_e2e",
    PromoAssetBayScopedPlankDetect,
    _GeneralizableHierarchicalSpineJev,
    dataset="gtmerch_golden",
    use_case="merchandising",
    also_calls=[MODEL],
    architecture=(
        "EXP-53..57 Promo-Bay Merchandising Spine: Stage 0 Dual-Buffer CLAHE -> "
        "Stage 1 AWP Bay Router + Physical Plank-Support Law + Pre-Embedding Bay Gate -> "
        "Stage 2 Unstarved Brand-Gate + Projective Scale + Lidstone Pack + Stage 6 Planogram/OOS"
    ),
    steps=[
        "Setup: build multi-prototype brand & SKU centroids + 16-class Lidstone container profiles",
        "Stage 0 + 1 (EXP-53 + EXP-57): Dual-Buffer CLAHE + AWP Bay Router + Plank-Support Filter + Pre-Embedding Bay Gate",
        "Stage 2A (EXP-50 + EXP-52 + EXP-54): Unstarved Brand-Gate + Global Top-2K Floor + CLM InfoNCE + Projective Scale & Lidstone Pack",
        "Stage 2B + Stage 6: Batched Contact-Sheet Verification + Class 0 Bay-Scoped Planogram & OOS Void Compliance",
    ],
)
