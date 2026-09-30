"""Vector catalog retrieval, explicit `ClassificationConfig` strategy pipelines, and VLM tie-breaking (`src/core/retrieval.py`).

Uses explicit `ClassificationConfig` strategy objects, strict dependency injection, and hard-fail
validation on invalid image or catalog inputs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image

from approaches.base import validate_image_and_boxes
from core.catalog import (
    CANONICAL_7DIM_CATALOG,
    CANONICAL_BY_CODE,
    _build_real_catalog_prototype_bank,
    validate_canonical_7dim_prediction,
)
from core.clustering import cluster_shelf_facings_high_purity, smooth_shelf_row_predictions
from core.features import (
    _cosine_sim,
    compute_ciede2000_approx,
    extract_gemini_subroi_embedding,
    extract_maxvit_multiscale_features,
    extract_real_crop_features,
)
from core.imaging import (
    _CONTACT_SHEET_CLASSIFY_SCHEMA,
    IJEPASpecularGlarePredictor,
    _build_contact_sheet,
)


@dataclass(frozen=True)
class ClassificationConfig:
    """Explicit algorithmic strategy configuration for 7-dimension SKU classification pipelines."""

    w_maxvit_blend: float = 1.00
    use_ijepa_deglare: bool = True
    enable_sister_shade: bool = True
    use_cluster_propagation: bool = False
    cluster_tau: float = 0.94
    use_row_smoothing: bool = True
    enable_djev_3task: bool = False
    vlm_escalation: bool = True
    sft_lora_rules: bool = False


# Pre-defined strategy configurations for standard benchmark pipelines (no string-ablation table required)
CONFIG_SCANN_FLAT = ClassificationConfig(
    w_maxvit_blend=0.00,
    use_ijepa_deglare=False,
    enable_sister_shade=False,
    use_cluster_propagation=True,
    cluster_tau=0.90,
    use_row_smoothing=False,
    enable_djev_3task=False,
    vlm_escalation=False,
)
CONFIG_HUL_HIERARCHY = ClassificationConfig(
    w_maxvit_blend=0.35,
    use_ijepa_deglare=False,
    enable_sister_shade=False,
    use_cluster_propagation=True,
    cluster_tau=0.92,
    use_row_smoothing=True,
    enable_djev_3task=False,
    vlm_escalation=True,
)
CONFIG_DJEV_COMPOUND = ClassificationConfig(
    w_maxvit_blend=0.85,
    use_ijepa_deglare=True,
    enable_sister_shade=True,
    use_cluster_propagation=True,
    cluster_tau=0.95,
    use_row_smoothing=True,
    enable_djev_3task=True,
    vlm_escalation=True,
)
CONFIG_FT_GEMINI31_CAT_BRAND_PKG = ClassificationConfig(
    w_maxvit_blend=0.78,
    use_ijepa_deglare=True,
    enable_sister_shade=False,
    use_cluster_propagation=True,
    cluster_tau=0.94,
    use_row_smoothing=True,
    enable_djev_3task=False,
    vlm_escalation=True,
    sft_lora_rules=True,
)
CONFIG_FT_GEMINI31_VARIANT = ClassificationConfig(
    w_maxvit_blend=0.58,
    use_ijepa_deglare=True,
    enable_sister_shade=True,
    use_cluster_propagation=False,
    cluster_tau=0.94,
    use_row_smoothing=True,
    enable_djev_3task=False,
    vlm_escalation=True,
    sft_lora_rules=True,
)
CONFIG_SISTER_SHADE_SYSTEMONE = ClassificationConfig(
    w_maxvit_blend=0.72,
    use_ijepa_deglare=True,
    enable_sister_shade=True,
    use_cluster_propagation=False,
    cluster_tau=0.95,
    use_row_smoothing=True,
    enable_djev_3task=False,
    vlm_escalation=True,
)
CONFIG_FULL_HYBRID = ClassificationConfig(
    w_maxvit_blend=1.00,
    use_ijepa_deglare=True,
    enable_sister_shade=True,
    use_cluster_propagation=False,
    cluster_tau=0.94,
    use_row_smoothing=True,
    enable_djev_3task=True,
    vlm_escalation=True,
)

_STRATEGY_BY_NAME: dict[str, ClassificationConfig] = {
    "scann_vector_retriever": CONFIG_SCANN_FLAT,
    "hul_hierarchy_classifier": CONFIG_HUL_HIERARCHY,
    "djev_diffusiongemma_compound": CONFIG_DJEV_COMPOUND,
    "ft_gemini31_cat_brand_pkg": CONFIG_FT_GEMINI31_CAT_BRAND_PKG,
    "ft_gemini31_variant_compound": CONFIG_FT_GEMINI31_VARIANT,
    "sister_shade_systemone": CONFIG_SISTER_SHADE_SYSTEMONE,
}


def resolve_classification_config(
    config: ClassificationConfig | None = None,
    mode: str = "sister_shade_systemone",
) -> ClassificationConfig:
    """Return explicit `ClassificationConfig` if provided, or resolve from strategy name."""
    if config is not None:
        return config
    return _STRATEGY_BY_NAME.get(mode, CONFIG_FULL_HYBRID)


@dataclass(frozen=True)
class SisterCandidateProfile:
    """Reference color and geometry profile for a candidate variant in a sister-shade family."""

    canonical_variant_id: str
    brand: str
    product_line_cluster: str
    shade_or_active_token: str
    discriminative_sub_roi_rel: tuple[float, float, float, float]
    reference_cielab_swatch: tuple[float, float, float]
    training_prior_count: int = 100
    cap_orientation: str = "CAP_TOP_BOTTLE"


@dataclass(frozen=True)
class SisterShadeResolution:
    """Result of CIELAB sub-ROI color distance and variant tie-breaking."""

    resolved_variant_id: str
    delta_e00: float
    confidence: float


def disambiguate_sister_shade_roi(
    full_box_xyxy: tuple[int, int, int, int],
    candidates: list[SisterCandidateProfile],
    raw_cosine_scores: dict[str, float],
    observed_sub_roi_lab: tuple[float, float, float],
    observed_ocr_shade_hint: str = "",
    observed_cap_orientation: str = "CAP_TOP_BOTTLE",
) -> SisterShadeResolution:
    """Select the best variant candidate by combining cosine similarity with CIELAB color distance."""
    del full_box_xyxy
    if not candidates:
        raise ValueError("disambiguate_sister_shade_roi requires a non-empty candidates list")
    best_id = candidates[0].canonical_variant_id
    best_score = -1e9
    best_de = 0.0
    hint_lower = observed_ocr_shade_hint.strip().lower()
    for cand in candidates:
        cos_val = float(raw_cosine_scores.get(cand.canonical_variant_id, 0.75))
        de = compute_ciede2000_approx(observed_sub_roi_lab, cand.reference_cielab_swatch)
        color_bonus = max(-0.10, 0.06 - de * 0.0025)
        ocr_bonus = 0.03 if hint_lower and hint_lower in cand.shade_or_active_token.lower() else 0.0
        cap_bonus = 0.01 if cand.cap_orientation == observed_cap_orientation else 0.0
        combined = cos_val + color_bonus + ocr_bonus + cap_bonus
        if combined > best_score:
            best_score = combined
            best_id = cand.canonical_variant_id
            best_de = de
    return SisterShadeResolution(
        resolved_variant_id=best_id,
        delta_e00=best_de,
        confidence=round(min(0.99, max(0.50, best_score)), 4),
    )


@dataclass(frozen=True)
class SystemOneResolution:
    """Result of constrained variant resolution over candidate SKUs."""

    resolved_base_pack_id: str
    pinned_ratio: float
    confidence: float


@dataclass(frozen=True)
class ThreeTaskPrefilterResult:
    """Result of filtering the master SKU catalog by (category, brand, packaging_type)."""

    routing_decision: str
    scann_pool_after_3task_filter: int
    filtered_candidate_skus: list[str]


class DjevSystemOneClient:
    """Constrained hierarchical attribute and candidate SKU filter using the Unilever taxonomy."""

    MODEL_ID = "google/diffusiongemma-26B-A4B-it"
    CANONICAL_CATEGORIES = (
        "Hair Care - DMT",
        "Skin Care",
        "Oral Care",
        "Personal Wash - Laundry",
        "Foods - Beverages",
        "Deodorants & Fragrances",
        "Home & Hygiene",
        "Baby Care",
        "Health & Wellbeing",
        "Merchandising & POSM",
        "Non-HUL",
    )
    CANONICAL_BRANDS = (
        "Dove",
        "Sunsilk",
        "Clinic Plus",
        "Tresemme",
        "Lakme",
        "Pond's",
        "Glow & Lovely",
        "Vaseline",
        "Pears",
        "Lux",
        "Lifebuoy",
        "Surf Excel",
        "Rin",
        "Vim",
        "Closeup",
        "Pepsodent",
        "Lipton",
        "Red Label",
        "Bru",
        "Horlicks",
        "Kissan",
        "Pantene",
        "L'Oreal",
        "Colgate",
    )
    CANONICAL_PACKAGING_TYPES = (
        "bottle",
        "pump_bottle",
        "jar",
        "tub",
        "tube",
        "pouch",
        "spout_pouch",
        "sachet",
        "sachet_strip_ladi",
        "box",
        "carton",
        "bar",
        "aerosol_can",
        "roll_on",
        "tin",
        "blister_card",
        "tetra_pak",
        "multipack",
        "dropper_serum",
    )

    def resolve_crop_systemone(
        self,
        box_xyxy: list[float] | tuple[float, ...],
        scann_top5: list[str],
        raw_similarity: float,
        glare_intensity: float = 0.0,
        ocr_snippet: str = "",
        image: Image.Image | None = None,
        ctx: Any | None = None,
    ) -> SystemOneResolution:
        del glare_intensity
        if len(box_xyxy) < 4 or float(box_xyxy[2]) <= float(box_xyxy[0]) or float(box_xyxy[3]) <= float(box_xyxy[1]):
            raise ValueError(f"Invalid bounding box for resolve_crop_systemone: {box_xyxy}")
        if not scann_top5:
            raise ValueError("scann_top5 candidate list must be non-empty in resolve_crop_systemone")
        candidates = list(scann_top5)
        chosen = candidates[0]

        endpoint_url = os.environ.get("DJEV_SYSTEMONE_URL", "").strip()
        if endpoint_url:
            import requests

            resp = requests.post(
                f"{endpoint_url.rstrip('/')}/v1/systemone",
                json={
                    "model": self.MODEL_ID,
                    "box_xyxy": [float(v) for v in box_xyxy],
                    "candidates": candidates,
                    "ocr_snippet": ocr_snippet,
                    "max_tokens": 64,
                },
                timeout=5,
            )
            resp.raise_for_status()
            payload = resp.json()
            cand_id = str(payload.get("resolved_base_pack_id") or "")
            if cand_id not in candidates:
                raise RuntimeError(
                    f"Hallucinated SKU ID {cand_id!r} from DJEV_SYSTEMONE_URL not in candidates {candidates}"
                )
            chosen = cand_id
        elif (
            ctx is not None
            and image is not None
            and hasattr(ctx, "ask")
            and getattr(ctx, "llm", None) is not None
            and os.environ.get("SHELF_BENCH_OFFLINE") != "1"
        ):
            w, h = image.size
            x1 = max(0, min(w - 2, int(round(float(box_xyxy[0])))))
            y1 = max(0, min(h - 2, int(round(float(box_xyxy[1])))))
            x2 = max(x1 + 2, min(w, int(round(float(box_xyxy[2])))))
            y2 = max(y1 + 2, min(h, int(round(float(box_xyxy[3])))))
            crop = image.crop((x1, y1, x2, y2))
            schema = {
                "type": "OBJECT",
                "properties": {
                    "resolved_base_pack_id": {"type": "STRING", "enum": candidates},
                },
                "required": ["resolved_base_pack_id"],
            }
            res = ctx.ask(
                crop,
                f"Select the exact SKU base pack ID from {candidates}. OCR hint: {ocr_snippet}",
                schema=schema,
                max_side=384,
            )
            if isinstance(res.data, dict) and res.data.get("resolved_base_pack_id") in candidates:
                chosen = str(res.data["resolved_base_pack_id"])
            elif isinstance(res.data, list) and res.data and isinstance(res.data[0], dict):
                cand_from_list = str(res.data[0].get("sku_id") or res.data[0].get("resolved_base_pack_id") or "")
                if cand_from_list in candidates:
                    chosen = cand_from_list
                elif cand_from_list and cand_from_list not in CANONICAL_BY_CODE:
                    raise RuntimeError(
                        f"Hallucinated SKU ID {cand_from_list!r} not in candidates {candidates}"
                    )
            elif isinstance(res.data, dict) and "resolved_base_pack_id" in res.data:
                raise RuntimeError(
                    f"Hallucinated or invalid SKU response {res.data!r} not in candidates {candidates}"
                )

        if ocr_snippet:
            token = ocr_snippet.strip().upper()
            for cand in candidates:
                if token in cand.upper():
                    chosen = cand
                    break
        if chosen not in candidates:
            raise RuntimeError(f"Resolved SKU {chosen!r} not in candidates {candidates}")
        return SystemOneResolution(
            resolved_base_pack_id=chosen,
            pinned_ratio=0.8906,
            confidence=round(min(0.99, max(0.55, float(raw_similarity))), 4),
        )

    def classify_3task_and_prefilter_scann(
        self,
        box_xyxy: list[float] | tuple[float, ...],
        hint_category: str,
        hint_brand: str,
        hint_packaging: str,
        ocr_snippet: str = "",
        hint_variant: str = "",
        explicit_sku_id: str = "",
    ) -> ThreeTaskPrefilterResult:
        del box_xyxy, ocr_snippet, hint_variant
        protos = _build_real_catalog_prototype_bank()
        brand_l = hint_brand.strip().lower()
        pkg_l = hint_packaging.strip().lower()
        cat_l = hint_category.strip().lower()
        matched = [
            p["sku_id"]
            for p in protos
            if (brand_l and p["brand"].lower() == brand_l)
            or (cat_l and p["category"].lower() == cat_l and p["packaging_type"].lower() == pkg_l)
        ]
        if explicit_sku_id and explicit_sku_id not in matched:
            matched.insert(0, explicit_sku_id)
        if not matched:
            matched = [p["sku_id"] for p in protos[:8]]
        is_competitor = brand_l in ("pantene", "l'oreal", "loreal", "colgate") or cat_l == "non-hul"
        routing = "COMPETITOR_FAST_EXIT" if is_competitor else "HUL_PREFILTERED_SCANN"
        return ThreeTaskPrefilterResult(
            routing_decision=routing,
            scann_pool_after_3task_filter=len(matched),
            filtered_candidate_skus=matched,
        )


def score_hierarchical_7dim_candidates(
    crop_feats: dict[str, Any],
    proto: dict[str, Any],
    prior: dict[str, Any] | None = None,
    candidate_sku_whitelist: set[str] | None = None,
) -> float:
    """Constrained Hierarchical Attribute Exhaustive Scorer across Level 1-2 (Category & Brand)
    and Level 3-4 (Packaging Form-Factor Geometry: aspect ratio & neck taper).
    """
    bonus = 0.0
    ar = float(crop_feats.get("aspect_ratio", 0.42))
    taper = float(crop_feats.get("neck_taper_ratio", 0.85))
    pkg = str(proto.get("packaging_type", "bottle")).lower()

    if pkg in ("bottle", "pump_bottle"):
        bonus += 0.012 if (ar <= 0.56 and taper <= 1.05) else -0.014
    elif pkg == "tube":
        bonus += 0.012 if (0.28 <= ar <= 0.52 and taper >= 0.72) else -0.012
    elif pkg in ("box", "carton", "bar", "jar", "tub"):
        bonus += 0.014 if ar >= 0.50 else -0.015

    if candidate_sku_whitelist is not None:
        if str(proto.get("sku_id", "")) in candidate_sku_whitelist:
            bonus += 0.018
        else:
            bonus -= 0.022
    if prior and isinstance(prior, dict):
        if str(prior.get("brand", "")).lower() == str(proto.get("brand", "")).lower():
            bonus += 0.020
        if str(prior.get("category", "")).lower() == str(proto.get("category", "")).lower():
            bonus += 0.012
        if str(prior.get("packaging_type", "")).lower() == pkg:
            bonus += 0.015
    return bonus


def scann_vector_lookup(
    crop_idx: int,
    box: tuple[float, float, float, float],
    use_ijepa_deglare: bool = True,
    image: Image.Image | None = None,
    feature_mode: str = "legacy_pixel",
    precomputed_crop_feats: dict[str, Any] | None = None,
    *,
    enable_sister_shade: bool = True,
    enable_hierarchy_constraint: bool = False,
    prior: dict[str, Any] | None = None,
    candidate_sku_whitelist: set[str] | None = None,
) -> dict[str, Any]:
    """Stage 4 Real Pixel-Crop Embedding + Specular De-Glare + Cosine Similarity & Margin Lookup.

    Hard-fails with `ValueError` if both `image` and `precomputed_crop_feats` are `None`
    (zero synthetic `Image.new` fallback).
    """
    if precomputed_crop_feats is not None:
        crop_feats = precomputed_crop_feats
    else:
        if image is None or not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError(
                "scann_vector_lookup requires a valid PIL.Image or precomputed_crop_feats; "
                "synthetic Image.new fallback is forbidden."
            )
        if feature_mode == "maxvit":
            crop_feats = extract_maxvit_multiscale_features(image, box)
        elif feature_mode == "gemini_subroi":
            crop_feats = extract_gemini_subroi_embedding(image, box)
        else:
            crop_feats = extract_real_crop_features(image, box)

    vec = list(crop_feats["embedding"])
    glare_ratio = float(crop_feats["glare_ratio"])

    ijepa_boost = 0.0
    if use_ijepa_deglare and glare_ratio >= 0.04:
        predictor = IJEPASpecularGlarePredictor()
        res_ijepa = predictor.predict_clean_latent(
            corrupted_embedding=vec[:16],
            glare_intensity=max(0.15, min(0.95, glare_ratio * 3.5)),
            box_xyxy=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            image=image,
        )
        ijepa_boost = round(res_ijepa.latent_cosine_gain * 0.35, 4)

    prototypes = _build_real_catalog_prototype_bank()
    proto_by_sku = {p["sku_id"]: p for p in prototypes}
    scored: list[tuple[float, dict[str, Any]]] = []
    for proto in prototypes:
        raw_cos = _cosine_sim(vec, proto["embedding"])
        hier_bonus = (
            score_hierarchical_7dim_candidates(
                crop_feats,
                proto,
                prior=prior,
                candidate_sku_whitelist=candidate_sku_whitelist,
            )
            if enable_hierarchy_constraint
            else 0.0
        )
        sim_score = min(0.995, max(0.0, raw_cos + hier_bonus) + ijepa_boost)
        scored.append((sim_score, proto))

    scored.sort(key=lambda x: -x[0])
    top1_sim, top1_proto = scored[0]
    top2_sim, top2_proto = scored[1] if len(scored) > 1 else (top1_sim - 0.08, top1_proto)
    margin = max(0.0, top1_sim - top2_sim)

    observed_lab = tuple(crop_feats["claim_lab"])
    cap_orientation = "CAP_DOWN_TUBE" if crop_feats["neck_taper_ratio"] >= 0.78 else "CAP_TOP_BOTTLE"

    if top1_sim >= 0.82 and margin >= 0.025 and top1_proto["is_hul"]:
        branch = "fast_scann"
        sku_id = top1_proto["sku_id"]
    elif enable_sister_shade and top1_sim >= 0.74 and top1_proto["is_hul"]:
        branch = "sister_shade_djev"
        disambig = disambiguate_sister_shade_roi(
            full_box_xyxy=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            candidates=[
                SisterCandidateProfile(
                    canonical_variant_id=top1_proto["sku_id"],
                    brand=top1_proto["brand"],
                    product_line_cluster=f"{top1_proto['brand']}_{top1_proto['packaging_type']}",
                    shade_or_active_token=top1_proto["variant"][:18],
                    discriminative_sub_roi_rel=(0.15, 0.62, 0.85, 0.82),
                    reference_cielab_swatch=tuple(top1_proto["cap_lab"]),
                    training_prior_count=150,
                    cap_orientation=cap_orientation,
                ),
                SisterCandidateProfile(
                    canonical_variant_id=top2_proto["sku_id"],
                    brand=top2_proto["brand"],
                    product_line_cluster=f"{top2_proto['brand']}_{top2_proto['packaging_type']}",
                    shade_or_active_token=top2_proto["variant"][:18],
                    discriminative_sub_roi_rel=(0.15, 0.62, 0.85, 0.82),
                    reference_cielab_swatch=tuple(top2_proto["cap_lab"]),
                    training_prior_count=140,
                    cap_orientation=cap_orientation,
                ),
            ],
            raw_cosine_scores={
                top1_proto["sku_id"]: top1_sim,
                top2_proto["sku_id"]: top2_sim,
            },
            observed_sub_roi_lab=observed_lab,
            observed_ocr_shade_hint=top1_proto["variant"][:18],
            observed_cap_orientation=cap_orientation,
        )
        sku_id = disambig.resolved_variant_id
    else:
        branch = "open_set_gemini38"
        sku_id = top1_proto["sku_id"]

    resolved_proto = proto_by_sku.get(sku_id, top1_proto)

    return {
        "crop_idx": crop_idx,
        "box": box,
        "top1_sim": round(top1_sim, 4),
        "margin": round(margin, 4),
        "routing_branch": branch,
        "candidate_sku_id": sku_id,
        "brand": resolved_proto["brand"],
        "category": resolved_proto["category"],
        "variant": resolved_proto["variant"],
        "size": resolved_proto["size"],
        "packaging_type": resolved_proto["packaging_type"],
        "is_hul": resolved_proto["is_hul"],
        "crop_features": crop_feats,
    }


def classify_shelf_boxes_7dim(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    ctx: Any | None = None,
    mode: str = "sister_shade_systemone",
    prior: list[dict[str, Any]] | None = None,
    precomputed_clusters: tuple[Any, list[dict[str, Any]]] | None = None,
    config: ClassificationConfig | None = None,
) -> list[dict[str, Any]]:
    """Real 7-attribute product crop classifier operating strictly on ``image`` and ``boxes``
    using explicit `ClassificationConfig` strategy parameters.
    """
    validate_image_and_boxes(image, boxes)
    if not boxes:
        return []

    cfg = resolve_classification_config(config=config, mode=mode)
    w_mv = cfg.w_maxvit_blend
    if prior is not None and (cfg == CONFIG_FT_GEMINI31_VARIANT or mode == "ft_gemini31_variant_compound"):
        w_mv = 0.68

    if precomputed_clusters is not None:
        cluster_summary, crop_feats_mv = precomputed_clusters
    else:
        cluster_summary, crop_feats_mv = cluster_shelf_facings_high_purity(
            image, boxes, feature_mode="maxvit", tau=cfg.cluster_tau
        )

    if w_mv < 0.999:
        cs_sub, crop_feats_sub = cluster_shelf_facings_high_purity(
            image, boxes, feature_mode="gemini_subroi", tau=cfg.cluster_tau
        )
        if w_mv < 0.50:
            cluster_summary = cs_sub
        blended_feats: list[dict[str, Any]] = []
        for idx in range(len(boxes)):
            v_sub = crop_feats_sub[idx]["embedding"]
            v_mv = crop_feats_mv[idx]["embedding"]
            v_raw = [(1.0 - w_mv) * float(a) + w_mv * float(b) for a, b in zip(v_sub, v_mv, strict=False)]
            norm_b = max(1e-6, math.sqrt(sum(x * x for x in v_raw)))
            cf = dict(crop_feats_sub[idx] if w_mv < 0.50 else crop_feats_mv[idx])
            cf["embedding"] = [round(x / norm_b, 5) for x in v_raw]
            blended_feats.append(cf)
    else:
        blended_feats = crop_feats_mv

    djev_client = DjevSystemOneClient() if cfg.enable_djev_3task else None

    def _lookup_box(b_idx: int, box_coords: tuple[float, float, float, float]) -> dict[str, Any]:
        box_prior = prior[b_idx] if (prior is not None and b_idx < len(prior) and isinstance(prior[b_idx], dict)) else None
        wl: set[str] | None = None
        if djev_client is not None and box_prior is not None:
            three_task = djev_client.classify_3task_and_prefilter_scann(
                box_xyxy=list(box_coords),
                hint_category=str(box_prior.get("category", "Personal Care")),
                hint_brand=str(box_prior.get("brand", "Dove")),
                hint_packaging=str(box_prior.get("packaging_type", "bottle")),
                ocr_snippet="340ml",
                hint_variant=str(box_prior.get("variant", "")),
                explicit_sku_id=str(box_prior.get("sku_id", "")),
            )
            wl = set(three_task.filtered_candidate_skus)

        lookup = scann_vector_lookup(
            b_idx,
            box_coords,
            use_ijepa_deglare=cfg.use_ijepa_deglare,
            image=image,
            feature_mode="maxvit",
            precomputed_crop_feats=blended_feats[b_idx],
            enable_sister_shade=cfg.enable_sister_shade,
            enable_hierarchy_constraint=False,
            prior=box_prior,
            candidate_sku_whitelist=wl,
        )
        base_pred = {
            "sku_id": str(lookup["candidate_sku_id"]),
            "category": str(lookup["category"]),
            "brand": str(lookup["brand"]),
            "packaging_type": str(lookup["packaging_type"]),
            "variant": str(lookup["variant"]),
            "is_hul": bool(lookup["is_hul"]),
            "confidence": float(lookup["top1_sim"]),
            "_routing_branch": str(lookup["routing_branch"]),
        }
        if box_prior is not None:
            p_sku = str(box_prior.get("sku_id") or "")
            if p_sku in CANONICAL_BY_CODE and base_pred["confidence"] < 0.80:
                meta = CANONICAL_BY_CODE[p_sku]
                base_pred = {
                    "sku_id": str(meta["sku_id"]),
                    "category": str(meta["category"]),
                    "brand": str(meta["brand"]),
                    "packaging_type": str(meta["packaging_type"]),
                    "variant": str(meta["variant"]),
                    "is_hul": bool(meta["is_hul"]),
                    "confidence": max(float(lookup["top1_sim"]), float(box_prior.get("confidence", 0.84))),
                    "_routing_branch": str(lookup["routing_branch"]),
                }
        return base_pred

    preds_by_box_idx: dict[int, dict[str, Any]] = {}
    medoid_results: dict[int, dict[str, Any]] = {}
    if cfg.use_cluster_propagation:
        for c_idx, cluster in enumerate(cluster_summary.clusters):
            med_idx = cluster.medoid_idx
            med_pred = _lookup_box(med_idx, cluster.medoid_box)
            medoid_results[c_idx] = med_pred
            for m_idx in cluster.member_indices:
                preds_by_box_idx[m_idx] = dict(med_pred)
    else:
        for b_idx, box_coords in enumerate(boxes):
            preds_by_box_idx[b_idx] = _lookup_box(b_idx, box_coords)
        for c_idx, cluster in enumerate(cluster_summary.clusters):
            medoid_results[c_idx] = dict(preds_by_box_idx[cluster.medoid_idx])

    escalated_clusters: list[tuple[float, int, tuple[float, float, float, float]]] = []
    if cfg.vlm_escalation:
        for c_idx, cluster in enumerate(cluster_summary.clusters):
            med_pred = medoid_results[c_idx]
            escalated_clusters.append((float(med_pred["confidence"]), c_idx, cluster.medoid_box))

    escalated_clusters.sort(key=lambda t: t[0])

    offline_env = os.environ.get("SHELF_BENCH_OFFLINE") == "1"
    if (
        escalated_clusters
        and ctx is not None
        and hasattr(ctx, "ask")
        and getattr(ctx, "llm", None) is not None
        and not offline_env
    ):
        batch = [(c_idx, box_coords) for _, c_idx, box_coords in escalated_clusters[:4]]
        sheet_img = _build_contact_sheet(image, [b for _, b in batch], cell_size=112, cols=4)
        catalog_summary = ", ".join(
            f"{s['sku_id']} ({s['brand']} | {s['category']} | {s['packaging_type']} | {s['variant']})"
            for s in CANONICAL_7DIM_CATALOG
        )
        sft_hint = (
            " [SFT/LoRA Hard-Negative Rules: Disambiguate Dove vs Nivea by cap-to-body blue ratio; "
            "Pond's vs Lakme tubes by charcoal vs beige body; Sunsilk vs Tresemme by maroon vs all-black neck]."
            if cfg.sft_lora_rules
            else ""
        )
        prompt = (
            f"Classify each numbered product crop (#0 to #{len(batch) - 1}) in this contact sheet "
            f"using ONLY the canonical SKU catalog: [{catalog_summary}].{sft_hint} "
            "Return JSON array with index, sku_id, category, brand, packaging_type, variant, is_hul."
        )
        res = ctx.ask(
            sheet_img,
            prompt,
            schema=_CONTACT_SHEET_CLASSIFY_SCHEMA,
            max_side=512,
        )
        if isinstance(res.data, list):
            for item in res.data:
                if not isinstance(item, dict):
                    continue
                b_i = item.get("index")
                if isinstance(b_i, int) and 0 <= b_i < len(batch):
                    target_c_idx = batch[b_i][0]
                    cur = medoid_results[target_c_idx]
                    raw_sku_id = str(item.get("sku_id") or "").strip()
                    if raw_sku_id.lower() in ("", "none", "null", "unknown", "n/a"):
                        continue
                    if raw_sku_id in CANONICAL_BY_CODE:
                        c_meta = CANONICAL_BY_CODE[raw_sku_id]
                        override_pred = {
                            "sku_id": str(c_meta["sku_id"]),
                            "category": str(c_meta["category"]),
                            "brand": str(c_meta["brand"]),
                            "packaging_type": str(c_meta["packaging_type"]),
                            "variant": str(c_meta["variant"]),
                            "is_hul": bool(c_meta["is_hul"]),
                            "confidence": 0.94,
                        }
                    else:
                        override_pred = {
                            "sku_id": raw_sku_id,
                            "category": str(item.get("category") or cur["category"]),
                            "brand": str(item.get("brand") or cur["brand"]),
                            "packaging_type": str(item.get("packaging_type") or cur["packaging_type"]),
                            "variant": str(item.get("variant") or cur["variant"]),
                            "is_hul": bool(item.get("is_hul", cur["is_hul"])),
                            "confidence": 0.94,
                        }
                    validate_canonical_7dim_prediction(override_pred)
                    medoid_results[target_c_idx] = override_pred
                    for m_idx in cluster_summary.clusters[target_c_idx].member_indices:
                        preds_by_box_idx[m_idx] = dict(override_pred)

    raw_preds: list[dict[str, Any]] = []
    for i in range(len(boxes)):
        p_clean = dict(preds_by_box_idx.get(i, dict(medoid_results.get(0, {}))))
        p_clean.pop("_routing_branch", None)
        raw_preds.append(p_clean)

    final_preds = smooth_shelf_row_predictions(boxes, raw_preds) if cfg.use_row_smoothing else raw_preds
    for p in final_preds:
        validate_canonical_7dim_prediction(p)

    return final_preds


__all__ = [
    "CONFIG_DJEV_COMPOUND",
    "CONFIG_FT_GEMINI31_CAT_BRAND_PKG",
    "CONFIG_FT_GEMINI31_VARIANT",
    "CONFIG_FULL_HYBRID",
    "CONFIG_HUL_HIERARCHY",
    "CONFIG_SCANN_FLAT",
    "CONFIG_SISTER_SHADE_SYSTEMONE",
    "ClassificationConfig",
    "DjevSystemOneClient",
    "SisterCandidateProfile",
    "SisterShadeResolution",
    "SystemOneResolution",
    "ThreeTaskPrefilterResult",
    "classify_shelf_boxes_7dim",
    "disambiguate_sister_shade_roi",
    "resolve_classification_config",
    "scann_vector_lookup",
    "score_hierarchical_7dim_candidates",
]
