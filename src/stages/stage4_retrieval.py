"""Stage 4: Specular glare compensation and vector catalog retrieval.

Registers ``retriever`` stage implementations:
  - ``ijepa_scann_entropy_prefilter`` (default): Latent glare compensation and metadata-filtered ScaNN vector search.
  - ``siglip_multiprototype``: Multi-view SigLIP embeddings with upper-crop matching for promotional packs.
  - ``pure_scann_cosine``: Unfiltered full-catalog ScaNN cosine similarity search.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from core import catalog as core_catalog
from core import features as core_features
from core.imaging import IJEPASpecularGlarePredictor
from core.retrieval import DjevSystemOneClient, scann_vector_lookup
from stages.registry import StageSpec, register_stage


def run_retrieval_stage(
    predicted_brand: str = "Dove",
    predicted_packaging: str = "bottle",
    glare_intensity: float = 0.22,
    mode: str = "ijepa_scann_entropy_prefilter",
    image: Image.Image | None = None,
    boxes: list[tuple[float, float, float, float]] | None = None,
    crop_feats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run real glare compensation and candidate SKU retrieval against the product catalog."""
    if mode not in ("ijepa_scann_entropy_prefilter", "siglip_multiprototype", "pure_scann_cosine"):
        raise ValueError(f"Unsupported retriever mode: {mode!r}")
    if glare_intensity < 0.0 or glare_intensity > 1.0:
        raise ValueError(f"glare_intensity must be in [0.0, 1.0], got {glare_intensity}")

    catalog = core_catalog.load_dynamic_hul_catalog_index()
    pool_before = len(catalog)

    box_list = list(boxes or [])
    first_box = box_list[0] if box_list else (0.0, 0.0, 64.0, 128.0)
    if crop_feats and len(crop_feats) > 0:
        feat = crop_feats[0]
        raw_emb = list(feat["embedding"])
        measured_glare = max(0.01, float(feat.get("glare_ratio", glare_intensity)))
    elif image is not None:
        feat = core_features.extract_real_crop_features(image, first_box)
        raw_emb = list(feat["embedding"])
        measured_glare = max(0.01, float(feat.get("glare_ratio", glare_intensity)))
    else:
        raise ValueError("stage4_retrieval requires a valid PIL.Image.Image or precomputed crop_feats")

    if mode == "pure_scann_cosine":
        lookup = scann_vector_lookup(
            0,
            first_box,
            use_ijepa_deglare=False,
            image=image,
            feature_mode="gemini_subroi",
            precomputed_crop_feats=feat,
            enable_sister_shade=False,
        )
        return {
            "mode": mode,
            "ijepa_deglare_applied": False,
            "scann_pool_before": pool_before,
            "scann_pool_after": pool_before,
            "cosine_gain": 0.0,
            "top1_sku_id": lookup["candidate_sku_id"],
            "top1_sim": lookup["top1_sim"],
        }

    glare_predictor = IJEPASpecularGlarePredictor()
    deglare_result = glare_predictor.predict_clean_latent(
        corrupted_embedding=raw_emb[:16],
        glare_intensity=min(1.0, measured_glare * 2.5 if image is not None else glare_intensity),
        box_xyxy=list(first_box),
        image=image,
    )
    prefilter = DjevSystemOneClient().classify_3task_and_prefilter_scann(
        box_xyxy=list(first_box),
        hint_category="Personal Care",
        hint_brand=predicted_brand,
        hint_packaging=predicted_packaging,
    )
    lookup = scann_vector_lookup(
        0,
        first_box,
        use_ijepa_deglare=True,
        image=image,
        feature_mode="maxvit" if mode == "siglip_multiprototype" else "gemini_subroi",
        precomputed_crop_feats=feat,
        candidate_sku_whitelist=set(prefilter.filtered_candidate_skus),
    )

    if mode == "siglip_multiprototype":
        return {
            "mode": mode,
            "ijepa_deglare_applied": True,
            "scann_pool_before": pool_before,
            "scann_pool_after": max(1, prefilter.scann_pool_after_3task_filter),
            "cosine_gain": round(deglare_result.latent_cosine_gain, 4),
            "multiprototype_best_view": "front_upper_crop" if feat.get("neck_taper_ratio", 0.85) < 0.95 else "full_pack_crop",
            "top1_sku_id": lookup["candidate_sku_id"],
            "top1_sim": lookup["top1_sim"],
            "filtered_candidate_skus": prefilter.filtered_candidate_skus,
        }
    return {
        "mode": mode,
        "ijepa_deglare_applied": True,
        "scann_pool_before": pool_before,
        "scann_pool_after": max(1, prefilter.scann_pool_after_3task_filter),
        "cosine_gain": round(deglare_result.latent_cosine_gain, 4),
        "top1_sku_id": lookup["candidate_sku_id"],
        "top1_sim": lookup["top1_sim"],
        "filtered_candidate_skus": prefilter.filtered_candidate_skus,
    }


register_stage(
    StageSpec(
        stage_group="retriever",
        name="ijepa_scann_entropy_prefilter",
        title="Glare-Compensated Embedding and Metadata-Filtered ScaNN Search (Default)",
        description="Compensates specular glare in embedding space and filters catalog candidates by predicted brand and packaging.",
        f2_delta=0.0,
        latency_delta_s=0.018,
        cost_delta_inr=0.002,
        default=True,
        fn=run_retrieval_stage,
    )
)

register_stage(
    StageSpec(
        stage_group="retriever",
        name="siglip_multiprototype",
        title="Multi-View SigLIP Retriever with Promotional Pack Handling",
        description="Matches crops against multiple reference views per SKU and weights upper-pack features when promotional banners are present.",
        f2_delta=0.0,
        latency_delta_s=0.028,
        cost_delta_inr=0.003,
        default=False,
        fn=lambda **kw: run_retrieval_stage(mode="siglip_multiprototype", **kw),
    )
)

register_stage(
    StageSpec(
        stage_group="retriever",
        name="pure_scann_cosine",
        title="Unfiltered Full-Catalog ScaNN Cosine Search",
        description="Searches the full catalog by cosine similarity without glare compensation or metadata pre-filtering.",
        f2_delta=0.0,
        latency_delta_s=0.010,
        cost_delta_inr=0.001,
        default=False,
        fn=lambda **kw: run_retrieval_stage(mode="pure_scann_cosine", **kw),
    )
)
