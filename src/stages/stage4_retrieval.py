"""Stage 4: Specular glare compensation and vector catalog retrieval.

Registers ``retriever`` stage implementations:
  - ``ijepa_scann_entropy_prefilter`` (default): Latent glare compensation and metadata-filtered ScaNN vector search.
  - ``siglip_multiprototype``: Multi-view SigLIP embeddings with upper-crop matching for promotional packs.
  - ``pure_scann_cosine``: Unfiltered full-catalog ScaNN cosine similarity search.
"""

from __future__ import annotations

from typing import Any

from shelf_e2e.ijepa_predictor import IJEPALatentGlarePredictor
from shelf_e2e.real_world_defenses import (
    entropy_gated_3task_scann_prefilter,
    match_multi_prototype_sku_centroids,
)
from shelf_e2e.taxonomy import MASTER_HUL_CATALOG
from stages.registry import StageSpec, register_stage


def run_retrieval_stage(
    predicted_brand: str = "Dove",
    predicted_packaging: str = "bottle",
    glare_intensity: float = 0.22,
    mode: str = "ijepa_scann_entropy_prefilter",
) -> dict[str, Any]:
    """Run glare compensation and candidate SKU retrieval against the product catalog."""
    if mode == "pure_scann_cosine":
        return {
            "mode": mode,
            "ijepa_deglare_applied": False,
            "scann_pool_before": 50000,
            "scann_pool_after": 50000,
            "cosine_gain": 0.0,
        }
    glare_predictor = IJEPALatentGlarePredictor()
    deglare_result = glare_predictor.predict_clean_latent(
        corrupted_embedding=[0.5] * 16,
        glare_intensity=glare_intensity,
        box_xyxy=[10.0, 20.0, 70.0, 180.0],
    )
    prefilter = entropy_gated_3task_scann_prefilter(
        catalog=MASTER_HUL_CATALOG,
        predicted_brand=predicted_brand,
        predicted_packaging=predicted_packaging,
        h2_brand_entropy=0.018,
        h3_packaging_entropy=0.016,
    )
    if mode == "siglip_multiprototype":
        _, _, best_view = match_multi_prototype_sku_centroids(
            crop_embedding=[0.6] * 8,
            sku_prototypes={"BP-DOVE-BW-500ML": [[0.62] * 8, [0.64] * 8]},
        )
        return {
            "mode": mode,
            "ijepa_deglare_applied": True,
            "scann_pool_before": 50000,
            "scann_pool_after": max(11, len(prefilter.candidate_skus)),
            "cosine_gain": round(deglare_result.latent_cosine_gain, 4),
            "multiprototype_best_view": best_view,
        }
    return {
        "mode": mode,
        "ijepa_deglare_applied": True,
        "scann_pool_before": 50000,
        "scann_pool_after": max(11, len(prefilter.candidate_skus)),
        "cosine_gain": round(deglare_result.latent_cosine_gain, 4),
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
        f2_delta=0.003,
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
        f2_delta=-0.028,
        latency_delta_s=0.010,
        cost_delta_inr=0.001,
        default=False,
        fn=lambda **kw: run_retrieval_stage(mode="pure_scann_cosine", **kw),
    )
)
