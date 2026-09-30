"""Stage 3.5: Adjacent crop clustering and deduplication before VLM inference.

Registers ``clusterer`` stage implementations:
  - ``connected_component_adjacency`` (default): Shelf-row spatial and visual cosine connected-component clustering.
  - ``maxvit_agglomerative``: Multi-axis attention features with agglomerative cosine clustering.
  - ``none``: No crop clustering (evaluates every crop individually).
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from core import catalog as core_catalog
from core import clustering as core_clustering
from stages.registry import StageSpec, register_stage


def run_crop_clustering(
    boxes: list[tuple[float, float, float, float]],
    mode: str = "connected_component_adjacency",
    image: Image.Image | None = None,
) -> dict[str, Any]:
    """Group visually identical adjacent crops using real pixel/neural features and return cluster summary."""
    if mode not in ("connected_component_adjacency", "maxvit_agglomerative", "none"):
        raise ValueError(f"Unsupported clusterer mode: {mode!r}")

    num_boxes = len(boxes)
    if num_boxes == 0:
        return {
            "mode": mode,
            "input_crops": 0,
            "medoid_calls": 0,
            "compression_ratio": 1.0,
            "cluster_purity": 1.0,
            "cluster_summary": None,
            "crop_feats": [],
        }

    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError("stage3_5_clustering requires a valid non-empty PIL.Image.Image")

    feat_mode = "maxvit" if mode == "maxvit_agglomerative" else "gemini_subroi"
    tau_val = 0.9999 if mode == "none" else (0.93 if mode == "maxvit_agglomerative" else 0.94)
    summary, crop_feats = core_clustering.cluster_shelf_facings_high_purity(
        image, boxes, feature_mode=feat_mode, tau=tau_val
    )
    catalog = core_catalog.load_dynamic_hul_catalog_index()
    if mode == "none":
        return {
            "mode": "none",
            "input_crops": num_boxes,
            "medoid_calls": num_boxes,
            "compression_ratio": 1.0,
            "cluster_purity": 1.0,
            "catalog_size": len(catalog),
            "cluster_summary": summary,
            "crop_feats": crop_feats,
        }

    return {
        "mode": mode,
        "input_crops": summary.total_facings,
        "medoid_calls": summary.num_clusters,
        "compression_ratio": summary.compression_ratio,
        "cluster_purity": summary.estimated_node_purity,
        "catalog_size": len(catalog),
        "cluster_summary": summary,
        "crop_feats": crop_feats,
    }


register_stage(
    StageSpec(
        stage_group="clusterer",
        name="connected_component_adjacency",
        title="Connected-Component Row Adjacency and Cosine Clustering (Default)",
        description="Groups adjacent identical products on the same shelf row before VLM classification.",
        f2_delta=0.0,
        latency_delta_s=0.0,
        cost_delta_inr=0.0,
        default=True,
        fn=run_crop_clustering,
    )
)

register_stage(
    StageSpec(
        stage_group="clusterer",
        name="maxvit_agglomerative",
        title="MaxViT Feature Agglomerative Clustering",
        description="Clusters crops across shelf bays using MaxViT visual embeddings and cosine distance thresholding.",
        f2_delta=0.0,
        latency_delta_s=0.020,
        cost_delta_inr=0.001,
        default=False,
        fn=lambda boxes, image=None: run_crop_clustering(boxes, mode="maxvit_agglomerative", image=image),
    )
)

register_stage(
    StageSpec(
        stage_group="clusterer",
        name="none",
        title="No Crop Clustering (Classify Every Crop Individually)",
        description="Skips crop deduplication and invokes the classifier on every detected bounding box.",
        f2_delta=0.0,
        latency_delta_s=0.480,
        cost_delta_inr=0.032,
        default=False,
        fn=lambda boxes, image=None: run_crop_clustering(boxes, mode="none", image=image),
    )
)
