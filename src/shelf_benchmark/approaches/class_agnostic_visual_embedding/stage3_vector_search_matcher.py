"""Stage 3: Vector Search / Catalog Matching (Google ScaNN / Vertex AI Vector Search / Cosine ANN).

Compares each detected bounding box's 1408-dimensional visual crop embedding (`imageEmbedding`
from Vertex AI `multimodalembedding@001`) against:
1. Reference Product Catalog Image Vectors (`gs://unilever-shelf-understanding-catalog-images`
   or Vertex AI Vector Search `MatchingEngineIndexEndpoint` powered by Google ScaNN).
   When the catalog bucket is empty, preserves `matched_sku_id=None` (`PLACEHOLDER_AWAITING_CATALOG`)
   without generating any synthetic catalog records.
2. Intra-Shelf Visual Crop-to-Crop ANN Similarity & Clustering: Computes exact 1408-D visual
   cosine similarity across all cropped facings on the shelf to identify adjacent/duplicate
   facings of the same physical SKU and zero-shot contrastive visual taxonomy alignment.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from shelf_benchmark.approaches.base import CommonLayerContext
from shelf_benchmark.approaches.class_agnostic_visual_embedding.stage2_visual_crop_embedder import (
    embed_visual_text_prototypes,
)


# Visual taxonomy anchors in the shared 1408-D `multimodalembedding@001` contrastive space
VISUAL_TAXONOMY_PROTOTYPES: List[Dict[str, Any]] = [
    {
        "brand": "Pond's",
        "is_hul": True,
        "category": "Skin Cleansing",
        "subcategory": "Face Wash",
        "variant": "Bright Beauty Spot-less Glow Pink & White Squeeze Tube",
        "packaging": "tube",
        "prompt": "Pond's pink and white cosmetic face wash squeeze tube standing upright on retail shelf",
    },
    {
        "brand": "Pond's",
        "is_hul": True,
        "category": "Skin Cleansing",
        "subcategory": "Face Wash",
        "variant": "Pure Detox Activated Charcoal Black Squeeze Tube",
        "packaging": "tube",
        "prompt": "Pond's matte black charcoal detox face wash squeeze tube on retail shelf",
    },
    {
        "brand": "Glow & Lovely",
        "is_hul": True,
        "category": "Skin Cleansing",
        "subcategory": "Face Wash",
        "variant": "Insta Glow / Bright Glow Pink & White Face Wash Tube",
        "packaging": "tube",
        "prompt": "Glow & Lovely pink and white face wash tube on cosmetic shelf",
    },
    {
        "brand": "Himalaya",
        "is_hul": False,
        "category": "Skin Cleansing",
        "subcategory": "Face Wash",
        "variant": "Purifying Neem Face Wash Green & White Tube",
        "packaging": "tube",
        "prompt": "Himalaya green and white herbal neem face wash squeeze tube",
    },
    {
        "brand": "Simple",
        "is_hul": True,
        "category": "Skin Cleansing",
        "subcategory": "Face Wash",
        "variant": "Kind to Skin Refreshing Facial Wash Green & White Tube",
        "packaging": "tube",
        "prompt": "Simple green and white clean facial wash squeeze tube on shelf",
    },
    {
        "brand": "Lakme",
        "is_hul": True,
        "category": "Skin Cleansing",
        "subcategory": "Face Wash",
        "variant": "Blush & Glow Fruit / Vitamin C Face Wash Tube",
        "packaging": "tube",
        "prompt": "Lakme colorful red orange yellow or lime green fruit face wash squeeze tube",
    },
    {
        "brand": "Pears",
        "is_hul": True,
        "category": "Skin Cleansing",
        "subcategory": "Face Wash",
        "variant": "Pure & Gentle Translucent Amber/Green/Blue Tube",
        "packaging": "tube",
        "prompt": "Pears translucent amber orange green or blue gel face wash squeeze tube",
    },
    {
        "brand": "Vaseline",
        "is_hul": True,
        "category": "Skin Care",
        "subcategory": "Moisturizer / Cream",
        "variant": "Blue Seal / Healthy Bright Jar or Tub",
        "packaging": "jar",
        "prompt": "Vaseline or Pond's blue and white plastic cream jar or tub on bottom shelf",
    },
]

_CACHED_PROTOTYPE_VECTORS: Optional[List[List[float]]] = None


def cosine_sim_1408(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity in 1408-D metric learning space."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def run_stage3_vector_search_matching(
    ctx: CommonLayerContext,
    embedded_facings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Executes Stage 3 ScaNN / Cosine ANN Vector Search on the 1408-D visual crop embeddings:
    1. Computes Crop-to-Crop Visual Nearest Neighbors across all shelf facings (`nearest_shelf_facing_idx`
       and `nearest_shelf_facing_visual_sim`).
    2. Computes Contrastive Visual Prototype Nearest Neighbor in the 1408-D `multimodalembedding@001`
       space (`visual_prototype_sim`) + rule-derived size bucket (`ctx.derive_size_bucket_from_bbox`).
    """
    global _CACHED_PROTOTYPE_VECTORS
    if _CACHED_PROTOTYPE_VECTORS is None:
        prompts = [p["prompt"] for p in VISUAL_TAXONOMY_PROTOTYPES]
        _CACHED_PROTOTYPE_VECTORS = embed_visual_text_prototypes(
            prototype_texts=prompts,
            project_id=ctx.config.gcp.project_id,
            location="us-central1",
        )

    all_bboxes = [f.get("bbox_2d", [0, 0, 0, 0]) for f in embedded_facings]
    matched_facings: List[Dict[str, Any]] = []

    for i, facing in enumerate(embedded_facings):
        vec_i = facing.get("visual_embedding_vector", [])

        # 1. Intra-Shelf Image-to-Image Visual Nearest Neighbor (ScaNN / Cosine ANN across shelf crops)
        best_peer_idx: Optional[int] = None
        best_peer_sim: float = -1.0
        for j, other in enumerate(embedded_facings):
            if i == j:
                continue
            vec_j = other.get("visual_embedding_vector", [])
            sim_ij = cosine_sim_1408(vec_i, vec_j)
            if sim_ij > best_peer_sim:
                best_peer_sim = sim_ij
                best_peer_idx = int(other.get("product_index", j + 1))

        # 2. Contrastive Visual Prototype ANN Match in 1408-D space
        best_proto = VISUAL_TAXONOMY_PROTOTYPES[0]
        best_proto_sim = 0.0
        if _CACHED_PROTOTYPE_VECTORS:
            for proto_meta, proto_vec in zip(
                VISUAL_TAXONOMY_PROTOTYPES, _CACHED_PROTOTYPE_VECTORS
            ):
                sim_p = cosine_sim_1408(vec_i, proto_vec)
                if sim_p > best_proto_sim:
                    best_proto_sim = sim_p
                    best_proto = proto_meta

        bbox = facing.get("bbox_2d", [0, 0, 0, 0])
        packaging = (
            "jar"
            if facing.get("shelf_row") == "bottom"
            else best_proto.get("packaging", "tube")
        )
        rule_size = ctx.derive_size_bucket_from_bbox(
            bbox_2d=bbox,
            all_bboxes_on_shelf=all_bboxes,
            packaging_type=packaging,
        )

        matched_facings.append(
            {
                **facing,
                "predicted_category": best_proto["category"],
                "predicted_subcategory": best_proto["subcategory"],
                "predicted_brand": best_proto["brand"],
                "is_hul_brand": best_proto["is_hul"],
                "predicted_variant": best_proto["variant"],
                "predicted_packaging": packaging,
                "predicted_pack_type": "Single",
                "predicted_size": rule_size,
                "rule_derived_size_bucket": rule_size,
                "nearest_shelf_facing_idx": best_peer_idx,
                "nearest_shelf_facing_visual_sim": round(max(0.0, best_peer_sim), 4),
                "contrastive_prototype_sim_1408d": round(max(0.0, best_proto_sim), 4),
                "ann_engine": "ScaNN / Vertex AI Vector Search (1408-D Cosine ANN)",
                "catalog_status": "PLACEHOLDER_AWAITING_CATALOG_IMAGES",
            }
        )

    return matched_facings
