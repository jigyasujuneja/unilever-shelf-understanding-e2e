"""Stage 2: Cropped Bounding-Box Visual Embedding & Metric Learning (`multimodalembedding@001` 1408-D).

Extracts physical pixel crops for every class-agnostic 'product' bounding box via
`ctx.crop_facing_images()` and passes each image crop through Google Cloud Vertex AI's
Vision-Transformer Contrastive Metric Learning model (`multimodalembedding@001`, or a
custom Vertex AI Endpoint hosting an ArcFace / Triplet-Loss ResNet/ViT backbone) to produce
a 1408-dimensional visual vector representation (`imageEmbedding`) for each shelf facing.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from shelf_benchmark.approaches.base import CommonLayerContext
from shelf_benchmark.auth import get_gcp_credentials


# Vertex AI Multimodal Embedding (`multimodalembedding@001`) pricing: $0.0001 per image
VERTEX_IMAGE_EMBEDDING_COST_PER_CROP_USD = 0.0001


def _embed_single_crop_bytes(
    crop_bytes: bytes,
    project_id: str,
    location: str,
    auth_token: str,
) -> List[float]:
    """Calls Vertex AI `multimodalembedding@001` on raw cropped product bytes to get a 1408-D visual vector."""
    img_b64 = base64.b64encode(crop_bytes).decode("utf-8")
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{location}/publishers/google/models/multimodalembedding@001:predict"
    )
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {auth_token}",
            "x-goog-user-project": project_id,
            "Content-Type": "application/json",
        },
        json={"instances": [{"image": {"bytesBase64Encoded": img_b64}}]},
        timeout=30,
    )
    if resp.status_code == 200:
        preds = resp.json().get("predictions", [])
        if preds:
            return preds[0].get("imageEmbedding", [])
    return []


def embed_visual_text_prototypes(
    prototype_texts: List[str],
    project_id: str,
    location: str = "us-central1",
) -> List[List[float]]:
    """Projects reference catalog prototype strings into the same 1408-D `multimodalembedding@001`
    contrastive visual-semantic vector space.
    """
    creds = get_gcp_credentials(project_id=project_id)
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{location}/publishers/google/models/multimodalembedding@001:predict"
    )
    vectors: List[List[float]] = []

    def _embed_one_text(txt: str) -> List[float]:
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {creds.token}",
                "x-goog-user-project": project_id,
                "Content-Type": "application/json",
            },
            json={"instances": [{"text": txt}]},
            timeout=30,
        )
        if r.status_code == 200:
            preds = r.json().get("predictions", [])
            if preds:
                return preds[0].get("textEmbedding", [])
        return []

    with ThreadPoolExecutor(max_workers=8) as pool:
        vectors = list(pool.map(_embed_one_text, prototype_texts))
    return vectors


def run_stage2_visual_crop_embedding(
    ctx: CommonLayerContext,
    shelf_image_uri: str,
    detected_facings: List[Dict[str, Any]],
    model_tag: str,
    embedding_location: str = "us-central1",
) -> Tuple[List[Dict[str, Any]], Optional[str], float]:
    """Physically crops each detected bounding box via `ctx.crop_facing_images()` and
    passes each crop through Vertex AI `multimodalembedding@001` (1408-D visual vector).

    Returns:
        (enriched_facings_with_1408d_vectors, montage_path, stage2_embedding_cost_usd)
    """
    crop_tuples, _, montage_path = ctx.crop_facing_images(
        shelf_image_uri=shelf_image_uri,
        facings=detected_facings,
        model_tag=model_tag,
    )

    crop_map: Dict[int, Tuple[bytes, str]] = {
        idx: (c_bytes, c_path) for idx, c_bytes, c_path in crop_tuples
    }

    creds = get_gcp_credentials(project_id=ctx.config.gcp.project_id)
    project_id = ctx.config.gcp.project_id

    def _embed_facing(facing: Dict[str, Any]) -> Dict[str, Any]:
        p_idx = int(facing.get("product_index", 1))
        c_bytes, c_path = crop_map.get(p_idx, (b"", ""))
        vec_1408: List[float] = []
        if c_bytes:
            vec_1408 = _embed_single_crop_bytes(
                crop_bytes=c_bytes,
                project_id=project_id,
                location=embedding_location,
                auth_token=creds.token,
            )
        return {
            **facing,
            "crop_image_path": c_path,
            "visual_embedding_model": "multimodalembedding@001",
            "visual_embedding_dim": len(vec_1408),
            "visual_embedding_vector": vec_1408,
        }

    with ThreadPoolExecutor(max_workers=6) as executor:
        enriched_facings = list(executor.map(_embed_facing, detected_facings))

    stage2_cost_usd = round(
        len(enriched_facings) * VERTEX_IMAGE_EMBEDDING_COST_PER_CROP_USD, 8
    )
    return enriched_facings, montage_path, stage2_cost_usd
