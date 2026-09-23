"""Stage 2: Cropped Bounding-Box Visual Embedding & Metric Learning (`multimodalembedding@001` 1408-D).

Extracts physical pixel crops for every class-agnostic 'product' bounding box via
`ctx.crop_facing_images()` and passes each image crop through Google Cloud Vertex AI's
Vision-Transformer Contrastive Metric Learning model (`multimodalembedding@001`, or a
custom Vertex AI Endpoint hosting an ArcFace / Triplet-Loss ResNet/ViT backbone) to produce
a 1408-dimensional visual vector representation (`imageEmbedding`) for each shelf facing.
"""

from __future__ import annotations

import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import requests

from shelf_benchmark.approaches.base import CommonLayerContext
from shelf_benchmark.auth import get_gcp_credentials

logger = logging.getLogger(__name__)

# Vertex AI Multimodal Embedding (`multimodalembedding@001`) pricing: $0.0001 per image
VERTEX_IMAGE_EMBEDDING_COST_PER_CROP_USD = 0.0001

# Per-facing embedding outcome. This is deliberately not a bare empty vector:
# an all-zero vector is indistinguishable from a real product that simply did not
# match the catalog, so a service outage used to be reported as a model-quality
# result ("BELOW_MATCH_THRESHOLD"). Stage 3 reads this to emit EMBEDDING_FAILED.
EMBEDDING_STATUS_OK = "OK"
EMBEDDING_STATUS_CROP_UNAVAILABLE = "CROP_UNAVAILABLE"
EMBEDDING_STATUS_SERVICE_ERROR = "SERVICE_ERROR"


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding service did not return a usable vector.

    Callers must either propagate this or record an explicit non-OK status. They
    must never substitute an empty/zero vector and let it flow into scoring.
    """


def _embed_single_crop_bytes(
    crop_bytes: bytes,
    project_id: str,
    location: str,
    auth_token: str,
) -> List[float]:
    """Calls Vertex AI `multimodalembedding@001` on raw cropped product bytes to get a 1408-D visual vector.

    Raises:
        EmbeddingUnavailable: if the service returned a non-200, or a 200 with no
            usable `imageEmbedding`. Never returns an empty list.
    """
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
    if resp.status_code != 200:
        raise EmbeddingUnavailable(
            f"multimodalembedding@001 image predict -> HTTP {resp.status_code}: {resp.text[:200]}"
        )
    preds = resp.json().get("predictions") or []
    vector = preds[0].get("imageEmbedding") if preds else None
    if not vector:
        raise EmbeddingUnavailable(
            "multimodalembedding@001 image predict -> HTTP 200 with no imageEmbedding"
        )
    return vector


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
        if r.status_code != 200:
            raise EmbeddingUnavailable(
                f"multimodalembedding@001 text predict -> HTTP {r.status_code}: {r.text[:200]}"
            )
        preds = r.json().get("predictions") or []
        vector = preds[0].get("textEmbedding") if preds else None
        if not vector:
            raise EmbeddingUnavailable(
                f"multimodalembedding@001 text predict -> HTTP 200 with no textEmbedding for {txt!r}"
            )
        return vector

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

    if ctx.config.offline.enabled:
        enriched_facings = []
        for facing in detected_facings:
            p_idx = int(facing.get("product_index", 1))
            _, c_path = crop_map.get(p_idx, (b"", ""))
            vec_1408 = [1.0] + [0.0] * 1407
            enriched_facings.append(
                {
                    **facing,
                    "crop_image_path": c_path,
                    "visual_embedding_model": "multimodalembedding@001",
                    "visual_embedding_dim": len(vec_1408),
                    "visual_embedding_vector": vec_1408,
                    "visual_embedding_status": EMBEDDING_STATUS_OK,
                }
            )
        stage2_cost_usd = round(
            len(enriched_facings) * VERTEX_IMAGE_EMBEDDING_COST_PER_CROP_USD, 8
        )
        return enriched_facings, montage_path, stage2_cost_usd

    creds = get_gcp_credentials(project_id=ctx.config.gcp.project_id)
    project_id = ctx.config.gcp.project_id

    def _embed_facing(facing: Dict[str, Any]) -> Dict[str, Any]:
        p_idx = int(facing.get("product_index", 1))
        c_bytes, c_path = crop_map.get(p_idx, (b"", ""))
        vec_1408: List[float] = []
        if not c_bytes:
            status = EMBEDDING_STATUS_CROP_UNAVAILABLE
            logger.warning(
                "Stage 2: no crop bytes for product_index=%s on %s; "
                "facing will be reported as %s, not as a failed match.",
                p_idx,
                shelf_image_uri,
                status,
            )
        else:
            try:
                vec_1408 = _embed_single_crop_bytes(
                    crop_bytes=c_bytes,
                    project_id=project_id,
                    location=embedding_location,
                    auth_token=creds.token,
                )
                status = EMBEDDING_STATUS_OK
            except (EmbeddingUnavailable, requests.RequestException) as exc:
                # Deliberately not re-raised: one bad crop should not void the whole
                # shelf. But the status is carried so stage 3 reports EMBEDDING_FAILED
                # rather than scoring this as a genuine below-threshold non-match.
                status = EMBEDDING_STATUS_SERVICE_ERROR
                logger.warning(
                    "Stage 2: embedding failed for product_index=%s on %s: %s",
                    p_idx,
                    shelf_image_uri,
                    exc,
                )
        return {
            **facing,
            "crop_image_path": c_path,
            "visual_embedding_model": "multimodalembedding@001",
            "visual_embedding_dim": len(vec_1408),
            "visual_embedding_vector": vec_1408,
            "visual_embedding_status": status,
        }

    with ThreadPoolExecutor(max_workers=6) as executor:
        enriched_facings = list(executor.map(_embed_facing, detected_facings))

    # Bill only the crops that actually produced a vector. Charging for failed
    # calls inflated the reported cost of an outage.
    billable = sum(
        1 for f in enriched_facings if f.get("visual_embedding_status") == EMBEDDING_STATUS_OK
    )
    failed = len(enriched_facings) - billable
    if failed:
        logger.warning(
            "Stage 2: %d of %d facings on %s have no usable embedding.",
            failed,
            len(enriched_facings),
            shelf_image_uri,
        )
    stage2_cost_usd = round(billable * VERTEX_IMAGE_EMBEDDING_COST_PER_CROP_USD, 8)
    return enriched_facings, montage_path, stage2_cost_usd
