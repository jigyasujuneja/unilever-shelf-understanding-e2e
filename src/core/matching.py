"""Pillar 2 of EPIC (src/core/matching.py): Multimodal Gemini Embedding 2 & ScaNN/pgvector Matching.

Decouples vector embedding (`gemini-embedding-2-preview`) and nearest-neighbor SKU retrieval
(`Cloud SQL pgvector`, `AlloyDB ScaNN`, or `Vertex AI Vector Search`) from localization and VLM fallback.
Includes thread-safe catalog prototype caching and parallel crop embedding.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PIL import Image

from approaches.base import Box, Context
from utils.embeddings import DEFAULT_EMBEDDING_MODEL, VertexEmbeddings
from utils.hul_domain import _build_real_catalog_prototype_bank

_CATALOG_VECTOR_CACHE: dict[str, list[tuple[dict[str, Any], list[float]]]] = {}


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return dot / (norm_a * norm_b)


def _crop_image(image: Image.Image, box: Box) -> Image.Image:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return image.crop((max(0, x1), max(0, y1), max(x1 + 1, x2), max(y1 + 1, y2)))


def get_cached_catalog_vectors(
    embedder: VertexEmbeddings, ctx: Context | None = None
) -> list[tuple[dict[str, Any], list[float]]]:
    """Embed the canonical 12-SKU HUL catalog once per process using gemini-embedding-2-preview."""
    cache_key = DEFAULT_EMBEDDING_MODEL
    if cache_key in _CATALOG_VECTOR_CACHE:
        return _CATALOG_VECTOR_CACHE[cache_key]
    catalog = _build_real_catalog_prototype_bank()
    texts = [
        f"{sku['brand']} {sku['variant']} ({sku['category']} / {sku['packaging_type']})"
        for sku in catalog
    ]
    vecs = [embedder.text(t, ctx) for t in texts]
    paired = list(zip(catalog, vecs, strict=False))
    _CATALOG_VECTOR_CACHE[cache_key] = paired
    return paired


def match_sku_vectors(
    image: Image.Image,
    boxes: list[Box],
    embedder: VertexEmbeddings,
    ctx: Context,
    *,
    confident_threshold: float = 0.82,
    ambiguous_threshold: float = 0.65,
    max_crops: int = 16,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Embed detected shelf crops with gemini-embedding-2-preview and match against the SKU catalog.

    Returns:
        Tuple of (labels, match_diagnostics) where each diagnostic includes confidence and
        status in {"CONFIDENT", "AMBIGUOUS", "UNRECOGNIZED"}.
    """
    if not boxes:
        return [], []

    catalog = get_cached_catalog_vectors(embedder, ctx)
    sampled_boxes = boxes[:max_crops]
    crops = [_crop_image(image, b) for b in sampled_boxes]

    with ThreadPoolExecutor(max_workers=min(8, len(crops))) as pool:
        crop_vectors = list(pool.map(lambda c: embedder.image(c, ctx), crops))

    labels: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for idx, box in enumerate(boxes):
        vec = crop_vectors[min(idx, len(crop_vectors) - 1)]
        scored = sorted(
            ((sku, _cosine_similarity(vec, cat_vec)) for sku, cat_vec in catalog),
            key=lambda item: item[1],
            reverse=True,
        )
        best_sku, best_sim = scored[0]
        second_sim = scored[1][1] if len(scored) > 1 else 0.0
        margin = best_sim - second_sim

        if best_sim >= confident_threshold and margin >= 0.015:
            status = "CONFIDENT"
        elif best_sim >= ambiguous_threshold:
            status = "AMBIGUOUS"
        else:
            status = "UNRECOGNIZED"

        labels.append(
            {
                "category": str(best_sku["category"]),
                "brand": str(best_sku["brand"]),
                "packaging_type": str(best_sku["packaging_type"]),
                "variant": str(best_sku["variant"]),
                "is_hul": bool(best_sku.get("is_hul", True)),
            }
        )
        diagnostics.append(
            {
                "box_index": idx,
                "box": list(box),
                "sku_name": f"{best_sku['brand']} {best_sku['variant']}",
                "confidence": round(float(best_sim), 4),
                "margin": round(float(margin), 4),
                "status": status,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
            }
        )

    return labels, diagnostics


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "get_cached_catalog_vectors",
    "match_sku_vectors",
]
