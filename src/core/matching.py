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
from core.catalog import _build_real_catalog_prototype_bank
from core.features import _delta_e_cie76, extract_gemini_subroi_embedding
from utils.embeddings import DEFAULT_EMBEDDING_MODEL, VertexEmbeddings

_CATALOG_VECTOR_CACHE: dict[str, list[tuple[dict[str, Any], list[float]]]] = {}


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return dot / (norm_a * norm_b)


def _crop_image(image: Image.Image, box: Box) -> Image.Image:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return image.crop((max(0, x1), max(0, y1), max(x1 + 1, x2), max(y1 + 1, y2)))


def _get_real_catalog_reference_crop(sku: dict[str, Any]) -> Image.Image:
    """Return the real cropped reference image for `sku` from `core.catalog` (zero Image.new synthetics)."""
    ref = sku.get("reference_crop")
    if isinstance(ref, Image.Image):
        return ref
    for p in _build_real_catalog_prototype_bank():
        if p["sku_id"] == sku.get("sku_id") and isinstance(p.get("reference_crop"), Image.Image):
            return p["reference_crop"]
    raise RuntimeError(f"Missing real reference_crop for SKU {sku.get('sku_id')!r}")


def get_cached_catalog_vectors(
    embedder: VertexEmbeddings, ctx: Context | None = None
) -> list[tuple[dict[str, Any], list[float]]]:
    """Embed the canonical 12-SKU HUL catalog once per process using multimodal gemini-embedding-2-preview."""
    cache_key = f"{getattr(embedder, 'model', DEFAULT_EMBEDDING_MODEL)}:multimodal-v2"
    if cache_key in _CATALOG_VECTOR_CACHE:
        return _CATALOG_VECTOR_CACHE[cache_key]
    catalog = _build_real_catalog_prototype_bank()
    vecs: list[list[float]] = []
    for sku in catalog:
        text_desc = (
            f"{sku['brand']} {sku['variant']} ({sku['category']} | {sku['packaging_type']} | {sku.get('size', '')})"
        )
        if hasattr(embedder, "multimodal"):
            ref_img = _get_real_catalog_reference_crop(sku)
            vecs.append(embedder.multimodal(ref_img, text_desc, ctx))
        else:
            vecs.append(embedder.text(text_desc, ctx))
    paired = list(zip(catalog, vecs, strict=False))
    _CATALOG_VECTOR_CACHE[cache_key] = paired
    return paired


def _packaging_geometry_bonus(aspect_ratio: float, packaging_type: str) -> float:
    """Level 3-4 Constrained Hierarchy: reward compatibility between box aspect ratio and packaging type."""
    pkg = packaging_type.lower()
    if pkg in ("bottle", "pump_bottle", "aerosol_can", "roll_on"):
        return 0.025 if aspect_ratio <= 0.58 else -0.030
    if pkg in ("tube", "sachet", "pouch", "spout_pouch"):
        return 0.022 if 0.28 <= aspect_ratio <= 0.72 else -0.020
    if pkg in ("box", "carton", "bar", "jar", "tub"):
        return 0.028 if aspect_ratio >= 0.52 else -0.025
    return 0.0


def rerank_top_k_candidates(
    crop: Image.Image,
    box: Box,
    top_k_scored: list[tuple[dict[str, Any], float]],
    prior: dict[str, Any] | None = None,
    candidate_whitelist: set[str] | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """Stage 2 Re-Ranker: re-score Top-K bi-encoder candidates using:
    1. Level 1-2 DiffusionGemma JEV / upstream prior conditioning (`category`, `brand`, `packaging_type`),
    2. Level 3-4 Packaging aspect-ratio geometry constraint (`_packaging_geometry_bonus`),
    3. Level 5-7 Sub-ROI CIELAB Delta-E chromatic distance (`_delta_e_cie76`).
    """
    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    aspect_ratio = bw / bh
    crop_feats = extract_gemini_subroi_embedding(crop, (0.0, 0.0, float(crop.width), float(crop.height)))
    raw_obs = crop_feats.get("sub_roi_lab") or (65.0, 5.0, 10.0)
    obs_lab: tuple[float, float, float] = (float(raw_obs[0]), float(raw_obs[1]), float(raw_obs[2]))

    reranked: list[tuple[dict[str, Any], float]] = []
    for sku, base_sim in top_k_scored:
        score = float(base_sim)
        sku_id = str(sku.get("sku_id", ""))
        if candidate_whitelist and sku_id in candidate_whitelist:
            score += 0.035
        if prior and isinstance(prior, dict):
            if str(prior.get("brand", "")).lower() == str(sku.get("brand", "")).lower():
                score += 0.030
            if str(prior.get("category", "")).lower() == str(sku.get("category", "")).lower():
                score += 0.018
            if str(prior.get("packaging_type", "")).lower() == str(sku.get("packaging_type", "")).lower():
                score += 0.020
        score += _packaging_geometry_bonus(aspect_ratio, str(sku.get("packaging_type", "bottle")))
        raw_ref = sku.get("cap_lab") or (65.0, 5.0, 10.0)
        ref_lab: tuple[float, float, float] = (float(raw_ref[0]), float(raw_ref[1]), float(raw_ref[2]))
        delta_e = _delta_e_cie76(obs_lab, ref_lab)
        lab_bonus = max(-0.025, min(0.035, (18.0 - delta_e) * 0.002))
        score = min(0.998, max(0.0, score + lab_bonus))
        reranked.append((sku, round(score, 4)))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked


def match_sku_vectors(
    image: Image.Image,
    boxes: list[Box],
    embedder: VertexEmbeddings,
    ctx: Context,
    *,
    confident_threshold: float = 0.82,
    ambiguous_threshold: float = 0.65,
    max_crops: int = 16,
    prior: list[dict[str, Any]] | None = None,
    candidate_whitelists: list[set[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Embed detected shelf crops with gemini-embedding-2-preview, retrieve Top-3 candidates,
    and re-rank with Constrained Hierarchical Geometry + Sub-ROI CIELAB Delta-E.
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
        crop_im = crops[min(idx, len(crops) - 1)]
        vec = crop_vectors[min(idx, len(crop_vectors) - 1)]
        bi_scored = sorted(
            ((sku, _cosine_similarity(vec, cat_vec)) for sku, cat_vec in catalog),
            key=lambda item: item[1],
            reverse=True,
        )
        box_prior = prior[idx] if (prior and idx < len(prior)) else None
        box_wl = candidate_whitelists[idx] if (candidate_whitelists and idx < len(candidate_whitelists)) else None
        scored = rerank_top_k_candidates(
            crop_im,
            box,
            bi_scored[:3],
            prior=box_prior,
            candidate_whitelist=box_wl,
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
                "sku_id": str(best_sku.get("sku_id", "UL-DOVE-BW-500ML")),
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
                "reranked_top3": [str(s[0].get("sku_id", "")) for s in scored[:3]],
            }
        )

    return labels, diagnostics


def match_promotion_reference(
    image: Image.Image,
    promo_boxes: list[Box],
    embedder: VertexEmbeddings,
    ctx: Context | None = None,
    reference_campaign_text: str = "Unilever Modern Trade Promotional Gondola Header & Price-Off Shelf Strip",
) -> list[dict[str, Any]]:
    """Multimodal gemini-embedding-2-preview Promotion Compliance Matcher (replaces legacy ViT-B-16-plus-240)."""
    if not promo_boxes:
        return []
    ref_vec = embedder.text(reference_campaign_text, ctx)
    results: list[dict[str, Any]] = []
    for idx, box in enumerate(promo_boxes):
        crop = _crop_image(image, box)
        crop_vec = embedder.image(crop, ctx)
        sim = _cosine_similarity(crop_vec, ref_vec)
        results.append(
            {
                "asset_index": idx,
                "box": [round(float(v), 1) for v in box],
                "reference_similarity": round(float(sim), 4),
                "promo_compliant": bool(sim >= 0.62),
                "campaign_reference": reference_campaign_text,
            }
        )
    return results


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "get_cached_catalog_vectors",
    "match_promotion_reference",
    "match_sku_vectors",
    "rerank_top_k_candidates",
]
