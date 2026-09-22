"""Stage 3: Vector Search / Catalog Matching (Vertex AI Vector Search / ScaNN / cosine ANN).

Two independent things happen here, and it matters which is which:

1.  **Intra-shelf crop-to-crop similarity** (always real). Cosine similarity between the 1408-D
    `multimodalembedding@001` vectors of the shelf's own crops. This needs no external data and
    genuinely identifies adjacent/duplicate facings of the same physical SKU.

2.  **Reference catalog matching** (only as real as the catalog). A class-agnostic detector finds
    *things*; it cannot name them. Naming requires a reference catalog of known products to match
    against. Until one is configured via `embeddings.reference_catalog.source_uri`, this stage
    reports `Unknown` / `NO_CATALOG_INDEXED` and leaves `matched_sku_id` empty.

    An earlier version of this file carried eight hardcoded "visual taxonomy prototypes" describing
    the products on one sample face-wash shelf, and fell back to the first of them whenever
    similarity was low. That produced confident brand and variant strings for every facing on every
    image, which looked like working product recognition and scored well on that one shelf. It was
    measuring nothing. Prototypes now live in a catalog file you point at explicitly; the bundled
    `configs/demo_visual_prototypes.json` is labelled a demo for exactly this reason.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from shelf_benchmark.approaches.base import CommonLayerContext
from shelf_benchmark.approaches.class_agnostic_visual_embedding.stage2_visual_crop_embedder import (
    embed_visual_text_prototypes,
)

logger = logging.getLogger(__name__)

# Cache keyed by catalog URI so swapping catalogs mid-process re-embeds instead of silently
# reusing the previous catalog's vectors.
_CATALOG_VECTOR_CACHE: Dict[str, Tuple[List[Dict[str, Any]], List[List[float]]]] = {}

_REQUIRED_CATALOG_FIELDS = ("brand", "prompt")


class ReferenceCatalogError(RuntimeError):
    """Raised when a reference catalog is configured but unusable."""


def cosine_sim_1408(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity in the 1408-D metric learning space."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def load_reference_catalog(ctx: CommonLayerContext) -> List[Dict[str, Any]]:
    """Reads the configured reference catalog. Returns an empty list when none is configured.

    Raises `ReferenceCatalogError` if a catalog *is* configured but cannot be read or is malformed:
    a typo in the path must not silently degrade into unnamed predictions.
    """
    cat_cfg = ctx.config.embeddings.reference_catalog
    if not cat_cfg.source_uri:
        return []

    try:
        raw = ctx.storage.read_json(cat_cfg.source_uri)
    except Exception as exc:
        raise ReferenceCatalogError(
            f"Reference catalog '{cat_cfg.source_uri}' is configured but could not be read: {exc}. "
            f"Fix the path, or set embeddings.reference_catalog.source_uri to null to run "
            f"detection-only without product naming."
        ) from exc

    entries = raw.get("products", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ReferenceCatalogError(
            f"Reference catalog '{cat_cfg.source_uri}' contained no product entries. Expected a "
            f"JSON list (or an object with a 'products' list)."
        )

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ReferenceCatalogError(
                f"Reference catalog entry #{idx} is {type(entry).__name__}, expected an object."
            )
        missing = [f for f in _REQUIRED_CATALOG_FIELDS if not entry.get(f)]
        if missing:
            raise ReferenceCatalogError(
                f"Reference catalog entry #{idx} is missing required field(s): {', '.join(missing)}."
            )
    return list(entries)


def _get_catalog_vectors(
    ctx: CommonLayerContext,
) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    """Loads and embeds the reference catalog, memoized per catalog URI."""
    cat_cfg = ctx.config.embeddings.reference_catalog
    cache_key = cat_cfg.source_uri or ""
    if cache_key in _CATALOG_VECTOR_CACHE:
        return _CATALOG_VECTOR_CACHE[cache_key]

    entries = load_reference_catalog(ctx)
    if not entries:
        _CATALOG_VECTOR_CACHE[cache_key] = ([], [])
        return [], []

    vectors = embed_visual_text_prototypes(
        prototype_texts=[str(e["prompt"]) for e in entries],
        project_id=ctx.config.gcp.project_id,
        location="us-central1",
    )
    if not vectors or len(vectors) != len(entries):
        raise ReferenceCatalogError(
            f"Embedded {len(vectors) if vectors else 0} reference vectors for {len(entries)} "
            f"catalog entries. Refusing to match against a partially embedded catalog."
        )
    logger.info(
        "Indexed %d reference catalog entries from '%s' for 1408-D vector search.",
        len(entries), cat_cfg.source_uri,
    )
    _CATALOG_VECTOR_CACHE[cache_key] = (entries, vectors)
    return entries, vectors


def clear_catalog_cache() -> None:
    """Drops memoized catalog vectors. Exposed for tests and for hot-swapping catalogs."""
    _CATALOG_VECTOR_CACHE.clear()


def _best_catalog_match(
    crop_vector: List[float],
    entries: List[Dict[str, Any]],
    vectors: List[List[float]],
    min_similarity: float,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Returns the best-scoring catalog entry, or `None` when nothing clears `min_similarity`.

    Returning `None` rather than the argmax is the whole point: an unmatched facing is a real and
    reportable outcome, and pretending otherwise inflates every downstream classification metric.
    """
    best_entry: Optional[Dict[str, Any]] = None
    best_sim = 0.0
    for entry, vec in zip(entries, vectors):
        sim = cosine_sim_1408(crop_vector, vec)
        if sim > best_sim:
            best_sim = sim
            best_entry = entry
    if best_entry is None or best_sim < min_similarity:
        return None, best_sim
    return best_entry, best_sim


def run_stage3_vector_search_matching(
    ctx: CommonLayerContext,
    embedded_facings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Runs cosine ANN vector search over the 1408-D visual crop embeddings.

    For each facing this computes:
    - `nearest_shelf_facing_idx` / `nearest_shelf_facing_visual_sim`: the most visually similar
      other crop on the same shelf (always available, no catalog required).
    - `predicted_brand` / `predicted_variant` / `matched_sku_id`: the nearest reference catalog
      entry above `min_match_similarity`, or the configured unknown label when no catalog is
      indexed or nothing clears the floor.
    - `catalog_status`: one of `MATCHED`, `BELOW_MATCH_THRESHOLD`, `NO_CATALOG_INDEXED`.
    """
    cat_cfg = ctx.config.embeddings.reference_catalog
    entries, vectors = _get_catalog_vectors(ctx)
    unknown = cat_cfg.unknown_brand_label

    if not entries:
        logger.warning(
            "No reference catalog is configured (embeddings.reference_catalog.source_uri is unset), "
            "so this approach can localize facings but cannot identify products. Brand/variant will "
            "be reported as '%s' and classification accuracy will be 0. Point source_uri at a "
            "catalog JSON to enable product matching.",
            unknown,
        )

    all_bboxes = [f.get("bbox_2d", [0, 0, 0, 0]) for f in embedded_facings]
    matched_facings: List[Dict[str, Any]] = []

    for i, facing in enumerate(embedded_facings):
        vec_i = facing.get("visual_embedding_vector", [])

        # 1. Intra-shelf image-to-image nearest neighbor (real: needs no external catalog).
        best_peer_idx: Optional[int] = None
        best_peer_sim: float = -1.0
        for j, other in enumerate(embedded_facings):
            if i == j:
                continue
            sim_ij = cosine_sim_1408(vec_i, other.get("visual_embedding_vector", []))
            if sim_ij > best_peer_sim:
                best_peer_sim = sim_ij
                best_peer_idx = int(other.get("product_index", j + 1))

        # 2. Reference catalog ANN match (only as good as the configured catalog).
        match, match_sim = (
            _best_catalog_match(vec_i, entries, vectors, cat_cfg.min_match_similarity)
            if entries
            else (None, 0.0)
        )

        if not entries:
            catalog_status = "NO_CATALOG_INDEXED"
        elif match is None:
            catalog_status = "BELOW_MATCH_THRESHOLD"
        else:
            catalog_status = "MATCHED"

        bbox = facing.get("bbox_2d", [0, 0, 0, 0])
        packaging = str(match.get("packaging", "")) if match else ""
        rule_size = ctx.derive_size_bucket_from_bbox(
            bbox_2d=bbox,
            all_bboxes_on_shelf=all_bboxes,
            packaging_type=packaging or None,
        )

        matched_facings.append(
            {
                **facing,
                "predicted_category": str(match.get("category", "")) if match else "",
                "predicted_subcategory": str(match.get("subcategory", "")) if match else "",
                "predicted_brand": str(match.get("brand", unknown)) if match else unknown,
                "is_hul_brand": bool(match.get("is_hul", False)) if match else False,
                "predicted_variant": str(match.get("variant", "")) if match else "",
                "predicted_packaging": packaging,
                "predicted_pack_type": str(match.get("pack_type", "Single")) if match else "",
                "predicted_size": rule_size,
                "rule_derived_size_bucket": rule_size,
                "matched_sku_id": match.get("sku_id") if match else None,
                "nearest_shelf_facing_idx": best_peer_idx,
                "nearest_shelf_facing_visual_sim": round(max(0.0, best_peer_sim), 4),
                "catalog_match_similarity_1408d": round(max(0.0, match_sim), 4),
                "ann_engine": "cosine ANN over 1408-D multimodalembedding@001 vectors",
                "catalog_status": catalog_status,
            }
        )

    return matched_facings
