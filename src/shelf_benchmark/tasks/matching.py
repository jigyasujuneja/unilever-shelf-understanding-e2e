"""Separated Task 3: Product Matching via Hybrid Search (`ProductMatchingTask`).

Combines Front-Facing-Only Depth De-duplication + 7-Dimension HUL Taxonomy Filters
(`Category`, `Subcategory`, `Brand`, `Variant`, `Packaging type`, `Pack type`, `Rule-derived Size`)
with Dense Vector (`gemini-embedding-001` 3072-D) + Sparse Lexical BM25 Query Generation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from shelf_benchmark.config import TaxonomyConfig
from shelf_benchmark.evaluation.cost import extract_token_usage
from shelf_benchmark.models import (
    ProductMatchingOutput,
    RowLevelReportItem,
    TokenUsageMetrics,
)
from shelf_benchmark.tasks.base import BaseBenchmarkTask
from shelf_benchmark.tasks.facing_utils import (
    check_is_hul_brand,
    deduplicate_depth_stacked_facings,
    derive_size_bucket_from_bbox,
)


def build_hybrid_matching_prompt(taxonomy: Optional[TaxonomyConfig] = None) -> str:
    """Dynamically builds the Hybrid Matching prompt from centralized `TaxonomyConfig`."""
    tax = taxonomy or TaxonomyConfig.from_yaml_or_defaults()
    cats_str = ", ".join(f"`{c}`" for c in tax.categories)
    subcats_str = ", ".join(f"`{s}`" for s in tax.subcategories[:10])
    pkg_str = ", ".join(f"`{p}`" for p in tax.packaging_types)
    pack_str = " or ".join(f'`"{pt}"`' for pt in tax.pack_types)

    return f"""You are an expert retail Product Matching & Hybrid Search query generator using the configurable 7-Dimension Retail Taxonomy.
Analyze the main middle shelf of this shelf image (from left to right) and for EVERY distinct FRONT-FACING product slot (do NOT count products stacked in depth behind the front unit in the same facing slot), generate the structured signals needed for Hybrid Search (combining Dense Vector Search + Sparse Lexical/BM25 Metadata Filtering):

1. `product_index`: 1-based front-facing slot index from left to right.
2. `bbox_2d`: Normalized `[ymin, xmin, ymax, xmax]` (0 to 1000).
3. `category`: Configured category ({cats_str}).
4. `subcategory`: Configured subcategory (e.g., {subcats_str}).
5. `brand`: Brand name read from the packaging.
6. `variant`: Specific product line & variant read from the packaging.
7. `packaging_type`: Configured packaging type ({pkg_str}).
8. `pack_type`: {pack_str}.
9. `size`: Size cue / bucket.
10. `lexical_search_keywords`: List of specific OCR text keywords, category, subcategory, brand, variant, packaging, pack type, and size terms for sparse BM25 keyword search.
11. `dense_embedding_text`: A rich, self-contained semantic & visual description synthesized for dense vector embedding retrieval.
12. `matched_sku_id`: Set to `"HYBRID_SEARCH_READY"` unless an explicit Product Catalog with SKU IDs is provided in the prompt."""


HYBRID_MATCHING_PROMPT = build_hybrid_matching_prompt()


class ProductMatchingTask(BaseBenchmarkTask):
    """Product Matching benchmark task using Hybrid Search (Dense Vector + Sparse Lexical + 7 HUL Dimensions)."""

    task_type = "matching"

    def _compute_dense_embeddings(self, texts: List[str]) -> Optional[int]:
        """Call Vertex AI `gemini-embedding-001` to generate 3072-D dense vectors for hybrid search."""
        if not texts:
            return None
        try:
            emb_client = self.get_client(location="us-central1")
            resp = emb_client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts[:20],
            )
            if resp.embeddings and resp.embeddings[0].values:
                return len(resp.embeddings[0].values)
        except Exception:
            pass
        return None

    def invoke_model(
        self,
        model_name: str,
        shelf_image_uri: str,
        catalog_data: Optional[Dict[str, Any]] = None,
        planogram_data: Optional[Dict[str, Any]] = None,
        catalog_uri: Optional[str] = None,
        planogram_uri: Optional[str] = None,
        prior_classifications: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], TokenUsageMetrics, List[RowLevelReportItem]]:
        client = self.get_client()
        image_part = self.storage.to_genai_part(shelf_image_uri)

        if catalog_data is None and catalog_uri:
            catalog_data = self.storage.read_json(catalog_uri)
        if planogram_data is None and planogram_uri:
            planogram_data = self.storage.read_json(planogram_uri)

        prompt_parts = [HYBRID_MATCHING_PROMPT]

        if catalog_data:
            prompt_parts.append(
                "\n### CONNECTED PRODUCT CATALOG:\n" + json.dumps(catalog_data, indent=2)
            )
        else:
            prompt_parts.append(
                "\nNo SKU catalog is connected yet. Produce high-recall `lexical_search_keywords` and `dense_embedding_text` incorporating all 7 HUL dimensions for Hybrid Vector Search, and set `matched_sku_id` to `\"HYBRID_SEARCH_READY\"`."
            )

        if planogram_data:
            prompt_parts.append(
                "\n### OPTIONAL PLANOGRAM SPECIFICATION:\n" + json.dumps(planogram_data, indent=2)
            )

        if prior_classifications:
            prompt_parts.append(
                "\n### PRIOR STAGE CLASSIFICATIONS:\n" + json.dumps(prior_classifications, indent=2)
            )

        response = client.models.generate_content(
            model=model_name,
            contents=[image_part, "\n".join(prompt_parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProductMatchingOutput,
                temperature=0.1,
            ),
        )

        tokens = extract_token_usage(response)
        raw_text = response.text or "{}"
        parsed_dict = json.loads(raw_text)
        validated = ProductMatchingOutput.model_validate(parsed_dict)

        raw_items = [it.model_dump() for it in validated.matched_products]
        dedup_items, depth_filtered = deduplicate_depth_stacked_facings(raw_items)
        all_boxes = [it.get("bbox_2d", [0, 0, 0, 0]) for it in dedup_items]

        dense_texts = [
            it.get("dense_embedding_text") or f"{it.get('brand', '')} {it.get('product_name', '')} {it.get('variant', '')}"
            for it in dedup_items
        ]
        embedding_dim = self._compute_dense_embeddings(dense_texts)

        rows: List[RowLevelReportItem] = []
        for idx, item in enumerate(dedup_items, start=1):
            box = item.get("bbox_2d") or [0, 0, 0, 0]
            pkg = item.get("packaging_type") or "tube"
            model_size = item.get("size") or ""
            rule_size = derive_size_bucket_from_bbox(box, all_boxes, pkg, model_size)
            brand_val = item.get("brand") or ""
            keywords_str = ", ".join(item.get("lexical_search_keywords") or [])

            rows.append(
                RowLevelReportItem(
                    run_id="",
                    trace_id="",
                    span_id="",
                    task_type=self.task_type,
                    separation_approach="hybrid_search_facing_nms",
                    model_name=model_name,
                    shelf_image_uri=shelf_image_uri,
                    start_time="",
                    end_time="",
                    image_latency_ms=0.0,
                    product_index=idx,
                    shelf_row=item.get("shelf_row", "middle"),
                    position_on_shelf=idx,
                    bbox_ymin=box[0],
                    bbox_xmin=box[1],
                    bbox_ymax=box[2],
                    bbox_xmax=box[3],
                    predicted_category=item.get("category", "Skin Care"),
                    predicted_subcategory=item.get("subcategory", "Face Wash"),
                    predicted_brand=brand_val,
                    is_hul_brand=check_is_hul_brand(brand_val),
                    predicted_variant=item.get("variant", ""),
                    predicted_packaging=pkg,
                    predicted_pack_type=item.get("pack_type", "Single"),
                    predicted_size=model_size,
                    rule_derived_size_bucket=rule_size,
                    predicted_product_name=item.get("product_name", ""),
                    confidence=float(item.get("match_confidence", 0.95)),
                    lexical_search_keywords=keywords_str,
                    dense_embedding_text=item.get("dense_embedding_text", ""),
                    embedding_vector_dim=embedding_dim,
                    matched_sku_id=item.get("matched_sku_id") or "HYBRID_SEARCH_READY",
                    planogram_compliant=item.get("planogram_compliant"),
                )
            )

        out_dump = validated.model_dump()
        out_dump["depth_duplicates_filtered"] = depth_filtered
        out_dump["matched_products"] = dedup_items
        out_dump["hybrid_search_embedding_model"] = "gemini-embedding-001"
        out_dump["hybrid_search_embedding_dimension"] = embedding_dim
        return out_dump, tokens, rows
