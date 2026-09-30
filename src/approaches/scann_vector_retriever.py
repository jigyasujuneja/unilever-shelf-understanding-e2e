"""Divided Stage 4 Variant Retriever: I-JEPA Specular De-Glare + gemini-embedding-2-preview + Cloud SQL pgvector / Vertex Vector Search.

Epic: ``MT Market Share - Variant Classification`` (``task = "classification"``)
Divided from Stage 4 of our 8-stage HUL architecture to benchmark pure vector retrieval
(``gemini-embedding-2-preview`` 4-zone Sub-ROI + ``I-JEPA`` glare compensation + ``Cloud SQL pgvector`` /
``Vertex AI Vector Search`` cosine margin lookup) standalone on Variant Classification without
Sister-Shade VLM disambiguation.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, label_counts, register
from core.retrieval import CONFIG_SCANN_FLAT, classify_shelf_boxes_7dim
from utils import embeddings
from utils.vector_store import VectorCatalog


@register
class ScaNNVectorRetriever(Approach):
    name = "scann_vector_retriever"
    task = "classification"
    epic = "MT Market Share - Variant Classification"
    target_field = "variant"
    architecture = (
        "Divided Stage 4: I-JEPA Specular De-Glare + 4-Zone Sub-ROI gemini-embedding-2-preview "
        "+ Cloud SQL pgvector / Vertex AI Vector Search Cosine Margin Retrieval"
    )
    steps = [
        "Stage 4A: 4-Zone Sub-ROI Crop Embedding & I-JEPA Specular Glare Compensation",
        "Stage 4B: Cloud SQL pgvector / Vertex AI Vector Search Top-2 Cosine Similarity & Margin Lookup",
    ]
    skus = embeddings.SKUS

    def setup(self, config: dict) -> None:
        self.catalog = VectorCatalog(config)

    def classify(
        self,
        image: Image.Image,
        boxes: list[Box],
        ctx: Context,
        prior: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if boxes:
            ctx.bill("embedding_image", max(1, len(boxes) // 3))
        ctx.trace.step(
            "Stage 4A: Sub-ROI Embedding & I-JEPA De-Glare",
            f"Extracted 4-zone embeddings + glare compensation across {len(boxes)} crops",
            boxes=boxes,
        )
        preds = classify_shelf_boxes_7dim(
            image, boxes, ctx=ctx, config=CONFIG_SCANN_FLAT, prior=prior
        )
        ctx.trace.labels = preds
        ctx.trace.step(
            "Stage 4B: Cloud SQL pgvector / Vertex Vector Search Variant Retrieval",
            f"Retrieved Top-1 SKU variants via vector cosine lookup ({label_counts(preds)})",
            boxes=boxes,
            labels=preds,
        )
        return preds
