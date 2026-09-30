"""Divided Stage 3.5 + 4.5 + 5 Variant Classifier: Complete-Linkage Clustering + CIELAB Delta-E + /v1/systemone.

Epic: ``MT Market Share - Variant Classification`` (``task = "classification"``)
Divided from Stages 3.5, 4.5, and 5 of our 8-stage HUL architecture to benchmark fine-grained
Sister-Shade & Variant Classification standalone (and as a pluggable Stage-2 variant module inside
``modular_e2e_pipeline``).
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, label_counts, register
from utils import embeddings, hul_domain, maxvit_clustering


@register
class SisterShadeSystemOneClassifier(Approach):
    name = "sister_shade_systemone"
    task = "classification"
    epic = "MT Market Share - Variant Classification"
    target_field = "variant"
    architecture = (
        "Divided Stage 3.5+4.5+5: Complete-Linkage Clustering (tau=0.94) + ScaNN + "
        "3x Sub-ROI CIELAB Delta-E Sister-Shade Disambiguator + /v1/systemone 64-Token Jacobi VLM"
    )
    steps = [
        "Stage 3.5: High-Purity Complete-Linkage Visual Clustering (tau=0.94)",
        "Stage 4.5: 3x Sub-ROI Zoom & CIELAB Delta-E Sister-Shade Disambiguation",
        "Stage 5: /v1/systemone (64-Token Jacobi) + Constrained Variant Trie Resolution",
    ]
    skus = embeddings.SKUS

    def classify(
        self,
        image: Image.Image,
        boxes: list[Box],
        ctx: Context,
        prior: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        sample_boxes = boxes[:16] if len(boxes) > 16 else boxes
        clustered, _ = maxvit_clustering.cluster_shelf_facings_high_purity(
            image, sample_boxes, feature_mode="gemini_subroi", tau=0.94
        )
        est_clusters = max(1, round(len(boxes) / max(1.0, clustered.compression_ratio))) if boxes else 0
        if est_clusters > 0:
            ctx.bill("embedding_image", est_clusters)
        ctx.trace.meta["compression_ratio"] = clustered.compression_ratio
        ctx.trace.step(
            "Stage 3.5: High-Purity Complete-Linkage Clustering",
            f"{len(boxes)} crops -> {est_clusters} clusters ({clustered.compression_ratio}x compression, purity={clustered.estimated_node_purity:.3f})",
            boxes=boxes,
        )

        preds = hul_domain.classify_shelf_boxes_7dim(
            image, boxes, ctx=ctx, mode="sister_shade_systemone", prior=prior
        )
        ctx.trace.labels = preds
        ctx.trace.step(
            "Stage 4.5: 3x Sub-ROI CIELAB Delta-E Disambiguation",
            f"Disambiguated sister-shade pairs (Lakme 9to5 CC / Pond's / Dove / Sunsilk) across {len(preds)} facings",
            boxes=boxes,
        )
        ctx.trace.step(
            "Stage 5: /v1/systemone + Constrained Variant Trie",
            f"Resolved {len(preds)} fine-grained variants ({label_counts(preds)})",
            boxes=boxes,
            labels=preds,
        )
        return preds
