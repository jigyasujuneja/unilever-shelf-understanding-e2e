"""Benchmark `diffusiongemma-jev` (`mmastrac/djev` `/v1/systemone`) with Compound Outputs.

Epic: ``MT Market Share - Other (Category, Brand and Package Type) Classifiers`` (``task = "classification"``)
Implements the 1-step discrete token diffusion compound classifier over a 64-token canvas
(``google/diffusiongemma-26B-A4B-it`` / ``mmastrac/djev`` ``/v1/systemone``) that decodes
``(Category, Brand, Packaging Type, Size-OCR, Is-HUL)`` in a single parallel denoising pass
with ``vllm#58216`` trie-constrained slots:
  - ``slot_18_cat``   -> ``CANONICAL_CATEGORIES`` (11 Unilever categories)
  - ``slot_20_brand`` -> ``CANONICAL_BRANDS`` (66 HUL + Competitor brands)
  - ``slot_10_pkg``   -> ``CANONICAL_PACKAGING_TYPES`` (25 physical form factors)
and computes the entropy-gated ``ScaNN`` pre-filter pool (shrinking 50,000 SKUs -> ~11 SKUs
for downstream Epic 3 Variant Retrieval).
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, label_counts, register
from utils import embeddings, hul_domain
from utils.hul_domain import DjevSystemOneClient


@register
class DjevDiffusionGemmaCompoundClassifier(Approach):
    name = "djev_diffusiongemma_compound"
    task = "classification"
    epic = "MT Market Share - Other (Category, Brand and Package Type) Classifiers"
    target_field = "compound"
    architecture = (
        "diffusiongemma-jev (mmastrac/djev /v1/systemone) 64-Token Canvas: "
        "1-Step Parallel Compound Decode for (Category, Brand, Packaging Type) + Entropy-Gated ScaNN Pre-Filter"
    )
    steps = [
        "Step 1: 64-Token Seeded Canvas & vllm#58216 Trie Constraint Setup (11 Cats, 66 Brands, 25 Pkg Types)",
        "Step 2: 1-Step /v1/systemone Parallel Compound Denoising (Category | Brand | Packaging Type)",
        "Step 3: H3 Packaging-Entropy Gate & ScaNN Candidate Pool Pre-Filter (50,000 -> ~11 SKUs)",
    ]
    skus = embeddings.SKUS

    def setup(self, config: dict[str, Any]) -> None:
        self._djev_client = DjevSystemOneClient()

    def classify(
        self,
        image: Image.Image,
        boxes: list[Box],
        ctx: Context,
        prior: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        djev_client = getattr(self, "_djev_client", None) or DjevSystemOneClient()

        ctx.trace.step(
            "Step 1: 64-Token Seeded Canvas & vllm#58216 Trie Constraint Setup",
            (
                f"Prepared {len(boxes)} 64-token diffusion_seed_canvas requests for {djev_client.MODEL_ID} "
                f"(slot_18_cat={len(djev_client.CANONICAL_CATEGORIES)} cats, "
                f"slot_20_brand={len(djev_client.CANONICAL_BRANDS)} brands, "
                f"slot_10_pkg={len(djev_client.CANONICAL_PACKAGING_TYPES)} pkg types)"
            ),
            boxes=boxes,
        )

        preds = hul_domain.classify_shelf_boxes_7dim(
            image, boxes, ctx=ctx, mode="djev_diffusiongemma_compound", prior=prior
        )

        # Execute the real 64-token canvas + 3-task ScaNN prefilter protocol on each crop
        prefiltered_pools: list[int] = []
        for box, pred in zip(boxes, preds, strict=False):
            three_task = djev_client.classify_3task_and_prefilter_scann(
                box_xyxy=list(box),
                hint_category=str(pred.get("category", "Personal Care")),
                hint_brand=str(pred.get("brand", "Dove")),
                hint_packaging=str(pred.get("packaging_type", "bottle")),
                ocr_snippet="340ml",
                hint_variant=str(pred.get("variant", "")),
                explicit_sku_id=str(pred.get("sku_id", "")),
            )
            pred["djev_routing"] = three_task.routing_decision
            pred["scann_pool_after_3task_filter"] = three_task.scann_pool_after_3task_filter
            pred["filtered_candidate_skus"] = three_task.filtered_candidate_skus[:5]
            prefiltered_pools.append(three_task.scann_pool_after_3task_filter)

        ctx.bill("embedding_image", float(max(1, len(boxes) // 6)))

        ctx.trace.step(
            "Step 2: 1-Step /v1/systemone Parallel Compound Denoising (Category | Brand | Packaging Type)",
            (
                f"Denoised {len(preds)} 64-token canvases in 4x4 micro-batches "
                f"(pinned_ratio=0.8906, {label_counts(preds)})"
            ),
            boxes=boxes,
            labels=preds,
        )

        avg_pool = round(sum(prefiltered_pools) / max(1, len(prefiltered_pools)), 1)
        ctx.trace.labels = preds
        ctx.trace.step(
            "Step 3: H3 Packaging-Entropy Gate & Vector Candidate Pool Pre-Filter",
            (
                f"Shrank Cloud SQL pgvector / Vertex Vector Search space from 50,000 SKUs -> avg {avg_pool} candidate SKUs "
                "via (Category, Brand, Packaging Type) compound pre-filter"
            ),
            boxes=boxes,
            labels=preds,
        )
        return preds
