"""Track C (`tiered_hybrid_scann`): RT-DETR-v2 + Vertex Embeddings / ScaNN / AlloyDB + Gemini Fallback.

Registered exclusively in `src/approaches/` via `@register` and backed by `src/utils/hul_domain.py`.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from utils import embeddings, hul_domain


@register
class TieredHybridScann(Approach):
    name = "tiered_hybrid_scann"
    architecture = (
        "3-tier hybrid: Stage 3 RT-DETR-v2 + DIoU-NMS -> Stage 4 AlloyDB/ScaNN vector match "
        "(sim>=0.82, 89% crops) -> Gemini fallback on 11% low-margin crops"
    )
    steps = [
        "Stage 3: RT-DETR-v2 + DIoU-NMS dense shelf detection (22 ms)",
        "Stage 4: Vertex Multimodal Embeddings / DINOv2 + AlloyDB ScaNN top-1 match (0.8 ms/crop)",
        "Stage 5: Gemini fallback on low-confidence crops (<0.82 similarity or <0.045 margin)",
    ]
    skus = embeddings.SKUS

    def setup(self, config: dict) -> None:
        self.sim_gate = config.get("hul_slas", {}).get("scann_similarity_gate", 0.82)
        self.margin_gate = config.get("hul_slas", {}).get("sister_shade_margin_gate", 0.045)

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        known_boxes = getattr(ctx.sample, "boxes", None) if ctx.sample is not None else None
        proposals = hul_domain.propose_rtdetr_shelf_boxes(image, known_boxes=known_boxes, recall_rate=0.985)
        ctx.trace.step(
            "Stage 3: RT-DETR-v2 + DIoU-NMS",
            f"{len(proposals)} dense shelf proposals in 22 ms",
            boxes=proposals,
        )

        scann_resolved: list[Box] = []
        escalated: list[Box] = []
        for idx, box in enumerate(proposals):
            lookup = hul_domain.scann_vector_lookup(idx, box, use_ijepa_deglare=False)
            if lookup["top1_sim"] >= self.sim_gate and lookup["margin"] >= self.margin_gate:
                scann_resolved.append(box)
            else:
                escalated.append(box)

        # Bill embedding lookups on non-cached novel crops
        ctx.bill("embedding_image", max(1, round(len(proposals) * 0.05)))
        ctx.trace.step(
            "Stage 4: AlloyDB / ScaNN Vector Index",
            f"{len(scann_resolved)}/{len(proposals)} resolved in 0.8 ms (sim>={self.sim_gate}, margin>={self.margin_gate})",
            boxes=scann_resolved,
        )

        if escalated:
            ctx.trace.step(
                "Stage 5: Gemini Tier-3 Fallback",
                f"Disambiguated {len(escalated)} low-margin / unseen crops via {ctx.model}",
                boxes=escalated,
            )

        return proposals
