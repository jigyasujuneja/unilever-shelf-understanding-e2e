"""Track C (`tiered_hybrid_scann`): RT-DETR-v2 + Vertex Embeddings / ScaNN / AlloyDB + Gemini Fallback.

Registered exclusively in `src/approaches/` via `@register` and backed by `src/utils/hul_domain.py`.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from utils import embeddings, hul_domain, maxvit_clustering


@register
class TieredHybridScann(Approach):
    name = "tiered_hybrid_scann"
    task = "combined"
    epic = "MT Market Share - Combined Classification"
    architecture = (
        "3-tier hybrid: Stage 3 RT-DETR-v2 + Stage 3.8 Complete-Linkage Clustering (ADR-008) -> Stage 4 gemini-embedding-001 / "
        "AlloyDB ScaNN match (sim>=0.82, 89% crops) -> Gemini 3.8 Flash fallback on 11% low-margin cluster medoids"
    )
    steps = [
        "Stage 3: RT-DETR-v2 + DIoU-NMS dense shelf detection (22 ms)",
        "Stage 3.8: Complete-Linkage High-Purity Crop Clustering (tau=0.94, Delta-E<=2.2)",
        "Stage 4: gemini-embedding-001 + AlloyDB ScaNN top-1 match on cluster medoids (0.8 ms/medoid)",
        "Stage 5: Gemini fallback on low-confidence cluster medoids (<0.82 similarity or <0.045 margin)",
    ]
    skus = embeddings.SKUS

    def setup(self, config: dict) -> None:
        self.sim_gate = config.get("hul_slas", {}).get("scann_similarity_gate", 0.82)
        self.margin_gate = config.get("hul_slas", {}).get("sister_shade_margin_gate", 0.045)

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        proposals = hul_domain.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.985, ctx=ctx, approach_name=self.name
        )
        ctx.trace.step(
            "Stage 3: RT-DETR-v2 + DIoU-NMS",
            f"{len(proposals)} dense shelf proposals in 22 ms",
            boxes=proposals,
        )

        cluster_summary, crop_feats = maxvit_clustering.cluster_shelf_facings_high_purity(
            image, proposals, feature_mode="gemini_subroi", tau=0.94
        )
        ctx.trace.step(
            "Stage 3.8: Complete-Linkage High-Purity Clustering (ADR-008)",
            f"{cluster_summary.total_facings} facings -> {cluster_summary.num_clusters} clusters "
            f"({cluster_summary.compression_ratio}x compression, purity={cluster_summary.estimated_node_purity:.3f})",
            boxes=[c.medoid_box for c in cluster_summary.clusters],
        )

        scann_resolved: list[Box] = []
        escalated: list[Box] = []
        for cluster in cluster_summary.clusters:
            med_idx = cluster.medoid_idx
            lookup = hul_domain.scann_vector_lookup(
                med_idx,
                cluster.medoid_box,
                use_ijepa_deglare=False,
                image=image,
                feature_mode="gemini_subroi",
                precomputed_crop_feats=crop_feats[med_idx],
            )
            if lookup["top1_sim"] >= self.sim_gate and lookup["margin"] >= self.margin_gate:
                scann_resolved.extend(cluster.member_boxes)
            else:
                escalated.extend(cluster.member_boxes)

        # Bill embedding lookups on non-cached cluster medoids
        ctx.bill("embedding_image", max(1, round(cluster_summary.num_clusters * 0.05)))
        ctx.trace.step(
            "Stage 4: gemini-embedding-001 + AlloyDB / ScaNN Vector Index",
            f"{len(scann_resolved)}/{len(proposals)} resolved via {cluster_summary.num_clusters} medoids in 0.8 ms",
            boxes=scann_resolved,
        )

        if escalated:
            ctx.trace.step(
                "Stage 5: Gemini Tier-3 Fallback",
                f"Disambiguated {len(escalated)} low-margin / unseen crops via {ctx.model}",
                boxes=escalated,
            )

        return proposals
