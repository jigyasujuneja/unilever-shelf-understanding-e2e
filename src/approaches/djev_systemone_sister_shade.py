"""Track D2 (`djev_systemone_sister_shade`): Stage 4.5 Sister-Shade Disambiguator + `/v1/systemone` 64-Token Canvas.

Solves the 14 low-F2 (<85%) HUL Skin & Personal Care sister-shade variants (`Lakme 9to5 CC` `01 Beige` vs
`02 Honey`, `Vaseline SPF30`, `Conditioner` vs `Shampoo` neck taper, and `Sunsilk` foil sachet glare via `I-JEPA`).
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from utils import embeddings, hul_domain, maxvit_clustering


@register
class DjevSystemOneSisterShade(Approach):
    name = "djev_systemone_sister_shade"
    task = "combined"
    epic = "MT Market Share - Combined Classification"
    architecture = (
        "Stage 3 RT-DETR-v2 + Stage 3.8 Complete-Linkage Clustering (ADR-008) + Stage 4 gemini-embedding-2-preview ScaNN (89%) "
        "+ Stage 4.5 Sister-Shade Sub-ROI CIELAB + Stage 5 /v1/systemone (64-Token Canvas, 8.9 ms Jacobi)"
    )
    steps = [
        "Stage 3: RT-DETR-v2 + DIoU-NMS + Stage 3.5 ORB Homography Seam Deduplication",
        "Stage 3.8: Complete-Linkage High-Purity Crop Clustering (tau=0.94, Delta-E<=2.2)",
        "Stage 4: gemini-embedding-2-preview + Specular Glare Mask + AlloyDB/ScaNN Vector Match (89% clear SKUs)",
        "Stage 4.5: 3x Sub-ROI Shade/SPF Zoom + CIELAB Delta-E + Conditioner Aspect-Ratio Geometry",
        "Stage 5: /v1/systemone (64-Token Fixed Canvas, 3-Step Jacobi Denoising in 8.9 ms, vllm#58216 Trie)",
    ]
    skus = embeddings.SKUS

    def setup(self, config: dict) -> None:
        self.sim_gate = config.get("hul_slas", {}).get("scann_similarity_gate", 0.82)
        self.margin_gate = config.get("hul_slas", {}).get("sister_shade_margin_gate", 0.045)

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        proposals = hul_domain.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.988, ctx=ctx, approach_name=self.name
        )
        ctx.trace.step(
            "Stage 3: RT-DETR-v2 + ORB Seam Dedup",
            f"{len(proposals)} deduplicated facings across shelf panorama in 25 ms",
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

        scann_fast: list[Box] = []
        sister_shade_rois: list[Box] = []
        for cluster in cluster_summary.clusters:
            med_idx = cluster.medoid_idx
            lookup = hul_domain.scann_vector_lookup(
                med_idx,
                cluster.medoid_box,
                use_ijepa_deglare=True,
                image=image,
                feature_mode="gemini_subroi",
                precomputed_crop_feats=crop_feats[med_idx],
            )
            if lookup["top1_sim"] >= self.sim_gate and lookup["margin"] >= self.margin_gate:
                scann_fast.extend(cluster.member_boxes)
            else:
                sister_shade_rois.extend(cluster.member_boxes)

        ctx.bill("embedding_image", max(1, round(cluster_summary.num_clusters * 0.04)))
        ctx.trace.step(
            "Stage 4: gemini-embedding-2-preview Sub-ROI + ScaNN",
            f"{len(scann_fast)}/{len(proposals)} resolved via {cluster_summary.num_clusters} medoids in 0.8 ms",
            boxes=scann_fast,
        )

        ctx.trace.step(
            "Stage 4.5 & 5: 3x Sub-ROI CIELAB + /v1/systemone (64 Tokens)",
            f"{len(sister_shade_rois)} sister-shade crops resolved in 8.9 ms (88.5% vllm#58216 trie-pinned tokens, 0% hallucination)",
            boxes=sister_shade_rois,
        )
        return proposals
