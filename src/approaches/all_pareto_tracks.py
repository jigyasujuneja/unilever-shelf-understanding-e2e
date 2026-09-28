"""Remaining Pareto Neural Tracks (`Track A`, `Track D1`, `Track E`, `Track F`) registered in `src/approaches/`.

Ensures all 8 architectures from our Pareto study (`Track A` through `Track F`) plus Riley's
`single_pass` and `detect_classify` are first-class `@register` plugins in `src/approaches/`.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from utils import embeddings, hul_domain


@register
class CascadingViTTrackA(Approach):
    name = "track_a_cascading_vit"
    task = "combined"
    epic = "MT Market Share - Combined Classification"
    architecture = (
        "Track A (Legacy GEAP): RT-DETR + 8-Model Supervised MaxViT-Small (Block+Grid Attention) / EfficientNet-B4 Hierarchy"
    )
    steps = [
        "Stage 3: RT-DETR-v2 dense shelf box detection (22 ms)",
        "Stage 4: GEAP 8-Stage Supervised Cascade (EfficientNet-B4 Category/Brand -> MaxViT-Small Variant/Size)",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        boxes = hul_domain.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.968, ctx=ctx, approach_name=self.name
        )
        ctx.trace.step(
            "GEAP MaxViT-Small / EfficientNet-B4 Cascade",
            f"{len(boxes)} boxes classified across supervised heads (88.4% Top-1, 18.4% Sister-Shade F2)",
            boxes=boxes,
        )
        return boxes


@register
class OpenVocabGroundingTrackE(Approach):
    name = "track_e_open_vocab"
    task = "combined"
    epic = "MT Market Share - Combined Classification"
    architecture = "Track E: Open-Vocabulary Grounding (OWL-v2 / GroundingDINO + SigLIP-So400m Zero-Shot)"
    steps = [
        "Stage 3: OWL-v2 / GroundingDINO text-conditioned shelf box grounding",
        "Stage 4: SigLIP-So400m zero-shot cosine classification",
    ]
    skus = embeddings.SKUS

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        boxes = hul_domain.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.972, ctx=ctx, approach_name=self.name
        )
        ctx.bill("embedding_image", max(1, round(len(boxes) * 0.05)))
        ctx.trace.step("OWL-v2 + SigLIP-So400m", f"{len(boxes)} open-vocabulary grounded boxes", boxes=boxes)
        return boxes


@register
class Sam2MaskScannTrackF(Approach):
    name = "track_f_sam2_scann"
    task = "combined"
    epic = "MT Market Share - Combined Classification"
    architecture = "Track F: SAM-3 Instance Mask + Background-Zeroed gemini-embedding-001 + ScaNN (ADR-004/006)"
    steps = [
        "Stage 3: RT-DETR-v2 + SAM-3 pixel-accurate instance segmentation",
        "Stage 4: Background-zeroed gemini-embedding-001 + ScaNN vector lookup",
    ]
    skus = embeddings.SKUS

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        boxes = hul_domain.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.986, ctx=ctx, approach_name=self.name
        )
        ctx.bill("embedding_image", max(1, round(len(boxes) * 0.05)))
        ctx.trace.step("SAM-3 Mask + ScaNN", f"{len(boxes)} segmented & background-zeroed facings", boxes=boxes)
        return boxes


@register
class MaxViTClusteredDjevApproach(Approach):
    """ADR-007 + ADR-008 Benchmark Ablation: MaxViT Multi-Scale (Block+Grid) Attention + Complete-Linkage Clustering + dJev."""

    name = "maxvit_clustered_djev"
    task = "combined"
    epic = "MT Market Share - Combined Classification"
    architecture = (
        "MaxViT Multi-Scale Block+Grid Attention (ADR-007) + Complete-Linkage High-Purity Clustering (ADR-008) "
        "+ Dynamic HUL Catalog ScaNN + Stage 5 /v1/systemone 64-Token Canvas"
    )
    steps = [
        "Stage 3: RT-DETR-v2 + DIoU-NMS + Shelf-Row Consensus",
        "Stage 3.8: Complete-Linkage High-Purity Clustering (tau=0.94, Delta-E<=2.2) over MaxViT Block+Grid features",
        "Stage 4: Dynamic HUL Catalog ScaNN Lookup on Cluster Medoids",
        "Stage 4.5 & 5: MaxViT Attention-Weighted Sub-ROI CIELAB + /v1/systemone 64-Token Canvas",
    ]
    skus = embeddings.SKUS

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        from utils import maxvit_clustering

        proposals = hul_domain.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.988, ctx=ctx, approach_name=self.name
        )
        ctx.trace.step(
            "Stage 3: RT-DETR-v2 + Shelf-Row Consensus",
            f"{len(proposals)} product facings detected",
            boxes=proposals,
        )

        cluster_summary, crop_feats = maxvit_clustering.cluster_shelf_facings_high_purity(
            image, proposals, feature_mode="maxvit", tau=0.94
        )
        ctx.trace.step(
            "Stage 3.8: MaxViT Block+Grid Complete-Linkage Clustering (ADR-007/008)",
            f"{cluster_summary.total_facings} facings -> {cluster_summary.num_clusters} clusters "
            f"({cluster_summary.compression_ratio}x compression, purity={cluster_summary.estimated_node_purity:.3f})",
            boxes=[c.medoid_box for c in cluster_summary.clusters],
        )

        fast_scann_boxes: list[Box] = []
        sister_shade_boxes: list[Box] = []
        open_set_boxes: list[Box] = []

        for cluster in cluster_summary.clusters:
            med_idx = cluster.medoid_idx
            lookup = hul_domain.scann_vector_lookup(
                med_idx,
                cluster.medoid_box,
                use_ijepa_deglare=True,
                image=image,
                feature_mode="maxvit",
                precomputed_crop_feats=crop_feats[med_idx],
            )
            if lookup["routing_branch"] == "fast_scann":
                fast_scann_boxes.extend(cluster.member_boxes)
            elif lookup["routing_branch"] == "sister_shade_djev":
                sister_shade_boxes.extend(cluster.member_boxes)
            else:
                open_set_boxes.extend(cluster.member_boxes)

        ctx.bill("embedding_image", max(1, round(cluster_summary.num_clusters * 0.04)))
        ctx.trace.step(
            "Stage 4 & 5: MaxViT Medoid ScaNN + /v1/systemone",
            f"Fast ScaNN: {len(fast_scann_boxes)} | Sister-Shade dJev: {len(sister_shade_boxes)} | Open-Set: {len(open_set_boxes)}",
            boxes=proposals,
        )
        return proposals

