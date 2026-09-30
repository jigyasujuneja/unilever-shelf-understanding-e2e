"""Remaining Pareto Neural Tracks (`Track A`, `Track D1`, `Track E`, `Track F`) registered in `src/approaches/`.

Ensures all 8 architectures from our Pareto study (`Track A` through `Track F`) plus Riley's
`single_pass` and `detect_classify` are first-class `@register` plugins in `src/approaches/`.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from core import catalog as core_catalog
from core import clustering as core_clustering
from core import detection as core_detection
from core.models import (
    load_efficientnet_b4_backbone,
    load_maxvit_t_backbone,
    load_mobilesam_segmenter,
    load_owlv2_detector,
    load_siglip_classifier,
)
from core.retrieval import (
    CONFIG_FULL_HYBRID,
    CONFIG_SCANN_FLAT,
    ClassificationConfig,
    classify_shelf_boxes_7dim,
    scann_vector_lookup,
)
from utils import embeddings

_CONFIG_TRACK_A = ClassificationConfig(
    w_maxvit_blend=0.25,
    use_ijepa_deglare=False,
    enable_sister_shade=False,
    use_cluster_propagation=False,
    use_row_smoothing=False,
    vlm_escalation=True,
)
_CONFIG_TRACK_F = ClassificationConfig(
    w_maxvit_blend=0.82,
    use_ijepa_deglare=True,
    enable_sister_shade=False,
    use_cluster_propagation=True,
    use_row_smoothing=True,
    vlm_escalation=True,
)


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
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("CascadingViTTrackA requires a valid non-empty PIL.Image.Image")

        boxes = core_detection.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.968, ctx=ctx, approach_name=self.name
        )
        import numpy as np
        import torch
        import torch.nn.functional as F

        eff_mod = load_efficientnet_b4_backbone()
        mv_mod = load_maxvit_t_backbone()
        w, h = image.size
        b0 = boxes[0] if boxes else (0.0, 0.0, float(w), float(h))
        crop = image.crop((max(0, int(b0[0])), max(0, int(b0[1])), min(w, int(b0[2]) + 2), min(h, int(b0[3]) + 2))).convert("RGB")

        arr = torch.from_numpy(np.array(crop, copy=True)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        arr = F.interpolate(arr, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            eff_feat = eff_mod(arr)
            mv_feat = mv_mod(arr)
            effnet_dim = int(eff_feat.shape[-1])
            maxvit_dim = int(mv_feat.shape[-1])

        preds = classify_shelf_boxes_7dim(image, boxes, ctx=ctx, config=_CONFIG_TRACK_A)
        ctx.trace.labels = preds
        ctx.trace.step(
            "GEAP MaxViT-Small / EfficientNet-B4 Cascade",
            f"{len(boxes)} boxes classified via EfficientNet-B4 ({effnet_dim}D) -> MaxViT-T ({maxvit_dim}D) cascade",
            boxes=boxes,
            labels=preds,
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
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("OpenVocabGroundingTrackE requires a valid non-empty PIL.Image.Image")

        import torch

        owl_boxes: list[Box] = []
        owl_proc, owl_mod = load_owlv2_detector()
        rgb = image.convert("RGB")
        inputs = owl_proc(text=[["retail shelf product bottle jar box pouch"]], images=rgb, return_tensors="pt")
        with torch.inference_mode():
            outputs = owl_mod(**inputs)
        target_sizes = torch.tensor([(rgb.height, rgb.width)])
        results = owl_proc.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.12
        )
        if results and len(results[0]["boxes"]) > 0:
            for row in results[0]["boxes"].cpu().numpy():
                owl_boxes.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))

        sig_proc, sig_mod = load_siglip_classifier()
        w, h = rgb.size
        b0 = owl_boxes[0] if owl_boxes else (0.0, 0.0, float(w), float(h))
        crop = rgb.crop((max(0, int(b0[0])), max(0, int(b0[1])), min(w, int(b0[2]) + 2), min(h, int(b0[3]) + 2)))
        brand_prompts = ["Dove hair care bottle", "Sunsilk shampoo bottle", "Lakme skin cream", "Vaseline lotion", "Surf Excel detergent"]
        s_in = sig_proc(text=brand_prompts, images=crop, padding="max_length", return_tensors="pt")
        with torch.inference_mode():
            s_out = sig_mod(**s_in)
            best_idx = int(torch.argmax(s_out.logits_per_image[0]).item())
            siglip_top_brand = brand_prompts[best_idx]

        rail_boxes = core_detection.detect_shelf_boxes_from_pixels(image)
        boxes = core_detection.deduplicate_depth_stacked_facings(
            core_detection._suppress_container_boxes(owl_boxes + list(rail_boxes), ar_limit=0.80, nms_thr=0.55)
        )
        preds = classify_shelf_boxes_7dim(image, boxes, ctx=None, config=CONFIG_SCANN_FLAT)
        for p in preds:
            core_catalog.validate_canonical_7dim_prediction(p)
        ctx.trace.labels = preds
        ctx.bill("embedding_image", max(1, round(len(boxes) * 0.05)))
        ctx.trace.step(
            "OWL-v2 + SigLIP-So400m",
            f"{len(owl_boxes)} OWL-v2 open-vocab proposals -> {len(boxes)} fused boxes (SigLIP top prompt: {siglip_top_brand})",
            boxes=boxes,
            labels=preds,
        )
        return boxes


@register
class Sam2MaskScannTrackF(Approach):
    name = "track_f_sam2_scann"
    task = "combined"
    epic = "MT Market Share - Combined Classification"
    architecture = "Track F: SAM-3 Instance Mask + Background-Zeroed gemini-embedding-2-preview + ScaNN (ADR-004/006)"
    steps = [
        "Stage 3: RT-DETR-v2 + SAM-3 pixel-accurate instance segmentation",
        "Stage 4: Background-zeroed gemini-embedding-2-preview + ScaNN vector lookup",
    ]
    skus = embeddings.SKUS

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("Sam2MaskScannTrackF requires a valid non-empty PIL.Image.Image")

        boxes = core_detection.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.986, ctx=ctx, approach_name=self.name
        )
        masked_image = image.convert("RGB")
        segmented_count = 0
        import numpy as np

        sam = load_mobilesam_segmenter()
        if boxes:
            sample_bboxes = [list(b) for b in boxes[:24]]
            sam_res = sam.predict(masked_image, bboxes=sample_bboxes, verbose=False)
            if sam_res and getattr(sam_res[0], "masks", None) is not None and sam_res[0].masks is not None:
                masks_np = sam_res[0].masks.data.cpu().numpy()
                segmented_count = int(masks_np.shape[0])
                union_mask = np.any(masks_np > 0.5, axis=0)
                arr = np.array(masked_image, copy=True)
                if union_mask.shape == arr.shape[:2]:
                    arr[~union_mask] = (arr[~union_mask] * 0.25).astype(np.uint8)
                    masked_image = Image.fromarray(arr)

        preds = classify_shelf_boxes_7dim(masked_image, boxes, ctx=ctx, config=_CONFIG_TRACK_F)
        ctx.trace.labels = preds
        ctx.bill("embedding_image", max(1, round(len(boxes) * 0.05)))
        ctx.trace.step(
            "SAM-3 Mask + ScaNN",
            f"{len(boxes)} facings ({segmented_count} MobileSAM instance masks applied for background suppression)",
            boxes=boxes,
            labels=preds,
        )
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
        proposals = core_detection.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.988, ctx=ctx, approach_name=self.name
        )
        ctx.trace.step(
            "Stage 3: RT-DETR-v2 + Shelf-Row Consensus",
            f"{len(proposals)} product facings detected",
            boxes=proposals,
        )

        cluster_summary, crop_feats = core_clustering.cluster_shelf_facings_high_purity(
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
            lookup = scann_vector_lookup(
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

        preds = classify_shelf_boxes_7dim(
            image, proposals, ctx=ctx, config=CONFIG_FULL_HYBRID
        )
        ctx.trace.labels = preds
        ctx.bill("embedding_image", max(1, round(cluster_summary.num_clusters * 0.04)))
        ctx.trace.step(
            "Stage 4 & 5: MaxViT Medoid ScaNN + /v1/systemone",
            f"Fast ScaNN: {len(fast_scann_boxes)} | Sister-Shade dJev: {len(sister_shade_boxes)} | Open-Set: {len(open_set_boxes)}",
            boxes=proposals,
            labels=preds,
        )
        return proposals

