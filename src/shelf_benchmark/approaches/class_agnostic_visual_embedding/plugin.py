"""Plugin Registration for the 3-Stage Class-Agnostic Detection + Visual Crop Embedding + Vector Search Approach."""

from __future__ import annotations

import io
import time
from typing import List, Optional

from shelf_benchmark.approaches.base import BaseShelfApproachPlugin, CommonLayerContext
from shelf_benchmark.approaches.class_agnostic_visual_embedding.stage1_class_agnostic_detector import (
    run_stage1_class_agnostic_detection,
)
from shelf_benchmark.approaches.class_agnostic_visual_embedding.stage2_visual_crop_embedder import (
    run_stage2_visual_crop_embedding,
)
from shelf_benchmark.approaches.class_agnostic_visual_embedding.stage3_vector_search_matcher import (
    run_stage3_vector_search_matching,
)
from shelf_benchmark.models import (
    ImageGroundTruth,
    RowLevelReportItem,
    ShelfAssociationRecord,
    TaskExecutionResult,
)


class ClassAgnosticVisualEmbeddingApproach(BaseShelfApproachPlugin):
    """Implements the classic 3-Stage Retail Shelf CV Pipeline on Google Cloud:
    - Stage 1: Class-Agnostic Object Detection (single class: 'product') + Front-Facing Depth NMS
    - Stage 2: Cropped Bounding-Box Visual Metric Learning (`multimodalembedding@001` 1408-D ViT vector)
    - Stage 3: Vector Search / Catalog Matching (ScaNN / Cosine ANN in 1408-D visual space)
    """

    def __init__(
        self,
        detector_backend: str = "google_open_vocab_class_agnostic_box2d",
        custom_id: str = "class_agnostic_visual_embedding",
        custom_title: str = "3-Stage Class-Agnostic Detector + 1408-D Visual Crop Embedding + ScaNN Vector Search",
    ) -> None:
        self._detector_backend = detector_backend
        self._id = custom_id
        self._title = custom_title

    @property
    def approach_id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._title

    @property
    def category(self) -> str:
        return "classic_cv_metric_learning"

    @property
    def stages_description(self) -> List[str]:
        return [
            f"Stage 1: Class-Agnostic Object Detection (single class: 'product' via {self._detector_backend}) + Front-Facing Column Depth NMS",
            "Stage 2: Physical Bounding-Box Crop Extraction + Vertex AI multimodalembedding@001 (1408-D ViT Contrastive Metric Learning)",
            "Stage 3: ScaNN / Vertex AI Vector Search (1408-D Image-to-Image Cosine ANN + Reference Catalog Matching)",
        ]

    def execute(
        self,
        ctx: CommonLayerContext,
        model_name: str,
        record: ShelfAssociationRecord,
        gt_record: Optional[ImageGroundTruth] = None,
        prior_detection: Optional[TaskExecutionResult] = None,
    ) -> TaskExecutionResult:
        from shelf_benchmark.run_ids import build_run_id

        run_id = build_run_id(self.approach_id, model_name)
        start_dt = ctx.telemetry.now_utc()
        cpu_start_sec = time.process_time()

        pil_img = ctx.load_shelf_image(record)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # STAGE 1: Class-Agnostic ('product') Object Detection + Depth NMS (or reuse prior_detection)
        prior_boxes = ctx.get_prior_detected_boxes(prior_detection) if prior_detection else None
        if prior_detection is not None and prior_boxes:
            kept_facings = prior_boxes
            depth_filtered = int(prior_detection.accuracy.depth_duplicates_filtered or 0)
            s1_tokens = prior_detection.tokens
            s1_extra_cost = float(prior_detection.cost.vertex_ai_embeddings_and_vision_usd or 0.0)
            stage1_latency_ms = round(float(prior_detection.latency_ms or 0.0), 2)
        else:
            t_s1 = time.perf_counter()
            kept_facings, depth_filtered, s1_tokens, s1_extra_cost = (
                run_stage1_class_agnostic_detection(
                    ctx=ctx,
                    image_bytes=image_bytes,
                    model_name=model_name,
                    detector_backend=self._detector_backend,
                )
            )
            stage1_latency_ms = round((time.perf_counter() - t_s1) * 1000.0, 2)

        # STAGE 2: Physical Crop Extraction + 1408-D Visual Metric Learning (`multimodalembedding@001`)
        t_s2 = time.perf_counter()
        model_tag = f"{model_name}_{self.approach_id}"
        embedded_facings, montage_path, s2_embed_cost = (
            run_stage2_visual_crop_embedding(
                ctx=ctx,
                shelf_image_uri=record.shelf_image_uri,
                detected_facings=kept_facings,
                model_tag=model_tag,
                embedding_location="us-central1",
            )
        )
        stage2_latency_ms = round((time.perf_counter() - t_s2) * 1000.0, 2)

        # STAGE 3: ScaNN / Cosine ANN Vector Search & Visual Clustering
        t_s3 = time.perf_counter()
        matched_facings = run_stage3_vector_search_matching(
            ctx=ctx,
            embedded_facings=embedded_facings,
        )
        stage3_latency_ms = round((time.perf_counter() - t_s3) * 1000.0, 2)

        end_dt = ctx.telemetry.now_utc()
        cpu_active_ms = round(max(0.0, time.process_time() - cpu_start_sec) * 1000.0, 3)

        # Only prediction fields are populated here; run_id, timings, tokens, 5-bucket GCP cost,
        # ground-truth scoring, OTel span and execution-trace provenance are all stamped by the
        # shared `ctx.finalize()` pipeline (this method used to assemble them inline -- the third
        # independent copy in the codebase).
        row_items: List[RowLevelReportItem] = []
        for item in matched_facings:
            bbox = item.get("bbox_2d") or [0, 0, 0, 0]
            peer_idx = item.get("nearest_shelf_facing_idx")
            peer_sim = item.get("nearest_shelf_facing_visual_sim", 0.0)
            match_sim = item.get("catalog_match_similarity_1408d", 0.0)
            catalog_status = item.get("catalog_status", "NO_CATALOG_INDEXED")
            row_items.append(
                RowLevelReportItem(
                    product_index=int(item.get("product_index", 1)),
                    shelf_row=str(item.get("shelf_row", "")),
                    position_on_shelf=int(item.get("position_on_shelf", 1)),
                    bbox_ymin=int(bbox[0]),
                    bbox_xmin=int(bbox[1]),
                    bbox_ymax=int(bbox[2]),
                    bbox_xmax=int(bbox[3]),
                    crop_image_path=item.get("crop_image_path"),
                    predicted_category=item.get("predicted_category", ""),
                    predicted_subcategory=item.get("predicted_subcategory", ""),
                    predicted_brand=item.get("predicted_brand", "") or "",
                    is_hul_brand=item.get("is_hul_brand"),
                    predicted_variant=item.get("predicted_variant", ""),
                    predicted_packaging=item.get("predicted_packaging", ""),
                    predicted_pack_type=item.get("predicted_pack_type", ""),
                    predicted_size=item.get("predicted_size", ""),
                    rule_derived_size_bucket=item.get("rule_derived_size_bucket", ""),
                    predicted_product_name=" ".join(
                        p for p in (item.get("predicted_brand"), item.get("predicted_variant")) if p
                    ),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    lexical_search_keywords=(
                        f"class:product visual_peer_slot:#{peer_idx} (sim={peer_sim}) "
                        f"catalog_status={catalog_status} catalog_sim={match_sim}"
                    ),
                    dense_embedding_text=(
                        f"1408-D multimodalembedding@001 crop vector | "
                        f"closest shelf twin: slot #{peer_idx} (cosine={peer_sim}) | "
                        f"catalog match: {catalog_status} (cosine={match_sim})"
                    ),
                    embedding_vector_dim=int(item.get("visual_embedding_dim", 1408)),
                    matched_sku_id=item.get("matched_sku_id"),
                )
            )

        return ctx.finalize(
            approach_id=self.approach_id,
            model_name=model_name,
            record=record,
            rows=row_items,
            start_dt=start_dt,
            end_dt=end_dt,
            gt_record=gt_record,
            run_id=run_id,
            tokens=s1_tokens,
            extra_api_cost_usd=round(s1_extra_cost + s2_embed_cost, 8),
            depth_duplicates_filtered=depth_filtered,
            stages_description=self.stages_description,
            cpu_active_ms=cpu_active_ms,
            span_attributes={
                "shelf_benchmark.stage1_latency_ms": stage1_latency_ms,
                "shelf_benchmark.stage2_latency_ms": stage2_latency_ms,
                "shelf_benchmark.stage3_latency_ms": stage3_latency_ms,
                "shelf_benchmark.visual_embedding_model": "multimodalembedding@001",
                "shelf_benchmark.visual_embedding_dim": 1408,
            },
            raw_output_extra={
                "approach_id": self.approach_id,
                "display_name": self.display_name,
                "detector_backend": self._detector_backend,
                "stage1_class_agnostic_latency_ms": stage1_latency_ms,
                "stage2_visual_crop_embedding_latency_ms": stage2_latency_ms,
                "stage3_scann_vector_search_latency_ms": stage3_latency_ms,
                "stage2_crop_embedding_cost_usd": s2_embed_cost,
                "visual_embedding_model": "multimodalembedding@001",
                "visual_embedding_dimension": 1408,
                "montage_path": montage_path,
                "front_facings_detected": len(matched_facings),
                "catalog_status": (
                    matched_facings[0].get("catalog_status") if matched_facings else "NO_FACINGS"
                ),
                "reference_catalog_uri": ctx.config.embeddings.reference_catalog.source_uri,
                "facings_summary": [
                    {
                        "product_index": f["product_index"],
                        "class_label": "product",
                        "bbox_2d": f["bbox_2d"],
                        "visual_embedding_dim": f["visual_embedding_dim"],
                        "nearest_shelf_facing_idx": f["nearest_shelf_facing_idx"],
                        "nearest_shelf_facing_visual_sim": f["nearest_shelf_facing_visual_sim"],
                        "catalog_match_similarity_1408d": f["catalog_match_similarity_1408d"],
                        "catalog_status": f["catalog_status"],
                        "matched_sku_id": f.get("matched_sku_id"),
                        "predicted_brand": f["predicted_brand"],
                        "predicted_variant": f["predicted_variant"],
                    }
                    for f in matched_facings
                ],
            },
        )

def get_plugins() -> List[BaseShelfApproachPlugin]:
    """Returns the registered plugins for this approach directory."""
    return [
        ClassAgnosticVisualEmbeddingApproach(
            detector_backend="google_open_vocab_class_agnostic_box2d",
            custom_id="class_agnostic_visual_embedding",
            custom_title="3-Stage Class-Agnostic Detector + 1408-D Visual Crop Embedding (multimodalembedding@001) + ScaNN Vector Search",
        ),
        ClassAgnosticVisualEmbeddingApproach(
            detector_backend="cloud_vision_object_localization",
            custom_id="cloud_vision_visual_embedding",
            custom_title="3-Stage Google Cloud Vision OBJECT_LOCALIZATION + 1408-D Visual Crop Embedding + ScaNN Vector Search",
        ),
    ]
