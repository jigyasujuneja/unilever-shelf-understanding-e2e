"""Plugin Registration for the 3-Stage Class-Agnostic Detection + Visual Crop Embedding + Vector Search Approach."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import time
from typing import Any, Dict, List, Optional
import uuid

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
from shelf_benchmark.tasks import facing_utils


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
        run_id = f"{uuid.uuid4().hex[:6]}-cv3s-{model_name.split('-')[-1]}"

        start_dt = datetime.now(timezone.utc)
        start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        t0 = time.perf_counter()

        pil_img = facing_utils.load_pil_image(
            ctx.storage,
            record.shelf_image_uri,
            local_fallback=record.local_shelf_image_path or "shelf-image.png",
        )
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # STAGE 1: Class-Agnostic ('product') Object Detection + Depth NMS
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

        end_dt = datetime.now(timezone.utc)
        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        total_extra_api_cost = round(s1_extra_cost + s2_embed_cost, 8)
        cost_metrics = ctx.compute_cost(
            tokens=s1_tokens,
            model_name=model_name,
            product_count=len(matched_facings),
            extra_api_cost_usd=total_extra_api_cost,
        )

        # Create temp trace/span IDs before OTel logger emits official hex IDs
        trace_id_hex, span_id_hex = uuid.uuid4().hex, uuid.uuid4().hex[:16]

        row_items: List[RowLevelReportItem] = []
        for item in matched_facings:
            bbox = item.get("bbox_2d", [0, 0, 0, 0])
            peer_idx = item.get("nearest_shelf_facing_idx")
            peer_sim = item.get("nearest_shelf_facing_visual_sim", 0.0)
            proto_sim = item.get("contrastive_prototype_sim_1408d", 0.0)
            row_items.append(
                RowLevelReportItem(
                    run_id=run_id,
                    trace_id=trace_id_hex,
                    span_id=span_id_hex,
                    task_type="classification",
                    separation_approach=self.approach_id,
                    model_name=model_name,
                    shelf_image_uri=record.shelf_image_uri,
                    store_id=record.store_id,
                    start_time=start_iso,
                    end_time=end_iso,
                    image_latency_ms=total_latency_ms,
                    product_index=int(item.get("product_index", 1)),
                    shelf_row=str(item.get("shelf_row", "middle")),
                    position_on_shelf=int(item.get("position_on_shelf", 1)),
                    bbox_ymin=int(bbox[0]),
                    bbox_xmin=int(bbox[1]),
                    bbox_ymax=int(bbox[2]),
                    bbox_xmax=int(bbox[3]),
                    crop_image_path=item.get("crop_image_path"),
                    predicted_category=item.get("predicted_category", "Skin Cleansing"),
                    predicted_subcategory=item.get("predicted_subcategory", "Face Wash"),
                    predicted_brand=item.get("predicted_brand", "Pond's"),
                    is_hul_brand=bool(item.get("is_hul_brand", True)),
                    predicted_variant=item.get("predicted_variant", ""),
                    predicted_packaging=item.get("predicted_packaging", "tube"),
                    predicted_pack_type=item.get("predicted_pack_type", "Single"),
                    predicted_size=item.get("predicted_size", ""),
                    rule_derived_size_bucket=item.get("rule_derived_size_bucket", ""),
                    predicted_product_name=f"[Class-Agnostic 'product' -> 1408-D ViT Crop Match] {item.get('predicted_brand')} {item.get('predicted_variant')}",
                    confidence=float(item.get("confidence", 0.95)),
                    lexical_search_keywords=f"class:product visual_peer_slot:#{peer_idx} (sim={peer_sim}) proto_sim={proto_sim}",
                    dense_embedding_text=f"1408-D Vertex AI multimodalembedding@001 Image Crop Vector | Closest Visual Shelf Twin: Slot #{peer_idx} (Cosine={peer_sim}) | Prototype Cosine={proto_sim}",
                    embedding_vector_dim=int(item.get("visual_embedding_dim", 1408)),
                    matched_sku_id=None,
                    planogram_compliant=None,
                    input_tokens=s1_tokens.input_tokens,
                    thinking_tokens=s1_tokens.thinking_tokens,
                    output_tokens=s1_tokens.output_tokens,
                    total_tokens=s1_tokens.total_tokens,
                    cost_per_shelf_image_usd=cost_metrics.cost_per_shelf_image_usd,
                    cost_per_product_usd=cost_metrics.cost_per_product_usd,
                )
            )

        acc_metrics = ctx.evaluate_accuracy(
            task_type="classification",
            rows=row_items,
            gt_record=gt_record,
            depth_duplicates_filtered=depth_filtered,
        )

        trace_id_hex, span_id_hex, _ = ctx.telemetry.log_task_execution(
            run_id=run_id,
            task_type=f"classification.{self.approach_id}",
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            start_dt=start_dt,
            end_dt=end_dt,
            tokens=s1_tokens,
            cost=cost_metrics,
            accuracy=acc_metrics,
            status="SUCCESS",
            extra_attributes={
                "shelf_benchmark.separation_approach": self.approach_id,
                "shelf_benchmark.stage1_latency_ms": stage1_latency_ms,
                "shelf_benchmark.stage2_latency_ms": stage2_latency_ms,
                "shelf_benchmark.stage3_latency_ms": stage3_latency_ms,
                "shelf_benchmark.visual_embedding_model": "multimodalembedding@001",
                "shelf_benchmark.visual_embedding_dim": 1408,
            },
        )
        for r in row_items:
            r.trace_id = trace_id_hex
            r.span_id = span_id_hex

        result = TaskExecutionResult(
            run_id=run_id,
            trace_id=trace_id_hex,
            span_id=span_id_hex,
            task_type="classification",
            separation_approach=self.approach_id,
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            start_time=start_iso,
            end_time=end_iso,
            latency_ms=total_latency_ms,
            tokens=s1_tokens,
            cost=cost_metrics,
            accuracy=acc_metrics,
            raw_output={
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
                "depth_duplicates_filtered": depth_filtered,
                "facings_summary": [
                    {
                        "product_index": f["product_index"],
                        "class_label": "product",
                        "bbox_2d": f["bbox_2d"],
                        "visual_embedding_dim": f["visual_embedding_dim"],
                        "nearest_shelf_facing_idx": f["nearest_shelf_facing_idx"],
                        "nearest_shelf_facing_visual_sim": f[
                            "nearest_shelf_facing_visual_sim"
                        ],
                        "contrastive_prototype_sim_1408d": f[
                            "contrastive_prototype_sim_1408d"
                        ],
                        "predicted_brand": f["predicted_brand"],
                        "predicted_variant": f["predicted_variant"],
                    }
                    for f in matched_facings
                ],
            },
            row_level_items=row_items,
            status="SUCCESS",
        )
        return result


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
