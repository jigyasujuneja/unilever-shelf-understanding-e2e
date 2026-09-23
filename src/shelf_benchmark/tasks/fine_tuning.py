"""Separated Task 4: Gemini Supervised Fine-Tuning (`GeminiFineTuningTask`).

Provides end-to-end support for Vertex AI Gemini Supervised Fine-Tuning (SFT):
1. Generates Vertex AI-compliant SFT JSONL datasets (`contents` with `fileData` GCS URIs
   and target structured JSON outputs) from shelf images + ground truth annotations (or
   distilled zero-shot predictions when ground truth is still a placeholder).
2. Validates JSONL schema and uploads training/validation datasets to GCS.
3. Supports launching a live Vertex AI tuning job (`client.tunings.tune(...)`) OR running
   a structured tuning evaluation benchmark on GCP, complete with OpenTelemetry logs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from shelf_benchmark.evaluation.cost import extract_token_usage
from shelf_benchmark.models import (
    ImageGroundTruth,
    ProductClassificationOutput,
    RowLevelReportItem,
    TokenUsageMetrics,
)
from shelf_benchmark.tasks.base import BaseBenchmarkTask
from shelf_benchmark.tasks.classification import build_classification_prompt

logger = logging.getLogger(__name__)


class GeminiFineTuningTask(BaseBenchmarkTask):
    """Vertex AI Gemini Fine-Tuning dataset builder, job launcher, and benchmark evaluator."""

    task_type = "fine_tuning"
    default_separation_approach = "vertex_sft_fine_tuning"

    def build_sft_jsonl_example(
        self,
        shelf_image_uri: str,
        ground_truth: Optional[ImageGroundTruth] = None,
        distilled_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct a single Vertex AI Gemini SFT JSONL training example."""
        if ground_truth and ground_truth.items:
            target_items = []
            for idx, it in enumerate(ground_truth.items, start=1):
                # Only emit attributes the annotation actually carries. This block used to write
                # `"variant": it.sku_id` (a SKU id is not a variant) plus the constants
                # `"category": "Face Wash"` (which is a *sub*category, and the same one for every
                # product) and `"packaging_type": "Tube"` (the taxonomy uses lowercase "tube").
                # The real `category`, `subcategory`, `variant`, `packaging_type`, `pack_type` and
                # `size` fields were present on GroundTruthProductItem and simply never read, so a
                # model fine-tuned on this dataset learned to emit those two literals.
                item: Dict[str, Any] = {
                    "product_index": idx,
                    "bbox_2d": it.bbox_2d,
                    "shelf_row": it.shelf_row,
                    "position_on_shelf": idx,
                    "brand": it.brand,
                    "product_name": it.product_name,
                    "confidence": 1.0,
                }
                for key, value in (
                    ("category", it.category),
                    ("subcategory", it.subcategory),
                    ("variant", it.variant),
                    ("packaging_type", it.packaging_type),
                    ("pack_type", it.pack_type),
                    ("size", it.size),
                    ("sku_id", it.sku_id),
                ):
                    if value:
                        item[key] = value
                if it.extra_attributes:
                    item["extra_attributes"] = dict(it.extra_attributes)
                target_items.append(item)
            target_payload = {
                "total_classified_products": len(target_items),
                "distinct_brands_found": ground_truth.expected_brands,
                "classified_products": target_items,
            }
        elif distilled_output:
            target_payload = distilled_output
        else:
            target_payload = {
                "total_classified_products": 0,
                "distinct_brands_found": [],
                "classified_products": [],
            }

        dynamic_prompt = build_classification_prompt(self.config.taxonomy)
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": self.storage.guess_mime_type(shelf_image_uri),
                                "fileUri": shelf_image_uri,
                            }
                        },
                        {"text": dynamic_prompt},
                    ],
                },
                {
                    "role": "model",
                    "parts": [{"text": json.dumps(target_payload)}],
                },
            ]
        }

    def prepare_and_upload_sft_dataset(
        self,
        shelf_image_uri: str,
        ground_truth: Optional[ImageGroundTruth],
        distilled_output: Optional[Dict[str, Any]] = None,
        local_dir: str | Path = "reports/tuning_data",
        duplicate_single_example_times: int = 1,
    ) -> Tuple[str, str, int]:
        """Create a Vertex AI SFT JSONL dataset from ONE shelf image and upload it to GCS.

        This builds a single training example. `duplicate_single_example_times` writes that one
        example N times to clear Vertex AI's minimum-row check; it adds no information and the
        result is a smoke-test fixture, not a training set. It was previously named
        `num_synthetic_replicas` and defaulted to 16, which made a 16-line file of one repeated
        row look like a 16-example dataset in the report.

        Raises on upload failure -- the caller records a GCS URI in the report, and a silently
        swallowed exception meant that URI could point at nothing.
        """
        out_dir = Path(local_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        local_jsonl = out_dir / "shelf_sft_train.jsonl"

        example = self.build_sft_jsonl_example(
            shelf_image_uri=shelf_image_uri,
            ground_truth=ground_truth,
            distilled_output=distilled_output,
        )
        copies = max(1, int(duplicate_single_example_times))
        if copies > 1:
            logger.warning(
                "Writing the SAME SFT example %d times to '%s'. This is a smoke-test fixture to "
                "satisfy Vertex AI's minimum row count, not a training dataset: it contains one "
                "distinct shelf image and adds no supervision. Supply multiple annotated images "
                "to build a real dataset.",
                copies,
                local_jsonl,
            )
        with open(local_jsonl, "w", encoding="utf-8") as f:
            for _ in range(copies):
                f.write(json.dumps(example) + "\n")

        bucket_base = (
            self.config.buckets.artifacts_bucket
            or self.config.buckets.shelf_images_bucket
        ).rstrip("/")
        gcs_dest = f"{bucket_base}/tuning/shelf_sft_train.jsonl"
        self.storage.upload_file(local_jsonl, gcs_dest)
        return str(local_jsonl), gcs_dest, copies

    def submit_vertex_tuning_job(
        self,
        base_model: str,
        training_dataset_gcs_uri: str,
        tuned_model_display_name: str = "unilever-shelf-classifier-tuned",
        epochs: int = 1,
    ) -> Dict[str, Any]:
        """Submit a Supervised Fine-Tuning job on Vertex AI (`us-central1`)."""
        tuning_client = self.get_client(location=self.config.gcp.tuning_location)
        try:
            job = tuning_client.tunings.tune(
                base_model=base_model,
                training_dataset=types.TuningDataset(gcs_uri=training_dataset_gcs_uri),
                config=types.CreateTuningJobConfig(
                    tuned_model_display_name=tuned_model_display_name,
                    epoch_count=epochs,
                ),
            )
            return {
                "job_submitted": True,
                "job_name": getattr(job, "name", None),
                "state": str(getattr(job, "state", "JOB_STATE_PENDING")),
                "base_model": base_model,
                "training_dataset_gcs_uri": training_dataset_gcs_uri,
            }
        except Exception as exc:
            return {
                "job_submitted": False,
                "base_model": base_model,
                "training_dataset_gcs_uri": training_dataset_gcs_uri,
                "reason": str(exc),
            }

    def invoke_model(
        self,
        model_name: str,
        shelf_image_uri: str,
        ground_truth: Optional[ImageGroundTruth] = None,
        submit_live_tuning_job: bool = False,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], TokenUsageMetrics, List[RowLevelReportItem]]:
        """Run zero-shot structured inference, generate & upload Vertex AI SFT JSONL dataset to GCS, and optionally submit tuning job."""
        client = self.get_client()
        image_part = self.storage.to_genai_part(shelf_image_uri)
        system_instruction = (
            "You are a retail shelf understanding model configured for structured supervised fine-tuning (SFT). "
            "Extract all front-facing shelf products with their normalized 2D bounding boxes, brand names, "
            "product titles, variants, and packaging types directly from visual cues."
        )

        response = client.models.generate_content(
            model=model_name,
            contents=[image_part, build_classification_prompt(self.config.taxonomy)],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ProductClassificationOutput,
                temperature=0.0,
            ),
        )

        tokens = extract_token_usage(response)
        raw_text = response.text or "{}"
        parsed_dict = json.loads(raw_text)
        validated = ProductClassificationOutput.model_validate(parsed_dict)

        ft_cfg = self.config.fine_tuning
        local_jsonl, gcs_jsonl, jsonl_row_count = self.prepare_and_upload_sft_dataset(
            shelf_image_uri=shelf_image_uri,
            ground_truth=ground_truth,
            distilled_output=validated.model_dump(),
            duplicate_single_example_times=ft_cfg.duplicate_single_example_times,
        )

        tuning_job_info: Dict[str, Any] = {
            "sft_dataset_local_path": local_jsonl,
            "sft_dataset_gcs_uri": gcs_jsonl,
            # Distinct vs. written rows are reported separately. A single key called
            # "sft_example_count" that returned the duplication factor made a one-image fixture
            # indistinguishable from a real multi-image dataset in the report.
            "sft_distinct_example_count": 1,
            "sft_jsonl_row_count": jsonl_row_count,
            "label_source": "ground_truth" if (ground_truth and ground_truth.items) else "distilled_placeholder",
            "submit_live_tuning_job": submit_live_tuning_job,
        }
        if submit_live_tuning_job:
            tuning_job_info["vertex_tuning_job"] = self.submit_vertex_tuning_job(
                base_model=model_name,
                training_dataset_gcs_uri=gcs_jsonl,
                tuned_model_display_name=ft_cfg.tuned_model_display_name,
                epochs=ft_cfg.epochs,
            )

        rows: List[RowLevelReportItem] = []
        for idx, item in enumerate(validated.classified_products, start=1):
            box = item.bbox_2d if len(item.bbox_2d) >= 4 else [0, 0, 0, 0]
            rows.append(
                RowLevelReportItem(
                    run_id="",
                    trace_id="",
                    span_id="",
                    task_type=self.task_type,
                    model_name=model_name,
                    shelf_image_uri=shelf_image_uri,
                    start_time="",
                    end_time="",
                    image_latency_ms=0.0,
                    product_index=item.product_index or idx,
                    shelf_row=item.shelf_row,
                    position_on_shelf=item.position_on_shelf or idx,
                    bbox_ymin=box[0],
                    bbox_xmin=box[1],
                    bbox_ymax=box[2],
                    bbox_xmax=box[3],
                    predicted_brand=item.brand,
                    predicted_product_name=item.product_name,
                    predicted_variant=item.variant,
                    predicted_category=item.category,
                    predicted_packaging=item.packaging_type,
                    confidence=item.confidence,
                )
            )

        output_payload = validated.model_dump()
        output_payload["tuning_metadata"] = tuning_job_info
        return output_payload, tokens, rows
