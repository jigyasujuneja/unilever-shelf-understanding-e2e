"""BenchmarkRunner orchestrating separated Detection, 7-Dimension Classification (across multiple Bounding-Box Separation approaches), Hybrid Search Matching, and Fine-Tuning."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.data.associations import create_association_provider
from shelf_benchmark.data.ground_truth import create_ground_truth_provider
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.models import ShelfAssociationRecord, TaskExecutionResult
from shelf_benchmark.reporting.generator import BenchmarkReportGenerator
from shelf_benchmark.run_ids import build_run_id
from shelf_benchmark.tasks import (
    GeminiFineTuningTask,
    ProductClassificationTask,
    ProductDetectionTask,
    ProductMatchingTask,
)
from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """High-level runner for the Shelf Understanding Benchmark Suite."""

    def __init__(self, config: Optional[BenchmarkConfig] = None, genai_client: Optional[object] = None):
        self.config = config or BenchmarkConfig()
        self.storage = StorageManager(
            project_id=self.config.gcp.project_id,
            bucket_config=self.config.buckets,
            offline=self.config.offline.enabled,
        )
        self.telemetry = OpenTelemetryBenchmarkLogger(
            config=self.config.telemetry,
            project_id=self.config.gcp.project_id,
            location=self.config.gcp.location,
            bucket_name=self.config.buckets.shelf_images_bucket,
        )
        self.association_provider = create_association_provider(
            assoc_config=self.config.associations,
            bucket_config=self.config.buckets,
            storage_manager=self.storage,
            project_id=self.config.gcp.project_id,
        )
        self.gt_provider = create_ground_truth_provider(
            gt_config=self.config.ground_truth,
            storage_manager=self.storage,
            project_id=self.config.gcp.project_id,
        )
        self.report_generator = BenchmarkReportGenerator.from_config(self.config)

        if genai_client is None and self.config.offline.enabled:
            from shelf_benchmark.sdk import UniversalGenAIClientAdapter, UniversalModelSpec
            from shelf_benchmark.testing import offline_universal_payload_handler

            genai_client = UniversalGenAIClientAdapter(
                spec=UniversalModelSpec(
                    model_id="offline-runner-model",
                    provider_family="custom_callable",
                    custom_handler=offline_universal_payload_handler,
                ),
                default_project=self.config.gcp.project_id,
                default_location=self.config.gcp.location,
            )
        self._genai_client = genai_client

        self.detection_task = ProductDetectionTask(
            self.config, self.storage, self.telemetry, genai_client=self._genai_client  # type: ignore[arg-type]
        )
        self.classification_task = ProductClassificationTask(
            self.config, self.storage, self.telemetry, genai_client=self._genai_client  # type: ignore[arg-type]
        )
        self.matching_task = ProductMatchingTask(
            self.config, self.storage, self.telemetry, genai_client=self._genai_client  # type: ignore[arg-type]
        )
        self.fine_tuning_task = GeminiFineTuningTask(
            self.config, self.storage, self.telemetry, genai_client=self._genai_client  # type: ignore[arg-type]
        )

    def resolve_records(
        self,
        shelf_image_uri: Optional[str] = None,
        upload_to_gcs: bool = False,
    ) -> List[ShelfAssociationRecord]:
        """Resolve input shelf image records from explicit CLI input or the configured AssociationProvider."""
        if shelf_image_uri:
            final_uri = shelf_image_uri
            if upload_to_gcs and not shelf_image_uri.startswith("gs://"):
                local_path = Path(shelf_image_uri)
                dest_uri = f"{self.config.buckets.shelf_images_bucket.rstrip('/')}/{local_path.name}"
                final_uri = self.storage.upload_file(local_path, dest_uri)
            return [
                ShelfAssociationRecord(
                    association_id="cli-input-001",
                    shelf_image_uri=final_uri,
                    local_shelf_image_path=shelf_image_uri if not shelf_image_uri.startswith("gs://") else None,
                    store_id="STORE-DEFAULT-01",
                    catalog_uri=None,
                    planogram_uri=None,
                    ground_truth_id=Path(shelf_image_uri).name,
                )
            ]
        records = self.association_provider.load_associations()
        if self.config.offline.enabled:
            from shelf_benchmark.testing import OFFLINE_IMAGE_URI

            for rec in records:
                if rec.shelf_image_uri.startswith("gs://"):
                    if rec.local_shelf_image_path and Path(rec.local_shelf_image_path).exists():
                        rec.shelf_image_uri = str(Path(rec.local_shelf_image_path).resolve())
                    else:
                        rec.shelf_image_uri = OFFLINE_IMAGE_URI
        return records

    def run_benchmark(
        self,
        tasks: Sequence[str] = ("detection", "classification", "matching", "fine_tuning"),
        models: Optional[Sequence[str]] = None,
        classification_approaches: Optional[Sequence[str]] = None,
        shelf_image_uri: Optional[str] = None,
        upload_to_gcs: bool = False,
        submit_live_tuning_job: bool = False,
    ) -> Dict[str, object]:
        """Run the specified tasks and bounding-box separation approaches across all target models and inputs."""
        active_models = list(models) if models else list(self.config.models)
        active_approaches = (
            list(classification_approaches)
            if classification_approaches is not None
            else list(self.config.approaches)
        )
        records = self.resolve_records(shelf_image_uri=shelf_image_uri, upload_to_gcs=upload_to_gcs)
        batch_id = uuid.uuid4().hex[:6]

        results: List[TaskExecutionResult] = []

        for rec in records:
            gt = self.gt_provider.get_ground_truth(rec.shelf_image_uri, rec.ground_truth_id)
            for model_name in active_models:
                for task_name in tasks:
                    t_norm = task_name.lower().strip()

                    if t_norm == "detection":
                        run_id = f"{batch_id}-det-{model_name.split('-')[-1]}"
                        logger.info(f"[BenchmarkRunner] task='detection' model='{model_name}' ...")
                        res = self.detection_task.execute(
                            model_name=model_name,
                            shelf_image_uri=rec.shelf_image_uri,
                            run_id=run_id,
                            store_id=rec.store_id,
                            ground_truth=gt,
                        )
                        self._log_progress(res)
                        results.append(res)

                    elif t_norm == "classification":
                        from shelf_benchmark.approaches import (
                            GLOBAL_APPROACH_REGISTRY,
                            CommonLayerContext,
                        )

                        ctx = CommonLayerContext(
                            config=self.config,
                            storage=self.storage,
                            telemetry=self.telemetry,
                            reports_dir=Path(self.config.reporting.output_dir),
                            genai_client=self._genai_client,
                        )
                        for approach in active_approaches:
                            run_id = build_run_id(
                                batch_id, "cls", approach, model_name.split("-")[-1]
                            )
                            logger.info(
                                f"[BenchmarkRunner] task='classification' approach='{approach}' model='{model_name}' ..."
                            )
                            plugin = GLOBAL_APPROACH_REGISTRY.require(approach)
                            res = plugin.execute(
                                ctx=ctx,
                                model_name=model_name,
                                record=rec,
                                gt_record=gt,
                            )
                            self._log_progress(res)
                            results.append(res)

                    elif t_norm == "matching":
                        run_id = f"{batch_id}-mat-{model_name.split('-')[-1]}"
                        logger.info(f"[BenchmarkRunner] task='matching' model='{model_name}' ...")
                        res = self.matching_task.execute(
                            model_name=model_name,
                            shelf_image_uri=rec.shelf_image_uri,
                            run_id=run_id,
                            store_id=rec.store_id,
                            ground_truth=gt,
                            catalog_uri=rec.catalog_uri,
                            planogram_uri=rec.planogram_uri,
                        )
                        self._log_progress(res)
                        results.append(res)

                    elif t_norm in ("fine_tuning", "tuning"):
                        run_id = f"{batch_id}-sft-{model_name.split('-')[-1]}"
                        logger.info(f"[BenchmarkRunner] task='fine_tuning' model='{model_name}' ...")
                        res = self.fine_tuning_task.execute(
                            model_name=model_name,
                            shelf_image_uri=rec.shelf_image_uri,
                            run_id=run_id,
                            store_id=rec.store_id,
                            ground_truth=gt,
                            submit_live_tuning_job=submit_live_tuning_job,
                        )
                        self._log_progress(res)
                        results.append(res)

                    else:
                        raise ValueError(f"Unknown benchmark task: {task_name}")

        self.telemetry.flush()
        report_paths = self.report_generator.generate_all_reports(results)
        return {
            "results": results,
            "report_paths": report_paths,
            "artifacts": report_paths,
            "reports": report_paths,
            "otel_log_path": str(self.telemetry.log_path),
        }

    @staticmethod
    def _log_progress(res: TaskExecutionResult) -> None:
        logger.info(
            f"  -> status={res.status} approach={res.separation_approach} "
            f"facings={res.cost.product_count} depth_filtered={res.accuracy.depth_duplicates_filtered} "
            f"latency={res.latency_ms:.1f}ms tokens(in={res.tokens.input_tokens}, "
            f"think={res.tokens.thinking_tokens}, out={res.tokens.output_tokens}) "
            f"cost_img=${res.cost.cost_per_shelf_image_usd:.6f} "
            f"cost_facing=${res.cost.cost_per_product_usd:.6f}"
        )
