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
        self._model_specs = {}
        from shelf_benchmark.sdk import UniversalModelSpec

        for alias, ep_cfg in self.config.model_endpoints.items():
            self._model_specs[alias] = UniversalModelSpec(
                model_id=ep_cfg.model_id or alias,
                display_name=alias,
                provider_family=ep_cfg.provider_family,
                endpoint_uri=ep_cfg.endpoint_uri,
                api_version=ep_cfg.api_version,
                location=ep_cfg.location,
            )

        self.detection_task = ProductDetectionTask(
            self.config, self.storage, self.telemetry, genai_client=self._genai_client
        )
        self.classification_task = ProductClassificationTask(
            self.config, self.storage, self.telemetry, genai_client=self._genai_client
        )
        self.matching_task = ProductMatchingTask(
            self.config, self.storage, self.telemetry, genai_client=self._genai_client
        )
        self.fine_tuning_task = GeminiFineTuningTask(
            self.config, self.storage, self.telemetry, genai_client=self._genai_client
        )

    def _get_client_for_model(self, model_name: str) -> Optional[object]:
        """Return the injected client or a UniversalGenAIClientAdapter when model_name maps to model_endpoints/GEAP/Gemma/Tuned Endpoint."""
        if self._genai_client is not None:
            return self._genai_client
        from shelf_benchmark.sdk import UniversalGenAIClientAdapter, UniversalModelSpec

        spec = self._model_specs.get(model_name)
        if spec is None:
            for s in self._model_specs.values():
                if s.model_id == model_name:
                    spec = s
                    break
        if spec is None:
            lower = model_name.lower()
            if model_name.startswith("projects/") and "/endpoints/" in model_name:
                spec = UniversalModelSpec(
                    model_id=model_name,
                    provider_family="vertex_tuned_endpoint",
                    endpoint_uri=model_name,
                )
            elif "gemma-4" in lower or "gemma_4" in lower:
                spec = UniversalModelSpec(
                    model_id="google/gemma-4-26b-a4b-it-maas",
                    display_name="gemma-4-26b-a4b-it",
                    provider_family="vertex_gemma_maas",
                    location="global",
                    endpoint_uri=f"https://aiplatform.googleapis.com/v1/projects/{self.config.gcp.project_id}/locations/global/endpoints/openapi/chat/completions",
                )
            elif "gemma" in lower:
                spec = UniversalModelSpec(
                    model_id=model_name,
                    provider_family="vertex_gemma",
                )
            elif any(tag in lower for tag in ("geap", "exp", "preview")):
                spec = UniversalModelSpec(
                    model_id=model_name,
                    provider_family="vertex_geap",
                    api_version="v1beta1",
                )
            else:
                return None
        return UniversalGenAIClientAdapter(
            spec=spec,
            default_project=self.config.gcp.project_id,
            default_location=self.config.gcp.location,
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
        prior_detection: Optional[TaskExecutionResult] = None,
        reuse_prior_detection: bool = False,
        detector_approach: Optional[str] = None,
        max_workers: int = 1,
    ) -> Dict[str, object]:
        """Run the specified tasks and bounding-box separation approaches across all target models and inputs."""
        from concurrent.futures import ThreadPoolExecutor

        from shelf_benchmark.approaches import (
            GLOBAL_APPROACH_REGISTRY,
            CommonLayerContext,
        )
        from shelf_benchmark.scoring import write_predictions_file

        active_models = list(models) if models else list(self.config.models)
        active_approaches = (
            list(classification_approaches)
            if classification_approaches is not None
            else list(self.config.approaches)
        )
        records = self.resolve_records(shelf_image_uri=shelf_image_uri, upload_to_gcs=upload_to_gcs)
        batch_id = uuid.uuid4().hex[:6]
        normalized_tasks = [t.lower().strip() for t in tasks]

        def _run_for_record(rec: ShelfAssociationRecord) -> List[TaskExecutionResult]:
            rec_results: List[TaskExecutionResult] = []
            gt = self.gt_provider.get_ground_truth(rec.shelf_image_uri, rec.ground_truth_id)
            for model_name in active_models:
                model_client = self._get_client_for_model(model_name)
                det_task = ProductDetectionTask(
                    self.config, self.storage, self.telemetry, genai_client=model_client
                )
                mat_task = ProductMatchingTask(
                    self.config, self.storage, self.telemetry, genai_client=model_client
                )
                ft_task = GeminiFineTuningTask(
                    self.config, self.storage, self.telemetry, genai_client=model_client
                )
                ctx = CommonLayerContext(
                    config=self.config,
                    storage=self.storage,
                    telemetry=self.telemetry,
                    reports_dir=Path(self.config.reporting.output_dir),
                    genai_client=model_client,
                )
                active_prior_det = prior_detection

                if detector_approach:
                    det_plugin = GLOBAL_APPROACH_REGISTRY.require(detector_approach)
                    active_prior_det = det_plugin.execute(
                        ctx=ctx,
                        model_name=model_name,
                        record=rec,
                        gt_record=gt,
                    )
                    if "detection" in normalized_tasks:
                        self._log_progress(active_prior_det)
                        rec_results.append(active_prior_det)

                for task_name in tasks:
                    t_norm = task_name.lower().strip()

                    if t_norm == "detection":
                        if detector_approach:
                            continue
                        run_id = f"{batch_id}-det-{model_name.split('-')[-1]}"
                        logger.info(f"[BenchmarkRunner] task='detection' model='{model_name}' ...")
                        res = det_task.execute(
                            model_name=model_name,
                            shelf_image_uri=rec.shelf_image_uri,
                            run_id=run_id,
                            store_id=rec.store_id,
                            ground_truth=gt,
                        )
                        self._log_progress(res)
                        rec_results.append(res)
                        if reuse_prior_detection and active_prior_det is None:
                            active_prior_det = res

                        for approach in active_approaches:
                            maybe_plugin = GLOBAL_APPROACH_REGISTRY.get(approach)
                            if (
                                maybe_plugin is not None
                                and getattr(maybe_plugin, "task_type", "classification") == "detection"
                            ):
                                det_plugin_res = maybe_plugin.execute(
                                    ctx=ctx,
                                    model_name=model_name,
                                    record=rec,
                                    gt_record=gt,
                                )
                                self._log_progress(det_plugin_res)
                                rec_results.append(det_plugin_res)
                                if reuse_prior_detection:
                                    active_prior_det = det_plugin_res

                    elif t_norm == "classification":
                        for approach in active_approaches:
                            plugin = GLOBAL_APPROACH_REGISTRY.require(approach)
                            if (
                                getattr(plugin, "task_type", "classification") == "detection"
                                and "detection" in normalized_tasks
                            ):
                                continue
                            run_id = build_run_id(
                                batch_id, "cls", approach, model_name.split("-")[-1]
                            )
                            logger.info(
                                f"[BenchmarkRunner] task='classification' approach='{approach}' model='{model_name}' ..."
                            )
                            res = plugin.execute(
                                ctx=ctx,
                                model_name=model_name,
                                record=rec,
                                gt_record=gt,
                                prior_detection=active_prior_det,
                            )
                            self._log_progress(res)
                            rec_results.append(res)

                    elif t_norm == "matching":
                        run_id = f"{batch_id}-mat-{model_name.split('-')[-1]}"
                        logger.info(f"[BenchmarkRunner] task='matching' model='{model_name}' ...")
                        res = mat_task.execute(
                            model_name=model_name,
                            shelf_image_uri=rec.shelf_image_uri,
                            run_id=run_id,
                            store_id=rec.store_id,
                            ground_truth=gt,
                            catalog_uri=rec.catalog_uri,
                            planogram_uri=rec.planogram_uri,
                        )
                        self._log_progress(res)
                        rec_results.append(res)

                    elif t_norm in ("fine_tuning", "tuning"):
                        run_id = f"{batch_id}-sft-{model_name.split('-')[-1]}"
                        logger.info(f"[BenchmarkRunner] task='fine_tuning' model='{model_name}' ...")
                        res = ft_task.execute(
                            model_name=model_name,
                            shelf_image_uri=rec.shelf_image_uri,
                            run_id=run_id,
                            store_id=rec.store_id,
                            ground_truth=gt,
                            submit_live_tuning_job=submit_live_tuning_job,
                        )
                        self._log_progress(res)
                        rec_results.append(res)

                    else:
                        raise ValueError(f"Unknown benchmark task: {task_name}")
            return rec_results

        results: List[TaskExecutionResult] = []
        pred_checkpoint_path = Path(self.config.reporting.output_dir) / "predictions.json"

        if max_workers > 1 and len(records) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for rec_batch in pool.map(_run_for_record, records):
                    results.extend(rec_batch)
                    if self.config.reporting.write_predictions_file:
                        write_predictions_file(results, pred_checkpoint_path, config=self.config)
        else:
            for rec in records:
                results.extend(_run_for_record(rec))
                if len(records) > 1 and self.config.reporting.write_predictions_file:
                    write_predictions_file(results, pred_checkpoint_path, config=self.config)

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
