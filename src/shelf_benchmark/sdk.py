"""High-Level Developer SDK & Universal Model Router for the Unilever Shelf Benchmark Suite.

Provides a unified, easy-to-use Python API (`ShelfBenchmarkSDK`, `UniversalModelSpec`, `UniversalGenAIClientAdapter`,
and `@register_approach_function`) so developers can benchmark:
- Standard Vertex AI Gemini models (`gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.5-flash-lite`, etc.)
- Google Early Access Program (GEAP) & Preview / Experimental models (`v1beta1`, custom endpoints, early-access IDs)
- Open-Weights & Model Garden models (`Gemma 3`, `Gemma 2`, `PaliGemma`, Vertex AI Endpoints, or custom Python callables)
- Supervised Fine-Tuned (SFT) Vertex AI models (`projects/.../locations/.../endpoints/...`)
- Custom multi-stage CV/VLM/Embedding approaches

Every run automatically emits OpenTelemetry-compliant spans & JSONL logs (`start_time`, `end_time`,
`input_tokens`, `thinking_tokens`, `output_tokens`, `cost_per_shelf_image_usd`, `cost_per_product_usd`)
and generates row-level CSV/JSON and Markdown summary reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from google import genai
from google.genai import types

from shelf_benchmark.approaches import (
    GLOBAL_APPROACH_REGISTRY,
    BaseShelfApproachPlugin,
    CommonLayerContext,
)
from shelf_benchmark.config import BenchmarkConfig, ModelPricing, TaxonomyConfig
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.models import (
    ImageGroundTruth,
    ShelfAssociationRecord,
    TaskExecutionResult,
)
from shelf_benchmark.reporting.generator import BenchmarkReportGenerator
from shelf_benchmark.tasks.detection import ProductDetectionTask
from shelf_benchmark.tasks.fine_tuning import GeminiFineTuningTask
from shelf_benchmark.tasks.matching import ProductMatchingTask
from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger

logger = logging.getLogger(__name__)

ModelProviderFamily = Literal[
    "vertex_gemini",
    "vertex_geap",
    "vertex_gemma",
    "vertex_tuned_endpoint",
    "custom_callable",
]


@dataclass
class UniversalModelSpec:
    """Declarative specification for any model (Gemini, GEAP, Gemma, Fine-Tuned Endpoint, or Custom Callable)."""

    model_id: str
    provider_family: ModelProviderFamily = "vertex_gemini"
    display_name: Optional[str] = None
    project_id: Optional[str] = None
    location: Optional[str] = None
    api_version: Optional[str] = None  # e.g. "v1beta1" or "v1alpha" for GEAP preview models
    endpoint_uri: Optional[str] = None  # e.g. "projects/123/locations/us-central1/endpoints/456"
    pricing: Optional[ModelPricing] = None
    custom_handler: Optional[Callable[[str, str, Optional[Any]], Dict[str, Any]]] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_name(self) -> str:
        return self.display_name or self.model_id


class _NormalizedUsageMetadata:
    """OpenTelemetry-compatible token usage metadata wrapper for any model backend."""

    def __init__(self, input_tokens: int, thinking_tokens: int, output_tokens: int):
        self.prompt_token_count = input_tokens
        self.thoughts_token_count = thinking_tokens
        self.candidates_token_count = output_tokens
        self.total_token_count = input_tokens + thinking_tokens + output_tokens


class _NormalizedModelResponse:
    """Standardized response wrapper compatible with all `shelf_benchmark` tasks."""

    def __init__(
        self,
        text: str,
        input_tokens: int = 250,
        thinking_tokens: int = 0,
        output_tokens: int = 180,
    ):
        self.text = text
        self.usage_metadata = _NormalizedUsageMetadata(
            input_tokens=input_tokens,
            thinking_tokens=thinking_tokens,
            output_tokens=output_tokens,
        )


class UniversalGenAIClientAdapter:
    """Wraps GEAP models, Gemma endpoints, Fine-Tuned endpoints, or custom Python callables into the standard `genai.Client` interface."""

    def __init__(self, spec: UniversalModelSpec, default_project: str, default_location: str):
        self.spec = spec
        self.project_id = spec.project_id or default_project
        self.location = spec.location or default_location
        self._underlying_client: Optional[genai.Client] = None

        if spec.provider_family in ("vertex_gemini", "vertex_geap", "vertex_gemma", "vertex_tuned_endpoint"):
            http_opts = types.HttpOptions(api_version=spec.api_version) if spec.api_version else None
            self._underlying_client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
                http_options=http_opts,
            )

        self.models = self

    def generate_content(
        self,
        model: str,
        contents: Any,
        config: Optional[types.GenerateContentConfig] = None,
    ) -> Any:
        """Unified `generate_content` supporting Gemini, GEAP, Gemma, Tuned Endpoints, and custom handlers."""
        target_model = self.spec.endpoint_uri or self.spec.model_id or model

        # 1. Custom Python Callable Handler (e.g., local Gemma vLLM / HuggingFace / custom endpoint)
        if self.spec.provider_family == "custom_callable" or self.spec.custom_handler is not None:
            if self.spec.custom_handler is None:
                raise ValueError("UniversalModelSpec with provider_family='custom_callable' requires `custom_handler`.")
            prompt_text = ""
            image_uri = ""
            for item in (contents if isinstance(contents, list) else [contents]):
                if isinstance(item, str):
                    prompt_text += item + "\n"
                elif hasattr(item, "file_data") and item.file_data:
                    image_uri = getattr(item.file_data, "file_uri", "") or ""
            schema = getattr(config, "response_schema", None) if config else None
            raw_out = self.spec.custom_handler(prompt_text.strip(), image_uri, schema)
            if isinstance(raw_out, dict):
                usage = raw_out.pop("_token_usage", {})
                text_payload = json.dumps(raw_out)
                return _NormalizedModelResponse(
                    text=text_payload,
                    input_tokens=int(usage.get("input_tokens", max(120, len(prompt_text) // 4))),
                    thinking_tokens=int(usage.get("thinking_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", max(80, len(text_payload) // 4))),
                )
            text_str = str(raw_out)
            return _NormalizedModelResponse(
                text=text_str,
                input_tokens=max(120, len(prompt_text) // 4),
                thinking_tokens=0,
                output_tokens=max(80, len(text_str) // 4),
            )

        # 2. Gemma models on Vertex AI (which may not support strict `response_schema` JSON mode in all versions)
        if self.spec.provider_family == "vertex_gemma" and config is not None:
            schema = getattr(config, "response_schema", None)
            schema_instruction = ""
            if schema is not None and hasattr(schema, "model_json_schema"):
                schema_instruction = (
                    "\n\nIMPORTANT FOR GEMMA: Return ONLY valid JSON matching this JSON schema:\n"
                    + json.dumps(schema.model_json_schema())
                )
            patched_contents = list(contents) if isinstance(contents, list) else [contents]
            if schema_instruction:
                patched_contents.append(schema_instruction)
            gemma_cfg = types.GenerateContentConfig(
                temperature=getattr(config, "temperature", 0.1),
                response_mime_type="application/json",
            )
            assert self._underlying_client is not None
            return self._underlying_client.models.generate_content(
                model=target_model,
                contents=patched_contents,
                config=gemma_cfg,
            )

        # 3. Standard Vertex AI Gemini, GEAP (v1beta1/v1alpha), or Vertex AI Fine-Tuned Endpoint (`projects/.../endpoints/...`)
        assert self._underlying_client is not None
        return self._underlying_client.models.generate_content(
            model=target_model,
            contents=contents,
            config=config,
        )


def register_approach_function(
    approach_id: str,
    display_name: str,
    category: Literal["vlm_multimodal", "two_stage_vlm", "classic_cv_metric_learning"] = "two_stage_vlm",
    stages_description: Optional[List[str]] = None,
) -> Callable[[Callable[..., List[Dict[str, Any]]]], BaseShelfApproachPlugin]:
    """Decorator to turn a simple Python function `(ctx, model_name, shelf_image_uri) -> List[dict]` into a full Benchmark Approach Plugin with automatic OTel logging, depth deduplication, size rules, and cost calculation."""

    def decorator(func: Callable[..., List[Dict[str, Any]]]) -> BaseShelfApproachPlugin:
        _approach_id = approach_id
        _display_name = display_name
        _category = category
        _stages = stages_description or [
            f"Stage 1: Custom Function Pipeline ({approach_id})"
        ]

        class FunctionApproachPlugin(BaseShelfApproachPlugin):
            @property
            def approach_id(self) -> str:
                return _approach_id

            @property
            def display_name(self) -> str:
                return _display_name

            @property
            def category(self) -> str:
                return _category

            @property
            def stages_description(self) -> List[str]:
                return _stages

            def execute(
                self,
                ctx: CommonLayerContext,
                model_name: str,
                record: ShelfAssociationRecord,
                gt_record: Optional[ImageGroundTruth] = None,
                prior_detection: Optional[TaskExecutionResult] = None,
            ) -> TaskExecutionResult:
                start_dt = ctx.telemetry.now_utc()
                raw_outputs = func(ctx, model_name, record.shelf_image_uri)
                end_dt = ctx.telemetry.now_utc()
                return ctx.finalize(
                    approach_id=approach_id,
                    model_name=model_name,
                    record=record,
                    raw_outputs=raw_outputs,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    gt_record=gt_record,
                    run_id=f"custom-{approach_id}-{model_name}",
                    stages_description=_stages,
                )

        plugin_instance = FunctionApproachPlugin()
        GLOBAL_APPROACH_REGISTRY.register(plugin_instance)
        return plugin_instance

    return decorator


class ShelfBenchmarkSDK:
    """High-level Developer Facade for running the Shelf Understanding Benchmark Suite with any model or approach."""

    def __init__(
        self,
        config_path: str | Path = "configs/default_config.yaml",
        taxonomy_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
    ):
        config = BenchmarkConfig.from_yaml(config_path)
        if taxonomy_path is not None:
            config.taxonomy = TaxonomyConfig.from_yaml_or_defaults(taxonomy_path)
        self._init_from_config(config, output_dir=output_dir)

    @classmethod
    def from_config(
        cls,
        config: BenchmarkConfig,
        output_dir: Optional[str | Path] = None,
    ) -> "ShelfBenchmarkSDK":
        """Build an SDK around an already-constructed config.

        Use this when the config is assembled in code rather than read from YAML, notably
        `shelf_benchmark.testing.offline_config()` for laptop runs with no GCP access.
        """
        sdk = cls.__new__(cls)
        sdk._init_from_config(config, output_dir=output_dir)
        return sdk

    def _init_from_config(
        self,
        config: BenchmarkConfig,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        from shelf_benchmark.data.ground_truth import create_ground_truth_provider

        self.config = config
        if output_dir is not None:
            self.config.reporting.output_dir = str(output_dir)
            self.config.telemetry.otel_log_path = str(Path(output_dir) / "otel_logs.jsonl")

        self.storage = StorageManager(
            project_id=self.config.gcp.project_id,
            bucket_config=self.config.buckets,
            offline=self.config.offline.enabled,
        )
        self.telemetry = OpenTelemetryBenchmarkLogger(
            config=self.config.telemetry,
            project_id=self.config.gcp.project_id,
            location=self.config.gcp.location,
        )
        self.gt_provider = create_ground_truth_provider(
            gt_config=self.config.ground_truth,
            storage_manager=self.storage,
            project_id=self.config.gcp.project_id,
        )
        logger.info("Ground truth: %s", self.gt_provider.describe())
        # `from_config` so reporting settings (isolate_runs, GCS sync, predictions file) are all
        # honoured. Passing only output_dir previously dropped the rest on the floor.
        self.report_generator = BenchmarkReportGenerator.from_config(self.config)
        # Built-in approaches must be discovered before any user code registers its own, otherwise
        # a custom registration at import time can hide them.
        GLOBAL_APPROACH_REGISTRY.ensure_discovered()

        self._dataset_records: Optional[List[ShelfAssociationRecord]] = None
        self._model_specs: Dict[str, UniversalModelSpec] = {}
        for alias, ep_cfg in self.config.model_endpoints.items():
            self._model_specs[alias] = UniversalModelSpec(
                model_id=ep_cfg.model_id or alias,
                display_name=alias,
                provider_family=ep_cfg.provider_family,  # type: ignore[arg-type]
                endpoint_uri=ep_cfg.endpoint_uri,
                api_version=ep_cfg.api_version,
                location=ep_cfg.location,
            )

    def connect_dataset(
        self,
        provider_type: Literal["default", "json", "csv", "bigquery", "gcs_bucket"] = "json",
        source_uri: Optional[str] = None,
        schema_mapping: Optional[Dict[str, str]] = None,
        shelf_images_bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plug in any GCP dataset or shelf image manifest at runtime regardless of table/file schema.

        Supports BigQuery tables or SQL queries (`provider_type="bigquery"`), GCS image buckets
        (`provider_type="gcs_bucket"`), CSV files (`provider_type="csv"`), and JSON/JSONL manifests
        (`provider_type="json"`), including nested dot-paths (e.g. `metadata.store_id`) and
        automatic resolution of relative image filenames against `shelf_images_bucket`.
        """
        from shelf_benchmark.data.associations import create_association_provider

        self.config.associations.provider_type = provider_type
        self.config.associations.source_uri = source_uri
        if shelf_images_bucket is not None:
            self.config.buckets.shelf_images_bucket = shelf_images_bucket
        if schema_mapping:
            unknown = [
                k for k in schema_mapping
                if not hasattr(self.config.associations.schema_mapping, k)
            ]
            if unknown:
                valid = sorted(type(self.config.associations.schema_mapping).model_fields)
                raise ValueError(
                    f"Unknown dataset schema_mapping key(s): {unknown}. Valid keys: {valid}"
                )
            for k, v in schema_mapping.items():
                setattr(self.config.associations.schema_mapping, k, v)

        assoc_provider = create_association_provider(
            assoc_config=self.config.associations,
            bucket_config=self.config.buckets,
            storage_manager=self.storage,
            project_id=self.config.gcp.project_id,
        )
        records = assoc_provider.load_associations()
        self._dataset_records = records
        return {
            "provider_type": provider_type,
            "source_uri": source_uri or self.config.buckets.shelf_images_bucket,
            "records_loaded": len(records),
            "records": records,
            "image_uris": [r.shelf_image_uri for r in records],
        }

    def connect_ground_truth(
        self,
        provider_type: Literal["json", "jsonl", "csv", "coco", "bigquery", "none"] = "json",
        source_uri: Optional[str] = None,
        schema_mapping: Optional[Dict[str, str]] = None,
        bbox_format: Optional[str] = None,
        gt_version: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Plug in ground truth at runtime and report what was actually loaded.

        This is the seam the whole suite is built around: until annotations exist, runs produce
        predictions with `accuracy_status="NO_GROUND_TRUTH"`; the moment a file lands, point this
        at it and every metric populates with no other code change.

        Args:
            provider_type: Backing store. `coco` reads a standard COCO detection JSON.
            source_uri: Local path or GCS URI of the annotations.
            schema_mapping: Overrides for field names, mapping `GroundTruthSchemaMapping`
                attribute names onto the names your annotator used, e.g.
                `{"brand_field": "manufacturer", "bbox_field": "box"}`.
            bbox_format: One of `ymin_xmin_ymax_xmax_1000` (suite-native), `coco_xywh_px`,
                `xyxy_px`, `xyxy_norm`, `yxyx_norm`. Declare this: guessing it wrong is the single
                most common cause of detection metrics that look plausible but are meaningless.
            gt_version: Label stamped on every report row and OTel span, so results computed
                against different annotation revisions are never silently compared.
            strict: Raise if the source cannot be read or yields zero images.

        Returns:
            A dict of load statistics (images loaded, items loaded, and a human-readable summary).

        Raises:
            GroundTruthError: If the source is unreadable, empty, or malformed while `strict`.
        """
        from shelf_benchmark.data.ground_truth import create_ground_truth_provider

        self.config.ground_truth.provider_type = provider_type
        self.config.ground_truth.source_uri = source_uri
        if bbox_format is not None:
            self.config.ground_truth.schema_mapping.bbox_format = bbox_format
        if gt_version is not None:
            self.config.ground_truth.gt_version = gt_version
        if strict is not None:
            self.config.ground_truth.strict = strict
        if schema_mapping:
            normalized_mapping: Dict[str, str] = {}
            for k, v in schema_mapping.items():
                if k in ("image_uri_field", "shelf_image_uri_field"):
                    normalized_mapping["image_key_field"] = v
                else:
                    normalized_mapping[k] = v
            unknown = [
                k for k in normalized_mapping
                if not hasattr(self.config.ground_truth.schema_mapping, k)
            ]
            if unknown:
                valid = sorted(type(self.config.ground_truth.schema_mapping).model_fields)
                raise ValueError(
                    f"Unknown ground-truth schema_mapping key(s): {unknown}. "
                    f"A typo here used to be ignored, leaving the default field name in place and "
                    f"producing empty ground truth. Valid keys: {valid}"
                )
            for k, v in normalized_mapping.items():
                setattr(self.config.ground_truth.schema_mapping, k, v)

        self.gt_provider = create_ground_truth_provider(
            gt_config=self.config.ground_truth,
            storage_manager=self.storage,
            project_id=self.config.gcp.project_id,
        )
        stats = dict(getattr(self.gt_provider, "load_stats", {}) or {})
        stats["summary"] = self.gt_provider.describe()
        stats["gt_version"] = self.config.ground_truth.gt_version
        logger.info("Ground truth connected: %s", stats["summary"])
        return stats

    def register_model(
        self,
        spec: UniversalModelSpec | str,
        *,
        custom_handler: Optional[Callable[[str, str, Optional[Any]], Dict[str, Any]]] = None,
        provider_family: Optional[ModelProviderFamily] = None,
        display_name: Optional[str] = None,
        pricing: Optional[ModelPricing] = None,
        endpoint_uri: Optional[str] = None,
        api_version: Optional[str] = None,
        location: Optional[str] = None,
    ) -> UniversalModelSpec:
        """Register a GEAP model, Gemma model, Fine-Tuned endpoint, or custom model with optional pricing.

        Accepts either a `UniversalModelSpec` instance or a string `model_id` with keyword arguments:
            sdk.register_model("my-model", custom_handler=lambda p, uri, schema: sample_shelf_payload())
        """
        if isinstance(spec, str):
            fam: ModelProviderFamily = provider_family or (
                "custom_callable" if custom_handler is not None else "vertex_gemini"
            )
            spec = UniversalModelSpec(
                model_id=spec,
                display_name=display_name or spec,
                provider_family=fam,
                custom_handler=custom_handler,
                pricing=pricing,
                endpoint_uri=endpoint_uri,
                api_version=api_version,
                location=location,
            )
        self._model_specs[spec.effective_name] = spec
        if spec.pricing is not None:
            self.config.pricing_per_million_tokens[spec.effective_name] = spec.pricing
        if spec.effective_name not in self.config.models:
            self.config.models.append(spec.effective_name)
        return spec

    def swap_models(self, models: Sequence[str | UniversalModelSpec]) -> List[str]:
        """Plug-and-play helper to swap the active benchmark models in one call (accepts model IDs, endpoint URIs, or UniversalModelSpec objects)."""
        active_names: List[str] = []
        for item in models:
            if isinstance(item, UniversalModelSpec):
                self.register_model(item)
                active_names.append(item.effective_name)
            else:
                active_names.append(str(item))
        self.config.models = active_names
        return active_names

    def _get_client_for_model(self, model_name: str) -> Optional[Any]:
        spec = self._model_specs.get(model_name)
        if spec is None:
            if self.config.offline.enabled:
                from shelf_benchmark.testing import offline_universal_payload_handler

                spec = UniversalModelSpec(
                    model_id=model_name,
                    display_name=model_name,
                    provider_family="custom_callable",
                    custom_handler=offline_universal_payload_handler,
                )
            else:
                lower = model_name.lower()
                # Auto-detect Fine-Tuned Vertex AI Endpoints (`projects/.../locations/.../endpoints/...`)
                if model_name.startswith("projects/") and "/endpoints/" in model_name:
                    spec = UniversalModelSpec(
                        model_id=model_name,
                        provider_family="vertex_tuned_endpoint",
                        endpoint_uri=model_name,
                    )
                # Auto-detect Gemma / PaliGemma models
                elif "gemma" in lower:
                    spec = UniversalModelSpec(
                        model_id=model_name,
                        provider_family="vertex_gemma",
                    )
                # Auto-detect GEAP / Early-Access / Experimental / Preview models
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

    def prepare_and_launch_fine_tuning(
        self,
        base_model: str = "gemini-3.8-flash",
        shelf_image_uri: str = "gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
        submit_vertex_job: Optional[bool] = None,
        tuned_display_name: Optional[str] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """One-call helper to generate the SFT JSONL dataset (`shelf_sft_train.jsonl`), upload to GCS, optionally submit a Vertex AI tuning job, and return a ready-to-use `UniversalModelSpec`."""
        do_submit = submit_vertex_job if submit_vertex_job is not None else self.config.fine_tuning.submit_live_tuning_job
        display_name = tuned_display_name or self.config.fine_tuning.tuned_model_display_name
        epoch_count = epochs if epochs is not None else self.config.fine_tuning.epochs

        custom_client = self._get_client_for_model(base_model)
        ft_task = GeminiFineTuningTask(
            self.config, self.storage, self.telemetry, genai_client=custom_client
        )
        res = ft_task.execute(
            model_name=base_model,
            shelf_image_uri=shelf_image_uri,
            submit_live_tuning_job=do_submit,
        )
        gcs_uri = f"{(self.config.buckets.artifacts_bucket or self.config.buckets.shelf_images_bucket).rstrip('/')}/{self.config.fine_tuning.dataset_gcs_subpath}"
        job_details = (
            ft_task.submit_vertex_tuning_job(
                base_model=base_model,
                training_dataset_gcs_uri=gcs_uri,
                tuned_model_display_name=display_name,
                epochs=epoch_count,
            )
            if do_submit
            else {
                "job_submitted": False,
                "base_model": base_model,
                "training_dataset_gcs_uri": gcs_uri,
                "ready_to_submit": True,
            }
        )
        ready_spec = UniversalModelSpec(
            model_id=display_name,
            display_name=display_name,
            provider_family="vertex_tuned_endpoint",
            endpoint_uri=job_details.get("job_name") or base_model,
        )
        return {
            "benchmark_result": res,
            "sft_dataset_gcs_uri": gcs_uri,
            "tuning_job": job_details,
            "tuned_model_spec": ready_spec,
        }

    def run_suite(
        self,
        models: Optional[Sequence[str]] = None,
        tasks: Optional[Sequence[str]] = None,
        approaches: Optional[Sequence[str]] = None,
        shelf_image_uri: Optional[str] = None,
        shelf_image_uris: Optional[Sequence[str]] = None,
        ground_truth: Optional[ImageGroundTruth] = None,
        ground_truth_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the benchmark suite across the requested models, tasks, approaches, and shelf images with full OpenTelemetry logging and Ground Truth evaluation."""
        selected_models = list(models) if models is not None else list(self.config.models)
        selected_tasks = list(tasks) if tasks is not None else list(self.config.tasks)
        selected_approaches = list(approaches) if approaches is not None else list(self.config.approaches)

        if shelf_image_uris:
            target_records = [
                ShelfAssociationRecord(
                    association_id=f"sdk-run-{idx:03d}",
                    shelf_image_uri=uri,
                    store_id="store-sdk",
                    ground_truth_id=ground_truth_id or Path(uri).name,
                )
                for idx, uri in enumerate(shelf_image_uris, start=1)
            ]
        elif shelf_image_uri is not None:
            target_records = [
                ShelfAssociationRecord(
                    association_id="sdk-run-001",
                    shelf_image_uri=shelf_image_uri,
                    store_id="store-sdk",
                    ground_truth_id=ground_truth_id or Path(shelf_image_uri).name,
                )
            ]
        elif self._dataset_records:
            target_records = list(self._dataset_records)
        else:
            default_uri = f"{self.config.buckets.shelf_images_bucket.rstrip('/')}/shelf-image.png"
            target_records = [
                ShelfAssociationRecord(
                    association_id="sdk-run-001",
                    shelf_image_uri=default_uri,
                    store_id="store-sdk",
                    ground_truth_id=ground_truth_id or Path(default_uri).name,
                )
            ]

        results: List[TaskExecutionResult] = []

        for record in target_records:
            current_uri = record.shelf_image_uri
            gt_record = ground_truth or self.gt_provider.get_ground_truth(
                shelf_image_uri=current_uri,
                ground_truth_id=record.ground_truth_id,
            )

            for model_name in selected_models:
                custom_client = self._get_client_for_model(model_name)
                ctx = CommonLayerContext(
                    config=self.config,
                    storage=self.storage,
                    telemetry=self.telemetry,
                    reports_dir=Path(self.config.reporting.output_dir),
                    genai_client=custom_client,
                )

                if "detection" in selected_tasks:
                    det_task = ProductDetectionTask(
                        self.config, self.storage, self.telemetry, genai_client=custom_client
                    )
                    results.append(
                        det_task.execute(
                            model_name=model_name,
                            shelf_image_uri=current_uri,
                            ground_truth=gt_record,
                        )
                    )

                if "classification" in selected_tasks:
                    for app_id in selected_approaches:
                        plugin = GLOBAL_APPROACH_REGISTRY.require(app_id)
                        results.append(
                            plugin.execute(
                                ctx=ctx,
                                model_name=model_name,
                                record=record,
                                gt_record=gt_record,
                            )
                        )

                if "matching" in selected_tasks:
                    mat_task = ProductMatchingTask(
                        self.config, self.storage, self.telemetry, genai_client=custom_client
                    )
                    results.append(
                        mat_task.execute(
                            model_name=model_name,
                            shelf_image_uri=current_uri,
                            ground_truth=gt_record,
                        )
                    )

                if "fine_tuning" in selected_tasks:
                    ft_task = GeminiFineTuningTask(
                        self.config, self.storage, self.telemetry, genai_client=custom_client
                    )
                    results.append(
                        ft_task.execute(
                            model_name=model_name,
                            shelf_image_uri=current_uri,
                            ground_truth=gt_record,
                        )
                    )

        artifact_paths = self.report_generator.generate_all_reports(results)
        return {
            "results": results,
            "artifacts": artifact_paths,
            "reports": artifact_paths,
            "report_paths": artifact_paths,
            "otel_log_path": self.config.telemetry.otel_log_path,
        }
