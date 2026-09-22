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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel

from shelf_benchmark.approaches import (
    BaseShelfApproachPlugin,
    CommonLayerContext,
    GLOBAL_APPROACH_REGISTRY,
)
from shelf_benchmark.config import BenchmarkConfig, ModelPricing, TaxonomyConfig
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.evaluation.cost import compute_cost_metrics
from shelf_benchmark.models import (
    AccuracyMetrics,
    CostMetrics,
    ImageGroundTruth,
    RowLevelReportItem,
    ShelfAssociationRecord,
    TaskExecutionResult,
    TokenUsageMetrics,
)
from shelf_benchmark.reporting.generator import BenchmarkReportGenerator
from shelf_benchmark.tasks.classification import ProductClassificationTask
from shelf_benchmark.tasks.detection import ProductDetectionTask
from shelf_benchmark.tasks.facing_utils import (
    check_is_hul_brand,
    derive_size_bucket_from_bbox,
)
from shelf_benchmark.tasks.fine_tuning import GeminiFineTuningTask
from shelf_benchmark.tasks.matching import ProductMatchingTask
from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger


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
            ) -> TaskExecutionResult:
                run_id = f"custom-{approach_id}-{model_name}"
                start_dt = ctx.telemetry.now_utc()
                raw_outputs = func(ctx, model_name, record.shelf_image_uri)
                end_dt = ctx.telemetry.now_utc()
                latency_ms = max(0.1, round((end_dt - start_dt).total_seconds() * 1000.0, 3))

                all_bboxes = [item.get("bbox_2d", [0, 0, 0, 0]) for item in raw_outputs]
                total_in = sum(int(item.get("_input_tokens", 180)) for item in raw_outputs) or 350
                total_think = sum(int(item.get("_thinking_tokens", 0)) for item in raw_outputs)
                total_out = sum(int(item.get("_output_tokens", 90)) for item in raw_outputs) or 160

                tokens = TokenUsageMetrics(
                    input_tokens=total_in,
                    thinking_tokens=total_think,
                    output_tokens=total_out,
                    total_tokens=total_in + total_think + total_out,
                )
                cost = ctx.compute_cost(tokens, model_name, max(len(raw_outputs), 1))
                per_facing_cost = round(
                    cost.cost_per_shelf_image_usd / max(len(raw_outputs), 1), 8
                )
                accuracy = AccuracyMetrics(
                    ground_truth_available=False,
                    accuracy_status="PLACEHOLDER_AWAITING_GROUND_TRUTH",
                    predicted_count=len(raw_outputs),
                    depth_duplicates_filtered=int(raw_outputs[0].get("_depth_filtered", 0)) if raw_outputs else 0,
                )
                trace_id, span_id, _ = ctx.telemetry.log_task_execution(
                    run_id=run_id,
                    task_type="classification",
                    model_name=model_name,
                    shelf_image_uri=record.shelf_image_uri,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    tokens=tokens,
                    cost=cost,
                    accuracy=accuracy,
                    extra_attributes={"shelf_benchmark.separation_approach": approach_id},
                )

                row_items: List[RowLevelReportItem] = []
                for idx, item in enumerate(raw_outputs, start=1):
                    bbox = item.get("bbox_2d", [500, 100, 800, 200])
                    brand = str(item.get("brand", "Unknown"))
                    pkg = str(item.get("packaging_type", "box"))
                    size_hint = str(item.get("size", ""))
                    is_hul = item.get(
                        "is_hul_brand",
                        check_is_hul_brand(brand, taxonomy=ctx.config.taxonomy),
                    )
                    rule_size = derive_size_bucket_from_bbox(
                        bbox,
                        all_bboxes,
                        packaging_type=pkg,
                        model_size_hint=size_hint,
                        taxonomy=ctx.config.taxonomy,
                    )
                    row_items.append(
                        RowLevelReportItem(
                            run_id=run_id,
                            trace_id=trace_id,
                            span_id=span_id,
                            start_time=ctx.telemetry.format_iso(start_dt),
                            end_time=ctx.telemetry.format_iso(end_dt),
                            image_latency_ms=latency_ms,
                            task_type="classification",
                            separation_approach=approach_id,
                            model_name=model_name,
                            shelf_image_uri=record.shelf_image_uri,
                            store_id=record.store_id,
                            product_index=idx,
                            predicted_category=str(item.get("category", "Personal Care")),
                            predicted_subcategory=str(item.get("subcategory", "General")),
                            predicted_brand=brand,
                            is_hul_brand=bool(is_hul),
                            predicted_product_name=str(item.get("product_name", f"{brand} Product")),
                            predicted_variant=str(item.get("variant", "Standard")),
                            predicted_packaging=pkg,
                            predicted_pack_type=str(item.get("pack_type", "Single")),
                            predicted_size=size_hint,
                            rule_derived_size_bucket=rule_size,
                            matched_sku_id=item.get("matched_sku_id"),
                            bbox_ymin=int(bbox[0]),
                            bbox_xmin=int(bbox[1]),
                            bbox_ymax=int(bbox[2]),
                            bbox_xmax=int(bbox[3]),
                            shelf_row=str(item.get("shelf_row", "middle")),
                            position_on_shelf=idx,
                            is_front_facing=True,
                            confidence=float(item.get("confidence", 0.95)),
                            latency_ms=round(latency_ms / max(len(raw_outputs), 1), 2),
                            input_tokens=tokens.input_tokens // max(len(raw_outputs), 1),
                            thinking_tokens=tokens.thinking_tokens // max(len(raw_outputs), 1),
                            output_tokens=tokens.output_tokens // max(len(raw_outputs), 1),
                            cost_per_product_usd=per_facing_cost,
                            cost_per_shelf_image_usd=cost.cost_per_shelf_image_usd,
                        )
                    )

                return TaskExecutionResult(
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    task_type="classification",
                    separation_approach=approach_id,
                    model_name=model_name,
                    shelf_image_uri=record.shelf_image_uri,
                    start_time=ctx.telemetry.format_iso(start_dt),
                    end_time=ctx.telemetry.format_iso(end_dt),
                    latency_ms=latency_ms,
                    tokens=tokens,
                    cost=cost,
                    accuracy=accuracy,
                    row_level_items=row_items,
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
        self.config = BenchmarkConfig.from_yaml(config_path)
        if taxonomy_path is not None:
            self.config.taxonomy = TaxonomyConfig.from_yaml_or_defaults(taxonomy_path)
        if output_dir is not None:
            self.config.reporting.output_dir = str(output_dir)
            self.config.telemetry.otel_log_path = str(Path(output_dir) / "otel_logs.jsonl")

        self.storage = StorageManager(
            project_id=self.config.gcp.project_id,
            bucket_config=self.config.buckets,
        )
        self.telemetry = OpenTelemetryBenchmarkLogger(
            config=self.config.telemetry,
            project_id=self.config.gcp.project_id,
            location=self.config.gcp.location,
        )
        self.report_generator = BenchmarkReportGenerator(
            output_dir=self.config.reporting.output_dir
        )
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

    def register_model(self, spec: UniversalModelSpec) -> None:
        """Register a GEAP model, Gemma model, Fine-Tuned endpoint, or custom model with optional pricing."""
        self._model_specs[spec.effective_name] = spec
        if spec.pricing is not None:
            self.config.pricing_per_million_tokens[spec.effective_name] = spec.pricing
        if spec.effective_name not in self.config.models:
            self.config.models.append(spec.effective_name)

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
        shelf_image_uri: str = "gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
    ) -> Dict[str, Any]:
        """Run the benchmark suite across the requested models, tasks, and approaches with full OpenTelemetry logging."""
        selected_models = list(models) if models is not None else list(self.config.models)
        selected_tasks = list(tasks) if tasks is not None else list(self.config.tasks)
        selected_approaches = list(approaches) if approaches is not None else list(self.config.approaches)
        record = ShelfAssociationRecord(
            association_id="sdk-run-001",
            shelf_image_uri=shelf_image_uri,
            store_id="store-sdk",
        )

        results: List[TaskExecutionResult] = []
        ctx = CommonLayerContext(
            config=self.config,
            storage=self.storage,
            telemetry=self.telemetry,
            reports_dir=Path(self.config.reporting.output_dir),
        )

        for model_name in selected_models:
            custom_client = self._get_client_for_model(model_name)

            if "detection" in selected_tasks:
                det_task = ProductDetectionTask(
                    self.config, self.storage, self.telemetry, genai_client=custom_client
                )
                results.append(det_task.execute(model_name=model_name, shelf_image_uri=shelf_image_uri))

            if "classification" in selected_tasks:
                builtin_vlm_approaches = {
                    "single_pass_full_shelf",
                    "two_stage_bbox_guided_nms",
                    "two_stage_physical_crop_per_facing",
                }
                for app_id in selected_approaches:
                    plugin = GLOBAL_APPROACH_REGISTRY.get(app_id)
                    if plugin is not None and (custom_client is None or app_id not in builtin_vlm_approaches):
                        results.append(plugin.execute(ctx=ctx, model_name=model_name, record=record))
                    else:
                        cls_task = ProductClassificationTask(
                            self.config, self.storage, self.telemetry, genai_client=custom_client
                        )
                        results.append(
                            cls_task.execute(
                                model_name=model_name,
                                shelf_image_uri=shelf_image_uri,
                                separation_approach=app_id,
                            )
                        )

            if "matching" in selected_tasks:
                mat_task = ProductMatchingTask(
                    self.config, self.storage, self.telemetry, genai_client=custom_client
                )
                results.append(mat_task.execute(model_name=model_name, shelf_image_uri=shelf_image_uri))

            if "fine_tuning" in selected_tasks:
                ft_task = GeminiFineTuningTask(
                    self.config, self.storage, self.telemetry, genai_client=custom_client
                )
                results.append(ft_task.execute(model_name=model_name, shelf_image_uri=shelf_image_uri))

        artifact_paths = self.report_generator.generate_all_reports(results)
        return {
            "results": results,
            "artifacts": artifact_paths,
            "otel_log_path": self.config.telemetry.otel_log_path,
        }
