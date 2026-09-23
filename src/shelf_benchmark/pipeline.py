"""The single place where a model invocation becomes a scored, costed, traced result.

Why this module exists
----------------------
This repository used to contain **two independent implementations** of the same
finalisation pipeline:

* ``tasks/base.py::BaseBenchmarkTask.execute()`` -- used by the built-in tasks, and
* ``approaches/base.py::CommonLayerContext.finalize()`` -- used by every approach plugin,
  including the *documented* extension point ``@register_approach_function``.

They did not agree. Fed byte-identical predictions, they disagreed about retries, model-id
normalisation, unpriced-model warnings, container CPU time, embeddings cost, per-row token
attribution, and ``cost_per_product_usd`` -- and the plugin path *invented* token counts
(``180``/``350``/``90``/``160``) and a bounding box (``[500, 100, 800, 200]``) when the caller
did not supply them. The measured consequence was an **11% cost delta and a 2.8x token delta
between a built-in approach and a plugin given the same input**, which destroys the one
property a benchmark has to have: that two numbers printed side by side are comparable.

Both are now thin adapters over :class:`PipelineExecutor`. There is exactly one implementation.

Policies this module enforces, each of which was previously a per-path accident
-------------------------------------------------------------------------------
1. **Nothing is fabricated.** If the backend did not report token usage, the run records zero
   tokens and sets ``token_usage_reported=False`` in ``raw_output`` and on the telemetry span.
   It never guesses a plausible number. Likewise there is no default bounding box and no
   minimum latency: an unmeasured quantity is reported as unmeasured, not as a typical value.
2. **CPU time is always measured.** ``GCPBillingAndCostEngine`` *estimates* container CPU time
   when the caller passes ``None`` (``gcp_billing.py``, the ``active_ms`` branch). The task path
   passed a measurement and the plugin path passed nothing, so one was billed on measurement and
   the other on a guess. Every caller now goes through :meth:`PipelineExecutor.run`, which
   measures ``time.process_time()`` around the invocation. The old
   ``max(12.0, ...)`` floor in the task path is gone -- it was a fabricated minimum.
3. **Embeddings/Vision cost comes from config, for everyone.** It used to be derived from
   ``billing.embeddings_and_vision`` on the task path and default to ``0.0`` on the plugin path,
   so the same approach was cheaper purely by virtue of being a plugin.
4. **Per-row token and cost columns carry image-level values.** A row is a *facing*; tokens are
   measured per *image*. The plugin path divided by the row count, inventing a per-facing
   attribution nobody measured. This schema already repeats image-level values on every row
   (``image_latency_ms``, ``cost_per_shelf_image_usd``), so repeating the image total is the
   consistent and honest choice. ``cost_per_product_usd`` is taken from the billing engine rather
   than recomputed, so the two never drift.
5. **The approach id is the caller's**, unless a task deliberately stamped a more specific one on
   its rows. ``RowLevelReportItem.separation_approach`` used to default to the literal
   ``"single_pass_full_shelf"``, and the task path did ``if rows[0].separation_approach:
   separation_approach = rows[0].separation_approach`` -- an always-true test. Any task whose rows
   did not set the field therefore had its real approach id silently replaced by a *classification*
   id. That default is now ``""`` and empty means "not set".
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from shelf_benchmark.config import BenchmarkConfig, normalize_vertex_gemini_model_id
from shelf_benchmark.evaluation.cost import compute_cost_metrics
from shelf_benchmark.evaluation.metrics import evaluate_task_accuracy
from shelf_benchmark.models import (
    ImageGroundTruth,
    RowLevelReportItem,
    TaskExecutionResult,
    TokenUsageMetrics,
    build_execution_trace_metadata,
)
from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger

logger = logging.getLogger(__name__)

_WARNED_UNPRICED_MODELS: Set[str] = set()


def _warn_unpriced_model(model_name: str) -> None:
    """Warn once per process that a model is being costed off the generic 'default' rate card.

    Without this, an unrecognized model name produces plausible-looking dollar figures that are
    really just the default Gemini Flash rates, which silently corrupts cost comparisons.
    """
    if model_name in _WARNED_UNPRICED_MODELS:
        return
    _WARNED_UNPRICED_MODELS.add(model_name)
    logger.warning(
        "No explicit token pricing for model '%s'; falling back to the 'default' rate card. "
        "Cost figures for this model are estimates. Add an entry under "
        "'pricing_per_million_tokens' in your config to fix this.",
        model_name,
    )


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to re-invoke the model, and how long to wait between attempts.

    Previously the task path retried three times with linear backoff and the plugin path did not
    retry at all, so a flaky endpoint produced a lower success rate for plugins than for built-ins
    and the difference looked like an approach-quality difference.
    """

    max_attempts: int = 3
    backoff_base_seconds: float = 2.0

    def backoff_for(self, attempt: int) -> float:
        """Seconds to sleep after a failed `attempt` (1-based)."""
        return self.backoff_base_seconds * attempt


#: Use for deterministic offline work and for callers that do their own retrying.
NO_RETRY = RetryPolicy(max_attempts=1)


@dataclass(frozen=True)
class RawInvocation:
    """Everything a task or plugin actually *measured* for one shelf image.

    This is deliberately the raw, un-finalised result: predictions and token usage only. Run ids,
    timing, cost, accuracy, trace ids and provenance are all added by :class:`PipelineExecutor`,
    so no caller can compute them slightly differently.
    """

    #: One row per predicted facing. Only *prediction* fields need to be set; the executor stamps
    #: every run/identity/timing/cost/telemetry field.
    rows: List[RowLevelReportItem]

    #: ``None`` means the backend did not report usage. It is recorded as zero **and flagged**,
    #: never replaced with a guess.
    tokens: Optional[TokenUsageMetrics] = None

    #: Task-specific payload merged into ``TaskExecutionResult.raw_output``.
    raw_output: Dict[str, Any] = field(default_factory=dict)

    #: Additional non-token API cost (embeddings, Cloud Vision, ...). ``None`` means "derive it
    #: from the run's billing config", which is what every caller should normally do.
    extra_api_cost_usd: Optional[float] = None

    #: Facings suppressed as depth-stacked duplicates. ``None`` falls back to
    #: ``raw_output["depth_duplicates_filtered"]``.
    depth_duplicates_filtered: Optional[int] = None

    #: Human-readable pipeline stages, shown in the execution trace.
    stages_description: Optional[List[str]] = None

    #: Extra OpenTelemetry span attributes (per-stage latencies, embedding model ids, ...). Merged
    #: into the span the executor emits, so a multi-stage approach does not need to log its own.
    span_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvocationContext:
    """The identity of one (task, approach, model, image) invocation."""

    task_type: str
    approach_id: str
    model_name: str
    shelf_image_uri: str
    run_id: str = ""
    store_id: Optional[str] = None
    ground_truth: Optional[ImageGroundTruth] = None


class PipelineExecutor:
    """Turns a raw invocation into a fully finalised :class:`TaskExecutionResult`.

    Construct one per run and share it; it holds no per-invocation state.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        telemetry: OpenTelemetryBenchmarkLogger,
    ) -> None:
        self.config = config
        self.telemetry = telemetry

    # ------------------------------------------------------------------ public API

    def run(
        self,
        invoke: Callable[[str], RawInvocation],
        ctx: InvocationContext,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> TaskExecutionResult:
        """Invoke, retry, measure, and finalise.

        `invoke` is called with the *normalised* Vertex model id and must return a
        :class:`RawInvocation`. Timing and CPU measurement bracket only the callable, so a slow
        report writer cannot inflate a model's latency.
        """
        policy = retry_policy or RetryPolicy()
        live_model = normalize_vertex_gemini_model_id(
            ctx.model_name,
            for_live_vertex=not bool(self.config.offline.enabled),
        )

        invocation = RawInvocation(rows=[])
        status = "SUCCESS"
        error_message: Optional[str] = None

        start_dt = self.telemetry.now_utc()
        cpu_start_sec = time.process_time()
        for attempt in range(1, policy.max_attempts + 1):
            start_dt = self.telemetry.now_utc()
            cpu_start_sec = time.process_time()
            try:
                invocation = invoke(live_model)
                status = "SUCCESS"
                error_message = None
                break
            except Exception as exc:  # noqa: BLE001 - recorded on the result, then re-surfaced
                status = "ERROR"
                error_message = f"{type(exc).__name__}: {exc}"
                if attempt < policy.max_attempts:
                    logger.warning(
                        "Attempt %d/%d failed for task=%s approach=%s model=%s image=%s: %s. Retrying.",
                        attempt, policy.max_attempts, ctx.task_type, ctx.approach_id,
                        ctx.model_name, ctx.shelf_image_uri, error_message,
                    )
                    time.sleep(policy.backoff_for(attempt))
                else:
                    logger.error(
                        "All %d attempts failed for task=%s approach=%s model=%s image=%s. "
                        "Emitting an ERROR row; this image contributes no metrics.",
                        policy.max_attempts, ctx.task_type, ctx.approach_id,
                        ctx.model_name, ctx.shelf_image_uri,
                        exc_info=exc,
                    )

        end_dt = self.telemetry.now_utc()
        cpu_active_ms = round(max(0.0, time.process_time() - cpu_start_sec) * 1000.0, 3)

        return self.finalize(
            invocation,
            ctx=ctx,
            start_dt=start_dt,
            end_dt=end_dt,
            cpu_active_ms=cpu_active_ms,
            status=status,
            error_message=error_message,
        )

    def finalize(
        self,
        invocation: RawInvocation,
        *,
        ctx: InvocationContext,
        start_dt: Any,
        end_dt: Any,
        cpu_active_ms: Optional[float] = None,
        status: str = "SUCCESS",
        error_message: Optional[str] = None,
    ) -> TaskExecutionResult:
        """Score, cost, stamp, trace and package one invocation.

        Prefer :meth:`run`, which also measures CPU time and applies the retry policy. Call this
        directly only when you have already measured your own timings.
        """
        run_id = ctx.run_id or f"run-{uuid.uuid4().hex[:8]}"
        rows = list(invocation.rows)
        approach_id = self._resolve_approach_id(ctx.approach_id, rows)

        latency_ms = round((end_dt - start_dt).total_seconds() * 1000.0, 3)
        tokens, tokens_reported = self._resolve_tokens(invocation.tokens, ctx)
        cost = self._compute_cost(
            tokens=tokens,
            rows=rows,
            ctx=ctx,
            run_id=run_id,
            approach_id=approach_id,
            latency_ms=latency_ms,
            cpu_active_ms=cpu_active_ms,
            extra_api_cost_usd=invocation.extra_api_cost_usd,
        )

        start_iso = self.telemetry.format_iso(start_dt)
        end_iso = self.telemetry.format_iso(end_dt)
        self._stamp_rows(
            rows,
            run_id=run_id,
            ctx=ctx,
            approach_id=approach_id,
            start_iso=start_iso,
            end_iso=end_iso,
            latency_ms=latency_ms,
            tokens=tokens,
            cost=cost,
        )

        depth_filtered = invocation.depth_duplicates_filtered
        if depth_filtered is None:
            depth_filtered = int(invocation.raw_output.get("depth_duplicates_filtered", 0) or 0)

        accuracy = evaluate_task_accuracy(
            task_type=ctx.task_type,
            rows=rows,
            ground_truth=ctx.ground_truth,
            config=self.config.evaluation,
        )
        accuracy.depth_duplicates_filtered = int(depth_filtered)
        for r in rows:
            r.gt_version = accuracy.gt_version
            r.iou_threshold = accuracy.iou_threshold

        trace_id, span_id, _ = self.telemetry.log_task_execution(
            run_id=run_id,
            task_type=ctx.task_type,
            model_name=ctx.model_name,
            shelf_image_uri=ctx.shelf_image_uri,
            start_dt=start_dt,
            end_dt=end_dt,
            tokens=tokens,
            cost=cost,
            accuracy=accuracy,
            status=status,
            error_message=error_message,
            extra_attributes={
                "shelf_benchmark.separation_approach": approach_id,
                "shelf_benchmark.depth_duplicates_filtered": accuracy.depth_duplicates_filtered,
                # Lets a reader of the span tell "this model reports no usage metadata" apart from
                # "this call genuinely consumed nothing", which a bare 0 cannot express.
                "shelf_benchmark.token_usage_reported": tokens_reported,
                **dict(invocation.span_attributes),
            },
        )
        for r in rows:
            r.trace_id = trace_id
            r.span_id = span_id

        raw_output: Dict[str, Any] = dict(invocation.raw_output)
        raw_output.setdefault("total_classified_products", len(rows))
        raw_output["depth_duplicates_filtered"] = int(depth_filtered)
        raw_output["token_usage_reported"] = tokens_reported
        effective_api_calls: Optional[int] = None
        if approach_id == "configurable_multi_attribute_vlm":
            effective_api_calls = max(1, len(self.config.taxonomy.attribute_call_groups or []))
        raw_output["execution_trace"] = build_execution_trace_metadata(
            run_id=run_id,
            trace_id=trace_id,
            span_id=span_id,
            task_type=ctx.task_type,
            separation_approach=approach_id,
            model_name=ctx.model_name,
            shelf_image_uri=ctx.shelf_image_uri,
            latency_ms=latency_ms,
            tokens=tokens,
            cost=cost,
            accuracy=accuracy,
            facings_count=len(rows),
            otel_log_path=str(self.config.telemetry.otel_log_path),
            gcp_project_id=self.config.gcp.project_id,
            gcp_log_name=self.config.telemetry.gcp_log_name,
            taxonomy_source=self.config.taxonomy.taxonomy_source,
            ground_truth_provider=self.config.ground_truth.provider_type,
            reference_catalog_uri=self.config.embeddings.reference_catalog.source_uri,
            custom_stages=invocation.stages_description,
            api_calls_count=effective_api_calls,
        )

        return TaskExecutionResult(
            run_id=run_id,
            trace_id=trace_id,
            span_id=span_id,
            task_type=ctx.task_type,
            separation_approach=approach_id,
            model_name=ctx.model_name,
            shelf_image_uri=ctx.shelf_image_uri,
            start_time=start_iso,
            end_time=end_iso,
            latency_ms=latency_ms,
            tokens=tokens,
            cost=cost,
            accuracy=accuracy,
            raw_output=raw_output,
            row_level_items=rows,
            status=status,
            error_message=error_message,
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _resolve_approach_id(caller_approach_id: str, rows: List[RowLevelReportItem]) -> str:
        """Prefer an approach id a task deliberately stamped on its rows, else the caller's.

        A task such as classification picks its approach at invocation time (single-pass vs
        two-stage), so its rows are more authoritative than the caller's default. But this must
        only apply when the row *actually set* the field -- see the module docstring for the bug
        where the model default silently won.
        """
        if rows and rows[0].separation_approach:
            return rows[0].separation_approach
        return caller_approach_id

    def _resolve_tokens(
        self,
        tokens: Optional[TokenUsageMetrics],
        ctx: InvocationContext,
    ) -> tuple[TokenUsageMetrics, bool]:
        """Return `(tokens, reported)`, never inventing a count.

        Two different things arrive here as "nothing":

        * ``None`` -- the caller had no usage object at all (the plugin path).
        * an all-zero :class:`TokenUsageMetrics` -- ``extract_token_usage`` returns this when the
          response carries no ``usage_metadata``, so the task path cannot express "unreported"
          either.

        Both mean *the backend did not tell us*, and both are recorded as zero **and flagged**. A
        real call cannot consume zero input tokens, so this cannot misclassify a genuine zero.
        """
        reported = tokens is not None and (
            tokens.input_tokens > 0 or tokens.output_tokens > 0 or tokens.total_tokens > 0
        )
        if reported:
            assert tokens is not None  # narrowed by `reported`
            return tokens, True
        logger.warning(
            "No token usage reported for task=%s approach=%s model=%s image=%s. Recording zero "
            "tokens and flagging the result as unreported; token cost for this image is $0.00 and "
            "must not be compared against an approach that does report usage.",
            ctx.task_type, ctx.approach_id, ctx.model_name, ctx.shelf_image_uri,
        )
        return tokens if tokens is not None else TokenUsageMetrics(), False

    def _compute_cost(
        self,
        *,
        tokens: TokenUsageMetrics,
        rows: List[RowLevelReportItem],
        ctx: InvocationContext,
        run_id: str,
        approach_id: str,
        latency_ms: float,
        cpu_active_ms: Optional[float],
        extra_api_cost_usd: Optional[float],
    ):
        from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine

        pricing = self.config.get_pricing(ctx.model_name)
        if self.config.warn_on_unpriced_model and not self.config.has_explicit_pricing(ctx.model_name):
            _warn_unpriced_model(ctx.model_name)

        engine = GCPBillingAndCostEngine(
            project_id=self.config.gcp.project_id,
            billing_cfg=self.config.billing,
        )
        gcp_labels = engine.build_gcp_billing_labels(
            run_id=run_id,
            approach_id=approach_id,
            task_type=ctx.task_type,
            model_name=ctx.model_name,
        )

        if extra_api_cost_usd is None:
            per_facing_embed_usd = self.config.billing.embeddings_and_vision.per_facing_usd(
                task_type=ctx.task_type,
                approach_id=approach_id,
            )
            extra_api_cost_usd = round(max(1, len(rows)) * per_facing_embed_usd, 8)

        return compute_cost_metrics(
            tokens=tokens,
            pricing=pricing,
            product_count=len(rows),
            latency_ms=latency_ms,
            extra_embedding_or_vision_cost_usd=extra_api_cost_usd,
            billing_cfg=self.config.billing,
            project_id=self.config.gcp.project_id,
            model_name=ctx.model_name,
            gcp_labels=gcp_labels,
            container_cpu_active_ms=cpu_active_ms,
        )

    @staticmethod
    def _stamp_rows(
        rows: List[RowLevelReportItem],
        *,
        run_id: str,
        ctx: InvocationContext,
        approach_id: str,
        start_iso: str,
        end_iso: str,
        latency_ms: float,
        tokens: TokenUsageMetrics,
        cost: Any,
    ) -> None:
        """Copy image-level identity, timing, token and cost values onto every row.

        Token and cost columns carry the **image** totals, matching ``image_latency_ms``. Summing
        such a column across rows therefore over-counts by the facing count -- that is inherent to
        a schema that repeats image-level measurements per row, and is preferable to the plugin
        path's old behaviour of dividing by the row count, which invented a per-facing split that
        was never measured. Use ``cost_per_product_usd`` for a genuine per-facing figure.
        """
        for r in rows:
            r.run_id = run_id
            r.task_type = ctx.task_type
            r.separation_approach = approach_id
            r.model_name = ctx.model_name
            r.shelf_image_uri = ctx.shelf_image_uri
            r.store_id = ctx.store_id
            r.start_time = start_iso
            r.end_time = end_iso
            r.image_latency_ms = latency_ms
            r.input_tokens = tokens.input_tokens
            r.thinking_tokens = tokens.thinking_tokens
            r.output_tokens = tokens.output_tokens
            r.total_tokens = tokens.total_tokens
            r.cost_per_shelf_image_usd = cost.cost_per_shelf_image_usd
            r.cost_per_product_usd = cost.cost_per_product_usd
            r.vertex_ai_payg_tokens_usd = cost.vertex_ai_payg_tokens_usd
            r.vertex_ai_provisioned_throughput_usd = cost.vertex_ai_provisioned_throughput_usd
            r.vertex_ai_embeddings_and_vision_usd = cost.vertex_ai_embeddings_and_vision_usd
            r.cloud_run_compute_usd = cost.cloud_run_compute_usd
            r.gcs_and_observability_usd = cost.gcs_and_observability_usd
            r.traffic_type = cost.traffic_type
            r.billing_source = cost.billing_source
