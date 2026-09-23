"""Abstract base task: declares *how to call the model*, and delegates everything else.

A task's only job is `invoke_model()`. Timing, retry, token accounting, the separated GCP cost
model, ground-truth scoring, OpenTelemetry spans and execution-trace provenance all live in
`shelf_benchmark.pipeline`, which the approach-plugin path shares. Before that module existed,
this file and `approaches/base.py` each implemented that logic separately and disagreed -- see
the `pipeline` module docstring for the list of divergences and what they cost.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from shelf_benchmark.auth import create_genai_client
from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.models import (
    ImageGroundTruth,
    RowLevelReportItem,
    TaskExecutionResult,
    TokenUsageMetrics,
)
from shelf_benchmark.pipeline import (
    InvocationContext,
    PipelineExecutor,
    RawInvocation,
    RetryPolicy,
)
from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger

logger = logging.getLogger(__name__)


class BaseBenchmarkTask(ABC):
    """Base class for separated Shelf Understanding tasks."""

    task_type: str = "base"

    #: Approach id reported when the caller does not pass `separation_approach=`.
    #:
    #: Declared per subclass rather than defaulted inside `execute()`. The fallback used to be the
    #: literal "single_pass_full_shelf" -- a *classification* approach id -- so the fine-tuning
    #: task, which never sets one, labelled every row, cost record and GCP billing label as if it
    #: had run classification. `__init_subclass__` below turns that omission into an import-time
    #: error instead of a plausible-looking wrong label in a report.
    default_separation_approach: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip intermediate abstract bases that do not declare a concrete task.
        if getattr(cls, "task_type", "base") == "base":
            return
        if not getattr(cls, "default_separation_approach", ""):
            raise TypeError(
                f"{cls.__name__} sets task_type={cls.task_type!r} but no "
                f"`default_separation_approach`. Declare one; reports, cost attribution and GCP "
                f"billing labels are all keyed on it."
            )

    def __init__(
        self,
        config: BenchmarkConfig,
        storage_manager: StorageManager,
        telemetry_logger: OpenTelemetryBenchmarkLogger,
        genai_client: Optional[Any] = None,
    ):
        self.config = config
        self.storage = storage_manager
        self.telemetry = telemetry_logger
        self._genai_client = genai_client
        self._executor = PipelineExecutor(config=config, telemetry=telemetry_logger)

    def get_client(self, location: Optional[str] = None) -> Any:
        if self._genai_client is not None and location is None:
            return self._genai_client
        return create_genai_client(
            project_id=self.config.gcp.project_id,
            location=location or self.config.gcp.location,
        )

    @abstractmethod
    def invoke_model(
        self,
        model_name: str,
        shelf_image_uri: str,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], TokenUsageMetrics, List[RowLevelReportItem]]:
        """Execute the task logic against the model and return (parsed_output_dict, token_usage, row_items)."""

    def execute(
        self,
        model_name: str,
        shelf_image_uri: str,
        run_id: Optional[str] = None,
        store_id: Optional[str] = None,
        ground_truth: Optional[ImageGroundTruth] = None,
        **kwargs: Any,
    ) -> TaskExecutionResult:
        """Invoke the task and finalise it through the shared pipeline.

        All of the timing, retry, cost, scoring, telemetry and trace logic that used to live here
        now lives in `shelf_benchmark.pipeline`, shared with the approach-plugin path. See that
        module's docstring for the divergences this collapse fixed.
        """
        # Copy before popping: the previous implementation did `kwargs.pop("max_attempts", 3)` on
        # the caller's own dict when it was splatted in, silently removing the key for the caller.
        invoke_kwargs = dict(kwargs)
        max_attempts = int(invoke_kwargs.pop("max_attempts", 3))
        approach_id = str(
            invoke_kwargs.get("separation_approach") or self.default_separation_approach
        )

        def _invoke(live_vertex_model: str) -> RawInvocation:
            raw_output, tokens, rows = self.invoke_model(
                model_name=live_vertex_model,
                shelf_image_uri=shelf_image_uri,
                ground_truth=ground_truth,
                **invoke_kwargs,
            )
            return RawInvocation(
                rows=list(rows or []),
                tokens=tokens,
                raw_output=dict(raw_output or {}),
            )

        return self._executor.run(
            _invoke,
            InvocationContext(
                task_type=self.task_type,
                approach_id=approach_id,
                model_name=model_name,
                shelf_image_uri=shelf_image_uri,
                run_id=run_id or "",
                store_id=store_id,
                ground_truth=ground_truth,
            ),
            RetryPolicy(max_attempts=max_attempts),
        )

