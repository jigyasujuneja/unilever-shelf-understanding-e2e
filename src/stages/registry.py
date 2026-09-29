"""Registry for pluggable pipeline stages.

Approaches in ``src/approaches/`` implement top-level benchmark tasks:
  - ``detection``: Product bounding-box detection.
  - ``classification`` (``target_field="compound"``): Category, brand, and packaging classification.
  - ``classification`` (``target_field="variant"``): Product variant and shade classification.

Modules in ``src/stages/`` register individual pipeline steps (rectification, post-detection
filtering, crop clustering, catalog retrieval, shade disambiguation, and shelf metric evaluation)
so callers can swap a single step via CLI flags or configuration dicts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

STAGE_GROUP_ALIASES: dict[str, str] = {
    "gondola_kpi": "shelf_metrics",
}


@dataclass
class StageSpec:
    """Metadata and callable for a registered pipeline stage."""

    stage_group: str
    name: str
    title: str
    description: str
    f2_delta: float = 0.0
    latency_delta_s: float = 0.0
    cost_delta_inr: float = 0.0
    default: bool = False
    fn: Callable[..., Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("fn", None)
        return data


# Backward-compatible alias for callers importing MicroStageSpec.
MicroStageSpec = StageSpec


STAGE_REGISTRY: dict[str, dict[str, StageSpec]] = {
    "rectifier": {},
    "post_detector": {},
    "clusterer": {},
    "retriever": {},
    "tiebreaker": {},
    "shelf_metrics": {},
}


def _resolve_group_name(stage_group: str) -> str:
    return STAGE_GROUP_ALIASES.get(stage_group, stage_group)


def register_stage(spec: StageSpec) -> StageSpec:
    """Register a pipeline stage specification under its stage group."""
    canonical_group = _resolve_group_name(spec.stage_group)
    spec.stage_group = canonical_group
    group = STAGE_REGISTRY.setdefault(canonical_group, {})
    group[spec.name] = spec
    return spec


def get_stage(stage_group: str, name: str | None = None) -> StageSpec:
    """Return the requested stage specification, falling back to the group default."""
    canonical_group = _resolve_group_name(stage_group)
    group = STAGE_REGISTRY.get(canonical_group, {})
    if name and name in group:
        return group[name]
    for spec in group.values():
        if spec.default:
            return spec
    if group:
        return next(iter(group.values()))
    raise KeyError(f"Unknown stage group {stage_group!r} or stage {name!r}")


def all_stages() -> dict[str, list[dict[str, Any]]]:
    """Return all registered stage specifications serialized as dictionaries."""
    return {
        group: [spec.to_dict() for spec in specs.values()]
        for group, specs in STAGE_REGISTRY.items()
    }


def default_stage_config() -> dict[str, str]:
    """Return a mapping from each stage group name to its default stage name."""
    defaults: dict[str, str] = {}
    for group, specs in STAGE_REGISTRY.items():
        for spec in specs.values():
            if spec.default:
                defaults[group] = spec.name
                break
    return defaults
