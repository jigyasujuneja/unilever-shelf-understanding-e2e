"""Auto-discovery plugin registry for modular Shelf Understanding approaches.

Scans `src/shelf_benchmark/approaches/*/plugin.py` and registers any `BaseShelfApproachPlugin`
subclass or `get_plugins()` factory it finds.

Two behaviours matter for correctness, both learned the hard way:

* **Discovery runs exactly once, eagerly.** The previous version only discovered when the registry
  was empty, so any script that registered a custom approach at import time (which every engineer
  script does, via `@register_approach_function`) permanently hid the built-in plugins. Callers
  then silently fell back to a different algorithm while still labelling the results with the
  requested approach id.
* **A broken plugin is an error, not a warning.** Import failures used to be printed and swallowed,
  so an approach could vanish from a comparison table without anyone noticing. Set
  `SHELF_BENCH_TOLERANT_PLUGINS=1` to downgrade failures to warnings (useful when deliberately
  working on a half-finished plugin).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from shelf_benchmark.approaches.base import BaseShelfApproachPlugin

logger = logging.getLogger(__name__)

TOLERANT_PLUGINS_ENV = "SHELF_BENCH_TOLERANT_PLUGINS"

# Approaches implemented by the built-in `ProductClassificationTask` rather than by a standalone
# plugin class. These are the only ids for which it is legitimate to bypass the registry, and they
# are defined here (not duplicated in runner.py and sdk.py) so the two execution paths cannot drift.
BUILTIN_VLM_CLASSIFICATION_APPROACHES = frozenset(
    {
        "single_pass_full_shelf",
        "open_vocab_brand_plus_catalog_resolver",
        "configurable_multi_attribute_vlm",
        "two_stage_bbox_guided_nms",
        "two_stage_physical_crop_per_facing",
    }
)


class ApproachPluginError(RuntimeError):
    """Raised when an approach plugin module cannot be imported or instantiated."""


class ApproachRegistry:
    """Central registry for pluggable Shelf Understanding approaches."""

    def __init__(self) -> None:
        self._plugins: Dict[str, BaseShelfApproachPlugin] = {}
        self._discovered = False
        self.discovery_errors: Dict[str, str] = {}

    def register(self, plugin: BaseShelfApproachPlugin) -> None:
        """Register (or replace) a plugin under its `approach_id`."""
        existing = self._plugins.get(plugin.approach_id)
        if existing is not None and existing is not plugin:
            logger.warning(
                "Approach id '%s' is being re-registered by %s (previously %s). "
                "The most recent registration wins.",
                plugin.approach_id,
                type(plugin).__name__,
                type(existing).__name__,
            )
        self._plugins[plugin.approach_id] = plugin

    def ensure_discovered(self) -> None:
        """Run built-in plugin discovery once, regardless of what is already registered."""
        if not self._discovered:
            self.discover_all()

    def get(self, approach_id: str) -> Optional[BaseShelfApproachPlugin]:
        """Return the plugin for `approach_id`, or `None` if it is not registered."""
        self.ensure_discovered()
        return self._plugins.get(approach_id)

    def require(self, approach_id: str) -> BaseShelfApproachPlugin:
        """Return the plugin for `approach_id` or raise with the list of valid ids.

        Callers should prefer this over `get()`: falling back to a default pipeline when an id is
        unknown produces results labelled with an approach that never ran.
        """
        plugin = self.get(approach_id)
        if plugin is None:
            raise KeyError(
                f"Unknown approach '{approach_id}'. Registered approaches: {self.list_ids()}. "
                f"{self._error_hint()}"
            )
        return plugin

    def list_all(self) -> List[BaseShelfApproachPlugin]:
        """All registered plugin instances."""
        self.ensure_discovered()
        return list(self._plugins.values())

    def list_ids(self) -> List[str]:
        """All registered approach ids, sorted for stable output."""
        self.ensure_discovered()
        return sorted(self._plugins)

    def _error_hint(self) -> str:
        if not self.discovery_errors:
            return ""
        return f"Note: {len(self.discovery_errors)} plugin module(s) failed to import: {self.discovery_errors}"

    def discover_all(self) -> None:
        """Import every `<subfolder>/plugin.py` under `shelf_benchmark.approaches`."""
        self.discovery_errors = {}
        tolerant = os.environ.get(TOLERANT_PLUGINS_ENV, "").strip().lower() in {"1", "true", "yes"}
        approaches_dir = Path(__file__).resolve().parent

        for subdir in sorted(approaches_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith((".", "_")):
                continue
            if not (subdir / "plugin.py").exists():
                continue

            module_name = f"shelf_benchmark.approaches.{subdir.name}.plugin"
            try:
                module = importlib.import_module(module_name)
                factory = getattr(module, "get_plugins", None)
                if callable(factory):
                    for plugin in factory():
                        if isinstance(plugin, BaseShelfApproachPlugin):
                            self.register(plugin)
                    continue
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseShelfApproachPlugin)
                        and obj is not BaseShelfApproachPlugin
                        and not inspect.isabstract(obj)
                        and obj.__module__ == module_name
                    ):
                        self.register(obj())
            except Exception as exc:
                self.discovery_errors[module_name] = f"{type(exc).__name__}: {exc}"
                if tolerant:
                    logger.warning("Skipping approach plugin '%s': %s", module_name, exc)
                else:
                    raise ApproachPluginError(
                        f"Failed to load approach plugin '{module_name}': {exc}. "
                        f"Fix the plugin, or set {TOLERANT_PLUGINS_ENV}=1 to skip broken plugins."
                    ) from exc
        self._discovered = True


GLOBAL_APPROACH_REGISTRY = ApproachRegistry()
