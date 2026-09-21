"""Auto-Discovery Plugin Registry for Modular Shelf Understanding Approaches.

Scans all subdirectories under `src/shelf_benchmark/approaches/*/plugin.py` and registers
any `BaseShelfApproachPlugin` subclasses or `get_plugins()` factory functions.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type

from shelf_benchmark.approaches.base import BaseShelfApproachPlugin


class ApproachRegistry:
    """Central registry for pluggable Shelf Understanding approaches."""

    def __init__(self) -> None:
        self._plugins: Dict[str, BaseShelfApproachPlugin] = {}

    def register(self, plugin: BaseShelfApproachPlugin) -> None:
        self._plugins[plugin.approach_id] = plugin

    def get(self, approach_id: str) -> Optional[BaseShelfApproachPlugin]:
        if not self._plugins:
            self.discover_all()
        return self._plugins.get(approach_id)

    def list_all(self) -> List[BaseShelfApproachPlugin]:
        if not self._plugins:
            self.discover_all()
        return list(self._plugins.values())

    def list_ids(self) -> List[str]:
        if not self._plugins:
            self.discover_all()
        return list(self._plugins.keys())

    def discover_all(self) -> None:
        """Auto-discovers every `<subfolder>/plugin.py` inside `shelf_benchmark.approaches`."""
        approaches_dir = Path(__file__).resolve().parent
        for subdir in sorted(approaches_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue
            plugin_file = subdir / "plugin.py"
            if not plugin_file.exists():
                continue

            module_name = f"shelf_benchmark.approaches.{subdir.name}.plugin"
            try:
                mod = importlib.import_module(module_name)
                if hasattr(mod, "get_plugins") and callable(mod.get_plugins):
                    for p in mod.get_plugins():
                        if isinstance(p, BaseShelfApproachPlugin):
                            self.register(p)
                else:
                    for _, obj in inspect.getmembers(mod, inspect.isclass):
                        if (
                            issubclass(obj, BaseShelfApproachPlugin)
                            and obj is not BaseShelfApproachPlugin
                        ):
                            self.register(obj())
            except Exception as exc:
                print(f"[ApproachRegistry] Warning: failed to load {module_name}: {exc}")


GLOBAL_APPROACH_REGISTRY = ApproachRegistry()
