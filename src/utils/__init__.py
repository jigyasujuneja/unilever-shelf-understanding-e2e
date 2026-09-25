"""Unified `src/utils/` package with automatic fallback shims when running outside a pip venv."""

from __future__ import annotations

import sys
from pathlib import Path

_SHIMS = str(Path(__file__).resolve().parent / "_local_shims")
if _SHIMS not in sys.path:
    sys.path.append(_SHIMS)  # appended last so real pip-installed PIL/opentelemetry take precedence
