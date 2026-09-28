"""Unified `src/utils/` package with automatic fallback shims when running outside a pip venv."""

from __future__ import annotations

import sys
from pathlib import Path

_VENV_SITE = "/usr/local/google/home/jjuneja/.gemini/jetski/brain/5fc8d969-761d-461f-99ae-02c8ca30923f/scratch/venv/lib/python3.13/site-packages"
if Path(_VENV_SITE).is_dir() and _VENV_SITE not in sys.path:
    sys.path.insert(1, _VENV_SITE)

_SYS_DIST = "/usr/lib/python3/dist-packages"
if Path(_SYS_DIST).is_dir() and _SYS_DIST not in sys.path:
    sys.path.append(_SYS_DIST)

_SHIMS = str(Path(__file__).resolve().parent / "_local_shims")
if _SHIMS not in sys.path:
    sys.path.append(_SHIMS)  # appended last so real pip-installed PIL/opentelemetry take precedence
