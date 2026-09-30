"""Detection approaches. Each module registers itself with ``@register``."""

from typing import Any

import utils  # noqa: F401
from approaches.base import (
    Approach,
    Context,
    Trace,
    all_approaches,
    get,
    register,
)


def load(name: str, config: dict[str, Any] | None = None) -> Approach:
    """Instantiate a registered approach and call its ``setup(config)`` hook."""
    app = get(name)
    app.setup(config or {})
    return app


__all__ = ["Approach", "Context", "Trace", "all_approaches", "get", "load", "register"]
