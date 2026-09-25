"""Detection approaches. Each module registers itself with ``@register``."""

from approaches.base import (
    Approach,
    Context,
    Trace,
    all_approaches,
    get,
    register,
)

__all__ = ["Approach", "Context", "Trace", "all_approaches", "get", "register"]
