"""Detection approaches. Each module registers itself with ``@register``."""

import utils  # noqa: F401
from approaches.base import (
    Approach,
    Context,
    Trace,
    all_approaches,
    get,
    register,
)

__all__ = ["Approach", "Context", "Trace", "all_approaches", "get", "register"]
