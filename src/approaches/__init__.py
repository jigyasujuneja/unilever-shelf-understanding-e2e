"""Approaches organized by business use case (``market_share/``, ``merchandising/``) and task
(``detection/``, ``classification/``, ``retrieval/``, ``end_to_end/``). Each module registers
itself with ``@register`` or ``compose(...)``."""

from approaches.base import (
    Approach,
    Context,
    Trace,
    _discover,
    all_approaches,
    compose,
    get,
    register,
)

_discover()

__all__ = ["Approach", "Context", "Trace", "all_approaches", "compose", "get", "register"]
