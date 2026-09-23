"""The single error type raised by the ground-truth layer.

This lives in its own module purely to keep the import graph acyclic: the geometry
(`data.bbox`), the schema mapper (`data.ground_truth_schema`), the provider framework
(`data.ground_truth_providers`) and the concrete readers (`data.ground_truth_sources`) all
need to raise it, and they are layered on top of one another in that order.

Callers of the ground-truth layer are documented to only have to handle `GroundTruthError`;
the provider layer translates backend-specific I/O failures into it at the boundary.
"""

from __future__ import annotations


class GroundTruthError(RuntimeError):
    """Raised when a configured ground-truth source cannot be loaded or is unusable."""
