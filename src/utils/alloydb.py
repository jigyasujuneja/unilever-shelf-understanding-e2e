"""Backward-compatible shim for legacy imports of ``utils.alloydb``.

Delegates all queries to ``VectorCatalog`` / ``CloudSQL`` in ``src/utils/vector_store.py`` so
existing approaches or notebooks calling ``AlloyDB(**config.get("cloudsql", {})).query(...)``
execute against Cloud SQL for PostgreSQL (``pgvector``), Vertex AI Vector Search, BigQuery, or
the in-memory GCS catalog without requiring an AlloyDB cluster.
"""

from __future__ import annotations

from typing import Any

from utils.vector_store import CloudSQL, VectorCatalog, iam_user


def pgvector(vector: list[float]) -> str:
    """A Python list as a pgvector literal; use as ``%s::vector`` in SQL."""
    return "[" + ",".join(map(str, vector)) + "]"


class AlloyDB(CloudSQL):
    """Deprecated alias for ``CloudSQL`` / ``VectorCatalog``."""

    def __init__(self, instance: str = "", database: str = "postgres", user: str | None = None,
                 ip_type: str = "PRIVATE", **kwargs: Any):
        super().__init__(
            instance=instance,
            database=database,
            user=user,
            ip_type=ip_type,
            fallback_local=bool(kwargs.get("fallback_local_scann", True)),
        )


__all__ = ["AlloyDB", "CloudSQL", "VectorCatalog", "iam_user", "pgvector"]
