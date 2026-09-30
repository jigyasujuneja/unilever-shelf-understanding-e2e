"""Cloud SQL for PostgreSQL (`pgvector`) connection helper.

Replaces AlloyDB with Cloud SQL for PostgreSQL (`CREATE EXTENSION vector; USING hnsw`) and
automatically falls back to the in-memory/GCS HUL catalog when running offline or in unit tests.
"""

from __future__ import annotations

from utils.vector_store import CloudSQL, VectorCatalog, VectorMatch, iam_user, pgvector

__all__ = ["CloudSQL", "VectorCatalog", "VectorMatch", "iam_user", "pgvector"]
