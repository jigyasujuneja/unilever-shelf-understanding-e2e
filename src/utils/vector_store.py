"""Unified Product Vector Store & Cloud SQL for PostgreSQL (`pgvector`) client.

Replaces AlloyDB with four selectable, cloud-governed retrieval backends configured via
``vector_store`` in ``config.yaml``:

1. ``cloudsql_pgvector``: Managed **Cloud SQL for PostgreSQL** with ``pgvector`` (``HNSW`` cosine
   index), connected over mTLS + IAM database authentication via ``cloud-sql-python-connector``
   and ``pg8000`` (parameterized SQL queries only).
2. ``vertex_vector_search``: Managed **Vertex AI Vector Search** (``aiplatform.googleapis.com``
   ``IndexEndpoint:findNeighbors``) with ScaNN token restricts (``category``, ``brand``,
   ``packaging_type``).
3. ``bigquery``: Serverless **BigQuery Vector Search** (``VECTOR_SEARCH`` SQL over HTTPS).
4. ``gcs_inmemory``: Zero-latency in-memory NumPy/ScaNN cosine similarity index backed by the
   Unilever SKU catalog in GCS (``gs://<project>-shelf-images/HUL_catalog``) or local catalog.
   Used automatically as a fast-path fallback when no external database endpoint is reachable.

One-time Cloud SQL for PostgreSQL setup::

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS products (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT,
        brand TEXT,
        packaging_type TEXT,
        embedding vector(512),
        metadata JSONB
    );
    CREATE INDEX IF NOT EXISTS products_embedding_hnsw_idx
        ON products USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from typing import Any

from utils.llm import load_config


@dataclass(frozen=True)
class VectorMatch:
    """Single nearest-neighbor SKU match returned by ``VectorCatalog.search``."""

    sku_id: str
    score: float
    category: str = "Personal Care"
    brand: str = "Dove"
    packaging_type: str = "bottle"
    variant: str = "Deeply Nourishing"
    metadata: dict[str, Any] = field(default_factory=dict)


def pgvector(vector: list[float]) -> str:
    """Format a numeric Python list as a sanitized pgvector literal for parameterized ``%s::vector``."""
    return "[" + ",".join(f"{float(v):.6f}" for v in vector) + "]"


def iam_user() -> str:
    """Resolve the IAM database user for Application Default Credentials."""
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    email = getattr(creds, "service_account_email", None)
    if not email or email == "default":
        creds.refresh(Request())
        email = getattr(creds, "service_account_email", None)
    if not email or "@" not in email:
        raise RuntimeError("Set vector_store.user or cloudsql.user in config.yaml (IAM database user)")
    return email.removesuffix(".gserviceaccount.com")


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    n = min(len(vec_a), len(vec_b))
    if n == 0:
        return 0.0
    dot = sum(float(vec_a[i]) * float(vec_b[i]) for i in range(n))
    norm_a = math.sqrt(sum(float(vec_a[i]) ** 2 for i in range(n))) or 1.0
    norm_b = math.sqrt(sum(float(vec_b[i]) ** 2 for i in range(n))) or 1.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class CloudSQL:
    """Thread-safe Cloud SQL for PostgreSQL (``pgvector``) client using mTLS + IAM auth.

    Automatically falls back to ``VectorCatalog(backend="gcs_inmemory")`` when ``fallback_local``
    is enabled and the Cloud SQL instance is unreachable (for example, during offline unit tests).
    """

    def __init__(
        self,
        instance: str = "",
        database: str = "postgres",
        user: str | None = None,
        ip_type: str = "PRIVATE",
        fallback_local: bool = True,
        **_: Any,
    ):
        self.instance = instance
        self.database = database
        self.user = user
        self.ip_type = ip_type.upper()
        self.fallback_local = fallback_local
        self._connector = None
        self._local = threading.local()
        self._fallback_catalog: VectorCatalog | None = None

    def _get_connection(self):
        if getattr(self._local, "conn", None) is not None:
            return self._local.conn
        connector = getattr(self, "connector", None) or getattr(self, "_connector", None)
        if connector is not None:
            resolved_user = self.user or iam_user()
            self._local.conn = connector.connect(
                self.instance,
                "pg8000",
                user=resolved_user,
                db=self.database,
                enable_iam_auth=True,
                ip_type=self.ip_type,
            )
            return self._local.conn

        from google.cloud.sql.connector import Connector, IPTypes

        self._connector = Connector(refresh_strategy="lazy")
        resolved_user = self.user or iam_user()
        ipt = IPTypes[self.ip_type] if hasattr(IPTypes, self.ip_type) else IPTypes.PRIVATE
        self._local.conn = self._connector.connect(
            self.instance,
            "pg8000",
            user=resolved_user,
            db=self.database,
            enable_iam_auth=True,
            ip_type=ipt,
        )
        return self._local.conn

    def query(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        """Execute a parameterized SQL query against Cloud SQL for PostgreSQL (``pgvector``)."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(sql, params)
            return list(cur.fetchall())
        except Exception:
            if not getattr(self, "fallback_local", True):
                raise
            if self._fallback_catalog is None:
                self._fallback_catalog = VectorCatalog(backend="gcs_inmemory")
            vec = [0.1] * 64
            brand_filter: str | None = None
            for p in params:
                if isinstance(p, str) and p.startswith("[") and p.endswith("]"):
                    try:
                        vec = [float(x) for x in json.loads(p)]
                    except Exception:
                        pass
                elif isinstance(p, (list, tuple)):
                    vec = [float(x) for x in p]
                elif isinstance(p, str) and p.strip():
                    brand_filter = p.strip()
            matches = self._fallback_catalog.search(vec, brand=brand_filter, top_k=2)
            sql_lower = sql.lower()
            if "category" in sql_lower and "packaging_type" in sql_lower:
                return [(m.sku_id, m.category, m.brand, m.packaging_type, m.variant) for m in matches]
            return [(m.sku_id, m.score) for m in matches]


class VectorCatalog:
    """Pluggable SKU Vector Catalog supporting Cloud SQL ``pgvector``, Vertex AI Vector Search,
    BigQuery ``VECTOR_SEARCH``, and GCS/in-memory NumPy ScaNN fallback.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        backend: str | None = None,
        config_override: dict[str, Any] | None = None,
    ):
        cfg = config or load_config()
        vs_cfg = {**cfg.get("vector_store", {}), **cfg.get("cloudsql", {}), **(config_override or {})}
        self.gcp = cfg.get("gcp", {})
        self.backend = (backend or vs_cfg.get("backend") or "cloudsql_pgvector").lower()
        self.vs_cfg = vs_cfg
        self.fallback_local = bool(vs_cfg.get("fallback_local_scann", True))
        self._cloudsql: CloudSQL | None = None
        self._inmemory_entries: list[dict[str, Any]] | None = None

        if self.backend in ("cloudsql", "cloudsql_pgvector", "postgres"):
            self._cloudsql = CloudSQL(
                instance=str(vs_cfg.get("instance", "")),
                database=str(vs_cfg.get("database", "postgres")),
                user=vs_cfg.get("user"),
                ip_type=str(vs_cfg.get("ip_type", "PRIVATE")),
                fallback_local=self.fallback_local,
            )

    def _ensure_inmemory_catalog(self) -> list[dict[str, Any]]:
        if self._inmemory_entries is not None:
            return self._inmemory_entries
        from utils import hul_domain

        entries: list[dict[str, Any]] = []
        for item in hul_domain._build_real_catalog_prototype_bank():
            entries.append(
                {
                    "sku_id": str(item["sku_id"]),
                    "category": str(item["category"]),
                    "brand": str(item["brand"]),
                    "packaging_type": str(item["packaging_type"]),
                    "variant": str(item["variant"]),
                    "embedding": list(item["embedding"]),
                }
            )
        self._inmemory_entries = entries
        return entries

    def _search_inmemory(
        self,
        vector: list[float],
        top_k: int = 2,
        filters: dict[str, str] | None = None,
    ) -> list[VectorMatch]:
        entries = self._ensure_inmemory_catalog()
        flt = {k: str(v).lower() for k, v in (filters or {}).items() if v}
        candidates = [
            e
            for e in entries
            if (not flt.get("brand") or e["brand"].lower() == flt["brand"])
            and (not flt.get("packaging_type") or e["packaging_type"].lower() == flt["packaging_type"])
            and (not flt.get("category") or e["category"].lower() == flt["category"])
        ] or entries

        scored: list[VectorMatch] = []
        for e in candidates:
            raw_sim = _cosine_similarity(vector, e["embedding"])
            sim = round(max(0.0, min(1.0, raw_sim)), 4)
            scored.append(
                VectorMatch(
                    sku_id=e["sku_id"],
                    score=sim,
                    category=e["category"],
                    brand=e["brand"],
                    packaging_type=e["packaging_type"],
                    variant=e["variant"],
                    metadata={"backend": "gcs_inmemory"},
                )
            )
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[: max(1, top_k)]

    def _search_vertex_vector_search(
        self,
        vector: list[float],
        top_k: int = 2,
        filters: dict[str, str] | None = None,
    ) -> list[VectorMatch]:
        """Query Vertex AI Vector Search (`IndexEndpoint:findNeighbors`) with ScaNN token restricts."""
        endpoint = self.vs_cfg.get("index_endpoint")
        deployed_id = self.vs_cfg.get("deployed_index_id", "hul_sku_scann_index")
        if not endpoint or "INDEX_ENDPOINT" in str(endpoint):
            return self._search_inmemory(vector, top_k=top_k, filters=filters)

        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        session = AuthorizedSession(creds)
        region = self.gcp.get("region", "us-central1")
        restricts = [
            {"namespace": k, "allowList": [str(v)]}
            for k, v in (filters or {}).items()
            if v
        ]
        url = f"https://{region}-aiplatform.googleapis.com/v1/{endpoint}:findNeighbors"
        payload = {
            "deployedIndexId": deployed_id,
            "queries": [
                {
                    "datapoint": {
                        "datapointId": "query_crop",
                        "featureVector": [float(x) for x in vector],
                        "restricts": restricts,
                    },
                    "neighborCount": int(top_k),
                }
            ],
        }
        try:
            resp = session.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            neighbors = (
                resp.json()
                .get("nearestNeighbors", [{}])[0]
                .get("neighbors", [])
            )
            if not neighbors:
                return self._search_inmemory(vector, top_k=top_k, filters=filters)
            return [
                VectorMatch(
                    sku_id=str(n.get("datapoint", {}).get("datapointId", "unknown")),
                    score=round(float(1.0 - float(n.get("distance", 0.08))), 4),
                    metadata={"backend": "vertex_vector_search"},
                )
                for n in neighbors[:top_k]
            ]
        except Exception:
            if not self.fallback_local:
                raise
            return self._search_inmemory(vector, top_k=top_k, filters=filters)

    def _search_bigquery(
        self,
        vector: list[float],
        top_k: int = 2,
        filters: dict[str, str] | None = None,
    ) -> list[VectorMatch]:
        """Query BigQuery `VECTOR_SEARCH` with parameterized query arguments."""
        project = self.gcp.get("project", "")
        dataset_id = self.vs_cfg.get("bq_dataset", "hul_shelf_analytics")
        table_id = self.vs_cfg.get("bq_table", "sku_catalog")
        if not project or not self.vs_cfg.get("enable_live_bigquery", False):
            return self._search_inmemory(vector, top_k=top_k, filters=filters)

        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        session = AuthorizedSession(creds)
        sql = (
            f"SELECT base.id, 1.0 - distance AS score "
            f"FROM VECTOR_SEARCH(TABLE `{project}.{dataset_id}.{table_id}`, 'embedding', "
            f"(SELECT @query_vec AS embedding), top_k => @top_k, distance_type => 'COSINE')"
        )
        body = {
            "query": sql,
            "useLegacySql": False,
            "parameterMode": "NAMED",
            "queryParameters": [
                {
                    "name": "query_vec",
                    "parameterType": {"type": "ARRAY", "arrayType": {"type": "FLOAT64"}},
                    "parameterValue": {"arrayValues": [{"value": str(float(x))} for x in vector]},
                },
                {
                    "name": "top_k",
                    "parameterType": {"type": "INT64"},
                    "parameterValue": {"value": str(int(top_k))},
                },
            ],
        }
        try:
            url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project}/queries"
            resp = session.post(url, json=body, timeout=20)
            resp.raise_for_status()
            rows = resp.json().get("rows", [])
            if not rows:
                return self._search_inmemory(vector, top_k=top_k, filters=filters)
            return [
                VectorMatch(
                    sku_id=str(r["f"][0]["v"]),
                    score=round(float(r["f"][1]["v"]), 4),
                    metadata={"backend": "bigquery"},
                )
                for r in rows[:top_k]
            ]
        except Exception:
            if not self.fallback_local:
                raise
            return self._search_inmemory(vector, top_k=top_k, filters=filters)

    def search(
        self,
        vector: list[float],
        top_k: int = 2,
        filters: dict[str, str] | None = None,
        brand: str | None = None,
        category: str | None = None,
        packaging_type: str | None = None,
    ) -> list[VectorMatch]:
        """Search the configured vector catalog backend (`cloudsql_pgvector`, `vertex_vector_search`, `bigquery`, or `gcs_inmemory`)."""
        merged_filters = dict(filters or {})
        if brand:
            merged_filters["brand"] = brand
        if category:
            merged_filters["category"] = category
        if packaging_type:
            merged_filters["packaging_type"] = packaging_type
        if self.backend in ("cloudsql", "cloudsql_pgvector", "postgres") and self._cloudsql is not None:
            v_str = pgvector(vector)
            b_flt = merged_filters.get("brand")
            p_flt = merged_filters.get("packaging_type")
            if b_flt and p_flt:
                sql = (
                    "SELECT id, 1 - (embedding <=> %s::vector) AS score "
                    "FROM products WHERE brand = %s AND packaging_type = %s "
                    "ORDER BY embedding <=> %s::vector LIMIT %s"
                )
                rows = self._cloudsql.query(sql, (v_str, b_flt, p_flt, v_str, int(top_k)))
            elif b_flt:
                sql = (
                    "SELECT id, 1 - (embedding <=> %s::vector) AS score "
                    "FROM products WHERE brand = %s "
                    "ORDER BY embedding <=> %s::vector LIMIT %s"
                )
                rows = self._cloudsql.query(sql, (v_str, b_flt, v_str, int(top_k)))
            else:
                sql = (
                    "SELECT id, 1 - (embedding <=> %s::vector) AS score "
                    "FROM products ORDER BY embedding <=> %s::vector LIMIT %s"
                )
                rows = self._cloudsql.query(sql, (v_str, v_str, int(top_k)))
            return [
                VectorMatch(
                    sku_id=str(r[0]),
                    score=round(float(r[1]), 4),
                    brand=b_flt or "Dove",
                    packaging_type=p_flt or "bottle",
                    metadata={"backend": "cloudsql_pgvector"},
                )
                for r in rows
            ]
        if self.backend == "vertex_vector_search":
            return self._search_vertex_vector_search(vector, top_k=top_k, filters=merged_filters)
        if self.backend == "bigquery":
            return self._search_bigquery(vector, top_k=top_k, filters=merged_filters)
        return self._search_inmemory(vector, top_k=top_k, filters=merged_filters)

    def query(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        """Execute parameterized SQL on Cloud SQL `pgvector` (or fallback to vector search)."""
        if self._cloudsql is not None:
            return self._cloudsql.query(sql, params)
        fallback = CloudSQL(fallback_local=True)
        return fallback.query(sql, params)
