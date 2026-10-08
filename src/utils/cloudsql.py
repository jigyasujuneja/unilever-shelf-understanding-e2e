"""Cloud SQL for PostgreSQL (`pgvector`) connection helper. Mirrors ``src/utils/alloydb.py``.

Connects through either:
1. ``cloud-sql-python-connector`` (if installed), or
2. Built-in Cloud SQL Admin API ephemeral client certificate mTLS (TLS 1.3) on port 3307 + ``pg8000``
   authorised by IAM using Application Default Credentials (locally) or the Cloud Run service
   account (in the cloud). Zero passwords, zero IP allow-lists, and zero extra pip dependencies.

One-time setup (psql / Cloud SQL Studio)::

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS products (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT DEFAULT '',
        brand TEXT DEFAULT '',
        pack_form TEXT DEFAULT '',
        size TEXT DEFAULT '',
        variant TEXT DEFAULT '',
        is_unilever BOOLEAN DEFAULT FALSE,
        embedding vector(768),
        lab_vec vector(13),
        metadata JSONB DEFAULT '{}'::jsonb
    );
    CREATE INDEX IF NOT EXISTS products_embedding_hnsw_idx
        ON products USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);

IAM: the caller needs ``roles/cloudsql.client`` + ``roles/cloudsql.instanceUser`` and an IAM
database user (``gcloud sql users create <sa-email minus .gserviceaccount.com>
--instance=<INSTANCE> --type=CLOUD_IAM_SERVICE_ACCOUNT``) with least-privilege ``SELECT`` on the
catalog tables.

Cost advantage in sandboxes: unlike AlloyDB (which runs 24/7 at >= 2 vCPU), a Cloud SQL instance
can be stopped between benchmark runs (``gcloud sql instances patch <INSTANCE>
--activation-policy=NEVER``), dropping idle compute cost to $0.00/hr.
"""

from __future__ import annotations

import socket
import ssl
import tempfile
import threading
from typing import Any

from utils.alloydb import iam_user, pgvector

__all__ = ["CloudSQL", "iam_user", "pgvector", "SCHEMA_SQL_CLOUDSQL", "HYBRID_RRF_SQL"]

SCHEMA_SQL_CLOUDSQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT DEFAULT '',
    brand TEXT DEFAULT '',
    pack_form TEXT DEFAULT '',
    size TEXT DEFAULT '',
    variant TEXT DEFAULT '',
    is_unilever BOOLEAN DEFAULT FALSE,
    embedding vector(768),
    lab_vec vector(13),
    metadata JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS products_embedding_hnsw_idx
    ON products USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS products_category_pack_idx
    ON products (category, pack_form);
CREATE TABLE IF NOT EXISTS training_crops (
    md5_hash TEXT PRIMARY KEY,
    crop_uri TEXT NOT NULL,
    brand_data_uri TEXT,
    variant_uri TEXT,
    pack_data_uri TEXT,
    join_mode TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'clean',
    session_id TEXT DEFAULT '',
    category TEXT DEFAULT '',
    brand TEXT DEFAULT '',
    variant TEXT DEFAULT '',
    base_variant TEXT DEFAULT '',
    basepack_id TEXT DEFAULT '',
    packaging_type TEXT DEFAULT '',
    is_hul BOOLEAN DEFAULT FALSE,
    embedding vector(768)
);
CREATE INDEX IF NOT EXISTS training_crops_embedding_hnsw_idx
    ON training_crops USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""

# Parameterized Hybrid Vector (HNSW) + Full-Text (tsvector) Reciprocal Rank Fusion + Category/Pack filter.
# Both CTEs (`v` and `k`) enforce identical `category` and `pack_form` predicates before `LIMIT 20`,
# and `FULL OUTER JOIN` ensures exact keyword matches outside the top-20 vector hits still contribute
# their RRF rank (`1 / (60 + r_kw)`).
HYBRID_RRF_SQL = """
WITH v AS (
    SELECT id, name, category, pack_form, size,
           1 - (embedding <=> %s::vector) AS vec_sim,
           RANK() OVER (ORDER BY embedding <=> %s::vector) AS r_vec
    FROM products
    WHERE (%s = '' OR category = %s)
      AND (%s = '' OR pack_form = %s)
    ORDER BY embedding <=> %s::vector
    LIMIT 20
),
k AS (
    SELECT id,
           RANK() OVER (
               ORDER BY ts_rank(
                   to_tsvector('english', name || ' ' || size || ' ' || variant),
                   plainto_tsquery('english', %s)
               ) DESC
           ) AS r_kw
    FROM products
    WHERE %s <> ''
      AND (%s = '' OR category = %s)
      AND (%s = '' OR pack_form = %s)
      AND to_tsvector('english', name || ' ' || size || ' ' || variant)
          @@ plainto_tsquery('english', %s)
    LIMIT 20
)
SELECT COALESCE(v.id, k.id) AS id,
       COALESCE(v.vec_sim, 0.0) AS vec_sim,
       COALESCE(1.0 / (60 + v.r_vec), 0.0) + COALESCE(1.0 / (60 + k.r_kw), 0.0) AS rrf_score
FROM v
FULL OUTER JOIN k ON v.id = k.id
ORDER BY rrf_score DESC, COALESCE(v.vec_sim, 0.0) DESC
LIMIT %s;
"""


class _AdminApiCloudSQLConnector:
    """Zero-dependency Cloud SQL Admin API mTLS (TLS 1.3) + IAM auth connector for ``pg8000``.

    Follows the exact protocol of ``cloud-sql-python-connector``:
    1. Generates an ephemeral 2048-bit RSA keypair in memory.
    2. Fetches instance CA certificate and IP addresses from ``connectSettings``.
    3. Requests a short-lived mTLS client certificate bound to the IAM access token via
       ``generateEphemeralCert``.
    4. Establishes a verified TLS 1.3 socket to port 3307 (Cloud SQL Auth Proxy server port)
       and hands it to ``pg8000.dbapi.connect``.
    """

    def __init__(self) -> None:
        import google.auth
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from google.auth.transport.requests import AuthorizedSession

        self._creds, _ = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/sqlservice.admin",
                "https://www.googleapis.com/auth/cloud-platform",
            ]
        )
        self._session = AuthorizedSession(self._creds)
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self._pub_pem = (
            self._key.public_key()
            .public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode("utf-8")
        )
        self._priv_pem = self._key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def connect(
        self,
        instance_connection_string: str,
        driver: str,
        *,
        user: str,
        db: str,
        enable_iam_auth: bool = True,
        ip_type: str = "PUBLIC",
    ):
        import pg8000.dbapi
        from google.auth.transport.requests import Request

        parts = instance_connection_string.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Cloud SQL instance must be 'project:region:instance', "
                f"got {instance_connection_string!r}"
            )
        project, _region, instance = parts
        base = f"https://sqladmin.googleapis.com/sql/v1beta4/projects/{project}/instances/{instance}"

        settings_resp = self._session.get(f"{base}/connectSettings", timeout=30)
        settings_resp.raise_for_status()
        settings = settings_resp.json()
        server_ca_pem = settings["serverCaCert"]["cert"]

        ip_map = { entry["type"]: entry["ipAddress"] for entry in settings.get("ipAddresses", []) }
        target_type = "PRIVATE" if str(ip_type).upper().endswith("PRIVATE") else "PRIMARY"
        host_ip = ip_map.get(target_type) or ip_map.get("PRIMARY") or ip_map.get("PRIVATE")
        if not host_ip:
            raise RuntimeError(f"No usable IP address found for Cloud SQL instance {instance_connection_string}")

        if not self._creds.valid:
            self._creds.refresh(Request())

        body: dict[str, Any] = {"public_key": self._pub_pem}
        if enable_iam_auth and self._creds.token:
            body["access_token"] = self._creds.token

        cert_resp = self._session.post(f"{base}:generateEphemeralCert", json=body, timeout=30)
        cert_resp.raise_for_status()
        client_cert_pem = cert_resp.json()["ephemeralCert"]["cert"]

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.check_hostname = False  # Cloud SQL server cert CN is project:instance, verified via CA chain
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cadata=server_ca_pem)

        with tempfile.NamedTemporaryFile(mode="wb", delete=True) as cert_file, \
             tempfile.NamedTemporaryFile(mode="wb", delete=True) as key_file:
            cert_file.write(client_cert_pem.encode("utf-8"))
            cert_file.flush()
            key_file.write(self._priv_pem)
            key_file.flush()
            ctx.load_cert_chain(certfile=cert_file.name, keyfile=key_file.name)

        raw_sock = socket.create_connection((host_ip, 3307), timeout=30)
        tls_sock = ctx.wrap_socket(raw_sock, server_hostname=host_ip)
        password = self._creds.token if enable_iam_auth else None
        return pg8000.dbapi.connect(user=user, database=db, password=password, sock=tls_sock)


class CloudSQL:
    """``CloudSQL(**config["cloudsql"]).query(sql, params)`` -> list of row tuples. Thread-safe.

    Strictly enforces parameterized SQL queries (`%s`) and mTLS + IAM database authentication.
    """

    def __init__(
        self,
        instance: str,
        database: str = "postgres",
        user: str | None = None,
        ip_type: str = "PUBLIC",
        **_: Any,
    ):
        if not instance or ":" not in instance:
            raise ValueError(
                f"CloudSQL instance must be 'project:region:instance', got {instance!r}"
            )
        self.instance = instance
        self.database = database
        self.user = user or iam_user()
        self.ip_type = ip_type.upper()
        try:
            from google.cloud.sql.connector import (  # type: ignore[import-not-found]
                Connector,
                IPTypes,
            )

            self.ip_type = IPTypes[self.ip_type]
            self.connector: Any = Connector(refresh_strategy="lazy")
        except ImportError:
            self.connector = _AdminApiCloudSQLConnector()
        self._local = threading.local()  # pg8000 connections aren't thread-safe: one per worker

    def _conn(self):
        if getattr(self._local, "conn", None) is None:
            self._local.conn = self.connector.connect(
                self.instance,
                "pg8000",
                user=self.user,
                db=self.database,
                enable_iam_auth=True,
                ip_type=self.ip_type,
            )
        return self._local.conn

    def query(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        """Execute a parameterized SELECT query and return all rows."""
        if not isinstance(params, (tuple, list)):
            raise TypeError("SQL parameters must be passed as a tuple or list")
        cur = self._conn().cursor()
        cur.execute(sql, params)
        return cur.fetchall()

    def execute(self, sql: str, params: tuple | list = ()) -> int:
        """Execute a parameterized DDL/DML statement and commit the transaction."""
        if not isinstance(params, (tuple, list)):
            raise TypeError("SQL parameters must be passed as a tuple or list")
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        return getattr(cur, "rowcount", 0)
