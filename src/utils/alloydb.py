"""AlloyDB for PostgreSQL connection helper. Approaches write their own SQL on top of it.

Connects through the AlloyDB Python Connector: an mTLS connection authorised by IAM with the
same credentials as everything else (your ADC locally, the Cloud Run job's service account in
the cloud). No passwords, no IP allow-lists. Private-IP instances are reached from Cloud Run
via Direct VPC egress (``cloud_run.network`` / ``cloud_run.subnet`` in config.yaml).

One-time setup (psql / AlloyDB Studio)::

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS alloydb_scann;      -- optional, fast ANN index
    CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, embedding vector(512), metadata JSONB);
    CREATE INDEX ON products USING scann (embedding cosine) WITH (num_leaves = 100);

IAM: the caller needs ``roles/alloydb.client`` + ``roles/serviceusage.serviceUsageConsumer`` and
an IAM database user (``gcloud alloydb users create <sa-email minus .gserviceaccount.com>
--type=IAM_BASED ...``) with SELECT on the table.

Billed per instance vCPU/memory-hour, not per query: a fixed monthly cost, not per image.
"""

from __future__ import annotations

import threading
from typing import Any


class AlloyDB:
    """``AlloyDB(**config["alloydb"]).query(sql, params)`` -> list of row tuples. Thread-safe."""

    def __init__(self, instance: str, database: str = "postgres", user: str | None = None,
                 ip_type: str = "PRIVATE", **_: Any):
        from google.cloud.alloydb.connector import Connector, IPTypes

        self.instance, self.database = instance, database
        self.user = user or iam_user()
        self.ip_type = IPTypes[ip_type.upper()]
        self.connector = Connector(refresh_strategy="lazy")
        self._local = threading.local()  # pg8000 connections aren't thread-safe: one per worker

    def query(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        if getattr(self._local, "conn", None) is None:
            self._local.conn = self.connector.connect(
                self.instance, "pg8000", user=self.user, db=self.database,
                enable_iam_auth=True, ip_type=self.ip_type)
        cur = self._local.conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()


def pgvector(vector: list[float]) -> str:
    """A Python list as a pgvector literal; use as ``%s::vector`` in SQL."""
    return "[" + ",".join(map(str, vector)) + "]"


def iam_user() -> str:
    """IAM database user for the current credentials (SA email minus '.gserviceaccount.com')."""
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    email = getattr(creds, "service_account_email", None)
    if not email or email == "default":
        creds.refresh(Request())
        email = getattr(creds, "service_account_email", None)
    if not email or "@" not in email:
        raise RuntimeError("Set alloydb.user in config.yaml (your IAM database user)")
    return email.removesuffix(".gserviceaccount.com")
