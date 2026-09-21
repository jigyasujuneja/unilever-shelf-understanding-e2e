"""Pluggable Association Table Adapters & Schema Mapper.

Because the location (BigQuery, GCS JSON/CSV, local file, or bucket convention)
and column schema of the association table linking:
  - input shelf images bucket
  - product catalog images bucket
  - optional planograms bucket
are not known in advance, this module provides a swappable adapter pattern
(`BaseAssociationProvider` + `AssociationSchemaMapping`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import io
import json
from typing import Any, Callable, Dict, List, Optional

from shelf_benchmark.auth import create_bigquery_client
from shelf_benchmark.config import AssociationConfig, AssociationSchemaMapping, BucketConfig
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.models import ShelfAssociationRecord


class BaseAssociationProvider(ABC):
    """Abstract interface for loading Shelf <-> Catalog <-> Planogram associations."""

    def __init__(self, schema_mapping: AssociationSchemaMapping):
        self.mapping = schema_mapping

    def map_raw_row(self, row: Dict[str, Any], index: int = 1) -> ShelfAssociationRecord:
        """Map an arbitrary row dictionary into a canonical ShelfAssociationRecord."""
        m = self.mapping
        assoc_id = str(row.get(m.association_id_field) or row.get("id") or f"assoc-{index:04d}")
        shelf_uri = str(
            row.get(m.shelf_image_uri_field)
            or row.get("image_uri")
            or row.get("shelf_uri")
            or ""
        )
        return ShelfAssociationRecord(
            association_id=assoc_id,
            shelf_image_uri=shelf_uri,
            local_shelf_image_path=row.get("local_shelf_image_path"),
            store_id=row.get(m.store_id_field),
            aisle_category=row.get("aisle_category"),
            catalog_uri=row.get(m.catalog_uri_field),
            planogram_uri=row.get(m.planogram_uri_field),
            ground_truth_id=row.get(m.ground_truth_id_field) or shelf_uri.split("/")[-1],
            metadata={
                k: v
                for k, v in row.items()
                if k
                not in {
                    m.association_id_field,
                    m.shelf_image_uri_field,
                    m.catalog_uri_field,
                    m.planogram_uri_field,
                    m.store_id_field,
                    m.ground_truth_id_field,
                }
            },
        )

    @abstractmethod
    def load_associations(self) -> List[ShelfAssociationRecord]:
        """Return canonical association records."""


class JSONAssociationProvider(BaseAssociationProvider):
    """Loads associations from a JSON array or JSONL file (local or `gs://`)."""

    def __init__(self, source_uri: str, storage_manager: StorageManager, schema_mapping: AssociationSchemaMapping):
        super().__init__(schema_mapping)
        self.source_uri = source_uri
        self.storage = storage_manager

    def load_associations(self) -> List[ShelfAssociationRecord]:
        text = self.storage.read_text(self.source_uri).strip()
        if not text:
            return []
        if text.startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [self.map_raw_row(r, i + 1) for i, r in enumerate(rows)]


class CSVAssociationProvider(BaseAssociationProvider):
    """Loads associations from a CSV file (local or `gs://`)."""

    def __init__(self, source_uri: str, storage_manager: StorageManager, schema_mapping: AssociationSchemaMapping):
        super().__init__(schema_mapping)
        self.source_uri = source_uri
        self.storage = storage_manager

    def load_associations(self) -> List[ShelfAssociationRecord]:
        text = self.storage.read_text(self.source_uri)
        reader = csv.DictReader(io.StringIO(text))
        return [self.map_raw_row(dict(r), i + 1) for i, r in enumerate(reader)]


class BigQueryAssociationProvider(BaseAssociationProvider):
    """Loads associations from a BigQuery table or SQL query (`project.dataset.table` or `SELECT ...`)."""

    def __init__(self, project_id: str, table_or_query: str, schema_mapping: AssociationSchemaMapping):
        super().__init__(schema_mapping)
        self.project_id = project_id
        self.table_or_query = table_or_query

    def load_associations(self) -> List[ShelfAssociationRecord]:
        bq = create_bigquery_client(self.project_id)
        sql = (
            self.table_or_query
            if self.table_or_query.strip().upper().startswith("SELECT")
            else f"SELECT * FROM `{self.table_or_query}`"
        )
        rows = [dict(r) for r in bq.query(sql).result()]
        return [self.map_raw_row(r, i + 1) for i, r in enumerate(rows)]


class BucketDiscoveryAssociationProvider(BaseAssociationProvider):
    """Fallback provider when no association table exists yet: discovers images in `shelf_images_bucket`."""

    def __init__(self, bucket_config: BucketConfig, storage_manager: StorageManager, schema_mapping: AssociationSchemaMapping):
        super().__init__(schema_mapping)
        self.buckets = bucket_config
        self.storage = storage_manager

    def load_associations(self) -> List[ShelfAssociationRecord]:
        uris = self.storage.list_gcs_images(self.buckets.shelf_images_bucket)
        records: List[ShelfAssociationRecord] = []
        default_catalog = (
            f"{self.buckets.catalog_images_bucket.rstrip('/')}/sample_catalog.json"
            if self.buckets.catalog_images_bucket
            else None
        )
        default_planogram = (
            f"{self.buckets.planograms_bucket.rstrip('/')}/sample_planogram.json"
            if self.buckets.planograms_bucket
            else None
        )
        for idx, uri in enumerate(uris, start=1):
            records.append(
                ShelfAssociationRecord(
                    association_id=f"auto-{idx:04d}",
                    shelf_image_uri=uri,
                    catalog_uri=default_catalog,
                    planogram_uri=default_planogram,
                    ground_truth_id=uri.split("/")[-1],
                )
            )
        return records


class CustomCallableAssociationProvider(BaseAssociationProvider):
    """Allows callers to inject an arbitrary Python function returning raw dicts."""

    def __init__(self, fn: Callable[[], List[Dict[str, Any]]], schema_mapping: AssociationSchemaMapping):
        super().__init__(schema_mapping)
        self.fn = fn

    def load_associations(self) -> List[ShelfAssociationRecord]:
        return [self.map_raw_row(r, i + 1) for i, r in enumerate(self.fn())]


def create_association_provider(
    assoc_config: AssociationConfig,
    bucket_config: BucketConfig,
    storage_manager: StorageManager,
    project_id: str = "unilever-shelf-understanding",
) -> BaseAssociationProvider:
    """Factory to build the configured association provider."""
    ptype = assoc_config.provider_type.lower()
    if ptype in ("json", "jsonl") and assoc_config.source_uri:
        return JSONAssociationProvider(assoc_config.source_uri, storage_manager, assoc_config.schema_mapping)
    if ptype == "csv" and assoc_config.source_uri:
        return CSVAssociationProvider(assoc_config.source_uri, storage_manager, assoc_config.schema_mapping)
    if ptype in ("bigquery", "bq") and assoc_config.source_uri:
        return BigQueryAssociationProvider(project_id, assoc_config.source_uri, assoc_config.schema_mapping)
    return BucketDiscoveryAssociationProvider(bucket_config, storage_manager, assoc_config.schema_mapping)
