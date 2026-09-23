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

import csv
import io
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from shelf_benchmark.auth import create_bigquery_client
from shelf_benchmark.config import AssociationConfig, AssociationSchemaMapping, BucketConfig
from shelf_benchmark.data.field_access import extract_field
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.models import ShelfAssociationRecord

logger = logging.getLogger(__name__)

VALID_ASSOCIATION_PROVIDER_TYPES = ("json", "jsonl", "csv", "bigquery", "bq", "bucket_discovery")

#: Provider types that read a named source. `bucket_discovery` is the only one that does not.
_SOURCE_BACKED_PROVIDER_TYPES = frozenset({"json", "jsonl", "csv", "bigquery", "bq"})


class AssociationConfigError(ValueError):
    """Raised when `AssociationConfig` cannot be resolved to a concrete provider."""


class AssociationParseError(ValueError):
    """Raised when an association source is reachable but cannot be parsed."""


#: Deprecated private alias kept so any in-tree caller of the old name keeps working.
_extract_nested = extract_field



class BaseAssociationProvider(ABC):
    """Abstract interface for loading Shelf <-> Catalog <-> Planogram associations across any table/file schema."""

    def __init__(
        self,
        schema_mapping: AssociationSchemaMapping,
        default_shelf_bucket: Optional[str] = None,
    ):
        self.mapping = schema_mapping
        self.default_shelf_bucket = default_shelf_bucket

    def map_raw_row(self, row: Dict[str, Any], index: int = 1) -> ShelfAssociationRecord:
        """Map an arbitrary row dictionary (BigQuery, CSV, JSON, AutoML manifest) into a canonical ShelfAssociationRecord."""
        m = self.mapping
        assoc_id = str(
            extract_field(row, m.association_id_field, "association_id", "id", "record_id", "session_id")
            or f"assoc-{index:04d}"
        )
        raw_uri = str(
            extract_field(
                row,
                m.shelf_image_uri_field,
                "shelf_image_uri",
                "image_uri",
                "gcs_uri",
                "image_gcs_uri",
                "file_uri",
                "photo_url",
                "image_path",
                "blob_name",
                "file_name",
            )
            or ""
        ).strip()

        local_path = extract_field(row, "local_shelf_image_path", "local_path")
        if (
            raw_uri
            and not raw_uri.startswith(("gs://", "http://", "https://", "/"))
            and self.default_shelf_bucket
            and not (local_path or Path(raw_uri).exists())
        ):
            shelf_uri = f"{self.default_shelf_bucket.rstrip('/')}/{raw_uri.lstrip('/')}"
        else:
            shelf_uri = raw_uri

        store_id = extract_field(row, m.store_id_field, "store_id", "store_code", "outlet_id", "metadata.store_id")
        catalog_uri = extract_field(row, m.catalog_uri_field, "catalog_uri", "catalog_gcs_uri")
        planogram_uri = extract_field(row, m.planogram_uri_field, "planogram_uri", "planogram_gcs_uri")
        gt_id = extract_field(row, m.ground_truth_id_field, "ground_truth_id", "image_id") or (
            shelf_uri.split("/")[-1] if shelf_uri else f"img-{index:04d}"
        )

        return ShelfAssociationRecord(
            association_id=assoc_id,
            shelf_image_uri=shelf_uri,
            local_shelf_image_path=str(local_path) if local_path else None,
            store_id=str(store_id) if store_id is not None else None,
            aisle_category=extract_field(row, "aisle_category", "category", "department"),
            catalog_uri=str(catalog_uri) if catalog_uri else None,
            planogram_uri=str(planogram_uri) if planogram_uri else None,
            ground_truth_id=str(gt_id),
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
    """Loads associations from a JSON array, `{records/images: [...]}` object, or JSONL file (local or `gs://`)."""

    def __init__(
        self,
        source_uri: str,
        storage_manager: StorageManager,
        schema_mapping: AssociationSchemaMapping,
        default_shelf_bucket: Optional[str] = None,
    ):
        super().__init__(schema_mapping, default_shelf_bucket=default_shelf_bucket)
        self.source_uri = source_uri
        self.storage = storage_manager

    def load_associations(self) -> List[ShelfAssociationRecord]:
        text = self.storage.read_text(self.source_uri).strip()
        if not text:
            return []
        rows = self._parse_rows(text)
        return [self.map_raw_row(r, i + 1) for i, r in enumerate(rows)]

    def _parse_rows(self, text: str) -> List[Dict[str, Any]]:
        """Parse a JSON array, a JSON object wrapping a record list, or a JSONL document.

        The document is parsed as a whole first. The previous heuristic routed any source
        containing a newline into the JSONL branch unless it started with `[`, so an
        ordinary pretty-printed JSON object failed with a confusing
        "Expecting value: line 1 column 2" instead of loading.
        """
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return self._parse_jsonl(text)

        if isinstance(parsed, list):
            return self._require_dict_rows(parsed)
        if isinstance(parsed, dict):
            for wrapper_key in ("records", "associations", "images", "items"):
                if wrapper_key in parsed:
                    value = parsed[wrapper_key]
                    if not isinstance(value, list):
                        raise AssociationParseError(
                            f"Association source '{self.source_uri}' has "
                            f"'{wrapper_key}' of type {type(value).__name__}; expected a list."
                        )
                    # An explicitly present but empty list means "no associations", not
                    # "keep looking under another key".
                    return self._require_dict_rows(value)
            return [parsed]
        raise AssociationParseError(
            f"Association source '{self.source_uri}' has unsupported JSON root type "
            f"{type(parsed).__name__}; expected an array, an object, or JSONL."
        )

    def _parse_jsonl(self, text: str) -> List[Dict[str, Any]]:
        rows: List[Any] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssociationParseError(
                    f"Association source '{self.source_uri}' is neither valid JSON nor "
                    f"valid JSONL (line {line_no}): {exc}"
                ) from exc
        return self._require_dict_rows(rows)

    def _require_dict_rows(self, rows: List[Any]) -> List[Dict[str, Any]]:
        """Reject non-object rows rather than skipping them.

        Dropping them would shrink the benchmarked image set without any signal, which is
        exactly the failure mode this module is meant to avoid.
        """
        for position, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise AssociationParseError(
                    f"Association source '{self.source_uri}' record #{position} is a "
                    f"{type(row).__name__}, not an object: {row!r}."
                )
        return list(rows)



class CSVAssociationProvider(BaseAssociationProvider):
    """Loads associations from a CSV file (local or `gs://`)."""

    def __init__(
        self,
        source_uri: str,
        storage_manager: StorageManager,
        schema_mapping: AssociationSchemaMapping,
        default_shelf_bucket: Optional[str] = None,
    ):
        super().__init__(schema_mapping, default_shelf_bucket=default_shelf_bucket)
        self.source_uri = source_uri
        self.storage = storage_manager

    def load_associations(self) -> List[ShelfAssociationRecord]:
        text = self.storage.read_text(self.source_uri)
        reader = csv.DictReader(io.StringIO(text))
        return [self.map_raw_row(dict(r), i + 1) for i, r in enumerate(reader)]


class BigQueryAssociationProvider(BaseAssociationProvider):
    """Loads associations from a BigQuery table or SQL query (`project.dataset.table` or `SELECT ...`)."""

    def __init__(
        self,
        project_id: str,
        table_or_query: str,
        schema_mapping: AssociationSchemaMapping,
        default_shelf_bucket: Optional[str] = None,
    ):
        super().__init__(schema_mapping, default_shelf_bucket=default_shelf_bucket)
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
    """Discovers images directly in `shelf_images_bucket` when no association table exists yet.

    This provider *invents* the catalog and planogram URIs by convention
    (`<bucket>/sample_catalog.json`). Those objects frequently do not exist, and the
    optional-artifact readers downstream return `None` for a missing catalog, so the
    assumption is logged rather than made silently.
    """

    def __init__(self, bucket_config: BucketConfig, storage_manager: StorageManager, schema_mapping: AssociationSchemaMapping):
        super().__init__(schema_mapping, default_shelf_bucket=bucket_config.shelf_images_bucket)
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
        if default_catalog or default_planogram:
            logger.warning(
                "Bucket discovery is assuming catalog_uri=%r and planogram_uri=%r by naming "
                "convention; these objects are not verified to exist.",
                default_catalog,
                default_planogram,
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
        if not records:
            logger.warning(
                "Bucket discovery found no images in %r. The run will benchmark 0 images.",
                self.buckets.shelf_images_bucket,
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
    """Factory to build the configured association provider.

    Raises `AssociationConfigError` for an unrecognised `provider_type`, and for a
    source-backed `provider_type` with no `source_uri`, instead of falling through to
    `BucketDiscoveryAssociationProvider`. That fall-through meant a typo such as
    `provider_type: bigqeury` silently swapped the benchmark's input set for a scan of the
    shelf-images bucket: the run still succeeded, still produced numbers, and nothing in
    the report said the association table had never been read. Selecting bucket discovery
    is now something a config has to ask for by name.
    """
    ptype = (assoc_config.provider_type or "").strip().lower()
    default_bucket = bucket_config.shelf_images_bucket

    if ptype not in VALID_ASSOCIATION_PROVIDER_TYPES:
        raise AssociationConfigError(
            f"Unknown associations.provider_type {assoc_config.provider_type!r}. "
            f"Expected one of: {', '.join(VALID_ASSOCIATION_PROVIDER_TYPES)}. "
            f"Refusing to default to bucket discovery, because that would silently "
            f"benchmark every image in {default_bucket!r} instead of the configured "
            f"association table."
        )

    if ptype in _SOURCE_BACKED_PROVIDER_TYPES and not assoc_config.source_uri:
        raise AssociationConfigError(
            f"associations.provider_type={ptype!r} needs `associations.source_uri` to "
            f"point at the file or table to read, but it is empty. Set it, or set "
            f"provider_type='bucket_discovery' to deliberately enumerate "
            f"{default_bucket!r} instead."
        )

    if ptype in ("json", "jsonl"):
        return JSONAssociationProvider(
            assoc_config.source_uri,
            storage_manager,
            assoc_config.schema_mapping,
            default_shelf_bucket=default_bucket,
        )
    if ptype == "csv":
        return CSVAssociationProvider(
            assoc_config.source_uri,
            storage_manager,
            assoc_config.schema_mapping,
            default_shelf_bucket=default_bucket,
        )
    if ptype in ("bigquery", "bq"):
        return BigQueryAssociationProvider(
            project_id,
            assoc_config.source_uri,
            assoc_config.schema_mapping,
            default_shelf_bucket=default_bucket,
        )
    return BucketDiscoveryAssociationProvider(bucket_config, storage_manager, assoc_config.schema_mapping)

