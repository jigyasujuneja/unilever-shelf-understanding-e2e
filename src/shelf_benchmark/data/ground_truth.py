"""Pluggable Ground Truth Providers & Schema Mapper.

Because the Ground Truth schema and storage location are not finalized yet,
this module allows swapping any JSON/JSONL, CSV, COCO, BigQuery, or custom Python
ground-truth source via configurable field mappings (`GroundTruthSchemaMapping`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import io
import json
from typing import Any, Dict, List, Optional

from shelf_benchmark.auth import create_bigquery_client
from shelf_benchmark.config import GroundTruthConfig, GroundTruthSchemaMapping
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.models import GroundTruthProductItem, ImageGroundTruth


class BaseGroundTruthProvider(ABC):
    """Abstract interface for loading ground truth annotations."""

    def __init__(self, schema_mapping: GroundTruthSchemaMapping):
        self.mapping = schema_mapping

    def map_raw_item(self, raw_item: Dict[str, Any], idx: int = 1) -> GroundTruthProductItem:
        m = self.mapping
        bbox = raw_item.get(m.bbox_field) or raw_item.get("bbox") or [0, 0, 0, 0]
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        return GroundTruthProductItem(
            item_id=int(raw_item.get("item_id") or idx),
            brand=str(raw_item.get(m.brand_field) or raw_item.get("brand_name") or ""),
            product_name=str(raw_item.get(m.product_name_field) or raw_item.get("title") or ""),
            sku_id=raw_item.get(m.sku_id_field) or raw_item.get("sku"),
            bbox_2d=[int(x) for x in bbox[:4]] if len(bbox) >= 4 else [0, 0, 0, 0],
            shelf_row=str(raw_item.get(m.shelf_row_field) or "middle"),
        )

    @abstractmethod
    def get_ground_truth(self, shelf_image_uri: str, ground_truth_id: Optional[str] = None) -> Optional[ImageGroundTruth]:
        """Lookup ground truth for a given shelf image URI or ground truth ID."""


class NullGroundTruthProvider(BaseGroundTruthProvider):
    """Used when no ground truth data is provided yet."""

    def __init__(self):
        super().__init__(GroundTruthSchemaMapping())

    def get_ground_truth(self, shelf_image_uri: str, ground_truth_id: Optional[str] = None) -> Optional[ImageGroundTruth]:
        return None


class JSONGroundTruthProvider(BaseGroundTruthProvider):
    """Loads ground truth from a JSON or JSONL file (local or `gs://`) with flexible schema mapping."""

    def __init__(self, source_uri: str, storage_manager: StorageManager, schema_mapping: GroundTruthSchemaMapping):
        super().__init__(schema_mapping)
        self.source_uri = source_uri
        self.storage = storage_manager
        self._cache: Dict[str, ImageGroundTruth] = {}
        self._load()

    def _load(self) -> None:
        raw_data = self.storage.read_json(self.source_uri)
        if not raw_data:
            return
        m = self.mapping
        images_dict = raw_data.get("images", raw_data) if isinstance(raw_data, dict) else {}
        if isinstance(images_dict, dict):
            for key, val in images_dict.items():
                if not isinstance(val, dict):
                    continue
                raw_items = val.get(m.items_list_field, [])
                items = [self.map_raw_item(it, idx + 1) for idx, it in enumerate(raw_items)]
                brands = val.get("expected_brands") or sorted({it.brand for it in items if it.brand})
                gt = ImageGroundTruth(
                    image_id=str(val.get(m.image_key_field) or key),
                    total_main_shelf_facings=int(val.get("total_main_shelf_facings") or len(items)),
                    expected_brands=brands,
                    items=items,
                )
                self._cache[key] = gt
                self._cache[gt.image_id] = gt
                self._cache[key.split("/")[-1]] = gt
        elif isinstance(raw_data, list):
            # List of image ground truth objects
            for entry in raw_data:
                key = str(entry.get(m.image_key_field) or entry.get("shelf_image_uri") or "")
                raw_items = entry.get(m.items_list_field, [])
                items = [self.map_raw_item(it, idx + 1) for idx, it in enumerate(raw_items)]
                brands = entry.get("expected_brands") or sorted({it.brand for it in items if it.brand})
                gt = ImageGroundTruth(
                    image_id=key,
                    total_main_shelf_facings=len(items),
                    expected_brands=brands,
                    items=items,
                )
                self._cache[key] = gt
                self._cache[key.split("/")[-1]] = gt

    def get_ground_truth(self, shelf_image_uri: str, ground_truth_id: Optional[str] = None) -> Optional[ImageGroundTruth]:
        for candidate in (shelf_image_uri, ground_truth_id, shelf_image_uri.split("/")[-1]):
            if candidate and candidate in self._cache:
                return self._cache[candidate]
        return None


class CSVGroundTruthProvider(BaseGroundTruthProvider):
    """Loads flat row-per-product ground truth from a CSV file (local or `gs://`)."""

    def __init__(self, source_uri: str, storage_manager: StorageManager, schema_mapping: GroundTruthSchemaMapping):
        super().__init__(schema_mapping)
        self.source_uri = source_uri
        self.storage = storage_manager
        self._cache: Dict[str, ImageGroundTruth] = {}
        self._load()

    def _load(self) -> None:
        text = self.storage.read_text(self.source_uri)
        reader = csv.DictReader(io.StringIO(text))
        grouped: Dict[str, List[GroundTruthProductItem]] = {}
        for idx, row in enumerate(reader, start=1):
            img_key = str(row.get(self.mapping.image_key_field) or row.get("shelf_image_uri") or "")
            grouped.setdefault(img_key, []).append(self.map_raw_item(dict(row), idx))
        for img_key, items in grouped.items():
            gt = ImageGroundTruth(
                image_id=img_key,
                total_main_shelf_facings=len(items),
                expected_brands=sorted({it.brand for it in items if it.brand}),
                items=items,
            )
            self._cache[img_key] = gt
            self._cache[img_key.split("/")[-1]] = gt

    def get_ground_truth(self, shelf_image_uri: str, ground_truth_id: Optional[str] = None) -> Optional[ImageGroundTruth]:
        for candidate in (shelf_image_uri, ground_truth_id, shelf_image_uri.split("/")[-1]):
            if candidate and candidate in self._cache:
                return self._cache[candidate]
        return None


class BigQueryGroundTruthProvider(BaseGroundTruthProvider):
    """Loads ground truth rows from a BigQuery table (`project.dataset.table` or SQL query)."""

    def __init__(self, project_id: str, table_or_query: str, schema_mapping: GroundTruthSchemaMapping):
        super().__init__(schema_mapping)
        self.project_id = project_id
        self.table_or_query = table_or_query
        self._cache: Dict[str, ImageGroundTruth] = {}
        self._load()

    def _load(self) -> None:
        bq = create_bigquery_client(self.project_id)
        sql = (
            self.table_or_query
            if self.table_or_query.strip().upper().startswith("SELECT")
            else f"SELECT * FROM `{self.table_or_query}`"
        )
        grouped: Dict[str, List[GroundTruthProductItem]] = {}
        for idx, r in enumerate(bq.query(sql).result(), start=1):
            row = dict(r)
            img_key = str(row.get(self.mapping.image_key_field) or row.get("shelf_image_uri") or "")
            grouped.setdefault(img_key, []).append(self.map_raw_item(row, idx))
        for img_key, items in grouped.items():
            gt = ImageGroundTruth(
                image_id=img_key,
                total_main_shelf_facings=len(items),
                expected_brands=sorted({it.brand for it in items if it.brand}),
                items=items,
            )
            self._cache[img_key] = gt
            self._cache[img_key.split("/")[-1]] = gt

    def get_ground_truth(self, shelf_image_uri: str, ground_truth_id: Optional[str] = None) -> Optional[ImageGroundTruth]:
        for candidate in (shelf_image_uri, ground_truth_id, shelf_image_uri.split("/")[-1]):
            if candidate and candidate in self._cache:
                return self._cache[candidate]
        return None


def create_ground_truth_provider(
    gt_config: GroundTruthConfig,
    storage_manager: StorageManager,
    project_id: str = "unilever-shelf-understanding",
) -> BaseGroundTruthProvider:
    """Factory to instantiate the configured Ground Truth provider."""
    ptype = gt_config.provider_type.lower()
    if ptype == "none" or not gt_config.source_uri:
        return NullGroundTruthProvider()
    if ptype in ("json", "jsonl"):
        return JSONGroundTruthProvider(gt_config.source_uri, storage_manager, gt_config.schema_mapping)
    if ptype == "csv":
        return CSVGroundTruthProvider(gt_config.source_uri, storage_manager, gt_config.schema_mapping)
    if ptype in ("bigquery", "bq"):
        return BigQueryGroundTruthProvider(project_id, gt_config.source_uri, gt_config.schema_mapping)
    return NullGroundTruthProvider()
