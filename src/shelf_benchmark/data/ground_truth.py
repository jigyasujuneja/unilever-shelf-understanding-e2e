"""Pluggable Ground Truth providers and schema mapper.

The ground-truth schema and storage location are owned by the annotation team, so this module
adapts any JSON/JSONL, CSV, COCO or BigQuery source onto the canonical `ImageGroundTruth` model
via `GroundTruthSchemaMapping` - no code changes required when the real dataset lands.

Two properties matter more than flexibility here:

* **Fail loudly.** A typo in a URI or a field name used to look exactly like "ground truth is not
  connected yet". Providers now raise on unreadable sources and on empty results (unless
  `GroundTruthConfig.strict` is disabled), and every provider reports load statistics.
* **Never guess the bbox convention.** Annotation vendors typically ship COCO `[x, y, w, h]` in
  pixels; this suite scores in `[ymin, xmin, ymax, xmax]` normalized to 0..1000. The format is
  declared in the mapping and converted explicitly by `convert_bbox`.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from shelf_benchmark.config import GroundTruthConfig, GroundTruthSchemaMapping
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.models import GroundTruthProductItem, ImageGroundTruth

logger = logging.getLogger(__name__)

SUPPORTED_BBOX_FORMATS = (
    "ymin_xmin_ymax_xmax_1000",
    "coco_xywh_px",
    "xyxy_px",
    "xyxy_norm",
    "yxyx_norm",
)

_PIXEL_FORMATS = ("coco_xywh_px", "xyxy_px")

_TRUTHY = {"true", "1", "yes", "y", "t"}


class GroundTruthError(RuntimeError):
    """Raised when a configured ground-truth source cannot be loaded or is unusable."""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


def convert_bbox(
    raw_bbox: Sequence[float],
    bbox_format: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> List[int]:
    """Convert an annotator bounding box into canonical `[ymin, xmin, ymax, xmax]` 0..1000.

    Raises `GroundTruthError` for unknown formats or for pixel formats supplied without image
    dimensions - silently mis-scaling boxes would corrupt every detection metric downstream.
    """
    if bbox_format not in SUPPORTED_BBOX_FORMATS:
        raise GroundTruthError(
            f"Unsupported bbox_format '{bbox_format}'. Supported: {', '.join(SUPPORTED_BBOX_FORMATS)}."
        )
    if len(raw_bbox) < 4:
        raise GroundTruthError(f"Bounding box needs 4 values, got {list(raw_bbox)!r}.")

    a, b, c, d = (float(v) for v in raw_bbox[:4])

    if bbox_format == "ymin_xmin_ymax_xmax_1000":
        ymin, xmin, ymax, xmax = a, b, c, d
    elif bbox_format == "yxyx_norm":
        ymin, xmin, ymax, xmax = a * 1000.0, b * 1000.0, c * 1000.0, d * 1000.0
    elif bbox_format == "xyxy_norm":
        ymin, xmin, ymax, xmax = b * 1000.0, a * 1000.0, d * 1000.0, c * 1000.0
    else:  # pixel formats
        if not image_width or not image_height:
            raise GroundTruthError(
                f"bbox_format '{bbox_format}' is in absolute pixels, so image_width/image_height "
                f"are required. Set them per image or via "
                f"GroundTruthSchemaMapping.default_image_width/height."
            )
        if bbox_format == "coco_xywh_px":
            x, y, w, h = a, b, c, d
            x2, y2 = x + w, y + h
        else:  # xyxy_px
            x, y, x2, y2 = a, b, c, d
        ymin = (y / image_height) * 1000.0
        xmin = (x / image_width) * 1000.0
        ymax = (y2 / image_height) * 1000.0
        xmax = (x2 / image_width) * 1000.0

    box = [int(round(v)) for v in (ymin, xmin, ymax, xmax)]
    return [max(0, min(1000, v)) for v in box]


class BaseGroundTruthProvider(ABC):
    """Abstract interface for loading ground-truth annotations."""

    def __init__(
        self,
        schema_mapping: GroundTruthSchemaMapping,
        gt_version: str = "unversioned",
        exclude_back_row_items: bool = True,
    ):
        self.mapping = schema_mapping
        self.gt_version = gt_version
        self.exclude_back_row_items = exclude_back_row_items
        self._cache: Dict[str, ImageGroundTruth] = {}
        self.load_stats: Dict[str, Any] = {"images": 0, "items": 0, "excluded_back_row": 0}
        self._lookup_misses: List[str] = []

    def parsed_entries(self) -> List[Tuple[str, ImageGroundTruth]]:
        """Return one (key, ground truth) pair per distinct annotated image.

        The lookup cache deliberately holds several aliases for the same image (bare id,
        basename, full `gs://` URI) so a run can find its ground truth however the image
        happens to be referenced. Those aliases are an implementation detail, so they are
        collapsed here: the number of entries returned equals `load_stats["images"]`.

        Exposed so CLI and notebook tooling can eyeball what was actually parsed without
        reaching into a private attribute. A caller that guesses the private name gets a
        silently empty list instead of an error, which is how `validate-gt` ended up
        printing no sample entries at all.
        """
        seen: set[int] = set()
        unique: List[Tuple[str, ImageGroundTruth]] = []
        for key, gt in self._cache.items():
            if id(gt) in seen:
                continue
            seen.add(id(gt))
            unique.append((key, gt))
        return unique

    # ----------------------------------------------------------------- mapping

    def map_raw_item(
        self,
        raw_item: Dict[str, Any],
        idx: int = 1,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> GroundTruthProductItem:
        """Map one annotator record onto the canonical item model."""
        m = self.mapping
        raw_bbox: Any = raw_item.get(m.bbox_field)
        if raw_bbox is None:
            raw_bbox = raw_item.get("bbox")
        if isinstance(raw_bbox, str):
            raw_bbox = json.loads(raw_bbox)
        if raw_bbox is None:
            raise GroundTruthError(
                f"Ground-truth item #{idx} has no bounding box under field '{m.bbox_field}'. "
                f"Available fields: {sorted(raw_item)}"
            )

        bbox = convert_bbox(
            raw_bbox,
            bbox_format=m.bbox_format,
            image_width=image_width or m.default_image_width,
            image_height=image_height or m.default_image_height,
        )

        raw_id = raw_item.get(m.item_id_field)
        try:
            item_id = int(raw_id) if raw_id is not None and str(raw_id).strip() != "" else idx
        except (TypeError, ValueError):
            item_id = idx

        return GroundTruthProductItem(
            item_id=item_id,
            brand=str(raw_item.get(m.brand_field) or ""),
            product_name=str(raw_item.get(m.product_name_field) or ""),
            category=raw_item.get(m.category_field),
            subcategory=raw_item.get(m.subcategory_field),
            variant=raw_item.get(m.variant_field),
            packaging_type=raw_item.get(m.packaging_type_field),
            pack_type=raw_item.get(m.pack_type_field),
            size=raw_item.get(m.size_field),
            sku_id=raw_item.get(m.sku_id_field),
            bbox_2d=bbox,
            shelf_row=str(raw_item.get(m.shelf_row_field) or "middle"),
            back_row=_as_bool(raw_item.get(m.back_row_field)),
            occluded=_as_bool(raw_item.get(m.occluded_field)),
        )

    def _filter_items(self, items: List[GroundTruthProductItem]) -> List[GroundTruthProductItem]:
        if not self.exclude_back_row_items:
            return items
        kept = [it for it in items if not it.back_row]
        self.load_stats["excluded_back_row"] += len(items) - len(kept)
        return kept

    def _index(self, key: str, gt: ImageGroundTruth) -> None:
        """Index a ground-truth entry under every key callers might legitimately look it up by."""
        for candidate in {key, gt.image_id, key.split("/")[-1], gt.image_id.split("/")[-1]}:
            if candidate:
                self._cache[candidate] = gt
        self.load_stats["images"] = len({id(v) for v in self._cache.values()})
        self.load_stats["items"] += len(gt.items)

    def _build(
        self,
        key: str,
        items: List[GroundTruthProductItem],
        image_id: Optional[str] = None,
        expected_brands: Optional[List[str]] = None,
        total_facings: Optional[int] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> ImageGroundTruth:
        kept = self._filter_items(items)
        return ImageGroundTruth(
            image_id=str(image_id or key),
            total_main_shelf_facings=int(total_facings or len(kept)),
            expected_brands=expected_brands or sorted({it.brand for it in kept if it.brand}),
            items=kept,
            gt_version=self.gt_version,
            source_key=key,
            image_width=image_width,
            image_height=image_height,
        )

    # ----------------------------------------------------------------- lookup

    def get_ground_truth(
        self, shelf_image_uri: str, ground_truth_id: Optional[str] = None
    ) -> Optional[ImageGroundTruth]:
        """Look up ground truth by image URI, explicit ground-truth ID, or file basename."""
        for candidate in (shelf_image_uri, ground_truth_id, (shelf_image_uri or "").split("/")[-1]):
            if candidate and candidate in self._cache:
                return self._cache[candidate]
        if shelf_image_uri:
            self._lookup_misses.append(shelf_image_uri)
        return None

    @property
    def is_connected(self) -> bool:
        """True when this provider holds usable annotations."""
        return bool(self._cache)

    @property
    def unmatched_lookups(self) -> List[str]:
        """Image URIs that were requested but had no ground-truth entry (join debugging)."""
        return list(dict.fromkeys(self._lookup_misses))

    def describe(self) -> str:
        """Human-readable load summary, printed by the runner/SDK so silence is never mistaken for success."""
        return (
            f"{type(self).__name__}: {self.load_stats['images']} image(s), "
            f"{self.load_stats['items']} item(s), "
            f"{self.load_stats['excluded_back_row']} back-row item(s) excluded, "
            f"gt_version={self.gt_version}"
        )

    @abstractmethod
    def _load(self) -> None:
        """Populate `self._cache` from the configured source."""


class NullGroundTruthProvider(BaseGroundTruthProvider):
    """Used when no ground truth is configured yet. Always returns `None`, and says so."""

    def __init__(self) -> None:
        super().__init__(GroundTruthSchemaMapping())

    def _load(self) -> None:  # pragma: no cover - nothing to load
        return None

    def get_ground_truth(
        self, shelf_image_uri: str, ground_truth_id: Optional[str] = None
    ) -> Optional[ImageGroundTruth]:
        return None

    def describe(self) -> str:
        return "NullGroundTruthProvider: no ground truth configured (accuracy will be reported as PLACEHOLDER)."


class _FileBackedProvider(BaseGroundTruthProvider):
    """Shared plumbing for providers that read a single local or `gs://` artifact."""

    def __init__(
        self,
        source_uri: str,
        storage_manager: StorageManager,
        schema_mapping: GroundTruthSchemaMapping,
        gt_version: str = "unversioned",
        exclude_back_row_items: bool = True,
        strict: bool = True,
    ):
        super().__init__(schema_mapping, gt_version, exclude_back_row_items)
        self.source_uri = source_uri
        self.storage = storage_manager
        self.strict = strict
        self._load()
        _assert_loaded(self, strict)

    def _read_source_text(self) -> str:
        """Read the configured source, converting any I/O failure into GroundTruthError.

        Callers of the ground-truth layer are documented to only have to handle
        GroundTruthError. Letting a raw FileNotFoundError or PermissionError escape from
        here forces every caller to know which storage backend is in play, so we translate
        once, at the boundary, and keep the original exception chained for debugging.
        """
        try:
            return self.storage.read_text(self.source_uri)
        except FileNotFoundError as exc:
            raise GroundTruthError(
                f"Ground-truth source '{self.source_uri}' does not exist. Check "
                f"ground_truth.source_uri in your config (or the --ground-truth-uri flag); "
                f"relative paths are resolved from the current working directory."
            ) from exc
        except PermissionError as exc:
            raise GroundTruthError(
                f"Ground-truth source '{self.source_uri}' exists but is not readable by this "
                f"process. Check file permissions, or your GCS credentials if this is a "
                f"gs:// URI."
            ) from exc
        except UnicodeDecodeError as exc:
            raise GroundTruthError(
                f"Ground-truth source '{self.source_uri}' is not valid UTF-8 text. If this is "
                f"a spreadsheet or an archive, export it to JSON, JSONL, or CSV first."
            ) from exc
        except GroundTruthError:
            raise
        except Exception as exc:
            raise GroundTruthError(
                f"Failed to read ground-truth source '{self.source_uri}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc


class JSONGroundTruthProvider(_FileBackedProvider):
    """Loads ground truth from a JSON or JSONL document (local path or `gs://` URI)."""

    def _load(self) -> None:
        text = self._read_source_text()
        raw_data = _parse_json_or_jsonl(text, self.source_uri)
        m = self.mapping

        if isinstance(raw_data, dict):
            self.gt_version = str(raw_data.get("gt_version") or self.gt_version)
            declared_format = raw_data.get("bbox_format")
            if declared_format and declared_format != m.bbox_format:
                raise GroundTruthError(
                    f"Ground-truth file declares bbox_format='{declared_format}' but the config "
                    f"says '{m.bbox_format}'. Fix the config rather than letting boxes be "
                    f"misinterpreted."
                )
            default_w = raw_data.get("image_width") or m.default_image_width
            default_h = raw_data.get("image_height") or m.default_image_height
            images = raw_data.get("images", raw_data)
            if not isinstance(images, dict):
                raise GroundTruthError(
                    f"Expected an object of images in '{self.source_uri}', got {type(images).__name__}."
                )
            for key, value in images.items():
                if not isinstance(value, dict):
                    continue
                self._ingest_entry(str(key), value, default_w, default_h)
        elif isinstance(raw_data, list):
            for entry in raw_data:
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get(m.image_key_field) or entry.get("shelf_image_uri") or "")
                if not key:
                    raise GroundTruthError(
                        f"Ground-truth entry is missing the image key field '{m.image_key_field}'. "
                        f"Available fields: {sorted(entry)}"
                    )
                self._ingest_entry(key, entry, m.default_image_width, m.default_image_height)
        else:
            raise GroundTruthError(
                f"Unsupported ground-truth JSON root type: {type(raw_data).__name__}."
            )

    def _ingest_entry(
        self,
        key: str,
        entry: Dict[str, Any],
        default_w: Optional[int],
        default_h: Optional[int],
    ) -> None:
        m = self.mapping
        width = entry.get(m.image_width_field) or default_w
        height = entry.get(m.image_height_field) or default_h
        raw_items = entry.get(m.items_list_field)
        if raw_items is None:
            raise GroundTruthError(
                f"Ground-truth entry '{key}' has no item list under '{m.items_list_field}'. "
                f"Available fields: {sorted(entry)}"
            )
        items = [
            self.map_raw_item(item, idx + 1, width, height)
            for idx, item in enumerate(raw_items)
        ]
        gt = self._build(
            key=key,
            items=items,
            image_id=entry.get(m.image_key_field) or key,
            expected_brands=entry.get("expected_brands"),
            total_facings=entry.get("total_main_shelf_facings"),
            image_width=width,
            image_height=height,
        )
        self._index(key, gt)


class CSVGroundTruthProvider(_FileBackedProvider):
    """Loads flat row-per-product ground truth from a CSV file (local path or `gs://` URI)."""

    def _load(self) -> None:
        text = self._read_source_text()
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise GroundTruthError(f"Ground-truth CSV '{self.source_uri}' has no header row.")
        _assert_columns(reader.fieldnames, self.mapping, self.source_uri)
        self._ingest_rows(list(reader))

    def _ingest_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        m = self.mapping
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get(m.image_key_field) or row.get("shelf_image_uri") or "")
            if not key:
                raise GroundTruthError(
                    f"Ground-truth row is missing image key '{m.image_key_field}': {row!r}"
                )
            grouped.setdefault(key, []).append(dict(row))

        for key, raw_rows in grouped.items():
            width = _first_int(raw_rows, m.image_width_field) or m.default_image_width
            height = _first_int(raw_rows, m.image_height_field) or m.default_image_height
            items = [
                self.map_raw_item(row, idx + 1, width, height)
                for idx, row in enumerate(raw_rows)
            ]
            self._index(key, self._build(key, items, image_width=width, image_height=height))


class COCOGroundTruthProvider(_FileBackedProvider):
    """Loads a COCO-format annotation file (`images` + `annotations` + `categories`).

    COCO boxes are `[x, y, width, height]` in absolute pixels, and image dimensions come from the
    `images` array, so no extra configuration is needed for the geometry.
    """

    def _load(self) -> None:
        payload = _parse_json_or_jsonl(self._read_source_text(), self.source_uri)
        if not isinstance(payload, dict):
            raise GroundTruthError(f"COCO file '{self.source_uri}' must contain a JSON object.")

        images = {img["id"]: img for img in payload.get("images", [])}
        if not images:
            raise GroundTruthError(f"COCO file '{self.source_uri}' contains no 'images' entries.")
        categories = {c["id"]: c.get("name", "") for c in payload.get("categories", [])}

        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for ann in payload.get("annotations", []):
            grouped.setdefault(ann.get("image_id"), []).append(ann)

        for image_id, image in images.items():
            width, height = image.get("width"), image.get("height")
            key = str(image.get("coco_url") or image.get("file_name") or image_id)
            items: List[GroundTruthProductItem] = []
            for idx, ann in enumerate(grouped.get(image_id, []), start=1):
                category_name = categories.get(ann.get("category_id"), "")
                extras = ann.get("attributes", {}) or {}
                items.append(
                    GroundTruthProductItem(
                        item_id=int(ann.get("id", idx)),
                        brand=str(extras.get("brand") or category_name),
                        product_name=str(extras.get("product_name") or category_name),
                        sku_id=extras.get("sku_id"),
                        bbox_2d=convert_bbox(ann["bbox"], "coco_xywh_px", width, height),
                        shelf_row=str(extras.get("shelf_row") or "middle"),
                        back_row=_as_bool(extras.get("back_row")),
                        occluded=_as_bool(extras.get("occluded") or ann.get("iscrowd")),
                    )
                )
            self._index(key, self._build(key, items, image_width=width, image_height=height))


class BigQueryGroundTruthProvider(BaseGroundTruthProvider):
    """Loads ground-truth rows from a BigQuery table (`project.dataset.table`) or a SQL query."""

    def __init__(
        self,
        project_id: str,
        table_or_query: str,
        schema_mapping: GroundTruthSchemaMapping,
        gt_version: str = "unversioned",
        exclude_back_row_items: bool = True,
        strict: bool = True,
    ):
        super().__init__(schema_mapping, gt_version, exclude_back_row_items)
        self.project_id = project_id
        self.table_or_query = table_or_query
        self.strict = strict
        self._load()
        _assert_loaded(self, strict)

    def _load(self) -> None:
        from shelf_benchmark.auth import create_bigquery_client

        bq = create_bigquery_client(self.project_id)
        sql = (
            self.table_or_query
            if self.table_or_query.strip().upper().startswith("SELECT")
            else f"SELECT * FROM `{self.table_or_query}`"
        )
        m = self.mapping
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in bq.query(sql).result():
            record = dict(row)
            key = str(record.get(m.image_key_field) or record.get("shelf_image_uri") or "")
            if not key:
                raise GroundTruthError(
                    f"BigQuery ground-truth row missing image key column '{m.image_key_field}'. "
                    f"Columns present: {sorted(record)}"
                )
            grouped.setdefault(key, []).append(record)

        for key, records in grouped.items():
            width = _first_int(records, m.image_width_field) or m.default_image_width
            height = _first_int(records, m.image_height_field) or m.default_image_height
            items = [
                self.map_raw_item(rec, idx + 1, width, height) for idx, rec in enumerate(records)
            ]
            self._index(key, self._build(key, items, image_width=width, image_height=height))


# ---------------------------------------------------------------------- helpers


def _parse_json_or_jsonl(text: str, source_uri: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise GroundTruthError(f"Ground-truth source '{source_uri}' is empty.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        records = []
        for line_no, line in enumerate(stripped.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise GroundTruthError(
                    f"'{source_uri}' is neither valid JSON nor JSONL (line {line_no}): {exc}"
                ) from exc
        return records


def _assert_columns(
    fieldnames: Sequence[str], mapping: GroundTruthSchemaMapping, source_uri: str
) -> None:
    required = {
        "image_key_field": mapping.image_key_field,
        "brand_field": mapping.brand_field,
        "bbox_field": mapping.bbox_field,
    }
    missing = {name: column for name, column in required.items() if column not in fieldnames}
    if missing:
        raise GroundTruthError(
            f"Ground-truth source '{source_uri}' is missing mapped column(s) "
            f"{missing}. Columns present: {list(fieldnames)}. "
            f"Update GroundTruthSchemaMapping so the join cannot silently return nothing."
        )


def _first_int(records: Sequence[Dict[str, Any]], field: str) -> Optional[int]:
    for record in records:
        value = record.get(field)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _assert_loaded(provider: BaseGroundTruthProvider, strict: bool) -> None:
    if provider.is_connected:
        logger.info(provider.describe())
        return
    message = (
        f"{type(provider).__name__} loaded 0 ground-truth images from "
        f"'{getattr(provider, 'source_uri', getattr(provider, 'table_or_query', 'unknown'))}'. "
        f"This usually means a wrong URI or a wrong `image_key_field` in the schema mapping."
    )
    if strict:
        raise GroundTruthError(message)
    logger.warning("%s (strict=False, continuing without ground truth)", message)


def create_ground_truth_provider(
    gt_config: GroundTruthConfig,
    storage_manager: StorageManager,
    project_id: str = "unilever-shelf-understanding",
) -> BaseGroundTruthProvider:
    """Instantiate the configured ground-truth provider.

    Raises `GroundTruthError` for an unknown provider type, or for a configured provider whose
    source cannot be read, instead of degrading to `NullGroundTruthProvider` - "GT not configured"
    and "GT misconfigured" must never look identical in a report.
    """
    provider_type = (gt_config.provider_type or "none").lower()
    if provider_type == "none":
        return NullGroundTruthProvider()

    if not gt_config.source_uri:
        if gt_config.strict:
            raise GroundTruthError(
                f"ground_truth.provider_type='{provider_type}' requires a `source_uri`."
            )
        return NullGroundTruthProvider()

    common = {
        "schema_mapping": gt_config.schema_mapping,
        "gt_version": gt_config.gt_version,
        "exclude_back_row_items": gt_config.exclude_back_row_items,
        "strict": gt_config.strict,
    }

    if provider_type in ("json", "jsonl"):
        return JSONGroundTruthProvider(gt_config.source_uri, storage_manager, **common)
    if provider_type == "csv":
        return CSVGroundTruthProvider(gt_config.source_uri, storage_manager, **common)
    if provider_type == "coco":
        return COCOGroundTruthProvider(gt_config.source_uri, storage_manager, **common)
    if provider_type in ("bigquery", "bq"):
        return BigQueryGroundTruthProvider(project_id, gt_config.source_uri, **common)

    raise GroundTruthError(
        f"Unknown ground_truth.provider_type '{gt_config.provider_type}'. "
        f"Expected one of: none, json, jsonl, csv, coco, bigquery."
    )
