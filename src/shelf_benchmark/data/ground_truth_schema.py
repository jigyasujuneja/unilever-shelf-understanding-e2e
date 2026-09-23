"""Adapter from an arbitrary annotator row onto the canonical `GroundTruthProductItem`.

Separated from the provider framework because this is the part that changes when a new
annotation vendor lands, and it is pure data transformation: give it a `dict` and a
`GroundTruthSchemaMapping` and it returns a canonical item. It needs no storage manager, no
network and no provider instance, so it can be tested directly against a sample row.

`BaseGroundTruthProvider.map_raw_item` delegates here, so existing provider subclasses and
callers are unaffected.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from shelf_benchmark.config import GroundTruthSchemaMapping
from shelf_benchmark.data.bbox import convert_bbox
from shelf_benchmark.data.field_access import extract_field
from shelf_benchmark.data.ground_truth_errors import GroundTruthError
from shelf_benchmark.models import GroundTruthProductItem

logger = logging.getLogger(__name__)

_TRUTHY = {"true", "1", "yes", "y", "t"}

#: Largest coordinate value still treated as "probably normalized 0..1" when a row supplies
#: bare corner columns without declaring a format.
_NORMALIZED_MAX = 1.05


def as_bool(value: Any) -> bool:
    """Coerce an annotator truthiness value ("1", "yes", "true", True) to `bool`."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


class GroundTruthItemMapper:
    """Maps one annotator record onto `GroundTruthProductItem` using a schema mapping."""

    def __init__(self, mapping: GroundTruthSchemaMapping):
        self.mapping = mapping

    def map_raw_item(
        self,
        raw_item: Dict[str, Any],
        idx: int = 1,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> GroundTruthProductItem:
        """Map one annotator record onto the canonical item model across any JSON/CSV/BigQuery/AutoML schema."""
        m = self.mapping
        raw_bbox, effective_format = self._resolve_bbox(raw_item, idx, image_width)

        bbox = convert_bbox(
            raw_bbox,
            bbox_format=effective_format,
            image_width=image_width or m.default_image_width,
            image_height=image_height or m.default_image_height,
        )

        return GroundTruthProductItem(
            item_id=self._resolve_item_id(raw_item, idx),
            brand=str(extract_field(raw_item, m.brand_field, "brand", "brand_name", "manufacturer", "attributes.brand") or ""),
            product_name=str(extract_field(raw_item, m.product_name_field, "product_name", "sku_name", "title", "attributes.product_name") or ""),
            category=extract_field(raw_item, m.category_field, "category", "attributes.category"),
            subcategory=extract_field(raw_item, m.subcategory_field, "subcategory", "attributes.subcategory"),
            variant=extract_field(raw_item, m.variant_field, "variant", "attributes.variant"),
            packaging_type=extract_field(raw_item, m.packaging_type_field, "packaging_type", "attributes.packaging_type"),
            pack_type=extract_field(raw_item, m.pack_type_field, "pack_type", "attributes.pack_type"),
            size=extract_field(raw_item, m.size_field, "size", "pack_size", "attributes.size"),
            sku_id=extract_field(raw_item, m.sku_id_field, "sku_id", "sku_code", "ean", "gtin", "attributes.sku_id"),
            bbox_2d=bbox,
            shelf_row=str(extract_field(raw_item, m.shelf_row_field, "shelf_row", "row") or "middle"),
            back_row=as_bool(extract_field(raw_item, m.back_row_field, "back_row", "is_back_row")),
            occluded=as_bool(extract_field(raw_item, m.occluded_field, "occluded", "is_occluded")),
            extra_attributes=self._collect_extra_attributes(raw_item),
        )

    # ------------------------------------------------------------------ bbox

    def _resolve_bbox(
        self, raw_item: Dict[str, Any], idx: int, image_width: Optional[int]
    ) -> tuple[Any, str]:
        """Locate the bounding box in the row and decide which format it is in.

        The module contract is "never guess the bbox convention", and for the mapped
        `bbox_field` that holds: the declared `bbox_format` is used as-is. The guessing
        below only applies to rows that ship bare corner/extent columns
        (`xmin/ymin/xmax/ymax`, `x/y/width/height`) or polygon vertices, where there is no
        declared format to honour. Every inference is logged so it is visible in the run
        output rather than being an invisible assumption behind a metric.
        """
        m = self.mapping
        raw_bbox: Any = extract_field(raw_item, m.bbox_field, "bbox", "bbox_2d", "box_2d", "bounding_box")
        effective_format = m.bbox_format

        if raw_bbox is None:
            # Check for polygon vertices (Cloud Vision / Vertex AI AutoML Vision)
            raw_bbox = extract_field(
                raw_item, "normalizedVertices", "vertices", "boundingPoly.normalizedVertices", "boundingPoly.vertices"
            )
            if raw_bbox is None:
                raw_bbox, effective_format = self._bbox_from_corner_columns(raw_item, effective_format)

        if isinstance(raw_bbox, str):
            raw_bbox = json.loads(raw_bbox)
        if isinstance(raw_bbox, list) and len(raw_bbox) >= 3 and isinstance(raw_bbox[0], dict):
            raw_bbox, effective_format = self._bbox_from_vertices(raw_bbox, image_width)
        if raw_bbox is None:
            raise GroundTruthError(
                f"Ground-truth item #{idx} has no bounding box under field '{m.bbox_field}'. "
                f"Available fields: {sorted(raw_item)}"
            )
        return raw_bbox, effective_format

    def _bbox_from_corner_columns(
        self, raw_item: Dict[str, Any], effective_format: str
    ) -> tuple[Any, str]:
        """Assemble a box from separate 4-column layouts used by BigQuery / CSV tables."""
        ex = extract_field
        if all(ex(raw_item, k) is not None for k in ("ymin", "xmin", "ymax", "xmax")):
            return [ex(raw_item, "ymin"), ex(raw_item, "xmin"), ex(raw_item, "ymax"), ex(raw_item, "xmax")], effective_format
        if all(ex(raw_item, k) is not None for k in ("y_min", "x_min", "y_max", "x_max")):
            return [ex(raw_item, "y_min"), ex(raw_item, "x_min"), ex(raw_item, "y_max"), ex(raw_item, "x_max")], effective_format
        if all(ex(raw_item, k) is not None for k in ("xmin", "ymin", "xmax", "ymax")):
            raw_bbox = [ex(raw_item, "xmin"), ex(raw_item, "ymin"), ex(raw_item, "xmax"), ex(raw_item, "ymax")]
            if effective_format == "ymin_xmin_ymax_xmax_1000":
                inferred = "xyxy_norm" if max(float(v) for v in raw_bbox) <= _NORMALIZED_MAX else "xyxy_px"
                _log_inferred_format(inferred, "xmin/ymin/xmax/ymax columns", raw_bbox)
                return raw_bbox, inferred
            return raw_bbox, effective_format
        if all(ex(raw_item, k) is not None for k in ("x", "y", "width", "height")):
            raw_bbox = [ex(raw_item, "x"), ex(raw_item, "y"), ex(raw_item, "width"), ex(raw_item, "height")]
            if effective_format == "ymin_xmin_ymax_xmax_1000":
                _log_inferred_format("coco_xywh_px", "x/y/width/height columns", raw_bbox)
                return raw_bbox, "coco_xywh_px"
            return raw_bbox, effective_format
        return None, effective_format

    def _bbox_from_vertices(self, vertices: list, image_width: Optional[int]) -> tuple[Any, str]:
        """Reduce a polygon vertex list to its axis-aligned bounding box."""
        xs = [float(v.get("x", 0.0)) for v in vertices]
        ys = [float(v.get("y", 0.0)) for v in vertices]
        inferred = (
            "yxyx_norm"
            if max(xs + ys) <= _NORMALIZED_MAX
            else ("xyxy_px" if (image_width or self.mapping.default_image_width) else "ymin_xmin_ymax_xmax_1000")
        )
        _log_inferred_format(inferred, "polygon vertices", vertices)
        if inferred == "xyxy_px":
            return [min(xs), min(ys), max(xs), max(ys)], inferred
        return [min(ys), min(xs), max(ys), max(xs)], inferred

    # ------------------------------------------------------------------ scalars

    def _resolve_item_id(self, raw_item: Dict[str, Any], idx: int) -> int:
        """Return the annotator's integer item id, or the positional index if it is not one.

        The canonical model types `item_id` as `int`, so a non-numeric annotator id such as
        `"SKU-A17"` cannot be preserved. Falling back to the position keeps ids unique
        within an image but breaks traceability back to the annotation, so the substitution
        is logged rather than performed silently.
        """
        raw_id = extract_field(raw_item, self.mapping.item_id_field, "item_id", "id")
        if raw_id is None or str(raw_id).strip() == "":
            return idx
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            logger.warning(
                "Ground-truth item id %r is not an integer; using positional index %d instead. "
                "The reported item_id will not match the annotation source.",
                raw_id,
                idx,
            )
            return idx

    def _collect_extra_attributes(self, raw_item: Dict[str, Any]) -> Dict[str, Any]:
        """Gather annotator columns beyond the canonical fields into `extra_attributes`."""
        m = self.mapping
        reserved_keys = {
            m.item_id_field, "item_id", "id",
            m.brand_field, "brand", "brand_name", "manufacturer",
            m.product_name_field, "product_name", "sku_name", "title",
            m.category_field, "category",
            m.subcategory_field, "subcategory",
            m.variant_field, "variant",
            m.packaging_type_field, "packaging_type",
            m.pack_type_field, "pack_type",
            m.size_field, "size", "pack_size",
            m.sku_id_field, "sku_id", "sku_code", "ean", "gtin",
            m.bbox_field, "bbox", "bbox_2d", "box_2d", "bounding_box",
            "ymin", "xmin", "ymax", "xmax", "y_min", "x_min", "y_max", "x_max", "x", "y", "width", "height",
            "boundingPoly", "normalizedVertices", "vertices",
            m.shelf_row_field, "shelf_row", "row",
            m.back_row_field, "back_row", "is_back_row",
            m.occluded_field, "occluded", "is_occluded",
            m.image_key_field, "image_id", "shelf_image_uri", "image_uri", "gcs_uri", "file_name",
            m.image_width_field, m.image_height_field, "image_width", "image_height",
            "extra_attributes", "attributes",
        }
        extra_attrs: Dict[str, Any] = {}
        if isinstance(raw_item.get("attributes"), dict):
            for k, v in raw_item["attributes"].items():
                if k not in reserved_keys and v is not None:
                    extra_attrs[str(k)] = v
        if isinstance(raw_item.get("extra_attributes"), dict):
            for k, v in raw_item["extra_attributes"].items():
                if v is not None:
                    extra_attrs[str(k)] = v
        for k, v in raw_item.items():
            if k not in reserved_keys and v is not None and not isinstance(v, (dict, list)):
                extra_attrs[str(k)] = v
        return extra_attrs


def _log_inferred_format(inferred: str, source: str, raw_bbox: Any) -> None:
    logger.info(
        "Inferred bbox_format='%s' from %s (no declared format applies to this layout): %r",
        inferred,
        source,
        raw_bbox,
    )
