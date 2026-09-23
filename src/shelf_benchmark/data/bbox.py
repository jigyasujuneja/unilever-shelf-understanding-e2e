"""Bounding-box format conversion onto the suite's canonical box convention.

Isolated from the provider/IO code on purpose. This is pure geometry with no storage,
config or network dependency, and it is the highest-consequence code in the data layer: get
the conversion wrong and every detection metric in the benchmark is wrong in a way that
still looks plausible. Keeping it standalone means it can be exhaustively unit-tested
without constructing a provider.

The canonical convention used everywhere downstream is `[ymin, xmin, ymax, xmax]`
normalized to 0..1000.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from shelf_benchmark.data.ground_truth_errors import GroundTruthError

logger = logging.getLogger(__name__)

SUPPORTED_BBOX_FORMATS = (
    "ymin_xmin_ymax_xmax_1000",
    "coco_xywh_px",
    "xyxy_px",
    "xyxy_norm",
    "yxyx_norm",
    "yolo_xywh_norm",
)

#: Formats expressed in absolute pixels, which therefore require image dimensions.
PIXEL_BBOX_FORMATS = ("coco_xywh_px", "xyxy_px")

#: Slack allowed before an out-of-range coordinate is treated as a likely format error
#: rather than ordinary rounding noise. 0..1000 is the canonical range.
_CLAMP_WARN_MARGIN = 1.0


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
    elif bbox_format == "yolo_xywh_norm":
        x_c, y_c, w, h = a, b, c, d
        ymin = (y_c - h / 2.0) * 1000.0
        xmin = (x_c - w / 2.0) * 1000.0
        ymax = (y_c + h / 2.0) * 1000.0
        xmax = (x_c + w / 2.0) * 1000.0
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

    scaled = (ymin, xmin, ymax, xmax)
    _warn_if_out_of_range(scaled, raw_bbox, bbox_format)
    box = [int(round(v)) for v in scaled]
    return [max(0, min(1000, v)) for v in box]


def _warn_if_out_of_range(
    scaled: Sequence[float], raw_bbox: Sequence[float], bbox_format: str
) -> None:
    """Log when a converted box lands materially outside 0..1000 before being clamped.

    The clamp is kept (a box a fraction of a pixel over the edge is not worth failing a run
    for), but a box that converts to, say, 5000 almost always means `bbox_format` is wrong.
    Clamping that silently produces a box flush against the image edge that looks entirely
    reasonable in a report, so the raw values are surfaced here instead.
    """
    if any(v < -_CLAMP_WARN_MARGIN or v > 1000.0 + _CLAMP_WARN_MARGIN for v in scaled):
        logger.warning(
            "Bounding box %r in format '%s' converted to %s, outside the canonical 0..1000 "
            "range, and was clamped. This usually means bbox_format or the image dimensions "
            "are wrong.",
            list(raw_bbox[:4]),
            bbox_format,
            [round(v, 1) for v in scaled],
        )
