"""Detect, then identify, on real store shelves (the Shelf end-to-end tab).

The ``detect_*`` pipelines run on HoloSelecta shelf photos instead of RPC checkout photos; the
gallery is product crops from other sessions' photos (see ``dataset.prepare_shelves``).
"""

from __future__ import annotations

from approaches.base import register
from approaches.detect_identify import DetectRerank, DetectRetrieve
from approaches.tiered_hybrid import DetectTiered


@register
class ShelfDetectRetrieve(DetectRetrieve):
    name = "shelf_detect_retrieve"
    dataset = "shelves"


@register
class ShelfDetectRerank(DetectRerank):
    name = "shelf_detect_rerank"
    dataset = "shelves"


@register
class ShelfDetectTiered(DetectTiered):
    name = "shelf_detect_tiered"
    dataset = "shelves"
