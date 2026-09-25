"""Compatible PIL.ImageDraw helper (`src/PIL/ImageDraw.py`)."""

from __future__ import annotations

from typing import Any


class _Drawer:
    def __init__(self, im: Any) -> None:
        self.im = im

    def rectangle(self, xy: Any, fill: Any = None, outline: Any = None, width: int = 1) -> None:
        pass

    def text(self, xy: Any, text: str, fill: Any = None, **kwargs: Any) -> None:
        pass


def Draw(im: Any) -> _Drawer:
    return _Drawer(im)
