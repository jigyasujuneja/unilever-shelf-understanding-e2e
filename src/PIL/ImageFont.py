"""Compatible PIL.ImageFont helper (`src/PIL/ImageFont.py`)."""

from __future__ import annotations

from typing import Any


class _Font:
    def getbbox(self, text: str) -> tuple[int, int, int, int]:
        return (0, 0, max(1, len(text) * 6), 12)


def load_default(size: int = 12) -> _Font:
    return _Font()


def truetype(*args: Any, **kwargs: Any) -> _Font:
    return _Font()
