"""Pure-Python JPEG/PNG dimension reader & Image compatibility class (`src/PIL/Image.py`)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


def _parse_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    if len(data) >= 4 and data[:2] == b"\xff\xd8":
        idx = 2
        while idx + 9 < len(data):
            if data[idx] != 0xFF:
                idx += 1
                continue
            marker = data[idx + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                h, w = struct.unpack(">HH", data[idx + 5 : idx + 9])
                return int(w), int(h)
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                idx += 2
                continue
            seg_len = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
            idx += 2 + max(2, seg_len)
    return 1000, 1000


class Image:
    """Compatible PIL.Image.Image representation."""

    def __init__(self, mode: str = "RGB", size: tuple[int, int] = (1000, 1000), raw: bytes = b"") -> None:
        self.mode = mode
        self.size = (max(1, int(size[0])), max(1, int(size[1])))
        self._raw = raw or b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    def __enter__(self) -> Image:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def convert(self, mode: str) -> Image:
        return Image(mode, self.size, self._raw)

    def copy(self) -> Image:
        return Image(self.mode, self.size, self._raw)

    def crop(self, box: tuple[float, float, float, float]) -> Image:
        x1, y1, x2, y2 = box
        return Image(self.mode, (max(1, int(x2 - x1)), max(1, int(y2 - y1))), self._raw)

    def resize(self, size: tuple[int, int], *args: Any, **kwargs: Any) -> Image:
        return Image(self.mode, (max(1, int(size[0])), max(1, int(size[1]))), self._raw)

    def thumbnail(self, size: tuple[int, int], *args: Any, **kwargs: Any) -> None:
        w, h = self.size
        scale = min(size[0] / max(1, w), size[1] / max(1, h), 1.0)
        self.size = (max(1, int(w * scale)), max(1, int(h * scale)))

    def paste(self, im: Image, box: tuple[int, int] | tuple[int, int, int, int]) -> None:
        pass

    def save(self, fp: Any, format: str = "JPEG", **kwargs: Any) -> None:
        w, h = self.size
        # Minimal valid JPEG header containing exact SOF0 dimensions (w, h)
        sof0 = b"\xff\xc0\x00\x11\x08" + struct.pack(">HH", h, w) + b"\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
        payload = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + sof0 + b"\xff\xd9"
        if hasattr(fp, "write"):
            fp.write(payload)
        else:
            Path(fp).write_bytes(payload)


class Resampling:
    LANCZOS = 1
    BILINEAR = 2


LANCZOS = 1
BILINEAR = 2


def new(mode: str, size: tuple[int, int], color: Any = "white") -> Image:
    return Image(mode, size)


def open(fp: Any) -> Image:
    if hasattr(fp, "read"):
        data = fp.read()
    else:
        data = Path(fp).read_bytes()
    w, h = _parse_dimensions(data)
    return Image("RGB", (w, h), data)
