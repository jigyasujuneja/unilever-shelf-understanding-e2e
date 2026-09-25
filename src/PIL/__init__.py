"""Lightweight stdlib-backed PIL shim (`src/PIL/__init__.py`) for environments without C-Pillow installed."""

from PIL import Image, ImageDraw, ImageFont

__all__ = ["Image", "ImageDraw", "ImageFont"]
