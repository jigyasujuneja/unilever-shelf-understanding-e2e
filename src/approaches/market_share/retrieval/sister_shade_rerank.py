"""Embedding shortlist re-ranked by pack colour (Retrieval tab). No Gemini call.

Ported from Jigyasu's ``sister_shade_disambiguator``: products of one brand often differ only
in pack colour (flavours, shades), which image embeddings blur together. So the top
``K`` products from ``embedding_retrieval`` are re-scored by

    cosine / 0.07 + max(-2, 2.5 - 0.15 * dE)

where dE is the colour distance (CIELAB, lightness weighted 0.65) between the mean colour of the
centre of the crop and of the candidate's closest reference photo. His version also added a
brand prior and an orientation term and measured a shade band at fixed coordinates of a
lipstick-style pack; those don't apply to arbitrary products and were dropped. The centre
region (middle 60% each way) is our choice.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from approaches.base import Context, register
from approaches.market_share.retrieval.embedding_retrieval import EmbeddingRetrieval

K = 5
TEMPERATURE = 0.07
ROI = (0.2, 0.2, 0.8, 0.8)  # centre of the crop / photo, as fractions (x0, y0, x1, y1)


def mean_lab(image: Image.Image) -> np.ndarray:
    """Mean CIELAB colour (D65) of the centre region."""
    w, h = image.size
    im = image.crop((int(ROI[0] * w), int(ROI[1] * h), max(int(ROI[2] * w), 1), max(int(ROI[3] * h), 1)))
    im.thumbnail((128, 128))
    rgb = np.asarray(im.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805],
                          [0.2126, 0.7152, 0.0722],
                          [0.0193, 0.1192, 0.9505]]).T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    lab = np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)
    return lab.mean(axis=0)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt((0.65 * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2))


def colour_score(cosine: float, de: float) -> float:
    return cosine / TEMPERATURE + max(-2.0, 2.5 - 0.15 * de)


@register
class SisterShadeRerank(EmbeddingRetrieval):
    name = "sister_shade_rerank"
    architecture = f"Embedding shortlist (top {K}) -> re-rank by cosine + pack colour distance"
    steps = [
        "Setup (once per run, reported apart from cost/img): embed every reference photo and "
        "measure its mean colour",
        f"Per crop: embed it and shortlist the {K} products with the most similar reference photo",
        "Re-rank the shortlist by cosine/0.07 + max(-2, 2.5 - 0.15 x colour distance)",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        super().setup(config, ctx)
        self.lab = {path: mean_lab(Image.open(io.BytesIO(data))) for path, data in self.jpeg.items()}

    def identify(self, image: Image.Image, ctx: Context,
                 allowed_ids: set[int] | None = None) -> int | None:
        top = self.ranked(image, ctx, allowed_ids)[:K]
        if not top:
            return None
        q = mean_lab(image)
        return max((colour_score(s, delta_e(q, self.lab[path])), pid) for s, pid, path in top)[1]
