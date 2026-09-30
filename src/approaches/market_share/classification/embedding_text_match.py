"""Nearest catalog product by text embedding (the Classification tab). No Gemini call.

Vertex AI multimodal embeddings (``multimodalembedding@001`` or ``gemini-embedding-2-preview``)
put images and text in one vector space. At setup every
catalog product's text ("brand product-with-size (category)") is embedded once (reported as the
run's setup cost); per photo we embed the image and return the catalog product with the highest
cosine similarity. Scored on the labelled products set, like ``gemini_classify``.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from approaches.base import Approach, Context, register
from utils import dataset
from utils.embeddings import MODELS, SKUS, VertexEmbeddings


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


@register
class EmbeddingTextMatch(Approach):
    name = "embedding_text_match"
    task = "classification"
    dataset = "products"
    models = MODELS
    skus = SKUS
    architecture = "Photo embedding -> nearest catalog product text embedding"
    steps = [
        "Setup (once per run, reported apart from cost/img): embed each catalog product's text",
        "Embed the photo with the same model",
        "Return the catalog product with the highest cosine similarity",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.emb = VertexEmbeddings(config, model=ctx.model)
        self.catalog = dataset.catalog(name=self.dataset)
        ids = list(self.catalog)
        texts = [f"{p.get('brand', '')} {p['product']} ({p.get('category', '')})".strip()
                 for p in self.catalog.values()]
        with ThreadPoolExecutor(max_workers=8) as pool:
            vecs = list(pool.map(lambda t: self.emb.text(t, ctx), texts))
        self.index = list(zip(ids, (unit(v) for v in vecs), strict=True))

    def identify(self, image: Image.Image, ctx: Context,
                 allowed_ids: set[int] | None = None) -> int | None:
        q = unit(self.emb.image(image, ctx))
        pool = [(i, v) for i, v in self.index if not allowed_ids or i in allowed_ids] or self.index
        ranked = sorted(((sum(a * b for a, b in zip(q, v, strict=True)), i) for i, v in pool),
                        reverse=True)
        top = ", ".join(f"{self.catalog[i]['product']} {s:.3f}" for s, i in ranked[:3])
        ctx.trace.step("Nearest catalog products", f"top 3 by cosine: {top}")
        return ranked[0][1]
