"""Nearest reference photo by image embedding (the Retrieval tab). No Gemini call.

The RPC reference gallery has ``REFS_PER_PRODUCT`` photos of every product on its own, from
different angles. At setup each is embedded with the run's embedding model (``multimodalembedding@001``
or ``gemini-embedding-2-preview``; reported as the run's setup cost, like building a vector index
once). Per product crop we embed the crop and return the product whose closest reference photo is
most similar (cosine). That's what a vector database (``utils/alloydb.py``) would do; with 800
vectors a Python loop is enough.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from approaches.base import Approach, Context, register
from approaches.market_share.classification.embedding_text_match import unit
from utils import dataset
from utils.embeddings import MODELS, MULTIMODAL, SKUS, VertexEmbeddings

MODEL = MULTIMODAL  # the fixed shortlist model of the Gemini approaches built on this one


@register
class EmbeddingRetrieval(Approach):
    name = "embedding_retrieval"
    task = "retrieval"
    dataset = "rpc"
    models = MODELS
    skus = SKUS
    architecture = "Crop embedding -> nearest reference photo"
    steps = [
        "Setup (once per run, reported apart from cost/img): embed every reference photo",
        "Embed each product crop with the same model",
        "Return the product whose reference photo is most similar (cosine)",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        # Run as an embedding approach: the run's model. Under a Gemini model (the rerank and
        # detect approaches): the fixed shortlist model they list in ``also_calls``.
        self.emb = VertexEmbeddings(config, model=ctx.model if ctx.model in MODELS else MODEL)
        gallery = dataset.rpc_gallery(dataset.data_root(self.dataset))
        refs = [(pid, path) for pid, paths in gallery.items() for path in paths]
        self.jpeg: dict[str, bytes] = {}  # reference photos, for the rerank contact sheet

        def embed(ref: tuple[int, str]) -> tuple[int, str, list[float]]:
            pid, path = ref
            self.jpeg[path] = dataset.read_bytes(path)
            with Image.open(io.BytesIO(self.jpeg[path])) as im:
                return pid, path, unit(self.emb.image(im.convert("RGB"), ctx))

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.index = list(pool.map(embed, refs))

    def ranked(self, image: Image.Image, ctx: Context,
               allowed_ids: set[int] | None = None) -> list[tuple[float, int, str]]:
        """Products by similarity of their closest reference photo: (cosine, product, photo)."""
        q = unit(self.emb.image(image, ctx))
        best: dict[int, tuple[float, int, str]] = {}
        for pid, path, v in self.index:
            if allowed_ids and pid not in allowed_ids:
                continue
            s = sum(a * b for a, b in zip(q, v, strict=True))
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, pid, path)
        return sorted(best.values(), reverse=True) if best else (
            self.ranked(image, ctx, None) if allowed_ids else []
        )

    def identify(self, image: Image.Image, ctx: Context,
                 allowed_ids: set[int] | None = None) -> int | None:
        r = self.ranked(image, ctx, allowed_ids)
        return r[0][1] if r else None
