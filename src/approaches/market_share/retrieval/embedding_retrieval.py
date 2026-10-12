"""Nearest reference photo by image embedding (the Retrieval tab). No Gemini call.

The RPC reference gallery has ``REFS_PER_PRODUCT`` photos of every product on its own, from
different angles. At setup each is embedded with the run's embedding model (``multimodalembedding@001``
or ``gemini-embedding-2``; reported as the run's setup cost, like building a vector index
once). Per product crop we embed the crop and return the product whose closest reference photo is
most similar (cosine). That's what a vector database (``utils/alloydb.py``) would do; with 800
vectors a Python loop is enough.

When this step runs under a Gemini model (the ``detect_*`` pipelines), the embedder is fixed:
``shortlist_model``. It is ``gemini-embedding-2`` because on RPC it identified 89.9% of
crops vs 74.2% for ``multimodalembedding@001`` at about the same price per crop.
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from approaches.base import Approach, Context, register
from approaches.market_share.classification.embedding_text_match import unit
from utils import dataset
from utils.embeddings import GEMINI_EMBEDDING, MODELS, SKUS, VertexEmbeddings
from utils.photometry import restore_photometry

MODEL = GEMINI_EMBEDDING  # the fixed embedder of the Gemini pipelines built on this one
_GALLERY_CACHE: dict[tuple[str, bool, str], tuple[bytes, list[float]]] = {}


class _LazyJpegDict(dict):
    """On-demand JPEG byte loader for contact-sheet rerankers so 10,000+ gallery images are not held in RAM."""

    def __getitem__(self, key: str) -> bytes:
        if key in self:
            return super().__getitem__(key)
        raw_b = dataset.read_bytes(key)
        if len(self) < 256:
            self[key] = raw_b
        return raw_b

    def get(self, key: str, default: bytes | None = None) -> bytes | None:  # type: ignore[override]
        try:
            return self[key]
        except Exception:
            return default


@register
class EmbeddingRetrieval(Approach):
    name = "embedding_retrieval"
    task = "retrieval"
    dataset = "rpc"
    models = MODELS
    skus = SKUS
    shortlist_model = MODEL  # embedder used when the run's -m is a Gemini model
    architecture = "Crop embedding -> nearest reference photo"
    steps = [
        "Setup (once per run, reported apart from cost/img): embed every reference photo",
        "Embed each product crop with the same model",
        "Return the product whose reference photo is most similar (cosine)",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        # Run as an embedding approach: the run's model. Under a Gemini model (the rerank and
        # detect approaches): the fixed ``shortlist_model`` they list in ``also_calls``.
        emb_model = ctx.model if ctx.model in MODELS else self.shortlist_model
        self.emb = VertexEmbeddings(config, model=emb_model)
        root = dataset.data_root(self.dataset)
        gallery = dataset.rpc_gallery(root)
        max_skus = int(os.environ.get("SHELF_BENCH_MAX_GALLERY_SKUS", "400"))
        if max_skus > 0 and len(gallery) > max_skus and os.environ.get("SHELF_BENCH_FULL_GALLERY") != "1":
            cat = dataset.catalog(root, self.dataset)
            # Label-blind stratified gallery selection across all catalog brands (zero eval_samples access)
            by_brand: dict[str, list[int]] = {}
            for pid in sorted(gallery):
                b = str((cat.get(pid) or {}).get("brand") or f"brand_{pid}")
                by_brand.setdefault(b, []).append(pid)
            chosen_pids: set[int] = set()
            brand_keys = sorted(by_brand)
            round_idx = 0
            while len(chosen_pids) < max_skus and brand_keys:
                added_any = False
                for b in brand_keys:
                    pids_b = by_brand[b]
                    if round_idx < len(pids_b):
                        chosen_pids.add(pids_b[round_idx])
                        added_any = True
                        if len(chosen_pids) >= max_skus:
                            break
                if not added_any:
                    break
                round_idx += 1
            gallery = {pid: paths for pid, paths in gallery.items() if pid in chosen_pids}

        refs = [(pid, path) for pid, paths in gallery.items() for path in paths]
        self.jpeg: dict[str, bytes] = _LazyJpegDict()  # lazy reference photos for rerank contact sheets
        is_real_shelf_gallery = self.dataset.startswith("hul") or self.dataset.startswith("gtmerch")

        def embed(ref: tuple[int, str]) -> tuple[int, str, list[float]]:
            pid, path = ref
            cache_key = (emb_model, is_real_shelf_gallery, path)
            cached = _GALLERY_CACHE.get(cache_key)
            if cached is not None:
                _, vec = cached
                return pid, path, vec
            raw_bytes = dataset.read_bytes(path)
            with Image.open(io.BytesIO(raw_bytes)) as im:
                rgb = im.convert("RGB")
                if is_real_shelf_gallery:
                    rgb, _ = restore_photometry(rgb)
                vec = unit(self.emb.image(rgb, ctx))
            _GALLERY_CACHE[cache_key] = (b"", vec)
            return pid, path, vec

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.index = list(pool.map(embed, refs))

        from utils.embeddings import flush_embedding_disk_cache

        flush_embedding_disk_cache()

        self.index_pids = [pid for pid, _, _ in self.index]
        self.index_paths = [path for _, path, _ in self.index]
        self.index_matrix = (
            np.asarray([v for _, _, v in self.index], dtype=np.float32)
            if self.index
            else np.zeros((0, 0), dtype=np.float32)
        )

    def ranked(self, image: Image.Image, ctx: Context,
               allowed_ids: set[int] | None = None) -> list[tuple[float, int, str]]:
        """Products by similarity of their closest reference photo: (cosine, product, photo)."""
        q = unit(self.emb.image(image, ctx))
        best: dict[int, tuple[float, int, str]] = {}
        if (
            hasattr(self, "index_matrix")
            and self.index_matrix.ndim == 2
            and self.index_matrix.shape[0] == len(self.index)
            and self.index_matrix.shape[0] > 0
        ):
            sims = (self.index_matrix @ np.asarray(q, dtype=np.float32)).tolist()
            for s, pid, path in zip(sims, self.index_pids, self.index_paths, strict=True):
                if allowed_ids and pid not in allowed_ids:
                    continue
                if pid not in best or s > best[pid][0]:
                    best[pid] = (float(s), pid, path)
        else:
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
