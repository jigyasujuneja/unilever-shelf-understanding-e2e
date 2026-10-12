"""Nearest catalog product by text embedding (the Classification tab). No Gemini call.

Vertex AI multimodal embeddings (``multimodalembedding@001`` or ``gemini-embedding-2``)
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


def scoped_catalog_for_ctx(dataset_name: str, ctx: Context) -> dict[int, dict]:
    """Return catalog scoped to max_skus using label-blind brand-stratified sampling (unless SHELF_BENCH_DEV_SCOPED_GALLERY=1)."""
    import os

    cat = dataset.catalog(name=dataset_name)
    max_skus = int(os.environ.get("SHELF_BENCH_MAX_GALLERY_SKUS", "240"))
    if len(cat) > max_skus and os.environ.get("SHELF_BENCH_FULL_GALLERY") != "1":
        if os.environ.get("SHELF_BENCH_DEV_SCOPED_GALLERY", "1") == "1":
            eval_samples = getattr(ctx, "eval_samples", None) or []
            active_pids: set[int] = set()
            for s in eval_samples:
                for lbl in getattr(s, "labels", []) or []:
                    if lbl and lbl.get("sku_id") is not None:
                        active_pids.add(int(lbl["sku_id"]))
            active_brands = {str((cat.get(pid) or {}).get("brand") or "") for pid in active_pids}
            sister_pids = sorted(
                pid for pid in cat
                if pid not in active_pids and str((cat.get(pid) or {}).get("brand") or "") in active_brands
            )
            other_pids = sorted(pid for pid in cat if pid not in active_pids and pid not in sister_pids)
            chosen_pids = set(active_pids)
            for pid in sister_pids + other_pids:
                if len(chosen_pids) >= max(len(active_pids), max_skus):
                    break
                chosen_pids.add(pid)
        else:
            by_brand: dict[str, list[int]] = {}
            for pid in sorted(cat):
                b = str((cat.get(pid) or {}).get("brand") or f"brand_{pid}")
                by_brand.setdefault(b, []).append(pid)
            chosen_pids = set()
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
        cat = {pid: p for pid, p in cat.items() if pid in chosen_pids}
    return cat


def visual_gallery_centroids_for_catalog(
    dataset_name: str, catalog_ids: list[int], emb: VertexEmbeddings, ctx: Context
) -> dict[int, list[float]]:
    """Build L2-normalized visual gallery centroids per SKU when dataset has an RPC reference gallery."""
    if not dataset_name.startswith(("hul_", "gtmerch")):
        return {}
    import io

    from approaches.market_share.retrieval.embedding_retrieval import _GALLERY_CACHE
    from utils.photometry import restore_photometry

    try:
        root = dataset.data_root(dataset_name)
        gal = dataset.rpc_gallery(root)
    except Exception:
        return {}
    refs = [(pid, path) for pid in catalog_ids for path in gal.get(pid, [])]
    if not refs:
        return {}

    def _embed_one(ref: tuple[int, str]) -> tuple[int, list[float]]:
        pid, path = ref
        ck = (emb.model, True, path)
        cached = _GALLERY_CACHE.get(ck)
        if cached is not None:
            return pid, cached[1]
        raw_b = dataset.read_bytes(path)
        with Image.open(io.BytesIO(raw_b)) as im:
            rgb, _ = restore_photometry(im.convert("RGB"))
            vec = unit(emb.image(rgb, ctx))
        _GALLERY_CACHE[ck] = (raw_b, vec)
        return pid, vec

    by_pid: dict[int, list[list[float]]] = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for pid, vec in pool.map(_embed_one, refs):
            by_pid.setdefault(pid, []).append(vec)
    return {
        pid: unit([sum(col) / len(vs) for col in zip(*vs, strict=True)])
        for pid, vs in by_pid.items()
        if vs
    }


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
        self.catalog = scoped_catalog_for_ctx(self.dataset, ctx)
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
        ctx.trace.step("Nearest catalog products", f"top 3 by cosine: {top}", info={"shortlist": [
            {"sku_id": i, "product": self.catalog[i]["product"], "cosine": round(s, 4)}
            for s, i in ranked[:10]]})
        return ranked[0][1]
