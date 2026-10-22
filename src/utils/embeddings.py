"""Vertex AI multimodal embeddings client: ``multimodalembedding@001`` or ``gemini-embedding-2``.

Crops and text share one vector space, so a shelf crop can be matched against catalog product
photos or names. An approach creates one in ``setup()`` and calls ``image(crop, ctx)``; passing
``ctx`` bills the call to the image being scored (priced from the ``SKUS`` below, which the
approach lists in its own ``skus``). ``multimodalembedding@001`` is billed per image / text
character; Gemini Embedding 2 (support from Jigyasu Juneja's branch) per input token, as reported
in each response.
"""

from __future__ import annotations

import base64
import logging
import random
import threading
import time
from contextlib import nullcontext

import google.auth
import requests
from google.auth.transport.requests import AuthorizedSession
from PIL import Image

from utils import telemetry
from utils.llm import image_to_jpeg, load_config

VERTEX_AI = "C7E2-9256-1C43"  # Billing Catalog service id
MULTIMODAL = "multimodalembedding@001"
GEMINI_EMBEDDING = "gemini-embedding-2"
MODELS = [MULTIMODAL, GEMINI_EMBEDDING]  # approaches that only embed can run either

# Billable unit -> (Billing Catalog service, SKU description). Add to Approach.skus.
SKUS = {"embedding_image": (VERTEX_AI, "Embeddings for multimodal - Image (input)"),
        "embedding_text_char": (VERTEX_AI, "Embeddings for multimodal - Text (input)"),
        "gemini_embedding_image_token": (VERTEX_AI, "Gemini MM Embedding - Image Input"),
        "gemini_embedding_text_token": (VERTEX_AI, "Gemini MM Embedding - Text Input")}

_EMB_CACHE_LOCK = threading.Lock()
_EMB_RESPONSE_CACHE: dict[tuple[str, str], tuple[list[float], dict[str, int | float], int]] = {}
_DISK_CACHE_LOADED = False
_DISK_CACHE_DIRTY = 0


GCS_EMB_CACHE_URI = "gs://ul-fs-s-sandbx-16-prj-shelf-images/cache/embeddings_response_cache.pkl"


def _disk_cache_path():
    import tempfile
    from pathlib import Path

    p = Path(tempfile.gettempdir()) / "shelf_bench_emb_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p / "embeddings_response_cache.pkl"


def _ensure_disk_cache_loaded() -> None:
    global _DISK_CACHE_LOADED
    with _EMB_CACHE_LOCK:
        if _DISK_CACHE_LOADED:
            return
        _DISK_CACHE_LOADED = True
    import os
    import pickle

    cp = _disk_cache_path()
    if cp.exists():
        try:
            data = pickle.loads(cp.read_bytes())
            if isinstance(data, dict):
                with _EMB_CACHE_LOCK:
                    _EMB_RESPONSE_CACHE.update(data)
            if os.environ.get("CLOUD_RUN_JOB"):
                cp.unlink(missing_ok=True)
            return
        except Exception:
            pass
    try:
        from utils import dataset

        raw = dataset.read_bytes(GCS_EMB_CACHE_URI)
        data = pickle.loads(raw)
        if isinstance(data, dict):
            with _EMB_CACHE_LOCK:
                _EMB_RESPONSE_CACHE.update(data)
        if not os.environ.get("CLOUD_RUN_JOB"):
            cp.write_bytes(raw)
    except Exception:
        pass


def flush_embedding_disk_cache() -> None:
    global _DISK_CACHE_DIRTY
    with _EMB_CACHE_LOCK:
        if _DISK_CACHE_DIRTY <= 0:
            return
        snapshot = dict(_EMB_RESPONSE_CACHE)
        _DISK_CACHE_DIRTY = 0
    import os
    import pickle

    cp = _disk_cache_path()
    tmp = cp.with_suffix(".pkl.tmp")
    try:
        payload = pickle.dumps(snapshot)
        if not os.environ.get("CLOUD_RUN_JOB"):
            cp.unlink(missing_ok=True)
            tmp.write_bytes(payload)
            tmp.replace(cp)
        try:
            from utils import dataset

            bucket, blob_name = dataset._split_gs(GCS_EMB_CACHE_URI)
            dataset._gcs().bucket(bucket).blob(blob_name).upload_from_string(payload)
        except Exception:
            pass
    except Exception:
        pass


def _build_session(creds) -> AuthorizedSession:
    sess = AuthorizedSession(creds)
    if hasattr(sess, "mount"):
        from requests.adapters import HTTPAdapter

        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
    return sess


class VertexEmbeddings:
    def __init__(self, config: dict | None = None, model: str = MULTIMODAL, dimension: int | None = None):
        if model not in MODELS:
            raise ValueError(f"embedding model must be one of {MODELS}, got {model!r}")
        _ensure_disk_cache_loaded()
        gcp = (config or load_config())["gcp"]
        # Use the global Vertex AI endpoint for all embedding models for lower latency/cost (and GA gemini-embedding-2 requires global).
        loc = gcp.get("location", "global")
        host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
        self.model = model
        self.gemini = model == GEMINI_EMBEDDING
        self.url = (f"https://{host}/v1/projects/{gcp['project']}"
                    f"/locations/{loc}/publishers/google/models/{model}:"
                    + ("embedContent" if self.gemini else "predict"))
        self.dimension = dimension or (768 if self.gemini else 512)
        self._creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self._owner_thread = threading.current_thread()
        self._tls = threading.local()
        self.session = _build_session(self._creds)
        self._default_session = self.session

    def _get_session(self):
        # Preserve test mock compatibility when `self.session` is replaced or called on the owner thread
        if self.session is not self._default_session or threading.current_thread() is self._owner_thread:
            return self.session
        sess = getattr(self._tls, "session", None)
        if sess is None:
            sess = _build_session(self._creds)
            self._tls.session = sess
        return sess

    def image(self, image: Image.Image, ctx=None) -> list[float]:
        import hashlib

        b64 = base64.b64encode(image_to_jpeg(image, 1024)).decode()
        cache_key = (self.url, hashlib.sha256(b64.encode()).hexdigest())

        def _do():
            global _DISK_CACHE_DIRTY
            if "ul-fs" in self.url:
                with _EMB_CACHE_LOCK:
                    cached = _EMB_RESPONSE_CACHE.get(cache_key)
                if cached is not None:
                    return cached
            res = (
                self._embed({"inlineData": {"mimeType": "image/jpeg", "data": b64}})
                if self.gemini
                else self._predict({"image": {"bytesBase64Encoded": b64}}, "imageEmbedding", {"embedding_image": 1})
            )
            if "ul-fs" in self.url:
                should_flush = False
                with _EMB_CACHE_LOCK:
                    _EMB_RESPONSE_CACHE[cache_key] = res
                    _DISK_CACHE_DIRTY += 1
                    if _DISK_CACHE_DIRTY >= 40:
                        should_flush = True
                if should_flush:
                    flush_embedding_disk_cache()
            return res

        return self._traced("image", ctx, _do)

    def text(self, text: str, ctx=None) -> list[float]:
        text = text[:1024]
        cache_key = (self.url, f"txt:{text}")

        def _do():
            global _DISK_CACHE_DIRTY
            if "ul-fs" in self.url:
                with _EMB_CACHE_LOCK:
                    cached = _EMB_RESPONSE_CACHE.get(cache_key)
                if cached is not None:
                    return cached
            res = (
                self._embed({"text": text})
                if self.gemini
                else self._predict({"text": text}, "textEmbedding", {"embedding_text_char": len(text)})
            )
            if "ul-fs" in self.url:
                should_flush = False
                with _EMB_CACHE_LOCK:
                    _EMB_RESPONSE_CACHE[cache_key] = res
                    _DISK_CACHE_DIRTY += 1
                    if _DISK_CACHE_DIRTY >= 40:
                        should_flush = True
                if should_flush:
                    flush_embedding_disk_cache()
            return res

        return self._traced("text", ctx, _do)

    def _traced(self, kind: str, ctx, call) -> list[float]:
        """Run one embedding request, bill it to ``ctx`` and make it auditable: a span + an
        ``embedding_call`` log entry under the image's span, and a record on the next step.
        Setup calls (no image span) are billed but not traced: 800 reference photos would
        otherwise be 800 root traces."""
        parent = getattr(ctx, "otel_parent", None)
        attrs = {"shelf_bench.embedding.model": self.model, "shelf_bench.embedding.kind": kind,
                 "shelf_bench.embedding.dimension": self.dimension}
        span_cm = (telemetry.span(f"embedding {self.model}", parent=parent, **attrs)
                   if parent is not None else nullcontext())
        with span_cm as span:
            t0 = time.perf_counter()
            try:
                vec, billed, attempts = call()
            except Exception as e:
                if span is not None:
                    telemetry.fail(span, e)
                    telemetry.log(span, "embedding_call_failed", {**attrs, "error": f"{type(e).__name__}: {e}"[:2000]},
                                  level=logging.ERROR)
                raise
            seconds = round(time.perf_counter() - t0, 3)
            if ctx is None:
                return vec
            for unit, n in billed.items():
                ctx.bill(unit, n)
            if span is not None:
                fields = {**attrs, "shelf_bench.embedding.seconds": seconds,
                          "shelf_bench.embedding.attempts": attempts,
                          **{f"shelf_bench.billed.{u}": n for u, n in billed.items()}}
                telemetry.set_attrs(span, fields)
                telemetry.log(span, "embedding_call", fields)
                ctx.trace.record_call({"span_id": telemetry.ids(span)[1], "kind": "embedding",
                                       "model": self.model, "input": kind, "dimension": self.dimension,
                                       "seconds": seconds, "attempts": attempts, "billed": billed})
        return vec

    def _embed(self, part: dict) -> tuple[list[float], dict[str, float], int]:
        """Gemini Embedding 2 (``:embedContent``); billed by the input tokens it reports."""
        data, attempts = self._post({"content": {"parts": [part]}, "outputDimensionality": self.dimension})
        billed = {f"gemini_embedding_{d['modality'].lower()}_token": d.get("tokenCount", 0)
                  for d in data.get("usageMetadata", {}).get("promptTokensDetails", [])}
        return data["embedding"]["values"], billed, attempts

    def _predict(self, instance: dict, key: str, billed: dict[str, float]) -> tuple[list[float], dict[str, float], int]:
        data, attempts = self._post({"instances": [instance], "parameters": {"dimension": self.dimension}})
        return data["predictions"][0][key], billed, attempts

    def _post(self, body: dict) -> tuple[dict, int]:
        # 429 / 5xx and transient TLS/socket back-off with jitter across concurrent worker threads.
        sess = self._get_session()
        last_exc: Exception | None = None
        r = None
        attempt = 0
        for attempt in range(12):
            try:
                r = sess.post(self.url, json=body, timeout=60)
                if r.status_code not in (429, 500, 502, 503, 504):
                    break
            except (requests.exceptions.RequestException, OSError) as exc:
                last_exc = exc
                if attempt == 11:
                    raise
            time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        if r is None:
            raise last_exc or RuntimeError("embedding request returned no response")
        r.raise_for_status()
        return r.json(), attempt + 1
