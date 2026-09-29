"""Vertex AI Gemini Embedding 2 client (``gemini-embedding-2-preview``).

Product crops and text share one multimodal vector space via ``:embedContent``, so a shelf crop
can be matched against catalog product photos, SKU titles, or multimodal (image + text) entries.
An approach creates one in ``setup()`` and calls ``image(crop, ctx)``, ``text(query, ctx)``, or
``multimodal(crop, text_hint, ctx)``; passing ``ctx`` bills the call to the image being scored
(priced from the ``SKUS`` below, which the approach lists in its own ``skus``).
"""

from __future__ import annotations

import base64
import time

import google.auth
from google.auth.transport.requests import AuthorizedSession
from PIL import Image

from utils.llm import image_to_jpeg, load_config

VERTEX_AI = "C7E2-9256-1C43"  # Billing Catalog service id
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2-preview"

# Billable unit -> (Billing Catalog service, SKU description). Add to Approach.skus.
SKUS = {
    "embedding_image": (VERTEX_AI, "Embeddings for multimodal - Image (input)"),
    "embedding_text_char": (VERTEX_AI, "Embeddings for multimodal - Text (input)"),
}


class VertexEmbeddings:
    """Multimodal image and text embeddings via Vertex AI ``gemini-embedding-2-preview``."""

    def __init__(
        self,
        config: dict | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = 768,
    ):
        gcp = (config or load_config())["gcp"]
        region = gcp.get("region", "us-central1")
        self.model = model
        self.dimension = dimension
        method = "embedContent" if model.startswith("gemini-embedding") else "predict"
        self.url = (
            f"https://{region}-aiplatform.googleapis.com/v1/projects/{gcp['project']}"
            f"/locations/{region}/publishers/google/models/{model}:{method}"
        )
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.session = AuthorizedSession(creds)

    def image(
        self,
        image: Image.Image,
        ctx=None,
        text_hint: str | None = None,
    ) -> list[float]:
        b64 = base64.b64encode(image_to_jpeg(image, 1024)).decode()
        if ctx:
            ctx.bill("embedding_image", 1)
            if text_hint:
                ctx.bill("embedding_text_char", len(text_hint[:1024]))
        parts: list[dict] = [{"inlineData": {"mimeType": "image/jpeg", "data": b64}}]
        if text_hint:
            parts.append({"text": text_hint[:1024]})
        return self._embed_parts(parts, fallback_instance={"image": {"bytesBase64Encoded": b64}}, fallback_key="imageEmbedding")

    def text(self, text: str, ctx=None) -> list[float]:
        text = text[:1024]
        if ctx:
            ctx.bill("embedding_text_char", len(text))
        return self._embed_parts(
            [{"text": text}],
            fallback_instance={"text": text},
            fallback_key="textEmbedding",
        )

    def multimodal(self, image: Image.Image, text: str, ctx=None) -> list[float]:
        return self.image(image, ctx=ctx, text_hint=text)

    def _embed_parts(
        self,
        parts: list[dict],
        fallback_instance: dict,
        fallback_key: str,
    ) -> list[float]:
        if self.url.endswith(":embedContent"):
            body: dict = {
                "content": {"parts": parts},
                "outputDimensionality": self.dimension,
            }
        else:
            body = {"instances": [fallback_instance], "parameters": {"dimension": self.dimension}}
        for attempt in range(5):  # 429 / 5xx back-off
            r = self.session.post(self.url, json=body, timeout=60)
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        payload = r.json()
        if "embedding" in payload:
            return [float(v) for v in payload["embedding"].get("values", [])]
        return [float(v) for v in payload["predictions"][0][fallback_key]]
