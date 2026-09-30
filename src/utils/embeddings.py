"""Vertex AI multimodal embeddings client: ``multimodalembedding@001`` or ``gemini-embedding-2-preview``.

Crops and text share one vector space, so a shelf crop can be matched against catalog product
photos or names. An approach creates one in ``setup()`` and calls ``image(crop, ctx)``; passing
``ctx`` bills the call to the image being scored (priced from the ``SKUS`` below, which the
approach lists in its own ``skus``). ``multimodalembedding@001`` is billed per image / text
character; Gemini Embedding 2 (support from Jigyasu Juneja's branch) per input token, as reported
in each response.
"""

from __future__ import annotations

import base64
import random
import time

import google.auth
from google.auth.transport.requests import AuthorizedSession
from PIL import Image

from utils.llm import image_to_jpeg, load_config

VERTEX_AI = "C7E2-9256-1C43"  # Billing Catalog service id
MULTIMODAL = "multimodalembedding@001"
GEMINI_EMBEDDING = "gemini-embedding-2-preview"
MODELS = [MULTIMODAL, GEMINI_EMBEDDING]  # approaches that only embed can run either

# Billable unit -> (Billing Catalog service, SKU description). Add to Approach.skus.
SKUS = {"embedding_image": (VERTEX_AI, "Embeddings for multimodal - Image (input)"),
        "embedding_text_char": (VERTEX_AI, "Embeddings for multimodal - Text (input)"),
        "gemini_embedding_image_token": (VERTEX_AI, "Gemini MM Embedding - Image Input"),
        "gemini_embedding_text_token": (VERTEX_AI, "Gemini MM Embedding - Text Input")}


class VertexEmbeddings:
    def __init__(self, config: dict | None = None, model: str = MULTIMODAL, dimension: int | None = None):
        if model not in MODELS:
            raise ValueError(f"embedding model must be one of {MODELS}, got {model!r}")
        gcp = (config or load_config())["gcp"]
        region = gcp.get("region", "us-central1")
        self.gemini = model == GEMINI_EMBEDDING
        self.url = (f"https://{region}-aiplatform.googleapis.com/v1/projects/{gcp['project']}"
                    f"/locations/{region}/publishers/google/models/{model}:"
                    + ("embedContent" if self.gemini else "predict"))
        self.dimension = dimension or (768 if self.gemini else 512)
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.session = AuthorizedSession(creds)

    def image(self, image: Image.Image, ctx=None) -> list[float]:
        b64 = base64.b64encode(image_to_jpeg(image, 1024)).decode()
        if self.gemini:
            return self._embed({"inlineData": {"mimeType": "image/jpeg", "data": b64}}, ctx)
        if ctx:
            ctx.bill("embedding_image", 1)
        return self._predict({"image": {"bytesBase64Encoded": b64}}, "imageEmbedding")

    def text(self, text: str, ctx=None) -> list[float]:
        text = text[:1024]
        if self.gemini:
            return self._embed({"text": text}, ctx)
        if ctx:
            ctx.bill("embedding_text_char", len(text))
        return self._predict({"text": text}, "textEmbedding")

    def _embed(self, part: dict, ctx) -> list[float]:
        """Gemini Embedding 2 (``:embedContent``); bills the input tokens it reports."""
        data = self._post({"content": {"parts": [part]}, "outputDimensionality": self.dimension})
        if ctx:
            for d in data.get("usageMetadata", {}).get("promptTokensDetails", []):
                ctx.bill(f"gemini_embedding_{d['modality'].lower()}_token", d.get("tokenCount", 0))
        return data["embedding"]["values"]

    def _predict(self, instance: dict, key: str) -> list[float]:
        data = self._post({"instances": [instance], "parameters": {"dimension": self.dimension}})
        return data["predictions"][0][key]

    def _post(self, body: dict) -> dict:
        # 429 / 5xx back-off. The quota is per minute, so wait long enough to reach the next
        # minute, with jitter so parallel threads (and Cloud Run tasks) don't retry in step.
        for attempt in range(12):
            r = self.session.post(self.url, json=body, timeout=60)
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        r.raise_for_status()
        return r.json()
