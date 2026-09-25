"""Vertex AI multimodal embeddings client (``multimodalembedding@001``).

Crops and text share one vector space, so a shelf crop can be matched against catalog product
photos or names. An approach creates one in ``setup()`` and calls ``image(crop, ctx)``; passing
``ctx`` bills the call to the image being scored (priced from the ``SKUS`` below, which the
approach lists in its own ``skus``).
"""

from __future__ import annotations

import base64
import time

from PIL import Image

from utils.llm import image_to_jpeg, load_config

VERTEX_AI = "C7E2-9256-1C43"  # Billing Catalog service id

# Billable unit -> (Billing Catalog service, SKU description). Add to Approach.skus.
SKUS = {"embedding_image": (VERTEX_AI, "Embeddings for multimodal - Image (input)"),
        "embedding_text_char": (VERTEX_AI, "Embeddings for multimodal - Text (input)")}


class VertexEmbeddings:
    def __init__(self, config: dict | None = None, model: str = "multimodalembedding@001",
                 dimension: int = 512):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        gcp = (config or load_config())["gcp"]
        region = gcp.get("region", "us-central1")
        self.url = (f"https://{region}-aiplatform.googleapis.com/v1/projects/{gcp['project']}"
                    f"/locations/{region}/publishers/google/models/{model}:predict")
        self.dimension = dimension
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.session = AuthorizedSession(creds)

    def image(self, image: Image.Image, ctx=None) -> list[float]:
        b64 = base64.b64encode(image_to_jpeg(image, 1024)).decode()
        if ctx:
            ctx.bill("embedding_image", 1)
        return self._predict({"image": {"bytesBase64Encoded": b64}}, "imageEmbedding")

    def text(self, text: str, ctx=None) -> list[float]:
        text = text[:1024]
        if ctx:
            ctx.bill("embedding_text_char", len(text))
        return self._predict({"text": text}, "textEmbedding")

    def _predict(self, instance: dict, key: str) -> list[float]:
        body = {"instances": [instance], "parameters": {"dimension": self.dimension}}
        for attempt in range(5):  # 429 / 5xx back-off
            r = self.session.post(self.url, json=body, timeout=60)
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        return r.json()["predictions"][0][key]
