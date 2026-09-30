"""TEMPLATE (not registered: files starting with ``_`` are skipped). Copy to
``detect_retrieve.py`` to benchmark detection + vector retrieval.

Detect boxes with Gemini, then identify each product by embedding its crop with
``gemini-embedding-2-preview`` and looking it up in ``VectorCatalog`` (Cloud SQL for PostgreSQL
``pgvector``, Vertex AI Vector Search, BigQuery ``VECTOR_SEARCH``, or GCS in-memory catalog).
"""

from __future__ import annotations

from PIL import Image

from approaches.base import (
    BOX_LIST_SCHEMA,
    DETECT_PROMPT,
    Approach,
    Box,
    Context,
    register,
    to_pixels,
)
from utils import embeddings
from utils.vector_store import CloudSQL, VectorCatalog, pgvector

VECTOR_SQL = """
    SELECT id, 1 - (embedding <=> %s::vector) AS score
    FROM products ORDER BY embedding <=> %s::vector LIMIT 1
"""


@register
class DetectRetrieve(Approach):
    name = "detect_retrieve"
    architecture = "Gemini detects boxes, Vertex embeddings + Cloud SQL pgvector / Vertex Vector Search identify each product"
    steps = ["Gemini detects every product box",
             "Embed each crop (Vertex multimodal) and look it up in Cloud SQL pgvector / VectorCatalog"]
    skus = embeddings.SKUS  # priced from the Billing Catalog at run start

    def setup(self, config: dict) -> None:
        self.embed = embeddings.VertexEmbeddings(config)
        self.catalog = VectorCatalog(config)
        self.db = CloudSQL(**config.get("cloudsql", config.get("vector_store", {})))

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        res = ctx.ask(image, DETECT_PROMPT, schema=BOX_LIST_SCHEMA, max_side=2048)
        boxes = to_pixels(res.data, 0, 0, *image.size)
        ctx.trace.step("Gemini detection", f"{len(boxes)} boxes", boxes=boxes)
        ids = []
        for b in boxes:
            vec = self.embed.image(image.crop(b), ctx)
            v = pgvector(vec)
            rows = self.db.query(VECTOR_SQL, (v, v))
            ids.append(rows[0][0] if rows else "unknown")
        ctx.trace.step("Identify products", f"{len(set(ids))} distinct products", boxes=boxes)
        return boxes
