"""TEMPLATE (not registered: files starting with ``_`` are skipped). Copy to
``detect_retrieve.py`` once an AlloyDB catalog exists.

Detect boxes with Gemini, then identify each product by embedding its crop and looking it up in
AlloyDB. Everything that varies between retrieval experiments lives here, in the approach:
which embedder, which database, and the SQL (pure vector below; a hybrid variant in comments).
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
from utils.alloydb import AlloyDB, pgvector

VECTOR_SQL = """
    SELECT id, 1 - (embedding <=> %s::vector) AS score
    FROM products ORDER BY embedding <=> %s::vector LIMIT 1
"""
# Hybrid example (vector + keyword, reciprocal-rank fusion) - swap in and pass (vec, text):
# WITH v AS (SELECT id, RANK() OVER (ORDER BY embedding <=> %s::vector) r
#            FROM products ORDER BY r LIMIT 20),
#      k AS (SELECT id, RANK() OVER (ORDER BY ts_rank(to_tsvector(name), plainto_tsquery(%s)) DESC) r
#            FROM products WHERE to_tsvector(name) @@ plainto_tsquery(%s) ORDER BY r LIMIT 20)
# SELECT id, SUM(1.0 / (60 + r)) score FROM (SELECT * FROM v UNION ALL SELECT * FROM k) u
# GROUP BY id ORDER BY score DESC LIMIT 1


@register
class DetectRetrieve(Approach):
    name = "detect_retrieve"
    architecture = "Gemini detects boxes, Vertex embeddings + AlloyDB identify each product"
    steps = ["Gemini detects every product box",
             "Embed each crop (Vertex multimodal) and look it up in AlloyDB"]
    skus = embeddings.SKUS  # priced from the Billing Catalog at run start

    def setup(self, config: dict) -> None:
        self.embed = embeddings.VertexEmbeddings(config)
        self.db = AlloyDB(**config["alloydb"])

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        res = ctx.ask(image, DETECT_PROMPT, schema=BOX_LIST_SCHEMA, max_side=2048)
        boxes = to_pixels(res.data, 0, 0, *image.size)
        ctx.trace.step("Gemini detection", f"{len(boxes)} boxes", boxes=boxes)
        ids = []
        for b in boxes:
            v = pgvector(self.embed.image(image.crop(b), ctx))
            rows = self.db.query(VECTOR_SQL, (v, v))
            ids.append(rows[0][0] if rows else "unknown")
        ctx.trace.step("Identify products", f"{len(set(ids))} distinct products", boxes=boxes)
        return boxes
