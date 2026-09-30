"""TEMPLATE (not registered: files starting with ``_`` are skipped). Copy to
``detect_retrieve_alloydb.py`` once an AlloyDB catalog exists.

``detect_retrieve`` (detect_identify.py) with the reference vectors in AlloyDB instead of in
memory: what a real catalog of thousands of products needs. Gemini detects boxes; the runner
crops each box and calls ``identify``, which embeds the crop and looks it up in AlloyDB.
Everything that varies between retrieval experiments lives here: which embedder, which
database, and the SQL (pure vector below; a hybrid variant in comments).
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Box, Context, register
from approaches.market_share.end_to_end.detect_identify import gemini_detect
from approaches.market_share.retrieval.embedding_retrieval import MODEL, EmbeddingRetrieval
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
class DetectRetrieveAlloyDB(EmbeddingRetrieval):
    name = "detect_retrieve_alloydb"
    task = "end_to_end"
    models: list[str] | None = None  # the Gemini detector
    also_calls = [MODEL]
    architecture = "Gemini detects boxes, Vertex embeddings + AlloyDB identify each product"
    steps = ["Gemini detects every product box",
             "Embed each crop (gemini-embedding-2-preview) and look it up in AlloyDB"]
    skus = embeddings.SKUS  # priced from the Billing Catalog at run start

    def setup(self, config: dict, ctx: Context) -> None:
        # The products table must hold the reference-photo embeddings (same model, 768-d),
        # with id = the RPC product id; see the setup SQL in utils/alloydb.py.
        self.emb = embeddings.VertexEmbeddings(config, model=MODEL)
        self.db = AlloyDB(**config["alloydb"])

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        return gemini_detect(image, ctx)

    def identify(self, image: Image.Image, ctx: Context) -> int | None:
        v = pgvector(self.emb.image(image, ctx))
        rows = self.db.query(VECTOR_SQL, (v, v))
        return int(rows[0][0]) if rows else None
