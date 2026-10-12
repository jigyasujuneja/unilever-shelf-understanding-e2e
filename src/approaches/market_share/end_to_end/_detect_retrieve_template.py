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
from utils.cloudsql import HYBRID_RRF_SQL, CloudSQL, build_hybrid_rrf_params

VECTOR_SQL = """
    SELECT id, 1 - (embedding <=> %s::vector) AS score
    FROM products ORDER BY embedding <=> %s::vector LIMIT 1
"""


@register
class DetectRetrieveAlloyDB(EmbeddingRetrieval):
    name = "detect_retrieve_alloydb"
    task = "end_to_end"
    models: list[str] | None = None  # the Gemini detector
    also_calls = [MODEL]
    architecture = "Gemini detects boxes, Vertex embeddings + AlloyDB (ScaNN) identify each product"
    steps = ["Gemini detects every product box",
             "Embed each crop (gemini-embedding-2) and look it up in AlloyDB (alloydb_scann)"]
    skus = embeddings.SKUS  # priced from the Billing Catalog at run start

    def setup(self, config: dict, ctx: Context) -> None:
        self.emb = embeddings.VertexEmbeddings(config, model=MODEL)
        self.db = AlloyDB(**config["alloydb"])

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        return gemini_detect(image, ctx)

    def identify(self, image: Image.Image, ctx: Context,
                 allowed_ids: set[int] | None = None) -> int | None:
        v = pgvector(self.emb.image(image, ctx))
        rows = self.db.query(VECTOR_SQL, (v, v))
        return int(rows[0][0]) if rows else None


@register
class DetectRetrieveCloudSQL(EmbeddingRetrieval):
    name = "detect_retrieve_cloudsql"
    task = "end_to_end"
    models: list[str] | None = None  # the Gemini detector
    also_calls = [MODEL]
    architecture = "Gemini detects boxes, Vertex embeddings + Cloud SQL Postgres (pgvector HNSW) identify each product"
    steps = ["Gemini detects every product box",
             "Embed each crop (gemini-embedding-2) and look it up in Cloud SQL (pgvector HNSW)"]
    skus = embeddings.SKUS

    def setup(self, config: dict, ctx: Context) -> None:
        self.emb = embeddings.VertexEmbeddings(config, model=MODEL)
        self.db = CloudSQL(**config["cloudsql"])

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        return gemini_detect(image, ctx)

    def identify(self, image: Image.Image, ctx: Context,
                 allowed_ids: set[int] | None = None,
                 category: str = "", pack_form: str = "", keyword: str = "") -> int | None:
        v = pgvector(self.emb.image(image, ctx))
        if category or pack_form or keyword:
            rows = self.db.query(
                HYBRID_RRF_SQL,
                build_hybrid_rrf_params(
                    v,
                    kw_query=keyword,
                    category=category,
                    pack_form=pack_form,
                    limit=1,
                ),
            )
            return int(rows[0][0]) if rows else None
        rows = self.db.query(VECTOR_SQL, (v, v))
        return int(rows[0][0]) if rows else None
