"""Serverless BigQuery Golden Catalog, Vector Search & Unstructured AI Functions.

Why BigQuery Over Always-On Cloud SQL for Analytical & Unstructured Catalog Workloads:
  1. **Zero Idle Compute Cost**: Stays alive indefinitely in ``ul-fs-s-sandbx-16-prj.hul_golden_catalog``
     (``us-central1``) with $0/hr idle instance cost, unlike Cloud SQL / AlloyDB PostgreSQL instances.
  2. **Unstructured GCS Object Tables + BigQuery AI Functions**: Native ``OBJECT_METADATA`` external tables
     over ``gs://marketshare-datasets/**`` allow direct SQL execution of ``ML.GENERATE_EMBEDDING``
     (``gemini-embedding-2``) and ``AI.GENERATE_TABLE`` / ``ML.GENERATE_TEXT``
     (``gemini-3.5-flash-lite`` and ``gemini-3.8-flash``).
  3. **Native Vector Search & Web-Crawled Catalog Joins**: Supports ``VECTOR_SEARCH`` and ``COSINE_DISTANCE``
     on ``ARRAY<FLOAT64>`` columns and joins directly with ``brand_variants_1104`` for web-crawled reference galleries.
"""

from __future__ import annotations

from typing import Any

from utils.llm import load_config


def _load_bq_defaults() -> tuple[str, str, str]:
    """Load BigQuery project, dataset, and regional location from ``config.yaml``.

    While Vertex AI publisher models default to ``location: global``, BigQuery datasets
    and external Vertex AI connections require a concrete region, so ``global`` automatically
    falls back to ``gcp.region`` (``us-central1``).
    """
    try:
        gcp_cfg = load_config().get("gcp", {})
    except Exception:
        gcp_cfg = {}
    project = str(gcp_cfg.get("project") or "ul-fs-s-sandbx-16-prj")
    dataset = str(gcp_cfg.get("bigquery_dataset") or "hul_golden_catalog")
    region = str(gcp_cfg.get("region") or "us-central1")
    loc = str(gcp_cfg.get("location") or region)
    resolved_loc = region if loc.lower() == "global" else loc
    return project, dataset, resolved_loc


DEFAULT_BQ_PROJECT, DEFAULT_BQ_DATASET, DEFAULT_BQ_LOCATION = _load_bq_defaults()

BIGQUERY_GOLDEN_CROPS_DDL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.golden_crops` (
  md5 STRING NOT NULL,
  canonical_uri STRING NOT NULL,
  brand_data_uri STRING,
  variant_uri STRING,
  pack_data_uri STRING,
  join_mode STRING NOT NULL,
  join_tier STRING NOT NULL,
  join_confidence FLOAT64 NOT NULL,
  is_conflicting_md5 BOOL NOT NULL,
  status STRING NOT NULL,
  conflict_reasons ARRAY<STRING>,
  session_id STRING,
  crop_idx INT64,
  bbox ARRAY<INT64>,
  is_planogram_render BOOL,
  split STRING NOT NULL,
  is_hul BOOL NOT NULL,
  category STRING NOT NULL,
  raw_silo_category STRING,
  category_source STRING,
  brand STRING,
  variant STRING,
  base_variant STRING,
  basepack_id STRING,
  packaging_type STRING,
  packaging_type_source STRING,
  allowed_packaging_types ARRAY<STRING>,
  is_catchall_variant BOOL,
  variant_enrichment_source STRING,
  has_promo BOOL,
  promo_type STRING,
  has_price_tag BOOL,
  artwork_gen STRING,
  embedding ARRAY<FLOAT64>,
  pack_sub_centroid ARRAY<FLOAT64>,
  lab_descriptor ARRAY<FLOAT64>
)
PARTITION BY RANGE_BUCKET(crop_idx, GENERATE_ARRAY(0, 500, 50))
CLUSTER BY status, is_hul, category, brand;
""".strip()

BIGQUERY_JOIN_VALIDATION_SQL = """
-- Inspect side-by-side Cloud Storage URIs across all 3 tables (brand_data, variant_*, pack_data)
SELECT
  md5,
  join_mode,
  join_tier,
  join_confidence,
  brand,
  base_variant,
  basepack_id,
  category,
  raw_silo_category,
  packaging_type,
  brand_data_uri,
  variant_uri,
  pack_data_uri
FROM `{project}.{dataset}.golden_crops`
WHERE join_mode = '3way_brand_variant_pack'
  AND is_conflicting_md5 = FALSE
LIMIT 100;
""".strip()

BIGQUERY_GTMERCH_DETECTIONS_DDL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.gtmerch_shelf_detections` (
  md5 STRING NOT NULL,
  uri STRING NOT NULL,
  status STRING NOT NULL,
  split STRING NOT NULL,
  outlet_id STRING NOT NULL,
  asset_id STRING,
  capture_date STRING,
  timestamp_ms INT64,
  session_id STRING NOT NULL,
  width INT64,
  height INT64,
  asset_box_count INT64,
  product_box_count INT64,
  inside_asset_product_count INT64,
  asset_boxes ARRAY<STRUCT<bbox ARRAY<INT64>>>,
  product_boxes ARRAY<STRUCT<
    product_idx INT64,
    bbox ARRAY<INT64>,
    inside_asset_idx INT64,
    asset_containment_ratio FLOAT64,
    is_inside_asset BOOL,
    linked_crop_md5 STRING,
    linked_crop_uri STRING,
    brand STRING,
    base_variant STRING,
    basepack_id STRING,
    category STRING,
    packaging_type STRING,
    is_hul BOOL
  >>
)
CLUSTER BY split, outlet_id, session_id;
""".strip()

BIGQUERY_OBJECT_TABLE_DDL = """
-- 1. Create Object Table over gs://marketshare-datasets and gs://gtmerch-marketshare for unstructured AI analysis
CREATE EXTERNAL TABLE IF NOT EXISTS `{project}.{dataset}.gcs_shelf_objects`
WITH CONNECTION `{project}.{location}.vertex_ai_conn`
OPTIONS (
  object_metadata = 'SIMPLE',
  uris = ['gs://marketshare-datasets/brand_data/**',
          'gs://marketshare-datasets/input_data/**',
          'gs://marketshare-datasets/pack_data/**',
          'gs://gtmerch-marketshare/**']
);

-- 2. Register Remote Gemini Embedding 2, Gemini 3.5 Flash-Lite, and Gemini 3.8 Flash Models in BigQuery ML
CREATE OR REPLACE MODEL `{project}.{dataset}.gemini_embedding_2`
REMOTE WITH CONNECTION `{project}.{location}.vertex_ai_conn`
OPTIONS (endpoint = 'gemini-embedding-2');

CREATE OR REPLACE MODEL `{project}.{dataset}.gemini_3_5_flash_lite`
REMOTE WITH CONNECTION `{project}.{location}.vertex_ai_conn`
OPTIONS (endpoint = 'gemini-3.5-flash-lite');

CREATE OR REPLACE MODEL `{project}.{dataset}.gemini_3_8_flash`
REMOTE WITH CONNECTION `{project}.{location}.vertex_ai_conn`
OPTIONS (endpoint = 'gemini-3.8-flash');
""".strip()

BIGQUERY_AI_UNSTRUCTURED_AUDIT_SQL = r"""
-- Unstructured AI Function Query: Pre-filter to real image crops (excluding .jsonl and
-- synthetic copy/DA_TBC_ files) BEFORE invoking ML.GENERATE_EMBEDDING on gemini-embedding-2.
SELECT
  uri,
  md5_hash,
  ml_generate_embedding_result AS embedding_768d
FROM ML.GENERATE_EMBEDDING(
  MODEL `{project}.{dataset}.gemini_embedding_2`,
  (
    SELECT *
    FROM `{project}.{dataset}.gcs_shelf_objects`
    WHERE content_type IN ('image/jpeg', 'image/png')
      AND NOT REGEXP_CONTAINS(
        uri,
        r'/jsonl/|(?i)(?:^|/)copy\d*\.(?:jpe?g|png)$|(?i)\s-\s*Copy|(?i)\(copy\)|/DA_TBC_'
      )
  ),
  STRUCT(TRUE AS flatten_json_output, 768 AS output_dimensionality)
);
""".strip()

BIGQUERY_VECTOR_SEARCH_SQL = """
-- Vector Search in BigQuery over Golden Catalog / Web-Crawled Reference Gallery
SELECT
  query.uri AS query_uri,
  base.brand,
  base.base_variant,
  base.basepack_id,
  base.packaging_type,
  base.is_hul,
  (1.0 - distance) AS cosine_similarity
FROM VECTOR_SEARCH(
  TABLE `{project}.{dataset}.golden_crops`,
  'embedding',
  (SELECT uri, embedding FROM `{project}.{dataset}.query_crops`),
  top_k => 5,
  distance_type => 'COSINE'
);
""".strip()


BIGQUERY_BRAND_SKU_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.brand_sku_registry` (
  canonical_brand STRING NOT NULL,
  is_hul BOOL NOT NULL,
  parent_company STRING NOT NULL,
  category STRING NOT NULL,
  dual_umbrella BOOL NOT NULL,
  valid_from DATE,
  valid_to DATE,
  verification_source STRING NOT NULL,
  confidence FLOAT64 NOT NULL,
  sku_count INT64 NOT NULL,
  aliases ARRAY<STRING>,
  sample_variants ARRAY<STRING>
)
CLUSTER BY is_hul, parent_company, category;
""".strip()

PARENT_COMPANY_SOS_SQL = """
-- Competitor Conglomerate Share-of-Shelf (SOS) Roll-up (HUL vs P&G vs ITC vs Dabur vs Godrej vs Marico)
SELECT
  COALESCE(r.parent_company, IF(g.is_hul, 'Hindustan Unilever Limited (HUL)', 'Regional / Other Competitor FMCG')) AS parent_company,
  g.category,
  COUNT(*) AS facing_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY g.category), 2) AS category_facing_sos_pct
FROM `{project}.{dataset}.golden_crops` AS g
LEFT JOIN `{project}.{dataset}.brand_sku_registry` AS r
  ON LOWER(g.brand) = LOWER(r.canonical_brand)
WHERE g.status = 'clean'
GROUP BY parent_company, g.category
ORDER BY g.category, facing_count DESC;
""".strip()


def golden_item_to_bq_row(item: dict[str, Any]) -> dict[str, Any]:
    """Flatten a ``golden_payload['items']`` record into a row matching ``BIGQUERY_GOLDEN_CROPS_DDL``."""
    lab = item.get("labels", {})
    srcs = item.get("source_uris", {})
    return {
        "md5": str(item["md5"]),
        "canonical_uri": str(item["canonical_uri"]),
        "brand_data_uri": item.get("brand_data_uri") or srcs.get("brand_data"),
        "variant_uri": item.get("variant_uri") or srcs.get("variant"),
        "pack_data_uri": item.get("pack_data_uri") or srcs.get("pack_data"),
        "join_mode": str(item["join_mode"]),
        "join_tier": str(item.get("join_tier", "tier1_exact_md5")),
        "join_confidence": float(item.get("join_confidence", 1.0)),
        "is_conflicting_md5": bool(item.get("is_conflicting_md5", False)),
        "status": str(item["status"]),
        "conflict_reasons": list(item.get("conflict_reasons") or []),
        "session_id": str(item.get("session_id") or ""),
        "crop_idx": int(item.get("crop_idx") or 0),
        "bbox": [int(x) for x in item["bbox"]] if item.get("bbox") else [],
        "is_planogram_render": bool(item.get("is_planogram_render", False)),
        "split": str(item["split"]),
        "is_hul": bool(lab.get("is_hul", False)),
        "category": str(lab.get("category") or "Unknown"),
        "raw_silo_category": lab.get("raw_silo_category"),
        "category_source": str(lab.get("category_source") or "unknown"),
        "brand": lab.get("brand"),
        "variant": lab.get("variant"),
        "base_variant": lab.get("base_variant"),
        "basepack_id": lab.get("basepack_id"),
        "packaging_type": str(lab.get("packaging_type") or "Unknown"),
        "packaging_type_source": str(lab.get("packaging_type_source") or "unknown"),
        "allowed_packaging_types": list(lab.get("allowed_packaging_types") or []),
        "is_catchall_variant": bool(lab.get("is_catchall_variant", False)),
        "variant_enrichment_source": str(lab.get("variant_enrichment_source") or "variant_folder"),
        "has_promo": bool(lab.get("has_promo", False)),
        "promo_type": lab.get("promo_type"),
        "has_price_tag": bool(lab.get("has_price_tag", False)),
        "artwork_gen": str(lab.get("artwork_gen") or "standard"),
    }


class BigQueryCatalogClient:
    """Lightweight BigQuery v2 REST client for ``hul_golden_catalog`` (defaults to ``config.yaml``)."""

    def __init__(
        self,
        project: str = DEFAULT_BQ_PROJECT,
        dataset: str = DEFAULT_BQ_DATASET,
        location: str = DEFAULT_BQ_LOCATION,
        session: Any | None = None,
    ) -> None:
        self.project = project
        self.dataset = dataset
        # Fall back to regional location (us-central1) if caller passes Vertex AI's 'global'
        self.location = DEFAULT_BQ_LOCATION if str(location).lower() == "global" else location
        self._session = session

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self._session = AuthorizedSession(creds)
        return self._session

    def query(self, sql: str, params: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Execute a Standard SQL query in BigQuery and return the JSON response."""
        body: dict[str, Any] = {
            "query": sql,
            "useLegacySql": False,
            "location": self.location,
        }
        if params:
            body["parameterMode"] = "NAMED"
            body["queryParameters"] = params
        resp = self._get_session().post(
            f"https://bigquery.googleapis.com/bigquery/v2/projects/{self.project}/queries",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    def render_ddls(self) -> dict[str, str]:
        """Return rendered DDL and AI Function SQL strings for ``project`` and ``dataset``."""
        fmt = {"project": self.project, "dataset": self.dataset, "location": self.location}
        return {
            "golden_crops_ddl": BIGQUERY_GOLDEN_CROPS_DDL.format(**fmt),
            "gtmerch_detections_ddl": BIGQUERY_GTMERCH_DETECTIONS_DDL.format(**fmt),
            "brand_sku_registry_ddl": BIGQUERY_BRAND_SKU_REGISTRY_DDL.format(**fmt),
            "parent_company_sos_sql": PARENT_COMPANY_SOS_SQL.format(**fmt),
            "join_validation_sql": BIGQUERY_JOIN_VALIDATION_SQL.format(**fmt),
            "object_table_and_models_ddl": BIGQUERY_OBJECT_TABLE_DDL.format(**fmt),
            "ai_unstructured_audit_sql": BIGQUERY_AI_UNSTRUCTURED_AUDIT_SQL.format(**fmt),
            "vector_search_sql": BIGQUERY_VECTOR_SEARCH_SQL.format(**fmt),
        }

