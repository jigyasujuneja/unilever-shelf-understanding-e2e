"""Training Data Analytics, Cross-Silo Linkage & Embedding Cohesion Audit (`src/utils/hul_audit.py`).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  1. Three-Tier Cross-Silo / Twin-Folder Matching                           │
  │     • Tier 1 (Risky): Filename / Session+Crop-Index collision              │
  │     • Tier 2 (Exact Bytes): GCS Blob MD5 hash match                        │
  │     • Tier 3 (Visual Semantic): Gemini Embedding 2 cosine (centroid & pair)│
  ├────────────────────────────────────────────────────────────────────────────┤
  │  2. Intra-Silo Multi-Folder MD5 Conflict & Cohesion Audit                  │
  │     • Detects byte-identical MD5 crops placed in >1 variant folder         │
  │     • Computes L2 class centroids mu_c across prepared gallery/train crops │
  │     • Flags suspect mislabelled crops where cos(v_i, mu_own) < cos(v_i,r)  │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  3. Cloud SQL (pgvector) & BigQuery Vector Audit SQL Queries               │
  │     • Ready-to-run SQL for at-scale catalog & training crop hygiene checks │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

# Cloud SQL (pgvector) query to find suspect mislabelled training crops whose embedding is
# closer to another class's centroid than to their assigned folder label's centroid.
CLOUDSQL_TRAINING_AUDIT_SQL = """
WITH class_centroids AS (
    SELECT category, brand, variant,
           AVG(embedding) AS centroid_vec,
           COUNT(*) AS n_crops
    FROM training_crops
    GROUP BY category, brand, variant
),
own_vs_rival AS (
    SELECT t.crop_uri,
           t.md5_hash,
           t.session_id,
           t.brand AS assigned_brand,
           t.variant AS assigned_variant,
           1 - (t.embedding <=> own_c.centroid_vec) AS own_centroid_cos,
           rival.brand AS rival_brand,
           rival.variant AS rival_variant,
           1 - (t.embedding <=> rival.centroid_vec) AS rival_centroid_cos
    FROM training_crops t
    JOIN class_centroids own_c
      ON t.category = own_c.category AND t.brand = own_c.brand AND t.variant = own_c.variant
    CROSS JOIN LATERAL (
        SELECT c2.brand, c2.variant, c2.centroid_vec
        FROM class_centroids c2
        WHERE NOT (c2.brand = t.brand AND c2.variant = t.variant)
        ORDER BY t.embedding <=> c2.centroid_vec
        LIMIT 1
    ) rival
)
SELECT crop_uri, md5_hash, session_id, assigned_brand, assigned_variant,
       ROUND(own_centroid_cos::numeric, 4) AS own_cos,
       rival_brand, rival_variant,
       ROUND(rival_centroid_cos::numeric, 4) AS rival_cos,
       ROUND((own_centroid_cos - rival_centroid_cos)::numeric, 4) AS margin
FROM own_vs_rival
WHERE own_centroid_cos < rival_centroid_cos
ORDER BY margin ASC;
"""

# BigQuery SQL query comparing filename match vs MD5 match vs ML.DISTANCE cosine similarity
# across the 3 siloed trees (brand_data, skin_cleansing, Pack).
BIGQUERY_CROSS_SILO_AUDIT_SQL = """
SELECT
  a.silo_tree AS tree_a,
  b.silo_tree AS tree_b,
  a.assigned_label AS label_a,
  b.assigned_label AS label_b,
  COUNT(*) AS filename_matches,
  COUNTIF(a.md5_hash = b.md5_hash) AS exact_md5_matches,
  COUNTIF(1.0 - ML.DISTANCE(a.embedding, b.embedding, 'COSINE') >= 0.985) AS embedding_identical_matches,
  ROUND(AVG(1.0 - ML.DISTANCE(a.embedding, b.embedding, 'COSINE')), 4) AS mean_embedding_cosine
FROM `hul_poc.training_crops_embeddings` a
JOIN `hul_poc.training_crops_embeddings` b
  ON a.crop_filename = b.crop_filename
 AND a.silo_tree < b.silo_tree
GROUP BY 1, 2, 3, 4
ORDER BY filename_matches DESC;
"""


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def audit_intra_silo_md5_conflicts(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Identify byte-identical MD5 hashes placed in >1 distinct class folder within a silo.

    Each item in ``records`` is ``{"uri": str, "md5": str, "folder": str}`` (if ``folder`` is omitted,
    it is parsed as the parent directory of ``uri``).
    """
    md5_to_folders: dict[str, set[str]] = defaultdict(set)
    md5_to_uris: dict[str, list[str]] = defaultdict(list)
    for r in records:
        uri = str(r["uri"])
        md5 = str(r.get("md5") or "")
        if not md5:
            continue
        folder = str(r.get("folder") or (uri.split("/")[-2] if "/" in uri else ""))
        md5_to_folders[md5].add(folder)
        md5_to_uris[md5].append(uri)

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    conflicted_md5s: dict[str, list[str]] = {}
    for md5, folders in md5_to_folders.items():
        if len(folders) > 1:
            sorted_f = sorted(folders)
            conflicted_md5s[md5] = sorted_f
            for i in range(len(sorted_f)):
                for j in range(i + 1, len(sorted_f)):
                    pair_counts[(sorted_f[i], sorted_f[j])] += 1

    top_pairs = [
        {"folder_a": a, "folder_b": b, "shared_md5_count": c}
        for (a, b), c in sorted(pair_counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    return {
        "total_records": len(records),
        "unique_md5s": len(md5_to_folders),
        "conflicted_md5_count": len(conflicted_md5s),
        "conflicted_md5s": conflicted_md5s,
        "conflicted_folder_pairs": top_pairs,
    }


def audit_embedding_cohesion(
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit intra-class embedding cohesion and detect suspect mislabelled training crops.

    Each item in ``items`` must contain ``{"uri": str, "class_name": str, "embedding": list[float]}``
    and optionally ``"md5"`` and ``"session_id"``.
    """
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_class[str(it["class_name"])].append({**it, "vec": _unit(list(it["embedding"]))})

    centroids: dict[str, list[float]] = {}
    for cls_name, rows in by_class.items():
        vecs = [r["vec"] for r in rows]
        centroids[cls_name] = _unit([sum(col) / len(vecs) for col in zip(*vecs, strict=True)])

    per_class: list[dict[str, Any]] = []
    suspect_items: list[dict[str, Any]] = []

    for cls_name in sorted(by_class):
        rows = by_class[cls_name]
        own_c = centroids[cls_name]
        own_scores: list[float] = []
        for r in rows:
            v = r["vec"]
            s_own = _dot(v, own_c)
            own_scores.append(s_own)
            rivals = [(rc, _dot(v, rv)) for rc, rv in centroids.items() if rc != cls_name]
            if rivals:
                rival_cls, s_rival = max(rivals, key=lambda x: x[1])
                margin = s_own - s_rival
                if margin < 0.0:
                    suspect_items.append({
                        "uri": r["uri"],
                        "md5": r.get("md5", ""),
                        "session_id": r.get("session_id", ""),
                        "assigned_class": cls_name,
                        "own_centroid_cos": round(s_own, 4),
                        "nearest_rival_class": rival_cls,
                        "rival_centroid_cos": round(s_rival, 4),
                        "margin": round(margin, 4),
                    })
        per_class.append({
            "class_name": cls_name,
            "n_crops": len(rows),
            "mean_own_centroid_cos": round(sum(own_scores) / len(own_scores), 4),
            "min_own_centroid_cos": round(min(own_scores), 4),
        })

    suspect_items.sort(key=lambda x: x["margin"])
    mean_cohesion = (
        round(sum(c["mean_own_centroid_cos"] for c in per_class) / len(per_class), 4)
        if per_class
        else 0.0
    )
    return {
        "total_crops": len(items),
        "total_classes": len(per_class),
        "mean_cohesion": mean_cohesion,
        "suspect_count": len(suspect_items),
        "suspect_pct": round(100.0 * len(suspect_items) / max(1, len(items)), 2),
        "suspect_items": suspect_items,
        "per_class": per_class,
    }


def compare_folder_embeddings(
    vecs_a: list[list[float]],
    vecs_b: list[list[float]],
    names_a: list[str] | None = None,
    names_b: list[str] | None = None,
    md5_a: list[str] | None = None,
    md5_b: list[str] | None = None,
) -> dict[str, Any]:
    """Compare two folder populations by filename overlap, MD5 overlap, and embedding similarity."""
    ua = [_unit(v) for v in vecs_a]
    ub = [_unit(v) for v in vecs_b]
    cent_a = _unit([sum(col) / len(ua) for col in zip(*ua, strict=True)])
    cent_b = _unit([sum(col) / len(ub) for col in zip(*ub, strict=True)])
    centroid_cos = round(_dot(cent_a, cent_b), 4)
    max_pair_cos = round(max(_dot(va, vb) for va in ua for vb in ub), 4)
    fname_overlap = len(set(names_a or []) & set(names_b or []))
    md5_overlap = len(set(md5_a or []) & set(md5_b or []))
    return {
        "centroid_cosine": centroid_cos,
        "max_cross_crop_cosine": max_pair_cos,
        "filename_overlap": fname_overlap,
        "md5_overlap": md5_overlap,
        "same_visual_concept": centroid_cos >= 0.88,
    }


def validate_golden_joins(payload: dict[str, Any]) -> dict[str, Any]:
    """Programmatically verify a Golden JSON payload across all 5 join and split invariants."""
    items = payload.get("items", [])
    three_way = [it for it in items if it.get("join_mode") == "3way_brand_variant_pack"]
    two_way_bv = [it for it in items if it.get("join_mode") == "2way_brand_variant"]
    two_way_pv = [it for it in items if it.get("join_mode") == "2way_pack_variant"]
    two_way_pb = [it for it in items if it.get("join_mode") == "2way_pack_brand"]

    three_way_complete = sum(
        1 for it in three_way
        if it.get("brand_data_uri") and it.get("variant_uri") and it.get("pack_data_uri")
    )
    two_way_bv_complete = sum(
        1 for it in two_way_bv if it.get("brand_data_uri") and it.get("variant_uri")
    )
    two_way_pv_complete = sum(
        1 for it in two_way_pv if it.get("pack_data_uri") and it.get("variant_uri")
    )
    two_way_pb_complete = sum(
        1 for it in two_way_pb if it.get("pack_data_uri") and it.get("brand_data_uri")
    )

    clean_items = [it for it in items if it.get("status") == "clean"]
    unquarantined_conflicts = sum(1 for it in clean_items if it.get("conflict_reasons"))

    # Verify zero store-visit session leakage across gallery/val/test among real store crops
    split_sessions: dict[str, set[str]] = {"gallery": set(), "val": set(), "test": set()}
    render_leakage = 0
    for it in clean_items:
        sp = str(it.get("split", "gallery"))
        if it.get("is_planogram_render"):
            if sp in ("val", "test"):
                render_leakage += 1
            continue
        sid = str(it.get("session_id", ""))
        if sid and sp in split_sessions:
            split_sessions[sp].add(sid)

    gal_val_overlap = len(split_sessions["gallery"] & split_sessions["val"])
    gal_test_overlap = len(split_sessions["gallery"] & split_sessions["test"])
    val_test_overlap = len(split_sessions["val"] & split_sessions["test"])
    total_session_leakage = gal_val_overlap + gal_test_overlap + val_test_overlap

    sample_3way = [
        {
            "md5": it["md5"],
            "brand_data_uri": it.get("brand_data_uri"),
            "variant_uri": it.get("variant_uri"),
            "pack_data_uri": it.get("pack_data_uri"),
            "brand": it["labels"].get("brand"),
            "base_variant": it["labels"].get("base_variant"),
            "packaging_type": it["labels"].get("packaging_type"),
            "join_tier": it.get("join_tier"),
            "join_confidence": it.get("join_confidence"),
        }
        for it in three_way
        if it.get("status") == "clean"
    ][:20]

    sample_quarantined = [
        {
            "md5": it["md5"],
            "conflict_reasons": it.get("conflict_reasons"),
            "brand_data_uri": it.get("brand_data_uri"),
            "variant_uri": it.get("variant_uri"),
            "pack_data_uri": it.get("pack_data_uri"),
        }
        for it in items
        if it.get("status") == "quarantined_conflict"
    ][:20]

    all_passed = (
        three_way_complete == len(three_way)
        and two_way_bv_complete == len(two_way_bv)
        and two_way_pv_complete == len(two_way_pv)
        and two_way_pb_complete == len(two_way_pb)
        and unquarantined_conflicts == 0
        and total_session_leakage == 0
        and render_leakage == 0
    )
    return {
        "all_invariants_passed": all_passed,
        "total_items": len(items),
        "clean_items": len(clean_items),
        "quarantined_items": len(items) - len(clean_items),
        "uri_column_completeness": {
            "3way_total": len(three_way),
            "3way_with_all_3_uris": three_way_complete,
            "2way_brand_variant_total": len(two_way_bv),
            "2way_brand_variant_with_both_uris": two_way_bv_complete,
            "2way_pack_variant_total": len(two_way_pv),
            "2way_pack_variant_with_both_uris": two_way_pv_complete,
            "2way_pack_brand_total": len(two_way_pb),
            "2way_pack_brand_with_both_uris": two_way_pb_complete,
        },
        "cross_silo_clean_brand_agreement_rate": 1.0 if unquarantined_conflicts == 0 else 0.0,
        "unquarantined_conflicts": unquarantined_conflicts,
        "session_split_leakage_count": total_session_leakage,
        "planogram_render_leakage_count": render_leakage,
        "sample_verified_3way_joins": sample_3way,
        "sample_quarantined_conflicts": sample_quarantined,
    }

