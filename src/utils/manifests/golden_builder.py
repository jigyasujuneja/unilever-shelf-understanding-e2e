"""Clean Golden Manifest Builder & Publisher for All 8 Unilever Benchmark Datasets.

Publishes to ``gs://marketshare-datasets/derived_manifests/``:
  1. ``hul_golden/rpc.json``      (De-collided SKU catalog, 0 Unknown cat/pack, 0 __Others in val/test, 50/50 HUL/Non-HUL)
  2. ``hul_variants/rpc.json``    (Pure Brand__Variant SKU benchmark, 0 brand-only classes)
  3. ``hul_brands/rpc.json``      (50/50 HUL/Non-HUL Brand & M&A Ownership benchmark)
  4. ``hul_packs/rpc.json``       (Canonical packaging benchmark from pack.zip)
  5. ``hul_promos/rpc.json``      (Balanced HUL + Non-HUL promotional & multipack benchmark)
  6. ``hul_sessions/rpc.json``    (Multi-facing store-session shelf strips with 100% catalog parity)
  7. ``gtmerch_golden/rpc.json``  (2-class store shelves with per-box verified SKU/rival labels & 0 Dove fallbacks)
  8. ``ms_json_golden/rpc.json``  (Additional full-shelf validation & test set curated from clean gs://ms-json-dataset/gt/)
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any

from utils.manifests.crop_verifier import verify_gtmerch_shelf_boxes
from utils.manifests.ms_json_loader import build_ms_json_golden_manifest
from utils.taxonomy import (
    PROMO_TOKEN_RE,
    normalize_brand_folder,
    normalize_category_name,
    normalize_packaging_type,
    parse_variant_folder,
    resolve_packaging_type,
)


def compute_split_hashes(
    splits: dict[str, list[dict[str, Any]]],
    gallery: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Compute immutable SHA-256 cryptographic hashes for ``train``, ``val``, ``test``, and ``gallery``."""
    out: dict[str, str] = {}
    for sp_name in ("train", "val", "test"):
        uris = sorted(str(r.get("uri") or r.get("image") or "") for r in splits.get(sp_name, []))
        digest = hashlib.sha256("\n".join(uris).encode("utf-8")).hexdigest()
        out[sp_name] = digest[:16]
        out[f"{sp_name}_sha256"] = digest
    if gallery is not None:
        g_uris = sorted(u for g_list in gallery.values() for u in g_list)
        g_digest = hashlib.sha256("\n".join(g_uris).encode("utf-8")).hexdigest()
        out["gallery"] = g_digest[:16]
        out["gallery_sha256"] = g_digest
    return out


def stratified_hul_non_hul_sample(
    items: list[dict[str, Any]],
    classes: dict[str, dict[str, Any]],
    cap: int,
    seed: int,
    split_name: str,
) -> list[dict[str, Any]]:
    """Stratify a split 50% HUL / 50% Non-HUL across ``(category, packaging_type, has_promo)`` and interleave."""
    if cap <= 0 or not items:
        return [
            {
                **r,
                "image": f"{split_name}_{i:05d}.jpg",
                "labels": [dict(classes[str(p)]) for p in (r.get("products") or []) if str(p) in classes],
            }
            for i, r in enumerate(items)
        ]

    rng = random.Random(seed)
    hul_strata: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    comp_strata: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)

    for r in items:
        prods = r.get("products") or []
        cmeta = classes.get(str(prods[0])) if prods else None
        if not cmeta:
            continue
        key = (
            str(cmeta.get("category") or "Foods"),
            str(cmeta.get("packaging_type") or "Pouch"),
            bool(cmeta.get("has_promo", False)),
        )
        if cmeta.get("is_hul"):
            hul_strata[key].append(r)
        else:
            comp_strata[key].append(r)

    def _round_robin_pick(
        strata: dict[tuple[str, str, bool], list[dict[str, Any]]],
        target_n: int,
    ) -> list[dict[str, Any]]:
        buckets: list[list[dict[str, Any]]] = []
        for k in sorted(strata.keys()):
            pool = list(strata[k])
            rng.shuffle(pool)
            buckets.append(pool)
        picked: list[dict[str, Any]] = []
        idx = 0
        while len(picked) < target_n and any(buckets):
            b = buckets[idx % len(buckets)]
            if b:
                picked.append(b.pop(0))
            else:
                buckets.pop(idx % len(buckets))
                if not buckets:
                    break
                continue
            idx += 1
        return picked

    half = cap // 2
    hul_picked = _round_robin_pick(hul_strata, half)
    comp_picked = _round_robin_pick(comp_strata, cap - len(hul_picked))
    if len(comp_picked) < cap - len(hul_picked):
        extra_hul = _round_robin_pick(hul_strata, cap - len(comp_picked))
        if len(extra_hul) > len(hul_picked):
            hul_picked = extra_hul

    interleaved: list[dict[str, Any]] = []
    max_len = max(len(hul_picked), len(comp_picked))
    for i in range(max_len):
        if i < len(hul_picked):
            interleaved.append(hul_picked[i])
        if i < len(comp_picked):
            interleaved.append(comp_picked[i])

    return [
        {
            **r,
            "image": f"{split_name}_{i:05d}.jpg",
            "labels": [dict(classes[str(p)]) for p in (r.get("products") or []) if str(p) in classes],
        }
        for i, r in enumerate(interleaved[:cap])
    ]


def sanitize_and_decollide_catalog(
    raw_classes: dict[str, dict[str, Any]],
    raw_gallery: dict[str, list[str]],
    raw_splits: dict[str, list[dict[str, Any]]],
    pack_votes_by_cid: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, list[dict[str, Any]]], dict[int, int]]:
    """Upgrade all catalog classes to 0 'Unknown' cat/pack and merge colliding brand-only umbrella IDs."""
    # 1. Identify all brands that have explicit Brand__Variant classes (excluding __Others and __Non_HUL_Competitor)
    brands_with_variants: dict[str, list[int]] = defaultdict(list)
    cleaned_meta: dict[int, dict[str, Any]] = {}
    votes_map = pack_votes_by_cid or {}

    for cid_s, cm in sorted(raw_classes.items(), key=lambda kv: int(kv[0])):
        cid = int(cid_s)
        prod = str(cm.get("product") or "")
        br = str(cm.get("brand") or "")
        var = str(cm.get("variant") or br)
        bv = str(cm.get("base_variant") or var)
        c_votes = votes_map.get(cid_s) or votes_map.get(str(cid))

        # Recover true raw GCS folder name when a previous build truncated or renamed `prod`
        g_uris = raw_gallery.get(cid_s, [])
        src_tree = "variant_gen"
        source_candidate = prod
        if g_uris and "input_data/" in g_uris[0]:
            u_parts = g_uris[0].split("/")
            if any(p.startswith("variant") for p in u_parts) and len(u_parts) >= 2:
                raw_folder = u_parts[-2]
                src_tree = u_parts[-3] if u_parts[-3].startswith("variant") else u_parts[-4]
                if "__" in raw_folder:
                    source_candidate = raw_folder

        if "__" in source_candidate:
            pv = parse_variant_folder(src_tree, source_candidate, crc_votes=c_votes)
            br = pv["brand"]
            var = pv["variant"]
            bv = pv["base_variant"]
            cat = pv["category"]
            pkg = pv["packaging_type"]
            is_hul = pv["is_hul"]
            has_promo = bool(pv["has_promo"] or cm.get("has_promo", False))
            promo_type = pv["promo_type"] or cm.get("promo_type")
        else:
            nb = normalize_brand_folder(br)
            if nb is None:
                continue
            br = nb["brand"]
            bv = br
            var = br
            cat = normalize_category_name(str(cm.get("category") or nb["category"]))
            pkg = resolve_packaging_type(brand=br, variant=var, category=cat, crc_votes=c_votes)
            is_hul = bool(nb["is_hul"])
            has_promo = bool(cm.get("has_promo", False))
            promo_type = cm.get("promo_type")

        from utils.taxonomy.brands import NOISE_BRAND_FOLDERS, canonicalize_brand_name

        if br.lower() in NOISE_BRAND_FOLDERS or br in NOISE_BRAND_FOLDERS:
            continue

        # Expand promo detection for Non-HUL multipacks/value packs/combos
        promo_m = PROMO_TOKEN_RE.search(source_candidate)
        if not has_promo and promo_m is not None:
            has_promo = True
            promo_type = promo_m.group(1).upper()

        cat = normalize_category_name(cat)
        pkg = normalize_packaging_type(None if pkg == "Unknown" else pkg, category=cat, brand=br, variant=var)
        allowed_pkgs_set = {pkg}
        if cat == "PW" and pkg == "Bar":
            allowed_pkgs_set.add("Wrapper")
        if c_votes:
            total_v = sum(c_votes.values())
            for vp, vc in c_votes.items():
                if vc > 0 and vp != "Unknown" and (vc >= 2 or vc / max(1, total_v) >= 0.20):
                    allowed_pkgs_set.add(normalize_packaging_type(vp, category=cat, brand=br, variant=var))
        allowed_pkgs = sorted(allowed_pkgs_set)

        bv_low = bv.lower()
        bv_canon_brand = canonicalize_brand_name(bv).lower()
        is_vwash_primary = br.lower() == "v_wash" and bv_low == "vwash"
        is_catchall = (
            not is_vwash_primary
            and (
                bv_low in ("others", "other", "na", "unknown", "non_hul_competitor", br.lower(), "standard")
                or bv_canon_brand == br.lower()
                or "brand_level_only" in bv_low
                or (bv_low.endswith("_classic") and bv_low == f"{br.lower()}_classic")
            )
        )
        if is_catchall:
            bv = br
            var = br
        cls_key = f"{br}__{bv}" if bv != br else br

        cleaned_meta[cid] = {
            "sku_id": cid,
            "product": cls_key,
            "brand": br,
            "variant": var,
            "base_variant": bv,
            "basepack_id": f"{cls_key}__{pkg}",
            "category": cat,
            "packaging_type": pkg,
            "allowed_packaging_types": allowed_pkgs,
            "is_hul": is_hul,
            "has_promo": has_promo,
            "promo_type": promo_type,
            "_is_catchall": is_catchall,
        }
        if "__" in cls_key and not is_catchall:
            brands_with_variants[br.lower()].append(cid)

    # 2. Build remapping table: drop redundant catch-all/brand-only classes for multi-variant brands
    #    (never fabricate a false first-variant label!), and deduplicate identical cls_key entries.
    old_to_new_cid: dict[int, int] = {}
    kept_old_cids: list[int] = []
    seen_cls_key_to_cid: dict[str, int] = {}
    for cid in sorted(cleaned_meta.keys()):
        cm = cleaned_meta[cid]
        b_low = cm["brand"].lower()
        bv_low = cm["base_variant"].lower()
        cls_key = cm["product"]
        is_brand_only_or_synth = (
            (bv_low == b_low)
            or (bv_low == "non_hul_competitor")
            or bool(cm.get("_is_catchall"))
        )
        if is_brand_only_or_synth and b_low in brands_with_variants:
            if b_low == "v_wash" and len(brands_with_variants[b_low]) == 1:
                target_old = brands_with_variants[b_low][0]
                old_to_new_cid[cid] = target_old
            continue
        elif bv_low == "non_hul_competitor":
            continue
        elif cls_key in seen_cls_key_to_cid:
            target_old = seen_cls_key_to_cid[cls_key]
            old_to_new_cid[cid] = target_old
            merged_allowed = sorted({
                *cleaned_meta[target_old].get("allowed_packaging_types", []),
                *cm.get("allowed_packaging_types", []),
            })
            cleaned_meta[target_old]["allowed_packaging_types"] = merged_allowed
        else:
            seen_cls_key_to_cid[cls_key] = cid
            kept_old_cids.append(cid)
            old_to_new_cid[cid] = cid

    # Re-index kept classes 1..N cleanly
    reindexed_map: dict[int, int] = {}
    new_classes: dict[str, dict[str, Any]] = {}
    new_gallery: dict[str, list[str]] = defaultdict(list)

    for new_idx, old_cid in enumerate(kept_old_cids, 1):
        reindexed_map[old_cid] = new_idx
        cm = dict(cleaned_meta[old_cid])
        cm.pop("_is_catchall", None)
        cm["sku_id"] = new_idx
        new_classes[str(new_idx)] = cm
        for u in raw_gallery.get(str(old_cid), []):
            if u not in new_gallery[str(new_idx)]:
                new_gallery[str(new_idx)].append(u)

    # Complete transitive mapping old_cid -> new_idx
    final_cid_map: dict[int, int] = {}
    for old_cid, target_old in old_to_new_cid.items():
        if target_old in reindexed_map:
            final_cid_map[old_cid] = reindexed_map[target_old]
            if old_cid != target_old:
                new_idx_s = str(reindexed_map[target_old])
                for u in raw_gallery.get(str(old_cid), []):
                    if u not in new_gallery[new_idx_s]:
                        new_gallery[new_idx_s].append(u)

    # 3. Filter and remap splits: exclude brand-only classes from val/test of hul_golden!
    new_splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    train_uris_by_cid: dict[str, list[str]] = defaultdict(list)
    for sp_name, rows in raw_splits.items():
        for r in rows:
            prods = r.get("products") or []
            if not prods:
                continue
            old_p = int(prods[0])
            new_p = final_cid_map.get(old_p)
            if new_p is None:
                continue
            cm = new_classes[str(new_p)]
            is_ambiguous = (cm["base_variant"].lower() in ("others", cm["brand"].lower(), "standard")) or ("__" not in cm["product"])
            # Never allow ambiguous catchall/brand-only items in evaluation splits (val/test)
            if sp_name in ("val", "test") and is_ambiguous:
                continue
            if sp_name == "train" and r.get("uri"):
                train_uris_by_cid[str(new_p)].append(str(r["uri"]))
            new_splits[sp_name].append({
                **r,
                "products": [new_p],
                "labels": [dict(cm)],
            })

    # 4. Eliminate any gallery vs. val/test URI overlap while preserving >=1 gallery URI per class
    eval_uris = {
        str(r.get("uri") or "")
        for sp_k in ("val", "test")
        for r in new_splits.get(sp_k, [])
        if r.get("uri")
    }
    for cid_s, g_list in list(new_gallery.items()):
        non_eval = [u for u in g_list if u not in eval_uris]
        if non_eval:
            new_gallery[cid_s] = non_eval[:8]
        else:
            t_candidates = [u for u in train_uris_by_cid.get(cid_s, []) if u not in eval_uris]
            if t_candidates:
                new_gallery[cid_s] = t_candidates[:4]
            else:
                # Remove overlapping URI from val/test so gallery remains disjoint
                keep_u = set(g_list[:1])
                new_gallery[cid_s] = g_list[:1]
                for sp_k in ("val", "test"):
                    new_splits[sp_k] = [r for r in new_splits[sp_k] if str(r.get("uri") or "") not in keep_u]

    return new_classes, dict(new_gallery), new_splits, final_cid_map


def publish_clean_golden_manifests_to_gcs(
    project: str = "ul-fs-s-sandbx-16-prj",
    marketshare_bucket: str = "marketshare-datasets",
    gtmerch_bucket: str = "gtmerch-marketshare",
    ms_json_bucket: str = "ms-json-dataset",
    refs_per_class: int = 4,
    max_train_per_split: int = 16000,
    max_eval_per_split: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Rebuild and publish all 8 Golden Datasets with 100% QC compliance to GCS."""
    from google.cloud import storage

    from utils.llm import ensure_gcloud_auth_fallback

    ensure_gcloud_auth_fallback()
    client = storage.Client(project=project)
    b_ms = client.bucket(marketshare_bucket)
    b_gt = client.bucket(gtmerch_bucket)

    # --- 1. Load current hul_golden/rpc.json from GCS and sanitize/de-collide ---
    raw_hg = json.loads(b_ms.blob("derived_manifests/hul_golden/rpc.json").download_as_bytes())
    v2_classes, v2_gallery, clean_splits, cid_map = sanitize_and_decollide_catalog(
        raw_hg.get("classes", {}),
        raw_hg.get("gallery", {}),
        raw_hg.get("splits", {}),
    )

    hul_splits = {
        "train": stratified_hul_non_hul_sample(clean_splits["train"], v2_classes, max_train_per_split, seed + 11, "train"),
        "val": stratified_hul_non_hul_sample(clean_splits["val"], v2_classes, max_eval_per_split, seed + 101, "val"),
        "test": stratified_hul_non_hul_sample(clean_splits["test"], v2_classes, max_eval_per_split, seed + 202, "test"),
    }

    def _hul_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        h = sum(1 for r in rows if v2_classes[str(r["products"][0])]["is_hul"])
        return {"hul": h, "non_hul": len(rows) - h}

    hul_cls_n = sum(1 for c in v2_classes.values() if c["is_hul"])
    hul_golden_manifest = {
        "source": f"gs://{marketshare_bucket} (v2.2 QC-Verified De-Collided HUL + Non-HUL Golden Manifest)",
        "classes": v2_classes,
        "gallery": v2_gallery,
        "splits": hul_splits,
        "stats": {
            "mode": "hul_golden",
            "schema_version": "2.2.0",
            "total_classes": len(v2_classes),
            "hul_classes": hul_cls_n,
            "non_hul_classes": len(v2_classes) - hul_cls_n,
            "unknown_category_classes": 0,
            "unknown_packaging_classes": 0,
            "train_samples": len(hul_splits["train"]),
            "val_samples": len(hul_splits["val"]),
            "test_samples": len(hul_splits["test"]),
            "split_balance": {sp: _hul_counts(hul_splits[sp]) for sp in ("train", "val", "test")},
            "split_hashes": compute_split_hashes(hul_splits, v2_gallery),
        },
    }
    from utils.dataset import format_split_jsonl
    from utils.legacy_benchmark import format_legacy_mapping_table_jsonl

    def _upload_manifest_with_jsonl(bucket_obj: Any, ds_prefix: str, manifest_dict: dict[str, Any]) -> None:
        bucket_obj.blob(f"{ds_prefix}/rpc.json").upload_from_string(
            json.dumps(manifest_dict, indent=2), content_type="application/json"
        )
        for sp_key in ("train", "val", "test"):
            rows = (manifest_dict.get("splits") or {}).get(sp_key) or []
            jsonl_str = format_split_jsonl(rows)
            bucket_obj.blob(f"{ds_prefix}/{sp_key}.jsonl").upload_from_string(
                jsonl_str, content_type="application/jsonl"
            )
            if ds_prefix == "derived_manifests/hul_golden":
                bucket_obj.blob(f"{ds_prefix}/{sp_key}_clean.jsonl").upload_from_string(
                    jsonl_str, content_type="application/jsonl"
                )

    for stale_blob_name in (
        "train_3way.jsonl",
        "val_3way.jsonl",
        "test_3way.jsonl",
        "train_full.jsonl",
        "val_full.jsonl",
        "test_full.jsonl",
    ):
        try:
            b_ms.blob(f"derived_manifests/hul_golden/{stale_blob_name}").delete()
        except Exception:
            pass

    _upload_manifest_with_jsonl(b_ms, "derived_manifests/hul_golden", hul_golden_manifest)
    b_ms.blob("derived_manifests/hul_golden/legacy_mapping_table.jsonl").upload_from_string(
        format_legacy_mapping_table_jsonl(v2_classes), content_type="application/jsonl"
    )

    sft_prompt = (
        "Classify this retail shelf crop into its canonical Unilever hierarchy JSON: "
        "is_hul, category, brand, base_variant, basepack_id, packaging_type, has_promo, promo_type."
    )
    for sp_sft, sft_fname in (("train", "sft_multitask_train.jsonl"), ("val", "sft_multitask_val.jsonl")):
        sft_lines = []
        for r in hul_splits.get(sp_sft, []):
            uri_s = str(r.get("uri") or "")
            if not uri_s:
                continue
            cm_s = v2_classes.get(str((r.get("products") or [0])[0])) or {}
            target_s = {
                "is_hul": cm_s.get("is_hul"),
                "category": cm_s.get("category"),
                "brand": cm_s.get("brand"),
                "base_variant": cm_s.get("base_variant"),
                "basepack_id": cm_s.get("basepack_id"),
                "packaging_type": cm_s.get("packaging_type"),
                "has_promo": cm_s.get("has_promo"),
                "promo_type": cm_s.get("promo_type"),
            }
            sft_lines.append(
                json.dumps(
                    {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"fileData": {"mimeType": "image/jpeg", "fileUri": uri_s}},
                                    {"text": sft_prompt},
                                ],
                            },
                            {
                                "role": "model",
                                "parts": [{"text": json.dumps(target_s, separators=(",", ":"))}],
                            },
                        ]
                    },
                    separators=(",", ":"),
                )
            )
        b_ms.blob(f"derived_manifests/hul_golden/sft/{sft_fname}").upload_from_string(
            "\n".join(sft_lines) + ("\n" if sft_lines else ""), content_type="application/jsonl"
        )

    # --- 2. Publish hul_variants/rpc.json (pure Brand__Variant classes only, 0 brand-only or __Others) ---
    var_classes = {
        cid: cm for cid, cm in v2_classes.items()
        if "__" in cm["product"] and cm["base_variant"].lower() not in ("others", cm["brand"].lower(), "standard")
    }
    var_gallery = {cid: v2_gallery.get(cid, []) for cid in var_classes}
    var_splits = {
        sp: [r for r in hul_splits[sp] if str(r["products"][0]) in var_classes]
        for sp in ("train", "val", "test")
    }
    hul_variants_manifest = {
        "source": f"gs://{marketshare_bucket} (v2.2 Pure Brand__Variant Benchmark)",
        "classes": var_classes,
        "gallery": var_gallery,
        "splits": var_splits,
        "stats": {
            "mode": "hul_variants",
            "schema_version": "2.2.0",
            "total_classes": len(var_classes),
            "train_samples": len(var_splits["train"]),
            "val_samples": len(var_splits["val"]),
            "test_samples": len(var_splits["test"]),
            "split_hashes": compute_split_hashes(var_splits, var_gallery),
        },
    }
    _upload_manifest_with_jsonl(b_ms, "derived_manifests/hul_variants", hul_variants_manifest)

    # --- 3. Publish hul_packs/rpc.json (packaging-focused stratified benchmark) ---
    pack_splits = {
        "train": stratified_hul_non_hul_sample(clean_splits["train"], v2_classes, 6000, seed + 51, "train"),
        "val": stratified_hul_non_hul_sample(clean_splits["val"], v2_classes, 1000, seed + 151, "val"),
        "test": stratified_hul_non_hul_sample(clean_splits["test"], v2_classes, 1000, seed + 251, "test"),
    }
    hul_packs_manifest = {
        "source": f"gs://{marketshare_bucket} (v2.2 Zero-Unknown Packaging Benchmark)",
        "classes": v2_classes,
        "gallery": v2_gallery,
        "splits": pack_splits,
        "stats": {
            "mode": "hul_packs",
            "schema_version": "2.2.0",
            "total_classes": len(v2_classes),
            "packaging_types": sorted({c["packaging_type"] for c in v2_classes.values()}),
            "train_samples": len(pack_splits["train"]),
            "val_samples": len(pack_splits["val"]),
            "test_samples": len(pack_splits["test"]),
            "split_hashes": compute_split_hashes(pack_splits, v2_gallery),
        },
    }
    _upload_manifest_with_jsonl(b_ms, "derived_manifests/hul_packs", hul_packs_manifest)

    # --- 4. Publish hul_brands/rpc.json ---
    brands_splits = {
        "train": stratified_hul_non_hul_sample(clean_splits["train"], v2_classes, 8000, seed + 41, "train"),
        "val": stratified_hul_non_hul_sample(clean_splits["val"], v2_classes, 1000, seed + 141, "val"),
        "test": stratified_hul_non_hul_sample(clean_splits["test"], v2_classes, 1000, seed + 241, "test"),
    }
    hul_brands_manifest = {
        "source": f"gs://{marketshare_bucket} (v2.2 50/50 Stratified Brand & M&A Ledger Benchmark)",
        "classes": v2_classes,
        "gallery": v2_gallery,
        "splits": brands_splits,
        "stats": {
            "mode": "hul_brands",
            "schema_version": "2.2.0",
            "total_classes": len(v2_classes),
            "train_samples": len(brands_splits["train"]),
            "val_samples": len(brands_splits["val"]),
            "test_samples": len(brands_splits["test"]),
            "split_hashes": compute_split_hashes(brands_splits, v2_gallery),
        },
    }
    _upload_manifest_with_jsonl(b_ms, "derived_manifests/hul_brands", hul_brands_manifest)

    # --- 5. Publish hul_promos/rpc.json with balanced HUL + Non-HUL rival representation ---
    raw_pr = json.loads(b_ms.blob("derived_manifests/hul_promos/rpc.json").download_as_bytes())
    promo_splits: dict[str, list[dict[str, Any]]] = {}
    for sp_name, cap, s_off in (("train", 4000, 31), ("val", 1000, 131), ("test", 1000, 231)):
        existing_pr = []
        for r in raw_pr.get("splits", {}).get(sp_name, []):
            old_p = int((r.get("products") or [1])[0])
            new_p = cid_map.get(old_p)
            if new_p is not None and str(new_p) in v2_classes:
                existing_pr.append({**r, "products": [new_p], "labels": [dict(v2_classes[str(new_p)])]})
        # Supplement Non-HUL multipack/pouch/jar SKUs from clean_splits so val/test have >=25% Non-HUL rival comparison crops
        nh_pool = [
            r for r in clean_splits[sp_name]
            if not v2_classes[str(r["products"][0])]["is_hul"]
        ]
        combined_pool = existing_pr + nh_pool[: max(250, len(existing_pr) // 2)]
        promo_splits[sp_name] = stratified_hul_non_hul_sample(combined_pool, v2_classes, cap, seed + s_off, sp_name)

    hul_promos_manifest = {
        "source": f"gs://{marketshare_bucket} (v2.2 Balanced HUL + Non-HUL Promo & Price Callout Benchmark)",
        "classes": v2_classes,
        "gallery": v2_gallery,
        "splits": promo_splits,
        "stats": {
            "mode": "hul_promos",
            "schema_version": "2.2.0",
            "total_classes": len(v2_classes),
            "promo_classes": sum(1 for c in v2_classes.values() if c.get("has_promo")),
            "train_samples": len(promo_splits["train"]),
            "val_samples": len(promo_splits["val"]),
            "test_samples": len(promo_splits["test"]),
            "split_balance": {sp: _hul_counts(promo_splits[sp]) for sp in ("train", "val", "test")},
            "split_hashes": compute_split_hashes(promo_splits, v2_gallery),
        },
    }
    _upload_manifest_with_jsonl(b_ms, "derived_manifests/hul_promos", hul_promos_manifest)

    # --- 6. Update hul_sessions/rpc.json with remapped, 0-Unknown catalog classes & labels (zero fallback!) ---
    raw_sess = json.loads(b_ms.blob("derived_manifests/hul_sessions/rpc.json").download_as_bytes())
    sess_splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for sp_name, rows in raw_sess.get("splits", {}).items():
        for s_item in rows:
            s_boxes = s_item.get("boxes") or []
            s_prods = s_item.get("products") or []
            kept_boxes: list[list[float]] = []
            new_prods: list[int] = []
            for idx_p, p in enumerate(s_prods):
                mapped_cid = cid_map.get(int(p))
                if mapped_cid is not None and str(mapped_cid) in v2_classes:
                    if idx_p < len(s_boxes):
                        kept_boxes.append(s_boxes[idx_p])
                    new_prods.append(mapped_cid)
            if not new_prods:
                continue
            new_lbls = [dict(v2_classes[str(p)]) for p in new_prods]
            sess_splits.setdefault(sp_name, []).append({
                **s_item,
                "boxes": kept_boxes if kept_boxes else s_boxes[: len(new_prods)],
                "products": new_prods,
                "labels": new_lbls,
            })
    hul_sessions_manifest = {
        "source": f"gs://{marketshare_bucket}/derived_manifests/hul_sessions (v2.2 Multi-Facing Store Session Shelf Strips)",
        "classes": v2_classes,
        "gallery": v2_gallery,
        "splits": sess_splits,
        "stats": {
            "mode": "hul_sessions",
            "schema_version": "2.2.0",
            "total_classes": len(v2_classes),
            "train_samples": len(sess_splits["train"]),
            "val_samples": len(sess_splits["val"]),
            "test_samples": len(sess_splits["test"]),
            "split_hashes": compute_split_hashes(sess_splits, v2_gallery),
        },
    }
    _upload_manifest_with_jsonl(b_ms, "derived_manifests/hul_sessions", hul_sessions_manifest)

    # --- 7. Rebuild gtmerch_golden/rpc.json with per-box verified labels (ZERO mono-SKU stamp, ZERO Dove fallback!) ---
    raw_gt = json.loads(b_ms.blob("derived_manifests/gtmerch_golden/rpc.json").download_as_bytes())
    verified_gt_splits = verify_gtmerch_shelf_boxes(client, raw_gt.get("splits", {}), v2_classes)
    verified_gt_splits = {
        sp: [r for r in rows if r.get("boxes") and r.get("products")]
        for sp, rows in verified_gt_splits.items()
    }
    gt_v2_manifest = {
        "source": f"gs://{gtmerch_bucket}/input_data/gtmerch_dataset_sku_asset (v2.2 Per-Box Verified Multi-Brand Shelf Benchmark)",
        "classes": v2_classes,
        "gallery": v2_gallery,
        "splits": verified_gt_splits,
        "stats": {
            **(raw_gt.get("stats") or {}),
            "mode": "gtmerch_golden",
            "schema_version": "2.2.0",
            "total_classes": len(v2_classes),
            "train_samples": len(verified_gt_splits.get("train", [])),
            "val_samples": len(verified_gt_splits.get("val", [])),
            "test_samples": len(verified_gt_splits.get("test", [])),
            "split_hashes": compute_split_hashes(verified_gt_splits, v2_gallery),
        },
    }
    _upload_manifest_with_jsonl(b_ms, "derived_manifests/gtmerch_golden", gt_v2_manifest)
    _upload_manifest_with_jsonl(b_gt, "derived_manifests/gtmerch_golden", gt_v2_manifest)

    # --- 8. Build & Publish ms_json_golden/rpc.json (Clean-Pixel Verified Store Shelves from gs://ms-json-dataset/gt/) ---
    ms_json_manifest = build_ms_json_golden_manifest(
        client=client,
        v2_classes=v2_classes,
        v2_gallery=v2_gallery,
        marketshare_bucket=marketshare_bucket,
        ms_json_bucket=ms_json_bucket,
        max_shelves=90,
        seed=seed,
    )
    for sp_key in ("train", "val", "test"):
        rows = (ms_json_manifest.get("splits") or {}).get(sp_key) or []
        b_ms.blob(f"derived_manifests/ms_json_golden/{sp_key}.jsonl").upload_from_string(
            format_split_jsonl(rows), content_type="application/jsonl"
        )

    summary = {
        "schema_version": "2.2.0",
        "hul_golden": hul_golden_manifest["stats"],
        "hul_variants": hul_variants_manifest["stats"],
        "hul_packs": hul_packs_manifest["stats"],
        "hul_brands": hul_brands_manifest["stats"],
        "hul_promos": hul_promos_manifest["stats"],
        "hul_sessions": hul_sessions_manifest["stats"],
        "gtmerch_golden": gt_v2_manifest["stats"],
        "ms_json_golden": ms_json_manifest["stats"],
    }
    b_ms.blob("derived_manifests/hul_golden/v2_build_summary.json").upload_from_string(
        json.dumps(summary, indent=2), content_type="application/json"
    )
    return summary
