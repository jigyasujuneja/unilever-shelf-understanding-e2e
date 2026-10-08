"""Store-Session Leakage-Free Splitter & Manifest Builder for ``gs://marketshare-datasets``.

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Input: Stream of GCS blob metadata records (name, md5_hash, size)         │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  1. Content, Copy & Multi-Folder Conflict Hygiene (0 Image Downloads)      │
  │     • Drops synthetic padding files (copy0.jpg..copy99.jpg, - Copy (2).jpg)│
  │     • Quarantines intra-silo multi-folder MD5 conflicts from eval splits   │
  │     • Deduplicates identical crops within a class via GCS blob md5_hash    │
  │  2. Cross-Silo Packaging Backfill                                          │
  │     • Links pack_data crops via MD5 / filename to backfill packaging_type  │
  │  3. Store-Visit Session-Grouped Split (80% gallery / 10% val / 10% test)   │
  │     • Hashes session_id -> bucket in [0..99] so crops from the same store  │
  │       visit NEVER appear in both reference gallery and val/test splits.    │
  │  4. RPC-Compatible Manifest Generation (classes, gallery, splits)          │
  │     • Supports hul_variants (W4), hul_brands (W3), hul_packs (W2+W6),      │
  │       hul_promos (W9), and hul_sessions (Store-Session SOS).               │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from utils.hul_taxonomy import (
    PKG_SUFFIX_RE,
    STRIP_PROMO_ART_SUFFIX_RE,
    normalize_brand_folder,
    parse_crop_filename,
    parse_pack_filename_brand_variant,
    parse_pack_folder,
    parse_variant_folder,
)

HUL_DATASETS = ("hul_variants", "hul_brands", "hul_packs", "hul_promos", "hul_sessions")


def _is_twin_pack_or_promo_suffix(shortest_bv: str, candidate_bv: str) -> bool:
    """Return True only if ``candidate_bv`` extends ``shortest_bv`` solely by a packaging, promo, or artwork suffix.

    Prevents distinct product variants that happen to share a prefix (e.g.
    ``Ponds__Bright_Beauty_Facewash`` vs ``Ponds__Bright_Beauty_Facewash_Scrub``,
    or ``Lux__Rose`` vs ``Lux__Rose_And_Aloe_Vera``) from being merged as twin
    folders when they collide on an MD5 hash.
    """
    if not candidate_bv.lower().startswith(shortest_bv.lower()):
        return False
    rem = candidate_bv[len(shortest_bv) :]
    if not rem.startswith("_"):
        return False
    return bool(PKG_SUFFIX_RE.fullmatch(rem) or STRIP_PROMO_ART_SUFFIX_RE.fullmatch(rem))


def session_split_bucket(session_id: str, seed: int = 42) -> str:
    """Deterministic 80/10/10 split ('gallery', 'val', 'test') keyed strictly by ``session_id``."""
    digest = hashlib.sha1(f"{seed}:{session_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "gallery"
    if bucket < 90:
        return "val"
    return "test"


def _resolve_record_class(
    rec: dict[str, Any],
    mode: str,
    pack_lookup_by_md5: dict[str, str] | None = None,
    pack_lookup_by_fname: dict[str, str] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Resolve ``(cls_key, class_meta_dict, filename_info)`` for a single GCS blob record."""
    uri = str(rec["uri"])
    parts = uri.replace("gs://", "").split("/")
    fname = parts[-1]
    folder = parts[-2] if len(parts) >= 2 else ""
    f_info = parse_crop_filename(fname)
    if f_info["is_synthetic_copy"]:
        return None

    md5 = str(rec.get("md5") or "")
    if mode == "hul_brands":
        b_info = normalize_brand_folder(folder)
        if b_info is None:
            return None
        cls_key = b_info["brand"]
        meta = {
            "product": b_info["brand"],
            "brand": b_info["brand"],
            "variant": b_info["brand"],
            "base_variant": b_info["brand"],
            "category": b_info["category"],
            "packaging_type": "Unknown",
            "is_hul": b_info["is_hul"],
            "has_promo": False,
            "promo_type": None,
        }
        return cls_key, meta, f_info

    if mode == "hul_packs":
        p_info = parse_pack_folder(folder)
        ebv = parse_pack_filename_brand_variant(fname)
        cls_key = p_info["raw_folder"]
        meta = {
            "product": p_info["raw_folder"],
            "brand": ebv["brand"] if ebv else p_info["category"],
            "variant": ebv["variant"] if ebv else p_info["packaging_type"],
            "base_variant": ebv["base_variant"] if ebv else p_info["packaging_type"],
            "category": p_info["category"],
            "packaging_type": p_info["packaging_type"],
            "is_hul": ebv["is_hul"] if ebv else True,
            "has_promo": ebv["has_promo"] if ebv else False,
            "promo_type": ebv["promo_type"] if ebv else None,
        }
        return cls_key, meta, f_info

    # Variant-level or session/promo modes
    if "brand_data" in parts and mode == "hul_sessions":
        b_info = normalize_brand_folder(folder)
        if b_info is None:
            return None
        cls_key = f"{b_info['brand']}__Brand_Level"
        meta = {
            "product": cls_key,
            "brand": b_info["brand"],
            "variant": "Brand_Level",
            "base_variant": "Brand_Level",
            "category": b_info["category"],
            "packaging_type": "Unknown",
            "is_hul": b_info["is_hul"],
            "has_promo": False,
            "promo_type": None,
        }
        return cls_key, meta, f_info

    tree = next((p for p in parts if p.startswith("variant_")), "variant_skin")
    v_info = parse_variant_folder(tree, folder)
    if mode == "hul_promos" and not v_info["has_promo"]:
        return None

    pack_type = v_info["packaging_type"]
    if pack_type == "Unknown":
        if pack_lookup_by_md5 and md5 in pack_lookup_by_md5:
            pack_type = pack_lookup_by_md5[md5]
        elif pack_lookup_by_fname and fname in pack_lookup_by_fname:
            pack_type = pack_lookup_by_fname[fname]

    cls_key = v_info["raw_folder"]
    meta = {
        "product": f"{v_info['brand']}__{v_info['variant']}",
        "brand": v_info["brand"],
        "variant": v_info["variant"],
        "base_variant": v_info["base_variant"],
        "category": v_info["category"],
        "packaging_type": pack_type,
        "is_hul": v_info["is_hul"],
        "has_promo": v_info["has_promo"],
        "promo_type": v_info["promo_type"],
        "has_price_tag": v_info["has_price_tag"],
        "artwork_gen": v_info["artwork_gen"],
    }
    return cls_key, meta, f_info


def dedup_and_split_blobs(
    records: list[dict[str, Any]],
    mode: str = "hul_variants",
    refs_per_class: int = 4,
    max_eval_per_split: int = 100,
    seed: int = 42,
    pack_lookup_by_md5: dict[str, str] | None = None,
    pack_lookup_by_fname: dict[str, str] | None = None,
    quarantine_multi_folder_md5: bool = True,
) -> dict[str, Any]:
    """Deduplicate GCS crop records by MD5 & filename and build a leakage-free ``rpc.json`` manifest.

    Each item in ``records`` is ``{"uri": "gs://...", "md5": "...", "width": 256, "height": 256}``.
    """
    # Pass 1: Detect intra-silo multi-folder MD5 conflicts (identical bytes in >1 distinct class folder)
    md5_to_classes: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        uri = str(rec["uri"])
        fname = uri.rsplit("/", 1)[-1]
        if parse_crop_filename(fname)["is_synthetic_copy"]:
            continue
        resolved = _resolve_record_class(rec, mode, pack_lookup_by_md5, pack_lookup_by_fname)
        if resolved is None:
            continue
        cls_key, _, _ = resolved
        md5 = str(rec.get("md5") or "")
        if md5:
            md5_to_classes[md5].add(cls_key)

    conflicted_md5s = {
        m for m, cls_set in md5_to_classes.items() if len(cls_set) > 1
    } if quarantine_multi_folder_md5 else set()

    class_meta: dict[str, dict[str, Any]] = {}
    by_class_split: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    seen_md5_per_class: dict[str, set[str]] = defaultdict(set)
    dropped_copies = 0
    dropped_md5 = 0
    quarantined_conflicts = 0

    for rec in records:
        uri = str(rec["uri"])
        fname = uri.rsplit("/", 1)[-1]
        f_info = parse_crop_filename(fname)
        if f_info["is_synthetic_copy"]:
            dropped_copies += 1
            continue

        resolved = _resolve_record_class(rec, mode, pack_lookup_by_md5, pack_lookup_by_fname)
        if resolved is None:
            continue
        cls_key, meta, f_info = resolved

        md5 = str(rec.get("md5") or uri)
        if md5 in seen_md5_per_class[cls_key]:
            dropped_md5 += 1
            continue
        seen_md5_per_class[cls_key].add(md5)
        class_meta[cls_key] = meta

        sp = session_split_bucket(f_info["session_id"], seed=seed)
        if md5 in conflicted_md5s:
            # Quarantine ambiguous multi-folder MD5 crops from gallery, val, and test splits
            # so two competing classes never share a byte-identical gallery reference or test crop.
            quarantined_conflicts += 1
            continue

        by_class_split[cls_key][sp].append({
            "uri": uri,
            "md5": md5,
            "session_id": f_info["session_id"],
            "crop_idx": f_info["crop_idx"],
            "bbox": f_info["bbox"],
            "width": int(rec.get("width", 256)),
            "height": int(rec.get("height", 256)),
        })

    sorted_keys = sorted(class_meta)
    classes: dict[str, dict[str, Any]] = {}
    cls_key_to_id: dict[str, int] = {}
    gallery: dict[str, list[str]] = {}
    splits: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}

    for idx, cls_key in enumerate(sorted_keys, 1):
        cid = str(idx)
        cls_key_to_id[cls_key] = idx
        classes[cid] = {"sku_id": idx, **class_meta[cls_key]}
        gal_pool = sorted(by_class_split[cls_key]["gallery"], key=lambda x: x["uri"])
        if not gal_pool:
            all_items = by_class_split[cls_key]["val"] + by_class_split[cls_key]["test"]
            if all_items:
                gal_pool = [all_items[0]]
                gal_sess = gal_pool[0]["session_id"]
                by_class_split[cls_key]["val"] = [
                    x for x in by_class_split[cls_key]["val"] if x["session_id"] != gal_sess
                ]
                by_class_split[cls_key]["test"] = [
                    x for x in by_class_split[cls_key]["test"] if x["session_id"] != gal_sess
                ]
        step = max(1, len(gal_pool) // max(1, refs_per_class))
        gallery[cid] = [gal_pool[i]["uri"] for i in range(0, len(gal_pool), step)[:refs_per_class]]

    if mode == "hul_sessions":
        # Group val/test crops by store visit session_id into multi-facing store session samples
        for sp_name in ("val", "test"):
            by_sess: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
            for cls_key in sorted_keys:
                idx = cls_key_to_id[cls_key]
                for item in by_class_split[cls_key][sp_name]:
                    by_sess[item["session_id"]].append((idx, item))
            for sess_id in sorted(by_sess):
                sess_items = sorted(by_sess[sess_id], key=lambda x: (x[1]["crop_idx"], x[1]["uri"]))
                tile_w = 256
                tile_h = 256
                boxes: list[list[float]] = []
                products: list[int] = []
                crop_uris: list[str] = []
                for col_i, (sku_id, it) in enumerate(sess_items[:12]):
                    x1 = float(col_i * tile_w)
                    boxes.append([x1, 0.0, x1 + float(tile_w), float(tile_h)])
                    products.append(sku_id)
                    crop_uris.append(it["uri"])
                if not products:
                    continue
                img_name = f"{sp_name}_{len(splits[sp_name]):05d}.jpg"
                splits[sp_name].append({
                    "image": img_name,
                    "uri": crop_uris[0] if len(crop_uris) == 1 else f"images/{img_name}",
                    "crop_uris": crop_uris,
                    "session_id": sess_id,
                    "width": tile_w * len(products),
                    "height": tile_h,
                    "boxes": boxes,
                    "products": products,
                })
    else:
        for idx, cls_key in enumerate(sorted_keys, 1):
            for sp_name in ("val", "test"):
                for item in sorted(by_class_split[cls_key][sp_name], key=lambda x: x["uri"]):
                    w, h = item["width"], item["height"]
                    splits[sp_name].append({
                        "image": f"{sp_name}_{len(splits[sp_name]):05d}.jpg",
                        "uri": item["uri"],
                        "session_id": item["session_id"],
                        "width": w,
                        "height": h,
                        "boxes": [[0.0, 0.0, float(w), float(h)]],
                        "products": [idx],
                    })

    for sp_name in ("val", "test"):
        if max_eval_per_split > 0 and len(splits[sp_name]) > max_eval_per_split:
            step = max(1, len(splits[sp_name]) // max_eval_per_split)
            splits[sp_name] = [
                {**r, "image": f"{sp_name}_{i:05d}.jpg"}
                for i, r in enumerate(splits[sp_name][::step][:max_eval_per_split])
            ]

    return {
        "classes": classes,
        "gallery": gallery,
        "splits": splits,
        "stats": {
            "mode": mode,
            "total_classes": len(classes),
            "dropped_synthetic_copies": dropped_copies,
            "dropped_md5_duplicates": dropped_md5,
            "quarantined_multi_folder_md5s": len(conflicted_md5s),
            "quarantined_eval_conflicts": quarantined_conflicts,
            "val_samples": len(splits["val"]),
            "test_samples": len(splits["test"]),
        },
    }


def build_golden_json_payload(records: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    """Join identical images across ``brand_data``, ``variant_*``, and ``pack_data`` into a single Golden JSON payload.

    Join hierarchy:
      1. **Tier 1 (Exact MD5 Join)**: Unifies ``brand_data``, ``variant_*``, and ``pack_data`` blobs sharing
         identical ``md5`` bytes (188,376 ``brand <-> variant`` + 38,723 ``pack <-> variant`` + 45,481 ``pack <-> brand``).
      2. **Tier 2 (Compatible Twin Union)**: When an MD5 appears in both a suffix-free folder
         (``Dove__Dryness_Care__Conditioner``) and its container/promo twin (``_Tube`` / ``_Rs_10``),
         merges the container and promo attributes into one enriched record instead of discarding.
      3. **Tier 3 (Same Session + Crop Index + Matching Brand)**: Links re-encoded JPEG pairs
         (e.g. ``10724_1.jpg`` in ``AAyush`` and ``Ayush__Sensitivity_Relief``) when ``(session_id, crop_idx, brand)``
         agree across silos, while rejecting cross-category counter collisions (e.g. ``1098_53.jpg`` ``AAyush`` vs ``Rexona``).
      4. **Conflict Quarantine**: Marks true label disagreements (different ``brand``, ``base_variant``, or ``packaging_type``)
         as ``status="quarantined_conflict"``.
    """
    by_md5: dict[str, dict[str, Any]] = {}
    dropped_copies = 0

    for rec in records:
        uri = str(rec["uri"])
        parts = uri.replace("gs://", "").split("/")
        fname = parts[-1]
        folder = parts[-2] if len(parts) >= 2 else ""
        f_info = parse_crop_filename(fname)
        if f_info["is_synthetic_copy"]:
            dropped_copies += 1
            continue

        md5 = str(rec.get("md5") or uri)
        entry = by_md5.setdefault(
            md5,
            {
                "md5": md5,
                "session_id": f_info["session_id"],
                "crop_idx": f_info["crop_idx"],
                "bbox": f_info["bbox"],
                "width": rec.get("width"),
                "height": rec.get("height"),
                "embedding": rec.get("embedding"),
                "is_planogram_render": fname.startswith("HUL-") and f_info["bbox"] is None,
                "is_tier3_merged": False,
                "source_uris": {},
                "brand_obs": [],
                "variant_obs": [],
                "pack_obs": [],
            },
        )
        if rec.get("width") and not entry["width"]:
            entry["width"] = rec.get("width")
            entry["height"] = rec.get("height")
        if rec.get("embedding") and not entry["embedding"]:
            entry["embedding"] = rec.get("embedding")
        if f_info["bbox"] and not entry["bbox"]:
            entry["bbox"] = f_info["bbox"]
            entry["is_planogram_render"] = False
        # Prefer numeric store-visit session_id or HUL shelf session_id over generic image counters
        cur_sid = str(entry["session_id"])
        new_sid = str(f_info["session_id"])
        if not cur_sid.isdigit() and (new_sid.isdigit() or ("HUL-" in new_sid and "image" in cur_sid.lower())):
            entry["session_id"] = f_info["session_id"]
            entry["crop_idx"] = f_info["crop_idx"]

        if "brand_data" in parts:
            b_info = normalize_brand_folder(folder)
            if b_info:
                entry["source_uris"]["brand_data"] = uri
                entry["brand_obs"].append(b_info)
        elif "pack_data" in parts:
            p_info = parse_pack_folder(folder)
            ebv = parse_pack_filename_brand_variant(fname)
            entry["source_uris"]["pack_data"] = uri
            entry["pack_obs"].append((p_info, ebv))
        else:
            tree = next((p for p in parts if p.startswith("variant_")), "variant_skin")
            v_info = parse_variant_folder(tree, folder)
            entry["source_uris"]["variant"] = uri
            entry["variant_obs"].append(v_info)

    def _entry_brand(e: dict[str, Any]) -> str | None:
        if e["variant_obs"]:
            return str(e["variant_obs"][0]["brand"])
        if e["brand_obs"]:
            return str(e["brand_obs"][0]["brand"])
        for _, ebv in e["pack_obs"]:
            if ebv and ebv.get("brand"):
                return str(ebv["brand"])
        return None

    def _dims_compatible(e1: dict[str, Any], e2: dict[str, Any]) -> bool:
        w1, h1 = e1.get("width"), e1.get("height")
        w2, h2 = e2.get("width"), e2.get("height")
        if w1 is not None and h1 is not None and w2 is not None and h2 is not None:
            return int(w1) == int(w2) and int(h1) == int(h2)
        return True

    def _emb_compatible(e1: dict[str, Any], e2: dict[str, Any]) -> bool:
        v1, v2 = e1.get("embedding"), e2.get("embedding")
        if v1 and v2 and len(v1) == len(v2):
            import math

            n1 = math.sqrt(sum(float(x) ** 2 for x in v1)) or 1.0
            n2 = math.sqrt(sum(float(x) ** 2 for x in v2)) or 1.0
            cos = sum(float(a) * float(b) for a, b in zip(v1, v2, strict=True)) / (n1 * n2)
            return cos >= 0.985
        return True

    # Tier 3: Link re-encoded JPEG groups sharing (session_id, crop_idx) when brand, dimensions, and optional embedding agree
    sess_crop_index: dict[tuple[str, int], list[str]] = defaultdict(list)
    for md5, entry in by_md5.items():
        sid = str(entry["session_id"])
        if sid.isdigit():
            sess_crop_index[(sid, int(entry["crop_idx"]))].append(md5)

    merged_into: set[str] = set()
    tier3_merges = 0
    for (_, _), md5_list in sess_crop_index.items():
        if not (2 <= len(md5_list) <= 3):
            continue
        entries = [by_md5[m] for m in md5_list]
        var_entries = [e for e in entries if e["variant_obs"]]
        other_entries = [e for e in entries if not e["variant_obs"]]
        if len(var_entries) != 1 or not other_entries:
            continue
        target = var_entries[0]
        t_brand = _entry_brand(target)
        if not t_brand:
            continue
        for src in other_entries:
            s_brand = _entry_brand(src)
            if (
                s_brand
                and s_brand.lower() == t_brand.lower()
                and _dims_compatible(target, src)
                and _emb_compatible(target, src)
            ):
                target["source_uris"].update(src["source_uris"])
                target["brand_obs"].extend(src["brand_obs"])
                target["pack_obs"].extend(src["pack_obs"])
                if src["bbox"] and not target["bbox"]:
                    target["bbox"] = src["bbox"]
                    target["is_planogram_render"] = False
                target["is_tier3_merged"] = True
                merged_into.add(src["md5"])
                tier3_merges += 1

    golden_items: list[dict[str, Any]] = []
    # Key by (brand, base_variant) so generic variant names (e.g. 'Dust', 'Natural_Care', 'Others')
    # never collide across brands (e.g. 3_Roses__Dust vs Red_Label__Dust).
    packs_by_base_variant: dict[tuple[str, str], set[str]] = defaultdict(set)
    stats = {
        "total_input_records": len(records),
        "dropped_synthetic_copies": dropped_copies,
        "unique_golden_records": 0,
        "joined_3way_brand_variant_pack": 0,
        "joined_2way_brand_variant": 0,
        "joined_2way_pack_variant": 0,
        "joined_2way_pack_brand": 0,
        "competitor_pack_joins": 0,
        "hul_brand_only_pack_joins": 0,
        "compatible_twin_unions": 0,
        "tier3_reencoded_session_merges": tier3_merges,
        "multimodal_pack_base_variants": 0,
        "planogram_renders_pinned_to_gallery": 0,
        "tail_classes_promoted_to_gallery": 0,
        "quarantined_conflicts": 0,
    }

    # Pre-compute variant-folder MD5 consensus from exact pack_data joins so all crops in an SKU folder share consistent pack/category
    from collections import Counter

    vfolder_pack_counts: dict[str, Counter[str]] = defaultdict(Counter)
    vfolder_cat_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for md5, entry in by_md5.items():
        if md5 in merged_into or not entry["variant_obs"] or not entry["pack_obs"]:
            continue
        vf_key = str(entry["variant_obs"][0]["raw_folder"])
        p0_info = entry["pack_obs"][0][0]
        if p0_info["packaging_type"] != "Unknown":
            vfolder_pack_counts[vf_key][p0_info["packaging_type"]] += 1
        if p0_info["category"] != "Unknown":
            vfolder_cat_counts[vf_key][p0_info["category"]] += 1

    vfolder_consensus_pack: dict[str, str] = {}
    for vf_key, cnts in vfolder_pack_counts.items():
        top_p, top_n = cnts.most_common(1)[0]
        if top_n / sum(cnts.values()) >= 0.80:
            vfolder_consensus_pack[vf_key] = top_p

    vfolder_consensus_cat: dict[str, str] = {}
    for vf_key, cnts in vfolder_cat_counts.items():
        top_c, top_n = cnts.most_common(1)[0]
        if top_n / sum(cnts.values()) >= 0.80:
            vfolder_consensus_cat[vf_key] = top_c

    for md5, entry in sorted(by_md5.items()):
        if md5 in merged_into:
            continue
        v_list = entry["variant_obs"]
        b_list = entry["brand_obs"]
        p_list = entry["pack_obs"]
        if not v_list and not b_list and not p_list:
            continue

        conflict_reasons: list[str] = []
        brand = None
        category = "Unknown"
        raw_silo_category = None
        category_source = "unknown"
        variant = None
        base_variant = None
        pkg_type = "Unknown"
        pkg_source = "unknown"
        is_hul = False
        has_promo = False
        promo_type = None
        has_price_tag = False
        artwork_gen = "standard"
        had_twin_union = False

        if v_list:
            v0 = v_list[0]
            vf_key = str(v0["raw_folder"])
            brand = v0["brand"]
            category = v0["category"]
            raw_silo_category = v0.get("raw_silo_category")
            category_source = "variant_folder"
            if vf_key in vfolder_consensus_cat and category != vfolder_consensus_cat[vf_key]:
                category = vfolder_consensus_cat[vf_key]
                category_source = "pack_folder_consensus"
            variant = v0["variant"]
            base_variant = v0["base_variant"]
            pkg_type = v0["packaging_type"]
            if pkg_type != "Unknown":
                pkg_source = "variant_suffix"
            elif vf_key in vfolder_consensus_pack:
                pkg_type = vfolder_consensus_pack[vf_key]
                pkg_source = "pack_folder_consensus"
            is_hul = v0["is_hul"]
            has_promo = v0["has_promo"]
            promo_type = v0["promo_type"]
            has_price_tag = v0["has_price_tag"]
            artwork_gen = v0["artwork_gen"]

            if len(v_list) > 1:
                v_brands = {v["brand"].lower() for v in v_list}
                if len(v_brands) > 1:
                    conflict_reasons.append(f"variant_brand_conflict:{sorted(v_brands)}")
                else:
                    base_vars = sorted({v["base_variant"] for v in v_list}, key=len)
                    if len(base_vars) > 1:
                        shortest = base_vars[0]
                        if all(_is_twin_pack_or_promo_suffix(shortest, bv) for bv in base_vars[1:]):
                            base_variant = shortest
                            stats["compatible_twin_unions"] += 1
                            had_twin_union = True
                        else:
                            conflict_reasons.append(f"base_variant_conflict:{base_vars}")
                    else:
                        stats["compatible_twin_unions"] += 1
                        had_twin_union = True
                    for v_extra in v_list[1:]:
                        if pkg_type == "Unknown" and v_extra["packaging_type"] != "Unknown":
                            pkg_type = v_extra["packaging_type"]
                            pkg_source = "variant_twin_union"
                        if not has_promo and v_extra["has_promo"]:
                            has_promo = True
                            promo_type = v_extra["promo_type"]
                        if not has_price_tag and v_extra["has_price_tag"]:
                            has_price_tag = True
                        if artwork_gen == "standard" and v_extra["artwork_gen"] != "standard":
                            artwork_gen = v_extra["artwork_gen"]

        if b_list:
            b_brands = {b["brand"].lower() for b in b_list}
            if len(b_brands) > 1:
                conflict_reasons.append(f"brand_data_multi_folder_conflict:{sorted(b_brands)}")
            b0 = b_list[0]
            if brand is None:
                brand = b0["brand"]
                category = b0["category"]
                category_source = "brand_default"
                is_hul = b0["is_hul"]
            elif brand.lower() != b0["brand"].lower():
                conflict_reasons.append(f"cross_silo_brand_conflict:{brand}_vs_{b0['brand']}")

        if p_list:
            p_types = {p[0]["packaging_type"] for p in p_list}
            if len(p_types) > 1:
                conflict_reasons.append(f"pack_data_multi_folder_conflict:{sorted(p_types)}")
            p0, ebv0 = p_list[0]
            if pkg_type == "Unknown" or pkg_source in ("variant_suffix", "pack_folder_consensus"):
                if p0["packaging_type"] != "Unknown":
                    pkg_type = p0["packaging_type"]
                    pkg_source = "pack_folder"
            # Cross-silo brand check against pack_data embedded filename brand
            if brand is not None and ebv0 and ebv0.get("brand"):
                if brand.lower() != str(ebv0["brand"]).lower():
                    conflict_reasons.append(f"cross_silo_pack_brand_conflict:{brand}_vs_{ebv0['brand']}")
            # Pack folder ground-truth category overrides umbrella brand or folder defaults
            if p0["category"] != "Unknown":
                category = p0["category"]
                category_source = "pack_folder"
            if brand is None and ebv0:
                brand = ebv0["brand"]
                variant = ebv0["variant"]
                base_variant = ebv0["base_variant"]
                is_hul = ebv0["is_hul"]
                has_promo = ebv0["has_promo"]
                promo_type = ebv0["promo_type"]
                has_price_tag = ebv0["has_price_tag"]
            elif brand is not None and base_variant is None:
                # 2-way pack <-> brand join without variant folder: tag Competitor vs HUL Brand-Only pack
                if not is_hul:
                    variant = "Non_HUL_Competitor"
                    base_variant = "Non_HUL_Competitor"
                    stats["competitor_pack_joins"] += 1
                else:
                    variant = f"{brand}__Brand_Level_Only"
                    base_variant = f"{brand}__Brand_Level_Only"
                    stats["hul_brand_only_pack_joins"] += 1

        srcs = set(entry["source_uris"])
        if {"brand_data", "variant", "pack_data"}.issubset(srcs):
            join_mode = "3way_brand_variant_pack"
            stats["joined_3way_brand_variant_pack"] += 1
        elif {"brand_data", "variant"}.issubset(srcs):
            join_mode = "2way_brand_variant"
            stats["joined_2way_brand_variant"] += 1
        elif {"pack_data", "variant"}.issubset(srcs):
            join_mode = "2way_pack_variant"
            stats["joined_2way_pack_variant"] += 1
        elif {"pack_data", "brand_data"}.issubset(srcs):
            join_mode = "2way_pack_brand"
            stats["joined_2way_pack_brand"] += 1
        else:
            join_mode = f"1way_{next(iter(srcs))}"

        is_conflicting = bool(conflict_reasons)
        status = "quarantined_conflict" if is_conflicting else "clean"
        if is_conflicting:
            stats["quarantined_conflicts"] += 1

        if entry.get("is_tier3_merged"):
            join_tier = "tier3_reencoded_session"
            join_confidence = 0.0 if is_conflicting else 0.985
        elif had_twin_union:
            join_tier = "tier2_twin_union"
            join_confidence = 0.0 if is_conflicting else 0.99
        elif len(srcs) >= 2:
            join_tier = "tier1_exact_md5"
            join_confidence = 0.0 if is_conflicting else 1.0
        else:
            join_tier = "tier4_single_silo"
            join_confidence = 0.0 if is_conflicting else 0.95

        if status == "clean" and base_variant and pkg_type != "Unknown":
            packs_by_base_variant[(str(brand or ""), base_variant)].add(pkg_type)

        # Prefix basepack_id with brand so generic base_variant names ('Dust', 'Others', '3x')
        # never collide across brands in Cloud SQL or contrastive triplet mining.
        bv_with_brand = (
            f"{brand}__{base_variant}"
            if brand and base_variant and not base_variant.lower().startswith(f"{brand.lower()}__")
            else base_variant
        )
        basepack_id = (
            f"{bv_with_brand}__{pkg_type}"
            if bv_with_brand and pkg_type != "Unknown"
            else bv_with_brand
        )

        # Pin studio/planogram renders (HUL-* without shelf bbox) to gallery so val/test only score real store crops
        if entry["is_planogram_render"]:
            split = "gallery"
            stats["planogram_renders_pinned_to_gallery"] += 1
        else:
            split = session_split_bucket(str(entry["session_id"]), seed=seed)

        brand_data_uri = entry["source_uris"].get("brand_data")
        variant_uri = entry["source_uris"].get("variant")
        pack_data_uri = entry["source_uris"].get("pack_data")
        canonical_uri = variant_uri or brand_data_uri or pack_data_uri

        golden_items.append({
            "md5": md5,
            "canonical_uri": canonical_uri,
            "brand_data_uri": brand_data_uri,
            "variant_uri": variant_uri,
            "pack_data_uri": pack_data_uri,
            "source_uris": entry["source_uris"],
            "join_mode": join_mode,
            "join_tier": join_tier,
            "join_confidence": join_confidence,
            "is_conflicting_md5": is_conflicting,
            "status": status,
            "conflict_reasons": conflict_reasons,
            "session_id": entry["session_id"],
            "crop_idx": entry["crop_idx"],
            "bbox": list(entry["bbox"]) if entry["bbox"] else None,
            "width": int(entry["width"]) if entry.get("width") else 256,
            "height": int(entry["height"]) if entry.get("height") else 256,
            "is_planogram_render": entry["is_planogram_render"],
            "split": split,
            "labels": {
                "is_hul": is_hul,
                "category": category,
                "raw_silo_category": raw_silo_category,
                "category_source": category_source,
                "brand": brand,
                "variant": variant,
                "base_variant": base_variant,
                "basepack_id": basepack_id,
                "packaging_type": pkg_type,
                "packaging_type_source": pkg_source,
                "has_promo": has_promo,
                "promo_type": promo_type,
                "has_price_tag": has_price_tag,
                "artwork_gen": artwork_gen,
            },
        })

    # Edge-case safeguard: ensure every clean (brand, base_variant) pair has >= 1 gallery anchor.
    # Promote the entire store session atomically across all items so zero session leakage occurs.
    by_bv: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in golden_items:
        if item["status"] == "clean" and item["labels"]["base_variant"]:
            by_bv[(str(item["labels"].get("brand") or ""), str(item["labels"]["base_variant"]))].append(item)
    promoted_sessions: set[str] = set()
    for _, bv_items in sorted(by_bv.items()):
        if not any(it["split"] == "gallery" or it["session_id"] in promoted_sessions for it in bv_items):
            target_sess = str(bv_items[0]["session_id"])
            promoted_sessions.add(target_sess)
            stats["tail_classes_promoted_to_gallery"] += 1
    if promoted_sessions:
        for item in golden_items:
            if str(item["session_id"]) in promoted_sessions:
                item["split"] = "gallery"

    stats["multimodal_pack_base_variants"] = sum(
        1 for pset in packs_by_base_variant.values() if len(pset) > 1
    )
    stats["unique_golden_records"] = len(golden_items)
    return {"stats": stats, "items": golden_items}


def export_golden_rpc_manifest(
    payload: dict[str, Any],
    refs_per_class: int = 4,
    max_eval_per_split: int = 100,
) -> dict[str, Any]:
    """Build an ``rpc.json`` benchmark manifest from a Golden JSON payload for ``hul_golden`` evaluation."""
    by_cls_split: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    class_meta: dict[str, dict[str, Any]] = {}

    for it in payload.get("items", []):
        if it.get("status") != "clean":
            continue
        lab = it.get("labels", {})
        b = lab.get("brand")
        bv = lab.get("base_variant") or b
        if not b or not bv:
            continue
        cls_key = f"{b}__{bv}" if bv != b else str(b)
        class_meta[cls_key] = {
            "product": cls_key,
            "brand": b,
            "variant": lab.get("variant") or bv,
            "base_variant": bv,
            "basepack_id": lab.get("basepack_id") or bv,
            "category": lab.get("category") or "Unknown",
            "packaging_type": lab.get("packaging_type") or "Unknown",
            "is_hul": bool(lab.get("is_hul", False)),
            "has_promo": bool(lab.get("has_promo", False)),
            "promo_type": lab.get("promo_type"),
        }
        by_cls_split[cls_key][str(it.get("split", "gallery"))].append(it)

    sorted_keys = sorted(class_meta)
    classes: dict[str, dict[str, Any]] = {}
    gallery: dict[str, list[str]] = {}
    splits: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}

    for idx, cls_key in enumerate(sorted_keys, 1):
        cid = str(idx)
        classes[cid] = {"sku_id": idx, **class_meta[cls_key]}
        gal_pool = sorted(by_cls_split[cls_key]["gallery"], key=lambda x: str(x["canonical_uri"]))
        if not gal_pool:
            fallback = by_cls_split[cls_key]["val"] + by_cls_split[cls_key]["test"]
            if fallback:
                gal_pool = [fallback[0]]
        step = max(1, len(gal_pool) // max(1, refs_per_class))
        gallery[cid] = [
            str(gal_pool[i]["canonical_uri"])
            for i in range(0, len(gal_pool), step)[:refs_per_class]
        ]
        for sp_name in ("val", "test"):
            for item in sorted(by_cls_split[cls_key][sp_name], key=lambda x: str(x["canonical_uri"])):
                w = int(item.get("width", 256))
                h = int(item.get("height", 256))
                splits[sp_name].append({
                    "image": f"{sp_name}_{len(splits[sp_name]):05d}.jpg",
                    "uri": str(item["canonical_uri"]),
                    "brand_data_uri": item.get("brand_data_uri"),
                    "variant_uri": item.get("variant_uri"),
                    "pack_data_uri": item.get("pack_data_uri"),
                    "session_id": str(item.get("session_id", "")),
                    "width": w,
                    "height": h,
                    "boxes": [[0.0, 0.0, float(w), float(h)]],
                    "products": [idx],
                })

    for sp_name in ("val", "test"):
        if max_eval_per_split > 0 and len(splits[sp_name]) > max_eval_per_split:
            step = max(1, len(splits[sp_name]) // max_eval_per_split)
            splits[sp_name] = [
                {**r, "image": f"{sp_name}_{i:05d}.jpg"}
                for i, r in enumerate(splits[sp_name][::step][:max_eval_per_split])
            ]

    return {
        "classes": classes,
        "gallery": gallery,
        "splits": splits,
        "stats": {
            "mode": "hul_golden",
            "total_classes": len(classes),
            "val_samples": len(splits["val"]),
            "test_samples": len(splits["test"]),
        },
    }


def export_golden_sft_jsonl(
    payload: dict[str, Any],
    output_dir: str = "data/hul_golden/sft",
) -> dict[str, Any]:
    """Translate a Golden JSON payload into zero-leakage Vertex AI SFT and Contrastive JSONL files.

    Strictly enforces:
      - Only ``status == "clean"`` and ``is_planogram_render == False`` real store crops are exported.
      - ``split == "gallery"`` -> ``sft_multitask_train.jsonl`` and ``contrastive_triplets_train.jsonl``.
      - ``split == "val"``     -> ``sft_multitask_val.jsonl``.
      - ``split == "test"``    -> NEVER exported to any training or validation JSONL file.
    """
    import json
    from pathlib import Path

    norm_dir = str(output_dir).strip().rstrip("/")
    for protected in PROTECTED_RAW_PREFIXES:
        if norm_dir == protected or norm_dir.startswith(protected + "/") or norm_dir.endswith("/" + protected):
            raise ValueError(f"Refusing to write SFT JSONL into protected path: {output_dir}")

    out_path = Path(norm_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prompt = (
        "Classify this retail shelf crop into its canonical Unilever hierarchy JSON: "
        "is_hul, category, brand, base_variant, basepack_id, packaging_type, has_promo, promo_type."
    )
    train_lines: list[str] = []
    val_lines: list[str] = []
    gallery_by_basepack: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    gallery_by_brand_pack: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for it in payload.get("items", []):
        if it.get("status") != "clean" or it.get("is_planogram_render"):
            continue
        sp = it.get("split")
        if sp not in ("gallery", "val"):
            continue
        lab = it.get("labels", {})
        target = {
            "is_hul": lab.get("is_hul"),
            "category": lab.get("category"),
            "brand": lab.get("brand"),
            "base_variant": lab.get("base_variant"),
            "basepack_id": lab.get("basepack_id"),
            "packaging_type": lab.get("packaging_type"),
            "has_promo": lab.get("has_promo"),
            "promo_type": lab.get("promo_type"),
        }
        sft_row = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"fileData": {"mimeType": "image/jpeg", "fileUri": it["canonical_uri"]}},
                        {"text": prompt},
                    ],
                },
                {
                    "role": "model",
                    "parts": [{"text": json.dumps(target, separators=(",", ":"))}],
                },
            ]
        }
        if sp == "gallery":
            train_lines.append(json.dumps(sft_row))
            bp = lab.get("basepack_id")
            br = lab.get("brand")
            pk = lab.get("packaging_type")
            if bp:
                gallery_by_basepack[(str(br or ""), str(bp))].append(it)
            if br and pk:
                gallery_by_brand_pack[(str(br), str(pk))].append(it)
        elif sp == "val":
            val_lines.append(json.dumps(sft_row))

    triplet_lines: list[str] = []
    for (br_key, bp), items in sorted(gallery_by_basepack.items()):
        for anchor in items:
            a_sess = anchor.get("session_id")
            positives = [x for x in items if x.get("session_id") != a_sess and x["md5"] != anchor["md5"]]
            if not positives:
                positives = [x for x in items if x["md5"] != anchor["md5"]]
            if not positives:
                continue
            pos = positives[0]
            lab = anchor["labels"]
            br = str(lab.get("brand") or br_key)
            pk = str(lab.get("packaging_type") or "")
            siblings = [
                x for x in gallery_by_brand_pack.get((br, pk), [])
                if x["labels"].get("basepack_id") != bp
            ]
            if not siblings:
                siblings = [
                    x for (other_br, other_bp), other_items in gallery_by_basepack.items()
                    if (other_br, other_bp) != (br_key, bp)
                    for x in other_items[:1]
                ]
            if not siblings:
                continue
            neg = siblings[0]
            triplet_lines.append(
                json.dumps({
                    "anchor_uri": anchor["canonical_uri"],
                    "positive_uri": pos["canonical_uri"],
                    "hard_negative_uri": neg["canonical_uri"],
                    "anchor_session_id": a_sess,
                    "positive_session_id": pos.get("session_id"),
                    "basepack_id": bp,
                    "brand": br,
                    "packaging_type": pk,
                })
            )

    train_path = out_path / "sft_multitask_train.jsonl"
    val_path = out_path / "sft_multitask_val.jsonl"
    triplet_path = out_path / "contrastive_triplets_train.jsonl"
    train_path.write_text("\n".join(train_lines) + ("\n" if train_lines else ""))
    val_path.write_text("\n".join(val_lines) + ("\n" if val_lines else ""))
    triplet_path.write_text("\n".join(triplet_lines) + ("\n" if triplet_lines else ""))
    return {
        "sft_multitask_train_path": str(train_path),
        "sft_multitask_val_path": str(val_path),
        "contrastive_triplets_train_path": str(triplet_path),
        "train_count": len(train_lines),
        "val_count": len(val_lines),
        "triplet_count": len(triplet_lines),
    }


PROTECTED_RAW_PREFIXES = (
    "gs://marketshare-datasets/brand_data",
    "gs://marketshare-datasets/input_data",
    "gs://marketshare-datasets/pack_data",
    "data/products",
    "data/rpc",
    "data/sku110k",
)


def save_golden_json_payload(payload: dict[str, Any], output_dir: str = "data/hul_golden") -> dict[str, str]:
    """Write the Golden JSON payload, RPC manifest, and quarantined conflict manifest to an isolated directory.

    Enforces a strict non-destructive guard: raises ``ValueError`` if ``output_dir`` points to
    any raw GCS prefix or existing baseline dataset directory.
    """
    import json
    from pathlib import Path

    norm_dir = str(output_dir).strip().rstrip("/")
    for protected in PROTECTED_RAW_PREFIXES:
        if norm_dir == protected or norm_dir.startswith(protected + "/") or norm_dir.endswith("/" + protected):
            raise ValueError(
                f"Refusing to write Golden JSON payload into protected raw/baseline path: {output_dir}"
            )

    out_path = Path(norm_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    golden_file = out_path / "golden_payload.json"
    quarantine_file = out_path / "quarantined_conflicts.json"
    rpc_file = out_path / "rpc.json"

    quarantined = [it for it in payload.get("items", []) if it.get("status") == "quarantined_conflict"]
    rpc_manifest = export_golden_rpc_manifest(payload)
    golden_file.write_text(json.dumps(payload, indent=2))
    quarantine_file.write_text(
        json.dumps({"count": len(quarantined), "items": quarantined}, indent=2)
    )
    rpc_file.write_text(json.dumps(rpc_manifest, indent=2))
    return {
        "golden_payload_path": str(golden_file),
        "quarantined_conflicts_path": str(quarantine_file),
        "rpc_manifest_path": str(rpc_file),
    }



