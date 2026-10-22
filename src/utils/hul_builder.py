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
    GTMERCH_BAY_BRAND_RULES as _GTMERCH_FILENAME_BRAND_RULES,
)
from utils.hul_taxonomy import (
    PKG_SUFFIX_RE,
    STRIP_PROMO_ART_SUFFIX_RE,
    normalize_brand_folder,
    parse_crop_filename,
    parse_gtmerch_bay_brand,
    parse_gtmerch_filename,
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

    # Tier 3: Link re-encoded JPEG groups sharing (session_id, crop_idx) when brand, dimensions, and optional embedding agree.
    # Also promotes catch-all (<Brand>__Others / <Brand>__<Brand>) re-encoded duplicates into specific sibling variants.
    sess_crop_index: dict[tuple[str, int], list[str]] = defaultdict(list)
    for md5, entry in by_md5.items():
        sid = str(entry["session_id"])
        if sid.isdigit():
            sess_crop_index[(sid, int(entry["crop_idx"]))].append(md5)

    merged_into: set[str] = set()
    tier3_merges = 0
    catchall_promoted_by_session = 0
    for (_, _), md5_list in sess_crop_index.items():
        if not (2 <= len(md5_list) <= 3):
            continue
        entries = [by_md5[m] for m in md5_list]
        var_entries = [e for e in entries if e["variant_obs"]]
        other_entries = [e for e in entries if not e["variant_obs"]]
        if len(var_entries) == 2 and not other_entries:
            # Check if one is specific and the other is catch-all (<Brand>__Others / <Brand>__<Brand>)
            spec = [e for e in var_entries if any(not v.get("is_catchall_variant") for v in e["variant_obs"])]
            catch = [e for e in var_entries if all(v.get("is_catchall_variant") for v in e["variant_obs"])]
            if len(spec) == 1 and len(catch) == 1:
                var_entries = spec
                other_entries = catch
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
                if src["variant_obs"]:
                    catchall_promoted_by_session += 1
                    target["is_catchall_session_promoted"] = True
                for k_src, u_src in src["source_uris"].items():
                    if k_src not in target["source_uris"]:
                        target["source_uris"][k_src] = u_src
                target["brand_obs"].extend(src["brand_obs"])
                target["variant_obs"].extend(src["variant_obs"])
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
        "catchall_promoted_by_md5": 0,
        "catchall_promoted_by_session": catchall_promoted_by_session,
        "catchall_promoted_by_prototype": 0,
        "tier3_reencoded_session_merges": tier3_merges,
        "multimodal_pack_base_variants": 0,
        "multimodal_subcluster_resolved": 0,
        "planogram_renders_pinned_to_gallery": 0,
        "tail_classes_promoted_to_gallery": 0,
        "quarantined_conflicts": 0,
    }

    # Pre-compute variant-folder MD5 consensus AND multi-modal (<80%) sub-cluster prototypes from exact pack_data joins
    import math
    from collections import Counter

    vfolder_pack_counts: dict[str, Counter[str]] = defaultdict(Counter)
    vfolder_cat_counts: dict[str, Counter[str]] = defaultdict(Counter)
    vfolder_pack_embs: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    vfolder_pack_log_aspect: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def _v_sort_key(v: dict[str, Any]) -> tuple[int, int, int, str]:
        return (
            1 if v.get("is_catchall_variant") else 0,
            1 if v.get("packaging_type") == "Unknown" else 0,
            len(str(v.get("base_variant") or "")),
            str(v.get("raw_folder") or ""),
        )

    for md5, entry in by_md5.items():
        if md5 in merged_into or not entry["variant_obs"] or not entry["pack_obs"]:
            continue
        v_sorted = sorted(entry["variant_obs"], key=_v_sort_key)
        vf_key = str(v_sorted[0]["raw_folder"])
        p0_info = sorted(entry["pack_obs"], key=lambda p: str(p[0]["raw_folder"]))[0][0]
        pk = p0_info["packaging_type"]
        if pk != "Unknown":
            vfolder_pack_counts[vf_key][pk] += 1
            if entry.get("embedding"):
                vfolder_pack_embs[vf_key][pk].append([float(x) for x in entry["embedding"]])
            w = float(entry.get("width") or 256.0)
            h = float(entry.get("height") or 256.0)
            if w > 0 and h > 0:
                vfolder_pack_log_aspect[vf_key][pk].append(math.log(h / w))
        if p0_info["category"] != "Unknown":
            vfolder_cat_counts[vf_key][p0_info["category"]] += 1

    vfolder_consensus_pack: dict[str, str] = {}
    vfolder_allowed_packs: dict[str, list[str]] = {}
    vfolder_pack_subclusters: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for vf_key, cnts in vfolder_pack_counts.items():
        top_p, top_n = cnts.most_common(1)[0]
        total_n = sum(cnts.values())
        if top_n / total_n >= 0.80:
            vfolder_consensus_pack[vf_key] = top_p
        else:
            vfolder_allowed_packs[vf_key] = sorted(cnts.keys())
            for pk in vfolder_allowed_packs[vf_key]:
                embs = vfolder_pack_embs[vf_key].get(pk, [])
                centroid: list[float] | None = None
                if embs:
                    dim = len(embs[0])
                    mean_v = [sum(vec[d] for vec in embs) / len(embs) for d in range(dim)]
                    norm = math.sqrt(sum(x * x for x in mean_v)) or 1.0
                    centroid = [x / norm for x in mean_v]
                aspects = vfolder_pack_log_aspect[vf_key].get(pk, [])
                mean_log_ar = sum(aspects) / len(aspects) if aspects else None
                vfolder_pack_subclusters[vf_key][pk] = {
                    "prior": cnts[pk] / total_n,
                    "centroid": centroid,
                    "mean_log_ar": mean_log_ar,
                }

    vfolder_consensus_cat: dict[str, str] = {}
    for vf_key, cnts in vfolder_cat_counts.items():
        top_c, top_n = cnts.most_common(1)[0]
        if top_n / sum(cnts.values()) >= 0.80:
            vfolder_consensus_cat[vf_key] = top_c

    def _resolve_multimodal_pack(vf_key: str, entry_rec: dict[str, Any]) -> tuple[str, str]:
        """Resolve <80% multi-modal SKU packaging via embedding sub-centroid + aspect-ratio likelihood."""
        sub = vfolder_pack_subclusters.get(vf_key, {})
        if not sub:
            return "Unknown", "unknown"
        emb = entry_rec.get("embedding")
        w = float(entry_rec.get("width") or 0.0)
        h = float(entry_rec.get("height") or 0.0)
        log_ar = math.log(h / w) if (w > 0 and h > 0 and (int(w) != 256 or int(h) != 256)) else None
        has_signal = False
        best_pk = max(sub, key=lambda k: sub[k]["prior"])
        best_score = -1e9
        for pk, info in sorted(sub.items()):
            score = 0.15 * float(info["prior"])
            c_vec = info["centroid"]
            if emb and c_vec and len(emb) == len(c_vec):
                n_e = math.sqrt(sum(float(x) ** 2 for x in emb)) or 1.0
                cos = sum(float(a) * float(b) for a, b in zip(emb, c_vec, strict=True)) / n_e
                score += 0.70 * cos
                has_signal = True
            if log_ar is not None and info["mean_log_ar"] is not None:
                ar_sim = math.exp(-2.0 * abs(log_ar - float(info["mean_log_ar"])))
                score += 0.30 * ar_sim
                has_signal = True
            if score > best_score:
                best_score = score
                best_pk = pk
        return best_pk, ("multimodal_subcluster" if has_signal else "multimodal_allowed_set")

    for md5, entry in sorted(by_md5.items()):
        if md5 in merged_into:
            continue
        v_list = sorted(entry["variant_obs"], key=_v_sort_key)
        b_list = sorted(entry["brand_obs"], key=lambda b: str(b["raw_folder"]))
        p_list = sorted(entry["pack_obs"], key=lambda p: str(p[0]["raw_folder"]))
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
        allowed_pkg_types: list[str] = []
        is_hul = False
        has_promo = False
        promo_type = None
        has_price_tag = False
        artwork_gen = "standard"
        is_catchall_variant = False
        variant_enrichment_source = "explicit_folder"
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
            is_catchall_variant = bool(v0.get("is_catchall_variant", False))
            if is_catchall_variant:
                variant_enrichment_source = "open_set_catchall"
            elif entry.get("is_catchall_session_promoted"):
                variant_enrichment_source = "session_specific_sibling"
            from utils.taxonomy.packaging import PKG_SUFFIX_RE
            pkg_type = v0["packaging_type"]
            has_explicit_suffix = bool(PKG_SUFFIX_RE.search(str(v0.get("variant") or "")))
            if vf_key in vfolder_allowed_packs:
                allowed_pkg_types = list(vfolder_allowed_packs[vf_key])
            if has_explicit_suffix and pkg_type != "Unknown":
                pkg_source = "variant_suffix"
            elif vf_key in vfolder_consensus_pack:
                pkg_type = vfolder_consensus_pack[vf_key]
                pkg_source = "pack_folder_consensus"
            elif vf_key in vfolder_allowed_packs:
                mm_pkg, pkg_source = _resolve_multimodal_pack(vf_key, entry)
                if mm_pkg != "Unknown":
                    pkg_type = mm_pkg
                if pkg_source == "multimodal_subcluster":
                    stats["multimodal_subcluster_resolved"] += 1
            elif pkg_type != "Unknown":
                pkg_source = "taxonomy_prior"
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
                    specific_obs = [v for v in v_list if not v.get("is_catchall_variant")]
                    catchall_obs = [v for v in v_list if v.get("is_catchall_variant")]
                    eval_obs = specific_obs if specific_obs else v_list
                    if specific_obs and catchall_obs:
                        stats["catchall_promoted_by_md5"] += 1
                        variant_enrichment_source = "md5_specific_sibling"
                        had_twin_union = True
                    base_vars = sorted({v["base_variant"] for v in eval_obs}, key=len)
                    if len(base_vars) > 1:
                        shortest = base_vars[0]
                        if all(_is_twin_pack_or_promo_suffix(shortest, bv) for bv in base_vars[1:]):
                            base_variant = shortest
                            stats["compatible_twin_unions"] += 1
                            had_twin_union = True
                        else:
                            conflict_reasons.append(f"base_variant_conflict:{base_vars}")
                    elif not (specific_obs and catchall_obs):
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
            if pkg_type == "Unknown" or pkg_source in (
                "variant_suffix",
                "taxonomy_prior",
                "pack_folder_consensus",
                "multimodal_subcluster",
                "multimodal_allowed_set",
            ):
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

        if (
            is_catchall_variant
            and pkg_type != "Unknown"
            and pkg_source not in ("unknown", "taxonomy_prior")
            and variant_enrichment_source == "open_set_catchall"
        ):
            variant_enrichment_source = "pack_container_enriched"

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
        if not allowed_pkg_types and pkg_type != "Unknown":
            allowed_pkg_types = [pkg_type]

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

        item_dict: dict[str, Any] = {
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
                "allowed_packaging_types": allowed_pkg_types,
                "is_catchall_variant": is_catchall_variant,
                "variant_enrichment_source": variant_enrichment_source,
                "has_promo": has_promo,
                "promo_type": promo_type,
                "has_price_tag": has_price_tag,
                "artwork_gen": artwork_gen,
            },
        }
        if entry.get("embedding"):
            item_dict["embedding"] = entry["embedding"]
        golden_items.append(item_dict)

    # Tier-2 Session-Sibling Catch-All Promotion: when a store session has crops of brand B where
    # all specific crops in that session belong to a single base_variant, promote catch-all sibling
    # crops of brand B in the same session to that specific base_variant.
    sess_brand_specific: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in golden_items:
        sid = str(item.get("session_id") or "")
        lab = item["labels"]
        b = str(lab.get("brand") or "")
        bv = str(lab.get("base_variant") or "")
        if (
            item["status"] == "clean"
            and sid
            and not sid.startswith("single_")
            and b
            and bv
            and not lab.get("is_catchall_variant")
            and bv not in ("Non_HUL_Competitor", f"{b}__Brand_Level_Only")
        ):
            sess_brand_specific[(sid, b)].add(bv)

    for item in golden_items:
        sid = str(item.get("session_id") or "")
        lab = item["labels"]
        b = str(lab.get("brand") or "")
        if item["status"] == "clean" and lab.get("is_catchall_variant") and (sid, b) in sess_brand_specific:
            spec_bvs = sess_brand_specific[(sid, b)]
            if len(spec_bvs) == 1:
                promoted_bv = next(iter(spec_bvs))
                lab["variant"] = promoted_bv
                lab["base_variant"] = promoted_bv
                lab["is_catchall_variant"] = False
                lab["variant_enrichment_source"] = "session_specific_sibling"
                pk = str(lab.get("packaging_type") or "Unknown")
                bv_with_brand = (
                    f"{b}__{promoted_bv}"
                    if not promoted_bv.lower().startswith(f"{b.lower()}__")
                    else promoted_bv
                )
                lab["basepack_id"] = f"{bv_with_brand}__{pk}" if pk != "Unknown" else bv_with_brand
                stats["catchall_promoted_by_session"] += 1

    # Backfill allowed_packaging_types across all items sharing a multi-modal (brand, base_variant)
    for item in golden_items:
        if item["status"] == "clean" and item["labels"].get("base_variant"):
            key_bv = (str(item["labels"].get("brand") or ""), str(item["labels"]["base_variant"]))
            pset = packs_by_base_variant.get(key_bv)
            if pset and len(pset) > 1:
                item["labels"]["allowed_packaging_types"] = sorted(pset)

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
    payload = {"stats": stats, "items": golden_items}
    return enrich_catchall_variants_by_prototype(payload)


def enrich_catchall_variants_by_prototype(
    payload: dict[str, Any],
    min_cosine: float = 0.90,
    min_margin: float = 0.04,
) -> dict[str, Any]:
    """Tier-3 Prototype Enrichment for ``<Brand>__Others`` and ``<Brand>__<Brand>`` catch-all folders.

    Computes L2-normalized gallery prototypes for all specific sibling variants of each brand.
    When a catch-all crop has an embedding whose cosine similarity to a specific sibling variant
    of the same brand satisfies ``cos >= min_cosine`` and ``(cos_1 - cos_2) >= min_margin``,
    promotes its ``base_variant`` and ``basepack_id`` to that specific sibling.
    """
    import math

    items = payload.get("items", [])
    by_brand_bv_embs: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    by_brand_bv_pack: dict[tuple[str, str], str] = {}

    for it in items:
        if it.get("status") != "clean" or not it.get("embedding"):
            continue
        lab = it.get("labels", {})
        if lab.get("is_catchall_variant"):
            continue
        b = str(lab.get("brand") or "")
        bv = str(lab.get("base_variant") or "")
        if not b or not bv or bv in ("Non_HUL_Competitor", f"{b}__Brand_Level_Only"):
            continue
        if it.get("split") == "gallery":
            by_brand_bv_embs[b][bv].append([float(x) for x in it["embedding"]])
            if lab.get("packaging_type") and lab["packaging_type"] != "Unknown":
                by_brand_bv_pack[(b, bv)] = str(lab["packaging_type"])

    prototypes: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for b, bv_map in by_brand_bv_embs.items():
        for bv, vecs in bv_map.items():
            if not vecs:
                continue
            dim = len(vecs[0])
            mean_v = [sum(v[d] for v in vecs) / len(vecs) for d in range(dim)]
            norm = math.sqrt(sum(x * x for x in mean_v)) or 1.0
            prototypes[b][bv] = [x / norm for x in mean_v]

    promoted = 0
    for it in items:
        if it.get("status") != "clean" or not it.get("embedding"):
            continue
        lab = it.get("labels", {})
        if not lab.get("is_catchall_variant"):
            continue
        b = str(lab.get("brand") or "")
        cand_map = prototypes.get(b)
        if not cand_map:
            continue
        emb = [float(x) for x in it["embedding"]]
        n_e = math.sqrt(sum(x * x for x in emb)) or 1.0
        emb_norm = [x / n_e for x in emb]
        scored: list[tuple[float, str]] = []
        for bv, proto in sorted(cand_map.items()):
            if len(proto) != len(emb_norm):
                continue
            cos = sum(a * c for a, c in zip(emb_norm, proto, strict=True))
            scored.append((cos, bv))
        if not scored:
            continue
        scored.sort(reverse=True)
        top1_cos, top1_bv = scored[0]
        top2_cos = scored[1][0] if len(scored) > 1 else -1.0
        if top1_cos >= min_cosine and (len(scored) == 1 or (top1_cos - top2_cos) >= min_margin):
            lab["variant"] = top1_bv
            lab["base_variant"] = top1_bv
            lab["is_catchall_variant"] = False
            lab["variant_enrichment_source"] = "sibling_prototype_cluster"
            if lab.get("packaging_type") == "Unknown" and (b, top1_bv) in by_brand_bv_pack:
                lab["packaging_type"] = by_brand_bv_pack[(b, top1_bv)]
                lab["packaging_type_source"] = "sibling_prototype_cluster"
            pk = str(lab.get("packaging_type") or "Unknown")
            bv_with_brand = (
                f"{b}__{top1_bv}"
                if not top1_bv.lower().startswith(f"{b.lower()}__")
                else top1_bv
            )
            lab["basepack_id"] = f"{bv_with_brand}__{pk}" if pk != "Unknown" else bv_with_brand
            promoted += 1

    if "stats" in payload:
        payload["stats"]["catchall_promoted_by_prototype"] = (
            int(payload["stats"].get("catchall_promoted_by_prototype", 0)) + promoted
        )
    return payload


def validate_golden_data_contract(payload: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    """Enforce hard data-engineering invariants on a Golden JSON payload before training or evaluation.

    Checks:
      1. ``unknown_category_clean_variants == 0`` (every clean variant/pack item has a known category).
      2. ``cross_split_session_leaks == 0`` (zero store ``session_id`` overlap across gallery, val, test).
      3. ``cross_split_md5_leaks == 0`` (zero ``md5`` overlap across gallery, val, test).
      4. ``eval_synthetic_or_augmented_copies == 0`` (zero synthetic or ``DA_*`` copies in val or test).
      5. ``incomplete_3way_uris == 0`` (every 3-way join has ``brand_data_uri``, ``variant_uri``, ``pack_data_uri``).
    """
    items = payload.get("items", [])
    clean = [it for it in items if it.get("status") == "clean"]

    unknown_cat_variants = [
        it["canonical_uri"]
        for it in clean
        if it.get("labels", {}).get("category", "Unknown") == "Unknown"
        and (it.get("variant_uri") or it.get("pack_data_uri") or it.get("labels", {}).get("is_hul"))
    ]

    split_sessions: dict[str, set[str]] = {"gallery": set(), "val": set(), "test": set()}
    split_md5s: dict[str, set[str]] = {"gallery": set(), "val": set(), "test": set()}
    eval_copies: list[str] = []
    incomplete_3way: list[str] = []

    for it in clean:
        sp = str(it.get("split", "gallery"))
        if sp in split_md5s and it.get("md5"):
            split_md5s[sp].add(str(it["md5"]))
        if sp in split_sessions and not it.get("is_planogram_render") and it.get("session_id"):
            split_sessions[sp].add(str(it["session_id"]))
        if sp in ("val", "test"):
            fname = str(it.get("canonical_uri") or "").rsplit("/", 1)[-1]
            f_info = parse_crop_filename(fname)
            if f_info["is_synthetic_copy"] or f_info["is_augmented_copy"]:
                eval_copies.append(str(it.get("canonical_uri")))
        if it.get("join_mode") == "3way_brand_variant_pack":
            if not (it.get("brand_data_uri") and it.get("variant_uri") and it.get("pack_data_uri")):
                incomplete_3way.append(str(it.get("md5")))

    sess_leaks = (
        (split_sessions["gallery"] & split_sessions["val"])
        | (split_sessions["gallery"] & split_sessions["test"])
        | (split_sessions["val"] & split_sessions["test"])
    )
    md5_leaks = (
        (split_md5s["gallery"] & split_md5s["val"])
        | (split_md5s["gallery"] & split_md5s["test"])
        | (split_md5s["val"] & split_md5s["test"])
    )

    report = {
        "valid": (
            len(unknown_cat_variants) == 0
            and len(sess_leaks) == 0
            and len(md5_leaks) == 0
            and len(eval_copies) == 0
            and len(incomplete_3way) == 0
        ),
        "unknown_category_clean_variants": len(unknown_cat_variants),
        "cross_split_session_leaks": len(sess_leaks),
        "cross_split_md5_leaks": len(md5_leaks),
        "eval_synthetic_or_augmented_copies": len(eval_copies),
        "incomplete_3way_uris": len(incomplete_3way),
    }
    if strict and not report["valid"]:
        raise ValueError(f"Golden data contract violation: {report}")
    return report


def export_golden_rpc_manifest(
    payload: dict[str, Any],
    refs_per_class: int = 4,
    max_eval_per_split: int = 100,
    max_train_per_split: int = 800,
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
            "allowed_packaging_types": list(lab.get("allowed_packaging_types") or []),
            "is_hul": bool(lab.get("is_hul", False)),
            "has_promo": bool(lab.get("has_promo", False)),
            "promo_type": lab.get("promo_type"),
        }
        by_cls_split[cls_key][str(it.get("split", "gallery"))].append(it)

    sorted_keys = sorted(class_meta)
    classes: dict[str, dict[str, Any]] = {}
    gallery: dict[str, list[str]] = {}
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for idx, cls_key in enumerate(sorted_keys, 1):
        cid = str(idx)
        classes[cid] = {"sku_id": idx, **class_meta[cls_key]}
        gal_pool = sorted(by_cls_split[cls_key]["gallery"], key=lambda x: str(x["canonical_uri"]))
        if not gal_pool:
            raise RuntimeError(
                f"Zero-leakage invariant violated: class {cls_key!r} (sku_id={idx}) has 0 train/gallery references; "
                "refusing to populate gallery from val or test splits."
            )
        step = max(1, len(gal_pool) // max(1, refs_per_class))
        gallery[cid] = [
            str(gal_pool[i]["canonical_uri"])
            for i in range(0, len(gal_pool), step)[:refs_per_class]
        ]
        for sp_src, sp_name in (("gallery", "train"), ("val", "val"), ("test", "test")):
            for item in sorted(by_cls_split[cls_key][sp_src], key=lambda x: str(x["canonical_uri"])):
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

    if max_train_per_split > 0 and len(splits["train"]) > max_train_per_split:
        step = max(1, len(splits["train"]) // max_train_per_split)
        splits["train"] = [
            {**r, "image": f"train_{i:05d}.jpg"}
            for i, r in enumerate(splits["train"][::step][:max_train_per_split])
        ]
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
            "train_samples": len(splits["train"]),
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
    "gs://gtmerch-marketshare",
    "data/products",
    "data/rpc",
    "data/sku110k",
)


def save_golden_json_payload(payload: dict[str, Any], output_dir: str = "data/hul_golden") -> dict[str, str]:
    """Write the Golden JSON payload, RPC manifest, and quarantined conflict manifest to an isolated directory.

    Enforces a strict non-destructive guard: raises ``ValueError`` if ``output_dir`` points to
    any raw GCS prefix or existing baseline dataset directory, and validates the data contract.
    """
    import json
    from pathlib import Path

    norm_dir = str(output_dir).strip().rstrip("/")
    for protected in PROTECTED_RAW_PREFIXES:
        if norm_dir == protected or norm_dir.startswith(protected + "/") or norm_dir.endswith("/" + protected):
            raise ValueError(
                f"Refusing to write Golden JSON payload into protected raw/baseline path: {output_dir}"
            )

    validate_golden_data_contract(payload, strict=True)

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
    from utils.dataset import export_rpc_splits_jsonl

    jsonl_paths = export_rpc_splits_jsonl(rpc_manifest, out_path)
    return {
        "golden_payload_path": str(golden_file),
        "quarantined_conflicts_path": str(quarantine_file),
        "rpc_manifest_path": str(rpc_file),
        "train_jsonl_path": jsonl_paths["train"],
        "val_jsonl_path": jsonl_paths["val"],
        "test_jsonl_path": jsonl_paths["test"],
    }


def _box_iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _box_containment_xyxy(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """Fraction of ``inner`` box area contained within ``outer`` box."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_in = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return inter / area_in if area_in > 0 else 0.0


def _parse_gtmerch_boxes(rec: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    """Parse YOLO ``labels_txt`` or structured ``boxes`` into ``[{'class_id': 0|1, 'class_name': 'asset'|'product', 'bbox': [x1, y1, x2, y2]}]``."""
    out: list[dict[str, Any]] = []
    if rec.get("boxes") is not None:
        for bx in rec["boxes"]:
            cid = int(bx.get("class_id", 1))
            coords = [int(round(float(v))) for v in bx["bbox"]]
            out.append({
                "class_id": cid,
                "class_name": "asset" if cid == 0 else "product",
                "bbox": coords[:4],
            })
        return out

    labels_txt = rec.get("labels_txt")
    if labels_txt is None:
        return out
    for raw_line in str(labels_txt).splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = max(0, int(round((cx - bw / 2.0) * width)))
        y1 = max(0, int(round((cy - bh / 2.0) * height)))
        x2 = min(width, int(round((cx + bw / 2.0) * width)))
        y2 = min(height, int(round((cy + bh / 2.0) * height)))
        if x2 > x1 and y2 > y1:
            out.append({
                "class_id": cid,
                "class_name": "asset" if cid == 0 else "product",
                "bbox": [x1, y1, x2, y2],
            })
    return out


def audit_and_split_gtmerch_dataset(
    records: list[dict[str, Any]],
    iou_dedup_threshold: float = 0.92,
    containment_threshold: float = 0.70,
    seed: int = 42,
) -> dict[str, Any]:
    """Audit, deduplicate, and split ``gs://gtmerch-marketshare`` 2-class YOLO shelf images.

    Enforces all data quality invariants discovered during the ``gtmerch`` audit:
      1. Strips non-image artifacts (``.amlignore``, ``.amlignore.amltmp``, ``labels.cache``).
      2. Quarantines orphan images without ``.txt`` label files (e.g. the 3 orphan JPEGs in ``train/images``).
      3. Resolves ``train <-> test`` MD5 leaks (49 overlapping MD5s including 5 renamed files and 11
         Class-0 box-splitting edits) and collapses intra-train MD5 duplicates (30 duplicate pairs),
         preferring the finer multi-bay Class-0 decomposition when Class-0 counts differ.
      4. Deduplicates double-clicked boxes of the same class with ``IoU > iou_dedup_threshold`` (0.92).
      5. Computes ``Class 1`` (``product``) spatial containment inside ``Class 0`` (``asset``) display bays.
      6. Groups splits by ``outlet_id`` / ``session_id`` so zero store spans both train and val/test.
    """
    stats = {
        "total_input_records": len(records),
        "dropped_artifact_files": 0,
        "quarantined_orphan_images": 0,
        "train_test_md5_leaks_resolved": 0,
        "intra_train_md5_duplicates_collapsed": 0,
        "renamed_md5_duplicates": 0,
        "class0_box_split_edits_resolved": 0,
        "deduped_double_click_boxes": 0,
        "class0_total_count": 0,
        "class1_total_count": 0,
        "class1_inside_class0_count": 0,
        "clean_shelf_images": 0,
    }

    quarantined_orphans: list[dict[str, Any]] = []
    by_md5: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rec in records:
        uri = str(rec.get("uri") or "")
        meta = parse_gtmerch_filename(uri)
        if meta["is_artifact_file"]:
            stats["dropped_artifact_files"] += 1
            continue
        has_lbl = rec.get("has_label_file")
        if has_lbl is False or (has_lbl is None and rec.get("labels_txt") is None and rec.get("boxes") is None):
            stats["quarantined_orphan_images"] += 1
            quarantined_orphans.append({
                "uri": uri,
                "md5": rec.get("md5"),
                "status": "quarantined_orphan",
                "outlet_id": meta["outlet_id"],
                "session_id": meta["session_id"],
            })
            continue
        w = int(rec.get("width") or 1200)
        h = int(rec.get("height") or 1600)
        parsed_boxes = _parse_gtmerch_boxes(rec, w, h)
        by_md5[str(rec.get("md5") or uri)].append({
            "uri": uri,
            "md5": str(rec.get("md5") or uri),
            "raw_split": str(rec.get("split") or ("test" if "/test/" in uri else "train")),
            "width": w,
            "height": h,
            "meta": meta,
            "boxes": parsed_boxes,
        })

    canonical_records: list[dict[str, Any]] = []
    test_pinned_outlets: set[str] = set()

    for _md5, group in sorted(by_md5.items()):
        splits_present = {g["raw_split"] for g in group}
        stems_present = {g["meta"]["stem"] for g in group}
        c0_counts = {sum(1 for b in g["boxes"] if b["class_id"] == 0) for g in group}

        if len(group) > 1:
            if "train" in splits_present and "test" in splits_present:
                stats["train_test_md5_leaks_resolved"] += 1
            else:
                stats["intra_train_md5_duplicates_collapsed"] += len(group) - 1
            if len(stems_present) > 1:
                stats["renamed_md5_duplicates"] += 1
            if len(c0_counts) > 1:
                stats["class0_box_split_edits_resolved"] += 1

        # Prefer the annotation with more granular Class 0 display bays (then non-copy filename, then test URI)
        group_sorted = sorted(
            group,
            key=lambda g: (
                -sum(1 for b in g["boxes"] if b["class_id"] == 0),
                -len(g["boxes"]),
                1 if g["meta"]["is_copy_variant"] else 0,
                0 if g["raw_split"] == "test" else 1,
                g["uri"],
            ),
        )
        chosen = group_sorted[0]
        if "test" in splits_present and chosen["meta"]["outlet_id"]:
            test_pinned_outlets.add(str(chosen["meta"]["outlet_id"]))
        canonical_records.append(chosen)

    items: list[dict[str, Any]] = []
    for chosen in canonical_records:
        meta = chosen["meta"]
        outlet_id = str(meta["outlet_id"] or meta["session_id"] or "unknown_outlet")
        session_id = str(meta["session_id"] or outlet_id)
        if outlet_id in test_pinned_outlets or chosen["raw_split"] == "test":
            split = "test"
        else:
            bkt = session_split_bucket(outlet_id, seed=seed)
            split = "train" if bkt == "gallery" else bkt

        # Deduplicate double-clicked boxes of the same class with IoU > iou_dedup_threshold
        deduped_boxes: list[dict[str, Any]] = []
        for bx in chosen["boxes"]:
            t_coords = (bx["bbox"][0], bx["bbox"][1], bx["bbox"][2], bx["bbox"][3])
            is_dup = False
            for kept in deduped_boxes:
                if kept["class_id"] == bx["class_id"]:
                    k_coords = (kept["bbox"][0], kept["bbox"][1], kept["bbox"][2], kept["bbox"][3])
                    if _box_iou_xyxy(t_coords, k_coords) > iou_dedup_threshold:
                        is_dup = True
                        stats["deduped_double_click_boxes"] += 1
                        break
            if not is_dup:
                deduped_boxes.append(dict(bx))

        asset_boxes = [b for b in deduped_boxes if b["class_id"] == 0]
        product_boxes = [b for b in deduped_boxes if b["class_id"] == 1]
        stats["class0_total_count"] += len(asset_boxes)
        stats["class1_total_count"] += len(product_boxes)

        for p_idx, pb in enumerate(product_boxes):
            p_coords = (pb["bbox"][0], pb["bbox"][1], pb["bbox"][2], pb["bbox"][3])
            best_a_idx: int | None = None
            best_cont = 0.0
            for a_idx, ab in enumerate(asset_boxes):
                a_coords = (ab["bbox"][0], ab["bbox"][1], ab["bbox"][2], ab["bbox"][3])
                cont = _box_containment_xyxy(p_coords, a_coords)
                if cont > best_cont:
                    best_cont = cont
                    best_a_idx = a_idx
            is_in = best_cont >= containment_threshold
            pb["product_idx"] = p_idx
            pb["inside_asset_idx"] = best_a_idx if is_in else None
            pb["asset_containment_ratio"] = round(best_cont, 4)
            pb["is_inside_asset"] = is_in
            if is_in:
                stats["class1_inside_class0_count"] += 1

        stats["clean_shelf_images"] += 1
        items.append({
            "uri": chosen["uri"],
            "md5": chosen["md5"],
            "status": "clean",
            "split": split,
            "outlet_id": outlet_id,
            "asset_id": meta["asset_id"],
            "capture_date": meta["capture_date"],
            "timestamp_ms": meta["timestamp_ms"],
            "session_id": session_id,
            "width": chosen["width"],
            "height": chosen["height"],
            "asset_boxes": asset_boxes,
            "product_boxes": product_boxes,
        })

    return {
        "stats": stats,
        "items": items,
        "quarantined_orphans": quarantined_orphans,
    }


def link_gtmerch_shelves_to_golden_crops(
    gtmerch_manifest: dict[str, Any],
    golden_payload: dict[str, Any],
    iou_match_threshold: float = 0.50,
) -> dict[str, Any]:
    """Synchronize cross-bucket store splits and link ``gtmerch`` 2D shelf boxes to ``hul_golden`` SKU crops.

    1. **Cross-Bucket Split Synchronization (`FDE-A`)**: Any ``session_id`` shared between
       ``gs://gtmerch-marketshare`` and ``gs://marketshare-datasets`` is assigned a single unified
       split across both datasets (preventing a store shelf in ``gtmerch/test`` from leaking its
       isolated crops into ``hul_golden/gallery``).
    2. **2D Shelf Box -> SKU Crop Ground-Truth Attribution**: Matches ``Class 1`` (``product``)
       boxes on ``gtmerch`` shelves against isolated crop records in ``golden_payload`` via
       ``(session_id, bbox)`` IoU or ``(session_id, crop_idx)`` and attaches full ``labels``
       (``brand``, ``base_variant``, ``basepack_id``, ``category``, ``packaging_type``, ``is_hul``).
    """
    crops_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for crop in golden_payload.get("items", []):
        if crop.get("status") == "clean" and crop.get("session_id"):
            crops_by_session[str(crop["session_id"])].append(crop)

    sync_sessions = 0
    sync_crops = 0
    linked_boxes = 0

    for shelf in gtmerch_manifest.get("items", []):
        if shelf.get("status") != "clean":
            continue
        sid = str(shelf.get("session_id") or "")
        sess_crops = crops_by_session.get(sid, [])
        if not sess_crops:
            continue
        sync_sessions += 1
        shelf_split = str(shelf.get("split") or "train")
        target_crop_split = "gallery" if shelf_split == "train" else shelf_split
        for c in sess_crops:
            if not c.get("is_planogram_render") and c.get("split") != target_crop_split:
                c["split"] = target_crop_split
                sync_crops += 1

        used_crop_md5s: set[str] = set()
        # Pass 1: Match by explicit crop bbox IoU when available
        for pb in shelf.get("product_boxes", []):
            p_coords = (pb["bbox"][0], pb["bbox"][1], pb["bbox"][2], pb["bbox"][3])
            best_crop: dict[str, Any] | None = None
            best_iou = 0.0
            for c in sess_crops:
                if c["md5"] in used_crop_md5s or not c.get("bbox"):
                    continue
                cb = c["bbox"]
                c_coords = (int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3]))
                iou = _box_iou_xyxy(p_coords, c_coords)
                if iou >= iou_match_threshold and iou > best_iou:
                    best_iou = iou
                    best_crop = c
            if best_crop is not None:
                used_crop_md5s.add(str(best_crop["md5"]))
                pb["linked_crop_md5"] = best_crop["md5"]
                pb["linked_crop_uri"] = best_crop["canonical_uri"]
                pb["link_mode"] = "bbox_iou"
                pb["labels"] = dict(best_crop["labels"])
                linked_boxes += 1

        # Pass 2: Match remaining session boxes by deterministic crop_idx order
        by_idx = {
            int(c.get("crop_idx") or -1): c
            for c in sess_crops
            if c["md5"] not in used_crop_md5s and c.get("crop_idx") is not None
        }
        for pb in shelf.get("product_boxes", []):
            if pb.get("linked_crop_md5"):
                continue
            cand = by_idx.get(int(pb.get("product_idx", -1))) or by_idx.get(int(pb.get("product_idx", -1)) + 1)
            if cand is not None and cand["md5"] not in used_crop_md5s:
                used_crop_md5s.add(str(cand["md5"]))
                pb["linked_crop_md5"] = cand["md5"]
                pb["linked_crop_uri"] = cand["canonical_uri"]
                pb["link_mode"] = "session_crop_idx"
                pb["labels"] = dict(cand["labels"])
                linked_boxes += 1

    gtmerch_manifest.setdefault("stats", {})["synchronized_cross_bucket_sessions"] = sync_sessions
    gtmerch_manifest["stats"]["synchronized_crop_splits"] = sync_crops
    gtmerch_manifest["stats"]["linked_product_boxes_to_golden_crops"] = linked_boxes
    return {
        "synchronized_cross_bucket_sessions": sync_sessions,
        "synchronized_crop_splits": sync_crops,
        "linked_product_boxes_to_golden_crops": linked_boxes,
    }


class GCSRangeReader:
    """Seekable HTTP Range reader over a GCS blob for zero-download ZIP central-directory inspection."""

    def __init__(self, blob: Any) -> None:
        self.blob = blob
        self.size = int(blob.size or 0)
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        end = self.size - 1 if (n is None or n < 0 or self.pos + n > self.size) else self.pos + n - 1
        if end < self.pos:
            return b""
        data = self.blob.download_as_bytes(start=self.pos, end=end)
        self.pos += len(data)
        return data


def _compute_split_hashes(
    splits: dict[str, list[dict[str, Any]]],
    gallery: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Compute immutable SHA-256 cryptographic hashes for ``train``, ``val``, ``test``, and ``gallery``."""
    from utils.manifests import compute_split_hashes

    return compute_split_hashes(splits, gallery)


def _stratified_hul_non_hul_sample(
    items: list[dict[str, Any]],
    classes: dict[str, dict[str, Any]],
    cap: int,
    seed: int,
    split_name: str,
) -> list[dict[str, Any]]:
    """Stratify a split 50% HUL / 50% Non-HUL across ``(category, packaging_type, has_promo)`` and interleave."""
    from utils.manifests import stratified_hul_non_hul_sample

    return stratified_hul_non_hul_sample(items, classes, cap, seed, split_name)


def infer_gtmerch_shelf_label(
    fname: str,
    classes: dict[str, dict[str, Any]],
    brand_to_cid: dict[str, int],
    default_cid: int,
) -> tuple[int, dict[str, Any]]:
    """Infer canonical ``(sku_id, label_dict)`` for a ``gtmerch`` shelf bay with strict catalog parity."""
    bay = parse_gtmerch_bay_brand(fname)
    if bay is not None:
        b_low = bay["brand"].lower()
        cid = brand_to_cid.get(b_low, default_cid)
        if str(cid) in classes:
            return cid, dict(classes[str(cid)])
        return cid, {
            "sku_id": cid,
            "product": f"{bay['brand']}__{bay['brand']}",
            "brand": bay["brand"],
            "variant": bay["brand"],
            "base_variant": bay["brand"],
            "basepack_id": f"{bay['brand']}__{bay['packaging_type']}",
            "category": bay["category"],
            "packaging_type": bay["packaging_type"],
            "is_hul": bay["is_hul"],
            "has_promo": bay["has_promo"],
            "promo_type": bay["promo_type"],
        }
    return default_cid, dict(classes.get(str(default_cid)) or {"sku_id": default_cid})



def export_gtmerch_rpc_manifest(
    gtmerch_manifest: dict[str, Any],
    hul_golden_rpc: dict[str, Any] | None = None,
    max_train_samples: int = 800,
    max_val_samples: int = 100,
    max_test_samples: int = 100,
) -> dict[str, Any]:
    """Export an ``rpc.json`` benchmark manifest from ``audit_and_split_gtmerch_dataset()`` output."""
    classes: dict[str, dict[str, Any]] = {}
    gallery: dict[str, list[str]] = {}
    prod_to_id: dict[str, int] = {}
    brand_to_cid: dict[str, int] = {}

    if hul_golden_rpc and hul_golden_rpc.get("classes"):
        for cid, cmeta in hul_golden_rpc["classes"].items():
            cid_int = int(cid)
            classes[str(cid)] = dict(cmeta)
            prod_to_id[str(cmeta.get("product") or "")] = cid_int
            prod_to_id[f"{cmeta.get('brand')}__{cmeta.get('base_variant')}"] = cid_int
            b_low = str(cmeta.get("brand") or "").lower()
            if b_low and (
                b_low not in brand_to_cid
                or str(cmeta.get("base_variant") or "").lower() == b_low
                or bool(gallery.get(str(cid)))
            ):
                brand_to_cid[b_low] = cid_int
        for cid, g_list in hul_golden_rpc.get("gallery", {}).items():
            gallery[str(cid)] = list(g_list)
    else:
        classes["1"] = {
            "sku_id": 1,
            "product": "Dove__Dove",
            "brand": "Dove",
            "variant": "Dove",
            "base_variant": "Dove",
            "basepack_id": "Dove__Bottle",
            "category": "Skin_Care",
            "packaging_type": "Bottle",
            "is_hul": True,
            "has_promo": False,
            "promo_type": None,
        }
        gallery["1"] = []
        brand_to_cid["dove"] = 1

    default_cls_id = brand_to_cid.get("dove") or int(next(iter(classes.keys()), "1"))
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for item in gtmerch_manifest.get("items", []):
        if item.get("status") != "clean":
            continue
        sp = str(item.get("split") or "train")
        if sp not in splits:
            sp = "train"
        uri = str(item["uri"])
        fname = uri.rsplit("/", 1)[-1]
        p_boxes = item.get("product_boxes", [])
        a_boxes = item.get("asset_boxes", [])
        boxes_xyxy = [[float(b["bbox"][0]), float(b["bbox"][1]), float(b["bbox"][2]), float(b["bbox"][3])] for b in p_boxes]
        assets_xyxy = [[float(b["bbox"][0]), float(b["bbox"][1]), float(b["bbox"][2]), float(b["bbox"][3])] for b in a_boxes]

        inferred_cid, inferred_lbl = infer_gtmerch_shelf_label(fname, classes, brand_to_cid, default_cls_id)
        products: list[int] = []
        labels: list[dict[str, Any]] = []
        for pb in p_boxes:
            lbl = pb.get("labels")
            if lbl:
                pkey = f"{lbl.get('brand')}__{lbl.get('base_variant')}"
                cid_int = (
                    prod_to_id.get(pkey)
                    or prod_to_id.get(str(lbl.get("product") or ""))
                    or brand_to_cid.get(str(lbl.get("brand") or "").lower())
                    or inferred_cid
                )
                products.append(cid_int)
                labels.append({"sku_id": cid_int, **lbl})
            else:
                products.append(inferred_cid)
                labels.append(dict(inferred_lbl))

        splits[sp].append({
            "image": fname,
            "uri": uri,
            "outlet_id": item.get("outlet_id"),
            "session_id": item.get("session_id"),
            "width": int(item.get("width") or 1200),
            "height": int(item.get("height") or 1600),
            "boxes": boxes_xyxy,
            "asset_boxes": assets_xyxy,
            "products": products,
            "labels": labels,
        })

    # Prioritize brand/campaign-explicit shelves first so val/test have rich multi-brand ground truth
    def _has_brand_token(rec: dict[str, Any]) -> int:
        low = str(rec.get("image") or "").lower()
        return 0 if any(any(t in low for t in toks) for toks, *_ in _GTMERCH_FILENAME_BRAND_RULES) else 1

    for sp_name, cap in (("train", max_train_samples), ("val", max_val_samples), ("test", max_test_samples)):
        splits[sp_name].sort(key=lambda r: (_has_brand_token(r), str(r.get("image") or "")))
        if cap > 0 and len(splits[sp_name]) > cap:
            branded = [r for r in splits[sp_name] if _has_brand_token(r) == 0]
            generic = [r for r in splits[sp_name] if _has_brand_token(r) == 1]
            if len(branded) >= cap:
                step = max(1, len(branded) // cap)
                splits[sp_name] = branded[::step][:cap]
            else:
                rem = cap - len(branded)
                step = max(1, len(generic) // max(1, rem))
                splits[sp_name] = branded + generic[::step][:rem]

    split_hashes = _compute_split_hashes(splits, gallery)
    return {
        "source": "gs://gtmerch-marketshare/input_data/gtmerch_dataset_sku_asset (v2.1 Golden 2-class shelf + asset)",
        "classes": classes,
        "gallery": gallery,
        "splits": splits,
        "stats": {
            "mode": "gtmerch_golden",
            "schema_version": "2.1.0",
            "total_classes": len(classes),
            "train_samples": len(splits["train"]),
            "val_samples": len(splits["val"]),
            "test_samples": len(splits["test"]),
            "split_hashes": split_hashes,
            **(gtmerch_manifest.get("stats") or {}),
        },
    }


def publish_v2_golden_manifests_to_gcs(
    project: str = "ul-fs-s-sandbx-16-prj",
    marketshare_bucket: str = "marketshare-datasets",
    gtmerch_bucket: str = "gtmerch-marketshare",
    ms_json_bucket: str = "ms-json-dataset",
    refs_per_class: int = 4,
    max_train_per_split: int = 16000,
    max_eval_per_split: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build and publish v2.2 clean Golden manifests to GCS via ``utils.manifests.golden_builder``."""
    from utils.manifests.golden_builder import publish_clean_golden_manifests_to_gcs

    return publish_clean_golden_manifests_to_gcs(
        project=project,
        marketshare_bucket=marketshare_bucket,
        gtmerch_bucket=gtmerch_bucket,
        ms_json_bucket=ms_json_bucket,
        refs_per_class=refs_per_class,
        max_train_per_split=max_train_per_split,
        max_eval_per_split=max_eval_per_split,
        seed=seed,
    )







