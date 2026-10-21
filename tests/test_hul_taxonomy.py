"""Unit tests for ``src/utils/hul_taxonomy.py``, ``src/utils/hul_builder.py``, and ``src/utils/hul_audit.py``."""

from __future__ import annotations

from utils.hul_audit import (
    audit_embedding_cohesion,
    audit_intra_silo_md5_conflicts,
    compare_folder_embeddings,
)
from utils.hul_builder import dedup_and_split_blobs
from utils.hul_taxonomy import (
    normalize_brand_folder,
    parse_crop_filename,
    parse_pack_filename_brand_variant,
    parse_pack_folder,
    parse_variant_folder,
)


def test_normalize_brand_folder_strips_keras_prefixes_aliases_and_preserves_short_brands():
    assert normalize_brand_folder("") is None
    for short_brand in ("W2", "Z", "An", "Fr", "He"):
        sb = normalize_brand_folder(short_brand)
        assert sb is not None and sb["brand"] == short_brand and sb["is_hul"] is False

    colgate = normalize_brand_folder("AColgate")
    assert colgate == {
        "raw_folder": "AColgate",
        "brand": "Colgate",
        "is_hul": False,
        "category": "Oral_Care",
        "legacy_tier": "priority_a_prefix",
    }

    comfort = normalize_brand_folder("AComfort")
    assert comfort["brand"] == "Comfort" and comfort["is_hul"] is True and comfort["category"] == "Laundry"

    dr_wash_a = normalize_brand_folder("ADr_Wash")
    dr_wash_b = normalize_brand_folder("Dr_Wash")
    assert dr_wash_a["brand"] == "Dr_Wash" and dr_wash_a["legacy_tier"] == "a_prefix_twin"
    assert dr_wash_b["brand"] == "Dr_Wash"

    brook_bond = normalize_brand_folder("ABrook_Bond")
    assert brook_bond["brand"] == "Brooke_Bond" and brook_bond["is_hul"] is True

    axe = normalize_brand_folder("Axe")
    assert axe["brand"] == "Axe" and axe["is_hul"] is True and axe["legacy_tier"] == "hul_umbrella_brand_only"

    seven_up = normalize_brand_folder("Z_7_up")
    assert seven_up["brand"] == "7_up" and seven_up["legacy_tier"] == "numeric_z_prefix"

    amul = normalize_brand_folder("Amul")
    assert amul["brand"] == "Amul" and amul["legacy_tier"] == "standard"


def test_parse_variant_and_pack_folders_extract_w2_to_w9_and_backfill_rules():
    v1 = parse_variant_folder("variant_wash", "Comfort__Blue_Morning_Fresh_Bottle")
    assert v1 == {
        "tree": "variant_wash",
        "raw_silo_category": "wash",
        "raw_folder": "Comfort__Blue_Morning_Fresh_Bottle",
        "category": "Laundry",
        "brand": "Comfort",
        "variant": "Blue_Morning_Fresh_Bottle",
        "base_variant": "Blue_Morning_Fresh_Bottle",
        "packaging_type": "Bottle",
        "is_hul": True,
        "is_catchall_variant": False,
        "has_promo": False,
        "promo_type": None,
        "has_price_tag": False,
        "artwork_gen": "standard",
    }

    # Cross-category override: Ayush oral in variant_skin -> Oral_Care, Tube
    v_ayush = parse_variant_folder("variant_skin", "Ayush__Anti_Cavity_Clove_Oil_Toothpaste")
    assert v_ayush["category"] == "Oral_Care" and v_ayush["packaging_type"] == "Tube"
    v_ayush_gel = parse_variant_folder("variant_skin", "Ayush__Freshness_Gel_Cardamom")
    assert v_ayush_gel["category"] == "Oral_Care"

    # Cross-category override: Dove hair conditioner & hair mask in variant_skin -> Hair_Care,
    # and Dove bathing bar in variant_skin -> PW (no special-case Skin override).
    v_dove = parse_variant_folder("variant_skin", "Dove__Daily_Shine_Conditioner")
    assert v_dove["category"] == "Hair_Care" and v_dove["packaging_type"] == "Bottle"
    v_dove_mask = parse_variant_folder("variant_skin", "Dove__Intense_Repair__Mask")
    assert v_dove_mask["category"] == "Hair_Care" and v_dove_mask["packaging_type"] == "Jar"
    v_dove_cond_typo = parse_variant_folder("variant_skin", "Dove__Peptide_Bond_Strength_Conditionar")
    assert v_dove_cond_typo["category"] == "Hair_Care" and v_dove_cond_typo["packaging_type"] == "Tube"
    v_dove_bar = parse_variant_folder("variant_skin", "Dove__Cream_Beauty_Bathing_Bar")
    assert v_dove_bar["category"] == "PW" and v_dove_bar["packaging_type"] == "Bar"

    # UK/Commonwealth spelling support in VARIANT_CATEGORY_KEYWORD_RULES
    v_uk_moist = parse_variant_folder("variant_gen", "Lbp__Murumuru_Butter_Moisturiser")
    assert v_uk_moist["category"] == "Skin"
    v_uk_san = parse_variant_folder("variant_skin", "Lifebuoy__Total_Hand_Sanitiser")
    assert v_uk_san["category"] == "HHC"

    # Wellbeing_Nutrition stays HFD even when variant mentions Hair_Fall_Control
    v_wn = parse_variant_folder("variant_p5", "Wellbeing_Nutrition__Hair_Fall_Control")
    assert v_wn["category"] == "HFD" and v_wn["raw_silo_category"] == "p5"

    # Lbp bodylotion in variant_gen -> Skin, Bottle
    v_lbp = parse_variant_folder("variant_gen", "Lbp__Cherry_Blossom_And_Tea_Rose_Bodylotion")
    assert v_lbp["category"] == "Skin" and v_lbp["packaging_type"] == "Bottle"

    # W9 Promo / Multipack + artwork_gen extraction
    v_promo = parse_variant_folder("variant_skin", "Lifebuoy__Total_10_Soap_Bar_B3G1")
    assert (
        v_promo["has_promo"] is True
        and v_promo["promo_type"] == "B3G1"
        and v_promo["base_variant"] == "Total_10_Soap_Bar"
    )

    v_new = parse_variant_folder("variant_p5", "Boost__Jar_New")
    assert v_new["artwork_gen"] == "new" and v_new["base_variant"] == "Jar" and v_new["packaging_type"] == "Jar"

    p1 = parse_pack_folder("Laundry__Pouch")
    assert p1 == {"raw_folder": "Laundry__Pouch", "category": "Laundry", "packaging_type": "Pouch"}

    p_hair = parse_pack_folder("Hair__Bottle")
    assert p_hair == {"raw_folder": "Hair__Bottle", "category": "Hair_Care", "packaging_type": "Bottle"}

    # Embedded Brand__Variant inside pack_data filenames (including _image0261 and 3_Roses)
    ebv = parse_pack_filename_brand_variant("Bru__Instant_Sachet_Rs10_1803799_23.jpg")
    assert ebv is not None
    assert ebv["brand"] == "Bru" and ebv["has_promo"] is True and ebv["has_price_tag"] is True

    ebv_img = parse_pack_filename_brand_variant("Comfort__Blue_Morning_Fresh_Pouch_image0261.jpg")
    assert ebv_img is not None
    assert ebv_img["brand"] == "Comfort" and ebv_img["variant"] == "Blue_Morning_Fresh_Pouch"

    ebv_3roses = parse_pack_filename_brand_variant("3_Roses__Dust_image1450.jpg")
    assert ebv_3roses is not None
    assert ebv_3roses["brand"] == "3_Roses" and ebv_3roses["variant"] == "Dust"


def test_parse_crop_filename_handles_all_five_gcs_patterns():
    f1 = parse_crop_filename("3401195_34.jpg")
    assert f1 == {
        "session_id": "3401195",
        "crop_idx": 34,
        "bbox": None,
        "is_synthetic_copy": False,
        "is_augmented_copy": False,
    }

    f2 = parse_crop_filename("processed_3514704_47.jpg")
    assert f2 == {
        "session_id": "3514704",
        "crop_idx": 47,
        "bbox": None,
        "is_synthetic_copy": False,
        "is_augmented_copy": False,
    }

    f3 = parse_crop_filename("852414836922296320_852414836922296320_7__605_421_691_454.jpg")
    assert f3 == {
        "session_id": "852414836922296320",
        "crop_idx": 7,
        "bbox": (605, 421, 691, 454),
        "is_synthetic_copy": False,
        "is_augmented_copy": False,
    }

    f4 = parse_crop_filename("copy12.jpg")
    assert f4["is_synthetic_copy"] is True and f4["bbox"] is None

    f5 = parse_crop_filename("3401195_34 - Copy (2).jpg")
    assert f5["session_id"] == "3401195" and f5["crop_idx"] == 34 and f5["is_synthetic_copy"] is True

    # Newly discovered pack_data & NH patterns: DA_TBC_ / DA_ data augmentation, (copy), GT1_PCLW_, and space/bracket session IDs
    f_da = parse_crop_filename("DA_TBC_100_10.jpg")
    assert f_da == {
        "session_id": "100",
        "crop_idx": 10,
        "bbox": None,
        "is_synthetic_copy": True,
        "is_augmented_copy": True,
    }

    f_da_nh = parse_crop_filename("DA_4501_9.jpg")
    assert f_da_nh["session_id"] == "4501" and f_da_nh["is_augmented_copy"] is True

    f_space_sess = parse_crop_filename("11548 (2)_3.jpg")
    assert (
        f_space_sess["session_id"] == "11548"
        and f_space_sess["crop_idx"] == 3
        and f_space_sess["is_synthetic_copy"] is False
    )

    f_gt = parse_crop_filename("GT1_PCLW_112_66.jpg")
    assert f_gt == {
        "session_id": "112",
        "crop_idx": 66,
        "bbox": None,
        "is_synthetic_copy": False,
        "is_augmented_copy": False,
    }

    f_cls = parse_crop_filename("50_1317_53.jpg")
    assert f_cls == {
        "session_id": "1317",
        "crop_idx": 53,
        "bbox": None,
        "is_synthetic_copy": False,
        "is_augmented_copy": False,
    }

    f_paren = parse_crop_filename("2343_58 (copy).jpg")
    assert f_paren["session_id"] == "2343" and f_paren["crop_idx"] == 58 and f_paren["is_synthetic_copy"] is True

    # Duplicate of session_crop vs Windows Explorer bulk-renamed carton crop
    f_dup_num = parse_crop_filename("3401195_34 (1).jpg")
    assert f_dup_num["is_synthetic_copy"] is True

    f_win_bulk = parse_crop_filename("tz (1).jpg")
    assert f_win_bulk["is_synthetic_copy"] is False


def test_dedup_and_split_blobs_enforces_zero_session_leakage_and_md5_conflict_quarantine(tmp_path):
    import json

    from utils import dataset

    records = []
    for sess_idx in range(1, 31):
        sess = f"340{sess_idx:04d}"
        records.append({
            "uri": f"gs://marketshare-datasets/input_data/variant_wash/train/Comfort__Blue_Bottle/{sess}_1.jpg",
            "md5": f"md5_{sess}_1",
        })
        records.append({
            "uri": f"gs://marketshare-datasets/input_data/variant_wash/train/Comfort__Blue_Bottle/{sess}_2.jpg",
            "md5": f"md5_{sess}_2",
        })
    # Add synthetic copy and exact MD5 duplicate -> both must be dropped
    records.append({
        "uri": "gs://marketshare-datasets/input_data/variant_wash/train/Comfort__Blue_Bottle/copy0.jpg",
        "md5": "md5_copy0",
    })
    records.append({
        "uri": "gs://marketshare-datasets/input_data/variant_wash/train/Comfort__Blue_Bottle/3400001_99.jpg",
        "md5": "md5_3400001_1",
    })
    # Add a multi-folder MD5 conflict in a second class (same md5_3400004_1 as Comfort__Blue_Bottle)
    records.append({
        "uri": "gs://marketshare-datasets/input_data/variant_wash/train/Comfort__Pink_Bottle/3400004_1.jpg",
        "md5": "md5_3400004_1",
    })
    records.append({
        "uri": "gs://marketshare-datasets/input_data/variant_wash/train/Comfort__Pink_Bottle/3400001_3.jpg",
        "md5": "md5_pink_gallery",
    })

    manifest = dedup_and_split_blobs(records, mode="hul_variants", refs_per_class=4, seed=42)
    assert manifest["stats"]["dropped_synthetic_copies"] == 1
    assert manifest["stats"]["dropped_md5_duplicates"] == 1
    assert manifest["stats"]["quarantined_multi_folder_md5s"] == 1
    assert manifest["stats"]["quarantined_eval_conflicts"] == 2
    assert manifest["stats"]["total_classes"] == 2

    gal_uris = set(manifest["gallery"]["1"])
    gal_uris_2 = set(manifest["gallery"]["2"])
    assert all(not u.endswith("3400004_1.jpg") for u in gal_uris | gal_uris_2)
    val_sessions = {r["session_id"] for r in manifest["splits"]["val"]}
    test_sessions = {r["session_id"] for r in manifest["splits"]["test"]}
    assert val_sessions.isdisjoint(test_sessions)
    for u in gal_uris:
        gal_sess = u.rsplit("/", 1)[-1].split("_")[0]
        assert gal_sess not in val_sessions and gal_sess not in test_sessions

    out_dir = tmp_path / "hul_variants"
    out_dir.mkdir()
    (out_dir / "rpc.json").write_text(json.dumps(manifest))

    cat = dataset.catalog(str(out_dir), name="hul_variants")
    assert cat[1]["brand"] == "Comfort" and cat[1]["packaging_type"] == "Bottle"
    samples = dataset.sample_images("test", limit=5, seed=0, root=str(out_dir), name="hul_variants")
    assert len(samples) >= 1 and samples[0].path.startswith("gs://marketshare-datasets/")


def test_audit_embedding_cohesion_and_intra_silo_md5_conflicts():
    items = [
        {"uri": "gs://b/Dove/1.jpg", "class_name": "Dove", "embedding": [1.0, 0.0, 0.0]},
        {"uri": "gs://b/Dove/2.jpg", "class_name": "Dove", "embedding": [0.98, 0.02, 0.0]},
        {"uri": "gs://b/Dove/3.jpg", "class_name": "Dove", "embedding": [0.99, 0.01, 0.0]},
        # Mislabelled Colgate crop sitting inside Dove folder:
        {"uri": "gs://b/Dove/mislabelled.jpg", "class_name": "Dove", "embedding": [0.0, 1.0, 0.0]},
        {"uri": "gs://b/Colgate/1.jpg", "class_name": "Colgate", "embedding": [0.0, 1.0, 0.0]},
        {"uri": "gs://b/Colgate/2.jpg", "class_name": "Colgate", "embedding": [0.02, 0.98, 0.0]},
    ]
    res = audit_embedding_cohesion(items)
    assert res["total_crops"] == 6 and res["total_classes"] == 2
    assert res["suspect_count"] == 1
    assert res["suspect_items"][0]["uri"] == "gs://b/Dove/mislabelled.jpg"
    assert res["suspect_items"][0]["nearest_rival_class"] == "Colgate"

    comp_same = compare_folder_embeddings(
        [[1.0, 0.0], [0.99, 0.01]],
        [[0.98, 0.02], [1.0, 0.0]],
        names_a=["a.jpg", "b.jpg"],
        names_b=["b.jpg", "c.jpg"],
        md5_a=["m1", "m2"],
        md5_b=["m2", "m3"],
    )
    assert comp_same["same_visual_concept"] is True
    assert comp_same["filename_overlap"] == 1 and comp_same["md5_overlap"] == 1

    conflicts = audit_intra_silo_md5_conflicts([
        {"uri": "gs://b/Pepsodent__2_In_1/1.jpg", "md5": "dup_hash"},
        {"uri": "gs://b/Pepsodent__Gel_Fresh/1.jpg", "md5": "dup_hash"},
        {"uri": "gs://b/Pepsodent__2_In_1/2.jpg", "md5": "unique_hash"},
    ])
    assert conflicts["conflicted_md5_count"] == 1
    assert conflicts["conflicted_folder_pairs"][0]["shared_md5_count"] == 1


def test_build_golden_json_payload_joins_3_silos_and_unions_twins():
    from utils.hul_audit import validate_golden_joins
    from utils.hul_builder import build_golden_json_payload

    records = [
        # 1. 3-way exact MD5 join across brand_data, variant_p5 (twin + promo), and pack_data
        {
            "uri": "gs://marketshare-datasets/brand_data/brand/brand_data/ARed_Label/2282_7.jpg",
            "md5": "md5_red_label_1",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_p5/train/Red_Label__Natural_Care_Poly/2282_7.jpg",
            "md5": "md5_red_label_1",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_p5/train/Red_Label__Natural_Care_Poly_Rs_10/2282_7.jpg",
            "md5": "md5_red_label_1",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/Beverages__Pouch/2282_7.jpg",
            "md5": "md5_red_label_1",
        },
        # 2. Tier-3 re-encoded JPEG pair sharing (session_id=10724, crop_idx=1) and matching brand (Ayush) + dims
        {
            "uri": "gs://marketshare-datasets/brand_data/brand/brand_data/AAyush/10724_1.jpg",
            "md5": "md5_ayush_brand_reencoded",
            "width": 111,
            "height": 484,
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Ayush__Sensitivity_Relief/10724_1.jpg",
            "md5": "md5_ayush_var_orig",
            "width": 111,
            "height": 484,
        },
        # 3. Cross-category counter collision on 1098_53.jpg (different MD5, different brand: Ayush vs Rexona) -> MUST NOT merge!
        {
            "uri": "gs://marketshare-datasets/brand_data/brand/brand_data/AAyush/1098_53.jpg",
            "md5": "md5_ayush_1098",
            "width": 93,
            "height": 295,
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_gen/train/Rexona__Sport_Defence_Roll_On/1098_53.jpg",
            "md5": "md5_rexona_1098",
            "width": 73,
            "height": 248,
        },
        # 4. True intra-pack conflict (same MD5 in HFD__Bottle and HFD__Jar) -> quarantined_conflict
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/HFD__Bottle/Horlicks__Standard_Jar_image10.jpg",
            "md5": "md5_horlicks_conflict",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/HFD__Jar/Horlicks__Standard_Jar_image10.jpg",
            "md5": "md5_horlicks_conflict",
        },
        # 5. Distinct prefix-sharing variants (Facewash vs Facewash_Scrub) colliding on MD5 -> MUST be flagged as base_variant_conflict!
        {
            "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Ponds__Bright_Beauty_Facewash/4401_1.jpg",
            "md5": "md5_ponds_prefix_conflict",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Ponds__Bright_Beauty_Facewash_Scrub/4401_1.jpg",
            "md5": "md5_ponds_prefix_conflict",
        },
    ]

    payload = build_golden_json_payload(records)
    st = payload["stats"]
    assert st["joined_3way_brand_variant_pack"] == 1
    assert st["compatible_twin_unions"] == 1
    assert st["tier3_reencoded_session_merges"] == 1
    assert st["quarantined_conflicts"] == 2

    by_md5 = {it["md5"]: it for it in payload["items"]}
    rl = by_md5["md5_red_label_1"]
    assert rl["join_mode"] == "3way_brand_variant_pack"
    assert rl["join_tier"] == "tier2_twin_union"
    assert rl["join_confidence"] == 0.99
    assert rl["is_conflicting_md5"] is False
    assert rl["brand_data_uri"].endswith("ARed_Label/2282_7.jpg")
    assert rl["variant_uri"].endswith("Red_Label__Natural_Care_Poly_Rs_10/2282_7.jpg")
    assert rl["pack_data_uri"].endswith("Beverages__Pouch/2282_7.jpg")
    assert rl["status"] == "clean"
    assert rl["labels"]["brand"] == "Red_Label"
    assert rl["labels"]["base_variant"] == "Natural_Care_Poly"
    assert rl["labels"]["basepack_id"] == "Red_Label__Natural_Care_Poly__Pouch"
    assert rl["labels"]["packaging_type"] == "Pouch"
    assert rl["labels"]["has_promo"] is True and rl["labels"]["promo_type"] == "RS10"

    # Distinct prefix-sharing variants (Facewash vs Facewash_Scrub) must be quarantined, not merged
    ponds_conf = by_md5["md5_ponds_prefix_conflict"]
    assert ponds_conf["status"] == "quarantined_conflict"
    assert any("base_variant_conflict" in r for r in ponds_conf["conflict_reasons"])

    ay = by_md5["md5_ayush_var_orig"]
    assert ay["join_mode"] == "2way_brand_variant"
    assert ay["join_tier"] == "tier3_reencoded_session"
    assert ay["join_confidence"] == 0.985
    assert ay["brand_data_uri"].endswith("AAyush/10724_1.jpg")
    assert ay["variant_uri"].endswith("Ayush__Sensitivity_Relief/10724_1.jpg")
    assert ay["labels"]["brand"] == "Ayush"

    # Counter collision (1098_53.jpg) remained separate
    assert "md5_ayush_1098" in by_md5 and "md5_rexona_1098" in by_md5

    # True pack conflict flagged
    hk = by_md5["md5_horlicks_conflict"]
    assert hk["status"] == "quarantined_conflict"
    assert hk["is_conflicting_md5"] is True
    assert hk["join_confidence"] == 0.0

    report = validate_golden_joins(payload)
    assert report["all_invariants_passed"] is True
    assert report["uri_column_completeness"]["3way_with_all_3_uris"] == 1
    assert report["session_split_leakage_count"] == 0


def test_golden_json_l8_edge_cases_and_isolation(tmp_path) -> None:
    """Verify L8 edge cases: umbrella brand pack override, competitor pack joins, brand-qualified basepack_id, studio render pinning, SFT export, and raw dataset protection."""
    import pytest

    from utils import dataset
    from utils.hul_builder import (
        build_golden_json_payload,
        export_golden_sft_jsonl,
        save_golden_json_payload,
    )

    records = [
        # 1. Umbrella brand (ADove defaults to Skin) joined with Hair__Bottle -> category overridden to Hair_Care
        {
            "uri": "gs://marketshare-datasets/brand_data/brand/brand_data/ADove/9001_1.jpg",
            "md5": "md5_dove_hair_bottle",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/Hair__Bottle/9001_1.jpg",
            "md5": "md5_dove_hair_bottle",
        },
        # 2. Non-HUL competitor brand (AColgate) joined with Oral_Care__Tube -> Non_HUL_Competitor with Tube pack
        {
            "uri": "gs://marketshare-datasets/brand_data/brand/brand_data/AColgate/9002_4.jpg",
            "md5": "md5_colgate_tube",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/Oral_Care__Tube/9002_4.jpg",
            "md5": "md5_colgate_tube",
        },
        # 3. Multimodal packaging for same base_variant (Boost__3x in Jar and Pouch) + studio render pinning
        {
            "uri": "gs://marketshare-datasets/input_data/variant_p5/train/Boost__3x/HUL-410166D-FG050_12.jpg",
            "md5": "md5_boost_jar_render",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/HFD__Jar/HUL-410166D-FG050_12.jpg",
            "md5": "md5_boost_jar_render",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_p5/train/Boost__3x/9005_2.jpg",
            "md5": "md5_boost_pouch_store",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/HFD__Pouch/9005_2.jpg",
            "md5": "md5_boost_pouch_store",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_p5/train/Boost__3x/9006_2.jpg",
            "md5": "md5_boost_pouch_store_2",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/HFD__Pouch/9006_2.jpg",
            "md5": "md5_boost_pouch_store_2",
        },
        # 4. Cross-brand shared base_variant name ('Dust' under 3_Roses vs Red_Label) must not collide
        {
            "uri": "gs://marketshare-datasets/input_data/variant_p5/train/3_Roses__Dust/9007_1.jpg",
            "md5": "md5_3roses_dust_pouch",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/Beverages__Pouch/9007_1.jpg",
            "md5": "md5_3roses_dust_pouch",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_p5/train/Red_Label__Dust/9008_1.jpg",
            "md5": "md5_redlabel_dust_carton",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/Beverages__Carton/9008_1.jpg",
            "md5": "md5_redlabel_dust_carton",
        },
    ]

    payload = build_golden_json_payload(records)
    st = payload["stats"]
    assert st["competitor_pack_joins"] == 1
    assert st["hul_brand_only_pack_joins"] == 1
    # Only Boost__3x has multiple packaging types; 3_Roses__Dust and Red_Label__Dust are separate brands!
    assert st["multimodal_pack_base_variants"] == 1
    assert st["planogram_renders_pinned_to_gallery"] == 1

    by_md5 = {it["md5"]: it for it in payload["items"]}
    dove = by_md5["md5_dove_hair_bottle"]
    assert dove["labels"]["brand"] == "Dove"
    assert dove["labels"]["category"] == "Hair_Care"
    assert dove["labels"]["category_source"] == "pack_folder"
    assert dove["labels"]["packaging_type"] == "Bottle"

    colg = by_md5["md5_colgate_tube"]
    assert colg["labels"]["is_hul"] is False
    assert colg["labels"]["brand"] == "Colgate"
    assert colg["labels"]["category"] == "Oral_Care"
    assert colg["labels"]["base_variant"] == "Non_HUL_Competitor"
    assert colg["labels"]["packaging_type"] == "Tube"

    boost_render = by_md5["md5_boost_jar_render"]
    assert boost_render["is_planogram_render"] is True
    assert boost_render["split"] == "gallery"
    assert boost_render["labels"]["basepack_id"] == "Boost__3x__Jar"

    boost_store = by_md5["md5_boost_pouch_store"]
    assert boost_store["labels"]["basepack_id"] == "Boost__3x__Pouch"

    assert by_md5["md5_3roses_dust_pouch"]["labels"]["basepack_id"] == "3_Roses__Dust__Pouch"
    assert by_md5["md5_redlabel_dust_carton"]["labels"]["basepack_id"] == "Red_Label__Dust__Carton"

    # Verify strict directory isolation guard
    for protected in ("data/products", "data/rpc", "data/sku110k", "gs://marketshare-datasets/brand_data"):
        with pytest.raises(ValueError, match="Refusing to write Golden JSON payload"):
            save_golden_json_payload(payload, output_dir=protected)

    golden_dir = tmp_path / "hul_golden"
    out_paths = save_golden_json_payload(payload, output_dir=str(golden_dir))
    assert "golden_payload.json" in out_paths["golden_payload_path"]
    assert "rpc.json" in out_paths["rpc_manifest_path"]
    cat = dataset.catalog(str(golden_dir), name="hul_golden")
    assert len(cat) >= 1

    sft_out = export_golden_sft_jsonl(payload, output_dir=str(golden_dir / "sft"))
    assert sft_out["train_count"] >= 1
    assert sft_out["triplet_count"] >= 1


def test_bigquery_catalog_ddl_and_query_client() -> None:
    """Verify BigQueryCatalogClient, golden_item_to_bq_row, and CloudSQL training_crops DDL + HYBRID_RRF_SQL."""
    from utils.bigquery_catalog import BigQueryCatalogClient, golden_item_to_bq_row
    from utils.cloudsql import HYBRID_RRF_SQL, SCHEMA_SQL_CLOUDSQL

    client = BigQueryCatalogClient()
    assert client.project == "ul-fs-s-sandbx-16-prj"
    assert client.location == "us-central1"
    ddls = client.render_ddls()
    assert "ul-fs-s-sandbx-16-prj.hul_golden_catalog.golden_crops" in ddls["golden_crops_ddl"]
    assert "brand_data_uri" in ddls["join_validation_sql"]
    assert "ML.GENERATE_EMBEDDING" in ddls["ai_unstructured_audit_sql"]
    assert "NOT REGEXP_CONTAINS" in ddls["ai_unstructured_audit_sql"]
    assert "gemini-embedding-2" in ddls["object_table_and_models_ddl"]
    assert "gemini-3.5-flash-lite" in ddls["object_table_and_models_ddl"]
    assert "gemini-3.8-flash-lite" not in ddls["object_table_and_models_ddl"]
    assert "VECTOR_SEARCH" in ddls["vector_search_sql"]
    assert "CREATE TABLE IF NOT EXISTS training_crops" in SCHEMA_SQL_CLOUDSQL
    assert "FULL OUTER JOIN k ON v.id = k.id" in HYBRID_RRF_SQL

    sample_row = golden_item_to_bq_row({
        "md5": "abc",
        "canonical_uri": "gs://b/v/1.jpg",
        "brand_data_uri": "gs://b/b/1.jpg",
        "variant_uri": "gs://b/v/1.jpg",
        "pack_data_uri": "gs://b/p/1.jpg",
        "join_mode": "3way_brand_variant_pack",
        "join_tier": "tier1_exact_md5",
        "join_confidence": 1.0,
        "is_conflicting_md5": False,
        "status": "clean",
        "conflict_reasons": [],
        "session_id": "2282",
        "crop_idx": 7,
        "bbox": None,
        "is_planogram_render": False,
        "split": "gallery",
        "labels": {
            "is_hul": True,
            "category": "Beverages",
            "category_source": "variant_folder",
            "brand": "Red_Label",
            "variant": "Natural_Care_Poly",
            "base_variant": "Natural_Care_Poly",
            "basepack_id": "Red_Label__Natural_Care_Poly__Pouch",
            "packaging_type": "Pouch",
            "packaging_type_source": "pack_folder",
            "has_promo": False,
            "promo_type": None,
            "has_price_tag": False,
            "artwork_gen": "standard",
        },
    })
    assert sample_row["brand_data_uri"] == "gs://b/b/1.jpg"
    assert sample_row["variant_uri"] == "gs://b/v/1.jpg"
    assert sample_row["pack_data_uri"] == "gs://b/p/1.jpg"
    assert sample_row["basepack_id"] == "Red_Label__Natural_Care_Poly__Pouch"
    assert sample_row["allowed_packaging_types"] == []
    assert sample_row["is_catchall_variant"] is False


def test_de_pr5_non_hul_taxonomy_compound_splitter_and_ownership_fixes() -> None:
    """Verify Non-HUL variant_NH_gen & variant_NH_P6 taxonomy, ownership corrections, compound splitter, and typo matcher."""
    from utils.hul_taxonomy import (
        normalize_brand_folder,
        normalize_variant_tokens,
        parse_variant_folder,
    )

    # Ownership corrections: Oral_B (P&G) and Freshwrapp (Aditya Birla) are Non-HUL; Oziva is HUL (HFD)
    oralb = parse_variant_folder("variant_NH_gen", "Oral_B__Pro_Health_Toothbrushes")
    assert oralb["is_hul"] is False and oralb["category"] == "Oral_Care"
    assert "tooth_brushes" in normalize_variant_tokens("Pro_Health_Toothbrushes")

    freshwrapp = normalize_brand_folder("Freshwrapp")
    assert freshwrapp is not None and freshwrapp["is_hul"] is False and freshwrapp["category"] == "HHC"

    oziva = parse_variant_folder("variant_NH_P6", "Oziva__Plant_Based_Collagen_Builder")
    assert oziva["is_hul"] is True and oziva["category"] == "HFD"

    # Cross-silo spelling aliases: Tetely -> Tetley, Yipee -> Yippee
    tetley = parse_variant_folder("variant_NH_P6", "Tetely__Green_Tea_Immune")
    assert tetley["brand"] == "Tetley" and tetley["is_hul"] is False and tetley["category"] == "Beverages"

    yippee = parse_variant_folder("variant_NH_P6", "Yipee__Magic_Masala_Noodles")
    assert yippee["brand"] == "Yippee" and yippee["is_hul"] is False and yippee["category"] == "Foods"

    # Multi-category brand profiles: Tata (Foods vs Beverages), Nivea (Deo vs PW vs Skin), Dettol (PW vs HHC)
    tata_spice = parse_variant_folder("variant_NH_P6", "Tata__Sampann_Chilli_Powder")
    assert tata_spice["category"] == "Foods"
    tata_tea = parse_variant_folder("variant_NH_P6", "Tata__Agni_Leaf_Tea")
    assert tata_tea["category"] == "Beverages"

    nivea_deo = parse_variant_folder("variant_NH_gen", "Nivea__Pearl_And_Beauty_Roll_On")
    assert nivea_deo["category"] == "DMT" and nivea_deo["packaging_type"] == "Roll_On"
    nivea_pw = parse_variant_folder("variant_NH_gen", "Nivea__Waterlily_And_Oil_Shower_Gel")
    assert nivea_pw["category"] == "PW"
    nivea_skin = parse_variant_folder("variant_NH_gen", "Nivea__Nourishing_Body_Milk")
    assert nivea_skin["category"] == "Skin"

    dettol_pw = parse_variant_folder("variant_NH_gen", "Dettol__Skincare_Liquid_Handwash")
    assert dettol_pw["category"] == "PW"
    dettol_hhc = parse_variant_folder("variant_NH_gen", "Dettol__Antiseptic_Disinfectant_Liquid")
    assert dettol_hhc["category"] == "HHC"

    # Damerau-Levenshtein <= 1 typo resilience (shmapoo -> Hair_Care)
    typo_shampoo = parse_variant_folder("variant_NH_gen", "Chick__Egg_Protein_Shmapoo")
    assert typo_shampoo["category"] == "Hair_Care"

    # Catch-all detection (<Brand>__Others and <Brand>__<Brand>)
    catchall_others = parse_variant_folder("variant_NH_gen", "Colgate__Others")
    assert catchall_others["is_catchall_variant"] is True
    catchall_self = parse_variant_folder("variant_p5", "Boost__Boost")
    assert catchall_self["is_catchall_variant"] is True


def test_de_pr5_4tier_multimodal_packaging_and_4tier_catchall_enrichment() -> None:
    """Verify 4-Tier (<80%) multimodal packaging resolution and 4-Tier <Brand>__Others enrichment."""
    from utils.hul_builder import (
        build_golden_json_payload,
        validate_golden_data_contract,
    )

    records = [
        # --- 1. Unimodal (>=80%) packaging propagation: 4 Pouch crops + 1 unjoined crop ---
        {
            "uri": "gs://marketshare-datasets/input_data/variant_wash/train/Surf_Excel__Easy_Wash/1001_1.jpg",
            "md5": "md5_se_1",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/Laundry__Pouch/1001_1.jpg",
            "md5": "md5_se_1",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_wash/train/Surf_Excel__Easy_Wash/1002_1.jpg",
            "md5": "md5_se_2",
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/Laundry__Pouch/1002_1.jpg",
            "md5": "md5_se_2",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_wash/train/Surf_Excel__Easy_Wash/1003_1.jpg",
            "md5": "md5_se_unjoined",
        },
        # --- 2. Multimodal (<80%) packaging resolution: 50% Bottle (tall) vs 50% Box (wide) ---
        {
            "uri": "gs://marketshare-datasets/input_data/variant_NH_gen/train/Godrej__Expert_Rich_Crème/2001_1.jpg",
            "md5": "md5_godrej_bottle",
            "width": 100,
            "height": 300,
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/Hair__Bottle/2001_1.jpg",
            "md5": "md5_godrej_bottle",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_NH_gen/train/Godrej__Expert_Rich_Crème/2002_1.jpg",
            "md5": "md5_godrej_box",
            "width": 250,
            "height": 150,
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "uri": "gs://marketshare-datasets/pack_data/pack/train/Hair__Box/2002_1.jpg",
            "md5": "md5_godrej_box",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_NH_gen/train/Godrej__Expert_Rich_Crème/2003_1.jpg",
            "md5": "md5_godrej_unjoined_tall",
            "width": 95,
            "height": 290,
            "embedding": [0.99, 0.01, 0.0],
        },
        # --- 3. 4-Tier Catch-All Enrichment (<Brand>__Others & <Brand>__<Brand>) ---
        # Tier-1 MD5 Twin Promotion: same MD5 in Vaseline__Others and Vaseline__Deep_Restore
        {
            "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Vaseline__Others/3001_1.jpg",
            "md5": "md5_vas_tier1",
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Vaseline__Deep_Restore/3001_1.jpg",
            "md5": "md5_vas_tier1",
            "embedding": [0.0, 0.0, 1.0],
        },
        # Tier-2 Session-Sibling Promotion: session 3002 has ONLY Vaseline__Deep_Restore + Vaseline__Others crop
        {
            "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Vaseline__Deep_Restore/3002_1.jpg",
            "md5": "md5_vas_sess_sibling_named",
            "embedding": [0.0, 0.02, 0.98],
        },
        {
            "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Vaseline__Others/3002_2.jpg",
            "md5": "md5_vas_tier2_sess",
        },
        # Tier-3 Visual Prototype Promotion: session 3003 has Vaseline__Others crop matching Deep_Restore prototype
        {
            "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Vaseline__Others/3003_1.jpg",
            "md5": "md5_vas_tier3_proto",
            "embedding": [0.0, 0.01, 0.99],
        },
        # Tier-4 Open-Set Tail Preservation: orthogonal embedding in Vaseline__Others stays Vaseline__Others
        {
            "uri": "gs://marketshare-datasets/input_data/variant_skin/train/Vaseline__Others/3004_1.jpg",
            "md5": "md5_vas_tier4_tail",
            "embedding": [1.0, 0.0, 0.0],
        },
    ]

    payload = build_golden_json_payload(records)
    by_md5 = {it["md5"]: it for it in payload["items"]}

    # Unimodal >=80% consensus
    se_unjoined = by_md5["md5_se_unjoined"]
    assert se_unjoined["labels"]["packaging_type"] == "Pouch"
    assert se_unjoined["labels"]["packaging_type_source"] == "pack_folder_consensus"
    assert se_unjoined["labels"]["allowed_packaging_types"] == ["Pouch"]

    # Multimodal <80% subcluster assignment + allowed_packaging_types
    godrej_tall = by_md5["md5_godrej_unjoined_tall"]
    assert godrej_tall["labels"]["packaging_type"] == "Bottle"
    assert godrej_tall["labels"]["packaging_type_source"] == "multimodal_subcluster"
    assert set(godrej_tall["labels"]["allowed_packaging_types"]) == {"Bottle", "Box"}

    # Tier-1 MD5 catch-all promotion
    vas_t1 = by_md5["md5_vas_tier1"]
    assert vas_t1["labels"]["base_variant"] == "Deep_Restore"
    assert vas_t1["labels"]["is_catchall_variant"] is False
    assert vas_t1["labels"]["variant_enrichment_source"] == "md5_specific_sibling"

    # Tier-2 Session-sibling catch-all promotion
    vas_t2 = by_md5["md5_vas_tier2_sess"]
    assert vas_t2["labels"]["base_variant"] == "Deep_Restore"
    assert vas_t2["labels"]["is_catchall_variant"] is False
    assert vas_t2["labels"]["variant_enrichment_source"] == "session_specific_sibling"

    # Tier-3 Prototype promotion vs Tier-4 Open-Set Tail preservation
    assert payload["stats"]["catchall_promoted_by_prototype"] == 1
    assert by_md5["md5_vas_tier3_proto"]["labels"]["base_variant"] == "Deep_Restore"
    assert by_md5["md5_vas_tier3_proto"]["labels"]["variant_enrichment_source"] == "sibling_prototype_cluster"
    assert by_md5["md5_vas_tier4_tail"]["labels"]["is_catchall_variant"] is True
    assert by_md5["md5_vas_tier4_tail"]["labels"]["variant_enrichment_source"] == "open_set_catchall"

    # Validate Golden Data Contract passes cleanly
    contract = validate_golden_data_contract(payload, strict=True)
    assert contract["valid"] is True


def test_de_pr5_metrics_open_rgb_normalized_and_cloudsql_rrf_params(tmp_path) -> None:
    """Verify open_rgb_normalized, granularity-aware match_product, bootstrap_ci95, and 15-param build_hybrid_rrf_params."""
    from PIL import Image

    from utils import dataset, metrics
    from utils.cloudsql import HYBRID_RRF_SQL, build_hybrid_rrf_params

    # 1. Verify open_rgb_normalized composites RGBA transparency onto neutral (240, 240, 240) instead of black (0, 0, 0)
    rgba_path = tmp_path / "transparent_crop.png"
    rgba_img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    rgba_img.save(rgba_path)
    rgb_out = dataset.open_rgb_normalized(rgba_path)
    assert rgb_out.mode == "RGB"
    assert rgb_out.getpixel((5, 5)) == (240, 240, 240)

    # 2. Verify granularity-aware metrics.match_product (allowed_packaging_types + Non_HUL_Competitor / catchall)
    gt_multimodal = {
        "brand": "Godrej",
        "category": "Hair_Care",
        "base_variant": "Expert_Rich_Crème",
        "packaging_type": "Bottle",
        "allowed_packaging_types": ["Bottle", "Box"],
        "is_hul": False,
    }
    pred_box = {
        "brand": "Godrej",
        "category": "Hair_Care",
        "base_variant": "Expert_Rich_Crème",
        "packaging_type": "Box",
        "is_hul": False,
    }
    assert metrics.match_product(
        pred_box,
        gt_multimodal,
        fields=("brand", "category", "base_variant", "packaging_type", "is_hul"),
    )

    gt_brand_only = {
        "brand": "Colgate",
        "category": "Oral_Care",
        "base_variant": "Non_HUL_Competitor",
        "packaging_type": "Tube",
        "is_hul": False,
    }
    pred_specific_colgate = {
        "brand": "Colgate",
        "category": "Oral_Care",
        "base_variant": "Strong_Teeth",
        "packaging_type": "Tube",
        "is_hul": False,
    }
    assert metrics.match_product(
        pred_specific_colgate,
        gt_brand_only,
        fields=("brand", "category", "base_variant", "packaging_type", "is_hul"),
    )

    h_scores = metrics.hierarchical_field_scores([{"pred": pred_box, "gt": gt_multimodal}])
    assert h_scores["level5_packaging_type"] == 1.0 and h_scores["level3_brand"] == 1.0

    ci = metrics.bootstrap_ci95([1.0, 1.0, 0.8, 0.9, 1.0], n_boot=200, seed=42)
    assert 0.8 <= ci["mean"] <= 1.0 and ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]

    # 3. Verify CloudSQL build_hybrid_rrf_params produces exact 15 parameters matching HYBRID_RRF_SQL %s placeholders
    params = build_hybrid_rrf_params(
        "[0.1,0.2,0.3]",
        kw_query="Colgate Strong Teeth",
        category="Oral_Care",
        pack_form="Tube",
        limit=10,
    )
    assert len(params) == 15
    assert HYBRID_RRF_SQL.count("%s") == 15


def test_de_pr6_gtmerch_audit_split_and_cross_bucket_linker(tmp_path) -> None:
    """Verify parse_gtmerch_filename, audit_and_split_gtmerch_dataset, link_gtmerch_shelves_to_golden_crops, and gtmerch_golden."""
    import json

    from utils import dataset
    from utils.bigquery_catalog import BigQueryCatalogClient
    from utils.hul_builder import (
        audit_and_split_gtmerch_dataset,
        build_golden_json_payload,
        link_gtmerch_shelves_to_golden_crops,
    )
    from utils.hul_taxonomy import parse_gtmerch_filename

    # 1. Filename parser: U-* / IN* outlet + asset + date + ts, numeric session ID, and artifact files
    m_u = parse_gtmerch_filename("U-105232550_A-3522429_2025-07-02_1751453692219.jpg")
    assert (
        m_u["outlet_id"] == "U-105232550"
        and m_u["asset_id"] == "A-3522429"
        and m_u["capture_date"] == "2025-07-02"
        and m_u["timestamp_ms"] == 1751453692219
        and m_u["session_id"] == "U-105232550"
        and m_u["is_artifact_file"] is False
    )

    m_in = parse_gtmerch_filename("IN5008777715_A-1589707_2025-07-02_1751452105528 (1).jpg")
    assert (
        m_in["outlet_id"] == "IN5008777715"
        and m_in["asset_id"] == "A-1589707"
        and m_in["is_copy_variant"] is True
    )

    m_num = parse_gtmerch_filename("3514704.jpg")
    assert m_num["session_id"] == "3514704" and m_num["outlet_id"] == "STORE_3514704"

    m_art = parse_gtmerch_filename(".amlignore.amltmp")
    assert m_art["is_artifact_file"] is True

    # 2. Audit & split gtmerch dataset: artifacts, orphans, train<->test MD5 leak with Class-0 box-splitting,
    #    intra-train duplicates, double-clicked IoU>0.92 boxes, and Class-1 inside Class-0 containment
    gt_records = [
        {"uri": "gs://gtmerch-marketshare/train/images/.amlignore", "md5": "md5_art"},
        {
            "uri": "gs://gtmerch-marketshare/train/images/U-999999_A-111_2025-07-01_1751000000000.jpg",
            "md5": "md5_orphan",
            "has_label_file": False,
        },
        # Train<->Test MD5 leak with Class-0 box-splitting: train has 1 coarse Class-0 bay, test has 2 split Class-0 bays
        {
            "uri": "gs://gtmerch-marketshare/train/images/U-105232550_A-3522429_2025-07-02_1751453692219 (1).jpg",
            "md5": "md5_leak_box_split",
            "split": "train",
            "width": 1000,
            "height": 1000,
            "boxes": [
                {"class_id": 0, "bbox": [50, 50, 950, 950]},
                {"class_id": 1, "bbox": [100, 100, 200, 250]},
            ],
        },
        {
            "uri": "gs://gtmerch-marketshare/test/images/U-105232550_A-3522429_2025-07-02_1751453692219.jpg",
            "md5": "md5_leak_box_split",
            "split": "test",
            "width": 1000,
            "height": 1000,
            "boxes": [
                {"class_id": 0, "bbox": [50, 50, 950, 480]},
                {"class_id": 0, "bbox": [50, 500, 950, 950]},
                {"class_id": 1, "bbox": [100, 100, 200, 250]},
                # Double-clicked duplicate box (IoU > 0.92 with [100, 100, 200, 250])
                {"class_id": 1, "bbox": [101, 101, 200, 250]},
            ],
        },
        # Numeric session shelf in test (session_id="3514704") to test cross-bucket linker with marketshare-datasets
        {
            "uri": "gs://gtmerch-marketshare/test/images/3514704.jpg",
            "md5": "md5_shelf_3514704",
            "split": "test",
            "width": 1000,
            "height": 1000,
            "boxes": [
                {"class_id": 0, "bbox": [20, 20, 980, 980]},
                {"class_id": 1, "bbox": [600, 420, 690, 455]},
            ],
        },
    ]

    gt_manifest = audit_and_split_gtmerch_dataset(gt_records)
    st = gt_manifest["stats"]
    assert st["dropped_artifact_files"] == 1
    assert st["quarantined_orphan_images"] == 1
    assert st["train_test_md5_leaks_resolved"] == 1
    assert st["renamed_md5_duplicates"] == 1
    assert st["class0_box_split_edits_resolved"] == 1
    assert st["deduped_double_click_boxes"] == 1
    assert st["clean_shelf_images"] == 2
    assert st["class1_inside_class0_count"] == 2

    by_md5 = {it["md5"]: it for it in gt_manifest["items"]}
    resolved_leak = by_md5["md5_leak_box_split"]
    assert resolved_leak["split"] == "test"
    assert len(resolved_leak["asset_boxes"]) == 2
    assert len(resolved_leak["product_boxes"]) == 1
    assert resolved_leak["product_boxes"][0]["is_inside_asset"] is True
    assert resolved_leak["product_boxes"][0]["inside_asset_idx"] == 0

    # 3. Cross-Bucket Session & BBox Linker (FDE-A): synchronize split & attach SKU labels onto gtmerch shelf boxes
    crop_records = [
        {
            "uri": "gs://marketshare-datasets/input_data/variant_wash/train/Comfort__Blue_Bottle/3514704_3514704_1__605_421_691_454.jpg",
            "md5": "md5_crop_3514704_1",
        },
        # Second session so Comfort__Blue_Bottle also has a gallery anchor
        {
            "uri": "gs://marketshare-datasets/input_data/variant_wash/train/Comfort__Blue_Bottle/9990001_1.jpg",
            "md5": "md5_crop_9990001_1",
        },
    ]
    golden_payload = build_golden_json_payload(crop_records)
    # Force crop 3514704 into gallery initially to verify cross-bucket linker moves it to 'test'
    for it in golden_payload["items"]:
        if it["session_id"] == "3514704":
            it["split"] = "gallery"

    link_res = link_gtmerch_shelves_to_golden_crops(gt_manifest, golden_payload)
    assert link_res["synchronized_cross_bucket_sessions"] == 1
    assert link_res["synchronized_crop_splits"] == 1
    assert link_res["linked_product_boxes_to_golden_crops"] == 1

    crop_3514704 = next(it for it in golden_payload["items"] if it["session_id"] == "3514704")
    assert crop_3514704["split"] == "test"

    shelf_3514704 = by_md5["md5_shelf_3514704"]
    pb0 = shelf_3514704["product_boxes"][0]
    assert pb0["linked_crop_md5"] == "md5_crop_3514704_1"
    assert pb0["link_mode"] == "bbox_iou"
    assert pb0["labels"]["brand"] == "Comfort" and pb0["labels"]["packaging_type"] == "Bottle"

    # 4. Verify dataset.sample_images loads gtmerch_golden with asset_boxes
    gt_dir = tmp_path / "gtmerch_golden"
    gt_dir.mkdir()
    rpc_gt = {
        "classes": {"1": pb0["labels"]},
        "gallery": {"1": ["gs://marketshare-datasets/input_data/variant_wash/train/Comfort__Blue_Bottle/9990001_1.jpg"]},
        "splits": {
            "test": [
                {
                    "image": "3514704.jpg",
                    "uri": shelf_3514704["uri"],
                    "width": 1000,
                    "height": 1000,
                    "boxes": [pb0["bbox"]],
                    "products": [1],
                    "asset_boxes": [ab["bbox"] for ab in shelf_3514704["asset_boxes"]],
                }
            ],
            "val": [],
        },
    }
    (gt_dir / "rpc.json").write_text(json.dumps(rpc_gt))
    samples = dataset.sample_images("test", limit=5, seed=0, root=str(gt_dir), name="gtmerch_golden")
    assert len(samples) == 1
    assert len(samples[0].asset_boxes) == 1 and samples[0].asset_boxes[0] == (20, 20, 980, 980)

    # 5. Verify BigQuery gtmerch_detections_ddl and Object Table URI
    ddls = BigQueryCatalogClient().render_ddls()
    assert "gtmerch_shelf_detections" in ddls["gtmerch_detections_ddl"]
    assert "gs://gtmerch-marketshare/**" in ddls["object_table_and_models_ddl"]


def test_match_product_requires_all_evaluated_fields_and_infers_is_hul_from_brand():
    from utils import metrics

    fields = ("product", "brand", "base_variant", "category", "packaging_type", "is_hul")
    gt = {
        "product": "Comfort__Blue_Bottle",
        "brand": "Comfort",
        "base_variant": "Blue_Bottle",
        "category": "Laundry",
        "packaging_type": "Bottle",
        "is_hul": True,
    }
    pred_wrong_pack = {
        "product": "Comfort__Blue_Bottle",
        "brand": "Comfort",
        "base_variant": "Blue_Bottle",
        "category": "Laundry",
        "packaging_type": "Pouch",
    }
    m_wrong = metrics.match_product(pred_wrong_pack, gt, fields)
    assert m_wrong["correct"]["product"] is True
    assert m_wrong["correct"]["packaging_type"] is False
    assert m_wrong["correct"]["is_hul"] is True
    assert m_wrong["tp"] == 0 and m_wrong["fp"] == 1

    pred_all_right_no_is_hul = {
        "product": "Comfort__Blue_Bottle",
        "brand": "Comfort",
        "base_variant": "Blue_Bottle",
        "category": "Laundry",
        "packaging_type": "Bottle",
    }
    m_right = metrics.match_product(pred_all_right_no_is_hul, gt, fields)
    assert m_right["correct"]["is_hul"] is True
    assert m_right["tp"] == 1 and m_right["fp"] == 0

    hier = metrics.hierarchical_field_scores([{"gt": gt, "pred": pred_all_right_no_is_hul, "correct": m_right["correct"]}])
    assert hier["level1_is_hul"] == 1.0
    assert hier["compound_all_5_levels"] == 1.0


def test_generate_run_methodology_uses_gemini_schema_and_falls_back_cleanly(monkeypatch):
    import json
    import types as pytypes

    from google import genai

    import runner

    class DummyAppr:
        name = "cv_classify"
        steps = ["Brand .h5", "Pack .h5", "Variant .h5"]

    summary = {
        "run_id": "1016-093317-cv_classify-marketshare_brand.h5",
        "approach": "cv_classify",
        "task": "classification",
        "dataset": "hul_golden",
        "model": "marketshare_brand.h5",
        "evaluated_fields": ["product", "brand", "base_variant", "category", "packaging_type"],
        "accuracy": 0.4232,
        "field_accuracy": {"brand": 0.892, "packaging_type": 0.562},
        "p50_latency_s": 0.25,
        "cost_per_image_usd": 0.0001,
        "usd_to_inr": 86.2,
    }
    rows = [{"steps": [{"name": "Brand .h5", "detail": "Comfort (0.98)"}]}]

    monkeypatch.setenv("SHELF_BENCH_AUTO_METHODOLOGY", "0")
    fallback = runner.generate_run_methodology(summary, rows, appr=DummyAppr(), config={})
    assert fallback is not None and fallback["generated_by"] in ("dynamic_introspection", "deterministic_fallback")

    monkeypatch.setenv("SHELF_BENCH_AUTO_METHODOLOGY", "1")
    fake_payload = {
        "summary_header": "Run #093317 Methodology (<code>cv_classify</code>): 9-Model Chain",
        "flow_summary": "Chained InceptionV3 classifiers on <code>hul_golden</code>.",
        "bullets": [
            "<b>Stage 1:</b> Brand classification.",
            "<b>Stage 2:</b> Pack classification.",
            "<b>Stage 3:</b> Variant routing.",
        ],
    }

    class FakeModels:
        def generate_content(self, model, contents, config):
            assert model == "gemini-3.5-flash-lite"
            assert "cv_classify" in contents
            return pytypes.SimpleNamespace(text=json.dumps(fake_payload))

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    out = runner.generate_run_methodology(summary, rows, appr=DummyAppr(), config={"gcp": {"project": "test-p"}})
    assert out is not None
    assert out["summary_header"] == fake_payload["summary_header"]
    assert len(out["bullets"]) == 3
    assert out["generated_by"] == "gemini-3.5-flash-lite"
