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
        "category": "Unknown",
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
    assert f1 == {"session_id": "3401195", "crop_idx": 34, "bbox": None, "is_synthetic_copy": False}

    f2 = parse_crop_filename("processed_3514704_47.jpg")
    assert f2 == {"session_id": "3514704", "crop_idx": 47, "bbox": None, "is_synthetic_copy": False}

    f3 = parse_crop_filename("852414836922296320_852414836922296320_7__605_421_691_454.jpg")
    assert f3 == {
        "session_id": "852414836922296320",
        "crop_idx": 7,
        "bbox": (605, 421, 691, 454),
        "is_synthetic_copy": False,
    }

    f4 = parse_crop_filename("copy12.jpg")
    assert f4["is_synthetic_copy"] is True and f4["bbox"] is None

    f5 = parse_crop_filename("3401195_34 - Copy (2).jpg")
    assert f5["session_id"] == "3401195" and f5["crop_idx"] == 34 and f5["is_synthetic_copy"] is True

    # Newly discovered pack_data patterns: DA_TBC_ data augmentation, (copy), GT1_PCLW_, and class-prefixed crops
    f_da = parse_crop_filename("DA_TBC_100_10.jpg")
    assert f_da == {"session_id": "100", "crop_idx": 10, "bbox": None, "is_synthetic_copy": True}

    f_gt = parse_crop_filename("GT1_PCLW_112_66.jpg")
    assert f_gt == {"session_id": "112", "crop_idx": 66, "bbox": None, "is_synthetic_copy": False}

    f_cls = parse_crop_filename("50_1317_53.jpg")
    assert f_cls == {"session_id": "1317", "crop_idx": 53, "bbox": None, "is_synthetic_copy": False}

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




