"""Unit tests for the Federated Brand & SKU Live Registry, Parent-Company Resolver, and Temporal M&A Ledger."""

from __future__ import annotations

import json
from pathlib import Path

from cli import main as cli_main
from utils.bigquery_catalog import BIGQUERY_BRAND_SKU_REGISTRY_DDL, PARENT_COMPANY_SOS_SQL
from utils.brand_registry_crawler import (
    BRAND_TEMPORAL_OWNERSHIP,
    DEFAULT_REGISTRY_PATH,
    build_live_brand_sku_registry,
    resolve_parent_company,
)
from utils.hul_taxonomy import load_live_brand_registry, verify_brand_ownership


def test_live_brand_sku_registry_snapshot_completeness() -> None:
    """Verify the persisted or in-memory brand_sku_live_registry covers HUL and Non-HUL brands and SKUs."""
    reg = load_live_brand_registry()
    summary = reg["summary"]
    assert summary["hul_brands"] >= 60
    assert summary["non_hul_brands"] >= 350
    if DEFAULT_REGISTRY_PATH.exists():
        assert summary["total_brands"] >= 934
        assert summary["total_skus"] == 2070
    assert "Tata" in summary["dual_ownership_umbrellas"]
    assert "pureit" in summary["temporal_ma_tracked_brands"]


def test_verify_brand_ownership_hul_and_competitor_conglomerates() -> None:
    """Verify HUL and major rival FMCG parent conglomerates resolve accurately."""
    dove = verify_brand_ownership("Dove", variant="Hair_Fall_Rescue_Shampoo")
    assert dove["is_hul"] is True
    assert dove["parent_company"] == "Hindustan Unilever Limited (HUL)"

    ariel = verify_brand_ownership("Ariel", variant="Matic_Front_Load_Pouch")
    assert ariel["is_hul"] is False
    assert ariel["parent_company"] == "Procter & Gamble (P&G)"

    dettol = verify_brand_ownership("Dettol", variant="Original_Soap_Bar")
    assert dettol["is_hul"] is False
    assert dettol["parent_company"] == "Reckitt Benckiser"

    savlon = verify_brand_ownership("Savlon", variant="Moisture_Shield_Handwash")
    assert savlon["is_hul"] is False
    assert savlon["parent_company"] == "ITC Limited"

    dabur = verify_brand_ownership("Dabur", variant="Red_Toothpaste")
    assert dabur["is_hul"] is False
    assert dabur["parent_company"] == "Dabur India"

    cinthol = verify_brand_ownership("Cinthol", variant="Lime_Soap")
    assert cinthol["is_hul"] is False
    assert cinthol["parent_company"] == "Godrej Consumer Products (GCPL)"

    parachute = verify_brand_ownership("Parachute", variant="Advansed_Aloe_Vera_Hair_Oil")
    assert parachute["is_hul"] is False
    assert parachute["parent_company"] == "Marico Limited"

    colgate = verify_brand_ownership("Colgate", variant="Strong_Teeth_Toothpaste")
    assert colgate["is_hul"] is False
    assert colgate["parent_company"] == "Colgate-Palmolive"


def test_verify_brand_ownership_tata_dual_umbrella() -> None:
    """Verify Tata dual-umbrella disambiguates HUL tea/coffee SKUs vs Tata Consumer Products SKUs."""
    tata_taj = verify_brand_ownership("Tata", variant="Taj_Mahal_Rich_Taste_Carton")
    assert tata_taj["brand"] == "Tata"
    assert tata_taj["dual_umbrella"] is True
    assert tata_taj["is_hul"] is True
    assert tata_taj["parent_company"] == "Hindustan Unilever Limited (HUL)"

    tata_agni = verify_brand_ownership("Tata", variant="Agni_Leaf_Pouch")
    assert tata_agni["brand"] == "Tata"
    assert tata_agni["dual_umbrella"] is True
    assert tata_agni["is_hul"] is False
    assert tata_agni["parent_company"] == "Tata Consumer Products"

    tata_salt = verify_brand_ownership("Tata", variant="Salt_Crystal_Pouch")
    assert tata_salt["is_hul"] is False
    assert tata_salt["parent_company"] == "Tata Consumer Products"


def test_verify_brand_ownership_temporal_ma_and_divestitures() -> None:
    """Verify temporal M&A and divestiture windows (Pureit, Horlicks, Annapurna) via as_of_date."""
    assert "pureit" in BRAND_TEMPORAL_OWNERSHIP
    pureit_2023 = verify_brand_ownership("Pureit", as_of_date="2023-06-15")
    assert pureit_2023["is_hul"] is True
    assert pureit_2023["parent_company"] == "Hindustan Unilever Limited (HUL)"

    pureit_2025 = verify_brand_ownership("Pureit", as_of_date="2025-02-01")
    assert pureit_2025["is_hul"] is False
    assert pureit_2025["parent_company"] == "A. O. Smith India"

    horlicks_2018 = verify_brand_ownership("Horlicks", as_of_date="2018-05-01")
    assert horlicks_2018["is_hul"] is False
    assert horlicks_2018["parent_company"] == "GSK Consumer Healthcare"

    horlicks_2024 = verify_brand_ownership("Horlicks", as_of_date="2024-05-01")
    assert horlicks_2024["is_hul"] is True
    assert horlicks_2024["parent_company"] == "Hindustan Unilever Limited (HUL)"


def test_bigquery_brand_registry_ddl_and_cli_sync(tmp_path: Path) -> None:
    """Verify BigQuery DDL/SQL constants and CLI sync-brands subcommand."""
    assert "brand_sku_registry" in BIGQUERY_BRAND_SKU_REGISTRY_DDL
    assert "parent_company" in BIGQUERY_BRAND_SKU_REGISTRY_DDL
    assert "valid_from" in BIGQUERY_BRAND_SKU_REGISTRY_DDL
    assert "parent_company" in PARENT_COMPANY_SOS_SQL

    out_file = tmp_path / "test_registry.json"
    rc = cli_main(["sync-brands", "--output", str(out_file)])
    assert rc == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["summary"]["total_brands"] >= 400
    if DEFAULT_REGISTRY_PATH.exists():
        assert data["summary"]["total_brands"] >= 934
        assert data["summary"]["total_skus"] == 2070

    custom = build_live_brand_sku_registry(
        hul_variant_folders=["Dove__Intense_Repair_Shampoo_Bottle"],
        nh_variant_folders=["Ariel__Matic_Liquid_Pouch"],
        crawl_online=False,
    )
    assert custom["summary"]["total_skus"] == 2
    assert resolve_parent_company("Nescafe", is_hul=False) == "Nestle India"
