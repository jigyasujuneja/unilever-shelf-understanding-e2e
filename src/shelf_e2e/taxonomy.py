"""Unilever 7-Dimension Product Taxonomy, 6-Domain Variant Hierarchy, Brand Canonicalization,
25-Form-Factor Packaging & POSM Registry, and Master Base-Pack Catalog.

Covers all 6 Official Unilever Variant Classification Domains from the Scope Deck (`Model Inventory`):
  1. Hair Care - DMT
  2. Skin Care
  3. Oral Care
  4. Personal Wash - Laundry
  5. Foods - Beverages
  6. Non-HUL (Competitor Brands)
Plus the 3 Merchandising Models (`Asset SKU Detection`, `Promotion Recognition`, `Merchandising Product Recognition`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "configs" / "unilever_taxonomy.json"

UNILEVER_VARIANT_DOMAINS = [
    "Hair Care - DMT",
    "Skin Care",
    "Oral Care",
    "Personal Wash - Laundry",
    "Foods - Beverages",
    "Non-HUL",
    "Merchandising & POSM",
]

HUL_BRANDS_CANONICAL = {
    # 1. Hair Care - DMT
    "dove",
    "tresemme",
    "sunsilk",
    "clinic plus",
    "clear",
    "indulekha",
    "love beauty and planet",
    "toni&guy",
    "brylcreem",
    "lever ayush",
    "ayush",
    "nexxus",
    "sheamoisture",
    "living proof",
    "nutrafol",
    "k18",
    "suave",
    # 2. Skin Care & Color Cosmetics
    "pond's",
    "glow & lovely",
    "fair & lovely",
    "lakme",
    "vaseline",
    "simple",
    "elle 18",
    "citra",
    "st. ives",
    "minimalist",
    "dermalogica",
    "paula's choice",
    "tatcha",
    "hourglass",
    "noxzema",
    "aviance",
    "hazeline",
    "eskinol",
    # 3. Oral Care
    "closeup",
    "pepsodent",
    "signal",
    "zendium",
    # 4. Personal Wash & Deodorants + Laundry & Home Care
    "lux",
    "lifebuoy",
    "pears",
    "hamam",
    "liril",
    "motibaug",
    "jai",
    "breeze",
    "rexona",
    "sure",
    "degree",
    "shield",
    "axe",
    "lynx",
    "radox",
    "badedas",
    "duschdas",
    "schmidt's",
    "surf excel",
    "surf",
    "omo",
    "persil",
    "skip",
    "rin",
    "wheel",
    "active wheel",
    "sunlight",
    "vim",
    "cif",
    "jif",
    "domex",
    "domestos",
    "comfort",
    "snuggle",
    "coccolino",
    "ala",
    "nature protect",
    "seventh generation",
    "pureit",
    # 5. Foods, Beverages, Tea, Coffee, Nutrition & Ice Cream
    "lipton",
    "brooke bond",
    "red label",
    "taj mahal",
    "taaza",
    "3 roses",
    "pg tips",
    "pukka",
    "t2",
    "sariwangi",
    "bru",
    "horlicks",
    "junior horlicks",
    "women's horlicks",
    "boost",
    "maltova",
    "viva",
    "kissan",
    "knorr",
    "hellmann's",
    "best foods",
    "lady's choice",
    "annapurna",
    "colman's",
    "maille",
    "calve",
    "maizena",
    "bovril",
    "marmite",
    "bango",
    "royco",
    "robertsons",
    "oziva",
    "wellbeing nutrition",
    "liquid i.v.",
    "kwality wall's",
    "wall's",
    "cornetto",
    "magnum",
    "feast",
    "paddle pop",
    "ben & jerry's",
    "breyers",
    "talenti",
    "klondike",
    "popsicle",
    "viennetta",
    "carte d'or",
    "solero",
}

BRAND_ALIASES = {
    "tresemmé": "Tresemme",
    "tresemme": "Tresemme",
    "ponds": "Pond's",
    "pond": "Pond's",
    "pond's": "Pond's",
    "lakmé": "Lakme",
    "lakme": "Lakme",
    "glow and lovely": "Glow & Lovely",
    "glow & lovely": "Glow & Lovely",
    "fair & lovely": "Glow & Lovely",
    "fair and lovely": "Glow & Lovely",
    "lipton": "Lipton",
    "lipton green tea": "Lipton",
    "lipton yellow label": "Lipton",
    "brooke bond": "Brooke Bond",
    "brooke bond red label": "Red Label",
    "red label": "Red Label",
    "brooke bond taj mahal": "Taj Mahal",
    "taj mahal": "Taj Mahal",
    "brooke bond taaza": "Taaza",
    "taaza": "Taaza",
    "brooke bond 3 roses": "3 Roses",
    "3 roses": "3 Roses",
    "three roses": "3 Roses",
    "active wheel": "Wheel",
    "wheel": "Wheel",
    "surf": "Surf Excel",
    "surf excel": "Surf Excel",
    "close up": "Closeup",
    "close-up": "Closeup",
    "closeup": "Closeup",
    "clinic+": "Clinic Plus",
    "clinic plus": "Clinic Plus",
    "ayush": "Lever Ayush",
    "lever ayush": "Lever Ayush",
    "kwality walls": "Kwality Wall's",
    "kwality wall's": "Kwality Wall's",
    "walls": "Kwality Wall's",
    "wall's": "Kwality Wall's",
    "hellmanns": "Hellmann's",
    "hellmann's": "Hellmann's",
    "lbp": "Love Beauty and Planet",
    "love beauty & planet": "Love Beauty and Planet",
    "love beauty and planet": "Love Beauty and Planet",
    "st ives": "St. Ives",
    "st. ives": "St. Ives",
    "paulas choice": "Paula's Choice",
    "paula's choice": "Paula's Choice",
    "liquid iv": "Liquid I.V.",
    "liquid i.v.": "Liquid I.V.",
    "ben and jerrys": "Ben & Jerry's",
    "ben & jerry's": "Ben & Jerry's",
    "loreal": "L'Oreal",
    "l'oréal": "L'Oreal",
    "l'oreal": "L'Oreal",
    "l'oreal paris": "L'Oreal",
    "h&s": "Head & Shoulders",
    "head and shoulders": "Head & Shoulders",
    "head & shoulders": "Head & Shoulders",
}

CANONICAL_PACKAGING_TYPES = [
    "bottle",
    "pump_bottle",
    "jar",
    "tub",
    "tube",
    "pouch",
    "spout_pouch",
    "sachet",
    "sachet_strip_ladi",
    "box",
    "carton",
    "bar",
    "aerosol_can",
    "roll_on",
    "tin",
    "blister_card",
    "tetra_pak",
    "multipack",
    "dropper_serum",
    "window_header",
    "side_fin",
    "shelf_strip",
    "toker_talker",
    "parasite_hanger",
    "floor_standee",
]

PACKAGING_ALIASES = {
    "sachet_ladi": "sachet_strip_ladi",
    "ladi": "sachet_strip_ladi",
    "sachet strip": "sachet_strip_ladi",
    "hanging strip": "sachet_strip_ladi",
    "mono_carton": "box",
    "paperboard box": "box",
    "tea box": "box",
    "soap bar": "bar",
    "wrapper_bar": "bar",
    "aerosol": "aerosol_can",
    "spray": "aerosol_can",
    "deodorant spray": "aerosol_can",
    "stick": "roll_on",
    "canister": "tin",
    "can": "tin",
    "serum": "dropper_serum",
    "dropper": "dropper_serum",
    "posm_window_header": "window_header",
    "branded_window": "window_header",
    "posm_side_fin": "side_fin",
    "posm_shelf_strip": "shelf_strip",
    "channel_strip": "shelf_strip",
    "posm_toker": "toker_talker",
    "toker": "toker_talker",
    "shelf_talker": "toker_talker",
    "standee": "floor_standee",
    "fsu": "floor_standee",
}

BRAND_TO_DEFAULT_DOMAIN = {
    "Dove": "Hair Care - DMT",
    "Tresemme": "Hair Care - DMT",
    "Sunsilk": "Hair Care - DMT",
    "Clinic Plus": "Hair Care - DMT",
    "Clear": "Hair Care - DMT",
    "Indulekha": "Hair Care - DMT",
    "Love Beauty and Planet": "Hair Care - DMT",
    "Toni&Guy": "Hair Care - DMT",
    "Brylcreem": "Hair Care - DMT",
    "Lever Ayush": "Hair Care - DMT",
    "Pond's": "Skin Care",
    "Glow & Lovely": "Skin Care",
    "Lakme": "Skin Care",
    "Vaseline": "Skin Care",
    "Simple": "Skin Care",
    "Elle 18": "Skin Care",
    "Citra": "Skin Care",
    "St. Ives": "Skin Care",
    "Minimalist": "Skin Care",
    "Dermalogica": "Skin Care",
    "Paula's Choice": "Skin Care",
    "Closeup": "Oral Care",
    "Pepsodent": "Oral Care",
    "Signal": "Oral Care",
    "Zendium": "Oral Care",
    "Lux": "Personal Wash - Laundry",
    "Lifebuoy": "Personal Wash - Laundry",
    "Pears": "Personal Wash - Laundry",
    "Hamam": "Personal Wash - Laundry",
    "Liril": "Personal Wash - Laundry",
    "Motibaug": "Personal Wash - Laundry",
    "Breeze": "Personal Wash - Laundry",
    "Rexona": "Personal Wash - Laundry",
    "Axe": "Personal Wash - Laundry",
    "Radox": "Personal Wash - Laundry",
    "Surf Excel": "Personal Wash - Laundry",
    "Rin": "Personal Wash - Laundry",
    "Wheel": "Personal Wash - Laundry",
    "Sunlight": "Personal Wash - Laundry",
    "Vim": "Personal Wash - Laundry",
    "Cif": "Personal Wash - Laundry",
    "Domex": "Personal Wash - Laundry",
    "Comfort": "Personal Wash - Laundry",
    "Ala": "Personal Wash - Laundry",
    "Nature Protect": "Personal Wash - Laundry",
    "Pureit": "Personal Wash - Laundry",
    "Lipton": "Foods - Beverages",
    "Brooke Bond": "Foods - Beverages",
    "Red Label": "Foods - Beverages",
    "Taj Mahal": "Foods - Beverages",
    "Taaza": "Foods - Beverages",
    "3 Roses": "Foods - Beverages",
    "Bru": "Foods - Beverages",
    "Horlicks": "Foods - Beverages",
    "Boost": "Foods - Beverages",
    "Maltova": "Foods - Beverages",
    "Viva": "Foods - Beverages",
    "Kissan": "Foods - Beverages",
    "Knorr": "Foods - Beverages",
    "Hellmann's": "Foods - Beverages",
    "Lady's Choice": "Foods - Beverages",
    "Annapurna": "Foods - Beverages",
    "OZiva": "Foods - Beverages",
    "Wellbeing Nutrition": "Foods - Beverages",
    "Liquid I.V.": "Foods - Beverages",
    "Kwality Wall's": "Foods - Beverages",
    "Cornetto": "Foods - Beverages",
    "Magnum": "Foods - Beverages",
    "Feast": "Foods - Beverages",
    "Ben & Jerry's": "Foods - Beverages",
}

# Master Canonical HUL Base-Pack & POSM Catalog spanning all 6 Variant Classification Domains
MASTER_HUL_CATALOG: List[Dict[str, Any]] = [
    # --- Domain 1: Hair Care - DMT ---
    {"sku_id": "BP-HUL-DOVE-IR-340ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Intense Repair Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-DS-340ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Daily Shine Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-HFR-340ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Hair Fall Rescue Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-COND-180ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Intense Repair Conditioner", "size": "180ml", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-DOVE-MASK-300G", "category": "Hair Care - DMT", "brand": "Dove", "variant": "10-in-1 Deep Repair Hair Mask", "size": "300g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-DOVE-SACHET-6ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Hair Fall Rescue Sachet Ladi", "size": "6ml", "packaging_type": "sachet_strip_ladi"},
    {"sku_id": "BP-HUL-SUNSILK-BLK-340ML", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Lusciously Thick & Long / Black Shine", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SUNSILK-PINK-340ML", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Thick & Long Pink Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SUNSILK-YLW-180ML", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Nourishing Soft & Smooth", "size": "180ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SUNSILK-LADI-6MLx16", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Black Shine Hanging Sachet Ladi", "size": "6ml", "packaging_type": "sachet_strip_ladi"},
    {"sku_id": "BP-HUL-TRESEMME-KS-580ML", "category": "Hair Care - DMT", "brand": "Tresemme", "variant": "Keratin Smooth Shampoo", "size": "580ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-TRESEMME-HF-580ML", "category": "Hair Care - DMT", "brand": "Tresemme", "variant": "Hair Fall Defense Shampoo", "size": "580ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-SL-340ML", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Strong & Long Health Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-LADI-6MLx16", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Strong & Long Sachet Ladi", "size": "6ml", "packaging_type": "sachet_strip_ladi"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-SACHET-6ML", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Strong & Long Sachet", "size": "6ml", "packaging_type": "sachet"},
    {"sku_id": "BP-HUL-CLEAR-COOL-MENTHOL-330ML", "category": "Hair Care - DMT", "brand": "Clear", "variant": "Cool Sport Menthol Anti-Dandruff", "size": "330ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-INDULEKHA-BRINGHA-OIL-100ML", "category": "Hair Care - DMT", "brand": "Indulekha", "variant": "Bringha Selfie Comb Hair Oil", "size": "100ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-LBP-ARGAN-LAVENDER-400ML", "category": "Hair Care - DMT", "brand": "Love Beauty and Planet", "variant": "Argan Oil & Lavender Anti-Frizz", "size": "400ml", "packaging_type": "bottle"},

    # --- Domain 2: Skin Care & Cosmetics ---
    {"sku_id": "BP-HUL-PONDS-BRIGHT-BEAUTY-50G", "category": "Skin Care", "brand": "Pond's", "variant": "Bright Beauty Spot-less Glow Face Wash", "size": "50g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PONDS-BRIGHT-BEAUTY-100G", "category": "Skin Care", "brand": "Pond's", "variant": "Bright Beauty Spot-less Glow Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PONDS-PURE-DETOX-50G", "category": "Skin Care", "brand": "Pond's", "variant": "Pure Detox Activated Charcoal Face Wash", "size": "50g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PONDS-PURE-DETOX-100G", "category": "Skin Care", "brand": "Pond's", "variant": "Pure Detox Activated Charcoal Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PONDS-SUPER-LIGHT-GEL-100G", "category": "Skin Care", "brand": "Pond's", "variant": "Super Light Gel Oil-Free Moisturizer", "size": "100g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-PONDS-BRIGHT-MIRACLE-SERUM-30ML", "category": "Skin Care", "brand": "Pond's", "variant": "Bright Miracle Niasorcinol Serum", "size": "30ml", "packaging_type": "dropper_serum"},
    {"sku_id": "BP-HUL-PONDS-DREAMFLOWER-TALC-200G", "category": "Skin Care", "brand": "Pond's", "variant": "Dreamflower Fragrant Talc", "size": "200g", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-GAL-INSTA-GLOW-50G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "Advanced Multivitamin Insta Glow Face Wash", "size": "50g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-GAL-INSTA-GLOW-100G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "Advanced Multivitamin Insta Glow Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-GAL-GLASS-BRIGHT-100G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "Glass Bright Vitamin C Gel Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-GAL-MULTIVITAMIN-CREAM-50G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "Advanced Multivitamin Face Cream", "size": "50g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LAKME-BG-STRAWBERRY-50G", "category": "Skin Care", "brand": "Lakme", "variant": "Blush & Glow Strawberry Gel Face Wash", "size": "50g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-BG-STRAWBERRY-100G", "category": "Skin Care", "brand": "Lakme", "variant": "Blush & Glow Strawberry Gel Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-BG-KIWI-100G", "category": "Skin Care", "brand": "Lakme", "variant": "Blush & Glow Kiwi Crush Gel Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-BG-LEMON-100G", "category": "Skin Care", "brand": "Lakme", "variant": "Blush & Glow Lemon Fresh Gel Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-CC-BEIGE-30G", "category": "Skin Care", "brand": "Lakme", "variant": "9to5 CC Cream 01 Beige", "size": "30g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-CC-ALMOND-30G", "category": "Skin Care", "brand": "Lakme", "variant": "9to5 CC Cream 02 Almond", "size": "30g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-CC-HONEY-30G", "category": "Skin Care", "brand": "Lakme", "variant": "9to5 CC Cream 03 Honey", "size": "30g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-CC-BRONZE-30G", "category": "Skin Care", "brand": "Lakme", "variant": "9to5 CC Cream 04 Bronze", "size": "30g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-SUN-EXPERT-SPF50-100ML", "category": "Skin Care", "brand": "Lakme", "variant": "Sun Expert SPF 50 PA+++ Ultra Matte", "size": "100ml", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-VASELINE-IC-DEEP-RESTORE-400ML", "category": "Skin Care", "brand": "Vaseline", "variant": "Intensive Care Deep Restore Lotion", "size": "400ml", "packaging_type": "pump_bottle"},
    {"sku_id": "BP-HUL-VASELINE-COCOA-GLOW-400ML", "category": "Skin Care", "brand": "Vaseline", "variant": "Intensive Care Cocoa Glow Lotion", "size": "400ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-VASELINE-BLUESEAL-JELLY-100G", "category": "Skin Care", "brand": "Vaseline", "variant": "Original Pure Skin Jelly", "size": "100g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-SIMPLE-REFRESHING-FW-150ML", "category": "Skin Care", "brand": "Simple", "variant": "Kind to Skin Refreshing Facial Wash", "size": "150ml", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-MINIMALIST-NIACINAMIDE-10PCT-30ML", "category": "Skin Care", "brand": "Minimalist", "variant": "10% Niacinamide Face Serum", "size": "30ml", "packaging_type": "dropper_serum"},

    # --- Domain 3: Oral Care ---
    {"sku_id": "BP-HUL-CLOSEUP-EVERFRESH-RED-150G", "category": "Oral Care", "brand": "Closeup", "variant": "Everfresh Red Hot Gel Toothpaste", "size": "150g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-CLOSEUP-PEPPERMINT-TUBE-80G", "category": "Oral Care", "brand": "Closeup", "variant": "Everfresh Peppermint Splash", "size": "80g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PEPSODENT-GERMICHECK-150G", "category": "Oral Care", "brand": "Pepsodent", "variant": "Germicheck 8-Action Cavity Protection", "size": "150g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-PEPSODENT-2IN1-150G", "category": "Oral Care", "brand": "Pepsodent", "variant": "2-in-1 Paste + Gel", "size": "150g", "packaging_type": "tube"},

    # --- Domain 4: Personal Wash - Laundry & Home Care ---
    {"sku_id": "BP-HUL-PEARS-PURE-GENTLE-100G", "category": "Skin Care", "brand": "Pears", "variant": "Pure & Gentle Glycerine Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PEARS-OIL-CLEAR-100G", "category": "Skin Care", "brand": "Pears", "variant": "Oil Clear Mint Extract Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PEARS-SOFT-FRESH-100G", "category": "Skin Care", "brand": "Pears", "variant": "Soft & Fresh Blue Mint Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PEARS-SOAP-BAR-125G", "category": "Personal Wash - Laundry", "brand": "Pears", "variant": "Pure & Gentle Bathing Bar", "size": "125g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LUX-ROSE-VIT-E-100G", "category": "Personal Wash - Laundry", "brand": "Lux", "variant": "Velvet Glow Rose & Vitamin E Soap Bar", "size": "100g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-LUX-JASMINE-MULTIPACK-4x100G", "category": "Personal Wash - Laundry", "brand": "Lux", "variant": "Jasmine & Vitamin E 4-Bar Multipack", "size": "400g", "packaging_type": "multipack"},
    {"sku_id": "BP-HUL-LIFEBUOY-TOTAL10-BAR-125G", "category": "Personal Wash - Laundry", "brand": "Lifebuoy", "variant": "Total 10 Germ Protection Soap Bar", "size": "125g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-LIFEBUOY-HANDWASH-POUCH-750ML", "category": "Personal Wash - Laundry", "brand": "Lifebuoy", "variant": "Total 10 Handwash Refill Pouch", "size": "750ml", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-DOVE-CREAM-BAR-100G", "category": "Personal Wash - Laundry", "brand": "Dove", "variant": "Cream Beauty Bathing Bar", "size": "100g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-DOVE-BW-500", "category": "Personal Wash - Laundry", "brand": "Dove", "variant": "Deeply Nourishing Body Wash", "size": "500ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-BW-750", "category": "Personal Wash - Laundry", "brand": "Dove", "variant": "Deeply Nourishing Body Wash", "size": "750ml", "packaging_type": "pump_bottle"},
    {"sku_id": "BP-HUL-DOVE-HW-500-POUCH", "category": "Personal Wash - Laundry", "brand": "Dove", "variant": "Deep Moisture Refill Pouch", "size": "500ml", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-HAMAM-NEEM-TULSI-100G", "category": "Personal Wash - Laundry", "brand": "Hamam", "variant": "100% Pure Neem, Tulsi & Aloe Vera Bar", "size": "100g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-REXONA-ROLLON-50ML", "category": "Personal Wash - Laundry", "brand": "Rexona", "variant": "Powder Dry Underarm Roll-On", "size": "50ml", "packaging_type": "roll_on"},
    {"sku_id": "BP-HUL-AXE-DARK-TEMPTATION-150ML", "category": "Personal Wash - Laundry", "brand": "Axe", "variant": "Dark Temptation Deodorant Body Spray", "size": "150ml", "packaging_type": "aerosol_can"},
    {"sku_id": "BP-HUL-SURF-EXCEL-QUICKWASH-500G", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Quick Wash Detergent Powder", "size": "500g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-SURF-EXCEL-EASYWASH-1KG", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Easy Wash Detergent Powder", "size": "1kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-SURF-EXCEL-MATIC-LIQ-1L", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Matic Top Load Liquid Detergent", "size": "1L", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SURF-EXCEL-MATIC-POUCH-2L", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Matic Front Load Liquid Spout Pouch", "size": "2L", "packaging_type": "spout_pouch"},
    {"sku_id": "BP-HUL-SURF-EXCEL-BAR-250G", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Stain Eraser Detergent Bar", "size": "250g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-RIN-ADVANCED-POWDER-1KG", "category": "Personal Wash - Laundry", "brand": "Rin", "variant": "Advanced Brightening Detergent Powder", "size": "1kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-RIN-BAR-250G", "category": "Personal Wash - Laundry", "brand": "Rin", "variant": "Dazzling White Detergent Bar", "size": "250g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-WHEEL-ACTIVE-2IN1-1KG", "category": "Personal Wash - Laundry", "brand": "Wheel", "variant": "Active 2-in-1 Lemon & Jasmine Powder", "size": "1kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-VIM-DISHWASH-GEL-750ML", "category": "Personal Wash - Laundry", "brand": "Vim", "variant": "Dishwash Liquid Gel Power of 100 Lemons", "size": "750ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-VIM-BAR-300G", "category": "Personal Wash - Laundry", "brand": "Vim", "variant": "Lemon Dishwash Bar", "size": "300g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-DOMEX-FRESH-GUARD-500ML", "category": "Personal Wash - Laundry", "brand": "Domex", "variant": "Fresh Guard Disinfectant Toilet Cleaner", "size": "500ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-COMFORT-MORNING-FRESH-860ML", "category": "Personal Wash - Laundry", "brand": "Comfort", "variant": "After Wash Morning Fresh Fabric Conditioner", "size": "860ml", "packaging_type": "bottle"},

    # --- Domain 5: Foods - Beverages, Tea, Coffee, Nutrition & Ice Cream ---
    {"sku_id": "BP-HUL-LIPTON-GREEN-TEA-25TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Pure & Light Green Tea 25 Tea Bags", "size": "35g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-HONEY-LEMON-25TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Honey Lemon Green Tea 25 Tea Bags", "size": "35g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-TULSI-NATURO-25TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Tulsi Naturo Green Tea 25 Tea Bags", "size": "35g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-LEMON-ZEST-10TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Lemon Zest Green Tea 10 Tea Bags", "size": "14g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-MINT-BURST-25TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Mint Burst Green Tea 25 Tea Bags", "size": "35g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-YELLOW-LABEL-250G", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Yellow Label Finest Black Tea Carton", "size": "250g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-RED-LABEL-NATURAL-CARE-500G", "category": "Foods - Beverages", "brand": "Red Label", "variant": "Brooke Bond Natural Care 5-Herbs Tea", "size": "500g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-RED-LABEL-POUCH-250G", "category": "Foods - Beverages", "brand": "Red Label", "variant": "Brooke Bond Red Label Tea Pouch", "size": "250g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-TAJ-MAHAL-TEA-250G", "category": "Foods - Beverages", "brand": "Taj Mahal", "variant": "Brooke Bond Taj Mahal Rich & Flavourful Tea", "size": "250g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-TAAZA-MASALA-CHAI-250G", "category": "Foods - Beverages", "brand": "Taaza", "variant": "Brooke Bond Taaza Leaf Tea Pouch", "size": "250g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-3ROSES-TOP-STAR-250G", "category": "Foods - Beverages", "brand": "3 Roses", "variant": "Brooke Bond 3 Roses Dust Tea", "size": "250g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-BRU-INSTANT-POUCH-100G", "category": "Foods - Beverages", "brand": "Bru", "variant": "Instant Chicory Blend Coffee Pouch", "size": "100g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-BRU-GOLD-JAR-100G", "category": "Foods - Beverages", "brand": "Bru", "variant": "Gold 100% Pure Freeze-Dried Coffee Jar", "size": "100g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-HORLICKS-CLASSIC-MALT-500G", "category": "Foods - Beverages", "brand": "Horlicks", "variant": "Classic Malt Health & Nutrition Drink", "size": "500g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-HORLICKS-WOMENS-PLUS-400G", "category": "Foods - Beverages", "brand": "Horlicks", "variant": "Women's Plus Caramel Bone Health", "size": "400g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-BOOST-ENERGY-JAR-500G", "category": "Foods - Beverages", "brand": "Boost", "variant": "3X Stamina Malt Energy Drink Jar", "size": "500g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-KISSAN-FRESH-TOMATO-KETCHUP-950G", "category": "Foods - Beverages", "brand": "Kissan", "variant": "Fresh Tomato Ketchup Spout Pouch", "size": "950g", "packaging_type": "spout_pouch"},
    {"sku_id": "BP-HUL-KISSAN-MIXED-FRUIT-JAM-500G", "category": "Foods - Beverages", "brand": "Kissan", "variant": "Mixed Fruit Jam Glass Jar", "size": "500g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-KNORR-CLASSIC-TOMATO-SOUP-53G", "category": "Foods - Beverages", "brand": "Knorr", "variant": "Classic Thick Tomato Soup Pouch", "size": "53g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-HELLMANNS-REAL-MAYO-400G", "category": "Foods - Beverages", "brand": "Hellmann's", "variant": "Real Mayonnaise Jar", "size": "400g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-KWALITY-WALLS-SHAMELESS-VANILLA-700ML", "category": "Foods - Beverages", "brand": "Kwality Wall's", "variant": "Vanilla Frozen Dessert Party Tub", "size": "700ml", "packaging_type": "tub"},

    # --- Domain 6: Merchandising & POSM 6-Asset Registry ---
    {"sku_id": "POSM-HUL-LIPTON-WINDOW-HEADER", "category": "Merchandising & POSM", "brand": "Lipton", "variant": "Reduce Belly Fat With Tasty Green Tea Window Header", "size": "POSM", "packaging_type": "window_header"},
    {"sku_id": "POSM-HUL-LIPTON-REF-ASSET", "category": "Merchandising & POSM", "brand": "Lipton", "variant": "Reference Business Promotion Planogram Header & Fins", "size": "POSM", "packaging_type": "window_header"},
    {"sku_id": "POSM-HUL-LAKME-PONDS-SHELF-STRIP", "category": "Merchandising & POSM", "brand": "Lakme", "variant": "Detox Facewash / Lakme Expert Face Cleansers Shelf Strip", "size": "POSM", "packaging_type": "shelf_strip"},
    {"sku_id": "POSM-HUL-DOVE-PROMO-TOKER", "category": "Merchandising & POSM", "brand": "Dove", "variant": "Promotional Price-Off Rail Toker / Talker", "size": "POSM", "packaging_type": "toker_talker"},
]


@dataclass(frozen=True)
class SevenDimensionProductAttributes:
    """Official Unilever 7-Dimension shelf recognition hierarchy."""

    category: str
    subcategory: str
    brand: str
    is_hul_brand: bool
    variant: str
    packaging_type: str
    pack_type: str
    rule_derived_size_bucket: str


def normalize_brand_and_hul_flag(raw_brand: Optional[str]) -> Tuple[str, bool]:
    """Canonicalize brand names and deterministically resolve HUL ownership (`is_hul_brand`)."""
    if not raw_brand or not raw_brand.strip():
        return ("Unknown", False)
    cleaned = re.sub(r"\s+", " ", raw_brand.strip())
    lower = cleaned.lower().replace("’", "'")
    canonical = BRAND_ALIASES.get(lower, cleaned)
    is_hul = canonical.lower() in HUL_BRANDS_CANONICAL
    if not is_hul:
        # Check prefix/substring match against canonical multi-word HUL brands (e.g., "Brooke Bond Red Label", "Lipton Green Tea")
        for hul_b in HUL_BRANDS_CANONICAL:
            if len(hul_b) >= 4 and re.search(rf"\b{re.escape(hul_b)}\b", lower):
                canonical = BRAND_ALIASES.get(hul_b, hul_b.title())
                is_hul = True
                break
    return (canonical, is_hul)


def normalize_packaging_type(raw_packaging: Optional[str]) -> str:
    """Canonicalize packaging or POSM form factor across all 25 Unilever form factors."""
    if not raw_packaging or not raw_packaging.strip():
        return "bottle"
    lower = raw_packaging.strip().lower().replace("-", "_").replace(" ", "_")
    if lower in CANONICAL_PACKAGING_TYPES:
        return lower
    raw_space = raw_packaging.strip().lower()
    if raw_space in PACKAGING_ALIASES:
        return PACKAGING_ALIASES[raw_space]
    if lower in PACKAGING_ALIASES:
        return PACKAGING_ALIASES[lower]
    for k, v in PACKAGING_ALIASES.items():
        if k in raw_space:
            return v
    return "bottle"


def resolve_variant_domain(brand: str, hint_category: str = "", packaging_type: str = "") -> str:
    """Map any product into one of the 6 Official Unilever Variant Classification Domains (+ Merchandising)."""
    canonical_brand, is_hul = normalize_brand_and_hul_flag(brand)
    pkg = normalize_packaging_type(packaging_type)
    if pkg in ("window_header", "side_fin", "shelf_strip", "toker_talker", "parasite_hanger", "floor_standee"):
        return "Merchandising & POSM"
    if not is_hul:
        return "Non-HUL"
    if canonical_brand in BRAND_TO_DEFAULT_DOMAIN:
        # Check if Dove/Pears is in Skin Care or Personal Wash based on hint_category/packaging
        if canonical_brand == "Pears" and ("face" in hint_category.lower() or pkg == "tube"):
            return "Skin Care"
        if canonical_brand == "Dove" and ("bar" in pkg or "pouch" in pkg or "wash" in hint_category.lower()):
            return "Personal Wash - Laundry"
        return BRAND_TO_DEFAULT_DOMAIN[canonical_brand]
    return hint_category or "Personal Care"


def resolve_or_synthesize_base_pack(
    brand: str,
    category: str,
    packaging_type: str,
    variant: str = "",
    size_text: str = "",
    explicit_sku_id: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Resolve an exact HUL Base-Pack code from MASTER_HUL_CATALOG or dynamically synthesize a canonical
    Base-Pack ID (`BP-HUL-<BRAND>-<VARIANT>-<SIZE>`) for novel variants/sizes so zero SKUs are ever missed.
    """
    canonical_brand, is_hul = normalize_brand_and_hul_flag(brand)
    pkg = normalize_packaging_type(packaging_type)

    if not is_hul:
        clean_cat = "".join(ch for ch in (category or "GEN").upper() if ch.isalnum())[:4]
        clean_br = "".join(ch for ch in canonical_brand.upper() if ch.isalnum())[:6]
        clean_sz = "".join(ch for ch in (size_text or "STD").upper() if ch.isalnum())[:5]
        non_hul_id = explicit_sku_id or f"NON-HUL-{clean_cat}-{clean_br}-{clean_sz}"
        return (non_hul_id, [])

    # Filter catalog by canonical brand
    brand_candidates = [
        item for item in MASTER_HUL_CATALOG
        if normalize_brand_and_hul_flag(item["brand"])[0].lower() == canonical_brand.lower()
    ]
    # Further score candidates by packaging type, variant token overlap, and size match
    var_tokens = {t for t in re.split(r"[^a-z0-9]+", (variant or "").lower()) if len(t) >= 3}
    size_clean = (size_text or "").lower().strip()

    scored: List[Tuple[float, str]] = []
    for item in brand_candidates:
        score = 1.0
        item_pkg = normalize_packaging_type(item["packaging_type"])
        if item_pkg == pkg:
            score += 3.0
        elif {item_pkg, pkg} <= {"sachet", "sachet_strip_ladi", "pouch", "spout_pouch"}:
            score += 1.5
        elif {item_pkg, pkg} <= {"box", "carton", "multipack"}:
            score += 1.5
        item_var_tokens = {t for t in re.split(r"[^a-z0-9]+", item.get("variant", "").lower()) if len(t) >= 3}
        overlap = len(var_tokens & item_var_tokens)
        score += overlap * 2.0
        if size_clean and size_clean in item.get("size", "").lower():
            score += 2.5
        scored.append((score, item["sku_id"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidate_ids = [sku for _, sku in scored[:8]]

    if explicit_sku_id:
        if explicit_sku_id not in candidate_ids:
            candidate_ids.insert(0, explicit_sku_id)
        return (explicit_sku_id, candidate_ids[:8])

    if scored and scored[0][0] >= 5.5:
        return (scored[0][1], candidate_ids[:8])

    # Open-vocabulary synthesis for valid HUL brand + unseen variant/size
    br_slug = re.sub(r"[^A-Z0-9]+", "", canonical_brand.upper())[:8]
    var_words = [w.upper()[:5] for w in re.split(r"[^a-zA-Z0-9]+", variant or pkg) if len(w) >= 2][:2]
    var_slug = "-".join(var_words) if var_words else pkg.upper()[:6]
    sz_slug = re.sub(r"[^A-Z0-9]+", "", (size_text or "STD").upper())[:6]
    prefix = "POSM-HUL" if pkg in ("window_header", "side_fin", "shelf_strip", "toker_talker", "parasite_hanger", "floor_standee") else "BP-HUL"
    synthesized = f"{prefix}-{br_slug}-{var_slug}-{sz_slug}"
    if synthesized not in candidate_ids:
        candidate_ids.insert(0, synthesized)
    return (synthesized, candidate_ids[:8])


def resolve_rule_derived_size_bucket(
    product_name_or_size_text: str = "",
    bbox_2d: Optional[List[int]] = None,
) -> str:
    """Deterministic size bucket resolver (Small <=55g/ml, Medium 56-110g/ml, Large >110g/ml).

    Uses explicit ml/g/tb regex extraction first, falling back to normalized bounding-box height priors.
    """
    text = (product_name_or_size_text or "").lower()
    tb_match = re.search(r"(\d+)\s*(?:tb|teabags|tea\s*bags)\b", text)
    if tb_match:
        cnt = int(tb_match.group(1))
        if cnt <= 10:
            return "Small / Trial / Sachet (<=55g/ml)"
        if cnt <= 25:
            return "Medium / Regular (56-110g/ml)"
        return "Large / Family (>110g/ml)"

    match = re.search(r"(\d+(?:\.\d+)?)\s*(ml|g|gm|grams|l|kg)\b", text)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit in ("l", "kg"):
            val *= 1000.0
        if val <= 55.0:
            return "Small / Trial / Sachet (<=55g/ml)"
        if val <= 110.0:
            return "Medium / Regular (56-110g/ml)"
        return "Large / Family (>110g/ml)"

    if bbox_2d and len(bbox_2d) == 4:
        height_norm = max(1, bbox_2d[2] - bbox_2d[0])
        if height_norm <= 48:
            return "Small / Trial / Sachet (<=55g/ml)"
        if height_norm <= 85:
            return "Medium / Regular (56-110g/ml)"
    return "Large / Family (>110g/ml)"


def enrich_with_7dim_taxonomy(
    catalog_entry: Dict[str, Any],
    bbox_2d: Optional[List[int]] = None,
) -> SevenDimensionProductAttributes:
    """Convert a catalog item + bounding box into a complete 7-Dimension Unilever attribute record."""
    brand_raw = str(catalog_entry.get("brand", "Unknown"))
    canonical_brand, is_hul = normalize_brand_and_hul_flag(brand_raw)
    prod_name = str(catalog_entry.get("product_name", "") or catalog_entry.get("variant", ""))
    size_bucket = str(
        catalog_entry.get("size_bucket")
        or resolve_rule_derived_size_bucket(prod_name, bbox_2d)
    )
    pkg = normalize_packaging_type(str(catalog_entry.get("packaging_type", "bottle")))
    cat = str(catalog_entry.get("category") or resolve_variant_domain(canonical_brand, "", pkg))
    return SevenDimensionProductAttributes(
        category=cat,
        subcategory=str(catalog_entry.get("subcategory", "General")),
        brand=canonical_brand,
        is_hul_brand=bool(catalog_entry.get("is_hul_brand", is_hul)),
        variant=str(catalog_entry.get("variant", "Standard")),
        packaging_type=pkg,
        pack_type=str(catalog_entry.get("pack_type", "Multipack" if pkg in ("sachet_strip_ladi", "multipack") else "Single")),
        rule_derived_size_bucket=size_bucket,
    )
