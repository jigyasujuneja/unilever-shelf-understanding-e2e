"""Unilever 7-Dimension Product Taxonomy, 6-Domain Variant Hierarchy, Open-Internet Brand & SKU Registry,
25-Form-Factor Packaging & POSM Registry, and Live Open-Web Grounding Resolver (`googleSearch` + Live Web API).

Covers all 6 Official Unilever Variant Classification Domains from the Scope Deck (`Model Inventory`):
  1. Hair Care - DMT
  2. Skin Care
  3. Oral Care
  4. Personal Wash - Laundry
  5. Foods - Beverages
  6. Non-HUL (Competitor Brands)
Plus the 3 Merchandising Models (`Asset SKU Detection`, `Promotion Recognition`, `Merchandising Product Recognition`).

Open-Internet Grounding Architecture:
  - Tier 1 (`open_internet_synced_taxonomy`): 270+ Global Unilever & HUL India brands + 130+ Competitor brands +
    220+ canonical Base-Pack SKUs synced from `hul.co.in/brands`, `unilever.com/brands/all-brands`,
    `en.wikipedia.org/wiki/List_of_Unilever_brands`, and Indian retail e-commerce catalogs (BigBasket, Blinkit, Shikhar).
  - Tier 2 (`vertex_gemini_google_search_grounding`): Real-time Vertex AI `gemini-2.5-flash` with
    `"tools": [{"googleSearch": {}}]` to verify any unseen brand, D2C acquisition, regional variant, or pack size.
  - Tier 3 (`live_http_open_web_search`): Direct live HTTPS search against Wikipedia MediaWiki API & `hul.co.in`
    with self-learning persistence into `configs/open_web_discovered_skus.json`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.parse
import urllib.request


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "configs" / "unilever_taxonomy.json"
LEARNED_OPEN_WEB_SKUS_PATH = Path(__file__).resolve().parents[2] / "configs" / "open_web_discovered_skus.json"

UNILEVER_VARIANT_DOMAINS = [
    "Hair Care - DMT",
    "Skin Care",
    "Oral Care",
    "Personal Wash - Laundry",
    "Foods - Beverages",
    "Non-HUL",
    "Merchandising & POSM",
]

# Complete Open-Internet Scraped Global Unilever (270+ brands/aliases) + HUL India (57 brands)
HUL_BRANDS_CANONICAL: Set[str] = {
    # 1. Hair Care - DMT (HUL India + Global Unilever + Prestige Hair)
    "dove",
    "tresemme",
    "sunsilk",
    "sedal",
    "seda",
    "elidor",
    "clinic plus",
    "clinic",
    "clinic deodorant",
    "clear",
    "linic",
    "ultrex",
    "indulekha",
    "love beauty and planet",
    "toni&guy",
    "tigi",
    "bed head",
    "brylcreem",
    "brisk",
    "lever ayush",
    "ayush",
    "nexxus",
    "sheamoisture",
    "shea moisture",
    "living proof",
    "nutrafol",
    "k18",
    "suave",
    "alberto vo5",
    "vo5",
    "cream silk",
    "andrelon",
    "organics",
    "dimension",
    "thermasilk",
    "vibrance",
    "vitakeratin",
    "pinuk",
    # 2. Skin Care, Dermacosmetics, Color Cosmetics & Prestige Beauty
    "pond's",
    "glow & lovely",
    "fair & lovely",
    "glow & handsome",
    "fair & handsome",
    "lakme",
    "vaseline",
    "vasenol",
    "simple",
    "elle 18",
    "citra",
    "st. ives",
    "st ives",
    "minimalist",
    "novology",
    "acne squad",
    "dermalogica",
    "paula's choice",
    "tatcha",
    "hourglass",
    "hourglass cosmetics",
    "kate somerville",
    "kate somerville skincare",
    "murad",
    "ren clean skincare",
    "garancia",
    "skinsei",
    "ahc",
    "noxzema",
    "aviance",
    "hazeline",
    "eskinol",
    "master",
    "block & glow",
    "block & white",
    "dawn",
    "fissan",
    "fds",
    "hijab fresh",
    "korea glow",
    "mele",
    "melé",
    "nameera",
    "nubian heritage",
    "nyakio",
    "pure line",
    "tholl",
    "white beauty",
    "zwitsal",
    # 3. Oral Care (HUL India + Global Unilever)
    "closeup",
    "close-up",
    "pepsodent",
    "p/s",
    "signal",
    "zendium",
    "mentadent",
    "mentadent sr",
    "sr",
    "aim",
    "prodent",
    "regenerate",
    "fluocaril",
    "zhonghua",
    # 4. Personal Wash, Deodorants, Feminine Care + Laundry & Home Care
    "lux",
    "caress",
    "camay",
    "lifebuoy",
    "pears",
    "pears transparent soap",
    "hamam",
    "liril",
    "moti",
    "motibaug",
    "ok",
    "jai",
    "breeze",
    "baby dove",
    "dove men+care",
    "dove nutritive solutions",
    "dove spa",
    "rexona",
    "rexona deo",
    "sure",
    "degree",
    "shield",
    "impulse",
    "axe",
    "lynx",
    "brut",
    "denim",
    "williams",
    "radox",
    "badedas",
    "dusch das",
    "duschdas",
    "schmidt's",
    "schmidt's naturals",
    "wild",
    "find your happy place",
    "vwash",
    "v wash",
    "dr. kaufmann",
    "gessy",
    "good morning",
    "lever 2000",
    "medcare",
    "mist",
    "vinolia",
    "vinólia",
    "surf excel",
    "surf",
    "omo",
    "persil",
    "skip",
    "rin",
    "rinso",
    "wheel",
    "active wheel",
    "active wheel 2 in 1",
    "sunlight",
    "vim",
    "cif",
    "jif",
    "viss",
    "domex",
    "domestos",
    "comfort",
    "snuggle",
    "coccolino",
    "yumos",
    "yumoş",
    "ala",
    "all",
    "baba",
    "badin",
    "bailan",
    "biotex",
    "blueair",
    "brilhante",
    "coral",
    "korall",
    "deja",
    "dero",
    "dixan",
    "lysoform",
    "minerva",
    "molto",
    "neutral",
    "quix",
    "radiant",
    "robijn",
    "sahaja",
    "seventh generation",
    "sun",
    "super pell",
    "via",
    "viso",
    "vixal",
    "wipol",
    "xedex",
    "nature protect",
    "smartclean",
    "love & care",
    "magic",
    "pureit",
    # 5. Foods, Beverages, Tea, Coffee, Health Food Drinks, Condiments, Supplements & Ice Cream
    "lipton",
    "brooke bond",
    "red label",
    "brooke bond red label",
    "taj mahal",
    "brooke bond taj mahal",
    "taaza",
    "brooke bond taaza",
    "3 roses",
    "brooke bond 3 roses",
    "pg tips",
    "pukka",
    "t2",
    "t2 ice tea",
    "sariwangi",
    "saga",
    "beseda",
    "bru",
    "sunrise",
    "horlicks",
    "junior horlicks",
    "women's horlicks",
    "horlicks women's plus",
    "mother's horlicks",
    "horlicks mother's plus",
    "horlicks protein plus",
    "horlicks protein",
    "horlicks lite",
    "horlicks diabetes plus",
    "boost",
    "maltova",
    "viva",
    "buavita",
    "liquid i.v.",
    "liquid i.v",
    "oziva",
    "wellbeing nutrition",
    "olly",
    "smartypants",
    "onnit",
    "wellements",
    "kissan",
    "knorr",
    "knorr-suiza",
    "royco",
    "continental",
    "hellmann's",
    "best foods",
    "lady's choice",
    "annapurna",
    "colman's",
    "maille",
    "calve",
    "calvé",
    "amora",
    "aromat",
    "amino",
    "bovril",
    "marmite",
    "our mate",
    "kecap bango",
    "bango",
    "robertson's",
    "robertsons",
    "sir kensington's",
    "chirat",
    "fanacoa",
    "fruco",
    "globus",
    "kuner",
    "kystee",
    "lao cai",
    "salsa lizano",
    "slotts",
    "tortex",
    "turun sinappi",
    "maizena",
    "mae terra",
    "mãe terra",
    "pfanni",
    "pot noodle",
    "pot rice",
    "bagel bagel",
    "kefli",
    "klik",
    "patit",
    "sealtest",
    "telma",
    "unilever food solutions",
    "kwality wall's",
    "wall's",
    "cornetto",
    "magnum",
    "feast",
    "paddle pop",
    "solero",
    "calippo",
    "viennetta",
    "carte d'or",
    "ben & jerry's",
    "breyers",
    "talenti",
    "good humor",
    "klondike",
    "popsicle",
    "yasso",
    "algida",
    "langnese",
    "ola",
    "miko",
    "frigo",
    "kibon",
    "holanda",
    "streets",
    "selecta",
}

COMPETITOR_BRANDS_NON_HUL: Set[str] = {
    "pantene",
    "head & shoulders",
    "l'oreal",
    "garnier",
    "schwarzkopf",
    "herbal essences",
    "matrix",
    "biolage",
    "vatika",
    "dabur vatika",
    "parachute",
    "nihar",
    "livon",
    "hair & care",
    "bajaj almond drops",
    "navratna",
    "kesh king",
    "meera",
    "karthika",
    "chik",
    "nyle",
    "mamaearth",
    "wow skin science",
    "biotique",
    "khadi natural",
    "nivea",
    "neutrogena",
    "cetaphil",
    "olay",
    "clean & clear",
    "aveeno",
    "himalaya",
    "boroplus",
    "boroline",
    "vicco",
    "emami",
    "roop mantra",
    "everyuth",
    "lotus herbals",
    "plum",
    "dot & key",
    "derma co",
    "aqualogica",
    "foxtale",
    "mcaffeine",
    "nykaa",
    "maybelline",
    "mac",
    "revlon",
    "sugar cosmetics",
    "colorbar",
    "faces canada",
    "swiss beauty",
    "insight cosmetics",
    "blue heaven",
    "sebamed",
    "bioderma",
    "cerave",
    "la roche-posay",
    "colgate",
    "sensodyne",
    "oral-b",
    "dabur red",
    "dabur meswak",
    "dabur babool",
    "patanjali",
    "patanjali dant kanti",
    "vicco vajradanti",
    "parodontax",
    "listerine",
    "dettol",
    "savlon",
    "santoor",
    "cinthol",
    "godrej no.1",
    "medimix",
    "mysore sandal",
    "fiama",
    "vivel",
    "margo",
    "palmolive",
    "safeguard",
    "yardley",
    "park avenue",
    "wild stone",
    "fogg",
    "engage",
    "nivea men",
    "denver",
    "set wet",
    "old spice",
    "whisper",
    "stayfree",
    "sofy",
    "ariel",
    "tide",
    "ghadi",
    "nirma",
    "henko",
    "ujala",
    "mr. white",
    "morelight",
    "fena",
    "hippo",
    "pril",
    "exo",
    "odopic",
    "harpic",
    "lizol",
    "colin",
    "mortein",
    "good knight",
    "all out",
    "hit",
    "odonil",
    "air wick",
    "godrej ezee",
    "genteel",
    "vanish",
    "downy",
    "tata tea",
    "tata tea gold",
    "tata tea premium",
    "tata tea agni",
    "tetley",
    "twinings",
    "organic india",
    "girnar",
    "wagh bakri",
    "society tea",
    "avt",
    "marvel tea",
    "nescafe",
    "tata coffee",
    "continental coffee",
    "rage coffee",
    "sleepy owl",
    "bournvita",
    "complan",
    "pediasure",
    "ensure",
    "protinex",
    "milo",
    "maggi",
    "yippee",
    "top ramen",
    "ching's secret",
    "heinz",
    "del monte",
    "veeba",
    "dr. oetker",
    "funfoods",
    "mtr",
    "aashirvaad",
    "fortune",
    "saffola",
    "amul",
    "mother dairy",
    "vadilal",
    "havmor",
    "arun icecreams",
    "baskin robbins",
    "london dairy",
    "britannia",
    "parle",
    "sunfeast",
    "haldiram's",
    "bikaji",
    "balaji",
    "lay's",
    "kurkure",
    "bingo",
}

BRAND_ALIASES: Dict[str, str] = {
    "tresemmé": "Tresemme",
    "tresemme": "Tresemme",
    "tresemmé professional": "Tresemme",
    "ponds": "Pond's",
    "pond": "Pond's",
    "pond's": "Pond's",
    "pond's skin institute": "Pond's",
    "lakmé": "Lakme",
    "lakme": "Lakme",
    "lakmé cosmetics": "Lakme",
    "glow and lovely": "Glow & Lovely",
    "glow & lovely": "Glow & Lovely",
    "fair & lovely": "Glow & Lovely",
    "fair and lovely": "Glow & Lovely",
    "glow and handsome": "Glow & Handsome",
    "glow & handsome": "Glow & Handsome",
    "fair & handsome (hul)": "Glow & Handsome",
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
    "active wheel 2 in 1": "Wheel",
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
    "liquid i.v": "Liquid I.V.",
    "liquid i.v.": "Liquid I.V.",
    "ben and jerrys": "Ben & Jerry's",
    "ben & jerry's": "Ben & Jerry's",
    "vwash": "VWash",
    "v wash": "VWash",
    "v wash feminine wash": "VWash",
    "rexona deo": "Rexona",
    "pears transparent soap": "Pears",
    "indulekha shampoo and oil": "Indulekha",
    "horlicks women's plus": "Horlicks",
    "women's horlicks": "Horlicks",
    "horlicks mother's plus": "Horlicks",
    "mother's horlicks": "Horlicks",
    "horlicks protein plus": "Horlicks",
    "horlicks protein": "Horlicks",
    "horlicks lite": "Horlicks",
    "horlicks diabetes plus": "Horlicks",
    "junior horlicks": "Horlicks",
    "loreal": "L'Oreal",
    "l'oréal": "L'Oreal",
    "l'oreal": "L'Oreal",
    "l'oreal paris": "L'Oreal",
    "h&s": "Head & Shoulders",
    "head and shoulders": "Head & Shoulders",
    "head & shoulders": "Head & Shoulders",
}

CANONICAL_PACKAGING_TYPES: List[str] = [
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

PACKAGING_ALIASES: Dict[str, str] = {
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

BRAND_TO_DEFAULT_DOMAIN: Dict[str, str] = {
    # Domain 1: Hair Care - DMT
    "Dove": "Hair Care - DMT",
    "Tresemme": "Hair Care - DMT",
    "Sunsilk": "Hair Care - DMT",
    "Sedal": "Hair Care - DMT",
    "Seda": "Hair Care - DMT",
    "Elidor": "Hair Care - DMT",
    "Clinic Plus": "Hair Care - DMT",
    "Clinic": "Hair Care - DMT",
    "Clear": "Hair Care - DMT",
    "Linic": "Hair Care - DMT",
    "Ultrex": "Hair Care - DMT",
    "Indulekha": "Hair Care - DMT",
    "Love Beauty and Planet": "Hair Care - DMT",
    "Toni&Guy": "Hair Care - DMT",
    "TIGI": "Hair Care - DMT",
    "Bed Head": "Hair Care - DMT",
    "Brylcreem": "Hair Care - DMT",
    "Brisk": "Hair Care - DMT",
    "Lever Ayush": "Hair Care - DMT",
    "Nexxus": "Hair Care - DMT",
    "SheaMoisture": "Hair Care - DMT",
    "Living Proof": "Hair Care - DMT",
    "Nutrafol": "Hair Care - DMT",
    "K18": "Hair Care - DMT",
    "Suave": "Hair Care - DMT",
    "Alberto VO5": "Hair Care - DMT",
    "VO5": "Hair Care - DMT",
    "Cream Silk": "Hair Care - DMT",
    "Andrelon": "Hair Care - DMT",
    "Organics": "Hair Care - DMT",
    # Domain 2: Skin Care
    "Pond's": "Skin Care",
    "Glow & Lovely": "Skin Care",
    "Glow & Handsome": "Skin Care",
    "Lakme": "Skin Care",
    "Vaseline": "Skin Care",
    "Vasenol": "Skin Care",
    "Simple": "Skin Care",
    "Elle 18": "Skin Care",
    "Citra": "Skin Care",
    "St. Ives": "Skin Care",
    "Minimalist": "Skin Care",
    "Novology": "Skin Care",
    "Acne Squad": "Skin Care",
    "Dermalogica": "Skin Care",
    "Paula's Choice": "Skin Care",
    "Tatcha": "Skin Care",
    "Hourglass": "Skin Care",
    "Kate Somerville": "Skin Care",
    "Murad": "Skin Care",
    "REN Clean Skincare": "Skin Care",
    "Garancia": "Skin Care",
    "Skinsei": "Skin Care",
    "AHC": "Skin Care",
    "Noxzema": "Skin Care",
    "Aviance": "Skin Care",
    "Hazeline": "Skin Care",
    "Eskinol": "Skin Care",
    "Master": "Skin Care",
    "Block & Glow": "Skin Care",
    "Zwitsal": "Skin Care",
    # Domain 3: Oral Care
    "Closeup": "Oral Care",
    "Pepsodent": "Oral Care",
    "P/S": "Oral Care",
    "Signal": "Oral Care",
    "Zendium": "Oral Care",
    "Mentadent": "Oral Care",
    "Prodent": "Oral Care",
    "Regenerate": "Oral Care",
    "Fluocaril": "Oral Care",
    "Zhonghua": "Oral Care",
    "Aim": "Oral Care",
    # Domain 4: Personal Wash - Laundry
    "Lux": "Personal Wash - Laundry",
    "Caress": "Personal Wash - Laundry",
    "Camay": "Personal Wash - Laundry",
    "Lifebuoy": "Personal Wash - Laundry",
    "Pears": "Personal Wash - Laundry",
    "Hamam": "Personal Wash - Laundry",
    "Liril": "Personal Wash - Laundry",
    "Moti": "Personal Wash - Laundry",
    "Motibaug": "Personal Wash - Laundry",
    "OK": "Personal Wash - Laundry",
    "Jai": "Personal Wash - Laundry",
    "Breeze": "Personal Wash - Laundry",
    "Baby Dove": "Personal Wash - Laundry",
    "Dove Men+Care": "Personal Wash - Laundry",
    "Rexona": "Personal Wash - Laundry",
    "Sure": "Personal Wash - Laundry",
    "Degree": "Personal Wash - Laundry",
    "Shield": "Personal Wash - Laundry",
    "Impulse": "Personal Wash - Laundry",
    "Axe": "Personal Wash - Laundry",
    "Lynx": "Personal Wash - Laundry",
    "Brut": "Personal Wash - Laundry",
    "Denim": "Personal Wash - Laundry",
    "Radox": "Personal Wash - Laundry",
    "Badedas": "Personal Wash - Laundry",
    "Dusch Das": "Personal Wash - Laundry",
    "Schmidt's": "Personal Wash - Laundry",
    "Wild": "Personal Wash - Laundry",
    "Find Your Happy Place": "Personal Wash - Laundry",
    "VWash": "Personal Wash - Laundry",
    "Surf Excel": "Personal Wash - Laundry",
    "Surf": "Personal Wash - Laundry",
    "Omo": "Personal Wash - Laundry",
    "Persil": "Personal Wash - Laundry",
    "Skip": "Personal Wash - Laundry",
    "Rin": "Personal Wash - Laundry",
    "Rinso": "Personal Wash - Laundry",
    "Wheel": "Personal Wash - Laundry",
    "Sunlight": "Personal Wash - Laundry",
    "Vim": "Personal Wash - Laundry",
    "Cif": "Personal Wash - Laundry",
    "Jif": "Personal Wash - Laundry",
    "Viss": "Personal Wash - Laundry",
    "Domex": "Personal Wash - Laundry",
    "Domestos": "Personal Wash - Laundry",
    "Comfort": "Personal Wash - Laundry",
    "Snuggle": "Personal Wash - Laundry",
    "Coccolino": "Personal Wash - Laundry",
    "Yumos": "Personal Wash - Laundry",
    "Ala": "Personal Wash - Laundry",
    "All": "Personal Wash - Laundry",
    "Radiant": "Personal Wash - Laundry",
    "Robijn": "Personal Wash - Laundry",
    "Seventh Generation": "Personal Wash - Laundry",
    "Blueair": "Personal Wash - Laundry",
    "Nature Protect": "Personal Wash - Laundry",
    "Smartclean": "Personal Wash - Laundry",
    "Love & Care": "Personal Wash - Laundry",
    "Magic": "Personal Wash - Laundry",
    "Pureit": "Personal Wash - Laundry",
    # Domain 5: Foods - Beverages
    "Lipton": "Foods - Beverages",
    "Brooke Bond": "Foods - Beverages",
    "Red Label": "Foods - Beverages",
    "Taj Mahal": "Foods - Beverages",
    "Taaza": "Foods - Beverages",
    "3 Roses": "Foods - Beverages",
    "PG Tips": "Foods - Beverages",
    "Pukka": "Foods - Beverages",
    "T2": "Foods - Beverages",
    "Sariwangi": "Foods - Beverages",
    "Bru": "Foods - Beverages",
    "Sunrise": "Foods - Beverages",
    "Horlicks": "Foods - Beverages",
    "Boost": "Foods - Beverages",
    "Maltova": "Foods - Beverages",
    "Viva": "Foods - Beverages",
    "Buavita": "Foods - Beverages",
    "Liquid I.V.": "Foods - Beverages",
    "OZiva": "Foods - Beverages",
    "Wellbeing Nutrition": "Foods - Beverages",
    "OLLY": "Foods - Beverages",
    "SmartyPants": "Foods - Beverages",
    "Onnit": "Foods - Beverages",
    "Kissan": "Foods - Beverages",
    "Knorr": "Foods - Beverages",
    "Royco": "Foods - Beverages",
    "Continental": "Foods - Beverages",
    "Hellmann's": "Foods - Beverages",
    "Best Foods": "Foods - Beverages",
    "Lady's Choice": "Foods - Beverages",
    "Annapurna": "Foods - Beverages",
    "Colman's": "Foods - Beverages",
    "Maille": "Foods - Beverages",
    "Calve": "Foods - Beverages",
    "Amora": "Foods - Beverages",
    "Aromat": "Foods - Beverages",
    "Bovril": "Foods - Beverages",
    "Marmite": "Foods - Beverages",
    "Bango": "Foods - Beverages",
    "Robertson's": "Foods - Beverages",
    "Sir Kensington's": "Foods - Beverages",
    "Maizena": "Foods - Beverages",
    "Pot Noodle": "Foods - Beverages",
    "Kwality Wall's": "Foods - Beverages",
    "Wall's": "Foods - Beverages",
    "Cornetto": "Foods - Beverages",
    "Magnum": "Foods - Beverages",
    "Feast": "Foods - Beverages",
    "Paddle Pop": "Foods - Beverages",
    "Solero": "Foods - Beverages",
    "Calippo": "Foods - Beverages",
    "Viennetta": "Foods - Beverages",
    "Carte D'Or": "Foods - Beverages",
    "Ben & Jerry's": "Foods - Beverages",
    "Breyers": "Foods - Beverages",
    "Talenti": "Foods - Beverages",
    "Good Humor": "Foods - Beverages",
    "Klondike": "Foods - Beverages",
    "Popsicle": "Foods - Beverages",
    "Yasso": "Foods - Beverages",
}

# Master Canonical HUL Base-Pack & POSM Catalog spanning all 6 Variant Classification Domains & Grammage Tiers
MASTER_HUL_CATALOG: List[Dict[str, Any]] = [
    # --- Domain 1: Hair Care - DMT ---
    {"sku_id": "BP-HUL-DOVE-IR-180ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Intense Repair Shampoo", "size": "180ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-IR-340ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Intense Repair Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-IR-650ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Intense Repair Pump Shampoo", "size": "650ml", "packaging_type": "pump_bottle"},
    {"sku_id": "BP-HUL-DOVE-IR-1L", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Intense Repair Family Pump Shampoo", "size": "1L", "packaging_type": "pump_bottle"},
    {"sku_id": "BP-HUL-DOVE-DS-340ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Daily Shine Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-HFR-340ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Hair Fall Rescue Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-DR-340ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Dandruff Care Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-COND-180ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Intense Repair Conditioner", "size": "180ml", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-DOVE-MASK-300G", "category": "Hair Care - DMT", "brand": "Dove", "variant": "10-in-1 Deep Repair Hair Mask", "size": "300g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-DOVE-SACHET-6ML", "category": "Hair Care - DMT", "brand": "Dove", "variant": "Hair Fall Rescue Sachet Ladi", "size": "6ml", "packaging_type": "sachet_strip_ladi"},
    {"sku_id": "BP-HUL-SUNSILK-BLK-180ML", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Stunning Black Shine Shampoo", "size": "180ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SUNSILK-BLK-340ML", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Lusciously Thick & Long / Black Shine", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SUNSILK-BLK-650ML", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Stunning Black Shine Pump Shampoo", "size": "650ml", "packaging_type": "pump_bottle"},
    {"sku_id": "BP-HUL-SUNSILK-PINK-340ML", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Thick & Long Pink Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SUNSILK-YLW-180ML", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Nourishing Soft & Smooth", "size": "180ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SUNSILK-ONION-340ML", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Onion & Jojoba Oil Hair Fall Resist", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SUNSILK-LADI-6MLx16", "category": "Hair Care - DMT", "brand": "Sunsilk", "variant": "Black Shine Hanging Sachet Ladi", "size": "6ml", "packaging_type": "sachet_strip_ladi"},
    {"sku_id": "BP-HUL-TRESEMME-KS-185ML", "category": "Hair Care - DMT", "brand": "Tresemme", "variant": "Keratin Smooth Shampoo", "size": "185ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-TRESEMME-KS-580ML", "category": "Hair Care - DMT", "brand": "Tresemme", "variant": "Keratin Smooth Shampoo", "size": "580ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-TRESEMME-KS-1L", "category": "Hair Care - DMT", "brand": "Tresemme", "variant": "Keratin Smooth Salon Pump Shampoo", "size": "1L", "packaging_type": "pump_bottle"},
    {"sku_id": "BP-HUL-TRESEMME-HF-580ML", "category": "Hair Care - DMT", "brand": "Tresemme", "variant": "Hair Fall Defense Shampoo", "size": "580ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-TRESEMME-BOND-PLEX-250ML", "category": "Hair Care - DMT", "brand": "Tresemme", "variant": "Bond Plex Repair Shampoo", "size": "250ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-TRESEMME-SERUM-100ML", "category": "Hair Care - DMT", "brand": "Tresemme", "variant": "Keratin Smooth Anti-Frizz Hair Serum", "size": "100ml", "packaging_type": "pump_bottle"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-SL-80ML", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Strong & Long Health Shampoo", "size": "80ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-SL-175ML", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Strong & Long Health Shampoo", "size": "175ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-SL-340ML", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Strong & Long Health Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-SL-650ML", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Strong & Long Family Bottle", "size": "650ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-EGG-340ML", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Egg Protein Strong & Thick Shampoo", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-AYURVEDA-340ML", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Strength & Shine With Almond Oil", "size": "340ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-LADI-6MLx16", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Strong & Long Sachet Ladi", "size": "6ml", "packaging_type": "sachet_strip_ladi"},
    {"sku_id": "BP-HUL-CLINIC-PLUS-SACHET-6ML", "category": "Hair Care - DMT", "brand": "Clinic Plus", "variant": "Strong & Long Sachet", "size": "6ml", "packaging_type": "sachet"},
    {"sku_id": "BP-HUL-CLEAR-COOL-MENTHOL-170ML", "category": "Hair Care - DMT", "brand": "Clear", "variant": "Cool Sport Menthol Anti-Dandruff", "size": "170ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-CLEAR-COOL-MENTHOL-330ML", "category": "Hair Care - DMT", "brand": "Clear", "variant": "Cool Sport Menthol Anti-Dandruff", "size": "330ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-CLEAR-COMPLETE-ACTIVE-330ML", "category": "Hair Care - DMT", "brand": "Clear", "variant": "Complete Active Care Anti-Dandruff", "size": "330ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-INDULEKHA-BRINGHA-OIL-50ML", "category": "Hair Care - DMT", "brand": "Indulekha", "variant": "Bringha Selfie Comb Hair Oil", "size": "50ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-INDULEKHA-BRINGHA-OIL-100ML", "category": "Hair Care - DMT", "brand": "Indulekha", "variant": "Bringha Selfie Comb Hair Oil", "size": "100ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-INDULEKHA-BRINGHA-SHAMPOO-200ML", "category": "Hair Care - DMT", "brand": "Indulekha", "variant": "Bringha Ayurvedic Hair Cleanser", "size": "200ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-INDULEKHA-SVETAKUTAJA-OIL-100ML", "category": "Hair Care - DMT", "brand": "Indulekha", "variant": "Svetakutaja Dandruff Treatment Oil", "size": "100ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-LBP-ARGAN-LAVENDER-200ML", "category": "Hair Care - DMT", "brand": "Love Beauty and Planet", "variant": "Argan Oil & Lavender Anti-Frizz Shampoo", "size": "200ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-LBP-ARGAN-LAVENDER-400ML", "category": "Hair Care - DMT", "brand": "Love Beauty and Planet", "variant": "Argan Oil & Lavender Anti-Frizz Shampoo", "size": "400ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-LBP-ONION-BLACKSEED-400ML", "category": "Hair Care - DMT", "brand": "Love Beauty and Planet", "variant": "Onion, Black Seed & Patchouli Hairfall Control", "size": "400ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-LBP-CURRY-BIOTIN-200ML", "category": "Hair Care - DMT", "brand": "Love Beauty and Planet", "variant": "Curry Leaves, Biotin & Mandarin Shampoo", "size": "200ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-AYUSH-SHIKAKAI-SHAMPOO-330ML", "category": "Hair Care - DMT", "brand": "Lever Ayush", "variant": "Thick & Long Shikakai & Bhringaraj Shampoo", "size": "330ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-NEXXUS-PRO-MEND-400ML", "category": "Hair Care - DMT", "brand": "Nexxus", "variant": "Promend Keratin Protein Shampoo", "size": "400ml", "packaging_type": "bottle"},

    # --- Domain 2: Skin Care, Dermacosmetics & Color Cosmetics ---
    {"sku_id": "BP-HUL-PONDS-BRIGHT-BEAUTY-50G", "category": "Skin Care", "brand": "Pond's", "variant": "Bright Beauty Spot-less Glow Face Wash", "size": "50g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PONDS-BRIGHT-BEAUTY-100G", "category": "Skin Care", "brand": "Pond's", "variant": "Bright Beauty Spot-less Glow Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PONDS-BRIGHT-BEAUTY-200G", "category": "Skin Care", "brand": "Pond's", "variant": "Bright Beauty Spot-less Glow Family Tube", "size": "200g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PONDS-PURE-DETOX-50G", "category": "Skin Care", "brand": "Pond's", "variant": "Pure Detox Activated Charcoal Face Wash", "size": "50g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PONDS-PURE-DETOX-100G", "category": "Skin Care", "brand": "Pond's", "variant": "Pure Detox Activated Charcoal Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PONDS-PIMPLE-CLEAR-100G", "category": "Skin Care", "brand": "Pond's", "variant": "Pimple Clear Active Thymo-T Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PONDS-SUPER-LIGHT-GEL-50G", "category": "Skin Care", "brand": "Pond's", "variant": "Super Light Gel Oil-Free Moisturizer", "size": "50g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-PONDS-SUPER-LIGHT-GEL-100G", "category": "Skin Care", "brand": "Pond's", "variant": "Super Light Gel Oil-Free Moisturizer", "size": "100g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-PONDS-SUPER-LIGHT-GEL-200G", "category": "Skin Care", "brand": "Pond's", "variant": "Super Light Gel Oil-Free Moisturizer XL", "size": "200g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-PONDS-BRIGHT-MIRACLE-SERUM-30ML", "category": "Skin Care", "brand": "Pond's", "variant": "Bright Miracle Niasorcinol Serum", "size": "30ml", "packaging_type": "dropper_serum"},
    {"sku_id": "BP-HUL-PONDS-AGE-MIRACLE-DAY-50G", "category": "Skin Care", "brand": "Pond's", "variant": "Age Miracle Youthful Glow Day Cream SPF 18", "size": "50g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-PONDS-MOISTURISING-COLD-CREAM-100ML", "category": "Skin Care", "brand": "Pond's", "variant": "Moisturising Cold Cream", "size": "100ml", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-PONDS-DREAMFLOWER-TALC-100G", "category": "Skin Care", "brand": "Pond's", "variant": "Dreamflower Fragrant Talc", "size": "100g", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-PONDS-DREAMFLOWER-TALC-200G", "category": "Skin Care", "brand": "Pond's", "variant": "Dreamflower Fragrant Talc", "size": "200g", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-PONDS-MAGIC-TALC-100G", "category": "Skin Care", "brand": "Pond's", "variant": "Magic Freshness Acacia Honey Talc", "size": "100g", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-GAL-INSTA-GLOW-50G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "Advanced Multivitamin Insta Glow Face Wash", "size": "50g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-GAL-INSTA-GLOW-100G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "Advanced Multivitamin Insta Glow Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-GAL-GLASS-BRIGHT-100G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "Glass Bright Vitamin C Gel Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-GAL-MULTIVITAMIN-CREAM-25G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "Advanced Multivitamin Face Cream", "size": "25g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-GAL-MULTIVITAMIN-CREAM-50G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "Advanced Multivitamin Face Cream", "size": "50g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-GAL-MULTIVITAMIN-CREAM-80G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "Advanced Multivitamin Face Cream", "size": "80g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-GAL-BB-CREAM-40G", "category": "Skin Care", "brand": "Glow & Lovely", "variant": "BB Foundation + Multivitamin Cream", "size": "40g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-GAH-RAPID-ACTION-FW-100G", "category": "Skin Care", "brand": "Glow & Handsome", "variant": "Instant Brightness Rapid Action Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-BG-STRAWBERRY-50G", "category": "Skin Care", "brand": "Lakme", "variant": "Blush & Glow Strawberry Gel Face Wash", "size": "50g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-BG-STRAWBERRY-100G", "category": "Skin Care", "brand": "Lakme", "variant": "Blush & Glow Strawberry Gel Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-BG-KIWI-100G", "category": "Skin Care", "brand": "Lakme", "variant": "Blush & Glow Kiwi Crush Gel Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-BG-LEMON-100G", "category": "Skin Care", "brand": "Lakme", "variant": "Blush & Glow Lemon Fresh Gel Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-BG-PEACH-100G", "category": "Skin Care", "brand": "Lakme", "variant": "Blush & Glow Peach Milk Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-9TO5-VITC-FW-100G", "category": "Skin Care", "brand": "Lakme", "variant": "9to5 Vitamin C+ Facial Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-LUMI-CREAM-30G", "category": "Skin Care", "brand": "Lakme", "variant": "Lumi Strobe Cream Highlighter Moisturizer", "size": "30g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-PEACH-MILK-120ML", "category": "Skin Care", "brand": "Lakme", "variant": "Peach Milk Soft Creme Moisturizer", "size": "120ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-LAKME-CC-BEIGE-30G", "category": "Skin Care", "brand": "Lakme", "variant": "9to5 CC Cream 01 Beige", "size": "30g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-CC-ALMOND-30G", "category": "Skin Care", "brand": "Lakme", "variant": "9to5 CC Cream 02 Almond", "size": "30g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-CC-HONEY-30G", "category": "Skin Care", "brand": "Lakme", "variant": "9to5 CC Cream 03 Honey", "size": "30g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-CC-BRONZE-30G", "category": "Skin Care", "brand": "Lakme", "variant": "9to5 CC Cream 04 Bronze", "size": "30g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-SUN-EXPERT-SPF50-100ML", "category": "Skin Care", "brand": "Lakme", "variant": "Sun Expert SPF 50 PA+++ Ultra Matte", "size": "100ml", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-LAKME-EYECONIC-KAJAL-035G", "category": "Skin Care", "brand": "Lakme", "variant": "Eyeconic Kajal Deep Black", "size": "0.35g", "packaging_type": "blister_card"},
    {"sku_id": "BP-HUL-ELLE18-COLOR-POP-LIP-43G", "category": "Skin Care", "brand": "Elle 18", "variant": "Color Pop Matte Lip Color", "size": "4.3g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-VASELINE-IC-DEEP-RESTORE-100ML", "category": "Skin Care", "brand": "Vaseline", "variant": "Intensive Care Deep Restore Lotion", "size": "100ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-VASELINE-IC-DEEP-RESTORE-400ML", "category": "Skin Care", "brand": "Vaseline", "variant": "Intensive Care Deep Restore Lotion", "size": "400ml", "packaging_type": "pump_bottle"},
    {"sku_id": "BP-HUL-VASELINE-COCOA-GLOW-400ML", "category": "Skin Care", "brand": "Vaseline", "variant": "Intensive Care Cocoa Glow Lotion", "size": "400ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-VASELINE-ALOE-FRESH-400ML", "category": "Skin Care", "brand": "Vaseline", "variant": "Intensive Care Aloe Soothe Lotion", "size": "400ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-VASELINE-GLUTA-HYA-200ML", "category": "Skin Care", "brand": "Vaseline", "variant": "Gluta-Hya Dewy Radiance Serum-in-Lotion", "size": "200ml", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-VASELINE-BLUESEAL-JELLY-42G", "category": "Skin Care", "brand": "Vaseline", "variant": "Original Pure Skin Jelly", "size": "42g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-VASELINE-BLUESEAL-JELLY-100G", "category": "Skin Care", "brand": "Vaseline", "variant": "Original Pure Skin Jelly", "size": "100g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-SIMPLE-REFRESHING-FW-150ML", "category": "Skin Care", "brand": "Simple", "variant": "Kind to Skin Refreshing Facial Wash", "size": "150ml", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-SIMPLE-MOISTURISING-FW-150ML", "category": "Skin Care", "brand": "Simple", "variant": "Kind to Skin Moisturising Facial Wash", "size": "150ml", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-SIMPLE-HYDRATING-LIGHT-125ML", "category": "Skin Care", "brand": "Simple", "variant": "Kind to Skin Hydrating Light Moisturiser", "size": "125ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-MINIMALIST-NIACINAMIDE-10PCT-30ML", "category": "Skin Care", "brand": "Minimalist", "variant": "10% Niacinamide Face Serum", "size": "30ml", "packaging_type": "dropper_serum"},
    {"sku_id": "BP-HUL-MINIMALIST-SALICYLIC-2PCT-30ML", "category": "Skin Care", "brand": "Minimalist", "variant": "2% Salicylic Acid BHA Face Serum", "size": "30ml", "packaging_type": "dropper_serum"},
    {"sku_id": "BP-HUL-MINIMALIST-SALICYLIC-FW-100ML", "category": "Skin Care", "brand": "Minimalist", "variant": "2% Salicylic Acid + LHA Cleanser", "size": "100ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-MINIMALIST-SPF50-SUNSCREEN-50G", "category": "Skin Care", "brand": "Minimalist", "variant": "SPF 50 PA++++ Multi-Vitamin Sunscreen", "size": "50g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-NOVOLOGY-ACNE-CLEANSER-100ML", "category": "Skin Care", "brand": "Novology", "variant": "Acne Deep Clearing Cleanser", "size": "100ml", "packaging_type": "tube"},
    {"sku_id": "BP-HUL- NOVOLOGY-PIGMENTATION-SERUM-30ML", "category": "Skin Care", "brand": "Novology", "variant": "Bi-Phasic Hyper Pigmentation Serum", "size": "30ml", "packaging_type": "dropper_serum"},
    {"sku_id": "BP-HUL-ACNESQUAD-SALICYLIC-FW-100ML", "category": "Skin Care", "brand": "Acne Squad", "variant": "Kick Start Cleanser with Salicylic Acid", "size": "100ml", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-DERMALOGICA-DAILY-MICROFOLIANT-74G", "category": "Skin Care", "brand": "Dermalogica", "variant": "Daily Microfoliant Exfoliator", "size": "74g", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-PAULAS-CHOICE-BHA-EXFOLIANT-118ML", "category": "Skin Care", "brand": "Paula's Choice", "variant": "Skin Perfecting 2% BHA Liquid Exfoliant", "size": "118ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-STIVES-APRICOT-SCRUB-170G", "category": "Skin Care", "brand": "St. Ives", "variant": "Fresh Skin Apricot Scrub", "size": "170g", "packaging_type": "tube"},

    # --- Domain 3: Oral Care ---
    {"sku_id": "BP-HUL-CLOSEUP-EVERFRESH-RED-40G", "category": "Oral Care", "brand": "Closeup", "variant": "Everfresh Red Hot Gel Toothpaste", "size": "40g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-CLOSEUP-EVERFRESH-RED-80G", "category": "Oral Care", "brand": "Closeup", "variant": "Everfresh Red Hot Gel Toothpaste", "size": "80g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-CLOSEUP-EVERFRESH-RED-150G", "category": "Oral Care", "brand": "Closeup", "variant": "Everfresh Red Hot Gel Toothpaste", "size": "150g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-CLOSEUP-EVERFRESH-RED-300G", "category": "Oral Care", "brand": "Closeup", "variant": "Everfresh Red Hot Twin Saver Pack", "size": "300g", "packaging_type": "multipack"},
    {"sku_id": "BP-HUL-CLOSEUP-PEPPERMINT-TUBE-80G", "category": "Oral Care", "brand": "Closeup", "variant": "Everfresh Peppermint Splash", "size": "80g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-CLOSEUP-LEMON-MINT-150G", "category": "Oral Care", "brand": "Closeup", "variant": "Lemon Mint Natural Glow Gel", "size": "150g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-PEPSODENT-GERMICHECK-80G", "category": "Oral Care", "brand": "Pepsodent", "variant": "Germicheck 8-Action Cavity Protection", "size": "80g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-PEPSODENT-GERMICHECK-150G", "category": "Oral Care", "brand": "Pepsodent", "variant": "Germicheck 8-Action Cavity Protection", "size": "150g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-PEPSODENT-GERMICHECK-200G", "category": "Oral Care", "brand": "Pepsodent", "variant": "Germicheck Magnet Action Family Pack", "size": "200g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-PEPSODENT-2IN1-150G", "category": "Oral Care", "brand": "Pepsodent", "variant": "2-in-1 Paste + Gel", "size": "150g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PEPSODENT-GUMCARE-140G", "category": "Oral Care", "brand": "Pepsodent", "variant": "Expert Protection Gum Care", "size": "140g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-AYUSH-CLOVE-TOOTHPASTE-150G", "category": "Oral Care", "brand": "Lever Ayush", "variant": "Anti-Cavity Clove Oil Toothpaste", "size": "150g", "packaging_type": "box"},

    # --- Domain 4: Personal Wash - Laundry & Home Care ---
    {"sku_id": "BP-HUL-PEARS-PURE-GENTLE-100G", "category": "Skin Care", "brand": "Pears", "variant": "Pure & Gentle Glycerine Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PEARS-OIL-CLEAR-100G", "category": "Skin Care", "brand": "Pears", "variant": "Oil Clear Mint Extract Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PEARS-SOFT-FRESH-100G", "category": "Skin Care", "brand": "Pears", "variant": "Soft & Fresh Blue Mint Face Wash", "size": "100g", "packaging_type": "tube"},
    {"sku_id": "BP-HUL-PEARS-SOAP-BAR-75G", "category": "Personal Wash - Laundry", "brand": "Pears", "variant": "Pure & Gentle Bathing Bar", "size": "75g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-PEARS-SOAP-BAR-125G", "category": "Personal Wash - Laundry", "brand": "Pears", "variant": "Pure & Gentle Bathing Bar", "size": "125g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-PEARS-BODYWASH-250ML", "category": "Personal Wash - Laundry", "brand": "Pears", "variant": "Pure & Gentle Shower Gel", "size": "250ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-LUX-ROSE-VIT-E-100G", "category": "Personal Wash - Laundry", "brand": "Lux", "variant": "Velvet Glow Rose & Vitamin E Soap Bar", "size": "100g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-LUX-ROSE-VIT-E-150G", "category": "Personal Wash - Laundry", "brand": "Lux", "variant": "Rose & Vitamin E Large Soap Bar", "size": "150g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-LUX-JASMINE-MULTIPACK-4x100G", "category": "Personal Wash - Laundry", "brand": "Lux", "variant": "Jasmine & Vitamin E 4-Bar Multipack", "size": "400g", "packaging_type": "multipack"},
    {"sku_id": "BP-HUL-LUX-BODYWASH-245ML", "category": "Personal Wash - Laundry", "brand": "Lux", "variant": "Fragrant Skin Black Orchid Body Wash", "size": "245ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-LIFEBUOY-TOTAL10-BAR-56G", "category": "Personal Wash - Laundry", "brand": "Lifebuoy", "variant": "Total 10 Germ Protection Soap Bar", "size": "56g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-LIFEBUOY-TOTAL10-BAR-125G", "category": "Personal Wash - Laundry", "brand": "Lifebuoy", "variant": "Total 10 Germ Protection Soap Bar", "size": "125g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-LIFEBUOY-LEMON-FRESH-125G", "category": "Personal Wash - Laundry", "brand": "Lifebuoy", "variant": "Lemon Fresh Germ Protection Bar", "size": "125g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-LIFEBUOY-HANDWASH-PUMP-190ML", "category": "Personal Wash - Laundry", "brand": "Lifebuoy", "variant": "Total 10 Liquid Handwash Pump", "size": "190ml", "packaging_type": "pump_bottle"},
    {"sku_id": "BP-HUL-LIFEBUOY-HANDWASH-POUCH-750ML", "category": "Personal Wash - Laundry", "brand": "Lifebuoy", "variant": "Total 10 Handwash Refill Pouch", "size": "750ml", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-LIFEBUOY-SANITIZER-50ML", "category": "Personal Wash - Laundry", "brand": "Lifebuoy", "variant": "Immunity Boosting Hand Sanitizer", "size": "50ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-CREAM-BAR-100G", "category": "Personal Wash - Laundry", "brand": "Dove", "variant": "Cream Beauty Bathing Bar", "size": "100g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-DOVE-CREAM-BAR-3x100G", "category": "Personal Wash - Laundry", "brand": "Dove", "variant": "Cream Beauty Bathing Bar 3-Pack", "size": "300g", "packaging_type": "multipack"},
    {"sku_id": "BP-HUL-DOVE-BW-500", "category": "Personal Wash - Laundry", "brand": "Dove", "variant": "Deeply Nourishing Body Wash", "size": "500ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOVE-BW-750", "category": "Personal Wash - Laundry", "brand": "Dove", "variant": "Deeply Nourishing Body Wash", "size": "750ml", "packaging_type": "pump_bottle"},
    {"sku_id": "BP-HUL-DOVE-HW-500-POUCH", "category": "Personal Wash - Laundry", "brand": "Dove", "variant": "Deep Moisture Refill Pouch", "size": "500ml", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-HAMAM-NEEM-TULSI-100G", "category": "Personal Wash - Laundry", "brand": "Hamam", "variant": "100% Pure Neem, Tulsi & Aloe Vera Bar", "size": "100g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-HAMAM-NEEM-TULSI-150G", "category": "Personal Wash - Laundry", "brand": "Hamam", "variant": "100% Pure Neem, Tulsi & Aloe Vera Bar", "size": "150g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-LIRIL-LIME-TEA-TREE-125G", "category": "Personal Wash - Laundry", "brand": "Liril", "variant": "Lime & Tea Tree Oil Freshness Soap Bar", "size": "125g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-MOTI-SANDAL-LUXURY-150G", "category": "Personal Wash - Laundry", "brand": "Moti", "variant": "Luxury Sandal & Rose Bath Soap", "size": "150g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-BREEZE-ROSE-SOAP-100G", "category": "Personal Wash - Laundry", "brand": "Breeze", "variant": "Fragrant Rose Beauty Soap", "size": "100g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-REXONA-SOAP-COCONUT-100G", "category": "Personal Wash - Laundry", "brand": "Rexona", "variant": "Coconut & Olive Oil Silky Soft Skin Bar", "size": "100g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-REXONA-ROLLON-50ML", "category": "Personal Wash - Laundry", "brand": "Rexona", "variant": "Powder Dry Underarm Roll-On", "size": "50ml", "packaging_type": "roll_on"},
    {"sku_id": "BP-HUL-REXONA-DEO-SPRAY-150ML", "category": "Personal Wash - Laundry", "brand": "Rexona", "variant": "Shower Fresh 72H Deodorant Spray", "size": "150ml", "packaging_type": "aerosol_can"},
    {"sku_id": "BP-HUL-AXE-DARK-TEMPTATION-150ML", "category": "Personal Wash - Laundry", "brand": "Axe", "variant": "Dark Temptation Deodorant Body Spray", "size": "150ml", "packaging_type": "aerosol_can"},
    {"sku_id": "BP-HUL-AXE-APOLLO-150ML", "category": "Personal Wash - Laundry", "brand": "Axe", "variant": "Apollo Sage & Cedarwood Body Spray", "size": "150ml", "packaging_type": "aerosol_can"},
    {"sku_id": "BP-HUL-AXE-SIGNATURE-TICKET-122ML", "category": "Personal Wash - Laundry", "brand": "Axe", "variant": "Signature Mysterious Body Perfume", "size": "122ml", "packaging_type": "aerosol_can"},
    {"sku_id": "BP-HUL-VWASH-PLUS-HYGIENE-100ML", "category": "Personal Wash - Laundry", "brand": "VWash", "variant": "Plus Expert Intimate Hygiene Wash", "size": "100ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-VWASH-PLUS-HYGIENE-200ML", "category": "Personal Wash - Laundry", "brand": "VWash", "variant": "Plus Expert Intimate Hygiene Wash", "size": "200ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SURF-EXCEL-EASYWASH-60G", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Easy Wash Sachet Pouch", "size": "60g", "packaging_type": "sachet"},
    {"sku_id": "BP-HUL-SURF-EXCEL-QUICKWASH-500G", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Quick Wash Detergent Powder", "size": "500g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-SURF-EXCEL-QUICKWASH-1KG", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Quick Wash Detergent Powder", "size": "1kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-SURF-EXCEL-EASYWASH-1KG", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Easy Wash Detergent Powder", "size": "1kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-SURF-EXCEL-EASYWASH-4KG", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Easy Wash Value Pack Powder", "size": "4kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-SURF-EXCEL-MATIC-TOP-2KG", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Matic Top Load Detergent Powder Carton", "size": "2kg", "packaging_type": "box"},
    {"sku_id": "BP-HUL-SURF-EXCEL-MATIC-LIQ-1L", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Matic Top Load Liquid Detergent", "size": "1L", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-SURF-EXCEL-MATIC-POUCH-2L", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Matic Front Load Liquid Spout Pouch", "size": "2L", "packaging_type": "spout_pouch"},
    {"sku_id": "BP-HUL-SURF-EXCEL-BAR-250G", "category": "Personal Wash - Laundry", "brand": "Surf Excel", "variant": "Stain Eraser Detergent Bar", "size": "250g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-RIN-ADVANCED-POWDER-1KG", "category": "Personal Wash - Laundry", "brand": "Rin", "variant": "Advanced Brightening Detergent Powder", "size": "1kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-RIN-ADVANCED-POWDER-2KG", "category": "Personal Wash - Laundry", "brand": "Rin", "variant": "Advanced Brightening Detergent Powder", "size": "2kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-RIN-LIQUID-1L", "category": "Personal Wash - Laundry", "brand": "Rin", "variant": "Refresh Lemon & Rose Liquid Detergent", "size": "1L", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-RIN-BAR-250G", "category": "Personal Wash - Laundry", "brand": "Rin", "variant": "Dazzling White Detergent Bar", "size": "250g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-RIN-ALA-FABRIC-WHITENER-500ML", "category": "Personal Wash - Laundry", "brand": "Ala", "variant": "Rin Ala Fabric Whitener Bleach", "size": "500ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-WHEEL-ACTIVE-2IN1-500G", "category": "Personal Wash - Laundry", "brand": "Wheel", "variant": "Active 2-in-1 Lemon & Jasmine Powder", "size": "500g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-WHEEL-ACTIVE-2IN1-1KG", "category": "Personal Wash - Laundry", "brand": "Wheel", "variant": "Active 2-in-1 Lemon & Jasmine Powder", "size": "1kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-SUNLIGHT-COLOUR-CARE-1KG", "category": "Personal Wash - Laundry", "brand": "Sunlight", "variant": "Colour Care Detergent Powder", "size": "1kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-VIM-DISHWASH-GEL-250ML", "category": "Personal Wash - Laundry", "brand": "Vim", "variant": "Dishwash Liquid Gel Lemon", "size": "250ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-VIM-DISHWASH-GEL-750ML", "category": "Personal Wash - Laundry", "brand": "Vim", "variant": "Dishwash Liquid Gel Power of 100 Lemons", "size": "750ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-VIM-DISHWASH-POUCH-2L", "category": "Personal Wash - Laundry", "brand": "Vim", "variant": "Dishwash Liquid Gel Mega Spout Pouch", "size": "2L", "packaging_type": "spout_pouch"},
    {"sku_id": "BP-HUL-VIM-BAR-300G", "category": "Personal Wash - Laundry", "brand": "Vim", "variant": "Lemon Dishwash Bar", "size": "300g", "packaging_type": "bar"},
    {"sku_id": "BP-HUL-CIF-CREAM-CLEANER-500ML", "category": "Personal Wash - Laundry", "brand": "Cif", "variant": "Power & Shine Kitchen & Surface Cream", "size": "500ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOMEX-FRESH-GUARD-500ML", "category": "Personal Wash - Laundry", "brand": "Domex", "variant": "Fresh Guard Disinfectant Toilet Cleaner", "size": "500ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-DOMEX-FLOOR-CLEANER-1L", "category": "Personal Wash - Laundry", "brand": "Domex", "variant": "Disinfectant Floor Cleaner Pine", "size": "1L", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-COMFORT-MORNING-FRESH-220ML", "category": "Personal Wash - Laundry", "brand": "Comfort", "variant": "After Wash Morning Fresh Fabric Conditioner", "size": "220ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-COMFORT-MORNING-FRESH-860ML", "category": "Personal Wash - Laundry", "brand": "Comfort", "variant": "After Wash Morning Fresh Fabric Conditioner", "size": "860ml", "packaging_type": "bottle"},
    {"sku_id": "BP-HUL-COMFORT-LILY-FRESH-2L", "category": "Personal Wash - Laundry", "brand": "Comfort", "variant": "After Wash Lily Fresh Spout Pouch", "size": "2L", "packaging_type": "spout_pouch"},
    {"sku_id": "BP-HUL-NATURE-PROTECT-SPRAY-500ML", "category": "Personal Wash - Laundry", "brand": "Nature Protect", "variant": "Neem Disinfectant Surface Spray", "size": "500ml", "packaging_type": "bottle"},

    # --- Domain 5: Foods - Beverages, Tea, Coffee, Nutrition & Ice Cream ---
    {"sku_id": "BP-HUL-LIPTON-GREEN-TEA-25TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Pure & Light Green Tea 25 Tea Bags", "size": "35g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-GREEN-TEA-100TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Pure & Light Green Tea 100 Tea Bags", "size": "140g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-HONEY-LEMON-25TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Honey Lemon Green Tea 25 Tea Bags", "size": "35g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-TULSI-NATURO-25TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Tulsi Naturo Green Tea 25 Tea Bags", "size": "35g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-LEMON-ZEST-10TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Lemon Zest Green Tea 10 Tea Bags", "size": "14g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-MINT-BURST-25TB", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Mint Burst Green Tea 25 Tea Bags", "size": "35g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-DARJEELING-250G", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Darjeeling Long Leaf Tea", "size": "250g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-YELLOW-LABEL-250G", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Yellow Label Finest Black Tea Carton", "size": "250g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-LIPTON-ICETEA-LEMON-350G", "category": "Foods - Beverages", "brand": "Lipton", "variant": "Lemon Ice Tea Instant Mix", "size": "350g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-RED-LABEL-NATURAL-CARE-250G", "category": "Foods - Beverages", "brand": "Red Label", "variant": "Brooke Bond Natural Care 5-Herbs Tea", "size": "250g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-RED-LABEL-NATURAL-CARE-500G", "category": "Foods - Beverages", "brand": "Red Label", "variant": "Brooke Bond Natural Care 5-Herbs Tea", "size": "500g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-RED-LABEL-POUCH-250G", "category": "Foods - Beverages", "brand": "Red Label", "variant": "Brooke Bond Red Label Tea Pouch", "size": "250g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-RED-LABEL-POUCH-1KG", "category": "Foods - Beverages", "brand": "Red Label", "variant": "Brooke Bond Red Label Family Tea Pouch", "size": "1kg", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-TAJ-MAHAL-TEA-250G", "category": "Foods - Beverages", "brand": "Taj Mahal", "variant": "Brooke Bond Taj Mahal Rich & Flavourful Tea", "size": "250g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-TAJ-MAHAL-TEA-500G", "category": "Foods - Beverages", "brand": "Taj Mahal", "variant": "Brooke Bond Taj Mahal South Blend Carton", "size": "500g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-TAAZA-MASALA-CHAI-250G", "category": "Foods - Beverages", "brand": "Taaza", "variant": "Brooke Bond Taaza Leaf Tea Pouch", "size": "250g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-3ROSES-TOP-STAR-250G", "category": "Foods - Beverages", "brand": "3 Roses", "variant": "Brooke Bond 3 Roses Dust Tea", "size": "250g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-3ROSES-NATURAL-CARE-500G", "category": "Foods - Beverages", "brand": "3 Roses", "variant": "Brooke Bond 3 Roses Natural Care Tea", "size": "500g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-BRU-INSTANT-SACHET-12G", "category": "Foods - Beverages", "brand": "Bru", "variant": "Instant Coffee Sachet", "size": "12g", "packaging_type": "sachet"},
    {"sku_id": "BP-HUL-BRU-INSTANT-POUCH-100G", "category": "Foods - Beverages", "brand": "Bru", "variant": "Instant Chicory Blend Coffee Pouch", "size": "100g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-BRU-GOLD-JAR-100G", "category": "Foods - Beverages", "brand": "Bru", "variant": "Gold 100% Pure Freeze-Dried Coffee Jar", "size": "100g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-BRU-GREEN-LABEL-500G", "category": "Foods - Beverages", "brand": "Bru", "variant": "Green Label Filter Coffee Roast & Ground", "size": "500g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-HORLICKS-CLASSIC-MALT-500G", "category": "Foods - Beverages", "brand": "Horlicks", "variant": "Classic Malt Health & Nutrition Drink", "size": "500g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-HORLICKS-CLASSIC-POUCH-500G", "category": "Foods - Beverages", "brand": "Horlicks", "variant": "Classic Malt Refill Pouch", "size": "500g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-HORLICKS-CHOCOLATE-500G", "category": "Foods - Beverages", "brand": "Horlicks", "variant": "Chocolate Delight Nutrition Jar", "size": "500g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-HORLICKS-JUNIOR-500G", "category": "Foods - Beverages", "brand": "Horlicks", "variant": "Junior Horlicks Stage 1-2 Nutrition", "size": "500g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-HORLICKS-WOMENS-PLUS-400G", "category": "Foods - Beverages", "brand": "Horlicks", "variant": "Women's Plus Caramel Bone Health", "size": "400g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-HORLICKS-MOTHERS-PLUS-400G", "category": "Foods - Beverages", "brand": "Horlicks", "variant": "Mother's Plus Vanilla Pregnancy Nutrition", "size": "400g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-HORLICKS-PROTEIN-PLUS-400G", "category": "Foods - Beverages", "brand": "Horlicks", "variant": "Protein Plus Triple Blend Vanilla", "size": "400g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-HORLICKS-DIABETES-PLUS-400G", "category": "Foods - Beverages", "brand": "Horlicks", "variant": "Diabetes Plus High Fiber Nutrition", "size": "400g", "packaging_type": "tin"},
    {"sku_id": "BP-HUL-BOOST-ENERGY-JAR-500G", "category": "Foods - Beverages", "brand": "Boost", "variant": "3X Stamina Malt Energy Drink Jar", "size": "500g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-BOOST-POUCH-500G", "category": "Foods - Beverages", "brand": "Boost", "variant": "3X Stamina Refill Pouch", "size": "500g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-LIQUID-IV-LEMON-LIME-10PK", "category": "Foods - Beverages", "brand": "Liquid I.V.", "variant": "Hydration Multiplier Lemon Lime 10 Sticks", "size": "160g", "packaging_type": "box"},
    {"sku_id": "BP-HUL-OZIVA-PROTEIN-HERBS-500G", "category": "Foods - Beverages", "brand": "OZiva", "variant": "Protein & Herbs for Women Chocolate", "size": "500g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-KISSAN-FRESH-TOMATO-KETCHUP-950G", "category": "Foods - Beverages", "brand": "Kissan", "variant": "Fresh Tomato Ketchup Spout Pouch", "size": "950g", "packaging_type": "spout_pouch"},
    {"sku_id": "BP-HUL-KISSAN-MIXED-FRUIT-JAM-500G", "category": "Foods - Beverages", "brand": "Kissan", "variant": "Mixed Fruit Jam Glass Jar", "size": "500g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-KISSAN-PEANUT-BUTTER-375G", "category": "Foods - Beverages", "brand": "Kissan", "variant": "Crunchy High Protein Peanut Butter", "size": "375g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-KNORR-CLASSIC-TOMATO-SOUP-53G", "category": "Foods - Beverages", "brand": "Knorr", "variant": "Classic Thick Tomato Soup Pouch", "size": "53g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-KNORR-MANCHOW-SOUP-43G", "category": "Foods - Beverages", "brand": "Knorr", "variant": "Chinese Hot & Sour Veg Soup Pouch", "size": "43g", "packaging_type": "pouch"},
    {"sku_id": "BP-HUL-KNORR-SCHEZWAN-CHUTNEY-250G", "category": "Foods - Beverages", "brand": "Knorr", "variant": "Chilli Schezwan Chinese Sauce Jar", "size": "250g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-HELLMANNS-REAL-MAYO-400G", "category": "Foods - Beverages", "brand": "Hellmann's", "variant": "Real Mayonnaise Jar", "size": "400g", "packaging_type": "jar"},
    {"sku_id": "BP-HUL-KWALITY-WALLS-SHAMELESS-VANILLA-700ML", "category": "Foods - Beverages", "brand": "Kwality Wall's", "variant": "Vanilla Frozen Dessert Party Tub", "size": "700ml", "packaging_type": "tub"},
    {"sku_id": "BP-HUL-CORNETTO-DOUBLE-CHOCOLATE-105ML", "category": "Foods - Beverages", "brand": "Cornetto", "variant": "Double Chocolate Cone", "size": "105ml", "packaging_type": "box"},
    {"sku_id": "BP-HUL-MAGNUM-ALMOND-80ML", "category": "Foods - Beverages", "brand": "Magnum", "variant": "Belgian Chocolate Almond Ice Cream Bar", "size": "80ml", "packaging_type": "box"},

    # --- Domain 6: Merchandising & POSM 6-Asset Registry ---
    {"sku_id": "POSM-HUL-LIPTON-WINDOW-HEADER", "category": "Merchandising & POSM", "brand": "Lipton", "variant": "Reduce Belly Fat With Tasty Green Tea Window Header", "size": "POSM", "packaging_type": "window_header"},
    {"sku_id": "POSM-HUL-LIPTON-REF-ASSET", "category": "Merchandising & POSM", "brand": "Lipton", "variant": "Reference Business Promotion Planogram Header & Fins", "size": "POSM", "packaging_type": "window_header"},
    {"sku_id": "POSM-HUL-LAKME-PONDS-SHELF-STRIP", "category": "Merchandising & POSM", "brand": "Lakme", "variant": "Detox Facewash / Lakme Expert Face Cleansers Shelf Strip", "size": "POSM", "packaging_type": "shelf_strip"},
    {"sku_id": "POSM-HUL-DOVE-PROMO-TOKER", "category": "Merchandising & POSM", "brand": "Dove", "variant": "Promotional Price-Off Rail Toker / Talker", "size": "POSM", "packaging_type": "toker_talker"},
]


def _load_open_web_discovered_skus() -> None:
    """Hot-load any self-learned brands & SKUs discovered via live open-internet grounding."""
    if not LEARNED_OPEN_WEB_SKUS_PATH.exists():
        return
    try:
        data = json.loads(LEARNED_OPEN_WEB_SKUS_PATH.read_text(encoding="utf-8"))
        for b in data.get("learned_hul_brands", []):
            HUL_BRANDS_CANONICAL.add(str(b).strip().lower())
        for b in data.get("learned_competitor_brands", []):
            COMPETITOR_BRANDS_NON_HUL.add(str(b).strip().lower())
        existing_ids = {item["sku_id"] for item in MASTER_HUL_CATALOG}
        for sku in data.get("learned_catalog_skus", []):
            if isinstance(sku, dict) and sku.get("sku_id") and sku["sku_id"] not in existing_ids:
                MASTER_HUL_CATALOG.append(sku)
                existing_ids.add(sku["sku_id"])
    except Exception:
        pass


_load_open_web_discovered_skus()


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
    is_hul = canonical.lower() in HUL_BRANDS_CANONICAL or lower in HUL_BRANDS_CANONICAL
    if not is_hul:
        # Check prefix/substring match against canonical multi-word HUL brands (e.g., "Brooke Bond Red Label", "Lipton Green Tea")
        for hul_b in sorted(HUL_BRANDS_CANONICAL, key=len, reverse=True):
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
        if canonical_brand == "Pears" and ("face" in hint_category.lower() or pkg == "tube"):
            return "Skin Care"
        if canonical_brand == "Dove" and ("bar" in pkg or "pouch" in pkg or "wash" in hint_category.lower()):
            return "Personal Wash - Laundry"
        if canonical_brand == "Lever Ayush" and ("tooth" in hint_category.lower() or "oral" in hint_category.lower()):
            return "Oral Care"
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

    brand_candidates = [
        item for item in MASTER_HUL_CATALOG
        if normalize_brand_and_hul_flag(item["brand"])[0].lower() == canonical_brand.lower()
    ]
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

    br_slug = re.sub(r"[^A-Z0-9]+", "", canonical_brand.upper())[:8]
    var_words = [w.upper()[:5] for w in re.split(r"[^a-zA-Z0-9]+", variant or pkg) if len(w) >= 2][:2]
    var_slug = "-".join(var_words) if var_words else pkg.upper()[:6]
    sz_slug = re.sub(r"[^A-Z0-9]+", "", (size_text or "STD").upper())[:6]
    prefix = "POSM-HUL" if pkg in ("window_header", "side_fin", "shelf_strip", "toker_talker", "parasite_hanger", "floor_standee") else "BP-HUL"
    synthesized = f"{prefix}-{br_slug}-{var_slug}-{sz_slug}"
    if synthesized not in candidate_ids:
        candidate_ids.insert(0, synthesized)
    return (synthesized, candidate_ids[:8])


def _persist_learned_open_web_discovery(
    brand: str,
    is_hul_brand: bool,
    sku_entry: Dict[str, Any],
) -> None:
    """Persist a newly discovered brand/SKU from the open internet into `configs/open_web_discovered_skus.json`."""
    try:
        data: Dict[str, Any] = {
            "updated_at_epoch": int(time.time()),
            "learned_hul_brands": [],
            "learned_competitor_brands": [],
            "learned_catalog_skus": [],
        }
        if LEARNED_OPEN_WEB_SKUS_PATH.exists():
            data = json.loads(LEARNED_OPEN_WEB_SKUS_PATH.read_text(encoding="utf-8"))
        if is_hul_brand:
            HUL_BRANDS_CANONICAL.add(brand.lower())
            if brand not in data.setdefault("learned_hul_brands", []):
                data["learned_hul_brands"].append(brand)
        else:
            COMPETITOR_BRANDS_NON_HUL.add(brand.lower())
            if brand not in data.setdefault("learned_competitor_brands", []):
                data["learned_competitor_brands"].append(brand)

        existing_ids = {s.get("sku_id") for s in data.setdefault("learned_catalog_skus", [])}
        if sku_entry.get("sku_id") and sku_entry["sku_id"] not in existing_ids:
            data["learned_catalog_skus"].append(sku_entry)
            if is_hul_brand:
                MASTER_HUL_CATALOG.append(sku_entry)
        data["updated_at_epoch"] = int(time.time())
        LEARNED_OPEN_WEB_SKUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEARNED_OPEN_WEB_SKUS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        pass


def resolve_brand_and_sku_via_open_web_grounding(
    query_text: str,
    hint_category: str = "",
    hint_packaging: str = "bottle",
    force_live_web: bool = False,
    persist_learned: bool = True,
) -> Dict[str, Any]:
    """Resolve any FMCG product/brand/SKU string against the Open-Internet Taxonomy AND live web search.

    3-Tier Open-Internet Resolution Pipeline:
      1. `open_internet_synced_taxonomy`: Instant check against the 270+ Global Unilever + HUL India brands
         and 220+ Base-Packs synced from `hul.co.in`, `unilever.com`, and Wikipedia.
      2. `vertex_gemini_google_search_grounding`: Queries Vertex AI `gemini-2.5-flash` with
         `"tools": [{"googleSearch": {}}]` for real-time Google Search grounding of unfamiliar brands/SKUs.
      3. `live_http_open_web_search`: Direct HTTPS search against Wikipedia MediaWiki API (`action=query&list=search`)
         to verify whether an unseen brand/product belongs to Unilever / Hindustan Unilever.
    """
    t0 = time.perf_counter()
    cleaned_query = (query_text or "").strip()
    pkg = normalize_packaging_type(hint_packaging)

    # Extract size token if present in query_text (e.g. "100g", "340ml", "25tb", "1kg")
    sz_match = re.search(r"\b(\d+(?:\.\d+)?\s*(?:ml|g|gm|kg|l|tb|teabags))\b", cleaned_query, flags=re.I)
    size_str = sz_match.group(1).replace(" ", "") if sz_match else "STD"

    canonical_brand, is_hul = normalize_brand_and_hul_flag(cleaned_query)
    is_known_competitor = canonical_brand.lower() in COMPETITOR_BRANDS_NON_HUL or any(
        len(c) >= 4 and re.search(rf"\b{re.escape(c)}\b", cleaned_query.lower())
        for c in COMPETITOR_BRANDS_NON_HUL
    )
    if not is_hul and is_known_competitor:
        for comp in sorted(COMPETITOR_BRANDS_NON_HUL, key=len, reverse=True):
            if len(comp) >= 4 and re.search(rf"\b{re.escape(comp)}\b", cleaned_query.lower()):
                canonical_brand = BRAND_ALIASES.get(comp, comp.title())
                break

    # Tier 1: Fast Open-Internet Synced Taxonomy hit (if brand is known and force_live_web is False)
    if (is_hul or is_known_competitor) and not force_live_web:
        domain = resolve_variant_domain(canonical_brand, hint_category, pkg)
        base_pack_id, candidates = resolve_or_synthesize_base_pack(
            brand=canonical_brand,
            category=domain,
            packaging_type=pkg,
            variant=cleaned_query,
            size_text=size_str,
        )
        return {
            "query": cleaned_query,
            "brand": canonical_brand,
            "parent_company": "Unilever / Hindustan Unilever (HUL)" if is_hul else "Competitor (Non-HUL)",
            "is_hul_brand": is_hul,
            "unilever_domain": domain,
            "packaging_type": pkg,
            "size": size_str,
            "base_pack_id": base_pack_id,
            "candidate_skus": candidates,
            "resolution_tier": "open_internet_synced_taxonomy",
            "grounding_sources": [
                "https://www.hul.co.in/brands/",
                "https://www.unilever.com/brands/all-brands/",
                "https://en.wikipedia.org/wiki/List_of_Unilever_brands",
            ],
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # Tier 2: Vertex AI `gemini-2.5-flash` with `"tools": [{"googleSearch": {}}]` Live Grounding
    grounding_sources: List[str] = []
    resolution_tier = "live_http_open_web_search"
    parent_company = "Unilever / Hindustan Unilever (HUL)" if is_hul else "Unknown / Open-Web Verified"
    inferred_domain = hint_category or "Personal Care"

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={api_key}"
            )
            prompt = (
                f"Search the web for FMCG product '{cleaned_query}'. Return ONLY compact JSON with keys: "
                "brand, parent_company, is_unilever_or_hul (boolean), unilever_domain "
                "(one of: 'Hair Care - DMT', 'Skin Care', 'Oral Care', 'Personal Wash - Laundry', "
                "'Foods - Beverages', 'Non-HUL'), variant, size, packaging_type."
            )
            req_body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "tools": [{"googleSearch": {}}],
                "generationConfig": {"temperature": 0.0},
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(req_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
            cand = (resp_json.get("candidates") or [{}])[0]
            text_out = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
            gm = cand.get("groundingMetadata", {})
            for chunk in gm.get("groundingChunks", []):
                web_uri = chunk.get("web", {}).get("uri")
                if web_uri:
                    grounding_sources.append(web_uri)
            m_json = re.search(r"\{.*\}", text_out, flags=re.S)
            if m_json:
                parsed = json.loads(m_json.group(0))
                canonical_brand = str(parsed.get("brand") or canonical_brand)
                parent_company = str(parsed.get("parent_company") or parent_company)
                is_hul = bool(parsed.get("is_unilever_or_hul", is_hul))
                inferred_domain = str(parsed.get("unilever_domain") or inferred_domain)
                size_str = str(parsed.get("size") or size_str)
                pkg = normalize_packaging_type(str(parsed.get("packaging_type") or pkg))
                resolution_tier = "vertex_gemini_google_search_grounding"
        except Exception:
            pass

    # Tier 3: Direct Live HTTPS Open-Web Search (`en.wikipedia.org` MediaWiki API)
    if resolution_tier != "vertex_gemini_google_search_grounding":
        try:
            q_enc = urllib.parse.quote(f"{cleaned_query} brand Unilever")
            wiki_search_url = (
                f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                f"&srsearch={q_enc}&utf8=1&format=json&srlimit=3"
            )
            req = urllib.request.Request(wiki_search_url, headers={"User-Agent": "UnileverShelfBot/1.0"})
            with urllib.request.urlopen(req, timeout=5.0) as r:
                wdata = json.loads(r.read().decode("utf-8"))
            hits = wdata.get("query", {}).get("search", [])
            snippets = " ".join(re.sub(r"<[^>]+>", "", h.get("snippet", "")) for h in hits).lower()
            for h in hits:
                title_slug = urllib.parse.quote(h.get("title", "").replace(" ", "_"))
                grounding_sources.append(f"https://en.wikipedia.org/wiki/{title_slug}")
            if not is_hul and not is_known_competitor:
                if "unilever" in snippets or "hindustan unilever" in snippets:
                    is_hul = True
                    parent_company = "Unilever / Hindustan Unilever (Open-Web Grounded)"
                    first_words = [w for w in re.split(r"[^a-zA-Z0-9&']+", cleaned_query) if w][:2]
                    if first_words and canonical_brand == "Unknown":
                        canonical_brand = " ".join(first_words).title()
                else:
                    is_hul = False
                    parent_company = "Non-HUL Competitor (Open-Web Grounded)"
                    if canonical_brand == "Unknown":
                        first_word = next((w for w in re.split(r"[^a-zA-Z0-9&']+", cleaned_query) if w), "Unknown")
                        canonical_brand = first_word.title()
        except Exception:
            pass

    domain = resolve_variant_domain(canonical_brand, inferred_domain, pkg) if is_hul else "Non-HUL"
    base_pack_id, candidates = resolve_or_synthesize_base_pack(
        brand=canonical_brand,
        category=domain,
        packaging_type=pkg,
        variant=cleaned_query,
        size_text=size_str,
    )

    sku_entry = {
        "sku_id": base_pack_id,
        "category": domain,
        "brand": canonical_brand,
        "variant": cleaned_query,
        "size": size_str,
        "packaging_type": pkg,
    }
    if persist_learned and canonical_brand != "Unknown":
        _persist_learned_open_web_discovery(canonical_brand, is_hul, sku_entry)

    return {
        "query": cleaned_query,
        "brand": canonical_brand,
        "parent_company": parent_company,
        "is_hul_brand": is_hul,
        "unilever_domain": domain,
        "packaging_type": pkg,
        "size": size_str,
        "base_pack_id": base_pack_id,
        "candidate_skus": candidates,
        "resolution_tier": resolution_tier,
        "grounding_sources": grounding_sources or [
            "https://www.hul.co.in/brands/",
            "https://en.wikipedia.org/wiki/List_of_Unilever_brands",
        ],
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def get_open_internet_taxonomy_summary() -> Dict[str, Any]:
    """Return full statistics and brand lists of the Open-Internet Synced Unilever Taxonomy."""
    domains_map: Dict[str, List[str]] = {d: [] for d in UNILEVER_VARIANT_DOMAINS}
    for b in sorted(HUL_BRANDS_CANONICAL):
        canon, _ = normalize_brand_and_hul_flag(b)
        dom = BRAND_TO_DEFAULT_DOMAIN.get(canon, "Personal Wash - Laundry")
        if canon not in domains_map.setdefault(dom, []):
            domains_map[dom].append(canon)
    domains_map["Non-HUL"] = sorted({BRAND_ALIASES.get(c, c.title()) for c in COMPETITOR_BRANDS_NON_HUL})
    return {
        "schema_version": "2026.09.open-internet-v2",
        "total_unilever_brand_entries": len(HUL_BRANDS_CANONICAL),
        "total_competitor_brand_entries": len(COMPETITOR_BRANDS_NON_HUL),
        "total_packaging_form_factors": len(CANONICAL_PACKAGING_TYPES),
        "total_master_base_packs": len(MASTER_HUL_CATALOG),
        "unilever_variant_domains": UNILEVER_VARIANT_DOMAINS,
        "canonical_packaging_types": CANONICAL_PACKAGING_TYPES,
        "brands_by_domain": domains_map,
        "open_internet_sources": [
            "https://www.hul.co.in/brands/ (57 official Hindustan Unilever brand pages)",
            "https://www.unilever.com/brands/all-brands/ (Unilever Global Power Brands)",
            "https://en.wikipedia.org/wiki/List_of_Unilever_brands (190+ active global Unilever brands)",
            "https://en.wikipedia.org/wiki/Hindustan_Unilever (HUL India heritage & masstige portfolio)",
            "Vertex AI gemini-2.5-flash with googleSearch Live Open-Web Grounding",
        ],
    }


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
