"""Canonical 10-Category Unilever Retail Taxonomy & Multi-Category Brand/Variant Resolver."""

from __future__ import annotations

import re

from utils.taxonomy.brands import A_PREFIX_TWINS, BRAND_ALIAS_MAP

CANONICAL_CATEGORIES: tuple[str, ...] = (
    "Skin",
    "PW",
    "Hair_Care",
    "Oral_Care",
    "Foods",
    "Beverages",
    "HFD",
    "Laundry",
    "HHC",
    "DMT",
)

CATEGORY_SYNONYM_MAP: dict[str, str] = {
    "skin_care": "Skin",
    "skincare": "Skin",
    "skin": "Skin",
    "skin_cleansing": "PW",
    "personal_wash": "PW",
    "pw": "PW",
    "hair": "Hair_Care",
    "hair_care": "Hair_Care",
    "haircare": "Hair_Care",
    "oral": "Oral_Care",
    "oral_care": "Oral_Care",
    "oralcare": "Oral_Care",
    "food": "Foods",
    "foods": "Foods",
    "beverage": "Beverages",
    "beverages": "Beverages",
    "hfd": "HFD",
    "nutrition": "HFD",
    "laundry": "Laundry",
    "fabric_care": "Laundry",
    "hhc": "HHC",
    "home_care": "HHC",
    "household_care": "HHC",
    "dmt": "DMT",
    "personal_care": "DMT",
    "deodorant": "DMT",
}

BRAND_CATEGORY_MAP: dict[str, str] = {
    "ayush": "Skin", "dove": "Skin", "elle18": "Skin", "fair_and_lovely": "Skin",
    "glow_and_lovely": "Skin", "lakme": "Skin", "novology": "Skin", "ponds": "Skin",
    "simple": "Skin", "vaseline": "Skin",
    "comfort": "Laundry", "rin": "Laundry", "sunlight": "Laundry", "surf_excel": "Laundry", "wheel": "Laundry",
    "breeze": "PW", "hamam": "PW", "lifebuoy": "PW", "liril": "PW", "lux": "PW", "moti": "PW",
    "nature_protect": "PW", "pears": "PW", "st_ives": "PW", "v_wash": "PW", "vwash": "PW",
    "clear": "Hair_Care", "clinic_plus": "Hair_Care", "indulekha": "Hair_Care", "lbp": "Hair_Care",
    "love_beauty_and_planet": "Hair_Care", "sunsilk": "Hair_Care", "tresemme": "Hair_Care",
    "rexona": "DMT", "axe": "DMT", "brut": "DMT",
    "closeup": "Oral_Care", "pepsodent": "Oral_Care",
    "3_roses": "Beverages", "brooke_bond": "Beverages", "bru": "Beverages", "lipton": "Beverages",
    "pukka": "Beverages", "red_label": "Beverages", "taaza": "Beverages", "taj_mahal": "Beverages",
    "boost": "HFD", "horlicks": "HFD", "maltova": "HFD", "oziva": "HFD", "viva": "HFD",
    "wellbeing_nutrition": "HFD",
    "annapurna": "Foods", "brown_and_polson": "Foods", "cornetto": "Foods", "hellmans": "Foods",
    "hellmanns": "Foods", "kissan": "Foods", "knorr": "Foods", "kwality": "Foods", "rex": "Foods",
    "cif": "HHC", "domex": "HHC", "ruby": "HHC", "sun": "HHC", "vim": "HHC",
}

_NON_HUL_BRANDS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "Laundry": (
        "ariel", "crisp_and_shine", "dr_wash", "enzo", "ezee", "fab", "fena", "genteel",
        "ghadi", "henko", "more_light", "mr_white", "nirma", "revive", "robin", "safe_wash",
        "snug", "soft_touch", "softouch", "sudz", "sun_plus", "tide", "ujala", "vanish",
        "doy", "ponvandu", "ranipal", "safed", "stainfree", "super_chek", "uni_wash", "woosh",
        "3_in_one", "amar", "amber", "arasan", "blue", "bold", "cargo", "classic", "clerby",
        "commando", "dazzl", "deepak", "discound", "do_it", "double_diamond", "fearless",
        "friends", "ganga", "hipolin", "just_out", "khatnil", "klia", "laser", "magic",
        "monkey_555", "more_on", "navrang", "novel", "orbit", "power", "pur", "race",
        "red_zx", "sams", "satto", "shashi", "sleek", "solar", "special_super", "spesh",
        "splitz", "starpower", "stop", "subhra", "sunny", "sunshine", "super", "surya",
        "thunder", "toofan", "triple_power", "tyro", "udaya", "ultimate_power", "wells",
        "wellworth", "zefrich",
    ),
    "Oral_Care": (
        "colgate", "dabur", "oralb", "oral_b", "sensodyne", "anchor", "babool", "crest", "gumcare",
        "k_p_namboodiris", "listerene", "listerine", "meswak", "promise", "vicco_vajradanti",
        "aim", "b_b", "charcoal", "herbatol_plus",
    ),
    "Hair_Care": (
        "ashwini", "bajaj", "biolage", "chik", "dhathri", "hair_vit", "head_and_shoulders",
        "karthika", "kesh_king", "khadi", "livon", "loreal", "matrix", "meera", "navratna",
        "nyle", "pantene", "parachute", "revlon", "sesa", "streax", "vasmol", "vatika", "vcare",
        "amla", "anuved", "bblunt", "bigen", "color_mate", "coconad", "cocosoft", "dr_batra",
        "gliss", "hair_care", "halo", "indica", "keo_karpin", "kaveri", "mediker", "neha_mehndi",
        "nihar", "raavi", "scalp", "shaumaa", "superia", "thulasi", "trichmey",
        "alayam", "arsi", "arasi", "ashtapathy", "ashwa", "ayucare", "ayurvedamix", "banjaras",
        "coco", "coco_plus", "cosmo_silky", "crown", "dhanadan", "full_bloom", "glori",
        "himyogi", "indraja", "jobha", "kairali", "kaivalya", "kanaka", "kasthuri", "kesar",
        "kesri", "krea", "lia", "madhuvana", "mahakal", "my_choice", "natural", "nature_power",
        "natures", "neo_neem", "nourish", "oorvasi", "prakruthi", "pushpaveda", "radhas",
        "raktachandana", "samudra", "spring_bloom", "sri_krishna", "sri_sri", "sundari",
        "surabhi", "thanda_cool", "thann_mann", "tulasi", "vaijayanthi", "vedganga", "vega",
    ),
    "PW": (
        "cinthol", "dermi_cool", "dettol", "dyna", "fiama", "get_real", "ghar", "glimmer",
        "godrej_no_1", "jive", "margo", "medimix", "mysore_sandal", "nycil", "puric",
        "safe_life", "santoor", "savlon", "vivel",
        "alife", "aramusk", "arasan", "camay", "chandrika", "cuticura", "fa", "hygienix",
        "irish_spring", "jo", "kaya_youth", "lervia", "limda", "moha", "mysore", "neem",
        "neko", "nima", "orchid_soap", "palmolive_soap", "parkavenue", "persona", "radox",
        "rexona_soap", "sandal", "sofwash", "super_max", "suthol", "tedibar", "think_skin",
        "aura", "aurlia", "care", "caremate", "cutee", "elaria", "evora", "excellence",
        "first_love", "girl", "goat_milk", "honey_citra", "hugs", "jass", "kiss_me", "leona",
        "lissa", "loli", "lora", "mee_mee", "merak", "mugi", "navish", "neo", "neu", "nine",
        "nova", "osr", "oxycool", "paris", "petals", "pigeon", "pixcy", "posh", "roshi",
        "skye", "soft", "stories", "svas", "symphony", "synx", "tryshy", "tulips", "vep",
        "waso", "zazio",
    ),
    "Skin": (
        "ayur", "banjaras", "biotique", "boroline", "boroplus", "cetaphil", "chicco",
        "clean_and_clear", "dermaco", "dot_and_key", "dr_rashee", "emami", "enchateur",
        "enchanteur", "everyuth", "fem", "foxtale", "garnier", "himalaya", "jergens",
        "johnsons", "joy", "lactoclamine", "lacto_calamine", "lotus", "mamearth", "mamaearth",
        "neutrogena", "nivea", "olay", "palmolive", "patanjali", "pilgrim", "plum",
        "seba_med", "veet", "vicco", "vlcc", "whitetone", "wow",
        "aloevera", "anne_french", "aquasoft", "azafran", "beardo", "Bombay_Shaving_Company",
        "charmis", "dermadew", "eva", "facia", "faiza", "gillette", "himani", "itch_gurad",
        "jovees", "krishna", "lafz", "maybelline", "m_caffeine", "oliva", "oshea", "oxylife",
        "ozone", "ring_guard", "roop_mantra", "skinella", "softsens", "st.botanica",
        "swiss_tempelle", "the_man_company", "ustraa", "viwa", "w2", "yardley",
        "adigo", "aeda", "alba", "austin", "avanti", "azista", "bella", "boric_acid",
        "butterfly", "durex", "faamle", "galaxia", "harmon", "hello", "huggies", "inflame",
        "kara", "mamy_poko_pants", "pampers", "stayfree", "vissco", "whisper",
    ),
    "DMT": (
        "denver", "adidas", "benetton", "charlie", "ck", "cobra", "engage", "fogg", "gatsby",
        "he", "hott", "jovan", "kamasutra", "ks", "layer_shot", "lomani", "nike", "old_spice",
        "ossum", "playboy", "ramsons", "rasasi", "royal_mirage", "secret", "setwet", "skinn",
        "spinz", "vanesa", "wildstone", "xxx", "z",
        "cigar", "jin_x", "kylo", "leonz", "lexus", "magnet", "otto", "predator", "smokey",
    ),
    "Beverages": (
        "avt", "continental", "cothas", "girnar", "levista", "mili", "narasus", "nescafe",
        "nestle", "organic_india", "society", "sunrise", "tata", "tetely", "tetley",
        "twinnings", "twinings", "wagh_bakri",
        "7_up", "adrak_chai", "advanced_filter_coffee", "alo_frut", "b_natural", "bayars",
        "beanies", "bevzilla", "bharath_tea", "cafe_blend", "cavins", "coco_cola", "davidoff",
        "fanta", "gemini", "green_coffee", "green_tea", "instant_filter_coffee", "jeeru",
        "jivraj", "kinley", "kwality_coffee", "lavazza", "limca", "maaza", "mirinda", "monster",
        "nestea", "ocean", "prakash_tea", "rage", "rasna", "raw", "real", "red_bull",
        "sleepy_owl", "smoodh", "sofit", "sprite", "storia", "suma_coffee", "sun_bean",
        "super_cup", "tang", "tea_valley_royal", "teaveda", "thumsup", "tropicana",
        "caspian", "gokul", "green_milk", "himalayan", "jk_royal", "leo", "marudhar",
        "maruti", "premier", "premium", "premium_gold", "royal_gold", "saburi_gold",
    ),
    "HFD": (
        "bio_health", "bournvita", "centrum", "chic_nutrix", "complan", "eno", "ensure",
        "fast_and_up", "fixuphealth", "hk_vitals", "pediasure", "plix",
        "aptamil", "b_fit", "baidyanath", "glucon_d", "glucose-d", "hajmola", "herbodaya",
        "iodex", "kapiva", "manna_rich", "max_protein", "moov", "nan_pro", "nutrela",
        "oveltime", "plantvita", "protinex", "safi", "satisabgoi", "similac", "sugar_free",
        "tiger_balm", "vicks", "volini", "woodwards", "zandu_balm",
        "beehive", "jeeni", "nutri", "nutri_value", "organic",
    ),
    "Foods": (
        "bakers", "chings", "del_monte", "funfoods", "hersheys", "keya", "maggi", "myfitness",
        "nutella", "pintola", "slurrp_farm", "sundrop", "top_ramen", "veeba", "wai_wai",
        "weikfield", "yipee", "yippee",
        "1_to_3", "24_mantra", "3_sisters", "9am", "aachi", "abbies", "act_ii", "ajanta",
        "american_garden", "amul", "amulya", "apis", "ashirvad", "badshah", "bagrrys",
        "balaji", "bambino", "barilla", "basso", "bechef", "bedekar", "betty_crocker",
        "beyond_snack", "bhuira", "bikaji", "bikano", "bingo", "blue_bird", "borges",
        "bragg", "brahmins", "britannia", "cadbury", "capilano", "capsico", "catch",
        "chef_boss", "chefs_basket", "chhedas", "corn_flakes", "cornitos", "cremica",
        "cup_noodles", "daawat", "dalda", "dhara", "doritos", "double_horse", "doublemint",
        "druk", "eastern", "english_oven", "everest", "figaro", "fortune", "foxs", "galaxy",
        "ganesh", "gery", "gits", "gold_winner", "gowardhan", "grb", "haldirams", "heinz",
        "hello_panda", "id", "india_gate", "indiras", "instant_popcorn", "itc", "jabsons",
        "kalverts", "kelloggys_chocos", "kellogs_corn_flakes", "kinder", "kitkat",
        "kitchen_treasures", "kohinoor", "koka", "koko_krunch", "kraft", "kurkure", "laocker",
        "lays", "lotte_choco_pie", "m&m", "m_d_h_chat_masala", "mackays", "madhusudan_desi_ghee",
        "malas", "malkist", "mapro", "mars", "mccain", "milkfood", "milky_mist", "monaco",
        "morton", "mother_garlic", "mothers_dairy", "mr_vities", "mtr", "muesli", "munchys",
        "nandos", "nanma foods", "nilons", "nutralite", "open_secret", "orbit_mints", "oreo",
        "panzani", "paras_ghee", "parle", "parrys", "pillsburry", "polo", "pravin", "pringles",
        "priya", "pro_nature", "quaker", "ragu", "rajasthan_garlic_chutney", "rajdhani",
        "rasoi magic", "real_thai", "remia", "sadguru_gram_flour", "safal", "saffola", "sakthi",
        "salsalito", "shan", "sil", "skippy", "skittles", "smith_jones", "snac_tac", "snapin",
        "snickers", "sorrentina", "spice_story", "sprig", "st.dalfour", "stute", "sugar_lite",
        "suhana", "sunfeast", "swad", "swastiks", "tabasco", "tamar_hind", "tasty_nibbles",
        "the_sauce_co", "the_snackhack", "tic_tac", "too_yumm", "tops", "twix", "unibic",
        "uttam_spices", "wingreens_form", "woh_hup", "zissto",
        "aarvees", "anil", "apna", "archi", "balian", "bbms", "benison", "bicano", "bihari_ji",
        "bites", "briyas", "bruno", "c_and_g", "cfs", "cheese", "chellam", "cherry", "chocolush",
        "com_pure", "cornado", "doko", "eatopia", "elite", "famus", "faramz", "farmers_harvest",
        "feasto", "fun_top", "garden", "geebeess", "global", "go_go", "gold", "goldrops",
        "grain_grace", "guud", "habit", "haeal", "hamly", "harima", "hillers", "hot_spicy",
        "ht_mart", "ikrish", "inchi", "italian_flavor", "jaambu", "jevton", "k_pra", "kato",
        "key_brand", "khaass", "kimball", "kims", "kishores", "krmp", "kudos", "lion", "mamos",
        "muffins", "ongs", "original_apna", "others3", "palat", "panvel", "pedigree", "pineapple",
        "pingani", "pipo", "popzo", "purepet", "raj", "rinkup", "rusk", "sao", "sarwar",
        "savour", "sds", "skr", "spicezzaa", "srb", "teju", "temptin", "this", "whiskas",
    ),
    "HHC": (
        "citric_plus", "colin", "exo", "expelz", "freshwrapp", "giffy", "harpic", "homeguard",
        "lizol", "max_kleen", "mopz", "mr_muscle", "nimyle", "pril", "sani_fresh", "scrubz",
        "aer", "air_wick", "all_out", "ambi_pur", "bygon_max", "cello", "clorox", "finish",
        "gainda", "good_home", "herbal_startegi", "hit", "indian_foil", "just_clean",
        "kadliwrap", "kiwi", "kiwi_dranex", "klean_x", "klinol", "laxmanrekha", "mangaldeep",
        "maxo", "mortein", "nip", "odomos", "odonil", "odopic", "origami", "pitambari",
        "prestige", "sabena", "sanigent", "scotch_brite", "stan_fresh", "steele_brite",
        "trishul", "vooki", "xpert", "zipwrap",
        "duracell", "eveready", "fevi_kwick", "graphite", "havells", "hem", "kangaroo",
        "philips", "vh_candles",
    ),
}

NON_HUL_BRAND_CATEGORY_MAP: dict[str, str] = {
    brand.lower(): cat
    for cat, brand_tuple in _NON_HUL_BRANDS_BY_CATEGORY.items()
    for brand in brand_tuple
}

HUL_BRAND_CANONICAL: dict[str, tuple[str, str]] = {
    b: (BRAND_ALIAS_MAP.get(b, b.replace("_", " ").title().replace(" ", "_")), cat)
    for b, cat in BRAND_CATEGORY_MAP.items()
}

NON_HUL_BRAND_CANONICAL: dict[str, tuple[str, str]] = {
    b.lower(): (BRAND_ALIAS_MAP.get(b.lower(), A_PREFIX_TWINS.get(b.lower(), b)), cat)
    for cat, brand_tuple in _NON_HUL_BRANDS_BY_CATEGORY.items()
    for b in brand_tuple
}

MULTI_CATEGORY_BRAND_PROFILES: dict[str, tuple[str, ...]] = {
    "dove": ("Skin", "Hair_Care", "PW", "DMT"),
    "ayush": ("Skin", "Oral_Care", "Hair_Care", "PW"),
    "lbp": ("Hair_Care", "Skin", "PW"),
    "lifebuoy": ("PW", "HHC", "Skin"),
    "rexona": ("DMT", "PW"),
    "tata": ("Beverages", "Foods"),
    "patanjali": ("Skin", "Oral_Care", "Hair_Care", "PW", "Foods", "HHC", "Laundry"),
    "himalaya": ("Skin", "Hair_Care", "Oral_Care", "PW", "HFD"),
    "dettol": ("PW", "HHC", "Laundry", "Skin"),
    "savlon": ("PW", "HHC", "Laundry", "Skin"),
    "dabur": ("Oral_Care", "Skin", "Hair_Care", "Foods", "HFD", "HHC"),
    "emami": ("Skin", "Hair_Care", "HFD"),
    "nivea": ("Skin", "PW", "DMT"),
    "garnier": ("Skin", "Hair_Care"),
    "biotique": ("Skin", "Hair_Care", "PW"),
    "lotus": ("Skin", "Hair_Care"),
    "joy": ("Skin", "PW"),
    "johnsons": ("Skin", "Hair_Care", "PW"),
    "cinthol": ("PW", "Skin", "DMT"),
    "dyna": ("PW", "Skin"),
    "medimix": ("PW", "Skin"),
    "santoor": ("PW", "Skin"),
    "chicco": ("Skin", "Hair_Care", "PW"),
    "palmolive": ("Skin", "PW"),
    "loreal": ("Hair_Care", "Skin"),
    "mamearth": ("Skin", "Hair_Care", "PW"),
    "mamaearth": ("Skin", "Hair_Care", "PW"),
    "nestle": ("Beverages", "Foods", "HFD"),
    "vicco": ("Skin", "Oral_Care"),
}

HFD_SUPPLEMENT_BRANDS: set[str] = {
    "wellbeing_nutrition", "oziva", "hk_vitals", "bio_health", "plix",
    "fast_and_up", "fixuphealth", "centrum", "complan", "bournvita",
    "pediasure", "ensure", "eno", "boost", "horlicks", "maltova", "viva",
}

TREE_DEFAULT_CATEGORY: dict[str, str] = {
    "variant_skin": "Skin",
    "variant_wash": "PW",
    "variant_p4": "Oral_Care",
    "variant_p5": "Foods",
    "variant_gen": "Hair_Care",
    "variant_oral": "Oral_Care",
}

PACK_CATEGORY_ALIAS: dict[str, str] = {"Hair": "Hair_Care"}

_COMPOUND_WORD_SUBS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"([a-z])([A-Z])"), r"\1_\2"),
    (re.compile(r"(?i)(hand|body|face|mouth|dish|fabric|hair|tooth|eye|lip|foot|intimate)(wash|paste|powder|brush|cream|lotion|scrub|pack|mask|gel|oil|color|colour|serum|soap|bars?|sanitizer|sanitiser|sanetizer|cleaner|cleanser)"), r"\1_\2"),
    (re.compile(r"(?i)(shaving|bathing|washing|cleansing|moisturizing|moisturising|nourishing|purifying|refreshing)(cream|bar|soap|bars|gel|lotion|milk|foam|powder)"), r"\1_\2"),
    (re.compile(r"(?i)(soap)(bars?)"), r"\1_\2"),
    (re.compile(r"(?i)(coffee|tea)(lovers?|bags?|granules?)"), r"\1_\2"),
]

_FUZZY_CATEGORY_TARGETS: list[tuple[str, str]] = [
    ("toothpaste", "Oral_Care"), ("toothpowder", "Oral_Care"), ("toothbrush", "Oral_Care"),
    ("mouthwash", "Oral_Care"), ("shampoo", "Hair_Care"), ("conditioner", "Hair_Care"),
    ("dandruff", "Hair_Care"), ("hairfall", "Hair_Care"), ("detergent", "Laundry"),
    ("disinfectant", "HHC"), ("dishwash", "HHC"), ("facewash", "Skin"),
    ("moisturizer", "Skin"), ("moisturiser", "Skin"), ("sunscreen", "Skin"),
    ("bodywash", "PW"), ("handwash", "PW"),
]

VARIANT_CATEGORY_KEYWORD_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?:^|_)(toothpaste|tooth_paste|tooth_gel|toothpowder|tooth_powder|toothbrush|tooth_brush|toothbrushes|tooth_brushes|mouthwash|mouth_wash|dental|anti_cavity|sensitivity_relief|clove_oil|dant_kanti|lal_dant_manjan|meswak|babool|freshness_gel_cardamom|purifying_turmeric_b4g1|purifying_turmeric)(?:$|_)",
            re.I,
        ),
        "Oral_Care",
    ),
    (
        re.compile(
            r"(?:^|_)(shampoo|shampoos|conditioner|conditioners|conditionar|hair_mask|deep_repair_mask|moisture_mask|strength_mask|hair_oil|hair_colour|hair_color|hair_cream|hair_serum|hair_care|scalp|dandruff|anti_dandruff|antidandruff|bringha|bhringaraj|shikakai|reetha|amla|hibiscus|hair_fall|hairfall|anti_hairfall|damage_repair|dryness_defense|detangler|almond_drops|cool_oil|6oil_nourish|black_naturals|color_naturals|color_natural|beautiful_curls|healthy_rituals|daily_shine|nourishing_oil_care|oxygen_moisture|environment_defence|peptide_bond_strength|mencare_[a-z0-9_]*2_in_1|intense_damage_repair__?mask|intense_repair__?mask)(?:$|_)",
            re.I,
        ),
        "Hair_Care",
    ),
    (
        re.compile(r"(?:^|_)(roll_on|deodorant|deodrant|deo_spray|body_spray|antiperspirant|anti_perspirant)(?:$|_)", re.I),
        "DMT",
    ),
    (
        re.compile(
            r"(?:^|_)(laundry_sanitiser|laundry_sanitizer|laundry_sanetizer|laundry_disinfectant|germ_removal_detergent|detergent_powder|detergent_liquid|liquid_detergent|detergent_bar|detergen_bar|detergent|washing_powder|fabric_conditioner|fab_conditioner|fabric_wash|matic_liquid|matic_front_load|matic_top_load|stain_remover|after_wash)(?:$|_)",
            re.I,
        ),
        "Laundry",
    ),
    (
        re.compile(
            r"(?:^|_)(surface_cleaner|floor_cleaner|glass_cleaner|kitchen_cleaner|bathroom_cleanser|toilet_cleaner|fruit_and_vegetable_wash|dishwash|dish_wash|surface_disinfectant|disinfectant_surface|disinfectant_liquid|liquid_disinfectant|disinfectant_spray|disinfictant_spray|surface_wipe|antiseptic_liquid|hand_sanitizer|hand_sanitiser|hand_sanetizer|sanitiser|sanitizer)(?:$|_)",
            re.I,
        ),
        "HHC",
    ),
    (
        re.compile(
            r"(?:^|_)(soap|soaps|soap_bar|soap_bars|bath_soap|bathing_bar|bodywash|body_wash|shower_gel|handwash|hand_wash|top_to_toe_bath|prickly_heat_powder)(?:$|_)",
            re.I,
        ),
        "PW",
    ),
    (
        re.compile(
            r"(?:^|_)(chai|tea_bags|green_tea|teaveda|darjeeling|kahwa|cappuccino|espresso|latte|filter_coffee|cold_coffee|coffee_gold|coffee_grand|coffee_pouch|coffelovers|chakra_gold|kanan_devan|agni)(?:$|_)",
            re.I,
        ),
        "Beverages",
    ),
    (
        re.compile(
            r"(?:^|_)(sampann|soulful|ragi_bites|rock_salt|salt_crystal|quick_quisine|custard_powder|mustard|mayonnaise|ketchup|pasta|noodles|masala|peanut_butter|barbeque)(?:$|_)",
            re.I,
        ),
        "Foods",
    ),
    (
        re.compile(
            r"(?:^|_)(facewash|face_wash|facwash|face_cream|day_cream|night_cream|skin_cream|shaving_cream|shavingcream|sunscreen|suncreen|sunblock|toner|astringent|cleanser|cleansing_milk|bodylotion|body_lotion|body_lotions|hand_body_lotion|body_milk|bodyscrub|body_scrub|apricot_scrub|walnut_scrub|face_pack|facepack|face_makeup|sheet_mask|clay_mask|peel_off_mask|lip_mask|sleeping_mask|moisturizer|moisturiser|moistrurizer|moisturising|moisturizing|petroleum_jelly|petroleumjelly|gulabari|aloe_vera_gel|aloevera_gel|saundarya)(?:$|_)",
            re.I,
        ),
        "Skin",
    ),
]


def _damerau_levenshtein_le1(a: str, b: str) -> bool:
    """Fast O(L) check returning True iff Damerau-Levenshtein edit distance between ``a`` and ``b`` is <= 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or min(la, lb) < 5:
        return False
    i = 0
    while i < la and i < lb and a[i] == b[i]:
        i += 1
    if la == lb:
        if a[i + 1 :] == b[i + 1 :]:
            return True
        return i + 1 < la and a[i] == b[i + 1] and a[i + 1] == b[i] and a[i + 2 :] == b[i + 2 :]
    return a[i + 1 :] == b[i:] if la > lb else a[i:] == b[i + 1 :]


def normalize_variant_tokens(variant: str) -> str:
    """Split camelCase, spaces, and fused compound words into lowercase underscore-delimited tokens."""
    s = variant.strip()
    for pat, repl in _COMPOUND_WORD_SUBS:
        s = pat.sub(repl, s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return re.sub(r"_+", "_", s)


def normalize_category_name(category: str) -> str:
    """Normalize any category string to one of the 10 canonical Unilever categories."""
    if category in CANONICAL_CATEGORIES:
        return category
    low = category.strip().lower().replace(" ", "_")
    return CATEGORY_SYNONYM_MAP.get(low, "Foods")


def _infer_brand_category(key: str) -> str:
    """Resolve brand-level default category from HUL_BRANDS, NON_HUL_BRAND_CATEGORY_MAP, or lexical tokens."""
    k = key.lower()
    if k in BRAND_CATEGORY_MAP:
        return BRAND_CATEGORY_MAP[k]
    if k in NON_HUL_BRAND_CATEGORY_MAP:
        return NON_HUL_BRAND_CATEGORY_MAP[k]
    norm = normalize_variant_tokens(k)
    for kw_re, cat in VARIANT_CATEGORY_KEYWORD_RULES:
        if kw_re.search(norm):
            return cat
    toks = set(norm.split("_"))
    if toks & {"tea", "coffee", "chai", "cola", "juice", "drink", "water"}:
        return "Beverages"
    if toks & {"ghee", "spices", "masala", "foods", "garlic", "chutney", "flour", "snacks", "biscuit", "chocolate", "mints", "popcorn", "sauce", "muesli", "flakes", "pickle"}:
        return "Foods"
    if toks & {"soap", "wash", "sandal", "bath"}:
        return "PW"
    if toks & {"balm", "health", "vita", "nutrition", "glucose", "protein"}:
        return "HFD"
    if toks & {"clean", "fresh", "foil", "wrap", "brite"}:
        return "HHC"
    return "Foods"
