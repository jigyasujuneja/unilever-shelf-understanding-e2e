"""Catalog Enrichment & Catalog Enrichment and Multi-Band Crop Canvas Utilities.

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  1. enrich_catalog_entry(p)                                                │
  │     • Recovers missing size (164/184 SKUs), variant (184/184), pack_form   │
  │       (bottle/pouch/sachet/tub/bar), and true is_unilever ownership        │
  │       (97 HUL vs 87 Competitor SKUs).                                      │
  │  2. build_pack_and_zoom_canvas(image)                                      │
  │     • Composes full product photo on left + middle variant band &          │
  │       bottom-35% net-weight band on right.                                 │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import re

from PIL import Image, ImageDraw

UNILEVER_BRANDS = {
    "axe", "breeze", "clear", "clinic plus", "closeup", "close-up", "comfort", "cornetto",
    "creamsilk", "cream silk", "domex", "dove", "eskinol", "fair & lovely", "glow & lovely",
    "hellmann's", "knorr", "lady's choice", "lifebuoy", "lipton", "liril", "lux", "magnum",
    "master", "pears", "pepsodent", "pond's", "ponds", "rexona", "selecta", "signal",
    "simple", "st. ives", "sun", "sunsilk", "surf", "tresemme", "tresemmé", "vaseline",
    "vim", "wall's",
}
SIZE_RE = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:ml|l|g|kg|oz|pcs|sachets?|pack|tabs?|cubes?))\b", re.I)
PACK_FORM_RE = re.compile(
    r"\b(sachet|pouch|bottle|jar|tub|tube|bar|box|can|stick|roll-?on|refill|multipack|cube|mix)\b",
    re.I,
)


def enrich_catalog_entry(p: dict) -> dict:
    """Auto-recover missing size, variant, pack form, and true HUL vs Competitor flag."""
    pname = str(p.get("product", ""))
    brand = str(p.get("brand", ""))
    cat = str(p.get("category", ""))
    size = str(p.get("size", "")).strip()
    if not size:
        m = SIZE_RE.search(pname)
        if m:
            size = m.group(1).replace(" ", "").lower()
    form_m = PACK_FORM_RE.search(pname)
    pack_form = form_m.group(1).lower() if form_m else "standard_pack"

    variant = re.sub(re.escape(brand), "", pname, flags=re.I)
    if size:
        variant = re.sub(re.escape(size), "", variant, flags=re.I)
    variant = re.sub(r"[^a-zA-Z0-9\s]", " ", variant)
    variant = " ".join(variant.split()).strip() or pname

    from utils.hul_taxonomy import normalize_brand_folder

    if "is_unilever" in p:
        is_hul = bool(p["is_unilever"])
    elif "is_hul" in p:
        is_hul = bool(p["is_hul"])
    else:
        nb = normalize_brand_folder(brand)
        is_hul = brand.strip().lower() in UNILEVER_BRANDS or bool(nb and nb.get("is_hul", False))
    return {
        **p,
        "size": size,
        "variant": variant,
        "pack_form": pack_form,
        "is_unilever": is_hul,
        "enriched_text": (
            f"{brand} | {cat} | {pname} | variant: {variant} | "
            f"form: {pack_form} | net weight size: {size or 'standard'}"
        ),
    }


def build_pack_and_zoom_canvas(image: Image.Image) -> Image.Image:
    """Compose crisp full product photo on left + 2x zoomed bottom-35% net-weight/variant strip on right."""
    rgb = image.convert("RGB")
    w, h = max(1, rgb.width), max(1, rgb.height)
    canvas = Image.new("RGB", (640, 380), "white")
    main = rgb.copy()
    main.thumbnail((310, 360))
    canvas.paste(main, ((320 - main.width) // 2, (380 - main.height) // 2))

    mid_band = rgb.crop((int(w * 0.08), int(h * 0.30), int(w * 0.92), int(h * 0.68)))
    mid_band.thumbnail((300, 175))
    canvas.paste(mid_band, (330 + (300 - mid_band.width) // 2, 10 + (175 - mid_band.height) // 2))

    bot_band = rgb.crop((int(w * 0.08), int(h * 0.65), int(w * 0.92), int(h * 0.96)))
    bot_band.thumbnail((300, 175))
    canvas.paste(bot_band, (330 + (300 - bot_band.width) // 2, 195 + (175 - bot_band.height) // 2))

    draw = ImageDraw.Draw(canvas)
    draw.line((320, 0, 320, 380), fill="#ccc", width=2)
    draw.line((320, 190, 640, 190), fill="#eee", width=1)
    return canvas
