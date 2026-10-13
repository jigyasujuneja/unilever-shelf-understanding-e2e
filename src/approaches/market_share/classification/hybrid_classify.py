"""Embedding-first hybrid cascade for all 5 HUL Golden attributes (``hybrid_classify``):
``product``, ``brand``, ``base_variant``, ``packaging_type``, and ``category``.

Uses the cheap embedding answer when it is confident across all evaluated fields and pays for one
LLM call only for the crops it isn't sure about.

  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ Stage A  kNN (1 embedding call): gemini-embedding-2 crop vector -> CLM InfoNCE ranking   │
  │          of the train-split reference gallery (``ClmStandaloneRetrieval.ranked``: de-    │
  │          glare, multi-view max cosine + SKU centroid). Labels = top-1 SKU's 5 fields     │
  │          (product, brand, base_variant, packaging_type, category); confidence per field  │
  │          = similarity-weighted vote share of that label among the top-k SKUs (softmax    │
  │          at 1/tau), plus the top-1 energy and the top1-top2 margin.                      │
  │ Gate     every evaluated field's vote share >= its threshold and top-1 energy >=         │
  │          min_energy -> return Stage A (0 LLM calls). Thresholds are calibrated on val.   │
  │ Stage B  escalation, one call to the run's -m model:                                     │
  │            contact_sheet  crop + top-k gallery references (CLEF zoom sheet, logprob      │
  │                           readout, ``ClefStandaloneRetrieval._clef_single``) -> a SKU    │
  │                           yielding all 5 canonical fields                                │
  │            sft            the HUL Golden 5-attr (or 3-attr) SFT prompt                   │
  │            sft_hints      same prompt + the kNN 5-attribute candidates as text hints     │
  │          merge=field keeps the confident Stage A fields and takes only the uncertain     │
  │          ones from Stage B (re-deriving product = Brand__Variant); merge=all takes all   │
  │          fields from Stage B.                                                            │
  │ Stage C  taxonomy & SKU consistency (free): snaps base_variant to canonical catalog      │
  │          variant for brand, aligns product = Brand__Variant, resolves category from      │
  │          SKU/brand catalog, and projects (category, packaging_type) onto a valid pair.   │
  └──────────────────────────────────────────────────────────────────────────────────────────┘

Parameters (class defaults < ``config.yaml`` ``hybrid_classify:`` section < env
``SHELF_BENCH_HYBRID_<NAME>``): ``stage_b``, ``merge``, ``tau_brand``, ``tau_base_variant``,
``tau_product``, ``tau_packaging_type``, ``tau_category``, ``min_energy``, ``inv_temp``,
``top_k``, ``taxonomy``, ``gallery``, ``gallery_exclude``, ``sheet_candidates``.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from PIL import Image

from approaches.base import Context, register
from approaches.market_share.retrieval.clef_standalone import ClefStandaloneRetrieval
from approaches.market_share.retrieval.clm_standalone import ClmStandaloneRetrieval
from approaches.market_share.retrieval.embedding_retrieval import MODEL, EmbeddingRetrieval
from utils import dataset
from utils.cv_models import resolve_evaluated_fields
from utils.embeddings import SKUS
from utils.gemini_fine_tuning.hul_golden_sft import (
    HUL_GOLDEN_3ATTR_FIELDS,
    HUL_GOLDEN_5ATTR_FIELDS,
    HUL_GOLDEN_5ATTR_INFERENCE_PROMPT,
    HUL_GOLDEN_INFERENCE_PROMPT,
    catalog_variant_lookup,
    normalize_hul_golden_brand,
    normalize_hul_golden_variant,
)
from utils.taxonomy.packaging import project_to_valid_category_pack

FIELDS: tuple[str, ...] = HUL_GOLDEN_5ATTR_FIELDS
FIELDS_3ATTR: tuple[str, ...] = HUL_GOLDEN_3ATTR_FIELDS
STAGE_B_VARIANTS = ("contact_sheet", "sft", "sft_hints", "none")
HINTS_MAX = 3

HUL_GOLDEN_5ATTR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "brand": {"type": "string"},
        "base_variant": {"type": "string"},
        "packaging_type": {"type": "string"},
        "category": {"type": "string"},
    },
    "required": list(FIELDS),
}

HUL_GOLDEN_3ATTR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "brand": {"type": "string"},
        "packaging_type": {"type": "string"},
        "category": {"type": "string"},
    },
    "required": list(FIELDS_3ATTR),
}

# Backward-compatible alias
HUL_GOLDEN_SCHEMA = HUL_GOLDEN_5ATTR_SCHEMA


def field_votes(
    shortlist: list[tuple[float, dict]],
    inv_temp: float = 80.0,
    k: int = 5,
    fields: Sequence[str] = FIELDS,
) -> dict[str, dict[str, Any]]:
    """Per-field similarity-weighted vote over the top-``k`` (score, labels) neighbours.

    Weights are ``softmax(inv_temp * (s_i - s_1))``. The answer is the top-1 neighbour's label (so
    all fields stay a real catalog SKU combination); ``conf`` is the vote share of that label
    and ``margin`` its lead over the best other label.
    """
    all_f = tuple(dict.fromkeys([*fields, *FIELDS]))
    top = shortlist[:k]
    if not top:
        return {f: {"label": None, "conf": 0.0, "margin": 0.0, "votes": {}} for f in all_f}
    s1 = top[0][0]
    w = [math.exp(max(-30.0, inv_temp * (s - s1))) for s, _ in top]
    z = sum(w) or 1.0
    out: dict[str, dict[str, Any]] = {}
    for f in all_f:
        votes: dict[str, float] = defaultdict(float)
        for (_, lab), wi in zip(top, w, strict=True):
            val = lab.get(f)
            if val is None and f == "product" and lab.get("brand"):
                bv = lab.get("base_variant") or lab.get("variant") or lab.get("brand")
                val = f"{lab['brand']}__{bv}" if bv != lab["brand"] else lab["brand"]
            elif val is None and f == "base_variant":
                val = lab.get("variant") or lab.get("brand")
            votes[str(val or "")] += wi / z
        top_lab = top[0][1]
        top_val = top_lab.get(f)
        if top_val is None and f == "product" and top_lab.get("brand"):
            bv0 = top_lab.get("base_variant") or top_lab.get("variant") or top_lab.get("brand")
            top_val = f"{top_lab['brand']}__{bv0}" if bv0 != top_lab["brand"] else top_lab["brand"]
        elif top_val is None and f == "base_variant":
            top_val = top_lab.get("variant") or top_lab.get("brand")
        label = str(top_val or "")
        conf = votes[label]
        others = [v for k2, v in votes.items() if k2 != label]
        out[f] = {
            "label": label,
            "conf": round(conf, 4),
            "margin": round(conf - (max(others) if others else 0.0), 4),
            "votes": {k2: round(v, 4) for k2, v in sorted(votes.items(), key=lambda kv: -kv[1])},
        }
    return out


def gate(
    votes: dict[str, dict[str, Any]],
    top_energy: float,
    thresholds: dict[str, float],
    min_energy: float,
    fields: Sequence[str] | None = None,
) -> tuple[list[str], str]:
    """Fields Stage A is unsure about (empty = accept Stage A) and a human-readable reason."""
    active_fields = [f for f in (fields or tuple(thresholds.keys())) if f in votes]
    if not active_fields:
        active_fields = [f for f in FIELDS if f in thresholds and f in votes]
    if top_energy < min_energy:
        return list(active_fields), f"top-1 energy {top_energy:.3f} < {min_energy}"
    unsure = [f for f in active_fields if votes[f]["conf"] < thresholds.get(f, 0.0)]
    if not unsure:
        return [], "all fields confident"
    return unsure, ", ".join(f"{f} vote {votes[f]['conf']:.2f} < {thresholds.get(f, 0.0)}" for f in unsure)


def merge_fields(
    stage_a: dict,
    stage_b: dict | None,
    unsure: list[str],
    merge: str,
    fields: Sequence[str] = FIELDS,
) -> dict:
    """``merge='field'``: uncertain fields from Stage B, the rest from Stage A; ``'all'``: Stage B."""
    all_f = tuple(dict.fromkeys([*fields, *(stage_a.keys()), *((stage_b or {}).keys())]))
    if not stage_b:
        return dict(stage_a)
    if merge == "all":
        merged = {f: stage_b.get(f) or stage_a.get(f) for f in all_f}
    else:
        unsure_set = set(unsure)
        if "brand" in unsure_set or "base_variant" in unsure_set:
            unsure_set.add("product")
        if "product" in unsure_set:
            unsure_set.add("base_variant")
        merged = {
            f: (stage_b.get(f) or stage_a.get(f)) if f in unsure_set else stage_a.get(f)
            for f in all_f
        }
    if merged.get("brand") and merged.get("base_variant"):
        b = str(merged["brand"])
        bv = str(merged["base_variant"])
        merged["product"] = f"{b}__{bv}" if bv != b else b
    return merged


def taxonomy_index(catalog: dict[int, dict]) -> tuple[dict[str, set[str]], set[tuple[str, str]]]:
    """``(brand -> categories, valid (category, packaging_type) pairs)`` seen in the catalog."""
    brand_cats: dict[str, set[str]] = defaultdict(set)
    pairs: set[tuple[str, str]] = set()
    for p in catalog.values():
        b, c = str(p.get("brand") or ""), str(p.get("category") or "")
        if b and c:
            brand_cats[b].add(c)
        for pt in [p.get("packaging_type"), *(p.get("allowed_packaging_types") or [])]:
            if c and pt:
                pairs.add((c, str(pt)))
    return dict(brand_cats), pairs


def project_taxonomy(
    pred: dict,
    brand_cats: dict[str, set[str]],
    valid_pairs: set[tuple[str, str]],
    *,
    variants_by_brand: dict[str, list[str]] | None = None,
    sku_meta_by_bv: dict[tuple[str, str], dict[str, str]] | None = None,
) -> tuple[dict, list[str]]:
    """Stage C: make (product, brand, base_variant, category, packaging_type) a valid combination."""
    out = dict(pred)
    changed: list[str] = []
    brand = str(out.get("brand") or "")

    if "base_variant" in out and out.get("base_variant"):
        raw_bv = str(out["base_variant"])
        norm_bv = normalize_hul_golden_variant(
            raw_bv,
            brand=brand,
            catalog_variants_by_brand=variants_by_brand,
        )
        if norm_bv and norm_bv != raw_bv:
            out["base_variant"] = norm_bv
            changed.append("base_variant")
        if brand and out.get("base_variant"):
            bv_final = str(out["base_variant"])
            expected_prod = f"{brand}__{bv_final}" if bv_final != brand else brand
            if out.get("product") != expected_prod:
                out["product"] = expected_prod
                if "product" not in changed:
                    changed.append("product")

    sku_meta = (
        sku_meta_by_bv.get((brand.lower(), str(out.get("base_variant") or "").lower()))
        if (sku_meta_by_bv and brand and out.get("base_variant"))
        else None
    )
    if sku_meta and sku_meta.get("category") and out.get("category") != sku_meta["category"]:
        out["category"] = sku_meta["category"]
        changed.append("category")
    else:
        cats = brand_cats.get(brand)
        if cats and len(cats) == 1 and out.get("category") not in cats:
            out["category"] = next(iter(cats))
            changed.append("category")

    cat, pt = str(out.get("category") or ""), str(out.get("packaging_type") or "")
    if cat and pt and (cat, pt) not in valid_pairs:
        proj = project_to_valid_category_pack(cat, pt)
        if proj != pt and (not valid_pairs or (cat, proj) in valid_pairs):
            out["packaging_type"] = proj
            changed.append("packaging_type")
    return out, changed


def hint_prompt(
    candidates: list[tuple[float, dict]],
    max_hints: int = HINTS_MAX,
    fields: Sequence[str] | None = None,
) -> str:
    """The SFT prompt plus up to ``max_hints`` distinct kNN label combinations as short hints."""
    if fields is None:
        has_5attr = any(bool(lab.get("base_variant") or lab.get("product")) for _, lab in candidates)
        use_5attr = has_5attr
    else:
        use_5attr = bool({"product", "base_variant"} & set(fields))

    base_prompt = HUL_GOLDEN_5ATTR_INFERENCE_PROMPT if use_5attr else HUL_GOLDEN_INFERENCE_PROMPT
    if not candidates:
        return base_prompt

    if use_5attr:
        seen5: list[tuple[str, str, str, str, str, float]] = []
        for s, lab in candidates:
            b = str(lab.get("brand") or "")
            bv = str(lab.get("base_variant") or lab.get("variant") or b)
            prod = str(lab.get("product") or (f"{b}__{bv}" if (b and bv and bv != b) else b))
            p = str(lab.get("packaging_type") or "")
            c = str(lab.get("category") or "")
            key5 = (prod, b, bv, p, c)
            if key5 not in [x[:5] for x in seen5]:
                seen5.append((*key5, s))
            if len(seen5) >= max_hints:
                break
        lines = "; ".join(
            f"product={prod}, brand={b}, base_variant={bv}, packaging_type={p}, category={c} (similarity {s:.2f})"
            for prod, b, bv, p, c, s in seen5
        )
        return (
            f"{base_prompt} Visual-search hints from similar catalog photos, most similar "
            f"first (they can be wrong; trust what you see in the image): {lines}."
        )

    seen3: list[tuple[str, str, str, float]] = []
    for s, lab in candidates:
        key3 = tuple(str(lab.get(f) or "") for f in FIELDS_3ATTR)
        if key3 not in [x[:3] for x in seen3]:
            seen3.append((*key3, s))
        if len(seen3) >= max_hints:
            break
    lines = "; ".join(f"brand={b}, packaging_type={p}, category={c} (similarity {s:.2f})" for b, p, c, s in seen3)
    return (
        f"{base_prompt} Visual-search hints from similar catalog photos, most similar "
        f"first (they can be wrong; trust what you see in the image): {lines}."
    )


def load_uri_list(src: str | None) -> list[str]:
    """A JSON list of URIs from a local path or ``gs://`` URI ('' -> [])."""
    if not src:
        return []
    import json

    raw = dataset.read_bytes(src) if str(src).startswith("gs://") else open(src, "rb").read()  # noqa: SIM115
    data = json.loads(raw)
    return [str(u) for u in (data.get("uris", []) if isinstance(data, dict) else data)]


def _env_or(cfg: dict, name: str, default: Any) -> Any:
    raw = os.environ.get(f"SHELF_BENCH_HYBRID_{name.upper()}")
    val = raw if raw not in (None, "") else cfg.get(name, default)
    if isinstance(default, bool):
        return str(val).strip().lower() in ("1", "true", "yes", "on") if isinstance(val, str) else bool(val)
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        return type(default)(val)
    return val


@register
class HybridClassify(ClmStandaloneRetrieval):
    """Embedding kNN when confident across all 5 fields, else one LLM call (contact sheet or SFT), then taxonomy."""

    name = "hybrid_classify"
    task = "classification"
    dataset = "hul_golden"
    models: list[str] | None = None  # any Gemini (base or tuned) for the escalations
    shortlist_model = MODEL
    also_calls = [MODEL]
    skus = SKUS

    stage_b = "contact_sheet"
    merge = "field"
    tau_brand = 0.9
    tau_base_variant = 0.85
    tau_product = 0.85
    tau_packaging_type = 0.8
    tau_category = 0.9
    min_energy = 0.85
    inv_temp = 160.0
    top_k = 5
    taxonomy = True
    gallery = "gallery"  # "gallery" (rpc.json refs) | "gallery+train" (adds the train split crops)
    gallery_exclude = ""  # JSON list of gallery URIs to drop (local path or gs:// URI)
    sheet_candidates = "clm"

    architecture = (
        f"Hybrid 5-attribute cascade: {MODEL} kNN vote over the reference gallery (0 LLM calls when "
        "product, brand, base_variant, packaging_type, and category are confident) -> else one LLM call "
        "(contact sheet of crop + top-5 refs, or 5-attr SFT + kNN hints) -> taxonomy & SKU projection"
    )
    steps = [
        "Setup: embed the train-split reference gallery (val/test URI/session duplicates excluded)",
        "Stage A: embed the crop, rank gallery SKUs (CLM InfoNCE), similarity-weighted vote across all 5 fields",
        "Gate: all evaluated fields confident and top-1 energy high -> answer with Stage A (no LLM call)",
        "Stage B: otherwise one LLM call (contact sheet / 5-attr SFT / SFT + kNN hints); merge uncertain fields",
        "Stage C: canonicalize base_variant, align product = Brand__Variant, and project (brand, category, packaging_type)",
    ]

    # ---- setup -------------------------------------------------------------------------------

    def _load_params(self, config: dict) -> None:
        cfg = dict(config.get("hybrid_classify") or {})
        for p in (
            "stage_b",
            "merge",
            "tau_brand",
            "tau_base_variant",
            "tau_product",
            "tau_packaging_type",
            "tau_category",
            "min_energy",
            "inv_temp",
            "top_k",
            "taxonomy",
            "gallery",
            "gallery_exclude",
            "sheet_candidates",
        ):
            setattr(self, p, _env_or(cfg, p, getattr(type(self), p)))
        if self.stage_b not in STAGE_B_VARIANTS:
            raise ValueError(f"stage_b must be one of {STAGE_B_VARIANTS}, got {self.stage_b!r}")
        if self.sheet_candidates not in ("clm", "cosine"):
            raise ValueError(f"sheet_candidates must be 'clm' or 'cosine', got {self.sheet_candidates!r}")
        if self.merge not in ("field", "all"):
            raise ValueError(f"merge must be 'field' or 'all', got {self.merge!r}")

    @property
    def params(self) -> dict[str, Any]:
        return {
            p: getattr(self, p)
            for p in (
                "stage_b",
                "merge",
                "tau_brand",
                "tau_base_variant",
                "tau_product",
                "tau_packaging_type",
                "tau_category",
                "min_energy",
                "inv_temp",
                "top_k",
                "taxonomy",
                "gallery",
                "gallery_exclude",
                "sheet_candidates",
            )
        }

    def gallery_refs(self, root: str) -> dict[int, list[str]]:
        """Reference gallery for kNN: rpc.json refs minus any held-out val/test URI or session."""
        from utils.taxonomy.filenames import parse_crop_filename

        def session_of(uri: str) -> str:
            try:
                return str(parse_crop_filename(uri.rsplit("/", 1)[-1]).get("session_id") or "")
            except Exception:  # noqa: BLE001 - unparseable names simply carry no session
                return ""

        refs = {pid: list(paths) for pid, paths in super().gallery_refs(root).items()}
        held_out = [r for sp in ("val", "test") for r in dataset._rpc_split_rows(sp, root)]
        held_uris = {str(r.get("uri") or "") for r in held_out} - {""}
        held_sessions = {str(r.get("session_id") or "") for r in held_out} - {""}
        blocked = set(load_uri_list(self.gallery_exclude)) | held_uris
        sessions: dict[str, str] = {}
        if "train" in str(self.gallery):
            for r in dataset._rpc_split_rows("train", root):
                if r.get("uri") and r.get("products"):
                    sessions[str(r["uri"])] = str(r.get("session_id") or "")
                    refs.setdefault(int(r["products"][0]), []).append(str(r["uri"]))
        out: dict[int, list[str]] = {}
        excluded = 0
        for pid, paths in refs.items():
            keep = [
                p
                for p in dict.fromkeys(paths)
                if p not in blocked and (sessions.get(p) or session_of(p)) not in held_sessions
            ]
            excluded += len(set(paths)) - len(keep)
            if keep:
                out[pid] = keep
        self.gallery_excluded = excluded
        return out

    def setup(self, config: dict, ctx: Context) -> None:
        self._load_params(config)
        if not self.evaluated_fields:
            self.evaluated_fields = tuple(resolve_evaluated_fields(self.dataset, with_variants=True))
        super().setup(config, ctx)
        self.catalog = dataset.catalog(name=self.dataset)
        self.brand_cats, self.valid_pairs = taxonomy_index(self.catalog)
        self.variants_by_brand, self.sku_meta_by_bv = catalog_variant_lookup(self.catalog)
        ctx.trace.step(
            "Hybrid setup",
            f"{len(self.index)} gallery vectors, "
            f"{getattr(self, 'gallery_excluded', 0)} held-out duplicates excluded",
            info={
                "params": self.params,
                "evaluated_fields": list(self.evaluated_fields),
                "gallery_vectors": len(self.index),
                "gallery_excluded": getattr(self, "gallery_excluded", 0),
            },
        )

    # ---- per image ---------------------------------------------------------------------------

    def _active_fields(self) -> tuple[str, ...]:
        return tuple(self.evaluated_fields or FIELDS)

    def _labels(self, sku: int | None) -> dict:
        p = self.catalog.get(sku) if sku is not None else None
        if not p:
            return {}
        brand = str(p.get("brand") or "")
        bv = str(p.get("base_variant") or p.get("variant") or brand)
        prod = str(p.get("product") or (f"{brand}__{bv}" if (brand and bv and bv != brand) else brand))
        full = {
            "product": prod,
            "brand": brand,
            "base_variant": bv,
            "packaging_type": p.get("packaging_type"),
            "category": p.get("category"),
        }
        return {f: full.get(f, p.get(f)) for f in dict.fromkeys([*self._active_fields(), *FIELDS])}

    def stage_a(self, image: Image.Image, ctx: Context) -> dict[str, Any]:
        active = self._active_fields()
        ranked = self.ranked(image, ctx)
        top = ranked[: self.top_k]
        cands = [(s, self._labels(pid)) for s, pid, _ in top]
        votes = field_votes(cands, inv_temp=self.inv_temp, k=self.top_k, fields=active)
        e1 = top[0][0] if top else -1.0
        e2 = top[1][0] if len(top) > 1 else -1.0
        thresholds = {
            "product": self.tau_product,
            "brand": self.tau_brand,
            "base_variant": self.tau_base_variant,
            "packaging_type": self.tau_packaging_type,
            "category": self.tau_category,
        }
        unsure, reason = (
            gate(votes, e1, thresholds, self.min_energy, fields=active)
            if top
            else (list(active), "no neighbours")
        )
        return {
            "ranked": top,
            "cands": cands,
            "labels": {f: votes[f]["label"] for f in dict.fromkeys([*active, *FIELDS])},
            "sku_id": top[0][1] if top else None,
            "votes": votes,
            "energy": round(e1, 4),
            "margin": round(e1 - e2, 4),
            "unsure": unsure,
            "reason": reason,
        }

    def stage_b_call(self, image: Image.Image, a: dict[str, Any], ctx: Context) -> tuple[dict, dict]:
        """One LLM call for an escalated crop; returns (labels, info)."""
        active = self._active_fields()
        use_5attr = bool({"product", "base_variant"} & set(active))
        if self.stage_b == "contact_sheet":
            top = (
                EmbeddingRetrieval.ranked(self, image, ctx)[: self.top_k]
                if self.sheet_candidates == "cosine"
                else a["ranked"]
            )
            if not top:
                return {}, {"error": "no candidates"}
            sku, meta = ClefStandaloneRetrieval._clef_single(self, image, top, ctx)
            return self._labels(sku), {**meta, "sku_id": sku, "candidates": [pid for _, pid, _ in top]}

        if self.stage_b == "sft_hints":
            prompt = hint_prompt(a["cands"], fields=active)
        else:
            prompt = HUL_GOLDEN_5ATTR_INFERENCE_PROMPT if use_5attr else HUL_GOLDEN_INFERENCE_PROMPT
        schema = HUL_GOLDEN_5ATTR_SCHEMA if use_5attr else HUL_GOLDEN_3ATTR_SCHEMA
        res = ctx.ask(image, prompt, schema=schema, max_side=1024)
        data = res.data if isinstance(res.data, dict) else {}
        brand = normalize_hul_golden_brand(str(data.get("brand") or ""))
        pt = str(data.get("packaging_type") or "")
        cat = str(data.get("category") or "")
        labels: dict[str, str] = {
            "brand": brand,
            "packaging_type": pt,
            "category": cat,
        }
        if use_5attr or data.get("base_variant") or data.get("variant") or data.get("product"):
            raw_bv = str(data.get("base_variant") or data.get("variant") or data.get("product") or "")
            bv = normalize_hul_golden_variant(
                raw_bv,
                brand=brand or str(a["labels"].get("brand") or ""),
                catalog_variants_by_brand=getattr(self, "variants_by_brand", None),
            )
            eff_brand = brand or str(a["labels"].get("brand") or "")
            prod = f"{eff_brand}__{bv}" if (eff_brand and bv and bv != eff_brand) else (eff_brand or bv)
            labels["base_variant"] = bv
            labels["product"] = prod
        return {f: v for f, v in labels.items() if v}, {"raw": data}

    def identify(self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None) -> dict | None:
        active = self._active_fields()
        a = self.stage_a(image, ctx)
        neigh = [{"sku_id": pid, "energy": round(s, 4), **self._labels(pid)} for s, pid, _ in a["ranked"]]
        ctx.trace.step(
            "Stage A · kNN vote",
            " · ".join(str(a["labels"].get(f) or "?") for f in active)
            + f" (energy {a['energy']:.3f}, margin {a['margin']:.3f}; "
            + ", ".join(f"{f} {a['votes'][f]['conf']:.2f}" for f in active)
            + ")",
            info={"neighbours": neigh, "votes": a["votes"], "energy": a["energy"], "margin": a["margin"]},
        )
        stage_a_labels = dict(a["labels"])
        escalate = bool(a["unsure"]) and self.stage_b != "none"
        b_labels: dict = {}
        b_info: dict = {}
        if escalate:
            b_labels, b_info = self.stage_b_call(image, a, ctx)
            ctx.trace.step(
                f"Stage B · {self.stage_b}",
                f"escalated ({a['reason']}) -> " + " · ".join(str(b_labels.get(f) or "?") for f in active),
                info={"labels": b_labels, **b_info},
            )
        merged = (
            merge_fields(stage_a_labels, b_labels, a["unsure"], self.merge, fields=active)
            if escalate
            else stage_a_labels
        )
        pre_taxonomy = dict(merged)
        changed: list[str] = []
        if self.taxonomy:
            merged, changed = project_taxonomy(
                merged,
                self.brand_cats,
                self.valid_pairs,
                variants_by_brand=getattr(self, "variants_by_brand", None),
                sku_meta_by_bv=getattr(self, "sku_meta_by_bv", None),
            )
        answered_by = (
            "A"
            if not escalate
            else ("B" if self.merge == "all" or len(a["unsure"]) >= len(active) else "A+B")
        )
        ctx.trace.step(
            "Hybrid decision",
            f"answered by Stage {answered_by}"
            + (f" ({a['reason']})" if escalate else "")
            + (f"; taxonomy fixed {', '.join(changed)}" if changed else ""),
            info={
                "answered_by": answered_by,
                "escalated": escalate,
                "unsure": a["unsure"],
                "reason": a["reason"],
                "confidence": {f: a["votes"][f]["conf"] for f in active},
                "energy": a["energy"],
                "margin": a["margin"],
                "stage_a": {f: stage_a_labels.get(f) for f in active},
                "stage_b": b_labels or None,
                "pre_taxonomy": {f: pre_taxonomy.get(f) for f in active},
                "taxonomy_changed": changed,
                "final": {f: merged.get(f) for f in active},
                "params": self.params,
            },
        )
        out = {f: merged.get(f) for f in active}
        if not escalate and a["sku_id"] is not None:
            out["sku_id"] = a["sku_id"]
        elif escalate and b_info.get("sku_id") is not None:
            out["sku_id"] = b_info["sku_id"]
        return out if any(out.get(f) for f in active) else None

    # ---- run summary -------------------------------------------------------------------------

    def summary_extra(self, rows: list[dict], fields: tuple[str, ...]) -> dict:
        """Escalation rate, per-stage accuracy and the taxonomy delta (``summary.json["hybrid"]``)."""
        from utils.metrics import match_product

        n = esc = tax = 0
        acc = defaultdict(lambda: [0, 0])  # bucket -> [all-correct, n]
        field_hits = defaultdict(lambda: defaultdict(int))
        for r in rows:
            dec = next((s.get("info") for s in r.get("steps", []) if s.get("name") == "Hybrid decision"), None)
            labs = r.get("labels") or []
            if not dec or not labs or not labs[0].get("gt"):
                continue
            gt = labs[0]["gt"]
            n += 1
            esc += bool(dec["escalated"])
            tax += bool(dec["taxonomy_changed"])
            final_ok = all(labs[0]["correct"].get(f) for f in fields)
            a_ok = match_product(dec["stage_a"], gt, fields)["correct"]
            pre_ok = match_product(dec["pre_taxonomy"], gt, fields)["correct"]
            buckets = {
                "final": final_ok,
                "stage_a_only": all(a_ok.values()),
                "pre_taxonomy": all(pre_ok.values()),
                ("escalated" if dec["escalated"] else "gated"): final_ok,
            }
            if dec["escalated"]:
                buckets["escalated_stage_a_would_be"] = all(a_ok.values())
            for k, ok in buckets.items():
                acc[k][0] += ok
                acc[k][1] += 1
            for f in fields:
                field_hits["stage_a_only"][f] += bool(a_ok[f])
        if not n:
            return {}
        return {
            "hybrid": {
                "params": self.params,
                "images": n,
                "escalation_rate": round(esc / n, 4),
                "taxonomy_changed_rate": round(tax / n, 4),
                "accuracy_all_fields": {k: round(v[0] / max(1, v[1]), 4) for k, v in acc.items()},
                "count": {k: v[1] for k, v in acc.items()},
                "stage_a_only_field_accuracy": {f: round(field_hits["stage_a_only"][f] / n, 4) for f in fields},
                "gallery_excluded": getattr(self, "gallery_excluded", 0),
            }
        }


@register
class HybridClassifySft(HybridClassify):
    """``hybrid_classify`` escalating to the HUL Golden 5-attribute SFT model with kNN hints in the prompt."""

    name = "hybrid_classify_sft"
    stage_b = "sft_hints"
    merge = "field"
    tau_brand = 0.6
    tau_base_variant = 0.6
    tau_product = 0.6
    tau_packaging_type = 0.8
    tau_category = 0.6
    min_energy = 0.0
    architecture = (
        f"Hybrid 5-attribute cascade: {MODEL} kNN vote (0 LLM calls when confident) -> else HUL Golden SFT "
        "single-crop call with 5-attribute kNN candidate hints -> taxonomy & SKU projection"
    )


@register
class HybridClassifyMax(HybridClassify):
    """``hybrid_classify`` at its max-accuracy gate (escalates more crops to the 5-attribute contact sheet)."""

    name = "hybrid_classify_max"
    tau_brand = 0.99
    tau_base_variant = 0.99
    tau_product = 0.99
    tau_packaging_type = 0.99
    tau_category = 0.99
    architecture = (
        f"Hybrid 5-attribute cascade (max-accuracy gate): {MODEL} kNN vote (0 LLM calls only when the top-5 "
        "neighbours agree across all 5 attributes) -> else contact sheet of crop + top-5 refs -> taxonomy projection"
    )
