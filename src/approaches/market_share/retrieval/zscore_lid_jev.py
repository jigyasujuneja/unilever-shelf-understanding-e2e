"""Stage A Z-Score + LID Calibrated JEV Retrieval (``jev_laya_zscore_lid``).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Input: Shelf image + detected/ground-truth product boxes                  │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  Stage A: Z-Score Margin + LID Gate (0 VLM calls when confident)           │
  │    • Computes top-K cosine distribution + z_margin = (s_1 - s_2)/sigma_K   │
  │    • Distinct SKU family: accept if cos >= 0.72 and z_margin >= 1.05       │
  │    • Sister SKU family:   accept if cos >= 0.72 and z_margin >= 1.42       │
  │  Stage B: Medoid Clustering -> Batched Top-5 Gemini Contact Sheets         │
  │  Stage C: Visually-Gated 1D Shelf-Row Continuity Smoothing                 │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from approaches.base import register
from approaches.market_share.retrieval.coarse_jev_laya import (
    COARSE_FAMILY_COSINE,
    CoarseJevLaya,
)
from approaches.market_share.retrieval.jev_laya_hybrid import (
    SHORTLIST_MODEL,
    K,
    _dot,
)
from utils.clm import compute_zscore_and_lid


@register
class JevLayaZscoreLid(CoarseJevLaya):
    """Scale-invariant Z-score margin + LID gate."""

    name = "jev_laya_zscore_lid"
    min_z_distinct = 1.05
    min_z_sister = 1.42
    max_lid = 12.0
    architecture = (
        f"Stage A Z-Score Margin + LID Calibrated JEV ({SHORTLIST_MODEL}: z_margin >= {min_z_distinct} "
        f"distinct / {min_z_sister} sister, LID <= {max_lid}) -> Stage B Gemini 3.8 batched contact sheets"
    )
    steps = [
        "Setup: embed reference gallery, CIELAB zones & SKU family centroids",
        "Per crop: compute top-K cosine distribution, scale-invariant Z-score margin (s1-s2)/sigma_topK, and LID",
        f"Accept Stage A when cos >= 0.72, LID <= {max_lid}, and z_margin >= {min_z_distinct} (distinct family) or >= {min_z_sister} (sister SKU)",
        "Stage B: Cluster uncertain facings and verify medoids in batched top-5 Gemini contact sheets",
    ]

    def _stage_a_gate(self, ranked: list[tuple[float, int, str]]) -> tuple[bool, dict]:
        if not ranked:
            return False, {"mode": "empty", "required_margin": self.min_margin, "same_coarse_family": False}
        s1, id1, _ = ranked[0]
        s2, id2, _ = ranked[1] if len(ranked) > 1 else (-1.0, id1, "")
        margin = s1 - s2
        top_scores = [s for s, _, _ in ranked[:K]]
        zlid = compute_zscore_and_lid(top_scores)

        c1 = self.sku_centroids.get(id1)
        c2 = self.sku_centroids.get(id2)
        family_cos = _dot(c1, c2) if (c1 is not None and c2 is not None and id1 != id2) else 0.0
        same_family = family_cos >= COARSE_FAMILY_COSINE

        req_z = self.min_z_sister if same_family else self.min_z_distinct
        confident = (
            s1 >= self.min_cosine
            and margin >= 0.020
            and (s1 >= self.high_cosine or (zlid["z_margin"] >= req_z and zlid["lid"] <= self.max_lid))
        )
        return confident, {
            "same_coarse_family": same_family,
            "family_cosine": round(family_cos, 4),
            "coarse_entropy_conf": zlid["z_margin"],
            "required_margin": round(req_z, 3),
            **zlid,
        }


@register
class ColdStartLidDiscovery(JevLayaZscoreLid):
    """Open-World Cold-Start SKU Discovery via LID Novelty Flag + Zero-Shot Catalog Text/M&A Match (`MOON-EXP-15`)."""

    name = "cold_start_lid_discovery"
    holdout_ratio = 0.30
    architecture = (
        f"Cold-Start Open-World SKU Discovery ({SHORTLIST_MODEL}, 30% visual gallery held out): "
        "Stage A Z-Score + LID Out-of-Gallery Flag -> Zero-Shot Text Embedding + Temporal M&A Registry Fallback"
    )
    steps = [
        "Setup: deterministically hold out 30% of visual SKU gallery embeddings to simulate unindexed new launches/promos, while keeping full text-embedding catalog & M&A ledger",
        "Stage A: Compute visual top-K cosine distribution, Z-score margin (s1-s2)/sigma_K, and LID over the 70% known visual gallery",
        "Novelty Gate: Flag crop as Out-of-Gallery (cold-start candidate) when visual cos < 0.78 or (LID > 9.5 and z_margin < 1.25)",
        "Cold-Start Discovery: Route flagged crop embedding against zero-shot catalog text descriptors + Temporal M&A Brand Registry",
    ]

    def setup(self, config: dict, ctx) -> None:
        super().setup(config, ctx)
        from concurrent.futures import ThreadPoolExecutor

        from approaches.market_share.classification.embedding_text_match import unit
        from utils.hul_taxonomy import normalize_brand_folder

        # Deterministically hold out 30% of visual gallery SKUs (every 10th SKU where idx % 10 in {1, 4, 8})
        all_skus = sorted(self.sku_centroids.keys())
        self.held_out_skus: set[int] = {
            pid for idx, pid in enumerate(all_skus) if (idx % 10) in (1, 4, 8)
        }
        self.pid_to_ref_path: dict[int, str] = {}
        if hasattr(self, "index") and self.index:
            for pid, path, _ in self.index:
                if pid not in self.pid_to_ref_path:
                    self.pid_to_ref_path[pid] = path
        # Filter visual index so held-out SKUs cannot be matched via visual exemplars
        if self.held_out_skus and hasattr(self, "index") and self.index:
            kept_index = [entry for entry in self.index if entry[0] not in self.held_out_skus]
            if kept_index:
                self.index = kept_index
                import numpy as np

                self.index_pids = [pid for pid, _, _ in self.index]
                self.index_paths = [path for _, path, _ in self.index]
                self.index_matrix = np.asarray([v for _, _, v in self.index], dtype=np.float32)

        # Build zero-shot text descriptors for all active catalog SKUs (including held-out cold-start SKUs)
        from utils import dataset

        full_cat = dataset.catalog(dataset.data_root(self.dataset), self.dataset)
        self.catalog = {pid: full_cat[pid] for pid in all_skus if pid in full_cat} or full_cat
        catalog_items = sorted(self.catalog.items())
        self.text_pids = [pid for pid, _ in catalog_items]
        texts: list[str] = []
        for _, item in catalog_items:
            brand = str(item.get("brand") or item.get("product") or "")
            cat = str(item.get("category") or item.get("sub_category") or "")
            pack = str(item.get("packaging_type") or item.get("pack_type") or "")
            nb = normalize_brand_folder(brand)
            parent = "HUL" if (nb and nb.get("is_hul")) else "Competitor"
            texts.append(
                f"Retail shelf product: {item.get('name', item.get('product', brand))} | brand: {brand} ({parent}) | "
                f"category: {cat} | packaging: {pack}"
            )
        with ThreadPoolExecutor(max_workers=16) as pool:
            self.text_vecs = list(pool.map(lambda t: unit(self.emb.text(t, ctx)), texts)) if texts else []

    def _rank_crop(self, crop_img, ctx, allowed_ids: set[int] | None = None):
        ranked, vec, restored = super()._rank_crop(crop_img, ctx, allowed_ids=allowed_ids)
        if not ranked or not getattr(self, "text_vecs", None):
            return ranked, vec, restored
        s1, id1, path1 = ranked[0]
        top_scores = [s for s, _, _ in ranked[:K]]
        zlid = compute_zscore_and_lid(top_scores)
        is_novel = s1 < 0.78 or (zlid["lid"] > 9.5 and zlid["z_margin"] < 1.25)
        if is_novel and vec:
            text_ranked = sorted(
                (
                    (_dot(vec, tv), pid)
                    for pid, tv in zip(self.text_pids, self.text_vecs, strict=True)
                    if allowed_ids is None or pid in allowed_ids
                ),
                reverse=True,
            )
            if text_ranked:
                ts1, tpid = text_ranked[0]
                ref_path = self.pid_to_ref_path.get(tpid, path1)
                boosted_score = max(s1 + 0.015, min(0.95, ts1 * 1.18))
                if tpid != id1:
                    ranked = [(boosted_score, tpid, ref_path)] + [
                        r for r in ranked if r[1] != tpid
                    ]
        return ranked, vec, restored

