"""Two-Stage Hybrid Retrieval: Stage A Coarse JEV Family Triage + Stage B Gemini Dense Contact Sheets.

Combines System-1 Coarse JEV visual family triage (`jev_entropy_confidence` + reference centroid
family clustering) with System-2 Gemini dense multi-crop contact sheets:

1. **Stage A — Coarse JEV Visual Family Triage & Entropy Gate:**
   - Computes `jev + CIELAB` similarity over the 4-view gallery and checks whether the top-2
     shortlist candidates belong to **different coarse visual families** (`centroid_cos < 0.80`)
     or are **sister variants within the same coarse family** (`centroid_cos >= 0.80`).
   - When top-1 and top-2 belong to **different coarse families** (`centroid_cos < 0.80`) and pass
     the Stage A JEV entropy gate (`1 - H(p)/ln(N) >= 0.20`, `margin >= 0.026`), Stage A accepts
     the crop immediately with **0 Gemini VLM calls**.
   - When top-1 and top-2 are **sister variants in the same coarse family** (`centroid_cos >= 0.80`),
     Stage A enforces a stricter sister-SKU margin (`>= 0.036`) and routes ambiguous sister
     variants to Stage B (`Gemini`).
2. **Stage B — Gemini Dense Multi-Crop Contact-Sheet Disambiguation + Row Smoothing:**
   - Clusters duplicate uncertain facings (`cluster_uncertain_crops`), packs up to 4 uncertain
     medoids into a single `2x2` Contact Sheet for Gemini with `response_logprobs=True`,
     calibrates token logprobs via `resolve_jev_choice`, and applies `smooth_shelf_rows`.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from approaches.base import Box, Context, crop, register
from approaches.market_share.classification.embedding_text_match import unit
from approaches.market_share.retrieval.jev_laya_hybrid import (
    BATCH_SIZE,
    CLUSTER_COSINE,
    SHORTLIST_MODEL,
    SMOOTH_COSINE,
    JevLayaHybrid,
    K,
    _dot,
    cluster_uncertain_crops,
    smooth_shelf_rows,
)
from utils.openjev import jev_entropy_confidence

COARSE_FAMILY_COSINE = 0.80  # gallery SKU centroid cosine >= 0.80 => same coarse product family
COARSE_DISTINCT_MARGIN = 0.026  # margin when top-1 & top-2 belong to different coarse families
SISTER_FAMILY_MARGIN = 0.036  # stricter margin when top-1 & top-2 are sister SKUs in same family
MIN_COARSE_ENTROPY_CONF = 0.20  # minimum Stage A 1 - H(p)/ln(N) entropy confidence


@register
class CoarseJevLaya(JevLayaHybrid):
    name = "coarse_jev_laya"
    architecture = (
        f"Stage A Coarse JEV family triage ({SHORTLIST_MODEL} + 1-H(p)/ln(N) gate: distinct family "
        f">={COARSE_DISTINCT_MARGIN} vs sister SKU >={SISTER_FAMILY_MARGIN}) -> Stage B Gemini "
        f"batched top-{K} contact sheets & row smoothing"
    )
    steps = [
        "Setup (once per run, reported apart from cost/img): embed reference photos, CIELAB zones & SKU family centroids",
        f"Stage A: Photometric de-glare + {SHORTLIST_MODEL} + Coarse JEV family triage (distinct family vs sister SKU)",
        f"Accept Stage A when distinct-family margin >= {COARSE_DISTINCT_MARGIN} (or sister margin >= {SISTER_FAMILY_MARGIN})",
        f"Stage B: Cluster uncertain facings (cos >= {CLUSTER_COSINE}) and verify medoids in batched top-{K} Gemini contact sheets",
        f"Visually-gated 1D shelf-row smoothing ([P, None/?, P] -> P when neighbour cosine >= {SMOOTH_COSINE})",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        super().setup(config, ctx)
        by_sku: dict[int, list[list[float]]] = {}
        for pid, _, vec in self.index:
            by_sku.setdefault(pid, []).append(vec)
        self.sku_centroids: dict[int, list[float]] = {}
        for pid, vecs in by_sku.items():
            mean_v = [sum(col) / len(vecs) for col in zip(*vecs, strict=True)]
            self.sku_centroids[pid] = unit(mean_v)

    def _stage_a_gate(self, ranked: list[tuple[float, int, str]]) -> tuple[bool, dict]:
        """Stage A Coarse JEV gate separating distinct coarse families from sister-SKU pairs."""
        if not ranked:
            return False, {"mode": "empty"}
        s1, id1, _ = ranked[0]
        s2, id2, _ = ranked[1] if len(ranked) > 1 else (-1.0, id1, "")
        margin = s1 - s2

        top_k = ranked[:K]
        weights = [math.exp((s - s1) / 0.04) for s, _, _ in top_k]
        ent_conf = jev_entropy_confidence(weights, n_options=len(top_k))

        c1 = self.sku_centroids.get(id1)
        c2 = self.sku_centroids.get(id2)
        family_cos = _dot(c1, c2) if (c1 is not None and c2 is not None and id1 != id2) else 0.0
        same_family = family_cos >= COARSE_FAMILY_COSINE

        if s1 >= self.high_cosine:
            req_margin = self.high_cosine_margin
        elif same_family:
            req_margin = SISTER_FAMILY_MARGIN
        else:
            req_margin = COARSE_DISTINCT_MARGIN

        confident = s1 >= self.min_cosine and margin >= req_margin and ent_conf >= MIN_COARSE_ENTROPY_CONF
        return confident, {
            "same_coarse_family": same_family,
            "family_cosine": round(family_cos, 4),
            "coarse_entropy_conf": ent_conf,
            "required_margin": req_margin,
        }

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        ranked, _, restored = self._rank_crop(image, ctx, allowed_ids)
        if not ranked:
            return None
        (s1, pid, _) = ranked[0]
        s2 = max((s for s, _, _ in ranked[1:]), default=-1.0)
        confident, gate_meta = self._stage_a_gate(ranked)
        info = {
            "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in ranked[:K]],
            "thresholds": {"min_cosine": self.min_cosine, "min_margin": gate_meta["required_margin"]},
            **gate_meta,
        }
        if confident:
            ctx.trace.step(
                "Stage A Coarse JEV tier",
                f"#{pid} accepted (cos {s1:.3f}, margin {s1 - s2:.3f}, "
                f"family={'sister' if gate_meta['same_coarse_family'] else 'distinct'}, "
                f"jev_ent {gate_meta['coarse_entropy_conf']:.2f})",
                info={**info, "answer": pid},
            )
            return pid
        choice, conf, jev_meta = self.pick_with_confidence(restored, ranked[:K], ctx)
        ctx.trace.step(
            "Stage B Gemini tier",
            f"escalated (cos {s1:.3f}, margin {s1 - s2:.3f}, jev_conf {conf:.2f}) -> "
            + (f"#{choice}" if choice is not None else "none"),
            info={**info, **jev_meta, "answer": choice},
        )
        return choice

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        if not boxes:
            return []
        crops = [crop(image, b) for b in boxes]
        with ThreadPoolExecutor(max_workers=8) as pool:
            crop_data = list(pool.map(lambda c: self._rank_crop(c, ctx), crops))

        ids: list[int | None] = [None] * len(boxes)
        confidences: list[float] = [1.0] * len(boxes)
        vecs: list[list[float]] = []
        shortlists: list[set[int]] = []
        top_ids: list[int | None] = []
        restored_crops: list[Image.Image] = []
        ranked_lists: list[list[tuple[float, int, str]]] = []
        uncertain: list[int] = []

        for idx, (ranked, q, restored) in enumerate(crop_data):
            vecs.append(q)
            restored_crops.append(restored)
            ranked_lists.append(ranked)
            shortlists.append({pid for _, pid, _ in ranked[:K]})
            if not ranked:
                top_ids.append(None)
                confidences[idx] = 0.0
                continue
            (s1, pid, _) = ranked[0]
            s2 = max((s for s, _, _ in ranked[1:]), default=-1.0)
            margin = s1 - s2
            top_ids.append(pid)
            confident, gate_meta = self._stage_a_gate(ranked)
            info = {
                "box_index": idx,
                "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in ranked[:K]],
                "thresholds": {"min_cosine": self.min_cosine, "min_margin": gate_meta["required_margin"]},
                **gate_meta,
            }
            if confident:
                ids[idx] = pid
                confidences[idx] = 1.0
                ctx.trace.step(
                    "Stage A Coarse JEV tier",
                    f"box {idx + 1}: #{pid} accepted (cos {s1:.3f}, margin {margin:.3f}, "
                    f"family={'sister' if gate_meta['same_coarse_family'] else 'distinct'})",
                    info={**info, "answer": pid},
                )
            else:
                uncertain.append(idx)

        if uncertain:
            clusters = cluster_uncertain_crops(uncertain, boxes, vecs, top_ids)
            medoids = [
                max(cl, key=lambda i: ranked_lists[i][0][0] if ranked_lists[i] else -1.0)
                for cl in clusters
            ]
            batches = [medoids[i : i + BATCH_SIZE] for i in range(0, len(medoids), BATCH_SIZE)]

            def run_batch(batch: list[int]) -> list[tuple[int | None, float, dict]]:
                qs = [restored_crops[m] for m in batch]
                ts = [ranked_lists[m][:K] for m in batch]
                return self.pick_batch_with_confidence(qs, ts, ctx)

            with ThreadPoolExecutor(max_workers=4) as pool:
                batch_results = list(pool.map(run_batch, batches))

            medoid_outcomes = [item for b_res in batch_results for item in b_res]
            for cl, med, (ans, conf, jev_meta) in zip(clusters, medoids, medoid_outcomes, strict=True):
                for member in cl:
                    ids[member] = ans
                    confidences[member] = conf
                s1 = ranked_lists[med][0][0]
                s2 = max((s for s, _, _ in ranked_lists[med][1:]), default=-1.0)
                twins = f" (+{len(cl) - 1} cluster twins)" if len(cl) > 1 else ""
                ctx.trace.step(
                    "Stage B Gemini tier",
                    f"box {med + 1}{twins}: escalated (cos {s1:.3f}, margin {s1 - s2:.3f}, "
                    f"jev_conf {conf:.2f}) -> " + (f"#{ans}" if ans is not None else "none"),
                    info={
                        "medoid_box": med,
                        "cluster_boxes": cl,
                        "shortlist": [
                            {"sku_id": i, "cosine": round(s, 4)} for s, i, _ in ranked_lists[med][:K]
                        ],
                        "thresholds": {"min_cosine": self.min_cosine, "min_margin": self.min_margin},
                        **jev_meta,
                        "answer": ans,
                    },
                )

        ids, smoothed = smooth_shelf_rows(
            boxes,
            ids,
            vecs,
            shortlists,
            confidences=confidences,
            max_smooth_confidence=self.smooth_max_confidence,
            asset_boxes=getattr(ctx, "detected_asset_boxes", None),
        )
        if smoothed:
            ctx.trace.step(
                "Shelf-row smoothing",
                f"{len(smoothed)} occluded/low-confidence facing(s) recovered via flanking row context",
                info={"smoothed": smoothed},
            )
        return ids
