"""jev + Laya hybrid retrieval & shelf disambiguator (Retrieval and End-to-end tabs).

Combines ``SHORTLIST_MODEL`` (``jev``) with the 4-layer ``Laya`` visual/layout system:

1. **Photometric De-glare Restoration:** Cleans specular glare highlights on shelf/counter crops
   before embedding and contact-sheet generation (``utils.photometry.restore_photometry``).
2. **jev + CIELAB Sister-Shade Ranking:** Embeds reference gallery photos and query crops with
   ``gemini-embedding-2`` on the ``global`` endpoint and breaks close cosine ties using 3-zone
   CIELAB chromatic + log-aspect similarity (``utils.photometry.lab_zones``).
3. **Spatial-Visual Medoid Clustering + Batched Multi-Crop Escalation:** On multi-product photos
   (``identify_boxes``), adjacent uncertain crops on the same shelf row with embedding cosine
   ``>= 0.92`` and identical top-1 candidate are grouped into a medoid cluster, and up to
   ``BATCH_SIZE = 4`` uncertain medoids are stacked onto a single multi-row contact sheet per
   Gemini call — cutting Gemini escalation calls and latency by 4x-8x.
4. **Visually-Gated 1D Shelf-Row Markov Smoothing:** Smooths an abstained interior
   facing ``[P, None, P] -> [P, P, P]`` on a shelf row only when the middle crop's visual embedding
   cosine with its flanking neighbours is ``>= 0.86`` and ``P`` is in its shortlist, preventing
   false overwrites across adjacent distinct products.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

from approaches.base import Box, Context, crop, register
from approaches.market_share.classification.embedding_text_match import unit
from approaches.market_share.retrieval.embedding_retrieval import EmbeddingRetrieval
from approaches.market_share.retrieval.tiered_hybrid import CELL, PROMPT, SCHEMA, K, sheet
from utils.embeddings import GEMINI_EMBEDDING
from utils.openjev import OpenJevClient, extract_slot_distributions, resolve_jev_choice
from utils.photometry import lab_similarity, lab_zones, restore_photometry

# Hyperparameters tuned on the RPC / Shelves validation splits (see class attributes on
# JevLayaHybrid so individual runs or ablation experiments can override them per instance/subclass).
MIN_COSINE = 0.72
MIN_MARGIN = 0.034
HIGH_COSINE = 0.905
HIGH_COSINE_MARGIN = 0.020
FALLBACK_COSINE = 0.82  # fallback threshold when Gemini returns malformed/missing JSON (choice is None)
MIN_JEV_CONFIDENCE = 0.35  # djev normalized entropy confidence (1 - H(p)/ln N) below which visual fusion triggers
SMOOTH_MAX_CONFIDENCE = 0.40  # max djev confidence eligible for [P, ?, P] -> [P, P, P] neighbour smoothing
LAB_WEIGHT = 0.035      # CIELAB tie-breaker weight (breaks sister-shade ties within ~0.03 cosine)
CLUSTER_COSINE = 0.92   # crop-to-crop embedding cosine to merge adjacent uncertain shelf facings
SMOOTH_COSINE = 0.86    # visual agreement gate required before [P, None, P] -> [P, P, P] row smoothing
BATCH_SIZE = 4          # uncertain medoids resolved per batched multi-row Gemini contact sheet
SHORTLIST_MODEL = GEMINI_EMBEDDING

BATCH_SCHEMA = {
    "type": "object",
    "properties": {"choices": {"type": "array", "items": {"type": "integer"}}},
    "required": ["choices"],
}
BATCH_PROMPT = (
    "The image shows {m} rows (Q1 to Q{m}). In each row, the red-labelled left panel Qi is a "
    "product cut out of a photo, followed by reference photos 1-{k} of candidate catalog products. "
    "For each row Qi (in order 1..{m}), pick the numbered panel (1-{k}) that shows the same "
    "product (same brand, product, flavour or variant, and pack size). The reference may be "
    "photographed from another angle. If none of them is the same product, answer 0. "
    'Return ONLY JSON like {{"choices": [2, 1]}} with {m} integers.'
)


def multi_sheet(queries: list[Image.Image], refs_per_query: list[list[Image.Image]]) -> Image.Image:
    """Stack ``m`` query rows (Q1..Qm), each with its ``1..K`` reference panels."""
    cols = max((len(r) for r in refs_per_query), default=K) + 1
    row_h = CELL + 30
    out = Image.new("RGB", (CELL * cols, row_h * len(queries)), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    for r_idx, (q, refs) in enumerate(zip(queries, refs_per_query, strict=True)):
        y0 = r_idx * row_h
        for c_idx, im in enumerate([q, *refs]):
            im = im.copy()
            im.thumbnail((CELL - 10, CELL - 10))
            x0 = c_idx * CELL
            out.paste(im, (x0 + (CELL - im.width) // 2, y0 + 30 + (CELL - im.height) // 2))
            draw.rectangle((x0, y0, x0 + CELL - 1, y0 + row_h - 1), outline="#999")
            label = f"Q{r_idx + 1}" if c_idx == 0 else str(c_idx)
            draw.text((x0 + 6, y0 + 2), label, fill="red" if c_idx == 0 else "black", font=font)
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def cluster_uncertain_crops(
    indices: list[int],
    boxes: list[Box],
    vecs: list[list[float]],
    top_ids: list[int | None],
) -> list[list[int]]:
    """Group spatially adjacent uncertain boxes on the same shelf row with cosine >= CLUSTER_COSINE."""
    clusters: list[list[int]] = []
    assigned: set[int] = set()
    # Sort by vertical row then left-to-right
    ordered = sorted(indices, key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
    for i in ordered:
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        cy_i = (boxes[i][1] + boxes[i][3]) / 2
        h_i = max(1.0, boxes[i][3] - boxes[i][1])
        for j in ordered:
            if j in assigned:
                continue
            cy_j = (boxes[j][1] + boxes[j][3]) / 2
            h_j = max(1.0, boxes[j][3] - boxes[j][1])
            same_row = abs(cy_i - cy_j) <= 0.55 * max(h_i, h_j)
            if (
                same_row
                and top_ids[i] is not None
                and top_ids[i] == top_ids[j]
                and _dot(vecs[i], vecs[j]) >= CLUSTER_COSINE
            ):
                group.append(j)
                assigned.add(j)
        clusters.append(group)
    return clusters


def smooth_shelf_rows(
    boxes: list[Box],
    ids: list[int | None],
    vecs: list[list[float]],
    shortlists: list[set[int]],
    confidences: list[float] | None = None,
    max_smooth_confidence: float = SMOOTH_MAX_CONFIDENCE,
    asset_boxes: list[Box] | None = None,
    smooth_cosine: float = SMOOTH_COSINE,
) -> tuple[list[int | None], list[dict]]:
    """Visually-gated 1D Markov shelf-row smoothing: [P, None/?, P] -> [P, P, P].

    Smooths an interior slot only when it was either abstained (``None``) or had low ``djev``
    entropy confidence (``< max_smooth_confidence``), so confident Gemini-verified or high-cosine
    predictions are never overwritten by neighbour smoothing. When ``asset_boxes`` (Class 0 display
    headers/bays) are present, blocks smoothing across distinct display-bay boundaries.
    """
    n = len(boxes)
    if n < 3:
        return ids, []
    out = list(ids)
    # Group boxes into horizontal shelf rows
    order = sorted(range(n), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows: list[list[int]] = []
    for i in order:
        cy = (boxes[i][1] + boxes[i][3]) / 2
        h = max(1.0, boxes[i][3] - boxes[i][1])
        placed = False
        for row in rows:
            ref_cy = sum((boxes[k][1] + boxes[k][3]) / 2 for k in row) / len(row)
            ref_h = sum(max(1.0, boxes[k][3] - boxes[k][1]) for k in row) / len(row)
            if abs(cy - ref_cy) <= 0.45 * max(h, ref_h):
                row.append(i)
                placed = True
                break
        if not placed:
            rows.append([i])

    def _bay_index(b: Box) -> int | None:
        if not asset_boxes:
            return None
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        for a_idx, ab in enumerate(asset_boxes):
            if ab[0] <= cx <= ab[2] and ab[1] <= cy <= ab[3]:
                return a_idx
        return -1

    smoothed: list[dict] = []
    for row in rows:
        row.sort(key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)
        for pos in range(1, len(row) - 1):
            left, mid, right = row[pos - 1], row[pos], row[pos + 1]
            pid = out[left]
            mid_conf = confidences[mid] if confidences is not None and mid < len(confidences) else 1.0
            can_smooth_mid = out[mid] is None or (mid_conf < max_smooth_confidence and out[mid] != pid)
            if pid is None or out[right] != pid or not can_smooth_mid:
                continue
            if asset_boxes:
                bay_l, bay_m, bay_r = _bay_index(boxes[left]), _bay_index(boxes[mid]), _bay_index(boxes[right])
                if len({bay_l, bay_m, bay_r}) > 1:
                    continue
            avg_w = ((boxes[left][2] - boxes[left][0]) + (boxes[right][2] - boxes[right][0])) / 2
            gap_l = max(0.0, boxes[mid][0] - boxes[left][2])
            gap_r = max(0.0, boxes[right][0] - boxes[mid][2])
            if gap_l > 0.85 * avg_w or gap_r > 0.85 * avg_w:
                continue
            cos_l = _dot(vecs[mid], vecs[left])
            cos_r = _dot(vecs[mid], vecs[right])
            if cos_l >= smooth_cosine and cos_r >= smooth_cosine and pid in shortlists[mid]:
                prev = out[mid]
                out[mid] = pid
                smoothed.append({
                    "box_index": mid,
                    "from": prev,
                    "to": pid,
                    "mid_confidence": round(mid_conf, 4),
                    "cos_left": round(cos_l, 4),
                    "cos_right": round(cos_r, 4),
                })
    return out, smoothed


@register
class JevLayaHybrid(EmbeddingRetrieval):
    name = "jev_laya_hybrid"
    models: list[str] | None = None
    shortlist_model = SHORTLIST_MODEL
    also_calls = [SHORTLIST_MODEL]
    min_cosine = MIN_COSINE
    min_margin = MIN_MARGIN
    high_cosine = HIGH_COSINE
    high_cosine_margin = HIGH_COSINE_MARGIN
    fallback_cosine = FALLBACK_COSINE
    min_jev_confidence = MIN_JEV_CONFIDENCE
    smooth_max_confidence = SMOOTH_MAX_CONFIDENCE
    architecture = (
        f"Photometric de-glare -> {SHORTLIST_MODEL} + CIELAB Laya ranking "
        f"(accept if cos >= {MIN_COSINE}, margin >= {MIN_MARGIN}) -> medoid clustering & "
        f"batched top-{K} djev logprob-calibrated verification -> visually-gated shelf-row smoothing"
    )
    steps = [
        "Setup (once per run, reported apart from cost/img): embed reference photos + CIELAB zones",
        f"Per crop: photometric de-glare + {SHORTLIST_MODEL} + CIELAB sister-shade ranking",
        f"Accept embedding answer when cosine >= {MIN_COSINE} and margin >= {MIN_MARGIN}",
        f"Cluster visually identical uncertain facings (cos >= {CLUSTER_COSINE}) and verify medoids "
        f"via batched top-{K} contact sheets with djev 1 - H(p)/ln(N) entropy calibration",
        f"Visually-gated 1D shelf-row smoothing ([P, None/?, P] -> P when neighbour cosine >= {SMOOTH_COSINE})",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        super().setup(config, ctx)
        self.openjev = OpenJevClient(config)
        self.lab_index: dict[str, list[float]] = {}
        for _, path, _ in self.index:
            if path not in self.lab_index and path in self.jpeg:
                with Image.open(io.BytesIO(self.jpeg[path])) as im:
                    self.lab_index[path] = lab_zones(im.convert("RGB"))

    def _rank_crop(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> tuple[list[tuple[float, int, str]], list[float], Image.Image]:
        restored, _ = restore_photometry(image)
        q = unit(self.emb.image(restored, ctx))
        pool = (
            [(pid, path, v) for pid, path, v in self.index if pid in allowed_ids]
            if allowed_ids
            else []
        ) or self.index
        best: dict[int, tuple[float, int, str]] = {}
        for pid, path, v in pool:
            s = _dot(q, v)
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, pid, path)
        coarse = sorted(best.values(), reverse=True)
        if not coarse:
            return [], q, restored

        # Laya CIELAB sister-shade tie-breaking across the top 2*K candidates
        q_lab = lab_zones(restored)
        head = []
        for cos, pid, path in coarse[: 2 * K]:
            ref_lab = self.lab_index.get(path)
            s_lab = lab_similarity(q_lab, ref_lab) if ref_lab else 0.5
            joint = min(1.0, cos + LAB_WEIGHT * (s_lab - 0.5))
            head.append((joint, cos, pid, path))
        head.sort(reverse=True)
        reranked = [(joint, pid, path) for joint, _, pid, path in head] + coarse[2 * K :]
        return reranked, q, restored

    def pick_with_confidence(
        self, image: Image.Image, top: list[tuple[float, int, str]], ctx: Context
    ) -> tuple[int | None, float, dict]:
        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        sheet_im = sheet(image, refs)
        openjev = getattr(self, "openjev", None)
        if openjev is not None and openjev.enabled:
            choices, slots, usage, _ = openjev.ask_choices(
                sheet_im, 1, len(top), PROMPT, tier=getattr(ctx.llm, "tier", None)
            )
            with ctx.trace._lock:
                ctx.trace.usage = ctx.trace.usage + usage
            raw_choice = choices[0] if choices else None
            slot = slots[0] if slots else None
        else:
            res = ctx.ask(sheet_im, PROMPT, schema=SCHEMA, logprobs=len(top) + 1)
            data = res.data
            raw_choice = data.get("choice") if isinstance(data, dict) else None
            valid_opts = set(range(0, len(top) + 1))
            slots = extract_slot_distributions(
                res.meta.get("logprobs"), valid_options=valid_opts, n_options=len(top) + 1
            )
            slot = slots[0] if slots else None
        return resolve_jev_choice(
            raw_choice,
            slot,
            top,
            fallback_cosine=self.fallback_cosine,
            min_confidence=self.min_jev_confidence,
        )

    def pick(self, image: Image.Image, top: list[tuple[float, int, str]], ctx: Context) -> int | None:
        ans, _, _ = self.pick_with_confidence(image, top, ctx)
        return ans

    def pick_batch_with_confidence(
        self,
        queries: list[Image.Image],
        tops: list[list[tuple[float, int, str]]],
        ctx: Context,
    ) -> list[tuple[int | None, float, dict]]:
        if len(queries) == 1:
            return [self.pick_with_confidence(queries[0], tops[0], ctx)]
        refs_per_q = [
            [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
            for top in tops
        ]
        prompt = BATCH_PROMPT.format(m=len(queries), k=K)
        sheet_im = multi_sheet(queries, refs_per_q)
        openjev = getattr(self, "openjev", None)
        if openjev is not None and openjev.enabled:
            choices, slots, usage, _ = openjev.ask_choices(
                sheet_im, len(queries), K, prompt, tier=getattr(ctx.llm, "tier", None)
            )
            with ctx.trace._lock:
                ctx.trace.usage = ctx.trace.usage + usage
        else:
            res = ctx.ask(sheet_im, prompt, schema=BATCH_SCHEMA, logprobs=K + 1)
            data = res.data
            raw = data.get("choices") if isinstance(data, dict) else None
            choices = raw if isinstance(raw, list) else []
            valid_opts = set(range(0, K + 1))
            slots = extract_slot_distributions(
                res.meta.get("logprobs"), valid_options=valid_opts, n_options=K + 1
            )
        out: list[tuple[int | None, float, dict]] = []
        for idx, top in enumerate(tops):
            c = choices[idx] if idx < len(choices) else None
            slot = slots[idx] if idx < len(slots) else None
            out.append(
                resolve_jev_choice(
                    c,
                    slot,
                    top,
                    fallback_cosine=self.fallback_cosine,
                    min_confidence=self.min_jev_confidence,
                )
            )
        return out

    def pick_batch(
        self,
        queries: list[Image.Image],
        tops: list[list[tuple[float, int, str]]],
        ctx: Context,
    ) -> list[int | None]:
        return [ans for ans, _, _ in self.pick_batch_with_confidence(queries, tops, ctx)]

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        ranked, _, restored = self._rank_crop(image, ctx, allowed_ids)
        if not ranked:
            return None
        (s1, pid, _) = ranked[0]
        s2 = max((s for s, _, _ in ranked[1:]), default=-1.0)
        info = {
            "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in ranked[:K]],
            "thresholds": {"min_cosine": self.min_cosine, "min_margin": self.min_margin},
        }
        req_margin = self.high_cosine_margin if s1 >= self.high_cosine else self.min_margin
        if s1 >= self.min_cosine and s1 - s2 >= req_margin:
            ctx.trace.step(
                "Embedding tier",
                f"#{pid} accepted (cos {s1:.3f}, margin {s1 - s2:.3f})",
                info={**info, "answer": pid},
            )
            return pid
        choice, conf, jev_meta = self.pick_with_confidence(restored, ranked[:K], ctx)
        ctx.trace.step(
            "Gemini tier",
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
        confidences: list[float] = [0.0] * len(boxes)
        vecs: list[list[float]] = []
        shortlists: list[set[int]] = []
        top_ids: list[int | None] = []
        restored_crops: list[Image.Image] = []
        ranked_lists: list[list[tuple[float, int, str]]] = []
        uncertain: list[int] = []

        from utils.openjev import calibrated_shortlist_confidence

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
            req_margin = self.high_cosine_margin if s1 >= self.high_cosine else self.min_margin
            cal = calibrated_shortlist_confidence(1, ranked[:K], catalog=getattr(self, "catalog", None))
            info = {
                "box_index": idx,
                "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in ranked[:K]],
                "thresholds": {"min_cosine": self.min_cosine, "min_margin": req_margin},
                "sku_confidence": cal["sku_confidence"],
                "brand_confidence": cal["brand_confidence"],
            }
            if s1 >= self.min_cosine and margin >= req_margin:
                ids[idx] = pid
                confidences[idx] = max(0.75, cal["sku_confidence"])
                ctx.trace.step(
                    "Embedding tier",
                    f"box {idx + 1}: #{pid} accepted (cos {s1:.3f}, margin {margin:.3f})",
                    info={**info, "answer": pid},
                )
            else:
                uncertain.append(idx)

        if uncertain:
            clusters = cluster_uncertain_crops(uncertain, boxes, vecs, top_ids)
            # Choose the highest-confidence crop in each cluster as the medoid
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
                    "Gemini tier",
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
                "Laya row smoothing",
                f"smoothed {len(smoothed)} shelf-row box(es) via visual neighbour continuity",
                info={"smoothed": smoothed},
            )
        return ids
