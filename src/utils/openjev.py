"""System One (``djev`` / ``openjev``) logprob & entropy calibration + ``/v1/systemone`` client.

Implements both:
1. **Option A — Vertex AI Gemini Single-Step Logprob & Entropy Calibration:**
   Extracts the token probability distribution over candidate option slots from Vertex AI
   ``response_logprobs=True`` (``LogprobsResult``), computes normalized Shannon entropy confidence
   ``1 - H(p) / ln(N)`` from `mmastrac/djev <https://github.com/mmastrac/djev>`_ and
   `razorback16/openjev <https://github.com/razorback16/openjev>`_, and fuses uncertain
   (high-entropy) VLM token distributions with the ``jev + CIELAB`` visual similarity prior.
2. **Option B — Direct ``/v1/systemone`` OpenJEV Client (``OpenJevClient``):**
   Speaks the ``/v1/systemone`` structured-decision protocol (single-pass masked-canvas read on
   ``google/diffusiongemma-26B-A4B-it`` / ``openjev-latest`` with ``0`` output tokens) when
   ``$OPENJEV_BASE_URL`` or ``config["openjev"]["base_url"]`` is configured.
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
import urllib.request
from typing import Any

from PIL import Image

from utils.llm import Usage, image_to_jpeg, load_config


def jev_entropy_confidence(
    probs: dict[Any, float] | list[float], n_options: int | None = None
) -> float:
    """Normalized Shannon entropy confidence ``1 - H(p) / ln(N)`` in ``[0.0, 1.0]``.

    Matches the ``djev`` / ``openjev`` System One calibration formula:
    - ``1.0`` when 100% of probability mass is on a single option (zero entropy).
    - ``0.0`` when probability mass is uniform across ``N`` options (maximum entropy ``ln(N)``).
    """
    vals = list(probs.values()) if isinstance(probs, dict) else list(probs)
    pos = [float(p) for p in vals if p > 0.0]
    if not pos:
        return 0.0
    total = sum(pos)
    norm = [p / total for p in pos]
    n = max(n_options or len(norm), len(norm), 2)
    entropy = -sum(p * math.log(p) for p in norm if p > 1e-12)
    max_entropy = math.log(n)
    if max_entropy <= 0.0:
        return 1.0
    return round(max(0.0, min(1.0, 1.0 - (entropy / max_entropy))), 4)


def serialize_logprobs(logprobs_result: Any) -> list[dict[str, Any]]:
    """Convert a ``google.genai.types.LogprobsResult`` into a compact JSON-serializable list."""
    if logprobs_result is None:
        return []
    chosen = getattr(logprobs_result, "chosen_candidates", None) or []
    tops = getattr(logprobs_result, "top_candidates", None) or []
    steps: list[dict[str, Any]] = []
    for idx, ch in enumerate(chosen):
        tok = getattr(ch, "token", None)
        lp = getattr(ch, "log_probability", None)
        top_list = []
        if idx < len(tops):
            for c in getattr(tops[idx], "candidates", None) or []:
                c_tok = getattr(c, "token", None)
                c_lp = getattr(c, "log_probability", None)
                if c_tok is not None and c_lp is not None:
                    top_list.append({"token": str(c_tok), "logprob": round(float(c_lp), 5)})
        if tok is not None and lp is not None:
            steps.append({
                "token": str(tok),
                "logprob": round(float(lp), 5),
                "top": top_list,
            })
    return steps


def _parse_int_token(tok: str) -> int | None:
    s = tok.strip()
    if not s:
        return None
    if s.startswith("-") and s[1:].isdigit():
        return int(s)
    if s.isdigit():
        return int(s)
    return None


def _slice_after_choice_key(logprobs_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anchor logprob slot scanning strictly after the JSON ``choice`` / ``choices`` / ``sku_id`` key."""
    anchor_idx = -1
    for idx, step in enumerate(logprobs_steps):
        tok = str(step.get("token", "")).strip().strip('"').strip("'").lower()
        if tok in {"choice", "choices", "sku_id"} or any(
            k in str(step.get("token", "")).lower() for k in ('"choice"', '"choices"', '"sku_id"')
        ):
            anchor_idx = idx
    return logprobs_steps[anchor_idx + 1 :] if anchor_idx >= 0 else logprobs_steps


def extract_slot_distributions(
    logprobs_steps: list[dict[str, Any]] | None,
    valid_options: set[int] | None = None,
    n_options: int = 6,
) -> list[dict[str, Any]]:
    """Extract per-slot categorical probability distributions from token logprobs.

    Scans the decoded token stream strictly after the ``choice`` / ``choices`` / ``sku_id`` JSON key
    (when present) for integer answer slots, aggregates the token probabilities ``exp(logprob)``
    over all candidate integer tokens at each slot, and computes the ``djev`` normalized entropy
    confidence ``1 - H(p) / ln(N)``.
    """
    if not logprobs_steps:
        return []
    active_steps = _slice_after_choice_key(logprobs_steps)
    slots: list[dict[str, Any]] = []
    for step in active_steps:
        chosen_int = _parse_int_token(str(step.get("token", "")))
        if chosen_int is None:
            continue
        if valid_options is not None and chosen_int not in valid_options:
            continue

        raw_lps: dict[int, float] = {chosen_int: float(step.get("logprob", 0.0))}
        for cand in step.get("top") or []:
            c_int = _parse_int_token(str(cand.get("token", "")))
            if c_int is None:
                continue
            if valid_options is not None and c_int not in valid_options:
                continue
            lp = float(cand.get("logprob", -20.0))
            if c_int not in raw_lps or lp > raw_lps[c_int]:
                raw_lps[c_int] = lp

        max_lp = max(raw_lps.values())
        unnorm = {k: math.exp(lp - max_lp) for k, lp in raw_lps.items()}
        # Account for any unobserved probability mass outside the top-K returned tokens
        seen_prob_mass = sum(math.exp(min(0.0, lp)) for lp in raw_lps.values())
        residual_mass = max(0.0, 1.0 - seen_prob_mass)
        denom = sum(unnorm.values()) or 1.0
        probs = {k: round((v / denom) * min(1.0, seen_prob_mass), 4) for k, v in unnorm.items()}

        # Include residual mass across remaining options when computing entropy
        dist_for_entropy = list(probs.values())
        unseen_count = max(1, n_options - len(probs))
        if residual_mass > 1e-4:
            dist_for_entropy.extend([residual_mass / unseen_count] * unseen_count)

        conf = jev_entropy_confidence(dist_for_entropy, n_options=max(n_options, len(probs)))
        sorted_p = sorted(probs.values(), reverse=True)
        top2_margin = round(sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0), 4)
        slots.append({
            "choice": chosen_int,
            "probabilities": probs,
            "confidence": conf,
            "top2_margin": top2_margin,
        })
    return slots


def resolve_jev_choice(
    raw_choice: int | None,
    slot: dict[str, Any] | None,
    top: list[tuple[float, int, str]],
    *,
    fallback_cosine: float = 0.82,
    min_confidence: float = 0.35,
    fusion_alpha: float = 0.65,
    cosine_temp: float = 0.04,
    allow_abstain: bool = False,
    max_energy_drop: float = 0.045,
) -> tuple[int | None, float, dict[str, Any]]:
    """Resolve a 1-indexed panel choice using ``djev`` entropy calibration + visual fusion + energy veto.

    - When ``allow_abstain=False`` (default for closed-set box attribution), an uncertain VLM choice
      of ``0`` ("none of these") falls back to the highest-energy CLM rank-1 candidate ``top[0][1]``
      whenever ``top[0][0] >= 0.45``, preserving L1–L4 hierarchy accuracy instead of abstaining to ``None``.
    - Enforces a **Log-Linear Energy Veto Guardrail** (``max_energy_drop=0.045``): if the VLM selects
      a candidate ``k > 1`` whose visual/CLM similarity ``s_k`` drops more than ``max_energy_drop``
      below the CLM rank-1 score ``s_1`` (unless VLM slot confidence ``>= 0.90``), vetoes the VLM
      override and keeps ``top[0][1]``.
    """
    k_len = len(top)
    if not top:
        return None, 0.0, {"mode": "empty_shortlist"}

    s1 = float(top[0][0])
    closed_set_floor = 0.45

    def _apply_energy_veto(chosen_k: int, conf_val: float, meta_d: dict[str, Any]) -> tuple[int | None, float, dict[str, Any]]:
        if chosen_k <= 1 or chosen_k > k_len:
            return top[0][1], conf_val, meta_d
        sk = float(top[chosen_k - 1][0])
        effective_max_drop = 0.008 if (slot is None and s1 >= 0.885) else max_energy_drop
        if (s1 - sk) > effective_max_drop and (slot is None or conf_val < 0.90):
            return (
                top[0][1],
                conf_val,
                {
                    **meta_d,
                    "mode": f"{meta_d.get('mode', 'jev')}_energy_veto_top1",
                    "vetoed_panel": chosen_k,
                    "energy_drop": round(s1 - sk, 4),
                },
            )
        return top[chosen_k - 1][1], conf_val, meta_d

    if slot is None:
        if isinstance(raw_choice, int) and 1 <= raw_choice <= k_len:
            return _apply_energy_veto(raw_choice, 1.0, {"mode": "greedy", "choice": raw_choice})
        if raw_choice == 0:
            if not allow_abstain and s1 >= closed_set_floor:
                return top[0][1], round(s1, 4), {"mode": "closed_set_top1_fallback", "choice": 0}
            return None, 1.0, {"mode": "greedy_abstain", "choice": 0}
        if s1 >= fallback_cosine or (not allow_abstain and s1 >= closed_set_floor):
            return top[0][1], round(s1, 4), {"mode": "cosine_fallback"}
        return None, 0.0, {"mode": "unparsed_abstain"}

    conf = float(slot.get("confidence", 1.0))
    probs: dict[int, float] = {int(k): float(v) for k, v in (slot.get("probabilities") or {}).items()}
    meta: dict[str, Any] = {
        "jev_confidence": conf,
        "jev_probabilities": {str(k): round(v, 4) for k, v in probs.items()},
        "jev_top2_margin": slot.get("top2_margin"),
    }

    # Confident System-1 read: accept choice (subject to energy veto) or handle abstention (0)
    if conf >= min_confidence:
        if isinstance(raw_choice, int) and 1 <= raw_choice <= k_len:
            return _apply_energy_veto(raw_choice, conf, {**meta, "mode": "jev_confident"})
        if raw_choice == 0:
            if not allow_abstain and s1 >= closed_set_floor:
                return top[0][1], conf, {**meta, "mode": "closed_set_top1_fallback", "choice": 0}
            return None, conf, {**meta, "mode": "jev_confident_abstain"}

    # High-entropy / uncertain slot: fuse VLM token distribution with visual similarity prior
    max_cos = max(s for s, _, _ in top)
    vis_unnorm = {
        idx + 1: math.exp((cos - max_cos) / max(1e-4, cosine_temp))
        for idx, (cos, _, _) in enumerate(top)
    }
    vis_denom = sum(vis_unnorm.values()) or 1.0
    vis_probs = {k: v / vis_denom for k, v in vis_unnorm.items()}

    p_abstain = probs.get(0, 0.0)
    fused_scores: dict[int, float] = {}
    for k in range(1, k_len + 1):
        p_jev = max(probs.get(k, 1e-3), 1e-3)
        p_vis = max(vis_probs.get(k, 1e-3), 1e-3)
        fused_scores[k] = (p_jev ** fusion_alpha) * (p_vis ** (1.0 - fusion_alpha))

    best_k = max(fused_scores, key=lambda k: fused_scores[k])
    best_jev_p = probs.get(best_k, 0.0)
    if (raw_choice == 0 or p_abstain > best_jev_p) and s1 < fallback_cosine:
        if not allow_abstain and s1 >= closed_set_floor:
            return top[0][1], conf, {**meta, "mode": "closed_set_top1_fallback"}
        return None, conf, {**meta, "mode": "jev_fused_abstain"}

    return _apply_energy_veto(
        best_k,
        conf,
        {**meta, "mode": "jev_visual_fusion", "fused_panel": best_k},
    )


class OpenJevClient:
    """Client for a ``/v1/systemone`` server (`mmastrac/djev`, `razorback16/openjev`).

    Activated when ``$OPENJEV_BASE_URL`` or ``config["openjev"]["base_url"]`` is set.
    Performs single-pass masked-canvas reads (``0`` output tokens) returning exact categorical
    distributions and ``1 - H(p) / ln(N)`` confidence for up to 12 questions simultaneously.
    """

    def __init__(self, config: dict | None = None, tier: str = "priority"):
        full_cfg = config or load_config()
        cfg = full_cfg.get("openjev") or {}
        self.base_url = (
            os.environ.get("OPENJEV_BASE_URL") or str(cfg.get("base_url") or "")
        ).rstrip("/")
        self.api_key = os.environ.get("OPENJEV_API_KEY") or str(cfg.get("api_key") or "")
        self.model = (
            os.environ.get("OPENJEV_MODEL")
            or str(cfg.get("model") or "openjev-latest")
        )
        self.timeout_s = float(cfg.get("timeout_s") or 30.0)
        self.tier = tier

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def ask_choices(
        self,
        image: Image.Image,
        n_queries: int,
        k_options: int,
        prompt: str,
        max_side: int | None = None,
        tier: str | None = None,
    ) -> tuple[list[int | None], list[dict[str, Any]], Usage, float]:
        """Query ``POST {base_url}/v1/systemone`` for ``n_queries`` choice slots (``0..k_options``)."""
        jpeg = image_to_jpeg(image, max_side=max_side)
        b64 = base64.b64encode(jpeg).decode("ascii")
        criteria = {"0": "None of the candidate reference panels match the query product"}
        for idx in range(1, k_options + 1):
            criteria[str(idx)] = f"Candidate reference panel {idx} matches the query product"

        questions = {
            f"q{q_idx}": {
                "text": f"Which candidate panel (1..{k_options}) matches row Q{q_idx} (or 0 if none)?",
                "type": "choice",
                "criteria": criteria,
            }
            for q_idx in range(1, n_queries + 1)
        }
        payload = json.dumps({
            "prompt": prompt,
            "images": [f"data:image/jpeg;base64,{b64}"],
            "questions": questions,
            "model": self.model,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone", data=payload, headers=headers, method="POST"
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        seconds = time.perf_counter() - t0

        results = body.get("results") or {}
        choices: list[int | None] = []
        slots: list[dict[str, Any]] = []
        for q_idx in range(1, n_queries + 1):
            r = results.get(f"q{q_idx}") or {}
            raw_c = r.get("choice")
            c_int = _parse_int_token(str(raw_c)) if raw_c is not None else None
            raw_probs = r.get("probabilities") or {}
            probs = {
                int(k): float(v)
                for k, v in raw_probs.items()
                if _parse_int_token(str(k)) is not None
            }
            conf = float(
                r.get(
                    "confidence",
                    jev_entropy_confidence(probs, n_options=k_options + 1),
                )
            )
            sorted_p = sorted(probs.values(), reverse=True)
            top2_margin = round(sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0), 4)
            choices.append(c_int)
            slots.append({
                "choice": c_int,
                "probabilities": probs,
                "confidence": round(conf, 4),
                "top2_margin": top2_margin,
            })

        active_tier = tier or self.tier or "priority"
        u = body.get("usage") or {}
        in_tok = int(u.get("prompt_tokens") or 0)
        out_tok = int(u.get("output_tokens") or 0)
        buckets: dict[str, int] = {}
        if in_tok:
            buckets[f"{active_tier}/text_input"] = in_tok
        if out_tok:
            buckets[f"{active_tier}/output"] = out_tok
        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=0,
            calls=1,
            traffic={active_tier: 1},
            buckets=buckets,
        )
        return choices, slots, usage, seconds


def coarse_jev_triage(
    ranked: list[tuple[float, int]],
    catalog: dict[int, dict[str, Any]],
    *,
    temp: float = 0.04,
    top_k: int = 12,
) -> dict[str, Any]:
    """Stage A Coarse JEV Multi-Attribute Triage over ``(category, brand, sku_id)``.

    Aggregates temperature-scaled Boltzmann probability mass ``exp((cos - max_cos) / temp)``
    across the top-``k`` multimodal embedding matches into hierarchical attribute slots
    (``category``, ``brand``, and intra-brand ``sku_id``) and computes the ``djev`` normalized
    Shannon entropy confidence ``1 - H(p) / ln(N)`` for each slot.
    """
    if not ranked:
        return {
            "top_id": None,
            "top_score": 0.0,
            "margin": 0.0,
            "intra_brand_margin": 0.0,
            "top_category": "",
            "category_confidence": 0.0,
            "top_brand": "",
            "brand_confidence": 0.0,
            "sku_confidence": 0.0,
            "brand_candidates": {},
        }

    top_slice = ranked[: max(2, min(top_k, len(ranked)))]
    max_cos = top_slice[0][0]
    cat_mass: dict[str, float] = {}
    brand_mass: dict[str, float] = {}
    sku_mass: dict[int, float] = {}

    for cos, pid in top_slice:
        w = math.exp((cos - max_cos) / max(1e-4, temp))
        p = catalog.get(pid) or {}
        cat = str(p.get("category") or "unknown")
        brand = str(p.get("brand") or cat)
        cat_mass[cat] = cat_mass.get(cat, 0.0) + w
        brand_mass[brand] = brand_mass.get(brand, 0.0) + w
        sku_mass[pid] = w

    all_cats = {str(p.get("category") or "unknown") for p in catalog.values()}
    all_brands = {str(p.get("brand") or p.get("category") or "unknown") for p in catalog.values()}

    top_id = ranked[0][1]
    top_score = ranked[0][0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = top_score - second_score

    top_p = catalog[top_id]
    top_cat = str(top_p.get("category") or "unknown")
    top_brand = str(top_p.get("brand") or top_cat)

    cat_conf = jev_entropy_confidence(cat_mass, n_options=max(2, len(all_cats)))
    brand_conf = jev_entropy_confidence(brand_mass, n_options=max(2, len(all_brands)))
    sku_conf = jev_entropy_confidence(sku_mass, n_options=len(top_slice))

    cos_by_id = {pid: cos for cos, pid in ranked}
    brand_ids = [
        pid
        for _, pid in ranked
        if str(catalog[pid].get("brand") or catalog[pid].get("category") or "") == top_brand
    ]
    if len(brand_ids) >= 2:
        intra_brand_margin = cos_by_id[brand_ids[0]] - cos_by_id[brand_ids[1]]
    else:
        intra_brand_margin = 1.0

    brand_candidates = {pid: catalog[pid] for pid in brand_ids}
    return {
        "top_id": top_id,
        "top_score": round(top_score, 4),
        "margin": round(margin, 4),
        "intra_brand_margin": round(intra_brand_margin, 4),
        "top_category": top_cat,
        "category_confidence": cat_conf,
        "top_brand": top_brand,
        "brand_confidence": brand_conf,
        "sku_confidence": sku_conf,
        "brand_candidates": brand_candidates,
    }

