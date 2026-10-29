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


def calibrated_shortlist_confidence(
    choice: int | None,
    top: list[tuple[float, int, str]],
    slot: dict[str, Any] | None = None,
    *,
    inv_temp: float = 28.0,
    open_set_threshold: float = 0.68,
    catalog: dict[int, dict[str, Any]] | None = None,
    vetoed: bool = False,
    pack_size_verified: bool = False,
) -> dict[str, float]:
    """Compute dustbin-calibrated SKU and Brand confidence in ``[0.01, 0.99]`` (never hardcoded ``1.0``).

    Combines temperature-scaled InfoNCE shortlist probabilities (with an open-set dustbin at
    ``open_set_threshold``), an absolute cosine clarity factor, and (when available) VLM token
    slot entropy confidence. Explicitly penalizes forced ``choice == 0`` fallbacks and energy-veto
    disagreements so downstream confidence thresholding can reliably filter low-evidence predictions.
    """
    if not top:
        return {
            "sku_confidence": 0.0,
            "brand_confidence": 0.0,
            "clm_top1_prob": 0.0,
            "c1_confidence": 0.0,
        }
    s1 = float(top[0][0])
    s2 = float(top[1][0]) if len(top) > 1 else max(0.0, s1 - 0.08)
    margin = max(0.0, s1 - s2)
    exps = [math.exp(max(-30.0, min(30.0, inv_temp * (float(s) - s1)))) for s, _, _ in top]
    dustbin = math.exp(max(-30.0, min(30.0, inv_temp * (open_set_threshold - s1))))
    z = (sum(exps) + dustbin) or 1.0
    probs = [e / z for e in exps]
    p_clm1 = probs[0]
    c1_conf = min(0.985, jev_entropy_confidence(probs + [dustbin / z], n_options=len(top) + 1))

    choice_int = choice if (isinstance(choice, int) and 1 <= choice <= len(top)) else None
    is_valid_panel = choice_int is not None
    chosen_idx = (choice_int - 1) if choice_int is not None else 0
    s_chosen = float(top[chosen_idx][0])
    p_chosen = probs[chosen_idx]

    # Absolute visual clarity factor in [0.08, 0.985] based on cosine similarity and top-2 margin
    cos_clarity = max(0.08, min(0.985, (s_chosen - 0.42) / 0.54))
    margin_bonus = min(0.08, margin * 1.2)

    if choice_int is not None:
        base_sku = 0.52 * p_chosen + 0.40 * cos_clarity + margin_bonus
        if choice_int > 1:
            drop = max(0.0, s1 - s_chosen)
            # When VLM selects panel k > 1, weight by visual proximity to rank-1
            base_sku = (0.38 * p_chosen + 0.52 * cos_clarity + 0.05) * max(0.45, 1.0 - 2.5 * drop)
        if slot is not None and "confidence" in slot:
            slot_c = max(0.05, min(0.985, float(slot["confidence"])))
            base_sku = 0.55 * slot_c + 0.45 * base_sku
        if vetoed:
            base_sku = min(0.46, base_sku * 0.62)
        elif pack_size_verified:
            base_sku = min(0.985, base_sku + 0.06)
        sku_conf = round(max(0.05, min(0.985, base_sku)), 4)
    else:
        # Forced fallback on choice == 0 ("none match / unreadable") or unparsed response:
        # Cap strictly <= 0.34 so any standard confidence threshold filters it out.
        sku_conf = round(max(0.02, min(0.34, 0.38 * p_clm1 * cos_clarity)), 4)

    # Brand-level probability mass across the shortlist (discounted by open-set dustbin & clarity)
    chosen_pid = int(top[chosen_idx][1])
    if catalog and chosen_pid in catalog:
        chosen_brand = str((catalog.get(chosen_pid) or {}).get("brand") or "").strip().lower()
        if chosen_brand:
            brand_mass = sum(
                p
                for p, (_, pid, _) in zip(probs, top, strict=True)
                if str((catalog.get(int(pid)) or {}).get("brand") or "").strip().lower() == chosen_brand
            )
        else:
            brand_mass = p_chosen
    else:
        brand_mass = max(p_chosen, min(0.96, p_clm1 * 1.15))

    if is_valid_panel:
        brand_conf = round(
            max(sku_conf, min(0.985, (0.60 * brand_mass + 0.40 * cos_clarity) * (0.72 if vetoed else 1.0))),
            4,
        )
    else:
        brand_conf = round(max(sku_conf, min(0.38, 0.45 * brand_mass * cos_clarity)), 4)

    return {
        "sku_confidence": sku_conf,
        "brand_confidence": brand_conf,
        "clm_top1_prob": round(p_clm1, 4),
        "c1_confidence": round(c1_conf, 4),
    }


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
    catalog: dict[int, dict[str, Any]] | None = None,
) -> tuple[int | None, float, dict[str, Any]]:
    """Resolve a 1-indexed panel choice using ``djev`` entropy calibration + visual fusion + energy veto.

    - When ``allow_abstain=False`` (default for closed-set box attribution), an uncertain VLM choice
      of ``0`` ("none of these") falls back to the highest-energy CLM rank-1 candidate ``top[0][1]``
      whenever ``top[0][0] >= 0.45``, preserving L1–L4 hierarchy accuracy instead of abstaining to ``None``.
    - Enforces a **Log-Linear Energy Veto Guardrail** (``max_energy_drop=0.045``): if the VLM selects
      a candidate ``k > 1`` whose visual/CLM similarity ``s_k`` drops more than ``max_energy_drop``
      below the CLM rank-1 score ``s_1`` (unless VLM slot confidence ``>= 0.90``), vetoes the VLM
      override and keeps ``top[0][1]``.
    - Always computes dustbin-calibrated ``calibrated_confidence`` and ``brand_confidence`` in
      ``[0.01, 0.985]`` (never hardcoded ``1.0``), and honors ``$SHELF_BENCH_MIN_CONFIDENCE`` for
      confidence-threshold abstention.
    """
    k_len = len(top)
    if not top:
        return None, 0.0, {"mode": "empty_shortlist", "calibrated_confidence": 0.0, "brand_confidence": 0.0}

    s1 = float(top[0][0])
    closed_set_floor = 0.45
    env_min_conf = float(os.environ.get("SHELF_BENCH_MIN_CONFIDENCE") or 0.0)

    def _apply_energy_veto(
        chosen_k: int, conf_val: float, meta_d: dict[str, Any]
    ) -> tuple[int | None, float, dict[str, Any]]:
        if chosen_k <= 1 or chosen_k > k_len:
            cal = calibrated_shortlist_confidence(1, top, slot, catalog=catalog, vetoed=False)
            eff_conf = cal["sku_confidence"] if slot is None else conf_val
            out_meta = {
                **meta_d,
                "calibrated_confidence": cal["sku_confidence"],
                "brand_confidence": cal["brand_confidence"],
                "abstain_recommended": False,
            }
            if slot is None:
                out_meta["jev_confidence"] = eff_conf
                out_meta["c1_confidence"] = cal["c1_confidence"]
            if env_min_conf > 0.0 and cal["sku_confidence"] < env_min_conf:
                return None, eff_conf, {**out_meta, "mode": "confidence_threshold_abstain"}
            return top[0][1], eff_conf, out_meta
        sk = float(top[chosen_k - 1][0])
        effective_max_drop = 0.008 if (slot is None and s1 >= 0.885) else max_energy_drop
        if (s1 - sk) > effective_max_drop and (slot is None or conf_val < 0.90):
            cal_v = calibrated_shortlist_confidence(chosen_k, top, slot, catalog=catalog, vetoed=True)
            eff_conf = cal_v["sku_confidence"] if slot is None else conf_val
            out_meta = {
                **meta_d,
                "mode": f"{meta_d.get('mode', 'jev')}_energy_veto_top1",
                "vetoed_panel": chosen_k,
                "energy_drop": round(s1 - sk, 4),
                "calibrated_confidence": cal_v["sku_confidence"],
                "brand_confidence": cal_v["brand_confidence"],
                "abstain_recommended": cal_v["sku_confidence"] < 0.35,
            }
            if slot is None:
                out_meta["jev_confidence"] = eff_conf
                out_meta["c1_confidence"] = cal_v["c1_confidence"]
            if env_min_conf > 0.0 and cal_v["sku_confidence"] < env_min_conf:
                return None, eff_conf, {**out_meta, "mode": "confidence_threshold_abstain"}
            return top[0][1], eff_conf, out_meta

        cal_ok = calibrated_shortlist_confidence(chosen_k, top, slot, catalog=catalog, vetoed=False)
        eff_conf = cal_ok["sku_confidence"] if slot is None else conf_val
        out_meta = {
            **meta_d,
            "calibrated_confidence": cal_ok["sku_confidence"],
            "brand_confidence": cal_ok["brand_confidence"],
            "abstain_recommended": False,
        }
        if slot is None:
            out_meta["jev_confidence"] = eff_conf
            out_meta["c1_confidence"] = cal_ok["c1_confidence"]
        if env_min_conf > 0.0 and cal_ok["sku_confidence"] < env_min_conf:
            return None, eff_conf, {**out_meta, "mode": "confidence_threshold_abstain"}
        return top[chosen_k - 1][1], eff_conf, out_meta

    if slot is None:
        if isinstance(raw_choice, int) and 1 <= raw_choice <= k_len:
            cal_g = calibrated_shortlist_confidence(raw_choice, top, None, catalog=catalog)
            return _apply_energy_veto(
                raw_choice,
                cal_g["sku_confidence"],
                {"mode": "greedy", "choice": raw_choice},
            )
        cal_fb = calibrated_shortlist_confidence(0, top, None, catalog=catalog)
        fb_meta = {
            "calibrated_confidence": cal_fb["sku_confidence"],
            "brand_confidence": cal_fb["brand_confidence"],
            "jev_confidence": cal_fb["sku_confidence"],
            "c1_confidence": cal_fb["c1_confidence"],
            "abstain_recommended": True,
        }
        if raw_choice == 0:
            if not allow_abstain and s1 >= closed_set_floor and (env_min_conf <= 0.0 or cal_fb["sku_confidence"] >= env_min_conf):
                return top[0][1], cal_fb["sku_confidence"], {**fb_meta, "mode": "closed_set_top1_fallback", "choice": 0}
            return None, cal_fb["sku_confidence"], {**fb_meta, "mode": "greedy_abstain", "choice": 0}
        if (s1 >= fallback_cosine or (not allow_abstain and s1 >= closed_set_floor)) and (
            env_min_conf <= 0.0 or cal_fb["sku_confidence"] >= env_min_conf
        ):
            return top[0][1], cal_fb["sku_confidence"], {**fb_meta, "mode": "cosine_fallback"}
        return None, 0.0, {**fb_meta, "mode": "unparsed_abstain"}

    conf = float(slot.get("confidence", 0.5))
    probs: dict[int, float] = {int(k): float(v) for k, v in (slot.get("probabilities") or slot.get("probs") or {}).items()}
    max_cos = max(s for s, _, _ in top)
    vis_unnorm = {
        idx + 1: math.exp((cos - max_cos) / max(1e-4, cosine_temp))
        for idx, (cos, _, _) in enumerate(top)
    }
    vis_denom = sum(vis_unnorm.values()) or 1.0
    vis_probs = {k: v / vis_denom for k, v in vis_unnorm.items()}
    c1_conf = jev_entropy_confidence(list(vis_probs.values()), n_options=max(2, k_len))
    w2_poe = round(conf / max(1e-6, c1_conf + conf), 4)
    cal_slot = calibrated_shortlist_confidence(raw_choice, top, slot, catalog=catalog)

    meta: dict[str, Any] = {
        "jev_confidence": conf,
        "c1_confidence": c1_conf,
        "c2_confidence": conf,
        "calibrated_confidence": cal_slot["sku_confidence"],
        "brand_confidence": cal_slot["brand_confidence"],
        "poe_weight_w2": w2_poe,
        "jev_probabilities": {str(k): round(v, 4) for k, v in probs.items()},
        "jev_top2_margin": slot.get("top2_margin"),
    }

    # Confident System-2 read when C_2 >= min_confidence and System 2 certainty dominates System 1:
    if conf >= min_confidence and w2_poe >= 0.45:
        if isinstance(raw_choice, int) and 1 <= raw_choice <= k_len:
            return _apply_energy_veto(raw_choice, conf, {**meta, "mode": "jev_confident"})
        if raw_choice == 0:
            if not allow_abstain and s1 >= closed_set_floor and (env_min_conf <= 0.0 or cal_slot["sku_confidence"] >= env_min_conf):
                return (
                    top[0][1],
                    conf,
                    {**meta, "mode": "closed_set_top1_fallback", "choice": 0, "abstain_recommended": True},
                )
            return None, conf, {**meta, "mode": "jev_confident_abstain", "abstain_recommended": True}

    # Continuous Entropy-Weighted Bayesian Product-of-Experts (PoE) fusion:
    # log P_joint(k) = w2 * log P_2(k) + (1 - w2) * log P_1(k)
    eff_alpha = w2_poe if fusion_alpha == 0.65 else fusion_alpha
    p_abstain = probs.get(0, 0.0)
    fused_scores: dict[int, float] = {}
    for k in range(1, k_len + 1):
        p_jev = max(probs.get(k, 1e-3), 1e-3)
        p_vis = max(vis_probs.get(k, 1e-3), 1e-3)
        fused_scores[k] = (p_jev ** eff_alpha) * (p_vis ** (1.0 - eff_alpha))

    best_k = max(fused_scores, key=lambda k: fused_scores[k])
    best_jev_p = probs.get(best_k, 0.0)
    if (raw_choice == 0 or p_abstain > best_jev_p) and s1 < fallback_cosine:
        if not allow_abstain and s1 >= closed_set_floor and (env_min_conf <= 0.0 or cal_slot["sku_confidence"] >= env_min_conf):
            return (
                top[0][1],
                conf,
                {**meta, "mode": "closed_set_top1_fallback", "abstain_recommended": True},
            )
        return None, conf, {**meta, "mode": "jev_fused_abstain", "abstain_recommended": True}

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

    # Include an open-set dustbin mass so homogeneous neighbor slices never collapse entropy to 0
    total_w = sum(sku_mass.values()) or 1.0
    dustbin_w = max(0.015 * total_w, math.exp((0.52 - max_cos) / max(1e-4, temp)))
    cat_mass["__open_set__"] = dustbin_w
    brand_mass["__open_set__"] = dustbin_w
    sku_mass[-1] = dustbin_w

    all_cats = {str(p.get("category") or "unknown") for p in catalog.values()}
    all_brands = {str(p.get("brand") or p.get("category") or "unknown") for p in catalog.values()}

    top_id = ranked[0][1]
    top_score = ranked[0][0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = top_score - second_score

    top_p = catalog[top_id]
    top_cat = str(top_p.get("category") or "unknown")
    top_brand = str(top_p.get("brand") or top_cat)

    cos_clarity = max(0.08, min(0.985, (top_score - 0.42) / 0.54))
    raw_cat_conf = jev_entropy_confidence(cat_mass, n_options=max(2, len(all_cats) + 1))
    raw_brand_conf = jev_entropy_confidence(brand_mass, n_options=max(2, len(all_brands) + 1))
    raw_sku_conf = jev_entropy_confidence(sku_mass, n_options=len(top_slice) + 1)

    cat_conf = round(max(0.05, min(0.985, 0.65 * raw_cat_conf + 0.35 * cos_clarity)), 4)
    brand_conf = round(max(0.05, min(0.985, 0.65 * raw_brand_conf + 0.35 * cos_clarity)), 4)
    sku_conf = round(max(0.03, min(0.985, 0.65 * raw_sku_conf + 0.35 * cos_clarity)), 4)

    cos_by_id = {pid: cos for cos, pid in ranked}
    brand_ids = [
        pid
        for _, pid in ranked
        if str(catalog[pid].get("brand") or catalog[pid].get("category") or "") == top_brand
    ]
    if len(brand_ids) >= 2:
        intra_brand_margin = cos_by_id[brand_ids[0]] - cos_by_id[brand_ids[1]]
    else:
        # Single-SKU brand: use margin against the runner-up neighbor (capped < 1.0)
        intra_brand_margin = max(SISTER_MARGIN_DEFAULT := 0.05, min(0.98, margin))

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

