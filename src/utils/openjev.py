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


def extract_slot_distributions(
    logprobs_steps: list[dict[str, Any]] | None,
    valid_options: set[int] | None = None,
    n_options: int = 6,
) -> list[dict[str, Any]]:
    """Extract per-slot categorical probability distributions from token logprobs.

    Scans the decoded token stream for integer answer slots (e.g. ``{"choice": 2}``,
    ``{"sku_id": 14}``, or ``{"choices": [2, 1, 0, 3]}``), aggregates the token probabilities
    ``exp(logprob)`` over all candidate integer tokens at each slot, and computes the ``djev``
    normalized entropy confidence ``1 - H(p) / ln(N)``.
    """
    if not logprobs_steps:
        return []
    slots: list[dict[str, Any]] = []
    for step in logprobs_steps:
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
) -> tuple[int | None, float, dict[str, Any]]:
    """Resolve a 1-indexed panel choice using ``djev`` entropy calibration + visual fusion.

    - When ``slot`` is absent (e.g. offline test fakes without logprobs), preserves standard greedy
      resolution with confidence ``1.0`` for valid choices.
    - When ``slot["confidence"] >= min_confidence``, trusts the model's choice (or abstention ``0``).
    - When ``slot["confidence"] < min_confidence`` (high token entropy across look-alike sister
      variants), performs Bayesian log-linear fusion between the VLM slot distribution ``p_jev(k)``
      and the temperature-scaled ``jev + CIELAB`` visual similarity prior ``p_vis(k)``.
    """
    k_len = len(top)
    if not top:
        return None, 0.0, {"mode": "empty_shortlist"}

    if slot is None:
        if isinstance(raw_choice, int) and 1 <= raw_choice <= k_len:
            return top[raw_choice - 1][1], 1.0, {"mode": "greedy", "choice": raw_choice}
        if raw_choice == 0:
            return None, 1.0, {"mode": "greedy_abstain", "choice": 0}
        if top[0][0] >= fallback_cosine:
            return top[0][1], round(float(top[0][0]), 4), {"mode": "cosine_fallback"}
        return None, 0.0, {"mode": "unparsed_abstain"}

    conf = float(slot.get("confidence", 1.0))
    probs: dict[int, float] = {int(k): float(v) for k, v in (slot.get("probabilities") or {}).items()}
    meta: dict[str, Any] = {
        "jev_confidence": conf,
        "jev_probabilities": {str(k): round(v, 4) for k, v in probs.items()},
        "jev_top2_margin": slot.get("top2_margin"),
    }

    # Confident System-1 read: accept choice or explicit abstention (0) directly
    if conf >= min_confidence:
        if isinstance(raw_choice, int) and 1 <= raw_choice <= k_len:
            return top[raw_choice - 1][1], conf, {**meta, "mode": "jev_confident"}
        if raw_choice == 0:
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
    if (raw_choice == 0 or p_abstain > best_jev_p) and top[0][0] < fallback_cosine:
        return None, conf, {**meta, "mode": "jev_fused_abstain"}

    return (
        top[best_k - 1][1],
        conf,
        {**meta, "mode": "jev_visual_fusion", "fused_panel": best_k},
    )


class OpenJevClient:
    """Client for a ``/v1/systemone`` server (`mmastrac/djev`, `razorback16/openjev`).

    Activated when ``$OPENJEV_BASE_URL`` or ``config["openjev"]["base_url"]`` is set.
    Performs single-pass masked-canvas reads (``0`` output tokens) returning exact categorical
    distributions and ``1 - H(p) / ln(N)`` confidence for up to 12 questions simultaneously.
    """

    def __init__(self, config: dict | None = None):
        cfg = (config or load_config()).get("openjev") or {}
        self.base_url = (
            os.environ.get("OPENJEV_BASE_URL") or str(cfg.get("base_url") or "")
        ).rstrip("/")
        self.api_key = os.environ.get("OPENJEV_API_KEY") or str(cfg.get("api_key") or "")
        self.model = (
            os.environ.get("OPENJEV_MODEL")
            or str(cfg.get("model") or "openjev-latest")
        )
        self.timeout_s = float(cfg.get("timeout_s") or 30.0)

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

        u = body.get("usage") or {}
        in_tok = int(u.get("prompt_tokens") or 0)
        out_tok = int(u.get("output_tokens") or 0)
        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=0,
            calls=1,
            traffic={"standard": 1},
            buckets={"standard/text_input": in_tok} if in_tok else {},
        )
        return choices, slots, usage, seconds
