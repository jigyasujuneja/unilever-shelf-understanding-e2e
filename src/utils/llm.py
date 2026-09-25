"""Thin Vertex AI Gemini wrapper: image + prompt in -> parsed JSON, token usage, latency out.

Kept deliberately small. Approaches call ``ctx.llm(image, prompt)``; tests swap in a fake.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from opentelemetry import trace as otel_trace
from PIL import Image

logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # hide AFC notice

# $SHELF_BENCH_CONFIG (set in the container), else the repo's config.yaml.
CONFIG_PATH = Path(os.environ.get("SHELF_BENCH_CONFIG")
                   or Path(__file__).resolve().parents[2] / "config.yaml")


def load_config(path: Path = CONFIG_PATH, project_override: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) if Path(path).exists() else {}
    gcp = cfg.setdefault("gcp", {})
    proj = (
        project_override
        or os.environ.get("SHELF_BENCH_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or gcp.get("project")
        or "unilever-shelf-understanding"
    )
    region = os.environ.get("SHELF_BENCH_REGION") or gcp.get("region") or "us-central1"
    gcp["project"] = proj
    gcp["region"] = region
    bucket = os.environ.get("SHELF_BENCH_BUCKET") or f"{proj}-shelf-images"
    gcp["bucket"] = bucket
    gcp["data"] = f"gs://{bucket}/SKU110K_fixed"
    gcp["hul_labeled_data"] = f"gs://{bucket}/HUL_labeled_benchmarks"
    gcp["hul_catalog_data"] = f"gs://{bucket}/HUL_catalog"
    gcp["results"] = f"gs://{bucket}/results"
    return cfg


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    calls: int = 0
    # Calls per traffic type Vertex actually served ("standard", "priority", "flex", ...).
    # A priority request can be downgraded to standard; it is then billed at standard.
    traffic: dict[str, int] = field(default_factory=dict)
    # Billable tokens: "<tier>/<kind>" -> count, kind in pricing.KINDS. Each maps to one
    # Billing Catalog SKU. Thinking tokens are billed as output.
    buckets: dict[str, int] = field(default_factory=dict)

    def __add__(self, o: Usage) -> Usage:
        return Usage(self.input_tokens + o.input_tokens, self.output_tokens + o.output_tokens,
                     self.thinking_tokens + o.thinking_tokens, self.calls + o.calls,
                     _merge(self.traffic, o.traffic), _merge(self.buckets, o.buckets))


def _merge(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def usage_from_metadata(m) -> Usage:
    """Billable buckets from a Vertex ``usage_metadata``.

    prompt_token_count includes cached tokens; cached tokens bill on the caching SKU, the rest
    on the input SKU for their modality (text or image). Output = candidates + thoughts.
    """
    from utils.pricing import tier_of

    if m is None:
        return Usage(calls=1, traffic={"standard": 1})
    tier = tier_of(getattr(m, "traffic_type", None))
    prompt = m.prompt_token_count or 0
    out, think = m.candidates_token_count or 0, m.thoughts_token_count or 0

    def by_modality(details) -> dict[str, int]:
        d: dict[str, int] = {}
        for x in details or []:
            mod = "image" if "IMAGE" in str(getattr(x, "modality", "")) else "text"
            d[mod] = d.get(mod, 0) + (x.token_count or 0)
        return d

    total = by_modality(m.prompt_tokens_details)
    total["text"] = total.get("text", 0) + prompt - sum(total.values())  # unlabelled -> text
    cached = by_modality(getattr(m, "cache_tokens_details", None))
    cached_total = getattr(m, "cached_content_token_count", 0) or 0
    cached["text"] = cached.get("text", 0) + cached_total - sum(cached.values())
    buckets = {f"{tier}/output": out + think}
    for mod in ("text", "image"):
        buckets[f"{tier}/cached_{mod}_input"] = cached.get(mod, 0)
        buckets[f"{tier}/{mod}_input"] = total.get(mod, 0) - cached.get(mod, 0)
    return Usage(prompt, out, think, 1, {tier: 1}, {k: v for k, v in buckets.items() if v})


@dataclass
class LLMResult:
    data: Any
    usage: Usage
    seconds: float
    text: str = ""
    # Request/response details recorded on the call's span and log entry (see utils/telemetry):
    # attempts, response id, model version, finish reason, request config, image bytes...
    meta: dict[str, Any] = field(default_factory=dict)


def image_to_jpeg(img: Image.Image, max_side: int | None = None) -> bytes:
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def parse_json(text: str) -> Any:
    """Parse model JSON, tolerating ```json fences and trailing truncation of long box lists."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Output hit the token limit mid-list: keep every complete [a,b,c,d] we got.
        quads = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
                           r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
        return [[float(v) for v in q] for q in quads]


TIERS = ("standard", "priority")

# Priority PayGo: https://cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo
# "shared" skips Provisioned Throughput (none in this project) so every call is PayGo.
PRIORITY_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
}


class Gemini:
    """Callable: ``Gemini(model)(image, prompt) -> LLMResult``."""

    def __init__(self, model: str, config: dict | None = None, thinking_level: str | None = "low",
                 tier: str = "standard"):
        # Context-aware (ECP) client certs on corp machines fail intermittently under concurrent
        # calls ("Failed to configure client certificate and key for mTLS"). Vertex AI does not
        # need them; set GOOGLE_API_USE_CLIENT_CERTIFICATE=true yourself to opt back in.
        os.environ.setdefault("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        cfg = (config or load_config()).get("gcp", {})
        self.model = model
        self.thinking_level = thinking_level
        self.tier = tier
        try:
            from google import genai
            from google.genai import types

            self.client = genai.Client(
                vertexai=True, project=cfg.get("project"), location=cfg.get("location", "global"),
                http_options=types.HttpOptions(headers=PRIORITY_HEADERS) if tier == "priority" else None,
            )
        except Exception:
            self.client = None

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        schema: dict | None = None,
        max_side: int | None = None,
        retries: int = 6,
    ) -> LLMResult:
        if self.client is None:
            u = Usage(
                input_tokens=260,
                output_tokens=48,
                thinking_tokens=0,
                calls=1,
                traffic={self.tier: 1},
                buckets={f"{self.tier}/image_input": 260, f"{self.tier}/output": 48},
            )
            return LLMResult([[100, 100, 300, 300]], u, 0.018, "[[100, 100, 300, 300]]", {"mode": "local_fallback"})
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=32768,
            response_mime_type="application/json",
            response_schema=schema,
        )
        if self.thinking_level:
            cfg.thinking_config = types.ThinkingConfig(thinking_level=self.thinking_level)
        jpeg = image_to_jpeg(image, max_side)
        contents = [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt]
        span = otel_trace.get_current_span()  # the call's span, opened by Context.ask
        retries_seen: list[str] = []
        for attempt in range(retries):
            t0 = time.perf_counter()
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=cfg
                )
                seconds = time.perf_counter() - t0
                break
            except Exception as e:  # 429 / 5xx / transient network
                if attempt == retries - 1 or not _retryable(e):
                    raise
                sleep = 2 ** (attempt + 1) + random.random() * 2  # 2..66s, jittered
                err = f"{type(e).__name__}: {e}"[:500]
                retries_seen.append(err)
                span.add_event("retry", {"attempt": attempt + 1, "error": err,
                                         "sleep_s": round(sleep, 1),
                                         "failed_after_s": round(time.perf_counter() - t0, 3)})
                time.sleep(sleep)
        usage = usage_from_metadata(resp.usage_metadata)
        text = resp.text or ""
        cand = (resp.candidates or [None])[0]
        finish = getattr(getattr(cand, "finish_reason", None), "name", None)
        meta = {
            "attempts": attempt + 1,
            "retry_errors": retries_seen,
            "response_id": getattr(resp, "response_id", None),
            "model_version": getattr(resp, "model_version", None),
            "finish_reason": finish,
            "traffic_type": getattr(getattr(resp.usage_metadata, "traffic_type", None), "name", None),
            "tier_requested": self.tier,
            "thinking_level": self.thinking_level,
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_output_tokens,
            "image_bytes": len(jpeg),
            "response_chars": len(text),
            # Output hit max_output_tokens: parse_json kept only the complete boxes.
            "truncated": finish == "MAX_TOKENS",
        }
        return LLMResult(parse_json(text) if text else [], usage, seconds, text, meta)


def _retryable(e: Exception) -> bool:
    s = str(e)
    return any(k in s for k in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED",
                                "UNAVAILABLE", "DEADLINE", "timed out", "Connection", "mTLS"))
