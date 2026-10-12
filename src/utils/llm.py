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
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import google.auth
import google.auth._default
import yaml
from opentelemetry import trace as otel_trace
from PIL import Image

logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # hide AFC notice

# $SHELF_BENCH_CONFIG (set in the container), else the repo's config.yaml.
CONFIG_PATH = Path(os.environ.get("SHELF_BENCH_CONFIG")
                   or Path(__file__).resolve().parents[2] / "config.yaml")

_GCLOUD_TOKEN_CACHE: dict[str, Any] = {"token": None, "ts": 0.0}


def _fetch_gcloud_cli_token() -> str | None:
    """Fetch a fresh OAuth2 access token from `gcloud auth print-access-token` (cached 45m)."""
    now = time.time()
    if _GCLOUD_TOKEN_CACHE["token"] and (now - float(_GCLOUD_TOKEN_CACHE["ts"])) < 2700:
        return str(_GCLOUD_TOKEN_CACHE["token"])
    try:
        tok = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
        if tok:
            _GCLOUD_TOKEN_CACHE["token"] = tok
            _GCLOUD_TOKEN_CACHE["ts"] = now
            return tok
    except Exception:
        pass
    return None


def ensure_gcloud_auth_fallback() -> None:
    """Patch `google.auth.default` so `gcloud auth login` seamlessly backs expired ADC files."""
    if getattr(google.auth.default, "_gcloud_fallback_wrapped", False):
        return
    orig_default = google.auth.default

    def _wrap_creds(creds: Any) -> Any:
        if creds is None or getattr(creds, "_gcloud_refresh_wrapped", False):
            return creds
        orig_refresh = getattr(creds, "refresh", None)
        if callable(orig_refresh):
            def _refresh(request: Any) -> None:
                try:
                    orig_refresh(request)
                except Exception:
                    tok = _fetch_gcloud_cli_token()
                    if not tok:
                        raise
                    creds.token = tok
                    creds.expiry = None

            creds.refresh = _refresh
            creds._gcloud_refresh_wrapped = True
        return creds

    def _default_with_fallback(*args: Any, **kwargs: Any) -> tuple[Any, str | None]:
        try:
            creds, project = orig_default(*args, **kwargs)
            return _wrap_creds(creds), project
        except Exception:
            tok = _fetch_gcloud_cli_token()
            if not tok:
                raise
            from google.oauth2.credentials import Credentials

            c = _wrap_creds(Credentials(token=tok))
            proj = load_config().get("gcp", {}).get("project", "ul-fs-s-sandbx-16-prj")
            return c, proj

    _default_with_fallback._gcloud_fallback_wrapped = True  # type: ignore[attr-defined]
    google.auth.default = _default_with_fallback
    google.auth._default.default = _default_with_fallback


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text()) if Path(path).exists() else {}


ensure_gcloud_auth_fallback()


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


def usage_from_metadata(m, default_tier: str = "priority") -> Usage:
    """Billable buckets from a Vertex ``usage_metadata``.

    prompt_token_count includes cached tokens; cached tokens bill on the caching SKU, the rest
    on the input SKU for their modality (text or image). Output = candidates + thoughts.
    """
    from utils.pricing import tier_of

    if m is None:
        return Usage(calls=1, traffic={default_tier: 1})
    tier = tier_of(getattr(m, "traffic_type", None), default=default_tier)
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


TIERS = ("standard", "priority", "flex")

# Priority PayGo: https://cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo
# Flex PayGo: https://cloud.google.com/vertex-ai/generative-ai/docs/flex-paygo (50% of Standard
# per token, for latency-tolerant work such as batch shelf processing).
# "shared" skips Provisioned Throughput (none in this project) so every call is PayGo.
TIER_HEADERS = {
    tier: {"X-Vertex-AI-LLM-Request-Type": "shared", "X-Vertex-AI-LLM-Shared-Request-Type": tier}
    for tier in ("priority", "flex")
}


def resolve_model(model: str, config: dict | None = None) -> tuple[str, str, str]:
    """Resolve ``model`` -> ``(target_model_or_endpoint, billing_base_model, location)``.

    Uses ``global`` as the default Vertex AI location for standard publisher models
    (``gemini-3.5-flash-lite``, ``gemini-3.5-flash``, ``gemini-3.8-flash``), and automatically
    falls back to the regional location (``us-central1``) when calling a deployed Vertex AI
    ``/endpoints/`` resource.
    """
    cfg = config or load_config()
    gcp = cfg.get("gcp", {})
    default_loc = gcp.get("location", "global")
    fallback_region = gcp.get("region", "us-central1")
    tuned = cfg.get("tuned_models", {}).get(model)
    if isinstance(tuned, dict):
        base = tuned.get("base_model", model.removesuffix("-sft"))
        cfg_ep = str(tuned.get("endpoint") or "")
        if "ENDPOINT_ID" in cfg_ep:
            cfg_ep = ""
        endpoint = os.environ.get("SHELF_BENCH_SFT_ENDPOINT") or cfg_ep or base
        if "endpoints/" in str(endpoint):
            parts = str(endpoint).split("/")
            ep_loc = parts[parts.index("locations") + 1] if "locations" in parts else None
            tuned_loc = tuned.get("location")
            loc = ep_loc or (tuned_loc if tuned_loc and tuned_loc != "global" else fallback_region)
        else:
            loc = default_loc
        return str(endpoint), str(base), str(loc)
    if model.startswith("projects/") and "/endpoints/" in model:
        parts = model.split("/")
        loc = parts[parts.index("locations") + 1] if "locations" in parts else fallback_region
        base = os.environ.get("SHELF_BENCH_SFT_BASE_MODEL", "gemini-3.5-flash-lite")
        return model, base, loc
    return model, model, default_loc


class Gemini:
    """Callable: ``Gemini(model)(image, prompt) -> LLMResult``."""

    def __init__(self, model: str, config: dict | None = None, thinking_level: str | None = "low",
                 tier: str = "priority"):
        # Context-aware (ECP) client certs on corp machines fail intermittently under concurrent
        # calls ("Failed to configure client certificate and key for mTLS"). Vertex AI does not
        # need them; set GOOGLE_API_USE_CLIENT_CERTIFICATE=true yourself to opt back in.
        os.environ.setdefault("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
        from google import genai
        from google.genai import types

        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        full_cfg = config or load_config()
        cfg = full_cfg.get("gcp", {})
        hf_cfg = (full_cfg.get("hf_gpu_models") or {}).get(model)
        self.hf_spec = hf_cfg if isinstance(hf_cfg, dict) else (
            {"hf_repo": model} if "/" in model and not model.startswith("projects/") else None
        )
        self.hf_base_url = (
            os.environ.get("HF_VLLM_BASE_URL")
            or (str(self.hf_spec.get("base_url") or "") if self.hf_spec else "")
        ).rstrip("/")
        target, base_model, loc = resolve_model(model, full_cfg)
        self.model = model
        self.target_model = target
        self.base_model = base_model
        self.project = cfg.get("project")
        self.location = loc
        self.fallback_region = cfg.get("region", "us-central1")
        self.thinking_level = thinking_level if base_model.startswith("gemini-3") else None
        self.tier = tier
        self._logprobs_unsupported = False
        self._http_options = (
            types.HttpOptions(headers=TIER_HEADERS[tier]) if tier in TIER_HEADERS else None
        )
        self.client = (
            None
            if self.hf_spec is not None
            else genai.Client(
                vertexai=True, project=self.project, location=self.location,
                http_options=self._http_options,
            )
        )

    def _call_hf_vllm(
        self,
        image: Image.Image,
        prompt: str,
        schema: dict | None = None,
        max_side: int | None = None,
        logprobs: int | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        import base64
        import urllib.request

        if not self.hf_base_url:
            raise RuntimeError(
                f"Hugging Face open-weight model {self.model!r} requires a live GPU vLLM endpoint "
                "($HF_VLLM_BASE_URL) or Vertex AI Custom Job execution (`uv run shelf-bench hf-gpu-job "
                f"-a clef_standalone -m {self.model} --dataset hul_golden`). Zero offline fallback permitted."
            )
        jpeg = image_to_jpeg(image, max_side)
        b64_img = base64.b64encode(jpeg).decode("ascii")
        hf_repo = str((self.hf_spec or {}).get("hf_repo") or self.model)
        payload: dict[str, Any] = {
            "model": hf_repo,
            "temperature": 0.0,
            "max_tokens": int(max_output_tokens or 1024),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        if logprobs and logprobs > 0:
            payload["logprobs"] = True
            payload["top_logprobs"] = int(min(16, logprobs))

        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{self.hf_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            resp_json = json.loads(r.read().decode("utf-8"))
        seconds = time.perf_counter() - t0

        choice0 = (resp_json.get("choices") or [{}])[0]
        text = str((choice0.get("message") or {}).get("content") or "")
        u_raw = resp_json.get("usage") or {}
        in_tok = int(u_raw.get("prompt_tokens") or 0)
        out_tok = int(u_raw.get("completion_tokens") or 0)
        usage = Usage(in_tok, out_tok, 0, 1, {self.tier: 1}, {f"{self.tier}/text_input": in_tok, f"{self.tier}/output": out_tok})

        lp_steps: list[dict[str, Any]] = []
        for item in ((choice0.get("logprobs") or {}).get("content") or []):
            tok = item.get("token")
            lp = item.get("logprob")
            if tok is not None and lp is not None:
                top_list = [
                    {"token": str(t.get("token")), "logprob": round(float(t.get("logprob", -20.0)), 5)}
                    for t in (item.get("top_logprobs") or [])
                    if t.get("token") is not None and t.get("logprob") is not None
                ]
                lp_steps.append({"token": str(tok), "logprob": round(float(lp), 5), "top": top_list})

        meta: dict[str, Any] = {
            "attempts": 1,
            "model_version": hf_repo,
            "finish_reason": choice0.get("finish_reason"),
            "tier_requested": self.tier,
            "temperature": 0.0,
            "image_bytes": len(jpeg),
            "response_chars": len(text),
        }
        if lp_steps:
            meta["logprobs"] = lp_steps
        return LLMResult(parse_json(text) if text else [], usage, seconds, text, meta)

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        schema: dict | None = None,
        max_side: int | None = None,
        retries: int = 6,
        logprobs: int | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        if self.hf_spec is not None:
            return self._call_hf_vllm(
                image, prompt, schema=schema, max_side=max_side,
                logprobs=logprobs, max_output_tokens=max_output_tokens,
            )
        from google import genai
        from google.genai import types

        from utils.openjev import serialize_logprobs

        resolved_max_tokens = (
            int(max_output_tokens)
            if max_output_tokens is not None
            else (2048 if (schema is not None or (logprobs and logprobs > 0)) else 32768)
        )
        cfg = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=resolved_max_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )
        if self.thinking_level:
            cfg.thinking_config = types.ThinkingConfig(thinking_level=self.thinking_level)
        if logprobs and logprobs > 0 and not self._logprobs_unsupported:
            cfg.response_logprobs = True
            cfg.logprobs = int(logprobs)
        jpeg = image_to_jpeg(image, max_side)
        contents = [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt]
        span = otel_trace.get_current_span()  # the call's span, opened by Context.ask
        retries_seen: list[str] = []
        resp: Any = None
        seconds = 0.0
        attempt = 0
        assert self.client is not None
        for attempt in range(retries):
            t0 = time.perf_counter()
            try:
                resp = self.client.models.generate_content(
                    model=self.target_model, contents=contents, config=cfg
                )
                seconds = time.perf_counter() - t0
                break
            except Exception as e:  # 429 / 5xx / transient network / logprobs unsupported / global fallback
                err_lower = str(e).lower()
                if cfg.response_logprobs and "logprob" in err_lower:
                    self._logprobs_unsupported = True
                    cfg.response_logprobs = None
                    cfg.logprobs = None
                    continue
                if (
                    self.location == "global"
                    and self.fallback_region
                    and self.fallback_region != "global"
                    and any(k in err_lower for k in ("not found", "404", "location", "region", "unsupported"))
                ):
                    self.location = self.fallback_region
                    self.client = genai.Client(
                        vertexai=True, project=self.project, location=self.location,
                        http_options=self._http_options,
                    )
                    continue
                if attempt == retries - 1 or not _retryable(e):
                    raise
                sleep = 2 ** (attempt + 1) + random.random() * 2  # 2..66s, jittered
                err = f"{type(e).__name__}: {e}"[:500]
                retries_seen.append(err)
                span.add_event("retry", {"attempt": attempt + 1, "error": err,
                                         "sleep_s": round(sleep, 1),
                                         "failed_after_s": round(time.perf_counter() - t0, 3)})
                time.sleep(sleep)
        usage = usage_from_metadata(resp.usage_metadata, default_tier=self.tier)
        text = resp.text or ""
        cand = (resp.candidates or [None])[0]
        finish = getattr(getattr(cand, "finish_reason", None), "name", None)
        lp_steps = serialize_logprobs(getattr(cand, "logprobs_result", None))
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
        if lp_steps:
            meta["logprobs"] = lp_steps
        return LLMResult(parse_json(text) if text else [], usage, seconds, text, meta)


def _retryable(e: Exception) -> bool:
    s = str(e)
    return any(k in s for k in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED",
                                "UNAVAILABLE", "DEADLINE", "timed out", "Connection", "mTLS"))
