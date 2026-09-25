"""Approach interface + registry.

An approach turns one shelf image into product boxes and records the steps it took.
To add one: drop a module in ``src/approaches/`` that defines a subclass and
decorates it with ``@register``. It then shows up in ``shelf-bench list`` and can be run
with ``shelf-bench run -a <name> -m <model>``. See ``single_pass.py`` for a ~30-line example.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from utils import telemetry
from utils.llm import LLMResult, Usage

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in original-image pixels

# JSON schema for "a list of [ymin, xmin, ymax, xmax] boxes normalised to 0..1000",
# which is Gemini's native box format.
BOX_LIST_SCHEMA = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "integer"}},
}


@dataclass
class Trace:
    """What the UI shows as 'step by step'. Keep steps few and meaningful."""

    steps: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    billed: dict[str, float] = field(default_factory=dict)  # non-Gemini usage, see Context.bill
    span: Any = None  # the image's OpenTelemetry span; each step is also an event on it
    _t0: float = field(default_factory=time.perf_counter)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def step(self, name: str, detail: str = "", boxes: list[Box] | None = None,
             regions: list[Box] | None = None) -> None:
        """Record a step. ``boxes`` are detections to overlay; ``regions`` e.g. tiles."""
        now = time.perf_counter()
        s: dict[str, Any] = {"name": name, "detail": detail, "ms": round((now - self._t0) * 1000)}
        if boxes is not None:
            s["boxes"] = [[round(v, 1) for v in b] for b in boxes]
        if regions is not None:
            s["regions"] = [[round(v, 1) for v in r] for r in regions]
        self.steps.append(s)
        self._t0 = now
        if self.span is not None:
            self.span.add_event(f"step: {name}", telemetry.clean({
                "detail": detail, "ms": s["ms"],
                "boxes": len(boxes) if boxes is not None else None,
                "regions": len(regions) if regions is not None else None}))


@dataclass
class Context:
    """Everything an approach may use.

    ``llm`` is a Gemini (or a fake in tests). Anything else an approach uses (embeddings,
    AlloyDB, ...) it creates itself in ``Approach.setup`` from the helpers in ``utils/``.
    """

    model: str
    llm: Callable[..., LLMResult]
    trace: Trace = field(default_factory=Trace)
    # Set by the runner: OTel parent for model-call spans (works from any thread), and
    # usage -> Gemini cost dict (list_usd / credit_usd / net_usd) for the span.
    otel_parent: Any = None
    price: Callable[[Usage], dict] | None = None
    sample: Any = None

    def ask(self, image: Image.Image, prompt: str, **kw) -> LLMResult:
        """Call the run's Gemini model. kw: schema=<JSON schema>, max_side=<px downscale>.

        Every call is a span (tokens, traffic type, retries, cost) plus a log entry with the
        prompt and response, so it can be inspected in Cloud Trace / Cloud Logging.
        """
        req = {"gen_ai.operation.name": "generate_content", "gen_ai.provider.name": "gcp.vertex_ai",
               "gen_ai.request.model": self.model, "shelf_bench.image.width": image.width,
               "shelf_bench.image.height": image.height, "shelf_bench.max_side": kw.get("max_side"),
               "shelf_bench.schema": bool(kw.get("schema")), "shelf_bench.prompt_chars": len(prompt)}
        with telemetry.span(f"gemini {self.model}", parent=self.otel_parent, **req) as span:
            t0 = time.perf_counter()
            try:
                res = self.llm(image, prompt, **kw)
            except Exception as e:
                telemetry.fail(span, e)
                telemetry.log(span, "gemini_call_failed",
                              {**req, "error": f"{type(e).__name__}: {e}"[:2000],
                               "seconds": round(time.perf_counter() - t0, 3),
                               "prompt": telemetry.truncate(prompt)}, level=logging.ERROR)
                raise
            with self.trace._lock:
                self.trace.usage = self.trace.usage + res.usage
            cost = self.price(res.usage) if self.price else {}
            m = res.meta
            fields = {
                **req, **telemetry.usage_attrs(res.usage),
                "gen_ai.response.model": m.get("model_version"),
                "gen_ai.response.id": m.get("response_id"),
                "gen_ai.response.finish_reasons": [m["finish_reason"]] if m.get("finish_reason") else None,
                "shelf_bench.traffic_type": ",".join(res.usage.traffic) or None,
                "shelf_bench.tier_requested": m.get("tier_requested"),
                "shelf_bench.attempts": m.get("attempts", 1),
                "shelf_bench.truncated": m.get("truncated"),
                "shelf_bench.thinking_level": m.get("thinking_level"),
                "gen_ai.request.temperature": m.get("temperature"),
                "gen_ai.request.max_tokens": m.get("max_output_tokens"),
                "shelf_bench.image_bytes": m.get("image_bytes"),
                "shelf_bench.response_chars": len(res.text or ""),
                "shelf_bench.model_seconds": round(res.seconds, 3),
                **{f"shelf_bench.cost.{k}": v for k, v in cost.items() if isinstance(v, (int, float))},
            }
            telemetry.set_attrs(span, fields)
            telemetry.log(span, "gemini_call", {
                **{k: v for k, v in fields.items() if v is not None},
                "token_buckets": res.usage.buckets, "retry_errors": m.get("retry_errors") or None,
                "prompt": telemetry.truncate(prompt), "response": telemetry.truncate(res.text)})
        return res

    def bill(self, unit: str, amount: float = 1) -> None:
        """Record billable non-Gemini usage for this image (e.g. ``bill("embedding_image")``).

        ``unit`` must be a key of the approach's ``skus``; it's priced from the Billing Catalog.
        The ``utils`` clients call this for you when you pass them ``ctx``.
        """
        with self.trace._lock:
            self.trace.billed[unit] = self.trace.billed.get(unit, 0) + amount


class Approach:
    name: str = ""          # CLI id
    architecture: str = ""  # one line for the leaderboard
    steps: list[str] = []   # static description of the pipeline for the UI
    # Non-Gemini billable units -> (Billing Catalog service id, SKU description), e.g.
    # utils.embeddings.SKUS. Priced at run start; charge them with ctx.bill(unit, amount).
    skus: dict[str, tuple[str, str]] = {}

    def setup(self, config: dict) -> None:
        """Called once per run before any image: create clients (embeddings, AlloyDB, ...)."""

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        raise NotImplementedError


REGISTRY: dict[str, Approach] = {}


def register(cls: type[Approach]) -> type[Approach]:
    inst = cls()
    REGISTRY[inst.name] = inst
    return cls


def get(name: str) -> Approach:
    _discover()
    if name not in REGISTRY:
        raise KeyError(f"Unknown approach {name!r}. Available: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]


def all_approaches() -> dict[str, Approach]:
    _discover()
    return dict(sorted(REGISTRY.items()))


def _discover() -> None:
    import approaches as pkg

    for m in pkgutil.iter_modules(pkg.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"{pkg.__name__}.{m.name}")


# ---- helpers shared by the Gemini approaches -------------------------------------------------

def to_pixels(raw: Any, x0: float, y0: float, w: float, h: float) -> list[Box]:
    """Gemini [ymin,xmin,ymax,xmax] in 0..1000 of a (x0,y0,w,h) region -> absolute pixel boxes."""
    if isinstance(raw, dict):  # tolerate {"boxes": [...]}
        raw = next((v for v in raw.values() if isinstance(v, list)), [])
    out: list[Box] = []
    for b in raw or []:
        if isinstance(b, dict):
            b = b.get("box_2d") or b.get("bbox_2d") or b.get("box")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        try:
            ymin, xmin, ymax, xmax = (min(1000.0, max(0.0, float(v))) for v in b)
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        out.append((x0 + xmin / 1000 * w, y0 + ymin / 1000 * h,
                    x0 + xmax / 1000 * w, y0 + ymax / 1000 * h))
    return out


DETECT_PROMPT = (
    "Detect every individual retail product visible on the shelves in this image - every "
    "box, bottle, can, bag and package, including small, partially visible and edge-of-frame "
    "items. Each physical item gets its own tight box. Do not merge adjacent identical products. "
    "Return ONLY a JSON array of boxes, each [ymin, xmin, ymax, xmax] normalized to 0-1000."
)

# Product classes both approaches assign. SKU-110K only labels "object", so classes are shown
# in the UI but not scored; boxes labelled not_a_product are dropped (that part is scored).
CATEGORIES = ["food", "beverage", "personal_care", "home_care", "other_product", "not_a_product"]
NOT_PRODUCT = "not_a_product"


def label_counts(labels: list[str]) -> str:
    from collections import Counter

    return ", ".join(f"{n} {c}" for n, c in Counter(labels).most_common())
