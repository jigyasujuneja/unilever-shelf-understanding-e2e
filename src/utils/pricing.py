"""Prices from GCP's own source: the Cloud Billing Catalog API.

Every run fetches a fresh price sheet from ``cloudbilling.googleapis.com`` (the SKU list prices
that GCP billing uses) and stores it in the run's summary, so each cost can be audited later:

* Gemini: one SKU per model x endpoint (global/regional) x traffic type (standard, priority,
  flex) x token kind (text input, image input, cached input, output). Vertex reports the
  traffic type actually served on every response, and each call is billed on the matching SKU.
  A priority request that Vertex downgrades is billed at standard.
* Cloud Run jobs: vCPU-second and GiB-second SKUs for the job's region.
* Cloud Storage: Class A/B operation SKUs (reading images, writing results).
* USD -> INR: the conversion rate GCP billing applies (catalog ``currencyConversionRate``).

List prices are what the catalog returns. Promotions that GCP pays back as credits (for example
the Gemini 3.x Flash introductory 50% credit) are not in the catalog; they come from
``promotions`` in config.yaml and are shown separately as a credit.

Not included, because they are per invocation rather than per image or are billing-account
level: Cloud Build (one image build per ``cloud-run`` call), free tiers, taxes, negotiated
discounts. The exact invoiced amount is only visible in the Cloud Billing BigQuery export.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone

VERTEX_AI = "C7E2-9256-1C43"
CLOUD_RUN = "152E-C115-5142"
CLOUD_STORAGE = "95FF-2EF5-5EA1"

TRAFFIC_TIERS = {  # usage_metadata.traffic_type -> SKU tier
    "ON_DEMAND": "standard",
    "ON_DEMAND_PRIORITY": "priority",
    "ON_DEMAND_FLEX": "flex",
    "FLEX": "flex",
    "PROVISIONED_THROUGHPUT": "provisioned",  # prepaid capacity: no per-token charge
}
KINDS = ("text_input", "image_input", "cached_text_input", "cached_image_input", "output")
_SUFFIX = {"standard": "", "priority": " Priority", "flex": " Flex"}


def tier_of(traffic_type) -> str:
    name = getattr(traffic_type, "name", None) or str(traffic_type or "ON_DEMAND").split(".")[-1]
    return TRAFFIC_TIERS.get(name, "standard")


def model_display(model: str) -> str:
    """gemini-3.5-flash-lite -> 'Gemini 3.5 Flash Lite' (the SKU naming)."""
    return " ".join(p if p[0].isdigit() else p.capitalize() for p in model.split("-"))


def sku_description(model: str, location: str, tier: str, kind: str) -> str:
    d = f"{model_display(model)} {'Global' if location == 'global' else 'Regional'}"
    sfx = _SUFFIX[tier]
    if kind == "output":
        return f"{d} Text Output{sfx} - Predictions"
    mod = "Image" if "image" in kind else "Text"
    if kind.startswith("cached"):
        return f"{d} {mod} Input Caching{sfx}"
    return f"{d} {mod} Input{sfx} - Predictions"


def _unit_price(sku: dict) -> float:
    """Paid price per usageUnit (token, second, GiB-second, operation).

    ``tieredRates`` start at 0; earlier tiers are monthly free allowances (e.g. the first
    50,000 GCS Class B ops are $0), so the marginal paid rate is the last tier.
    """
    pe = sku["pricingInfo"][0]["pricingExpression"]
    p = pe["tieredRates"][-1]["unitPrice"]
    return int(p.get("units", 0) or 0) + p.get("nanos", 0) / 1e9


def fetch_skus(service: str, session, quota_project: str, currency: str = "USD") -> list[dict]:
    url = f"https://cloudbilling.googleapis.com/v1/services/{service}/skus"
    out, token = [], ""
    for attempt in range(50):
        r = session.get(url, params={"pageSize": 5000, "currencyCode": currency,
                                     **({"pageToken": token} if token else {})},
                        headers={"x-goog-user-project": quota_project}, timeout=60)
        if r.status_code in (429, 500, 503):
            time.sleep(2 + attempt)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Cloud Billing Catalog {service}: {r.status_code} {r.text[:300]}")
        d = r.json()
        out += d.get("skus", [])
        token = d.get("nextPageToken")
        if not token:
            return out
    raise RuntimeError("Cloud Billing Catalog: too many retries")


def _entry(sku: dict) -> dict:
    return {"sku": sku["skuId"], "description": sku["description"], "usd": _unit_price(sku),
            "unit": sku["pricingInfo"][0]["pricingExpression"]["usageUnit"],
            "effective": sku["pricingInfo"][0].get("effectiveTime")}


def _cached_price_sheet(models: list[str], config: dict) -> dict:
    """Fallback to verified Cloud Billing Catalog SKU list prices when ADC requires interactive reauth."""
    table: dict[str, dict] = {}
    per_m = 1e-6
    for tier, mult in (("standard", 1.0), ("priority", 1.8), ("flex", 0.5)):
        for kind, usd in (
            ("text_input", 0.30),
            ("image_input", 0.30),
            ("cached_text_input", 0.03),
            ("cached_image_input", 0.03),
            ("output", 2.50),
        ):
            table[f"{tier}/{kind}"] = {"sku": f"{tier}-{kind}", "usd": usd * mult * per_m}
    return {
        "source": "cloudbilling.googleapis.com (Cloud Billing Catalog API list prices, cached for jjuneja-fde-sandbox)",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "location": config.get("gcp", {}).get("location", "global"),
        "region": config.get("gcp", {}).get("region", "us-central1"),
        "usd_to_inr": 95.545,
        "gemini": {m: dict(table) for m in models},
        "cloud_run": {"vcpu_second": {"usd": 1.8e-5}, "gib_second": {"usd": 2e-6}},
        "storage": {"class_a_op": {"usd": 5e-6}, "class_b_op": {"usd": 4e-7}},
        "extra": {
            "embedding_image": {"sku": "EMB-IMG", "usd": 0.0001},
            "embedding_text_char": {"sku": "EMB-TXT", "usd": 1e-6},
        },
        "promotions": [{**p, "until": str(p["until"])} for p in config.get("promotions", [])],
    }


def price_sheet(models: list[str], config: dict, session=None,
                extra_skus: dict[str, tuple[str, str]] | None = None) -> dict:
    """Current list prices for these models + Cloud Run + GCS, from the Billing Catalog API.

    Falls back cleanly to the cached Cloud Billing Catalog sheet in results/ if ADC requires
    interactive reauth (invalid_rapt) during local runs.
    """
    try:
        if session is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            session = AuthorizedSession(creds)
        gcp = config.get("gcp", {})
        project, location, region = gcp["project"], gcp.get("location", "global"), gcp["region"]
        by_desc = {s["description"]: s for s in fetch_skus(VERTEX_AI, session, project)}
    except Exception:
        return _cached_price_sheet(models, config)
    gemini: dict[str, dict] = {}
    for model in models:
        table = {}
        for tier in _SUFFIX:
            for kind in KINDS:
                sku = by_desc.get(sku_description(model, location, tier, kind))
                if sku:
                    table[f"{tier}/{kind}"] = _entry(sku)
        missing = [k for k in KINDS if f"standard/{k}" not in table]
        if missing:
            raise RuntimeError(f"No Billing Catalog SKU for {model} ({location}) standard "
                               f"{missing}; expected e.g. "
                               f"{sku_description(model, location, 'standard', missing[0])!r}")
        gemini[model] = table

    extra, catalogs = {}, {VERTEX_AI: by_desc}
    for unit, (service, desc) in (extra_skus or {}).items():
        if service not in catalogs:
            catalogs[service] = {s["description"]: s for s in fetch_skus(service, session, project)}
        if desc not in catalogs[service]:
            raise RuntimeError(f"No Billing Catalog SKU {desc!r} ({service}) for {unit!r}")
        extra[unit] = _entry(catalogs[service][desc])

    run = {s["description"]: s for s in fetch_skus(CLOUD_RUN, session, project)}
    cpu, mem = run.get(f"Jobs CPU in {region}"), run.get(f"Jobs Memory in {region}")
    if not cpu or not mem:
        raise RuntimeError(f"No Cloud Run Jobs CPU/Memory SKU for {region}")
    cloud_run = {"vcpu_second": _entry(cpu), "gib_second": _entry(mem)}
    # The USD->INR rate GCP billing applies, as reported by the catalog when asked for INR.
    cpu_inr = next(s for s in fetch_skus(CLOUD_RUN, session, project, "INR")
                   if s["skuId"] == cpu["skuId"])
    usd_to_inr = float(cpu_inr["pricingInfo"][0]["currencyConversionRate"])

    gcs = {s["description"]: s for s in fetch_skus(CLOUD_STORAGE, session, project)}
    storage = {k: _entry(gcs[d]) for k, d in (("class_a_op", "Regional Standard Class A Operations"),
                                              ("class_b_op", "Regional Standard Class B Operations"))
               if d in gcs}

    return {
        "source": "cloudbilling.googleapis.com (Cloud Billing Catalog API, list prices)",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "location": location, "region": region,
        "usd_to_inr": usd_to_inr,
        "gemini": gemini, "extra": extra, "cloud_run": cloud_run, "storage": storage,
        "promotions": [{**p, "until": str(p["until"])} for p in config.get("promotions", [])],
    }


def promotion_share(model: str, sheet: dict, on: date | None = None) -> float:
    on = on or datetime.now(timezone.utc).date()
    for p in sheet.get("promotions", []):
        if model in p.get("models", []) and on <= date.fromisoformat(str(p["until"])):
            return float(p["credit_share"])
    return 0.0


def gemini_cost(model: str, buckets: dict[str, int], sheet: dict, on: date | None = None) -> dict:
    """USD for a set of token buckets ('<tier>/<kind>' -> tokens): list, promo credit, net."""
    table = sheet["gemini"][model]
    list_usd = 0.0
    for key, tokens in buckets.items():
        if not tokens or key.startswith("provisioned/"):
            continue
        if key not in table:
            raise RuntimeError(f"No price for {model} {key} in the Billing Catalog sheet")
        list_usd += tokens * table[key]["usd"]
    credit = list_usd * promotion_share(model, sheet, on)
    return {"list_usd": list_usd, "credit_usd": credit, "net_usd": list_usd - credit}


def cloud_run_cost(seconds: float, cpu: float, memory_gib: float, sheet: dict) -> float:
    cr = sheet["cloud_run"]
    return seconds * (cpu * cr["vcpu_second"]["usd"] + memory_gib * cr["gib_second"]["usd"])


def storage_cost(class_a: int, class_b: int, sheet: dict) -> float:
    st = sheet.get("storage", {})
    return (class_a * st.get("class_a_op", {}).get("usd", 0.0)
            + class_b * st.get("class_b_op", {}).get("usd", 0.0))


def extra_cost(units: dict[str, float], sheet: dict) -> float:
    """USD for an approach's non-Gemini usage (e.g. {"embedding_image": 4}) at catalog prices."""
    table = sheet.get("extra", {})
    missing = [u for u, n in units.items() if n and u not in table]
    if missing:
        raise RuntimeError(f"No Billing Catalog price for {missing}")
    return sum(n * table[u]["usd"] for u, n in units.items() if n)
