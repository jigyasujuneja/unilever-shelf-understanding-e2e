"""Universal Input & Output JSON Schema Contracts for SPEC-001 (Zero-Dependency & Pydantic-Compatible)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class SchemaValidationError(ValueError):
    """Raised when an InputContract or OutputContract payload violates the strict SPEC-001 JSON schema."""


def _enforce_exact_keys(data: Dict[str, Any], required: Set[str], optional: Optional[Set[str]] = None) -> None:
    if not isinstance(data, dict):
        raise SchemaValidationError(f"Expected dict, got {type(data).__name__}")
    allowed = set(required) | (set(optional) if optional else set())
    missing = required - set(data.keys())
    extra = set(data.keys()) - allowed
    if missing:
        raise SchemaValidationError(f"Missing required keys: {sorted(missing)}")
    if extra:
        raise SchemaValidationError(f"Unexpected extra keys: {sorted(extra)}")


class ComplianceStatus(str, Enum):
    """Binary compliance flag for Toker and Display compliance."""

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"


@dataclass
class StoreMetadata:
    """Store context metadata."""

    store_id: str
    channel: str = "MODERN_TRADE"
    planogram_id: str = ""

    def __post_init__(self) -> None:
        if not self.store_id or not isinstance(self.store_id, str):
            raise SchemaValidationError("store_id must be a non-empty string")
        if self.channel not in ("MODERN_TRADE", "GENERAL_TRADE"):
            raise SchemaValidationError(f"Invalid channel: {self.channel}")
        if not self.planogram_id or not isinstance(self.planogram_id, str):
            raise SchemaValidationError("planogram_id must be a non-empty string")

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "StoreMetadata":
        _enforce_exact_keys(data, {"store_id", "channel", "planogram_id"})
        return cls(**data)


@dataclass
class PromoRules:
    """Promotional Toker and display count thresholds."""

    toker_text: str
    min_display_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.toker_text, str):
            raise SchemaValidationError("toker_text must be a string")
        if not isinstance(self.min_display_count, int) or self.min_display_count < 1:
            raise SchemaValidationError("min_display_count must be an int >= 1")

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "PromoRules":
        _enforce_exact_keys(data, {"toker_text", "min_display_count"})
        return cls(**data)


@dataclass
class PlanogramContract:
    """Planogram target SKUs and promotional compliance rules."""

    target_skus: List[str]
    promo_rules: PromoRules

    def __post_init__(self) -> None:
        if not isinstance(self.target_skus, list) or not self.target_skus:
            raise SchemaValidationError("target_skus must be a non-empty list of strings")

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "PlanogramContract":
        _enforce_exact_keys(data, {"target_skus", "promo_rules"})
        rules = (
            data["promo_rules"]
            if isinstance(data["promo_rules"], PromoRules)
            else PromoRules.model_validate(data["promo_rules"])
        )
        return cls(target_skus=list(data["target_skus"]), promo_rules=rules)


@dataclass
class InputContract:
    """Universal Input Contract (SPEC-001 Section 3)."""

    image_path: str
    store_metadata: StoreMetadata
    planogram_contract: PlanogramContract

    def __post_init__(self) -> None:
        if not self.image_path or not isinstance(self.image_path, str):
            raise SchemaValidationError("image_path must be a non-empty string")

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "InputContract":
        _enforce_exact_keys(data, {"image_path", "store_metadata", "planogram_contract"})
        sm = (
            data["store_metadata"]
            if isinstance(data["store_metadata"], StoreMetadata)
            else StoreMetadata.model_validate(data["store_metadata"])
        )
        pc = (
            data["planogram_contract"]
            if isinstance(data["planogram_contract"], PlanogramContract)
            else PlanogramContract.model_validate(data["planogram_contract"])
        )
        return cls(image_path=data["image_path"], store_metadata=sm, planogram_contract=pc)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatencyBreakdownMs:
    """Per-tier and total end-to-end latency in milliseconds."""

    tier1_detection: float
    tier2_catalog_match: float
    tier3_compliance: float
    total_e2e: float

    def __post_init__(self) -> None:
        for name in ("tier1_detection", "tier2_catalog_match", "tier3_compliance", "total_e2e"):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or float(val) < 0.0:
                raise SchemaValidationError(f"{name} must be a non-negative float")

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "LatencyBreakdownMs":
        _enforce_exact_keys(
            data, {"tier1_detection", "tier2_catalog_match", "tier3_compliance", "total_e2e"}
        )
        return cls(**data)


@dataclass
class MetricsPayload:
    """Operational, SOS, and cost metrics per shelf image."""

    total_detected: int
    share_of_shelf_pct: float
    latency_ms: LatencyBreakdownMs
    estimated_cost_inr: float

    def __post_init__(self) -> None:
        if not isinstance(self.total_detected, int) or self.total_detected < 0:
            raise SchemaValidationError("total_detected must be an int >= 0")
        if not (0.0 <= float(self.share_of_shelf_pct) <= 100.0):
            raise SchemaValidationError("share_of_shelf_pct must be between 0 and 100")
        if float(self.estimated_cost_inr) < 0.0:
            raise SchemaValidationError("estimated_cost_inr must be >= 0")

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "MetricsPayload":
        _enforce_exact_keys(
            data, {"total_detected", "share_of_shelf_pct", "latency_ms", "estimated_cost_inr"}
        )
        lat = (
            data["latency_ms"]
            if isinstance(data["latency_ms"], LatencyBreakdownMs)
            else LatencyBreakdownMs.model_validate(data["latency_ms"])
        )
        return cls(
            total_detected=int(data["total_detected"]),
            share_of_shelf_pct=float(data["share_of_shelf_pct"]),
            latency_ms=lat,
            estimated_cost_inr=float(data["estimated_cost_inr"]),
        )


@dataclass
class ResolvedSKU:
    """Localized SKU facing with Base Pack ID and confidence."""

    box_xyxy: List[float]
    base_pack_id: str
    confidence: float
    candidate_ranking: Optional[List[str]] = None
    category: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.box_xyxy, list) or len(self.box_xyxy) != 4:
            raise SchemaValidationError("box_xyxy must be a 4-element [x1, y1, x2, y2] list")
        x1, y1, x2, y2 = [float(v) for v in self.box_xyxy]
        if x2 <= x1 or y2 <= y1:
            raise SchemaValidationError(f"Invalid box_xyxy {self.box_xyxy}")
        self.box_xyxy = [x1, y1, x2, y2]
        if not self.base_pack_id or not isinstance(self.base_pack_id, str):
            raise SchemaValidationError("base_pack_id must be a non-empty string")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise SchemaValidationError("confidence must be in [0.0, 1.0]")

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "ResolvedSKU":
        _enforce_exact_keys(
            data,
            {"box_xyxy", "base_pack_id", "confidence"},
            {"candidate_ranking", "category"},
        )
        return cls(**data)


@dataclass
class MarketSharePayload:
    """MarketShare and Planogram Gap insights."""

    resolved_skus: List[ResolvedSKU]
    red_line_gaps: List[str] = field(default_factory=list)
    width_pack_gaps: List[str] = field(default_factory=list)

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "MarketSharePayload":
        _enforce_exact_keys(data, {"resolved_skus", "red_line_gaps", "width_pack_gaps"})
        skus = [
            s if isinstance(s, ResolvedSKU) else ResolvedSKU.model_validate(s)
            for s in data["resolved_skus"]
        ]
        return cls(
            resolved_skus=skus,
            red_line_gaps=list(data["red_line_gaps"]),
            width_pack_gaps=list(data["width_pack_gaps"]),
        )


@dataclass
class CompliancePayload:
    """Promotional Toker and Display Compliance audit output."""

    toker_status: ComplianceStatus
    display_status: ComplianceStatus
    coaching_message: str

    def __post_init__(self) -> None:
        self.toker_status = ComplianceStatus(self.toker_status)
        self.display_status = ComplianceStatus(self.display_status)
        if not self.coaching_message or not isinstance(self.coaching_message, str):
            raise SchemaValidationError("coaching_message must be a non-empty string")

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "CompliancePayload":
        _enforce_exact_keys(data, {"toker_status", "display_status", "coaching_message"})
        return cls(
            toker_status=ComplianceStatus(data["toker_status"]),
            display_status=ComplianceStatus(data["display_status"]),
            coaching_message=str(data["coaching_message"]),
        )


@dataclass
class OutputContract:
    """Universal Output Contract (SPEC-001 Section 3)."""

    metrics: MetricsPayload
    marketshare: MarketSharePayload
    compliance: CompliancePayload

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "OutputContract":
        _enforce_exact_keys(data, {"metrics", "marketshare", "compliance"})
        m = (
            data["metrics"]
            if isinstance(data["metrics"], MetricsPayload)
            else MetricsPayload.model_validate(data["metrics"])
        )
        ms = (
            data["marketshare"]
            if isinstance(data["marketshare"], MarketSharePayload)
            else MarketSharePayload.model_validate(data["marketshare"])
        )
        comp = (
            data["compliance"]
            if isinstance(data["compliance"], CompliancePayload)
            else CompliancePayload.model_validate(data["compliance"])
        )
        return cls(metrics=m, marketshare=ms, compliance=comp)

    def model_dump(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["compliance"]["toker_status"] = self.compliance.toker_status.value
        raw["compliance"]["display_status"] = self.compliance.display_status.value
        return raw
