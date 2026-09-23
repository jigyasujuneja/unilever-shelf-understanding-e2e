"""Unilever Shelf Understanding Benchmark Suite on GCP & Vertex AI Gemini."""

from shelf_benchmark.approaches import (
    GLOBAL_APPROACH_REGISTRY,
    BaseShelfApproachPlugin,
    CommonLayerContext,
    SimpleShelfApproachPlugin,
)
from shelf_benchmark.config import (
    BenchmarkConfig,
    CustomAttributeSpec,
    ModelPricing,
    TaxonomyConfig,
)
from shelf_benchmark.runner import BenchmarkRunner
from shelf_benchmark.sdk import (
    ShelfBenchmarkSDK,
    UniversalGenAIClientAdapter,
    UniversalModelSpec,
    register_approach_function,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkConfig",
    "CustomAttributeSpec",
    "ModelPricing",
    "TaxonomyConfig",
    "BenchmarkRunner",
    "ShelfBenchmarkSDK",
    "UniversalGenAIClientAdapter",
    "UniversalModelSpec",
    "register_approach_function",
    "BaseShelfApproachPlugin",
    "SimpleShelfApproachPlugin",
    "CommonLayerContext",
    "GLOBAL_APPROACH_REGISTRY",
]

