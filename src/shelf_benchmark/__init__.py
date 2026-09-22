"""Unilever Shelf Understanding Benchmark Suite on GCP & Vertex AI Gemini."""

from shelf_benchmark.config import BenchmarkConfig, ModelPricing, TaxonomyConfig
from shelf_benchmark.sdk import (
    ShelfBenchmarkSDK,
    UniversalGenAIClientAdapter,
    UniversalModelSpec,
    register_approach_function,
)

__version__ = "0.2.0"

__all__ = [
    "BenchmarkConfig",
    "ModelPricing",
    "TaxonomyConfig",
    "ShelfBenchmarkSDK",
    "UniversalGenAIClientAdapter",
    "UniversalModelSpec",
    "register_approach_function",
]

