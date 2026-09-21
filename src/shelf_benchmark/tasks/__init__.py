"""Separated Benchmark Tasks: Detection, Classification, Matching, and Fine-Tuning."""

from shelf_benchmark.tasks.classification import ProductClassificationTask
from shelf_benchmark.tasks.detection import ProductDetectionTask
from shelf_benchmark.tasks.fine_tuning import GeminiFineTuningTask
from shelf_benchmark.tasks.matching import ProductMatchingTask

__all__ = [
    "ProductDetectionTask",
    "ProductClassificationTask",
    "ProductMatchingTask",
    "GeminiFineTuningTask",
]
