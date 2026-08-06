"""Inference pipelines."""

from .baseline_a import BaselineAPipeline, PipelineResult
from .baseline_b import BaselineBPipeline, BaselineBResult

__all__ = ["BaselineAPipeline", "BaselineBPipeline", "BaselineBResult", "PipelineResult"]
