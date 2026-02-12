"""CooperBench integration for cooperative coding evaluation.

Provides dataset loading, sandbox evaluation, reward computation,
and a cooperation pipeline for the CooperBench benchmark.

CooperBench measures AI agent cooperation on collaborative coding tasks:
two agents each implement a feature in a shared codebase, then their
patches are merged and tested together.
"""

from src.evaluation.cooperbench.models import (
    CooperBenchConfig,
    CooperBenchEvalResult,
    CooperBenchJudgment,
    CooperBenchPipelineResult,
    CooperBenchProblem,
    CooperBenchResponse,
    CooperBenchVerification,
    FeatureResult,
    FeatureSpec,
)

__all__ = [
    "CooperBenchConfig",
    "CooperBenchProblem",
    "CooperBenchResponse",
    "CooperBenchVerification",
    "CooperBenchJudgment",
    "CooperBenchEvalResult",
    "CooperBenchPipelineResult",
    "FeatureSpec",
    "FeatureResult",
]
