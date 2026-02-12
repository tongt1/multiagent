"""CooperBench integration for cooperative coding evaluation.

Provides dataset loading, sandbox evaluation, reward computation,
and a cooperation pipeline for the CooperBench benchmark.

CooperBench measures AI agent cooperation on collaborative coding tasks:
two agents each implement a feature in a shared codebase, then their
patches are merged and tested together.
"""

from pathlib import Path
from typing import Any, Union

import yaml

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


def load_config(config_path: Union[str, Path]) -> CooperBenchConfig:
    """Load CooperBenchConfig from a nested YAML file.

    Flattens the nested YAML structure (dataset.path, agents.solver.model, etc.)
    into the flat field names expected by CooperBenchConfig.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        CooperBenchConfig populated from the YAML file.
    """
    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    flat: dict[str, Any] = {}

    # Top-level fields
    if "mode" in raw:
        flat["mode"] = raw["mode"]

    # Dataset settings
    ds = raw.get("dataset", {})
    if "path" in ds:
        flat["dataset_path"] = ds["path"]
    if "subset" in ds:
        flat["subset"] = ds["subset"]

    # Agent settings
    agents = raw.get("agents", {})
    solver = agents.get("solver", {})
    verifier = agents.get("verifier", {})
    if "model" in solver:
        flat["solver_model"] = solver["model"]
    if "model" in verifier:
        flat["verifier_model"] = verifier["model"]
    if "temperature" in solver:
        flat["temperature"] = solver["temperature"]
    if "max_tokens" in solver:
        flat["max_tokens"] = solver["max_tokens"]

    # Evaluation settings
    ev = raw.get("evaluation", {})
    if "backend" in ev:
        flat["backend"] = ev["backend"]
    if "timeout" in ev:
        flat["timeout"] = ev["timeout"]

    # Cooperation settings
    coop = raw.get("cooperation", {})
    if "max_rounds" in coop:
        flat["max_rounds"] = coop["max_rounds"]
    if "messaging_enabled" in coop:
        flat["messaging_enabled"] = coop["messaging_enabled"]
    if "git_enabled" in coop:
        flat["git_enabled"] = coop["git_enabled"]

    # Execution settings
    ex = raw.get("execution", {})
    if "max_parallel_tasks" in ex:
        flat["max_parallel_tasks"] = ex["max_parallel_tasks"]

    return CooperBenchConfig(**flat)


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
    "load_config",
]
