"""SWEEP integration helpers for multi-model training config generation.

Translates MultiModelConfig into Flink-compatible configuration dicts,
including dual vLLM sidecar configs, GPU partition splits, sampler configs,
and patch_run_config with multi_model section.

These helpers are consumed by the SWEEP config class
(configs/sweep_math_debate_multimodel_grpo.py) to programmatically build
the FlinkZordConfig for dual-model debate training.

Design notes:
- Pure Python, no Flink/JAX/sweep imports (testable in isolation)
- GPU split is proportional to model sizes (parsed from "7B", "3B" strings)
- Single-model fallback preserves backward compatibility with existing config
"""

from __future__ import annotations

import math
import re

from src.training.multi_model.config import MultiModelConfig


def _parse_model_size(size_str: str) -> float:
    """Parse a model size string like '7B', '3B', '13B', '0.5B' into a numeric value.

    Args:
        size_str: Size string with optional decimal and 'B' suffix (e.g., "7B", "13B", "0.5B").

    Returns:
        Numeric value in billions (e.g., 7.0 for "7B").

    Raises:
        ValueError: If size_str cannot be parsed.
    """
    match = re.match(r"^(\d+(?:\.\d+)?)B$", size_str.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Cannot parse model size '{size_str}'. Expected format: '7B', '3B', '0.5B', etc."
        )
    return float(match.group(1))


def compute_gpu_split(
    total_sampling_gpus: int,
    solver_size: str | None,
    verifier_size: str | None,
) -> tuple[int, int]:
    """Compute GPU allocation split between solver and verifier models.

    If both model sizes are specified, GPUs are allocated proportionally to
    parameter count (e.g., 7B + 3B on 16 GPUs -> 11 + 5). If sizes are not
    specified, GPUs are split evenly.

    Each model is guaranteed at least 1 GPU.

    Args:
        total_sampling_gpus: Total number of sampling GPUs to partition.
        solver_size: Solver model size string (e.g., "7B") or None.
        verifier_size: Verifier model size string (e.g., "3B") or None.

    Returns:
        Tuple of (solver_gpus, verifier_gpus) that sum to total_sampling_gpus.
    """
    if total_sampling_gpus < 2:
        raise ValueError(
            f"Need at least 2 sampling GPUs for dual-model, got {total_sampling_gpus}"
        )

    if solver_size is not None and verifier_size is not None:
        solver_val = _parse_model_size(solver_size)
        verifier_val = _parse_model_size(verifier_size)
        total_val = solver_val + verifier_val

        # Proportional allocation, rounded to nearest integer
        solver_gpus = max(1, round(total_sampling_gpus * solver_val / total_val))
        verifier_gpus = total_sampling_gpus - solver_gpus

        # Ensure verifier also gets at least 1 GPU
        if verifier_gpus < 1:
            verifier_gpus = 1
            solver_gpus = total_sampling_gpus - 1
    else:
        # Even split (solver gets the extra GPU if odd total)
        solver_gpus = total_sampling_gpus // 2
        verifier_gpus = total_sampling_gpus - solver_gpus

    return (solver_gpus, verifier_gpus)


def build_sidecar_configs(
    config: MultiModelConfig,
    base_export_dir: str,
    total_sampling_gpus: int = 16,
) -> list[dict]:
    """Build vLLM sidecar configuration dicts from MultiModelConfig.

    For single-model mode, returns a single sidecar config (unchanged from
    existing behavior). For multi-model mode, returns two sidecar configs
    with separate names, GPU partitions, export directories, and checkpoint paths.

    Args:
        config: Multi-model training configuration.
        base_export_dir: Base export directory path (e.g., "/data/1d/post-training/...").
        total_sampling_gpus: Total sampling GPUs available for partitioning.

    Returns:
        List of sidecar config dicts. Single-model: 1 item. Multi-model: 2 items.
    """
    if not config.is_multi_model:
        return [
            {
                "name": "vllm",
                "partition": f"gpu_{total_sampling_gpus}",
                "export_dir": base_export_dir,
                "ckpt_path": config.solver_ckpt,
            }
        ]

    solver_gpus, verifier_gpus = compute_gpu_split(
        total_sampling_gpus, config.solver_model_size, config.verifier_model_size
    )

    return [
        {
            "name": "vllm_solver",
            "partition": f"gpu_{solver_gpus}",
            "export_dir": f"{base_export_dir}/solver",
            "ckpt_path": config.solver_ckpt,
        },
        {
            "name": "vllm_verifier",
            "partition": f"gpu_{verifier_gpus}",
            "export_dir": f"{base_export_dir}/verifier",
            "ckpt_path": config.verifier_ckpt,
        },
    ]


def build_sampler_configs(
    config: MultiModelConfig,
    base_port: int = 8000,
) -> dict[str, dict]:
    """Build sampler endpoint configuration dicts for FlinkZordConfig.samplers.

    For single-model, returns a single sampler config keyed by "sampler_key".
    For multi-model, returns two sampler configs keyed by "solver_sampler_key"
    and "verifier_sampler_key", each pointing to its own sidecar and port.

    Args:
        config: Multi-model training configuration.
        base_port: Base port number. Solver uses base_port, verifier uses base_port+1.

    Returns:
        Dict mapping sampler key names to sidecar connection config dicts.
    """
    if not config.is_multi_model:
        return {
            "sampler_key": {
                "sidecar_name": "vllm",
                "port": base_port,
            }
        }

    return {
        "solver_sampler_key": {
            "sidecar_name": "vllm_solver",
            "port": base_port,
        },
        "verifier_sampler_key": {
            "sidecar_name": "vllm_verifier",
            "port": base_port + 1,
        },
    }


def build_patch_run_config(
    config: MultiModelConfig,
    base_config: dict,
) -> dict:
    """Extend a base run config dict with multi_model section.

    For multi-model mode, adds a `multi_model` key with solver/verifier
    checkpoint paths and freeze_roles configuration. For single-model mode,
    returns the base_config unchanged.

    Args:
        config: Multi-model training configuration.
        base_config: Existing patch_run_config dict to extend.

    Returns:
        Updated config dict (new dict in multi-model mode, same dict in single-model mode).
    """
    if not config.is_multi_model:
        return base_config

    # Create a shallow copy to avoid mutating the caller's dict
    result = dict(base_config)
    # Store multi_model config under metadata to avoid RunConfig extra_forbidden
    # validation error. RunConfig.metadata is Dict[str, Any] and accepts arbitrary
    # keys, whereas top-level RunConfig fields are strictly typed (extra="forbid").
    result.setdefault("metadata", {})["multi_model"] = {
        "enabled": True,
        "solver_ckpt": config.solver_ckpt,
        "verifier_ckpt": config.verifier_ckpt,
        "freeze_roles": list(config.freeze_roles),
    }
    return result
