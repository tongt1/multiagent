"""Sweep integration helpers for dual-model training.

Generates sidecar configuration dicts used by sweep configs to set up
dual vLLM sidecars (solver + verifier).
"""
from __future__ import annotations
from src.training.multi_model.config import MultiModelConfig


def build_sidecar_configs(
    config: MultiModelConfig,
    base_export_dir: str,
    total_sampling_gpus: int = 2,
) -> list[dict]:
    """Build sidecar configuration dicts for dual-model sweep.

    Args:
        config: Multi-model configuration.
        base_export_dir: Base directory for model exports.
        total_sampling_gpus: Total GPUs across all sidecars.

    Returns:
        List of 2 dicts, one per sidecar (solver, verifier).
        Each dict has keys: name, ckpt_path, partition, export_subdir.
    """
    gpus_per_sidecar = total_sampling_gpus // 2
    return [
        {
            "name": "vllm_solver",
            "ckpt_path": config.solver_ckpt,
            "partition": f"gpu_{gpus_per_sidecar}",
            "export_subdir": f"{base_export_dir}/solver",
        },
        {
            "name": "vllm_verifier",
            "ckpt_path": config.verifier_ckpt,
            "partition": f"gpu_{gpus_per_sidecar}",
            "export_subdir": f"{base_export_dir}/verifier",
        },
    ]
