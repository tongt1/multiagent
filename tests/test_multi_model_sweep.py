"""Tests for SWEEP config generation and multi-model integration.

Covers: sidecar config generation, GPU split logic, sampler configs,
and patch_run_config for both single-model and multi-model modes.
"""

from __future__ import annotations

import pytest

from src.training.multi_model.config import MultiModelConfig
from src.training.multi_model.sweep_integration import (
    build_patch_run_config,
    build_sampler_configs,
    build_sidecar_configs,
    compute_gpu_split,
)


# ---------------------------------------------------------------------------
# Single-model backward compatibility
# ---------------------------------------------------------------------------


class TestSingleModelBackwardCompat:
    """Single-model configs must produce unchanged behavior."""

    def test_single_model_sidecar_unchanged(self):
        """Single-model config produces one sidecar named 'vllm' (backward compat)."""
        config = MultiModelConfig()  # Single-model default
        sidecars = build_sidecar_configs(config, base_export_dir="/data/export")

        assert len(sidecars) == 1
        assert sidecars[0]["name"] == "vllm"
        assert sidecars[0]["export_dir"] == "/data/export"


# ---------------------------------------------------------------------------
# Dual-sidecar config generation
# ---------------------------------------------------------------------------


class TestDualSidecarConfigs:
    """Multi-model config generates two properly named sidecars."""

    def test_dual_sidecar_names(self):
        """Multi-model produces two sidecars named 'vllm_solver' and 'vllm_verifier'."""
        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        sidecars = build_sidecar_configs(config, base_export_dir="/data/export")

        assert len(sidecars) == 2
        names = {s["name"] for s in sidecars}
        assert names == {"vllm_solver", "vllm_verifier"}

    def test_dual_sidecar_export_dirs(self):
        """Each sidecar has a separate export directory to prevent model weight collision."""
        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        sidecars = build_sidecar_configs(config, base_export_dir="/data/export")

        solver_sidecar = next(s for s in sidecars if s["name"] == "vllm_solver")
        verifier_sidecar = next(s for s in sidecars if s["name"] == "vllm_verifier")

        assert solver_sidecar["export_dir"] == "/data/export/solver"
        assert verifier_sidecar["export_dir"] == "/data/export/verifier"
        # Verify they are different
        assert solver_sidecar["export_dir"] != verifier_sidecar["export_dir"]


# ---------------------------------------------------------------------------
# GPU split logic
# ---------------------------------------------------------------------------


class TestGPUSplit:
    """GPU allocation logic for proportional and even splits."""

    def test_gpu_split_even(self):
        """Equal model sizes (or unspecified) split GPUs evenly."""
        solver_gpus, verifier_gpus = compute_gpu_split(16, None, None)
        assert solver_gpus + verifier_gpus == 16
        assert solver_gpus == 8
        assert verifier_gpus == 8

    def test_gpu_split_proportional(self):
        """7B + 3B on 16 GPUs -> proportional split (11 + 5)."""
        solver_gpus, verifier_gpus = compute_gpu_split(16, "7B", "3B")
        assert solver_gpus + verifier_gpus == 16
        # 7/(7+3) * 16 = 11.2 -> rounds to 11
        assert solver_gpus == 11
        assert verifier_gpus == 5

    def test_gpu_split_minimum(self):
        """Minimum 1 GPU per model even with extreme asymmetry."""
        # 100B + 0.5B on 2 GPUs -- verifier must still get 1
        solver_gpus, verifier_gpus = compute_gpu_split(2, "100B", "0.5B")
        assert solver_gpus >= 1
        assert verifier_gpus >= 1
        assert solver_gpus + verifier_gpus == 2


# ---------------------------------------------------------------------------
# Sampler configs
# ---------------------------------------------------------------------------


class TestSamplerConfigs:
    """Sampler endpoint configuration for single and multi-model."""

    def test_sampler_configs_dual(self):
        """Multi-model produces two sampler configs with correct endpoint keys."""
        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        samplers = build_sampler_configs(config)

        assert "solver_sampler_key" in samplers
        assert "verifier_sampler_key" in samplers
        assert len(samplers) == 2

        # Different sidecar names
        assert samplers["solver_sampler_key"]["sidecar_name"] == "vllm_solver"
        assert samplers["verifier_sampler_key"]["sidecar_name"] == "vllm_verifier"

        # Different ports
        assert samplers["solver_sampler_key"]["port"] == 8000
        assert samplers["verifier_sampler_key"]["port"] == 8001


# ---------------------------------------------------------------------------
# Patch run config
# ---------------------------------------------------------------------------


class TestPatchRunConfig:
    """patch_run_config extension with multi_model section."""

    def test_patch_run_config_multi_model(self):
        """Multi-model config adds multi_model section to run config."""
        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        base = {"train_batch_size": 8, "lr": 3e-6}
        result = build_patch_run_config(config, base)

        assert "multi_model" in result
        assert result["multi_model"]["enabled"] is True
        assert result["multi_model"]["solver_ckpt"] == "/path/solver"
        assert result["multi_model"]["verifier_ckpt"] == "/path/verifier"

        # Original keys preserved
        assert result["train_batch_size"] == 8

    def test_patch_run_config_single_model(self):
        """Single-model config returns base config unchanged."""
        config = MultiModelConfig()  # Single-model default
        base = {"train_batch_size": 8, "lr": 3e-6}
        result = build_patch_run_config(config, base)

        assert result is base  # Same object, not a copy
        assert "multi_model" not in result

    def test_freeze_roles_in_patch(self):
        """freeze_roles propagated to patch_run_config multi_model section."""
        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            freeze_roles=["verifier", "judge"],
        )
        base = {}
        result = build_patch_run_config(config, base)

        assert result["multi_model"]["freeze_roles"] == ["verifier", "judge"]
