"""Unit tests for dual learner config, gradient router, and per-model LR schedules.

TDD RED phase: These tests define the expected behavior of DualLearnerConfig,
GradientRouter, and PerModelLRSchedule for independent per-model optimizer state,
gradient routing by role mask, and freeze behavior.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# DualLearnerConfig tests
# ---------------------------------------------------------------------------

class TestDualLearnerConfigDefault:
    """Tests for DualLearnerConfig default (single-model) behavior."""

    def test_default_single_model(self):
        """Default config produces single optimizer config with one LR schedule.
        When multi_model_config is single-model (default), get_optimizer_configs
        returns a single entry keyed by 'policy' with base LR values."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.dual_learner import DualLearnerConfig

        mm_config = MultiModelConfig()
        dl_config = DualLearnerConfig(multi_model_config=mm_config)
        opt_configs = dl_config.get_optimizer_configs()

        assert len(opt_configs) == 1
        assert "policy" in opt_configs
        assert opt_configs["policy"].peak_lr == dl_config.base_peak_lr
        assert opt_configs["policy"].end_lr == dl_config.base_end_lr


class TestDualLearnerConfigMultiModel:
    """Tests for DualLearnerConfig in multi-model mode."""

    def test_dual_model_optimizer_configs(self):
        """When multi-model, returns dict mapping model_key -> OptimizerConfig
        with separate peak_lr, end_lr, warmup_steps."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.dual_learner import (
            DualLearnerConfig,
            PerModelOptimizerConfig,
        )

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        dl_config = DualLearnerConfig(
            multi_model_config=mm_config,
            solver_optimizer=PerModelOptimizerConfig(
                peak_lr=3e-6, end_lr=1e-6, warmup_steps=100
            ),
            verifier_optimizer=PerModelOptimizerConfig(
                peak_lr=1e-6, end_lr=5e-7, warmup_steps=50
            ),
        )
        opt_configs = dl_config.get_optimizer_configs()

        assert len(opt_configs) == 2
        assert "solver_model" in opt_configs
        assert "verifier_model" in opt_configs

        assert opt_configs["solver_model"].peak_lr == 3e-6
        assert opt_configs["solver_model"].end_lr == 1e-6
        assert opt_configs["solver_model"].warmup_steps == 100

        assert opt_configs["verifier_model"].peak_lr == 1e-6
        assert opt_configs["verifier_model"].end_lr == 5e-7
        assert opt_configs["verifier_model"].warmup_steps == 50

    def test_solver_custom_lr(self):
        """Can set solver-specific learning rate (e.g., 3e-6) different from
        verifier (e.g., 1e-6)."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.dual_learner import (
            DualLearnerConfig,
            PerModelOptimizerConfig,
        )

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        dl_config = DualLearnerConfig(
            multi_model_config=mm_config,
            solver_optimizer=PerModelOptimizerConfig(peak_lr=3e-6),
            verifier_optimizer=PerModelOptimizerConfig(peak_lr=1e-6),
        )
        opt_configs = dl_config.get_optimizer_configs()

        assert opt_configs["solver_model"].peak_lr == 3e-6
        assert opt_configs["verifier_model"].peak_lr == 1e-6
        assert opt_configs["solver_model"].peak_lr != opt_configs["verifier_model"].peak_lr

    def test_frozen_model_no_optimizer(self):
        """Frozen model key excluded from optimizer configs (no Adam state needed)."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.dual_learner import DualLearnerConfig

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            freeze_roles=["verifier", "judge"],
        )
        dl_config = DualLearnerConfig(multi_model_config=mm_config)
        opt_configs = dl_config.get_optimizer_configs()

        assert "solver_model" in opt_configs
        assert "verifier_model" not in opt_configs  # Frozen: no optimizer needed
        assert len(opt_configs) == 1

    def test_get_active_model_keys(self):
        """Returns only non-frozen model keys that need gradient computation."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.dual_learner import DualLearnerConfig

        # Freeze verifier+judge -> only solver is active
        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            freeze_roles=["verifier", "judge"],
        )
        dl_config = DualLearnerConfig(multi_model_config=mm_config)
        active = dl_config.get_active_model_keys()

        assert active == ["solver_model"]

        # No freeze -> both active
        mm_config_no_freeze = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        dl_config_no_freeze = DualLearnerConfig(multi_model_config=mm_config_no_freeze)
        active_all = dl_config_no_freeze.get_active_model_keys()
        assert "solver_model" in active_all
        assert "verifier_model" in active_all


# ---------------------------------------------------------------------------
# GradientRouter tests
# ---------------------------------------------------------------------------

class TestGradientRouterMasking:
    """Tests for GradientRouter per-token loss masking by role."""

    def test_mask_gradient_by_role(self):
        """Given per-token loss [B, T-1] and role masks, produces per-model masked loss.
        solver_model loss = loss * solver_mask[:, :-1]
        verifier_model loss = loss * (verifier_mask | judge_mask)[:, :-1]"""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.gradient_router import GradientRouter
        from src.training.multi_model.model_manager import MultiModelManager

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        manager = MultiModelManager(mm_config)
        router = GradientRouter(manager)

        # B=2, T=5 -> per_token_loss shape [2, 4] (T-1)
        per_token_loss = np.ones((2, 4), dtype=np.float32)

        # Role masks: [B, T] = [2, 5]
        solver_mask = np.array([
            [True, True, False, False, False],
            [True, False, False, False, False],
        ])
        verifier_mask = np.array([
            [False, False, True, False, False],
            [False, True, False, False, False],
        ])
        judge_mask = np.array([
            [False, False, False, True, True],
            [False, False, True, True, True],
        ])

        role_masks = {
            "debate/role_mask_solver": solver_mask,
            "debate/role_mask_verifier": verifier_mask,
            "debate/role_mask_judge": judge_mask,
        }

        result = router.mask_loss_by_model(per_token_loss, role_masks)

        assert "solver_model" in result
        assert "verifier_model" in result

        # solver_model gets solver_mask[:, :-1] = [[T,T,F,F], [T,F,F,F]]
        expected_solver = per_token_loss * solver_mask[:, :-1].astype(np.float32)
        np.testing.assert_array_almost_equal(result["solver_model"], expected_solver)

        # verifier_model gets (verifier|judge)[:, :-1]
        combined_mask = (verifier_mask | judge_mask)[:, :-1].astype(np.float32)
        expected_verifier = per_token_loss * combined_mask
        np.testing.assert_array_almost_equal(result["verifier_model"], expected_verifier)

    def test_gradient_isolation(self):
        """Solver tokens produce zero gradient for verifier model and vice versa."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.gradient_router import GradientRouter
        from src.training.multi_model.model_manager import MultiModelManager

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        manager = MultiModelManager(mm_config)
        router = GradientRouter(manager)

        per_token_loss = np.ones((1, 3), dtype=np.float32) * 5.0

        # Only solver tokens -- no verifier/judge
        solver_mask = np.array([[True, True, True, True]])
        verifier_mask = np.array([[False, False, False, False]])
        judge_mask = np.array([[False, False, False, False]])

        role_masks = {
            "debate/role_mask_solver": solver_mask,
            "debate/role_mask_verifier": verifier_mask,
            "debate/role_mask_judge": judge_mask,
        }

        result = router.mask_loss_by_model(per_token_loss, role_masks)

        # Solver gets full loss
        np.testing.assert_array_almost_equal(result["solver_model"], per_token_loss)
        # Verifier gets zero loss (no verifier/judge tokens)
        np.testing.assert_array_almost_equal(
            result["verifier_model"], np.zeros_like(per_token_loss)
        )

    def test_frozen_model_zero_gradient(self):
        """Frozen model always gets zero gradient mask regardless of role mask."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.gradient_router import GradientRouter
        from src.training.multi_model.model_manager import MultiModelManager

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            freeze_roles=["verifier", "judge"],
        )
        manager = MultiModelManager(mm_config)
        router = GradientRouter(manager)

        per_token_loss = np.ones((2, 3), dtype=np.float32) * 10.0

        # Both solver and verifier have tokens
        solver_mask = np.array([
            [True, True, False, False],
            [True, False, False, False],
        ])
        verifier_mask = np.array([
            [False, False, True, False],
            [False, True, False, False],
        ])
        judge_mask = np.array([
            [False, False, False, True],
            [False, False, True, True],
        ])

        role_masks = {
            "debate/role_mask_solver": solver_mask,
            "debate/role_mask_verifier": verifier_mask,
            "debate/role_mask_judge": judge_mask,
        }

        result = router.mask_loss_by_model(per_token_loss, role_masks)

        # verifier_model is frozen -> zero loss everywhere
        np.testing.assert_array_almost_equal(
            result["verifier_model"], np.zeros((2, 3), dtype=np.float32)
        )
        # solver_model still gets its masked loss
        expected_solver = per_token_loss * solver_mask[:, :-1].astype(np.float32)
        np.testing.assert_array_almost_equal(result["solver_model"], expected_solver)

    def test_single_model_passthrough(self):
        """In single-model mode, returns original loss unmasked (all roles train one model)."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.gradient_router import GradientRouter
        from src.training.multi_model.model_manager import MultiModelManager

        mm_config = MultiModelConfig()  # Single-model mode
        manager = MultiModelManager(mm_config)
        router = GradientRouter(manager)

        per_token_loss = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        # Role masks exist but in single-model mode they're irrelevant
        solver_mask = np.array([[True, True, False, False]])
        verifier_mask = np.array([[False, False, True, False]])
        judge_mask = np.array([[False, False, False, True]])

        role_masks = {
            "debate/role_mask_solver": solver_mask,
            "debate/role_mask_verifier": verifier_mask,
            "debate/role_mask_judge": judge_mask,
        }

        result = router.mask_loss_by_model(per_token_loss, role_masks)

        assert "policy" in result
        assert len(result) == 1
        np.testing.assert_array_almost_equal(result["policy"], per_token_loss)

    def test_mask_shape_alignment(self):
        """Role masks [B, T] correctly sliced to [B, T-1] to match GRPO objective shape."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.gradient_router import GradientRouter
        from src.training.multi_model.model_manager import MultiModelManager

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        manager = MultiModelManager(mm_config)
        router = GradientRouter(manager)

        B, T = 4, 8
        per_token_loss = np.ones((B, T - 1), dtype=np.float32)  # [4, 7]
        solver_mask = np.ones((B, T), dtype=bool)     # [4, 8]
        verifier_mask = np.zeros((B, T), dtype=bool)  # [4, 8]
        judge_mask = np.zeros((B, T), dtype=bool)     # [4, 8]

        role_masks = {
            "debate/role_mask_solver": solver_mask,
            "debate/role_mask_verifier": verifier_mask,
            "debate/role_mask_judge": judge_mask,
        }

        result = router.mask_loss_by_model(per_token_loss, role_masks)

        # Output should match per_token_loss shape [B, T-1]
        assert result["solver_model"].shape == (B, T - 1)
        assert result["verifier_model"].shape == (B, T - 1)


# ---------------------------------------------------------------------------
# GradientRouter helper method tests
# ---------------------------------------------------------------------------

class TestGradientRouterHelpers:
    """Tests for GradientRouter helper methods."""

    def test_should_compute_gradient(self):
        """should_compute_gradient returns False for frozen model keys."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.gradient_router import GradientRouter
        from src.training.multi_model.model_manager import MultiModelManager

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            freeze_roles=["verifier", "judge"],
        )
        manager = MultiModelManager(mm_config)
        router = GradientRouter(manager)

        assert router.should_compute_gradient("solver_model") is True
        assert router.should_compute_gradient("verifier_model") is False

    def test_get_gradient_mask_frozen(self):
        """get_gradient_mask returns zero mask for frozen models."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.gradient_router import GradientRouter
        from src.training.multi_model.model_manager import MultiModelManager

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            freeze_roles=["verifier", "judge"],
        )
        manager = MultiModelManager(mm_config)
        router = GradientRouter(manager)

        solver_mask = np.ones((2, 5), dtype=bool)
        verifier_mask = np.ones((2, 5), dtype=bool)
        judge_mask = np.ones((2, 5), dtype=bool)

        role_masks = {
            "debate/role_mask_solver": solver_mask,
            "debate/role_mask_verifier": verifier_mask,
            "debate/role_mask_judge": judge_mask,
        }

        frozen_mask = router.get_gradient_mask("verifier_model", role_masks)
        # Frozen -> all zeros, sliced to [B, T-1]
        assert frozen_mask.shape == (2, 4)
        np.testing.assert_array_equal(frozen_mask, np.zeros((2, 4), dtype=bool))

    def test_get_gradient_mask_active(self):
        """get_gradient_mask returns combined role mask sliced to [B, T-1] for active model."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.gradient_router import GradientRouter
        from src.training.multi_model.model_manager import MultiModelManager

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        manager = MultiModelManager(mm_config)
        router = GradientRouter(manager)

        solver_mask = np.array([[True, True, False, False, False]])
        verifier_mask = np.array([[False, False, True, False, False]])
        judge_mask = np.array([[False, False, False, True, True]])

        role_masks = {
            "debate/role_mask_solver": solver_mask,
            "debate/role_mask_verifier": verifier_mask,
            "debate/role_mask_judge": judge_mask,
        }

        solver_grad_mask = router.get_gradient_mask("solver_model", role_masks)
        assert solver_grad_mask.shape == (1, 4)
        expected = solver_mask[:, :-1]
        np.testing.assert_array_equal(solver_grad_mask, expected)


# ---------------------------------------------------------------------------
# PerModelLRSchedule tests
# ---------------------------------------------------------------------------

class TestPerModelLRSchedule:
    """Tests for per-model learning rate schedule configuration."""

    def test_independent_lr_schedules(self):
        """Solver and verifier can have different warmup_steps, peak_lr, end_lr."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.dual_learner import (
            DualLearnerConfig,
            PerModelOptimizerConfig,
        )

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        dl_config = DualLearnerConfig(
            multi_model_config=mm_config,
            solver_optimizer=PerModelOptimizerConfig(
                peak_lr=3e-6, end_lr=1e-6, warmup_steps=100
            ),
            verifier_optimizer=PerModelOptimizerConfig(
                peak_lr=1e-6, end_lr=5e-7, warmup_steps=200
            ),
        )

        solver_params = dl_config.get_lr_schedule_params("solver_model")
        verifier_params = dl_config.get_lr_schedule_params("verifier_model")

        assert solver_params is not None
        assert verifier_params is not None
        assert solver_params["peak_lr"] == 3e-6
        assert solver_params["warmup_steps"] == 100
        assert verifier_params["peak_lr"] == 1e-6
        assert verifier_params["warmup_steps"] == 200

    def test_default_schedule_matches_base(self):
        """When no per-model override, schedule matches the base training config LR."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.dual_learner import DualLearnerConfig

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        dl_config = DualLearnerConfig(
            multi_model_config=mm_config,
            base_peak_lr=5e-6,
            base_end_lr=1e-6,
        )

        solver_params = dl_config.get_lr_schedule_params("solver_model")
        verifier_params = dl_config.get_lr_schedule_params("verifier_model")

        assert solver_params is not None
        assert solver_params["peak_lr"] == 5e-6
        assert solver_params["end_lr"] == 1e-6

        assert verifier_params is not None
        assert verifier_params["peak_lr"] == 5e-6
        assert verifier_params["end_lr"] == 1e-6

    def test_frozen_model_no_schedule(self):
        """Frozen model has no LR schedule (returns None)."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.dual_learner import DualLearnerConfig

        mm_config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            freeze_roles=["verifier", "judge"],
        )
        dl_config = DualLearnerConfig(multi_model_config=mm_config)

        solver_params = dl_config.get_lr_schedule_params("solver_model")
        verifier_params = dl_config.get_lr_schedule_params("verifier_model")

        assert solver_params is not None  # Solver is active
        assert verifier_params is None    # Verifier is frozen -> no schedule
