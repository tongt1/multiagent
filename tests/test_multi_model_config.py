"""Unit tests for multi-model training configuration, role routing, and model management.

TDD RED phase: These tests define the expected behavior of the multi-model
training foundation -- MultiModelConfig, RoleRouter, and MultiModelManager.
Tests cover config validation, role-to-model routing, batch mask routing,
single vs multi-model modes, and trainable/frozen model key queries.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# MultiModelConfig tests
# ---------------------------------------------------------------------------

class TestMultiModelConfigDefaults:
    """Tests for default MultiModelConfig behavior (single-model mode)."""

    def test_default_config(self):
        """Default config has solver_ckpt=None, verifier_ckpt=None (single-model mode),
        freeze_roles=[]."""
        from src.training.multi_model.config import MultiModelConfig

        config = MultiModelConfig()
        assert config.solver_ckpt is None
        assert config.verifier_ckpt is None
        assert config.freeze_roles == []

    def test_dual_model_config(self):
        """Setting both ckpt paths creates valid dual-model config."""
        from src.training.multi_model.config import MultiModelConfig

        config = MultiModelConfig(
            solver_ckpt="/path/to/solver",
            verifier_ckpt="/path/to/verifier",
        )
        assert config.solver_ckpt == "/path/to/solver"
        assert config.verifier_ckpt == "/path/to/verifier"

    def test_is_multi_model(self):
        """Property returns True only when both ckpts are set."""
        from src.training.multi_model.config import MultiModelConfig

        # Single-model: neither set
        config_none = MultiModelConfig()
        assert config_none.is_multi_model is False

        # Only solver set
        config_solver = MultiModelConfig(solver_ckpt="/path/to/solver")
        assert config_solver.is_multi_model is False

        # Only verifier set
        config_verifier = MultiModelConfig(verifier_ckpt="/path/to/verifier")
        assert config_verifier.is_multi_model is False

        # Both set
        config_both = MultiModelConfig(
            solver_ckpt="/path/to/solver",
            verifier_ckpt="/path/to/verifier",
        )
        assert config_both.is_multi_model is True

    def test_freeze_roles_validation(self):
        """freeze_roles only accepts 'solver', 'verifier', 'judge'; rejects unknown roles."""
        from src.training.multi_model.config import MultiModelConfig

        # Valid roles
        config = MultiModelConfig(freeze_roles=["solver", "verifier"])
        config.validate()  # Should not raise

        config_judge = MultiModelConfig(freeze_roles=["judge"])
        config_judge.validate()  # Should not raise

        # Invalid role
        config_bad = MultiModelConfig(freeze_roles=["attacker"])
        with pytest.raises(ValueError, match="attacker"):
            config_bad.validate()

    def test_model_sizes_optional(self):
        """solver_model_size and verifier_model_size default to None (same size)."""
        from src.training.multi_model.config import MultiModelConfig

        config = MultiModelConfig()
        assert config.solver_model_size is None
        assert config.verifier_model_size is None

    def test_asymmetric_sizes(self):
        """Can set solver_model_size='7B', verifier_model_size='3B'."""
        from src.training.multi_model.config import MultiModelConfig

        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            solver_model_size="7B",
            verifier_model_size="3B",
        )
        assert config.solver_model_size == "7B"
        assert config.verifier_model_size == "3B"


# ---------------------------------------------------------------------------
# RoleRouter tests
# ---------------------------------------------------------------------------

class TestRoleRouterMultiModel:
    """Tests for RoleRouter in multi-model mode."""

    def test_get_model_key_solver(self):
        """'solver' role maps to 'solver_model' key."""
        from src.training.multi_model.role_router import RoleRouter

        router = RoleRouter(is_multi_model=True)
        assert router.get_model_key("solver") == "solver_model"

    def test_get_model_key_verifier_judge(self):
        """'verifier' and 'judge' roles both map to 'verifier_model' key."""
        from src.training.multi_model.role_router import RoleRouter

        router = RoleRouter(is_multi_model=True)
        assert router.get_model_key("verifier") == "verifier_model"
        assert router.get_model_key("judge") == "verifier_model"

    def test_route_batch_masks(self):
        """Given batch role masks, returns dict mapping model_key -> combined mask.
        verifier_model mask = verifier_mask | judge_mask."""
        from src.training.multi_model.role_router import RoleRouter

        router = RoleRouter(is_multi_model=True)

        # Batch of 2 items, sequence length 4
        solver_mask = np.array([
            [True, False, False, False],
            [True, True, False, False],
        ])
        verifier_mask = np.array([
            [False, True, False, False],
            [False, False, True, False],
        ])
        judge_mask = np.array([
            [False, False, True, True],
            [False, False, False, True],
        ])

        role_masks = {
            "solver": solver_mask,
            "verifier": verifier_mask,
            "judge": judge_mask,
        }

        result = router.route_batch_masks(role_masks)

        assert "solver_model" in result
        assert "verifier_model" in result

        np.testing.assert_array_equal(result["solver_model"], solver_mask)
        # verifier_model = verifier | judge
        expected_verifier = verifier_mask | judge_mask
        np.testing.assert_array_equal(result["verifier_model"], expected_verifier)


class TestRoleRouterSingleModel:
    """Tests for RoleRouter in single-model mode."""

    def test_single_model_routing(self):
        """When is_multi_model=False, all roles map to 'policy' (the default Flink model key)."""
        from src.training.multi_model.role_router import RoleRouter

        router = RoleRouter(is_multi_model=False)
        assert router.get_model_key("solver") == "policy"
        assert router.get_model_key("verifier") == "policy"
        assert router.get_model_key("judge") == "policy"


# ---------------------------------------------------------------------------
# MultiModelManager tests
# ---------------------------------------------------------------------------

class TestMultiModelManager:
    """Tests for MultiModelManager model key queries."""

    def test_model_keys_multi(self):
        """Returns ['solver_model', 'verifier_model'] for multi-model config."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.model_manager import MultiModelManager

        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        manager = MultiModelManager(config)
        keys = manager.model_keys
        assert "solver_model" in keys
        assert "verifier_model" in keys
        assert len(keys) == 2

    def test_model_keys_single(self):
        """Returns ['policy'] for single-model config."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.model_manager import MultiModelManager

        config = MultiModelConfig()
        manager = MultiModelManager(config)
        assert manager.model_keys == ["policy"]

    def test_is_role_frozen(self):
        """Respects freeze_roles from config."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.model_manager import MultiModelManager

        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            freeze_roles=["verifier", "judge"],
        )
        manager = MultiModelManager(config)
        assert manager.is_role_frozen("verifier") is True
        assert manager.is_role_frozen("judge") is True
        assert manager.is_role_frozen("solver") is False

    def test_get_trainable_model_keys(self):
        """Returns only model keys whose roles are NOT all frozen."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.model_manager import MultiModelManager

        # Freeze verifier and judge -> verifier_model is fully frozen
        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            freeze_roles=["verifier", "judge"],
        )
        manager = MultiModelManager(config)
        trainable = manager.get_trainable_model_keys()
        assert trainable == ["solver_model"]

    def test_get_frozen_model_keys(self):
        """Returns model keys whose roles are ALL frozen."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.model_manager import MultiModelManager

        # Freeze verifier and judge -> verifier_model is fully frozen
        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
            freeze_roles=["verifier", "judge"],
        )
        manager = MultiModelManager(config)
        frozen = manager.get_frozen_model_keys()
        assert frozen == ["verifier_model"]

    def test_get_trainable_keys_none_frozen(self):
        """When no roles frozen, all model keys are trainable."""
        from src.training.multi_model.config import MultiModelConfig
        from src.training.multi_model.model_manager import MultiModelManager

        config = MultiModelConfig(
            solver_ckpt="/path/solver",
            verifier_ckpt="/path/verifier",
        )
        manager = MultiModelManager(config)
        trainable = manager.get_trainable_model_keys()
        assert "solver_model" in trainable
        assert "verifier_model" in trainable
        assert manager.get_frozen_model_keys() == []
