"""Unit tests for reward shaping registry, base class, and identity strategy.

TDD RED phase: These tests define the expected behavior of the reward shaping
registry foundation. Tests are written first, then implementation follows.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestRewardShaperBaseClass:
    """Tests for the abstract RewardShaper base class."""

    def test_base_class_is_abstract(self):
        """Instantiating RewardShaper directly raises TypeError."""
        from src.training.reward_shaping.base import RewardShaper

        with pytest.raises(TypeError):
            RewardShaper()


class TestIdentityRewardShaper:
    """Tests for the IdentityRewardShaper passthrough strategy."""

    def test_identity_shaper_returns_rewards_unchanged(self):
        """IdentityRewardShaper.shape_rewards returns rewards array unchanged."""
        from src.training.reward_shaping.identity import IdentityRewardShaper

        shaper = IdentityRewardShaper()
        rewards = np.array([0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0])
        role_masks = {
            "solver": np.ones((8, 128), dtype=bool),
            "verifier": np.zeros((8, 128), dtype=bool),
            "judge": np.zeros((8, 128), dtype=bool),
        }
        trajectory_metadata = [{"problem_id": f"p{i}"} for i in range(8)]

        result = shaper.shape_rewards(rewards, role_masks, trajectory_metadata)

        np.testing.assert_array_equal(result, rewards)

    def test_identity_shaper_preserves_shape(self):
        """shape_rewards with (B,) shape returns (B,) shape."""
        from src.training.reward_shaping.identity import IdentityRewardShaper

        shaper = IdentityRewardShaper()

        # Test with various batch sizes
        for batch_size in [1, 4, 8, 16, 32]:
            rewards = np.random.rand(batch_size)
            result = shaper.shape_rewards(rewards, None, None)
            assert result.shape == (batch_size,), (
                f"Expected shape ({batch_size},), got {result.shape}"
            )
            np.testing.assert_array_equal(result, rewards)

    def test_identity_shaper_name(self):
        """IdentityRewardShaper.name returns 'identity'."""
        from src.training.reward_shaping.identity import IdentityRewardShaper

        shaper = IdentityRewardShaper()
        assert shaper.name == "identity"


class TestRegistry:
    """Tests for the reward shaping strategy registry."""

    def test_registry_register_and_retrieve(self):
        """register_strategy then get_strategy returns the class."""
        from src.training.reward_shaping.base import RewardShaper
        from src.training.reward_shaping.registry import (
            _REGISTRY,
            get_strategy,
            register_strategy,
        )

        # Create a test strategy
        class _TestShaper(RewardShaper):
            @property
            def name(self) -> str:
                return "test"

            def shape_rewards(self, rewards, role_masks, trajectory_metadata):
                return rewards

        register_strategy("test_strategy", _TestShaper)
        try:
            retrieved = get_strategy("test_strategy")
            assert retrieved is _TestShaper
        finally:
            # Clean up registry
            _REGISTRY.pop("test_strategy", None)

    def test_registry_unknown_strategy_raises(self):
        """get_strategy('nonexistent') raises KeyError with helpful message."""
        from src.training.reward_shaping.registry import get_strategy

        with pytest.raises(KeyError, match="nonexistent"):
            get_strategy("nonexistent")

    def test_registry_register_non_subclass_raises(self):
        """register_strategy with non-RewardShaper subclass raises TypeError."""
        from src.training.reward_shaping.registry import register_strategy

        with pytest.raises(TypeError, match="subclass"):
            register_strategy("bad", dict)  # type: ignore[arg-type]


class TestCreateStrategyFromConfig:
    """Tests for create_strategy_from_config factory function."""

    def test_create_strategy_from_config_identity(self):
        """create_strategy_from_config({'strategy_name': 'identity'}) returns IdentityRewardShaper."""
        from src.training.reward_shaping.identity import IdentityRewardShaper
        from src.training.reward_shaping.registry import create_strategy_from_config

        strategy = create_strategy_from_config({"strategy_name": "identity"})
        assert isinstance(strategy, IdentityRewardShaper)

    def test_create_strategy_from_config_with_params(self):
        """create_strategy_from_config with strategy_params passes params to constructor."""
        from src.training.reward_shaping.registry import create_strategy_from_config

        strategy = create_strategy_from_config(
            {"strategy_name": "identity", "strategy_params": {}}
        )
        assert strategy.name == "identity"

    def test_create_strategy_from_config_default_none(self):
        """create_strategy_from_config(None) returns IdentityRewardShaper."""
        from src.training.reward_shaping.identity import IdentityRewardShaper
        from src.training.reward_shaping.registry import create_strategy_from_config

        strategy = create_strategy_from_config(None)
        assert isinstance(strategy, IdentityRewardShaper)

    def test_create_strategy_from_config_default_empty(self):
        """create_strategy_from_config({}) returns IdentityRewardShaper."""
        from src.training.reward_shaping.identity import IdentityRewardShaper
        from src.training.reward_shaping.registry import create_strategy_from_config

        strategy = create_strategy_from_config({})
        assert isinstance(strategy, IdentityRewardShaper)


class TestListStrategies:
    """Tests for list_strategies utility."""

    def test_list_strategies_includes_identity(self):
        """list_strategies() includes 'identity' after identity module import."""
        from src.training.reward_shaping.registry import list_strategies

        # Import identity to trigger auto-registration
        import src.training.reward_shaping.identity  # noqa: F401

        strategies = list_strategies()
        assert "identity" in strategies
