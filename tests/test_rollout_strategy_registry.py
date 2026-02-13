"""Unit tests for rollout strategy registry, base class, and identity strategy.

TDD RED phase: These tests define the expected behavior of the rollout strategy
registry foundation. Tests are written first, then implementation follows.

Mirrors test_reward_shaping_registry.py pattern for consistency.
"""

from __future__ import annotations

import numpy as np
import pytest


class MockItem:
    """Mock actor output item for testing rollout strategies."""

    def __init__(self, reward: float, prompt_id: int = 0):
        self.data = {"rewards": np.array(reward)}
        self.metadata = {"prompt_id": prompt_id}


class TestIdentityRolloutStrategy:
    """Tests for the IdentityRolloutStrategy passthrough strategy."""

    def test_identity_strategy_registered(self):
        """After importing rollout_strategy package, 'identity' is in list_strategies()."""
        from src.training.rollout_strategy import list_strategies

        strategies = list_strategies()
        assert "identity" in strategies

    def test_identity_strategy_passthrough(self):
        """IdentityRolloutStrategy.select_rollouts returns items unchanged."""
        from src.training.rollout_strategy.identity import IdentityRolloutStrategy

        strategy = IdentityRolloutStrategy()
        items = [MockItem(reward=float(i), prompt_id=i // 4) for i in range(8)]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert result is items
        assert len(result) == 8

    def test_identity_strategy_name(self):
        """IdentityRolloutStrategy().name == 'identity'."""
        from src.training.rollout_strategy.identity import IdentityRolloutStrategy

        strategy = IdentityRolloutStrategy()
        assert strategy.name == "identity"


class TestRegistry:
    """Tests for the rollout strategy registry."""

    def test_registry_get_strategy(self):
        """get_strategy('identity') returns IdentityRolloutStrategy class."""
        from src.training.rollout_strategy.identity import IdentityRolloutStrategy
        from src.training.rollout_strategy.registry import get_strategy

        cls = get_strategy("identity")
        assert cls is IdentityRolloutStrategy

    def test_registry_get_unknown_raises(self):
        """get_strategy('nonexistent') raises KeyError with available strategies listed."""
        from src.training.rollout_strategy.registry import get_strategy

        with pytest.raises(KeyError, match="nonexistent"):
            get_strategy("nonexistent")

    def test_register_non_subclass_raises(self):
        """register_strategy with non-RolloutStrategy subclass raises TypeError."""
        from src.training.rollout_strategy.registry import register_strategy

        with pytest.raises(TypeError, match="subclass"):
            register_strategy("bad", str)  # type: ignore[arg-type]


class TestCreateStrategyFromConfig:
    """Tests for create_strategy_from_config factory function."""

    def test_create_from_config_default(self):
        """create_strategy_from_config(None) returns IdentityRolloutStrategy instance."""
        from src.training.rollout_strategy.identity import IdentityRolloutStrategy
        from src.training.rollout_strategy.registry import create_strategy_from_config

        strategy = create_strategy_from_config(None)
        assert isinstance(strategy, IdentityRolloutStrategy)

    def test_create_from_config_empty(self):
        """create_strategy_from_config({}) returns IdentityRolloutStrategy instance."""
        from src.training.rollout_strategy.identity import IdentityRolloutStrategy
        from src.training.rollout_strategy.registry import create_strategy_from_config

        strategy = create_strategy_from_config({})
        assert isinstance(strategy, IdentityRolloutStrategy)

    def test_create_from_config_identity(self):
        """create_strategy_from_config({'strategy_name': 'identity'}) returns IdentityRolloutStrategy."""
        from src.training.rollout_strategy.identity import IdentityRolloutStrategy
        from src.training.rollout_strategy.registry import create_strategy_from_config

        strategy = create_strategy_from_config({"strategy_name": "identity"})
        assert isinstance(strategy, IdentityRolloutStrategy)

    def test_create_from_config_best_of_n(self):
        """create_strategy_from_config({'strategy_name': 'best_of_n'}) returns BestOfNStrategy."""
        from src.training.rollout_strategy.best_of_n import BestOfNStrategy
        from src.training.rollout_strategy.registry import create_strategy_from_config

        strategy = create_strategy_from_config({"strategy_name": "best_of_n"})
        assert isinstance(strategy, BestOfNStrategy)
