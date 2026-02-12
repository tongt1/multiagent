"""Unit tests for reward mixing shaping strategy.

TDD RED phase: Tests define expected behavior for RewardMixingShaper
which blends global team reward G with per-role local signals via
r_i = alpha*G + (1-alpha)*r_i_local.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestRewardMixingAlphaOne:
    """Alpha=1.0: pure global reward."""

    def test_reward_mixing_alpha_1_returns_global(self):
        """alpha=1.0: r_i = 1.0*G + 0.0*r_local = G.
        Output equals input rewards for all roles."""
        from src.training.reward_shaping.reward_mixing import RewardMixingShaper

        shaper = RewardMixingShaper(alpha=1.0)
        rewards = np.array([5.0, 0.0, 5.0])
        trajectory_metadata = [
            {
                "local_reward_solver": 1.0,
                "local_reward_verifier": 2.0,
                "local_reward_judge": 3.0,
            },
            {
                "local_reward_solver": 4.0,
                "local_reward_verifier": 5.0,
                "local_reward_judge": 6.0,
            },
            {
                "local_reward_solver": 7.0,
                "local_reward_verifier": 8.0,
                "local_reward_judge": 9.0,
            },
        ]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        assert isinstance(result, dict)
        np.testing.assert_array_almost_equal(result["solver"], rewards)
        np.testing.assert_array_almost_equal(result["verifier"], rewards)
        np.testing.assert_array_almost_equal(result["judge"], rewards)


class TestRewardMixingAlphaZero:
    """Alpha=0.0: pure local reward."""

    def test_reward_mixing_alpha_0_returns_local(self):
        """alpha=0.0: r_i = 0.0*G + 1.0*r_local.
        Output equals per-role local rewards only."""
        from src.training.reward_shaping.reward_mixing import RewardMixingShaper

        shaper = RewardMixingShaper(alpha=0.0)
        rewards = np.array([5.0, 0.0])
        trajectory_metadata = [
            {
                "local_reward_solver": 1.0,
                "local_reward_verifier": 2.0,
                "local_reward_judge": 3.0,
            },
            {
                "local_reward_solver": 4.0,
                "local_reward_verifier": 5.0,
                "local_reward_judge": 6.0,
            },
        ]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        assert isinstance(result, dict)
        np.testing.assert_array_almost_equal(
            result["solver"], np.array([1.0, 4.0])
        )
        np.testing.assert_array_almost_equal(
            result["verifier"], np.array([2.0, 5.0])
        )
        np.testing.assert_array_almost_equal(
            result["judge"], np.array([3.0, 6.0])
        )


class TestRewardMixingAlphaHalf:
    """Alpha=0.5: equal blend."""

    def test_reward_mixing_alpha_half(self):
        """alpha=0.5: r_i = 0.5*G + 0.5*r_local. Verify numeric result."""
        from src.training.reward_shaping.reward_mixing import RewardMixingShaper

        shaper = RewardMixingShaper(alpha=0.5)
        rewards = np.array([5.0, 0.0])
        trajectory_metadata = [
            {
                "local_reward_solver": 3.0,
                "local_reward_verifier": 1.0,
                "local_reward_judge": 4.0,
            },
            {
                "local_reward_solver": 2.0,
                "local_reward_verifier": 6.0,
                "local_reward_judge": 0.0,
            },
        ]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        # Rollout 0: G=5.0, solver_local=3.0 -> 0.5*5 + 0.5*3 = 4.0
        # Rollout 0: G=5.0, verifier_local=1.0 -> 0.5*5 + 0.5*1 = 3.0
        # Rollout 0: G=5.0, judge_local=4.0 -> 0.5*5 + 0.5*4 = 4.5
        # Rollout 1: G=0.0, solver_local=2.0 -> 0.5*0 + 0.5*2 = 1.0
        # Rollout 1: G=0.0, verifier_local=6.0 -> 0.5*0 + 0.5*6 = 3.0
        # Rollout 1: G=0.0, judge_local=0.0 -> 0.5*0 + 0.5*0 = 0.0
        np.testing.assert_array_almost_equal(
            result["solver"], np.array([4.0, 1.0])
        )
        np.testing.assert_array_almost_equal(
            result["verifier"], np.array([3.0, 3.0])
        )
        np.testing.assert_array_almost_equal(
            result["judge"], np.array([4.5, 0.0])
        )


class TestRewardMixingDefaultAlpha:
    """Default alpha value."""

    def test_reward_mixing_default_alpha(self):
        """Default alpha=0.5 when not specified."""
        from src.training.reward_shaping.reward_mixing import RewardMixingShaper

        shaper = RewardMixingShaper()
        assert shaper.alpha == 0.5


class TestRewardMixingNoLocalSignals:
    """Fallback when local signals are absent."""

    def test_reward_mixing_no_local_signals_uses_global(self):
        """When trajectory_metadata has no local_reward_* keys,
        falls back to global reward for local component."""
        from src.training.reward_shaping.reward_mixing import RewardMixingShaper

        shaper = RewardMixingShaper(alpha=0.5)
        rewards = np.array([5.0, 0.0])
        # Metadata without local reward keys
        trajectory_metadata = [{"other_key": 42}, {"other_key": 99}]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        # With no local signals, local falls back to G, so:
        # r_i = 0.5*G + 0.5*G = G
        np.testing.assert_array_almost_equal(result["solver"], rewards)
        np.testing.assert_array_almost_equal(result["verifier"], rewards)
        np.testing.assert_array_almost_equal(result["judge"], rewards)


class TestRewardMixingReturnsDict:
    """Output format."""

    def test_reward_mixing_returns_per_role_dict(self):
        """Output is dict {"solver": [B,], "verifier": [B,], "judge": [B,]}."""
        from src.training.reward_shaping.reward_mixing import RewardMixingShaper

        shaper = RewardMixingShaper()
        batch_size = 4
        rewards = np.array([5.0, 0.0, 5.0, 0.0])
        trajectory_metadata = [
            {
                "local_reward_solver": 1.0,
                "local_reward_verifier": 2.0,
                "local_reward_judge": 3.0,
            }
            for _ in range(batch_size)
        ]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"solver", "verifier", "judge"}
        for role in ["solver", "verifier", "judge"]:
            assert isinstance(result[role], np.ndarray)
            assert result[role].shape == (batch_size,)


class TestRewardMixingRegistered:
    """Registry integration."""

    def test_reward_mixing_registered(self):
        """get_strategy('reward_mixing') returns RewardMixingShaper."""
        from src.training.reward_shaping.registry import get_strategy
        from src.training.reward_shaping.reward_mixing import RewardMixingShaper

        cls = get_strategy("reward_mixing")
        assert cls is RewardMixingShaper


class TestRewardMixingConfigCreation:
    """Config-based creation."""

    def test_reward_mixing_alpha_from_config(self):
        """create_strategy_from_config with strategy_params alpha=0.8
        creates shaper with alpha=0.8."""
        from src.training.reward_shaping.registry import create_strategy_from_config

        # Import to trigger registration
        import src.training.reward_shaping.reward_mixing  # noqa: F401

        strategy = create_strategy_from_config(
            {"strategy_name": "reward_mixing", "strategy_params": {"alpha": 0.8}}
        )
        assert strategy.name == "reward_mixing"
        assert strategy.alpha == 0.8
