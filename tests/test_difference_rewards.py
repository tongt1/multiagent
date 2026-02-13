"""Unit tests for difference rewards shaping strategy.

TDD RED phase: Tests define expected behavior for DifferenceRewardShaper
which computes per-agent marginal contribution D_i = G(z) - G(z_{-i}).
"""

from __future__ import annotations

import numpy as np
import pytest


class TestDifferenceRewardsAllCorrect:
    """When G(z) is high and removing any agent drops reward to 0."""

    def test_difference_rewards_all_correct(self):
        """When G(z) = 5.0 and all counterfactuals G(z_{-i}) = 0.0,
        each agent gets D_i = 5.0 - 0.0 = 5.0 (all agents essential)."""
        from src.training.reward_shaping.difference_rewards import (
            DifferenceRewardShaper,
        )

        shaper = DifferenceRewardShaper()
        rewards = np.array([5.0])
        trajectory_metadata = [
            {
                "counterfactual_solver": 0.0,
                "counterfactual_verifier": 0.0,
                "counterfactual_judge": 0.0,
            }
        ]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        assert isinstance(result, dict)
        np.testing.assert_array_almost_equal(result["solver"], np.array([5.0]))
        np.testing.assert_array_almost_equal(result["verifier"], np.array([5.0]))
        np.testing.assert_array_almost_equal(result["judge"], np.array([5.0]))


class TestDifferenceRewardsSolverOnlyMatters:
    """When only removing the solver affects the outcome."""

    def test_difference_rewards_solver_only_matters(self):
        """When G(z) = 5.0 and G(z_{-solver}) = 0.0 but
        G(z_{-verifier}) = 5.0 and G(z_{-judge}) = 5.0,
        solver gets D = 5.0, verifier and judge get D = 0.0."""
        from src.training.reward_shaping.difference_rewards import (
            DifferenceRewardShaper,
        )

        shaper = DifferenceRewardShaper()
        rewards = np.array([5.0])
        trajectory_metadata = [
            {
                "counterfactual_solver": 0.0,
                "counterfactual_verifier": 5.0,
                "counterfactual_judge": 5.0,
            }
        ]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        assert isinstance(result, dict)
        np.testing.assert_array_almost_equal(result["solver"], np.array([5.0]))
        np.testing.assert_array_almost_equal(result["verifier"], np.array([0.0]))
        np.testing.assert_array_almost_equal(result["judge"], np.array([0.0]))


class TestDifferenceRewardsAllIncorrect:
    """When G(z) = 0.0, all agents get D_i = 0.0."""

    def test_difference_rewards_all_incorrect(self):
        """When G(z) = 0.0, all agents get D_i = 0.0 regardless of
        counterfactuals (no marginal contribution to a failed outcome)."""
        from src.training.reward_shaping.difference_rewards import (
            DifferenceRewardShaper,
        )

        shaper = DifferenceRewardShaper()
        rewards = np.array([0.0])
        trajectory_metadata = [
            {
                "counterfactual_solver": 0.0,
                "counterfactual_verifier": 0.0,
                "counterfactual_judge": 0.0,
            }
        ]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        assert isinstance(result, dict)
        np.testing.assert_array_almost_equal(result["solver"], np.array([0.0]))
        np.testing.assert_array_almost_equal(result["verifier"], np.array([0.0]))
        np.testing.assert_array_almost_equal(result["judge"], np.array([0.0]))


class TestDifferenceRewardsReturnsDict:
    """Output format is a per-role dictionary."""

    def test_difference_rewards_returns_per_role_dict(self):
        """Output is dict {"solver": [B,], "verifier": [B,], "judge": [B,]}."""
        from src.training.reward_shaping.difference_rewards import (
            DifferenceRewardShaper,
        )

        shaper = DifferenceRewardShaper()
        batch_size = 4
        rewards = np.array([5.0, 0.0, 5.0, 0.0])
        trajectory_metadata = [
            {
                "counterfactual_solver": 0.0,
                "counterfactual_verifier": 0.0,
                "counterfactual_judge": 0.0,
            }
            for _ in range(batch_size)
        ]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"solver", "verifier", "judge"}
        for role in ["solver", "verifier", "judge"]:
            assert isinstance(result[role], np.ndarray)
            assert result[role].shape == (batch_size,)


class TestDifferenceRewardsNoRoleMasksFallback:
    """Fallback behavior when no counterfactual data is available."""

    def test_difference_rewards_no_role_masks_falls_back(self):
        """When trajectory_metadata is None, falls back to returning
        G(z) for all agents (no counterfactual possible)."""
        from src.training.reward_shaping.difference_rewards import (
            DifferenceRewardShaper,
        )

        shaper = DifferenceRewardShaper()
        rewards = np.array([5.0, 0.0, 5.0])

        result = shaper.shape_rewards(rewards, None, None)

        assert isinstance(result, dict)
        np.testing.assert_array_equal(result["solver"], rewards)
        np.testing.assert_array_equal(result["verifier"], rewards)
        np.testing.assert_array_equal(result["judge"], rewards)


class TestDifferenceRewardsBatchMixed:
    """Batch processing with mixed correct/incorrect rollouts."""

    def test_difference_rewards_batch_mixed(self):
        """Batch of 4 rollouts with mixed correct/incorrect, verify per-element."""
        from src.training.reward_shaping.difference_rewards import (
            DifferenceRewardShaper,
        )

        shaper = DifferenceRewardShaper()
        rewards = np.array([5.0, 0.0, 5.0, 5.0])
        trajectory_metadata = [
            # Rollout 0: all agents essential
            {
                "counterfactual_solver": 0.0,
                "counterfactual_verifier": 0.0,
                "counterfactual_judge": 0.0,
            },
            # Rollout 1: failed, all zeros
            {
                "counterfactual_solver": 0.0,
                "counterfactual_verifier": 0.0,
                "counterfactual_judge": 0.0,
            },
            # Rollout 2: only solver matters
            {
                "counterfactual_solver": 0.0,
                "counterfactual_verifier": 5.0,
                "counterfactual_judge": 5.0,
            },
            # Rollout 3: solver and judge matter, not verifier
            {
                "counterfactual_solver": 0.0,
                "counterfactual_verifier": 5.0,
                "counterfactual_judge": 0.0,
            },
        ]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        # Rollout 0: D = 5-0 = 5 for all
        # Rollout 1: D = 0-0 = 0 for all
        # Rollout 2: solver=5, verifier=0, judge=0
        # Rollout 3: solver=5, verifier=0, judge=5
        np.testing.assert_array_almost_equal(
            result["solver"], np.array([5.0, 0.0, 5.0, 5.0])
        )
        np.testing.assert_array_almost_equal(
            result["verifier"], np.array([5.0, 0.0, 0.0, 0.0])
        )
        np.testing.assert_array_almost_equal(
            result["judge"], np.array([5.0, 0.0, 0.0, 5.0])
        )


class TestDifferenceRewardsRegistered:
    """Registry integration."""

    def test_difference_rewards_registered(self):
        """get_strategy('difference_rewards') returns DifferenceRewardShaper."""
        from src.training.reward_shaping.difference_rewards import (
            DifferenceRewardShaper,
        )
        from src.training.reward_shaping.registry import get_strategy

        cls = get_strategy("difference_rewards")
        assert cls is DifferenceRewardShaper
