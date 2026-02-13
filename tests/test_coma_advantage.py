"""Unit tests for COMA counterfactual advantage reward shaping strategy.

TDD RED phase: These tests define the expected behavior of the
COMAAdvantageShaper strategy. Tests cover per-agent advantage computation,
fallback to GRPO group mean, per-role output, and registry integration.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestCOMAAdvantageNoBaseline:
    """Tests for COMA advantage fallback when no baseline metadata is available."""

    def test_coma_no_baseline_returns_raw_advantage(self):
        """When trajectory_metadata has no baseline_reward_* keys, advantage
        equals reward minus mean reward across the GRPO group (standard GRPO advantage)."""
        from src.training.reward_shaping.coma_advantage import COMAAdvantageShaper

        shaper = COMAAdvantageShaper(n_rollouts_per_prompt=4)
        # 1 prompt, 4 rollouts: rewards [0, 5, 0, 5] -> mean=2.5 -> advantages [-2.5, 2.5, -2.5, 2.5]
        rewards = np.array([0.0, 5.0, 0.0, 5.0])
        # No baseline keys in metadata
        metadata = [{"problem_id": "p0"} for _ in range(4)]

        result = shaper.shape_rewards(rewards, None, metadata)

        # Fallback: same advantage for all roles
        assert isinstance(result, dict)
        for role in ["solver", "verifier", "judge"]:
            assert role in result
            np.testing.assert_array_almost_equal(
                result[role], np.array([-2.5, 2.5, -2.5, 2.5])
            )


class TestCOMAAdvantageWithBaseline:
    """Tests for COMA advantage with per-role baseline metadata."""

    def test_coma_with_baseline_computes_advantage(self):
        """When trajectory_metadata[i] has baseline_reward_solver etc.,
        advantage A_i = Q(s,a_i) - baseline_i."""
        from src.training.reward_shaping.coma_advantage import COMAAdvantageShaper

        shaper = COMAAdvantageShaper(n_rollouts_per_prompt=2)
        rewards = np.array([5.0, 0.0])
        metadata = [
            {
                "baseline_reward_solver": 3.0,
                "baseline_reward_verifier": 2.0,
                "baseline_reward_judge": 4.0,
            },
            {
                "baseline_reward_solver": 1.0,
                "baseline_reward_verifier": 3.0,
                "baseline_reward_judge": 2.0,
            },
        ]

        result = shaper.shape_rewards(rewards, None, metadata)

        assert isinstance(result, dict)
        # A_solver = [5.0 - 3.0, 0.0 - 1.0] = [2.0, -1.0]
        np.testing.assert_array_almost_equal(result["solver"], np.array([2.0, -1.0]))
        # A_verifier = [5.0 - 2.0, 0.0 - 3.0] = [3.0, -3.0]
        np.testing.assert_array_almost_equal(
            result["verifier"], np.array([3.0, -3.0])
        )
        # A_judge = [5.0 - 4.0, 0.0 - 2.0] = [1.0, -2.0]
        np.testing.assert_array_almost_equal(result["judge"], np.array([1.0, -2.0]))

    def test_coma_advantage_positive_for_above_baseline(self):
        """Agent i chose action better than baseline: A_i > 0."""
        from src.training.reward_shaping.coma_advantage import COMAAdvantageShaper

        shaper = COMAAdvantageShaper(n_rollouts_per_prompt=1)
        rewards = np.array([5.0])
        metadata = [{"baseline_reward_solver": 2.0, "baseline_reward_verifier": 1.0, "baseline_reward_judge": 3.0}]

        result = shaper.shape_rewards(rewards, None, metadata)

        assert result["solver"][0] > 0  # 5.0 - 2.0 = 3.0
        assert result["verifier"][0] > 0  # 5.0 - 1.0 = 4.0
        assert result["judge"][0] > 0  # 5.0 - 3.0 = 2.0

    def test_coma_advantage_negative_for_below_baseline(self):
        """Agent i chose action worse than baseline: A_i < 0."""
        from src.training.reward_shaping.coma_advantage import COMAAdvantageShaper

        shaper = COMAAdvantageShaper(n_rollouts_per_prompt=1)
        rewards = np.array([0.0])
        metadata = [{"baseline_reward_solver": 2.0, "baseline_reward_verifier": 3.0, "baseline_reward_judge": 1.0}]

        result = shaper.shape_rewards(rewards, None, metadata)

        assert result["solver"][0] < 0  # 0.0 - 2.0 = -2.0
        assert result["verifier"][0] < 0  # 0.0 - 3.0 = -3.0
        assert result["judge"][0] < 0  # 0.0 - 1.0 = -1.0

    def test_coma_advantage_zero_at_baseline(self):
        """Agent i matches baseline exactly: A_i = 0."""
        from src.training.reward_shaping.coma_advantage import COMAAdvantageShaper

        shaper = COMAAdvantageShaper(n_rollouts_per_prompt=1)
        rewards = np.array([5.0])
        metadata = [{"baseline_reward_solver": 5.0, "baseline_reward_verifier": 5.0, "baseline_reward_judge": 5.0}]

        result = shaper.shape_rewards(rewards, None, metadata)

        np.testing.assert_array_almost_equal(result["solver"], np.array([0.0]))
        np.testing.assert_array_almost_equal(result["verifier"], np.array([0.0]))
        np.testing.assert_array_almost_equal(result["judge"], np.array([0.0]))


class TestCOMAAdvantageOutput:
    """Tests for COMA advantage output format and batch computation."""

    def test_coma_returns_per_role_dict(self):
        """Output is dict {'solver': [B,], 'verifier': [B,], 'judge': [B,]}."""
        from src.training.reward_shaping.coma_advantage import COMAAdvantageShaper

        shaper = COMAAdvantageShaper(n_rollouts_per_prompt=4)
        rewards = np.array([0.0, 5.0, 0.0, 5.0])
        metadata = [{"problem_id": "p0"} for _ in range(4)]

        result = shaper.shape_rewards(rewards, None, metadata)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"solver", "verifier", "judge"}
        for role in ["solver", "verifier", "judge"]:
            assert isinstance(result[role], np.ndarray)
            assert result[role].shape == (4,)

    def test_coma_batch_computation(self):
        """Batch of 8 rollouts (1 prompt, 8 GRPO samples), verify per-role advantage."""
        from src.training.reward_shaping.coma_advantage import COMAAdvantageShaper

        shaper = COMAAdvantageShaper(n_rollouts_per_prompt=8)
        # 1 prompt, 8 rollouts with mixed rewards
        rewards = np.array([5.0, 0.0, 5.0, 5.0, 0.0, 0.0, 5.0, 0.0])
        metadata = [{"problem_id": "p0"} for _ in range(8)]

        result = shaper.shape_rewards(rewards, None, metadata)

        # mean reward = (5+0+5+5+0+0+5+0)/8 = 20/8 = 2.5
        expected = rewards - 2.5
        for role in ["solver", "verifier", "judge"]:
            np.testing.assert_array_almost_equal(result[role], expected)


class TestCOMAAdvantageRegistry:
    """Tests for COMA advantage registry integration."""

    def test_coma_registered(self):
        """get_strategy('coma_advantage') returns COMAAdvantageShaper class."""
        from src.training.reward_shaping.coma_advantage import COMAAdvantageShaper
        from src.training.reward_shaping.registry import get_strategy

        cls = get_strategy("coma_advantage")
        assert cls is COMAAdvantageShaper
