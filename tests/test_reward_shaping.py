"""Tests for reward shaping strategies."""

import numpy as np
import pytest

from src.training.reward_shaping import (
    RewardShapingConfig,
    RewardShapingMode,
    apply_reward_shaping,
    apply_reward_shaping_batch,
)


class TestRewardShaping:
    """Tests for reward shaping functions."""

    def test_quality_mode_basic(self):
        """Test quality mode reward shaping with basic trajectory."""
        # Create simple trajectory: 2 agents, 3 turns
        raw_rewards = np.array([
            [1.0, 1.0, 1.0],  # Agent 0: consistent high rewards
            [0.0, 0.0, 0.0],  # Agent 1: consistent low rewards
        ])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="quality",
            alpha=0.5,
            beta=0.0,  # No outcome reward for this test
        )

        # Turn 0: Should be unchanged (no history)
        assert shaped[0, 0] == raw_rewards[0, 0]
        assert shaped[1, 0] == raw_rewards[1, 0]

        # Turn 1: Q_t = mean([1.0]) = 1.0, R_t = 1.0
        # Quality term = Q_t * R_t - (1 - Q_t) * (1 - R_t) = 1.0 * 1.0 - 0.0 * 0.0 = 1.0
        # Shaped = R_t + alpha * term = 1.0 + 0.5 * 1.0 = 1.5
        assert shaped[0, 1] == pytest.approx(1.5)

        # Turn 1: Q_t = mean([0.0]) = 0.0, R_t = 0.0
        # Quality term = 0.0 * 0.0 - 1.0 * 1.0 = -1.0
        # Shaped = 0.0 + 0.5 * (-1.0) = -0.5
        assert shaped[1, 1] == pytest.approx(-0.5)

    def test_margin_mode_basic(self):
        """Test margin mode reward shaping with basic trajectory."""
        raw_rewards = np.array([
            [0.5, 0.8, 1.0],  # Agent 0: improving rewards
            [1.0, 0.5, 0.0],  # Agent 1: declining rewards
        ])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="margin",
            alpha=0.5,
            beta=0.0,
        )

        # Turn 0: Should be unchanged
        assert shaped[0, 0] == 0.5
        assert shaped[1, 0] == 1.0

        # Turn 1: Q_t = mean([0.5]) = 0.5, R_t = 0.8
        # Margin term = R_t - Q_t = 0.8 - 0.5 = 0.3
        # Shaped = 0.8 + 0.5 * 0.3 = 0.95
        assert shaped[0, 1] == pytest.approx(0.95)

        # Turn 2: Q_t = mean([0.5, 0.8]) = 0.65, R_t = 1.0
        # Margin term = 1.0 - 0.65 = 0.35
        # Shaped = 1.0 + 0.5 * 0.35 = 1.175
        assert shaped[0, 2] == pytest.approx(1.175)

        # Turn 1: Q_t = mean([1.0]) = 1.0, R_t = 0.5
        # Margin term = 0.5 - 1.0 = -0.5
        # Shaped = 0.5 + 0.5 * (-0.5) = 0.25
        assert shaped[1, 1] == pytest.approx(0.25)

    def test_turn_0_uses_raw_reward(self):
        """Test that turn 0 always uses raw reward (no shaping)."""
        raw_rewards = np.array([
            [0.8, 0.5, 0.3],
        ])

        # Test both modes
        for mode in ["quality", "margin"]:
            shaped = apply_reward_shaping(
                raw_rewards=raw_rewards,
                mode=mode,
                alpha=1.0,  # Max shaping
                beta=0.0,
            )

            # Turn 0 should be unchanged
            assert shaped[0, 0] == 0.8

    def test_quality_mode_formula(self):
        """Test quality mode formula: Q*R - (1-Q)*(1-R)."""
        # Single agent, 2 turns
        raw_rewards = np.array([[0.6, 0.8]])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="quality",
            alpha=1.0,  # Full shaping weight
            beta=0.0,
        )

        # Turn 1: Q_t = 0.6, R_t = 0.8
        # Quality term = 0.6 * 0.8 - (1 - 0.6) * (1 - 0.8)
        #              = 0.48 - 0.4 * 0.2
        #              = 0.48 - 0.08 = 0.4
        # Shaped = 0.8 + 1.0 * 0.4 = 1.2
        assert shaped[0, 1] == pytest.approx(1.2)

    def test_margin_mode_formula(self):
        """Test margin mode formula: R - Q."""
        raw_rewards = np.array([[0.6, 0.8]])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="margin",
            alpha=1.0,
            beta=0.0,
        )

        # Turn 1: Q_t = 0.6, R_t = 0.8
        # Margin term = 0.8 - 0.6 = 0.2
        # Shaped = 0.8 + 1.0 * 0.2 = 1.0
        assert shaped[0, 1] == pytest.approx(1.0)

    def test_historical_average_computation(self):
        """Test that Q_t is correctly computed as mean of previous turns."""
        raw_rewards = np.array([[0.2, 0.4, 0.6, 0.8]])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="margin",
            alpha=1.0,
            beta=0.0,
        )

        # Turn 1: Q_t = mean([0.2]) = 0.2
        # Margin = 0.4 - 0.2 = 0.2, Shaped = 0.4 + 0.2 = 0.6
        assert shaped[0, 1] == pytest.approx(0.6)

        # Turn 2: Q_t = mean([0.2, 0.4]) = 0.3
        # Margin = 0.6 - 0.3 = 0.3, Shaped = 0.6 + 0.3 = 0.9
        assert shaped[0, 2] == pytest.approx(0.9)

        # Turn 3: Q_t = mean([0.2, 0.4, 0.6]) = 0.4
        # Margin = 0.8 - 0.4 = 0.4, Shaped = 0.8 + 0.4 = 1.2
        assert shaped[0, 3] == pytest.approx(1.2)

    def test_beta_parameter_outcome_reward(self):
        """Test beta parameter adds outcome reward based on final turn."""
        raw_rewards = np.array([[0.5, 0.6, 1.0]])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="none",  # No mode shaping, only beta
            alpha=0.0,
            beta=0.3,  # Add 0.3 * final_reward to all turns
        )

        # Final reward = 1.0
        # Each turn should have beta * final_reward added
        # Turn 0: 0.5 + 0.3 * 1.0 = 0.8
        # Turn 1: 0.6 + 0.3 * 1.0 = 0.9
        # Turn 2: 1.0 + 0.3 * 1.0 = 1.3
        assert shaped[0, 0] == pytest.approx(0.8)
        assert shaped[0, 1] == pytest.approx(0.9)
        assert shaped[0, 2] == pytest.approx(1.3)

    def test_alpha_parameter_shaping_weight(self):
        """Test alpha parameter controls shaping strength."""
        raw_rewards = np.array([[0.5, 1.0]])

        # No shaping (alpha=0)
        shaped_no_alpha = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="margin",
            alpha=0.0,
            beta=0.0,
        )
        assert np.allclose(shaped_no_alpha, raw_rewards)

        # Full shaping (alpha=1)
        shaped_full_alpha = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="margin",
            alpha=1.0,
            beta=0.0,
        )

        # Turn 1: Q_t = 0.5, R_t = 1.0, margin = 0.5
        # Shaped = 1.0 + 1.0 * 0.5 = 1.5
        assert shaped_full_alpha[0, 1] == pytest.approx(1.5)

        # Half shaping (alpha=0.5)
        shaped_half_alpha = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="margin",
            alpha=0.5,
            beta=0.0,
        )

        # Shaped = 1.0 + 0.5 * 0.5 = 1.25
        assert shaped_half_alpha[0, 1] == pytest.approx(1.25)

    def test_none_mode_returns_raw_rewards(self):
        """Test that 'none' mode returns raw rewards unchanged."""
        raw_rewards = np.array([
            [0.5, 0.8, 1.0],
            [1.0, 0.5, 0.0],
        ])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="none",
            alpha=1.0,
            beta=0.0,
        )

        assert np.allclose(shaped, raw_rewards)

    def test_single_turn_trajectory(self):
        """Test reward shaping with single turn (no shaping possible)."""
        raw_rewards = np.array([[1.0]])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="margin",
            alpha=0.5,
            beta=0.0,
        )

        # Single turn - no shaping, should return raw rewards
        assert shaped[0, 0] == 1.0

    def test_edge_case_all_zeros(self):
        """Test reward shaping with all zero rewards."""
        raw_rewards = np.array([[0.0, 0.0, 0.0]])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="margin",
            alpha=0.5,
            beta=0.0,
        )

        # All zeros - Q_t = 0, R_t = 0, margin = 0
        # Shaped should still be all zeros
        assert np.allclose(shaped, raw_rewards)

    def test_edge_case_all_ones(self):
        """Test reward shaping with all one rewards."""
        raw_rewards = np.array([[1.0, 1.0, 1.0]])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="quality",
            alpha=0.5,
            beta=0.0,
        )

        # Turn 0: 1.0
        assert shaped[0, 0] == 1.0

        # Turn 1+: Q_t = 1.0, R_t = 1.0
        # Quality = 1.0 * 1.0 - 0.0 * 0.0 = 1.0
        # Shaped = 1.0 + 0.5 * 1.0 = 1.5
        assert shaped[0, 1] == pytest.approx(1.5)
        assert shaped[0, 2] == pytest.approx(1.5)

    def test_multiple_agents_independent(self):
        """Test that shaping is applied independently per agent."""
        raw_rewards = np.array([
            [0.5, 0.8],
            [0.9, 0.3],
        ])

        shaped = apply_reward_shaping(
            raw_rewards=raw_rewards,
            mode="margin",
            alpha=1.0,
            beta=0.0,
        )

        # Agent 0, Turn 1: Q_t = 0.5, R_t = 0.8, margin = 0.3
        # Shaped = 0.8 + 0.3 = 1.1
        assert shaped[0, 1] == pytest.approx(1.1)

        # Agent 1, Turn 1: Q_t = 0.9, R_t = 0.3, margin = -0.6
        # Shaped = 0.3 + (-0.6) = -0.3
        assert shaped[1, 1] == pytest.approx(-0.3)

    def test_apply_reward_shaping_batch(self):
        """Test batch reward shaping."""
        trajectories = [
            np.array([[0.5, 0.8]]),
            np.array([[0.9, 0.3]]),
        ]

        config = RewardShapingConfig(
            mode=RewardShapingMode.MARGIN,
            alpha=1.0,
            beta=0.0,
        )

        shaped_batch = apply_reward_shaping_batch(trajectories, config)

        assert len(shaped_batch) == 2

        # First trajectory
        assert shaped_batch[0][0, 1] == pytest.approx(1.1)  # 0.8 + (0.8 - 0.5)

        # Second trajectory
        assert shaped_batch[1][0, 1] == pytest.approx(-0.3)  # 0.3 + (0.3 - 0.9)

    def test_batch_with_variable_length_trajectories(self):
        """Test batch shaping with different trajectory lengths."""
        trajectories = [
            np.array([[0.5, 0.8, 1.0]]),  # 3 turns
            np.array([[0.9, 0.3]]),  # 2 turns
            np.array([[0.7]]),  # 1 turn
        ]

        config = RewardShapingConfig(
            mode=RewardShapingMode.QUALITY,
            alpha=0.5,
            beta=0.0,
        )

        shaped_batch = apply_reward_shaping_batch(trajectories, config)

        assert len(shaped_batch) == 3
        assert shaped_batch[0].shape == (1, 3)
        assert shaped_batch[1].shape == (1, 2)
        assert shaped_batch[2].shape == (1, 1)

    def test_config_enum_usage(self):
        """Test RewardShapingConfig with enum values."""
        # Quality mode
        config_quality = RewardShapingConfig(
            mode=RewardShapingMode.QUALITY,
            alpha=0.5,
            beta=0.5,
        )
        assert config_quality.mode == RewardShapingMode.QUALITY
        assert config_quality.alpha == 0.5
        assert config_quality.beta == 0.5

        # Margin mode
        config_margin = RewardShapingConfig(
            mode=RewardShapingMode.MARGIN,
        )
        assert config_margin.mode == RewardShapingMode.MARGIN

        # None mode
        config_none = RewardShapingConfig(
            mode=RewardShapingMode.NONE,
        )
        assert config_none.mode == RewardShapingMode.NONE
