"""Unit tests for potential-based reward shaping strategy.

TDD RED phase: These tests define the expected behavior of the
PotentialBasedShaper strategy. Tests cover zero/constant potentials,
built-in potential functions, gamma scaling, custom callables,
output format, and registry integration.

Reference: Ng, Harada, Russell 1999 -- potential-based shaping preserves
optimal policy guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestPotentialZeroAndConstant:
    """Tests that trivial potential functions behave as identity."""

    def test_potential_zero_is_identity(self):
        """Phi=0 everywhere: r' = r + gamma*0 - 0 = r. Shaped reward equals original."""
        from src.training.reward_shaping.potential_shaping import PotentialBasedShaper

        shaper = PotentialBasedShaper(gamma=0.99, potential_type="zero")
        rewards = np.array([0.0, 5.0, 0.0, 5.0])
        metadata = [
            {"state_start": {}, "state_end": {}} for _ in range(4)
        ]

        result = shaper.shape_rewards(rewards, None, metadata)

        np.testing.assert_array_almost_equal(result, rewards)

    def test_potential_constant_is_identity(self):
        """Phi=c for all states: r' = r + gamma*c - c. For gamma=1.0, equals r."""
        from src.training.reward_shaping.potential_shaping import PotentialBasedShaper

        # With gamma=1.0, constant potential cancels: r + 1.0*c - c = r
        def const_phi(state_info: dict) -> float:
            return 42.0

        shaper = PotentialBasedShaper(gamma=1.0, potential_fn=const_phi)
        rewards = np.array([0.0, 5.0, 3.0, 1.0])
        metadata = [
            {"state_start": {"x": 1}, "state_end": {"x": 2}} for _ in range(4)
        ]

        result = shaper.shape_rewards(rewards, None, metadata)

        np.testing.assert_array_almost_equal(result, rewards)


class TestPotentialBuiltIns:
    """Tests for built-in potential functions."""

    def test_potential_debate_length_potential(self):
        """Built-in 'debate_length' potential: Phi(s) = -0.1 * num_turns.
        Longer debates get penalized (encourages efficient resolution)."""
        from src.training.reward_shaping.potential_shaping import PotentialBasedShaper

        shaper = PotentialBasedShaper(
            gamma=1.0, potential_type="debate_length", penalty=0.1
        )
        rewards = np.array([5.0, 5.0])
        metadata = [
            {
                "state_start": {"num_turns": 0},
                "state_end": {"num_turns": 3},
            },
            {
                "state_start": {"num_turns": 0},
                "state_end": {"num_turns": 5},
            },
        ]

        result = shaper.shape_rewards(rewards, None, metadata)

        # r' = r + gamma * Phi(s') - Phi(s)
        # Rollout 0: 5.0 + 1.0 * (-0.1*3) - (-0.1*0) = 5.0 - 0.3 = 4.7
        # Rollout 1: 5.0 + 1.0 * (-0.1*5) - (-0.1*0) = 5.0 - 0.5 = 4.5
        np.testing.assert_array_almost_equal(result, np.array([4.7, 4.5]))

    def test_potential_correctness_progress_potential(self):
        """Built-in 'correctness_progress' potential: Phi(s) based on
        intermediate correctness progress."""
        from src.training.reward_shaping.potential_shaping import PotentialBasedShaper

        shaper = PotentialBasedShaper(
            gamma=1.0, potential_type="correctness_progress"
        )
        rewards = np.array([5.0])
        metadata = [
            {
                "state_start": {
                    "intermediate_correct_count": 0,
                    "total_turns": 4,
                },
                "state_end": {
                    "intermediate_correct_count": 2,
                    "total_turns": 4,
                },
            },
        ]

        result = shaper.shape_rewards(rewards, None, metadata)

        # Phi(start) = 0.5 * 0/4 = 0.0
        # Phi(end) = 0.5 * 2/4 = 0.25
        # r' = 5.0 + 1.0 * 0.25 - 0.0 = 5.25
        np.testing.assert_array_almost_equal(result, np.array([5.25]))


class TestPotentialGammaAndCustom:
    """Tests for gamma scaling and custom potential functions."""

    def test_potential_gamma_scaling(self):
        """Verify gamma correctly scales the potential difference."""
        from src.training.reward_shaping.potential_shaping import PotentialBasedShaper

        shaper = PotentialBasedShaper(
            gamma=0.5, potential_type="debate_length", penalty=1.0
        )
        rewards = np.array([5.0])
        metadata = [
            {
                "state_start": {"num_turns": 0},
                "state_end": {"num_turns": 2},
            },
        ]

        result = shaper.shape_rewards(rewards, None, metadata)

        # Phi(start) = -1.0 * 0 = 0.0
        # Phi(end) = -1.0 * 2 = -2.0
        # r' = 5.0 + 0.5 * (-2.0) - 0.0 = 5.0 - 1.0 = 4.0
        np.testing.assert_array_almost_equal(result, np.array([4.0]))

    def test_potential_custom_phi_function(self):
        """User can pass a callable Phi(state_info) -> float as potential function."""
        from src.training.reward_shaping.potential_shaping import PotentialBasedShaper

        def custom_phi(state_info: dict) -> float:
            return state_info.get("quality_score", 0.0) * 10.0

        shaper = PotentialBasedShaper(gamma=1.0, potential_fn=custom_phi)
        rewards = np.array([5.0])
        metadata = [
            {
                "state_start": {"quality_score": 0.2},
                "state_end": {"quality_score": 0.8},
            },
        ]

        result = shaper.shape_rewards(rewards, None, metadata)

        # Phi(start) = 0.2 * 10 = 2.0
        # Phi(end) = 0.8 * 10 = 8.0
        # r' = 5.0 + 1.0 * 8.0 - 2.0 = 11.0
        np.testing.assert_array_almost_equal(result, np.array([11.0]))


class TestPotentialOutput:
    """Tests for output format."""

    def test_potential_returns_flat_array(self):
        """Output is np.ndarray shape (B,), not per-role dict
        (potential shaping applies uniformly to team reward)."""
        from src.training.reward_shaping.potential_shaping import PotentialBasedShaper

        shaper = PotentialBasedShaper(gamma=0.99, potential_type="zero")
        rewards = np.array([0.0, 5.0, 0.0, 5.0, 3.0])

        result = shaper.shape_rewards(rewards, None, None)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)


class TestPotentialRegistry:
    """Tests for registry integration and config-based creation."""

    def test_potential_registered(self):
        """get_strategy('potential_based') returns PotentialBasedShaper class."""
        from src.training.reward_shaping.potential_shaping import PotentialBasedShaper
        from src.training.reward_shaping.registry import get_strategy

        cls = get_strategy("potential_based")
        assert cls is PotentialBasedShaper

    def test_potential_from_config(self):
        """create_strategy_from_config with potential_based strategy works."""
        from src.training.reward_shaping.registry import create_strategy_from_config

        strategy = create_strategy_from_config(
            {
                "strategy_name": "potential_based",
                "strategy_params": {
                    "gamma": 0.99,
                    "potential_type": "debate_length",
                },
            }
        )
        assert strategy.name == "potential_based"
