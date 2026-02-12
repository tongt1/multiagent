"""Unit tests for SelfConsistencyStrategy rollout selection.

TDD RED phase: These tests define the expected behavior of the self-consistency
rollout strategy. Tests are written first, implementation follows.

Self-consistency (Wang et al., 2022) uses majority vote across N rollouts per prompt
to determine consensus correctness, then scales rewards by agreement fraction.

Key semantics:
- Classify each rollout as correct (reward > 0) or incorrect (reward == 0)
- Majority vote determines the consensus label for the prompt
- agreement_fraction = count_of_majority / total_rollouts_per_prompt
- Rollouts that agree with majority: reward = original_reward * agreement_fraction
- Rollouts that disagree with majority: reward = 0.0 (suppressed)
- When agreement_fraction < agreement_threshold: pass through original rewards unchanged
- Tie-break: favor correct (reward > 0)
"""

from __future__ import annotations

import numpy as np
import pytest


class MockItem:
    """Mock actor output item for testing rollout strategies."""

    def __init__(self, reward: float):
        self.data = {"rewards": np.array(reward)}
        self.metadata = {}


class TestSelfConsistencyStrategy:
    """Tests for the SelfConsistencyStrategy."""

    def test_self_consistency_unanimous_correct(self):
        """1 prompt x 4 rollouts, all correct (reward=5.0).

        Agreement fraction = 4/4 = 1.0.
        All rollouts agree with majority (correct).
        Reward = 5.0 * 1.0 = 5.0 for all.
        """
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = SelfConsistencyStrategy()
        items = [MockItem(reward=5.0) for _ in range(4)]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 4
        for item in result:
            assert item.data["rewards"].item() == pytest.approx(5.0)

    def test_self_consistency_unanimous_incorrect(self):
        """1 prompt x 4 rollouts, all incorrect (reward=0.0).

        All agree on incorrect. Majority = incorrect.
        Agreement fraction = 4/4 = 1.0.
        Incorrect rollouts stay 0.0 (0.0 * 1.0 = 0.0).
        """
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = SelfConsistencyStrategy()
        items = [MockItem(reward=0.0) for _ in range(4)]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 4
        for item in result:
            assert item.data["rewards"].item() == pytest.approx(0.0)

    def test_self_consistency_majority_correct(self):
        """1 prompt x 4 rollouts: 3 correct (5.0), 1 incorrect (0.0).

        Majority = correct (3/4 = 0.75 agreement).
        Correct rollouts: 5.0 * 0.75 = 3.75.
        Incorrect rollout: 0.0 (disagrees with majority, suppressed).
        """
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = SelfConsistencyStrategy()
        items = [
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            MockItem(reward=0.0),
        ]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 4
        # First 3 (correct, agree with majority): 5.0 * 0.75
        assert result[0].data["rewards"].item() == pytest.approx(3.75)
        assert result[1].data["rewards"].item() == pytest.approx(3.75)
        assert result[2].data["rewards"].item() == pytest.approx(3.75)
        # Last (incorrect, disagrees): 0.0
        assert result[3].data["rewards"].item() == pytest.approx(0.0)

    def test_self_consistency_majority_incorrect(self):
        """1 prompt x 4 rollouts: 1 correct (5.0), 3 incorrect (0.0).

        Majority = incorrect (3/4 = 0.75 agreement).
        Correct rollout: 0.0 (disagrees with majority, suppressed).
        Incorrect rollouts: 0.0 * 0.75 = 0.0 (agree with majority but reward is 0).
        """
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = SelfConsistencyStrategy()
        items = [
            MockItem(reward=5.0),
            MockItem(reward=0.0),
            MockItem(reward=0.0),
            MockItem(reward=0.0),
        ]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 4
        # First (correct, disagrees with majority): 0.0
        assert result[0].data["rewards"].item() == pytest.approx(0.0)
        # Last 3 (incorrect, agree with majority): 0.0 * 0.75 = 0.0
        assert result[1].data["rewards"].item() == pytest.approx(0.0)
        assert result[2].data["rewards"].item() == pytest.approx(0.0)
        assert result[3].data["rewards"].item() == pytest.approx(0.0)

    def test_self_consistency_split_vote(self):
        """1 prompt x 4 rollouts: 2 correct, 2 incorrect.

        Tie: agreement = 0.5. With default threshold 0.5, meets threshold.
        Tie-break: favor correct (reward > 0).
        Majority = correct. Agreement = 2/4 = 0.5.
        Correct rollouts: 5.0 * 0.5 = 2.5.
        Incorrect rollouts: 0.0 (disagree with majority).
        """
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = SelfConsistencyStrategy()
        items = [
            MockItem(reward=5.0),
            MockItem(reward=0.0),
            MockItem(reward=5.0),
            MockItem(reward=0.0),
        ]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 4
        assert result[0].data["rewards"].item() == pytest.approx(2.5)
        assert result[1].data["rewards"].item() == pytest.approx(0.0)
        assert result[2].data["rewards"].item() == pytest.approx(2.5)
        assert result[3].data["rewards"].item() == pytest.approx(0.0)

    def test_self_consistency_threshold_parameter(self):
        """1 prompt x 4 rollouts: 3 correct, 1 incorrect.

        With agreement_threshold=0.9, agreement = 0.75 < 0.9.
        Below threshold: all rollouts get original reward unchanged.
        """
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = SelfConsistencyStrategy(agreement_threshold=0.9)
        items = [
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            MockItem(reward=0.0),
        ]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 4
        # All unchanged -- below threshold
        assert result[0].data["rewards"].item() == pytest.approx(5.0)
        assert result[1].data["rewards"].item() == pytest.approx(5.0)
        assert result[2].data["rewards"].item() == pytest.approx(5.0)
        assert result[3].data["rewards"].item() == pytest.approx(0.0)

    def test_self_consistency_multiple_prompts(self):
        """2 prompts x 4 rollouts each.

        Prompt 1: 4 correct (unanimous, agreement = 1.0).
        Prompt 2: 3 correct, 1 incorrect (agreement = 0.75).
        Verify independent majority vote per prompt.
        """
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = SelfConsistencyStrategy()
        items = [
            # Prompt 1: all correct
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            # Prompt 2: 3 correct, 1 incorrect
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            MockItem(reward=0.0),
        ]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 8
        # Prompt 1: 5.0 * 1.0 = 5.0
        for i in range(4):
            assert result[i].data["rewards"].item() == pytest.approx(5.0)
        # Prompt 2: correct -> 5.0 * 0.75 = 3.75, incorrect -> 0.0
        assert result[4].data["rewards"].item() == pytest.approx(3.75)
        assert result[5].data["rewards"].item() == pytest.approx(3.75)
        assert result[6].data["rewards"].item() == pytest.approx(3.75)
        assert result[7].data["rewards"].item() == pytest.approx(0.0)

    def test_self_consistency_returns_all_items(self):
        """Unlike best-of-N, self-consistency returns ALL N*P items.

        Verify len(output) == len(input).
        """
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = SelfConsistencyStrategy()
        items = [MockItem(reward=float(i % 2) * 5.0) for i in range(12)]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == len(items) == 12

    def test_self_consistency_name(self):
        """SelfConsistencyStrategy().name == 'self_consistency'."""
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = SelfConsistencyStrategy()
        assert strategy.name == "self_consistency"

    def test_self_consistency_registry_creation(self):
        """create_strategy_from_config creates SelfConsistencyStrategy with correct threshold."""
        from src.training.rollout_strategy.registry import create_strategy_from_config
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = create_strategy_from_config(
            {
                "strategy_name": "self_consistency",
                "strategy_params": {"agreement_threshold": 0.6},
            }
        )

        assert isinstance(strategy, SelfConsistencyStrategy)
        assert strategy._agreement_threshold == pytest.approx(0.6)

    def test_self_consistency_incomplete_group(self):
        """If batch not evenly divisible, handle gracefully.

        Incomplete group items pass through unchanged.
        """
        from src.training.rollout_strategy.self_consistency import (
            SelfConsistencyStrategy,
        )

        strategy = SelfConsistencyStrategy()
        # 5 items with n_rollouts_per_prompt=4: 1 complete group + 1 leftover
        items = [
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            MockItem(reward=5.0),
            MockItem(reward=0.0),
            MockItem(reward=5.0),  # leftover
        ]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 5
        # First group processed (3 correct, 1 incorrect, agreement=0.75)
        assert result[0].data["rewards"].item() == pytest.approx(3.75)
        assert result[1].data["rewards"].item() == pytest.approx(3.75)
        assert result[2].data["rewards"].item() == pytest.approx(3.75)
        assert result[3].data["rewards"].item() == pytest.approx(0.0)
        # Leftover item passes through unchanged
        assert result[4].data["rewards"].item() == pytest.approx(5.0)
