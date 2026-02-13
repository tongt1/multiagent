"""Unit tests for BestOfNStrategy rollout selection.

TDD RED phase: These tests define the expected behavior of the best-of-N
rollout selection strategy. Tests are written first, then implementation follows.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest


class MockItem:
    """Mock actor output item for testing rollout strategies."""

    def __init__(self, reward: float, prompt_id: int = 0):
        self.data = {"rewards": np.array(reward)}
        self.metadata = {"prompt_id": prompt_id}


class TestBestOfNStrategy:
    """Tests for BestOfNStrategy rollout selection."""

    def test_best_of_n_selects_highest_reward(self):
        """2 prompts x 4 rollouts: returns 2 items, each the highest-reward rollout."""
        from src.training.rollout_strategy.best_of_n import BestOfNStrategy

        strategy = BestOfNStrategy()

        # Prompt 0: rewards [1.0, 3.0, 2.0, 0.5] -> best is index 1 (reward=3.0)
        # Prompt 1: rewards [0.0, 5.0, 4.0, 1.0] -> best is index 1 (reward=5.0)
        items = [
            MockItem(reward=1.0, prompt_id=0),
            MockItem(reward=3.0, prompt_id=0),
            MockItem(reward=2.0, prompt_id=0),
            MockItem(reward=0.5, prompt_id=0),
            MockItem(reward=0.0, prompt_id=1),
            MockItem(reward=5.0, prompt_id=1),
            MockItem(reward=4.0, prompt_id=1),
            MockItem(reward=1.0, prompt_id=1),
        ]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 2
        assert result[0].data["rewards"].item() == 3.0
        assert result[1].data["rewards"].item() == 5.0

    def test_best_of_n_single_prompt(self):
        """1 prompt x 8 rollouts: returns 1 item with highest reward."""
        from src.training.rollout_strategy.best_of_n import BestOfNStrategy

        strategy = BestOfNStrategy()

        rewards = [0.5, 1.0, 3.0, 2.0, 4.5, 0.0, 1.5, 2.5]
        items = [MockItem(reward=r, prompt_id=0) for r in rewards]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=8)

        assert len(result) == 1
        assert result[0].data["rewards"].item() == 4.5

    def test_best_of_n_all_same_rewards(self):
        """All rollouts have reward=1.0: returns 1 per prompt (first in group for determinism)."""
        from src.training.rollout_strategy.best_of_n import BestOfNStrategy

        strategy = BestOfNStrategy()

        items = [MockItem(reward=1.0, prompt_id=i // 4) for i in range(8)]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 2
        # np.argmax returns first occurrence for ties, so first item in each group
        assert result[0] is items[0]
        assert result[1] is items[4]

    def test_best_of_n_preserves_item_data(self):
        """The returned item is the actual item object (not a copy)."""
        from src.training.rollout_strategy.best_of_n import BestOfNStrategy

        strategy = BestOfNStrategy()

        items = [
            MockItem(reward=1.0, prompt_id=0),
            MockItem(reward=5.0, prompt_id=0),
            MockItem(reward=2.0, prompt_id=0),
            MockItem(reward=0.0, prompt_id=0),
        ]
        # Add extra data to verify preservation
        items[1].metadata["extra_field"] = "preserved"

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 1
        assert result[0] is items[1]  # Same object, not a copy
        assert result[0].metadata["extra_field"] == "preserved"

    def test_best_of_n_name(self):
        """BestOfNStrategy().name == 'best_of_n'."""
        from src.training.rollout_strategy.best_of_n import BestOfNStrategy

        strategy = BestOfNStrategy()
        assert strategy.name == "best_of_n"

    def test_best_of_n_registry_creation(self):
        """create_strategy_from_config({'strategy_name': 'best_of_n'}) creates BestOfNStrategy."""
        from src.training.rollout_strategy.best_of_n import BestOfNStrategy
        from src.training.rollout_strategy.registry import create_strategy_from_config

        strategy = create_strategy_from_config({"strategy_name": "best_of_n"})
        assert isinstance(strategy, BestOfNStrategy)

    def test_best_of_n_binary_rewards(self):
        """Rewards are 0.0 and 5.0 (binary SymPy format). Selects 5.0 rollouts correctly."""
        from src.training.rollout_strategy.best_of_n import BestOfNStrategy

        strategy = BestOfNStrategy()

        # 3 prompts x 4 rollouts, binary rewards
        items = [
            # Prompt 0: [0, 0, 5, 0] -> selects index 2
            MockItem(reward=0.0, prompt_id=0),
            MockItem(reward=0.0, prompt_id=0),
            MockItem(reward=5.0, prompt_id=0),
            MockItem(reward=0.0, prompt_id=0),
            # Prompt 1: [5, 0, 0, 0] -> selects index 0
            MockItem(reward=5.0, prompt_id=1),
            MockItem(reward=0.0, prompt_id=1),
            MockItem(reward=0.0, prompt_id=1),
            MockItem(reward=0.0, prompt_id=1),
            # Prompt 2: [0, 0, 0, 5] -> selects index 3
            MockItem(reward=0.0, prompt_id=2),
            MockItem(reward=0.0, prompt_id=2),
            MockItem(reward=0.0, prompt_id=2),
            MockItem(reward=5.0, prompt_id=2),
        ]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 3
        assert result[0].data["rewards"].item() == 5.0
        assert result[1].data["rewards"].item() == 5.0
        assert result[2].data["rewards"].item() == 5.0

    def test_best_of_n_shaped_rewards(self):
        """Rewards have continuous values (e.g., from potential-based shaping). Still selects highest."""
        from src.training.rollout_strategy.best_of_n import BestOfNStrategy

        strategy = BestOfNStrategy()

        # 2 prompts x 4 rollouts, continuous rewards
        items = [
            MockItem(reward=0.3, prompt_id=0),
            MockItem(reward=2.7, prompt_id=0),
            MockItem(reward=1.5, prompt_id=0),
            MockItem(reward=4.1, prompt_id=0),  # highest in group 0
            MockItem(reward=3.8, prompt_id=1),  # highest in group 1
            MockItem(reward=1.2, prompt_id=1),
            MockItem(reward=0.9, prompt_id=1),
            MockItem(reward=2.0, prompt_id=1),
        ]

        result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        assert len(result) == 2
        assert result[0].data["rewards"].item() == pytest.approx(4.1)
        assert result[1].data["rewards"].item() == pytest.approx(3.8)

    def test_best_of_n_incomplete_group_warning(self, caplog):
        """If batch_size not divisible by n_rollouts_per_prompt, truncates with warning."""
        from src.training.rollout_strategy.best_of_n import BestOfNStrategy

        strategy = BestOfNStrategy()

        # 10 items with n_rollouts_per_prompt=4 -> 2 complete groups, 2 leftover
        items = [MockItem(reward=float(i), prompt_id=i // 4) for i in range(10)]

        with caplog.at_level(logging.WARNING):
            result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

        # Should process 2 complete groups (8 items), ignoring 2 leftover
        assert len(result) == 2
        # Group 0: rewards [0, 1, 2, 3] -> selects index 3 (reward=3.0)
        assert result[0].data["rewards"].item() == 3.0
        # Group 1: rewards [4, 5, 6, 7] -> selects index 3 (reward=7.0)
        assert result[1].data["rewards"].item() == 7.0
        # Check warning was logged
        assert any("truncat" in record.message.lower() or "incomplete" in record.message.lower()
                    for record in caplog.records)
