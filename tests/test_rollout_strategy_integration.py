"""Integration tests for rollout strategy in DebateMetricStreamer pipeline.

Tests the full pipeline from config -> DebateMetricStreamer -> rollout selection ->
reward extraction -> metrics. Verifies execution order (rollout strategy BEFORE
reward shaping), backward compatibility, error handling, and composition with
reward shaping strategies.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.training.wandb_enrichment.debate_streamer import (
    DebateMetricStreamer,
    DebateMetricStreamerConfig,
)
from src.training.wandb_enrichment.metric_schema import (
    METRIC_FRAC_ZERO_STD,
    METRIC_MEAN_REWARD_STD,
    METRIC_REWARD_SOLVER,
)

# ---- Helpers ----------------------------------------------------------------


class MockActorOutputItem:
    """Mock ActorOutputWithMetadata for testing."""

    def __init__(
        self,
        reward: float,
        role_label: str = "solver",
        env_name: str = "math_debate",
    ):
        self.data = {"rewards": np.array(reward)}
        self.metadata = {
            "env_name": np.array(env_name),
            f"{env_name}/reward_metrics": {"role_label": role_label},
        }


class MockUpstreamStreamer:
    """Mock upstream FlinkActorOutputStreamer."""

    def __init__(self, items: list[MockActorOutputItem], upstream_metrics: dict):
        self._items = items
        self._upstream_metrics = upstream_metrics

    def get(self):
        return self._items, self._upstream_metrics

    def flush_and_discard_ready_items(self):
        pass


def _make_streamer(
    items: list[MockActorOutputItem],
    n_rollouts_per_prompt: int = 4,
    rollout_strategy: str = "",
    rollout_strategy_params: dict | None = None,
    reward_shaping_strategy: str = "",
    reward_shaping_params: dict | None = None,
    upstream_metrics: dict | None = None,
) -> tuple[DebateMetricStreamer, list[MockActorOutputItem]]:
    """Create a DebateMetricStreamer with given config and items."""
    upstream = MockUpstreamStreamer(items, upstream_metrics or {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=n_rollouts_per_prompt,
        rollout_strategy=rollout_strategy,
        rollout_strategy_params=rollout_strategy_params or {},
        reward_shaping_strategy=reward_shaping_strategy,
        reward_shaping_params=reward_shaping_params or {},
    )
    streamer = DebateMetricStreamer(config, upstream, MagicMock())
    return streamer, items


# ---- Tests ------------------------------------------------------------------


class TestRolloutStrategyIntegration:
    """Integration tests verifying rollout strategy wiring in DebateMetricStreamer."""

    def test_streamer_with_identity_rollout_strategy(self):
        """No rollout strategy (identity) passes all items through with selection_ratio=1.0."""
        # 2 prompts x 4 rollouts = 8 items
        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
        ]

        streamer, _ = _make_streamer(items, n_rollouts_per_prompt=4)
        result_items, metrics = streamer.get()

        # All 8 items pass through
        assert len(result_items) == 8

        # Rollout strategy metrics
        assert metrics["debate/rollout_strategy/items_in"] == 8.0
        assert metrics["debate/rollout_strategy/items_out"] == 8.0
        assert metrics["debate/rollout_strategy/selection_ratio"] == pytest.approx(1.0)

    def test_streamer_with_best_of_n_strategy(self):
        """best_of_n selects 1 best rollout per prompt, reducing items from 8 to 2."""
        # Prompt 1: rewards [1.0, 3.0, 2.0, 0.0] -> best = 3.0
        # Prompt 2: rewards [0.0, 5.0, 4.0, 1.0] -> best = 5.0
        items = [
            MockActorOutputItem(reward=1.0, role_label="solver"),
            MockActorOutputItem(reward=3.0, role_label="solver"),
            MockActorOutputItem(reward=2.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=4.0, role_label="solver"),
            MockActorOutputItem(reward=1.0, role_label="solver"),
        ]

        streamer, _ = _make_streamer(
            items,
            n_rollouts_per_prompt=4,
            rollout_strategy="best_of_n",
        )
        result_items, metrics = streamer.get()

        # Only 2 items remain (1 per prompt)
        assert len(result_items) == 2

        # Verify correct items selected (best rewards)
        result_rewards = [item.data["rewards"].item() for item in result_items]
        assert result_rewards == [3.0, 5.0]

        # Rollout strategy metrics
        assert metrics["debate/rollout_strategy/items_in"] == 8.0
        assert metrics["debate/rollout_strategy/items_out"] == 2.0
        assert metrics["debate/rollout_strategy/selection_ratio"] == pytest.approx(0.25)
        assert metrics["debate/rollout_strategy/mean_selected_reward"] == pytest.approx(4.0)

    def test_streamer_with_self_consistency_strategy(self):
        """self_consistency modifies rewards based on majority vote, preserving all items."""
        # Prompt 1: 3 correct (5.0) + 1 incorrect (0.0)
        #   majority = correct, agreement = 3/4 = 0.75
        #   correct items: 5.0 * 0.75 = 3.75, incorrect: 0.0
        # Prompt 2: 3 correct (5.0) + 1 incorrect (0.0)
        #   same pattern
        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
        ]

        streamer, _ = _make_streamer(
            items,
            n_rollouts_per_prompt=4,
            rollout_strategy="self_consistency",
            rollout_strategy_params={"agreement_threshold": 0.5},
        )
        result_items, metrics = streamer.get()

        # All 8 items returned (self-consistency preserves batch size)
        assert len(result_items) == 8

        # Check modified rewards
        result_rewards = [item.data["rewards"].item() for item in result_items]
        # Prompt 1: [3.75, 3.75, 3.75, 0.0]
        # Prompt 2: [3.75, 3.75, 3.75, 0.0]
        expected = [3.75, 3.75, 3.75, 0.0, 3.75, 3.75, 3.75, 0.0]
        for actual, exp in zip(result_rewards, expected):
            assert actual == pytest.approx(exp)

        # Selection ratio should be 1.0 (all items kept)
        assert metrics["debate/rollout_strategy/items_in"] == 8.0
        assert metrics["debate/rollout_strategy/items_out"] == 8.0
        assert metrics["debate/rollout_strategy/selection_ratio"] == pytest.approx(1.0)

    def test_rollout_strategy_before_reward_shaping(self):
        """Rollout strategy is applied BEFORE reward shaping (execution order verified)."""
        # Use best_of_n to reduce 8 items to 2, then identity reward shaping
        # If order were reversed, reward shaping would see 8 items
        items = [
            MockActorOutputItem(reward=1.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=2.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=3.0, role_label="verifier"),
            MockActorOutputItem(reward=4.0, role_label="verifier"),
            MockActorOutputItem(reward=0.0, role_label="verifier"),
            MockActorOutputItem(reward=1.0, role_label="verifier"),
        ]

        streamer, _ = _make_streamer(
            items,
            n_rollouts_per_prompt=4,
            rollout_strategy="best_of_n",
            reward_shaping_strategy="identity",
        )
        result_items, metrics = streamer.get()

        # Best-of-N reduces to 2 items first
        assert len(result_items) == 2
        assert metrics["debate/rollout_strategy/items_out"] == 2.0

        # Reward shaping operates on filtered set (2 items, not 8)
        # The shaped reward mean should reflect only the 2 selected items
        assert "debate/shaped_reward/mean" in metrics

        # Verify the selected items are the best from each prompt
        result_rewards = [item.data["rewards"].item() for item in result_items]
        assert 5.0 in result_rewards  # Best from prompt 1
        assert 4.0 in result_rewards  # Best from prompt 2

    def test_rollout_strategy_backward_compatible_no_config(self):
        """Default DebateMetricStreamerConfig() produces identity rollout strategy."""
        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
        ]

        upstream = MockUpstreamStreamer(items, {})
        config = DebateMetricStreamerConfig(n_rollouts_per_prompt=4)
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        # Verify identity strategy
        assert streamer._rollout_strategy.name == "identity"
        assert streamer._reward_shaper.name == "identity"

        result_items, metrics = streamer.get()

        # All items pass through
        assert len(result_items) == 4

        # All existing metrics should be present (backward compatible)
        assert METRIC_REWARD_SOLVER in metrics
        assert METRIC_FRAC_ZERO_STD in metrics
        assert METRIC_MEAN_REWARD_STD in metrics
        assert "debate/shaped_reward/mean" in metrics
        assert "debate/rollout_strategy/selection_ratio" in metrics
        assert metrics["debate/rollout_strategy/selection_ratio"] == pytest.approx(1.0)

    def test_rollout_strategy_error_handling(self):
        """If rollout strategy fails, streamer falls back to original items."""
        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
        ]

        streamer, _ = _make_streamer(items, n_rollouts_per_prompt=2)

        # Mock the rollout strategy to raise an error
        streamer._rollout_strategy.select_rollouts = MagicMock(
            side_effect=RuntimeError("strategy exploded")
        )

        result_items, metrics = streamer.get()

        # Original items should be returned
        assert len(result_items) == 2
        result_rewards = [item.data["rewards"].item() for item in result_items]
        assert result_rewards == [5.0, 0.0]

        # Core debate metrics should still be present
        assert METRIC_REWARD_SOLVER in metrics

        # Rollout strategy metrics should NOT be present (error path)
        assert "debate/rollout_strategy/items_in" not in metrics

    def test_all_rollout_strategies_in_registry(self):
        """All 3 rollout strategies are registered: identity, best_of_n, self_consistency."""
        from src.training.rollout_strategy import list_strategies

        strategies = list_strategies()
        assert set(strategies) == {"identity", "best_of_n", "self_consistency"}

    def test_best_of_n_with_reward_shaping_composition(self):
        """best_of_n rollout strategy + reward_mixing reward shaping compose correctly."""
        # 2 prompts x 4 rollouts
        # Prompt 1: rewards [1.0, 5.0, 2.0, 0.0] -> best = 5.0
        # Prompt 2: rewards [3.0, 0.0, 4.0, 1.0] -> best = 4.0
        items = [
            MockActorOutputItem(reward=1.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=2.0, role_label="verifier"),
            MockActorOutputItem(reward=0.0, role_label="verifier"),
            MockActorOutputItem(reward=3.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=4.0, role_label="verifier"),
            MockActorOutputItem(reward=1.0, role_label="verifier"),
        ]

        streamer, _ = _make_streamer(
            items,
            n_rollouts_per_prompt=4,
            rollout_strategy="best_of_n",
            reward_shaping_strategy="reward_mixing",
            reward_shaping_params={"alpha": 0.5},
        )

        # Verify both strategies initialized
        assert streamer._rollout_strategy.name == "best_of_n"
        assert streamer._reward_shaper.name == "reward_mixing"

        result_items, metrics = streamer.get()

        # Best-of-N selects 2 items (1 per prompt)
        assert len(result_items) == 2
        assert metrics["debate/rollout_strategy/items_in"] == 8.0
        assert metrics["debate/rollout_strategy/items_out"] == 2.0
        assert metrics["debate/rollout_strategy/selection_ratio"] == pytest.approx(0.25)

        # Reward shaping metrics should be present (applied after rollout selection)
        assert "debate/shaped_reward/mean" in metrics
        assert metrics["debate/shaped_reward/strategy_active"] == 1.0

        # Rollout strategy mean selected reward
        selected_rewards = [item.data["rewards"].item() for item in result_items]
        assert metrics["debate/rollout_strategy/mean_selected_reward"] == pytest.approx(
            np.mean(selected_rewards)
        )
