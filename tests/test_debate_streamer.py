"""Tests for DebateMetricStreamer.

Tests that the custom FlinkActorOutputStreamer correctly computes debate metrics
from actor outputs and merges them into the metrics dict.
"""

from __future__ import annotations

from typing import Any
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
    METRIC_REWARD_VERIFIER,
)


class MockActorOutputItem:
    """Mock ActorOutputWithMetadata for testing."""

    def __init__(self, reward: float, role_label: str = "solver", env_name: str = "math_debate"):
        self.data = {"rewards": np.array(reward)}
        self.metadata = {
            "env_name": np.array(env_name),
            f"{env_name}/reward_metrics": {"role_label": role_label},
        }


class MockUpstreamStreamer:
    """Mock upstream FlinkActorOutputStreamer."""

    def __init__(self, items: list[MockActorOutputItem], upstream_metrics: dict[str, float]):
        self._items = items
        self._upstream_metrics = upstream_metrics

    def get(self) -> tuple[list[MockActorOutputItem], dict[str, float]]:
        return self._items, self._upstream_metrics

    def flush_and_discard_ready_items(self):
        pass


def test_debate_streamer_computes_per_role_rewards():
    """Test that DebateMetricStreamer computes per-role rewards correctly."""
    # Create mock items with different roles
    items = [
        MockActorOutputItem(reward=0.5, role_label="solver"),
        MockActorOutputItem(reward=1.0, role_label="solver"),
        MockActorOutputItem(reward=0.0, role_label="verifier"),
        MockActorOutputItem(reward=0.8, role_label="verifier"),
    ]

    upstream = MockUpstreamStreamer(items, {"upstream_metric": 42.0})
    config = DebateMetricStreamerConfig(n_rollouts_per_prompt=4)
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    result_items, result_metrics = streamer.get()

    # Check items pass through unchanged
    assert result_items == items

    # Check per-role rewards
    assert METRIC_REWARD_SOLVER in result_metrics
    assert METRIC_REWARD_VERIFIER in result_metrics
    assert result_metrics[METRIC_REWARD_SOLVER] == pytest.approx(0.75)  # (0.5 + 1.0) / 2
    assert result_metrics[METRIC_REWARD_VERIFIER] == pytest.approx(0.4)  # (0.0 + 0.8) / 2

    # Check upstream metrics preserved
    assert result_metrics["upstream_metric"] == 42.0


def test_debate_streamer_computes_zero_advantage_metrics():
    """Test that zero-advantage detection metrics are computed correctly."""
    # 2 prompts: first has all-same rewards (zero std), second has varied rewards
    items = [
        # Prompt 1: all rewards == 1.0
        MockActorOutputItem(reward=1.0, role_label="solver"),
        MockActorOutputItem(reward=1.0, role_label="solver"),
        MockActorOutputItem(reward=1.0, role_label="solver"),
        MockActorOutputItem(reward=1.0, role_label="solver"),
        # Prompt 2: varied rewards
        MockActorOutputItem(reward=0.5, role_label="solver"),
        MockActorOutputItem(reward=0.8, role_label="solver"),
        MockActorOutputItem(reward=1.0, role_label="solver"),
        MockActorOutputItem(reward=0.2, role_label="solver"),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(n_rollouts_per_prompt=4)
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    _, result_metrics = streamer.get()

    # Check zero-advantage metrics
    assert METRIC_FRAC_ZERO_STD in result_metrics
    assert METRIC_MEAN_REWARD_STD in result_metrics

    # 1 out of 2 prompts has zero std
    assert result_metrics[METRIC_FRAC_ZERO_STD] == pytest.approx(0.5)

    # Mean reward std: (0.0 + std([0.5, 0.8, 1.0, 0.2])) / 2
    expected_mean_std = np.std([0.5, 0.8, 1.0, 0.2]) / 2
    assert result_metrics[METRIC_MEAN_REWARD_STD] == pytest.approx(expected_mean_std)


def test_debate_streamer_with_missing_role_labels():
    """Test that missing role labels default to 'solver'."""
    # Create items without role_label in metadata
    items = [
        MockActorOutputItem(reward=0.5, role_label="solver"),
        MockActorOutputItem(reward=1.0, role_label="solver"),
    ]
    # Remove role_label from second item's metadata
    items[1].metadata = {"env_name": np.array("math_debate")}

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(n_rollouts_per_prompt=2)
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    _, result_metrics = streamer.get()

    # Should compute solver reward using both items (default to solver)
    assert METRIC_REWARD_SOLVER in result_metrics
    assert result_metrics[METRIC_REWARD_SOLVER] == pytest.approx(0.75)


def test_debate_streamer_upstream_metrics_take_precedence():
    """Test that upstream metrics take precedence on key conflict."""
    items = [
        MockActorOutputItem(reward=0.5, role_label="solver"),
        MockActorOutputItem(reward=1.0, role_label="solver"),
    ]

    # Upstream has a conflicting metric
    upstream = MockUpstreamStreamer(items, {METRIC_REWARD_SOLVER: 999.0})
    config = DebateMetricStreamerConfig(n_rollouts_per_prompt=2)
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    _, result_metrics = streamer.get()

    # Upstream value should win
    assert result_metrics[METRIC_REWARD_SOLVER] == 999.0


def test_debate_streamer_error_handling():
    """Test that errors in metric computation return upstream metrics unchanged."""
    # Create items with malformed data that will raise an error
    class BadItem:
        @property
        def data(self):
            raise AttributeError("Simulated error")

    items = [BadItem()]

    upstream = MockUpstreamStreamer(items, {"upstream_metric": 42.0})
    config = DebateMetricStreamerConfig(n_rollouts_per_prompt=2)
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    result_items, result_metrics = streamer.get()

    # Should return items and upstream metrics unchanged
    assert result_items == items
    assert result_metrics == {"upstream_metric": 42.0}


def test_debate_streamer_flush():
    """Test that flush_and_discard_ready_items calls upstream."""
    upstream = MagicMock()
    config = DebateMetricStreamerConfig(n_rollouts_per_prompt=4)
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    streamer.flush_and_discard_ready_items()

    # Verify upstream.flush_and_discard_ready_items() was called
    upstream.flush_and_discard_ready_items.assert_called_once()


def test_debate_streamer_config_create_streamer():
    """Test that DebateMetricStreamerConfig.create_streamer() works."""
    config = DebateMetricStreamerConfig(n_rollouts_per_prompt=8)
    upstream = MagicMock()
    fax_config = MagicMock()
    metrics_collector = MagicMock()

    streamer = config.create_streamer(upstream, fax_config, metrics_collector)

    assert isinstance(streamer, DebateMetricStreamer)
    assert streamer._config == config
    assert streamer._upstream == upstream
    assert streamer._metrics_collector == metrics_collector


def test_debate_streamer_with_multiple_roles():
    """Test per-role rewards with all three roles present."""
    items = [
        MockActorOutputItem(reward=0.5, role_label="solver"),
        MockActorOutputItem(reward=1.0, role_label="solver"),
        MockActorOutputItem(reward=0.0, role_label="verifier"),
        MockActorOutputItem(reward=0.8, role_label="verifier"),
        MockActorOutputItem(reward=0.6, role_label="judge"),
        MockActorOutputItem(reward=0.4, role_label="judge"),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(n_rollouts_per_prompt=6)
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    _, result_metrics = streamer.get()

    # Check all three role rewards
    from src.training.wandb_enrichment.metric_schema import METRIC_REWARD_JUDGE

    assert result_metrics[METRIC_REWARD_SOLVER] == pytest.approx(0.75)
    assert result_metrics[METRIC_REWARD_VERIFIER] == pytest.approx(0.4)
    assert result_metrics[METRIC_REWARD_JUDGE] == pytest.approx(0.5)


def test_rollout_table_called_at_interval():
    """Test that rollout table logging is called at configured intervals."""
    from unittest.mock import patch

    items = [
        MockActorOutputItem(reward=0.5, role_label="solver"),
        MockActorOutputItem(reward=1.0, role_label="solver"),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=2,
        log_rollout_table_every_n_gets=5  # Log every 5 get() calls
    )
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)

    with patch("src.training.wandb_enrichment.rollout_integration.log_debate_rollout_table") as mock_log:
        # Call get() 12 times
        for _ in range(12):
            streamer.get()

        # Should have been called at get() #5 and get() #10 (2 times total)
        assert mock_log.call_count == 2

        # Verify calls were at step 5 and 10
        call_steps = [call.kwargs["step"] for call in mock_log.call_args_list]
        assert call_steps == [5, 10]


def test_workspace_init_called_on_first_get():
    """Test that workspace initialization is called on first get() call."""
    from unittest.mock import patch

    items = [MockActorOutputItem(reward=0.5, role_label="solver")]
    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(n_rollouts_per_prompt=1)
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)

    with patch("src.training.wandb_enrichment.workspace_init.init_debate_workspace") as mock_init:
        # First get() should trigger workspace init
        streamer.get()
        assert mock_init.call_count == 1

        # Second get() should NOT trigger workspace init again
        streamer.get()
        assert mock_init.call_count == 1
