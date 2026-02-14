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

    def __init__(
        self,
        reward: float,
        role_label: str = "solver",
        env_name: str = "math_debate",
        trajectory: str | None = None,
        unique_sample_id: str | None = None,
    ):
        self.data = {"rewards": np.array(reward)}
        self.metadata = {
            "env_name": np.array(env_name),
            f"{env_name}/reward_metrics": {"role_label": role_label},
        }
        if trajectory is not None:
            self.metadata["trajectory"] = np.array(trajectory)
        if unique_sample_id is not None:
            self.metadata["unique_sample_id"] = np.array(unique_sample_id)


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


def test_debug_data_written_at_interval():
    """Test that write_debate_debug_data is called at configured intervals when debug_data_output_dir is set."""
    from unittest.mock import patch

    items = [
        MockActorOutputItem(
            reward=0.5, role_label="solver",
            trajectory="Turn 1 text", unique_sample_id="math:1",
        ),
        MockActorOutputItem(
            reward=1.0, role_label="solver",
            trajectory="Turn 2 text", unique_sample_id="math:2",
        ),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=2,
        log_rollout_table_every_n_gets=3,
        debug_data_output_dir="/tmp/test_debug",
    )
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)

    with patch("src.training.wandb_enrichment.workspace_init.init_debate_workspace"), \
         patch("src.training.wandb_enrichment.rollout_integration.log_debate_rollout_table"), \
         patch("src.training.wandb_enrichment.rollout_integration.write_debate_debug_data") as mock_write:
        # Call get() 7 times
        for _ in range(7):
            streamer.get()

        # Should have been called at get #3 and #6 (2 times total)
        assert mock_write.call_count == 2

        # Verify call args
        first_call = mock_write.call_args_list[0]
        assert first_call.kwargs["items"] == items
        assert first_call.kwargs["step"] == 3
        assert first_call.kwargs["output_dir"] == "/tmp/test_debug"

        second_call = mock_write.call_args_list[1]
        assert second_call.kwargs["step"] == 6
        assert second_call.kwargs["output_dir"] == "/tmp/test_debug"


def test_debug_data_not_written_when_dir_empty():
    """Test that write_debate_debug_data is NOT called when debug_data_output_dir is empty (default)."""
    from unittest.mock import patch

    items = [
        MockActorOutputItem(
            reward=0.5, role_label="solver",
            trajectory="Turn 1 text", unique_sample_id="math:1",
        ),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=1,
        log_rollout_table_every_n_gets=1,
        debug_data_output_dir="",  # Empty = disabled (default)
    )
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)

    with patch("src.training.wandb_enrichment.workspace_init.init_debate_workspace"), \
         patch("src.training.wandb_enrichment.rollout_integration.log_debate_rollout_table"), \
         patch("src.training.wandb_enrichment.rollout_integration.write_debate_debug_data") as mock_write:
        # Call get() 5 times
        for _ in range(5):
            streamer.get()

        # Should never be called since debug_data_output_dir is empty
        assert mock_write.call_count == 0


def test_debug_data_error_does_not_crash():
    """Test that errors in write_debate_debug_data do not propagate to caller."""
    from unittest.mock import patch

    items = [
        MockActorOutputItem(
            reward=0.5, role_label="solver",
            trajectory="Turn 1 text", unique_sample_id="math:1",
        ),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=1,
        log_rollout_table_every_n_gets=1,
        debug_data_output_dir="/tmp/test",
    )
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)

    with patch("src.training.wandb_enrichment.workspace_init.init_debate_workspace"), \
         patch("src.training.wandb_enrichment.rollout_integration.log_debate_rollout_table"), \
         patch("src.training.wandb_enrichment.rollout_integration.write_debate_debug_data",
               side_effect=RuntimeError("disk full")) as mock_write:
        # Call get() once -- should NOT raise
        result_items, result_metrics = streamer.get()

        # Should return items and metrics successfully
        assert result_items == items
        assert isinstance(result_metrics, dict)

        # Verify the function WAS called (attempted) but error was caught
        assert mock_write.call_count == 1


# ─── Shaped reward write-back tests ───────────────────────────────────────────


def test_shaped_rewards_written_to_items_identity():
    """Identity strategy: item.data['rewards'] values remain unchanged after get()."""
    items = [
        MockActorOutputItem(reward=5.0, role_label="solver"),
        MockActorOutputItem(reward=0.0, role_label="solver"),
        MockActorOutputItem(reward=5.0, role_label="solver"),
        MockActorOutputItem(reward=0.0, role_label="solver"),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(n_rollouts_per_prompt=4)
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    assert streamer._reward_shaper.name == "identity"

    result_items, _ = streamer.get()

    # Identity is passthrough: rewards should be unchanged
    assert np.isclose(result_items[0].data["rewards"].item(), 5.0)
    assert np.isclose(result_items[1].data["rewards"].item(), 0.0)
    assert np.isclose(result_items[2].data["rewards"].item(), 5.0)
    assert np.isclose(result_items[3].data["rewards"].item(), 0.0)


def test_shaped_rewards_written_to_items_global_strategy():
    """Global ndarray strategy (potential_based): write-back path works."""
    items = [
        MockActorOutputItem(reward=5.0, role_label="solver"),
        MockActorOutputItem(reward=0.0, role_label="solver"),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=2,
        reward_shaping_strategy="potential_based",
        reward_shaping_params={"gamma": 0.99, "potential_type": "zero"},
    )
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    result_items, _ = streamer.get()

    # With zero potential, shaped equals original (gamma*0 - 0 = 0)
    assert np.isclose(result_items[0].data["rewards"].item(), 5.0)
    assert np.isclose(result_items[1].data["rewards"].item(), 0.0)


def test_shaped_rewards_written_to_items_per_role_strategy():
    """Per-role dict strategy (coma_advantage): items get role-specific shaped values."""
    items = [
        MockActorOutputItem(reward=5.0, role_label="solver"),
        MockActorOutputItem(reward=0.0, role_label="solver"),
        MockActorOutputItem(reward=5.0, role_label="verifier"),
        MockActorOutputItem(reward=0.0, role_label="verifier"),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=4,
        reward_shaping_strategy="coma_advantage",
        reward_shaping_params={"n_rollouts_per_prompt": 4},
    )
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    result_items, _ = streamer.get()

    # COMA with 4 rollouts per prompt, mean=2.5, advantages=[2.5, -2.5, 2.5, -2.5]
    # Solver items (indices 0,1) get solver advantage values
    assert result_items[0].data["rewards"].item() == pytest.approx(2.5)
    assert result_items[1].data["rewards"].item() == pytest.approx(-2.5)
    # Verifier items (indices 2,3) get verifier advantage values
    assert result_items[2].data["rewards"].item() == pytest.approx(2.5)
    assert result_items[3].data["rewards"].item() == pytest.approx(-2.5)


def test_judge_items_get_zero_reward():
    """Judge items get zero reward regardless of strategy."""
    items = [
        MockActorOutputItem(reward=5.0, role_label="solver"),
        MockActorOutputItem(reward=5.0, role_label="verifier"),
        MockActorOutputItem(reward=5.0, role_label="judge"),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=3,
        reward_shaping_strategy="difference_rewards",
    )
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    result_items, _ = streamer.get()

    # Judge item should be zero regardless of strategy
    assert result_items[2].data["rewards"].item() == pytest.approx(0.0)
    # Solver and verifier: difference_rewards without metadata falls back to raw
    assert result_items[0].data["rewards"].item() == pytest.approx(5.0)
    assert result_items[1].data["rewards"].item() == pytest.approx(5.0)


def test_unshaped_metrics_unchanged_after_mutation():
    """Unshaped debate metrics use original raw rewards, not mutated values."""
    items = [
        MockActorOutputItem(reward=5.0, role_label="solver"),
        MockActorOutputItem(reward=0.0, role_label="solver"),
        MockActorOutputItem(reward=5.0, role_label="solver"),
        MockActorOutputItem(reward=0.0, role_label="solver"),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=4,
        reward_shaping_strategy="coma_advantage",
        reward_shaping_params={"n_rollouts_per_prompt": 4},
    )
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    _, result_metrics = streamer.get()

    # Unshaped debate/reward/solver should be mean of ORIGINAL raw rewards: (5+0+5+0)/4 = 2.5
    assert result_metrics[METRIC_REWARD_SOLVER] == pytest.approx(2.5)

    # Shaped reward metrics should reflect COMA advantage: mean of [2.5, -2.5, 2.5, -2.5] = 0.0
    # (COMA advantages are zero-mean by construction)
    assert result_metrics["debate/shaped_reward/mean"] == pytest.approx(0.0)


def test_reward_mutation_fallback_on_missing_role():
    """Items with unknown role fall back to raw reward value."""
    items = [
        MockActorOutputItem(reward=5.0, role_label="solver"),
        MockActorOutputItem(reward=3.0, role_label="solver"),
    ]
    # Override second item's role to a custom unknown role
    items[1].metadata["math_debate/reward_metrics"]["role_label"] = "custom_role"

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=2,
        reward_shaping_strategy="difference_rewards",
    )
    metrics_collector = MagicMock()

    streamer = DebateMetricStreamer(config, upstream, metrics_collector)
    result_items, _ = streamer.get()

    # Solver item: difference_rewards without metadata falls back to raw
    assert result_items[0].data["rewards"].item() == pytest.approx(5.0)
    # Custom role: not in shaped dict, should fall back to raw reward (3.0)
    assert result_items[1].data["rewards"].item() == pytest.approx(3.0)
