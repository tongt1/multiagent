"""Integration tests for reward shaping end-to-end flow.

Tests the full pipeline from config dict -> create_strategy_from_config ->
shape_rewards -> correct output for each strategy, and verifies the
DebateMetricStreamer integration produces shaped reward metrics.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


# ─── Helpers ───────────────────────────────────────────────────────────────────


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


# ─── Config-to-strategy creation tests ────────────────────────────────────────


class TestConfigToStrategyCreation:
    """Test that all 5 strategies can be created from config dicts."""

    def test_identity_from_config(self):
        """identity strategy created from config dict."""
        from src.training.reward_shaping import create_strategy_from_config
        from src.training.reward_shaping.identity import IdentityRewardShaper

        strategy = create_strategy_from_config({"strategy_name": "identity"})
        assert isinstance(strategy, IdentityRewardShaper)
        assert strategy.name == "identity"

    def test_difference_rewards_from_config(self):
        """difference_rewards strategy created from config dict."""
        from src.training.reward_shaping import create_strategy_from_config
        from src.training.reward_shaping.difference_rewards import DifferenceRewardShaper

        strategy = create_strategy_from_config({"strategy_name": "difference_rewards"})
        assert isinstance(strategy, DifferenceRewardShaper)
        assert strategy.name == "difference_rewards"

    def test_reward_mixing_from_config(self):
        """reward_mixing strategy created from config dict with alpha param."""
        from src.training.reward_shaping import create_strategy_from_config
        from src.training.reward_shaping.reward_mixing import RewardMixingShaper

        strategy = create_strategy_from_config(
            {"strategy_name": "reward_mixing", "strategy_params": {"alpha": 0.7}}
        )
        assert isinstance(strategy, RewardMixingShaper)
        assert strategy.name == "reward_mixing"
        assert strategy.alpha == 0.7

    def test_coma_advantage_from_config(self):
        """coma_advantage strategy created from config dict."""
        from src.training.reward_shaping import create_strategy_from_config
        from src.training.reward_shaping.coma_advantage import COMAAdvantageShaper

        strategy = create_strategy_from_config(
            {"strategy_name": "coma_advantage", "strategy_params": {"n_rollouts_per_prompt": 4}}
        )
        assert isinstance(strategy, COMAAdvantageShaper)
        assert strategy.name == "coma_advantage"
        assert strategy.n_rollouts_per_prompt == 4

    def test_potential_based_from_config(self):
        """potential_based strategy created from config dict."""
        from src.training.reward_shaping import create_strategy_from_config
        from src.training.reward_shaping.potential_shaping import PotentialBasedShaper

        strategy = create_strategy_from_config(
            {
                "strategy_name": "potential_based",
                "strategy_params": {"gamma": 0.95, "potential_type": "debate_length"},
            }
        )
        assert isinstance(strategy, PotentialBasedShaper)
        assert strategy.name == "potential_based"
        assert strategy.gamma == 0.95

    def test_none_config_returns_identity(self):
        """None config returns identity (backward compatibility)."""
        from src.training.reward_shaping import create_strategy_from_config
        from src.training.reward_shaping.identity import IdentityRewardShaper

        strategy = create_strategy_from_config(None)
        assert isinstance(strategy, IdentityRewardShaper)

    def test_empty_config_returns_identity(self):
        """Empty config dict returns identity."""
        from src.training.reward_shaping import create_strategy_from_config
        from src.training.reward_shaping.identity import IdentityRewardShaper

        strategy = create_strategy_from_config({})
        assert isinstance(strategy, IdentityRewardShaper)

    def test_unknown_strategy_raises(self):
        """Unknown strategy name raises KeyError."""
        from src.training.reward_shaping import create_strategy_from_config

        with pytest.raises(KeyError, match="nonexistent"):
            create_strategy_from_config({"strategy_name": "nonexistent"})

    def test_all_strategies_listed(self):
        """list_strategies returns all 5 registered strategies."""
        from src.training.reward_shaping import list_strategies

        strategies = list_strategies()
        assert set(strategies) == {
            "identity",
            "difference_rewards",
            "reward_mixing",
            "coma_advantage",
            "potential_based",
        }


# ─── End-to-end strategy execution tests ──────────────────────────────────────


class TestEndToEndStrategyExecution:
    """Test full config -> strategy -> shape_rewards pipeline for each strategy."""

    def _make_batch(self, batch_size: int = 8):
        """Create a standard test batch of rewards."""
        return np.array([5.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 0.0][:batch_size])

    def test_identity_end_to_end(self):
        """identity: config -> create -> shape -> output unchanged."""
        from src.training.reward_shaping import create_strategy_from_config

        strategy = create_strategy_from_config({"strategy_name": "identity"})
        rewards = self._make_batch()
        result = strategy.shape_rewards(rewards, None, None)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, rewards)

    def test_difference_rewards_end_to_end(self):
        """difference_rewards: config -> create -> shape -> per-role dict."""
        from src.training.reward_shaping import create_strategy_from_config

        strategy = create_strategy_from_config({"strategy_name": "difference_rewards"})
        rewards = np.array([5.0, 0.0, 5.0, 0.0])
        metadata = [
            {"counterfactual_solver": 0.0, "counterfactual_verifier": 5.0, "counterfactual_judge": 0.0},
            {"counterfactual_solver": 0.0, "counterfactual_verifier": 0.0, "counterfactual_judge": 0.0},
            {"counterfactual_solver": 0.0, "counterfactual_verifier": 5.0, "counterfactual_judge": 5.0},
            {"counterfactual_solver": 0.0, "counterfactual_verifier": 0.0, "counterfactual_judge": 0.0},
        ]

        result = strategy.shape_rewards(rewards, None, metadata)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"solver", "verifier", "judge"}
        for role in result:
            assert result[role].shape == (4,)

        # Verify: D_solver[0] = G[0] - G_{-solver}[0] = 5.0 - 0.0 = 5.0
        assert result["solver"][0] == pytest.approx(5.0)
        # Verify: D_verifier[0] = G[0] - G_{-verifier}[0] = 5.0 - 5.0 = 0.0
        assert result["verifier"][0] == pytest.approx(0.0)

    def test_reward_mixing_end_to_end(self):
        """reward_mixing: config -> create -> shape -> blended per-role dict."""
        from src.training.reward_shaping import create_strategy_from_config

        strategy = create_strategy_from_config(
            {"strategy_name": "reward_mixing", "strategy_params": {"alpha": 0.5}}
        )
        rewards = np.array([5.0, 0.0])
        metadata = [
            {"local_reward_solver": 3.0, "local_reward_verifier": 1.0, "local_reward_judge": 4.0},
            {"local_reward_solver": 2.0, "local_reward_verifier": 6.0, "local_reward_judge": 0.0},
        ]

        result = strategy.shape_rewards(rewards, None, metadata)

        assert isinstance(result, dict)
        # r_solver[0] = 0.5*5.0 + 0.5*3.0 = 4.0
        assert result["solver"][0] == pytest.approx(4.0)
        # r_verifier[1] = 0.5*0.0 + 0.5*6.0 = 3.0
        assert result["verifier"][1] == pytest.approx(3.0)

    def test_coma_advantage_end_to_end(self):
        """coma_advantage: config -> create -> shape -> advantage dict."""
        from src.training.reward_shaping import create_strategy_from_config

        strategy = create_strategy_from_config(
            {"strategy_name": "coma_advantage", "strategy_params": {"n_rollouts_per_prompt": 4}}
        )
        # 4 rollouts for 1 prompt: mean = 2.5
        rewards = np.array([5.0, 0.0, 5.0, 0.0])

        result = strategy.shape_rewards(rewards, None, None)

        assert isinstance(result, dict)
        # GRPO fallback: advantage = reward - mean
        # mean = 2.5, so advantages = [2.5, -2.5, 2.5, -2.5]
        np.testing.assert_array_almost_equal(
            result["solver"], np.array([2.5, -2.5, 2.5, -2.5])
        )
        # All roles same in fallback mode
        np.testing.assert_array_almost_equal(result["solver"], result["verifier"])
        np.testing.assert_array_almost_equal(result["solver"], result["judge"])

    def test_potential_based_end_to_end(self):
        """potential_based: config -> create -> shape -> shaped array."""
        from src.training.reward_shaping import create_strategy_from_config

        strategy = create_strategy_from_config(
            {
                "strategy_name": "potential_based",
                "strategy_params": {"gamma": 0.99, "potential_type": "debate_length", "penalty": 0.1},
            }
        )
        rewards = np.array([5.0, 0.0])
        metadata = [
            {"state_start": {"num_turns": 0}, "state_end": {"num_turns": 3}},
            {"state_start": {"num_turns": 0}, "state_end": {"num_turns": 5}},
        ]

        result = strategy.shape_rewards(rewards, None, metadata)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        # r'[0] = 5.0 + 0.99 * (-0.1 * 3) - (-0.1 * 0) = 5.0 - 0.297 = 4.703
        assert result[0] == pytest.approx(5.0 + 0.99 * (-0.3) - 0.0)
        # r'[1] = 0.0 + 0.99 * (-0.1 * 5) - (-0.1 * 0) = 0.0 - 0.495 = -0.495
        assert result[1] == pytest.approx(0.0 + 0.99 * (-0.5) - 0.0)


# ─── DebateMetricStreamer integration tests ───────────────────────────────────


class TestDebateMetricStreamerRewardShaping:
    """Test reward shaping integration in DebateMetricStreamer."""

    def test_default_config_uses_identity(self):
        """Default config (no reward_shaping_strategy) uses identity strategy."""
        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )

        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="verifier"),
        ]

        upstream = MockUpstreamStreamer(items, {})
        config = DebateMetricStreamerConfig(n_rollouts_per_prompt=2)
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        assert streamer._reward_shaper.name == "identity"

        _, metrics = streamer.get()

        # Shaped reward metrics should exist
        assert "debate/shaped_reward/mean" in metrics
        assert "debate/shaped_reward/strategy_active" in metrics
        # Strategy active = 0.0 for identity
        assert metrics["debate/shaped_reward/strategy_active"] == 0.0

    def test_identity_shaped_rewards_match_unshaped(self):
        """Identity strategy shaped rewards match original rewards."""
        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )
        from src.training.wandb_enrichment.metric_schema import (
            METRIC_REWARD_SOLVER,
            METRIC_REWARD_VERIFIER,
        )

        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=1.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="verifier"),
            MockActorOutputItem(reward=3.0, role_label="verifier"),
        ]

        upstream = MockUpstreamStreamer(items, {})
        config = DebateMetricStreamerConfig(n_rollouts_per_prompt=4)
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        _, metrics = streamer.get()

        # For identity, shaped per-role should match unshaped per-role
        assert metrics["debate/shaped_reward/solver"] == pytest.approx(
            metrics[METRIC_REWARD_SOLVER]
        )
        assert metrics["debate/shaped_reward/verifier"] == pytest.approx(
            metrics[METRIC_REWARD_VERIFIER]
        )

    def test_reward_mixing_shaped_metrics(self):
        """reward_mixing strategy produces shaped reward metrics in streamer."""
        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )

        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="verifier"),
            MockActorOutputItem(reward=5.0, role_label="judge"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
        ]

        upstream = MockUpstreamStreamer(items, {})
        config = DebateMetricStreamerConfig(
            n_rollouts_per_prompt=4,
            reward_shaping_strategy="reward_mixing",
            reward_shaping_params={"alpha": 0.5},
        )
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        assert streamer._reward_shaper.name == "reward_mixing"

        _, metrics = streamer.get()

        # Shaped reward metrics should exist
        assert "debate/shaped_reward/solver" in metrics
        assert "debate/shaped_reward/verifier" in metrics
        assert "debate/shaped_reward/judge" in metrics
        assert "debate/shaped_reward/mean" in metrics
        # Strategy active = 1.0 for non-identity
        assert metrics["debate/shaped_reward/strategy_active"] == 1.0

    def test_coma_advantage_shaped_metrics(self):
        """coma_advantage strategy produces shaped reward metrics in streamer."""
        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )

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
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        assert streamer._reward_shaper.name == "coma_advantage"

        _, metrics = streamer.get()

        # Shaped reward metrics should exist for all roles
        assert "debate/shaped_reward/solver" in metrics
        assert "debate/shaped_reward/verifier" in metrics
        assert "debate/shaped_reward/judge" in metrics
        assert "debate/shaped_reward/mean" in metrics
        assert metrics["debate/shaped_reward/strategy_active"] == 1.0

    def test_potential_based_shaped_metrics(self):
        """potential_based strategy produces global shaped reward metrics."""
        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )

        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="verifier"),
        ]

        upstream = MockUpstreamStreamer(items, {})
        config = DebateMetricStreamerConfig(
            n_rollouts_per_prompt=2,
            reward_shaping_strategy="potential_based",
            reward_shaping_params={"gamma": 0.99, "potential_type": "zero"},
        )
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        assert streamer._reward_shaper.name == "potential_based"

        _, metrics = streamer.get()

        # With zero potential, shaped should equal unshaped
        assert "debate/shaped_reward/mean" in metrics
        assert metrics["debate/shaped_reward/strategy_active"] == 1.0

    def test_original_metrics_unchanged_with_shaping(self):
        """Original (unshaped) debate metrics remain unchanged when shaping is active."""
        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )
        from src.training.wandb_enrichment.metric_schema import (
            METRIC_FRAC_ZERO_STD,
            METRIC_MEAN_REWARD_STD,
            METRIC_REWARD_SOLVER,
        )

        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
        ]

        upstream = MockUpstreamStreamer(items, {})

        # Get metrics without shaping
        config_no_shape = DebateMetricStreamerConfig(n_rollouts_per_prompt=4)
        streamer_no_shape = DebateMetricStreamer(config_no_shape, upstream, MagicMock())
        _, metrics_no_shape = streamer_no_shape.get()

        # Get metrics with shaping
        config_shape = DebateMetricStreamerConfig(
            n_rollouts_per_prompt=4,
            reward_shaping_strategy="reward_mixing",
            reward_shaping_params={"alpha": 0.3},
        )
        streamer_shape = DebateMetricStreamer(config_shape, upstream, MagicMock())
        _, metrics_shape = streamer_shape.get()

        # Original metrics should be identical
        assert metrics_shape[METRIC_REWARD_SOLVER] == pytest.approx(
            metrics_no_shape[METRIC_REWARD_SOLVER]
        )
        assert metrics_shape[METRIC_FRAC_ZERO_STD] == pytest.approx(
            metrics_no_shape[METRIC_FRAC_ZERO_STD]
        )
        assert metrics_shape[METRIC_MEAN_REWARD_STD] == pytest.approx(
            metrics_no_shape[METRIC_MEAN_REWARD_STD]
        )

    def test_reward_shaping_error_does_not_crash(self):
        """Errors in reward shaping do not crash the streamer."""
        from unittest.mock import patch

        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )
        from src.training.wandb_enrichment.metric_schema import METRIC_REWARD_SOLVER

        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
        ]

        upstream = MockUpstreamStreamer(items, {})
        config = DebateMetricStreamerConfig(n_rollouts_per_prompt=2)
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        # Patch the shaper to raise an error
        streamer._reward_shaper.shape_rewards = MagicMock(side_effect=RuntimeError("boom"))

        _, metrics = streamer.get()

        # Original metrics should still be present
        assert METRIC_REWARD_SOLVER in metrics
        # Shaped metrics should NOT be present (error was caught)
        assert "debate/shaped_reward/mean" not in metrics

    def test_empty_strategy_name_uses_identity(self):
        """Empty reward_shaping_strategy string uses identity."""
        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )

        items = [MockActorOutputItem(reward=5.0, role_label="solver")]

        upstream = MockUpstreamStreamer(items, {})
        config = DebateMetricStreamerConfig(
            n_rollouts_per_prompt=1,
            reward_shaping_strategy="",  # Empty = identity
        )
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        assert streamer._reward_shaper.name == "identity"

    def test_difference_rewards_shaped_metrics(self):
        """difference_rewards strategy produces shaped reward metrics."""
        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )

        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="verifier"),
            MockActorOutputItem(reward=5.0, role_label="judge"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
        ]

        upstream = MockUpstreamStreamer(items, {})
        config = DebateMetricStreamerConfig(
            n_rollouts_per_prompt=4,
            reward_shaping_strategy="difference_rewards",
        )
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        assert streamer._reward_shaper.name == "difference_rewards"

        _, metrics = streamer.get()

        # Shaped reward metrics should exist
        assert "debate/shaped_reward/solver" in metrics
        assert "debate/shaped_reward/verifier" in metrics
        assert "debate/shaped_reward/judge" in metrics
        assert "debate/shaped_reward/mean" in metrics
        assert metrics["debate/shaped_reward/strategy_active"] == 1.0

    def test_reward_mixing_mutates_item_rewards(self):
        """reward_mixing strategy writes shaped values back to item.data['rewards']."""
        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )

        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="verifier"),
            MockActorOutputItem(reward=0.0, role_label="verifier"),
        ]

        upstream = MockUpstreamStreamer(items, {})
        config = DebateMetricStreamerConfig(
            n_rollouts_per_prompt=4,
            reward_shaping_strategy="reward_mixing",
            reward_shaping_params={"alpha": 0.5},
        )
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        result_items, _ = streamer.get()

        # reward_mixing without metadata: local falls back to G, so r = 0.5*G + 0.5*G = G
        # Shaped equals raw, but write-back still happens (type-safe)
        assert result_items[0].data["rewards"].item() == pytest.approx(5.0)
        assert result_items[1].data["rewards"].item() == pytest.approx(0.0)
        assert result_items[2].data["rewards"].item() == pytest.approx(5.0)
        assert result_items[3].data["rewards"].item() == pytest.approx(0.0)
        # Verify dtype preserved
        assert result_items[0].data["rewards"].dtype == np.float64

    def test_difference_rewards_mutates_item_rewards(self):
        """difference_rewards strategy writes shaped values back to item.data['rewards']."""
        from src.training.wandb_enrichment.debate_streamer import (
            DebateMetricStreamer,
            DebateMetricStreamerConfig,
        )

        items = [
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
            MockActorOutputItem(reward=5.0, role_label="solver"),
            MockActorOutputItem(reward=0.0, role_label="solver"),
        ]

        upstream = MockUpstreamStreamer(items, {})
        config = DebateMetricStreamerConfig(
            n_rollouts_per_prompt=4,
            reward_shaping_strategy="difference_rewards",
        )
        streamer = DebateMetricStreamer(config, upstream, MagicMock())

        result_items, _ = streamer.get()

        # difference_rewards without metadata falls back to raw reward copy
        # Each item's data["rewards"] should be the solver's difference reward
        assert result_items[0].data["rewards"].item() == pytest.approx(5.0)
        assert result_items[1].data["rewards"].item() == pytest.approx(0.0)
        assert result_items[2].data["rewards"].item() == pytest.approx(5.0)
        assert result_items[3].data["rewards"].item() == pytest.approx(0.0)
        # Verify dtype preserved
        assert result_items[0].data["rewards"].dtype == np.float64


# ─── Identity regression and gradient-path tests ─────────────────────────────


def _run_streamer_and_collect_rewards(items_spec, config_kwargs):
    """Run DebateMetricStreamer.get() and return post-mutation reward values.

    Creates fresh MockActorOutputItem copies (to avoid mutation contamination),
    runs streamer.get(), and returns the mutated item.data["rewards"] values
    as a numpy array.

    Args:
        items_spec: List of (reward, role_label) tuples to create MockActorOutputItems from.
        config_kwargs: Dict of DebateMetricStreamerConfig kwargs.

    Returns:
        np.ndarray of shape (len(items_spec),) with post-mutation reward values.
    """
    from src.training.wandb_enrichment.debate_streamer import (
        DebateMetricStreamer,
        DebateMetricStreamerConfig,
    )

    mock_items = [MockActorOutputItem(reward=r, role_label=rl) for r, rl in items_spec]
    upstream = MockUpstreamStreamer(mock_items, {})
    config = DebateMetricStreamerConfig(**config_kwargs)
    streamer = DebateMetricStreamer(config, upstream, MagicMock())
    result_items, _ = streamer.get()
    return np.array([item.data["rewards"].item() for item in result_items])


class TestIdentityRegressionAndGradientPath:
    """RINT-04: Identity regression tests and gradient-path liveness tests.

    Verifies:
    - Identity strategy == no strategy configured (backward compat)
    - Non-identity strategies produce observably different shaped rewards
    - Gradient-norm comparison proves shaped rewards affect the gradient path
    """

    def test_identity_regression_single_step(self):
        """No-strategy and explicit identity produce identical rewards matching raw."""
        raw_rewards = [5.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 0.0]
        items_spec = [(r, "solver") for r in raw_rewards]

        # Run with no reward_shaping_strategy (defaults to identity)
        rewards_no_strategy = _run_streamer_and_collect_rewards(
            items_spec, {"n_rollouts_per_prompt": 8}
        )

        # Run with explicit identity
        rewards_identity = _run_streamer_and_collect_rewards(
            items_spec, {"n_rollouts_per_prompt": 8, "reward_shaping_strategy": "identity"}
        )

        # Both should match each other
        np.testing.assert_allclose(
            rewards_no_strategy, rewards_identity, atol=1e-6, rtol=1e-5
        )

        # Both should match the original raw rewards
        np.testing.assert_allclose(
            rewards_no_strategy, np.array(raw_rewards), atol=1e-6, rtol=1e-5
        )
        np.testing.assert_allclose(
            rewards_identity, np.array(raw_rewards), atol=1e-6, rtol=1e-5
        )

    def test_identity_regression_with_mixed_roles(self):
        """Identity regression holds with mixed solver/verifier roles."""
        items_spec = [
            (5.0, "solver"),
            (0.0, "solver"),
            (5.0, "solver"),
            (0.0, "solver"),
            (0.0, "verifier"),
            (5.0, "verifier"),
            (0.0, "verifier"),
            (5.0, "verifier"),
        ]
        raw_rewards = [r for r, _ in items_spec]

        rewards_no_strategy = _run_streamer_and_collect_rewards(
            items_spec, {"n_rollouts_per_prompt": 8}
        )
        rewards_identity = _run_streamer_and_collect_rewards(
            items_spec, {"n_rollouts_per_prompt": 8, "reward_shaping_strategy": "identity"}
        )

        np.testing.assert_allclose(
            rewards_no_strategy, rewards_identity, atol=1e-6, rtol=1e-5
        )
        np.testing.assert_allclose(
            rewards_no_strategy, np.array(raw_rewards), atol=1e-6, rtol=1e-5
        )

    def test_multi_step_identity_vs_no_strategy(self):
        """Multi-step (5 steps) identity regression: no-strategy == explicit identity at each step."""
        for step in range(5):
            rng = np.random.RandomState(seed=step)
            raw_rewards = rng.uniform(0.0, 5.0, size=8).tolist()
            items_spec = [(r, "solver") for r in raw_rewards]

            rewards_no_strategy = _run_streamer_and_collect_rewards(
                items_spec, {"n_rollouts_per_prompt": 8}
            )
            rewards_identity = _run_streamer_and_collect_rewards(
                items_spec, {"n_rollouts_per_prompt": 8, "reward_shaping_strategy": "identity"}
            )

            np.testing.assert_allclose(
                rewards_no_strategy,
                rewards_identity,
                atol=1e-6,
                err_msg=f"Identity regression failed at step {step}",
            )

    def test_non_identity_produces_different_rewards(self):
        """COMA advantage produces different rewards than identity (non-trivial shaping)."""
        raw_rewards = [5.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 0.0]
        items_spec = [(r, "solver") for r in raw_rewards]

        rewards_identity = _run_streamer_and_collect_rewards(
            items_spec, {"n_rollouts_per_prompt": 8}
        )
        rewards_coma = _run_streamer_and_collect_rewards(
            items_spec,
            {
                "n_rollouts_per_prompt": 8,
                "reward_shaping_strategy": "coma_advantage",
                "reward_shaping_params": {"n_rollouts_per_prompt": 8},
            },
        )

        # COMA advantage: mean=2.5, so advantages = reward - mean
        expected_coma = np.array([2.5, -2.5, 2.5, -2.5, 2.5, 2.5, -2.5, -2.5])

        # Shaped rewards differ from identity
        assert not np.allclose(rewards_identity, rewards_coma)

        # Shaped rewards match expected COMA advantage values
        np.testing.assert_allclose(rewards_coma, expected_coma, atol=1e-6)

    def test_multi_step_non_identity_differs_from_identity(self):
        """Multi-step (5 steps) divergence: non-identity rewards differ from identity."""
        cumulative_diff = 0.0
        at_least_one_differs = False

        for step in range(5):
            rng = np.random.RandomState(seed=step + 100)
            raw_rewards = rng.uniform(0.0, 5.0, size=8).tolist()
            items_spec = [(r, "solver") for r in raw_rewards]

            rewards_identity = _run_streamer_and_collect_rewards(
                items_spec, {"n_rollouts_per_prompt": 8}
            )
            rewards_coma = _run_streamer_and_collect_rewards(
                items_spec,
                {
                    "n_rollouts_per_prompt": 8,
                    "reward_shaping_strategy": "coma_advantage",
                    "reward_shaping_params": {"n_rollouts_per_prompt": 8},
                },
            )

            step_diff = np.abs(rewards_identity - rewards_coma).sum()
            cumulative_diff += step_diff
            if not np.allclose(rewards_identity, rewards_coma):
                at_least_one_differs = True

        assert at_least_one_differs, "Expected at least one step where COMA differs from identity"
        assert cumulative_diff > 0, "Expected cumulative difference > 0 across 5 steps"

    def test_gradient_norm_differs_with_shaping(self):
        """Gradient-norm comparison (5 steps): shaped rewards produce different gradients.

        Uses torch.backward() per locked decision R12/R13 to prove gradient-path liveness.
        """
        torch = pytest.importorskip("torch")

        raw_norms = []
        shaped_norms = []
        identity_norms_a = []
        identity_norms_b = []

        for step in range(5):
            rng = np.random.RandomState(seed=step + 200)
            raw_rewards_list = rng.uniform(0.0, 5.0, size=8).tolist()
            items_spec = [(r, "solver") for r in raw_rewards_list]

            # Collect raw (identity) rewards
            raw_rewards = _run_streamer_and_collect_rewards(
                items_spec, {"n_rollouts_per_prompt": 8}
            )

            # Collect shaped (COMA advantage) rewards
            shaped_rewards = _run_streamer_and_collect_rewards(
                items_spec,
                {
                    "n_rollouts_per_prompt": 8,
                    "reward_shaping_strategy": "coma_advantage",
                    "reward_shaping_params": {"n_rollouts_per_prompt": 8},
                },
            )

            # Simulate gradient computation with raw rewards
            param_raw = torch.randn(8, requires_grad=True)
            loss_raw = (torch.tensor(raw_rewards, dtype=torch.float32) * param_raw).sum()
            loss_raw.backward()
            raw_norms.append(param_raw.grad.norm().item())

            # Simulate gradient computation with shaped rewards
            param_shaped = param_raw.detach().clone().requires_grad_(True)
            loss_shaped = (torch.tensor(shaped_rewards, dtype=torch.float32) * param_shaped).sum()
            loss_shaped.backward()
            shaped_norms.append(param_shaped.grad.norm().item())

            # Also test identity vs no-strategy gradient norms
            identity_rewards = _run_streamer_and_collect_rewards(
                items_spec,
                {"n_rollouts_per_prompt": 8, "reward_shaping_strategy": "identity"},
            )

            param_id_a = param_raw.detach().clone().requires_grad_(True)
            loss_id_a = (torch.tensor(raw_rewards, dtype=torch.float32) * param_id_a).sum()
            loss_id_a.backward()
            identity_norms_a.append(param_id_a.grad.norm().item())

            param_id_b = param_raw.detach().clone().requires_grad_(True)
            loss_id_b = (torch.tensor(identity_rewards, dtype=torch.float32) * param_id_b).sum()
            loss_id_b.backward()
            identity_norms_b.append(param_id_b.grad.norm().item())

        # Shaped rewards should produce different gradient norms in at least one step
        assert any(
            abs(raw - shaped) > 1e-6
            for raw, shaped in zip(raw_norms, shaped_norms)
        ), "Expected gradient norms to differ between raw and shaped rewards"

        # Identity vs no-strategy should produce identical gradient norms across all steps
        for step_idx, (norm_a, norm_b) in enumerate(zip(identity_norms_a, identity_norms_b)):
            assert abs(norm_a - norm_b) < 1e-5, (
                f"Identity vs no-strategy gradient norms differ at step {step_idx}: "
                f"{norm_a} vs {norm_b}"
            )

    @pytest.mark.ci
    def test_gradient_norm_ci_lightweight(self):
        """Lightweight CI gradient-norm check (2 steps, fixed rewards).

        Per locked decision R13: 1-2 step CI variant of gradient-norm comparison.
        """
        torch = pytest.importorskip("torch")

        fixed_rewards_sets = [
            [5.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 5.0, 0.0, 0.0, 5.0, 5.0],
        ]

        raw_norms = []
        shaped_norms = []

        for rewards_list in fixed_rewards_sets:
            items_spec = [(r, "solver") for r in rewards_list]

            raw_rewards = _run_streamer_and_collect_rewards(
                items_spec, {"n_rollouts_per_prompt": 8}
            )
            shaped_rewards = _run_streamer_and_collect_rewards(
                items_spec,
                {
                    "n_rollouts_per_prompt": 8,
                    "reward_shaping_strategy": "coma_advantage",
                    "reward_shaping_params": {"n_rollouts_per_prompt": 8},
                },
            )

            param_raw = torch.randn(8, requires_grad=True)
            loss_raw = (torch.tensor(raw_rewards, dtype=torch.float32) * param_raw).sum()
            loss_raw.backward()
            raw_norms.append(param_raw.grad.norm().item())

            param_shaped = param_raw.detach().clone().requires_grad_(True)
            loss_shaped = (torch.tensor(shaped_rewards, dtype=torch.float32) * param_shaped).sum()
            loss_shaped.backward()
            shaped_norms.append(param_shaped.grad.norm().item())

        # At least 1 of 2 steps should show different gradient norms
        assert any(
            abs(raw - shaped) > 1e-6
            for raw, shaped in zip(raw_norms, shaped_norms)
        ), "CI check: expected gradient norms to differ for at least 1 of 2 steps"

    def test_potential_based_produces_different_rewards(self):
        """Potential-based strategy produces different rewards when state metadata is provided."""
        from src.training.reward_shaping.potential_shaping import PotentialBasedShaper

        rewards = np.array([5.0, 0.0, 5.0, 0.0])
        metadata = [
            {"state_start": {"num_turns": 0}, "state_end": {"num_turns": 3}},
            {"state_start": {"num_turns": 0}, "state_end": {"num_turns": 5}},
            {"state_start": {"num_turns": 1}, "state_end": {"num_turns": 4}},
            {"state_start": {"num_turns": 2}, "state_end": {"num_turns": 6}},
        ]

        shaper = PotentialBasedShaper(gamma=0.99, potential_type="debate_length", penalty=0.1)
        shaped = shaper.shape_rewards(rewards, None, metadata)

        # Output should differ from input (potential term adds shaping)
        assert not np.allclose(shaped, rewards), (
            "Potential-based shaping should produce different values with state metadata"
        )

        # Verify specific values:
        # r'[0] = 5.0 + 0.99 * (-0.1*3) - (-0.1*0) = 5.0 - 0.297 = 4.703
        assert shaped[0] == pytest.approx(5.0 + 0.99 * (-0.3) - 0.0, abs=1e-6)
        # r'[1] = 0.0 + 0.99 * (-0.1*5) - (-0.1*0) = 0.0 - 0.495 = -0.495
        assert shaped[1] == pytest.approx(0.0 + 0.99 * (-0.5) - 0.0, abs=1e-6)
