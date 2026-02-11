"""Custom Flink actor output streamer for debate metric enrichment.

This module provides a FlinkActorOutputStreamer that intercepts actor outputs,
computes debate-specific scalar metrics from batch rewards, and returns them
in the Metrics dict from get(). These metrics flow through flink_batching's
weighted average into actor_metrics -> learner_metrics -> W&B plotter.

The streamer is wired into the training pipeline via SWEEP config's
actor_outputs_streamers extension point (no Flink core changes).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

# Import debate metrics (always available)
from src.training.wandb_enrichment.debate_metrics import (
    compute_per_role_rewards,
    compute_zero_advantage_metrics,
)

# Conditional imports for Flink infrastructure (not available in local testing)
if TYPE_CHECKING:
    from fax import config as fax_config
    from post_training import relax
    from post_training.flink import flink_types
    from post_training.flink.components.flink_learning_filter import async_metrics_collector

try:
    from fax.config import components
    from post_training.flink import flink_types

    FLINK_AVAILABLE = True
except ImportError:
    FLINK_AVAILABLE = False
    flink_types = None

    # Minimal stub for components
    class _ComponentBaseStub:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class components:
        ComponentBase = _ComponentBaseStub

logger = logging.getLogger(__name__)


# Implementation mixin with the actual logic (shared between both class definitions)
class _DebateMetricStreamerImpl:
    """Implementation mixin for DebateMetricStreamer.

    Wraps an upstream streamer. On each get() call:
    1. Calls upstream.get() to receive items + metrics
    2. Extracts rewards and role labels from items
    3. Computes per-role rewards and zero-advantage metrics
    4. Merges debate metrics into the returned Metrics dict

    The returned metrics flow through flink_batching -> actor_metrics -> learner_metrics -> W&B.
    """

    def __init__(self, config, upstream, metrics_collector):
        """Initialize DebateMetricStreamer.

        Args:
            config: DebateMetricStreamerConfig
            upstream: Upstream FlinkActorOutputStreamer
            metrics_collector: AsyncMetricsCollector
        """
        self._config = config
        self._upstream = upstream
        self._metrics_collector = metrics_collector
        self._get_count = 0  # Track get() calls for rollout table logging

    def get(self):
        """Fetch items from upstream and enrich with debate metrics.

        Also handles rollout table logging and workspace initialization.

        Returns:
            Tuple of (items, metrics) where metrics includes debate-specific scalars.
        """
        items, upstream_metrics = self._upstream.get()
        self._get_count += 1

        # Initialize W&B workspace on first get() call
        if self._get_count == 1:
            try:
                from src.training.wandb_enrichment.workspace_init import init_debate_workspace
                init_debate_workspace()
            except Exception as e:
                logger.warning(f"DebateMetricStreamer: workspace init failed: {e}")

        try:
            # Extract rewards from items
            # items[i].data is a dict with "rewards" key containing numpy scalar
            rewards = np.array([item.data["rewards"].item() for item in items])

            # Extract role labels from item metadata
            # Metadata follows pattern: item.metadata["{env_name}/reward_metrics"] = dict with "role_label"
            role_labels = []
            for item in items:
                # Get env_name from metadata
                # Metadata keys are flat strings, values are numpy arrays or dicts
                env_name_arr = item.metadata.get("env_name", np.array(""))
                env_name = env_name_arr.item() if hasattr(env_name_arr, 'item') else str(env_name_arr)

                # Construct reward_metrics key
                reward_metrics_key = f"{env_name}/reward_metrics"
                reward_metrics = item.metadata.get(reward_metrics_key, {})

                # Extract role_label from reward_metrics
                if isinstance(reward_metrics, dict) and "role_label" in reward_metrics:
                    label = reward_metrics["role_label"]
                    role_labels.append(label.item() if hasattr(label, 'item') else str(label))
                else:
                    # Default to "solver" if role_label not found
                    role_labels.append("solver")

            # Compute debate metrics
            debate_metrics = {}
            debate_metrics.update(compute_per_role_rewards(rewards, role_labels))
            debate_metrics.update(compute_zero_advantage_metrics(rewards, self._config.n_rollouts_per_prompt))

            # Merge: upstream metrics take precedence on conflict (unlikely)
            all_metrics = {**debate_metrics, **upstream_metrics}

            # Log rollout table at configured intervals
            if self._get_count % self._config.log_rollout_table_every_n_gets == 0:
                try:
                    from src.training.wandb_enrichment.rollout_integration import log_debate_rollout_table
                    log_debate_rollout_table(items=items, step=self._get_count)
                except Exception as e:
                    logger.warning(f"DebateMetricStreamer: rollout table logging failed: {e}")

            # Write Parquet debug data at configured intervals (for Streamlit viewer)
            if self._config.debug_data_output_dir and self._get_count % self._config.log_rollout_table_every_n_gets == 0:
                try:
                    from src.training.wandb_enrichment.rollout_integration import write_debate_debug_data
                    write_debate_debug_data(
                        items=items,
                        step=self._get_count,
                        output_dir=self._config.debug_data_output_dir,
                    )
                except Exception as e:
                    logger.warning(f"DebateMetricStreamer: debug data writing failed: {e}")

            return items, all_metrics

        except Exception as e:
            logger.warning(f"DebateMetricStreamer: failed to compute debate metrics: {e}", exc_info=True)
            # Return upstream metrics unchanged on error
            return items, upstream_metrics

    def flush_and_discard_ready_items(self):
        """Flush upstream streamer's ready items."""
        self._upstream.flush_and_discard_ready_items()


# Define classes based on Flink availability
# IMPORTANT: component_name MUST be a literal argument, not dynamic **kwargs,
# because Ray/cloudpickle doesn't preserve dynamically unpacked class keywords
if FLINK_AVAILABLE:
    class DebateMetricStreamerConfig(flink_types.FlinkActorOutputStreamerConfig, component_name="debate_metric_streamer"):
        """Configuration for DebateMetricStreamer.

        Attributes:
            n_rollouts_per_prompt: GRPO group size (number of rollouts per prompt).
                Must match generations_per_prompt in SWEEP config.
            log_rollout_table_every_n_gets: Log W&B rollout table every N get() calls.
                Roughly corresponds to log_train_generations_every_steps in Flink.
        """

        n_rollouts_per_prompt: int = 8  # GRPO group size
        log_rollout_table_every_n_gets: int = 25  # Log W&B table every N get() calls
        debug_data_output_dir: str = ""  # Empty = disabled; set to dir path to write Parquet debug data

        def create_streamer(self, upstream, config, metrics_collector):
            """Create a DebateMetricStreamer instance."""
            return DebateMetricStreamer(self, upstream, metrics_collector)

    class DebateMetricStreamer(_DebateMetricStreamerImpl, flink_types.FlinkActorOutputStreamer):
        """Flink-compatible DebateMetricStreamer."""
        pass

else:
    # Stubs for local testing without Flink
    class DebateMetricStreamerConfig(components.ComponentBase):
        """Configuration for DebateMetricStreamer (test stub)."""

        n_rollouts_per_prompt: int = 8
        log_rollout_table_every_n_gets: int = 25
        debug_data_output_dir: str = ""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            # Set defaults for missing attributes
            if not hasattr(self, 'n_rollouts_per_prompt'):
                self.n_rollouts_per_prompt = 8
            if not hasattr(self, 'log_rollout_table_every_n_gets'):
                self.log_rollout_table_every_n_gets = 25
            if not hasattr(self, 'debug_data_output_dir'):
                self.debug_data_output_dir = ""

        def create_streamer(self, upstream, config, metrics_collector):
            """Create a DebateMetricStreamer instance."""
            return DebateMetricStreamer(self, upstream, metrics_collector)

    class DebateMetricStreamer(_DebateMetricStreamerImpl):
        """Test stub for DebateMetricStreamer."""
        pass
