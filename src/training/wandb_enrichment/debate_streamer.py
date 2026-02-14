"""Custom Flink actor output streamer for debate metric enrichment.

This module provides a FlinkActorOutputStreamer that intercepts actor outputs,
computes debate-specific scalar metrics from batch rewards, and returns them
in the Metrics dict from get(). These metrics flow through flink_batching's
weighted average into actor_metrics -> learner_metrics -> W&B plotter.

The streamer is wired into the training pipeline via SWEEP config's
actor_outputs_streamers extension point (no Flink core changes).

Rollout strategy integration (Phase 9):
The streamer can optionally apply a rollout selection strategy to filter or
modify rollouts BEFORE reward extraction and shaping. Configure via
rollout_strategy and rollout_strategy_params in DebateMetricStreamerConfig.
Execution order: rollout selection -> reward extraction -> reward shaping -> metrics.

Reward shaping integration (Phase 8):
The streamer can optionally apply a reward shaping strategy to transform
raw binary rewards before logging. Shaped rewards are logged as additional
debate/shaped_reward/* metrics. The original (unshaped) metrics remain
unchanged for backward compatibility. Configure via reward_shaping_strategy
and reward_shaping_params in DebateMetricStreamerConfig.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

# Import reward shaping and rollout strategy registries (always available)
from src.training.reward_shaping import create_strategy_from_config as create_reward_strategy
from src.training.rollout_strategy import create_strategy_from_config as create_rollout_strategy

# Import debate metrics (always available)
from src.training.wandb_enrichment.debate_metrics import (
    compute_per_role_rewards,
    compute_zero_advantage_metrics,
)

# Import shaped reward metric constants
from src.training.wandb_enrichment.metric_schema import (
    METRIC_SHAPED_REWARD_MEAN,
    METRIC_SHAPED_REWARD_SOLVER,
    METRIC_SHAPED_REWARD_VERIFIER,
    METRIC_SHAPED_REWARD_JUDGE,
    METRIC_SHAPED_REWARD_STRATEGY_ACTIVE,
)

# Conditional imports for Flink infrastructure (not available in local testing)
if TYPE_CHECKING:
    from post_training.flink import flink_types

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
        self._logged_mutation = False  # Log shaped reward write-back only once

        # Initialize reward shaping strategy from config (Phase 8)
        reward_shaping_config = None
        strategy_name = getattr(config, "reward_shaping_strategy", "")
        if strategy_name:
            reward_shaping_config = {
                "strategy_name": strategy_name,
                "strategy_params": getattr(config, "reward_shaping_params", {}) or {},
            }
        self._reward_shaper = create_reward_strategy(reward_shaping_config)
        logger.info(
            f"DebateMetricStreamer: initialized reward shaping strategy '{self._reward_shaper.name}'"
        )

        # Initialize rollout strategy from config (Phase 9)
        rollout_config = None
        rollout_strategy_name = getattr(config, "rollout_strategy", "")
        if rollout_strategy_name:
            rollout_config = {
                "strategy_name": rollout_strategy_name,
                "strategy_params": getattr(config, "rollout_strategy_params", {}) or {},
            }
        self._rollout_strategy = create_rollout_strategy(rollout_config)
        logger.info(
            f"DebateMetricStreamer: using rollout strategy: {self._rollout_strategy.name}"
        )

    def _update_wandb_config(self):
        """Surface reward shaping config as top-level WandB run config keys."""
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update({
                    "reward_shaping_strategy": self._reward_shaper.name,
                    "reward_shaping_params": getattr(self._config, "reward_shaping_params", {}) or {},
                }, allow_val_change=True)
                logger.info(
                    f"DebateMetricStreamer: updated WandB config with "
                    f"reward_shaping_strategy='{self._reward_shaper.name}'"
                )
        except Exception as e:
            logger.warning(f"DebateMetricStreamer: failed to update WandB config: {e}")

    def get(self):
        """Fetch items from upstream and enrich with debate metrics.

        Also handles rollout table logging, workspace initialization,
        and shaped reward write-back into item.data["rewards"].

        Returns:
            Tuple of (items, metrics) where metrics includes debate-specific scalars.
            Items have their data["rewards"] mutated with shaped values when a
            non-identity strategy is configured.
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
            self._update_wandb_config()

        # Apply rollout strategy BEFORE reward extraction (Phase 9)
        # Execution order: rollout selection -> reward extraction -> reward shaping -> metrics
        rollout_strategy_metrics = {}
        items_in = len(items)
        try:
            items = self._rollout_strategy.select_rollouts(
                items, self._config.n_rollouts_per_prompt
            )
            items_out = len(items)
            selection_ratio = items_out / items_in if items_in > 0 else 1.0

            # Compute mean selected reward
            selected_rewards = [item.data["rewards"].item() for item in items]
            mean_selected_reward = float(np.mean(selected_rewards)) if selected_rewards else 0.0

            rollout_strategy_metrics = {
                "debate/rollout_strategy/items_in": float(items_in),
                "debate/rollout_strategy/items_out": float(items_out),
                "debate/rollout_strategy/selection_ratio": selection_ratio,
                "debate/rollout_strategy/mean_selected_reward": mean_selected_reward,
            }
        except Exception as e:
            logger.warning(
                f"DebateMetricStreamer: rollout strategy failed, using original items: {e}"
            )

        try:
            # STEP A: Extract rewards from items (possibly filtered by rollout strategy)
            # items[i].data is a dict with "rewards" key containing numpy scalar
            rewards = np.array([item.data["rewards"].item() for item in items])

            # STEP B: Capture original dtype for write-back
            original_dtype = items[0].data["rewards"].dtype

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

            # STEP C: Compute debate metrics (unshaped, for backward compatibility)
            # These use the ORIGINAL raw rewards array. Do NOT move these after mutation.
            debate_metrics = {}
            debate_metrics.update(compute_per_role_rewards(rewards, role_labels))

            # Adjust n_rollouts_per_prompt for zero-advantage computation:
            # If rollout strategy filtered items (e.g., best-of-N: N*P -> P),
            # each "group" is now 1 item, so use n_rollouts_per_prompt=1.
            effective_n_rollouts = self._config.n_rollouts_per_prompt
            if len(items) < items_in:
                effective_n_rollouts = 1
            debate_metrics.update(compute_zero_advantage_metrics(rewards, effective_n_rollouts))

            # STEP D: Apply reward shaping, compute per-item shaped values, and log metrics
            try:
                shaped_metrics, shaped_per_item = self._compute_and_apply_shaped_rewards(
                    rewards, role_labels
                )
                debate_metrics.update(shaped_metrics)

                # STEP E: Write shaped rewards back into items for learner consumption
                # items from get() feed into FlinkRlooLearner -> GRPO loss reads
                # item.data["rewards"] for advantage computation
                for idx, item in enumerate(items):
                    item.data["rewards"] = np.array(shaped_per_item[idx], dtype=original_dtype)

                # One-time INFO log when reward mutation is active (non-identity strategy)
                if not self._logged_mutation and self._reward_shaper.name != "identity":
                    logger.info(
                        f"DebateMetricStreamer: shaped rewards written to item.data['rewards'] "
                        f"using strategy '{self._reward_shaper.name}'"
                    )
                    self._logged_mutation = True
            except Exception as e:
                logger.warning(f"DebateMetricStreamer: reward shaping failed: {e}")

            # Merge: rollout strategy + debate + upstream (upstream takes precedence on conflict)
            all_metrics = {**rollout_strategy_metrics, **debate_metrics, **upstream_metrics}

            # STEP F: Log rollout table at configured intervals (now sees shaped rewards)
            if self._get_count % self._config.log_rollout_table_every_n_gets == 0:
                try:
                    from src.training.wandb_enrichment.rollout_integration import (
                        log_debate_rollout_table,
                    )
                    log_debate_rollout_table(items=items, step=self._get_count)
                except Exception as e:
                    logger.warning(f"DebateMetricStreamer: rollout table logging failed: {e}")

            # Write Parquet debug data at configured intervals (for Streamlit viewer)
            if self._config.debug_data_output_dir and self._get_count % self._config.log_rollout_table_every_n_gets == 0:
                try:
                    from src.training.wandb_enrichment.rollout_integration import (
                        write_debate_debug_data,
                    )
                    write_debate_debug_data(
                        items=items,
                        step=self._get_count,
                        output_dir=self._config.debug_data_output_dir,
                    )
                except Exception as e:
                    logger.warning(f"DebateMetricStreamer: debug data writing failed: {e}")

            # STEP G: Return (items, metrics) as before
            return items, all_metrics

        except Exception as e:
            logger.warning(f"DebateMetricStreamer: failed to compute debate metrics: {e}", exc_info=True)
            # Return upstream metrics unchanged on error
            return items, upstream_metrics

    def _compute_and_apply_shaped_rewards(
        self,
        rewards: np.ndarray,
        role_labels: list[str],
    ) -> tuple[dict[str, float], np.ndarray]:
        """Compute shaped reward metrics and per-item shaped values.

        Applies the reward shaping strategy, builds a per-item shaped reward
        array for write-back into item.data["rewards"], and logs per-role
        shaped rewards as debate/shaped_reward/* metrics. For identity strategy,
        shaped rewards equal original rewards.

        Args:
            rewards: Shape (B,) raw rewards per rollout.
            role_labels: Role label per rollout (length B).

        Returns:
            Tuple of (metrics_dict, shaped_per_item) where:
            - metrics_dict: Dict with shaped reward metrics using debate/shaped_reward/ prefix.
            - shaped_per_item: np.ndarray shape (B,) with shaped reward per item.
        """
        # Build role_masks from role_labels for strategies that need them
        batch_size = len(rewards)
        role_masks = {}
        for role in ("solver", "verifier", "judge"):
            mask = np.array([label == role for label in role_labels], dtype=bool)
            if mask.any():
                role_masks[role] = mask

        # Apply reward shaping (trajectory_metadata not available in streamer context)
        shaped = self._reward_shaper.shape_rewards(rewards, role_masks, None)

        # Convert shaped output to metrics dict and build per-item shaped array
        metrics = {}

        # Lookup dict for role -> metric constant
        _ROLE_METRIC = {
            "solver": METRIC_SHAPED_REWARD_SOLVER,
            "verifier": METRIC_SHAPED_REWARD_VERIFIER,
            "judge": METRIC_SHAPED_REWARD_JUDGE,
        }

        if isinstance(shaped, dict):
            # Per-role shaped rewards (e.g., difference_rewards, reward_mixing, coma_advantage)
            for role, role_rewards in shaped.items():
                if len(role_rewards) > 0:
                    metrics[_ROLE_METRIC.get(role, f"debate/shaped_reward/{role}")] = float(role_rewards.mean())
            # Also log overall mean shaped reward
            all_shaped = np.concatenate(list(shaped.values()))
            if len(all_shaped) > 0:
                metrics[METRIC_SHAPED_REWARD_MEAN] = float(all_shaped.mean())

            # Build per-item shaped array from per-role dict
            # Contract: all per-role strategies populate all B indices for every role key
            # (verified in difference_rewards.py, coma_advantage.py, reward_mixing.py)
            shaped_per_item = np.copy(rewards)  # start from raw as default fallback
            for idx in range(batch_size):
                role = role_labels[idx]
                if role == "judge":
                    shaped_per_item[idx] = 0.0
                elif role in shaped:
                    shaped_per_item[idx] = shaped[role][idx]
                # else: keep rewards[idx] (raw fallback -- per locked decision)
        else:
            # Global shaped rewards (e.g., identity, potential_based)
            metrics[METRIC_SHAPED_REWARD_MEAN] = float(shaped.mean())
            # Break down by role
            for role in ("solver", "verifier", "judge"):
                if role in role_masks:
                    role_shaped = shaped[role_masks[role]]
                    if len(role_shaped) > 0:
                        metrics[_ROLE_METRIC[role]] = float(role_shaped.mean())

            # Build per-item shaped array from global ndarray
            shaped_per_item = np.copy(shaped)
            for idx in range(batch_size):
                if role_labels[idx] == "judge":
                    shaped_per_item[idx] = 0.0

        # Log which strategy is active
        metrics[METRIC_SHAPED_REWARD_STRATEGY_ACTIVE] = 1.0 if self._reward_shaper.name != "identity" else 0.0

        return metrics, shaped_per_item

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
            rollout_strategy: Name of registered rollout selection strategy.
                Available: "identity" (default), "best_of_n", "self_consistency".
                Applied BEFORE reward shaping. Empty string uses identity.
            rollout_strategy_params: Strategy-specific parameters for rollout selection.
                E.g., {"agreement_threshold": 0.5} for self_consistency.
            reward_shaping_strategy: Name of registered reward shaping strategy.
                Available: "identity" (default), "difference_rewards", "reward_mixing",
                "coma_advantage", "potential_based". Empty string uses identity.
            reward_shaping_params: Strategy-specific parameters passed to the
                strategy constructor. E.g., {"alpha": 0.7} for reward_mixing,
                {"n_rollouts_per_prompt": 4} for coma_advantage.
        """

        n_rollouts_per_prompt: int = 8  # GRPO group size
        log_rollout_table_every_n_gets: int = 25  # Log W&B table every N get() calls
        debug_data_output_dir: str = ""  # Empty = disabled; set to dir path to write Parquet debug data
        rollout_strategy: str = ""  # Empty = identity passthrough (default)
        rollout_strategy_params: dict = {}  # Strategy-specific params
        reward_shaping_strategy: str = ""  # Empty = identity passthrough (default)
        reward_shaping_params: dict = {}  # Strategy-specific params

        def create_streamer(self, upstream, config, metrics_collector):
            """Create a DebateMetricStreamer instance."""
            return DebateMetricStreamer(self, upstream, metrics_collector)

    class DebateMetricStreamer(_DebateMetricStreamerImpl, flink_types.FlinkActorOutputStreamer):
        """Flink-compatible DebateMetricStreamer."""
        pass

else:
    # Stubs for local testing without Flink
    class DebateMetricStreamerConfig(components.ComponentBase):
        """Configuration for DebateMetricStreamer (test stub).

        Attributes:
            n_rollouts_per_prompt: GRPO group size (number of rollouts per prompt).
            log_rollout_table_every_n_gets: Log W&B rollout table every N get() calls.
            rollout_strategy: Name of registered rollout selection strategy.
                Available: "identity" (default), "best_of_n", "self_consistency".
                Applied BEFORE reward shaping. Empty string uses identity.
            rollout_strategy_params: Strategy-specific parameters for rollout selection.
            reward_shaping_strategy: Name of registered reward shaping strategy.
                Available: "identity" (default), "difference_rewards", "reward_mixing",
                "coma_advantage", "potential_based". Empty string uses identity.
            reward_shaping_params: Strategy-specific parameters passed to the
                strategy constructor.
        """

        n_rollouts_per_prompt: int = 8
        log_rollout_table_every_n_gets: int = 25
        debug_data_output_dir: str = ""
        rollout_strategy: str = ""
        rollout_strategy_params: dict = {}
        reward_shaping_strategy: str = ""
        reward_shaping_params: dict = {}

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
            if not hasattr(self, 'rollout_strategy'):
                self.rollout_strategy = ""
            if not hasattr(self, 'rollout_strategy_params'):
                self.rollout_strategy_params = {}
            if not hasattr(self, 'reward_shaping_strategy'):
                self.reward_shaping_strategy = ""
            if not hasattr(self, 'reward_shaping_params'):
                self.reward_shaping_params = {}

        def create_streamer(self, upstream, config, metrics_collector):
            """Create a DebateMetricStreamer instance."""
            return DebateMetricStreamer(self, upstream, metrics_collector)

    class DebateMetricStreamer(_DebateMetricStreamerImpl):
        """Test stub for DebateMetricStreamer."""
        pass
