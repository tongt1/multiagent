"""Dual learner configuration for independent per-model optimizer state.

Defines DualLearnerConfig and PerModelOptimizerConfig for managing separate
optimizer parameters (learning rate, warmup, weight decay) for each model
in multi-model debate training.

Design notes:
- Pure Python dataclass, no Flink/JAX dependencies (matching config.py pattern)
- Each model key (solver_model, verifier_model) can have independent optimizer params
- Frozen models are excluded from optimizer configs (no Adam moments needed)
- In single-model mode, falls back to a single optimizer config keyed by "policy"
- Per-model LR schedule params can be queried for schedule construction
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.training.multi_model.advantage_alignment import compute_aligned_advantages
from src.training.multi_model.config import MultiModelConfig
from src.training.multi_model.model_manager import MultiModelManager

# Reward strategy is optional -- may not be available on cluster
try:
    from src.training.multi_model.reward_strategy_adapter import RewardStrategyAdapter
    _HAS_REWARD_STRATEGY = True
except ImportError:
    _HAS_REWARD_STRATEGY = False
    RewardStrategyAdapter = None


@dataclass
class PerModelOptimizerConfig:
    """Optimizer configuration for a single model.

    Stores the learning rate schedule parameters and weight decay for one
    model's optimizer (e.g., solver_model or verifier_model).

    Attributes:
        peak_lr: Peak learning rate after warmup.
        end_lr: Final learning rate after decay.
        warmup_steps: Number of steps for linear warmup.
        weight_decay: L2 weight decay coefficient.
    """

    peak_lr: float = 3e-6
    end_lr: float = 3e-6
    warmup_steps: int = 0
    weight_decay: float = 0.0


@dataclass
class DualLearnerConfig:
    """Configuration for dual-learner training with independent optimizer state per model.

    Manages per-model optimizer parameters and learning rate schedules. In multi-model
    mode, each model key (solver_model, verifier_model) can have separate peak_lr,
    end_lr, warmup_steps, and weight_decay. Frozen models are excluded from optimizer
    configs entirely (no Adam moments allocated).

    In single-model mode, a single optimizer config is returned keyed by "policy".

    Attributes:
        multi_model_config: The underlying MultiModelConfig specifying model keys
            and freeze configuration.
        base_peak_lr: Default peak learning rate when no per-model override is set.
        base_end_lr: Default end learning rate when no per-model override is set.
        solver_optimizer: Optional optimizer override for the solver model.
        verifier_optimizer: Optional optimizer override for the verifier model.
    """

    multi_model_config: MultiModelConfig
    base_peak_lr: float = 3e-6
    base_end_lr: float = 3e-6
    solver_optimizer: PerModelOptimizerConfig | None = None
    verifier_optimizer: PerModelOptimizerConfig | None = None

    def __post_init__(self) -> None:
        """Initialize the internal model manager."""
        self._manager = MultiModelManager(self.multi_model_config)

    def get_optimizer_configs(self) -> dict[str, PerModelOptimizerConfig]:
        """Return optimizer configs for non-frozen model keys only.

        For each active (non-frozen) model key, returns a PerModelOptimizerConfig.
        Uses per-model override if set, otherwise creates a config from base LR values.

        Returns:
            Dict mapping model_key -> PerModelOptimizerConfig for trainable models.
            Frozen model keys are excluded (they need no Adam state).
        """
        trainable_keys = self._manager.get_trainable_model_keys()
        result: dict[str, PerModelOptimizerConfig] = {}

        for model_key in trainable_keys:
            override = self._get_override_for_key(model_key)
            if override is not None:
                result[model_key] = override
            else:
                # Create default config from base LR values
                result[model_key] = PerModelOptimizerConfig(
                    peak_lr=self.base_peak_lr,
                    end_lr=self.base_end_lr,
                )

        return result

    def get_active_model_keys(self) -> list[str]:
        """Return model keys that need gradient computation (non-frozen).

        Returns:
            List of model key strings that are trainable (not fully frozen).
        """
        return self._manager.get_trainable_model_keys()

    def get_lr_schedule_params(self, model_key: str) -> dict | None:
        """Return LR schedule parameters for a given model key.

        Args:
            model_key: The model key to query (e.g., "solver_model", "verifier_model").

        Returns:
            Dict with keys "peak_lr", "end_lr", "warmup_steps" for active models.
            None if the model key is frozen (no schedule needed).
        """
        # Check if frozen
        frozen_keys = self._manager.get_frozen_model_keys()
        if model_key in frozen_keys:
            return None

        # Get the optimizer config for this key
        override = self._get_override_for_key(model_key)
        if override is not None:
            return {
                "peak_lr": override.peak_lr,
                "end_lr": override.end_lr,
                "warmup_steps": override.warmup_steps,
            }

        # Fall back to base values
        return {
            "peak_lr": self.base_peak_lr,
            "end_lr": self.base_end_lr,
            "warmup_steps": 0,
        }

    def align_advantages(
        self,
        solver_advantages: np.ndarray,
        verifier_advantages: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply advantage alignment if enabled in config.

        Implements the integration point described in AA-02: advantage alignment
        loss plugs into the dual learner's GRPO objective as a configurable option.

        Args:
            solver_advantages: GRPO advantages for solver, shape [B].
            verifier_advantages: GRPO advantages for verifier, shape [B].

        Returns:
            Tuple of (aligned_solver_advantages, aligned_verifier_advantages).
            If alignment is disabled or single-model mode, returns inputs unchanged.
        """
        aa_config = self.multi_model_config.advantage_alignment
        if not aa_config.enabled or not self.multi_model_config.is_multi_model:
            return solver_advantages, verifier_advantages
        return compute_aligned_advantages(solver_advantages, verifier_advantages, aa_config)

    def compute_alignment_metrics(
        self,
        solver_adv: np.ndarray,
        verifier_adv: np.ndarray,
        aligned_solver_adv: np.ndarray,
        aligned_verifier_adv: np.ndarray,
    ) -> dict[str, float]:
        """Compute advantage alignment monitoring metrics.

        Returns metrics tracking the magnitude of the alignment correction,
        aligned advantage means, and configuration values for W&B logging.

        Args:
            solver_adv: Original solver advantages, shape [B].
            verifier_adv: Original verifier advantages, shape [B].
            aligned_solver_adv: Aligned solver advantages, shape [B].
            aligned_verifier_adv: Aligned verifier advantages, shape [B].

        Returns:
            Dict of alignment metrics. Empty dict if alignment is disabled
            or single-model mode.
        """
        aa_config = self.multi_model_config.advantage_alignment
        if not aa_config.enabled or not self.multi_model_config.is_multi_model:
            return {}
        return {
            "advantage_alignment/solver_alignment_term": float(
                np.mean(aligned_solver_adv - solver_adv)
            ),
            "advantage_alignment/verifier_alignment_term": float(
                np.mean(aligned_verifier_adv - verifier_adv)
            ),
            "advantage_alignment/solver_aligned_mean": float(
                np.mean(aligned_solver_adv)
            ),
            "advantage_alignment/verifier_aligned_mean": float(
                np.mean(aligned_verifier_adv)
            ),
            "advantage_alignment/beta": float(aa_config.beta),
            "advantage_alignment/enabled": 1.0,
        }

    def shape_rewards(
        self,
        rewards: np.ndarray,
        role_masks: dict[str, np.ndarray] | None = None,
        trajectory_metadata: list[dict] | None = None,
    ) -> dict[str, np.ndarray]:
        """Apply reward shaping strategy to batch rewards.

        Uses the RewardStrategyAdapter to transform raw rewards through
        the configured strategy. Same strategy is applied to both roles.
        Frozen roles receive passthrough (unshaped) rewards.

        Pipeline position: called BEFORE advantage computation.
        raw_rewards -> shape_rewards() -> compute_advantages -> align_advantages()

        Args:
            rewards: Shape (B,) raw reward per rollout.
            role_masks: Optional per-role boolean masks from batch.data.
            trajectory_metadata: Optional per-rollout metadata.

        Returns:
            Dict with keys "solver", "verifier", "judge", each np.ndarray (B,).
            If reward strategy is not available, returns passthrough rewards for all roles.
        """
        if not _HAS_REWARD_STRATEGY:
            # Passthrough: same rewards for all roles when adapter not available
            return {"solver": rewards.copy(), "verifier": rewards.copy(), "judge": rewards.copy()}

        if not hasattr(self, "_reward_adapter"):
            self._reward_adapter = RewardStrategyAdapter(
                config=self.multi_model_config.reward_strategy,
                freeze_roles=list(self.multi_model_config.freeze_roles),
            )
        return self._reward_adapter.shape_rewards(rewards, role_masks, trajectory_metadata)

    def get_reward_strategy_name(self) -> str:
        """Return the active reward strategy name for W&B tagging."""
        if not _HAS_REWARD_STRATEGY:
            return "passthrough"

        if not hasattr(self, "_reward_adapter"):
            self._reward_adapter = RewardStrategyAdapter(
                config=self.multi_model_config.reward_strategy,
                freeze_roles=list(self.multi_model_config.freeze_roles),
            )
        return self._reward_adapter.name

    def _get_override_for_key(self, model_key: str) -> PerModelOptimizerConfig | None:
        """Get the per-model optimizer override for a model key, if any.

        Args:
            model_key: The model key to look up.

        Returns:
            The PerModelOptimizerConfig override, or None if no override is set.
        """
        if model_key == "solver_model":
            return self.solver_optimizer
        elif model_key == "verifier_model":
            return self.verifier_optimizer
        # Single-model "policy" key: check solver_optimizer as fallback
        elif model_key == "policy":
            return self.solver_optimizer
        return None
