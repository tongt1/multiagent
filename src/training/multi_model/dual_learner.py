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

from src.training.multi_model.config import MultiModelConfig
from src.training.multi_model.model_manager import MultiModelManager


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
