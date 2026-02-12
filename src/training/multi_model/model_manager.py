"""Multi-model manager for debate RL training.

Provides a high-level interface for managing multiple model identifiers
within a single training job. Determines which model keys are trainable
vs frozen based on the freeze_roles configuration.

Design notes:
- Pure Python, no Flink/JAX dependencies (matching reward_shaping pattern)
- In single-model mode, exposes a single "policy" key (standard Flink behavior)
- In multi-model mode, exposes "solver_model" and "verifier_model" keys
- Freeze logic: a model key is frozen iff ALL its assigned roles are frozen
  (e.g., verifier_model is frozen only when BOTH "verifier" and "judge" are frozen)
"""

from __future__ import annotations

from src.training.multi_model.config import MultiModelConfig
from src.training.multi_model.role_router import (
    MODEL_KEY_TO_ROLES,
    RoleRouter,
    SINGLE_MODEL_KEY,
)


class MultiModelManager:
    """Manages model identifiers and freeze state for multi-model training.

    Wraps a MultiModelConfig and RoleRouter to provide queries about which
    model keys exist, which are trainable, and which are frozen.

    Example:
        >>> from src.training.multi_model.config import MultiModelConfig
        >>> config = MultiModelConfig(
        ...     solver_ckpt="/path/solver",
        ...     verifier_ckpt="/path/verifier",
        ...     freeze_roles=["verifier", "judge"],
        ... )
        >>> manager = MultiModelManager(config)
        >>> manager.model_keys
        ['solver_model', 'verifier_model']
        >>> manager.get_trainable_model_keys()
        ['solver_model']
        >>> manager.get_frozen_model_keys()
        ['verifier_model']
    """

    def __init__(self, config: MultiModelConfig) -> None:
        """Initialize the manager.

        Args:
            config: Multi-model training configuration specifying checkpoints
                and freeze roles.
        """
        self._config = config
        self._router = RoleRouter(is_multi_model=config.is_multi_model)

    @property
    def config(self) -> MultiModelConfig:
        """The underlying multi-model configuration."""
        return self._config

    @property
    def router(self) -> RoleRouter:
        """The role router for mapping roles to model keys."""
        return self._router

    @property
    def model_keys(self) -> list[str]:
        """Return distinct model keys for the current configuration.

        Returns:
            In multi-model mode: ["solver_model", "verifier_model"]
            In single-model mode: ["policy"]
        """
        if self._config.is_multi_model:
            return ["solver_model", "verifier_model"]
        return [SINGLE_MODEL_KEY]

    def is_role_frozen(self, role: str) -> bool:
        """Check if a specific role is frozen (receives no gradient updates).

        Args:
            role: Debate role name ("solver", "verifier", or "judge").

        Returns:
            True if the role is in the config's freeze_roles list.
        """
        return role in self._config.freeze_roles

    def get_trainable_model_keys(self) -> list[str]:
        """Return model keys where at least one assigned role is NOT frozen.

        A model key is trainable if any of its roles are not in freeze_roles.
        For example, if only "verifier" is frozen but not "judge", then
        "verifier_model" is still trainable (judge is trainable).

        Returns:
            List of trainable model key strings.
        """
        if not self._config.is_multi_model:
            # Single model: trainable if any role is not frozen
            frozen = set(self._config.freeze_roles)
            if frozen >= {"solver", "verifier", "judge"}:
                return []
            return [SINGLE_MODEL_KEY]

        trainable = []
        for model_key in self.model_keys:
            roles = MODEL_KEY_TO_ROLES[model_key]
            # Trainable if at least one role is NOT frozen
            if any(role not in self._config.freeze_roles for role in roles):
                trainable.append(model_key)
        return trainable

    def get_frozen_model_keys(self) -> list[str]:
        """Return model keys where ALL assigned roles are frozen.

        A model key is fully frozen only when every role mapped to it
        is in freeze_roles. For verifier_model, both "verifier" and "judge"
        must be frozen.

        Returns:
            List of frozen model key strings.
        """
        if not self._config.is_multi_model:
            frozen = set(self._config.freeze_roles)
            if frozen >= {"solver", "verifier", "judge"}:
                return [SINGLE_MODEL_KEY]
            return []

        frozen_keys = []
        for model_key in self.model_keys:
            roles = MODEL_KEY_TO_ROLES[model_key]
            # Frozen if ALL roles are frozen
            if all(role in self._config.freeze_roles for role in roles):
                frozen_keys.append(model_key)
        return frozen_keys
