"""Role-to-model routing for multi-model debate training.

Maps debate roles (solver, verifier, judge) to model keys (solver_model,
verifier_model) and combines per-role batch masks by model key for gradient
routing.

In multi-model mode:
- solver -> solver_model
- verifier, judge -> verifier_model (shared verifier/judge model)

In single-model mode:
- all roles -> policy (the default Flink model key)

Design notes:
- Pure Python/numpy, no Flink/Comb/JAX dependencies
- verifier and judge share a model because both evaluate solver outputs
- "policy" is the standard Flink key for the single trainable model
"""

from __future__ import annotations

import numpy as np

# Multi-model role-to-key mapping
ROLE_TO_MODEL_KEY: dict[str, str] = {
    "solver": "solver_model",
    "verifier": "verifier_model",
    "judge": "verifier_model",
}

# Default Flink model key for single-model mode
SINGLE_MODEL_KEY = "policy"

# Inverse mapping: model_key -> list of roles assigned to it
MODEL_KEY_TO_ROLES: dict[str, list[str]] = {
    "solver_model": ["solver"],
    "verifier_model": ["verifier", "judge"],
}


class RoleRouter:
    """Routes debate roles to model keys and combines batch masks by model.

    In multi-model mode, solver tokens go to solver_model and verifier/judge
    tokens go to verifier_model. In single-model mode, all tokens go to the
    default "policy" key.

    Example:
        >>> router = RoleRouter(is_multi_model=True)
        >>> router.get_model_key("solver")
        'solver_model'
        >>> router.get_model_key("judge")
        'verifier_model'
    """

    def __init__(self, is_multi_model: bool) -> None:
        """Initialize the router.

        Args:
            is_multi_model: If True, use per-role model keys. If False,
                all roles map to the single "policy" key.
        """
        self._is_multi_model = is_multi_model

    @property
    def is_multi_model(self) -> bool:
        """Whether routing is in multi-model mode."""
        return self._is_multi_model

    def get_model_key(self, role: str) -> str:
        """Get the model key for a given debate role.

        Args:
            role: Debate role name ("solver", "verifier", or "judge").

        Returns:
            Model key string. In multi-model mode, returns the role-specific
            key. In single-model mode, always returns "policy".
        """
        if not self._is_multi_model:
            return SINGLE_MODEL_KEY
        return ROLE_TO_MODEL_KEY[role]

    def route_batch_masks(
        self, role_masks: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Combine per-role masks into per-model masks.

        For multi-model mode:
        - solver_model gets the solver mask
        - verifier_model gets verifier_mask | judge_mask

        For single-model mode:
        - policy gets all masks OR'd together

        Args:
            role_masks: Dict mapping role names to boolean mask arrays.
                Expected keys: "solver", "verifier", "judge".
                Values: np.ndarray of shape (B, T) or (B,).

        Returns:
            Dict mapping model keys to combined boolean mask arrays.
        """
        if not self._is_multi_model:
            # Single model: OR all masks together
            combined = None
            for mask in role_masks.values():
                if combined is None:
                    combined = mask.copy()
                else:
                    combined = combined | mask
            return {SINGLE_MODEL_KEY: combined}

        # Multi-model: group masks by model key
        result: dict[str, np.ndarray] = {}
        for role, mask in role_masks.items():
            model_key = ROLE_TO_MODEL_KEY[role]
            if model_key not in result:
                result[model_key] = mask.copy()
            else:
                result[model_key] = result[model_key] | mask

        return result
