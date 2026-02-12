"""Gradient routing for multi-model debate training.

Routes per-token loss gradients to the correct model by masking with role-specific
boolean masks. Each model only trains on tokens produced by its assigned debate roles.

Design notes:
- Pure Python/numpy, no Flink/Comb/JAX dependencies
- Role masks are [B, T] arrays; GRPO per-token loss is [B, T-1]
  -> masks are sliced [:, :-1] to align (established in Phase 6)
- Frozen models always get zero gradient masks regardless of role assignments
- Single-model mode passes through the original loss unchanged
- Verifier and judge masks are combined (OR) for verifier_model

Key integration points:
- Uses MultiModelManager for model key queries and freeze state
- Role masks come from batch.data keys: debate/role_mask_{solver,verifier,judge}
- Per-token loss comes from GRPOLoss._compute() output
"""

from __future__ import annotations

import numpy as np

from src.training.multi_model.model_manager import MultiModelManager
from src.training.multi_model.role_router import MODEL_KEY_TO_ROLES


# Mapping from batch.data role mask keys to role names
_MASK_KEY_TO_ROLE: dict[str, str] = {
    "debate/role_mask_solver": "solver",
    "debate/role_mask_verifier": "verifier",
    "debate/role_mask_judge": "judge",
}


class GradientRouter:
    """Routes per-token loss to the correct model via role-based masking.

    For multi-model training, each model should only receive gradients from
    tokens produced by its assigned roles. The GradientRouter combines role
    masks by model key, slices them to match the GRPO objective shape, and
    applies them to the per-token loss.

    Example:
        >>> from src.training.multi_model.config import MultiModelConfig
        >>> from src.training.multi_model.model_manager import MultiModelManager
        >>> config = MultiModelConfig(
        ...     solver_ckpt="/path/solver",
        ...     verifier_ckpt="/path/verifier",
        ... )
        >>> manager = MultiModelManager(config)
        >>> router = GradientRouter(manager)
        >>> masked_losses = router.mask_loss_by_model(per_token_loss, role_masks)
    """

    def __init__(self, multi_model_manager: MultiModelManager) -> None:
        """Initialize the gradient router.

        Args:
            multi_model_manager: Manager providing model key queries, routing,
                and freeze state.
        """
        self._manager = multi_model_manager

    def mask_loss_by_model(
        self,
        per_token_loss: np.ndarray,
        role_masks: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Mask per-token loss by model, routing gradients to the correct model.

        For each model key:
        1. Combines the role masks for that model's assigned roles
        2. Slices combined mask from [B, T] to [B, T-1] to match GRPO objective shape
        3. Multiplies per_token_loss by the mask
        4. For frozen model keys, returns zero array
        5. For single-model mode, returns {"policy": per_token_loss} unchanged

        Args:
            per_token_loss: Loss array of shape [B, T-1] from GRPO objective.
            role_masks: Dict mapping batch.data keys to boolean mask arrays [B, T].
                Expected keys: "debate/role_mask_solver", "debate/role_mask_verifier",
                "debate/role_mask_judge".

        Returns:
            Dict mapping model_key -> masked loss array of shape [B, T-1].
        """
        # Single-model mode: passthrough unchanged
        if not self._manager.config.is_multi_model:
            return {"policy": per_token_loss}

        # Convert batch.data keys to role names
        role_to_mask = self._resolve_role_masks(role_masks)

        frozen_keys = set(self._manager.get_frozen_model_keys())
        result: dict[str, np.ndarray] = {}

        for model_key in self._manager.model_keys:
            if model_key in frozen_keys:
                # Frozen model: zero loss
                result[model_key] = np.zeros_like(per_token_loss)
                continue

            # Combine masks for this model's roles
            combined_mask = self._combine_role_masks(model_key, role_to_mask)

            # Slice from [B, T] to [B, T-1] to match per_token_loss shape
            sliced_mask = combined_mask[:, :-1].astype(per_token_loss.dtype)

            # Apply mask to loss
            result[model_key] = per_token_loss * sliced_mask

        return result

    def should_compute_gradient(self, model_key: str) -> bool:
        """Check whether gradient computation is needed for a model key.

        Args:
            model_key: The model key to check (e.g., "solver_model").

        Returns:
            False if the model key is frozen, True otherwise.
        """
        return model_key not in self._manager.get_frozen_model_keys()

    def get_gradient_mask(
        self,
        model_key: str,
        role_masks: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Get the combined and sliced gradient mask for a specific model key.

        Args:
            model_key: The model key to get the mask for.
            role_masks: Dict mapping batch.data keys to boolean mask arrays [B, T].

        Returns:
            Boolean mask array of shape [B, T-1]. Zero mask for frozen models.
        """
        role_to_mask = self._resolve_role_masks(role_masks)

        # Check if frozen
        if model_key in self._manager.get_frozen_model_keys():
            # Return zero mask with shape [B, T-1]
            any_mask = next(iter(role_to_mask.values()))
            return np.zeros((any_mask.shape[0], any_mask.shape[1] - 1), dtype=bool)

        # Combine masks for this model's roles and slice
        combined = self._combine_role_masks(model_key, role_to_mask)
        return combined[:, :-1]

    def _resolve_role_masks(
        self, role_masks: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert batch.data mask keys to role-name-keyed masks.

        Args:
            role_masks: Dict with keys like "debate/role_mask_solver".

        Returns:
            Dict mapping role names ("solver", "verifier", "judge") to mask arrays.
        """
        resolved: dict[str, np.ndarray] = {}
        for key, mask in role_masks.items():
            if key in _MASK_KEY_TO_ROLE:
                resolved[_MASK_KEY_TO_ROLE[key]] = mask
            else:
                # Already a role name (for backward compatibility)
                resolved[key] = mask
        return resolved

    def _combine_role_masks(
        self,
        model_key: str,
        role_to_mask: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Combine role masks for all roles assigned to a model key.

        Args:
            model_key: The model key (e.g., "solver_model", "verifier_model").
            role_to_mask: Dict mapping role names to boolean mask arrays.

        Returns:
            Combined boolean mask array (OR of all assigned role masks).
        """
        roles = MODEL_KEY_TO_ROLES.get(model_key, [])
        combined: np.ndarray | None = None

        for role in roles:
            if role in role_to_mask:
                if combined is None:
                    combined = role_to_mask[role].copy()
                else:
                    combined = combined | role_to_mask[role]

        if combined is None:
            # No matching roles found -- return zero mask
            any_mask = next(iter(role_to_mask.values()))
            return np.zeros_like(any_mask)

        return combined
