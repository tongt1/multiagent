"""Identity (passthrough) reward shaping strategy.

Returns rewards unchanged, serving as the default strategy that preserves
existing binary SymPy reward behavior (0=incorrect, 5=correct).

Auto-registers as "identity" on import.
"""

from __future__ import annotations

import numpy as np

from src.training.reward_shaping.base import RewardShaper
from src.training.reward_shaping.registry import register_strategy


class IdentityRewardShaper(RewardShaper):
    """Passthrough strategy that returns rewards unchanged.

    This is the default strategy used when no reward shaping is configured.
    It preserves the existing binary SymPy reward behavior.

    Example:
        >>> shaper = IdentityRewardShaper()
        >>> rewards = np.array([0.0, 5.0, 0.0, 5.0])
        >>> shaper.shape_rewards(rewards, None, None)
        array([0., 5., 0., 5.])
    """

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "identity"

    def shape_rewards(
        self,
        rewards: np.ndarray,
        role_masks: dict[str, np.ndarray] | None,
        trajectory_metadata: list[dict] | None,
    ) -> np.ndarray:
        """Return rewards unchanged.

        Args:
            rewards: Shape (B,) flat reward per rollout
            role_masks: Ignored by identity strategy
            trajectory_metadata: Ignored by identity strategy

        Returns:
            The input rewards array, unchanged
        """
        return rewards


# Auto-register on import
register_strategy("identity", IdentityRewardShaper)
