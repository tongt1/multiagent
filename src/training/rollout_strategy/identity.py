"""Identity (passthrough) rollout selection strategy.

Returns all rollouts unchanged, serving as the default strategy that preserves
existing behavior where all N rollouts per prompt contribute to GRPO training.

Auto-registers as "identity" on import.
"""

from __future__ import annotations

from src.training.rollout_strategy.base import RolloutStrategy
from src.training.rollout_strategy.registry import register_strategy


class IdentityRolloutStrategy(RolloutStrategy):
    """Passthrough strategy that returns all rollouts unchanged.

    This is the default strategy used when no rollout selection is configured.
    It preserves the existing behavior where all N rollouts per prompt
    contribute to GRPO gradient updates.

    Example:
        >>> strategy = IdentityRolloutStrategy()
        >>> items = [item1, item2, item3, item4]
        >>> strategy.select_rollouts(items, n_rollouts_per_prompt=4)
        [item1, item2, item3, item4]
    """

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "identity"

    def select_rollouts(
        self,
        items: list,
        n_rollouts_per_prompt: int,
    ) -> list:
        """Return all rollouts unchanged.

        Args:
            items: Flat list of actor output items
            n_rollouts_per_prompt: Ignored by identity strategy

        Returns:
            The input items list, unchanged
        """
        return items


# Auto-register on import
register_strategy("identity", IdentityRolloutStrategy)
