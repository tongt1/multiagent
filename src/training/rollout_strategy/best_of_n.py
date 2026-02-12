"""Best-of-N rollout selection strategy.

Selects the highest-reward rollout per prompt group from N candidates.
This improves training signal quality by filtering out low-quality rollouts
before they contribute to GRPO gradient updates.

Auto-registers as "best_of_n" on import.
"""

from __future__ import annotations

import logging

import numpy as np

from src.training.rollout_strategy.base import RolloutStrategy
from src.training.rollout_strategy.registry import register_strategy

logger = logging.getLogger(__name__)


class BestOfNStrategy(RolloutStrategy):
    """Select the highest-reward rollout per prompt group.

    Given N rollouts per prompt, selects exactly 1 rollout per prompt --
    the one with the highest reward. This concentrates training signal
    on the best-performing rollouts.

    Uses np.argmax for deterministic tie-breaking (first highest wins).

    Example:
        >>> strategy = BestOfNStrategy()
        >>> # 2 prompts x 4 rollouts
        >>> result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)
        >>> len(result)  # 2 items (1 per prompt)
        2
    """

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "best_of_n"

    def select_rollouts(
        self,
        items: list,
        n_rollouts_per_prompt: int,
    ) -> list:
        """Select highest-reward rollout per prompt group.

        Args:
            items: Flat list of actor output items. Each item has
                item.data["rewards"] as a numpy scalar with the rollout reward.
            n_rollouts_per_prompt: Number of rollouts per prompt (GRPO group size).

        Returns:
            List of P items (one best rollout per prompt), where
            P = len(items) // n_rollouts_per_prompt.
        """
        n_items = len(items)
        n_prompts = n_items // n_rollouts_per_prompt

        # Handle incomplete group at end
        remainder = n_items % n_rollouts_per_prompt
        if remainder != 0:
            logger.warning(
                "Incomplete rollout group: %d items with n_rollouts_per_prompt=%d "
                "leaves %d leftover items. Truncating to %d complete groups (%d items).",
                n_items,
                n_rollouts_per_prompt,
                remainder,
                n_prompts,
                n_prompts * n_rollouts_per_prompt,
            )

        result = []
        for i in range(n_prompts):
            group_start = i * n_rollouts_per_prompt
            group_end = group_start + n_rollouts_per_prompt
            group = items[group_start:group_end]

            # Extract rewards and find argmax
            rewards = np.array([item.data["rewards"].item() for item in group])
            best_idx = int(np.argmax(rewards))

            result.append(group[best_idx])

        return result


# Auto-register on import
register_strategy("best_of_n", BestOfNStrategy)
