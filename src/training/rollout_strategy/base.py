"""Abstract base class for rollout selection strategies.

All rollout selection strategies (identity, best-of-N, etc.) must implement
this interface. The base class enforces a consistent API for selecting which
rollouts from a GRPO batch contribute to gradient updates.

Design notes:
- Pure numpy, no Flink/Comb/JAX dependencies (matching reward_shaping pattern)
- select_rollouts receives a flat list of items grouped by prompt and filters
  them according to the strategy's selection criteria
- Items arrive as N*P flat list where N is rollouts per prompt and P is prompts
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RolloutStrategy(ABC):
    """Abstract base class for rollout selection strategies.

    All rollout selection strategies must subclass this and implement
    select_rollouts() and the name property.

    Example:
        >>> class MyStrategy(RolloutStrategy):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_strategy"
        ...     def select_rollouts(self, items, n_rollouts_per_prompt):
        ...         return items  # passthrough
        >>> strategy = MyStrategy()
        >>> strategy.select_rollouts([item1, item2], n_rollouts_per_prompt=2)
        [item1, item2]
    """

    def __init__(self, **kwargs) -> None:
        """Initialize with arbitrary config parameters.

        Subclasses can accept strategy-specific params via **kwargs.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the registered name of this strategy."""
        ...

    @abstractmethod
    def select_rollouts(
        self,
        items: list,
        n_rollouts_per_prompt: int,
    ) -> list:
        """Select rollouts from a batch of items.

        Items arrive as a flat list of N*P items where N is rollouts per prompt
        and P is the number of prompts. Items are grouped consecutively: the first
        N items belong to prompt 0, the next N to prompt 1, etc.

        Args:
            items: Flat list of actor output items. Each item has item.data["rewards"]
                as a numpy scalar with the rollout reward.
            n_rollouts_per_prompt: Number of rollouts per prompt (GRPO group size).

        Returns:
            Filtered list of items to use as training signal. Length depends on
            strategy (e.g., best-of-N returns P items, identity returns all N*P).
        """
        ...
