"""Abstract base class for reward shaping strategies.

All reward shaping strategies (identity, difference rewards, COMA, etc.)
must implement this interface. The base class enforces a consistent API
for transforming raw binary SymPy rewards into shaped rewards that can
drive more nuanced multi-agent credit assignment.

Design notes:
- Pure numpy, no Flink/Comb/JAX dependencies (matching debate_metrics.py pattern)
- shape_rewards accepts full batch context (rewards + role_masks + metadata)
  so that strategies like difference rewards and COMA can access per-role structure
- Default return type is np.ndarray (B,) for backward compatibility;
  per-role dict is opt-in for strategies that need different rewards per role
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RewardShaper(ABC):
    """Abstract base class for reward shaping strategies.

    All reward shaping strategies must subclass this and implement
    shape_rewards() and the name property.

    Example:
        >>> class MyShaper(RewardShaper):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_strategy"
        ...     def shape_rewards(self, rewards, role_masks, trajectory_metadata):
        ...         return rewards * 2.0
        >>> shaper = MyShaper()
        >>> shaper.shape_rewards(np.array([0.0, 5.0]), None, None)
        array([ 0., 10.])
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
    def shape_rewards(
        self,
        rewards: np.ndarray,
        role_masks: dict[str, np.ndarray] | None,
        trajectory_metadata: list[dict] | None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Transform raw rewards into shaped rewards.

        Args:
            rewards: Shape (B,) flat reward per rollout. Currently binary
                0/5 from SymPy correctness checking.
            role_masks: Optional dict of per-role boolean masks.
                Keys: "solver", "verifier", "judge"
                Values: shape (B, T) boolean arrays indicating which tokens
                belong to each role.
            trajectory_metadata: Optional list of per-rollout metadata dicts.
                Length B. May contain counterfactual info, problem difficulty,
                turn structure, etc.

        Returns:
            Either:
            - np.ndarray shape (B,): global reshaped rewards (backward compatible)
            - dict[str, np.ndarray]: per-role rewards with keys like
              "solver", "verifier", "judge", each shape (B,)
        """
        ...
