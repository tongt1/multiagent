"""Difference rewards shaping strategy.

Computes each agent's marginal contribution D_i(z) = G(z) - G(z_{-i})
by comparing the team reward with the counterfactual reward obtained
when agent i's contribution is removed.

In online RL (no pre-generated trajectories), z_{-i} is approximated by:
1. Taking the existing debate trajectory
2. Masking agent i's tokens (setting them to padding/empty)
3. Re-evaluating the reward on the masked trajectory

Since the Comb environment computes reward on the final answer only
(SmartAnswerValidator on last chatbot turn), the counterfactual for
removing the judge is G(z_{-judge}) = 0 (no final answer).
For removing solver/verifier, the counterfactual depends on whether
the remaining agents can still produce a correct final answer.

Simplification for binary reward:
With binary reward (5=correct, 0=incorrect), the counterfactual
G(z_{-i}) is estimated from trajectory_metadata which carries
per-agent counterfactual indicators computed during rollout.
If no counterfactual data is available, falls back to G(z) for all.

Auto-registers as "difference_rewards" on import.
"""

from __future__ import annotations

import numpy as np

from src.training.reward_shaping.base import RewardShaper
from src.training.reward_shaping.registry import register_strategy

_ROLES = ("solver", "verifier", "judge")


class DifferenceRewardShaper(RewardShaper):
    """Difference rewards: D_i(z) = G(z) - G(z_{-i}).

    Computes each agent's marginal contribution by comparing the team reward
    with the counterfactual reward obtained when agent i's contribution is
    removed (masked).

    Counterfactual rewards G(z_{-i}) come from trajectory_metadata under
    keys ``counterfactual_solver``, ``counterfactual_verifier``,
    ``counterfactual_judge``. When metadata is absent, falls back to
    returning G(z) for all roles (no counterfactual possible).
    """

    def __init__(self, **kwargs) -> None:
        """Initialize difference rewards shaper.

        No special params needed beyond base class.
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "difference_rewards"

    def shape_rewards(
        self,
        rewards: np.ndarray,
        role_masks: dict[str, np.ndarray] | None,
        trajectory_metadata: list[dict] | None,
    ) -> dict[str, np.ndarray]:
        """Compute per-agent difference rewards.

        For each rollout i and role r:
            D_r[i] = G[i] - G_{-r}[i]

        where G_{-r}[i] is the counterfactual reward without role r,
        read from trajectory_metadata[i]["counterfactual_{r}"].

        Args:
            rewards: Shape (B,) team reward per rollout.
            role_masks: Ignored (counterfactual data comes from metadata).
            trajectory_metadata: List of B dicts, each optionally containing
                "counterfactual_solver", "counterfactual_verifier",
                "counterfactual_judge" float values.

        Returns:
            Dict with keys "solver", "verifier", "judge", each np.ndarray
            of shape (B,) containing the difference reward for that role.
        """
        batch_size = len(rewards)

        # Fallback: no metadata means no counterfactual data available
        if trajectory_metadata is None or len(trajectory_metadata) == 0:
            return {role: rewards.copy() for role in _ROLES}

        # Compute per-role difference rewards
        result: dict[str, np.ndarray] = {
            role: np.zeros(batch_size, dtype=np.float64) for role in _ROLES
        }

        for i in range(batch_size):
            g = rewards[i]
            meta = trajectory_metadata[i] if i < len(trajectory_metadata) else {}

            for role in _ROLES:
                # Counterfactual: reward without this role's contribution
                g_minus_r = meta.get(f"counterfactual_{role}", g)
                result[role][i] = g - g_minus_r

        return result


# Auto-register on import
register_strategy("difference_rewards", DifferenceRewardShaper)
