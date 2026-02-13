"""Reward mixing shaping strategy.

Blends global team reward G with per-role local proxy signals via
a tunable alpha coefficient:

    r_i = alpha * G + (1 - alpha) * r_i_local

Alpha controls the balance:
- alpha=1.0: pure global team reward (current behavior)
- alpha=0.0: pure local per-role reward
- alpha=0.5: equal blend (default)

Local proxy signals (r_i_local) come from trajectory_metadata:
- local_reward_solver: solver's local contribution signal
- local_reward_verifier: verifier's local contribution signal
- local_reward_judge: judge's local contribution signal

When local signals are absent, falls back to global reward G
for the local component (effectively alpha=1.0 behavior).

Auto-registers as "reward_mixing" on import.
"""

from __future__ import annotations

import numpy as np

from src.training.reward_shaping.base import RewardShaper
from src.training.reward_shaping.registry import register_strategy

_ROLES = ("solver", "verifier", "judge")


class RewardMixingShaper(RewardShaper):
    """Reward mixing: r_i = alpha*G + (1-alpha)*r_i_local.

    Blends global team reward G with per-role local proxy signals.
    Alpha controls the balance between global and local credit assignment.

    Attributes:
        alpha: Blend coefficient in [0, 1]. Default 0.5.
    """

    def __init__(self, alpha: float = 0.5, **kwargs) -> None:
        """Initialize reward mixing shaper.

        Args:
            alpha: Blend coefficient. 1.0 = pure global, 0.0 = pure local.
                Default 0.5 (equal blend).
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "reward_mixing"

    def shape_rewards(
        self,
        rewards: np.ndarray,
        role_masks: dict[str, np.ndarray] | None,
        trajectory_metadata: list[dict] | None,
    ) -> dict[str, np.ndarray]:
        """Blend global team reward with per-role local signals.

        For each rollout i and role r:
            r_r[i] = alpha * G[i] + (1 - alpha) * local_r[i]

        where local_r[i] = trajectory_metadata[i]["local_reward_{r}"],
        falling back to G[i] when absent.

        Args:
            rewards: Shape (B,) global team reward per rollout.
            role_masks: Ignored (local signals come from metadata).
            trajectory_metadata: List of B dicts, each optionally containing
                "local_reward_solver", "local_reward_verifier",
                "local_reward_judge" float values.

        Returns:
            Dict with keys "solver", "verifier", "judge", each np.ndarray
            of shape (B,) containing the blended reward for that role.
        """
        batch_size = len(rewards)

        result: dict[str, np.ndarray] = {
            role: np.zeros(batch_size, dtype=np.float64) for role in _ROLES
        }

        for i in range(batch_size):
            g = rewards[i]
            meta = (
                trajectory_metadata[i]
                if trajectory_metadata is not None and i < len(trajectory_metadata)
                else {}
            )

            for role in _ROLES:
                # Local signal, falling back to global if absent
                local_r = meta.get(f"local_reward_{role}", g)
                result[role][i] = self.alpha * g + (1 - self.alpha) * local_r

        return result


# Auto-register on import
register_strategy("reward_mixing", RewardMixingShaper)
