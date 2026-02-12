"""Difference rewards shaping strategy (stub for TDD RED phase)."""

from __future__ import annotations

import numpy as np

from src.training.reward_shaping.base import RewardShaper
from src.training.reward_shaping.registry import register_strategy


class DifferenceRewardShaper(RewardShaper):
    """Stub: not yet implemented."""

    @property
    def name(self) -> str:
        return "difference_rewards"

    def shape_rewards(
        self,
        rewards: np.ndarray,
        role_masks: dict[str, np.ndarray] | None,
        trajectory_metadata: list[dict] | None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        raise NotImplementedError("TDD RED phase: not yet implemented")


register_strategy("difference_rewards", DifferenceRewardShaper)
