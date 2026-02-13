"""Potential-based reward shaping strategy.

Implements r' = r + gamma * Phi(s') - Phi(s), adding a shaping term based
on a potential function Phi that encodes domain knowledge about desirable
states. This preserves optimal policy guarantees: the optimal policy under
shaped rewards is identical to the optimal policy under original rewards.

Reference: Ng, Harada, Russell 1999 -- "Policy invariance under reward
transformations: theory and application to reward shaping."

Built-in potential functions:
- "zero": Phi=0 (no shaping, identity behavior)
- "debate_length": Phi = -penalty * num_turns (penalizes long debates)
- "correctness_progress": Phi based on answer quality in intermediate turns

Custom potentials can be registered via potential_fn parameter.

Auto-registers as "potential_based" on import.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.training.reward_shaping.base import RewardShaper
from src.training.reward_shaping.registry import register_strategy


def _phi_zero(state_info: dict) -> float:
    """Zero potential function. Always returns 0.0 (no shaping)."""
    return 0.0


def _phi_debate_length(state_info: dict, penalty: float = 0.1) -> float:
    """Debate length potential: penalizes longer debates.

    Phi(s) = -penalty * num_turns

    Encourages efficient debate resolution by penalizing states
    with more turns.

    Args:
        state_info: Dict with optional "num_turns" key.
        penalty: Penalty per turn (default 0.1).

    Returns:
        Negative penalty proportional to debate length.
    """
    return -penalty * state_info.get("num_turns", 0)


def _phi_correctness_progress(state_info: dict) -> float:
    """Correctness progress potential: rewards intermediate correct answers.

    Phi(s) = 0.5 * intermediate_correct_count / max(total_turns, 1)

    Rewards states where intermediate turns contain correct answer
    elements, encouraging the debate to converge toward correctness.

    Args:
        state_info: Dict with optional "intermediate_correct_count"
            and "total_turns" keys.

    Returns:
        Progress score in [0, 0.5].
    """
    correct = state_info.get("intermediate_correct_count", 0)
    total = max(state_info.get("total_turns", 1), 1)
    return 0.5 * correct / total


# Map of built-in potential function names to their implementations
_BUILT_IN_POTENTIALS: dict[str, Callable] = {
    "zero": _phi_zero,
    "debate_length": _phi_debate_length,
    "correctness_progress": _phi_correctness_progress,
}


class PotentialBasedShaper(RewardShaper):
    """Potential-based reward shaping: r' = r + gamma*Phi(s') - Phi(s).

    Adds a shaping term based on a potential function Phi that encodes
    domain knowledge about desirable states. This preserves optimal
    policy guarantees (Ng, Harada, Russell 1999): the optimal policy
    under shaped rewards is identical to the optimal policy under
    original rewards.

    Built-in potential functions:
    - "zero": Phi=0 (no shaping, identity behavior)
    - "debate_length": Phi = -penalty * num_turns (penalizes long debates)
    - "correctness_progress": Phi based on answer quality in intermediate turns

    Custom potentials can be registered via potential_fn parameter.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        potential_type: str = "zero",
        potential_fn: Callable[[dict], float] | None = None,
        penalty: float = 0.1,
        **kwargs,
    ) -> None:
        """Initialize potential-based shaper.

        Args:
            gamma: Discount factor for scaling the potential difference.
                Controls how much future potential is valued.
            potential_type: Name of built-in potential function to use.
                One of "zero", "debate_length", "correctness_progress".
                Ignored if potential_fn is provided.
            potential_fn: Custom callable Phi(state_info) -> float.
                Overrides potential_type if provided.
            penalty: Penalty parameter for "debate_length" potential.
                Only used when potential_type="debate_length".
            **kwargs: Additional parameters passed to base class.
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.penalty = penalty

        if potential_fn is not None:
            self._phi = potential_fn
        elif potential_type == "debate_length":
            # Bind penalty parameter for debate_length
            self._phi = lambda state_info: _phi_debate_length(
                state_info, penalty=self.penalty
            )
        elif potential_type in _BUILT_IN_POTENTIALS:
            self._phi = _BUILT_IN_POTENTIALS[potential_type]
        else:
            raise ValueError(
                f"Unknown potential_type '{potential_type}'. "
                f"Available: {list(_BUILT_IN_POTENTIALS.keys())}. "
                f"Or pass a custom potential_fn callable."
            )

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "potential_based"

    def shape_rewards(
        self,
        rewards: np.ndarray,
        role_masks: dict[str, np.ndarray] | None,
        trajectory_metadata: list[dict] | None,
    ) -> np.ndarray:
        """Apply potential-based reward shaping.

        For each rollout i:
            r'[i] = r[i] + gamma * Phi(s'[i]) - Phi(s[i])

        Where s and s' are extracted from trajectory_metadata state_start
        and state_end dicts. If no metadata, Phi defaults to zero (no shaping).

        Args:
            rewards: Shape (B,) flat reward per rollout.
            role_masks: Optional per-role boolean masks (not used,
                potential shaping applies uniformly to team reward).
            trajectory_metadata: Optional list of per-rollout metadata dicts.
                May contain "state_start" and "state_end" dicts with state info.

        Returns:
            np.ndarray shape (B,) of shaped rewards.
        """
        if trajectory_metadata is None:
            # No metadata: Phi defaults to zero, reward unchanged
            return rewards.copy()

        batch_size = len(rewards)
        shaped = np.empty(batch_size, dtype=rewards.dtype)

        for i in range(batch_size):
            meta = trajectory_metadata[i] if i < len(trajectory_metadata) else {}
            state_start = meta.get("state_start", {})
            state_end = meta.get("state_end", {})

            phi_s = self._phi(state_start)
            phi_s_prime = self._phi(state_end)

            shaped[i] = rewards[i] + self.gamma * phi_s_prime - phi_s

        return shaped


# Auto-register on import
register_strategy("potential_based", PotentialBasedShaper)
