"""COMA-style counterfactual advantage reward shaping strategy.

Computes per-agent advantage by comparing the team reward (Q-value of
joint action) against a per-agent baseline. The baseline represents the
expected reward under agent i's own policy, marginalizing over its
possible actions.

In the GRPO setting with N rollouts per prompt, the baseline is naturally
approximated by the mean reward over the N rollouts for the same prompt.
If per-agent baselines are available in trajectory_metadata (from separate
counterfactual evaluations), those are used instead.

Reference: Foerster et al., "Counterfactual Multi-Agent Policy Gradients" (COMA), AAAI 2018.

Auto-registers as "coma_advantage" on import.
"""

from __future__ import annotations

import numpy as np

from src.training.reward_shaping.base import RewardShaper
from src.training.reward_shaping.registry import register_strategy

_ROLES = ("solver", "verifier", "judge")


class COMAAdvantageShaper(RewardShaper):
    """COMA-style counterfactual advantage: A_i(s,a) = Q(s,a) - baseline_i.

    Computes per-agent advantage by comparing the team reward (Q-value of
    joint action) against a per-agent baseline. The baseline represents the
    expected reward under agent i's own policy, marginalizing over its
    possible actions.

    A_i(s,a) = Q(s, a_i, a_{-i}) - sum(pi_i(a'_i|s) * Q(s, a'_i, a_{-i}))

    In GRPO with N rollouts per prompt, we approximate the baseline as:
    - For each agent i in each rollout: baseline_i = mean reward across
      the N GRPO rollouts (same prompt, different completions)
    - This approximates E_{a'_i ~ pi_i}[Q(s, a'_i, a_{-i})]

    If per-agent baselines from trajectory_metadata are available
    (e.g., from separate counterfactual evaluations), those are used instead.

    Fallback: When no metadata is available, returns standard GRPO advantage
    (reward - mean_reward_per_prompt) for all roles equally.
    """

    def __init__(self, n_rollouts_per_prompt: int = 8, **kwargs) -> None:
        """Initialize COMA advantage shaper.

        Args:
            n_rollouts_per_prompt: GRPO group size for baseline computation.
                Must match GENERATIONS_PER_PROMPT from SWEEP config.
            **kwargs: Additional parameters passed to base class.
        """
        super().__init__(**kwargs)
        self.n_rollouts_per_prompt = n_rollouts_per_prompt

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "coma_advantage"

    def shape_rewards(
        self,
        rewards: np.ndarray,
        role_masks: dict[str, np.ndarray] | None,
        trajectory_metadata: list[dict] | None,
    ) -> dict[str, np.ndarray]:
        """Compute per-agent counterfactual advantage.

        If trajectory_metadata provides per-role baselines (baseline_reward_solver,
        etc.), uses those for counterfactual advantage. Otherwise falls back to
        GRPO group mean advantage.

        Args:
            rewards: Shape (B,) flat reward per rollout.
            role_masks: Optional per-role boolean masks (not used for advantage
                computation but accepted for interface compatibility).
            trajectory_metadata: Optional list of per-rollout metadata dicts.
                May contain baseline_reward_solver, baseline_reward_verifier,
                baseline_reward_judge keys.

        Returns:
            Dict with keys "solver", "verifier", "judge", each containing
            np.ndarray shape (B,) of per-agent advantages.
        """
        batch_size = len(rewards)

        # Check if per-role baselines are available in metadata
        if trajectory_metadata and self._has_per_role_baselines(trajectory_metadata):
            return self._compute_with_baselines(rewards, trajectory_metadata)

        # Fallback: GRPO group mean advantage (same for all roles)
        return self._compute_grpo_fallback(rewards, batch_size)

    def _has_per_role_baselines(self, metadata: list[dict]) -> bool:
        """Check if any metadata entry has per-role baseline keys."""
        if not metadata:
            return False
        first = metadata[0]
        return any(f"baseline_reward_{role}" in first for role in _ROLES)

    def _compute_with_baselines(
        self,
        rewards: np.ndarray,
        metadata: list[dict],
    ) -> dict[str, np.ndarray]:
        """Compute advantage using per-role baselines from metadata.

        A_role[i] = Q(s, a_i) - baseline_role[i]
        where Q is the team reward and baseline_role is the expected
        reward under role's policy.
        """
        result = {}
        for role in _ROLES:
            baselines = np.array(
                [m.get(f"baseline_reward_{role}", 0.0) for m in metadata]
            )
            result[role] = rewards - baselines
        return result

    def _compute_grpo_fallback(
        self,
        rewards: np.ndarray,
        batch_size: int,
    ) -> dict[str, np.ndarray]:
        """Compute advantage using GRPO group mean as baseline.

        Groups rewards by prompt (reshape to [n_prompts, n_rollouts]),
        computes mean per prompt, subtracts to get advantage, flattens back.
        Same advantage assigned to all roles.
        """
        n = self.n_rollouts_per_prompt

        if batch_size % n != 0:
            # If batch doesn't divide evenly, use global mean
            mean_reward = rewards.mean()
            advantage = rewards - mean_reward
        else:
            n_prompts = batch_size // n
            grouped = rewards.reshape(n_prompts, n)
            mean_per_prompt = grouped.mean(axis=1, keepdims=True)
            advantage = (grouped - mean_per_prompt).reshape(-1)

        # Same advantage for all roles in fallback mode
        return {role: advantage.copy() for role in _ROLES}


# Auto-register on import
register_strategy("coma_advantage", COMAAdvantageShaper)
