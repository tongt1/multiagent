"""Debate metric computation functions for W&B logging.

This module provides pure computation functions for per-role rewards, zero-advantage
detection, and per-role KL divergence. All functions operate on numpy arrays and return
dicts with debate/-prefixed metric names from metric_schema.py.

Functions are designed to be:
- Pure (no side effects)
- Testable (no JAX dependencies)
- Context-agnostic (callable from any training context)
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from src.training.wandb_enrichment.metric_schema import (
    METRIC_FRAC_ZERO_STD,
    METRIC_FRAC_ZERO_STD_CORRECT,
    METRIC_FRAC_ZERO_STD_INCORRECT,
    METRIC_KL_JUDGE,
    METRIC_KL_SOLVER,
    METRIC_KL_VERIFIER,
    METRIC_MEAN_REWARD_STD,
    METRIC_REWARD_JUDGE,
    METRIC_REWARD_SOLVER,
    METRIC_REWARD_VERIFIER,
)


def compute_per_role_rewards(
    rewards: np.ndarray,  # Shape: (batch_size,) — reward per rollout
    role_labels: list[str],  # Length: batch_size — "solver", "verifier", or "judge" per rollout
) -> dict[str, float]:
    """Compute mean reward per debate role.

    Groups rollout rewards by role label and computes mean for each role.

    Args:
        rewards: Flat array of rewards, one per rollout
        role_labels: Role label per rollout ("solver", "verifier", or "judge")

    Returns:
        Dict with per-role reward metrics using debate/ prefixed keys.
        Omits roles not present in the batch.

    Example:
        >>> rewards = np.array([0.5, 1.0, 0.0, 0.8, 1.0, 0.2])
        >>> role_labels = ["solver", "solver", "verifier", "verifier", "judge", "judge"]
        >>> compute_per_role_rewards(rewards, role_labels)
        {'debate/reward/solver': 0.75, 'debate/reward/verifier': 0.4, 'debate/reward/judge': 0.6}
    """
    if len(rewards) == 0:
        return {}

    if len(rewards) != len(role_labels):
        logger.warning(
            f"Rewards length ({len(rewards)}) != role_labels length ({len(role_labels)}). "
            "Returning empty dict."
        )
        return {}

    result = {}

    # Map role names to metric constants
    role_metric_map = {
        "solver": METRIC_REWARD_SOLVER,
        "verifier": METRIC_REWARD_VERIFIER,
        "judge": METRIC_REWARD_JUDGE,
    }

    # Group rewards by role and compute means
    for role, metric_key in role_metric_map.items():
        role_mask = np.array([label == role for label in role_labels])
        if role_mask.any():
            result[metric_key] = float(rewards[role_mask].mean())

    return result


def compute_zero_advantage_metrics(
    rewards: np.ndarray,  # Shape: (batch_size,) — flat array of all rollout rewards
    n_rollouts_per_prompt: int = 8,  # GRPO group size
) -> dict[str, float]:
    """Detect zero-advantage GRPO training collapse.

    Zero-advantage occurs when all rollouts for a prompt receive identical rewards,
    causing GRPO advantages to become zero and training to stall. This function
    computes the fraction of prompts with zero reward variance and breaks down
    by correct vs incorrect outcomes.

    Args:
        rewards: Flat array of all rollout rewards
        n_rollouts_per_prompt: Number of rollouts per prompt (GRPO group size)

    Returns:
        Dict with zero-advantage detection metrics:
        - frac_reward_zero_std: Fraction of prompts with zero std
        - frac_zero_std_correct: Fraction where zero_std AND all rewards == 1.0
        - frac_zero_std_incorrect: Fraction where zero_std AND all rewards == 0.0
        - mean_reward_std: Mean of per-prompt standard deviations

    Example:
        >>> # 2 prompts: first has all-same rewards, second has varied rewards
        >>> rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ...                     0.5, 0.8, 1.0, 0.2, 0.9, 0.3, 0.7, 1.0])
        >>> compute_zero_advantage_metrics(rewards, n_rollouts_per_prompt=8)
        {'debate/frac_reward_zero_std': 0.5, 'debate/frac_zero_std_correct': 0.5, ...}
    """
    if len(rewards) < n_rollouts_per_prompt:
        # Not enough rewards to form even one complete group
        return {
            METRIC_FRAC_ZERO_STD: 0.0,
            METRIC_FRAC_ZERO_STD_CORRECT: 0.0,
            METRIC_FRAC_ZERO_STD_INCORRECT: 0.0,
            METRIC_MEAN_REWARD_STD: 0.0,
        }

    # Truncate to last complete group if not evenly divisible
    n_prompts = len(rewards) // n_rollouts_per_prompt
    n_complete_rewards = n_prompts * n_rollouts_per_prompt

    if len(rewards) > n_complete_rewards:
        logger.warning(
            f"Batch size {len(rewards)} not divisible by n_rollouts_per_prompt {n_rollouts_per_prompt}. "
            f"Truncating to {n_complete_rewards} rewards ({n_prompts} prompts)."
        )
        rewards = rewards[:n_complete_rewards]

    # Reshape to (n_prompts, n_rollouts_per_prompt)
    rewards_grouped = rewards.reshape(n_prompts, n_rollouts_per_prompt)

    # Compute per-prompt reward standard deviation
    reward_std = rewards_grouped.std(axis=1)

    # Zero-std mask (epsilon for numerical stability)
    zero_std_mask = reward_std < 1e-8

    # Compute fractions
    frac_zero_std = float(zero_std_mask.mean())

    # Breakdown by correct vs incorrect
    # Correct: all rewards in prompt == 1.0
    all_correct_mask = (rewards_grouped == 1.0).all(axis=1)
    frac_zero_std_correct = float((zero_std_mask & all_correct_mask).mean())

    # Incorrect: all rewards in prompt == 0.0
    all_incorrect_mask = (rewards_grouped == 0.0).all(axis=1)
    frac_zero_std_incorrect = float((zero_std_mask & all_incorrect_mask).mean())

    # Mean of reward std
    mean_reward_std = float(reward_std.mean())

    return {
        METRIC_FRAC_ZERO_STD: frac_zero_std,
        METRIC_FRAC_ZERO_STD_CORRECT: frac_zero_std_correct,
        METRIC_FRAC_ZERO_STD_INCORRECT: frac_zero_std_incorrect,
        METRIC_MEAN_REWARD_STD: mean_reward_std,
    }


def compute_per_role_kl(
    kl_per_token: np.ndarray,  # Shape: (batch_size, seq_len) — per-token KL divergence
    role_masks: dict[str, np.ndarray],  # Keys: "solver", "verifier", "judge", Values: (batch_size, seq_len)
) -> dict[str, float]:
    """Compute per-role KL divergence from reference policy.

    Aggregates per-token KL using role masks to compute mean KL per debate role.

    Args:
        kl_per_token: Per-token KL divergence from reference policy
        role_masks: Boolean masks indicating which tokens belong to each role

    Returns:
        Dict with per-role KL metrics using debate/ prefixed keys.
        Omits roles with no tokens (mask has no True values).

    Example:
        >>> kl = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        >>> masks = {
        ...     "solver": np.array([[True, True, False, False], [True, False, False, False]]),
        ...     "verifier": np.array([[False, False, True, True], [False, True, True, False]]),
        ... }
        >>> compute_per_role_kl(kl, masks)
        {'debate/kl/solver': 0.225, 'debate/kl/verifier': 0.45}
    """
    if role_masks is None or len(role_masks) == 0:
        return {}

    result = {}

    # Map role names to metric constants
    role_metric_map = {
        "solver": METRIC_KL_SOLVER,
        "verifier": METRIC_KL_VERIFIER,
        "judge": METRIC_KL_JUDGE,
    }

    # Compute per-role KL
    for role, metric_key in role_metric_map.items():
        if role not in role_masks:
            continue

        role_mask = role_masks[role]

        # Check if mask has any True values
        n_tokens = role_mask.sum()
        if n_tokens == 0:
            # No tokens for this role, skip
            continue

        # Compute mean KL for this role
        kl_role = (kl_per_token * role_mask).sum() / (n_tokens + 1e-8)
        result[metric_key] = float(kl_role)

    return result


def compute_all_scalar_metrics(
    rewards: np.ndarray,
    role_labels: list[str],
    n_rollouts_per_prompt: int = 8,
    kl_per_token: np.ndarray | None = None,
    role_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Convenience function to compute all debate scalar metrics.

    Calls all metric computation functions and merges results.

    Args:
        rewards: Flat array of rewards, one per rollout
        role_labels: Role label per rollout
        n_rollouts_per_prompt: GRPO group size
        kl_per_token: Per-token KL divergence (optional)
        role_masks: Boolean masks for per-role KL (optional)

    Returns:
        Dict with all computed metrics using debate/ prefixed keys.

    Example:
        >>> rewards = np.array([1.0, 1.0, 0.5, 0.8])
        >>> role_labels = ["solver", "verifier", "judge", "solver"]
        >>> compute_all_scalar_metrics(rewards, role_labels, n_rollouts_per_prompt=4)
        {'debate/reward/solver': 0.9, 'debate/reward/verifier': 0.5, ...}
    """
    result = {}

    # Per-role rewards
    result.update(compute_per_role_rewards(rewards, role_labels))

    # Zero-advantage detection
    result.update(compute_zero_advantage_metrics(rewards, n_rollouts_per_prompt))

    # Per-role KL (if data provided)
    if kl_per_token is not None and role_masks is not None:
        result.update(compute_per_role_kl(kl_per_token, role_masks))

    return result
