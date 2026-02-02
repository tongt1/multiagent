"""Reward shaping strategies for multi-agent RL training.

Implements quality mode and margin mode reward shaping following MARTI MAPoRL patterns:
- Quality mode: Rewards consistency with historical performance
- Margin mode: Rewards improvement over historical average
"""

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel


class RewardShapingMode(str, Enum):
    """Reward shaping mode."""

    QUALITY = "quality"
    MARGIN = "margin"
    NONE = "none"


class RewardShapingConfig(BaseModel):
    """Configuration for reward shaping."""

    mode: RewardShapingMode = RewardShapingMode.MARGIN
    alpha: float = 0.5
    beta: float = 0.5


def apply_reward_shaping(
    raw_rewards: np.ndarray,
    mode: str = "margin",
    alpha: float = 0.5,
    beta: float = 0.5,
) -> np.ndarray:
    """Apply reward shaping to multi-turn trajectory.

    Args:
        raw_rewards: Raw rewards array of shape (num_agents, num_turns)
        mode: Shaping mode ("quality", "margin", or "none")
        alpha: Shaping coefficient (default: 0.5)
        beta: Outcome reward coefficient (default: 0.5)

    Returns:
        Shaped rewards array of same shape as input

    Shaping formulas:
    - Quality mode: R_t + alpha * (Q_t * R_t - (1 - Q_t) * (1 - R_t))
      where Q_t = mean(raw_rewards[:t]) (historical average up to turn t)
      Rewards consistency with historical performance

    - Margin mode: R_t + alpha * (R_t - Q_t)
      where Q_t = mean(raw_rewards[:t]) (historical average up to turn t)
      Rewards improvement over historical average

    - None mode: Returns raw rewards unchanged

    Note: Turn 0 always uses raw reward (no history available)
    """
    if mode == "none":
        return raw_rewards.copy()

    shaped_rewards = raw_rewards.copy()
    num_agents, num_turns = raw_rewards.shape

    if num_turns <= 1:
        # Single turn or empty - no shaping possible
        return shaped_rewards

    # Apply shaping for each agent
    for m in range(num_agents):
        # Turn 0: Use raw reward (no history)
        # Turns 1+: Apply shaping based on historical average

        for t in range(1, num_turns):  # Start from turn 1
            # Compute historical average Q_t (mean of previous turns)
            Q_t = np.mean(raw_rewards[m, :t])

            R_t = raw_rewards[m, t]

            if mode == "quality":
                # Quality mode: Q_t * R_t - (1 - Q_t) * (1 - R_t)
                # Measures consistency with historical performance
                shaping_term = Q_t * R_t - (1.0 - Q_t) * (1.0 - R_t)
                shaped_rewards[m, t] = R_t + alpha * shaping_term

            elif mode == "margin":
                # Margin mode: R_t - Q_t
                # Measures improvement over historical average
                shaping_term = R_t - Q_t
                shaped_rewards[m, t] = R_t + alpha * shaping_term

    # Add outcome reward if beta > 0 (final turn performance)
    if beta > 0:
        for m in range(num_agents):
            final_reward = raw_rewards[m, -1]
            for t in range(num_turns):
                shaped_rewards[m, t] += beta * final_reward

    return shaped_rewards


def apply_reward_shaping_batch(
    trajectories: list[np.ndarray],
    config: RewardShapingConfig,
) -> list[np.ndarray]:
    """Apply reward shaping to batch of trajectories.

    Args:
        trajectories: List of reward arrays, each of shape (num_agents, num_turns)
        config: Reward shaping configuration

    Returns:
        List of shaped reward arrays with same shapes as input
    """
    shaped_trajectories = []

    for trajectory_rewards in trajectories:
        shaped = apply_reward_shaping(
            raw_rewards=trajectory_rewards,
            mode=config.mode.value,
            alpha=config.alpha,
            beta=config.beta,
        )
        shaped_trajectories.append(shaped)

    return shaped_trajectories
