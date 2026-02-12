"""Reward shaping strategies for multi-agent debate RL.

This package provides a pluggable registry of reward shaping strategies
that transform raw binary SymPy rewards into shaped rewards for more
nuanced multi-agent credit assignment.

Available strategies:
- identity: Passthrough (default), preserves existing binary rewards
- difference_rewards: Counterfactual marginal contribution D_i = G(z) - G(z_{-i})
- reward_mixing: Blends global team reward with per-role local signals

Usage:
    from src.training.reward_shaping import create_strategy_from_config

    # Create from training config
    strategy = create_strategy_from_config({"strategy_name": "identity"})
    shaped = strategy.shape_rewards(rewards, role_masks, metadata)

    # Default returns identity passthrough
    strategy = create_strategy_from_config(None)
"""

from src.training.reward_shaping.base import RewardShaper
from src.training.reward_shaping.registry import (
    create_strategy_from_config,
    get_strategy,
    list_strategies,
    register_strategy,
)

# Import strategies to trigger auto-registration
from src.training.reward_shaping.identity import IdentityRewardShaper
from src.training.reward_shaping.difference_rewards import DifferenceRewardShaper
from src.training.reward_shaping.reward_mixing import RewardMixingShaper

__all__ = [
    "RewardShaper",
    "IdentityRewardShaper",
    "DifferenceRewardShaper",
    "RewardMixingShaper",
    "register_strategy",
    "get_strategy",
    "create_strategy_from_config",
    "list_strategies",
]
