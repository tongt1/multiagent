"""Rollout selection strategies for GRPO training.

This package provides a pluggable registry of rollout selection strategies
that filter which rollouts from a GRPO batch contribute to gradient updates.

Available strategies:
- identity: Passthrough (default), all rollouts contribute to training
- best_of_n: Select highest-reward rollout per prompt group

Usage:
    from src.training.rollout_strategy import create_strategy_from_config

    # Create from training config
    strategy = create_strategy_from_config({"strategy_name": "best_of_n"})
    selected = strategy.select_rollouts(items, n_rollouts_per_prompt=4)

    # Default returns identity passthrough
    strategy = create_strategy_from_config(None)
"""

from src.training.rollout_strategy.base import RolloutStrategy
from src.training.rollout_strategy.best_of_n import BestOfNStrategy

# Import strategies to trigger auto-registration
from src.training.rollout_strategy.identity import IdentityRolloutStrategy
from src.training.rollout_strategy.registry import (
    create_strategy_from_config,
    get_strategy,
    list_strategies,
    register_strategy,
)

__all__ = [
    "RolloutStrategy",
    "IdentityRolloutStrategy",
    "BestOfNStrategy",
    "register_strategy",
    "get_strategy",
    "create_strategy_from_config",
    "list_strategies",
]
