"""Global registry for rollout selection strategies.

Provides registration, retrieval, and config-based instantiation of
rollout selection strategies. Strategies self-register at import time
(e.g., identity.py registers "identity" when imported).

Usage:
    from src.training.rollout_strategy.registry import create_strategy_from_config

    # From training config
    strategy = create_strategy_from_config({"strategy_name": "best_of_n"})

    # Default (no config) returns identity passthrough
    strategy = create_strategy_from_config(None)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.rollout_strategy.base import RolloutStrategy

_REGISTRY: dict[str, type[RolloutStrategy]] = {}


def register_strategy(name: str, cls: type[RolloutStrategy]) -> None:
    """Register a rollout selection strategy by name.

    Args:
        name: Strategy name (e.g., "identity", "best_of_n")
        cls: Strategy class, must be a subclass of RolloutStrategy

    Raises:
        TypeError: If cls is not a subclass of RolloutStrategy
    """
    from src.training.rollout_strategy.base import RolloutStrategy

    if not (isinstance(cls, type) and issubclass(cls, RolloutStrategy)):
        raise TypeError(
            f"Strategy class must be a subclass of RolloutStrategy, got {cls}. "
            f"Ensure your strategy extends RolloutStrategy."
        )
    _REGISTRY[name] = cls


def get_strategy(name: str) -> type[RolloutStrategy]:
    """Retrieve a registered strategy class by name.

    Args:
        name: Strategy name to look up

    Returns:
        The strategy class

    Raises:
        KeyError: If strategy name is not registered, with helpful message
            listing available strategies
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Unknown rollout strategy '{name}'. "
            f"Available strategies: {available}. "
            f"Did you forget to import the strategy module?"
        )
    return _REGISTRY[name]


def create_strategy_from_config(config: dict | None) -> RolloutStrategy:
    """Create a strategy instance from a config dict.

    If config is None, empty, or missing strategy_name, returns
    IdentityRolloutStrategy as the default (preserving existing
    behavior where all rollouts contribute to training).

    Args:
        config: Dict with optional keys:
            - strategy_name (str): Name of registered strategy
            - strategy_params (dict): Kwargs passed to strategy constructor

    Returns:
        Instantiated RolloutStrategy

    Example:
        >>> create_strategy_from_config({"strategy_name": "best_of_n"})
        <BestOfNStrategy>
        >>> create_strategy_from_config(None)
        <IdentityRolloutStrategy>
    """
    # Import identity to ensure it's registered as the default
    import src.training.rollout_strategy.identity  # noqa: F401

    if not config or "strategy_name" not in config:
        # Default to identity passthrough
        return _REGISTRY["identity"]()

    strategy_name = config["strategy_name"]
    strategy_params = config.get("strategy_params", {}) or {}

    cls = get_strategy(strategy_name)
    return cls(**strategy_params)


def list_strategies() -> list[str]:
    """Return list of registered strategy names.

    Returns:
        Sorted list of strategy name strings
    """
    return sorted(_REGISTRY.keys())
