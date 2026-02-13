"""Global registry for reward shaping strategies.

Provides registration, retrieval, and config-based instantiation of
reward shaping strategies. Strategies self-register at import time
(e.g., identity.py registers "identity" when imported).

Usage:
    from src.training.reward_shaping.registry import create_strategy_from_config

    # From training config
    strategy = create_strategy_from_config({"strategy_name": "identity"})

    # Default (no config) returns identity passthrough
    strategy = create_strategy_from_config(None)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.reward_shaping.base import RewardShaper

_REGISTRY: dict[str, type[RewardShaper]] = {}


def register_strategy(name: str, cls: type[RewardShaper]) -> None:
    """Register a reward shaping strategy by name.

    Args:
        name: Strategy name (e.g., "identity", "difference_rewards")
        cls: Strategy class, must be a subclass of RewardShaper

    Raises:
        TypeError: If cls is not a subclass of RewardShaper
    """
    from src.training.reward_shaping.base import RewardShaper

    if not (isinstance(cls, type) and issubclass(cls, RewardShaper)):
        raise TypeError(
            f"Strategy class must be a subclass of RewardShaper, got {cls}. "
            f"Ensure your strategy extends RewardShaper."
        )
    _REGISTRY[name] = cls


def get_strategy(name: str) -> type[RewardShaper]:
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
            f"Unknown reward shaping strategy '{name}'. "
            f"Available strategies: {available}. "
            f"Did you forget to import the strategy module?"
        )
    return _REGISTRY[name]


def create_strategy_from_config(config: dict | None) -> RewardShaper:
    """Create a strategy instance from a config dict.

    If config is None, empty, or missing strategy_name, returns
    IdentityRewardShaper as the default (preserving existing binary
    SymPy reward behavior).

    Args:
        config: Dict with optional keys:
            - strategy_name (str): Name of registered strategy
            - strategy_params (dict): Kwargs passed to strategy constructor

    Returns:
        Instantiated RewardShaper

    Example:
        >>> create_strategy_from_config({"strategy_name": "identity"})
        <IdentityRewardShaper>
        >>> create_strategy_from_config(None)
        <IdentityRewardShaper>
    """
    # Import identity to ensure it's registered as the default
    import src.training.reward_shaping.identity  # noqa: F401

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
