"""Experiment configuration module.

Provides override-only experiment configs and batch management.
"""

from .experiment_config import ExperimentBatchConfig, ExperimentConfig

__all__ = [
    "ExperimentConfig",
    "ExperimentBatchConfig",
]

# Lazy import to avoid circular dependencies
def __getattr__(name):
    if name in ["ModelConfig", "get_model_config", "SUPPORTED_MODELS"]:
        from .model_registry import SUPPORTED_MODELS, ModelConfig, get_model_config
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
