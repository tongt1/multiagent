"""Multi-model training support for dual-sidecar sweep configs."""
from src.training.multi_model.config import MultiModelConfig
from src.training.multi_model.sweep_integration import build_sidecar_configs

__all__ = ["MultiModelConfig", "build_sidecar_configs"]
