"""Multi-model training foundation for debate RL.

This package provides configuration, model management, role routing,
dual-learner optimizer config, and gradient routing for training separate
solver and verifier/judge models within a single training job.

Key components:
- MultiModelConfig: Dataclass specifying per-role checkpoint paths and freeze config
- MultiModelManager: Maps role names to model identifiers, queries trainable/frozen keys
- RoleRouter: Routes debate roles to model keys and combines batch masks by model
- DualLearnerConfig: Per-model optimizer state with independent LR schedules
- PerModelOptimizerConfig: Single-model optimizer parameters (peak_lr, warmup, decay)
- GradientRouter: Splits batch gradients by role mask and routes to correct model

Usage:
    from src.training.multi_model import MultiModelConfig, MultiModelManager, RoleRouter
    from src.training.multi_model import DualLearnerConfig, GradientRouter

    # Single-model mode (default, backward compatible)
    config = MultiModelConfig()
    manager = MultiModelManager(config)
    assert manager.model_keys == ["policy"]

    # Multi-model mode with dual learner
    config = MultiModelConfig(
        solver_ckpt="/path/to/solver",
        verifier_ckpt="/path/to/verifier",
        freeze_roles=["verifier", "judge"],
    )
    manager = MultiModelManager(config)
    dl_config = DualLearnerConfig(multi_model_config=config)
    router = GradientRouter(manager)
"""

from src.training.multi_model.config import MultiModelConfig
from src.training.multi_model.dual_learner import (
    DualLearnerConfig,
    PerModelOptimizerConfig,
)
from src.training.multi_model.gradient_router import GradientRouter
from src.training.multi_model.model_manager import MultiModelManager
from src.training.multi_model.role_router import RoleRouter

__all__ = [
    "DualLearnerConfig",
    "GradientRouter",
    "MultiModelConfig",
    "MultiModelManager",
    "PerModelOptimizerConfig",
    "RoleRouter",
]
