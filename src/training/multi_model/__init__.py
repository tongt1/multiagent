"""Multi-model training foundation for debate RL.

This package provides configuration, model management, and role routing
for training separate solver and verifier/judge models within a single
training job.

Key components:
- MultiModelConfig: Dataclass specifying per-role checkpoint paths and freeze config
- MultiModelManager: Maps role names to model identifiers, queries trainable/frozen keys
- RoleRouter: Routes debate roles to model keys and combines batch masks by model

Usage:
    from src.training.multi_model import MultiModelConfig, MultiModelManager, RoleRouter

    # Single-model mode (default, backward compatible)
    config = MultiModelConfig()
    manager = MultiModelManager(config)
    assert manager.model_keys == ["policy"]

    # Multi-model mode
    config = MultiModelConfig(
        solver_ckpt="/path/to/solver",
        verifier_ckpt="/path/to/verifier",
        freeze_roles=["verifier", "judge"],
    )
    manager = MultiModelManager(config)
    assert manager.model_keys == ["solver_model", "verifier_model"]
    assert manager.get_trainable_model_keys() == ["solver_model"]
"""

from src.training.multi_model.config import MultiModelConfig
from src.training.multi_model.model_manager import MultiModelManager
from src.training.multi_model.role_router import RoleRouter

__all__ = [
    "MultiModelConfig",
    "MultiModelManager",
    "RoleRouter",
]
