"""Multi-model configuration for dual-sidecar training.

Defines the configuration for training with separate solver and verifier
models, each served by its own vLLM sidecar.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MultiModelConfig:
    """Configuration for dual-model training.

    Attributes:
        solver_ckpt: Checkpoint path for the solver model.
        verifier_ckpt: Checkpoint path for the verifier model.
        solver_model_size: Human-readable solver model size (e.g., "0.135B").
        verifier_model_size: Human-readable verifier model size.
        freeze_roles: List of roles to freeze during training.
            Empty list = train both models.
            ["solver"] = freeze solver, train verifier+judge.
            ["verifier", "judge"] = freeze verifier+judge, train solver.
    """
    solver_ckpt: str = ""
    verifier_ckpt: str = ""
    solver_model_size: str = ""
    verifier_model_size: str = ""
    freeze_roles: list[str] = field(default_factory=list)
