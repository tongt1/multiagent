"""Multi-model training configuration.

Defines the MultiModelConfig dataclass that specifies per-role checkpoint paths,
freeze configuration, and model size metadata for training separate solver and
verifier/judge models within a single training job.

Design notes:
- Pure Python dataclass, no Flink/JAX dependencies (matching reward_shaping pattern)
- Single-model mode (default): both ckpt paths are None, behaves like standard training
- Multi-model mode: both solver_ckpt and verifier_ckpt are set
- freeze_roles controls which roles receive no gradient updates (e.g., freeze verifier
  while training solver only)
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_ROLES = frozenset({"solver", "verifier", "judge"})


@dataclass
class MultiModelConfig:
    """Configuration for multi-model training with per-role checkpoints.

    In single-model mode (default), a single model handles all debate roles.
    In multi-model mode, separate checkpoints are used for solver vs verifier/judge.

    Attributes:
        solver_ckpt: Checkpoint path for the solver model. None = use shared model.
        verifier_ckpt: Checkpoint path for the verifier/judge model. None = use shared model.
        solver_model_size: Informational size label (e.g., "7B"). Used for GPU partitioning.
        verifier_model_size: Informational size label (e.g., "3B"). Used for GPU partitioning.
        freeze_roles: List of role names whose model receives no gradient updates.
            Valid values: "solver", "verifier", "judge".
    """

    solver_ckpt: str | None = None
    verifier_ckpt: str | None = None
    solver_model_size: str | None = None
    verifier_model_size: str | None = None
    freeze_roles: list[str] = field(default_factory=list)

    @property
    def is_multi_model(self) -> bool:
        """True when both solver and verifier checkpoints are set (dual-model mode)."""
        return self.solver_ckpt is not None and self.verifier_ckpt is not None

    def validate(self) -> None:
        """Validate configuration fields.

        Raises:
            ValueError: If freeze_roles contains unknown role names.
        """
        for role in self.freeze_roles:
            if role not in VALID_ROLES:
                raise ValueError(
                    f"Unknown role '{role}' in freeze_roles. "
                    f"Valid roles: {sorted(VALID_ROLES)}"
                )
