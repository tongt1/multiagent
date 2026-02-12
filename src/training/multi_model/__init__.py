"""Multi-model training foundation for debate RL.

This package provides configuration, model management, and role routing
for training separate solver and verifier/judge models within a single
training job.

Key components:
- MultiModelConfig: Dataclass specifying per-role checkpoint paths and freeze config
- MultiModelManager: Maps role names to model identifiers, queries trainable/frozen keys
- RoleRouter: Routes debate roles to model keys and combines batch masks by model
"""
