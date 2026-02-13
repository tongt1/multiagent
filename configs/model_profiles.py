"""Model profiles for sweep configs.

Each profile encapsulates GPU requirements, checkpoint path, and
sequence length for a model size. Sweep configs import the profile
and read its attributes.

Usage:
    from configs.model_profiles import SMOLLM_135M
    _PROFILE = SMOLLM_135M
    NUM_TRAINING_GPUS = _PROFILE.num_training_gpus
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """Immutable model profile for sweep configs."""
    ckpt_path: str
    num_training_gpus: int
    num_sampling_gpus: int
    max_sequence_length: int
    needs_mesh_override: bool


SMOLLM_135M = ModelProfile(
    ckpt_path="gs://cohere-dev-central-2/users/roman_cohere_com/smollm-135M/megazord_weights/ckpt-0",
    num_training_gpus=1,
    num_sampling_gpus=1,
    max_sequence_length=2048,
    needs_mesh_override=False,
)
