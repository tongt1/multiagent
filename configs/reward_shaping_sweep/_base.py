"""Shared constants for reward shaping sweep configs (SmolLM-135M, MATH-500).

All 5 strategy-specific SWEEP configs import from this module.
The ONLY differences between configs are in DebateMetricStreamerConfig arguments.
This guarantees apples-to-apples comparison (ECFG-06).

OWNERS: Multiagent Debate RL Experiment
"""
from __future__ import annotations

from configs.model_profiles import SMOLLM_135M

# Model profile
_PROFILE = SMOLLM_135M
NUM_TRAINING_GPUS = _PROFILE.num_training_gpus      # 1
NUM_SAMPLING_GPUS = _PROFILE.num_sampling_gpus       # 1
MAX_SEQUENCE_LENGTH = _PROFILE.max_sequence_length   # 2048
CKPT_PATH = _PROFILE.ckpt_path  # gs://cohere-dev-central-2/users/roman_cohere_com/smollm-135M/megazord_weights/ckpt-0

# Training hyperparameters (shared across all 5 configs)
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
TOTAL_TRAIN_STEPS = 15           # SmolLM-135M: very fast iteration
LEARNING_RATE = 3e-6
KL_BETA = 0.03
GENERATIONS_PER_PROMPT = 4       # GRPO group size
EXPORT_EVERY_STEPS = 5
HARD_UPDATE_REF_EVERY_STEPS = 5
SEED = 42
N_GRADIENT_ACCUMULATION_STEPS = 1

# Queue and priority
PRIORITY_CLASS = "dev-low"

# Paths
RUN_CONFIG_PATH = "${HOME}/reward-training/run_configs/smollm_135m_rloo_math.run"
MATH_500_DATA_PATH = "gs://cohere-dev-central-2/comb/data/math_500/2025_05_15/scenarios_train.jsonl"
K8S_SECRETS_PATH = "${HOME}/repos/secrets_template.toml"
WANDB_PROJECT = "multiagent-debate-rl"

# vLLM sidecar configuration
_VLLM_SIDECAR = "vllm"
_VLLM_PORT = 8000
_VLLM_EXPORT_DIR = "/data/1d/post-training/${USER}/${SWEEP_NAME}/${TRIAL_IDX}"
