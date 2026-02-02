"""Training job configuration models for multi-agent RL with Ray + vLLM + OpenRLHF.

Provides configuration models for:
- Training algorithms (PPO, GRPO, REINFORCE, RLOO)
- vLLM engine configuration
- OpenRLHF training parameters
- Ray cluster resource configuration
- Complete multi-agent training job specification with reward shaping

Follows MARTI training patterns for distributed RL training.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.training.reward_shaping import RewardShapingConfig, RewardShapingMode


class TrainingAlgorithm(str, Enum):
    """RL training algorithm."""

    PPO = "ppo"
    GRPO = "grpo"
    REINFORCE = "reinforce"
    RLOO = "rloo"


class VLLMConfig(BaseModel):
    """vLLM engine configuration for inference during training.

    Controls vLLM batch inference engines used for policy rollouts.
    """

    num_engines: int = Field(default=2, description="Number of vLLM engine instances")
    tensor_parallel_size: int = Field(
        default=1, description="Tensor parallelism degree per engine"
    )
    gpu_memory_utilization: float = Field(
        default=0.6,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization fraction (0.1-1.0)",
    )
    max_model_len: int = Field(default=12288, description="Maximum model context length")
    enforce_eager: bool = Field(
        default=False, description="Use eager mode instead of CUDA graphs"
    )


class OpenRLHFConfig(BaseModel):
    """OpenRLHF training hyperparameters.

    Controls RL algorithm, batch sizes, learning rates, and PPO/GRPO parameters.
    """

    algorithm: TrainingAlgorithm = Field(
        default=TrainingAlgorithm.REINFORCE, description="RL training algorithm"
    )
    num_episodes: int = Field(default=5, ge=1, description="Number of training episodes")
    train_batch_size: int = Field(
        default=128, ge=1, description="Total batch size for training"
    )
    micro_train_batch_size: int = Field(
        default=4, ge=1, description="Micro-batch size for gradient accumulation"
    )
    rollout_batch_size: int = Field(
        default=128, ge=1, description="Total batch size for policy rollouts"
    )
    micro_rollout_batch_size: int = Field(
        default=8, ge=1, description="Micro-batch size for rollout generation"
    )
    n_samples_per_prompt: int = Field(
        default=8, ge=1, description="Number of samples per prompt for RLOO/GRPO"
    )
    actor_learning_rate: float = Field(
        default=1e-6, gt=0, description="Actor (policy) learning rate"
    )
    critic_learning_rate: float = Field(
        default=9e-6, gt=0, description="Critic (value) learning rate"
    )
    gamma: float = Field(
        default=1.0, ge=0, le=1.0, description="Discount factor for rewards"
    )
    lambd: float = Field(
        default=1.0, ge=0, le=1.0, description="GAE lambda parameter"
    )
    init_kl_coef: float = Field(
        default=0.0, ge=0, description="Initial KL penalty coefficient"
    )
    normalize_reward: bool = Field(
        default=True, description="Normalize rewards across batch"
    )


class RayJobConfig(BaseModel):
    """Ray cluster resource configuration for distributed training.

    Specifies compute resources for training job on Ray cluster.
    """

    num_nodes: int = Field(default=3, ge=1, description="Number of compute nodes")
    num_gpus_per_node: int = Field(default=2, ge=1, description="GPUs per node")
    cpu_per_node: int = Field(default=16, ge=1, description="CPUs per node")
    memory_per_node: str = Field(default="128Gi", description="Memory per node (e.g., '128Gi')")
    runtime_env: dict = Field(
        default_factory=dict,
        description="Ray runtime environment (pip packages, env vars)",
    )


class MultiAgentTrainingConfig(BaseModel):
    """Complete configuration for multi-agent RL training job.

    Combines model paths, trajectory data, infrastructure config, training hyperparameters,
    and reward shaping settings for a complete training run specification.

    Attributes:
        pretrain_path: Base model checkpoint to initialize from
        save_path: Output directory for trained checkpoints
        trajectory_path: Path to MARTI-format trajectory JSONL
        ray_config: Ray cluster resource configuration
        vllm_config: vLLM engine configuration
        openrlhf_config: OpenRLHF training hyperparameters
        reward_shaping: Reward shaping configuration
        num_agents: Number of agents (3 solvers + verifier + judge = 5)
        num_rounds: Number of dialogue rounds
        shared_agents: Whether to share policies across agent roles
        wandb_project: Optional W&B project name for logging
        wandb_run_name: Optional W&B run name
    """

    pretrain_path: str = Field(..., description="Base model checkpoint path")
    save_path: str = Field(..., description="Output checkpoint directory")
    trajectory_path: str = Field(..., description="MARTI trajectory JSONL path")
    ray_config: RayJobConfig = Field(
        default_factory=RayJobConfig, description="Ray cluster configuration"
    )
    vllm_config: VLLMConfig = Field(
        default_factory=VLLMConfig, description="vLLM engine configuration"
    )
    openrlhf_config: OpenRLHFConfig = Field(
        default_factory=OpenRLHFConfig, description="OpenRLHF training parameters"
    )
    reward_shaping: RewardShapingConfig = Field(
        default_factory=RewardShapingConfig,
        description="Reward shaping configuration from 03-01",
    )
    num_agents: int = Field(
        default=5, ge=1, description="Number of agents (3 solvers + verifier + judge)"
    )
    num_rounds: int = Field(default=3, ge=1, description="Number of dialogue rounds")
    shared_agents: bool = Field(
        default=False, description="Share policies across agent roles"
    )
    wandb_project: Optional[str] = Field(
        None, description="Weights & Biases project name"
    )
    wandb_run_name: Optional[str] = Field(None, description="W&B run name")


def build_default_config(
    pretrain_path: str,
    trajectory_path: str,
    save_path: str,
) -> MultiAgentTrainingConfig:
    """Create training config with sensible defaults.

    Args:
        pretrain_path: Base model checkpoint to initialize from
        trajectory_path: Path to MARTI-format trajectory JSONL
        save_path: Output directory for trained checkpoints

    Returns:
        MultiAgentTrainingConfig with default hyperparameters

    Example:
        >>> config = build_default_config(
        ...     pretrain_path="models/cohere-7b",
        ...     trajectory_path="trajectories/marti_train.jsonl",
        ...     save_path="checkpoints/run_001"
        ... )
        >>> config.openrlhf_config.algorithm
        <TrainingAlgorithm.REINFORCE: 'reinforce'>
    """
    return MultiAgentTrainingConfig(
        pretrain_path=pretrain_path,
        trajectory_path=trajectory_path,
        save_path=save_path,
        ray_config=RayJobConfig(),
        vllm_config=VLLMConfig(),
        openrlhf_config=OpenRLHFConfig(),
        reward_shaping=RewardShapingConfig(mode=RewardShapingMode.MARGIN, alpha=0.5),
    )


def validate_config(config: MultiAgentTrainingConfig) -> list[str]:
    """Validate training configuration.

    Checks for:
    - Required paths provided
    - Valid hyperparameter ranges
    - Consistent batch size configuration
    - Valid resource allocation

    Args:
        config: Training configuration to validate

    Returns:
        List of error messages (empty if valid)

    Example:
        >>> config = build_default_config("model", "data.jsonl", "output")
        >>> errors = validate_config(config)
        >>> len(errors)
        0
    """
    errors = []

    # Check required paths
    if not config.pretrain_path:
        errors.append("pretrain_path is required")
    if not config.trajectory_path:
        errors.append("trajectory_path is required")
    if not config.save_path:
        errors.append("save_path is required")

    # Validate batch size consistency
    if config.openrlhf_config.train_batch_size % config.openrlhf_config.micro_train_batch_size != 0:
        errors.append(
            f"train_batch_size ({config.openrlhf_config.train_batch_size}) "
            f"must be divisible by micro_train_batch_size ({config.openrlhf_config.micro_train_batch_size})"
        )

    if config.openrlhf_config.rollout_batch_size % config.openrlhf_config.micro_rollout_batch_size != 0:
        errors.append(
            f"rollout_batch_size ({config.openrlhf_config.rollout_batch_size}) "
            f"must be divisible by micro_rollout_batch_size ({config.openrlhf_config.micro_rollout_batch_size})"
        )

    # Validate resource allocation
    total_gpus = config.ray_config.num_nodes * config.ray_config.num_gpus_per_node
    vllm_gpus_needed = config.vllm_config.num_engines * config.vllm_config.tensor_parallel_size

    if vllm_gpus_needed > total_gpus:
        errors.append(
            f"vLLM requires {vllm_gpus_needed} GPUs "
            f"({config.vllm_config.num_engines} engines × {config.vllm_config.tensor_parallel_size} TP) "
            f"but only {total_gpus} GPUs available "
            f"({config.ray_config.num_nodes} nodes × {config.ray_config.num_gpus_per_node} GPUs/node)"
        )

    # Validate algorithm-specific requirements
    if config.openrlhf_config.algorithm in (TrainingAlgorithm.RLOO, TrainingAlgorithm.GRPO):
        if config.openrlhf_config.n_samples_per_prompt < 2:
            errors.append(
                f"{config.openrlhf_config.algorithm.value} requires n_samples_per_prompt >= 2"
            )

    # Validate num_agents matches expected MARTI structure
    if config.num_agents < 5:
        errors.append(
            f"num_agents ({config.num_agents}) should be at least 5 "
            "(3 solvers + 1 verifier + 1 judge)"
        )

    return errors
