"""Tests for training configuration models."""

import json

import pytest

from src.training.reward_shaping import RewardShapingConfig, RewardShapingMode
from src.training.training_config import (
    MultiAgentTrainingConfig,
    OpenRLHFConfig,
    RayJobConfig,
    TrainingAlgorithm,
    VLLMConfig,
    build_default_config,
    validate_config,
)


def test_training_algorithm_enum():
    """Test TrainingAlgorithm enum values."""
    assert TrainingAlgorithm.PPO.value == "ppo"
    assert TrainingAlgorithm.GRPO.value == "grpo"
    assert TrainingAlgorithm.REINFORCE.value == "reinforce"
    assert TrainingAlgorithm.RLOO.value == "rloo"


def test_vllm_config_defaults():
    """Test VLLMConfig default values."""
    config = VLLMConfig()
    assert config.num_engines == 2
    assert config.tensor_parallel_size == 1
    assert config.gpu_memory_utilization == 0.6
    assert config.max_model_len == 12288
    assert config.enforce_eager is False


def test_vllm_config_validation():
    """Test VLLMConfig validates GPU memory utilization."""
    # Valid range
    config = VLLMConfig(gpu_memory_utilization=0.5)
    assert config.gpu_memory_utilization == 0.5

    # Out of range - should raise ValidationError
    with pytest.raises(Exception):  # Pydantic ValidationError
        VLLMConfig(gpu_memory_utilization=1.5)

    with pytest.raises(Exception):
        VLLMConfig(gpu_memory_utilization=0.0)


def test_openrlhf_config_defaults():
    """Test OpenRLHFConfig default values."""
    config = OpenRLHFConfig()
    assert config.algorithm == TrainingAlgorithm.REINFORCE
    assert config.num_episodes == 5
    assert config.train_batch_size == 128
    assert config.micro_train_batch_size == 4
    assert config.rollout_batch_size == 128
    assert config.micro_rollout_batch_size == 8
    assert config.n_samples_per_prompt == 8
    assert config.actor_learning_rate == 1e-6
    assert config.critic_learning_rate == 9e-6
    assert config.gamma == 1.0
    assert config.lambd == 1.0
    assert config.init_kl_coef == 0.0
    assert config.normalize_reward is True


def test_ray_job_config_defaults():
    """Test RayJobConfig default values."""
    config = RayJobConfig()
    assert config.num_nodes == 3
    assert config.num_gpus_per_node == 2
    assert config.cpu_per_node == 16
    assert config.memory_per_node == "128Gi"
    assert config.runtime_env == {}


def test_ray_job_config_runtime_env():
    """Test RayJobConfig with custom runtime_env."""
    config = RayJobConfig(runtime_env={"pip": ["numpy", "torch"]})
    assert config.runtime_env == {"pip": ["numpy", "torch"]}


def test_build_default_config():
    """Test build_default_config creates valid config."""
    config = build_default_config(
        pretrain_path="models/cohere-7b",
        trajectory_path="trajectories/train.jsonl",
        save_path="checkpoints/run_001",
    )

    # Check required fields
    assert config.pretrain_path == "models/cohere-7b"
    assert config.trajectory_path == "trajectories/train.jsonl"
    assert config.save_path == "checkpoints/run_001"

    # Check defaults
    assert config.num_agents == 5
    assert config.num_rounds == 3
    assert config.shared_agents is False
    assert config.wandb_project is None
    assert config.wandb_run_name is None

    # Check nested configs initialized
    assert isinstance(config.ray_config, RayJobConfig)
    assert isinstance(config.vllm_config, VLLMConfig)
    assert isinstance(config.openrlhf_config, OpenRLHFConfig)
    assert isinstance(config.reward_shaping, RewardShapingConfig)

    # Check reward shaping defaults
    assert config.reward_shaping.mode == RewardShapingMode.MARGIN
    assert config.reward_shaping.alpha == 0.5


def test_multiagent_training_config_custom():
    """Test MultiAgentTrainingConfig with custom values."""
    config = MultiAgentTrainingConfig(
        pretrain_path="models/custom",
        trajectory_path="data/trajectories.jsonl",
        save_path="output/checkpoints",
        ray_config=RayJobConfig(num_nodes=5, num_gpus_per_node=4),
        vllm_config=VLLMConfig(num_engines=4, tensor_parallel_size=2),
        openrlhf_config=OpenRLHFConfig(algorithm=TrainingAlgorithm.PPO, num_episodes=10),
        reward_shaping=RewardShapingConfig(mode=RewardShapingMode.QUALITY, alpha=0.7),
        num_agents=7,
        num_rounds=5,
        shared_agents=True,
        wandb_project="my-project",
        wandb_run_name="experiment-001",
    )

    assert config.pretrain_path == "models/custom"
    assert config.ray_config.num_nodes == 5
    assert config.vllm_config.num_engines == 4
    assert config.openrlhf_config.algorithm == TrainingAlgorithm.PPO
    assert config.reward_shaping.mode == RewardShapingMode.QUALITY
    assert config.num_agents == 7
    assert config.shared_agents is True
    assert config.wandb_project == "my-project"


def test_validate_config_valid():
    """Test validate_config accepts valid configuration."""
    config = build_default_config(
        pretrain_path="models/cohere-7b",
        trajectory_path="trajectories/train.jsonl",
        save_path="checkpoints/run_001",
    )

    errors = validate_config(config)
    assert errors == []


def test_validate_config_missing_required_fields():
    """Test validate_config catches missing required fields."""
    config = MultiAgentTrainingConfig(
        pretrain_path="",
        trajectory_path="",
        save_path="",
    )

    errors = validate_config(config)
    assert len(errors) >= 3
    assert any("pretrain_path is required" in err for err in errors)
    assert any("trajectory_path is required" in err for err in errors)
    assert any("save_path is required" in err for err in errors)


def test_validate_config_batch_size_consistency():
    """Test validate_config catches inconsistent batch sizes."""
    config = build_default_config(
        pretrain_path="models/test",
        trajectory_path="data/test.jsonl",
        save_path="output/test",
    )

    # Make train_batch_size not divisible by micro_train_batch_size
    config.openrlhf_config.train_batch_size = 100
    config.openrlhf_config.micro_train_batch_size = 7

    errors = validate_config(config)
    assert any("train_batch_size" in err and "divisible" in err for err in errors)


def test_validate_config_gpu_allocation():
    """Test validate_config catches insufficient GPU allocation."""
    config = build_default_config(
        pretrain_path="models/test",
        trajectory_path="data/test.jsonl",
        save_path="output/test",
    )

    # Request more GPUs than available
    config.ray_config.num_nodes = 2
    config.ray_config.num_gpus_per_node = 2  # Total: 4 GPUs
    config.vllm_config.num_engines = 3
    config.vllm_config.tensor_parallel_size = 2  # Needs: 6 GPUs

    errors = validate_config(config)
    assert any("vLLM requires" in err and "GPUs available" in err for err in errors)


def test_validate_config_algorithm_requirements():
    """Test validate_config checks algorithm-specific requirements."""
    config = build_default_config(
        pretrain_path="models/test",
        trajectory_path="data/test.jsonl",
        save_path="output/test",
    )

    # RLOO requires n_samples_per_prompt >= 2
    config.openrlhf_config.algorithm = TrainingAlgorithm.RLOO
    config.openrlhf_config.n_samples_per_prompt = 1

    errors = validate_config(config)
    assert any("rloo requires n_samples_per_prompt >= 2" in err.lower() for err in errors)


def test_validate_config_num_agents():
    """Test validate_config checks minimum number of agents."""
    config = build_default_config(
        pretrain_path="models/test",
        trajectory_path="data/test.jsonl",
        save_path="output/test",
    )

    # Less than 5 agents (3 solvers + verifier + judge)
    config.num_agents = 3

    errors = validate_config(config)
    assert any("num_agents" in err and "at least 5" in err for err in errors)


def test_config_serialization():
    """Test config can be serialized to/from JSON."""
    config = build_default_config(
        pretrain_path="models/cohere-7b",
        trajectory_path="trajectories/train.jsonl",
        save_path="checkpoints/run_001",
    )

    # Serialize to JSON
    config_json = config.model_dump_json()
    config_dict = json.loads(config_json)

    # Check key fields present
    assert config_dict["pretrain_path"] == "models/cohere-7b"
    assert config_dict["trajectory_path"] == "trajectories/train.jsonl"
    assert config_dict["ray_config"]["num_nodes"] == 3
    assert config_dict["openrlhf_config"]["algorithm"] == "reinforce"
    assert config_dict["reward_shaping"]["mode"] == "margin"

    # Deserialize from JSON
    restored_config = MultiAgentTrainingConfig.model_validate_json(config_json)

    assert restored_config.pretrain_path == config.pretrain_path
    assert restored_config.trajectory_path == config.trajectory_path
    assert restored_config.ray_config.num_nodes == config.ray_config.num_nodes
    assert restored_config.openrlhf_config.algorithm == config.openrlhf_config.algorithm
    assert restored_config.reward_shaping.mode == config.reward_shaping.mode


def test_reward_shaping_integration():
    """Test RewardShapingConfig integration from 03-01."""
    config = build_default_config(
        pretrain_path="models/test",
        trajectory_path="data/test.jsonl",
        save_path="output/test",
    )

    # Verify RewardShapingConfig is properly integrated
    assert isinstance(config.reward_shaping, RewardShapingConfig)
    assert config.reward_shaping.mode == RewardShapingMode.MARGIN
    assert config.reward_shaping.alpha == 0.5
    assert config.reward_shaping.beta == 0.5

    # Test with different reward shaping modes
    config.reward_shaping = RewardShapingConfig(
        mode=RewardShapingMode.QUALITY,
        alpha=0.3,
        beta=0.7,
    )
    assert config.reward_shaping.mode == RewardShapingMode.QUALITY
    assert config.reward_shaping.alpha == 0.3
    assert config.reward_shaping.beta == 0.7


def test_pydantic_validation():
    """Test Pydantic validates types and ranges."""
    # Invalid type
    with pytest.raises(Exception):  # Pydantic ValidationError
        VLLMConfig(num_engines="not-a-number")

    # Out of range
    with pytest.raises(Exception):
        OpenRLHFConfig(gamma=1.5)  # Must be <= 1.0

    # Negative value
    with pytest.raises(Exception):
        OpenRLHFConfig(num_episodes=-1)  # Must be >= 1
