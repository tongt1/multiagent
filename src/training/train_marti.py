"""MARTI training entrypoint for multi-agent RL training jobs.

This is a basic entrypoint stub that:
1. Loads training configuration from environment variable
2. Validates setup (paths, config consistency)
3. Logs configuration for debugging
4. Placeholder for actual training loop (deferred to future work)

Expected environment variables:
- TRAINING_CONFIG_JSON: Serialized MultiAgentTrainingConfig JSON
- TRAJECTORY_PATH: Path to MARTI trajectory JSONL
- PRETRAIN_PATH: Base model checkpoint path
- SAVE_PATH: Output checkpoint directory

The actual training loop integration with OpenRLHF will be implemented in a future phase.
"""

import json
import os
import sys
from pathlib import Path

from loguru import logger

from src.training.training_config import MultiAgentTrainingConfig, validate_config


def load_config_from_env() -> MultiAgentTrainingConfig:
    """Load training configuration from environment variable.

    Returns:
        Parsed MultiAgentTrainingConfig

    Raises:
        ValueError: If TRAINING_CONFIG_JSON not set or invalid
    """
    config_json = os.getenv("TRAINING_CONFIG_JSON")
    if not config_json:
        raise ValueError("TRAINING_CONFIG_JSON environment variable not set")

    logger.info("Loading training configuration from environment")

    try:
        config = MultiAgentTrainingConfig.model_validate_json(config_json)
        logger.info("Successfully parsed training configuration")
        return config
    except Exception as e:
        logger.error(f"Failed to parse training config: {e}")
        raise ValueError(f"Invalid training config JSON: {e}")


def validate_setup(config: MultiAgentTrainingConfig) -> None:
    """Validate training setup before starting.

    Checks:
    - Config validation passes
    - Trajectory file exists
    - Pretrain path exists (if local)
    - Save path is writable

    Args:
        config: Training configuration

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating training setup")

    # Validate config
    errors = validate_config(config)
    if errors:
        logger.error(f"Configuration validation failed: {errors}")
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

    # Check trajectory file exists
    trajectory_path = Path(config.trajectory_path)
    if not trajectory_path.exists():
        raise ValueError(f"Trajectory file not found: {config.trajectory_path}")

    logger.info(f"Trajectory file found: {config.trajectory_path}")

    # Check pretrain path (skip if remote path like s3://)
    if not config.pretrain_path.startswith(("s3://", "gs://", "azure://")):
        pretrain_path = Path(config.pretrain_path)
        if not pretrain_path.exists():
            logger.warning(f"Pretrain path not found (may be remote): {config.pretrain_path}")

    # Check save path parent directory exists
    save_path = Path(config.save_path)
    if not save_path.parent.exists():
        logger.info(f"Creating save path parent directory: {save_path.parent}")
        save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Setup validation complete")


def log_configuration(config: MultiAgentTrainingConfig) -> None:
    """Log training configuration for debugging.

    Args:
        config: Training configuration
    """
    logger.info("=" * 80)
    logger.info("MARTI Training Configuration")
    logger.info("=" * 80)

    logger.info(f"Pretrain path: {config.pretrain_path}")
    logger.info(f"Trajectory path: {config.trajectory_path}")
    logger.info(f"Save path: {config.save_path}")

    logger.info("\nRay Configuration:")
    logger.info(f"  Nodes: {config.ray_config.num_nodes}")
    logger.info(f"  GPUs per node: {config.ray_config.num_gpus_per_node}")
    logger.info(
        f"  Total GPUs: {config.ray_config.num_nodes * config.ray_config.num_gpus_per_node}"
    )
    logger.info(f"  CPU per node: {config.ray_config.cpu_per_node}")
    logger.info(f"  Memory per node: {config.ray_config.memory_per_node}")

    logger.info("\nvLLM Configuration:")
    logger.info(f"  Engines: {config.vllm_config.num_engines}")
    logger.info(f"  Tensor parallel size: {config.vllm_config.tensor_parallel_size}")
    logger.info(f"  GPU memory utilization: {config.vllm_config.gpu_memory_utilization}")
    logger.info(f"  Max model length: {config.vllm_config.max_model_len}")

    logger.info("\nOpenRLHF Configuration:")
    logger.info(f"  Algorithm: {config.openrlhf_config.algorithm.value}")
    logger.info(f"  Episodes: {config.openrlhf_config.num_episodes}")
    logger.info(f"  Train batch size: {config.openrlhf_config.train_batch_size}")
    logger.info(f"  Rollout batch size: {config.openrlhf_config.rollout_batch_size}")
    logger.info(f"  Actor LR: {config.openrlhf_config.actor_learning_rate}")
    logger.info(f"  Critic LR: {config.openrlhf_config.critic_learning_rate}")

    logger.info("\nReward Shaping:")
    logger.info(f"  Mode: {config.reward_shaping.mode.value}")
    logger.info(f"  Alpha: {config.reward_shaping.alpha}")
    logger.info(f"  Beta: {config.reward_shaping.beta}")

    logger.info("\nMulti-Agent Configuration:")
    logger.info(f"  Num agents: {config.num_agents}")
    logger.info(f"  Num rounds: {config.num_rounds}")
    logger.info(f"  Shared agents: {config.shared_agents}")

    if config.wandb_project:
        logger.info(f"\nW&B Project: {config.wandb_project}")
        if config.wandb_run_name:
            logger.info(f"W&B Run: {config.wandb_run_name}")

    logger.info("=" * 80)


def main() -> int:
    """Main entrypoint for MARTI training.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger.info("Starting MARTI training entrypoint")

    try:
        # Load configuration from environment
        config = load_config_from_env()

        # Validate setup
        validate_setup(config)

        # Log configuration
        log_configuration(config)

        # TODO: Actual training loop integration
        # This will be implemented in a future phase with:
        # 1. Load MARTI trajectories
        # 2. Initialize vLLM engines
        # 3. Initialize OpenRLHF trainer
        # 4. Apply reward shaping
        # 5. Run training episodes
        # 6. Save checkpoints
        # 7. Log metrics to W&B

        logger.info("Training entrypoint stub complete (actual training loop not implemented)")
        logger.warning(
            "NOTE: This is a placeholder entrypoint. "
            "Actual training loop with OpenRLHF will be implemented in future work."
        )

        return 0

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
