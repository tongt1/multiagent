"""Ray training job executor for multi-agent RL training.

Provides Ray job submission for training jobs with:
- JobSubmissionClient integration for Ray clusters
- Runtime environment configuration with pip packages
- Environment variable passing for config and paths
- Job status monitoring and log retrieval
- Graceful handling of missing Ray dependency

Follows MARTI training patterns for distributed RL on Ray.
"""

import json
import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from src.infrastructure.distributed_executor import JobInfo, JobStatus
from src.training.training_config import MultiAgentTrainingConfig

# Check if ray[client] is available
try:
    from ray.job_submission import JobStatus as RayJobStatus
    from ray.job_submission import JobSubmissionClient

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.debug("ray[client] not installed - Ray training executor unavailable")


class RayTrainingExecutor:
    """Ray job submission for multi-agent RL training.

    Uses Ray JobSubmissionClient to submit training jobs to Ray cluster.
    Handles runtime environment setup, config serialization, and status monitoring.

    Example:
        >>> executor = RayTrainingExecutor("http://localhost:8265")
        >>> job_id = executor.submit_training_job(config)
        >>> status = executor.get_training_job_status(job_id)
        >>> print(status.status)
        JobStatus.RUNNING
    """

    def __init__(self, ray_address: str = "http://localhost:8265") -> None:
        """Initialize Ray JobSubmissionClient.

        Args:
            ray_address: Ray cluster dashboard address (default: http://localhost:8265)

        Raises:
            ImportError: If ray[client] not installed
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "ray[client] not installed. Install with: pip install 'ray[client]'"
            )

        self.ray_address = ray_address
        self.client = JobSubmissionClient(ray_address)
        logger.info(f"Initialized RayTrainingExecutor with address: {ray_address}")

    def submit_training_job(
        self, config: MultiAgentTrainingConfig, job_name: Optional[str] = None
    ) -> str:
        """Submit training job to Ray cluster.

        Creates runtime environment with required packages, serializes config to JSON,
        and submits job with train_marti.py entrypoint.

        Args:
            config: Training configuration with model paths, hyperparameters, etc.
            job_name: Optional job name (default: auto-generated with timestamp)

        Returns:
            job_id: Ray job ID for status monitoring

        Example:
            >>> config = build_default_config("model", "data.jsonl", "output")
            >>> job_id = executor.submit_training_job(config, job_name="my-training-run")
        """
        # Generate unique job name if not provided
        if job_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            job_name = f"marti-train-{timestamp}"

        logger.info(f"Submitting training job: {job_name}")

        # Build runtime environment with required packages
        runtime_env = {
            "pip": [
                "hydra-core",
                "openrlhf",
                "vllm",
                "deepspeed",
                "wandb",
                "numpy",
                "torch",
                "transformers",
            ],
            "env_vars": {},
        }

        # Merge user-provided runtime_env
        if config.ray_config.runtime_env:
            if "pip" in config.ray_config.runtime_env:
                runtime_env["pip"].extend(config.ray_config.runtime_env["pip"])
            if "env_vars" in config.ray_config.runtime_env:
                runtime_env["env_vars"].update(config.ray_config.runtime_env["env_vars"])

        # Serialize config to JSON for environment variable
        config_json = config.model_dump_json()

        # Set environment variables for training script
        runtime_env["env_vars"].update(
            {
                "TRAINING_CONFIG_JSON": config_json,
                "TRAJECTORY_PATH": config.trajectory_path,
                "PRETRAIN_PATH": config.pretrain_path,
                "SAVE_PATH": config.save_path,
                "REWARD_SHAPING_MODE": config.reward_shaping.mode.value,
                "REWARD_SHAPING_ALPHA": str(config.reward_shaping.alpha),
                "ALGORITHM": config.openrlhf_config.algorithm.value,
            }
        )

        # Add W&B config if provided
        if config.wandb_project:
            runtime_env["env_vars"]["WANDB_PROJECT"] = config.wandb_project
        if config.wandb_run_name:
            runtime_env["env_vars"]["WANDB_RUN_NAME"] = config.wandb_run_name

        # Build entrypoint command
        entrypoint = "python3 -m src.training.train_marti"

        logger.debug(f"Runtime env: {runtime_env}")
        logger.debug(f"Entrypoint: {entrypoint}")

        # Submit job to Ray
        job_id = self.client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            metadata={"job_name": job_name},
        )

        logger.info(f"Job submitted with ID: {job_id}")
        return job_id

    def get_training_job_status(self, job_id: str) -> JobInfo:
        """Get training job status.

        Queries Ray cluster for job status and maps to JobInfo model.

        Args:
            job_id: Ray job ID

        Returns:
            JobInfo with status, timestamps, and message

        Example:
            >>> status = executor.get_training_job_status(job_id)
            >>> print(f"{status.job_name}: {status.status}")
        """
        try:
            # Get status from Ray
            ray_status = self.client.get_job_status(job_id)

            # Map Ray JobStatus to our JobStatus enum
            status_map = {
                RayJobStatus.PENDING: JobStatus.PENDING,
                RayJobStatus.RUNNING: JobStatus.RUNNING,
                RayJobStatus.SUCCEEDED: JobStatus.SUCCEEDED,
                RayJobStatus.FAILED: JobStatus.FAILED,
                RayJobStatus.STOPPED: JobStatus.FAILED,
            }

            status = status_map.get(ray_status, JobStatus.UNKNOWN)

            # Get job metadata
            job_info_dict = self.client.get_job_info(job_id)
            metadata = job_info_dict.metadata or {}
            job_name = metadata.get("job_name", job_id)

            # Extract timestamps
            created_at = None
            completed_at = None
            if job_info_dict.start_time:
                created_at = datetime.fromtimestamp(
                    job_info_dict.start_time / 1000, tz=timezone.utc
                ).isoformat()
            if job_info_dict.end_time:
                completed_at = datetime.fromtimestamp(
                    job_info_dict.end_time / 1000, tz=timezone.utc
                ).isoformat()

            # Get error message if failed
            message = None
            if status == JobStatus.FAILED:
                message = job_info_dict.message or "Training job failed"

            return JobInfo(
                job_name=job_name,
                namespace="ray",
                status=status,
                created_at=created_at,
                completed_at=completed_at,
                num_pods=1,
                succeeded_pods=1 if status == JobStatus.SUCCEEDED else 0,
                failed_pods=1 if status == JobStatus.FAILED else 0,
                message=message,
            )

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return JobInfo(
                job_name=job_id,
                namespace="ray",
                status=JobStatus.UNKNOWN,
                message=f"Error getting status: {e}",
            )

    def cancel_training_job(self, job_id: str) -> bool:
        """Cancel training job.

        Stops the running job on Ray cluster.

        Args:
            job_id: Ray job ID

        Returns:
            True if cancelled, False if not found or already completed

        Example:
            >>> cancelled = executor.cancel_training_job(job_id)
            >>> print(f"Cancelled: {cancelled}")
        """
        try:
            self.client.stop_job(job_id)
            logger.info(f"Cancelled training job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False

    def get_training_job_logs(self, job_id: str) -> str:
        """Get training job logs.

        Retrieves logs from Ray cluster for the job.

        Args:
            job_id: Ray job ID

        Returns:
            Log string (may be empty if job not started)

        Example:
            >>> logs = executor.get_training_job_logs(job_id)
            >>> print(logs)
        """
        try:
            logs = self.client.get_job_logs(job_id)
            return logs
        except Exception as e:
            logger.error(f"Failed to get job logs: {e}")
            return f"Error retrieving logs: {e}"


def submit_training_job(
    config: MultiAgentTrainingConfig, ray_address: str = "http://localhost:8265"
) -> str:
    """Submit training job to Ray cluster (convenience function).

    Args:
        config: Training configuration
        ray_address: Ray cluster dashboard address

    Returns:
        job_id: Ray job ID for monitoring

    Example:
        >>> from src.training.training_config import build_default_config
        >>> config = build_default_config("model", "data.jsonl", "output")
        >>> job_id = submit_training_job(config)
    """
    executor = RayTrainingExecutor(ray_address)
    return executor.submit_training_job(config)


def get_training_job_status(
    job_id: str, ray_address: str = "http://localhost:8265"
) -> JobInfo:
    """Get training job status (convenience function).

    Args:
        job_id: Ray job ID
        ray_address: Ray cluster dashboard address

    Returns:
        JobInfo with current status

    Example:
        >>> status = get_training_job_status(job_id)
        >>> print(status.status)
    """
    executor = RayTrainingExecutor(ray_address)
    return executor.get_training_job_status(job_id)
