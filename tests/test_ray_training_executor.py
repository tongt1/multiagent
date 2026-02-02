"""Tests for Ray training executor."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.infrastructure.distributed_executor import JobInfo, JobStatus
from src.training.training_config import (
    MultiAgentTrainingConfig,
    RewardShapingMode,
    TrainingAlgorithm,
    build_default_config,
)


# Mock Ray imports since ray[client] may not be installed
@pytest.fixture
def mock_ray():
    """Mock ray.job_submission module."""
    with patch("src.infrastructure.ray_training_executor.RAY_AVAILABLE", True):
        # Mock RayJobStatus enum
        mock_status = Mock()
        mock_status.PENDING = "PENDING"
        mock_status.RUNNING = "RUNNING"
        mock_status.SUCCEEDED = "SUCCEEDED"
        mock_status.FAILED = "FAILED"
        mock_status.STOPPED = "STOPPED"

        with patch(
            "src.infrastructure.ray_training_executor.RayJobStatus", mock_status
        ), patch(
            "src.infrastructure.ray_training_executor.JobSubmissionClient"
        ) as mock_client_class:
            yield mock_client_class


def test_ray_training_executor_init(mock_ray):
    """Test RayTrainingExecutor initialization."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    executor = RayTrainingExecutor("http://localhost:8265")
    assert executor.ray_address == "http://localhost:8265"
    mock_ray.assert_called_once_with("http://localhost:8265")


def test_ray_training_executor_import_error():
    """Test RayTrainingExecutor raises ImportError when Ray not available."""
    with patch("src.infrastructure.ray_training_executor.RAY_AVAILABLE", False):
        from src.infrastructure.ray_training_executor import RayTrainingExecutor

        with pytest.raises(ImportError, match="ray\\[client\\] not installed"):
            RayTrainingExecutor()


def test_submit_training_job(mock_ray):
    """Test submit_training_job creates correct entrypoint and runtime_env."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    # Create mock client
    mock_client = MagicMock()
    mock_client.submit_job.return_value = "job-123"
    mock_ray.return_value = mock_client

    executor = RayTrainingExecutor("http://localhost:8265")

    # Create test config
    config = build_default_config(
        pretrain_path="models/cohere-7b",
        trajectory_path="trajectories/train.jsonl",
        save_path="checkpoints/run_001",
    )

    # Submit job
    job_id = executor.submit_training_job(config, job_name="test-training")

    assert job_id == "job-123"

    # Verify submit_job was called
    mock_client.submit_job.assert_called_once()
    call_args = mock_client.submit_job.call_args

    # Check entrypoint
    assert call_args.kwargs["entrypoint"] == "python3 -m src.training.train_marti"

    # Check runtime_env
    runtime_env = call_args.kwargs["runtime_env"]
    assert "pip" in runtime_env
    assert "openrlhf" in runtime_env["pip"]
    assert "vllm" in runtime_env["pip"]
    assert "deepspeed" in runtime_env["pip"]

    # Check env vars
    env_vars = runtime_env["env_vars"]
    assert "TRAINING_CONFIG_JSON" in env_vars
    assert env_vars["TRAJECTORY_PATH"] == "trajectories/train.jsonl"
    assert env_vars["PRETRAIN_PATH"] == "models/cohere-7b"
    assert env_vars["SAVE_PATH"] == "checkpoints/run_001"
    assert env_vars["REWARD_SHAPING_MODE"] == "margin"
    assert env_vars["ALGORITHM"] == "reinforce"

    # Check metadata
    assert call_args.kwargs["metadata"]["job_name"] == "test-training"


def test_submit_training_job_auto_name(mock_ray):
    """Test submit_training_job generates job name if not provided."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    mock_client = MagicMock()
    mock_client.submit_job.return_value = "job-456"
    mock_ray.return_value = mock_client

    executor = RayTrainingExecutor()
    config = build_default_config("model", "data.jsonl", "output")

    job_id = executor.submit_training_job(config)

    assert job_id == "job-456"

    # Check that job_name was auto-generated
    call_args = mock_client.submit_job.call_args
    job_name = call_args.kwargs["metadata"]["job_name"]
    assert job_name.startswith("marti-train-")


def test_submit_training_job_with_wandb(mock_ray):
    """Test submit_training_job includes W&B config when provided."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    mock_client = MagicMock()
    mock_client.submit_job.return_value = "job-789"
    mock_ray.return_value = mock_client

    executor = RayTrainingExecutor()

    config = build_default_config("model", "data.jsonl", "output")
    config.wandb_project = "my-project"
    config.wandb_run_name = "experiment-001"

    executor.submit_training_job(config)

    call_args = mock_client.submit_job.call_args
    env_vars = call_args.kwargs["runtime_env"]["env_vars"]

    assert env_vars["WANDB_PROJECT"] == "my-project"
    assert env_vars["WANDB_RUN_NAME"] == "experiment-001"


def test_get_training_job_status_running(mock_ray):
    """Test get_training_job_status maps Ray status to JobInfo."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    mock_client = MagicMock()
    mock_ray.return_value = mock_client

    # Mock job status
    mock_client.get_job_status.return_value = "RUNNING"

    # Mock job info
    mock_job_info = Mock()
    mock_job_info.start_time = 1609459200000  # 2021-01-01 00:00:00 UTC
    mock_job_info.end_time = None
    mock_job_info.message = None
    mock_job_info.metadata = {"job_name": "test-job"}
    mock_client.get_job_info.return_value = mock_job_info

    executor = RayTrainingExecutor()
    job_info = executor.get_training_job_status("job-123")

    assert job_info.job_name == "test-job"
    assert job_info.status == JobStatus.RUNNING
    assert job_info.namespace == "ray"
    assert job_info.created_at is not None
    assert job_info.completed_at is None


def test_get_training_job_status_succeeded(mock_ray):
    """Test get_training_job_status handles succeeded status."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    mock_client = MagicMock()
    mock_ray.return_value = mock_client

    mock_client.get_job_status.return_value = "SUCCEEDED"

    mock_job_info = Mock()
    mock_job_info.start_time = 1609459200000
    mock_job_info.end_time = 1609462800000  # 1 hour later
    mock_job_info.message = None
    mock_job_info.metadata = {"job_name": "completed-job"}
    mock_client.get_job_info.return_value = mock_job_info

    executor = RayTrainingExecutor()
    job_info = executor.get_training_job_status("job-456")

    assert job_info.job_name == "completed-job"
    assert job_info.status == JobStatus.SUCCEEDED
    assert job_info.succeeded_pods == 1
    assert job_info.failed_pods == 0
    assert job_info.completed_at is not None


def test_get_training_job_status_failed(mock_ray):
    """Test get_training_job_status handles failed status."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    mock_client = MagicMock()
    mock_ray.return_value = mock_client

    mock_client.get_job_status.return_value = "FAILED"

    mock_job_info = Mock()
    mock_job_info.start_time = 1609459200000
    mock_job_info.end_time = 1609460100000
    mock_job_info.message = "Out of memory"
    mock_job_info.metadata = {"job_name": "failed-job"}
    mock_client.get_job_info.return_value = mock_job_info

    executor = RayTrainingExecutor()
    job_info = executor.get_training_job_status("job-789")

    assert job_info.job_name == "failed-job"
    assert job_info.status == JobStatus.FAILED
    assert job_info.succeeded_pods == 0
    assert job_info.failed_pods == 1
    assert "Out of memory" in job_info.message


def test_get_training_job_status_error(mock_ray):
    """Test get_training_job_status handles errors gracefully."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    mock_client = MagicMock()
    mock_ray.return_value = mock_client

    mock_client.get_job_status.side_effect = Exception("Connection error")

    executor = RayTrainingExecutor()
    job_info = executor.get_training_job_status("job-404")

    assert job_info.job_name == "job-404"
    assert job_info.status == JobStatus.UNKNOWN
    assert "Connection error" in job_info.message


def test_cancel_training_job(mock_ray):
    """Test cancel_training_job calls stop_job."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    mock_client = MagicMock()
    mock_ray.return_value = mock_client

    executor = RayTrainingExecutor()
    result = executor.cancel_training_job("job-123")

    assert result is True
    mock_client.stop_job.assert_called_once_with("job-123")


def test_cancel_training_job_error(mock_ray):
    """Test cancel_training_job handles errors."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    mock_client = MagicMock()
    mock_ray.return_value = mock_client

    mock_client.stop_job.side_effect = Exception("Job not found")

    executor = RayTrainingExecutor()
    result = executor.cancel_training_job("job-404")

    assert result is False


def test_get_training_job_logs(mock_ray):
    """Test get_training_job_logs retrieves logs."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    mock_client = MagicMock()
    mock_ray.return_value = mock_client

    mock_client.get_job_logs.return_value = "Training started\nEpoch 1 complete\n"

    executor = RayTrainingExecutor()
    logs = executor.get_training_job_logs("job-123")

    assert "Training started" in logs
    assert "Epoch 1 complete" in logs
    mock_client.get_job_logs.assert_called_once_with("job-123")


def test_get_training_job_logs_error(mock_ray):
    """Test get_training_job_logs handles errors."""
    from src.infrastructure.ray_training_executor import RayTrainingExecutor

    mock_client = MagicMock()
    mock_ray.return_value = mock_client

    mock_client.get_job_logs.side_effect = Exception("Logs unavailable")

    executor = RayTrainingExecutor()
    logs = executor.get_training_job_logs("job-123")

    assert "Error retrieving logs" in logs


def test_submit_training_job_convenience_function(mock_ray):
    """Test submit_training_job convenience function."""
    from src.infrastructure.ray_training_executor import submit_training_job

    mock_client = MagicMock()
    mock_client.submit_job.return_value = "job-999"
    mock_ray.return_value = mock_client

    config = build_default_config("model", "data.jsonl", "output")
    job_id = submit_training_job(config, ray_address="http://custom:8265")

    assert job_id == "job-999"
    mock_ray.assert_called_with("http://custom:8265")


def test_get_training_job_status_convenience_function(mock_ray):
    """Test get_training_job_status convenience function."""
    from src.infrastructure.ray_training_executor import get_training_job_status

    mock_client = MagicMock()
    mock_ray.return_value = mock_client

    mock_client.get_job_status.return_value = "PENDING"
    mock_job_info = Mock()
    mock_job_info.start_time = None
    mock_job_info.end_time = None
    mock_job_info.message = None
    mock_job_info.metadata = {}
    mock_client.get_job_info.return_value = mock_job_info

    job_info = get_training_job_status("job-123", ray_address="http://custom:8265")

    assert job_info.status == JobStatus.PENDING
    mock_ray.assert_called_with("http://custom:8265")


def test_kjobs_training_executor_placeholder():
    """Test KjobsTrainingExecutor raises NotImplementedError."""
    from src.infrastructure.distributed_executor import KjobsTrainingExecutor

    executor = KjobsTrainingExecutor(
        api_endpoint="https://kjobs.example.com", api_key="test-key"
    )

    with pytest.raises(NotImplementedError, match="kjobs/apiary training API not yet available"):
        executor.submit_training_job({})

    with pytest.raises(NotImplementedError, match="kjobs/apiary training API not yet available"):
        executor.get_training_job_status("job-123")

    with pytest.raises(NotImplementedError, match="kjobs/apiary training API not yet available"):
        executor.cancel_training_job("job-123")

    with pytest.raises(NotImplementedError, match="kjobs/apiary training API not yet available"):
        executor.get_training_job_logs("job-123")
