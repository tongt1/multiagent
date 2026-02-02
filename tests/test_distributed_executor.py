"""Tests for distributed executor infrastructure."""

import sys
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.distributed_executor import (
    JobConfig,
    JobInfo,
    JobStatus,
    KjobsExecutor,
    KubernetesExecutor,
)


class TestJobModels:
    """Test Pydantic models for job configuration and status."""

    def test_job_config_defaults(self) -> None:
        """Test JobConfig with minimal required fields."""
        config = JobConfig(name="test-job", image="test:latest")

        assert config.name == "test-job"
        assert config.image == "test:latest"
        assert config.namespace == "default"
        assert config.cpu_request == "2"
        assert config.memory_limit == "16Gi"
        assert config.backoff_limit == 3
        assert config.env_vars == {}

    def test_job_config_custom_values(self) -> None:
        """Test JobConfig with custom values."""
        config = JobConfig(
            name="my-job",
            image="my-image:v1",
            namespace="ml-jobs",
            cpu_request="4",
            cpu_limit="8",
            memory_request="16Gi",
            memory_limit="32Gi",
            backoff_limit=5,
            env_vars={"API_KEY": "secret"},
        )

        assert config.namespace == "ml-jobs"
        assert config.cpu_limit == "8"
        assert config.memory_request == "16Gi"
        assert config.env_vars["API_KEY"] == "secret"

    def test_job_status_enum(self) -> None:
        """Test JobStatus enum values."""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.SUCCEEDED == "succeeded"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.UNKNOWN == "unknown"

    def test_job_info_creation(self) -> None:
        """Test JobInfo model creation."""
        info = JobInfo(
            job_name="test-job-123",
            namespace="default",
            status=JobStatus.RUNNING,
            created_at="2024-01-01T00:00:00Z",
            num_pods=2,
            succeeded_pods=1,
            failed_pods=0,
        )

        assert info.job_name == "test-job-123"
        assert info.status == JobStatus.RUNNING
        assert info.num_pods == 2
        assert info.succeeded_pods == 1


@pytest.mark.asyncio
class TestKubernetesExecutor:
    """Test KubernetesExecutor implementation."""

    @patch("src.infrastructure.distributed_executor.KUBERNETES_AVAILABLE", False)
    def test_init_without_kubernetes_library(self) -> None:
        """Test KubernetesExecutor raises ImportError when kubernetes not installed."""
        with pytest.raises(ImportError, match="kubernetes library not installed"):
            KubernetesExecutor(in_cluster=False)

    @patch("src.infrastructure.distributed_executor.KUBERNETES_AVAILABLE", True)
    @patch("src.infrastructure.distributed_executor.config")
    @patch("src.infrastructure.distributed_executor.client")
    def test_init_with_kubeconfig(
        self,
        mock_client: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test KubernetesExecutor initialization with kubeconfig."""
        executor = KubernetesExecutor(in_cluster=False)

        mock_config.load_kube_config.assert_called_once()
        assert executor.batch_api is not None
        assert executor.core_api is not None

    @patch("src.infrastructure.distributed_executor.KUBERNETES_AVAILABLE", True)
    @patch("src.infrastructure.distributed_executor.config")
    @patch("src.infrastructure.distributed_executor.client")
    async def test_submit_job(
        self,
        mock_client: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test job submission to Kubernetes."""
        # Setup mocks
        mock_batch_api = MagicMock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        executor = KubernetesExecutor(in_cluster=False)

        # Prepare test data
        problems = [
            {"id": "p1", "problem": "What is 2+2?", "domain": "math"},
            {"id": "p2", "problem": "What is 3+3?", "domain": "math"},
        ]

        job_config = JobConfig(
            name="test-job",
            image="test:latest",
            env_vars={"API_KEY": "test-key"},
        )

        # Mock the create_namespaced_job call
        async def mock_create(*args: Any, **kwargs: Any) -> None:
            # Verify job creation is called with correct parameters
            assert kwargs["namespace"] == "default"
            job_body = kwargs["body"]
            assert "test-job" in job_body.metadata.name
            assert job_body.spec.template.spec.containers[0].image == "test:latest"

        with patch("asyncio.to_thread", side_effect=mock_create):
            job_id = await executor.submit_job(
                problems=problems,
                pipeline_config_path="config/pipeline.yaml",
                job_config=job_config,
            )

        # Verify job_id format
        assert job_id.startswith("test-job-")
        assert len(job_id) > len("test-job-")

    @patch("src.infrastructure.distributed_executor.KUBERNETES_AVAILABLE", True)
    @patch("src.infrastructure.distributed_executor.config")
    @patch("src.infrastructure.distributed_executor.client")
    async def test_get_status_succeeded(
        self,
        mock_client: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test getting status of a succeeded job."""
        mock_batch_api = MagicMock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        executor = KubernetesExecutor(in_cluster=False)

        # Mock job object
        mock_job = MagicMock()
        mock_job.metadata.name = "test-job-123"
        mock_job.metadata.creation_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_job.status.succeeded = 1
        mock_job.status.failed = 0
        mock_job.status.active = 0
        mock_job.status.completion_time = datetime(2024, 1, 1, 0, 5, 0, tzinfo=timezone.utc)

        async def mock_read(*args: Any, **kwargs: Any) -> Any:
            return mock_job

        with patch("asyncio.to_thread", side_effect=mock_read):
            job_info = await executor.get_status("test-job-123")

        assert job_info.job_name == "test-job-123"
        assert job_info.status == JobStatus.SUCCEEDED
        assert job_info.succeeded_pods == 1
        assert job_info.failed_pods == 0
        assert job_info.completed_at is not None

    @patch("src.infrastructure.distributed_executor.KUBERNETES_AVAILABLE", True)
    @patch("src.infrastructure.distributed_executor.config")
    @patch("src.infrastructure.distributed_executor.client")
    async def test_get_status_running(
        self,
        mock_client: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test getting status of a running job."""
        mock_batch_api = MagicMock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        executor = KubernetesExecutor(in_cluster=False)

        # Mock running job
        mock_job = MagicMock()
        mock_job.metadata.name = "test-job-456"
        mock_job.metadata.creation_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_job.status.succeeded = 0
        mock_job.status.failed = 0
        mock_job.status.active = 1
        mock_job.status.completion_time = None

        async def mock_read(*args: Any, **kwargs: Any) -> Any:
            return mock_job

        with patch("asyncio.to_thread", side_effect=mock_read):
            job_info = await executor.get_status("test-job-456")

        assert job_info.job_name == "test-job-456"
        assert job_info.status == JobStatus.RUNNING
        assert job_info.num_pods == 1
        assert job_info.completed_at is None

    @patch("src.infrastructure.distributed_executor.KUBERNETES_AVAILABLE", True)
    @patch("src.infrastructure.distributed_executor.config")
    @patch("src.infrastructure.distributed_executor.client")
    async def test_get_status_not_found(
        self,
        mock_client: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test getting status of non-existent job."""
        from kubernetes.client.rest import ApiException

        mock_batch_api = MagicMock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        executor = KubernetesExecutor(in_cluster=False)

        # Mock 404 exception
        async def mock_read(*args: Any, **kwargs: Any) -> None:
            raise ApiException(status=404, reason="Not Found")

        with patch("asyncio.to_thread", side_effect=mock_read):
            job_info = await executor.get_status("nonexistent-job")

        assert job_info.job_name == "nonexistent-job"
        assert job_info.status == JobStatus.UNKNOWN
        assert job_info.message == "Job not found"

    @patch("src.infrastructure.distributed_executor.KUBERNETES_AVAILABLE", True)
    @patch("src.infrastructure.distributed_executor.config")
    @patch("src.infrastructure.distributed_executor.client")
    async def test_cancel_job(
        self,
        mock_client: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test cancelling a job."""
        mock_batch_api = MagicMock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        executor = KubernetesExecutor(in_cluster=False)

        # Mock successful deletion
        async def mock_delete(*args: Any, **kwargs: Any) -> None:
            assert kwargs["name"] == "test-job-789"
            assert kwargs["namespace"] == "default"
            assert kwargs["propagation_policy"] == "Background"

        with patch("asyncio.to_thread", side_effect=mock_delete):
            result = await executor.cancel_job("test-job-789")

        assert result is True

    @patch("src.infrastructure.distributed_executor.KUBERNETES_AVAILABLE", True)
    @patch("src.infrastructure.distributed_executor.config")
    @patch("src.infrastructure.distributed_executor.client")
    async def test_wait_for_completion_success(
        self,
        mock_client: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test waiting for job completion."""
        mock_batch_api = MagicMock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        executor = KubernetesExecutor(in_cluster=False)

        # Mock job transitioning from running to succeeded
        call_count = 0

        async def mock_get_status(job_id: str) -> JobInfo:
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                # First two calls: running
                return JobInfo(
                    job_name=job_id,
                    namespace="default",
                    status=JobStatus.RUNNING,
                    num_pods=1,
                )
            else:
                # Third call: succeeded
                return JobInfo(
                    job_name=job_id,
                    namespace="default",
                    status=JobStatus.SUCCEEDED,
                    num_pods=1,
                    succeeded_pods=1,
                )

        # Patch get_status method
        executor.get_status = mock_get_status  # type: ignore

        job_info = await executor.wait_for_completion(
            "test-job",
            poll_interval=0.1,  # Fast polling for test
            timeout=10.0,
        )

        assert job_info.status == JobStatus.SUCCEEDED
        assert call_count >= 3

    @patch("src.infrastructure.distributed_executor.KUBERNETES_AVAILABLE", True)
    @patch("src.infrastructure.distributed_executor.config")
    @patch("src.infrastructure.distributed_executor.client")
    async def test_wait_for_completion_timeout(
        self,
        mock_client: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test waiting for job completion with timeout."""
        mock_batch_api = MagicMock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        executor = KubernetesExecutor(in_cluster=False)

        # Mock job stuck in running state
        async def mock_get_status(job_id: str) -> JobInfo:
            return JobInfo(
                job_name=job_id,
                namespace="default",
                status=JobStatus.RUNNING,
                num_pods=1,
            )

        executor.get_status = mock_get_status  # type: ignore

        with pytest.raises(TimeoutError, match="did not complete within"):
            await executor.wait_for_completion(
                "test-job",
                poll_interval=0.1,
                timeout=0.5,  # Short timeout for test
            )

    @patch("src.infrastructure.distributed_executor.KUBERNETES_AVAILABLE", True)
    @patch("src.infrastructure.distributed_executor.config")
    @patch("src.infrastructure.distributed_executor.client")
    async def test_list_jobs(
        self,
        mock_client: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test listing jobs in namespace."""
        mock_batch_api = MagicMock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        executor = KubernetesExecutor(in_cluster=False)

        # Mock job list
        mock_job1 = MagicMock()
        mock_job1.metadata.name = "job1"
        mock_job1.metadata.creation_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_job1.status.succeeded = 1
        mock_job1.status.failed = 0
        mock_job1.status.active = 0
        mock_job1.status.completion_time = datetime(2024, 1, 1, 0, 5, 0, tzinfo=timezone.utc)

        mock_job2 = MagicMock()
        mock_job2.metadata.name = "job2"
        mock_job2.metadata.creation_timestamp = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        mock_job2.status.succeeded = 0
        mock_job2.status.failed = 0
        mock_job2.status.active = 1
        mock_job2.status.completion_time = None

        mock_jobs_list = MagicMock()
        mock_jobs_list.items = [mock_job1, mock_job2]

        async def mock_list(*args: Any, **kwargs: Any) -> Any:
            return mock_jobs_list

        with patch("asyncio.to_thread", side_effect=mock_list):
            jobs = await executor.list_jobs(namespace="default")

        assert len(jobs) == 2
        assert jobs[0].job_name == "job1"
        assert jobs[0].status == JobStatus.SUCCEEDED
        assert jobs[1].job_name == "job2"
        assert jobs[1].status == JobStatus.RUNNING


@pytest.mark.asyncio
class TestKjobsExecutor:
    """Test KjobsExecutor placeholder."""

    def test_init(self) -> None:
        """Test KjobsExecutor initialization."""
        executor = KjobsExecutor(api_endpoint="https://kjobs.api", api_key="key123")

        assert executor.api_endpoint == "https://kjobs.api"
        assert executor.api_key == "key123"

    async def test_submit_job_not_implemented(self) -> None:
        """Test submit_job raises NotImplementedError."""
        executor = KjobsExecutor(api_endpoint="https://kjobs.api", api_key="key123")

        with pytest.raises(NotImplementedError, match="kjobs/apiary integration not implemented"):
            await executor.submit_job(
                problems=[],
                pipeline_config_path="config.yaml",
                job_config=JobConfig(name="test", image="test:latest"),
            )

    async def test_get_status_not_implemented(self) -> None:
        """Test get_status raises NotImplementedError."""
        executor = KjobsExecutor(api_endpoint="https://kjobs.api", api_key="key123")

        with pytest.raises(NotImplementedError, match="kjobs/apiary integration not implemented"):
            await executor.get_status("job-123")

    async def test_cancel_job_not_implemented(self) -> None:
        """Test cancel_job raises NotImplementedError."""
        executor = KjobsExecutor(api_endpoint="https://kjobs.api", api_key="key123")

        with pytest.raises(NotImplementedError, match="kjobs/apiary integration not implemented"):
            await executor.cancel_job("job-123")

    async def test_wait_for_completion_not_implemented(self) -> None:
        """Test wait_for_completion raises NotImplementedError."""
        executor = KjobsExecutor(api_endpoint="https://kjobs.api", api_key="key123")

        with pytest.raises(NotImplementedError, match="kjobs/apiary integration not implemented"):
            await executor.wait_for_completion("job-123")

    async def test_list_jobs_not_implemented(self) -> None:
        """Test list_jobs raises NotImplementedError."""
        executor = KjobsExecutor(api_endpoint="https://kjobs.api", api_key="key123")

        with pytest.raises(NotImplementedError, match="kjobs/apiary integration not implemented"):
            await executor.list_jobs()


def test_import_when_kubernetes_not_installed() -> None:
    """Test that distributed_executor can be imported even when kubernetes not installed."""
    # This test verifies that KUBERNETES_AVAILABLE flag works correctly
    # and module doesn't crash on import when kubernetes is missing

    # Simulate missing kubernetes by removing it from sys.modules temporarily
    kubernetes_modules = [mod for mod in sys.modules.keys() if mod.startswith("kubernetes")]
    removed_modules = {}

    try:
        for mod in kubernetes_modules:
            removed_modules[mod] = sys.modules.pop(mod)

        # Re-import the module
        import importlib

        import src.infrastructure.distributed_executor

        importlib.reload(src.infrastructure.distributed_executor)

        # Should be able to import classes
        from src.infrastructure.distributed_executor import (
            JobConfig,
            JobInfo,
            JobStatus,
        )

        # Models should work fine
        config = JobConfig(name="test", image="test:latest")
        assert config.name == "test"

    finally:
        # Restore kubernetes modules
        sys.modules.update(removed_modules)
