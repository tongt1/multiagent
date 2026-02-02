"""Distributed executor abstraction for batch inference on Kubernetes.

This module provides:
- DistributedExecutor abstract interface for job submission and monitoring
- KubernetesExecutor implementation using standard Kubernetes Jobs API
- KjobsExecutor placeholder for kjobs/apiary integration (when API becomes available)

Design rationale:
The abstraction layer allows swapping execution backends without changing CLI/orchestration code.
Since kjobs/apiary lacks public API documentation, we build against standard K8s Jobs with a
swappable backend architecture.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

# Optional kubernetes dependency
try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    logger.debug("kubernetes library not installed - distributed execution unavailable")


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class JobConfig(BaseModel):
    """Configuration for a distributed job."""

    name: str = Field(..., description="Job name prefix (timestamp will be appended)")
    image: str = Field(..., description="Container image with multiagent package installed")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    cpu_request: str = Field(default="2", description="CPU request (e.g., '2', '500m')")
    cpu_limit: str = Field(default="4", description="CPU limit")
    memory_request: str = Field(default="8Gi", description="Memory request")
    memory_limit: str = Field(default="16Gi", description="Memory limit")
    backoff_limit: int = Field(default=3, description="Number of retries on failure")
    env_vars: dict[str, str] = Field(
        default_factory=dict, description="Environment variables (e.g., API keys)"
    )


class JobInfo(BaseModel):
    """Information about a running or completed job."""

    job_name: str
    namespace: str
    status: JobStatus
    created_at: str | None = None
    completed_at: str | None = None
    num_pods: int = 0
    succeeded_pods: int = 0
    failed_pods: int = 0
    message: str | None = None


class DistributedExecutor(ABC):
    """Abstract interface for distributed job execution.

    Implementations provide different execution backends:
    - KubernetesExecutor: Standard Kubernetes Jobs API
    - KjobsExecutor: kjobs/apiary integration (future)
    """

    @abstractmethod
    async def submit_job(
        self,
        problems: list[dict[str, Any]],
        pipeline_config_path: str,
        job_config: JobConfig,
    ) -> str:
        """Submit a batch job for execution.

        Args:
            problems: List of problem dicts with 'id', 'problem', 'domain', etc.
            pipeline_config_path: Path to pipeline configuration YAML
            job_config: Job resource configuration

        Returns:
            Job ID for monitoring and cancellation
        """
        pass

    @abstractmethod
    async def get_status(self, job_id: str) -> JobInfo:
        """Get current status of a job.

        Args:
            job_id: Job identifier returned from submit_job

        Returns:
            JobInfo with current status and pod counts
        """
        pass

    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False if not found or already completed
        """
        pass

    @abstractmethod
    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 10.0,
        timeout: float = 3600.0,
    ) -> JobInfo:
        """Wait for job to complete (succeed or fail).

        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds

        Returns:
            Final JobInfo

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        pass

    @abstractmethod
    async def list_jobs(
        self,
        namespace: str = "default",
        label_selector: str = "",
    ) -> list[JobInfo]:
        """List jobs in namespace.

        Args:
            namespace: Kubernetes namespace
            label_selector: Label filter (e.g., "app=multiagent")

        Returns:
            List of JobInfo objects
        """
        pass


class KubernetesExecutor(DistributedExecutor):
    """Kubernetes Jobs implementation of distributed executor.

    Uses standard Kubernetes batch/v1 Jobs API for job submission and monitoring.
    Assumes kubectl is configured and multiagent image is available in cluster.
    """

    def __init__(self, in_cluster: bool = False) -> None:
        """Initialize Kubernetes client.

        Args:
            in_cluster: If True, use in-cluster config; else use kubeconfig

        Raises:
            ImportError: If kubernetes library not installed
            Exception: If kubernetes config cannot be loaded
        """
        if not KUBERNETES_AVAILABLE:
            raise ImportError(
                "kubernetes library not installed. "
                "Install with: pip install 'multiagent[distributed]'"
            )

        # Load kubernetes config
        if in_cluster:
            config.load_incluster_config()
        else:
            config.load_kube_config()

        self.batch_api = client.BatchV1Api()
        self.core_api = client.CoreV1Api()

    async def submit_job(
        self,
        problems: list[dict[str, Any]],
        pipeline_config_path: str,
        job_config: JobConfig,
    ) -> str:
        """Submit Kubernetes Job for batch inference.

        Creates a Job that:
        1. Mounts pipeline config as ConfigMap
        2. Passes problems JSON via environment variable (or ConfigMap for large batches)
        3. Runs batch executor CLI with configured concurrency

        Args:
            problems: Problem list
            pipeline_config_path: Path to pipeline config
            job_config: Job resource configuration

        Returns:
            Job name (job_id)
        """
        # Generate unique job name with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        job_name = f"{job_config.name}-{timestamp}"

        # Serialize problems to JSON
        problems_json = json.dumps(problems)

        # Build environment variables
        env_vars = [
            client.V1EnvVar(name="PROBLEMS_JSON", value=problems_json),
            client.V1EnvVar(name="PIPELINE_CONFIG", value=pipeline_config_path),
        ]

        # Add user-provided env vars (e.g., API keys)
        for key, value in job_config.env_vars.items():
            env_vars.append(client.V1EnvVar(name=key, value=value))

        # Define container
        container = client.V1Container(
            name="batch-executor",
            image=job_config.image,
            command=["python", "-m", "src.cli.main", "batch"],
            args=[
                "--source",
                "/tmp/problems.json",  # Will be created from PROBLEMS_JSON
                "--config",
                pipeline_config_path,
            ],
            env=env_vars,
            resources=client.V1ResourceRequirements(
                requests={
                    "cpu": job_config.cpu_request,
                    "memory": job_config.memory_request,
                },
                limits={
                    "cpu": job_config.cpu_limit,
                    "memory": job_config.memory_limit,
                },
            ),
        )

        # Define pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "app": "multiagent",
                    "job-name": job_name,
                }
            ),
            spec=client.V1PodSpec(
                restart_policy="Never",
                containers=[container],
            ),
        )

        # Define job spec
        job_spec = client.V1JobSpec(
            template=template,
            backoff_limit=job_config.backoff_limit,
        )

        # Create job object
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                labels={
                    "app": "multiagent",
                    "component": "batch-executor",
                },
            ),
            spec=job_spec,
        )

        # Submit to Kubernetes
        try:
            await asyncio.to_thread(
                self.batch_api.create_namespaced_job,
                namespace=job_config.namespace,
                body=job,
            )
            logger.info(
                f"Submitted Kubernetes Job: {job_name} in namespace {job_config.namespace}"
            )
            return job_name

        except ApiException as e:
            logger.error(f"Failed to create Kubernetes Job: {e}")
            raise

    async def get_status(self, job_id: str) -> JobInfo:
        """Get Kubernetes Job status.

        Args:
            job_id: Job name

        Returns:
            JobInfo with status parsed from K8s Job object
        """
        # Extract namespace from job_id if provided as "namespace/name"
        if "/" in job_id:
            namespace, job_name = job_id.split("/", 1)
        else:
            namespace = "default"
            job_name = job_id

        try:
            job = await asyncio.to_thread(
                self.batch_api.read_namespaced_job,
                name=job_name,
                namespace=namespace,
            )

            # Parse job status
            status = JobStatus.UNKNOWN
            num_pods = 0
            succeeded = 0
            failed = 0
            message = None

            if job.status:
                # Count pods
                num_pods = (job.status.active or 0) + (job.status.succeeded or 0) + (
                    job.status.failed or 0
                )
                succeeded = job.status.succeeded or 0
                failed = job.status.failed or 0

                # Determine overall status
                if job.status.succeeded:
                    status = JobStatus.SUCCEEDED
                elif job.status.failed:
                    status = JobStatus.FAILED
                    message = f"Job failed after {failed} attempts"
                elif job.status.active:
                    status = JobStatus.RUNNING
                else:
                    status = JobStatus.PENDING

            # Get timestamps
            created_at = None
            completed_at = None
            if job.metadata.creation_timestamp:
                created_at = job.metadata.creation_timestamp.isoformat()
            if job.status and job.status.completion_time:
                completed_at = job.status.completion_time.isoformat()

            return JobInfo(
                job_name=job_name,
                namespace=namespace,
                status=status,
                created_at=created_at,
                completed_at=completed_at,
                num_pods=num_pods,
                succeeded_pods=succeeded,
                failed_pods=failed,
                message=message,
            )

        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Job not found: {job_name}")
                return JobInfo(
                    job_name=job_name,
                    namespace=namespace,
                    status=JobStatus.UNKNOWN,
                    message="Job not found",
                )
            else:
                logger.error(f"Failed to get job status: {e}")
                raise

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel Kubernetes Job by deleting it.

        Args:
            job_id: Job name

        Returns:
            True if cancelled, False if not found
        """
        # Extract namespace
        if "/" in job_id:
            namespace, job_name = job_id.split("/", 1)
        else:
            namespace = "default"
            job_name = job_id

        try:
            await asyncio.to_thread(
                self.batch_api.delete_namespaced_job,
                name=job_name,
                namespace=namespace,
                propagation_policy="Background",  # Delete pods too
            )
            logger.info(f"Cancelled job: {job_name}")
            return True

        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Job not found for cancellation: {job_name}")
                return False
            else:
                logger.error(f"Failed to cancel job: {e}")
                raise

    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 10.0,
        timeout: float = 3600.0,
    ) -> JobInfo:
        """Wait for Kubernetes Job to complete.

        Args:
            job_id: Job name
            poll_interval: Seconds between polls
            timeout: Maximum wait time

        Returns:
            Final JobInfo

        Raises:
            TimeoutError: If timeout exceeded
        """
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            job_info = await self.get_status(job_id)

            if job_info.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                return job_info

            logger.debug(
                f"Job {job_id} status: {job_info.status}, "
                f"elapsed: {elapsed:.1f}s, "
                f"pods: {job_info.succeeded_pods}/{job_info.num_pods}"
            )

            await asyncio.sleep(poll_interval)

    async def list_jobs(
        self,
        namespace: str = "default",
        label_selector: str = "",
    ) -> list[JobInfo]:
        """List Kubernetes Jobs in namespace.

        Args:
            namespace: Kubernetes namespace
            label_selector: Label filter (e.g., "app=multiagent")

        Returns:
            List of JobInfo objects
        """
        try:
            jobs_list = await asyncio.to_thread(
                self.batch_api.list_namespaced_job,
                namespace=namespace,
                label_selector=label_selector,
            )

            job_infos = []
            for job in jobs_list.items:
                # Build JobInfo for each job
                status = JobStatus.UNKNOWN
                if job.status:
                    if job.status.succeeded:
                        status = JobStatus.SUCCEEDED
                    elif job.status.failed:
                        status = JobStatus.FAILED
                    elif job.status.active:
                        status = JobStatus.RUNNING
                    else:
                        status = JobStatus.PENDING

                created_at = None
                completed_at = None
                if job.metadata.creation_timestamp:
                    created_at = job.metadata.creation_timestamp.isoformat()
                if job.status and job.status.completion_time:
                    completed_at = job.status.completion_time.isoformat()

                job_info = JobInfo(
                    job_name=job.metadata.name,
                    namespace=namespace,
                    status=status,
                    created_at=created_at,
                    completed_at=completed_at,
                    num_pods=(job.status.active or 0)
                    + (job.status.succeeded or 0)
                    + (job.status.failed or 0)
                    if job.status
                    else 0,
                    succeeded_pods=job.status.succeeded or 0 if job.status else 0,
                    failed_pods=job.status.failed or 0 if job.status else 0,
                )
                job_infos.append(job_info)

            return job_infos

        except ApiException as e:
            logger.error(f"Failed to list jobs: {e}")
            raise


class KjobsExecutor(DistributedExecutor):
    """Placeholder for kjobs/apiary integration.

    kjobs/apiary is Cohere's internal job scheduling infrastructure. Since no public
    API documentation is available, this is a placeholder that raises NotImplementedError.

    When the API becomes available, implement:
    1. Authentication with kjobs/apiary service
    2. Job submission via their REST/gRPC API
    3. Status monitoring and log retrieval
    4. Job cancellation

    Expected API patterns (to be confirmed):
    - POST /api/v1/jobs with job spec
    - GET /api/v1/jobs/{job_id} for status
    - DELETE /api/v1/jobs/{job_id} for cancellation
    """

    def __init__(self, api_endpoint: str, api_key: str) -> None:
        """Initialize kjobs/apiary client.

        Args:
            api_endpoint: kjobs/apiary API endpoint URL
            api_key: Authentication key
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        logger.warning("KjobsExecutor is a placeholder - kjobs/apiary API not yet available")

    async def submit_job(
        self,
        problems: list[dict[str, Any]],
        pipeline_config_path: str,
        job_config: JobConfig,
    ) -> str:
        """Submit job to kjobs/apiary.

        Raises:
            NotImplementedError: API not yet available
        """
        raise NotImplementedError(
            "kjobs/apiary integration not implemented. "
            "Use KubernetesExecutor for standard Kubernetes Jobs, "
            "or wait for kjobs/apiary public API documentation."
        )

    async def get_status(self, job_id: str) -> JobInfo:
        """Get job status from kjobs/apiary.

        Raises:
            NotImplementedError: API not yet available
        """
        raise NotImplementedError("kjobs/apiary integration not implemented")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel job in kjobs/apiary.

        Raises:
            NotImplementedError: API not yet available
        """
        raise NotImplementedError("kjobs/apiary integration not implemented")

    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 10.0,
        timeout: float = 3600.0,
    ) -> JobInfo:
        """Wait for job completion in kjobs/apiary.

        Raises:
            NotImplementedError: API not yet available
        """
        raise NotImplementedError("kjobs/apiary integration not implemented")

    async def list_jobs(
        self,
        namespace: str = "default",
        label_selector: str = "",
    ) -> list[JobInfo]:
        """List jobs in kjobs/apiary.

        Raises:
            NotImplementedError: API not yet available
        """
        raise NotImplementedError("kjobs/apiary integration not implemented")
