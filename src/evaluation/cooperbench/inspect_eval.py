"""Inspect AI evaluation module for CooperBench agentic RL training.

Provides an Inspect-based evaluation harness that uses the same agentic
OpenHands loop as training, enabling apples-to-apples comparison between
training reward and held-out evaluation metrics.

Key components:
- ``cooperbench_eval``: Inspect Task factory
- ``cooperbench_agentic_solver``: Solver running OpenHands agent loop per sample
- ``cooperbench_scorer``: Scorer computing pass@k and partial credit
- ``compute_passk``: Unbiased pass@k estimator

The evaluation mirrors the ``comb_cooperbench_env.py`` training loop:
Docker container with -oh image -> agent-server -> conversation loop
-> in-container test evaluation -> reward + metrics.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pass@k estimator
# ---------------------------------------------------------------------------

def compute_passk(scores: list[float], k: int) -> float:
    """Unbiased pass@k estimator.

    Implements the standard estimator: ``1 - C(n-c, k) / C(n, k)``
    where *n* = total samples, *c* = number passing, *k* = k value.

    Uses the log-space formulation to avoid overflow with large
    combinatorial values:
        ``1 - exp(sum_{i=0}^{k-1} log(n - c - i) - log(n - i))``

    Args:
        scores: List of per-rollout scores (> 0 means pass).
        k: The k in pass@k.

    Returns:
        Unbiased pass@k estimate in [0, 1].
    """
    n = len(scores)
    c = sum(1 for s in scores if s > 0)

    if n == 0:
        return 0.0
    if k > n:
        # When k > n, fall back to empirical rate
        return c / n if n > 0 else 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    if n - c < k:
        # Not enough failures to fill k slots -> guaranteed at least one pass
        return 1.0

    # Log-space computation: 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)
    log_prod = 0.0
    for i in range(k):
        log_prod += math.log(n - c - i) - math.log(n - i)

    return 1.0 - math.exp(log_prod)


# ---------------------------------------------------------------------------
# Data structures for evaluation results
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """A single evaluation sample (one rollout of one task)."""

    task_id: str
    rollout_idx: int
    input_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Filled after evaluation
    docker_reward: float = 0.0
    docker_tests_passed: int = 0
    docker_tests_total: int = 0
    agentic_steps: int = 0
    rollout_time_seconds: float = 0.0
    token_count: int = 0
    error: Optional[str] = None


@dataclass
class EvalTaskResult:
    """Aggregated result for a single task across all rollouts."""

    task_id: str
    samples: list[EvalSample] = field(default_factory=list)

    @property
    def pass_at_1(self) -> float:
        scores = [s.docker_reward for s in self.samples]
        return compute_passk(scores, k=1)

    @property
    def pass_at_3(self) -> float:
        scores = [s.docker_reward for s in self.samples]
        return compute_passk(scores, k=3)

    @property
    def pass_at_5(self) -> float:
        scores = [s.docker_reward for s in self.samples]
        return compute_passk(scores, k=5)

    @property
    def partial_credit(self) -> float:
        """Mean docker test pass rate across rollouts."""
        if not self.samples:
            return 0.0
        rates = []
        for s in self.samples:
            if s.docker_tests_total > 0:
                rates.append(s.docker_tests_passed / s.docker_tests_total)
            else:
                rates.append(s.docker_reward)  # fallback to binary
        return sum(rates) / len(rates)

    @property
    def mean_agent_turns(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.agentic_steps for s in self.samples) / len(self.samples)

    @property
    def mean_rollout_time(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.rollout_time_seconds for s in self.samples) / len(self.samples)

    @property
    def mean_token_count(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.token_count for s in self.samples) / len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "n_rollouts": len(self.samples),
            "pass_at_1": self.pass_at_1,
            "pass_at_3": self.pass_at_3,
            "pass_at_5": self.pass_at_5,
            "partial_credit": self.partial_credit,
            "mean_agent_turns": self.mean_agent_turns,
            "mean_rollout_time_s": self.mean_rollout_time,
            "mean_token_count": self.mean_token_count,
        }


# ---------------------------------------------------------------------------
# Agentic solver (OpenHands Docker loop)
# ---------------------------------------------------------------------------

def _run_openhands_agent_standalone(
    task_id: str,
    task_description: str,
    vllm_base_url: str,
    docker_image: str,
    runner_script: str,
    max_iterations: int = 20,
    timeout: int = 900,
) -> dict[str, Any]:
    """Standalone OpenHands agent loop for evaluation.

    Mirrors the training ``run_openhands_agent()`` from
    ``comb_cooperbench_env.py`` but adapted for standalone evaluation
    without the Flink/comb framework.

    The loop:
    1. Create Docker container from the -oh image
    2. Start the agent-server inside the container
    3. Run the OpenHands conversation loop (model <-> agent-server)
    4. Evaluate tests in-container via runner.sh
    5. Return reward + metrics

    This is a *stub* that returns placeholder results when the OpenHands
    infrastructure is not available. Real execution requires the -oh Docker
    images and a running vLLM sidecar.

    Args:
        task_id: CooperBench task identifier.
        task_description: Natural language task specification.
        vllm_base_url: URL of the vLLM inference server.
        docker_image: Docker image name for this task.
        runner_script: Path to the test runner script.
        max_iterations: Maximum agent conversation turns.
        timeout: Rollout timeout in seconds.

    Returns:
        Dict with keys: docker_reward, docker_tests_passed,
        docker_tests_total, agentic_steps, rollout_time_seconds,
        token_count, error.
    """
    # NOTE: This is the evaluation-mode stub. In production, this calls
    # the same OpenHands agent loop used in training:
    #   - docker run -d {docker_image}
    #   - POST /api/conversations to start agent
    #   - Loop: GET events, POST actions (model generates via vllm_base_url)
    #   - docker exec runner.sh for test evaluation
    #
    # For now, return a placeholder so the evaluation script can be tested
    # end-to-end without infrastructure.
    logger.warning(
        f"OpenHands agent stub called for {task_id}. "
        f"Real execution requires -oh Docker images and vLLM sidecar at {vllm_base_url}"
    )

    return {
        "docker_reward": 0.0,
        "docker_tests_passed": 0,
        "docker_tests_total": 0,
        "agentic_steps": 0,
        "rollout_time_seconds": 0.0,
        "token_count": 0,
        "error": "stub: OpenHands infrastructure not available",
    }


# ---------------------------------------------------------------------------
# Inspect-compatible task, solver, scorer interfaces
# ---------------------------------------------------------------------------

def load_eval_samples(
    task_ids: list[str],
    n_rollouts: int = 10,
    dataset_path: str | None = None,
) -> list[EvalSample]:
    """Load evaluation samples for the given task IDs.

    Creates ``n_rollouts`` samples per task. If ``dataset_path`` is provided,
    loads task descriptions from the CooperBench dataset on disk. Otherwise,
    uses the task_id as a placeholder input.

    Args:
        task_ids: List of task IDs to evaluate.
        n_rollouts: Number of independent rollouts per task.
        dataset_path: Optional path to CooperBench dataset for task descriptions.

    Returns:
        List of EvalSample objects ready for evaluation.
    """
    samples: list[EvalSample] = []

    for task_id in task_ids:
        # Try to load task description from dataset
        input_text = f"CooperBench task: {task_id}"
        metadata: dict[str, Any] = {"task_id": task_id}

        if dataset_path:
            task_dir = Path(dataset_path) / task_id.replace("/", "/")
            # Look for feature descriptions
            if task_dir.exists():
                feature_descs = []
                for feature_dir in sorted(task_dir.iterdir()):
                    if feature_dir.is_dir() and feature_dir.name.startswith("feature"):
                        feature_md = feature_dir / "feature.md"
                        if feature_md.exists():
                            feature_descs.append(feature_md.read_text(encoding="utf-8"))
                if feature_descs:
                    input_text = "\n\n---\n\n".join(feature_descs)

                # Gather Docker/runner metadata
                runner_sh = task_dir / "runner.sh"
                dockerfile = task_dir / "Dockerfile"
                repo_name = task_id.split("/")[0]
                task_name = task_id.split("/")[1]
                metadata.update({
                    "docker_image": f"cooperbench-{repo_name}-{task_name}".lower(),
                    "runner_script": str(runner_sh) if runner_sh.exists() else None,
                    "dockerfile": str(dockerfile) if dockerfile.exists() else None,
                    "dataset_path": dataset_path,
                })

        for rollout_idx in range(n_rollouts):
            samples.append(
                EvalSample(
                    task_id=task_id,
                    rollout_idx=rollout_idx,
                    input_text=input_text,
                    metadata={**metadata, "rollout_idx": rollout_idx},
                )
            )

    return samples


def run_eval_sample(
    sample: EvalSample,
    vllm_base_url: str = "http://localhost:8000/v1",
    max_iterations: int = 20,
    timeout: int = 900,
) -> EvalSample:
    """Run the agentic evaluation loop on a single sample.

    Calls the OpenHands agent standalone function and populates the
    sample's result fields.

    Args:
        sample: The evaluation sample to run.
        vllm_base_url: vLLM inference server URL.
        max_iterations: Max agent turns.
        timeout: Rollout timeout in seconds.

    Returns:
        The same sample with result fields populated.
    """
    start_time = time.time()

    docker_image = sample.metadata.get("docker_image", "")
    runner_script = sample.metadata.get("runner_script", "")

    result = _run_openhands_agent_standalone(
        task_id=sample.task_id,
        task_description=sample.input_text,
        vllm_base_url=vllm_base_url,
        docker_image=docker_image,
        runner_script=runner_script,
        max_iterations=max_iterations,
        timeout=timeout,
    )

    sample.docker_reward = result.get("docker_reward", 0.0)
    sample.docker_tests_passed = result.get("docker_tests_passed", 0)
    sample.docker_tests_total = result.get("docker_tests_total", 0)
    sample.agentic_steps = result.get("agentic_steps", 0)
    sample.rollout_time_seconds = time.time() - start_time
    sample.token_count = result.get("token_count", 0)
    sample.error = result.get("error")

    return sample


def cooperbench_eval(
    model_path: str,
    task_ids: list[str],
    n_rollouts: int = 10,
    vllm_base_url: str = "http://localhost:8000/v1",
    dataset_path: str | None = None,
    max_iterations: int = 20,
    timeout: int = 900,
) -> dict[str, Any]:
    """Run CooperBench evaluation and return structured results.

    This is the main evaluation entry point. It creates samples for each
    held-out task, runs the agentic OpenHands loop for each sample, and
    aggregates results including pass@k metrics.

    Args:
        model_path: Path to the model checkpoint (used for metadata/logging).
        task_ids: Task IDs to evaluate (typically the held-out eval set).
        n_rollouts: Independent rollouts per task (default 10).
        vllm_base_url: vLLM sidecar URL for model inference.
        dataset_path: Optional CooperBench dataset path for task descriptions.
        max_iterations: Max agent turns per rollout.
        timeout: Per-rollout timeout in seconds.

    Returns:
        Dict with overall metrics, per-task breakdown, and metadata.
    """
    logger.info(
        f"Starting CooperBench eval: model={model_path}, "
        f"tasks={len(task_ids)}, rollouts={n_rollouts}"
    )

    # Load samples
    samples = load_eval_samples(task_ids, n_rollouts, dataset_path)

    # Run evaluation
    for sample in samples:
        run_eval_sample(sample, vllm_base_url, max_iterations, timeout)

    # Group by task
    task_results: dict[str, EvalTaskResult] = {}
    for sample in samples:
        if sample.task_id not in task_results:
            task_results[sample.task_id] = EvalTaskResult(task_id=sample.task_id)
        task_results[sample.task_id].samples.append(sample)

    # Compute overall metrics
    all_scores = [s.docker_reward for s in samples]
    overall_pass1 = compute_passk(all_scores, k=1)
    overall_pass3 = compute_passk(all_scores, k=3)
    overall_pass5 = compute_passk(all_scores, k=5)

    # Per-task pass@1 mean (the "macro" metric)
    task_pass1_values = [tr.pass_at_1 for tr in task_results.values()]
    macro_pass1 = sum(task_pass1_values) / len(task_pass1_values) if task_pass1_values else 0.0

    all_partial = [
        s.docker_tests_passed / s.docker_tests_total
        if s.docker_tests_total > 0 else s.docker_reward
        for s in samples
    ]
    mean_partial = sum(all_partial) / len(all_partial) if all_partial else 0.0

    mean_turns = sum(s.agentic_steps for s in samples) / len(samples) if samples else 0.0
    mean_time = sum(s.rollout_time_seconds for s in samples) / len(samples) if samples else 0.0
    mean_tokens = sum(s.token_count for s in samples) / len(samples) if samples else 0.0

    return {
        "model_path": model_path,
        "n_tasks": len(task_ids),
        "n_rollouts": n_rollouts,
        "overall": {
            "pass_at_1": overall_pass1,
            "pass_at_3": overall_pass3,
            "pass_at_5": overall_pass5,
            "macro_pass_at_1": macro_pass1,
            "partial_credit": mean_partial,
            "mean_agent_turns": mean_turns,
            "mean_rollout_time_s": mean_time,
            "mean_token_count": mean_tokens,
        },
        "per_task": [tr.to_dict() for tr in task_results.values()],
    }
