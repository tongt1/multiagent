"""Pydantic models for CooperBench cooperative coding evaluation.

Defines all data models used across the CooperBench integration:
configuration, problem representation, agent responses, evaluation
results, and pipeline outputs.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CooperBenchConfig(BaseModel):
    """Configuration for CooperBench evaluation runs.

    Controls dataset selection, agent settings, evaluation backend,
    and cooperation parameters.
    """

    model_config = {"extra": "ignore"}

    # Dataset settings
    dataset_path: str = Field(
        default="dataset/",
        description="Path to CooperBench dataset directory",
    )
    subset: Optional[Literal["lite", "flash"]] = Field(
        default="lite",
        description="Dataset subset to use (lite=100 pairs, flash=larger)",
    )
    repo_filter: Optional[list[str]] = Field(
        default=None,
        description="Filter to specific repository names",
    )
    task_filter: Optional[list[str]] = Field(
        default=None,
        description="Filter to specific task IDs",
    )

    # Agent settings
    solver_model: str = Field(
        default="command-r-plus",
        description="LLM model for solver agent",
    )
    verifier_model: str = Field(
        default="command-r-plus",
        description="LLM model for verifier agent",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for agents",
    )
    max_tokens: int = Field(
        default=8192,
        ge=256,
        le=32768,
        description="Maximum tokens per agent response",
    )

    # Evaluation settings
    backend: Literal["docker", "modal", "gcp"] = Field(
        default="docker",
        description="Sandbox backend for test execution",
    )
    timeout: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Timeout in seconds for test execution",
    )

    # Cooperation settings
    mode: Literal["coop", "solo"] = Field(
        default="coop",
        description="Evaluation mode: coop (2 agents) or solo (1 agent)",
    )
    max_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum cooperation rounds (debate iterations)",
    )
    messaging_enabled: bool = Field(
        default=True,
        description="Enable inter-agent messaging",
    )
    git_enabled: bool = Field(
        default=False,
        description="Enable shared git server for collaboration",
    )

    # Parallelism
    max_parallel_tasks: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum parallel task evaluations",
    )


class FeatureSpec(BaseModel):
    """Specification for a single feature within a CooperBench task."""

    feature_id: int = Field(description="Feature number within the task")
    description: str = Field(description="Natural language feature specification (from feature.md)")
    patch_path: Optional[str] = Field(
        default=None,
        description="Path to gold implementation patch",
    )
    tests_patch_path: Optional[str] = Field(
        default=None,
        description="Path to test cases patch",
    )


class CooperBenchProblem(BaseModel):
    """A single CooperBench cooperative coding problem.

    Represents a task from the CooperBench dataset: a codebase with
    two or more features to implement, each with its own spec and tests.
    """

    repo: str = Field(description="Repository name (e.g., 'llama_index')")
    task_id: str = Field(description="Task identifier (e.g., 'task17244')")
    features: list[int] = Field(description="Feature IDs to implement in this problem")
    feature_specs: dict[int, FeatureSpec] = Field(
        description="Feature specifications keyed by feature ID",
    )
    dataset_path: str = Field(description="Root path to CooperBench dataset")
    setup_script: Optional[str] = Field(
        default=None,
        description="Path to setup.sh for this task",
    )
    run_tests_script: Optional[str] = Field(
        default=None,
        description="Path to run_tests.sh for this task",
    )
    runner_script: Optional[str] = Field(
        default=None,
        description="Path to runner.sh for standardized test execution",
    )
    dockerfile: Optional[str] = Field(
        default=None,
        description="Path to Dockerfile for this task",
    )
    combined_patch: Optional[str] = Field(
        default=None,
        description="Path to combined gold patch",
    )
    image_name: Optional[str] = Field(
        default=None,
        description="Docker image name for this task's sandbox",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata",
    )

    @property
    def task_dir(self) -> str:
        """Get the task directory path."""
        return f"{self.dataset_path}/{self.repo}/{self.task_id}"

    @property
    def feature_pair_key(self) -> str:
        """Get a unique key for this feature pair."""
        feature_str = "_".join(str(f) for f in sorted(self.features))
        return f"{self.repo}/{self.task_id}/features_{feature_str}"


class CooperBenchResponse(BaseModel):
    """Response from a CooperBench solver agent.

    Contains the generated code patch, reasoning, and optional
    messages to the partner agent.
    """

    patch: str = Field(description="Generated git diff patch")
    reasoning: str = Field(description="Agent's reasoning about their approach")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Agent's confidence in their solution",
    )
    messages_to_partner: list[str] = Field(
        default_factory=list,
        description="Messages to send to the partner agent",
    )
    approach_summary: str = Field(
        default="",
        description="Brief summary of approach for partner coordination",
    )


class CooperBenchVerification(BaseModel):
    """Verification result from reviewing a partner's approach.

    Used during cooperation rounds to detect potential conflicts
    between agents' changes.
    """

    has_conflicts: bool = Field(
        description="Whether the partner's approach conflicts with own changes",
    )
    conflict_description: str = Field(
        default="",
        description="Description of detected conflicts",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for resolving conflicts",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Confidence in the conflict assessment",
    )


class FeatureResult(BaseModel):
    """Test result for a single feature."""

    feature_id: int = Field(description="Feature ID")
    passed: bool = Field(description="Whether the feature's tests passed")
    test_output: str = Field(default="", description="Raw test output")
    error: Optional[str] = Field(default=None, description="Error message if tests failed")


class CooperBenchJudgment(BaseModel):
    """Judgment from merge + test evaluation.

    The ground-truth evaluation: whether both features' patches
    merge cleanly and pass their respective test suites.
    """

    both_passed: bool = Field(
        description="Whether both features' tests passed (primary metric)",
    )
    feature_results: list[FeatureResult] = Field(
        default_factory=list,
        description="Per-feature test results",
    )
    merge_status: Literal["clean", "conflicts", "failed"] = Field(
        default="clean",
        description="Status of patch merge operation",
    )
    merge_strategy: Literal["naive", "union", "manual"] = Field(
        default="naive",
        description="Merge strategy used",
    )
    test_output: str = Field(
        default="",
        description="Combined test output",
    )


class CooperBenchEvalResult(BaseModel):
    """Full evaluation result for a CooperBench task.

    Mirrors the CooperBench eval.json structure with all metrics
    needed for analysis and reward computation.
    """

    repo: str = Field(description="Repository name")
    task_id: str = Field(description="Task identifier")
    features: list[int] = Field(description="Feature IDs evaluated")
    mode: Literal["coop", "solo"] = Field(description="Evaluation mode")

    # Results
    both_passed: bool = Field(description="Primary metric: both features passed")
    feature_results: list[FeatureResult] = Field(
        default_factory=list,
        description="Per-feature test results",
    )
    merge_status: str = Field(default="clean", description="Merge status")
    merge_strategy: str = Field(default="naive", description="Merge strategy used")

    # Patches
    patches: list[str] = Field(
        default_factory=list,
        description="Applied patches (1 for solo, 2 for coop)",
    )
    merged_patch: Optional[str] = Field(
        default=None,
        description="Merged patch (coop mode only)",
    )

    # Execution details
    test_output: str = Field(default="", description="Raw test output")
    execution_time: float = Field(default=0.0, description="Evaluation wall time in seconds")
    error: Optional[str] = Field(default=None, description="Error if evaluation failed")

    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Evaluation timestamp",
    )


class CooperBenchPipelineResult(BaseModel):
    """Result of a complete CooperBench pipeline run.

    Includes the evaluation result plus trajectory information,
    cost tracking, and cooperation metrics.
    """

    problem: CooperBenchProblem = Field(description="The problem that was solved")
    eval_result: Optional[CooperBenchEvalResult] = Field(
        default=None,
        description="Evaluation result (None if evaluation was skipped)",
    )
    reward: float = Field(default=0.0, description="Computed reward")

    # Agent outputs
    patches: list[str] = Field(
        default_factory=list,
        description="Patches generated by agents",
    )
    approach_summaries: list[str] = Field(
        default_factory=list,
        description="Approach summaries exchanged between agents",
    )

    # Cooperation metrics
    rounds_completed: int = Field(default=0, description="Number of cooperation rounds completed")
    messages_exchanged: int = Field(default=0, description="Total messages between agents")

    # Cost tracking
    total_cost: float = Field(default=0.0, description="Total LLM API cost in USD")
    total_tokens: int = Field(default=0, description="Total tokens used")
    solver_tokens: int = Field(default=0, description="Tokens used by solver agent")
    verifier_tokens: int = Field(default=0, description="Tokens used by verifier agent")

    # Trajectory
    trajectory: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full execution trajectory",
    )

    # Timing
    wall_time: float = Field(default=0.0, description="Wall clock time in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Run timestamp",
    )
    error: Optional[str] = Field(default=None, description="Error if pipeline failed")
