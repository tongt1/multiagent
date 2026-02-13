"""Experiment state tracking with JSON persistence.

Tracks per-stage completion status and supports resume from last incomplete stage.

Usage:
    from pathlib import Path
    from src.orchestration.experiment_state import ExperimentState, StageStatus

    # Create new experiment state
    state = ExperimentState.create_new("exp_001", "debate_cmdR")

    # Mark stages as they complete
    state.mark_stage("data_generation", StageStatus.COMPLETE)
    state.record_output("data_generation", "train_path", "/tmp/train.jsonl")

    # Save to disk
    state.save(Path("exp_001_state.json"))

    # Load and resume
    loaded = ExperimentState.load(Path("exp_001_state.json"))
    resume_from = loaded.get_resume_point()  # "data_conversion"
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class StageStatus(Enum):
    """Stage execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


# Canonical stage order for the experiment pipeline
PIPELINE_STAGES = [
    "data_generation",
    "data_conversion",
    "training",
    "evaluation",
    "comparison",
]


@dataclass
class ExperimentState:
    """Experiment state tracking with stage-level granularity.

    Attributes:
        experiment_id: Unique experiment identifier
        experiment_name: Human-readable experiment name
        stages: Stage name to status mapping (stored as strings for JSON)
        retry_count: Stage name to retry count mapping
        outputs: Stage name to output artifacts mapping
        error_log: List of error records with stage, error, timestamp
        created_at: ISO timestamp of creation
        updated_at: ISO timestamp of last update
    """

    experiment_id: str
    experiment_name: str
    stages: dict[str, str] = field(default_factory=dict)
    retry_count: dict[str, int] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error_log: list[dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def create_new(cls, experiment_id: str, experiment_name: str) -> ExperimentState:
        """Create new experiment state with all stages initialized to PENDING.

        Args:
            experiment_id: Unique experiment identifier
            experiment_name: Human-readable experiment name

        Returns:
            New ExperimentState instance
        """
        now = datetime.utcnow().isoformat() + "Z"
        stages = {stage: StageStatus.PENDING.value for stage in PIPELINE_STAGES}
        retry_count = {stage: 0 for stage in PIPELINE_STAGES}

        return cls(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            stages=stages,
            retry_count=retry_count,
            created_at=now,
            updated_at=now,
        )

    def mark_stage(self, stage: str, status: StageStatus) -> None:
        """Update stage status and timestamp.

        Args:
            stage: Stage name
            status: New stage status
        """
        self.stages[stage] = status.value
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def record_output(self, stage: str, key: str, value: Any) -> None:
        """Record stage output artifact.

        Args:
            stage: Stage name
            key: Output key
            value: Output value (must be JSON-serializable)
        """
        if stage not in self.outputs:
            self.outputs[stage] = {}
        self.outputs[stage][key] = value
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def record_error(self, stage: str, error: str) -> None:
        """Record stage error and increment retry count.

        Args:
            stage: Stage name
            error: Error message
        """
        self.error_log.append({
            "stage": stage,
            "error": error,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
        if stage not in self.retry_count:
            self.retry_count[stage] = 0
        self.retry_count[stage] += 1
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def should_run_stage(self, stage: str, max_retries: int = 3) -> bool:
        """Check if stage should run (not complete and under retry limit).

        Args:
            stage: Stage name
            max_retries: Maximum retry count

        Returns:
            True if stage should run
        """
        if self.stages.get(stage) == StageStatus.COMPLETE.value:
            return False
        return self.retry_count.get(stage, 0) < max_retries

    def get_resume_point(self) -> str | None:
        """Get first incomplete stage in pipeline order.

        Returns:
            Stage name to resume from, or None if all complete
        """
        for stage in PIPELINE_STAGES:
            status = self.stages.get(stage, StageStatus.PENDING.value)
            if status not in [StageStatus.COMPLETE.value, StageStatus.SKIPPED.value]:
                return stage
        return None

    def is_complete(self) -> bool:
        """Check if all pipeline stages are complete or skipped.

        Returns:
            True if all stages are complete or skipped
        """
        for stage in PIPELINE_STAGES:
            status = self.stages.get(stage, StageStatus.PENDING.value)
            if status not in [StageStatus.COMPLETE.value, StageStatus.SKIPPED.value]:
                return False
        return True

    def save(self, state_file: Path) -> None:
        """Persist state to JSON file.

        Args:
            state_file: Path to state file
        """
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, state_file: Path) -> ExperimentState:
        """Load state from JSON file.

        Args:
            state_file: Path to state file

        Returns:
            ExperimentState instance

        Raises:
            FileNotFoundError: If state file doesn't exist
        """
        with open(state_file, "r") as f:
            data = json.load(f)
        return cls(**data)
