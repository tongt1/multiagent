"""Trajectory logger for recording agent actions."""

from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from src.models.trajectory import TrajectoryEntry


class TrajectoryLogger:
    """JSONL logger for trajectory entries with context manager support."""

    def __init__(self, output_path: Path, run_id: str, config_hash: str) -> None:
        """Initialize trajectory logger.

        Args:
            output_path: Path to JSONL output file
            run_id: Unique identifier for this run
            config_hash: Hash of pipeline configuration
        """
        self.output_path = output_path
        self.run_id = run_id
        self.config_hash = config_hash
        self.step_counter = 0
        self.file_handle: TextIO | None = None

    def __enter__(self) -> "TrajectoryLogger":
        """Enter context manager, create dirs and open file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = self.output_path.open("a")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, close file handle."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    async def __aenter__(self) -> "TrajectoryLogger":
        """Enter async context manager."""
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        self.__exit__(exc_type, exc_val, exc_tb)

    def log_step(
        self,
        agent: str,
        action: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a trajectory step.

        Args:
            agent: Agent identifier (solver/verifier/judge)
            action: Action name
            input_data: Input data dict
            output_data: Output data dict
            metadata: Optional metadata dict
        """
        if not self.file_handle:
            raise RuntimeError("TrajectoryLogger not opened (use context manager)")

        self.step_counter += 1

        entry = TrajectoryEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            run_id=self.run_id,
            step_id=self.step_counter,
            agent=agent,
            action=action,
            input=input_data,
            output=output_data,
            metadata=metadata or {},
        )

        # Write JSONL entry and flush immediately
        self.file_handle.write(entry.model_dump_json() + "\n")
        self.file_handle.flush()

    def log_error(
        self,
        agent: str,
        action: str,
        error: Exception,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an error as a trajectory entry.

        Args:
            agent: Agent identifier
            action: Action that failed
            error: Exception that occurred
            metadata: Optional metadata dict
        """
        error_output = {
            "error": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        self.log_step(
            agent=agent,
            action=action,
            input_data={},
            output_data=error_output,
            metadata=metadata,
        )
