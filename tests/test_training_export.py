"""Tests for training export functionality."""

import json
import tempfile
from pathlib import Path

from src.data.training_export import (
    TrainingTrajectory,
    TrainingTurn,
    convert_to_training_trajectory,
    export_batch_trajectories,
    export_trajectory_for_training,
    group_by_run_id,
    load_trajectory_entries,
)


def create_sample_trajectory_file(path: Path, num_runs: int = 2, steps_per_run: int = 3):
    """Create a sample trajectory JSONL file for testing."""
    with open(path, "w", encoding="utf-8") as f:
        for run_idx in range(num_runs):
            run_id = f"run-{run_idx}"
            for step_idx in range(steps_per_run):
                entry = {
                    "timestamp": f"2024-01-01T00:00:{step_idx:02d}",
                    "run_id": run_id,
                    "step_id": step_idx,
                    "agent": ["solver", "verifier", "judge"][step_idx % 3],
                    "action": ["solve", "verify", "judge"][step_idx % 3],
                    "input": {"problem": f"Problem {run_idx}"},
                    "output": {"solution": f"Solution {run_idx}-{step_idx}"},
                    "metadata": {"domain": "math"},
                    "reward": 0.5 if step_idx > 0 else 0.0,
                    "success": True if step_idx == steps_per_run - 1 else None,
                }
                f.write(json.dumps(entry) + "\n")


class TestLoadTrajectoryEntries:
    """Tests for loading trajectory entries."""

    def test_load_entries(self, tmp_path):
        """Test loading entries from JSONL file."""
        traj_file = tmp_path / "trajectory.jsonl"
        create_sample_trajectory_file(traj_file, num_runs=1, steps_per_run=2)

        entries = load_trajectory_entries(traj_file)

        assert len(entries) == 2
        assert entries[0]["run_id"] == "run-0"
        assert entries[0]["step_id"] == 0
        assert entries[1]["step_id"] == 1

    def test_load_empty_file(self, tmp_path):
        """Test loading empty file."""
        traj_file = tmp_path / "empty.jsonl"
        traj_file.write_text("")

        entries = load_trajectory_entries(traj_file)

        assert len(entries) == 0

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file."""
        traj_file = tmp_path / "nonexistent.jsonl"

        entries = load_trajectory_entries(traj_file)

        assert len(entries) == 0


class TestGroupByRunId:
    """Tests for grouping by run_id."""

    def test_group_single_run(self):
        """Test grouping entries from single run."""
        entries = [
            {"run_id": "run-1", "step_id": 0},
            {"run_id": "run-1", "step_id": 1},
        ]

        groups = group_by_run_id(entries)

        assert len(groups) == 1
        assert "run-1" in groups
        assert len(groups["run-1"]) == 2

    def test_group_multiple_runs(self):
        """Test grouping entries from multiple runs."""
        entries = [
            {"run_id": "run-1", "step_id": 0},
            {"run_id": "run-2", "step_id": 0},
            {"run_id": "run-1", "step_id": 1},
        ]

        groups = group_by_run_id(entries)

        assert len(groups) == 2
        assert len(groups["run-1"]) == 2
        assert len(groups["run-2"]) == 1

    def test_group_empty_entries(self):
        """Test grouping empty entries list."""
        groups = group_by_run_id([])

        assert len(groups) == 0


class TestConvertToTrainingTrajectory:
    """Tests for converting to training trajectory."""

    def test_convert_basic(self):
        """Test basic conversion."""
        steps = [
            {
                "run_id": "test-1",
                "step_id": 0,
                "agent": "solver",
                "action": "solve",
                "input": {"problem": "Test problem"},
                "output": {"solution": "Test solution"},
                "timestamp": "2024-01-01T00:00:00",
                "reward": 1.0,
            }
        ]

        traj = convert_to_training_trajectory("test-1", steps, "Test problem", "math")

        assert traj.id == "test-1"
        assert traj.problem == "Test problem"
        assert traj.domain == "math"
        assert len(traj.turns) == 1
        assert traj.total_reward == 1.0
        assert traj.turns[0].agent == "solver"

    def test_convert_multiple_steps(self):
        """Test conversion with multiple steps."""
        steps = [
            {
                "step_id": 0,
                "agent": "solver",
                "action": "solve",
                "input": {},
                "output": {},
                "timestamp": "2024-01-01T00:00:00",
                "reward": 0.5,
            },
            {
                "step_id": 1,
                "agent": "verifier",
                "action": "verify",
                "input": {},
                "output": {},
                "timestamp": "2024-01-01T00:00:01",
                "reward": 0.5,
            },
        ]

        traj = convert_to_training_trajectory("test-1", steps)

        assert len(traj.turns) == 2
        assert traj.total_reward == 1.0
        assert traj.turns[0].turn_id == 0
        assert traj.turns[1].turn_id == 1

    def test_convert_with_success(self):
        """Test conversion with success field."""
        steps = [
            {
                "step_id": 0,
                "agent": "solver",
                "action": "solve",
                "input": {},
                "output": {},
                "timestamp": "2024-01-01T00:00:00",
                "reward": 1.0,
                "success": True,
            }
        ]

        traj = convert_to_training_trajectory("test-1", steps)

        assert traj.success is True

    def test_convert_extracts_problem(self):
        """Test that problem is extracted from input if not provided."""
        steps = [
            {
                "step_id": 0,
                "agent": "solver",
                "action": "solve",
                "input": {"problem": "Extracted problem"},
                "output": {},
                "timestamp": "2024-01-01T00:00:00",
                "reward": 0.0,
            }
        ]

        traj = convert_to_training_trajectory("test-1", steps)

        assert traj.problem == "Extracted problem"


class TestExportTrajectoryForTraining:
    """Tests for exporting trajectory to training format."""

    def test_export_basic(self, tmp_path):
        """Test basic export functionality."""
        # Create input file
        input_file = tmp_path / "input.jsonl"
        create_sample_trajectory_file(input_file, num_runs=2, steps_per_run=3)

        # Export
        output_file = tmp_path / "output.jsonl"
        count = export_trajectory_for_training(input_file, output_file)

        assert count == 2
        assert output_file.exists()

        # Read back and validate
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Parse first trajectory
        traj_data = json.loads(lines[0])
        assert "id" in traj_data
        assert "turns" in traj_data
        assert "total_reward" in traj_data
        assert len(traj_data["turns"]) == 3

    def test_export_with_problem_map(self, tmp_path):
        """Test export with problem map enrichment."""
        # Create input file
        input_file = tmp_path / "input.jsonl"
        create_sample_trajectory_file(input_file, num_runs=1, steps_per_run=2)

        # Create problem map
        problem_map = {
            "run-0": {
                "problem": "Enriched problem",
                "domain": "code",
            }
        }

        # Export
        output_file = tmp_path / "output.jsonl"
        count = export_trajectory_for_training(input_file, output_file, problem_map)

        assert count == 1

        # Read back and validate
        with open(output_file, "r", encoding="utf-8") as f:
            traj_data = json.loads(f.readline())

        assert traj_data["problem"] == "Enriched problem"
        assert traj_data["domain"] == "code"

    def test_export_empty_file(self, tmp_path):
        """Test exporting empty file."""
        input_file = tmp_path / "empty.jsonl"
        input_file.write_text("")

        output_file = tmp_path / "output.jsonl"
        count = export_trajectory_for_training(input_file, output_file)

        assert count == 0

    def test_export_creates_directories(self, tmp_path):
        """Test that export creates output directories."""
        input_file = tmp_path / "input.jsonl"
        create_sample_trajectory_file(input_file, num_runs=1, steps_per_run=1)

        output_file = tmp_path / "subdir" / "output.jsonl"
        count = export_trajectory_for_training(input_file, output_file)

        assert count == 1
        assert output_file.exists()


class TestExportBatchTrajectories:
    """Tests for batch trajectory export."""

    def test_export_batch(self, tmp_path):
        """Test exporting multiple trajectory files."""
        # Create input directory with multiple files
        input_dir = tmp_path / "trajectories"
        input_dir.mkdir()

        file1 = input_dir / "traj1.jsonl"
        file2 = input_dir / "traj2.jsonl"
        create_sample_trajectory_file(file1, num_runs=1, steps_per_run=2)
        create_sample_trajectory_file(file2, num_runs=1, steps_per_run=2)

        # Export
        output_file = tmp_path / "output.jsonl"
        count = export_batch_trajectories(input_dir, output_file)

        assert count == 2
        assert output_file.exists()

        # Read back and validate
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_export_batch_empty_dir(self, tmp_path):
        """Test exporting from empty directory."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        output_file = tmp_path / "output.jsonl"
        count = export_batch_trajectories(input_dir, output_file)

        assert count == 0

    def test_export_batch_nonexistent_dir(self, tmp_path):
        """Test exporting from nonexistent directory."""
        input_dir = tmp_path / "nonexistent"

        output_file = tmp_path / "output.jsonl"
        count = export_batch_trajectories(input_dir, output_file)

        assert count == 0


class TestRoundTrip:
    """Tests for round-trip serialization."""

    def test_round_trip(self, tmp_path):
        """Test that exported data can be read back and validated."""
        # Create input
        input_file = tmp_path / "input.jsonl"
        create_sample_trajectory_file(input_file, num_runs=1, steps_per_run=3)

        # Export
        output_file = tmp_path / "output.jsonl"
        export_trajectory_for_training(input_file, output_file)

        # Read back
        with open(output_file, "r", encoding="utf-8") as f:
            traj_data = json.loads(f.readline())

        # Validate structure
        assert "id" in traj_data
        assert "turns" in traj_data
        assert "total_reward" in traj_data
        assert "problem" in traj_data
        assert "domain" in traj_data
        assert "success" in traj_data
        assert "metadata" in traj_data

        # Validate turns structure
        for turn in traj_data["turns"]:
            assert "turn_id" in turn
            assert "agent" in turn
            assert "action" in turn
            assert "input" in turn
            assert "output" in turn
            assert "reward" in turn
            assert "timestamp" in turn
