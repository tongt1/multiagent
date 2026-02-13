"""Tests for trajectory JSONL format validation.

Validates that trajectory entries contain all required fields and conform
to the expected format for both debate and baseline modes.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.infrastructure.trajectory_logger import TrajectoryLogger
from src.models.trajectory import TrajectoryEntry


def test_trajectory_entry_has_required_fields():
    """Test that trajectory entries contain all required fields."""
    # Create a sample trajectory entry
    entry = TrajectoryEntry(
        timestamp="2024-01-01T00:00:00Z",
        run_id="test_run_001",
        step_id=1,
        agent="solver",
        action="generate",
        input={"problem": "What is 2+2?"},
        output={"solution": "4"},
        metadata={"mode": "debate", "model": "test-model"},
    )

    # Convert to dict (mimics JSONL serialization)
    entry_dict = entry.model_dump()

    # Verify all required fields present
    required_fields = [
        "run_id",
        "step_id",
        "agent",
        "action",
        "input",
        "output",
        "metadata",
        "timestamp",
    ]

    for field in required_fields:
        assert field in entry_dict, f"Missing required field: {field}"

    # Verify field types
    assert isinstance(entry_dict["run_id"], str)
    assert isinstance(entry_dict["step_id"], int)
    assert isinstance(entry_dict["agent"], str)
    assert isinstance(entry_dict["action"], str)
    assert isinstance(entry_dict["input"], dict)
    assert isinstance(entry_dict["output"], dict)
    assert isinstance(entry_dict["metadata"], dict)
    assert isinstance(entry_dict["timestamp"], str)


def test_debate_trajectory_has_all_agents():
    """Test that debate trajectory contains solver, verifier, judge, and reward agents."""
    # Create synthetic debate trajectory
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectory_path = Path(tmpdir) / "debate_test.jsonl"

        with TrajectoryLogger(trajectory_path, "debate_run", "test_hash") as traj:
            # Solver step
            traj.log_step(
                agent="solver",
                action="generate",
                input_data={"problem": "Test problem"},
                output_data={"solution": "Test solution"},
                metadata={"mode": "debate", "iteration": 1},
            )

            # Verifier step
            traj.log_step(
                agent="verifier",
                action="verify",
                input_data={"solution": "Test solution"},
                output_data={"feedback": "Looks good"},
                metadata={"mode": "debate", "iteration": 1},
            )

            # Judge step
            traj.log_step(
                agent="judge",
                action="score",
                input_data={"solution": "Test solution"},
                output_data={"score": 0.9},
                metadata={"mode": "debate", "iteration": 1},
            )

            # Reward step (ground truth verification)
            traj.log_step(
                agent="reward",
                action="ground_truth_verify",
                input_data={"solution": "Test solution", "ground_truth": "Expected answer"},
                output_data={"reward": 1.0, "is_correct": True},
                metadata={"mode": "debate"},
            )

        # Read back and verify agents present
        agents_found = set()
        with open(trajectory_path) as f:
            for line in f:
                entry_dict = json.loads(line)
                agents_found.add(entry_dict["agent"])

        expected_agents = {"solver", "verifier", "judge", "reward"}
        assert agents_found == expected_agents, f"Expected {expected_agents}, found {agents_found}"


def test_baseline_trajectory_has_solver_and_reward():
    """Test that baseline trajectory has only solver and reward steps."""
    # Create synthetic baseline trajectory
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectory_path = Path(tmpdir) / "baseline_test.jsonl"

        with TrajectoryLogger(trajectory_path, "baseline_run", "test_hash") as traj:
            # Solver step (single call, no iteration)
            traj.log_step(
                agent="solver",
                action="generate",
                input_data={"problem": "Test problem", "feedback": None},
                output_data={"solution": "Test solution"},
                metadata={"mode": "baseline", "iteration": 1},
            )

            # Reward step (ground truth verification)
            traj.log_step(
                agent="reward",
                action="ground_truth_verify",
                input_data={"solution": "Test solution", "ground_truth": "Expected answer"},
                output_data={"reward": 1.0, "is_correct": True},
                metadata={"mode": "baseline", "termination_reason": "single_iteration"},
            )

        # Read back and verify only solver and reward present
        agents_found = set()
        with open(trajectory_path) as f:
            for line in f:
                entry_dict = json.loads(line)
                agents_found.add(entry_dict["agent"])

        expected_agents = {"solver", "reward"}
        assert agents_found == expected_agents, f"Expected {expected_agents}, found {agents_found}"

        # Verify no verifier or judge
        assert "verifier" not in agents_found
        assert "judge" not in agents_found


def test_reward_is_binary():
    """Test that reward values are binary (1.0, 0.0, or None)."""
    # Create trajectory with various reward values
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectory_path = Path(tmpdir) / "reward_test.jsonl"

        with TrajectoryLogger(trajectory_path, "reward_run", "test_hash") as traj:
            # Correct solution (reward=1.0)
            traj.log_step(
                agent="reward",
                action="ground_truth_verify",
                input_data={"solution": "Correct", "ground_truth": "Correct"},
                output_data={"reward": 1.0, "is_correct": True},
                metadata={"mode": "debate"},
            )

            # Incorrect solution (reward=0.0)
            traj.log_step(
                agent="reward",
                action="ground_truth_verify",
                input_data={"solution": "Wrong", "ground_truth": "Correct"},
                output_data={"reward": 0.0, "is_correct": False},
                metadata={"mode": "debate"},
            )

        # Read back and validate rewards
        with open(trajectory_path) as f:
            for line in f:
                entry_dict = json.loads(line)
                if entry_dict["agent"] == "reward":
                    reward = entry_dict["output"]["reward"]
                    assert reward in [0.0, 1.0, None], f"Reward must be binary, got: {reward}"


def test_trajectory_has_mode_in_metadata():
    """Test that each entry's metadata includes mode key."""
    # Create trajectory with mode metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectory_path = Path(tmpdir) / "mode_test.jsonl"

        with TrajectoryLogger(trajectory_path, "mode_run", "test_hash") as traj:
            # Debate mode entry
            traj.log_step(
                agent="solver",
                action="generate",
                input_data={"problem": "Test"},
                output_data={"solution": "Test"},
                metadata={"mode": "debate", "iteration": 1},
            )

            # Baseline mode entry
            traj.log_step(
                agent="solver",
                action="generate",
                input_data={"problem": "Test"},
                output_data={"solution": "Test"},
                metadata={"mode": "baseline", "iteration": 1},
            )

        # Verify mode present in all entries
        with open(trajectory_path) as f:
            for line in f:
                entry_dict = json.loads(line)
                assert "mode" in entry_dict["metadata"], "Missing 'mode' in metadata"
                assert entry_dict["metadata"]["mode"] in ["debate", "baseline"], \
                    f"Invalid mode: {entry_dict['metadata']['mode']}"


def test_trajectory_ids_sequential():
    """Test that trajectory_ids are sequential integers starting from 0."""
    # Simulate problem metadata with trajectory IDs
    trajectory_ids = [0, 1, 2, 3, 4]

    # Create trajectory entries with trajectory_id in metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectory_path = Path(tmpdir) / "id_test.jsonl"

        with TrajectoryLogger(trajectory_path, "id_run", "test_hash") as traj:
            for traj_id in trajectory_ids:
                traj.log_step(
                    agent="solver",
                    action="generate",
                    input_data={"problem": f"Problem {traj_id}"},
                    output_data={"solution": f"Solution {traj_id}"},
                    metadata={"mode": "debate", "trajectory_id": traj_id},
                )

        # Read back and verify sequential IDs
        found_ids = []
        with open(trajectory_path) as f:
            for line in f:
                entry_dict = json.loads(line)
                if "trajectory_id" in entry_dict["metadata"]:
                    found_ids.append(entry_dict["metadata"]["trajectory_id"])

        # Verify sequential starting from 0
        assert found_ids == list(range(len(found_ids))), \
            f"Expected sequential IDs [0, 1, 2, ...], got {found_ids}"


def test_trajectory_round_trip():
    """Test creating entries with TrajectoryLogger and reading them back."""
    # Create trajectory entries
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectory_path = Path(tmpdir) / "roundtrip_test.jsonl"

        with TrajectoryLogger(trajectory_path, "roundtrip_run", "test_hash") as traj:
            # Log multiple steps
            traj.log_step(
                agent="solver",
                action="generate",
                input_data={"problem": "What is 2+2?"},
                output_data={"solution": "4", "reasoning": "Simple addition"},
                metadata={"mode": "debate", "iteration": 1, "model": "test-model"},
            )

            traj.log_step(
                agent="verifier",
                action="verify",
                input_data={"solution": "4"},
                output_data={"feedback": "Correct", "agrees": True},
                metadata={"mode": "debate", "iteration": 1},
            )

            traj.log_step(
                agent="reward",
                action="ground_truth_verify",
                input_data={"solution": "4", "ground_truth": "4"},
                output_data={"reward": 1.0, "is_correct": True},
                metadata={"mode": "debate"},
            )

        # Read back and parse
        entries = []
        with open(trajectory_path) as f:
            for line in f:
                entry_dict = json.loads(line)
                # Validate can be parsed back to TrajectoryEntry
                entry = TrajectoryEntry(**entry_dict)
                entries.append(entry)

        # Verify integrity
        assert len(entries) == 3, f"Expected 3 entries, found {len(entries)}"

        # Check first entry (solver)
        assert entries[0].agent == "solver"
        assert entries[0].action == "generate"
        assert entries[0].input["problem"] == "What is 2+2?"
        assert entries[0].output["solution"] == "4"
        assert entries[0].metadata["mode"] == "debate"

        # Check second entry (verifier)
        assert entries[1].agent == "verifier"
        assert entries[1].action == "verify"
        assert entries[1].output["feedback"] == "Correct"

        # Check third entry (reward)
        assert entries[2].agent == "reward"
        assert entries[2].action == "ground_truth_verify"
        assert entries[2].output["reward"] == 1.0
        assert entries[2].output["is_correct"] is True
