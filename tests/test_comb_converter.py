"""Tests for Comb JSONL converter.

Validates that Phase 1 trajectory JSONL files convert correctly to
Comb-compatible CommandDataCombItem format.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.comb_converter import (
    convert_batch,
    convert_single_trajectory,
    convert_trajectories_to_comb,
)


def test_convert_single_debate_trajectory():
    """Test converting a single debate trajectory to Comb format."""
    # Create synthetic debate trajectory steps
    steps = [
        {
            "run_id": "debate_001",
            "step_id": 0,
            "agent": "solver",
            "action": "generate",
            "input": {"problem": "What is 2+2?", "ground_truth": "4"},
            "output": {"solution": "4"},
            "reward": 0.0,
            "metadata": {"mode": "debate", "trajectory_id": 0, "ground_truth": "4"},
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "run_id": "debate_001",
            "step_id": 1,
            "agent": "verifier",
            "action": "verify",
            "input": {"solution": "4"},
            "output": {"feedback": "Looks correct"},
            "reward": 0.0,
            "metadata": {"mode": "debate", "trajectory_id": 0},
            "timestamp": "2026-01-01T00:00:01Z",
        },
        {
            "run_id": "debate_001",
            "step_id": 2,
            "agent": "judge",
            "action": "score",
            "input": {"solution": "4", "feedback": "Looks correct"},
            "output": {"score": 0.9},
            "reward": 0.0,
            "metadata": {"mode": "debate", "trajectory_id": 0},
            "timestamp": "2026-01-01T00:00:02Z",
        },
        {
            "run_id": "debate_001",
            "step_id": 3,
            "agent": "reward",
            "action": "ground_truth_verify",
            "input": {"solution": "4", "ground_truth": "4"},
            "output": {"reward": 1.0, "is_correct": True},
            "reward": 1.0,
            "metadata": {"mode": "debate", "trajectory_id": 0},
            "timestamp": "2026-01-01T00:00:03Z",
        },
    ]

    # Convert to Comb format
    comb_item = convert_single_trajectory("debate_001", steps, "debate")

    # Verify structure
    assert "comb_env_name" in comb_item
    assert "agent_trajectory" in comb_item
    assert "validator_annotation" in comb_item
    assert "custom_data" in comb_item

    # Verify comb_env_name is "math_debate" for debate mode
    assert comb_item["comb_env_name"] == "math_debate"

    # Verify agent_trajectory structure
    agent_traj = comb_item["agent_trajectory"]
    assert "turns" in agent_traj
    assert "preamble" in agent_traj
    assert agent_traj["preamble"] is None

    # Verify user turn contains problem
    assert len(agent_traj["turns"]) == 1
    user_turn = agent_traj["turns"][0]
    assert user_turn["role"] == "user"
    assert "contents" in user_turn
    assert len(user_turn["contents"]) == 1
    assert user_turn["contents"][0]["type"] == "text"
    assert user_turn["contents"][0]["text"] == "What is 2+2?"

    # Verify validator_annotation contains gold_answer
    validator = comb_item["validator_annotation"]
    assert "spec" in validator
    assert "arguments" in validator["spec"]
    assert "gold_answer" in validator["spec"]["arguments"]
    assert validator["spec"]["arguments"]["gold_answer"] == "4"


def test_convert_single_baseline_trajectory():
    """Test converting a single baseline trajectory to Comb format."""
    # Create synthetic baseline trajectory steps
    steps = [
        {
            "run_id": "baseline_001",
            "step_id": 0,
            "agent": "solver",
            "action": "generate",
            "input": {"problem": "Solve x+3=7", "ground_truth": "x=4", "feedback": None},
            "output": {"solution": "x=4"},
            "reward": 0.0,
            "metadata": {"mode": "baseline", "trajectory_id": 0, "ground_truth": "x=4"},
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "run_id": "baseline_001",
            "step_id": 1,
            "agent": "reward",
            "action": "ground_truth_verify",
            "input": {"solution": "x=4", "ground_truth": "x=4"},
            "output": {"reward": 1.0, "is_correct": True},
            "reward": 1.0,
            "metadata": {
                "mode": "baseline",
                "trajectory_id": 0,
                "termination_reason": "single_iteration",
            },
            "timestamp": "2026-01-01T00:00:01Z",
        },
    ]

    # Convert to Comb format
    comb_item = convert_single_trajectory("baseline_001", steps, "baseline")

    # Verify comb_env_name is "math" for baseline mode
    assert comb_item["comb_env_name"] == "math"

    # Verify agent_trajectory contains problem
    user_turn = comb_item["agent_trajectory"]["turns"][0]
    assert user_turn["contents"][0]["text"] == "Solve x+3=7"

    # Verify validator_annotation contains ground truth
    gold_answer = comb_item["validator_annotation"]["spec"]["arguments"]["gold_answer"]
    assert gold_answer == "x=4"


def test_convert_trajectories_to_comb_file():
    """Test round-trip: write Phase 1 JSONL -> convert -> read Comb JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create Phase 1 trajectory JSONL
        input_path = Path(tmpdir) / "phase1_debate.jsonl"
        with open(input_path, "w", encoding="utf-8") as f:
            # Write steps for trajectory 1
            steps_1 = [
                {
                    "run_id": "traj_001",
                    "step_id": 0,
                    "agent": "solver",
                    "action": "generate",
                    "input": {"problem": "2+2=?", "ground_truth": "4"},
                    "output": {"solution": "4"},
                    "reward": 0.0,
                    "metadata": {"mode": "debate", "trajectory_id": 0, "ground_truth": "4"},
                    "timestamp": "2026-01-01T00:00:00Z",
                },
                {
                    "run_id": "traj_001",
                    "step_id": 1,
                    "agent": "reward",
                    "action": "ground_truth_verify",
                    "input": {"solution": "4", "ground_truth": "4"},
                    "output": {"reward": 1.0, "is_correct": True},
                    "reward": 1.0,
                    "metadata": {"mode": "debate", "trajectory_id": 0},
                    "timestamp": "2026-01-01T00:00:01Z",
                },
            ]
            for step in steps_1:
                f.write(json.dumps(step) + "\n")

            # Write steps for trajectory 2
            steps_2 = [
                {
                    "run_id": "traj_002",
                    "step_id": 0,
                    "agent": "solver",
                    "action": "generate",
                    "input": {"problem": "3*3=?", "ground_truth": "9"},
                    "output": {"solution": "9"},
                    "reward": 0.0,
                    "metadata": {"mode": "debate", "trajectory_id": 1, "ground_truth": "9"},
                    "timestamp": "2026-01-01T00:01:00Z",
                },
                {
                    "run_id": "traj_002",
                    "step_id": 1,
                    "agent": "reward",
                    "action": "ground_truth_verify",
                    "input": {"solution": "9", "ground_truth": "9"},
                    "output": {"reward": 1.0, "is_correct": True},
                    "reward": 1.0,
                    "metadata": {"mode": "debate", "trajectory_id": 1},
                    "timestamp": "2026-01-01T00:01:01Z",
                },
            ]
            for step in steps_2:
                f.write(json.dumps(step) + "\n")

        # Convert to Comb format
        output_path = Path(tmpdir) / "comb_output.jsonl"
        count = convert_trajectories_to_comb(input_path, output_path, mode="debate")

        # Verify conversion count
        assert count == 2, f"Expected 2 trajectories, got {count}"

        # Read back and verify Comb format
        comb_items = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    comb_items.append(json.loads(line))

        assert len(comb_items) == 2

        # Check first trajectory
        item_1 = comb_items[0]
        assert item_1["comb_env_name"] == "math_debate"
        assert item_1["agent_trajectory"]["turns"][0]["contents"][0]["text"] == "2+2=?"
        assert item_1["validator_annotation"]["spec"]["arguments"]["gold_answer"] == "4"

        # Check second trajectory
        item_2 = comb_items[1]
        assert item_2["comb_env_name"] == "math_debate"
        assert item_2["agent_trajectory"]["turns"][0]["contents"][0]["text"] == "3*3=?"
        assert item_2["validator_annotation"]["spec"]["arguments"]["gold_answer"] == "9"


def test_convert_batch_directory():
    """Test batch conversion of multiple JSONL files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "trajectories"
        input_dir.mkdir()

        # Create debate trajectory file
        debate_file = input_dir / "debate.jsonl"
        with open(debate_file, "w", encoding="utf-8") as f:
            steps = [
                {
                    "run_id": "debate_run",
                    "step_id": 0,
                    "agent": "solver",
                    "action": "generate",
                    "input": {"problem": "Debate problem", "ground_truth": "answer1"},
                    "output": {"solution": "answer1"},
                    "reward": 0.0,
                    "metadata": {"mode": "debate", "ground_truth": "answer1"},
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            ]
            for step in steps:
                f.write(json.dumps(step) + "\n")

        # Create baseline trajectory file
        baseline_file = input_dir / "baseline.jsonl"
        with open(baseline_file, "w", encoding="utf-8") as f:
            steps = [
                {
                    "run_id": "baseline_run",
                    "step_id": 0,
                    "agent": "solver",
                    "action": "generate",
                    "input": {"problem": "Baseline problem", "ground_truth": "answer2"},
                    "output": {"solution": "answer2"},
                    "reward": 0.0,
                    "metadata": {"mode": "baseline", "ground_truth": "answer2"},
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            ]
            for step in steps:
                f.write(json.dumps(step) + "\n")

        # Convert batch
        output_path = Path(tmpdir) / "batch_output.jsonl"
        count = convert_batch(input_dir, output_path)

        # Verify count
        assert count == 2

        # Read and verify
        comb_items = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    comb_items.append(json.loads(line))

        assert len(comb_items) == 2

        # Verify debate item has math_debate env
        debate_items = [
            item for item in comb_items if item["comb_env_name"] == "math_debate"
        ]
        assert len(debate_items) == 1
        assert (
            debate_items[0]["agent_trajectory"]["turns"][0]["contents"][0]["text"]
            == "Debate problem"
        )

        # Verify baseline item has math env
        baseline_items = [item for item in comb_items if item["comb_env_name"] == "math"]
        assert len(baseline_items) == 1
        assert (
            baseline_items[0]["agent_trajectory"]["turns"][0]["contents"][0]["text"]
            == "Baseline problem"
        )


def test_comb_item_has_all_required_fields():
    """Test that converted Comb items have all required CommandDataCombItem fields."""
    steps = [
        {
            "run_id": "test_run",
            "step_id": 0,
            "agent": "solver",
            "action": "generate",
            "input": {"problem": "Test problem", "ground_truth": "Test answer"},
            "output": {"solution": "Test answer"},
            "reward": 0.0,
            "metadata": {"mode": "debate", "ground_truth": "Test answer"},
            "timestamp": "2026-01-01T00:00:00Z",
        },
    ]

    comb_item = convert_single_trajectory("test_run", steps, "debate")

    # All required CommandDataCombItem fields
    required_fields = [
        "comb_env_name",
        "agent_trajectory",
        "validator_annotation",
        "custom_data",
    ]
    for field in required_fields:
        assert field in comb_item, f"Missing required field: {field}"

    # Verify nested structure
    assert "turns" in comb_item["agent_trajectory"]
    assert "preamble" in comb_item["agent_trajectory"]
    assert "spec" in comb_item["validator_annotation"]
    assert "arguments" in comb_item["validator_annotation"]["spec"]
    assert "gold_answer" in comb_item["validator_annotation"]["spec"]["arguments"]


def test_mode_detection_from_metadata():
    """Test that converter auto-detects mode from trajectory metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create file with baseline metadata
        input_path = Path(tmpdir) / "auto_detect.jsonl"
        with open(input_path, "w", encoding="utf-8") as f:
            steps = [
                {
                    "run_id": "auto_run",
                    "step_id": 0,
                    "agent": "solver",
                    "action": "generate",
                    "input": {"problem": "Auto-detect test", "ground_truth": "answer"},
                    "output": {"solution": "answer"},
                    "reward": 0.0,
                    "metadata": {
                        "mode": "baseline",
                        "ground_truth": "answer",
                    },  # baseline in metadata
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            ]
            for step in steps:
                f.write(json.dumps(step) + "\n")

        # Convert with default mode="debate" but should auto-detect "baseline"
        output_path = Path(tmpdir) / "auto_output.jsonl"
        convert_trajectories_to_comb(input_path, output_path, mode="debate")

        # Read and verify it detected baseline mode
        with open(output_path, "r", encoding="utf-8") as f:
            comb_item = json.loads(f.readline())

        # Should be "math" (baseline), not "math_debate" (debate)
        assert comb_item["comb_env_name"] == "math"


def test_missing_ground_truth_raises_error():
    """Test that converter raises error if ground truth is missing."""
    steps = [
        {
            "run_id": "no_gt_run",
            "step_id": 0,
            "agent": "solver",
            "action": "generate",
            "input": {"problem": "Test problem"},  # No ground_truth
            "output": {"solution": "Test answer"},
            "reward": 0.0,
            "metadata": {"mode": "debate"},  # No ground_truth here either
            "timestamp": "2026-01-01T00:00:00Z",
        },
    ]

    with pytest.raises(ValueError, match="Missing ground_truth"):
        convert_single_trajectory("no_gt_run", steps, "debate")


def test_missing_problem_raises_error():
    """Test that converter raises error if problem text is missing."""
    steps = [
        {
            "run_id": "no_problem_run",
            "step_id": 0,
            "agent": "solver",
            "action": "generate",
            "input": {"ground_truth": "answer"},  # No problem field
            "output": {"solution": "answer"},
            "reward": 0.0,
            "metadata": {"mode": "debate", "ground_truth": "answer"},
            "timestamp": "2026-01-01T00:00:00Z",
        },
    ]

    with pytest.raises(ValueError, match="Missing problem text"):
        convert_single_trajectory("no_problem_run", steps, "debate")


def test_empty_steps_raises_error():
    """Test that converter raises error for empty steps list."""
    with pytest.raises(ValueError, match="Empty steps list"):
        convert_single_trajectory("empty_run", [], "debate")
