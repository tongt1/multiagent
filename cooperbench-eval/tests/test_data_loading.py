"""Tests for data loading modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.data_loading.schemas import AgentRole, Message, TaskData
from src.data_loading.trajectory_loader import load_trajectories


class TestTrajectoryLoader:
    """Tests for JSONL trajectory loading."""

    def test_loads_valid_jsonl(self, tmp_path: Path):
        """Test loading a valid JSONL trajectory file."""
        entries = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "run_id": "run-001",
                "step_id": 0,
                "agent": "solver_0",
                "action": "generate_solution",
                "input": {"problem": "2+2=?"},
                "output": {"solution": "4"},
                "metadata": {"agent_role": "solver"},
                "reward": 1.0,
                "terminal": False,
                "success": None,
            },
            {
                "timestamp": "2026-01-01T00:00:01Z",
                "run_id": "run-001",
                "step_id": 1,
                "agent": "verifier",
                "action": "verify_solution",
                "input": {"problem": "2+2=?"},
                "output": {"feedback": "Correct!"},
                "metadata": {"agent_role": "verifier"},
                "reward": 1.0,
                "terminal": True,
                "success": True,
            },
        ]

        filepath = tmp_path / "test.jsonl"
        with open(filepath, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        tasks = load_trajectories(filepath)
        assert len(tasks) == 1

        task = tasks[0]
        assert task.task_id == "run-001"
        assert task.run_id == "run-001"
        assert len(task.messages) == 2
        assert task.agents == ["solver_0", "verifier"]
        assert task.task_description == "2+2=?"
        assert task.eval_result is not None
        assert task.eval_result.passed is True
        assert task.eval_result.score == 1.0

    def test_groups_by_run_id(self, tmp_path: Path):
        """Test that entries are grouped by run_id."""
        entries = []
        for run_id in ["run-001", "run-002"]:
            for step in range(3):
                entries.append({
                    "run_id": run_id,
                    "step_id": step,
                    "agent": f"solver_{step}",
                    "action": "generate_solution",
                    "input": {"problem": "test"},
                    "output": {"solution": "answer"},
                    "metadata": {},
                    "reward": 0.5,
                    "terminal": step == 2,
                    "success": step == 2,
                })

        filepath = tmp_path / "test.jsonl"
        with open(filepath, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        tasks = load_trajectories(filepath)
        assert len(tasks) == 2
        assert {t.run_id for t in tasks} == {"run-001", "run-002"}

    def test_handles_missing_file(self):
        tasks = load_trajectories("/nonexistent/path.jsonl")
        assert tasks == []

    def test_handles_empty_file(self, tmp_path: Path):
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("")
        tasks = load_trajectories(filepath)
        assert tasks == []

    def test_loads_real_data(self):
        """Integration test with actual trajectory data."""
        data_path = Path("/home/terry_tong_cohere_com/multiagent/data/sample_trajectories.jsonl")
        if not data_path.exists():
            pytest.skip("Sample trajectory data not available")

        tasks = load_trajectories(data_path)
        assert len(tasks) == 50
        for task in tasks:
            assert len(task.messages) > 0
            assert len(task.agents) > 1
            assert task.task_description


class TestSchemas:
    """Tests for data schemas."""

    def test_message_is_question(self):
        q = Message(agent="a", content="What is the answer?", index=0)
        assert q.is_question

        q2 = Message(agent="a", content="Can you verify this?", index=0)
        assert q2.is_question

        s = Message(agent="a", content="The answer is 42.", index=0)
        assert not s.is_question

    def test_message_word_count(self):
        m = Message(agent="a", content="one two three four", index=0)
        assert m.word_count == 4

    def test_task_data_is_solo(self):
        solo = TaskData(task_id="t1", run_id="r1", agents=["a"])
        assert solo.is_solo

        multi = TaskData(task_id="t2", run_id="r1", agents=["a", "b"])
        assert not multi.is_solo

    def test_task_data_agent_messages(self):
        task = TaskData(
            task_id="t1",
            run_id="r1",
            messages=[
                Message(agent="a", content="hello", index=0),
                Message(agent="b", content="world", index=1),
                Message(agent="a", content="again", index=2),
            ],
            agents=["a", "b"],
        )
        groups = task.agent_messages
        assert len(groups["a"]) == 2
        assert len(groups["b"]) == 1
