"""Tests for the inspect_logging conversion pipeline.

Covers trajectory reader, sample builder, log builder, round-trip file I/O,
and edge cases like malformed JSONL and missing agents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inspect_ai.log import EvalSample, read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Score

from src.infrastructure.inspect_logging import (
    build_baseline_sample,
    build_eval_log,
    convert_trajectories,
    read_and_group_trajectories,
)
from src.models.trajectory import TrajectoryEntry


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_trajectories.jsonl"


def _make_entry(
    *,
    agent: str = "solver_0",
    run_id: str = "test-run-001",
    step_id: int = 0,
    problem: str = "What is 2+2?",
    solution: str = "4",
    reward: float | None = None,
    score: float | None = None,
    reasoning: str | None = None,
    method: str | None = None,
    expected_answer: str | None = None,
    predicted_answer: str | None = None,
    tokens: int | dict = 100,
) -> TrajectoryEntry:
    """Factory for creating TrajectoryEntry test objects."""
    action = "generate_solution"
    output: dict = {"solution": solution}
    metadata: dict = {
        "model_version": "1.0",
        "config_hash": "test_hash",
        "tokens": tokens,
        "cost_usd": 0.001,
        "agent_name": agent,
    }

    if "judge" in agent:
        action = "judge_solution"
        output = {"score": score or 0.5, "reasoning": reasoning or "test reasoning"}
    elif agent == "reward":
        action = "compute_reward"
        output = {"reward": reward or 1.0, "method": method or "exact_match"}
        if expected_answer is not None:
            metadata["expected_answer"] = expected_answer
        if predicted_answer is not None:
            metadata["predicted_answer"] = predicted_answer

    return TrajectoryEntry(
        timestamp="2026-01-01T00:00:00+00:00",
        run_id=run_id,
        step_id=step_id,
        agent=agent,
        action=action,
        input={"problem": problem, "prompt": f"Solve: {problem}"},
        output=output,
        metadata=metadata,
        reward=reward,
        terminal=False,
        success=None,
    )


# ---------------------------------------------------------------------------
# Test: trajectory reader
# ---------------------------------------------------------------------------


class TestReadAndGroupTrajectories:
    """Tests for read_and_group_trajectories()."""

    def test_read_and_group_trajectories(self) -> None:
        """Read sample_trajectories.jsonl and verify grouping by run_id."""
        groups = read_and_group_trajectories(SAMPLE_DATA_PATH)

        # Returns dict with multiple run_id keys (sample data has 50 unique run_ids)
        assert isinstance(groups, dict)
        assert len(groups) > 1

        # Each group: all entries share the same run_id
        for run_id, entries in groups.items():
            assert len(entries) > 0
            for entry in entries:
                assert entry.run_id == run_id

        # Entries within each group should be in file order (step_id ascending)
        for entries in groups.values():
            step_ids = [e.step_id for e in entries]
            assert step_ids == sorted(step_ids), "Entries should be in step_id order"

    def test_malformed_jsonl_lines_skipped(self, tmp_path: Path) -> None:
        """Malformed lines are skipped; valid lines are still parsed."""
        jsonl_file = tmp_path / "mixed.jsonl"
        valid_entry = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "run_id": "valid-run",
            "step_id": 0,
            "agent": "solver_0",
            "action": "generate_solution",
            "input": {"problem": "test"},
            "output": {"solution": "42"},
            "metadata": {"tokens": 10},
            "reward": 1.0,
            "terminal": False,
            "success": None,
        }
        lines = [
            json.dumps(valid_entry),
            "THIS IS NOT JSON",
            '{"partial": "json but missing required fields"}',
            "",  # empty line (should be skipped)
        ]
        jsonl_file.write_text("\n".join(lines) + "\n")

        groups = read_and_group_trajectories(jsonl_file)

        # Only the valid entry should be parsed
        assert len(groups) == 1
        assert "valid-run" in groups
        assert len(groups["valid-run"]) == 1


# ---------------------------------------------------------------------------
# Test: sample builder
# ---------------------------------------------------------------------------


class TestBuildBaselineSample:
    """Tests for build_baseline_sample()."""

    def test_build_baseline_sample_with_scores(self) -> None:
        """Solver + reward + judge produces a sample with both scorers."""
        entries = [
            _make_entry(agent="solver_0", step_id=0, reward=1.0, solution="x = 6"),
            _make_entry(
                agent="reward",
                step_id=1,
                reward=1.0,
                method="exact_match",
                expected_answer="6",
                predicted_answer="6",
            ),
            _make_entry(agent="judge", step_id=2, score=0.85, reasoning="Correct approach"),
        ]

        sample = build_baseline_sample(entries, sample_id=1)

        # Input is a string (Pitfall 3)
        assert isinstance(sample.input, str)
        assert sample.epoch == 1

        # Messages: at least user + assistant
        assert len(sample.messages) >= 2
        assert any(isinstance(m, ChatMessageUser) for m in sample.messages)
        assert any(isinstance(m, ChatMessageAssistant) for m in sample.messages)

        # Scores: both ground_truth and judge present
        assert sample.scores is not None
        assert "ground_truth" in sample.scores
        assert isinstance(sample.scores["ground_truth"].value, float)
        assert "judge" in sample.scores
        assert isinstance(sample.scores["judge"].value, float)

        # Ground truth explanation has method description
        assert "exact_match" in sample.scores["ground_truth"].explanation

        # Target is non-empty string
        assert isinstance(sample.target, str)
        assert len(sample.target) > 0

    def test_build_baseline_sample_without_judge(self) -> None:
        """Solver + reward only (typical baseline) produces ground_truth but no judge."""
        entries = [
            _make_entry(agent="solver_0", step_id=0, reward=1.0, solution="42"),
            _make_entry(
                agent="reward",
                step_id=1,
                reward=1.0,
                method="exact_match",
                expected_answer="42",
            ),
        ]

        sample = build_baseline_sample(entries, sample_id=1)

        assert sample.scores is not None
        assert "ground_truth" in sample.scores
        assert "judge" not in sample.scores

    def test_build_baseline_sample_with_system_prompt(self) -> None:
        """System prompt appears as first ChatMessageSystem."""
        entries = [
            _make_entry(agent="solver_0", step_id=0, solution="x=6"),
        ]

        sample = build_baseline_sample(
            entries, sample_id=1, system_prompt="You are a math solver"
        )

        assert len(sample.messages) >= 1
        first_msg = sample.messages[0]
        assert isinstance(first_msg, ChatMessageSystem)
        assert first_msg.content == "You are a math solver"

    def test_build_baseline_sample_multiple_solvers(self) -> None:
        """All solver attempts appear in messages (per user decision)."""
        entries = [
            _make_entry(agent="solver_0", step_id=0, solution="wrong answer"),
            _make_entry(agent="solver_1", step_id=1, solution="still wrong"),
            _make_entry(agent="solver_2", step_id=2, solution="correct!"),
        ]

        sample = build_baseline_sample(entries, sample_id=1)

        # All solver attempts should produce messages
        assistant_msgs = [
            m for m in sample.messages if isinstance(m, ChatMessageAssistant)
        ]
        assert len(assistant_msgs) == 3
        assert assistant_msgs[0].content == "wrong answer"
        assert assistant_msgs[1].content == "still wrong"
        assert assistant_msgs[2].content == "correct!"


# ---------------------------------------------------------------------------
# Test: log builder
# ---------------------------------------------------------------------------


class TestBuildEvalLog:
    """Tests for build_eval_log()."""

    def test_build_eval_log_dual_scorers(self) -> None:
        """EvalLog from multiple samples has correct structure and dual scorers."""
        # Build two samples with both scorers
        entries_1 = [
            _make_entry(agent="solver_0", run_id="run-1", step_id=0, reward=1.0),
            _make_entry(agent="reward", run_id="run-1", step_id=1, reward=1.0, method="exact_match", expected_answer="6"),
            _make_entry(agent="judge", run_id="run-1", step_id=2, score=0.9),
        ]
        entries_2 = [
            _make_entry(agent="solver_0", run_id="run-2", step_id=0, reward=0.0),
            _make_entry(agent="reward", run_id="run-2", step_id=1, reward=0.0, method="exact_match", expected_answer="10"),
            _make_entry(agent="judge", run_id="run-2", step_id=2, score=0.3),
        ]

        sample_1 = build_baseline_sample(entries_1, sample_id=1)
        sample_2 = build_baseline_sample(entries_2, sample_id=2)

        log = build_eval_log(
            samples=[sample_1, sample_2],
            task_name="test_task",
            model_name="test-model",
        )

        assert log.status == "success"
        assert log.version == 2

        # Results must have scores for both scorers
        assert log.results is not None
        assert log.results.scores is not None
        assert len(log.results.scores) >= 2

        scorer_names = {s.name for s in log.results.scores}
        assert "ground_truth" in scorer_names
        assert "judge" in scorer_names

        # Each EvalScore has mean and count metrics
        for eval_score in log.results.scores:
            assert "mean" in eval_score.metrics
            assert "count" in eval_score.metrics

        # Scorer names in results must match scorer names in samples (Pitfall 2)
        sample_scorer_names = set()
        for sample in [sample_1, sample_2]:
            if sample.scores:
                sample_scorer_names.update(sample.scores.keys())
        assert scorer_names == sample_scorer_names


# ---------------------------------------------------------------------------
# Test: round-trip .eval file
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Tests for convert_trajectories() with file I/O round-trip."""

    def test_round_trip_eval_file(self, tmp_path: Path) -> None:
        """Convert sample JSONL -> write .eval -> read back; data preserved."""
        output_path = convert_trajectories(
            jsonl_path=SAMPLE_DATA_PATH,
            output_dir=tmp_path,
            task_name="roundtrip_test",
        )

        assert output_path.exists()
        assert output_path.suffix == ".eval"

        # Read back the written file
        loaded_log = read_eval_log(str(output_path))

        # Same number of samples as unique run_ids in source
        groups = read_and_group_trajectories(SAMPLE_DATA_PATH)
        assert loaded_log.samples is not None
        assert len(loaded_log.samples) == len(groups)

        # Check one sample has expected fields
        sample = loaded_log.samples[0]
        assert isinstance(sample.input, str)
        assert len(sample.input) > 0
        assert sample.messages is not None
        assert len(sample.messages) >= 2
        assert sample.scores is not None
        assert len(sample.scores) > 0
        assert sample.target is not None

        # Scorer names preserved in round-trip
        scorer_names_in_sample = set(sample.scores.keys())
        result_scorer_names = {s.name for s in loaded_log.results.scores}
        # Sample scorer names should be a subset of result scorer names
        assert scorer_names_in_sample.issubset(result_scorer_names)
