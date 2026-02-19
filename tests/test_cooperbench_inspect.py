"""Tests for the CooperBench → Inspect AI conversion pipeline.

Covers: run loading, sample building, log assembly, round-trip file I/O,
solo vs coop modes, inter-agent conversation handling, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inspect_ai.log import EvalSample, read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

# Import the converter functions directly from the script
import importlib.util
_script_path = Path(__file__).resolve().parent.parent / "scripts" / "convert_cooperbench_to_inspect.py"
_spec = importlib.util.spec_from_file_location("convert_cooperbench", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_run = _mod.load_run
build_sample_from_run = _mod.build_sample_from_run
build_eval_log = _mod.build_eval_log
discover_runs = _mod.discover_runs


# ---------------------------------------------------------------------------
# Fixtures: create CooperBench-format data on disk
# ---------------------------------------------------------------------------


def _make_traj(
    agent_id: str = "agent1",
    feature_id: int = 1,
    model: str = "test-model",
    status: str = "Submitted",
    cost: float = 0.1,
    steps: int = 5,
    messages: list[dict] | None = None,
) -> dict:
    """Create a CooperBench trajectory dict."""
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a coding agent.", "timestamp": 1000},
            {"role": "user", "content": "Implement feature X.", "timestamp": 1001},
            {"role": "assistant", "content": "I'll implement feature X.", "timestamp": 1002, "extra": {}},
            {"role": "user", "content": "Output of running tests...", "timestamp": 1003},
            {"role": "assistant", "content": "Done implementing.", "timestamp": 1004, "extra": {}},
        ]
    return {
        "repo": "test_repo_task",
        "task_id": 42,
        "feature_id": feature_id,
        "agent_id": agent_id,
        "model": model,
        "status": status,
        "cost": cost,
        "steps": steps,
        "messages": messages,
    }


def _make_result(
    setting: str = "coop",
    agents: dict | None = None,
    messages_sent: int = 0,
    features: list[int] | None = None,
) -> dict:
    """Create a CooperBench result.json dict."""
    if agents is None:
        agents = {
            "agent1": {
                "feature_id": 1,
                "status": "Submitted",
                "cost": 0.1,
                "steps": 5,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "patch_lines": 100,
                "error": None,
            },
        }
    total_cost = sum(a["cost"] for a in agents.values())
    total_steps = sum(a["steps"] for a in agents.values())
    return {
        "repo": "test_repo_task",
        "task_id": 42,
        "features": features or [1],
        "setting": setting,
        "run_id": "abc123",
        "run_name": f"test-{setting}",
        "agent_framework": "mini_swe_agent",
        "model": "test-model",
        "started_at": "2026-02-18T00:00:00",
        "ended_at": "2026-02-18T00:05:00",
        "duration_seconds": 300.0,
        "agents": agents,
        "total_cost": total_cost,
        "total_steps": total_steps,
        "messages_sent": messages_sent,
        "log_dir": "logs/test",
    }


@pytest.fixture
def solo_run_dir(tmp_path: Path) -> Path:
    """Create a solo run directory with one agent."""
    run_dir = tmp_path / "solo" / "test_repo_task" / "42" / "f1"
    run_dir.mkdir(parents=True)

    traj = _make_traj(agent_id="solo", feature_id=1)
    (run_dir / "solo_traj.json").write_text(json.dumps(traj))

    result = _make_result(
        setting="solo",
        agents={
            "solo": {
                "feature_id": 1,
                "status": "Submitted",
                "cost": 0.15,
                "steps": 8,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "patch_lines": 50,
                "error": None,
            },
        },
    )
    (run_dir / "result.json").write_text(json.dumps(result))

    return run_dir


@pytest.fixture
def coop_run_dir(tmp_path: Path) -> Path:
    """Create a coop run directory with two agents + conversation."""
    run_dir = tmp_path / "coop" / "test_repo_task" / "42" / "f1_f3"
    run_dir.mkdir(parents=True)

    traj1 = _make_traj(agent_id="agent1", feature_id=1, cost=0.2, steps=10)
    traj2 = _make_traj(
        agent_id="agent2",
        feature_id=3,
        cost=0.3,
        steps=8,
        messages=[
            {"role": "system", "content": "You are agent 2.", "timestamp": 2000},
            {"role": "user", "content": "Implement feature 3.", "timestamp": 2001},
            {"role": "assistant", "content": "Working on feature 3.", "timestamp": 2002},
        ],
    )
    (run_dir / "agent1_traj.json").write_text(json.dumps(traj1))
    (run_dir / "agent3_traj.json").write_text(json.dumps(traj2))

    result = _make_result(
        setting="coop",
        features=[1, 3],
        messages_sent=2,
        agents={
            "agent1": {
                "feature_id": 1,
                "status": "Submitted",
                "cost": 0.2,
                "steps": 10,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "patch_lines": 200,
                "error": None,
            },
            "agent2": {
                "feature_id": 3,
                "status": "Submitted",
                "cost": 0.3,
                "steps": 8,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "patch_lines": 50,
                "error": None,
            },
        },
    )
    (run_dir / "result.json").write_text(json.dumps(result))

    conversation = [
        {
            "from": "agent1",
            "to": "agent2",
            "message": "I'm modifying file X, please avoid it.",
            "timestamp": 1500,
            "feature_id": 1,
        },
        {
            "from": "agent2",
            "to": "agent1",
            "message": "Got it, I'll work on file Y instead.",
            "timestamp": 1600,
            "feature_id": 3,
        },
    ]
    (run_dir / "conversation.json").write_text(json.dumps(conversation))

    return run_dir


# ---------------------------------------------------------------------------
# Test: load_run
# ---------------------------------------------------------------------------


class TestLoadRun:
    def test_load_solo_run(self, solo_run_dir: Path) -> None:
        run = load_run(solo_run_dir)
        assert run is not None
        assert run["result"]["setting"] == "solo"
        assert "solo" in run["trajectories"]
        assert len(run["trajectories"]["solo"]["messages"]) == 5
        assert run["conversation"] == []

    def test_load_coop_run_with_conversation(self, coop_run_dir: Path) -> None:
        run = load_run(coop_run_dir)
        assert run is not None
        assert run["result"]["setting"] == "coop"
        assert len(run["trajectories"]) == 2
        assert len(run["conversation"]) == 2

    def test_load_run_missing_result(self, tmp_path: Path) -> None:
        """Returns None if result.json doesn't exist."""
        assert load_run(tmp_path) is None

    def test_load_run_no_conversation(self, solo_run_dir: Path) -> None:
        """Solo runs have no conversation.json — returns empty list."""
        run = load_run(solo_run_dir)
        assert run["conversation"] == []


# ---------------------------------------------------------------------------
# Test: build_sample_from_run
# ---------------------------------------------------------------------------


class TestBuildSampleFromRun:
    def test_solo_sample_fields(self, solo_run_dir: Path) -> None:
        """Solo run produces valid EvalSample with expected fields."""
        run = load_run(solo_run_dir)
        sample = build_sample_from_run(run, sample_id=1)

        assert isinstance(sample, EvalSample)
        assert isinstance(sample.input, str)
        assert "test_repo_task" in sample.input
        assert "42" in sample.input
        assert sample.epoch == 1
        assert sample.target is not None

    def test_solo_sample_messages(self, solo_run_dir: Path) -> None:
        """Solo run messages include system, user, and assistant."""
        run = load_run(solo_run_dir)
        sample = build_sample_from_run(run, sample_id=1)

        assert len(sample.messages) >= 3
        has_system = any(isinstance(m, ChatMessageSystem) for m in sample.messages)
        has_user = any(isinstance(m, ChatMessageUser) for m in sample.messages)
        has_assistant = any(isinstance(m, ChatMessageAssistant) for m in sample.messages)
        assert has_system
        assert has_user
        assert has_assistant

    def test_solo_sample_scores(self, solo_run_dir: Path) -> None:
        """Solo run has submission_rate, cost_usd, total_steps, patch_lines scores."""
        run = load_run(solo_run_dir)
        sample = build_sample_from_run(run, sample_id=1)

        assert sample.scores is not None
        assert "submission_rate" in sample.scores
        assert sample.scores["submission_rate"].value == 1.0
        assert "cost_usd" in sample.scores
        assert "total_steps" in sample.scores
        assert "patch_lines" in sample.scores
        # Solo has no messages_sent
        assert "messages_sent" not in sample.scores

    def test_coop_sample_has_conversation_messages(self, coop_run_dir: Path) -> None:
        """Coop run includes inter-agent communication in messages."""
        run = load_run(coop_run_dir)
        sample = build_sample_from_run(run, sample_id=1)

        # Should contain inter-agent communication separator
        system_msgs = [
            m for m in sample.messages
            if isinstance(m, ChatMessageSystem) and "Inter-Agent" in m.content
        ]
        assert len(system_msgs) == 1

        # Should have messages_sent score
        assert "messages_sent" in sample.scores
        assert sample.scores["messages_sent"].value == 2.0

    def test_coop_sample_metadata(self, coop_run_dir: Path) -> None:
        """Coop run metadata has per-agent details."""
        run = load_run(coop_run_dir)
        sample = build_sample_from_run(run, sample_id=1)

        assert sample.metadata["repo"] == "test_repo_task"
        assert sample.metadata["setting"] == "coop"
        assert sample.metadata["run_id"] == "abc123"

    def test_sample_with_limits_exceeded(self, tmp_path: Path) -> None:
        """Agent with LimitsExceeded status affects submission_rate."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        traj = _make_traj(agent_id="agent1", status="LimitsExceeded")
        (run_dir / "agent1_traj.json").write_text(json.dumps(traj))

        result = _make_result(
            setting="solo",
            agents={
                "agent1": {
                    "feature_id": 1,
                    "status": "LimitsExceeded",
                    "cost": 0.5,
                    "steps": 30,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "patch_lines": 0,
                    "error": "Max steps exceeded",
                },
            },
        )
        (run_dir / "result.json").write_text(json.dumps(result))

        run = load_run(run_dir)
        sample = build_sample_from_run(run, sample_id=1)

        assert sample.scores["submission_rate"].value == 0.0

    def test_coop_both_agents_messages_present(self, coop_run_dir: Path) -> None:
        """Both agents' trajectories appear in messages."""
        run = load_run(coop_run_dir)
        sample = build_sample_from_run(run, sample_id=1)

        # Should have agent separators for both agents
        agent_separators = [
            m for m in sample.messages
            if isinstance(m, ChatMessageSystem) and "Agent:" in m.content
        ]
        assert len(agent_separators) == 2


# ---------------------------------------------------------------------------
# Test: build_eval_log
# ---------------------------------------------------------------------------


class TestBuildEvalLog:
    def test_log_structure(self, solo_run_dir: Path) -> None:
        """EvalLog has correct version, status, and sample count."""
        run = load_run(solo_run_dir)
        sample = build_sample_from_run(run, sample_id=1)
        log = build_eval_log([sample], "test_task", "test-model", "solo")

        assert log.version == 2
        assert log.status == "success"
        assert log.samples is not None
        assert len(log.samples) == 1
        assert log.eval.task == "test_task"
        assert log.eval.model == "test-model"

    def test_log_aggregate_scores(self, solo_run_dir: Path, coop_run_dir: Path) -> None:
        """Aggregate scores computed correctly across samples."""
        run1 = load_run(solo_run_dir)
        run2 = load_run(coop_run_dir)
        s1 = build_sample_from_run(run1, sample_id=1)
        s2 = build_sample_from_run(run2, sample_id=2)
        log = build_eval_log([s1, s2], "test", "model", "mixed")

        assert log.results is not None
        scorer_names = {s.name for s in log.results.scores}
        assert "submission_rate" in scorer_names
        assert "cost_usd" in scorer_names

        for eval_score in log.results.scores:
            assert "mean" in eval_score.metrics
            assert "count" in eval_score.metrics

    def test_scorer_name_alignment(self, solo_run_dir: Path) -> None:
        """Scorer names in results match those in samples."""
        run = load_run(solo_run_dir)
        sample = build_sample_from_run(run, sample_id=1)
        log = build_eval_log([sample], "test", "model", "solo")

        result_names = {s.name for s in log.results.scores}
        sample_names = set(sample.scores.keys())
        assert sample_names == result_names


# ---------------------------------------------------------------------------
# Test: discover_runs
# ---------------------------------------------------------------------------


class TestDiscoverRuns:
    def test_discover_nested_runs(self, solo_run_dir: Path, coop_run_dir: Path, tmp_path: Path) -> None:
        """Discovers all run directories containing result.json."""
        runs = discover_runs(tmp_path)
        assert len(runs) == 2
        assert solo_run_dir in runs
        assert coop_run_dir in runs

    def test_discover_empty_dir(self, tmp_path: Path) -> None:
        """Returns empty list for directory with no runs."""
        assert discover_runs(tmp_path) == []


# ---------------------------------------------------------------------------
# Test: round-trip .eval file
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_solo(self, solo_run_dir: Path, tmp_path: Path) -> None:
        """Write .eval from solo run, read back, verify fields preserved."""
        run = load_run(solo_run_dir)
        sample = build_sample_from_run(run, sample_id=1)
        log = build_eval_log([sample], "test", "test-model", "solo")

        output_path = tmp_path / "output" / "test.eval"
        output_path.parent.mkdir(parents=True)
        from inspect_ai.log import write_eval_log
        write_eval_log(log, location=str(output_path), format="eval")

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        loaded = read_eval_log(str(output_path))
        assert loaded.status == "success"
        assert len(loaded.samples) == 1
        assert isinstance(loaded.samples[0].input, str)
        assert loaded.samples[0].scores is not None
        assert "submission_rate" in loaded.samples[0].scores

    def test_round_trip_coop(self, coop_run_dir: Path, tmp_path: Path) -> None:
        """Write .eval from coop run, read back, verify conversation preserved."""
        run = load_run(coop_run_dir)
        sample = build_sample_from_run(run, sample_id=1)
        log = build_eval_log([sample], "test", "test-model", "coop")

        output_path = tmp_path / "output" / "coop.eval"
        output_path.parent.mkdir(parents=True)
        from inspect_ai.log import write_eval_log
        write_eval_log(log, location=str(output_path), format="eval")

        loaded = read_eval_log(str(output_path))
        assert len(loaded.samples) == 1
        s = loaded.samples[0]
        assert "messages_sent" in s.scores


# ---------------------------------------------------------------------------
# Test: real CooperBench data (integration)
# ---------------------------------------------------------------------------

COOPERBENCH_LOGS = Path("/mnt/data/terry/home/cooperbench-repro/repos/CooperBench/logs")


@pytest.mark.skipif(
    not COOPERBENCH_LOGS.exists(),
    reason="CooperBench data not available",
)
class TestRealCooperBenchData:
    def test_discover_all_runs(self) -> None:
        """Discovers 300 runs from real CooperBench data."""
        runs = discover_runs(COOPERBENCH_LOGS)
        assert len(runs) >= 300  # At least 300 runs (may grow as new seeds are added)

    def test_load_real_coop_comm_run(self) -> None:
        """Load a real coop-comm run and build a sample."""
        runs = discover_runs(COOPERBENCH_LOGS / "command-a-coop-comm")
        assert len(runs) > 0
        run = load_run(runs[0])
        assert run is not None
        assert len(run["trajectories"]) >= 1

        sample = build_sample_from_run(run, sample_id=1)
        assert isinstance(sample.input, str)
        assert len(sample.messages) > 0
        assert sample.scores is not None

    def test_load_real_solo_run(self) -> None:
        """Load a real solo run and build a sample."""
        runs = discover_runs(COOPERBENCH_LOGS / "command-a-solo")
        assert len(runs) > 0
        run = load_run(runs[0])
        assert run is not None

        sample = build_sample_from_run(run, sample_id=1)
        # Solo runs should have scores (submission_rate after solo→agents normalization)
        assert sample.scores is not None
        assert "cost_usd" in sample.scores
        assert "total_steps" in sample.scores
        if "submission_rate" in sample.scores:
            assert sample.scores["submission_rate"].value in (0.0, 1.0)

    def test_full_conversion_round_trip(self, tmp_path: Path) -> None:
        """Convert 5 real runs to .eval and read back."""
        runs = discover_runs(COOPERBENCH_LOGS / "command-a-solo")[:5]
        samples = []
        for i, run_dir in enumerate(runs):
            run = load_run(run_dir)
            samples.append(build_sample_from_run(run, sample_id=i + 1))

        log = build_eval_log(samples, "cooperbench", "command-a", "solo")
        output_path = tmp_path / "real_test.eval"
        from inspect_ai.log import write_eval_log
        write_eval_log(log, location=str(output_path), format="eval")

        loaded = read_eval_log(str(output_path))
        assert len(loaded.samples) == 5
        for s in loaded.samples:
            assert s.scores is not None
            assert "cost_usd" in s.scores
            assert "total_steps" in s.scores
