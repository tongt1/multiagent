"""Shared test fixtures for CooperBench eval tests."""

from __future__ import annotations

import pytest

from src.data_loading.schemas import (
    AgentRole,
    EvalResult,
    Message,
    PatchInfo,
    TaskData,
)


@pytest.fixture
def solo_task() -> TaskData:
    """A single-agent task (should be skipped by multi-agent classifiers)."""
    return TaskData(
        task_id="solo-001",
        run_id="run-001",
        messages=[
            Message(agent="agent_a", content="Working on the solution.", index=0, role=AgentRole.AGENT_A),
            Message(agent="agent_a", content="Done with the implementation.", index=1, role=AgentRole.AGENT_A),
        ],
        agents=["agent_a"],
        task_description="Implement feature X",
    )


@pytest.fixture
def multi_agent_task() -> TaskData:
    """A multi-agent task with basic conversation."""
    return TaskData(
        task_id="multi-001",
        run_id="run-001",
        messages=[
            Message(
                agent="solver_0",
                content="[generate_solution] x = 7",
                index=0,
                role=AgentRole.AGENT_A,
                metadata={"agent_role": "solver", "reward": 0.0},
            ),
            Message(
                agent="solver_1",
                content="[generate_solution] x = 6",
                index=1,
                role=AgentRole.AGENT_B,
                metadata={"agent_role": "solver", "reward": 1.0},
            ),
            Message(
                agent="solver_2",
                content="[generate_solution] x = 6",
                index=2,
                role=AgentRole.UNKNOWN,
                metadata={"agent_role": "solver", "reward": 1.0},
            ),
            Message(
                agent="verifier",
                content="[verify_solution] The solution has errors. Let me recalculate.",
                index=3,
                role=AgentRole.UNKNOWN,
                metadata={"agent_role": "verifier", "reward": 0.3},
            ),
            Message(
                agent="judge",
                content="[judge_solution] Confidence: 0.64 based on verification.",
                index=4,
                role=AgentRole.UNKNOWN,
                metadata={"agent_role": "judge", "reward": 0.64},
            ),
        ],
        agents=["judge", "solver_0", "solver_1", "solver_2", "verifier"],
        task_description="Solve for x: 2x + 5 = 17",
        eval_result=EvalResult(task_id="multi-001", passed=True, score=0.64),
    )


@pytest.fixture
def repetitive_task() -> TaskData:
    """A task with repetitive messages from one agent."""
    return TaskData(
        task_id="rep-001",
        run_id="run-001",
        messages=[
            Message(
                agent="solver_0",
                content="Let me solve this equation. First subtract 5 from both sides to get 2x = 12, then divide by 2.",
                index=0,
                role=AgentRole.AGENT_A,
            ),
            Message(
                agent="solver_1",
                content="I'll work on this problem as well.",
                index=1,
                role=AgentRole.AGENT_B,
            ),
            Message(
                agent="solver_0",
                content="Let me solve this equation. First subtract 5 from both sides to get 2x = 12, then divide by 2.",
                index=2,
                role=AgentRole.AGENT_A,
            ),
            Message(
                agent="solver_1",
                content="My approach is different from the first solver.",
                index=3,
                role=AgentRole.AGENT_B,
            ),
        ],
        agents=["solver_0", "solver_1"],
        task_description="Solve: 2x + 5 = 17",
    )


@pytest.fixture
def unresponsive_task() -> TaskData:
    """A task with unanswered questions."""
    return TaskData(
        task_id="unresponsive-001",
        run_id="run-001",
        messages=[
            Message(
                agent="solver_0",
                content="I think we should use the quadratic formula. What approach are you taking solver_1?",
                index=0,
                role=AgentRole.AGENT_A,
            ),
            Message(
                agent="solver_0",
                content="Can you verify this approach works for negative discriminant cases?",
                index=1,
                role=AgentRole.AGENT_A,
            ),
            Message(
                agent="solver_0",
                content="How should we handle the edge case where x equals zero?",
                index=2,
                role=AgentRole.AGENT_A,
            ),
            Message(
                agent="solver_0",
                content="I'll just proceed with my approach since nobody responded.",
                index=3,
                role=AgentRole.AGENT_A,
            ),
        ],
        agents=["solver_0", "solver_1"],
        task_description="Solve the quadratic equation",
    )


@pytest.fixture
def patch_overlap_task() -> TaskData:
    """A task where agents modified overlapping files."""
    return TaskData(
        task_id="overlap-001",
        run_id="run-001",
        messages=[
            Message(agent="agent_a", content="I'll modify the solver module.", index=0, role=AgentRole.AGENT_A),
            Message(agent="agent_b", content="I'll also work on the solver.", index=1, role=AgentRole.AGENT_B),
        ],
        patches=[
            PatchInfo(
                agent="agent_a",
                raw_diff="",
                files_modified=["src/solver.py", "src/utils.py", "tests/test_solver.py"],
                functions_modified={"src/solver.py": ["solve", "validate"]},
            ),
            PatchInfo(
                agent="agent_b",
                raw_diff="",
                files_modified=["src/solver.py", "src/config.py"],
                functions_modified={"src/solver.py": ["solve", "parse_input"]},
            ),
        ],
        agents=["agent_a", "agent_b"],
        task_description="Improve the equation solver",
    )


@pytest.fixture
def placeholder_task() -> TaskData:
    """A task with placeholder code in patches."""
    return TaskData(
        task_id="placeholder-001",
        run_id="run-001",
        messages=[
            Message(agent="agent_a", content="Here's my implementation.", index=0, role=AgentRole.AGENT_A),
        ],
        patches=[
            PatchInfo(
                agent="agent_a",
                raw_diff="",
                files_modified=["src/solver.py"],
                added_lines=[
                    "def solve(equation):",
                    "    # TODO: implement proper parsing",
                    "    raise NotImplementedError",
                    "    pass",
                ],
            ),
        ],
        agents=["agent_a"],
        task_description="Implement equation solver",
    )
