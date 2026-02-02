"""Tests for MARTI trajectory export functionality."""

import numpy as np
import pytest

from src.models.trajectory import TrajectoryEntry
from src.training.marti_exporter import (
    AgentGraphNode,
    MARTITrajectory,
    MARTITurn,
    add_graph_metadata,
    build_agent_graph,
    export_to_marti_format,
)


def test_build_agent_graph_default():
    """Test building agent graph with default 3 solvers."""
    graph = build_agent_graph(num_solvers=3)

    # Check structure
    assert "agents" in graph
    assert "spatial_masks" in graph
    assert "temporal_masks" in graph
    assert "num_rounds" in graph

    # Check agents
    agents = graph["agents"]
    assert len(agents) == 5  # 3 solvers + 1 verifier + 1 judge

    # Check solver agents
    for i in range(3):
        assert agents[i]["agent_name"] == f"solver_{i}"
        assert agents[i]["agent_role"] == "solver"
        assert agents[i]["round_idx"] == 0

    # Check verifier
    assert agents[3]["agent_name"] == "verifier"
    assert agents[3]["agent_role"] == "verifier"
    assert agents[3]["round_idx"] == 1

    # Check judge
    assert agents[4]["agent_name"] == "judge"
    assert agents[4]["agent_role"] == "judge"
    assert agents[4]["round_idx"] == 2

    # Check num_rounds
    assert graph["num_rounds"] == 3


def test_build_agent_graph_spatial_masks():
    """Test spatial masks (intra-round communication)."""
    graph = build_agent_graph(num_solvers=3)
    spatial_masks = graph["spatial_masks"]

    # Round 0: Solvers see only themselves
    mask_r0 = spatial_masks[0]
    assert mask_r0.shape == (5, 5)
    # Solver 0 sees only itself
    assert mask_r0[0, 0] == 1.0
    assert mask_r0[0, 1] == 0.0
    assert mask_r0[0, 2] == 0.0
    # Solver 1 sees only itself
    assert mask_r0[1, 1] == 1.0
    assert mask_r0[1, 0] == 0.0
    # Solver 2 sees only itself
    assert mask_r0[2, 2] == 1.0
    assert mask_r0[2, 0] == 0.0

    # Round 1: Verifier sees all solvers
    mask_r1 = spatial_masks[1]
    verifier_idx = 3
    assert mask_r1[verifier_idx, 0] == 1.0  # sees solver_0
    assert mask_r1[verifier_idx, 1] == 1.0  # sees solver_1
    assert mask_r1[verifier_idx, 2] == 1.0  # sees solver_2
    assert mask_r1[verifier_idx, verifier_idx] == 1.0  # sees itself

    # Round 2: Judge sees verifier
    mask_r2 = spatial_masks[2]
    judge_idx = 4
    assert mask_r2[judge_idx, verifier_idx] == 1.0  # sees verifier
    assert mask_r2[judge_idx, judge_idx] == 1.0  # sees itself


def test_build_agent_graph_temporal_masks():
    """Test temporal masks (cross-round dependencies)."""
    graph = build_agent_graph(num_solvers=3)
    temporal_masks = graph["temporal_masks"]

    # Round 0: No temporal dependencies
    mask_r0 = temporal_masks[0]
    assert mask_r0.shape == (5, 5)
    assert np.sum(mask_r0) == 0.0  # All zeros

    # Round 1: Verifier accesses Round 0 solvers
    mask_r1 = temporal_masks[1]
    verifier_idx = 3
    assert mask_r1[verifier_idx, 0] == 1.0  # accesses solver_0
    assert mask_r1[verifier_idx, 1] == 1.0  # accesses solver_1
    assert mask_r1[verifier_idx, 2] == 1.0  # accesses solver_2

    # Round 2: Judge accesses Round 1 verifier
    mask_r2 = temporal_masks[2]
    judge_idx = 4
    assert mask_r2[judge_idx, verifier_idx] == 1.0  # accesses verifier


def test_build_agent_graph_custom_solvers():
    """Test building agent graph with custom number of solvers."""
    graph = build_agent_graph(num_solvers=5)

    agents = graph["agents"]
    assert len(agents) == 7  # 5 solvers + 1 verifier + 1 judge

    # Check all solvers
    for i in range(5):
        assert agents[i]["agent_name"] == f"solver_{i}"
        assert agents[i]["agent_role"] == "solver"
        assert agents[i]["round_idx"] == 0

    # Verifier and judge positions adjusted
    verifier_idx = 5
    judge_idx = 6
    assert agents[verifier_idx]["agent_role"] == "verifier"
    assert agents[judge_idx]["agent_role"] == "judge"

    # Check spatial mask dimensions
    spatial_masks = graph["spatial_masks"]
    assert spatial_masks[0].shape == (7, 7)
    assert spatial_masks[1].shape == (7, 7)
    assert spatial_masks[2].shape == (7, 7)


def test_add_graph_metadata():
    """Test adding graph metadata to trajectory entry."""
    graph = build_agent_graph(num_solvers=3)

    # Create entry for verifier (should see all solvers)
    entry = TrajectoryEntry(
        timestamp="2024-01-01T00:00:00",
        run_id="test_run",
        step_id=3,
        agent="verifier",
        action="verify",
        input={"problem": "test"},
        output={"result": "verified"},
        metadata={
            "agent_name": "verifier",
            "agent_role": "verifier",
            "round_idx": 1,
        },
    )

    updated_entry = add_graph_metadata(entry, graph)

    # Check spatial predecessors (should see solvers 0, 1, 2 in same round)
    spatial_preds = updated_entry.metadata["spatial_predecessors"]
    assert sorted(spatial_preds) == [0, 1, 2]

    # Check temporal predecessors (should access solvers from previous round)
    temporal_preds = updated_entry.metadata["temporal_predecessors"]
    assert sorted(temporal_preds) == [0, 1, 2]


def test_add_graph_metadata_solver():
    """Test adding graph metadata for solver agent."""
    graph = build_agent_graph(num_solvers=3)

    # Create entry for solver_0
    entry = TrajectoryEntry(
        timestamp="2024-01-01T00:00:00",
        run_id="test_run",
        step_id=0,
        agent="solver_0",
        action="solve",
        input={"problem": "test"},
        output={"solution": "answer"},
        metadata={
            "agent_name": "solver_0",
            "agent_role": "solver",
            "round_idx": 0,
        },
    )

    updated_entry = add_graph_metadata(entry, graph)

    # Solver sees only itself spatially, no other agents
    spatial_preds = updated_entry.metadata["spatial_predecessors"]
    assert spatial_preds == []  # Excludes self

    # No temporal predecessors in round 0
    temporal_preds = updated_entry.metadata["temporal_predecessors"]
    assert temporal_preds == []


def test_export_to_marti_format():
    """Test exporting trajectory entries to MARTI format."""
    # Create sample trajectory entries
    entries = [
        TrajectoryEntry(
            timestamp="2024-01-01T00:00:00",
            run_id="test_run",
            step_id=0,
            agent="solver_0",
            action="solve",
            input={"problem": "What is 2+2?"},
            output={"solution": "4"},
            metadata={
                "agent_name": "solver_0",
                "agent_role": "solver",
                "round_idx": 0,
                "spatial_predecessors": [],
                "temporal_predecessors": [],
            },
            reward=1.0,
        ),
        TrajectoryEntry(
            timestamp="2024-01-01T00:00:01",
            run_id="test_run",
            step_id=1,
            agent="solver_1",
            action="solve",
            input={"problem": "What is 2+2?"},
            output={"solution": "4"},
            metadata={
                "agent_name": "solver_1",
                "agent_role": "solver",
                "round_idx": 0,
                "spatial_predecessors": [],
                "temporal_predecessors": [],
            },
            reward=1.0,
        ),
        TrajectoryEntry(
            timestamp="2024-01-01T00:00:02",
            run_id="test_run",
            step_id=2,
            agent="verifier",
            action="verify",
            input={"solutions": ["4", "4"]},
            output={"verified": True},
            metadata={
                "agent_name": "verifier",
                "agent_role": "verifier",
                "round_idx": 1,
                "spatial_predecessors": [0, 1],
                "temporal_predecessors": [0, 1],
            },
            reward=1.0,
        ),
    ]

    marti_traj = export_to_marti_format(
        entries=entries,
        problem="What is 2+2?",
        label="4",
    )

    # Check top-level structure
    assert isinstance(marti_traj, MARTITrajectory)
    assert marti_traj.problem == "What is 2+2?"
    assert marti_traj.label == "4"

    # Check nested list structure
    trajectory = marti_traj.trajectory
    assert len(trajectory) == 2  # 2 rounds (round 0 and round 1)

    # Round 0: 2 agents (solver_0, solver_1)
    round_0 = trajectory[0]
    assert len(round_0) == 2

    # Check solver_0 turns
    solver_0_turns = round_0[0]
    assert len(solver_0_turns) == 1
    turn_0 = solver_0_turns[0]
    assert turn_0.agent_name == "solver_0"
    assert turn_0.agent_role == "solver"
    assert turn_0.round_idx == 0
    assert turn_0.rewards == 1.0
    assert turn_0.spatial_predecessors == []
    assert turn_0.temporal_predecessors == []

    # Check solver_1 turns
    solver_1_turns = round_0[1]
    assert len(solver_1_turns) == 1
    turn_1 = solver_1_turns[0]
    assert turn_1.agent_name == "solver_1"
    assert turn_1.agent_role == "solver"

    # Round 1: 1 agent (verifier)
    round_1 = trajectory[1]
    assert len(round_1) == 1

    # Check verifier turns
    verifier_turns = round_1[0]
    assert len(verifier_turns) == 1
    verifier_turn = verifier_turns[0]
    assert verifier_turn.agent_name == "verifier"
    assert verifier_turn.agent_role == "verifier"
    assert verifier_turn.round_idx == 1
    assert verifier_turn.rewards == 1.0
    assert sorted(verifier_turn.spatial_predecessors) == [0, 1]
    assert sorted(verifier_turn.temporal_predecessors) == [0, 1]


def test_export_to_marti_format_empty():
    """Test exporting empty trajectory."""
    entries = []

    marti_traj = export_to_marti_format(
        entries=entries,
        problem="Empty problem",
        label="",
    )

    assert marti_traj.problem == "Empty problem"
    assert marti_traj.label == ""
    assert len(marti_traj.trajectory) == 0


def test_export_to_marti_format_single_agent():
    """Test exporting trajectory with single agent."""
    entries = [
        TrajectoryEntry(
            timestamp="2024-01-01T00:00:00",
            run_id="test_run",
            step_id=0,
            agent="solver_0",
            action="solve",
            input={"problem": "test"},
            output={"solution": "answer"},
            metadata={
                "agent_name": "solver_0",
                "agent_role": "solver",
                "round_idx": 0,
                "spatial_predecessors": [],
                "temporal_predecessors": [],
            },
            reward=0.5,
        ),
    ]

    marti_traj = export_to_marti_format(
        entries=entries,
        problem="test",
        label="answer",
    )

    trajectory = marti_traj.trajectory
    assert len(trajectory) == 1  # 1 round
    assert len(trajectory[0]) == 1  # 1 agent
    assert len(trajectory[0][0]) == 1  # 1 turn

    turn = trajectory[0][0][0]
    assert turn.agent_name == "solver_0"
    assert turn.rewards == 0.5


def test_export_to_marti_format_multiple_turns_per_agent():
    """Test exporting trajectory with multiple turns per agent."""
    entries = [
        TrajectoryEntry(
            timestamp="2024-01-01T00:00:00",
            run_id="test_run",
            step_id=0,
            agent="solver_0",
            action="solve",
            input={"problem": "test"},
            output={"solution": "first attempt"},
            metadata={
                "agent_name": "solver_0",
                "agent_role": "solver",
                "round_idx": 0,
                "spatial_predecessors": [],
                "temporal_predecessors": [],
            },
            reward=0.5,
        ),
        TrajectoryEntry(
            timestamp="2024-01-01T00:00:01",
            run_id="test_run",
            step_id=1,
            agent="solver_0",
            action="solve",
            input={"problem": "test"},
            output={"solution": "second attempt"},
            metadata={
                "agent_name": "solver_0",
                "agent_role": "solver",
                "round_idx": 0,
                "spatial_predecessors": [],
                "temporal_predecessors": [],
            },
            reward=1.0,
        ),
    ]

    marti_traj = export_to_marti_format(
        entries=entries,
        problem="test",
        label="answer",
    )

    trajectory = marti_traj.trajectory
    assert len(trajectory) == 1  # 1 round
    assert len(trajectory[0]) == 1  # 1 agent
    assert len(trajectory[0][0]) == 2  # 2 turns

    # Check turns are sorted by step_id
    turn_0 = trajectory[0][0][0]
    turn_1 = trajectory[0][0][1]
    assert turn_0.rewards == 0.5
    assert turn_1.rewards == 1.0
