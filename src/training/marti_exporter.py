"""MARTI-compatible trajectory export with agent graph structure.

Implements trajectory export following MARTI framework:
- Nested list structure: [[[agent1_turns], [agent2_turns]], ...]
- Agent graph with spatial/temporal connection masks
- Multi-agent solver-verifier-judge architecture
"""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

from src.models.trajectory import TrajectoryEntry


class AgentGraphNode(BaseModel):
    """Agent node in the multi-agent graph."""

    agent_name: str
    agent_role: str
    round_idx: int


class MARTITurn(BaseModel):
    """A single turn in MARTI trajectory format."""

    agent_name: str
    agent_role: str
    pretrain: bool = False
    round_idx: int
    inputs: list[dict[str, Any]]
    outputs: dict[str, Any]
    rewards: float
    spatial_predecessors: list[int]
    temporal_predecessors: list[int]


class MARTITrajectory(BaseModel):
    """MARTI trajectory with nested list structure."""

    problem: str
    label: str
    trajectory: list[list[list[MARTITurn]]]  # [rounds][agents][turns]


def build_debate_agent_graph() -> dict:
    """Build agent graph for 1-solver debate architecture (Phase 1 decision).

    Returns:
        Dict with agents, spatial_masks, temporal_masks, num_rounds
    """
    return build_agent_graph(num_solvers=1)


def build_agent_graph(num_solvers: int = 3) -> dict:
    """Build solver-verifier-judge agent graph with spatial/temporal masks.

    Args:
        num_solvers: Number of solver agents (default: 3)

    Returns:
        Dict with agents, spatial_masks, temporal_masks, num_rounds
    """
    # Define agents: solvers (round 0), verifier (round 1), judge (round 2)
    agents = []

    # Round 0: Solvers
    for i in range(num_solvers):
        agents.append({
            "agent_name": f"solver_{i}",
            "agent_role": "solver",
            "round_idx": 0,
        })

    # Round 1: Verifier
    agents.append({
        "agent_name": "verifier",
        "agent_role": "verifier",
        "round_idx": 1,
    })

    # Round 2: Judge
    agents.append({
        "agent_name": "judge",
        "agent_role": "judge",
        "round_idx": 2,
    })

    num_agents = len(agents)

    # Build spatial masks (intra-round communication)
    # spatial_masks[round][agent_i, agent_j] = 1 if agent_i can see agent_j in same round
    spatial_masks = {}

    # Round 0: Solvers see only themselves (independent)
    spatial_mask_r0 = np.zeros((num_agents, num_agents), dtype=np.float32)
    for i in range(num_solvers):
        spatial_mask_r0[i, i] = 1.0
    spatial_masks[0] = spatial_mask_r0

    # Round 1: Verifier sees all solvers
    spatial_mask_r1 = np.zeros((num_agents, num_agents), dtype=np.float32)
    verifier_idx = num_solvers
    for i in range(num_solvers):
        spatial_mask_r1[verifier_idx, i] = 1.0
    spatial_mask_r1[verifier_idx, verifier_idx] = 1.0
    spatial_masks[1] = spatial_mask_r1

    # Round 2: Judge sees verifier
    spatial_mask_r2 = np.zeros((num_agents, num_agents), dtype=np.float32)
    judge_idx = num_solvers + 1
    spatial_mask_r2[judge_idx, verifier_idx] = 1.0
    spatial_mask_r2[judge_idx, judge_idx] = 1.0
    spatial_masks[2] = spatial_mask_r2

    # Build temporal masks (cross-round dependencies)
    # temporal_masks[round][agent_i, agent_j] = 1 if agent_i (current) can access agent_j (previous rounds)
    temporal_masks = {}

    # Round 0: No temporal dependencies (first round)
    temporal_masks[0] = np.zeros((num_agents, num_agents), dtype=np.float32)

    # Round 1: Verifier accesses all Round 0 solvers
    temporal_mask_r1 = np.zeros((num_agents, num_agents), dtype=np.float32)
    for i in range(num_solvers):
        temporal_mask_r1[verifier_idx, i] = 1.0
    temporal_masks[1] = temporal_mask_r1

    # Round 2: Judge accesses Round 1 verifier
    temporal_mask_r2 = np.zeros((num_agents, num_agents), dtype=np.float32)
    temporal_mask_r2[judge_idx, verifier_idx] = 1.0
    temporal_masks[2] = temporal_mask_r2

    return {
        "agents": agents,
        "spatial_masks": spatial_masks,
        "temporal_masks": temporal_masks,
        "num_rounds": 3,
    }


def add_graph_metadata(entry: TrajectoryEntry, agent_graph: dict) -> TrajectoryEntry:
    """Add spatial/temporal predecessors to trajectory entry metadata.

    Args:
        entry: TrajectoryEntry to annotate
        agent_graph: Agent graph from build_agent_graph()

    Returns:
        TrajectoryEntry with updated metadata
    """
    # Extract agent info from entry metadata
    agent_name = entry.metadata.get("agent_name", entry.agent)
    round_idx = entry.metadata.get("round_idx", 0)

    # Find agent index in graph
    agents = agent_graph["agents"]
    agent_idx = None
    for i, agent in enumerate(agents):
        if agent["agent_name"] == agent_name and agent["round_idx"] == round_idx:
            agent_idx = i
            break

    if agent_idx is None:
        # Agent not found in graph, return entry unchanged
        return entry

    # Get spatial predecessors (same round)
    spatial_mask = agent_graph["spatial_masks"].get(round_idx, np.array([]))
    if spatial_mask.size > 0:
        spatial_predecessors = [
            int(j) for j in range(len(agents))
            if spatial_mask[agent_idx, j] > 0 and j != agent_idx
        ]
    else:
        spatial_predecessors = []

    # Get temporal predecessors (previous rounds)
    temporal_mask = agent_graph["temporal_masks"].get(round_idx, np.array([]))
    if temporal_mask.size > 0:
        temporal_predecessors = [
            int(j) for j in range(len(agents))
            if temporal_mask[agent_idx, j] > 0
        ]
    else:
        temporal_predecessors = []

    # Update entry metadata
    entry.metadata["spatial_predecessors"] = spatial_predecessors
    entry.metadata["temporal_predecessors"] = temporal_predecessors

    return entry


def export_to_marti_format(
    entries: list[TrajectoryEntry],
    problem: str,
    label: str,
    mode: str = "debate",
) -> MARTITrajectory:
    """Convert pipeline TrajectoryEntry list to MARTI nested list format.

    Args:
        entries: List of TrajectoryEntry objects
        problem: Problem text
        label: Ground truth label/answer
        mode: Pipeline mode - "debate" for 1-solver, "baseline" for 3-solver (default: "debate")

    Returns:
        MARTITrajectory with nested list structure [rounds][agents][turns]
    """
    # Build agent graph based on mode
    if mode == "debate":
        agent_graph = build_debate_agent_graph()  # num_solvers=1
    else:
        agent_graph = build_agent_graph()  # default num_solvers=3

    # Group entries by (round_idx, agent_name)
    # Structure: {round_idx: {agent_name: [entries]}}
    grouped = defaultdict(lambda: defaultdict(list))

    for entry in entries:
        round_idx = entry.metadata.get("round_idx", 0)
        agent_name = entry.metadata.get("agent_name", entry.agent)
        grouped[round_idx][agent_name].append(entry)

    # Build nested list structure
    trajectory_nested = []

    # Sort rounds
    sorted_rounds = sorted(grouped.keys())

    for round_idx in sorted_rounds:
        agents_in_round = []

        # Sort agents within round
        sorted_agents = sorted(grouped[round_idx].keys())

        for agent_name in sorted_agents:
            turns_for_agent = []

            # Sort turns by step_id
            entries_for_agent = sorted(
                grouped[round_idx][agent_name],
                key=lambda e: e.step_id
            )

            for entry in entries_for_agent:
                # Extract agent role
                agent_role = entry.metadata.get("agent_role", "unknown")

                # Build inputs list (from entry.input dict)
                inputs = [entry.input] if entry.input else []

                # Build outputs dict
                outputs = {"output": entry.output} if entry.output else {"output": {}}

                # Get reward
                rewards = entry.reward if entry.reward is not None else 0.0

                # Get predecessors from metadata
                spatial_predecessors = entry.metadata.get("spatial_predecessors", [])
                temporal_predecessors = entry.metadata.get("temporal_predecessors", [])

                # Create MARTI turn
                turn = MARTITurn(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    pretrain=False,
                    round_idx=round_idx,
                    inputs=inputs,
                    outputs=outputs,
                    rewards=rewards,
                    spatial_predecessors=spatial_predecessors,
                    temporal_predecessors=temporal_predecessors,
                )

                turns_for_agent.append(turn)

            agents_in_round.append(turns_for_agent)

        trajectory_nested.append(agents_in_round)

    return MARTITrajectory(
        problem=problem,
        label=label,
        trajectory=trajectory_nested,
    )
