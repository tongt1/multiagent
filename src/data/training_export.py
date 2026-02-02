"""Export trajectory data for RL training."""

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel


class TrainingTurn(BaseModel):
    """A single turn in a training trajectory."""

    turn_id: int
    agent: str
    action: str
    input: dict
    output: dict
    reward: float = 0.0
    timestamp: str


class TrainingTrajectory(BaseModel):
    """A complete training trajectory."""

    id: str
    problem: str
    domain: str = "general"
    turns: list
    total_reward: float
    success: Optional[bool] = None
    metadata: dict = {}


def load_trajectory_entries(trajectory_path: Path) -> list:
    """Load trajectory entries from JSONL file.

    Args:
        trajectory_path: Path to JSONL trajectory file

    Returns:
        List of trajectory entry dicts
    """
    entries = []

    if not trajectory_path.exists():
        return entries

    try:
        with open(trajectory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception:
        # Handle empty or malformed files gracefully
        pass

    return entries


def group_by_run_id(entries: list) -> dict:
    """Group trajectory entries by run_id.

    Args:
        entries: List of trajectory entry dicts

    Returns:
        Dict mapping run_id to list of entries
    """
    groups = {}

    for entry in entries:
        run_id = entry.get("run_id")
        if run_id:
            if run_id not in groups:
                groups[run_id] = []
            groups[run_id].append(entry)

    return groups


def convert_to_training_trajectory(
    run_id: str, steps: list, problem: str = "", domain: str = "general"
) -> TrainingTrajectory:
    """Convert raw trajectory steps to TrainingTrajectory format.

    Args:
        run_id: Unique identifier for this trajectory
        steps: List of trajectory entry dicts
        problem: Problem text (optional)
        domain: Problem domain (default: "general")

    Returns:
        TrainingTrajectory with structured turns
    """
    # Sort steps by step_id to ensure correct order
    steps_sorted = sorted(steps, key=lambda x: x.get("step_id", 0))

    # Convert each step to a training turn
    turns = []
    total_reward = 0.0
    success = None

    for i, step in enumerate(steps_sorted):
        turn = TrainingTurn(
            turn_id=step.get("step_id", i),
            agent=step.get("agent", "unknown"),
            action=step.get("action", "unknown"),
            input=step.get("input", {}),
            output=step.get("output", {}),
            reward=step.get("reward", 0.0),
            timestamp=step.get("timestamp", ""),
        )
        turns.append(turn)
        total_reward += turn.reward

        # Track success from last step if available
        if "success" in step and step["success"] is not None:
            success = step["success"]

    # Extract problem from first step if not provided
    if not problem and steps_sorted:
        first_input = steps_sorted[0].get("input", {})
        problem = first_input.get("problem", first_input.get("text", ""))

    # Extract domain from metadata if available and not provided
    if domain == "general" and steps_sorted:
        metadata = steps_sorted[0].get("metadata", {})
        domain = metadata.get("domain", "general")

    trajectory = TrainingTrajectory(
        id=run_id,
        problem=problem,
        domain=domain,
        turns=turns,
        total_reward=total_reward,
        success=success,
        metadata={
            "num_steps": len(turns),
            "run_id": run_id,
        },
    )

    return trajectory


def export_trajectory_for_training(
    trajectory_path: Path,
    output_path: Path,
    problem_map: Optional[dict] = None,
) -> int:
    """Export trajectory file to training-structured JSONL format.

    Args:
        trajectory_path: Path to input trajectory JSONL file
        output_path: Path to output training JSONL file
        problem_map: Optional dict mapping run_id to problem info (text, domain)

    Returns:
        Number of trajectories exported
    """
    # Load entries
    entries = load_trajectory_entries(trajectory_path)

    if not entries:
        return 0

    # Group by run_id
    groups = group_by_run_id(entries)

    # Convert each group to training trajectory
    trajectories = []

    for run_id, steps in groups.items():
        # Get problem info from map if provided
        problem = ""
        domain = "general"

        if problem_map and run_id in problem_map:
            info = problem_map[run_id]
            problem = info.get("problem", info.get("text", ""))
            domain = info.get("domain", "general")

        trajectory = convert_to_training_trajectory(
            run_id=run_id, steps=steps, problem=problem, domain=domain
        )
        trajectories.append(trajectory)

    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(traj.model_dump_json() + "\n")

    return len(trajectories)


def export_batch_trajectories(
    trajectory_dir: Path,
    output_path: Path,
    problem_map: Optional[dict] = None,
) -> int:
    """Export all trajectory files in a directory to training format.

    Args:
        trajectory_dir: Directory containing trajectory JSONL files
        output_path: Path to output training JSONL file
        problem_map: Optional dict mapping run_id to problem info

    Returns:
        Total number of trajectories exported
    """
    if not trajectory_dir.exists() or not trajectory_dir.is_dir():
        return 0

    # Collect all JSONL files
    trajectory_files = list(trajectory_dir.glob("*.jsonl"))

    if not trajectory_files:
        return 0

    # Process each file and accumulate trajectories
    all_trajectories = []

    for traj_file in trajectory_files:
        entries = load_trajectory_entries(traj_file)
        if not entries:
            continue

        groups = group_by_run_id(entries)

        for run_id, steps in groups.items():
            problem = ""
            domain = "general"

            if problem_map and run_id in problem_map:
                info = problem_map[run_id]
                problem = info.get("problem", info.get("text", ""))
                domain = info.get("domain", "general")

            trajectory = convert_to_training_trajectory(
                run_id=run_id, steps=steps, problem=problem, domain=domain
            )
            all_trajectories.append(trajectory)

    # Write all trajectories to output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for traj in all_trajectories:
            f.write(traj.model_dump_json() + "\n")

    return len(all_trajectories)
