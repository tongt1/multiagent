"""Generate sample MARTI trajectory data for training validation.

Creates synthetic but realistic trajectory entries matching the format
produced by the inference pipeline (TrajectoryEntry model).

Usage:
    python scripts/generate_trajectories.py [--output data/sample_trajectories.jsonl] [--count 50]
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path


MATH_PROBLEMS = [
    ("What is the sum of all integers from 1 to 100?", "5050"),
    ("Solve for x: 2x + 5 = 17", "x = 6"),
    ("What is the derivative of x^3 + 2x^2 - 5x + 1?", "3x^2 + 4x - 5"),
    ("Find the area of a circle with radius 7.", "49*pi ≈ 153.94"),
    ("What is 15! / 13!?", "210"),
    ("Simplify: (3x^2 - 12) / (x - 2)", "3(x + 2) = 3x + 6"),
    ("What is the integral of 2x dx from 0 to 3?", "9"),
    ("Solve the system: x + y = 10, x - y = 4", "x = 7, y = 3"),
    ("What is the GCD of 84 and 126?", "42"),
    ("Find the 10th term of the arithmetic sequence 3, 7, 11, ...", "39"),
    ("What is log_2(256)?", "8"),
    ("Simplify: sqrt(50) + sqrt(18)", "8*sqrt(2)"),
    ("How many ways can you arrange 5 books on a shelf?", "120"),
    ("What is the probability of rolling a sum of 7 with two dice?", "6/36 = 1/6"),
    ("Solve: x^2 - 5x + 6 = 0", "x = 2 or x = 3"),
    ("What is the dot product of [1,2,3] and [4,5,6]?", "32"),
    ("Convert 255 from decimal to binary.", "11111111"),
    ("What is the limit of (sin x)/x as x approaches 0?", "1"),
    ("Find the median of: 3, 7, 1, 9, 5, 2, 8", "5"),
    ("What is the determinant of [[1,2],[3,4]]?", "-2"),
]

AGENTS = ["solver_0", "solver_1", "solver_2", "verifier", "judge"]
AGENT_ROLES = {
    "solver_0": "solver",
    "solver_1": "solver",
    "solver_2": "solver",
    "verifier": "verifier",
    "judge": "judge",
}


def generate_entry(
    problem: str,
    answer: str,
    agent: str,
    run_id: str,
    step_id: int,
    round_idx: int,
) -> dict:
    """Generate a single trajectory entry."""
    role = AGENT_ROLES[agent]

    if role == "solver":
        # Solver proposes a solution (sometimes wrong)
        correct = random.random() < 0.7
        if correct:
            solution = answer
            reward = 1.0
        else:
            # Generate a plausible but wrong answer
            solution = f"Approximately {random.randint(1, 100)}"
            reward = 0.0
        action = "generate_solution"
        output = {"solution": solution}

    elif role == "verifier":
        # Verifier checks the solution
        agrees = random.random() < 0.6
        feedback = "The solution is correct." if agrees else "The solution contains errors. Let me recalculate."
        reward = 0.8 if agrees else 0.3
        action = "verify_solution"
        output = {"feedback": feedback, "agrees": agrees}

    elif role == "judge":
        # Judge assigns final score
        score = round(random.uniform(0.3, 1.0), 2)
        reward = score
        action = "judge_solution"
        output = {"score": score, "reasoning": f"Based on verification, confidence={score:.2f}"}

    else:
        reward = 0.5
        action = "unknown"
        output = {"result": "unknown action"}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "step_id": step_id,
        "agent": agent,
        "action": action,
        "input": {"problem": problem, "prompt": f"Solve: {problem}"},
        "output": output,
        "metadata": {
            "model_version": "1.0",
            "config_hash": "sample_" + run_id[:8],
            "tokens": random.randint(50, 500),
            "cost_usd": round(random.uniform(0.001, 0.01), 4),
            "agent_name": agent,
            "agent_role": role,
            "round_idx": round_idx,
        },
        "reward": reward,
        "terminal": role == "judge",
        "success": reward > 0.5 if role == "judge" else None,
    }


def generate_trajectories(count: int) -> list[dict]:
    """Generate a set of trajectory entries."""
    entries = []

    for i in range(count):
        problem, answer = random.choice(MATH_PROBLEMS)
        run_id = str(uuid.uuid4())
        step_id = 0

        # Round 0: Solvers (independent)
        for solver in ["solver_0", "solver_1", "solver_2"]:
            entry = generate_entry(problem, answer, solver, run_id, step_id, round_idx=0)
            entries.append(entry)
            step_id += 1

        # Round 1: Verifier
        entry = generate_entry(problem, answer, "verifier", run_id, step_id, round_idx=1)
        entries.append(entry)
        step_id += 1

        # Round 2: Judge
        entry = generate_entry(problem, answer, "judge", run_id, step_id, round_idx=2)
        entries.append(entry)

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample trajectory data")
    parser.add_argument("--output", default="data/sample_trajectories.jsonl")
    parser.add_argument("--count", type=int, default=50, help="Number of problem runs")
    args = parser.parse_args()

    entries = generate_trajectories(args.count)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {len(entries)} trajectory entries ({args.count} problems × 5 agents)")
    print(f"Written to: {output_path}")

    # Stats
    rewards = [e["reward"] for e in entries]
    agents = set(e["agent"] for e in entries)
    print(f"Agents: {sorted(agents)}")
    print(f"Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    print(f"Mean reward: {sum(rewards) / len(rewards):.4f}")


if __name__ == "__main__":
    main()
