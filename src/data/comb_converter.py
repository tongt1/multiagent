"""Convert Phase 1 trajectory JSONL to Comb-compatible CommandDataCombItem format.

This converter transforms trajectory data from Phase 1 (debate/baseline runs) into
the JSONL format expected by Flink's CombLineEncoder for GRPO training.

Phase 1 format: Each JSONL line is a step with fields like run_id, step_id, agent, etc.
Target Comb format: Each JSONL line is a CommandDataCombItem with comb_env_name,
                    agent_trajectory, validator_annotation.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def convert_single_trajectory(run_id: str, steps: list[dict], mode: str) -> dict:
    """Convert a single trajectory (grouped steps by run_id) to Comb format.

    Args:
        run_id: Unique identifier for this trajectory
        steps: List of trajectory step dicts (all with same run_id)
        mode: "debate" or "baseline" - determines comb_env_name

    Returns:
        Dict in CommandDataCombItem format
    """
    # Sort steps by step_id to ensure correct order
    steps_sorted = sorted(steps, key=lambda x: x.get("step_id", 0))

    if not steps_sorted:
        raise ValueError(f"Empty steps list for run_id {run_id}")

    # Extract problem text from first step's input
    first_step = steps_sorted[0]
    first_input = first_step.get("input", {})
    problem_text = first_input.get("problem", "")

    if not problem_text:
        raise ValueError(f"Missing problem text in first step for run_id {run_id}")

    # Extract ground truth from first step's metadata or input
    metadata = first_step.get("metadata", {})
    ground_truth = metadata.get("ground_truth") or first_input.get("ground_truth", "")

    if not ground_truth:
        raise ValueError(f"Missing ground_truth for run_id {run_id}")

    # Set comb_env_name based on mode
    # CRITICAL: "math_debate" for debate (custom env from Plan 02-02)
    #           "math" for baseline (existing Comb env)
    if mode == "debate":
        comb_env_name = "math_debate"
    elif mode == "baseline":
        comb_env_name = "math"
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'debate' or 'baseline'")

    # Build CommandDataCombItem structure
    # Following schema from comb.interface.build_types.CommandDataCombItem
    comb_item = {
        "comb_env_name": comb_env_name,
        "agent_trajectory": {
            "turns": [
                {
                    "role": "user",
                    "contents": [{"type": "text", "text": problem_text}],
                }
            ],
            "preamble": None,  # Will be patched by CombItemsPreprocessor
        },
        "validator_annotation": {
            "spec": {
                "arguments": {
                    "gold_answer": ground_truth,
                }
            }
        },
        "custom_data": None,
    }

    return comb_item


def convert_trajectories_to_comb(
    input_path: Path, output_path: Path, mode: str = "debate"
) -> int:
    """Convert Phase 1 trajectory JSONL to Comb-compatible JSONL.

    Args:
        input_path: Path to Phase 1 trajectory JSONL file
        output_path: Path to output Comb-format JSONL file
        mode: "debate" or "baseline" (default: "debate")

    Returns:
        Number of trajectories converted
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load all steps from input JSONL
    steps = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    if not steps:
        return 0

    # Group steps by run_id
    groups: dict[str, list[dict]] = {}
    for step in steps:
        run_id = step.get("run_id")
        if run_id:
            if run_id not in groups:
                groups[run_id] = []
            groups[run_id].append(step)

    # Detect mode from trajectory metadata if not specified
    # (Use first step's metadata.mode if available)
    detected_mode = mode
    if groups and mode == "debate":  # Only auto-detect if using default
        first_run_id = next(iter(groups))
        first_steps = groups[first_run_id]
        if first_steps:
            first_metadata = first_steps[0].get("metadata", {})
            if "mode" in first_metadata:
                detected_mode = first_metadata["mode"]

    # Convert each trajectory
    comb_items = []
    for run_id, run_steps in groups.items():
        try:
            comb_item = convert_single_trajectory(run_id, run_steps, detected_mode)
            comb_items.append(comb_item)
        except ValueError as e:
            # Skip malformed trajectories with warning
            print(f"Warning: Skipping trajectory {run_id}: {e}")
            continue

    # Write output JSONL (one line per trajectory, not per step)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in comb_items:
            f.write(json.dumps(item) + "\n")

    return len(comb_items)


def convert_batch(input_dir: Path, output_path: Path) -> int:
    """Convert all JSONL files in a directory to single Comb JSONL output.

    Args:
        input_dir: Directory containing Phase 1 trajectory JSONL files
        output_path: Path to output Comb-format JSONL file

    Returns:
        Total number of trajectories converted
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    # Collect all JSONL files
    jsonl_files = list(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        return 0

    # Process each file and accumulate all Comb items
    all_comb_items = []

    for jsonl_file in jsonl_files:
        # Detect mode from filename or file content
        mode = "debate" if "debate" in jsonl_file.name else "baseline"

        # Load steps from file
        steps = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    steps.append(json.loads(line))

        if not steps:
            continue

        # Group by run_id
        groups: dict[str, list[dict]] = {}
        for step in steps:
            run_id = step.get("run_id")
            if run_id:
                if run_id not in groups:
                    groups[run_id] = []
                groups[run_id].append(step)

        # Convert each trajectory
        for run_id, run_steps in groups.items():
            # Detect mode from metadata if available
            detected_mode = mode
            if run_steps:
                first_metadata = run_steps[0].get("metadata", {})
                if "mode" in first_metadata:
                    detected_mode = first_metadata["mode"]

            try:
                comb_item = convert_single_trajectory(run_id, run_steps, detected_mode)
                all_comb_items.append(comb_item)
            except ValueError as e:
                print(f"Warning: Skipping trajectory {run_id} from {jsonl_file.name}: {e}")
                continue

    # Write all items to single output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_comb_items:
            f.write(json.dumps(item) + "\n")

    return len(all_comb_items)


def main():
    """CLI entrypoint for converter."""
    parser = argparse.ArgumentParser(
        description="Convert Phase 1 trajectory JSONL to Comb-compatible format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input trajectory JSONL file or directory",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output Comb-format JSONL file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debate", "baseline"],
        default="debate",
        help="Trajectory mode (default: debate)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat input as directory and process all JSONL files",
    )

    args = parser.parse_args()

    if args.batch:
        count = convert_batch(args.input, args.output)
    else:
        count = convert_trajectories_to_comb(args.input, args.output, args.mode)

    print(f"Converted {count} trajectories to {args.output}")


if __name__ == "__main__":
    main()
