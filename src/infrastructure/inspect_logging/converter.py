"""Trajectory-to-Inspect conversion orchestrator.

Reads trajectory JSONL files, builds EvalSamples from run_id groups,
assembles an EvalLog, and writes the result as a .eval file.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai.log import write_eval_log
from loguru import logger

from .log_builder import build_eval_log
from .sample_builder import build_baseline_sample
from .trajectory_reader import read_and_group_trajectories


def convert_trajectories(
    jsonl_path: Path,
    output_dir: Path,
    task_name: str = "baseline_eval",
    model_name: str | None = None,
    system_prompt: str = "",
) -> Path:
    """Convert a trajectory JSONL file to an Inspect .eval file.

    Reads and groups trajectory entries by run_id, builds an EvalSample
    for each group, assembles a complete EvalLog, and writes a .eval file.

    One JSONL file produces one .eval file (Phase 1 heuristic).

    Args:
        jsonl_path: Path to trajectory JSONL file.
        output_dir: Directory to write the .eval file.
        task_name: Task name for EvalSpec.
        model_name: Model name override. If None, auto-detect from first solver entry.
        system_prompt: Optional system prompt to include in messages.

    Returns:
        Path to the written .eval file.
    """
    # Read and group trajectories
    groups = read_and_group_trajectories(jsonl_path)
    if not groups:
        raise ValueError(f"No valid trajectory entries found in {jsonl_path}")

    # Auto-detect model name from first solver entry if not provided
    if model_name is None:
        for entries in groups.values():
            for entry in entries:
                if "solver" in entry.agent:
                    model_name = entry.metadata.get("model", "unknown")
                    break
            if model_name is not None:
                break
        if model_name is None:
            model_name = "unknown"

    # Build EvalSample for each run_id group
    samples = []
    for idx, (run_id, entries) in enumerate(groups.items(), 1):
        try:
            sample = build_baseline_sample(
                entries=entries,
                sample_id=idx,
                system_prompt=system_prompt,
            )
            samples.append(sample)
        except ValueError as e:
            logger.warning(f"Skipping run_id {run_id}: {e}")

    if not samples:
        raise ValueError(f"No valid samples could be built from {jsonl_path}")

    # Assemble EvalLog
    log = build_eval_log(
        samples=samples,
        task_name=task_name,
        model_name=model_name,
        dataset_name=jsonl_path.stem,
    )

    # Write .eval file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_{jsonl_path.stem}.eval"
    write_eval_log(log, location=str(output_path), format="eval")

    # Log summary
    scorer_names = [s.name for s in log.results.scores] if log.results and log.results.scores else []
    logger.info(
        f"Converted {len(samples)} samples to {output_path} "
        f"(scorers: {scorer_names})"
    )

    return output_path
