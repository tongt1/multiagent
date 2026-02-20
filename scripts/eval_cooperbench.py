#!/usr/bin/env python3
"""CLI for running CooperBench evaluations on model checkpoints.

Evaluates a model checkpoint on the held-out CooperBench task set using the
same agentic OpenHands loop as training. Computes pass@1/3/5, partial credit,
agent turns, rollout time, and token count. Optionally logs results to W&B.

Usage:
    # Dry-run: show eval plan without executing
    python scripts/eval_cooperbench.py \\
        --model-path /path/to/checkpoint \\
        --seed 42 --n-rollouts 10 --dry-run

    # Full evaluation with W&B logging
    python scripts/eval_cooperbench.py \\
        --model-path /path/to/checkpoint \\
        --seed 42 --n-rollouts 10 \\
        --wandb-project multiagent-debate-rl \\
        --wandb-run-id abc123

The ``run_cooperbench_eval`` function is also importable for use as a
post-training hook in sweep configs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Ensure reward-training root is on path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.cooperbench.task_split import (
    LITE_V4_TASK_IDS,
    get_train_eval_split,
)
from src.evaluation.cooperbench.inspect_eval import (
    compute_passk,
    cooperbench_eval,
)


def run_cooperbench_eval(
    model_path: str,
    seed: int = 42,
    n_rollouts: int = 10,
    vllm_base_url: str = "http://localhost:8000/v1",
    wandb_run_id: str | None = None,
    wandb_project: str = "multiagent-debate-rl",
    output_dir: str | None = None,
    dataset_path: str | None = None,
    tasks: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run CooperBench evaluation on a model checkpoint.

    This function is both the CLI entry point and an importable hook for
    post-training integration in sweep configs.

    Args:
        model_path: Path to model checkpoint or HF export directory.
        seed: Seed for task split (determines which tasks are held out).
        n_rollouts: Number of independent rollouts per held-out task.
        vllm_base_url: vLLM sidecar URL for model inference.
        wandb_run_id: Optional W&B run ID to attach results to.
        wandb_project: W&B project name for logging.
        output_dir: Directory to save JSON results.
        dataset_path: Path to CooperBench dataset on disk.
        tasks: Optional explicit task IDs (overrides split-based selection).
        dry_run: If True, print eval plan and return without executing.

    Returns:
        Results dict with overall metrics and per-task breakdown.
    """
    # Determine eval task set
    if tasks:
        eval_task_ids = tasks
        train_task_ids = [t for t in LITE_V4_TASK_IDS if t not in tasks]
    else:
        train_task_ids, eval_task_ids = get_train_eval_split(
            LITE_V4_TASK_IDS, seed=seed
        )

    # Print eval plan
    print("=" * 60)
    print("CooperBench Evaluation Plan")
    print("=" * 60)
    print(f"  Model path:    {model_path}")
    print(f"  Seed:          {seed}")
    print(f"  N rollouts:    {n_rollouts}")
    print(f"  vLLM URL:      {vllm_base_url}")
    print(f"  Total tasks:   {len(LITE_V4_TASK_IDS)}")
    print(f"  Train tasks:   {len(train_task_ids)}")
    print(f"  Eval tasks:    {len(eval_task_ids)}")
    print(f"  Total samples: {len(eval_task_ids) * n_rollouts}")
    print()
    print("Held-out eval tasks:")
    for tid in eval_task_ids:
        print(f"  - {tid}")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN] Would evaluate the above tasks. Exiting.")
        return {
            "model_path": model_path,
            "seed": seed,
            "n_rollouts": n_rollouts,
            "dry_run": True,
            "eval_task_ids": eval_task_ids,
            "n_eval_tasks": len(eval_task_ids),
            "n_total_samples": len(eval_task_ids) * n_rollouts,
        }

    # Run evaluation
    results = cooperbench_eval(
        model_path=model_path,
        task_ids=eval_task_ids,
        n_rollouts=n_rollouts,
        vllm_base_url=vllm_base_url,
        dataset_path=dataset_path,
    )

    # Add metadata
    results["seed"] = seed
    results["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Save results JSON
    if output_dir is None:
        output_dir = str(Path(model_path).parent / "eval_results")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"cooperbench_eval_seed{seed}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Log to W&B if configured
    if wandb_run_id:
        _log_to_wandb(
            results=results,
            wandb_project=wandb_project,
            wandb_run_id=wandb_run_id,
            seed=seed,
        )

    # Print summary table
    _print_summary(results)

    return results


def _log_to_wandb(
    results: dict[str, Any],
    wandb_project: str,
    wandb_run_id: str,
    seed: int,
) -> None:
    """Log evaluation results to Weights & Biases.

    Creates a W&B Table with per-task breakdown and logs overall
    metrics as summary values.

    Args:
        results: Evaluation results dict.
        wandb_project: W&B project name.
        wandb_run_id: W&B run ID to attach to.
        seed: Seed number for metric namespacing.
    """
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed, skipping W&B logging")
        return

    try:
        # Resume existing run
        run = wandb.init(
            project=wandb_project,
            id=wandb_run_id,
            resume="allow",
        )

        if run is None:
            print("WARNING: wandb.init returned None, skipping W&B logging")
            return

        # Log overall metrics
        overall = results.get("overall", {})
        prefix = f"eval/cooperbench/seed{seed}"
        run.summary[f"{prefix}/pass_at_1"] = overall.get("pass_at_1", 0)
        run.summary[f"{prefix}/pass_at_3"] = overall.get("pass_at_3", 0)
        run.summary[f"{prefix}/pass_at_5"] = overall.get("pass_at_5", 0)
        run.summary[f"{prefix}/partial_credit"] = overall.get("partial_credit", 0)
        run.summary[f"{prefix}/mean_agent_turns"] = overall.get("mean_agent_turns", 0)
        run.summary[f"{prefix}/mean_rollout_time_s"] = overall.get("mean_rollout_time_s", 0)
        run.summary[f"{prefix}/mean_token_count"] = overall.get("mean_token_count", 0)

        # Log per-task table
        columns = [
            "task_id", "pass_at_1", "pass_at_3", "pass_at_5",
            "partial_credit", "mean_agent_turns", "mean_rollout_time_s",
        ]
        table = wandb.Table(columns=columns)
        for task in results.get("per_task", []):
            table.add_data(
                task["task_id"],
                task.get("pass_at_1", 0),
                task.get("pass_at_3", 0),
                task.get("pass_at_5", 0),
                task.get("partial_credit", 0),
                task.get("mean_agent_turns", 0),
                task.get("mean_rollout_time_s", 0),
            )
        run.log({f"{prefix}/per_task": table})

        # Save as artifact
        artifact = wandb.Artifact(
            name=f"cooperbench-eval-seed{seed}",
            type="eval-results",
            description=f"CooperBench evaluation results for seed {seed}",
        )
        artifact.add(table, f"per_task_seed{seed}")
        run.log_artifact(artifact)

        run.finish()
        print(f"Results logged to W&B run {wandb_run_id}")

    except Exception as e:
        print(f"WARNING: Failed to log to W&B: {e}")


def _print_summary(results: dict[str, Any]) -> None:
    """Print a summary table of evaluation results.

    Args:
        results: Evaluation results dict.
    """
    overall = results.get("overall", {})
    print()
    print("=" * 50)
    print("Evaluation Results Summary")
    print("=" * 50)
    print(f"  pass@1:          {overall.get('pass_at_1', 0):.4f}")
    print(f"  pass@3:          {overall.get('pass_at_3', 0):.4f}")
    print(f"  pass@5:          {overall.get('pass_at_5', 0):.4f}")
    print(f"  partial credit:  {overall.get('partial_credit', 0):.4f}")
    print(f"  mean turns:      {overall.get('mean_agent_turns', 0):.1f}")
    print(f"  mean time (s):   {overall.get('mean_rollout_time_s', 0):.1f}")
    print(f"  mean tokens:     {overall.get('mean_token_count', 0):.0f}")
    print("=" * 50)

    per_task = results.get("per_task", [])
    if per_task:
        print(f"\nPer-task breakdown ({len(per_task)} tasks):")
        print(f"  {'Task ID':<45} {'pass@1':>7} {'pass@3':>7} {'pass@5':>7}")
        print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7}")
        for task in per_task:
            tid = task["task_id"]
            p1 = task.get("pass_at_1", 0)
            p3 = task.get("pass_at_3", 0)
            p5 = task.get("pass_at_5", 0)
            print(f"  {tid:<45} {p1:>7.3f} {p3:>7.3f} {p5:>7.3f}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run CooperBench evaluation on a model checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to model checkpoint or HF export directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for task split (default: 42)",
    )
    parser.add_argument(
        "--n-rollouts", type=int, default=10,
        help="Number of rollouts per held-out task (default: 10)",
    )
    parser.add_argument(
        "--vllm-base-url", default="http://localhost:8000/v1",
        help="vLLM sidecar URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--wandb-project", default="multiagent-debate-rl",
        help="W&B project name (default: multiagent-debate-rl)",
    )
    parser.add_argument(
        "--wandb-run-id", default=None,
        help="Attach results to existing W&B run",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for JSON results (default: next to model-path)",
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help="Path to CooperBench dataset on disk",
    )
    parser.add_argument(
        "--tasks", default=None,
        help="Comma-separated task IDs (overrides split-based selection)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print eval plan without executing",
    )

    args = parser.parse_args()

    task_list = None
    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",")]

    run_cooperbench_eval(
        model_path=args.model_path,
        seed=args.seed,
        n_rollouts=args.n_rollouts,
        vllm_base_url=args.vllm_base_url,
        wandb_run_id=args.wandb_run_id,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        tasks=task_list,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
