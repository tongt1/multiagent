#!/usr/bin/env python3
"""Checkpoint evaluation CLI script and importable module.

Runs BEE MATH evaluation on model checkpoints, producing per-difficulty
accuracy breakdowns and optional W&B artifact logging.

Usage:
    # Evaluate all checkpoints in an experiment (debate mode)
    python scripts/evaluate_checkpoints.py --experiment-dir experiments/20260202_120000 --mode debate

    # Evaluate both debate and baseline
    python scripts/evaluate_checkpoints.py --experiment-dir experiments/20260202_120000 --mode both

    # Evaluate with W&B logging
    python scripts/evaluate_checkpoints.py --experiment-dir experiments/20260202_120000 --wandb-run-id abc123

    # Dry run (preview what will be evaluated)
    python scripts/evaluate_checkpoints.py --experiment-dir experiments/20260202_120000 --dry-run

    # Evaluate single checkpoint
    python scripts/evaluate_checkpoints.py --checkpoint experiments/.../ckpt-5 --eval-data data/math500_cache.json

As importable module:
    from scripts.evaluate_checkpoints import run_evaluation

    result = run_evaluation(
        experiment_dir="experiments/20260202_120000",
        mode="debate",
    )
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct import to avoid __init__.py dependency issues
import importlib.util
_bee_eval_path = Path(__file__).parent.parent / "src" / "evaluation" / "bee_eval.py"
_spec = importlib.util.spec_from_file_location("bee_eval", _bee_eval_path)
bee_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bee_eval)

evaluate_checkpoint = bee_eval.evaluate_checkpoint
evaluate_all_checkpoints = bee_eval.evaluate_all_checkpoints
log_eval_to_wandb = bee_eval.log_eval_to_wandb


def run_evaluation(
    experiment_dir: str,
    mode: str = "both",
    eval_data: Optional[str] = None,
    wandb_project: str = "multiagent-debate-rl",
    wandb_run_id: Optional[str] = None,
    skip_wandb: bool = False,
    hive_estimator_class: str = "hive_estimator.command.CommandEstimator",
    hive_model: str = "command-a-03-2025",
) -> dict:
    """Run evaluation pipeline (importable function for post-training hook).

    Args:
        experiment_dir: Path to experiment directory
        mode: "debate", "baseline", or "both"
        eval_data: Path to MATH 500 eval data (default: data/math500_cache.json)
        wandb_project: W&B project name
        wandb_run_id: W&B run ID for logging (optional)
        skip_wandb: Skip W&B logging even if run ID provided
        hive_estimator_class: Hive estimator class name
        hive_model: Model name for Hive estimator

    Returns:
        Dict with evaluation summary:
        {
            "mode": str,
            "results": {mode: list[dict]},
            "status": "complete" | "error",
            "error": str | None
        }
    """
    try:
        # Default eval data path
        if eval_data is None:
            eval_data = str(Path(__file__).parent.parent / "data" / "math500_cache.json")

        # Verify experiment directory exists
        exp_path = Path(experiment_dir)
        if not exp_path.exists():
            return {
                "mode": mode,
                "results": {},
                "status": "error",
                "error": f"Experiment directory not found: {experiment_dir}",
            }

        # Verify eval data exists
        eval_data_path = Path(eval_data)
        if not eval_data_path.exists():
            warnings.warn(f"Eval data not found: {eval_data}. Evaluation will use default BEE data.")

        # Determine modes to evaluate
        modes = ["debate", "baseline"] if mode == "both" else [mode]

        # Hive estimator config
        hive_config = {
            "model": hive_model,
            "prod": True,
            "max_retries": 2,
        }

        # Evaluate each mode
        all_results = {}
        for eval_mode in modes:
            print(f"\n{'=' * 60}")
            print(f"Evaluating {eval_mode} checkpoints")
            print(f"{'=' * 60}")

            results = evaluate_all_checkpoints(
                experiment_dir=experiment_dir,
                eval_data_path=eval_data,
                mode=eval_mode,
                hive_config=hive_config,
            )

            if not results:
                print(f"Warning: No checkpoints evaluated for {eval_mode}")
                continue

            all_results[eval_mode] = results

            # Print summary table
            print(f"\n{eval_mode.upper()} Results Summary:")
            print("-" * 80)
            print(f"{'Step':<8} {'Overall':<10} {'L1':<8} {'L2':<8} {'L3':<8} {'L4':<8} {'L5':<8}")
            print("-" * 80)

            for result in results:
                if result["status"] != "complete":
                    continue

                # Extract step from checkpoint path
                checkpoint_name = Path(result["checkpoint"]).name
                if checkpoint_name.startswith("ckpt-"):
                    step = checkpoint_name.split("-")[1]
                else:
                    step = "?"

                overall = result["overall_accuracy"]
                by_diff = result["by_difficulty"]

                print(
                    f"{step:<8} {overall:<10.3f} {by_diff['1']:<8.3f} {by_diff['2']:<8.3f} "
                    f"{by_diff['3']:<8.3f} {by_diff['4']:<8.3f} {by_diff['5']:<8.3f}"
                )

            print("-" * 80)

            # W&B logging
            if wandb_run_id and not skip_wandb:
                print(f"\nLogging {eval_mode} results to W&B...")
                eval_results_dir = exp_path / eval_mode / "eval_results"

                for result in results:
                    if result["status"] != "complete":
                        continue

                    # Extract step
                    checkpoint_name = Path(result["checkpoint"]).name
                    if checkpoint_name.startswith("ckpt-"):
                        step = int(checkpoint_name.split("-")[1])
                    else:
                        continue

                    # Find results file
                    results_file = eval_results_dir / f"ckpt-{step}_eval.json"
                    if not results_file.exists():
                        warnings.warn(f"Results file not found: {results_file}")
                        continue

                    log_eval_to_wandb(
                        training_run_id=wandb_run_id,
                        checkpoint_step=step,
                        eval_results=result,
                        eval_results_path=str(results_file),
                        wandb_project=wandb_project,
                    )

        return {
            "mode": mode,
            "results": all_results,
            "status": "complete",
            "error": None,
        }

    except Exception as e:
        return {
            "mode": mode,
            "results": {},
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model checkpoints using BEE MATH task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--experiment-dir",
        type=str,
        help="Path to experiment directory (e.g., experiments/20260202_120000/)",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debate", "baseline", "both"],
        default="both",
        help="Evaluation mode: debate, baseline, or both (default: both)",
    )

    # Data paths
    parser.add_argument(
        "--eval-data",
        type=str,
        help="Path to MATH 500 eval data (default: data/math500_cache.json)",
    )

    # Single checkpoint evaluation
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Evaluate a single checkpoint path instead of all (optional)",
    )

    # W&B logging
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="multiagent-debate-rl",
        help="W&B project name (default: multiagent-debate-rl)",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        help="W&B run ID for logging (optional; if not provided, skip W&B logging)",
    )
    parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Skip W&B artifact logging even if run ID provided",
    )

    # Hive configuration
    parser.add_argument(
        "--hive-estimator-class",
        type=str,
        default="hive_estimator.command.CommandEstimator",
        help="Hive estimator class (default: hive_estimator.command.CommandEstimator)",
    )
    parser.add_argument(
        "--hive-model",
        type=str,
        default="command-a-03-2025",
        help="Model name for Hive estimator (default: command-a-03-2025)",
    )

    # Execution control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be evaluated without running",
    )

    args = parser.parse_args()

    # Validation
    if not args.checkpoint and not args.experiment_dir:
        parser.error("Either --experiment-dir or --checkpoint is required")

    # Single checkpoint evaluation
    if args.checkpoint:
        print("Evaluating single checkpoint:")
        print(f"  Checkpoint: {args.checkpoint}")

        if args.eval_data is None:
            args.eval_data = str(Path(__file__).parent.parent / "data" / "math500_cache.json")

        print(f"  Eval data: {args.eval_data}")

        if args.dry_run:
            print("\nDry run - would evaluate checkpoint")
            return

        result = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            eval_data_path=args.eval_data,
            output_dir=str(Path(args.checkpoint).parent.parent / "eval_results"),
            hive_estimator_class=args.hive_estimator_class,
            hive_estimator_config={"model": args.hive_model, "prod": True, "max_retries": 2},
        )

        print(f"\nResult: {result['status']}")
        if result["status"] == "complete":
            print(f"Overall accuracy: {result['overall_accuracy']:.3f}")
            print("Per-difficulty accuracy:")
            for level, acc in result["by_difficulty"].items():
                print(f"  Level {level}: {acc:.3f}")

        return

    # Full experiment evaluation
    print("Evaluating experiment checkpoints:")
    print(f"  Experiment: {args.experiment_dir}")
    print(f"  Mode: {args.mode}")

    # Verify experiment directory
    exp_path = Path(args.experiment_dir)
    if not exp_path.exists():
        print(f"\nError: Experiment directory not found: {args.experiment_dir}")
        sys.exit(1)

    # Find checkpoints
    modes = ["debate", "baseline"] if args.mode == "both" else [args.mode]

    print("\nScanning for checkpoints...")
    for mode in modes:
        ckpt_dir = exp_path / mode / "checkpoints"
        if not ckpt_dir.exists():
            print(f"  {mode}: No checkpoints directory found")
            continue

        checkpoints = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("ckpt-")],
            key=lambda d: int(d.name.split("-")[1]),
        )

        if not checkpoints:
            print(f"  {mode}: No checkpoints found")
        else:
            print(f"  {mode}: {len(checkpoints)} checkpoints found")
            for ckpt in checkpoints[:5]:  # Show first 5
                print(f"    - {ckpt.name}")
            if len(checkpoints) > 5:
                print(f"    ... and {len(checkpoints) - 5} more")

    if args.dry_run:
        print("\nDry run complete - no evaluation performed")
        return

    # Run evaluation
    print("\nStarting evaluation...")

    result = run_evaluation(
        experiment_dir=args.experiment_dir,
        mode=args.mode,
        eval_data=args.eval_data,
        wandb_project=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
        skip_wandb=args.skip_wandb,
        hive_estimator_class=args.hive_estimator_class,
        hive_model=args.hive_model,
    )

    # Print final status
    print("\n" + "=" * 60)
    if result["status"] == "complete":
        print("Evaluation complete!")
        print(f"Results saved in: {args.experiment_dir}/{{mode}}/eval_results/")
    else:
        print(f"Evaluation failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
