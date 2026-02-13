"""BEE checkpoint evaluation module.

Evaluates model checkpoints using BEE MATH task, producing per-difficulty
accuracy breakdowns and logging results both locally (JSON) and to W&B artifacts.

This module provides:
- evaluate_checkpoint: Run BEE MATH eval on a single checkpoint
- evaluate_all_checkpoints: Batch eval across all checkpoints in an experiment
- log_eval_to_wandb: Log evaluation results as W&B artifacts

Usage:
    # Evaluate a single checkpoint
    results = evaluate_checkpoint(
        checkpoint_path="experiments/20260202_120000/debate/checkpoints/ckpt-5",
        eval_data_path="data/math500_cache.json",
        output_dir="experiments/20260202_120000/debate/eval_results",
    )

    # Evaluate all checkpoints in an experiment
    all_results = evaluate_all_checkpoints(
        experiment_dir="experiments/20260202_120000",
        eval_data_path="data/math500_cache.json",
        mode="debate",
    )
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Check BEE/Hive availability
BEE_AVAILABLE = False
try:
    from bee.tasks.library.math.MATH import MATH  # type: ignore
    from hive import load_estimator  # type: ignore

    BEE_AVAILABLE = True
except ImportError:
    warnings.warn(
        "BEE/Hive not available. Evaluation will return stub results. "
        "Install BEE and Hive for actual evaluation."
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    eval_data_path: str,
    output_dir: str,
    hive_estimator_class: str = "hive_estimator.command.CommandEstimator",
    hive_estimator_config: Optional[dict] = None,
    num_samples: int = 500,
) -> dict[str, Any]:
    """Run BEE MATH evaluation on a single checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (S3 or local)
        eval_data_path: Path to MATH 500 eval data (JSONL)
        output_dir: Directory to save eval results JSON
        hive_estimator_class: Hive estimator class name
        hive_estimator_config: Hive estimator config dict
        num_samples: Number of samples to evaluate (default: 500)

    Returns:
        Dict with evaluation metrics:
        {
            "checkpoint": str,
            "overall_accuracy": float,
            "by_difficulty": {"1": float, "2": float, ...},
            "per_sample_results": [{"problem_id": str, "level": int, "correct": bool, ...}],
            "num_samples": int,
            "timestamp": str (ISO format UTC),
            "status": "complete" | "bee_not_available"
        }
    """
    if not BEE_AVAILABLE:
        warnings.warn("BEE not available, returning stub results")
        return {
            "status": "bee_not_available",
            "checkpoint": checkpoint_path,
            "overall_accuracy": 0.0,
            "by_difficulty": {str(i): 0.0 for i in range(1, 6)},
            "per_sample_results": [],
            "num_samples": 0,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": "BEE/Hive not available. Install for actual evaluation.",
        }

    # Default estimator config
    if hive_estimator_config is None:
        hive_estimator_config = {
            "model": "command-a-03-2025",
            "prod": True,
            "max_retries": 2,
        }

    print(f"\nEvaluating checkpoint: {checkpoint_path}")
    print(f"Using {num_samples} MATH samples from: {eval_data_path}")

    try:
        # Load checkpoint as Hive estimator
        print(f"Loading estimator: {hive_estimator_class}")
        estimator = load_estimator(
            cls_name=hive_estimator_class,
            checkpoint=checkpoint_path,
            usage="bee/eval",
            **hive_estimator_config,
        )

        # Load MATH 500 eval data
        # BEE MATH task expects data in specific format
        # For now, we'll use a simplified approach assuming eval_data_path is MATH format
        print(f"Loading MATH eval data from: {eval_data_path}")

        # Initialize BEE MATH task
        # Note: BEE tasks typically take a config object
        # This is a simplified version - may need adjustment based on actual BEE API
        from bee.core.config import Config  # type: ignore

        task_config = Config(
            {
                "path": eval_data_path,
                "num_generations": 1,  # Single generation per problem
                "sources": ["MATH"],
                "num_truncate_per_source": num_samples,
            }
        )

        task = MATH(task_config)
        samples = task.load_data()

        # Limit to num_samples if more loaded
        if len(samples) > num_samples:
            samples = samples[:num_samples]

        print(f"Running evaluation on {len(samples)} samples...")

        # Run evaluation
        results = task.run(estimator=estimator, samples=samples)

        # Parse results into our format
        # BEE tasks return nested dicts with overall and per-level metrics
        overall_accuracy = results.get("overall", {}).get("accuracy", 0.0)

        # Per-difficulty breakdown (MATH levels 1-5)
        by_difficulty = {}
        for level in range(1, 6):
            level_key = f"MATH_level_{level}"
            if level_key in results:
                by_difficulty[str(level)] = results[level_key].get("accuracy", 0.0)
            else:
                by_difficulty[str(level)] = 0.0

        # Per-sample results (if available in results)
        per_sample_results = []
        if "samples" in results:
            for sample in results["samples"]:
                per_sample_results.append(
                    {
                        "problem_id": sample.get("id", "unknown"),
                        "level": sample.get("level", 0),
                        "correct": sample.get("correct", False),
                        "predicted": sample.get("predicted", ""),
                        "expected": sample.get("expected", ""),
                    }
                )

        # Aggregate metrics
        metrics = {
            "status": "complete",
            "checkpoint": checkpoint_path,
            "overall_accuracy": overall_accuracy,
            "by_difficulty": by_difficulty,
            "per_sample_results": per_sample_results,
            "num_samples": len(samples),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract checkpoint step from path (e.g., "ckpt-5" -> "5")
        checkpoint_name = Path(checkpoint_path).name
        if checkpoint_name.startswith("ckpt-"):
            step = checkpoint_name.split("-")[1]
            results_file = output_path / f"ckpt-{step}_eval.json"
        else:
            results_file = output_path / "eval_results.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Results saved to: {results_file}")
        print(f"Overall accuracy: {overall_accuracy:.3f}")
        print("Per-difficulty accuracy:")
        for level, acc in by_difficulty.items():
            print(f"  Level {level}: {acc:.3f}")

        return metrics

    except Exception as e:
        warnings.warn(f"Evaluation failed: {e}")
        return {
            "status": "error",
            "checkpoint": checkpoint_path,
            "overall_accuracy": 0.0,
            "by_difficulty": {str(i): 0.0 for i in range(1, 6)},
            "per_sample_results": [],
            "num_samples": 0,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(e),
        }


def evaluate_all_checkpoints(
    experiment_dir: str,
    eval_data_path: str,
    mode: str = "debate",
    hive_config: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Evaluate all checkpoints in an experiment directory.

    Args:
        experiment_dir: Path to experiment directory (e.g., experiments/20260202_120000)
        eval_data_path: Path to MATH 500 eval data
        mode: "debate" or "baseline" (determines checkpoint subdirectory)
        hive_config: Optional Hive estimator config

    Returns:
        List of per-checkpoint evaluation result dicts, sorted by step number
    """
    experiment_path = Path(experiment_dir)
    checkpoints_dir = experiment_path / mode / "checkpoints"

    if not checkpoints_dir.exists():
        warnings.warn(f"Checkpoints directory not found: {checkpoints_dir}")
        return []

    print(f"\nScanning for checkpoints in: {checkpoints_dir}")

    # Find all checkpoint directories (ckpt-N pattern)
    checkpoint_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("ckpt-")],
        key=lambda d: int(d.name.split("-")[1]),  # Sort by step number
    )

    if not checkpoint_dirs:
        warnings.warn(f"No checkpoints found in {checkpoints_dir}")
        return []

    print(f"Found {len(checkpoint_dirs)} checkpoints:")
    for ckpt_dir in checkpoint_dirs:
        print(f"  - {ckpt_dir.name}")

    # Evaluate each checkpoint
    all_results = []
    eval_output_dir = experiment_path / mode / "eval_results"

    for ckpt_dir in checkpoint_dirs:
        print(f"\n{'=' * 60}")
        result = evaluate_checkpoint(
            checkpoint_path=str(ckpt_dir),
            eval_data_path=eval_data_path,
            output_dir=str(eval_output_dir),
            hive_estimator_config=hive_config,
        )
        all_results.append(result)

    # Save combined results
    combined_results_file = eval_output_dir / "all_results.json"
    combined_results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(combined_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Combined results saved to: {combined_results_file}")
    print(f"\nSummary: {len(all_results)} checkpoints evaluated")

    return all_results


def log_eval_to_wandb(
    training_run_id: str,
    checkpoint_step: int,
    eval_results: dict[str, Any],
    eval_results_path: str,
    wandb_project: str = "multiagent-debate-rl",
) -> None:
    """Log evaluation results to W&B as artifacts.

    Args:
        training_run_id: W&B run ID from training
        checkpoint_step: Training step number
        eval_results: Dict with accuracy metrics from evaluate_checkpoint
        eval_results_path: Local path to eval results JSON file
        wandb_project: W&B project name
    """
    try:
        import wandb  # type: ignore
    except ImportError:
        warnings.warn("W&B not available, skipping artifact logging")
        return

    try:
        print(f"\nLogging eval results to W&B (project: {wandb_project}, run: {training_run_id})")

        # Initialize/resume W&B run
        run = wandb.init(
            project=wandb_project,
            id=training_run_id,
            resume="allow",
        )

        # Create artifact for eval results
        artifact = wandb.Artifact(
            name=f"eval-results-step-{checkpoint_step}",
            type="evaluation",
            metadata={
                "checkpoint_step": checkpoint_step,
                "overall_accuracy": eval_results["overall_accuracy"],
                "num_samples": eval_results["num_samples"],
                "timestamp": eval_results["timestamp"],
            },
        )

        # Add eval results JSON
        artifact.add_file(eval_results_path, name="eval_results.json")

        # Log artifact
        run.log_artifact(artifact)

        # Also log as scalar metrics for easy tracking
        metrics = {
            f"eval/overall_accuracy": eval_results["overall_accuracy"],
        }

        # Add per-level metrics
        for level, acc in eval_results["by_difficulty"].items():
            metrics[f"eval/accuracy_level_{level}"] = acc

        # Log at the checkpoint step
        run.log(metrics, step=checkpoint_step)

        print(f"Logged artifact: eval-results-step-{checkpoint_step}")
        print(f"Logged metrics: eval/overall_accuracy = {eval_results['overall_accuracy']:.3f}")

        run.finish()

    except Exception as e:
        warnings.warn(f"Failed to log to W&B: {e}")
