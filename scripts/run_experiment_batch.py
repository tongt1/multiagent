#!/usr/bin/env python3
"""Run a batch of experiments end-to-end.

Loads experiment configurations from a Python config file,
validates them, and runs each experiment through the full pipeline:
data generation -> data conversion -> training -> evaluation -> comparison.

Usage:
    # Preview (dry run - DEFAULT, safe)
    python scripts/run_experiment_batch.py --config configs/experiments/example_batch.py

    # Submit experiments to cluster
    python scripts/run_experiment_batch.py --config configs/experiments/example_batch.py --submit

    # Resume previously failed experiments
    python scripts/run_experiment_batch.py --config configs/experiments/example_batch.py --submit --resume

    # Run specific experiment from batch
    python scripts/run_experiment_batch.py --config configs/experiments/example_batch.py --submit --only debate_cmdA
"""

import argparse
import concurrent.futures
import datetime
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.experiments.experiment_config import ExperimentConfig, ExperimentBatchConfig
from src.orchestration.experiment_runner import ExperimentRunner


def load_batch_config(config_path: str) -> ExperimentBatchConfig:
    """Load BATCH_CONFIG from Python config file.

    Args:
        config_path: Path to config .py file

    Returns:
        ExperimentBatchConfig instance

    Raises:
        ValueError: If BATCH_CONFIG not found or invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise ValueError(f"Config file not found: {config_path}")

    # Load module
    spec = importlib.util.spec_from_file_location("batch_config", config_file)
    if not spec or not spec.loader:
        raise ValueError(f"Failed to load config module: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get BATCH_CONFIG attribute
    if not hasattr(module, "BATCH_CONFIG"):
        raise ValueError(
            f"Config file must define BATCH_CONFIG variable\n"
            f"File: {config_path}\n"
            f"Available attributes: {[a for a in dir(module) if not a.startswith('_')]}"
        )

    batch_config = module.BATCH_CONFIG
    if not isinstance(batch_config, ExperimentBatchConfig):
        raise ValueError(
            f"BATCH_CONFIG must be ExperimentBatchConfig instance, got {type(batch_config)}"
        )

    return batch_config


def filter_experiments(
    experiments: list[ExperimentConfig],
    only: str | None
) -> list[ExperimentConfig]:
    """Filter experiments by name.

    Args:
        experiments: List of all experiments
        only: Comma-separated experiment names (or None for all)

    Returns:
        Filtered experiment list

    Raises:
        ValueError: If no matches found
    """
    if not only:
        return experiments

    target_names = [name.strip() for name in only.split(",")]
    filtered = [exp for exp in experiments if exp.name in target_names]

    if not filtered:
        available = [exp.name for exp in experiments]
        raise ValueError(
            f"No experiments match --only filter: {target_names}\n"
            f"Available experiments: {available}"
        )

    return filtered


def print_batch_header(
    batch_config: ExperimentBatchConfig,
    submit: bool,
    experiments: list[ExperimentConfig],
    max_retries: int,
    sequential: bool,
):
    """Print batch execution header."""
    print("\n" + "="*70)
    print(f"Experiment Batch: {batch_config.batch_name or 'unnamed'}")
    print("="*70)
    print(f"Mode: {'SUBMIT' if submit else 'DRY RUN (add --submit to execute)'}")
    print(f"Experiments: {len(experiments)}")
    print(f"Base checkpoint: {batch_config.base_ckpt_path}")
    print(f"Max retries: {max_retries}")
    print(f"Parallel: {'no (--sequential)' if sequential else ('yes' if batch_config.parallel else 'no')}")
    print("="*70 + "\n")


def print_experiment_table(experiments: list[ExperimentConfig]):
    """Print table of experiments with overrides."""
    print("Experiments:\n")
    print(f"{'#':<4} {'Name':<20} {'Mode':<10} {'Model':<30} {'Overrides':<30} {'Linked':<20}")
    print("-" * 120)

    for i, exp in enumerate(experiments, 1):
        # Collect non-None overrides
        overrides = []
        if exp.learning_rate is not None:
            overrides.append(f"lr={exp.learning_rate}")
        if exp.train_batch_size is not None:
            overrides.append(f"bs={exp.train_batch_size}")
        if exp.total_train_steps is not None:
            overrides.append(f"steps={exp.total_train_steps}")
        if exp.kl_beta is not None:
            overrides.append(f"kl={exp.kl_beta}")
        if exp.generations_per_prompt is not None:
            overrides.append(f"gens={exp.generations_per_prompt}")

        override_str = ", ".join(overrides) if overrides else "none (base defaults)"
        linked_str = exp.linked_experiment or "-"

        print(f"{i:<4} {exp.name:<20} {exp.mode:<10} {exp.model:<30} {override_str:<30} {linked_str:<20}")

    print()


def run_experiment_safe(
    exp: ExperimentConfig,
    base_ckpt_path: str,
    batch_dir: Path,
    max_retries: int,
    dry_run: bool,
) -> dict:
    """Run experiment with exception handling.

    Args:
        exp: Experiment config
        base_ckpt_path: Base checkpoint path
        batch_dir: Batch directory
        max_retries: Retry count
        dry_run: Dry run mode

    Returns:
        Result dict with experiment name, status, duration, outputs
    """
    start_time = time.time()

    try:
        runner = ExperimentRunner(
            config=exp,
            base_ckpt_path=base_ckpt_path,
            base_dir=batch_dir / "experiments",
            max_retries=max_retries,
            dry_run=dry_run,
        )
        result = runner.run()

        duration = time.time() - start_time

        return {
            "name": exp.name,
            "status": result.get("status", "unknown"),
            "duration": duration,
            "experiment_id": result.get("experiment_id"),
            "outputs": result.get("outputs", {}),
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[ERROR] Experiment '{exp.name}' failed with exception: {e}\n")

        return {
            "name": exp.name,
            "status": "failed",
            "duration": duration,
            "error": str(e),
        }


def run_batch_sequential(
    experiments: list[ExperimentConfig],
    base_ckpt_path: str,
    batch_dir: Path,
    max_retries: int,
    dry_run: bool,
) -> list[dict]:
    """Run experiments sequentially.

    Args:
        experiments: List of experiments
        base_ckpt_path: Base checkpoint path
        batch_dir: Batch directory
        max_retries: Retry count
        dry_run: Dry run mode

    Returns:
        List of result dicts
    """
    results = []

    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"Running experiment {i}/{len(experiments)}: {exp.name}")
        print(f"{'='*70}\n")

        result = run_experiment_safe(exp, base_ckpt_path, batch_dir, max_retries, dry_run)
        results.append(result)

        print(f"\n[{exp.name}] Status: {result['status']} (duration: {result['duration']:.1f}s)")

    return results


def run_batch_parallel(
    experiments: list[ExperimentConfig],
    base_ckpt_path: str,
    batch_dir: Path,
    max_retries: int,
    dry_run: bool,
    max_workers: int = 4,
) -> list[dict]:
    """Run independent experiments in parallel.

    Args:
        experiments: List of experiments
        base_ckpt_path: Base checkpoint path
        batch_dir: Batch directory
        max_retries: Retry count
        dry_run: Dry run mode
        max_workers: Max concurrent experiments

    Returns:
        List of result dicts
    """
    print(f"\nRunning {len(experiments)} experiments in parallel (max_workers={max_workers})...\n")

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(
                run_experiment_safe,
                exp,
                base_ckpt_path,
                batch_dir,
                max_retries,
                dry_run,
            ): exp
            for exp in experiments
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_exp):
            exp = future_to_exp[future]
            result = future.result()
            results.append(result)

            print(f"[{exp.name}] Complete - Status: {result['status']} (duration: {result['duration']:.1f}s)")

    return results


def print_batch_summary(results: list[dict], total_duration: float):
    """Print batch completion summary.

    Args:
        results: List of experiment results
        total_duration: Total batch duration in seconds
    """
    print("\n" + "="*70)
    print("BATCH COMPLETE")
    print("="*70 + "\n")

    print(f"{'Experiment':<20} {'Status':<15} {'Duration':<15} {'Final Accuracy':<15}")
    print("-" * 70)

    for result in results:
        name = result["name"]
        status = result["status"]
        duration = f"{result['duration']:.1f}s"

        # Extract accuracy if available
        outputs = result.get("outputs", {})
        eval_results = outputs.get("evaluation", {}).get("results", {})
        accuracy = eval_results.get("accuracy", "N/A")
        if isinstance(accuracy, (int, float)):
            accuracy = f"{accuracy:.1%}"

        print(f"{name:<20} {status:<15} {duration:<15} {accuracy:<15}")

    print()
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")

    # Count statuses
    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"Summary: {', '.join(f'{status}={count}' for status, count in sorted(status_counts.items()))}")
    print()


def main():
    """Main batch runner entry point."""
    parser = argparse.ArgumentParser(
        description="Run a batch of experiments end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to Python config file containing BATCH_CONFIG",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit to cluster (default: dry run / preview)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume failed experiments from their last completed stage",
    )
    parser.add_argument(
        "--only",
        help="Run only specific experiment(s) by name (comma-separated)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential execution even if batch.parallel=True",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Override batch max_retries (default: use batch config value)",
    )
    parser.add_argument(
        "--base-dir",
        default="experiments",
        help="Override base experiment output directory (default: experiments)",
    )

    args = parser.parse_args()

    # Load batch config
    try:
        batch_config = load_batch_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate batch config
    errors = batch_config.validate()
    if errors:
        print("Batch configuration validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    # Filter experiments if --only specified
    try:
        experiments = filter_experiments(batch_config.experiments, args.only)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine max_retries
    max_retries = args.max_retries if args.max_retries is not None else batch_config.max_retries

    # Print batch header
    print_batch_header(
        batch_config,
        args.submit,
        experiments,
        max_retries,
        args.sequential,
    )

    # Print experiment table
    print_experiment_table(experiments)

    # Dry run mode
    if not args.submit:
        print("\nDRY RUN MODE - Previewing each experiment:\n")

        for exp in experiments:
            runner = ExperimentRunner(
                config=exp,
                base_ckpt_path=batch_config.base_ckpt_path,
                max_retries=max_retries,
                dry_run=True,
            )
            runner.run()
            print()

        print("="*70)
        print("Dry run complete. Add --submit flag to execute.")
        print("="*70 + "\n")
        return

    # Create batch directory
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_name = batch_config.batch_name or "batch"
    batch_dir = Path(args.base_dir) / f"{timestamp}_{batch_name}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch directory: {batch_dir}\n")

    # Save batch manifest
    manifest = {
        "batch_name": batch_config.batch_name,
        "timestamp": timestamp,
        "base_ckpt_path": batch_config.base_ckpt_path,
        "gcs_bucket": batch_config.gcs_bucket,
        "max_retries": max_retries,
        "parallel": batch_config.parallel and not args.sequential,
        "experiments": [exp.name for exp in experiments],
        "config_file": args.config,
    }

    manifest_path = batch_dir / "batch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved batch manifest: {manifest_path}\n")

    # Run experiments
    batch_start = time.time()

    if args.sequential or not batch_config.parallel:
        results = run_batch_sequential(
            experiments,
            batch_config.base_ckpt_path,
            batch_dir,
            max_retries,
            dry_run=False,
        )
    else:
        # Parallel execution
        max_workers = min(4, len(experiments))
        results = run_batch_parallel(
            experiments,
            batch_config.base_ckpt_path,
            batch_dir,
            max_retries,
            dry_run=False,
            max_workers=max_workers,
        )

    batch_duration = time.time() - batch_start

    # Save batch results
    results_path = batch_dir / "batch_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "batch_name": batch_config.batch_name,
                "timestamp": timestamp,
                "duration": batch_duration,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved batch results: {results_path}")

    # Print summary
    print_batch_summary(results, batch_duration)

    # Note linked pairs for comparison
    linked_pairs = batch_config.get_linked_pairs()
    if linked_pairs:
        print("Linked experiment pairs for comparison:")
        seen_pairs = set()
        for exp, linked in linked_pairs:
            pair_key = tuple(sorted([exp.name, linked.name]))
            if pair_key not in seen_pairs:
                print(f"  - {exp.name} <-> {linked.name}")
                seen_pairs.add(pair_key)
        print()


if __name__ == "__main__":
    main()
