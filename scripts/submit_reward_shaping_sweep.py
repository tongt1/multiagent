#!/usr/bin/env python3
"""Batch submission and validation for reward shaping SWEEP comparison experiment.

Submits all 5 reward shaping strategy configs to the cluster and validates
that WandB runs completed with expected metrics.

Usage:
    # Dry run (print commands without submitting):
    uv run python scripts/submit_reward_shaping_sweep.py --dry-run

    # Submit all 5 configs:
    uv run python scripts/submit_reward_shaping_sweep.py --submit

    # Validate runs after completion:
    uv run python scripts/submit_reward_shaping_sweep.py --validate

OWNERS: Multiagent Debate RL Experiment
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap

from configs.reward_shaping_sweep._base import WANDB_PROJECT

# All 5 SWEEP config paths
SWEEP_CONFIGS = [
    "configs/reward_shaping_sweep/sweep_identity.py",
    "configs/reward_shaping_sweep/sweep_difference_rewards.py",
    "configs/reward_shaping_sweep/sweep_potential_based.py",
    "configs/reward_shaping_sweep/sweep_coma_advantage.py",
    "configs/reward_shaping_sweep/sweep_reward_mixing.py",
]

# Expected strategy names (one per config)
EXPECTED_STRATEGIES = {
    "identity",
    "difference_rewards",
    "potential_based",
    "coma_advantage",
    "reward_mixing",
}


def submit_all(dry_run: bool = False) -> None:
    """Submit all 5 SWEEP configs to the cluster.

    Args:
        dry_run: If True, print commands without executing them.
    """
    mode = "DRY RUN" if dry_run else "SUBMIT"
    print(f"\n[{mode}] Submitting {len(SWEEP_CONFIGS)} reward shaping configs")
    print(f"WandB project: {WANDB_PROJECT}\n")

    for config_path in SWEEP_CONFIGS:
        cmd = ["uv", "run", "python", config_path, "--submit", "start"]
        strategy = config_path.rsplit("/", 1)[-1].replace("sweep_", "").replace(".py", "")

        if dry_run:
            print(f"  [{strategy}] Would run: {' '.join(cmd)}")
        else:
            print(f"  [{strategy}] Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                print(f"  [{strategy}] SUCCESS")
            except subprocess.CalledProcessError as e:
                print(f"  [{strategy}] FAILED (exit code {e.returncode})")
                sys.exit(1)

    if dry_run:
        print(f"\nDry run complete. {len(SWEEP_CONFIGS)} commands printed.")
    else:
        print(f"\nAll {len(SWEEP_CONFIGS)} configs submitted.")


def validate_runs() -> None:
    """Validate that all 5 reward shaping runs completed in WandB.

    Queries the WandB API for runs with reward_shaping_strategy in their config
    and checks that all expected strategies are present.
    """
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    print(f"\nValidating runs in WandB project: cohere/{WANDB_PROJECT}")

    runs = api.runs(
        f"cohere/{WANDB_PROJECT}",
        filters={"config.reward_shaping_strategy": {"$exists": True}},
    )

    strategies_found: dict[str, list[dict]] = {}
    for run in runs:
        strategy = run.config.get("reward_shaping_strategy", "unknown")
        if strategy not in strategies_found:
            strategies_found[strategy] = []
        strategies_found[strategy].append({
            "name": run.name,
            "status": run.state,
            "steps": run.summary.get("_step", 0),
        })

    print(f"\nFound {len(strategies_found)} strategies with {sum(len(v) for v in strategies_found.values())} total runs:\n")

    for strategy in sorted(EXPECTED_STRATEGIES):
        if strategy in strategies_found:
            for run_info in strategies_found[strategy]:
                print(
                    f"  [FOUND] {strategy}: "
                    f"name={run_info['name']}, "
                    f"status={run_info['status']}, "
                    f"steps={run_info['steps']}"
                )
        else:
            print(f"  [MISSING] {strategy}: no run found")

    missing = EXPECTED_STRATEGIES - set(strategies_found.keys())
    if missing:
        print(f"\nMissing strategies: {sorted(missing)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(EXPECTED_STRATEGIES)} strategies found.")


def main() -> None:
    """Parse arguments and route to appropriate function."""
    parser = argparse.ArgumentParser(
        description="Batch submission and validation for reward shaping SWEEP comparison experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              uv run python scripts/submit_reward_shaping_sweep.py --dry-run
              uv run python scripts/submit_reward_shaping_sweep.py --submit
              uv run python scripts/submit_reward_shaping_sweep.py --validate
        """),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print submit commands without executing them",
    )
    group.add_argument(
        "--submit",
        action="store_true",
        help="Submit all 5 configs to the cluster",
    )
    group.add_argument(
        "--validate",
        action="store_true",
        help="Validate that all 5 runs completed in WandB",
    )

    args = parser.parse_args()

    if args.dry_run:
        submit_all(dry_run=True)
    elif args.submit:
        submit_all(dry_run=False)
    elif args.validate:
        validate_runs()


if __name__ == "__main__":
    main()
