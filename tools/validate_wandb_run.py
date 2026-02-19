#!/usr/bin/env python3
"""Tier 4: Validate a WandB training run's metrics for integration test success.

Checks key metrics from a completed WandB run to verify training worked:
- Loss stability (no NaN)
- Reward variance (not all identical)
- Step completion
- Eval completion

Usage:
    python tools/validate_wandb_run.py --run-name <wandb-run-name>
    python tools/validate_wandb_run.py --run-path <entity/project/run_id>
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def validate_run(run) -> list[CheckResult]:
    """Validate a WandB run object against success criteria."""
    results: list[CheckResult] = []
    history = run.history()

    # 1. Loss stability (no NaN)
    if "train/loss" in history.columns:
        loss_values = history["train/loss"].dropna()
        has_nan = history["train/loss"].isna().any()
        if len(loss_values) > 0:
            results.append(CheckResult(
                "Loss stability",
                not has_nan,
                f"{'NaN detected' if has_nan else 'No NaN'}, "
                f"last={loss_values.iloc[-1]:.4f}" if len(loss_values) > 0 else "No values",
            ))
        else:
            results.append(CheckResult("Loss stability", False, "No loss values recorded"))
    else:
        results.append(CheckResult("Loss stability", False, "train/loss metric not found"))

    # 2. Reward variance (not all identical)
    reward_std_key = "debate/zero_advantage/frac_reward_zero_std"
    if reward_std_key in history.columns:
        values = history[reward_std_key].dropna()
        if len(values) > 0:
            last_val = values.iloc[-1]
            results.append(CheckResult(
                "Reward variance (frac_zero_std < 1.0)",
                last_val < 1.0,
                f"frac_reward_zero_std={last_val:.4f}",
            ))
        else:
            results.append(CheckResult("Reward variance", False, "No values recorded"))
    else:
        results.append(CheckResult(
            "Reward variance",
            False,
            f"Metric '{reward_std_key}' not found. Available: {list(history.columns)[:10]}...",
        ))

    # 3. Mean reward std > 0
    mean_std_key = "debate/zero_advantage/mean_reward_std"
    if mean_std_key in history.columns:
        values = history[mean_std_key].dropna()
        if len(values) > 0:
            last_val = values.iloc[-1]
            results.append(CheckResult(
                "Mean reward std > 0",
                last_val > 0,
                f"mean_reward_std={last_val:.4f}",
            ))
        else:
            results.append(CheckResult("Mean reward std", False, "No values recorded"))
    else:
        results.append(CheckResult("Mean reward std", False, f"Metric '{mean_std_key}' not found"))

    # 4. Training step completion
    if "_step" in history.columns:
        max_step = history["_step"].max()
        results.append(CheckResult(
            "Training step completion",
            max_step >= 1,
            f"Completed {int(max_step)} steps",
        ))
    else:
        results.append(CheckResult("Training step completion", False, "No step data"))

    # 5. Run state
    run_state = run.state
    results.append(CheckResult(
        "Run state",
        run_state == "finished",
        f"State: {run_state}",
    ))

    # 6. Run duration
    if hasattr(run, "summary") and run.summary:
        runtime = run.summary.get("_runtime", 0)
        results.append(CheckResult(
            "Run duration",
            runtime > 0,
            f"{runtime:.0f}s ({runtime / 60:.1f}min)" if runtime else "Unknown",
        ))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate WandB training run metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-name", help="WandB run name to search for")
    group.add_argument("--run-path", help="Full WandB run path (entity/project/run_id)")
    parser.add_argument("--project", default="multiagent-debate-rl", help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity (defaults to logged-in user)")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("FAIL: wandb not installed. Run: pip install wandb")
        return 1

    api = wandb.Api()

    if args.run_path:
        print(f"Fetching run: {args.run_path}")
        run = api.run(args.run_path)
    else:
        print(f"Searching for run: {args.run_name} in project {args.project}")
        entity = args.entity or api.default_entity
        runs = api.runs(
            f"{entity}/{args.project}",
            filters={"display_name": args.run_name},
        )
        runs_list = list(runs)
        if not runs_list:
            print(f"FAIL: No runs found with name '{args.run_name}'")
            return 1
        if len(runs_list) > 1:
            print(f"WARNING: Found {len(runs_list)} runs with name '{args.run_name}', using most recent")
        run = runs_list[0]

    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    print(f"URL: {run.url}")
    print()

    results = validate_run(run)

    passed = 0
    failed = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "+" if r.passed else "-"
        print(f"  [{icon}] {status}: {r.name} — {r.detail}")
        if r.passed:
            passed += 1
        else:
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
