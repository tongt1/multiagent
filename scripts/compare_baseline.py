#!/usr/bin/env python3
"""Multi-seed comparison of trained vs baseline CooperBench evaluation results.

Aggregates evaluation results across 3+ seeds, computes statistical
significance, and produces a formatted comparison report with per-task
breakdown and agent efficiency metrics.

Usage:
    # Compare results with ASCII table output
    python scripts/compare_baseline.py \\
        --results-dir /path/to/eval/results \\
        --seeds 42,43,44 --print-table

    # Save comparison report JSON
    python scripts/compare_baseline.py \\
        --results-dir /path/to/eval/results \\
        --output comparison_report.json

The core functions are importable for programmatic use:
    from scripts.compare_baseline import compare_results, aggregate_seeds
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure reward-training root is on path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_seed_results(results_dir: str, seed: int) -> dict[str, Any]:
    """Load baseline (step-0) and final checkpoint eval results for a single seed.

    Expected directory structure:
        {results_dir}/seed{N}/step0_eval.json     (baseline)
        {results_dir}/seed{N}/final_eval.json      (trained)
        OR
        {results_dir}/seed{N}/cooperbench_eval_seed{N}.json  (single file with both)

    Falls back to looking for any JSON with "step0" or "final" in the name.

    Args:
        results_dir: Root directory containing per-seed result subdirectories.
        seed: The seed number to load.

    Returns:
        Dict with "baseline" and "trained" keys, each containing eval JSON.

    Raises:
        FileNotFoundError: If expected result files are not found.
    """
    seed_dir = Path(results_dir) / f"seed{seed}"

    if not seed_dir.exists():
        # Try flat layout
        seed_dir = Path(results_dir)

    result = {"baseline": None, "trained": None}

    # Try standard naming
    baseline_candidates = [
        seed_dir / "step0_eval.json",
        seed_dir / f"step0_eval_seed{seed}.json",
        seed_dir / f"cooperbench_eval_step0_seed{seed}.json",
        seed_dir / "baseline_eval.json",
    ]
    trained_candidates = [
        seed_dir / "final_eval.json",
        seed_dir / f"final_eval_seed{seed}.json",
        seed_dir / f"cooperbench_eval_seed{seed}.json",
        seed_dir / "trained_eval.json",
    ]

    for path in baseline_candidates:
        if path.exists():
            with open(path) as f:
                result["baseline"] = json.load(f)
            break

    for path in trained_candidates:
        if path.exists():
            with open(path) as f:
                result["trained"] = json.load(f)
            break

    # Fallback: scan directory for JSON files
    if result["baseline"] is None or result["trained"] is None:
        if seed_dir.exists():
            for json_file in sorted(seed_dir.glob("*.json")):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    name = json_file.stem.lower()
                    if "step0" in name or "baseline" in name:
                        if result["baseline"] is None:
                            result["baseline"] = data
                    elif "final" in name or "trained" in name:
                        if result["trained"] is None:
                            result["trained"] = data
                except (json.JSONDecodeError, KeyError):
                    continue

    if result["baseline"] is None and result["trained"] is None:
        raise FileNotFoundError(
            f"No evaluation results found for seed {seed} in {results_dir}. "
            f"Expected files at {seed_dir}/step0_eval.json and {seed_dir}/final_eval.json"
        )

    return result


def _extract_metric(eval_data: dict[str, Any], metric: str) -> float:
    """Extract a metric from evaluation results JSON.

    Args:
        eval_data: Evaluation results dict (output of eval_cooperbench.py).
        metric: Metric name (e.g., "pass_at_1", "partial_credit").

    Returns:
        Metric value, or 0.0 if not found.
    """
    overall = eval_data.get("overall", {})
    return overall.get(metric, 0.0)


def aggregate_seeds(results_dir: str, seeds: list[int]) -> dict[str, Any]:
    """Aggregate evaluation results across multiple seeds.

    Loads baseline and trained results for each seed, computes
    cross-seed statistics (mean, std), and builds per-task comparison.

    Args:
        results_dir: Directory containing per-seed result subdirectories.
        seeds: List of seed values to aggregate.

    Returns:
        Aggregated results dict with baseline, trained, and per-task stats.
    """
    metrics = ["pass_at_1", "pass_at_3", "pass_at_5", "partial_credit",
               "mean_agent_turns", "mean_rollout_time_s", "mean_token_count"]

    seed_data: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        try:
            seed_data[seed] = load_seed_results(results_dir, seed)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            continue

    if not seed_data:
        return {"error": "No seed data loaded", "seeds": seeds}

    # Aggregate per-metric
    baseline_metrics: dict[str, dict[str, Any]] = {}
    trained_metrics: dict[str, dict[str, Any]] = {}

    for metric in metrics:
        baseline_values = []
        trained_values = []
        for seed, data in seed_data.items():
            if data.get("baseline"):
                baseline_values.append(_extract_metric(data["baseline"], metric))
            if data.get("trained"):
                trained_values.append(_extract_metric(data["trained"], metric))

        if baseline_values:
            baseline_metrics[metric] = {
                "mean": _mean(baseline_values),
                "std": _std(baseline_values),
                "per_seed": baseline_values,
            }
        if trained_values:
            trained_metrics[metric] = {
                "mean": _mean(trained_values),
                "std": _std(trained_values),
                "per_seed": trained_values,
            }

    # Per-task comparison
    task_comparison = _aggregate_per_task(seed_data)

    return {
        "seeds": list(seed_data.keys()),
        "n_seeds": len(seed_data),
        "baseline": baseline_metrics,
        "trained": trained_metrics,
        "per_task": task_comparison,
    }


def _aggregate_per_task(seed_data: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per-task comparison across seeds.

    Since eval sets rotate per seed, not all tasks appear in all seeds.
    This tracks which seeds evaluated each task and computes means.

    Args:
        seed_data: Dict mapping seed -> {"baseline": ..., "trained": ...}.

    Returns:
        List of per-task comparison dicts.
    """
    task_stats: dict[str, dict[str, list[float]]] = {}

    for seed, data in seed_data.items():
        for phase in ["baseline", "trained"]:
            phase_data = data.get(phase)
            if not phase_data:
                continue
            for task in phase_data.get("per_task", []):
                task_id = task.get("task_id", "unknown")
                if task_id not in task_stats:
                    task_stats[task_id] = {
                        "baseline_pass1": [],
                        "trained_pass1": [],
                        "seeds_evaluated": [],
                    }
                if phase == "baseline":
                    task_stats[task_id]["baseline_pass1"].append(task.get("pass_at_1", 0))
                else:
                    task_stats[task_id]["trained_pass1"].append(task.get("pass_at_1", 0))
                if seed not in task_stats[task_id]["seeds_evaluated"]:
                    task_stats[task_id]["seeds_evaluated"].append(seed)

    result = []
    for task_id, stats in sorted(task_stats.items()):
        baseline_mean = _mean(stats["baseline_pass1"]) if stats["baseline_pass1"] else 0.0
        trained_mean = _mean(stats["trained_pass1"]) if stats["trained_pass1"] else 0.0
        result.append({
            "task_id": task_id,
            "seeds_evaluated": len(stats["seeds_evaluated"]),
            "baseline_pass1": baseline_mean,
            "trained_pass1": trained_mean,
            "improvement": trained_mean - baseline_mean,
        })

    # Sort by improvement descending
    result.sort(key=lambda x: x["improvement"], reverse=True)
    return result


def compute_statistical_test(
    baseline_scores: list[float],
    trained_scores: list[float],
) -> dict[str, Any]:
    """Compute paired t-test for baseline vs trained scores.

    Uses a paired t-test when scores correspond to the same seeds,
    which is the standard case for multi-seed comparison.

    With only 3 seeds, p-values may not reach significance thresholds,
    but reporting variance is still valuable for understanding reliability.

    Args:
        baseline_scores: Per-seed baseline metric values.
        trained_scores: Per-seed trained metric values.

    Returns:
        Dict with t_statistic, p_value, and significance flags.
    """
    n = min(len(baseline_scores), len(trained_scores))
    if n < 2:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "significant_at_005": False,
            "significant_at_001": False,
            "n_pairs": n,
            "note": "Insufficient data for t-test (need >= 2 pairs)",
        }

    # Paired differences
    diffs = [trained_scores[i] - baseline_scores[i] for i in range(n)]
    mean_diff = _mean(diffs)
    std_diff = _std(diffs)

    if std_diff == 0:
        # All differences identical
        if mean_diff > 0:
            return {
                "t_statistic": float("inf"),
                "p_value": 0.0,
                "significant_at_005": True,
                "significant_at_001": True,
                "n_pairs": n,
            }
        elif mean_diff < 0:
            return {
                "t_statistic": float("-inf"),
                "p_value": 0.0,
                "significant_at_005": True,
                "significant_at_001": True,
                "n_pairs": n,
            }
        else:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant_at_005": False,
                "significant_at_001": False,
                "n_pairs": n,
            }

    # t = mean_diff / (std_diff / sqrt(n))
    se = std_diff / math.sqrt(n)
    t_stat = mean_diff / se

    # Approximate p-value using Student's t-distribution
    # For small n, use a lookup table approximation
    df = n - 1
    p_value = _approx_t_pvalue(abs(t_stat), df)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
        "n_pairs": n,
        "degrees_of_freedom": df,
    }


def _approx_t_pvalue(t_abs: float, df: int) -> float:
    """Approximate two-tailed p-value for Student's t-distribution.

    Uses a simple approximation that's adequate for quick comparison.
    For rigorous analysis, use scipy.stats.ttest_rel.

    Args:
        t_abs: Absolute value of t-statistic.
        df: Degrees of freedom.

    Returns:
        Approximate two-tailed p-value.
    """
    try:
        from scipy.stats import t as t_dist
        return 2 * t_dist.sf(t_abs, df)
    except ImportError:
        pass

    # Fallback: crude approximation based on common critical values
    # For df=2 (3 seeds): t_0.05 = 4.303, t_0.01 = 9.925
    # For df=4 (5 seeds): t_0.05 = 2.776, t_0.01 = 4.604
    # For df=9 (10 seeds): t_0.05 = 2.262, t_0.01 = 3.250
    critical_values = {
        1: {0.05: 12.706, 0.01: 63.657},
        2: {0.05: 4.303, 0.01: 9.925},
        3: {0.05: 3.182, 0.01: 5.841},
        4: {0.05: 2.776, 0.01: 4.604},
        5: {0.05: 2.571, 0.01: 4.032},
        10: {0.05: 2.228, 0.01: 3.169},
        20: {0.05: 2.086, 0.01: 2.845},
        30: {0.05: 2.042, 0.01: 2.750},
    }

    # Find closest df
    closest_df = min(critical_values.keys(), key=lambda x: abs(x - df))
    cv = critical_values[closest_df]

    if t_abs >= cv[0.01]:
        return 0.005  # p < 0.01
    elif t_abs >= cv[0.05]:
        return 0.025  # 0.01 < p < 0.05
    else:
        # Rough estimate: p ~ 0.5 * exp(-t^2 / (2*df))
        # This is very approximate but gives a reasonable order
        return min(1.0, 2.0 * math.exp(-t_abs * t_abs / (2.0 + df)))


def compare_results(
    results_dir: str,
    seeds: list[int],
) -> dict[str, Any]:
    """Main comparison entry point: aggregate + statistical test.

    Combines ``aggregate_seeds`` with ``compute_statistical_test`` to
    produce a full comparison report.

    Args:
        results_dir: Directory containing per-seed result subdirectories.
        seeds: List of seed values.

    Returns:
        Full comparison report dict.
    """
    agg = aggregate_seeds(results_dir, seeds)

    if "error" in agg:
        return agg

    # Statistical test on pass@1
    baseline_p1 = agg.get("baseline", {}).get("pass_at_1", {})
    trained_p1 = agg.get("trained", {}).get("pass_at_1", {})

    stat_test = compute_statistical_test(
        baseline_p1.get("per_seed", []),
        trained_p1.get("per_seed", []),
    )

    # Compute improvement metrics
    improvement: dict[str, Any] = {}
    for metric in ["pass_at_1", "pass_at_3", "pass_at_5", "partial_credit"]:
        b = agg.get("baseline", {}).get(metric, {})
        t = agg.get("trained", {}).get(metric, {})
        b_mean = b.get("mean", 0)
        t_mean = t.get("mean", 0)
        diff = t_mean - b_mean
        rel_pct = (diff / b_mean * 100) if b_mean > 0 else 0
        improvement[metric] = {
            "mean": diff,
            "std": math.sqrt(b.get("std", 0) ** 2 + t.get("std", 0) ** 2),
            "relative_pct": rel_pct,
        }

    # Agent efficiency comparison
    b_turns = agg.get("baseline", {}).get("mean_agent_turns", {}).get("mean", 0)
    t_turns = agg.get("trained", {}).get("mean_agent_turns", {}).get("mean", 0)
    turns_reduction = b_turns - t_turns
    turns_reduction_pct = (turns_reduction / b_turns * 100) if b_turns > 0 else 0

    report = {
        "seeds": agg["seeds"],
        "n_seeds": agg["n_seeds"],
        "baseline": agg["baseline"],
        "trained": agg["trained"],
        "improvement": improvement,
        "statistical_test": stat_test,
        "per_task": agg["per_task"],
        "agent_efficiency": {
            "baseline_mean_turns": b_turns,
            "trained_mean_turns": t_turns,
            "turns_reduction": turns_reduction,
            "turns_reduction_pct": turns_reduction_pct,
        },
    }

    return report


def print_comparison_table(report: dict[str, Any]) -> None:
    """Pretty-print an ASCII comparison table.

    Args:
        report: Full comparison report from ``compare_results``.
    """
    n_seeds = report.get("n_seeds", 0)
    baseline = report.get("baseline", {})
    trained = report.get("trained", {})
    improvement = report.get("improvement", {})
    stat = report.get("statistical_test", {})
    efficiency = report.get("agent_efficiency", {})

    print()
    print(f"CooperBench Evaluation: Trained vs Baseline ({n_seeds} seeds)")
    print("=" * 70)
    print(f"{'Metric':<18} {'Baseline':>16} {'Trained':>16} {'Delta':>16}")
    print(f"{'-'*18} {'-'*16} {'-'*16} {'-'*16}")

    for metric in ["pass_at_1", "pass_at_3", "pass_at_5", "partial_credit"]:
        b = baseline.get(metric, {})
        t = trained.get(metric, {})
        imp = improvement.get(metric, {})
        b_str = f"{b.get('mean', 0):.3f} +/- {b.get('std', 0):.3f}" if b else "N/A"
        t_str = f"{t.get('mean', 0):.3f} +/- {t.get('std', 0):.3f}" if t else "N/A"
        d_mean = imp.get("mean", 0)
        d_pct = imp.get("relative_pct", 0)
        d_str = f"{d_mean:+.3f} ({d_pct:+.1f}%)" if imp else "N/A"
        print(f"{metric:<18} {b_str:>16} {t_str:>16} {d_str:>16}")

    # Agent turns
    b_turns = efficiency.get("baseline_mean_turns", 0)
    t_turns = efficiency.get("trained_mean_turns", 0)
    turns_red = efficiency.get("turns_reduction_pct", 0)
    b_turns_str = f"{b_turns:.1f}"
    t_turns_str = f"{t_turns:.1f}"
    turns_str = f"{t_turns - b_turns:+.1f} ({-turns_red:+.1f}%)" if b_turns > 0 else "N/A"
    print(f"{'mean_turns':<18} {b_turns_str:>16} {t_turns_str:>16} {turns_str:>16}")

    print("=" * 70)

    # Statistical test
    t_stat = stat.get("t_statistic", 0)
    p_val = stat.get("p_value", 1)
    sig_05 = stat.get("significant_at_005", False)
    sig_01 = stat.get("significant_at_001", False)
    sig_label = "p < 0.01" if sig_01 else ("p < 0.05" if sig_05 else "not significant")
    print(f"t-test: t={t_stat:.2f}, p={p_val:.4f} ({sig_label})")

    # Per-task breakdown
    per_task = report.get("per_task", [])
    if per_task:
        print(f"\nPer-task breakdown ({len(per_task)} tasks, sorted by improvement):")
        print(f"  {'Task ID':<40} {'Seeds':>5} {'Base':>6} {'Train':>6} {'Delta':>7}")
        print(f"  {'-'*40} {'-'*5} {'-'*6} {'-'*6} {'-'*7}")
        for task in per_task[:20]:  # Show top 20
            tid = task["task_id"][:40]
            seeds_n = task.get("seeds_evaluated", 0)
            b_p1 = task.get("baseline_pass1", 0)
            t_p1 = task.get("trained_pass1", 0)
            imp_val = task.get("improvement", 0)
            print(f"  {tid:<40} {seeds_n:>5} {b_p1:>6.3f} {t_p1:>6.3f} {imp_val:>+7.3f}")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    """Compute mean of a list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare trained vs baseline CooperBench evaluation results across seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory containing per-seed evaluation result JSONs",
    )
    parser.add_argument(
        "--seeds", default="42,43,44",
        help="Comma-separated seed values (default: 42,43,44)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for comparison report JSON",
    )
    parser.add_argument(
        "--wandb-project", default=None,
        help="W&B project for logging comparison results",
    )
    parser.add_argument(
        "--wandb-run-id", default=None,
        help="Existing W&B run to log to",
    )
    parser.add_argument(
        "--print-table", action="store_true",
        help="Print ASCII comparison table to stdout",
    )

    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    report = compare_results(args.results_dir, seeds)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Comparison report saved to {output_path}")

    if args.print_table or not args.output:
        print_comparison_table(report)

    # Log to W&B if configured
    if args.wandb_project and args.wandb_run_id:
        _log_comparison_to_wandb(report, args.wandb_project, args.wandb_run_id)


def _log_comparison_to_wandb(
    report: dict[str, Any],
    wandb_project: str,
    wandb_run_id: str,
) -> None:
    """Log comparison results to W&B.

    Args:
        report: Full comparison report.
        wandb_project: W&B project name.
        wandb_run_id: W&B run ID.
    """
    try:
        import wandb

        run = wandb.init(
            project=wandb_project,
            id=wandb_run_id,
            resume="allow",
        )
        if run is None:
            return

        # Log improvement metrics
        improvement = report.get("improvement", {})
        for metric, values in improvement.items():
            run.summary[f"comparison/{metric}_improvement"] = values.get("mean", 0)
            run.summary[f"comparison/{metric}_relative_pct"] = values.get("relative_pct", 0)

        # Log stat test
        stat = report.get("statistical_test", {})
        run.summary["comparison/t_statistic"] = stat.get("t_statistic", 0)
        run.summary["comparison/p_value"] = stat.get("p_value", 1)

        run.finish()
        print(f"Comparison logged to W&B run {wandb_run_id}")

    except Exception as e:
        print(f"WARNING: Failed to log comparison to W&B: {e}")


if __name__ == "__main__":
    main()
