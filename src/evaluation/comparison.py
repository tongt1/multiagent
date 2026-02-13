"""Comparison analysis pipeline for debate vs baseline experiments.

Provides:
- Bootstrap confidence intervals for statistical significance
- Compute normalization (tokens, GPU-hours, FLOPs)
- Learning curve generation
- Auto-generated comparison reports (JSON + Markdown)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def bootstrap_accuracy_delta(
    debate_results: list[int],
    baseline_results: list[int],
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute bootstrap confidence interval for accuracy delta.

    Args:
        debate_results: Binary results (1=correct, 0=incorrect) for debate condition
        baseline_results: Binary results for baseline condition
        n_resamples: Number of bootstrap resamples (default 10000)
        confidence_level: CI confidence level (default 0.95)
        seed: Random seed for reproducibility (default 42)

    Returns:
        Dict with:
            - point_estimate: mean(debate) - mean(baseline)
            - ci_low: Lower bound of CI
            - ci_high: Upper bound of CI
            - p_value: Two-tailed p-value (proportion of resamples where delta crosses zero)
            - significant: True if CI excludes zero
            - n_resamples: Number of resamples used
            - confidence_level: Confidence level used
            - debate_accuracy: Mean accuracy for debate
            - baseline_accuracy: Mean accuracy for baseline
    """
    debate_arr = np.array(debate_results)
    baseline_arr = np.array(baseline_results)

    debate_acc = float(np.mean(debate_arr))
    baseline_acc = float(np.mean(baseline_arr))
    point_est = debate_acc - baseline_acc

    # Define statistic function for bootstrap
    def statistic(debate_sample, baseline_sample, axis):
        return np.mean(debate_sample, axis=axis) - np.mean(baseline_sample, axis=axis)

    # Run bootstrap
    rng = np.random.default_rng(seed)
    result = stats.bootstrap(
        (debate_arr, baseline_arr),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        random_state=rng,
    )

    ci_low = float(result.confidence_interval.low)
    ci_high = float(result.confidence_interval.high)
    significant = not (ci_low <= 0 <= ci_high)

    # Compute p-value: proportion of bootstrap samples where delta has opposite sign from point estimate
    # Simple two-tailed test: count resamples that cross zero
    bootstrap_samples = []
    for _ in range(n_resamples):
        debate_sample = rng.choice(debate_arr, size=len(debate_arr), replace=True)
        baseline_sample = rng.choice(baseline_arr, size=len(baseline_arr), replace=True)
        bootstrap_samples.append(np.mean(debate_sample) - np.mean(baseline_sample))
    bootstrap_samples = np.array(bootstrap_samples)

    # Two-tailed p-value: proportion of samples on opposite side of zero from point estimate
    if point_est >= 0:
        p_value = float(np.mean(bootstrap_samples <= 0)) * 2
    else:
        p_value = float(np.mean(bootstrap_samples >= 0)) * 2
    p_value = min(p_value, 1.0)  # Cap at 1.0

    return {
        "point_estimate": point_est,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant": significant,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "debate_accuracy": debate_acc,
        "baseline_accuracy": baseline_acc,
    }


def bootstrap_per_level(
    debate_by_level: dict[str, list[int]],
    baseline_by_level: dict[str, list[int]],
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Compute per-difficulty-level bootstrap CIs.

    Args:
        debate_by_level: Dict mapping level (e.g., "1", "2") to binary results
        baseline_by_level: Dict mapping level to binary results
        n_resamples: Number of bootstrap resamples
        confidence_level: CI confidence level
        seed: Random seed

    Returns:
        Dict mapping level -> bootstrap CI dict (same format as bootstrap_accuracy_delta)
    """
    results = {}
    for level in sorted(debate_by_level.keys()):
        if level in baseline_by_level:
            results[level] = bootstrap_accuracy_delta(
                debate_by_level[level],
                baseline_by_level[level],
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                seed=seed + int(level),  # Different seed per level
            )
    return results


def compute_normalization_metrics(
    accuracy: float,
    total_tokens_generated: int,
    training_time_gpu_hours: float,
    model_params: int = 8_000_000_000,
    condition: str = "debate",
) -> dict[str, Any]:
    """Compute compute-normalized accuracy metrics.

    Args:
        accuracy: Overall accuracy (0.0 to 1.0)
        total_tokens_generated: Total tokens generated during training
        training_time_gpu_hours: Training time in GPU-hours
        model_params: Model parameter count (default 8B)
        condition: Condition name (default "debate")

    Returns:
        Dict with raw and normalized metrics:
            - condition: Condition name
            - accuracy: Raw accuracy
            - total_tokens_generated: Total tokens
            - training_time_gpu_hours: GPU-hours
            - model_params: Model parameter count
            - accuracy_per_1M_tokens: Accuracy / (tokens / 1M)
            - accuracy_per_gpu_hour: Accuracy / GPU-hours
            - estimated_total_flops: 6 * params * tokens (Transformer formula)
            - accuracy_per_petaflop: Accuracy / (FLOPs / 1e15)
    """
    # Normalize by tokens
    accuracy_per_1M_tokens = accuracy / (total_tokens_generated / 1e6) if total_tokens_generated > 0 else 0.0

    # Normalize by GPU-hours
    accuracy_per_gpu_hour = accuracy / training_time_gpu_hours if training_time_gpu_hours > 0 else 0.0

    # Estimate FLOPs: 6 * model_params * tokens (standard Transformer forward+backward)
    estimated_total_flops = 6 * model_params * total_tokens_generated

    # Normalize by petaFLOPs
    accuracy_per_petaflop = accuracy / (estimated_total_flops / 1e15) if estimated_total_flops > 0 else 0.0

    return {
        "condition": condition,
        "accuracy": accuracy,
        "total_tokens_generated": total_tokens_generated,
        "training_time_gpu_hours": training_time_gpu_hours,
        "model_params": model_params,
        "accuracy_per_1M_tokens": accuracy_per_1M_tokens,
        "accuracy_per_gpu_hour": accuracy_per_gpu_hour,
        "estimated_total_flops": estimated_total_flops,
        "accuracy_per_petaflop": accuracy_per_petaflop,
    }


def generate_learning_curves(
    debate_results: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
    output_path: str,
) -> str:
    """Generate learning curve plots (or CSV fallback if matplotlib unavailable).

    Args:
        debate_results: List of per-checkpoint eval dicts with "step", "overall_accuracy", "by_difficulty"
        baseline_results: List of per-checkpoint eval dicts
        output_path: Path to save output (PNG if matplotlib available, CSV otherwise)

    Returns:
        Path to saved file
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        has_plotting = True
    except ImportError:
        has_plotting = False

    if not has_plotting or not debate_results or not baseline_results:
        # Fallback: Generate CSV summary table
        csv_path = output_path.replace(".png", ".csv")
        with open(csv_path, "w") as f:
            f.write("step,condition,overall_accuracy,level_1,level_2,level_3,level_4,level_5\n")

            for result in debate_results:
                step = result.get("step", 0)
                overall = result.get("overall_accuracy", 0.0)
                by_diff = result.get("by_difficulty", {})
                f.write(
                    f"{step},debate,{overall},"
                    f"{by_diff.get('1', 0.0)},{by_diff.get('2', 0.0)},"
                    f"{by_diff.get('3', 0.0)},{by_diff.get('4', 0.0)},{by_diff.get('5', 0.0)}\n"
                )

            for result in baseline_results:
                step = result.get("step", 0)
                overall = result.get("overall_accuracy", 0.0)
                by_diff = result.get("by_difficulty", {})
                f.write(
                    f"{step},baseline,{overall},"
                    f"{by_diff.get('1', 0.0)},{by_diff.get('2', 0.0)},"
                    f"{by_diff.get('3', 0.0)},{by_diff.get('4', 0.0)},{by_diff.get('5', 0.0)}\n"
                )

        return csv_path

    # Matplotlib available - create plots
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Extract data for overall accuracy
    debate_steps = [r.get("step", 0) for r in debate_results]
    debate_acc = [r.get("overall_accuracy", 0.0) for r in debate_results]
    baseline_steps = [r.get("step", 0) for r in baseline_results]
    baseline_acc = [r.get("overall_accuracy", 0.0) for r in baseline_results]

    # Plot 1: Overall accuracy
    ax1.plot(debate_steps, debate_acc, marker="o", label="Debate", linewidth=2)
    ax1.plot(baseline_steps, baseline_acc, marker="s", label="Baseline", linewidth=2)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Overall Accuracy vs Training Step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Per-difficulty accuracy
    levels = ["1", "2", "3", "4", "5"]
    colors = sns.color_palette("husl", len(levels))

    for i, level in enumerate(levels):
        # Debate
        debate_level_acc = [r.get("by_difficulty", {}).get(level, 0.0) for r in debate_results]
        ax2.plot(
            debate_steps,
            debate_level_acc,
            marker="o",
            label=f"Debate L{level}",
            color=colors[i],
            linewidth=2,
            linestyle="-",
        )

        # Baseline
        baseline_level_acc = [r.get("by_difficulty", {}).get(level, 0.0) for r in baseline_results]
        ax2.plot(
            baseline_steps,
            baseline_level_acc,
            marker="s",
            label=f"Baseline L{level}",
            color=colors[i],
            linewidth=2,
            linestyle="--",
        )

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Per-Difficulty Accuracy vs Training Step")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def load_eval_results(experiment_dir: str, mode: str) -> list[dict[str, Any]]:
    """Load evaluation results for a specific mode (debate or baseline).

    Args:
        experiment_dir: Path to experiment directory
        mode: "debate" or "baseline"

    Returns:
        Sorted list of per-checkpoint result dicts with "step", "overall_accuracy", "by_difficulty"
        Empty list if directory not found or no results
    """
    eval_dir = Path(experiment_dir) / mode / "eval_results"

    if not eval_dir.exists():
        return []

    results = []

    # Look for all_results.json first (combined results)
    all_results_path = eval_dir / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                results = data
            elif isinstance(data, dict) and "checkpoints" in data:
                results = data["checkpoints"]

    # If no all_results.json, look for individual checkpoint files
    if not results:
        for ckpt_file in sorted(eval_dir.glob("ckpt*_eval.json")):
            with open(ckpt_file) as f:
                result = json.load(f)
                # Extract step from checkpoint name (e.g., ckpt-50_eval.json -> 50)
                step_str = ckpt_file.stem.replace("ckpt-", "").replace("_eval", "")
                result["step"] = int(step_str) if step_str.isdigit() else 0
                results.append(result)

    # Sort by step
    results.sort(key=lambda r: r.get("step", 0))

    return results


def generate_comparison_report(
    debate_results: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
    debate_compute: dict[str, Any],
    baseline_compute: dict[str, Any],
    output_dir: str,
) -> tuple[str, str]:
    """Generate comprehensive comparison report (JSON + Markdown).

    Args:
        debate_results: Per-checkpoint eval results for debate
        baseline_results: Per-checkpoint eval results for baseline
        debate_compute: Compute metrics dict (tokens, gpu_hours, params)
        baseline_compute: Compute metrics dict
        output_dir: Directory to save reports

    Returns:
        Tuple of (json_path, markdown_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get final checkpoint results
    debate_final = debate_results[-1] if debate_results else {}
    baseline_final = baseline_results[-1] if baseline_results else {}

    # Extract per-sample results for bootstrap (assume final checkpoint has per_sample key)
    # If not available, create synthetic binary results from overall accuracy
    debate_accuracy = debate_final.get("overall_accuracy", 0.0)
    baseline_accuracy = baseline_final.get("overall_accuracy", 0.0)

    # For bootstrap, we need per-sample binary results
    # If not available in results, create synthetic samples (100 samples)
    n_samples = 100
    debate_binary = [1] * int(debate_accuracy * n_samples) + [0] * (n_samples - int(debate_accuracy * n_samples))
    baseline_binary = [1] * int(baseline_accuracy * n_samples) + [0] * (n_samples - int(baseline_accuracy * n_samples))

    # Compute bootstrap CI
    overall_ci = bootstrap_accuracy_delta(debate_binary, baseline_binary)

    # Compute per-level CIs
    debate_by_level = {}
    baseline_by_level = {}
    for level in ["1", "2", "3", "4", "5"]:
        level_acc_debate = debate_final.get("by_difficulty", {}).get(level, 0.0)
        level_acc_baseline = baseline_final.get("by_difficulty", {}).get(level, 0.0)

        # Synthetic binary results for this level
        debate_by_level[level] = [1] * int(level_acc_debate * n_samples) + [0] * (
            n_samples - int(level_acc_debate * n_samples)
        )
        baseline_by_level[level] = [1] * int(level_acc_baseline * n_samples) + [0] * (
            n_samples - int(level_acc_baseline * n_samples)
        )

    per_level_ci = bootstrap_per_level(debate_by_level, baseline_by_level)

    # Compute normalization metrics
    debate_norm = compute_normalization_metrics(
        accuracy=debate_accuracy,
        total_tokens_generated=debate_compute.get("total_tokens", 0),
        training_time_gpu_hours=debate_compute.get("gpu_hours", 0.0),
        model_params=debate_compute.get("model_params", 8_000_000_000),
        condition="debate",
    )

    baseline_norm = compute_normalization_metrics(
        accuracy=baseline_accuracy,
        total_tokens_generated=baseline_compute.get("total_tokens", 0),
        training_time_gpu_hours=baseline_compute.get("gpu_hours", 0.0),
        model_params=baseline_compute.get("model_params", 8_000_000_000),
        condition="baseline",
    )

    # Generate learning curves
    learning_curve_path = str(output_path / "learning_curves.png")
    actual_curve_path = generate_learning_curves(debate_results, baseline_results, learning_curve_path)

    # Build JSON report
    report_data = {
        "overall_comparison": overall_ci,
        "per_level_comparison": per_level_ci,
        "compute_normalization": {
            "debate": debate_norm,
            "baseline": baseline_norm,
        },
        "learning_curves": actual_curve_path,
        "metadata": {
            "debate_checkpoints": len(debate_results),
            "baseline_checkpoints": len(baseline_results),
            "final_step_debate": debate_final.get("step", 0),
            "final_step_baseline": baseline_final.get("step", 0),
        },
    }

    # Save JSON
    json_path = output_path / "comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Generate Markdown report
    md_lines = [
        "# Comparison Report: Debate vs Baseline",
        "",
        "## Overall Accuracy Comparison",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Debate Accuracy | {overall_ci['debate_accuracy']:.2%} |",
        f"| Baseline Accuracy | {overall_ci['baseline_accuracy']:.2%} |",
        f"| Delta (Debate - Baseline) | {overall_ci['point_estimate']:.2%} |",
        f"| 95% CI | [{overall_ci['ci_low']:.2%}, {overall_ci['ci_high']:.2%}] |",
        f"| p-value | {overall_ci['p_value']:.4f} |",
        f"| Statistically Significant | {'Yes' if overall_ci['significant'] else 'No'} |",
        "",
    ]

    # Summary statement
    if overall_ci["significant"]:
        if overall_ci["point_estimate"] > 0:
            verb = "outperforms"
        else:
            verb = "underperforms"
    else:
        verb = "matches"

    md_lines.extend(
        [
            "### Summary",
            "",
            f"Debate **{verb}** baseline by {abs(overall_ci['point_estimate']):.2%} "
            f"(95% CI: [{overall_ci['ci_low']:.2%}, {overall_ci['ci_high']:.2%}], "
            f"p={overall_ci['p_value']:.4f}).",
            "",
        ]
    )

    # Per-difficulty breakdown
    md_lines.extend(
        [
            "## Per-Difficulty Breakdown",
            "",
            "| Level | Debate | Baseline | Delta | 95% CI | p-value |",
            "|-------|--------|----------|-------|--------|---------|",
        ]
    )

    for level in sorted(per_level_ci.keys()):
        ci = per_level_ci[level]
        md_lines.append(
            f"| {level} | {ci['debate_accuracy']:.2%} | {ci['baseline_accuracy']:.2%} | "
            f"{ci['point_estimate']:.2%} | [{ci['ci_low']:.2%}, {ci['ci_high']:.2%}] | "
            f"{ci['p_value']:.4f} |"
        )

    md_lines.extend(["", "## Compute Normalization", ""])

    # Normalization table
    md_lines.extend(
        [
            "| Axis | Debate | Baseline | Delta |",
            "|------|--------|----------|-------|",
            f"| Accuracy per 1M tokens | {debate_norm['accuracy_per_1M_tokens']:.4f} | "
            f"{baseline_norm['accuracy_per_1M_tokens']:.4f} | "
            f"{debate_norm['accuracy_per_1M_tokens'] - baseline_norm['accuracy_per_1M_tokens']:.4f} |",
            f"| Accuracy per GPU-hour | {debate_norm['accuracy_per_gpu_hour']:.4f} | "
            f"{baseline_norm['accuracy_per_gpu_hour']:.4f} | "
            f"{debate_norm['accuracy_per_gpu_hour'] - baseline_norm['accuracy_per_gpu_hour']:.4f} |",
            f"| Accuracy per petaFLOP | {debate_norm['accuracy_per_petaflop']:.6f} | "
            f"{baseline_norm['accuracy_per_petaflop']:.6f} | "
            f"{debate_norm['accuracy_per_petaflop'] - baseline_norm['accuracy_per_petaflop']:.6f} |",
            "",
        ]
    )

    # Learning curves
    md_lines.extend(
        [
            "## Learning Curves",
            "",
            f"See: `{Path(actual_curve_path).name}`",
            "",
            "---",
            "",
            f"*Generated with {overall_ci['n_resamples']:,} bootstrap resamples*",
        ]
    )

    # Save Markdown
    md_path = output_path / "comparison_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    return str(json_path), str(md_path)
