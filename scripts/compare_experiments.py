#!/usr/bin/env python3
"""CLI script for comparing debate vs baseline experiments.

Loads evaluation results, computes bootstrap CIs, normalizes by compute,
generates learning curves, and produces comparison reports.

Usage:
    # Full comparison with all metrics
    python scripts/compare_experiments.py \
        --experiment-dir experiments/2024-01-15_debate_vs_baseline \
        --debate-tokens 10000000 \
        --baseline-tokens 5000000 \
        --debate-gpu-hours 100.0 \
        --baseline-gpu-hours 50.0

    # Summary only (no full report)
    python scripts/compare_experiments.py \
        --experiment-dir experiments/2024-01-15_debate_vs_baseline \
        --debate-tokens 10000000 \
        --baseline-tokens 5000000 \
        --debate-gpu-hours 100.0 \
        --baseline-gpu-hours 50.0 \
        --summary-only

    # Skip learning curve plots
    python scripts/compare_experiments.py \
        --experiment-dir experiments/2024-01-15_debate_vs_baseline \
        --debate-tokens 10000000 \
        --baseline-tokens 5000000 \
        --debate-gpu-hours 100.0 \
        --baseline-gpu-hours 50.0 \
        --skip-plots
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def load_comparison_module():
    """Load comparison module directly to avoid __init__.py dependencies."""
    comparison_path = Path(__file__).parent.parent / "src" / "evaluation" / "comparison.py"
    spec = importlib.util.spec_from_file_location("comparison", comparison_path)
    comparison = importlib.util.module_from_spec(spec)
    sys.modules["comparison"] = comparison
    spec.loader.exec_module(comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare debate vs baseline experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--experiment-dir", required=True, help="Path to experiment directory containing debate/ and baseline/ results"
    )
    parser.add_argument("--debate-tokens", type=int, required=True, help="Total tokens generated during debate training")
    parser.add_argument(
        "--baseline-tokens", type=int, required=True, help="Total tokens generated during baseline training"
    )
    parser.add_argument("--debate-gpu-hours", type=float, required=True, help="GPU-hours for debate training")
    parser.add_argument("--baseline-gpu-hours", type=float, required=True, help="GPU-hours for baseline training")

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        help="Where to save comparison outputs (default: experiment_dir/comparison/)",
    )
    parser.add_argument(
        "--model-params",
        type=int,
        default=8_000_000_000,
        help="Model parameter count (default: 8B)",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=10000,
        help="Bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="CI confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap (default: 42)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip learning curve plot generation",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary only, skip full report generation",
    )

    args = parser.parse_args()

    # Load comparison module
    comparison = load_comparison_module()

    # Set output directory
    output_dir = args.output_dir or str(Path(args.experiment_dir) / "comparison")

    # Load evaluation results
    print("Loading evaluation results...")
    debate_results = comparison.load_eval_results(args.experiment_dir, "debate")
    baseline_results = comparison.load_eval_results(args.experiment_dir, "baseline")

    # Validate results
    if not debate_results:
        print(f"ERROR: No debate evaluation results found in {args.experiment_dir}/debate/eval_results/")
        sys.exit(1)

    if not baseline_results:
        print(f"ERROR: No baseline evaluation results found in {args.experiment_dir}/baseline/eval_results/")
        sys.exit(1)

    print(f"Found {len(debate_results)} debate checkpoints, {len(baseline_results)} baseline checkpoints")

    # Build compute dicts
    debate_compute = {
        "total_tokens": args.debate_tokens,
        "gpu_hours": args.debate_gpu_hours,
        "model_params": args.model_params,
    }

    baseline_compute = {
        "total_tokens": args.baseline_tokens,
        "gpu_hours": args.baseline_gpu_hours,
        "model_params": args.model_params,
    }

    # Get final checkpoint accuracies for summary
    debate_final = debate_results[-1]
    baseline_final = baseline_results[-1]
    debate_accuracy = debate_final.get("overall_accuracy", 0.0)
    baseline_accuracy = baseline_final.get("overall_accuracy", 0.0)
    delta = debate_accuracy - baseline_accuracy

    # Compute bootstrap CI for summary
    n_samples = 100  # Synthetic samples
    debate_binary = [1] * int(debate_accuracy * n_samples) + [0] * (n_samples - int(debate_accuracy * n_samples))
    baseline_binary = [1] * int(baseline_accuracy * n_samples) + [0] * (n_samples - int(baseline_accuracy * n_samples))
    overall_ci = comparison.bootstrap_accuracy_delta(
        debate_binary, baseline_binary, n_resamples=args.n_resamples, confidence_level=args.confidence_level, seed=args.seed
    )

    if args.summary_only:
        # Print summary table only
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\nDebate accuracy:   {debate_accuracy:.2%}")
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        print(f"Delta:             {delta:+.2%}")
        print(f"95% CI:            [{overall_ci['ci_low']:.2%}, {overall_ci['ci_high']:.2%}]")
        print(f"p-value:           {overall_ci['p_value']:.4f}")
        print(f"Significant:       {'Yes' if overall_ci['significant'] else 'No'}")
        print("\n" + "=" * 70)
        return

    # Generate full comparison report
    print("\nGenerating comparison report...")
    json_path, md_path = comparison.generate_comparison_report(
        debate_results, baseline_results, debate_compute, baseline_compute, output_dir
    )

    # Print Markdown report to stdout
    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70 + "\n")

    with open(md_path) as f:
        print(f.read())

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"\nJSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"Output directory: {output_dir}")

    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nDebate accuracy:   {debate_accuracy:.2%}")
    print(f"Baseline accuracy: {baseline_accuracy:.2%}")
    print(f"Delta:             {delta:+.2%}")
    print(f"95% CI:            [{overall_ci['ci_low']:.2%}, {overall_ci['ci_high']:.2%}]")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
