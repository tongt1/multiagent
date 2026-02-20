#!/usr/bin/env python3
"""Create W&B dashboard and formal report for CooperBench evaluation.

Builds W&B panels for training curves, eval comparison, and agent behavior
across multi-seed training runs. Optionally creates a formal W&B Report
with narrative sections and embedded charts.

Usage:
    # Log comparison data to W&B
    python scripts/create_wandb_dashboard.py \\
        --wandb-project multiagent-debate-rl \\
        --run-ids seed42_run,seed43_run,seed44_run

    # Create formal report with comparison data
    python scripts/create_wandb_dashboard.py \\
        --wandb-project multiagent-debate-rl \\
        --run-ids seed42_run,seed43_run,seed44_run \\
        --comparison-json comparison_report.json \\
        --create-report

The core functions are importable for programmatic use:
    from scripts.create_wandb_dashboard import create_comparison_tables
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure reward-training root is on path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def create_comparison_tables(
    comparison_data: dict[str, Any],
) -> dict[str, Any]:
    """Build W&B-compatible table data from comparison report.

    Transforms the comparison report JSON (from compare_baseline.py) into
    structured table data suitable for W&B visualization.

    Args:
        comparison_data: Comparison report from ``compare_results()``.

    Returns:
        Dict with table definitions for W&B logging:
        - "per_task_table": columns and rows for per-task comparison
        - "overall_table": columns and rows for overall metrics
        - "efficiency_table": columns and rows for agent efficiency
    """
    tables = {}

    # Per-task comparison table
    per_task = comparison_data.get("per_task", [])
    tables["per_task_table"] = {
        "columns": ["task_id", "seeds_evaluated", "baseline_pass1",
                     "trained_pass1", "improvement"],
        "data": [
            [
                t.get("task_id", ""),
                t.get("seeds_evaluated", 0),
                t.get("baseline_pass1", 0),
                t.get("trained_pass1", 0),
                t.get("improvement", 0),
            ]
            for t in per_task
        ],
    }

    # Overall metrics table
    metrics = ["pass_at_1", "pass_at_3", "pass_at_5", "partial_credit"]
    baseline = comparison_data.get("baseline", {})
    trained = comparison_data.get("trained", {})
    improvement = comparison_data.get("improvement", {})

    overall_rows = []
    for metric in metrics:
        b = baseline.get(metric, {})
        t = trained.get(metric, {})
        imp = improvement.get(metric, {})
        overall_rows.append([
            metric,
            b.get("mean", 0),
            b.get("std", 0),
            t.get("mean", 0),
            t.get("std", 0),
            imp.get("mean", 0),
            imp.get("relative_pct", 0),
        ])

    tables["overall_table"] = {
        "columns": ["metric", "baseline_mean", "baseline_std",
                     "trained_mean", "trained_std",
                     "improvement", "improvement_pct"],
        "data": overall_rows,
    }

    # Agent efficiency table
    efficiency = comparison_data.get("agent_efficiency", {})
    tables["efficiency_table"] = {
        "columns": ["metric", "baseline", "trained", "change_pct"],
        "data": [
            [
                "mean_turns",
                efficiency.get("baseline_mean_turns", 0),
                efficiency.get("trained_mean_turns", 0),
                -efficiency.get("turns_reduction_pct", 0),
            ],
        ],
    }

    return tables


def log_comparison_to_wandb(
    comparison_data: dict[str, Any],
    wandb_project: str,
    run_name: str = "eval-comparison",
    entity: str | None = None,
) -> str | None:
    """Log comparison data to a dedicated W&B run.

    Creates a new W&B run specifically for the evaluation comparison,
    logging all tables and summary metrics.

    Args:
        comparison_data: Comparison report from ``compare_results()``.
        wandb_project: W&B project name.
        run_name: Name for the comparison run.
        entity: W&B entity (team/user). None for default.

    Returns:
        W&B run ID if successful, None otherwise.
    """
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. Install with: pip install wandb")
        return None

    try:
        run = wandb.init(
            project=wandb_project,
            name=run_name,
            entity=entity,
            job_type="eval-comparison",
            config={
                "seeds": comparison_data.get("seeds", []),
                "n_seeds": comparison_data.get("n_seeds", 0),
            },
        )

        if run is None:
            print("ERROR: wandb.init returned None")
            return None

        tables = create_comparison_tables(comparison_data)

        # Log per-task comparison table
        pt = tables["per_task_table"]
        per_task_wandb = wandb.Table(columns=pt["columns"], data=pt["data"])
        run.log({"eval/per_task_comparison": per_task_wandb})

        # Create bar chart: baseline vs trained per task
        if pt["data"]:
            run.log({
                "eval/per_task_bar_chart": wandb.plot.bar(
                    per_task_wandb,
                    "task_id",
                    "improvement",
                    title="Per-Task Pass@1 Improvement (Trained - Baseline)",
                ),
            })

        # Log overall metrics table
        ot = tables["overall_table"]
        overall_wandb = wandb.Table(columns=ot["columns"], data=ot["data"])
        run.log({"eval/overall_comparison": overall_wandb})

        # Log summary metrics
        improvement = comparison_data.get("improvement", {})
        for metric, values in improvement.items():
            run.summary[f"eval/{metric}_improvement"] = values.get("mean", 0)
            run.summary[f"eval/{metric}_relative_pct"] = values.get("relative_pct", 0)

        # Log statistical test results
        stat = comparison_data.get("statistical_test", {})
        run.summary["eval/t_statistic"] = stat.get("t_statistic", 0)
        run.summary["eval/p_value"] = stat.get("p_value", 1)
        run.summary["eval/significant_at_005"] = stat.get("significant_at_005", False)

        # Log efficiency metrics
        efficiency = comparison_data.get("agent_efficiency", {})
        run.summary["eval/turns_reduction_pct"] = efficiency.get("turns_reduction_pct", 0)

        # Save comparison data as artifact
        artifact = wandb.Artifact(
            name="cooperbench-comparison",
            type="eval-comparison",
            description="Multi-seed CooperBench evaluation comparison",
        )
        with artifact.new_file("comparison_report.json") as f:
            json.dump(comparison_data, f, indent=2, default=str)
        run.log_artifact(artifact)

        run_id = run.id
        run.finish()
        print(f"Comparison data logged to W&B run {run_id}")
        return run_id

    except Exception as e:
        print(f"ERROR: Failed to log to W&B: {e}")
        return None


def fetch_training_curves(
    wandb_project: str,
    run_ids: list[str],
    entity: str | None = None,
) -> dict[str, Any]:
    """Fetch training curve data from existing W&B runs.

    Queries the W&B API to retrieve training metrics (docker_reward,
    test_pass_rate, etc.) from the multi-seed training runs.

    Args:
        wandb_project: W&B project name.
        run_ids: List of W&B run IDs for the seed runs.
        entity: W&B entity (team/user).

    Returns:
        Dict mapping run_id -> {metric_name: [(step, value), ...]}.
    """
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed")
        return {}

    api = wandb.Api()
    curves: dict[str, Any] = {}

    metrics_to_fetch = [
        "cooperbench/agentic/docker_reward",
        "cooperbench/agentic/docker_avg_test_rate",
        "cooperbench/agentic/mean_steps",
        "cooperbench/agentic/container_error_rate",
        "train/loss",
    ]

    for run_id in run_ids:
        try:
            path = f"{entity}/{wandb_project}/{run_id}" if entity else f"{wandb_project}/{run_id}"
            run = api.run(path)
            run_curves: dict[str, list] = {}

            history = run.scan_history(keys=metrics_to_fetch + ["_step"])
            for row in history:
                step = row.get("_step", 0)
                for metric in metrics_to_fetch:
                    if metric in row and row[metric] is not None:
                        if metric not in run_curves:
                            run_curves[metric] = []
                        run_curves[metric].append((step, row[metric]))

            curves[run_id] = {
                "name": run.name,
                "seed": run.config.get("seed", "unknown"),
                "metrics": run_curves,
            }
            print(f"  Fetched {sum(len(v) for v in run_curves.values())} data points from {run.name}")

        except Exception as e:
            print(f"  WARNING: Failed to fetch run {run_id}: {e}")
            curves[run_id] = {"error": str(e)}

    return curves


def create_wandb_report(
    wandb_project: str,
    run_ids: list[str],
    comparison_data: dict[str, Any] | None = None,
    title: str = "CooperBench Agentic RL Training: Evaluation Report",
    entity: str | None = None,
) -> str | None:
    """Create a formal W&B Report with embedded panels and narrative.

    Attempts to create a programmatic W&B Report. If the reports API is
    not available, falls back to logging data and printing instructions
    for manual report creation.

    Args:
        wandb_project: W&B project name.
        run_ids: W&B run IDs for the 3 seed runs.
        comparison_data: Optional comparison report from compare_baseline.py.
        title: Report title.
        entity: W&B entity (team/user).

    Returns:
        Report URL if successful, None otherwise.
    """
    try:
        import wandb
        import wandb.apis.reports as wr
    except (ImportError, AttributeError):
        print("WARNING: wandb.apis.reports not available. "
              "Falling back to data logging + manual report instructions.")
        return _fallback_report_instructions(wandb_project, run_ids, comparison_data, entity)

    try:
        report = wr.Report(
            project=wandb_project,
            entity=entity,
            title=title,
            description="Multi-seed evaluation comparing trained Qwen3-4B vs baseline on CooperBench",
        )

        # Section 1: Overview
        report.blocks = [
            wr.H1(text="Overview"),
            wr.P(text=(
                "This report presents the evaluation results of Qwen3-4B fine-tuned "
                "with GRPO (agentic mode, OpenHands Docker loop) on CooperBench "
                "cooperative coding tasks. Training used DAPO loss, 250 steps, "
                "3 seeds (42, 43, 44) for variance measurement."
            )),
            wr.P(text=(
                "Evaluation uses a deterministic 80/20 train/eval split rotated per seed, "
                "with 10 rollouts per held-out task. The step-0 checkpoint serves as the "
                "pre-training baseline."
            )),
        ]

        # Section 2: Training Curves
        report.blocks.extend([
            wr.H1(text="Training Curves"),
            wr.P(text="Docker reward and test pass rate over training steps, all 3 seeds overlaid."),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(
                        project=wandb_project,
                        entity=entity or "",
                        filters={"$or": [{"config.run_id": rid} for rid in run_ids]},
                    ),
                ],
                panels=[
                    wr.LinePlot(
                        title="Docker Reward vs Training Step",
                        x="Step",
                        y=["cooperbench/agentic/docker_reward"],
                    ),
                    wr.LinePlot(
                        title="Docker Test Pass Rate vs Training Step",
                        x="Step",
                        y=["cooperbench/agentic/docker_avg_test_rate"],
                    ),
                ],
            ),
        ])

        # Section 3: Baseline vs Trained
        report.blocks.extend([
            wr.H1(text="Baseline vs Trained Comparison"),
            wr.P(text="Per-task pass@1 comparison and overall metrics."),
        ])

        if comparison_data:
            stat = comparison_data.get("statistical_test", {})
            imp = comparison_data.get("improvement", {}).get("pass_at_1", {})
            report.blocks.append(wr.P(text=(
                f"**pass@1 improvement: {imp.get('mean', 0):+.3f} "
                f"({imp.get('relative_pct', 0):+.1f}%)** | "
                f"t={stat.get('t_statistic', 0):.2f}, "
                f"p={stat.get('p_value', 1):.4f}"
            )))

        # Section 4: Agent Behavior
        report.blocks.extend([
            wr.H1(text="Agent Behavior"),
            wr.P(text="Agent turns, container error rate, and finish rate over training."),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(
                        project=wandb_project,
                        entity=entity or "",
                        filters={"$or": [{"config.run_id": rid} for rid in run_ids]},
                    ),
                ],
                panels=[
                    wr.LinePlot(
                        title="Mean Agent Turns per Rollout",
                        x="Step",
                        y=["cooperbench/agentic/mean_steps"],
                    ),
                    wr.LinePlot(
                        title="Container Error Rate",
                        x="Step",
                        y=["cooperbench/agentic/container_error_rate"],
                    ),
                ],
            ),
        ])

        # Section 5: Conclusions (template)
        report.blocks.extend([
            wr.H1(text="Conclusions"),
            wr.P(text=(
                "[TODO: Fill in conclusions based on results.] "
                "Key findings: ... "
                "Observations on per-task improvements: ... "
                "Agent efficiency changes: ... "
                "Recommendations for next steps: ..."
            )),
        ])

        report.save()
        url = report.url
        print(f"W&B Report created: {url}")
        return url

    except Exception as e:
        print(f"WARNING: Failed to create W&B Report: {e}")
        return _fallback_report_instructions(wandb_project, run_ids, comparison_data, entity)


def _fallback_report_instructions(
    wandb_project: str,
    run_ids: list[str],
    comparison_data: dict[str, Any] | None,
    entity: str | None,
) -> str | None:
    """Print instructions for manual W&B report creation.

    Args:
        wandb_project: W&B project name.
        run_ids: W&B run IDs.
        comparison_data: Optional comparison data.
        entity: W&B entity.

    Returns:
        None (prints instructions to stdout).
    """
    # Log the data to a W&B run so it's available for manual report creation
    run_id = None
    if comparison_data:
        run_id = log_comparison_to_wandb(
            comparison_data, wandb_project, "eval-comparison",
        )

    print()
    print("=" * 60)
    print("Manual W&B Report Creation Instructions")
    print("=" * 60)
    print(f"1. Go to: https://wandb.ai/{entity or 'YOUR_ENTITY'}/{wandb_project}")
    print(f"2. Click 'Reports' > 'Create Report'")
    print(f"3. Add run filter for runs: {', '.join(run_ids)}")
    print(f"4. Add panels:")
    print(f"   - Line chart: cooperbench/agentic/docker_reward vs Step")
    print(f"   - Line chart: cooperbench/agentic/docker_avg_test_rate vs Step")
    print(f"   - Line chart: cooperbench/agentic/mean_steps vs Step")
    if run_id:
        print(f"5. Comparison data logged to run: {run_id}")
        print(f"   - Table: eval/per_task_comparison")
        print(f"   - Table: eval/overall_comparison")
    print("=" * 60)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create W&B dashboard and report for CooperBench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--wandb-project", required=True,
        help="W&B project name",
    )
    parser.add_argument(
        "--run-ids", required=True,
        help="Comma-separated W&B run IDs for the 3 seed runs",
    )
    parser.add_argument(
        "--comparison-json", default=None,
        help="Path to comparison report JSON from compare_baseline.py",
    )
    parser.add_argument(
        "--create-report", action="store_true",
        help="Create a formal W&B Report",
    )
    parser.add_argument(
        "--report-title",
        default="CooperBench Agentic RL Training: Evaluation Report",
        help="Title for the W&B Report",
    )
    parser.add_argument(
        "--entity", default=None,
        help="W&B entity (team/user)",
    )

    args = parser.parse_args()
    run_ids = [r.strip() for r in args.run_ids.split(",")]

    # Load comparison data if provided
    comparison_data = None
    if args.comparison_json:
        with open(args.comparison_json) as f:
            comparison_data = json.load(f)

    # Fetch training curves
    print("Fetching training curves from W&B...")
    curves = fetch_training_curves(
        args.wandb_project, run_ids, args.entity,
    )

    # Log comparison data
    if comparison_data:
        print("Logging comparison data to W&B...")
        log_comparison_to_wandb(
            comparison_data, args.wandb_project,
            run_name="eval-comparison",
            entity=args.entity,
        )

    # Create report
    if args.create_report:
        print("Creating W&B Report...")
        url = create_wandb_report(
            args.wandb_project,
            run_ids,
            comparison_data,
            args.report_title,
            args.entity,
        )
        if url:
            print(f"\nReport URL: {url}")

    print("\nDashboard setup complete.")


if __name__ == "__main__":
    main()
