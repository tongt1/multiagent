"""Overview page showing aggregate training statistics and multi-run comparison.

This page provides:
- Per-run summary stats (total steps, rollouts, data source info)
- Reward curves over training steps (multi-run overlay)
- Cross-run comparison charts (when 2+ runs loaded)
- Per-role loss trends (if available in data)
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st

from streamlit_viewer.config import DATA_SOURCE_PARQUET, DATA_SOURCE_WANDB
from streamlit_viewer.lib.data_loader import (
    discover_parquet_files,
    load_parquet_step,
    compute_step_summary,
)
from streamlit_viewer.lib.visualizations import (
    render_reward_curve,
    render_reward_distribution,
    render_per_role_loss_curves,
)


def _load_all_steps_data(run_config: dict) -> tuple[list[dict], str]:
    """Load summary data for all training steps in a run.

    Args:
        run_config: Run configuration dict with keys: label, source_type, path_or_run

    Returns:
        Tuple of (steps_data, error_message) where:
        - steps_data: List of {"step": int, "avg_reward": float, "reward_std": float,
                       "num_prompts": int, "num_rollouts": int}
        - error_message: Error string if loading failed, empty string if success
    """
    if run_config["source_type"] == DATA_SOURCE_PARQUET:
        parquet_dir = run_config["path_or_run"]
        step_files = discover_parquet_files(parquet_dir)

        if not step_files:
            return [], f"No Parquet files found in {parquet_dir}"

        steps_data = []
        for sf in step_files:
            df = load_parquet_step(sf["path"])
            if not df.empty:
                summary = compute_step_summary(df)
                steps_data.append({
                    "step": sf["step"],
                    "avg_reward": summary["avg_reward"],
                    "reward_std": summary["reward_std"],
                    "num_prompts": summary["num_prompts"],
                    "num_rollouts": summary["num_rollouts"],
                })

        return steps_data, ""

    elif run_config["source_type"] == DATA_SOURCE_WANDB:
        return [], "W&B data source not yet implemented for Overview page"

    else:
        return [], f"Unknown data source type: {run_config['source_type']}"


def _extract_per_role_losses(df: pd.DataFrame) -> Optional[dict]:
    """Extract per-role loss data from rollout DataFrame.

    Looks for per-role loss in reward_metrics JSON field or dedicated columns.

    Args:
        df: DataFrame with rollout data

    Returns:
        Dict with {"loss_solver": float, "loss_verifier": float, "loss_judge": float}
        or None if per-role loss data not available
    """
    # Try to extract from reward_metrics JSON column
    if "reward_metrics" in df.columns:
        try:
            # Parse first non-null reward_metrics entry
            for metrics_json in df["reward_metrics"].dropna():
                if isinstance(metrics_json, str):
                    metrics = json.loads(metrics_json)
                elif isinstance(metrics_json, dict):
                    metrics = metrics_json
                else:
                    continue

                # Check for per-role loss keys
                if all(k in metrics for k in ["loss_solver", "loss_verifier", "loss_judge"]):
                    return {
                        "loss_solver": metrics["loss_solver"],
                        "loss_verifier": metrics["loss_verifier"],
                        "loss_judge": metrics["loss_judge"],
                    }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Try dedicated columns
    if all(col in df.columns for col in ["loss_solver", "loss_verifier", "loss_judge"]):
        return {
            "loss_solver": df["loss_solver"].mean(),
            "loss_verifier": df["loss_verifier"].mean(),
            "loss_judge": df["loss_judge"].mean(),
        }

    return None


def render_overview(run_configs: list[dict]) -> None:
    """Render overview dashboard with aggregate statistics and comparison.

    Args:
        run_configs: List of run configuration dicts. Each dict has keys:
                     label, source_type, path_or_run
    """
    st.header("Overview Dashboard")

    if not run_configs:
        st.warning("No runs configured. Add a data source in the sidebar.")
        return

    # ========================================================================
    # Load data for all runs
    # ========================================================================

    all_runs_data = []
    for run_config in run_configs:
        with st.spinner(f"Loading data for {run_config['label']}..."):
            steps_data, error = _load_all_steps_data(run_config)
            if error:
                st.error(f"**{run_config['label']}**: {error}")
            else:
                all_runs_data.append({
                    "config": run_config,
                    "steps_data": steps_data,
                })

    if not all_runs_data:
        st.error("No data loaded from any run. Check data source configuration.")
        return

    # ========================================================================
    # Single-run statistics (always shown)
    # ========================================================================

    st.subheader("Run Statistics")

    for run_data in all_runs_data:
        config = run_data["config"]
        steps_data = run_data["steps_data"]

        with st.expander(f"📊 {config['label']}", expanded=(len(all_runs_data) == 1)):
            if not steps_data:
                st.warning("No step data available")
                continue

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            total_steps = len(steps_data)
            total_rollouts = sum(s["num_rollouts"] for s in steps_data)
            final_reward = steps_data[-1]["avg_reward"] if steps_data else 0.0
            avg_reward_all = sum(s["avg_reward"] for s in steps_data) / len(steps_data)

            col1.metric("Total Steps", total_steps)
            col2.metric("Total Rollouts", total_rollouts)
            col3.metric("Final Reward", f"{final_reward:.3f}")
            col4.metric("Avg Reward (All Steps)", f"{avg_reward_all:.3f}")

            # Data source info
            st.markdown("**Data Source:**")
            st.json({
                "type": config["source_type"],
                "path": config["path_or_run"],
                "steps_range": f"{steps_data[0]['step']} - {steps_data[-1]['step']}",
            })

    # ========================================================================
    # Reward curves (overlay all runs)
    # ========================================================================

    st.markdown("---")
    st.subheader("Reward Curves")

    if all_runs_data:
        # Color palette for multiple runs
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

        # Create figure with first run
        first_run = all_runs_data[0]
        fig = render_reward_curve(
            first_run["steps_data"],
            run_label=first_run["config"]["label"],
            color=colors[0],
        )

        # Add additional runs
        for i, run_data in enumerate(all_runs_data[1:], start=1):
            if run_data["steps_data"]:
                additional_fig = render_reward_curve(
                    run_data["steps_data"],
                    run_label=run_data["config"]["label"],
                    color=colors[i % len(colors)],
                )
                # Add traces from additional figure to main figure
                for trace in additional_fig.data:
                    fig.add_trace(trace)

        # Update layout and render
        fig.update_layout(
            title="Training Reward Curves (All Runs)",
            height=500,
            showlegend=True,
            legend=dict(x=1.05, y=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # Multi-run comparison (shown when 2+ runs)
    # ========================================================================

    if len(all_runs_data) >= 2:
        st.markdown("---")
        st.subheader("Cross-Run Comparison")

        # Side-by-side reward distributions
        st.markdown("#### Reward Distributions")

        cols = st.columns(len(all_runs_data))
        for i, (col, run_data) in enumerate(zip(cols, all_runs_data)):
            with col:
                st.markdown(f"**{run_data['config']['label']}**")

                # Collect all rewards across all steps
                all_rewards = []
                for step in run_data["steps_data"]:
                    # We only have avg_reward per step, but we can show distribution across steps
                    all_rewards.append(step["avg_reward"])

                if all_rewards:
                    fig = render_reward_distribution(all_rewards, run_data["config"]["label"])
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No reward data")

        # Summary comparison table
        st.markdown("#### Summary Comparison")

        comparison_data = []
        for run_data in all_runs_data:
            config = run_data["config"]
            steps_data = run_data["steps_data"]

            if steps_data:
                comparison_data.append({
                    "Run": config["label"],
                    "Total Steps": len(steps_data),
                    "Avg Reward": f"{sum(s['avg_reward'] for s in steps_data) / len(steps_data):.3f}",
                    "Final Reward": f"{steps_data[-1]['avg_reward']:.3f}",
                    "Reward Trend": "↑" if len(steps_data) > 1 and steps_data[-1]["avg_reward"] > steps_data[0]["avg_reward"] else "→",
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ========================================================================
    # Data source configuration summary
    # ========================================================================

    st.markdown("---")
    st.subheader("Configuration Summary")

    config_summary = []
    for run_data in all_runs_data:
        config = run_data["config"]
        steps_data = run_data["steps_data"]

        config_summary.append({
            "Run": config["label"],
            "Source Type": config["source_type"],
            "Path/ID": config["path_or_run"],
            "Steps Available": len(steps_data),
            "Status": "✓ Connected" if steps_data else "✗ No data",
        })

    if config_summary:
        summary_df = pd.DataFrame(config_summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
