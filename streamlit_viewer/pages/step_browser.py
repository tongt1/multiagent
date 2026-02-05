"""Step Browser page for navigating training steps and rollouts.

This page provides:
- Step selection with summary stats
- Prompt list sorted by reward variance (highest first)
- Rollout display with debate structure (best/worst by default, expandable to all 8)
- Multi-run side-by-side layout when 2+ runs loaded
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st

from streamlit_viewer.config import DATA_SOURCE_PARQUET, GRPO_ROLLOUTS_PER_PROMPT
from streamlit_viewer.lib.data_loader import (
    discover_parquet_files,
    load_parquet_step,
    compute_step_summary,
)
from streamlit_viewer.lib.debate_parser import (
    parse_debate_turns,
    attach_per_turn_rewards,
    compute_reward_variance,
)
from streamlit_viewer.lib.layout_helpers import (
    render_rollout_card,
    render_prompt_header,
)


def _render_single_run(run_config: dict, run_idx: int) -> None:
    """Render step browser for a single run.

    Args:
        run_config: Run configuration dict with keys: label, source_type, path_or_run
        run_idx: Run index for namespaced session state keys
    """
    st.subheader(run_config["label"])

    # ========================================================================
    # Step Selection
    # ========================================================================

    if run_config["source_type"] == DATA_SOURCE_PARQUET:
        # Discover available steps from Parquet files
        parquet_dir = run_config["path_or_run"]
        step_files = discover_parquet_files(parquet_dir)

        if not step_files:
            st.warning(f"No Parquet files found in {parquet_dir}")
            return

        # Display step selector
        step_numbers = [sf["step"] for sf in step_files]
        selected_step = st.selectbox(
            f"Select training step ({run_config['label']}):",
            step_numbers,
            key=f"selected_step_{run_idx}",
        )

        # Load selected step data
        selected_file = next(sf for sf in step_files if sf["step"] == selected_step)
        step_data = load_parquet_step(selected_file["path"])

        if step_data.empty:
            st.warning(f"No data found for step {selected_step}")
            return

        # Display step summary
        summary = compute_step_summary(step_data)
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Reward", f"{summary['avg_reward']:.3f}")
        col2.metric("Num Prompts", summary['num_prompts'])
        col3.metric("Num Rollouts", summary['num_rollouts'])

    else:
        # W&B source - placeholder for now
        st.info("W&B data source support coming soon")
        return

    # ========================================================================
    # Prompt List (sorted by reward variance)
    # ========================================================================

    st.markdown("---")
    st.markdown("### Prompts (sorted by reward variance)")

    # Group rollouts by prompt
    if "unique_sample_id" not in step_data.columns:
        st.error("unique_sample_id column not found in data")
        return

    prompt_groups = step_data.groupby("unique_sample_id")

    # Compute reward variance for each prompt
    prompt_stats = []
    for prompt_id, group in prompt_groups:
        rewards = group["reward"].tolist()
        variance = compute_reward_variance(rewards)
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        prompt_stats.append({
            "prompt_id": prompt_id,
            "variance": variance,
            "avg_reward": avg_reward,
            "num_rollouts": len(group),
        })

    # Sort by variance descending (most interesting first)
    prompt_stats.sort(key=lambda x: x["variance"], reverse=True)

    # Display prompt list
    if not prompt_stats:
        st.info("No prompts found in this step")
        return

    # Use selectbox for prompt selection
    prompt_labels = [
        f"{p['prompt_id'][:40]}... (var={p['variance']:.4f}, avg={p['avg_reward']:.2f})"
        for p in prompt_stats
    ]
    selected_prompt_idx = st.selectbox(
        "Select a prompt:",
        range(len(prompt_stats)),
        format_func=lambda i: prompt_labels[i],
        key=f"selected_prompt_{run_idx}",
    )

    selected_prompt = prompt_stats[selected_prompt_idx]
    prompt_id = selected_prompt["prompt_id"]

    # ========================================================================
    # Rollout Display
    # ========================================================================

    st.markdown("---")
    render_prompt_header(
        prompt_id=prompt_id,
        reward_variance=selected_prompt["variance"],
        num_rollouts=selected_prompt["num_rollouts"],
    )

    # Get rollouts for selected prompt
    prompt_rollouts = step_data[step_data["unique_sample_id"] == prompt_id]

    if prompt_rollouts.empty:
        st.info("No rollouts found for this prompt")
        return

    # Sort rollouts by reward descending
    prompt_rollouts = prompt_rollouts.sort_values("reward", ascending=False).reset_index(drop=True)

    # Parse debate turns for each rollout
    rollouts_with_turns = []
    for _, rollout in prompt_rollouts.iterrows():
        # Get completion text (trajectory column)
        completion = rollout.get("trajectory", "")

        # Parse debate turns
        turns = parse_debate_turns(completion)

        # Attach per-turn rewards if available
        per_turn_rewards = None
        if "solver_reward" in rollout and pd.notna(rollout.get("solver_reward")):
            per_turn_rewards = {
                "solver": rollout.get("solver_reward"),
                "verifier": rollout.get("verifier_reward"),
                "judge": rollout.get("judge_reward"),
            }
            # Filter out None values
            per_turn_rewards = {k: v for k, v in per_turn_rewards.items() if pd.notna(v)}

        if per_turn_rewards:
            turns = attach_per_turn_rewards(turns, per_turn_rewards)

        rollouts_with_turns.append({
            "rollout": rollout.to_dict(),
            "turns": turns,
        })

    # Default view: Best and worst rollout side-by-side
    if len(rollouts_with_turns) >= 2:
        st.markdown("#### Best vs Worst Rollout")

        col_best, col_worst = st.columns(2)

        with col_best:
            st.markdown("**Best Rollout** (highest reward)")
            best = rollouts_with_turns[0]
            render_rollout_card(
                rollout_data=best["rollout"],
                turns=best["turns"],
                expanded=True,
            )

        with col_worst:
            st.markdown("**Worst Rollout** (lowest reward)")
            worst = rollouts_with_turns[-1]
            render_rollout_card(
                rollout_data=worst["rollout"],
                turns=worst["turns"],
                expanded=True,
            )

    elif len(rollouts_with_turns) == 1:
        # Only one rollout
        st.markdown("#### Single Rollout")
        single = rollouts_with_turns[0]
        render_rollout_card(
            rollout_data=single["rollout"],
            turns=single["turns"],
            expanded=True,
        )

    # Expandable section for all rollouts
    if len(rollouts_with_turns) > 2:
        with st.expander(f"Show all {len(rollouts_with_turns)} rollouts"):
            for i, item in enumerate(rollouts_with_turns):
                st.markdown(f"**Rollout {i+1} of {len(rollouts_with_turns)}**")
                render_rollout_card(
                    rollout_data=item["rollout"],
                    turns=item["turns"],
                    expanded=True,
                )
                if i < len(rollouts_with_turns) - 1:
                    st.markdown("---")


def render_step_browser(run_configs: list[dict]) -> None:
    """Render Step Browser tab with multi-run support.

    This function handles the top-level layout for single or multiple runs.
    When 2+ runs are loaded, it creates side-by-side columns for comparison.

    Args:
        run_configs: List of run configuration dicts. Each dict has:
            - label: Display name for the run
            - source_type: "parquet" or "wandb"
            - path_or_run: Directory path (Parquet) or run ID (W&B)
    """
    if not run_configs:
        st.info("No data sources configured. Use the sidebar to add a run.")
        return

    # Multi-run layout: side-by-side columns for 2+ runs
    if len(run_configs) == 1:
        # Single run: full width
        _render_single_run(run_configs[0], run_idx=0)
    else:
        # Multiple runs: side-by-side columns
        st.markdown("### Side-by-Side Run Comparison")
        st.markdown(f"Comparing {len(run_configs)} runs with independent navigation")

        cols = st.columns(len(run_configs))

        for run_idx, (col, run_config) in enumerate(zip(cols, run_configs)):
            with col:
                _render_single_run(run_config, run_idx=run_idx)
