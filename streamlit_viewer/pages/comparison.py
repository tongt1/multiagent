"""Comparison page for side-by-side rollout and cross-run analysis.

This page provides:
- Within-run rollout comparison: Select any 2 of 8 rollouts for a prompt
- Side-by-side display with debate timelines and per-turn rewards
- Diff highlighting between rollout completions
- Cross-run comparison: Match prompts by unique_sample_id across runs
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
)
from streamlit_viewer.lib.debate_parser import (
    parse_debate_turns,
    attach_per_turn_rewards,
)
from streamlit_viewer.lib.diff_utils import render_rollout_diff
from streamlit_viewer.lib.visualizations import render_debate_timeline
from streamlit_viewer.lib.layout_helpers import render_rollout_card


def _get_per_turn_rewards(rollout_row: pd.Series) -> Optional[dict[str, float]]:
    """Extract per-turn rewards from a rollout row.

    Args:
        rollout_row: DataFrame row with rollout data

    Returns:
        Dict mapping role to reward, e.g. {"solver": 0.3, "verifier": 0.5, "judge": 0.8}
        or None if per-turn reward data not available
    """
    per_turn_rewards = {}

    # Check for dedicated per-role reward columns
    if "solver_reward" in rollout_row and pd.notna(rollout_row["solver_reward"]):
        per_turn_rewards["solver"] = float(rollout_row["solver_reward"])
    if "verifier_reward" in rollout_row and pd.notna(rollout_row["verifier_reward"]):
        per_turn_rewards["verifier"] = float(rollout_row["verifier_reward"])
    if "judge_reward" in rollout_row and pd.notna(rollout_row["judge_reward"]):
        per_turn_rewards["judge"] = float(rollout_row["judge_reward"])

    return per_turn_rewards if per_turn_rewards else None


def render_comparison(run_configs: list[dict]) -> None:
    """Render comparison page with within-run and cross-run comparison modes.

    Args:
        run_configs: List of run configuration dicts. Each dict has keys:
                     label, source_type, path_or_run
    """
    st.header("Rollout Comparison")

    if not run_configs:
        st.warning("No runs configured. Add a data source in the sidebar.")
        return

    # ========================================================================
    # Mode Selection
    # ========================================================================

    comparison_mode = st.radio(
        "Comparison mode:",
        ["Within-run (same step, different rollouts)", "Cross-run (same prompt, different runs)"],
        help="Choose whether to compare rollouts within a run or across runs",
    )

    # ========================================================================
    # Within-run rollout comparison
    # ========================================================================

    if comparison_mode.startswith("Within-run"):
        st.markdown("---")
        st.subheader("Within-Run Rollout Comparison")

        # Run selection (for multi-run scenarios)
        if len(run_configs) > 1:
            run_labels = [rc["label"] for rc in run_configs]
            selected_run_label = st.selectbox("Select run:", run_labels)
            run_config = next(rc for rc in run_configs if rc["label"] == selected_run_label)
        else:
            run_config = run_configs[0]
            st.markdown(f"**Run:** {run_config['label']}")

        if run_config["source_type"] != DATA_SOURCE_PARQUET:
            st.info("W&B data source support coming soon")
            return

        # Step selection
        parquet_dir = run_config["path_or_run"]
        step_files = discover_parquet_files(parquet_dir)

        if not step_files:
            st.warning(f"No Parquet files found in {parquet_dir}")
            return

        step_numbers = [sf["step"] for sf in step_files]
        selected_step = st.selectbox("Select training step:", step_numbers)

        # Load step data
        selected_file = next(sf for sf in step_files if sf["step"] == selected_step)
        step_data = load_parquet_step(selected_file["path"])

        if step_data.empty:
            st.warning(f"No data found for step {selected_step}")
            return

        # Prompt selection
        prompts = sorted(step_data["unique_sample_id"].unique())
        selected_prompt = st.selectbox("Select prompt:", prompts)

        # Get rollouts for selected prompt
        prompt_rollouts = step_data[step_data["unique_sample_id"] == selected_prompt]

        if len(prompt_rollouts) < 2:
            st.warning(f"Need at least 2 rollouts for comparison. Found {len(prompt_rollouts)}")
            return

        # Rollout selection (pick any 2 of the 8)
        rollout_options = []
        for idx, (_, row) in enumerate(prompt_rollouts.iterrows()):
            reward = row.get("reward", 0.0)
            rollout_options.append(f"Rollout {idx+1} (reward: {reward:.2f})")

        col1, col2 = st.columns(2)
        with col1:
            # Default to best (highest reward)
            best_idx = prompt_rollouts["reward"].idxmax()
            best_position = list(prompt_rollouts.index).index(best_idx)
            left_selection = st.selectbox(
                "Left rollout:",
                rollout_options,
                index=best_position,
                key="left_rollout",
            )
        with col2:
            # Default to worst (lowest reward)
            worst_idx = prompt_rollouts["reward"].idxmin()
            worst_position = list(prompt_rollouts.index).index(worst_idx)
            right_selection = st.selectbox(
                "Right rollout:",
                rollout_options,
                index=worst_position,
                key="right_rollout",
            )

        # Extract selected rollouts
        left_idx = int(left_selection.split()[1]) - 1
        right_idx = int(right_selection.split()[1]) - 1

        left_rollout = prompt_rollouts.iloc[left_idx]
        right_rollout = prompt_rollouts.iloc[right_idx]

        # ====================================================================
        # Side-by-side display
        # ====================================================================

        st.markdown("---")
        st.markdown("### Side-by-Side Comparison")

        col_left, col_right = st.columns(2)

        # Left rollout
        with col_left:
            st.markdown(f"**{left_selection}**")

            # Parse debate turns
            completion = left_rollout.get("trajectory", "")
            turns = parse_debate_turns(completion)
            per_turn_rewards = _get_per_turn_rewards(left_rollout)
            turns = attach_per_turn_rewards(turns, per_turn_rewards)

            # Render rollout card (convert Series to dict for layout helper)
            render_rollout_card(left_rollout.to_dict(), turns, expanded=False)

            # Render debate timeline
            if turns:
                render_debate_timeline(turns, title="Debate Flow Timeline (Left)")

        # Right rollout
        with col_right:
            st.markdown(f"**{right_selection}**")

            # Parse debate turns
            completion = right_rollout.get("trajectory", "")
            turns = parse_debate_turns(completion)
            per_turn_rewards = _get_per_turn_rewards(right_rollout)
            turns = attach_per_turn_rewards(turns, per_turn_rewards)

            # Render rollout card (convert Series to dict for layout helper)
            render_rollout_card(right_rollout.to_dict(), turns, expanded=False)

            # Render debate timeline
            if turns:
                render_debate_timeline(turns, title="Debate Flow Timeline (Right)")

        # ====================================================================
        # Diff highlighting
        # ====================================================================

        st.markdown("---")
        st.markdown("### Text Diff")

        rollout_dict_left = {
            "completion": left_rollout.get("trajectory", ""),
            "reward": left_rollout.get("reward", 0.0),
        }
        rollout_dict_right = {
            "completion": right_rollout.get("trajectory", ""),
            "reward": right_rollout.get("reward", 0.0),
        }

        render_rollout_diff(rollout_dict_left, rollout_dict_right)

    # ========================================================================
    # Cross-run comparison
    # ========================================================================

    else:  # Cross-run mode
        st.markdown("---")
        st.subheader("Cross-Run Comparison")

        if len(run_configs) < 2:
            st.warning("Cross-run comparison requires at least 2 runs. Add more runs in the sidebar.")
            return

        st.info("Cross-run comparison matches prompts by unique_sample_id across runs.")

        # Load first step from each run to find common prompts
        all_prompts = {}
        for run_config in run_configs:
            if run_config["source_type"] != DATA_SOURCE_PARQUET:
                continue

            parquet_dir = run_config["path_or_run"]
            step_files = discover_parquet_files(parquet_dir)

            if not step_files:
                continue

            # Load first step
            first_step_data = load_parquet_step(step_files[0]["path"])
            if not first_step_data.empty:
                prompts = set(first_step_data["unique_sample_id"].unique())
                all_prompts[run_config["label"]] = prompts

        if len(all_prompts) < 2:
            st.warning("Could not load data from at least 2 runs.")
            return

        # Find common prompts
        common_prompts = set.intersection(*all_prompts.values())

        if not common_prompts:
            st.warning("No common prompts found across runs.")
            st.json({run_label: len(prompts) for run_label, prompts in all_prompts.items()})
            return

        st.success(f"Found {len(common_prompts)} common prompts across {len(all_prompts)} runs.")

        # Prompt selection
        selected_prompt = st.selectbox("Select prompt:", sorted(common_prompts))

        # Step selection (apply to all runs)
        st.markdown("**Step selection** (applied to all runs):")
        selected_step = st.number_input("Training step:", min_value=0, value=0, step=1)

        st.markdown("---")

        # Load and display best rollout from each run
        cols = st.columns(len(run_configs))

        for col, run_config in zip(cols, run_configs):
            with col:
                st.markdown(f"**{run_config['label']}**")

                if run_config["source_type"] != DATA_SOURCE_PARQUET:
                    st.info("W&B not supported")
                    continue

                parquet_dir = run_config["path_or_run"]
                step_files = discover_parquet_files(parquet_dir)

                # Find step file
                step_file = next((sf for sf in step_files if sf["step"] == selected_step), None)

                if not step_file:
                    st.warning(f"Step {selected_step} not found")
                    continue

                # Load step data
                step_data = load_parquet_step(step_file["path"])
                prompt_rollouts = step_data[step_data["unique_sample_id"] == selected_prompt]

                if prompt_rollouts.empty:
                    st.warning("Prompt not found in this run")
                    continue

                # Get best rollout
                best_idx = prompt_rollouts["reward"].idxmax()
                best_rollout = prompt_rollouts.loc[best_idx]

                # Parse debate turns for this rollout
                completion = best_rollout.get("trajectory", "")
                turns = parse_debate_turns(completion)
                per_turn_rewards = _get_per_turn_rewards(best_rollout)
                turns = attach_per_turn_rewards(turns, per_turn_rewards)

                # Display
                st.metric("Best Reward", f"{best_rollout['reward']:.2f}")
                render_rollout_card(best_rollout.to_dict(), turns, expanded=False)

                if turns:
                    render_debate_timeline(turns, title="Best Rollout Timeline")
