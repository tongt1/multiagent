"""Data loading utilities for Parquet files and W&B API.

This module provides cached data loading functions for both local Parquet debug
data files and remote W&B Tables. All functions use @st.cache_data for performance.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from streamlit_viewer.config import DEFAULT_PARQUET_GLOB


@st.cache_data
def discover_parquet_files(directory: str) -> list[dict]:
    """Discover Parquet debug data files in a directory.

    Args:
        directory: Directory path to search

    Returns:
        List of dicts with {"step": int, "path": str} sorted by step ascending.
        Empty list if directory doesn't exist or no files found.
    """
    if not directory or not directory.strip():
        return []
    if not os.path.isdir(directory):
        return []

    pattern = os.path.join(directory, DEFAULT_PARQUET_GLOB)
    files = glob.glob(pattern)

    results = []
    # Pattern: batch_debug_data_train_{step}.parquet
    step_pattern = re.compile(r"batch_debug_data_train_(\d+)\.parquet")

    for file_path in files:
        basename = os.path.basename(file_path)
        match = step_pattern.search(basename)
        if match:
            step = int(match.group(1))
            results.append({"step": step, "path": file_path})

    # Sort by step ascending
    results.sort(key=lambda x: x["step"])
    return results


@st.cache_data
def load_parquet_step(file_path: str, step: Optional[int] = None) -> pd.DataFrame:
    """Load Parquet debug data file with optional step filtering.

    Args:
        file_path: Path to Parquet file
        step: Optional step number for predicate pushdown filtering

    Returns:
        DataFrame with selected columns from debug data schema.
        Empty DataFrame if file doesn't exist or read fails.

    Note:
        The Parquet files from debug_data_writer.py are per-step files,
        so the step parameter may not be needed for filtering. It's included
        for compatibility with multi-step Parquet files if they exist.
    """
    try:
        # Select relevant columns for viewer
        columns = [
            "unique_sample_id",
            "trajectory",
            "reward",
            "solver_reward",
            "verifier_reward",
            "judge_reward",
            "role_assignments",
            "reward_metrics",
            "schema_version",
        ]

        # Attempt to load with PyArrow engine for predicate pushdown
        if step is not None:
            # Try predicate pushdown (may not work if step column doesn't exist)
            try:
                df = pd.read_parquet(
                    file_path,
                    engine="pyarrow",
                    columns=columns,
                    filters=[("step", "==", step)],
                )
            except Exception:
                # Fall back to loading all and filtering in memory
                df = pd.read_parquet(file_path, engine="pyarrow", columns=columns)
        else:
            df = pd.read_parquet(file_path, engine="pyarrow", columns=columns)

        return df

    except Exception as e:
        st.error(f"Failed to load Parquet file {file_path}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_wandb_runs(entity: str, project: str) -> list[dict]:
    """Load W&B runs for a project.

    Args:
        entity: W&B entity name
        project: W&B project name

    Returns:
        List of dicts with run metadata: {"id": str, "name": str, "state": str, "config": dict}
        Empty list if API call fails or wandb not available.
    """
    try:
        import wandb

        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")

        results = []
        for run in runs:
            results.append({
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "config": dict(run.config),
            })

        return results

    except ImportError:
        st.error("wandb package not available. Install with: pip install wandb")
        return []
    except Exception as e:
        st.error(f"Failed to load W&B runs: {e}")
        return []


@st.cache_data(ttl=300)
def load_wandb_step_data(
    entity: str,
    project: str,
    run_id: str,
    table_key: str = "debate/rollouts",
) -> pd.DataFrame:
    """Fetch W&B Table as DataFrame.

    Args:
        entity: W&B entity name
        project: W&B project name
        run_id: Run ID
        table_key: Table key in W&B (default: "debate/rollouts")

    Returns:
        DataFrame with table data. Empty DataFrame if fetch fails.
    """
    try:
        import wandb

        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")

        # Try to get table from artifacts or logged data
        # This is a simplified implementation - actual W&B Table access may vary
        try:
            # Method 1: Try artifact approach
            for artifact in run.logged_artifacts():
                if table_key in artifact.name:
                    table = artifact.get(table_key)
                    return table.get_dataframe()
        except Exception:
            pass

        # Method 2: Try history approach (for scalar metrics)
        # Note: W&B Tables may require different access patterns
        history = run.history(keys=[table_key])
        if not history.empty:
            return history

        st.warning(f"Table {table_key} not found in run {run_id}")
        return pd.DataFrame()

    except ImportError:
        st.error("wandb package not available. Install with: pip install wandb")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load W&B table: {e}")
        return pd.DataFrame()


def auto_detect_source(
    parquet_dir: Optional[str],
    wandb_entity: Optional[str],
    wandb_project: Optional[str],
) -> tuple[str, dict]:
    """Auto-detect data source based on availability.

    Priority: Parquet files (local) > W&B API (remote).

    Args:
        parquet_dir: Directory to check for Parquet files
        wandb_entity: W&B entity name
        wandb_project: W&B project name

    Returns:
        Tuple of (source_type, metadata_dict) where:
        - source_type is "parquet" or "wandb"
        - metadata_dict contains source-specific info
    """
    from streamlit_viewer.config import DATA_SOURCE_PARQUET, DATA_SOURCE_WANDB

    # Try Parquet first
    if parquet_dir:
        files = discover_parquet_files(parquet_dir)
        if files:
            return DATA_SOURCE_PARQUET, {
                "directory": parquet_dir,
                "num_files": len(files),
                "steps": [f["step"] for f in files],
            }

    # Fall back to W&B
    if wandb_entity and wandb_project:
        runs = load_wandb_runs(wandb_entity, wandb_project)
        if runs:
            return DATA_SOURCE_WANDB, {
                "entity": wandb_entity,
                "project": wandb_project,
                "num_runs": len(runs),
            }

    # No source found
    return "none", {
        "error": "No data source configured",
        "hint": "Set a Parquet directory path or configure W&B credentials",
    }


def compute_step_summary(df: pd.DataFrame) -> dict:
    """Compute summary statistics for a step's data.

    Args:
        df: DataFrame with rollout data (must have "reward" and "unique_sample_id" columns)

    Returns:
        Dict with summary statistics: avg_reward, num_prompts, num_rollouts, reward_std
    """
    if df.empty:
        return {
            "avg_reward": 0.0,
            "num_prompts": 0,
            "num_rollouts": 0,
            "reward_std": 0.0,
        }

    return {
        "avg_reward": df["reward"].mean() if "reward" in df.columns else 0.0,
        "num_prompts": df["unique_sample_id"].nunique() if "unique_sample_id" in df.columns else 0,
        "num_rollouts": len(df),
        "reward_std": df["reward"].std() if "reward" in df.columns else 0.0,
    }
