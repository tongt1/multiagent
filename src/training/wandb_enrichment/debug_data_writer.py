"""Parquet debug data writer for downstream Streamlit consumption.

This module writes complete rollout records to Parquet files with snappy
compression. The Parquet files include schema versioning for forward-compatible
schema evolution when Phase 7 Streamlit analysis tools read them.

File naming: batch_debug_data_train_{step}.parquet
Compression: snappy
Parquet version: 2.6

The schema matches BatchDebugData from Flink with debate-specific extensions
(per-role rewards and KL divergence).
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from .metric_schema import CURRENT_SCHEMA_VERSION, get_debug_data_schema


def write_debug_data_parquet(
    step: int,
    prompt_ids: list[str],
    prompt_texts: list[str],
    completions: list[str],
    rewards: list[float],
    output_dir: str,
    solver_rewards: list[float] | None = None,
    verifier_rewards: list[float] | None = None,
    judge_rewards: list[float] | None = None,
    role_assignments: list[str] | None = None,
    unique_sample_ids: list[str] | None = None,
) -> str:
    """Write rollout debug data to Parquet file.

    Args:
        step: Training step number
        prompt_ids: List of prompt identifiers
        prompt_texts: List of prompt text strings
        completions: List of completion text (debate trajectories)
        rewards: List of final rewards
        output_dir: Directory to write Parquet file
        solver_rewards: Per-role reward for solver (optional)
        verifier_rewards: Per-role reward for verifier (optional)
        judge_rewards: Per-role reward for judge (optional)
        role_assignments: Role assignment strings (optional)
        unique_sample_ids: Unique sample identifiers (optional)

    Returns:
        Path to written Parquet file

    Raises:
        ValueError: If input lists have mismatched lengths

    File format:
        - Filename: batch_debug_data_train_{step}.parquet
        - Compression: snappy
        - Parquet version: 2.6
        - Schema: From get_debug_data_schema() with CURRENT_SCHEMA_VERSION
    """
    import pandas as pd
    import pyarrow.parquet as pq

    # Validate input lengths
    n = len(prompt_ids)
    if not all(
        len(lst) == n
        for lst in [prompt_texts, completions, rewards]
    ):
        raise ValueError(
            f"Mismatched input lengths: prompt_ids={len(prompt_ids)}, "
            f"prompt_texts={len(prompt_texts)}, completions={len(completions)}, "
            f"rewards={len(rewards)}"
        )

    # Build data dictionary matching schema
    data = {
        "schema_version": [CURRENT_SCHEMA_VERSION] * n,

        # Core Flink columns (using simplified values for now)
        "env_name": ["math_debate"] * n,  # Will be populated from actual env
        "trajectory": completions,  # The actual debate text
        "agent_trajectories": [""] * n,  # JSON-encoded agent data (if available)
        "exception_info": [None] * n,  # Error info if rollout failed
        "reward": rewards,
        "reward_metrics": ["{}"] * n,  # JSON metrics dict (if available)
        "reward_text_info": [""] * n,  # Human-readable reward info
        "unique_sample_id": unique_sample_ids if unique_sample_ids else [""] * n,

        # Debate-specific extensions
        "role_assignments": role_assignments if role_assignments else [None] * n,
        "solver_reward": solver_rewards if solver_rewards else [float('nan')] * n,
        "verifier_reward": verifier_rewards if verifier_rewards else [float('nan')] * n,
        "judge_reward": judge_rewards if judge_rewards else [float('nan')] * n,
        "solver_kl": [float('nan')] * n,  # KL divergence (to be added in Phase 5 Plan 2)
        "verifier_kl": [float('nan')] * n,
        "judge_kl": [float('nan')] * n,
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write to Parquet
    output_path = get_debug_data_path(output_dir, step)
    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        version="2.6",
        index=False,
    )

    logger.info(
        f"Wrote {len(df)} rollouts to {output_path} "
        f"(schema_version={CURRENT_SCHEMA_VERSION})"
    )
    return output_path


def read_debug_data_parquet(file_path: str) -> "pd.DataFrame":
    """Read debug data Parquet file and validate schema version.

    Args:
        file_path: Path to Parquet file

    Returns:
        DataFrame with debug data

    Logs warning if schema_version doesn't match CURRENT_SCHEMA_VERSION.
    """
    import pandas as pd

    df = pd.read_parquet(file_path, engine="pyarrow")

    # Validate schema version
    if "schema_version" not in df.columns:
        logger.warning(f"Parquet file {file_path} missing schema_version column")
    else:
        file_version = df["schema_version"].iloc[0]
        if file_version != CURRENT_SCHEMA_VERSION:
            logger.warning(
                f"Parquet file {file_path} has schema_version={file_version}, "
                f"expected {CURRENT_SCHEMA_VERSION}"
            )

    return df


def get_debug_data_path(output_dir: str, step: int) -> str:
    """Get expected Parquet file path for a given step.

    Args:
        output_dir: Base output directory
        step: Training step number

    Returns:
        Full path to Parquet file
    """
    return os.path.join(output_dir, f"batch_debug_data_train_{step}.parquet")
