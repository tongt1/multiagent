"""Integration layer bridging Flink actor outputs to rollout table logging.

This module converts Flink's batch metadata format to RolloutRecord objects and
orchestrates W&B Table logging, debug data persistence, and KL computation (when available).

Key functions:
- build_rollout_records_from_batch(): Convert Flink batch format to RolloutRecord list
- log_debate_rollout_table(): End-to-end W&B Table logging from actor outputs
- write_debate_debug_data(): Bridge to debug_data_writer.py
- compute_kl_if_available(): Hook for Phase 6 KL integration (returns empty dict until role_masks provided)

Design principles:
- All functions store RAW completion text -- NO debate parsing (Phase 7 handles that per CONTEXT.md)
- All functions are tolerant of missing/malformed data (never crash training)
- All W&B operations use try/except with logging warnings
- Functions operate on Python dicts/lists, NOT JAX arrays (for testability)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.training.wandb_enrichment.rollout_table import RolloutRecord

if TYPE_CHECKING:
    import wandb


def build_rollout_records_from_batch(
    trajectories: list[str],
    rewards: list[float],
    unique_sample_ids: list[str],
    env_names: list[str] | None = None,
    reward_metrics: dict | None = None,
) -> list[RolloutRecord]:
    """Convert Flink batch metadata format to RolloutRecord objects.

    Args:
        trajectories: Raw completion text from batch.metadata["trajectory"]
        rewards: Final rewards from batch.data["rewards"]
        unique_sample_ids: Sample IDs from batch.metadata["unique_sample_id"]
        env_names: Environment names from batch.metadata["env_name"] (optional)
        reward_metrics: Per-rollout reward metrics dict (optional)

    Returns:
        List of RolloutRecord objects, one per rollout in the batch.
        If env_names provided, filters to only "math_debate" entries.

    Design notes:
        - Uses unique_sample_id as prompt_id (Flink format: "math:42")
        - Sets completion to FULL trajectory string as-is (NO debate parsing)
        - Filters to math_debate env if env_names provided
        - Per-role rewards set to None (Phase 6 will populate from reward_metrics)
    """
    if not trajectories:
        return []

    # Validate input lengths
    n = len(trajectories)
    if len(rewards) != n or len(unique_sample_ids) != n:
        logger.warning(
            f"Mismatched input lengths: trajectories={len(trajectories)}, "
            f"rewards={len(rewards)}, unique_sample_ids={len(unique_sample_ids)}. "
            "Truncating to shortest."
        )
        n = min(len(trajectories), len(rewards), len(unique_sample_ids))
        trajectories = trajectories[:n]
        rewards = rewards[:n]
        unique_sample_ids = unique_sample_ids[:n]

    if env_names is not None and len(env_names) != n:
        logger.warning(
            f"env_names length ({len(env_names)}) != batch size ({n}). "
            "Ignoring env_names filter."
        )
        env_names = None

    records = []
    for i in range(n):
        # Filter to math_debate env if env_names provided
        if env_names is not None:
            if env_names[i] != "math_debate":
                continue

        # Extract prompt_id from unique_sample_id (format: "math:42")
        prompt_id = unique_sample_ids[i]

        # For now, we don't have access to the original prompt text
        # Phase 6 will add this to batch metadata
        prompt_text = f"[Prompt {prompt_id}]"

        # Create RolloutRecord with raw completion text (NO parsing)
        record = RolloutRecord(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            completion=trajectories[i],  # RAW text, no debate parsing
            reward=rewards[i],
            solver_reward=None,  # Phase 6 will populate from reward_metrics
            verifier_reward=None,
            judge_reward=None,
            role_assignments=None,  # Phase 6 will populate
        )
        records.append(record)

    return records


def log_debate_rollout_table(
    items: list,
    step: int,
    wandb_run: Any = None,
    n_prompts: int = 4,
    top_k: int = 2,
    bottom_k: int = 2,
) -> None:
    """Log debate rollout table to W&B from actor output items.

    This is the end-to-end function called by DebateMetricStreamer during training.
    It converts actor outputs to RolloutRecords, samples per-prompt top/bottom rollouts,
    creates a W&B Table, and logs it.

    Args:
        items: Actor output items from streamer.get()
        step: Training step number
        wandb_run: Active W&B run (or None to use wandb.run)
        n_prompts: Number of prompts to sample per step (default: 4)
        top_k: Top-performing rollouts per prompt (default: 2)
        bottom_k: Bottom-performing rollouts per prompt (default: 2)

    Returns:
        None. Logs to W&B and catches all exceptions (never crashes training).

    Design notes:
        - Extracts trajectories, rewards, unique_sample_ids from items
        - Calls build_rollout_records_from_batch() to convert format
        - Calls sample_rollouts_per_prompt() from rollout_table.py
        - Creates W&B Table and logs via log_rollout_table()
        - All errors logged as warnings (training continues)
    """
    try:
        # Lazy import to avoid import errors in test environments
        import wandb
        from src.training.wandb_enrichment.rollout_table import (
            add_sampled_rollouts_to_table,
            create_rollout_table,
            log_rollout_table,
            sample_rollouts_per_prompt,
        )

        # Get active W&B run
        if wandb_run is None:
            wandb_run = wandb.run
        if wandb_run is None:
            logger.debug("No active W&B run, skipping rollout table logging")
            return

        # Extract data from actor output items
        # items[i].metadata is a dict with string keys and numpy array values
        # items[i].data is a dict with string keys and numpy scalar values
        trajectories = []
        rewards = []
        unique_sample_ids = []
        env_names = []

        for item in items:
            # Extract trajectory
            traj = item.metadata.get("trajectory", "")
            if hasattr(traj, "item"):
                traj = traj.item()
            trajectories.append(str(traj))

            # Extract reward
            reward_val = item.data.get("rewards", 0.0)
            if hasattr(reward_val, "item"):
                reward_val = reward_val.item()
            rewards.append(float(reward_val))

            # Extract unique_sample_id
            sample_id = item.metadata.get("unique_sample_id", "")
            if hasattr(sample_id, "item"):
                sample_id = sample_id.item()
            unique_sample_ids.append(str(sample_id))

            # Extract env_name
            env_name = item.metadata.get("env_name", "")
            if hasattr(env_name, "item"):
                env_name = env_name.item()
            env_names.append(str(env_name))

        # Build RolloutRecord objects
        rollout_records = build_rollout_records_from_batch(
            trajectories=trajectories,
            rewards=rewards,
            unique_sample_ids=unique_sample_ids,
            env_names=env_names,
        )

        if not rollout_records:
            logger.debug(f"No rollout records to log at step {step}")
            return

        # Sample top-k and bottom-k per prompt
        sampled_rollouts = sample_rollouts_per_prompt(
            rollouts=rollout_records,
            n_prompts=n_prompts,
            top_k=top_k,
            bottom_k=bottom_k,
        )

        if not sampled_rollouts:
            logger.debug(f"No sampled rollouts at step {step}")
            return

        # Create W&B Table
        table = create_rollout_table(use_incremental=True)

        # Add sampled rollouts to table
        add_sampled_rollouts_to_table(
            table=table,
            step=step,
            sampled_rollouts=sampled_rollouts,
        )

        # Log table to W&B
        log_rollout_table(
            wandb_run=wandb_run,
            table=table,
            step=step,
        )

        logger.debug(f"Successfully logged rollout table at step {step}")

    except ImportError as e:
        logger.warning(f"Cannot log rollout table (missing dependency): {e}")
    except Exception as e:
        logger.warning(f"Failed to log rollout table at step {step}: {e}", exc_info=True)


def write_debate_debug_data(
    items: list,
    step: int,
    output_dir: str,
) -> str | None:
    """Write debate debug data to Parquet file.

    Bridge function that extracts data from actor output items and calls
    write_debug_data_parquet() from debug_data_writer.py.

    Args:
        items: Actor output items from streamer.get()
        step: Training step number
        output_dir: Directory to write Parquet file

    Returns:
        Path to written Parquet file, or None on error.

    Design notes:
        - Extracts trajectories, rewards, unique_sample_ids from items
        - Calls write_debug_data_parquet() from debug_data_writer.py
        - Catches all exceptions (never crashes training)
    """
    try:
        from src.training.wandb_enrichment.debug_data_writer import write_debug_data_parquet

        # Extract data from items
        prompt_ids = []
        prompt_texts = []
        completions = []
        rewards = []
        unique_sample_ids = []

        for item in items:
            # Extract unique_sample_id
            sample_id = item.metadata.get("unique_sample_id", "")
            if hasattr(sample_id, "item"):
                sample_id = sample_id.item()
            sample_id_str = str(sample_id)
            unique_sample_ids.append(sample_id_str)
            prompt_ids.append(sample_id_str)

            # Extract trajectory (completion)
            traj = item.metadata.get("trajectory", "")
            if hasattr(traj, "item"):
                traj = traj.item()
            completions.append(str(traj))

            # Placeholder prompt text (Phase 6 will add to metadata)
            prompt_texts.append(f"[Prompt {sample_id_str}]")

            # Extract reward
            reward_val = item.data.get("rewards", 0.0)
            if hasattr(reward_val, "item"):
                reward_val = reward_val.item()
            rewards.append(float(reward_val))

        # Write to Parquet
        output_path = write_debug_data_parquet(
            step=step,
            prompt_ids=prompt_ids,
            prompt_texts=prompt_texts,
            completions=completions,
            rewards=rewards,
            output_dir=output_dir,
            unique_sample_ids=unique_sample_ids,
        )

        logger.debug(f"Wrote debug data to {output_path}")
        return output_path

    except ImportError as e:
        logger.warning(f"Cannot write debug data (missing dependency): {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to write debug data at step {step}: {e}", exc_info=True)
        return None


def compute_kl_if_available(
    batch_metadata: dict,
) -> dict[str, float]:
    """Compute per-role KL divergence if role masks are available.

    This is a hook for Phase 6 integration. Currently returns empty dict because
    role_masks are not yet provided in batch_metadata. Phase 6 will add:
    - batch_metadata["debate/role_masks"] = {"solver": mask, "verifier": mask, "judge": mask}
    - batch_metadata["kl_per_token"] = per-token KL divergence array

    Args:
        batch_metadata: Batch metadata dict from training

    Returns:
        Dict with per-role KL metrics (debate/kl/solver, etc.) if role_masks available,
        otherwise empty dict.

    Design notes:
        - Checks for "debate/role_masks" in batch_metadata
        - If available, extracts kl_per_token and calls compute_per_role_kl()
        - If not available, returns empty dict with debug log
        - This hook will be "activated" when Phase 6 adds role_masks to metadata
    """
    try:
        from src.training.wandb_enrichment.debate_metrics import compute_per_role_kl

        # Check if role_masks are available (Phase 6 will add this)
        if "debate/role_masks" not in batch_metadata:
            logger.debug(
                "compute_kl_if_available: role_masks not in batch_metadata. "
                "Phase 6 will activate this hook by adding debate/role_masks."
            )
            return {}

        # Extract role_masks and kl_per_token
        role_masks = batch_metadata["debate/role_masks"]
        kl_per_token = batch_metadata.get("kl_per_token", None)

        if kl_per_token is None:
            logger.warning("compute_kl_if_available: role_masks present but kl_per_token missing")
            return {}

        # Compute per-role KL
        return compute_per_role_kl(kl_per_token=kl_per_token, role_masks=role_masks)

    except ImportError as e:
        logger.warning(f"Cannot compute per-role KL (missing dependency): {e}")
        return {}
    except Exception as e:
        logger.warning(f"Failed to compute per-role KL: {e}", exc_info=True)
        return {}
