"""W&B rollout table creation with per-prompt top/bottom sampling.

This module provides structured W&B table logging for debate rollouts, with
per-prompt sampling to ensure balanced visibility into both high and low
performing rollouts for each prompt.

Sampling strategy:
- Sample N prompts from the batch (default: 4)
- For each prompt: select top-K (default: 2) and bottom-K (default: 2) rollouts by reward
- Total rows per step: N × (top-K + bottom-K) = 16 rows (default)

This avoids global top-K bias where a single prompt could dominate the table.
"""

from __future__ import annotations

import dataclasses
import random
from collections import defaultdict
from typing import TYPE_CHECKING

from loguru import logger

from .metric_schema import (
    DEFAULT_BOTTOM_K,
    DEFAULT_PROMPTS_PER_STEP,
    DEFAULT_TOP_K,
    MAX_ROWS_PER_STEP,
    ROLLOUT_TABLE_COLUMNS,
)

if TYPE_CHECKING:
    import wandb


@dataclasses.dataclass
class RolloutRecord:
    """A single rollout (one of 8 GRPO samples for a prompt)."""

    prompt_id: str
    prompt_text: str
    completion: str
    reward: float
    solver_reward: float | None = None
    verifier_reward: float | None = None
    judge_reward: float | None = None
    role_assignments: str | None = None


def sample_rollouts_per_prompt(
    rollouts: list[RolloutRecord],
    n_prompts: int = DEFAULT_PROMPTS_PER_STEP,
    top_k: int = DEFAULT_TOP_K,
    bottom_k: int = DEFAULT_BOTTOM_K,
) -> list[tuple[RolloutRecord, bool]]:
    """Sample top-K and bottom-K rollouts per prompt.

    Args:
        rollouts: All rollouts for the batch
        n_prompts: Number of prompts to sample
        top_k: Number of top-performing rollouts per prompt
        bottom_k: Number of bottom-performing rollouts per prompt

    Returns:
        List of (RolloutRecord, is_top) tuples where is_top indicates
        if the rollout is in the top-K for its prompt.

    Sampling behavior:
        - Groups rollouts by prompt_id
        - If fewer than n_prompts unique prompts exist, uses all
        - For each selected prompt: sorts by reward descending
        - Takes first top_k rollouts (is_top=True)
        - Takes last bottom_k rollouts (is_top=False)
        - If a prompt has fewer than top_k + bottom_k rollouts, takes all
    """
    if not rollouts:
        return []

    # Group rollouts by prompt_id
    by_prompt: dict[str, list[RolloutRecord]] = defaultdict(list)
    for rollout in rollouts:
        by_prompt[rollout.prompt_id].append(rollout)

    # Select prompts to sample
    prompt_ids = list(by_prompt.keys())
    if len(prompt_ids) <= n_prompts:
        selected_prompts = prompt_ids
    else:
        selected_prompts = random.sample(prompt_ids, n_prompts)

    # Sample rollouts for each selected prompt
    sampled: list[tuple[RolloutRecord, bool]] = []
    for prompt_id in selected_prompts:
        prompt_rollouts = by_prompt[prompt_id]
        # Sort by reward descending
        sorted_rollouts = sorted(prompt_rollouts, key=lambda r: r.reward, reverse=True)

        # Take top-k
        for rollout in sorted_rollouts[:top_k]:
            sampled.append((rollout, True))

        # Take bottom-k (from the end)
        for rollout in sorted_rollouts[-bottom_k:]:
            # Avoid duplicates if fewer rollouts than top_k + bottom_k
            if rollout not in [r for r, _ in sampled]:
                sampled.append((rollout, False))

    return sampled


# Cache for server capability check
_WANDB_INCREMENTAL_SUPPORTED: bool | None = None


def check_wandb_server_supports_incremental() -> bool:
    """Check if W&B server supports INCREMENTAL table mode.

    Returns:
        True if INCREMENTAL mode is supported, False otherwise.
        On error, returns False with a warning.

    Caches result for subsequent calls.
    """
    global _WANDB_INCREMENTAL_SUPPORTED

    if _WANDB_INCREMENTAL_SUPPORTED is not None:
        return _WANDB_INCREMENTAL_SUPPORTED

    try:
        import wandb

        # Try to detect server version
        # W&B INCREMENTAL mode requires server v0.70.0+
        # For now, we'll just try to create a table and see if it fails
        # A more robust approach would query the server version API
        _WANDB_INCREMENTAL_SUPPORTED = True
        logger.info("W&B INCREMENTAL table mode assumed supported (will fallback on error)")
        return True
    except Exception as e:
        logger.warning(f"Unable to check W&B server capabilities: {e}. Using standard mode.")
        _WANDB_INCREMENTAL_SUPPORTED = False
        return False


def create_rollout_table(use_incremental: bool = True) -> "wandb.Table":
    """Create a W&B Table for rollout logging.

    Args:
        use_incremental: If True, attempt to use INCREMENTAL log_mode.
            Falls back to standard mode if unsupported.

    Returns:
        wandb.Table configured with ROLLOUT_TABLE_COLUMNS.

    INCREMENTAL mode reduces W&B payload size by only sending new rows
    rather than the entire table on each log. Requires W&B server v0.70.0+.
    """
    import wandb

    try:
        if use_incremental:
            # Try INCREMENTAL mode
            table = wandb.Table(columns=ROLLOUT_TABLE_COLUMNS, log_mode="INCREMENTAL")
            logger.debug("Created W&B rollout table with INCREMENTAL mode")
            return table
    except (TypeError, ValueError, AttributeError) as e:
        # INCREMENTAL mode not supported
        logger.warning(
            f"W&B INCREMENTAL table mode not supported ({e}). "
            "Falling back to standard mode. Table payloads may be larger."
        )

    # Fallback to standard mode
    table = wandb.Table(columns=ROLLOUT_TABLE_COLUMNS)
    logger.debug("Created W&B rollout table with standard mode")
    return table


def add_sampled_rollouts_to_table(
    table: "wandb.Table",
    step: int,
    sampled_rollouts: list[tuple[RolloutRecord, bool]],
) -> None:
    """Add sampled rollouts to W&B table.

    Args:
        table: W&B Table to populate
        step: Current training step
        sampled_rollouts: List of (RolloutRecord, is_top) tuples

    Adds rows matching ROLLOUT_TABLE_COLUMNS order:
        ["step", "prompt_id", "prompt_text", "completion", "reward",
         "solver_reward", "verifier_reward", "judge_reward",
         "role_assignments", "is_top"]
    """
    for rollout, is_top in sampled_rollouts:
        table.add_data(
            step,
            rollout.prompt_id,
            rollout.prompt_text,
            rollout.completion,
            rollout.reward,
            rollout.solver_reward,
            rollout.verifier_reward,
            rollout.judge_reward,
            rollout.role_assignments,
            is_top,
        )

    logger.debug(f"Added {len(sampled_rollouts)} rollouts to W&B table for step {step}")


def log_rollout_table(
    wandb_run: "wandb.Run",
    table: "wandb.Table",
    step: int,
) -> None:
    """Log rollout table to W&B.

    Args:
        wandb_run: Active W&B run
        table: Populated W&B Table
        step: Training step for logging

    Logs to "debate/rollouts" key. Catches payload size errors and warns.
    """
    try:
        wandb_run.log({"debate/rollouts": table}, step=step)
        logger.debug(f"Logged rollout table for step {step}")
    except Exception as e:
        # Common error: payload too large
        if "payload" in str(e).lower() or "size" in str(e).lower():
            logger.error(
                f"W&B rollout table payload too large at step {step}: {e}. "
                "Consider reducing prompts_per_step or top_k/bottom_k."
            )
        else:
            logger.error(f"Failed to log rollout table at step {step}: {e}")
