"""Canonical debate metric schema for W&B logging and debug data.

This module defines the authoritative metric names, W&B table columns, and Parquet
debug data schema for multi-agent debate training. All debate metrics use the
`debate/` namespace prefix to avoid collision with Flink built-in metrics.

Schema versioning supports future extension without breaking Phase 7 analysis.
"""

from __future__ import annotations

# ============================================================================
# Namespace Convention
# ============================================================================

DEBATE_PREFIX = "debate/"
"""All debate metrics must use this prefix to avoid collision with Flink metrics."""

# ============================================================================
# Per-Role Reward Metrics
# ============================================================================

METRIC_REWARD_SOLVER = f"{DEBATE_PREFIX}reward/solver"
"""Mean reward for solver role across batch."""

METRIC_REWARD_VERIFIER = f"{DEBATE_PREFIX}reward/verifier"
"""Mean reward for verifier role across batch."""

METRIC_REWARD_JUDGE = f"{DEBATE_PREFIX}reward/judge"
"""Mean reward for judge role across batch."""

# ============================================================================
# Zero-Advantage Detection Metrics
# ============================================================================

METRIC_FRAC_ZERO_STD = f"{DEBATE_PREFIX}frac_reward_zero_std"
"""Fraction of prompts with zero reward std across rollouts (all equal rewards)."""

METRIC_FRAC_ZERO_STD_CORRECT = f"{DEBATE_PREFIX}frac_zero_std_correct"
"""Fraction of zero-std prompts where all rollouts are correct."""

METRIC_FRAC_ZERO_STD_INCORRECT = f"{DEBATE_PREFIX}frac_zero_std_incorrect"
"""Fraction of zero-std prompts where all rollouts are incorrect."""

METRIC_MEAN_REWARD_STD = f"{DEBATE_PREFIX}mean_reward_std"
"""Mean of reward std across prompts (higher = more diversity)."""

# ============================================================================
# Per-Role KL Divergence Metrics
# ============================================================================

METRIC_KL_SOLVER = f"{DEBATE_PREFIX}kl/solver"
"""Mean KL divergence from reference policy for solver role."""

METRIC_KL_VERIFIER = f"{DEBATE_PREFIX}kl/verifier"
"""Mean KL divergence from reference policy for verifier role."""

METRIC_KL_JUDGE = f"{DEBATE_PREFIX}kl/judge"
"""Mean KL divergence from reference policy for judge role."""

# ============================================================================
# Gradient Metrics
# ============================================================================

METRIC_GRAD_GLOBAL_NORM = f"{DEBATE_PREFIX}grad/global_norm"
"""Global gradient norm (from advanced_logging or manual computation)."""

# ============================================================================
# W&B Table Schema
# ============================================================================

ROLLOUT_TABLE_COLUMNS = [
    "step",
    "prompt_id",
    "prompt_text",
    "completion",
    "reward",
    "solver_reward",
    "verifier_reward",
    "judge_reward",
    "role_assignments",
    "is_top",
]
"""
Column definitions for W&B rollout table.

Columns:
- step: Training step number
- prompt_id: Unique prompt identifier (from dataset)
- prompt_text: The input problem/question
- completion: Multi-turn debate trajectory as string
- reward: Final reward for this rollout
- solver_reward: Per-role reward contribution (solver)
- verifier_reward: Per-role reward contribution (verifier)
- judge_reward: Per-role reward contribution (judge)
- role_assignments: Role assignment string (e.g., "S:0,V:1,J:2")
- is_top: Boolean indicating if this is a top-k rollout for the prompt
"""

# ============================================================================
# Parquet Debug Data Schema
# ============================================================================

CURRENT_SCHEMA_VERSION = 1
"""Current schema version for debug data Parquet files."""


def get_debug_data_schema() -> dict[str, str]:
    """Get the Parquet debug data schema definition.

    Returns:
        Dictionary mapping column names to their data types.

    Schema versioning:
        - Version 1 (current): Base schema with per-role metrics
        - Future versions: Add columns, never remove (for backward compat)
    """
    return {
        # Schema version for forward compatibility
        "schema_version": "int32",

        # Existing Flink columns (from BatchDebugData)
        "env_name": "string",
        "trajectory": "string",
        "agent_trajectories": "string",
        "exception_info": "string",
        "reward": "float32",
        "reward_metrics": "string",  # JSON string
        "reward_text_info": "string",
        "unique_sample_id": "string",

        # Debate-specific extensions (Plan 05-03)
        "role_assignments": "string",
        "solver_reward": "float32",
        "verifier_reward": "float32",
        "judge_reward": "float32",
        "solver_kl": "float32",
        "verifier_kl": "float32",
        "judge_kl": "float32",
    }


# ============================================================================
# Sampling Configuration Constants
# ============================================================================

DEFAULT_PROMPTS_PER_STEP = 4
"""Default number of prompts to sample per training step for rollout logging."""

DEFAULT_TOP_K = 2
"""Default number of top-performing rollouts to log per prompt."""

DEFAULT_BOTTOM_K = 2
"""Default number of bottom-performing rollouts to log per prompt."""

ROLLOUTS_PER_PROMPT = DEFAULT_TOP_K + DEFAULT_BOTTOM_K
"""Total rollouts logged per prompt (top-k + bottom-k)."""

MAX_ROWS_PER_STEP = DEFAULT_PROMPTS_PER_STEP * ROLLOUTS_PER_PROMPT
"""Maximum W&B table rows per step (4 prompts × 4 rollouts = 16 rows)."""

# ============================================================================
# Metric Collections
# ============================================================================

ALL_DEBATE_METRICS = [
    # Rewards
    METRIC_REWARD_SOLVER,
    METRIC_REWARD_VERIFIER,
    METRIC_REWARD_JUDGE,

    # Zero-advantage detection
    METRIC_FRAC_ZERO_STD,
    METRIC_FRAC_ZERO_STD_CORRECT,
    METRIC_FRAC_ZERO_STD_INCORRECT,
    METRIC_MEAN_REWARD_STD,

    # KL divergence
    METRIC_KL_SOLVER,
    METRIC_KL_VERIFIER,
    METRIC_KL_JUDGE,

    # Gradients
    METRIC_GRAD_GLOBAL_NORM,
]
"""All debate metric names for validation and documentation."""
