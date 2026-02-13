"""Shared configuration constants for Streamlit viewer.

This module contains role colors, marker strings, and data source constants
used across all viewer components.
"""

from __future__ import annotations

# ============================================================================
# Role Colors for UI
# ============================================================================

ROLE_COLORS = {
    "solver": "#3498db",    # Blue
    "verifier": "#2ecc71",  # Green
    "judge": "#e67e22",     # Orange
}
"""Color scheme for role-based UI elements."""


# ============================================================================
# Marker Strings (imported from training code)
# ============================================================================

try:
    # Import from training code to ensure consistency
    from src.training.wandb_enrichment.role_mask_computer import (
        DEFAULT_VERIFICATION_MARKERS,
        DEFAULT_FINAL_ANSWER_MARKERS,
    )
except ImportError:
    # Fallback if running standalone or training code not available
    DEFAULT_VERIFICATION_MARKERS: list[str] = [
        "verify",
        "check your",
        "review the",
    ]
    """Character strings that indicate the start of verifier role."""

    DEFAULT_FINAL_ANSWER_MARKERS: list[str] = [
        "final answer",
        "provide your final",
        "give your final",
    ]
    """Character strings that indicate the start of judge role."""


# ============================================================================
# Data Source Constants
# ============================================================================

DATA_SOURCE_PARQUET = "parquet"
"""Data source type for local Parquet files."""

DATA_SOURCE_WANDB = "wandb"
"""Data source type for W&B Tables."""

DEFAULT_PARQUET_GLOB = "batch_debug_data_train_*.parquet"
"""Glob pattern for debug data Parquet files (matches debug_data_writer.py naming)."""

WANDB_ENTITY = "cohere"
"""Default W&B entity."""

WANDB_PROJECT = "multiagent-debate-rl"
"""Default W&B project name."""

GRPO_ROLLOUTS_PER_PROMPT = 8
"""Number of rollouts per prompt in GRPO training."""
