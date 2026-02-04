"""Workspace initialization wrapper for training startup.

This module provides a safe wrapper for calling workspace_template.py's
create_workspace_if_new_run() function at training startup. All errors are
caught and logged as warnings (never crashes training).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def init_debate_workspace(wandb_run: Any = None) -> None:
    """Initialize W&B workspace template for the active debate training run.

    Safe to call even if wandb is not available or no run is active.
    All errors are caught and logged as warnings.

    Args:
        wandb_run: The wandb.Run object from wandb.init(), or None to use wandb.run

    Usage:
        >>> import wandb
        >>> from src.training.wandb_enrichment.workspace_init import init_debate_workspace
        >>> run = wandb.init(project="debate-training")
        >>> init_debate_workspace(run)

    Design notes:
        - Lazy imports to avoid import errors in environments without wandb
        - All exceptions caught and logged as warnings
        - Safe to call multiple times (workspace creation is idempotent)
    """
    try:
        # Get active W&B run if not provided
        if wandb_run is None:
            import wandb
            wandb_run = wandb.run

        if wandb_run is None:
            logger.debug("No active W&B run, skipping workspace init")
            return

        # Import and call workspace creation
        from src.training.wandb_enrichment.workspace_template import create_workspace_if_new_run
        create_workspace_if_new_run(wandb_run)
        logger.info("Debate workspace template initialized")

    except ImportError as e:
        logger.warning(f"Cannot init workspace (missing dependency): {e}")
    except Exception as e:
        logger.warning(f"Workspace init failed (non-fatal): {e}")
