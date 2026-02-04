"""Programmatic W&B workspace template for debate training dashboards.

This module provides functions to create consistent W&B workspace layouts with
pre-configured panels for debate metrics. The template ensures every training
run gets the same dashboard organization automatically.

Usage:
    # At start of training run
    import wandb
    from src.training.wandb_enrichment.workspace_template import create_workspace_if_new_run

    run = wandb.init(project="debate-training", ...)
    create_workspace_if_new_run(run)

    # Or manually
    from src.training.wandb_enrichment.workspace_template import create_debate_workspace

    create_debate_workspace(entity="cohere", project="debate-training", run_id="abc123")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Import metric constants from metric_schema (DO NOT hardcode metric strings)
from src.training.wandb_enrichment.metric_schema import (
    METRIC_FRAC_ZERO_STD,
    METRIC_FRAC_ZERO_STD_CORRECT,
    METRIC_FRAC_ZERO_STD_INCORRECT,
    METRIC_GRAD_GLOBAL_NORM,
    METRIC_KL_JUDGE,
    METRIC_KL_SOLVER,
    METRIC_KL_VERIFIER,
    METRIC_MEAN_REWARD_STD,
    METRIC_REWARD_JUDGE,
    METRIC_REWARD_SOLVER,
    METRIC_REWARD_VERIFIER,
)

logger = logging.getLogger(__name__)


def create_debate_workspace(
    entity: str,
    project: str,
    run_id: str | None = None,
) -> None:
    """Create a W&B workspace template for debate training runs.

    This function programmatically generates a W&B workspace with 4 sections:
    1. Per-Role Rewards: Solver, Verifier, Judge reward curves
    2. Per-Role KL Divergence: KL from reference policy per role
    3. Training Health: Zero-advantage detection, gradient norms
    4. Rollout Samples: (collapsed by default, for W&B Tables)

    Args:
        entity: W&B team/user name
        project: W&B project name
        run_id: Optional specific run ID. If None, applies to all runs in project.

    Example:
        >>> create_debate_workspace("cohere", "debate-rlhf", "abc123")
    """
    try:
        import wandb_workspaces.reports.v2 as wr
    except ImportError:
        logger.warning(
            "wandb-workspaces library not available. "
            "Workspace template creation skipped. "
            "Install with: pip install wandb-workspaces"
        )
        return

    try:
        # Create workspace with 4 sections
        workspace = wr.Report(
            entity=entity,
            project=project,
            title="Debate Training Dashboard",
            description="Multi-agent debate RL training metrics (auto-generated)",
        )

        # =====================================================================
        # Section 1: Per-Role Rewards
        # =====================================================================
        workspace.blocks = workspace.blocks + [
            wr.H1(text="Per-Role Rewards"),
            wr.LinePlot(
                title="Per-Role Reward Curves",
                x="Step",
                y=[METRIC_REWARD_SOLVER, METRIC_REWARD_VERIFIER, METRIC_REWARD_JUDGE],
                smoothing_factor=0.6,
                range_x=None,
                range_y=None,
                legend_position="north",
            ),
        ]

        # =====================================================================
        # Section 2: Per-Role KL Divergence
        # =====================================================================
        workspace.blocks = workspace.blocks + [
            wr.H1(text="Per-Role KL Divergence"),
            wr.LinePlot(
                title="Per-Role KL Divergence",
                x="Step",
                y=[METRIC_KL_SOLVER, METRIC_KL_VERIFIER, METRIC_KL_JUDGE],
                smoothing_factor=0.6,
                range_x=None,
                range_y=None,
                legend_position="north",
            ),
        ]

        # =====================================================================
        # Section 3: Training Health
        # =====================================================================
        workspace.blocks = workspace.blocks + [
            wr.H1(text="Training Health"),
            wr.LinePlot(
                title="Zero-Advantage Detection",
                x="Step",
                y=[METRIC_FRAC_ZERO_STD, METRIC_MEAN_REWARD_STD],
                smoothing_factor=0.6,
                range_x=None,
                range_y=None,
                legend_position="north",
            ),
            wr.LinePlot(
                title="Zero-Std Breakdown (Correct vs Incorrect)",
                x="Step",
                y=[METRIC_FRAC_ZERO_STD_CORRECT, METRIC_FRAC_ZERO_STD_INCORRECT],
                smoothing_factor=0.6,
                range_x=None,
                range_y=None,
                legend_position="north",
            ),
            wr.LinePlot(
                title="Global Gradient Norm",
                x="Step",
                y=[METRIC_GRAD_GLOBAL_NORM],
                smoothing_factor=0.6,
                range_x=None,
                range_y=None,
                legend_position="north",
            ),
        ]

        # =====================================================================
        # Section 4: Rollout Samples
        # =====================================================================
        workspace.blocks = workspace.blocks + [
            wr.H1(text="Rollout Samples"),
            wr.P(
                text=(
                    "W&B Tables for rollout inspection will appear here during training. "
                    "Top-k and bottom-k rollouts logged per prompt for debugging."
                )
            ),
        ]

        # Save the workspace
        workspace.save()
        logger.info(f"Created W&B workspace template for {entity}/{project}")

    except Exception as e:
        logger.warning(f"Failed to create W&B workspace template: {e}")


def create_workspace_if_new_run(wandb_run: Any) -> None:
    """Create workspace template if this is a new run.

    This is a convenience wrapper for calling from wandb.init() context.

    Args:
        wandb_run: The wandb.Run object from wandb.init()

    Example:
        >>> import wandb
        >>> run = wandb.init(project="debate-training")
        >>> create_workspace_if_new_run(run)
    """
    if wandb_run is None:
        logger.warning("wandb_run is None, skipping workspace creation")
        return

    try:
        entity = wandb_run.entity
        project = wandb_run.project
        run_id = wandb_run.id

        # Only create workspace once per project (not per run)
        # The workspace template applies to all runs in the project
        # Check if this is step 0 to avoid recreating on resume
        if wandb_run.step == 0:
            create_debate_workspace(entity=entity, project=project, run_id=run_id)
    except Exception as e:
        logger.warning(f"Failed to create workspace in wandb.init() context: {e}")


def main() -> None:
    """CLI entrypoint for creating workspace templates.

    Usage:
        python -m src.training.wandb_enrichment.workspace_template \\
            --entity cohere \\
            --project debate-training \\
            --run-id abc123
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Create W&B workspace template for debate training"
    )
    parser.add_argument("--entity", required=True, help="W&B team/user name")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run-id", default=None, help="Specific run ID (optional)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    create_debate_workspace(
        entity=args.entity,
        project=args.project,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
