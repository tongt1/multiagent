"""Plotly-based visualization builders for debate analysis.

This module provides chart builders for reward curves, debate flow timelines,
and distribution plots. All charts use the ROLE_COLORS config for consistent styling.
"""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from streamlit_viewer.config import ROLE_COLORS
from streamlit_viewer.lib.debate_parser import DebateTurn


def render_debate_timeline(turns: list[DebateTurn], title: str = "Debate Flow Timeline") -> None:
    """Render horizontal swimlane timeline showing debate flow with per-turn rewards.

    Each turn is displayed as a horizontal bar on its role's swimlane. Bar width
    represents token count. Per-turn rewards are annotated directly on bars when
    available from DebateTurn.reward field.

    Args:
        turns: List of DebateTurn objects (from parse_debate_turns + attach_per_turn_rewards)
        title: Chart title

    Displays:
        Plotly horizontal bar chart with role swimlanes (Y-axis) and token positions (X-axis).
        Colors from ROLE_COLORS config. Hover shows full text preview.
    """
    if not turns:
        st.warning("No debate turns to display")
        return

    fig = go.Figure()

    # Build traces for each role's turns
    # X position tracks cumulative token position
    x_start = 0
    role_order = ["solver", "verifier", "judge"]  # Fixed Y-axis order

    for turn in turns:
        token_count = turn.token_count or len(turn.text.split())

        # Build bar annotation text
        if turn.reward is not None:
            bar_text = f"{token_count} tokens (r={turn.reward:.2f})"
        else:
            bar_text = f"{token_count} tokens"

        # Hover template with text preview
        text_preview = turn.text[:100] + ("..." if len(turn.text) > 100 else "")
        hover_template = (
            f"<b>{turn.role.capitalize()}</b><br>"
            f"Tokens: {token_count}<br>"
        )
        if turn.reward is not None:
            hover_template += f"Reward: {turn.reward:.2f}<br>"
        hover_template += f"Text: {text_preview}<extra></extra>"

        # Add horizontal bar
        fig.add_trace(go.Bar(
            x=[token_count],
            y=[turn.role],
            orientation='h',
            base=x_start,
            marker_color=ROLE_COLORS.get(turn.role, "#95a5a6"),
            text=bar_text,
            textposition="inside",
            textfont=dict(size=10, color="white"),
            hovertemplate=hover_template,
            showlegend=False,
        ))

        x_start += token_count

    # Layout configuration
    fig.update_layout(
        title=title,
        xaxis_title="Token Position",
        yaxis_title="Role",
        height=250,
        barmode='stack',
        showlegend=False,
        margin=dict(l=80, r=20, t=40, b=40),
        yaxis=dict(categoryorder='array', categoryarray=role_order),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_reward_curve(
    steps_data: list[dict],
    run_label: str = "",
    color: Optional[str] = None,
) -> go.Figure:
    """Build reward curve line chart for a single run.

    Args:
        steps_data: List of {"step": int, "avg_reward": float, "reward_std": float}
        run_label: Label for this run's trace
        color: Optional color hex string. If None, uses Plotly default sequence.

    Returns:
        go.Figure with line + error band traces. Caller can add more traces or render.
    """
    if not steps_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
        )
        return fig

    steps = [d["step"] for d in steps_data]
    rewards = [d["avg_reward"] for d in steps_data]
    stds = [d.get("reward_std", 0.0) for d in steps_data]

    fig = go.Figure()

    # Add error band (upper bound)
    fig.add_trace(go.Scatter(
        x=steps,
        y=[r + s for r, s in zip(rewards, stds)],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
        marker=dict(color=color) if color else {},
    ))

    # Main line trace
    trace_kwargs = {
        "x": steps,
        "y": rewards,
        "mode": 'lines+markers',
        "name": run_label or "Reward",
        "line": dict(width=2),
        "marker": dict(size=6),
    }
    if color:
        trace_kwargs["line"]["color"] = color
        trace_kwargs["marker"]["color"] = color

    fig.add_trace(go.Scatter(**trace_kwargs))

    # Add error band (lower bound with fill)
    fig.add_trace(go.Scatter(
        x=steps,
        y=[r - s for r, s in zip(rewards, stds)],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)" if color else "rgba(100, 100, 100, 0.2)",
        showlegend=False,
        hoverinfo='skip',
    ))

    fig.update_layout(
        xaxis_title="Training Step",
        yaxis_title="Average Reward",
        hovermode='x unified',
    )

    return fig


def render_reward_distribution(
    rewards: list[float],
    run_label: str = "",
) -> go.Figure:
    """Build histogram of reward distribution.

    Args:
        rewards: List of reward values
        run_label: Label for this distribution

    Returns:
        go.Figure with histogram trace
    """
    fig = go.Figure()

    if not rewards:
        fig.add_annotation(
            text="No rewards to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
        )
        return fig

    fig.add_trace(go.Histogram(
        x=rewards,
        nbinsx=20,
        name=run_label or "Reward Distribution",
        marker=dict(line=dict(width=1, color="white")),
    ))

    fig.update_layout(
        xaxis_title="Reward",
        yaxis_title="Count",
        showlegend=bool(run_label),
    )

    return fig


def render_per_role_loss_curves(steps_data: list[dict]) -> go.Figure:
    """Build per-role loss curves over training steps.

    Args:
        steps_data: List of dicts with {"step": int, "loss_solver": float,
                    "loss_verifier": float, "loss_judge": float}

    Returns:
        go.Figure with three loss traces colored by role
    """
    fig = go.Figure()

    if not steps_data:
        fig.add_annotation(
            text="No per-role loss data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
        )
        return fig

    steps = [d["step"] for d in steps_data]

    for role in ["solver", "verifier", "judge"]:
        loss_key = f"loss_{role}"
        losses = [d.get(loss_key, None) for d in steps_data]

        # Filter out None values
        valid_steps = [s for s, l in zip(steps, losses) if l is not None]
        valid_losses = [l for l in losses if l is not None]

        if valid_losses:
            fig.add_trace(go.Scatter(
                x=valid_steps,
                y=valid_losses,
                mode='lines+markers',
                name=role.capitalize(),
                line=dict(color=ROLE_COLORS[role], width=2),
                marker=dict(size=4),
            ))

    fig.update_layout(
        xaxis_title="Training Step",
        yaxis_title="Loss",
        hovermode='x unified',
        legend=dict(x=1.05, y=1),
    )

    return fig


def render_token_distribution_pie(turns: list[DebateTurn]) -> None:
    """Render pie chart showing token distribution across roles.

    Args:
        turns: List of DebateTurn objects

    Displays:
        Plotly pie chart with role colors from config
    """
    if not turns:
        st.warning("No debate turns to display")
        return

    # Aggregate tokens by role
    role_tokens = {}
    for turn in turns:
        token_count = turn.token_count or len(turn.text.split())
        role_tokens[turn.role] = role_tokens.get(turn.role, 0) + token_count

    # Build pie chart
    roles = list(role_tokens.keys())
    tokens = list(role_tokens.values())
    colors = [ROLE_COLORS.get(role, "#95a5a6") for role in roles]

    fig = go.Figure(data=[go.Pie(
        labels=[r.capitalize() for r in roles],
        values=tokens,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Tokens: %{value}<br>%{percent}<extra></extra>',
    )])

    fig.update_layout(
        title="Token Distribution by Role",
        height=300,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)
