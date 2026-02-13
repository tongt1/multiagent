"""Reusable UI components for Streamlit viewer.

This module provides chat bubble rendering, rollout cards, and LaTeX-aware
text rendering for debate visualizations.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import streamlit as st

from streamlit_viewer.config import ROLE_COLORS
from streamlit_viewer.lib.debate_parser import DebateTurn


def render_math_text(text: str) -> None:
    """Render text with LaTeX math expressions.

    Splits text by LaTeX delimiters and renders:
    - $$...$$ as display math (st.latex, centered)
    - $...$ as inline math (st.markdown, Streamlit native support)
    - Plain text as markdown (with $ escaping to prevent accidental LaTeX)

    Args:
        text: Text potentially containing LaTeX expressions
    """
    if not text or not text.strip():
        return

    # Split by LaTeX delimiters: $$...$$ (display) and $...$ (inline)
    # Use re.DOTALL to match across newlines
    parts = re.split(r'(\$\$.+?\$\$|\$.+?\$)', text, flags=re.DOTALL)

    for part in parts:
        if not part:
            continue

        # Display math: $$...$$
        if part.startswith('$$') and part.endswith('$$'):
            inner = part[2:-2].strip()
            if inner:
                st.latex(inner)
        # Inline math: $...$
        elif part.startswith('$') and part.endswith('$'):
            # Streamlit markdown natively supports inline LaTeX
            st.markdown(part, unsafe_allow_html=True)
        # Plain text: escape any stray $ to prevent accidental LaTeX
        else:
            escaped = part.replace('$', r'\$')
            st.markdown(escaped, unsafe_allow_html=True)


def render_debate_as_chat(turns: list[DebateTurn]) -> None:
    """Render debate turns as colored chat bubbles.

    Each role gets a colored bubble with:
    - Role name in UPPERCASE, bold, colored
    - Optional token count badge
    - Optional per-turn reward badge (colored: green if >= 0.5, red if < 0.5)
    - Debate text rendered with LaTeX support

    Args:
        turns: List of DebateTurn objects from parse_debate_turns
    """
    for turn in turns:
        role = turn.role
        color = ROLE_COLORS.get(role, "#95a5a6")  # Gray fallback

        # Build header with role name
        header_parts = [f"<strong style='color: {color}'>{role.upper()}</strong>"]

        # Add token count badge if available
        if turn.token_count is not None:
            header_parts.append(
                f"<span style='font-size: 0.85em; color: #7f8c8d; margin-left: 8px;'>"
                f"{turn.token_count} tokens</span>"
            )

        # Add reward badge if available
        if turn.reward is not None:
            reward_color = "#27ae60" if turn.reward >= 0.5 else "#e74c3c"  # Green or red
            header_parts.append(
                f"<span style='font-size: 0.85em; color: {reward_color}; margin-left: 8px;'>"
                f"reward: {turn.reward:.2f}</span>"
            )

        header = " ".join(header_parts)

        # Chat bubble style: background at 12% opacity, left border 4px solid
        bubble_bg = color + "20"  # Add 20 for 12% opacity in hex
        bubble_style = (
            f"background-color: {bubble_bg}; "
            f"border-left: 4px solid {color}; "
            f"padding: 12px; "
            f"margin: 8px 0; "
            f"border-radius: 4px;"
        )

        # Render bubble
        st.markdown(
            f"<div style='{bubble_style}'>{header}</div>",
            unsafe_allow_html=True,
        )

        # Render debate text with LaTeX support
        render_math_text(turn.text)


def render_rollout_card(
    rollout_data: dict,
    turns: list[DebateTurn],
    expanded: bool = False,
) -> None:
    """Render a rollout card with reward, token counts, and debate turns.

    Args:
        rollout_data: Dict with rollout metadata (must have "reward" key)
        turns: List of DebateTurn objects
        expanded: If True, always show debate turns. If False, can be used in
                  a collapsed/expandable context.
    """
    # Card header with reward
    reward = rollout_data.get("reward", 0.0)
    reward_color = "#27ae60" if reward >= 0.5 else "#e74c3c"  # Green or red

    st.markdown(
        f"<div style='font-size: 1.2em; font-weight: bold; color: {reward_color};'>"
        f"Reward: {reward:.2f}</div>",
        unsafe_allow_html=True,
    )

    # Token counts per role
    token_counts = {}
    for turn in turns:
        if turn.token_count is not None:
            token_counts[turn.role] = token_counts.get(turn.role, 0) + turn.token_count

    if token_counts:
        token_parts = []
        for role in ["solver", "verifier", "judge"]:
            if role in token_counts:
                color = ROLE_COLORS.get(role, "#95a5a6")
                token_parts.append(
                    f"<span style='color: {color}'>{role.capitalize()}: {token_counts[role]}</span>"
                )

        st.markdown(
            f"<div style='font-size: 0.9em; margin: 8px 0;'>{' | '.join(token_parts)}</div>",
            unsafe_allow_html=True,
        )

    # Per-role loss metrics if available in reward_metrics JSON
    reward_metrics_json = rollout_data.get("reward_metrics")
    if reward_metrics_json:
        try:
            # Parse JSON string if it's a string
            if isinstance(reward_metrics_json, str):
                reward_metrics = json.loads(reward_metrics_json)
            else:
                reward_metrics = reward_metrics_json

            # Extract per-role loss metrics from Phase 6
            loss_metrics = {}
            for role in ["solver", "verifier", "judge"]:
                loss_key = f"debate/loss/{role}"
                if loss_key in reward_metrics:
                    loss_metrics[role] = reward_metrics[loss_key]

            if loss_metrics:
                loss_parts = []
                for role in ["solver", "verifier", "judge"]:
                    if role in loss_metrics:
                        color = ROLE_COLORS.get(role, "#95a5a6")
                        loss_parts.append(
                            f"<span style='color: {color}'>{role}: {loss_metrics[role]:.3f}</span>"
                        )

                st.markdown(
                    f"<div style='font-size: 0.85em; margin: 8px 0; color: #7f8c8d;'>"
                    f"Per-role loss: {' | '.join(loss_parts)}</div>",
                    unsafe_allow_html=True,
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Silently skip if metrics parsing fails

    # Render debate turns if expanded
    if expanded:
        st.markdown("---")
        render_debate_as_chat(turns)


def render_prompt_header(prompt_id: str, reward_variance: float, num_rollouts: int) -> None:
    """Render prompt header with ID, reward variance, and rollout count.

    Args:
        prompt_id: Unique prompt identifier
        reward_variance: Variance of rewards across rollouts
        num_rollouts: Number of rollouts for this prompt
    """
    st.subheader(f"Prompt: {prompt_id}")
    st.caption(
        f"Reward variance: {reward_variance:.4f} | Rollouts: {num_rollouts}"
    )
