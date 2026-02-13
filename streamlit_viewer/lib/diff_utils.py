"""Text comparison utilities for rollout analysis.

This module provides diff highlighting for side-by-side rollout comparison using
Python's difflib library. Used by the Comparison page for visualizing differences
between debate trajectories.
"""

from __future__ import annotations

import difflib
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


def compute_text_diff(text1: str, text2: str) -> str:
    """Generate HTML diff table for two text strings.

    Args:
        text1: First text (left side)
        text2: Second text (right side)

    Returns:
        HTML string with side-by-side diff table wrapped in scrollable div
    """
    diff_maker = difflib.HtmlDiff(wrapcolumn=80)

    html_diff = diff_maker.make_table(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromdesc="Text 1",
        todesc="Text 2",
        context=True,
        numlines=3,
    )

    # Wrap in scrollable div
    wrapped_html = f"""
    <div style="max-height: 500px; overflow-y: auto; border: 1px solid #ddd;">
        {html_diff}
    </div>
    """

    return wrapped_html


def render_rollout_diff(rollout1: dict, rollout2: dict) -> None:
    """Display side-by-side comparison of two rollouts with diff highlighting.

    Args:
        rollout1: Rollout dict with "completion" and "reward" keys
        rollout2: Rollout dict with "completion" and "reward" keys

    Displays:
        - Header with reward comparison
        - HTML diff table via st.components
        - Text similarity metric
    """
    # Extract data
    text1 = rollout1.get("completion", "")
    text2 = rollout2.get("completion", "")
    reward1 = rollout1.get("reward", 0.0)
    reward2 = rollout2.get("reward", 0.0)

    # Header with rewards
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Left Rollout Reward", f"{reward1:.2f}")
    with col2:
        st.metric("Right Rollout Reward", f"{reward2:.2f}")

    # Generate diff
    diff_maker = difflib.HtmlDiff(wrapcolumn=80)

    html_diff = diff_maker.make_table(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromdesc=f"Rollout (Reward: {reward1:.2f})",
        todesc=f"Rollout (Reward: {reward2:.2f})",
        context=True,
        numlines=3,
    )

    # Render diff
    components.html(html_diff, height=500, scrolling=True)

    # Similarity metric
    similarity = compute_similarity(text1, text2)
    st.metric("Text Similarity", f"{similarity:.1%}")


def compute_similarity(text1: str, text2: str) -> float:
    """Compute text similarity ratio between two strings.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity ratio between 0.0 (completely different) and 1.0 (identical)
    """
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()
