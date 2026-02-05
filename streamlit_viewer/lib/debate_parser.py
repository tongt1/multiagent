"""Debate turn extraction from raw completion text.

This module parses multi-turn debate trajectories into individual role turns
(solver, verifier, judge) using configurable marker strings. It mirrors the
parsing logic from role_mask_computer.py but operates on complete text strings
rather than token sequences.

The reward field on DebateTurn is populated by attach_per_turn_rewards() when
per-role reward data is available from rollout metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from streamlit_viewer.config import (
    DEFAULT_VERIFICATION_MARKERS,
    DEFAULT_FINAL_ANSWER_MARKERS,
)


@dataclass
class DebateTurn:
    """A single role turn in a debate trajectory.

    Attributes:
        role: Role name ("solver", "verifier", or "judge")
        text: The text content of this turn
        token_count: Estimated token count (word-based approximation)
        reward: Per-turn reward value when available (populated by attach_per_turn_rewards)
    """
    role: str
    text: str
    token_count: Optional[int] = None
    reward: Optional[float] = None


def find_marker_in_text(text: str, markers: list[str]) -> int:
    """Search text for any marker string and return character offset.

    Args:
        text: Full trajectory text to search
        markers: List of marker strings to search for

    Returns:
        Character offset of the FIRST (earliest) marker found, or -1 if no marker found.
        Case-insensitive matching.
    """
    text_lower = text.lower()

    earliest_offset = -1
    for marker in markers:
        offset = text_lower.find(marker.lower())
        if offset >= 0:
            if earliest_offset < 0 or offset < earliest_offset:
                earliest_offset = offset

    return earliest_offset


def parse_debate_turns(
    completion: str,
    verification_markers: Optional[list[str]] = None,
    final_answer_markers: Optional[list[str]] = None,
) -> list[DebateTurn]:
    """Parse debate completion text into role turns.

    This function extracts up to 3 turns (solver, verifier, judge) by searching
    for marker strings that indicate role boundaries. The reward field on each
    turn is left as None - callers should use attach_per_turn_rewards() to populate.

    Args:
        completion: Raw debate trajectory text
        verification_markers: Marker strings for verifier role (None = use config defaults)
        final_answer_markers: Marker strings for judge role (None = use config defaults)

    Returns:
        List of DebateTurn objects (1-3 turns depending on markers found).
        Empty list if completion is empty.

    Behavior:
        - No markers found: Single solver turn (short debate fallback)
        - Only verification marker: Solver + verifier turns
        - Both markers: Solver + verifier + judge turns
    """
    if not completion or not completion.strip():
        return []

    if verification_markers is None:
        verification_markers = DEFAULT_VERIFICATION_MARKERS
    if final_answer_markers is None:
        final_answer_markers = DEFAULT_FINAL_ANSWER_MARKERS

    # Find marker positions
    verify_offset = find_marker_in_text(completion, verification_markers)
    final_offset = find_marker_in_text(completion, final_answer_markers)

    turns: list[DebateTurn] = []

    # Case 1: No markers found - single solver turn (short debate)
    if verify_offset < 0:
        solver_text = completion.strip()
        if solver_text:
            turns.append(DebateTurn(
                role="solver",
                text=solver_text,
                token_count=len(solver_text.split()),
            ))
        return turns

    # Case 2: Only verification marker found - solver + verifier
    if final_offset < 0:
        solver_text = completion[:verify_offset].strip()
        verifier_text = completion[verify_offset:].strip()

        if solver_text:
            turns.append(DebateTurn(
                role="solver",
                text=solver_text,
                token_count=len(solver_text.split()),
            ))
        if verifier_text:
            turns.append(DebateTurn(
                role="verifier",
                text=verifier_text,
                token_count=len(verifier_text.split()),
            ))
        return turns

    # Case 3: Both markers found - solver + verifier + judge
    # Edge case: final marker before verify marker (malformed)
    if final_offset < verify_offset:
        # Treat as short debate
        solver_text = completion.strip()
        if solver_text:
            turns.append(DebateTurn(
                role="solver",
                text=solver_text,
                token_count=len(solver_text.split()),
            ))
        return turns

    solver_text = completion[:verify_offset].strip()
    verifier_text = completion[verify_offset:final_offset].strip()
    judge_text = completion[final_offset:].strip()

    if solver_text:
        turns.append(DebateTurn(
            role="solver",
            text=solver_text,
            token_count=len(solver_text.split()),
        ))
    if verifier_text:
        turns.append(DebateTurn(
            role="verifier",
            text=verifier_text,
            token_count=len(verifier_text.split()),
        ))
    if judge_text:
        turns.append(DebateTurn(
            role="judge",
            text=judge_text,
            token_count=len(judge_text.split()),
        ))

    return turns


def attach_per_turn_rewards(
    turns: list[DebateTurn],
    per_turn_rewards: Optional[dict[str, float]],
) -> list[DebateTurn]:
    """Attach per-role rewards to parsed debate turns.

    This function populates the reward field on each DebateTurn when per-role
    reward data is available from rollout metadata (e.g., solver_reward,
    verifier_reward, judge_reward columns in Parquet debug data).

    Args:
        turns: List of parsed DebateTurn objects
        per_turn_rewards: Dict mapping role name to reward value, e.g.
            {"solver": 0.3, "verifier": 0.5, "judge": 0.8}
            If None, turns are returned unchanged.

    Returns:
        The same turns list with reward field populated (mutates in place and returns).
    """
    if per_turn_rewards is None:
        return turns

    for turn in turns:
        turn.reward = per_turn_rewards.get(turn.role, None)

    return turns


def compute_reward_variance(rewards: list[float]) -> float:
    """Compute variance of a list of rewards.

    Used for prompt sorting by reward diversity in the UI.

    Args:
        rewards: List of reward values

    Returns:
        Variance of rewards, or 0.0 if fewer than 2 rewards.
    """
    if len(rewards) < 2:
        return 0.0

    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    return variance
