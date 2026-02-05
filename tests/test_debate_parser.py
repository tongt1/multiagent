"""Tests for debate turn parsing logic."""

import pytest

from streamlit_viewer.lib.debate_parser import (
    DebateTurn,
    find_marker_in_text,
    parse_debate_turns,
    attach_per_turn_rewards,
    compute_reward_variance,
)


def test_find_marker_in_text():
    """Test marker search with case-insensitive matching."""
    text = "Hello world. Please VERIFY this answer. Thank you."
    markers = ["verify", "check"]

    offset = find_marker_in_text(text, markers)
    assert offset == 20  # Position of "VERIFY"


def test_find_marker_earliest():
    """Test that earliest marker is returned when multiple match."""
    text = "First check this. Then verify that."
    markers = ["verify", "check"]

    offset = find_marker_in_text(text, markers)
    assert offset == 6  # Position of "check" (earlier than "verify")


def test_find_marker_not_found():
    """Test marker search when no markers present."""
    text = "Hello world"
    markers = ["verify", "check"]

    offset = find_marker_in_text(text, markers)
    assert offset == -1


def test_parse_three_role_debate():
    """Test parsing full debate with all three roles."""
    completion = """Solver reasoning here.
Please verify this solution.
Verifier checking here.
Provide your final answer.
Judge decision here."""

    turns = parse_debate_turns(
        completion,
        verification_markers=["verify"],
        final_answer_markers=["final answer"],
    )

    assert len(turns) == 3
    assert turns[0].role == "solver"
    assert turns[1].role == "verifier"
    assert turns[2].role == "judge"
    assert "Solver reasoning" in turns[0].text
    assert "Verifier checking" in turns[1].text
    assert "Judge decision" in turns[2].text
    assert all(turn.reward is None for turn in turns)  # Rewards not attached yet


def test_parse_short_debate_no_markers():
    """Test parsing when no markers found (short debate fallback)."""
    completion = "Just solver output here, no markers."

    turns = parse_debate_turns(completion)

    assert len(turns) == 1
    assert turns[0].role == "solver"
    assert turns[0].text == completion.strip()
    assert turns[0].token_count == len(completion.split())


def test_parse_two_role_debate():
    """Test parsing with only verification marker (solver + verifier)."""
    completion = """Solver work here.
Please verify this.
Verifier work here."""

    turns = parse_debate_turns(
        completion,
        verification_markers=["verify"],
        final_answer_markers=["final answer"],
    )

    assert len(turns) == 2
    assert turns[0].role == "solver"
    assert turns[1].role == "verifier"


def test_parse_empty_text():
    """Test parsing empty string."""
    turns = parse_debate_turns("")
    assert len(turns) == 0

    turns = parse_debate_turns("   ")
    assert len(turns) == 0


def test_parse_markers_case_insensitive():
    """Test that marker matching is case-insensitive."""
    completion = """Solver text.
VERIFY THIS.
Verifier text."""

    turns = parse_debate_turns(
        completion,
        verification_markers=["verify"],  # lowercase
        final_answer_markers=["final answer"],
    )

    assert len(turns) == 2
    assert turns[0].role == "solver"
    assert turns[1].role == "verifier"


def test_debate_turn_reward_field():
    """Test that DebateTurn has reward field with correct default."""
    turn = DebateTurn(role="solver", text="test")
    assert turn.reward is None

    turn_with_reward = DebateTurn(role="solver", text="test", reward=0.5)
    assert turn_with_reward.reward == 0.5


def test_attach_per_turn_rewards():
    """Test attaching per-role rewards to parsed turns."""
    completion = """Solver reasoning.
Please verify this.
Verifier checking.
Provide your final answer.
Judge decision."""

    turns = parse_debate_turns(
        completion,
        verification_markers=["verify"],
        final_answer_markers=["final answer"],
    )

    # Initially rewards are None
    assert all(turn.reward is None for turn in turns)

    # Attach rewards
    per_turn_rewards = {
        "solver": 0.3,
        "verifier": 0.5,
        "judge": 0.8,
    }
    updated_turns = attach_per_turn_rewards(turns, per_turn_rewards)

    assert updated_turns[0].reward == 0.3  # solver
    assert updated_turns[1].reward == 0.5  # verifier
    assert updated_turns[2].reward == 0.8  # judge


def test_attach_per_turn_rewards_none_input():
    """Test attach_per_turn_rewards with None input (no-op)."""
    completion = "Solver text."
    turns = parse_debate_turns(completion)

    updated_turns = attach_per_turn_rewards(turns, None)
    assert updated_turns[0].reward is None


def test_attach_per_turn_rewards_partial():
    """Test attach_per_turn_rewards with partial reward data."""
    completion = "Solver text."
    turns = parse_debate_turns(completion)

    per_turn_rewards = {"verifier": 0.5}  # Only verifier reward, solver missing
    updated_turns = attach_per_turn_rewards(turns, per_turn_rewards)

    assert updated_turns[0].reward is None  # solver not in dict


def test_compute_reward_variance():
    """Test reward variance calculation."""
    rewards = [1.0, 0.0, 0.5, 0.5]
    variance = compute_reward_variance(rewards)

    # Mean = 0.5, variance = ((0.5)^2 + (-0.5)^2 + 0 + 0) / 4 = 0.125
    assert abs(variance - 0.125) < 1e-6


def test_compute_reward_variance_zero_std():
    """Test variance when all rewards are equal."""
    rewards = [0.5, 0.5, 0.5, 0.5]
    variance = compute_reward_variance(rewards)
    assert variance == 0.0


def test_compute_reward_variance_single_value():
    """Test variance with single value."""
    rewards = [1.0]
    variance = compute_reward_variance(rewards)
    assert variance == 0.0


def test_compute_reward_variance_empty():
    """Test variance with empty list."""
    rewards = []
    variance = compute_reward_variance(rewards)
    assert variance == 0.0


def test_token_count_estimation():
    """Test that token_count is estimated correctly."""
    completion = "This is a test with exactly eight words here."
    turns = parse_debate_turns(completion)

    assert len(turns) == 1
    assert turns[0].token_count == 9  # word count
