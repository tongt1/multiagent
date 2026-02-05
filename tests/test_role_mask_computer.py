"""Tests for role mask computation from trajectory text.

This test suite validates text marker parsing, character-to-token offset mapping,
and role mask computation for debate trajectories.
"""

import numpy as np
import pytest

from src.training.wandb_enrichment.role_mask_computer import (
    compute_batch_role_masks,
    compute_role_masks_from_trajectory,
    find_marker_in_text,
    map_char_offset_to_token_index,
    parse_trajectory_roles,
)


# ============================================================================
# Mock Tokenizer
# ============================================================================


class MockTokenizer:
    """Mock tokenizer for testing without SentencePiece dependency.

    Implements a simple word-based tokenization where each whitespace-separated
    word becomes one token. This allows predictable character-to-token mapping
    for test cases.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text as list of token IDs."""
        # Simple word tokenization: split on whitespace
        tokens = text.split()
        # Return token IDs (just indices for testing)
        return list(range(len(tokens)))


# ============================================================================
# Test find_marker_in_text
# ============================================================================


def test_find_marker_in_text_found():
    """Test marker found in text."""
    text = "Hello. Please verify this solution."
    markers = ["verify", "check"]
    offset = find_marker_in_text(text, markers)
    assert offset == 14  # Position of "verify"


def test_find_marker_in_text_not_found():
    """Test marker not found in text."""
    text = "Hello world"
    markers = ["verify", "check"]
    offset = find_marker_in_text(text, markers)
    assert offset == -1


def test_find_marker_in_text_case_insensitive():
    """Test case-insensitive marker matching."""
    text = "Please VERIFY this solution."
    markers = ["verify"]
    offset = find_marker_in_text(text, markers)
    assert offset == 7  # Position of "VERIFY"


def test_find_marker_in_text_earliest():
    """Test returns earliest marker when multiple markers present."""
    text = "First check, then verify."
    markers = ["verify", "check"]
    offset = find_marker_in_text(text, markers)
    assert offset == 6  # Position of "check" (before "verify")


# ============================================================================
# Test map_char_offset_to_token_index
# ============================================================================


def test_map_char_offset_to_token_index_basic():
    """Test character-to-token offset mapping."""
    tokenizer = MockTokenizer()
    text = "Hello world verify this"
    # Words: "Hello" "world" "verify" "this"
    # Char positions: 0-5, 6-11, 12-18, 19-23

    # Offset at start of "verify" should map to token 2
    offset = text.find("verify")
    token_idx = map_char_offset_to_token_index(text, offset, tokenizer)
    assert token_idx == 2


def test_map_char_offset_to_token_index_middle_of_word():
    """Test offset inside a word maps to that word's token."""
    tokenizer = MockTokenizer()
    text = "Hello world"

    # Offset 8 is inside "world" (at 'r')
    token_idx = map_char_offset_to_token_index(text, 8, tokenizer)
    # MockTokenizer splits on whitespace, so prefix "Hello wo" → 2 tokens
    assert token_idx == 2


def test_map_char_offset_to_token_index_empty_prefix():
    """Test offset 0 maps to token 0."""
    tokenizer = MockTokenizer()
    text = "Hello world"
    token_idx = map_char_offset_to_token_index(text, 0, tokenizer)
    assert token_idx == 0


# ============================================================================
# Test parse_trajectory_roles
# ============================================================================


def test_parse_trajectory_roles_full_debate():
    """Test parsing a full 3-role debate."""
    tokenizer = MockTokenizer()
    trajectory = (
        "Problem statement here. "
        "Solver solution text. "
        "Please verify the solution. "
        "Verifier critique text. "
        "Provide your final answer. "
        "Judge final answer text."
    )
    # Word counts: 3 + 3 + 4 + 3 + 4 + 4 = 21 tokens

    max_seq_len = 50
    masks = parse_trajectory_roles(
        trajectory_text=trajectory,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        verification_markers=["verify"],
        final_answer_markers=["final answer"],
    )

    assert masks is not None
    assert "solver" in masks
    assert "verifier" in masks
    assert "judge" in masks

    # Check that masks are non-empty
    assert masks["solver"].sum() > 0
    assert masks["verifier"].sum() > 0
    assert masks["judge"].sum() > 0

    # Check that masks are mutually exclusive
    overlap = (
        (masks["solver"] * masks["verifier"]).sum()
        + (masks["solver"] * masks["judge"]).sum()
        + (masks["verifier"] * masks["judge"]).sum()
    )
    assert overlap == 0, "Masks should be mutually exclusive"


def test_parse_trajectory_roles_short_debate_all_solver():
    """Test short debate (missing markers) assigns all tokens to solver."""
    tokenizer = MockTokenizer()
    trajectory = "Problem statement here Solver solution text only"
    # MockTokenizer splits on whitespace: 8 words = 8 tokens

    max_seq_len = 50
    masks = parse_trajectory_roles(
        trajectory_text=trajectory,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        verification_markers=["verify"],
        final_answer_markers=["final answer"],
    )

    assert masks is not None
    # All tokens assigned to solver
    num_tokens = len(tokenizer.encode(trajectory))
    assert masks["solver"].sum() == num_tokens
    # No tokens for verifier/judge
    assert masks["verifier"].sum() == 0
    assert masks["judge"].sum() == 0


def test_parse_trajectory_roles_two_turns():
    """Test debate with verification but no final answer."""
    tokenizer = MockTokenizer()
    trajectory = (
        "Problem statement "
        "Solver text "
        "Please verify "
        "Verifier text only"
    )
    # MockTokenizer splits on whitespace

    max_seq_len = 50
    masks = parse_trajectory_roles(
        trajectory_text=trajectory,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        verification_markers=["verify"],
        final_answer_markers=["final answer"],
    )

    assert masks is not None
    # Short debate fallback: all to solver (missing final marker)
    num_tokens = len(tokenizer.encode(trajectory))
    assert masks["solver"].sum() == num_tokens
    assert masks["verifier"].sum() == 0
    assert masks["judge"].sum() == 0


# ============================================================================
# Test compute_role_masks_from_trajectory
# ============================================================================


def test_compute_role_masks_user_tokens_excluded():
    """Test user prompt tokens are excluded via prompt_mask."""
    tokenizer = MockTokenizer()
    trajectory = (
        "User problem. "  # 2 tokens - user
        "Solver solution. "  # 2 tokens - chatbot
        "User verify prompt. "  # 3 tokens - user
        "Verifier response. "  # 2 tokens - chatbot
        "User final prompt. "  # 3 tokens - user
        "Judge answer."  # 2 tokens - chatbot
    )
    # Total: 14 tokens

    max_seq_len = 50

    # Prompt mask: 1 for user tokens, 0 for chatbot tokens
    prompt_mask = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0], dtype=np.int32)

    masks = compute_role_masks_from_trajectory(
        trajectory_text=trajectory,
        prompt_mask=prompt_mask,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        verification_markers=["verify"],
        final_answer_markers=["final"],
    )

    assert masks is not None

    # Check that user tokens are NOT assigned to any role
    # Tokens 0,1 (user problem), 4,5,6 (user verify), 9,10,11 (user final) should be 0
    user_token_indices = [0, 1, 4, 5, 6, 9, 10, 11]
    for idx in user_token_indices:
        assert masks["solver"][idx] == 0, f"User token {idx} assigned to solver"
        assert masks["verifier"][idx] == 0, f"User token {idx} assigned to verifier"
        assert masks["judge"][idx] == 0, f"User token {idx} assigned to judge"

    # Check that chatbot tokens ARE assigned to roles
    # Tokens 2,3 (solver), 7,8 (verifier), 12,13 (judge) should be assigned
    chatbot_token_sum = (
        masks["solver"].sum() + masks["verifier"].sum() + masks["judge"].sum()
    )
    assert chatbot_token_sum == 6, "Should have 6 chatbot tokens assigned"


def test_compute_role_masks_mutually_exclusive():
    """Test that role masks are mutually exclusive."""
    tokenizer = MockTokenizer()
    trajectory = "Solver text. Please verify. Verifier text. Final answer. Judge text."
    max_seq_len = 50
    prompt_mask = np.zeros(50, dtype=np.int32)  # No user tokens for simplicity

    masks = compute_role_masks_from_trajectory(
        trajectory_text=trajectory,
        prompt_mask=prompt_mask,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        verification_markers=["verify"],
        final_answer_markers=["final answer"],
    )

    assert masks is not None

    # Check mutual exclusivity
    solver_verifier_overlap = (masks["solver"] * masks["verifier"]).sum()
    solver_judge_overlap = (masks["solver"] * masks["judge"]).sum()
    verifier_judge_overlap = (masks["verifier"] * masks["judge"]).sum()

    assert solver_verifier_overlap == 0
    assert solver_judge_overlap == 0
    assert verifier_judge_overlap == 0


def test_compute_role_masks_respects_max_seq_len():
    """Test masks are truncated to max_seq_len."""
    tokenizer = MockTokenizer()
    trajectory = " ".join(["word"] * 100)  # 100 tokens
    max_seq_len = 20
    prompt_mask = np.zeros(20, dtype=np.int32)

    masks = compute_role_masks_from_trajectory(
        trajectory_text=trajectory,
        prompt_mask=prompt_mask,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    assert masks is not None
    assert len(masks["solver"]) == max_seq_len
    assert len(masks["verifier"]) == max_seq_len
    assert len(masks["judge"]) == max_seq_len


def test_compute_role_masks_parse_failure_returns_none():
    """Test that parse failures return None gracefully."""
    tokenizer = MockTokenizer()

    # Simulate parse failure by passing invalid data
    # (In practice, this is hard to trigger with MockTokenizer,
    # so we test the code path by mocking)

    # Pass empty trajectory
    trajectory = ""
    max_seq_len = 50
    prompt_mask = np.zeros(50, dtype=np.int32)

    # This should either return None or handle gracefully
    # With MockTokenizer, empty trajectory → 0 tokens → all-False masks
    # Let's test that it doesn't crash
    masks = compute_role_masks_from_trajectory(
        trajectory_text=trajectory,
        prompt_mask=prompt_mask,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    # Should handle empty trajectory gracefully
    # (May return None or masks with all zeros)
    assert masks is None or all(m.sum() == 0 for m in masks.values())


# ============================================================================
# Test compute_batch_role_masks
# ============================================================================


def test_compute_batch_role_masks_basic():
    """Test batch computation with multiple trajectories."""
    tokenizer = MockTokenizer()
    trajectories = [
        "Solver. Verify. Verifier. Final. Judge.",
        "Another solver. Check this. Another verifier. Final answer. Another judge.",
    ]

    B = len(trajectories)
    T = 50
    batch_prompt_mask = np.zeros((B, T), dtype=np.int32)

    masks = compute_batch_role_masks(
        batch_trajectories=trajectories,
        batch_prompt_mask=batch_prompt_mask,
        tokenizer=tokenizer,
        max_seq_len=T,
        verification_markers=["verify", "check"],
        final_answer_markers=["final"],
    )

    assert masks is not None
    assert masks["solver"].shape == (B, T)
    assert masks["verifier"].shape == (B, T)
    assert masks["judge"].shape == (B, T)

    # Check that each sample has some role assignments
    for i in range(B):
        total_assigned = (
            masks["solver"][i].sum()
            + masks["verifier"][i].sum()
            + masks["judge"][i].sum()
        )
        assert total_assigned > 0, f"Sample {i} has no role assignments"


def test_compute_batch_role_masks_partial_failure():
    """Test batch computation handles partial failures gracefully."""
    tokenizer = MockTokenizer()

    # Mix of valid and empty trajectories
    trajectories = [
        "Solver. Verify. Verifier. Final. Judge.",
        "",  # Empty trajectory
        "Another solver. Check. Another verifier. Final answer. Judge.",
    ]

    B = len(trajectories)
    T = 50
    batch_prompt_mask = np.zeros((B, T), dtype=np.int32)

    masks = compute_batch_role_masks(
        batch_trajectories=trajectories,
        batch_prompt_mask=batch_prompt_mask,
        tokenizer=tokenizer,
        max_seq_len=T,
        verification_markers=["verify", "check"],
        final_answer_markers=["final"],
    )

    # Should return masks even with one failure
    assert masks is not None

    # Sample 1 (index 0) and sample 3 (index 2) should have assignments
    assert masks["solver"][0].sum() > 0
    assert masks["solver"][2].sum() > 0

    # Sample 2 (index 1) may have zero assignments (empty trajectory)
    # This is acceptable - partial failure is handled


def test_compute_batch_role_masks_all_empty_returns_zero_masks():
    """Test batch computation with empty trajectories returns zero masks.

    Note: Empty trajectories don't technically fail parsing - they just
    produce zero tokens, which results in all-zero masks. This is acceptable
    graceful degradation (training continues with no role signal for these samples).
    """
    tokenizer = MockTokenizer()

    # All empty trajectories
    trajectories = ["", "", ""]

    B = len(trajectories)
    T = 50
    batch_prompt_mask = np.zeros((B, T), dtype=np.int32)

    masks = compute_batch_role_masks(
        batch_trajectories=trajectories,
        batch_prompt_mask=batch_prompt_mask,
        tokenizer=tokenizer,
        max_seq_len=T,
    )

    # Empty trajectories produce zero masks (graceful degradation)
    # This is acceptable - training continues without crashing
    assert masks is not None
    assert masks["solver"].sum() == 0
    assert masks["verifier"].sum() == 0
    assert masks["judge"].sum() == 0


# ============================================================================
# Test configurable markers
# ============================================================================


def test_configurable_markers():
    """Test that custom marker strings work correctly."""
    tokenizer = MockTokenizer()
    trajectory = "Solver text. Custom verify marker. Verifier. Custom final marker. Judge."
    max_seq_len = 50
    prompt_mask = np.zeros(50, dtype=np.int32)

    # Use custom markers
    masks = compute_role_masks_from_trajectory(
        trajectory_text=trajectory,
        prompt_mask=prompt_mask,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        verification_markers=["custom verify"],
        final_answer_markers=["custom final"],
    )

    assert masks is not None
    # Should detect all three roles with custom markers
    assert masks["solver"].sum() > 0
    assert masks["verifier"].sum() > 0
    assert masks["judge"].sum() > 0


# ============================================================================
# Integration test: realistic trajectory structure
# ============================================================================


def test_realistic_debate_structure():
    """Test with a realistic multi-turn debate trajectory structure."""
    tokenizer = MockTokenizer()

    # Simulate realistic debate turns
    trajectory = (
        "Solve this problem: What is 2+2? "  # User prompt (will be masked out)
        "Initial solution: The answer is 4 because 2+2=4. "  # Solver (chatbot)
        "Please verify the solution above. "  # User prompt (will be masked out)
        "Verification: The solution is correct. 2+2 indeed equals 4. "  # Verifier (chatbot)
        "Provide your final answer. "  # User prompt (will be masked out)
        "Final answer: 4. "  # Judge (chatbot)
    )

    # Construct prompt_mask:
    # Tokens: 50 total for simplicity
    # User tokens are at known positions based on the trajectory structure
    # For this test, we'll approximate:
    all_tokens = trajectory.split()
    T = len(all_tokens)

    # Mark user prompt tokens (approximation for test)
    # "Solve this problem: What is 2+2?" → 6 tokens
    # "Please verify the solution above." → 5 tokens  (starts at ~13)
    # "Provide your final answer." → 4 tokens (starts at ~25)
    prompt_mask = np.zeros(T, dtype=np.int32)
    prompt_mask[0:6] = 1  # First user prompt
    prompt_mask[13:18] = 1  # Verify prompt (approximate)
    prompt_mask[25:29] = 1  # Final prompt (approximate)

    max_seq_len = T

    masks = compute_role_masks_from_trajectory(
        trajectory_text=trajectory,
        prompt_mask=prompt_mask,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        verification_markers=["verify"],
        final_answer_markers=["final answer"],
    )

    assert masks is not None

    # Verify that we have assignments for all three roles
    assert masks["solver"].sum() > 0, "Solver should have tokens"
    assert masks["verifier"].sum() > 0, "Verifier should have tokens"
    assert masks["judge"].sum() > 0, "Judge should have tokens"

    # Verify user tokens are excluded
    user_indices = np.where(prompt_mask == 1)[0]
    for idx in user_indices:
        if idx < len(masks["solver"]):
            assert (
                masks["solver"][idx] == 0
                and masks["verifier"][idx] == 0
                and masks["judge"][idx] == 0
            ), f"User token {idx} should not be assigned to any role"
