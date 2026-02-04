"""Tests for W&B rollout table with per-prompt sampling."""

import pytest

from src.training.wandb_enrichment.rollout_table import (
    RolloutRecord,
    sample_rollouts_per_prompt,
)


def test_sample_rollouts_basic():
    """Test basic per-prompt sampling with 2 prompts, 4 rollouts each."""
    rollouts = [
        # Prompt A: rewards 1.0, 0.8, 0.5, 0.2
        RolloutRecord("prompt_a", "Problem A", "completion_a1", 1.0),
        RolloutRecord("prompt_a", "Problem A", "completion_a2", 0.8),
        RolloutRecord("prompt_a", "Problem A", "completion_a3", 0.5),
        RolloutRecord("prompt_a", "Problem A", "completion_a4", 0.2),
        # Prompt B: rewards 0.9, 0.7, 0.4, 0.1
        RolloutRecord("prompt_b", "Problem B", "completion_b1", 0.9),
        RolloutRecord("prompt_b", "Problem B", "completion_b2", 0.7),
        RolloutRecord("prompt_b", "Problem B", "completion_b3", 0.4),
        RolloutRecord("prompt_b", "Problem B", "completion_b4", 0.1),
    ]

    sampled = sample_rollouts_per_prompt(rollouts, n_prompts=2, top_k=1, bottom_k=1)

    # Should have 2 prompts × (1 top + 1 bottom) = 4 rollouts
    assert len(sampled) == 4

    # Check that we have top and bottom for each prompt
    prompt_a_sampled = [r for r, _ in sampled if r.prompt_id == "prompt_a"]
    prompt_b_sampled = [r for r, _ in sampled if r.prompt_id == "prompt_b"]

    assert len(prompt_a_sampled) == 2
    assert len(prompt_b_sampled) == 2

    # Top rollouts should have is_top=True
    top_rollouts = [r for r, is_top in sampled if is_top]
    assert len(top_rollouts) == 2

    # Bottom rollouts should have is_top=False
    bottom_rollouts = [r for r, is_top in sampled if not is_top]
    assert len(bottom_rollouts) == 2


def test_sample_rollouts_per_prompt_not_global():
    """Verify that sampling is per-prompt, not global top-K."""
    rollouts = [
        # Prompt A: all low rewards
        RolloutRecord("prompt_a", "Problem A", "completion_a1", 0.3),
        RolloutRecord("prompt_a", "Problem A", "completion_a2", 0.2),
        RolloutRecord("prompt_a", "Problem A", "completion_a3", 0.1),
        # Prompt B: all high rewards
        RolloutRecord("prompt_b", "Problem B", "completion_b1", 1.0),
        RolloutRecord("prompt_b", "Problem B", "completion_b2", 0.9),
        RolloutRecord("prompt_b", "Problem B", "completion_b3", 0.8),
        # Prompt C: mixed rewards
        RolloutRecord("prompt_c", "Problem C", "completion_c1", 0.6),
        RolloutRecord("prompt_c", "Problem C", "completion_c2", 0.5),
        RolloutRecord("prompt_c", "Problem C", "completion_c3", 0.4),
    ]

    sampled = sample_rollouts_per_prompt(rollouts, n_prompts=3, top_k=1, bottom_k=1)

    # All 3 prompts should be represented
    prompt_ids = {r.prompt_id for r, _ in sampled}
    assert len(prompt_ids) == 3

    # Prompt A's top rollout (0.3) should be included, even though it's globally low
    prompt_a_top = [r for r, is_top in sampled if r.prompt_id == "prompt_a" and is_top]
    assert len(prompt_a_top) == 1
    assert prompt_a_top[0].reward == 0.3

    # Prompt B's bottom rollout (0.8) should be included, even though it's globally high
    prompt_b_bottom = [r for r, is_top in sampled if r.prompt_id == "prompt_b" and not is_top]
    assert len(prompt_b_bottom) == 1
    assert prompt_b_bottom[0].reward == 0.8


def test_sample_rollouts_fewer_prompts_than_requested():
    """Test when batch has fewer prompts than n_prompts."""
    rollouts = [
        RolloutRecord("prompt_a", "Problem A", "completion_a1", 1.0),
        RolloutRecord("prompt_a", "Problem A", "completion_a2", 0.5),
        RolloutRecord("prompt_b", "Problem B", "completion_b1", 0.8),
        RolloutRecord("prompt_b", "Problem B", "completion_b2", 0.3),
    ]

    # Request 4 prompts but only 2 exist
    sampled = sample_rollouts_per_prompt(rollouts, n_prompts=4, top_k=1, bottom_k=1)

    # Should use all available prompts
    prompt_ids = {r.prompt_id for r, _ in sampled}
    assert len(prompt_ids) == 2
    assert len(sampled) == 4


def test_sample_rollouts_fewer_rollouts_than_k():
    """Test when a prompt has fewer rollouts than top_k + bottom_k."""
    rollouts = [
        # Only 2 rollouts for this prompt
        RolloutRecord("prompt_a", "Problem A", "completion_a1", 1.0),
        RolloutRecord("prompt_a", "Problem A", "completion_a2", 0.5),
    ]

    # Request top_k=2, bottom_k=2 (total 4) but only 2 exist
    sampled = sample_rollouts_per_prompt(rollouts, n_prompts=1, top_k=2, bottom_k=2)

    # Should take all available without duplicates
    assert len(sampled) == 2

    # Both should be present
    rewards = {r.reward for r, _ in sampled}
    assert rewards == {1.0, 0.5}


def test_sample_rollouts_is_top_flag():
    """Verify that is_top flag correctly identifies top vs bottom rollouts."""
    rollouts = [
        RolloutRecord("prompt_a", "Problem A", "completion_a1", 1.0),
        RolloutRecord("prompt_a", "Problem A", "completion_a2", 0.8),
        RolloutRecord("prompt_a", "Problem A", "completion_a3", 0.3),
        RolloutRecord("prompt_a", "Problem A", "completion_a4", 0.1),
    ]

    sampled = sample_rollouts_per_prompt(rollouts, n_prompts=1, top_k=2, bottom_k=2)

    # Top 2 should have is_top=True
    top_rewards = {r.reward for r, is_top in sampled if is_top}
    assert top_rewards == {1.0, 0.8}

    # Bottom 2 should have is_top=False
    bottom_rewards = {r.reward for r, is_top in sampled if not is_top}
    assert bottom_rewards == {0.3, 0.1}


def test_sample_rollouts_empty():
    """Test with empty rollout list."""
    sampled = sample_rollouts_per_prompt([], n_prompts=4, top_k=2, bottom_k=2)
    assert sampled == []


def test_rollout_record_dataclass():
    """Test RolloutRecord dataclass creation."""
    # With all fields
    record = RolloutRecord(
        prompt_id="test_prompt",
        prompt_text="What is 2+2?",
        completion="The answer is 4.",
        reward=1.0,
        solver_reward=0.9,
        verifier_reward=0.95,
        judge_reward=1.0,
        role_assignments="S:0,V:1,J:2",
    )

    assert record.prompt_id == "test_prompt"
    assert record.prompt_text == "What is 2+2?"
    assert record.completion == "The answer is 4."
    assert record.reward == 1.0
    assert record.solver_reward == 0.9
    assert record.verifier_reward == 0.95
    assert record.judge_reward == 1.0
    assert record.role_assignments == "S:0,V:1,J:2"

    # With minimal fields
    minimal_record = RolloutRecord(
        prompt_id="test_prompt",
        prompt_text="What is 2+2?",
        completion="The answer is 4.",
        reward=1.0,
    )

    assert minimal_record.solver_reward is None
    assert minimal_record.verifier_reward is None
    assert minimal_record.judge_reward is None
    assert minimal_record.role_assignments is None
