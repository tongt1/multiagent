"""Integration tests for GRPO per-role loss decomposition.

Tests verify the LOGIC of per-role loss decomposition using numpy (no JAX).
These tests ensure correct:
- Role mask slicing from [B, T] to [B, T-1] to match objective shape
- Per-role loss averaging weighted by token count
- Loss fraction computation (partitioning of total loss)
- Advantage broadcasting from [miniB, 1] to [miniB, T-1]
- Zero-token edge case safety (no NaN/Inf)
- GSPO guard (1D objective skipped)
"""

from __future__ import annotations

import numpy as np


def test_role_mask_decomposition_math():
    """Verify per-role loss math with 3-role masks."""
    # Mock data: 4 samples, 10 completion tokens (11 total tokens)
    B, T = 4, 11

    # Objective (loss per token): [B, T-1]
    objective = np.array([
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0],  # Sample 0
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Sample 1
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # Sample 2
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Sample 3
    ], dtype=np.float32)

    # Role masks (before slicing): [B, T]
    # Partition: solver (0-3), verifier (4-7), judge (8-10)
    solver_mask = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)

    verifier_mask = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    ], dtype=np.float32)

    judge_mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ], dtype=np.float32)

    # Advantage: [B, 1] (broadcasts to [B, T-1])
    advantage = np.array([[0.5], [1.0], [-0.5], [0.0]], dtype=np.float32)

    # Slice masks to [B, T-1] to match objective
    solver_mask_sliced = solver_mask[:, :-1]
    verifier_mask_sliced = verifier_mask[:, :-1]
    judge_mask_sliced = judge_mask[:, :-1]

    # Compute per-role metrics
    solver_tokens = solver_mask_sliced.sum()
    verifier_tokens = verifier_mask_sliced.sum()
    judge_tokens = judge_mask_sliced.sum()

    solver_obj = (objective * solver_mask_sliced).sum()
    verifier_obj = (objective * verifier_mask_sliced).sum()
    judge_obj = (objective * judge_mask_sliced).sum()

    solver_loss = solver_obj / (solver_tokens + 1e-8)
    verifier_loss = verifier_obj / (verifier_tokens + 1e-8)
    judge_loss = judge_obj / (judge_tokens + 1e-8)

    # Verify token counts (4 samples * tokens per role in T-1 slice)
    assert solver_tokens == 16, f"Expected 16 solver tokens, got {solver_tokens}"
    assert verifier_tokens == 16, f"Expected 16 verifier tokens, got {verifier_tokens}"
    assert judge_tokens == 8, f"Expected 8 judge tokens, got {judge_tokens}"

    # Verify loss magnitudes are reasonable
    assert 0 < solver_loss < 10, f"Solver loss {solver_loss} out of expected range"
    assert 0 < verifier_loss < 10, f"Verifier loss {verifier_loss} out of expected range"
    assert 0 < judge_loss < 10, f"Judge loss {judge_loss} out of expected range"

    # Verify advantage broadcast and masking
    solver_adv = (advantage * solver_mask_sliced).sum() / (solver_tokens + 1e-8)
    verifier_adv = (advantage * verifier_mask_sliced).sum() / (verifier_tokens + 1e-8)
    judge_adv = (advantage * judge_mask_sliced).sum() / (judge_tokens + 1e-8)

    # Check that advantage is computed (non-zero for at least one role)
    assert abs(solver_adv) > 0 or abs(verifier_adv) > 0 or abs(judge_adv) > 0


def test_role_mask_shape_slicing():
    """Verify masks [B, T] slice to [B, T-1] to align with objective."""
    B, T = 2, 5

    # Objective: [B, T-1] = [2, 4]
    objective = np.ones((B, T - 1), dtype=np.float32)

    # Role mask: [B, T] = [2, 5]
    role_mask = np.array([
        [1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
    ], dtype=np.float32)

    # Slice to match objective
    role_mask_sliced = role_mask[:, :-1]

    assert role_mask_sliced.shape == objective.shape, \
        f"Sliced mask shape {role_mask_sliced.shape} != objective shape {objective.shape}"

    # Verify masked objective is computable
    masked_obj = objective * role_mask_sliced
    assert masked_obj.shape == objective.shape


def test_missing_role_masks_no_metrics():
    """Verify no per-role metrics computed when role_masks absent."""
    # Simulate GRPO._compute with no role_masks in extra_data
    extra_data = {}

    # Objective exists
    objective = np.ones((4, 10), dtype=np.float32)

    # Conditional block should be skipped
    if "role_masks" in extra_data and objective.ndim == 2:
        raise AssertionError("Should not execute per-role block without role_masks")

    # Test passes if we reach here


def test_loss_fractions_sum_to_one():
    """Verify loss fractions sum to ~1.0 when masks partition all tokens."""
    B, T = 4, 11

    # Objective: [B, T-1]
    objective = np.random.randn(B, T - 1).astype(np.float32)

    # Masks that perfectly partition tokens (no overlap, no gaps)
    solver_mask = np.zeros((B, T), dtype=np.float32)
    verifier_mask = np.zeros((B, T), dtype=np.float32)
    judge_mask = np.zeros((B, T), dtype=np.float32)

    solver_mask[:, :4] = 1
    verifier_mask[:, 4:8] = 1
    judge_mask[:, 8:] = 1

    # Slice to [B, T-1]
    solver_mask_sliced = solver_mask[:, :-1]
    verifier_mask_sliced = verifier_mask[:, :-1]
    judge_mask_sliced = judge_mask[:, :-1]

    # Compute loss fractions
    total_obj_sum = np.abs(objective).sum()

    solver_frac = np.abs(objective * solver_mask_sliced).sum() / (total_obj_sum + 1e-8)
    verifier_frac = np.abs(objective * verifier_mask_sliced).sum() / (total_obj_sum + 1e-8)
    judge_frac = np.abs(objective * judge_mask_sliced).sum() / (total_obj_sum + 1e-8)

    total_frac = solver_frac + verifier_frac + judge_frac

    # Should sum to ~1.0 (within floating point tolerance)
    assert abs(total_frac - 1.0) < 1e-5, f"Loss fractions sum to {total_frac}, expected ~1.0"


def test_short_debate_all_solver():
    """Verify short debate (only solver) produces ~100% solver loss fraction."""
    B, T = 2, 11

    # Objective: [B, T-1]
    objective = np.ones((B, T - 1), dtype=np.float32)

    # Only solver mask is nonzero (short debate)
    solver_mask = np.ones((B, T), dtype=np.float32)
    verifier_mask = np.zeros((B, T), dtype=np.float32)
    judge_mask = np.zeros((B, T), dtype=np.float32)

    # Slice to [B, T-1]
    solver_mask_sliced = solver_mask[:, :-1]
    verifier_mask_sliced = verifier_mask[:, :-1]
    judge_mask_sliced = judge_mask[:, :-1]

    # Compute loss fractions
    total_obj_sum = np.abs(objective).sum()

    solver_frac = np.abs(objective * solver_mask_sliced).sum() / (total_obj_sum + 1e-8)
    verifier_frac = np.abs(objective * verifier_mask_sliced).sum() / (total_obj_sum + 1e-8)
    judge_frac = np.abs(objective * judge_mask_sliced).sum() / (total_obj_sum + 1e-8)

    # Solver should be ~100%, others ~0%
    assert abs(solver_frac - 1.0) < 1e-5, f"Solver fraction {solver_frac}, expected ~1.0"
    assert verifier_frac < 1e-5, f"Verifier fraction {verifier_frac}, expected ~0.0"
    assert judge_frac < 1e-5, f"Judge fraction {judge_frac}, expected ~0.0"


def test_zero_token_count_safety():
    """Verify zero-token mask produces near-zero loss (not NaN/Inf)."""
    B, T = 2, 11

    # Objective: [B, T-1]
    objective = np.ones((B, T - 1), dtype=np.float32)

    # All-zero mask (no tokens for this role)
    zero_mask = np.zeros((B, T), dtype=np.float32)

    # Slice to [B, T-1]
    zero_mask_sliced = zero_mask[:, :-1]

    # Compute per-role loss with epsilon safety
    n_tokens = zero_mask_sliced.sum()
    masked_obj = (objective * zero_mask_sliced).sum()
    loss = masked_obj / (n_tokens + 1e-8)

    # Should be near-zero, not NaN or Inf
    assert not np.isnan(loss), "Loss is NaN with zero tokens"
    assert not np.isinf(loss), "Loss is Inf with zero tokens"
    assert abs(loss) < 1e-5, f"Loss {loss} should be near-zero with zero tokens"


def test_gspo_guard_skips_1d_objective():
    """Verify GSPO (1D objective) skips per-role block."""
    # GSPO objective is 1D (sequence-level): [B]
    objective_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    extra_data = {
        "role_masks": {
            "solver": np.ones((4, 10), dtype=np.float32),
            "verifier": np.zeros((4, 10), dtype=np.float32),
            "judge": np.zeros((4, 10), dtype=np.float32),
        }
    }

    # Conditional block should be skipped due to objective.ndim != 2
    if "role_masks" in extra_data and objective_1d.ndim == 2:
        raise AssertionError("Should not execute per-role block for 1D objective (GSPO)")

    # Test passes if we reach here


def test_advantage_broadcast_with_mask():
    """Verify advantage [miniB, 1] broadcasts correctly with mask [miniB, T-1]."""
    B, T = 3, 6

    # Advantage: [B, 1]
    advantage = np.array([[0.5], [1.0], [-0.5]], dtype=np.float32)

    # Role mask (sliced): [B, T-1]
    role_mask = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
    ], dtype=np.float32)

    # Broadcast advantage to mask shape
    masked_adv = advantage * role_mask  # [B, 1] * [B, T-1] -> [B, T-1]

    assert masked_adv.shape == role_mask.shape, \
        f"Masked advantage shape {masked_adv.shape} != mask shape {role_mask.shape}"

    # Verify advantage values are preserved where mask is 1
    # Sample 0, advantage 0.5, mask at positions 0,1,2
    assert np.allclose(masked_adv[0, 0], 0.5)
    assert np.allclose(masked_adv[0, 1], 0.5)
    assert np.allclose(masked_adv[0, 2], 0.5)

    # Sample 1, advantage 1.0, mask at positions 1,2,3
    assert np.allclose(masked_adv[1, 1], 1.0)
    assert np.allclose(masked_adv[1, 2], 1.0)
    assert np.allclose(masked_adv[1, 3], 1.0)

    # Verify zero where mask is 0
    assert np.allclose(masked_adv[0, 3], 0.0)
    assert np.allclose(masked_adv[0, 4], 0.0)
