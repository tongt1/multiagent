"""Advantage alignment for multi-agent cooperative GRPO training.

Implements the advantage alignment algorithm from Duque Van et al.
(arXiv:2406.14662, "Multi-Agent Reinforcement Learning from Human Feedback:
Data Coverage and Algorithmic Techniques", ICLR 2025).

The key idea: each agent's advantage is augmented with a cross-agent signal
that captures how well the agent's historical performance correlates with
the opponent's current advantage. This creates a cooperative learning signal
where agents that historically contributed to good outcomes for their partner
receive a bonus.

Formula (Eq. 10 adapted for two-agent cooperative setting):

    A*_solver[i] = A_solver[i] + beta * (cumsum_solver[i] * A_verifier[i]) / max(i, 1)
    A*_verifier[i] = A_verifier[i] + beta * (cumsum_verifier[i] * A_solver[i]) / max(i, 1)

Where cumsum_solver[i] = sum_{k=0}^{i-1} gamma^{i-1-k} * A_solver[k]
is the discounted cumulative sum of past solver advantages.

Design decisions:
- Two agents only: solver + verifier (judge tokens are part of verifier's advantage)
- Cooperative only (positive beta). Adversarial mode is out of scope.
- Both models get aligned advantages (even frozen model's advantages are computed
  as opponent signal)
- Pure numpy implementation, no Flink/JAX dependencies
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class AdvantageAlignmentConfig:
    """Configuration for advantage alignment between solver and verifier.

    Attributes:
        enabled: Whether advantage alignment is active. Auto-enabled for dual learner.
        beta: Opponent advantage scaling weight (aa_weight in the paper). Paper default = 1.0.
        gamma: Discount factor for historical advantages. Paper default = 0.99.
    """

    enabled: bool = True
    beta: float = 1.0
    gamma: float = 0.99


def compute_discounted_cumsum(
    advantages: NDArray[np.floating],
    gamma: float,
) -> NDArray[np.floating]:
    """Compute discounted cumulative sum of past advantages.

    For each element i, computes the exponentially-discounted sum of all
    *preceding* elements (not including element i itself):

        result[i] = sum_{k=0}^{i-1} gamma^{i-1-k} * advantages[k]

    Note: result[0] = 0 (no history for the first element).

    Args:
        advantages: 1-D array of shape [B] with per-prompt advantages.
        gamma: Discount factor in [0, 1]. gamma=1.0 means no discounting;
            gamma=0.0 means no history contribution.

    Returns:
        Discounted cumulative sum array of shape [B].
    """
    if advantages.size == 0:
        return np.array([], dtype=advantages.dtype)

    B = len(advantages)
    result = np.zeros(B, dtype=np.float64)

    # result[i] = sum_{k=0}^{i-1} gamma^{i-1-k} * advantages[k]
    # Recurrence: result[i] = gamma * result[i-1] + advantages[i-1]
    for i in range(1, B):
        result[i] = gamma * result[i - 1] + advantages[i - 1]

    return result.astype(advantages.dtype)


def compute_aligned_advantages(
    solver_advantages: NDArray[np.floating],
    verifier_advantages: NDArray[np.floating],
    config: AdvantageAlignmentConfig,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute advantage-aligned advantages for solver and verifier.

    Implements Eq. 10 from Duque Van et al. for a two-agent cooperative setting.
    Each agent's advantage is augmented with a cross-agent term that captures
    the correlation between the agent's historical performance and the
    opponent's current advantage.

    When config.enabled is False, returns the original advantages unchanged.
    When config.beta is 0, the alignment term vanishes and original advantages
    are returned (numerically exact).

    Args:
        solver_advantages: 1-D array of shape [B] with per-prompt solver advantages.
        verifier_advantages: 1-D array of shape [B] with per-prompt verifier advantages.
        config: Advantage alignment configuration (beta, gamma, enabled).

    Returns:
        Tuple of (aligned_solver_advantages, aligned_verifier_advantages),
        each of shape [B].
    """
    if not config.enabled:
        return solver_advantages.copy(), verifier_advantages.copy()

    if config.beta == 0.0:
        return solver_advantages.copy(), verifier_advantages.copy()

    B = len(solver_advantages)

    # Compute discounted cumulative sums of past advantages
    cumsum_solver = compute_discounted_cumsum(solver_advantages, config.gamma)
    cumsum_verifier = compute_discounted_cumsum(verifier_advantages, config.gamma)

    # Normalization: divide by max(i, 1) for each batch index
    normalizer = np.maximum(np.arange(B, dtype=np.float64), 1.0)

    # Aligned advantages (Eq. 10):
    # A*_solver[i] = A_solver[i] + beta * (cumsum_solver[i] * A_verifier[i]) / max(i, 1)
    aligned_solver = (
        solver_advantages
        + config.beta * (cumsum_solver * verifier_advantages) / normalizer
    )

    # A*_verifier[i] = A_verifier[i] + beta * (cumsum_verifier[i] * A_solver[i]) / max(i, 1)
    aligned_verifier = (
        verifier_advantages
        + config.beta * (cumsum_verifier * solver_advantages) / normalizer
    )

    return aligned_solver.astype(solver_advantages.dtype), aligned_verifier.astype(
        verifier_advantages.dtype
    )
