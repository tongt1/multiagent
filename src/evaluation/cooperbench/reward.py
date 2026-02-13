"""Reward computation for CooperBench cooperative coding evaluation.

Provides binary and partial reward functions compatible with RLVR training.
The primary metric is binary: 1.0 if both features pass, 0.0 otherwise.
Partial credit awards 0.5 per passing feature.
"""

from loguru import logger

from src.evaluation.cooperbench.models import CooperBenchEvalResult


def compute_cooperbench_reward(eval_result: CooperBenchEvalResult) -> float:
    """Compute binary RLVR reward for CooperBench evaluation.

    Returns 1.0 if both features' tests passed, 0.0 otherwise.
    This is the primary CooperBench metric and is RLVR-compatible
    (strictly binary, no partial credit).

    Args:
        eval_result: Evaluation result from CooperBenchEvaluator.

    Returns:
        1.0 if both_passed is True, 0.0 otherwise.
    """
    reward = 1.0 if eval_result.both_passed else 0.0
    logger.debug(
        f"Binary reward for {eval_result.repo}/{eval_result.task_id}: "
        f"{reward} (both_passed={eval_result.both_passed})"
    )
    return reward


def compute_cooperbench_partial_reward(eval_result: CooperBenchEvalResult) -> float:
    """Compute partial reward for CooperBench evaluation.

    Awards 0.5 per feature that passes its test suite. This provides
    a denser reward signal during training but is not the official
    CooperBench metric.

    Reward breakdown:
        - 0 features pass: 0.0
        - 1 feature passes: 0.5
        - 2 features pass: 1.0

    Args:
        eval_result: Evaluation result from CooperBenchEvaluator.

    Returns:
        Float between 0.0 and 1.0 (0.5 increment per passing feature).
    """
    if not eval_result.feature_results:
        # No feature results available - fall back to binary
        return 1.0 if eval_result.both_passed else 0.0

    num_features = len(eval_result.feature_results)
    if num_features == 0:
        return 0.0

    passed_count = sum(1 for fr in eval_result.feature_results if fr.passed)
    reward = passed_count / num_features

    logger.debug(
        f"Partial reward for {eval_result.repo}/{eval_result.task_id}: "
        f"{reward:.2f} ({passed_count}/{num_features} features passed)"
    )
    return reward


def compute_cooperbench_shaped_reward(
    eval_result: CooperBenchEvalResult,
    merge_bonus: float = 0.1,
) -> float:
    """Compute shaped reward with merge bonus for training.

    Extends partial reward with a small bonus for clean merges,
    encouraging agents to produce non-conflicting patches.

    Reward components:
        - Base: 0.5 per passing feature (same as partial)
        - Merge bonus: +merge_bonus for clean merge status

    Total reward is clamped to [0.0, 1.0].

    Args:
        eval_result: Evaluation result from CooperBenchEvaluator.
        merge_bonus: Bonus for clean merge (default 0.1).

    Returns:
        Shaped reward float between 0.0 and 1.0.
    """
    # Start with partial reward
    base_reward = compute_cooperbench_partial_reward(eval_result)

    # Add merge bonus for clean merges
    bonus = merge_bonus if eval_result.merge_status == "clean" else 0.0

    reward = min(1.0, base_reward + bonus)

    logger.debug(
        f"Shaped reward for {eval_result.repo}/{eval_result.task_id}: "
        f"{reward:.2f} (base={base_reward:.2f}, merge_bonus={bonus:.2f})"
    )
    return reward
