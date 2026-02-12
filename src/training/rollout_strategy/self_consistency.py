"""Self-consistency rollout strategy using majority vote.

Implements the self-consistency approach (Wang et al., 2022) where multiple
rollouts per prompt are evaluated via majority vote to determine correctness,
and rewards are scaled by agreement fraction. This provides a more robust
training signal than any single rollout by leveraging the idea that multiple
correct reasoning paths tend to converge on the same answer.

Auto-registers as "self_consistency" on import.
"""

from __future__ import annotations

import logging

import numpy as np

from src.training.rollout_strategy.base import RolloutStrategy
from src.training.rollout_strategy.registry import register_strategy

logger = logging.getLogger(__name__)


class SelfConsistencyStrategy(RolloutStrategy):
    """Self-consistency rollout strategy using majority vote.

    For each prompt group of N rollouts:
    1. Classify each rollout as correct (reward > 0) or incorrect (reward == 0)
    2. Determine majority vote (correct or incorrect), tie-break favors correct
    3. Compute agreement_fraction = count_of_majority / N
    4. If agreement_fraction < agreement_threshold, pass through unchanged
    5. Rollouts agreeing with majority: reward *= agreement_fraction
    6. Rollouts disagreeing with majority: reward = 0.0

    Unlike best-of-N, self-consistency returns ALL items with modified rewards,
    preserving the batch size for GRPO training.

    Args:
        agreement_threshold: Minimum agreement fraction to apply strategy.
            Below this, original rewards pass through unchanged. Default 0.5.
        correct_reward_threshold: Threshold for classifying rollout as correct.
            Rollouts with reward > this value are "correct". Default 0.0.

    Example:
        >>> strategy = SelfConsistencyStrategy(agreement_threshold=0.5)
        >>> # 4 rollouts: 3 correct (5.0), 1 incorrect (0.0)
        >>> # Majority = correct, agreement = 0.75
        >>> # Correct rollouts: 5.0 * 0.75 = 3.75, incorrect: 0.0
        >>> result = strategy.select_rollouts(items, n_rollouts_per_prompt=4)
        >>> len(result)  # Still 4 (all items returned)
        4
    """

    def __init__(
        self,
        agreement_threshold: float = 0.5,
        correct_reward_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._agreement_threshold = agreement_threshold
        self._correct_reward_threshold = correct_reward_threshold

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "self_consistency"

    def select_rollouts(
        self,
        items: list,
        n_rollouts_per_prompt: int,
    ) -> list:
        """Apply majority vote and agreement fraction scaling to all rollouts.

        Args:
            items: Flat list of actor output items. Each item has
                item.data["rewards"] as a numpy scalar with the rollout reward.
            n_rollouts_per_prompt: Number of rollouts per prompt (GRPO group size).

        Returns:
            List of ALL items (same length as input) with rewards modified
            to reflect self-consistency signal.
        """
        n_total = len(items)
        n_prompts = n_total // n_rollouts_per_prompt
        remainder = n_total % n_rollouts_per_prompt

        if remainder > 0:
            logger.warning(
                "Batch size %d not evenly divisible by n_rollouts_per_prompt %d. "
                "%d leftover items will pass through unchanged.",
                n_total,
                n_rollouts_per_prompt,
                remainder,
            )

        result = []

        for i in range(n_prompts):
            start = i * n_rollouts_per_prompt
            end = start + n_rollouts_per_prompt
            group = items[start:end]

            # Extract rewards and classify each rollout
            rewards = [item.data["rewards"].item() for item in group]
            is_correct = [r > self._correct_reward_threshold for r in rewards]
            n_correct = sum(is_correct)
            n_incorrect = n_rollouts_per_prompt - n_correct

            # Determine majority vote (tie-break favors correct)
            majority_is_correct = n_correct >= n_incorrect

            # Compute agreement fraction
            if majority_is_correct:
                agreement_fraction = n_correct / n_rollouts_per_prompt
            else:
                agreement_fraction = n_incorrect / n_rollouts_per_prompt

            # If below threshold, pass through unchanged
            if agreement_fraction < self._agreement_threshold:
                result.extend(group)
                continue

            # Apply majority vote scaling
            for item, correct, reward in zip(group, is_correct, rewards):
                agrees_with_majority = correct == majority_is_correct

                if agrees_with_majority:
                    new_reward = reward * agreement_fraction
                else:
                    new_reward = 0.0

                # Create new data dict to avoid mutating original
                item.data = {**item.data, "rewards": np.array(new_reward)}
                result.append(item)

        # Append any leftover items unchanged
        if remainder > 0:
            leftover_start = n_prompts * n_rollouts_per_prompt
            result.extend(items[leftover_start:])

        return result


# Auto-register on import
register_strategy("self_consistency", SelfConsistencyStrategy)
