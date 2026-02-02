"""Iteration control for solver-verifier loop."""

from loguru import logger


class IterationController:
    """Controls iteration flow in the solver-verifier loop."""

    def __init__(self, max_iterations: int) -> None:
        """Initialize iteration controller.

        Args:
            max_iterations: Maximum number of solver-verifier iterations
        """
        self.max_iterations = max_iterations
        self.iteration_history: list[dict[str, str]] = []

    def should_continue(self, iteration: int, verification_passed: bool) -> bool:
        """Determine if iteration should continue.

        Args:
            iteration: Current iteration number (1-indexed)
            verification_passed: Whether verification passed

        Returns:
            True if iteration should continue, False otherwise
        """
        # Stop if verification passed
        if verification_passed:
            logger.info(f"Iteration {iteration}: Verification passed, stopping")
            return False

        # Stop if reached max iterations
        if iteration >= self.max_iterations:
            logger.warning(
                f"Iteration {iteration}: Reached max_iterations={self.max_iterations}, stopping"
            )
            return False

        # Continue iteration
        logger.info(f"Iteration {iteration}: Verification failed, continuing")
        return True

    def detect_circular_critique(self, critiques: list[str]) -> bool:
        """Detect if critiques are stuck in a circular pattern.

        Uses a simple heuristic: checks if the last 2-3 critiques show
        repetition (substring match). Requires at least 3 critiques to
        avoid false positives.

        Args:
            critiques: List of critique strings from all iterations

        Returns:
            True if circular pattern detected, False otherwise
        """
        # Need at least 3 critiques to detect a pattern
        if len(critiques) < 3:
            return False

        # Get last two critiques
        last_critique = critiques[-1].strip().lower()
        prev_critique = critiques[-2].strip().lower()

        # Check if one is a substring of the other or they're very similar
        if not last_critique or not prev_critique:
            return False

        shorter = min(last_critique, prev_critique, key=len)
        longer = max(last_critique, prev_critique, key=len)

        # Check substring containment
        if shorter in longer:
            logger.warning(
                "Circular critique detected: last critique very similar to previous"
            )
            return True

        return False

    def record_iteration(self, iteration: int, critique: str) -> None:
        """Record iteration details for observability.

        Args:
            iteration: Iteration number
            critique: Critique from verifier
        """
        self.iteration_history.append(
            {
                "iteration": str(iteration),
                "critique": critique,
            }
        )
