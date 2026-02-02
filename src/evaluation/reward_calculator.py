"""Reward calculation for math and code domains."""

from typing import Any, Optional

from src.evaluation.code_executor import execute_humaneval, execute_with_tests
from src.evaluation.math_verifier import (
    compute_math_reward as _compute_math_reward,
)


class RewardCalculator:
    """Calculator for domain-specific rewards."""

    def compute_reward(
        self,
        solution: str,
        ground_truth: str,
        domain: str,
        metadata: Optional[dict] = None,
    ) -> tuple:
        """Compute reward based on domain.

        Args:
            solution: Solution text
            ground_truth: Ground truth answer/solution
            domain: Domain type ("math", "code", "general")
            metadata: Optional metadata (used for code domain)

        Returns:
            Tuple of (reward_score, details_dict)
        """
        if domain == "math":
            return self.compute_math_reward(solution, ground_truth)
        elif domain == "code":
            return self.compute_code_reward(solution, ground_truth, metadata or {})
        else:
            # For general domain, default to 0.0 (no ground truth verification)
            return 0.0, {"domain": "general", "message": "No verification available"}

    def compute_math_reward(
        self, solution: str, ground_truth: str
    ) -> tuple:
        """Compute reward for math domain.

        Args:
            solution: Solution text with \\boxed{answer}
            ground_truth: Ground truth solution with \\boxed{answer}

        Returns:
            Tuple of (reward, details_dict)
        """
        reward, result = _compute_math_reward(solution, ground_truth)

        details = {
            "domain": "math",
            "is_correct": result.is_correct,
            "predicted_answer": result.predicted_answer,
            "expected_answer": result.expected_answer,
            "method": result.method,
        }

        return reward, details

    def compute_code_reward(
        self, solution: str, ground_truth: str, metadata: dict
    ) -> tuple:
        """Compute reward for code domain.

        Args:
            solution: Generated code
            ground_truth: Ground truth (may be unused if metadata has tests)
            metadata: Metadata containing test cases or HumanEval format

        Returns:
            Tuple of (reward, details_dict)
        """
        # Check if HumanEval format (has test and entry_point)
        if "test" in metadata and "entry_point" in metadata:
            result = execute_humaneval(
                code=solution,
                test_code=metadata["test"],
                entry_point=metadata["entry_point"],
                timeout=metadata.get("timeout", 10),
            )

            reward = 1.0 if result.passed else 0.0

            details = {
                "domain": "code",
                "format": "humaneval",
                "passed": result.passed,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
                "timed_out": result.timed_out,
                "error": result.error,
            }

            return reward, details

        # Check if test_cases format
        elif "test_cases" in metadata:
            pass_rate, results = execute_with_tests(
                code=solution,
                test_cases=metadata["test_cases"],
                timeout_per_test=metadata.get("timeout_per_test", 5),
            )

            details = {
                "domain": "code",
                "format": "test_cases",
                "pass_rate": pass_rate,
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results if r.passed),
                "results": [
                    {
                        "passed": r.passed,
                        "stdout": r.stdout,
                        "stderr": r.stderr,
                        "timed_out": r.timed_out,
                    }
                    for r in results
                ],
            }

            return pass_rate, details

        else:
            # No test format found
            return 0.0, {
                "domain": "code",
                "error": "No test cases or HumanEval format found in metadata",
            }

    def compute_rlvr_reward(
        self,
        solution: str,
        ground_truth: str,
        domain: str,
        metadata: Optional[dict] = None,
    ) -> tuple[float, dict]:
        """Compute RLVR binary verifiable reward (1.0 if correct, 0.0 if incorrect).

        RLVR rewards are strictly binary - no partial credit.

        Args:
            solution: Solution text or code
            ground_truth: Ground truth answer/solution
            domain: Domain type ("math", "code", "general")
            metadata: Optional metadata (used for code domain test cases)

        Returns:
            Tuple of (reward_float, details_dict)
            - reward_float: 1.0 if correct, 0.0 otherwise
            - details_dict: Includes domain, correctness indicators, method
        """
        if metadata is None:
            metadata = {}

        if domain == "math":
            # Use existing math reward computation
            reward, result = _compute_math_reward(solution, ground_truth)

            details = {
                "domain": "math",
                "is_correct": result.is_correct,
                "predicted_answer": result.predicted_answer,
                "expected_answer": result.expected_answer,
                "method": result.method,
            }

            # RLVR: strictly binary (1.0 or 0.0)
            rlvr_reward = 1.0 if result.is_correct else 0.0
            return rlvr_reward, details

        elif domain == "code":
            # Check if HumanEval format
            if "test" in metadata and "entry_point" in metadata:
                result = execute_humaneval(
                    code=solution,
                    test_code=metadata["test"],
                    entry_point=metadata["entry_point"],
                    timeout=metadata.get("timeout", 10),
                )

                details = {
                    "domain": "code",
                    "format": "humaneval",
                    "passed": result.passed,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.return_code,
                    "timed_out": result.timed_out,
                    "error": result.error,
                    "method": "humaneval_execution",
                }

                # RLVR: 1.0 if all tests pass, 0.0 otherwise
                rlvr_reward = 1.0 if result.passed else 0.0
                return rlvr_reward, details

            # Check if test_cases format
            elif "test_cases" in metadata:
                pass_rate, results = execute_with_tests(
                    code=solution,
                    test_cases=metadata["test_cases"],
                    timeout_per_test=metadata.get("timeout_per_test", 5),
                )

                details = {
                    "domain": "code",
                    "format": "test_cases",
                    "pass_rate": pass_rate,
                    "total_tests": len(results),
                    "passed_tests": sum(1 for r in results if r.passed),
                    "method": "test_cases_execution",
                    "results": [
                        {
                            "passed": r.passed,
                            "stdout": r.stdout,
                            "stderr": r.stderr,
                            "timed_out": r.timed_out,
                        }
                        for r in results
                    ],
                }

                # RLVR: 1.0 if ALL tests pass (pass_rate == 1.0), 0.0 otherwise
                rlvr_reward = 1.0 if pass_rate >= 1.0 else 0.0
                return rlvr_reward, details

            else:
                # No test format found
                return 0.0, {
                    "domain": "code",
                    "error": "No test cases or HumanEval format found in metadata",
                    "method": "no_verification",
                }

        else:
            # General domain: no verifiable reward available
            return 0.0, {
                "domain": "general",
                "error": "No verifiable reward available",
                "method": "no_verification",
            }


# Convenience functions
def compute_math_reward(
    solution: str, ground_truth: str
) -> tuple:
    """Convenience function for math reward computation."""
    calculator = RewardCalculator()
    return calculator.compute_math_reward(solution, ground_truth)


def compute_code_reward(
    solution: str, ground_truth: str, metadata: dict
) -> tuple:
    """Convenience function for code reward computation."""
    calculator = RewardCalculator()
    return calculator.compute_code_reward(solution, ground_truth, metadata)


def compute_rlvr_reward(
    solution: str,
    ground_truth: str,
    domain: str,
    metadata: Optional[dict] = None,
) -> tuple[float, dict]:
    """Convenience function for RLVR binary reward computation."""
    calculator = RewardCalculator()
    return calculator.compute_rlvr_reward(solution, ground_truth, domain, metadata)
