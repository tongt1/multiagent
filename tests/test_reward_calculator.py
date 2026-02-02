"""Tests for RLVR reward computation."""

import pytest

from src.evaluation.reward_calculator import (
    RewardCalculator,
    compute_rlvr_reward,
)


class TestRLVRRewardComputation:
    """Tests for RLVR binary reward computation."""

    def test_compute_rlvr_reward_math_correct(self):
        """Test RLVR reward for correct math answer."""
        calculator = RewardCalculator()
        solution = r"The answer is \boxed{42}"
        ground_truth = r"Solution: \boxed{42}"

        reward, details = calculator.compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="math",
        )

        # RLVR reward should be binary: 1.0 for correct
        assert reward == 1.0
        assert details["domain"] == "math"
        assert details["is_correct"] is True
        assert details["predicted_answer"] == "42"
        assert details["expected_answer"] == "42"
        assert "method" in details

    def test_compute_rlvr_reward_math_incorrect(self):
        """Test RLVR reward for incorrect math answer."""
        calculator = RewardCalculator()
        solution = r"The answer is \boxed{42}"
        ground_truth = r"Solution: \boxed{43}"

        reward, details = calculator.compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="math",
        )

        # RLVR reward should be binary: 0.0 for incorrect
        assert reward == 0.0
        assert details["domain"] == "math"
        assert details["is_correct"] is False
        assert details["predicted_answer"] == "42"
        assert details["expected_answer"] == "43"

    def test_compute_rlvr_reward_math_symbolic_equivalence(self):
        """Test RLVR reward with symbolic equivalence."""
        calculator = RewardCalculator()
        solution = r"The answer is \boxed{2/4}"
        ground_truth = r"Solution: \boxed{1/2}"

        reward, details = calculator.compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="math",
        )

        # Should be correct via symbolic equivalence
        assert reward == 1.0
        assert details["is_correct"] is True

    def test_compute_rlvr_reward_math_extraction_failed(self):
        """Test RLVR reward when answer extraction fails."""
        calculator = RewardCalculator()
        solution = "The answer is 42"  # No \boxed{}
        ground_truth = r"Solution: \boxed{42}"

        reward, details = calculator.compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="math",
        )

        # Extraction failure should give 0.0
        assert reward == 0.0
        assert details["is_correct"] is False
        assert details["method"] == "extraction_failed"

    def test_compute_rlvr_reward_code_humaneval_pass(self):
        """Test RLVR reward for code that passes HumanEval tests."""
        calculator = RewardCalculator()
        solution = """
def add(a, b):
    return a + b
"""
        ground_truth = ""  # Not used for code
        metadata = {
            "test": """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0
""",
            "entry_point": "add",
        }

        reward, details = calculator.compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="code",
            metadata=metadata,
        )

        # All tests pass -> 1.0
        assert reward == 1.0
        assert details["domain"] == "code"
        assert details["format"] == "humaneval"
        assert details["passed"] is True
        assert details["method"] == "humaneval_execution"

    def test_compute_rlvr_reward_code_humaneval_fail(self):
        """Test RLVR reward for code that fails HumanEval tests."""
        calculator = RewardCalculator()
        solution = """
def add(a, b):
    return a - b  # Wrong implementation
"""
        ground_truth = ""
        metadata = {
            "test": """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
""",
            "entry_point": "add",
        }

        reward, details = calculator.compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="code",
            metadata=metadata,
        )

        # Test fails -> 0.0
        assert reward == 0.0
        assert details["domain"] == "code"
        assert details["passed"] is False

    def test_compute_rlvr_reward_code_test_cases_all_pass(self):
        """Test RLVR reward for code with all test cases passing."""
        calculator = RewardCalculator()
        solution = """
x = int(input())
print(x * 2)
"""
        ground_truth = ""
        metadata = {
            "test_cases": [
                {"input": "5", "expected_output": "10"},
                {"input": "0", "expected_output": "0"},
                {"input": "-3", "expected_output": "-6"},
            ],
        }

        reward, details = calculator.compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="code",
            metadata=metadata,
        )

        # All tests pass -> 1.0
        assert reward == 1.0
        assert details["domain"] == "code"
        assert details["format"] == "test_cases"
        assert details["pass_rate"] == 1.0
        assert details["passed_tests"] == 3
        assert details["total_tests"] == 3
        assert details["method"] == "test_cases_execution"

    def test_compute_rlvr_reward_code_test_cases_partial_pass(self):
        """Test RLVR reward for code with partial test case pass."""
        calculator = RewardCalculator()
        solution = """
x = int(input())
if x > 0:
    print(x * 2)
else:
    print(0)  # Wrong for negative numbers
"""
        ground_truth = ""
        metadata = {
            "test_cases": [
                {"input": "5", "expected_output": "10"},
                {"input": "0", "expected_output": "0"},
                {"input": "-3", "expected_output": "-6"},  # Will fail
            ],
        }

        reward, details = calculator.compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="code",
            metadata=metadata,
        )

        # Partial pass (2/3) -> 0.0 (RLVR is binary, not partial credit)
        assert reward == 0.0
        assert details["domain"] == "code"
        assert details["pass_rate"] < 1.0
        assert details["passed_tests"] == 2
        assert details["total_tests"] == 3

    def test_compute_rlvr_reward_code_no_tests(self):
        """Test RLVR reward for code with no test format."""
        calculator = RewardCalculator()
        solution = "print('hello')"
        ground_truth = ""
        metadata = {}  # No test cases

        reward, details = calculator.compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="code",
            metadata=metadata,
        )

        # No verification available -> 0.0
        assert reward == 0.0
        assert details["domain"] == "code"
        assert "error" in details
        assert details["method"] == "no_verification"

    def test_compute_rlvr_reward_general_domain(self):
        """Test RLVR reward for general domain (no verification)."""
        calculator = RewardCalculator()
        solution = "This is a general response"
        ground_truth = "This is the expected response"

        reward, details = calculator.compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="general",
        )

        # General domain has no verifiable reward -> 0.0
        assert reward == 0.0
        assert details["domain"] == "general"
        assert "error" in details
        assert details["method"] == "no_verification"

    def test_compute_rlvr_reward_convenience_function(self):
        """Test convenience function for RLVR reward."""
        solution = r"The answer is \boxed{100}"
        ground_truth = r"Solution: \boxed{100}"

        reward, details = compute_rlvr_reward(
            solution=solution,
            ground_truth=ground_truth,
            domain="math",
        )

        assert reward == 1.0
        assert details["is_correct"] is True

    def test_compute_rlvr_reward_strictly_binary(self):
        """Test that RLVR rewards are strictly binary (no intermediate values)."""
        calculator = RewardCalculator()

        # Test various scenarios - all should return exactly 1.0 or 0.0
        test_cases = [
            # Math correct
            (r"\boxed{42}", r"\boxed{42}", "math", {}, 1.0),
            # Math incorrect
            (r"\boxed{42}", r"\boxed{43}", "math", {}, 0.0),
            # Code all pass
            (
                "x = int(input()); print(x * 2)",
                "",
                "code",
                {"test_cases": [{"input": "5", "expected_output": "10"}]},
                1.0,
            ),
            # Code partial pass (2/3) - should be 0.0, not 0.67
            (
                "x = int(input()); print(x if x > 0 else 0)",
                "",
                "code",
                {
                    "test_cases": [
                        {"input": "5", "expected_output": "5"},
                        {"input": "0", "expected_output": "0"},
                        {"input": "-3", "expected_output": "-3"},  # Fails
                    ]
                },
                0.0,
            ),
        ]

        for solution, ground_truth, domain, metadata, expected_reward in test_cases:
            reward, _ = calculator.compute_rlvr_reward(
                solution=solution,
                ground_truth=ground_truth,
                domain=domain,
                metadata=metadata,
            )
            # Check strictly binary
            assert reward in [0.0, 1.0], f"RLVR reward must be binary, got {reward}"
            assert reward == expected_reward
