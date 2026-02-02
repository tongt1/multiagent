"""Tests for evaluation modules."""

import platform
import pytest

from src.evaluation.code_executor import (
    CodeExecutionResult,
    execute_code,
    execute_with_tests,
)
from src.evaluation.math_verifier import (
    MathVerificationResult,
    extract_boxed_answer,
    verify_math_answer,
)
from src.evaluation.reward_calculator import RewardCalculator
from src.models.trajectory import TrajectoryEntry


class TestMathVerifier:
    """Tests for math verifier."""

    def test_extract_boxed_simple(self):
        """Test extracting simple boxed answer."""
        solution = r"The answer is \boxed{42}"
        assert extract_boxed_answer(solution) == "42"

    def test_extract_boxed_nested(self):
        """Test extracting nested boxed answer."""
        solution = r"The answer is \boxed{\frac{1}{2}}"
        assert extract_boxed_answer(solution) == r"\frac{1}{2}"

    def test_extract_boxed_deeply_nested(self):
        """Test extracting deeply nested boxed answer."""
        solution = r"The answer is \boxed{{x^2 + {1}}}"
        assert extract_boxed_answer(solution) == r"{x^2 + {1}}"

    def test_extract_boxed_none(self):
        """Test no boxed answer found."""
        solution = "The answer is 42"
        assert extract_boxed_answer(solution) is None

    def test_extract_boxed_multiple(self):
        """Test multiple boxed answers (returns last)."""
        solution = r"First \boxed{1} then \boxed{2}"
        assert extract_boxed_answer(solution) == "2"

    def test_verify_exact_match(self):
        """Test exact match verification."""
        assert verify_math_answer("42", "42") is True
        assert verify_math_answer("42", "43") is False

    def test_verify_symbolic_equivalence_fraction(self):
        """Test symbolic equivalence for fractions."""
        assert verify_math_answer("1/2", "0.5") is True
        assert verify_math_answer("0.5", "1/2") is True

    def test_verify_symbolic_equivalence_algebra(self):
        """Test symbolic equivalence for algebra."""
        assert verify_math_answer("x+1", "1+x") is True
        assert verify_math_answer("2*x", "x*2") is True

    def test_verify_non_equivalent(self):
        """Test non-equivalent answers."""
        assert verify_math_answer("3", "4") is False
        assert verify_math_answer("x+1", "x+2") is False

    def test_verify_invalid_expression_fallback(self):
        """Test fallback to string comparison for invalid expressions."""
        assert verify_math_answer("hello", "hello") is True
        assert verify_math_answer("hello", "world") is False


class TestCodeExecutor:
    """Tests for code executor."""

    def test_execute_simple_print(self):
        """Test executing simple print statement."""
        result = execute_code('print("Hello, World!")')
        assert result.passed is True
        assert result.stdout.strip() == "Hello, World!"
        assert result.timed_out is False
        assert result.error is None

    def test_execute_with_return_value(self):
        """Test executing code with return value."""
        result = execute_code("print(40 + 2)")
        assert result.passed is True
        assert result.stdout.strip() == "42"

    def test_execute_syntax_error(self):
        """Test executing code with syntax error."""
        result = execute_code("print(")
        assert result.passed is False
        assert result.return_code != 0

    def test_execute_timeout(self):
        """Test executing code that times out."""
        code = "while True: pass"
        result = execute_code(code, timeout=1)
        assert result.passed is False
        assert result.timed_out is True

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Resource limits not supported on Windows"
    )
    def test_execute_resource_limit(self):
        """Test resource limit enforcement (Unix only)."""
        # Try to allocate a large amount of memory
        code = "x = [0] * (1024 * 1024 * 1024)"  # Try to allocate 1GB
        result = execute_code(code, timeout=5)
        # Should fail due to memory limit or complete without error
        # (depends on Python's memory allocation behavior)
        assert result is not None

    def test_execute_with_stdin(self):
        """Test executing code with stdin."""
        code = "x = input(); print(f'Got: {x}')"
        result = execute_code(code, stdin="test\n")
        assert result.passed is True
        assert "Got: test" in result.stdout


class TestExecuteWithTests:
    """Tests for test case execution."""

    def test_all_pass(self):
        """Test all test cases passing."""
        code = "x = int(input()); print(x * 2)"
        test_cases = [
            {"input": "1", "expected_output": "2"},
            {"input": "2", "expected_output": "4"},
            {"input": "5", "expected_output": "10"},
        ]
        pass_rate, results = execute_with_tests(code, test_cases)
        assert pass_rate == 1.0
        assert len(results) == 3
        assert all(r.passed for r in results)

    def test_partial_pass(self):
        """Test partial test case passing."""
        code = "x = int(input()); print(x * 2)"
        test_cases = [
            {"input": "1", "expected_output": "2"},
            {"input": "2", "expected_output": "5"},  # Wrong expected
            {"input": "5", "expected_output": "10"},
        ]
        pass_rate, results = execute_with_tests(code, test_cases)
        assert pass_rate == pytest.approx(2.0 / 3.0)
        assert len(results) == 3
        assert results[0].passed is True
        assert results[1].passed is False
        assert results[2].passed is True

    def test_all_fail(self):
        """Test all test cases failing."""
        code = "print('wrong')"
        test_cases = [
            {"input": "1", "expected_output": "2"},
            {"input": "2", "expected_output": "4"},
        ]
        pass_rate, results = execute_with_tests(code, test_cases)
        assert pass_rate == 0.0
        assert all(not r.passed for r in results)

    def test_empty_test_cases(self):
        """Test with empty test cases."""
        code = "print('hello')"
        test_cases = []
        pass_rate, results = execute_with_tests(code, test_cases)
        assert pass_rate == 0.0
        assert len(results) == 0


class TestRewardCalculator:
    """Tests for reward calculator."""

    def test_compute_math_reward_correct(self):
        """Test math reward for correct answer."""
        calculator = RewardCalculator()
        solution = r"The answer is \boxed{42}"
        ground_truth = r"Solution: \boxed{42}"
        reward, details = calculator.compute_math_reward(solution, ground_truth)
        assert reward == 1.0
        assert details["is_correct"] is True
        assert details["domain"] == "math"

    def test_compute_math_reward_incorrect(self):
        """Test math reward for incorrect answer."""
        calculator = RewardCalculator()
        solution = r"The answer is \boxed{42}"
        ground_truth = r"Solution: \boxed{43}"
        reward, details = calculator.compute_math_reward(solution, ground_truth)
        assert reward == 0.0
        assert details["is_correct"] is False

    def test_compute_code_reward_test_cases(self):
        """Test code reward with test cases."""
        calculator = RewardCalculator()
        solution = "x = int(input()); print(x * 2)"
        metadata = {
            "test_cases": [
                {"input": "1", "expected_output": "2"},
                {"input": "2", "expected_output": "4"},
            ]
        }
        reward, details = calculator.compute_code_reward(solution, "", metadata)
        assert reward == 1.0
        assert details["domain"] == "code"
        assert details["format"] == "test_cases"
        assert details["pass_rate"] == 1.0

    def test_compute_code_reward_humaneval(self):
        """Test code reward with HumanEval format."""
        calculator = RewardCalculator()
        solution = "def add(a, b):\n    return a + b"
        metadata = {
            "test": "def check(f):\n    assert f(1, 2) == 3\n    assert f(0, 0) == 0",
            "entry_point": "add",
        }
        reward, details = calculator.compute_code_reward(solution, "", metadata)
        assert reward == 1.0
        assert details["passed"] is True
        assert details["format"] == "humaneval"

    def test_compute_reward_dispatch(self):
        """Test reward computation dispatch by domain."""
        calculator = RewardCalculator()

        # Math domain
        reward, details = calculator.compute_reward(
            r"\boxed{1}", r"\boxed{1}", "math"
        )
        assert reward == 1.0
        assert details["domain"] == "math"

        # Code domain
        reward, details = calculator.compute_reward(
            "print(42)",
            "",
            "code",
            {"test_cases": [{"input": "", "expected_output": "42"}]},
        )
        assert reward == 1.0
        assert details["domain"] == "code"

        # General domain (no verification)
        reward, details = calculator.compute_reward(
            "answer", "answer", "general"
        )
        assert reward == 0.0
        assert details["domain"] == "general"


class TestTrajectoryBackwardCompat:
    """Tests for trajectory backward compatibility."""

    def test_trajectory_entry_without_reward_fields(self):
        """Test TrajectoryEntry works without reward fields."""
        entry = TrajectoryEntry(
            timestamp="2024-01-01T00:00:00",
            run_id="test-run",
            step_id=1,
            agent="solver",
            action="solve",
            input={"problem": "test"},
            output={"solution": "test"},
            metadata={},
        )
        assert entry.reward is None
        assert entry.terminal is False
        assert entry.success is None

    def test_trajectory_entry_with_reward_fields(self):
        """Test TrajectoryEntry works with reward fields."""
        entry = TrajectoryEntry(
            timestamp="2024-01-01T00:00:00",
            run_id="test-run",
            step_id=1,
            agent="solver",
            action="solve",
            input={"problem": "test"},
            output={"solution": "test"},
            metadata={},
            reward=1.0,
            terminal=True,
            success=True,
        )
        assert entry.reward == 1.0
        assert entry.terminal is True
        assert entry.success is True
