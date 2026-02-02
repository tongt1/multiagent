"""Code execution with sandboxing and resource limits."""

import platform
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired
from typing import Any, Optional

from pydantic import BaseModel


class CodeExecutionResult(BaseModel):
    """Result of code execution."""

    passed: bool
    stdout: str
    stderr: str
    return_code: Optional[int]
    timed_out: bool
    error: Optional[str]


def _limit_resources() -> None:
    """Set resource limits for subprocess (Unix only).

    Sets:
    - RLIMIT_AS: 256MB memory limit
    - RLIMIT_CPU: 5s CPU time limit
    """
    if platform.system() == "Windows":
        return

    try:
        import resource

        # 256MB memory limit
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))

        # 5 second CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    except Exception:
        # If resource module not available or fails, continue without limits
        pass


def execute_code(code: str, stdin: str = "", timeout: int = 10) -> CodeExecutionResult:
    """Execute Python code in a subprocess with timeout and resource limits.

    Args:
        code: Python code to execute
        stdin: Input to provide to the code
        timeout: Timeout in seconds (default: 10)

    Returns:
        CodeExecutionResult with execution details
    """
    # Determine if we should use preexec_fn (Unix only)
    preexec_fn = _limit_resources if platform.system() != "Windows" else None

    # Use sys.executable as the most reliable option
    python_cmd = sys.executable

    try:
        result = subprocess.run(
            [python_cmd, "-c", code],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=preexec_fn,
        )

        return CodeExecutionResult(
            passed=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            timed_out=False,
            error=None,
        )

    except TimeoutExpired:
        return CodeExecutionResult(
            passed=False,
            stdout="",
            stderr="",
            return_code=None,
            timed_out=True,
            error="Execution timed out",
        )

    except Exception as e:
        return CodeExecutionResult(
            passed=False,
            stdout="",
            stderr="",
            return_code=None,
            timed_out=False,
            error=str(e),
        )


def execute_with_tests(
    code: str, test_cases: list, timeout_per_test: int = 5
) -> tuple:
    """Execute code against multiple test cases.

    Args:
        code: Python code to test
        test_cases: List of test cases with 'input' and 'expected_output' keys
        timeout_per_test: Timeout per test in seconds (default: 5)

    Returns:
        Tuple of (pass_rate, list of results)
        Pass rate is the fraction of tests passed (0.0 to 1.0)
    """
    results = []

    for test_case in test_cases:
        stdin = test_case.get("input", "")
        expected_output = test_case.get("expected_output", "")

        result = execute_code(code, stdin=stdin, timeout=timeout_per_test)

        # Check if output matches expected
        if result.passed and result.stdout.strip() == expected_output.strip():
            result.passed = True
        else:
            result.passed = False

        results.append(result)

    # Calculate pass rate
    passed_count = sum(1 for r in results if r.passed)
    pass_rate = passed_count / len(test_cases) if test_cases else 0.0

    return pass_rate, results


def execute_humaneval(
    code: str, test_code: str, entry_point: str, timeout: int = 10
) -> CodeExecutionResult:
    """Execute HumanEval-style code with test harness.

    Args:
        code: Generated code (typically a function implementation)
        test_code: Test harness code (typically contains check() function)
        entry_point: Name of the function being tested
        timeout: Timeout in seconds (default: 10)

    Returns:
        CodeExecutionResult with execution details
    """
    # Combine the code and test harness
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"

    return execute_code(full_code, timeout=timeout)
