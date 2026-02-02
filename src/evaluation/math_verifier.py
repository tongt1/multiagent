"""Math answer verification using symbolic equivalence."""

import re
from typing import Literal, Optional

import sympy as sp
from pydantic import BaseModel


class MathVerificationResult(BaseModel):
    """Result of math answer verification."""

    is_correct: bool
    predicted_answer: Optional[str]
    expected_answer: Optional[str]
    method: Literal["symbolic", "string_fallback", "extraction_failed"]


def extract_boxed_answer(solution: str) -> Optional[str]:
    """Extract answer from LaTeX \\boxed{...} with nested brace support.

    Args:
        solution: Solution text possibly containing \\boxed{...}

    Returns:
        Extracted answer string or None if not found
    """
    # Find all \boxed{ occurrences
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, solution))

    if not matches:
        return None

    # Use the last \boxed{ occurrence (in case of multiple)
    last_match = matches[-1]
    start_pos = last_match.end()

    # Count braces to handle nesting
    brace_depth = 1
    end_pos = start_pos

    while end_pos < len(solution) and brace_depth > 0:
        if solution[end_pos] == "{":
            brace_depth += 1
        elif solution[end_pos] == "}":
            brace_depth -= 1
        end_pos += 1

    if brace_depth == 0:
        return solution[start_pos : end_pos - 1]
    else:
        # Unclosed braces
        return None


def verify_math_answer(predicted: str, ground_truth: str) -> bool:
    """Verify if predicted answer is equivalent to ground truth using sympy.

    Args:
        predicted: Predicted answer string
        ground_truth: Expected answer string

    Returns:
        True if answers are equivalent, False otherwise
    """
    # Try symbolic equivalence first
    try:
        pred_expr = sp.sympify(predicted)
        truth_expr = sp.sympify(ground_truth)

        # Check equality
        result = pred_expr.equals(truth_expr)

        # If equals() returns None (can't determine), try simplification
        if result is None:
            try:
                diff = sp.simplify(pred_expr - truth_expr)
                result = diff == 0
            except Exception:
                # If simplification fails, fall back to string comparison
                result = None

        if result is not None:
            return bool(result)

    except Exception:
        # If sympify fails, fall through to string comparison
        pass

    # Fall back to string comparison
    return predicted.strip() == ground_truth.strip()


def compute_math_reward(
    solution: str, ground_truth_solution: str
) -> tuple[float, MathVerificationResult]:
    """Compute reward for math problem by extracting and verifying answers.

    Args:
        solution: Solution text with \\boxed{answer}
        ground_truth_solution: Ground truth solution with \\boxed{answer}

    Returns:
        Tuple of (reward, verification_result)
        Reward is 1.0 if correct, 0.0 otherwise
    """
    # Extract answers
    predicted_answer = extract_boxed_answer(solution)
    expected_answer = extract_boxed_answer(ground_truth_solution)

    # If extraction failed, can't verify
    if predicted_answer is None or expected_answer is None:
        return (
            0.0,
            MathVerificationResult(
                is_correct=False,
                predicted_answer=predicted_answer,
                expected_answer=expected_answer,
                method="extraction_failed",
            ),
        )

    # Verify equivalence
    is_correct = verify_math_answer(predicted_answer, expected_answer)

    # Determine method used (symbolic if sympify worked, else string_fallback)
    method: Literal["symbolic", "string_fallback"] = "symbolic"
    try:
        sp.sympify(predicted_answer)
        sp.sympify(expected_answer)
    except Exception:
        method = "string_fallback"

    result = MathVerificationResult(
        is_correct=is_correct,
        predicted_answer=predicted_answer,
        expected_answer=expected_answer,
        method=method,
    )

    reward = 1.0 if is_correct else 0.0
    return reward, result
