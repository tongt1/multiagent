"""Shared test fixtures."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.agents.solver import SolverResponse
from src.models.evaluation import Judgment, VerificationResult
from src.models.trajectory import TokenUsage


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Create a minimal pipeline.yaml in tmp_path.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        Path to the config file
    """
    config_data = {
        "config_version": "test-v1.0",
        "max_iterations": 3,
        "trajectory_output_dir": str(tmp_path / "trajectories"),
        "solver": {
            "role": "solver",
            "model": "command-r-plus",
            "temperature": 0.7,
            "max_tokens": 2048,
            "system_prompt": "You are a solver.",
            "prompt_template": "Problem: {problem}\n{feedback_section}\nSolve:",
        },
        "verifier": {
            "role": "verifier",
            "model": "command-r",
            "temperature": 0.0,
            "max_tokens": 1024,
            "system_prompt": "You are a verifier.",
            "prompt_template": "Problem: {problem}\nSolution: {solution}\nVerify:",
        },
        "judge": {
            "model": "command-r-plus",
            "temperature": 0.0,
            "max_tokens": 1024,
            "system_prompt": "You are a judge.",
            "prompt_template": "Problem: {problem}\nSolution: {solution}\n{rubric}\nScore:",
            "scoring_rubric": "Score 0-1 based on correctness.",
        },
    }

    config_path = tmp_path / "test_pipeline.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path


@pytest.fixture
def mock_litellm():
    """Patch litellm.acompletion to return canned responses.

    Returns:
        Patch context manager
    """
    # Create realistic mock responses with token usage
    solver_response = SolverResponse(
        solution="The answer is 42",
        reasoning="Through careful calculation",
        confidence=0.95,
    )

    verif_response = VerificationResult(
        passed=True,
        critique=None,
        confidence=0.9,
    )

    judgment = Judgment(
        score=0.85,
        reasoning="Good solution",
        strengths=["Clear reasoning", "Correct answer"],
        weaknesses=["Could be more detailed"],
    )

    # Create mock token usage
    token_usage = TokenUsage(
        prompt_tokens=150,
        completion_tokens=75,
        total_tokens=225,
    )

    # Patch LLMClient to avoid actual API calls
    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client:
        # Create mock client instance
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_llm_client.return_value = mock_client

        # Set up return values for generate calls
        mock_client.generate.side_effect = [
            (solver_response, token_usage),  # Solver iteration 1
            (verif_response, token_usage),  # Verifier iteration 1
            (judgment, token_usage),  # Judge final
        ]

        yield mock_llm_client
