"""Tests for CLI functionality and integration."""

import json
from pathlib import Path

import pytest
import yaml

from src.cli.runner import run_single
from src.models.config import PipelineConfig


@pytest.mark.asyncio
async def test_config_loading(tmp_config):
    """Test that pipeline.yaml loads correctly."""
    # Load the config file
    with open(tmp_config) as f:
        config_data = yaml.safe_load(f)

    # Parse into PipelineConfig
    config = PipelineConfig(**config_data)

    # Verify all fields parsed correctly
    assert config.config_version == "test-v1.0"
    assert config.max_iterations == 3
    assert config.solver.role == "solver"
    assert config.solver.model == "command-r-plus"
    assert config.verifier.role == "verifier"
    assert config.verifier.model == "command-r"
    assert config.judge.model == "command-r-plus"

    # Verify config_hash is deterministic
    hash1 = config.config_hash()
    hash2 = config.config_hash()
    assert hash1 == hash2
    assert len(hash1) == 8  # First 8 chars of SHA256


@pytest.mark.asyncio
async def test_runner_with_mocked_llm(tmp_config, mock_litellm, tmp_path):
    """Test run_single with mocked LLM."""
    # Run with direct problem string
    result = await run_single(
        config_path=str(tmp_config),
        problem="What is 2+2?",
    )

    # Verify PipelineResult has all fields
    assert result.problem_description == "What is 2+2?"
    assert result.solution == "The answer is 42"
    assert result.passed_verification is True
    assert result.judge_score == 0.85
    assert result.iterations == 1
    assert result.total_cost > 0
    assert result.trajectory_path != ""
    assert result.token_usage is not None
    assert result.cost_summary is not None


@pytest.mark.asyncio
async def test_per_agent_cost_breakdown(tmp_config, mock_litellm):
    """Test that cost_summary contains per-agent breakdown."""
    result = await run_single(
        config_path=str(tmp_config),
        problem="Test problem",
    )

    # Verify cost_summary structure
    cost_summary = result.cost_summary
    assert "total_cost_usd" in cost_summary
    assert "total_tokens" in cost_summary
    assert "by_agent" in cost_summary
    assert "by_model" in cost_summary

    # Verify per-agent breakdown
    by_agent = cost_summary["by_agent"]
    assert "solver" in by_agent
    assert "verifier" in by_agent
    assert "judge" in by_agent

    # Verify each agent has token counts and cost
    for agent_name in ["solver", "verifier", "judge"]:
        agent_data = by_agent[agent_name]
        assert "tokens" in agent_data
        assert "cost_usd" in agent_data

        tokens = agent_data["tokens"]
        assert tokens["prompt_tokens"] > 0
        assert tokens["completion_tokens"] > 0
        assert tokens["total_tokens"] > 0

        assert agent_data["cost_usd"] > 0


@pytest.mark.asyncio
async def test_trajectory_output(tmp_config, mock_litellm, tmp_path):
    """Test that trajectory JSONL file is created with correct entries."""
    result = await run_single(
        config_path=str(tmp_config),
        problem="Test problem",
    )

    # Verify trajectory file exists
    trajectory_path = Path(result.trajectory_path)
    assert trajectory_path.exists()

    # Read JSONL entries
    entries = []
    with open(trajectory_path) as f:
        for line in f:
            entries.append(json.loads(line))

    # Should have entries for solver, verifier, and judge
    assert len(entries) >= 3

    # Check that we have entries from each agent
    agents_seen = {entry["agent"] for entry in entries}
    assert "solver" in agents_seen
    assert "verifier" in agents_seen
    assert "judge" in agents_seen

    # Verify required fields are present
    for entry in entries:
        assert "run_id" in entry
        assert "timestamp" in entry
        assert "agent" in entry
        assert "action" in entry
        assert "step_id" in entry

        # Check for metadata field with model and tokens
        if "metadata" in entry:
            metadata = entry["metadata"]
            if "model" in metadata:
                assert metadata["model"] in ["command-r-plus", "command-r"]
            if "tokens" in metadata:
                tokens = metadata["tokens"]
                assert "prompt_tokens" in tokens
                assert "completion_tokens" in tokens
                assert "total_tokens" in tokens


@pytest.mark.asyncio
async def test_problem_from_file(tmp_config, mock_litellm, tmp_path):
    """Test loading problem from YAML file with metadata."""
    # Create a problem YAML file
    problem_data = {
        "description": "What is the capital of France?",
        "metadata": {
            "domain": "geography",
            "difficulty": "easy",
            "expected_answer": "Paris",
        },
    }

    problem_path = tmp_path / "test_problem.yaml"
    with open(problem_path, "w") as f:
        yaml.dump(problem_data, f)

    # Run with problem file path
    result = await run_single(
        config_path=str(tmp_config),
        problem=str(problem_path),
    )

    # Verify problem description loaded
    assert result.problem_description == "What is the capital of France?"


@pytest.mark.asyncio
async def test_cost_tracking(tmp_config, mock_litellm):
    """Test that total cost is calculated correctly."""
    result = await run_single(
        config_path=str(tmp_config),
        problem="Test problem",
    )

    # Verify total cost is positive
    assert result.total_cost > 0

    # Verify total cost matches sum of agent costs
    by_agent = result.cost_summary["by_agent"]
    agent_total = sum(agent["cost_usd"] for agent in by_agent.values())
    assert abs(result.total_cost - agent_total) < 0.000001  # Float comparison


@pytest.mark.asyncio
async def test_max_iterations_respected(tmp_config, mock_litellm, tmp_path):
    """Test that max_iterations is respected."""
    # Patch to make verifier always fail
    from unittest.mock import AsyncMock, MagicMock, patch

    from src.agents.solver import SolverResponse
    from src.models.evaluation import Judgment, VerificationResult
    from src.models.trajectory import TokenUsage

    solver_response = SolverResponse(
        solution="Wrong answer",
        reasoning="Guessing",
        confidence=0.5,
    )

    verif_response = VerificationResult(
        passed=False,  # Always fail
        critique="Solution is incorrect, try again",
        confidence=0.8,
    )

    judgment = Judgment(
        score=0.3,
        reasoning="Poor solution",
        strengths=[],
        weaknesses=["Incorrect"],
    )

    token_usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_llm_client.return_value = mock_client

        # Set up to always fail verification for max_iterations attempts
        # max_iterations=3, so we need 3 solver calls + 3 verifier calls + 1 judge call
        mock_client.generate.side_effect = [
            (solver_response, token_usage),  # Solver iter 1
            (verif_response, token_usage),  # Verifier iter 1
            (solver_response, token_usage),  # Solver iter 2
            (verif_response, token_usage),  # Verifier iter 2
            (solver_response, token_usage),  # Solver iter 3
            (verif_response, token_usage),  # Verifier iter 3
            (judgment, token_usage),  # Judge final
        ]

        result = await run_single(
            config_path=str(tmp_config),
            problem="Unsolvable problem",
        )

        # Verify iterations equals max_iterations
        assert result.iterations == 3
        assert result.passed_verification is False


@pytest.mark.asyncio
async def test_cli_arg_parsing():
    """Test CLI argument parsing."""
    import sys

    from src.cli.main import parse_args

    original_argv = sys.argv

    try:
        # Test basic args
        sys.argv = ["prog", "What is 2+2?"]
        args = parse_args()
        assert args.problem == "What is 2+2?"
        assert args.config == "config/pipeline.yaml"
        assert args.output_dir is None
        assert args.verbose is False

        # Test with all flags
        sys.argv = [
            "prog",
            "Test problem",
            "--config",
            "custom.yaml",
            "--output-dir",
            "./outputs",
            "-v",
        ]
        args = parse_args()
        assert args.problem == "Test problem"
        assert args.config == "custom.yaml"
        assert args.output_dir == "./outputs"
        assert args.verbose is True

    finally:
        sys.argv = original_argv
