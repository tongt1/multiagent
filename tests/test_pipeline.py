"""Tests for pipeline orchestration."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.solver import SolverResponse
from src.models.config import AgentConfig, JudgeConfig, PipelineConfig
from src.models.evaluation import Judgment, VerificationResult
from src.models.trajectory import TokenUsage
from src.orchestration.pipeline import PipelineResult, SolverVerifierJudgePipeline


@pytest.fixture
def mock_config():
    """Create a test pipeline configuration."""
    return PipelineConfig(
        solver=AgentConfig(
            role="solver",
            model="command-r-plus",
            temperature=0.0,
            max_tokens=4096,
            prompt_template="Problem: {problem}\n{feedback_section}\nProvide solution:",
            system_prompt="You are a problem solver.",
        ),
        verifier=AgentConfig(
            role="verifier",
            model="command-r",
            temperature=0.0,
            max_tokens=2048,
            prompt_template="Problem: {problem}\nSolution: {solution}\nVerify:",
            system_prompt="You are a verifier.",
        ),
        judge=JudgeConfig(
            model="command-r-plus",
            temperature=0.0,
            max_tokens=2048,
            prompt_template="Problem: {problem}\nSolution: {solution}\nRubric: {rubric}\nScore:",
            system_prompt="You are a judge.",
            scoring_rubric="Score 0-1 based on correctness and quality.",
        ),
        max_iterations=5,
        trajectory_output_dir="test_trajectories",
        config_version="1.0.0",
    )


@pytest.fixture
def mock_token_usage():
    """Create mock token usage."""
    return TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)


@pytest.mark.asyncio
async def test_pipeline_completes_successfully(mock_config, mock_token_usage, tmp_path):
    """Test that pipeline completes with mocked agents."""
    # Update config to use tmp_path
    mock_config.trajectory_output_dir = str(tmp_path / "trajectories")

    # Create mock responses
    mock_solver_response = SolverResponse(
        solution="The answer is 42",
        reasoning="I calculated it",
        confidence=0.95,
    )
    mock_verif_response = VerificationResult(
        passed=True,
        critique=None,
        confidence=0.9,
    )
    mock_judgment = Judgment(
        score=0.85,
        reasoning="Good solution",
        strengths=["Clear", "Correct"],
        weaknesses=["Could be more detailed"],
    )

    # Patch LLMClient to avoid actual API calls
    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client_cls:
        # Create mock client instances
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_llm_client_cls.return_value = mock_client

        # Set up return values for generate calls
        mock_client.generate.side_effect = [
            (mock_solver_response, mock_token_usage),  # Solver call
            (mock_verif_response, mock_token_usage),  # Verifier call
            (mock_judgment, mock_token_usage),  # Judge call
        ]

        # Create and run pipeline
        pipeline = SolverVerifierJudgePipeline(mock_config)
        result = await pipeline.run(
            problem_description="What is the meaning of life?",
        )

        # Verify result
        assert isinstance(result, PipelineResult)
        assert result.problem_description == "What is the meaning of life?"
        assert result.solution == "The answer is 42"
        assert result.passed_verification is True
        assert result.judge_score == 0.85
        assert result.iterations == 1
        assert result.total_cost > 0.0
        assert "cost_summary" in result.model_dump()
        assert Path(result.trajectory_path).exists()


@pytest.mark.asyncio
async def test_pipeline_respects_max_iterations(mock_config, mock_token_usage, tmp_path):
    """Test that pipeline stops at max_iterations when verification fails."""
    # Update config to use tmp_path and set low max_iterations
    mock_config.trajectory_output_dir = str(tmp_path / "trajectories")
    mock_config.max_iterations = 3

    # Create mock responses - verifier always fails
    mock_solver_response = SolverResponse(
        solution="Wrong answer",
        reasoning="Guessing",
        confidence=0.5,
    )
    mock_verif_response = VerificationResult(
        passed=False,
        critique="This is incorrect",
        confidence=0.8,
    )
    mock_judgment = Judgment(
        score=0.3,
        reasoning="Incorrect solution",
        strengths=["Attempted"],
        weaknesses=["Wrong"],
    )

    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_llm_client_cls.return_value = mock_client

        # Set up return values - 3 solver-verifier iterations + 1 judge
        mock_client.generate.side_effect = [
            (mock_solver_response, mock_token_usage),  # Solver iter 1
            (mock_verif_response, mock_token_usage),  # Verifier iter 1
            (mock_solver_response, mock_token_usage),  # Solver iter 2
            (mock_verif_response, mock_token_usage),  # Verifier iter 2
            (mock_solver_response, mock_token_usage),  # Solver iter 3
            (mock_verif_response, mock_token_usage),  # Verifier iter 3
            (mock_judgment, mock_token_usage),  # Judge
        ]

        pipeline = SolverVerifierJudgePipeline(mock_config)
        result = await pipeline.run(problem_description="Test problem")

        # Should stop at max_iterations
        assert result.iterations == 3
        assert result.passed_verification is False
        assert result.judge_score == 0.3


@pytest.mark.asyncio
async def test_pipeline_terminates_early_on_pass(mock_config, mock_token_usage, tmp_path):
    """Test that pipeline terminates early when verification passes."""
    mock_config.trajectory_output_dir = str(tmp_path / "trajectories")
    mock_config.max_iterations = 5

    # First iteration fails, second passes
    mock_solver_response_1 = SolverResponse(
        solution="First attempt",
        reasoning="Initial try",
        confidence=0.6,
    )
    mock_verif_response_1 = VerificationResult(
        passed=False,
        critique="Needs improvement",
        confidence=0.7,
    )
    mock_solver_response_2 = SolverResponse(
        solution="Improved solution",
        reasoning="Addressed feedback",
        confidence=0.9,
    )
    mock_verif_response_2 = VerificationResult(
        passed=True,
        critique=None,
        confidence=0.95,
    )
    mock_judgment = Judgment(
        score=0.9,
        reasoning="Excellent",
        strengths=["Correct", "Well-reasoned"],
        weaknesses=[],
    )

    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.side_effect = [
            (mock_solver_response_1, mock_token_usage),  # Solver iter 1
            (mock_verif_response_1, mock_token_usage),  # Verifier iter 1 (fail)
            (mock_solver_response_2, mock_token_usage),  # Solver iter 2
            (mock_verif_response_2, mock_token_usage),  # Verifier iter 2 (pass)
            (mock_judgment, mock_token_usage),  # Judge
        ]

        pipeline = SolverVerifierJudgePipeline(mock_config)
        result = await pipeline.run(problem_description="Test problem")

        # Should stop at iteration 2 when verification passes
        assert result.iterations == 2
        assert result.passed_verification is True
        assert result.solution == "Improved solution"


@pytest.mark.asyncio
async def test_pipeline_result_fields(mock_config, mock_token_usage, tmp_path):
    """Test that PipelineResult has all expected fields populated."""
    mock_config.trajectory_output_dir = str(tmp_path / "trajectories")

    mock_solver_response = SolverResponse(
        solution="Test solution", reasoning="Test", confidence=0.8
    )
    mock_verif_response = VerificationResult(passed=True, confidence=0.9)
    mock_judgment = Judgment(
        score=0.75, reasoning="OK", strengths=["Good"], weaknesses=["Meh"]
    )

    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.side_effect = [
            (mock_solver_response, mock_token_usage),
            (mock_verif_response, mock_token_usage),
            (mock_judgment, mock_token_usage),
        ]

        pipeline = SolverVerifierJudgePipeline(mock_config)
        result = await pipeline.run(problem_description="Test")

        # Check all fields
        assert result.problem_description == "Test"
        assert result.solution == "Test solution"
        assert result.passed_verification is True
        assert result.judge_score == 0.75
        assert result.iterations >= 1
        assert result.total_cost >= 0.0
        assert result.trajectory_path != ""
        assert isinstance(result.token_usage, TokenUsage)
        assert result.token_usage.total_tokens > 0
        assert isinstance(result.cost_summary, dict)


@pytest.mark.asyncio
async def test_pipeline_cost_summary_breakdown(mock_config, mock_token_usage, tmp_path):
    """Test that cost_summary contains per-model and per-agent breakdowns."""
    mock_config.trajectory_output_dir = str(tmp_path / "trajectories")

    mock_solver_response = SolverResponse(
        solution="Solution", reasoning="R", confidence=0.8
    )
    mock_verif_response = VerificationResult(passed=True, confidence=0.9)
    mock_judgment = Judgment(score=0.8, reasoning="R", strengths=["S"], weaknesses=["W"])

    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.side_effect = [
            (mock_solver_response, mock_token_usage),
            (mock_verif_response, mock_token_usage),
            (mock_judgment, mock_token_usage),
        ]

        pipeline = SolverVerifierJudgePipeline(mock_config)
        result = await pipeline.run(problem_description="Test")

        # Check cost_summary structure
        assert "total_cost_usd" in result.cost_summary
        assert "total_tokens" in result.cost_summary
        assert "by_model" in result.cost_summary
        assert "by_agent" in result.cost_summary

        # Check per-agent breakdown exists
        assert "solver" in result.cost_summary["by_agent"]
        assert "verifier" in result.cost_summary["by_agent"]
        assert "judge" in result.cost_summary["by_agent"]

        # Each agent should have tokens and cost
        for agent_data in result.cost_summary["by_agent"].values():
            assert "tokens" in agent_data
            assert "cost_usd" in agent_data


@pytest.mark.asyncio
async def test_pipeline_creates_trajectory_file(mock_config, mock_token_usage, tmp_path):
    """Test that trajectory file is created and contains expected entries."""
    mock_config.trajectory_output_dir = str(tmp_path / "trajectories")

    mock_solver_response = SolverResponse(
        solution="Solution", reasoning="R", confidence=0.8
    )
    mock_verif_response = VerificationResult(passed=True, confidence=0.9)
    mock_judgment = Judgment(score=0.8, reasoning="R", strengths=["S"], weaknesses=["W"])

    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.side_effect = [
            (mock_solver_response, mock_token_usage),
            (mock_verif_response, mock_token_usage),
            (mock_judgment, mock_token_usage),
        ]

        pipeline = SolverVerifierJudgePipeline(mock_config)
        result = await pipeline.run(problem_description="Test")

        # Check trajectory file exists
        traj_path = Path(result.trajectory_path)
        assert traj_path.exists()

        # Read and verify entries
        entries = []
        with traj_path.open("r") as f:
            for line in f:
                entries.append(json.loads(line))

        # Should have 3 entries: solver, verifier, judge
        assert len(entries) == 3
        assert entries[0]["agent"] == "solver"
        assert entries[0]["action"] == "generate"
        assert entries[1]["agent"] == "verifier"
        assert entries[1]["action"] == "validate"
        assert entries[2]["agent"] == "judge"
        assert entries[2]["action"] == "score"


@pytest.mark.asyncio
async def test_pipeline_cost_tracker_accumulates(mock_config, mock_token_usage, tmp_path):
    """Test that cost tracker accumulates across all agent calls."""
    mock_config.trajectory_output_dir = str(tmp_path / "trajectories")

    # Create different token usages for each agent
    solver_usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    verifier_usage = TokenUsage(prompt_tokens=80, completion_tokens=40, total_tokens=120)
    judge_usage = TokenUsage(prompt_tokens=90, completion_tokens=60, total_tokens=150)

    mock_solver_response = SolverResponse(
        solution="Solution", reasoning="R", confidence=0.8
    )
    mock_verif_response = VerificationResult(passed=True, confidence=0.9)
    mock_judgment = Judgment(score=0.8, reasoning="R", strengths=["S"], weaknesses=["W"])

    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.side_effect = [
            (mock_solver_response, solver_usage),
            (mock_verif_response, verifier_usage),
            (mock_judgment, judge_usage),
        ]

        pipeline = SolverVerifierJudgePipeline(mock_config)
        result = await pipeline.run(problem_description="Test")

        # Total tokens should be sum of all usages
        expected_total = 150 + 120 + 150
        assert result.token_usage.total_tokens == expected_total
        assert result.total_cost > 0.0

        # Check that cost_summary has correct total
        assert result.cost_summary["total_tokens"]["total_tokens"] == expected_total
