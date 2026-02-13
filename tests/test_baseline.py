"""Tests for baseline mode (single-agent without debate)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.solver import SolverResponse
from src.models.config import AgentConfig, JudgeConfig, PipelineConfig
from src.models.trajectory import TokenUsage
from src.orchestration.baseline_runner import BaselineRunner
from src.orchestration.pipeline import PipelineResult, SolverVerifierJudgePipeline


@pytest.fixture
def baseline_config():
    """Create a baseline mode pipeline configuration."""
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
        mode="baseline",  # Baseline mode
        max_iterations=5,
        trajectory_output_dir="test_trajectories",
        config_version="1.0.0",
    )


@pytest.fixture
def debate_config():
    """Create a debate mode pipeline configuration for comparison."""
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
        mode="debate",  # Debate mode
        max_iterations=5,
        trajectory_output_dir="test_trajectories",
        config_version="1.0.0",
    )


@pytest.fixture
def mock_token_usage():
    """Create mock token usage."""
    return TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)


@pytest.mark.asyncio
async def test_baseline_uses_same_solver_config(baseline_config):
    """Test that BaselineRunner uses identical solver model, temperature, prompt as debate."""
    runner = BaselineRunner(baseline_config)

    # Verify solver config matches
    assert runner.solver.config.model == baseline_config.solver.model
    assert runner.solver.config.temperature == baseline_config.solver.temperature
    assert runner.solver.config.prompt_template == baseline_config.solver.prompt_template
    assert runner.solver.config.system_prompt == baseline_config.solver.system_prompt

    # Verify client config
    assert runner.solver_client.model == baseline_config.solver.model
    assert runner.solver_client.temperature == baseline_config.solver.temperature


@pytest.mark.asyncio
async def test_baseline_calls_solver_without_feedback(baseline_config, mock_token_usage, tmp_path):
    """Test that BaselineRunner calls solver.generate() with feedback=None."""
    baseline_config.trajectory_output_dir = str(tmp_path / "trajectories")

    mock_solver_response = SolverResponse(
        solution="The answer is 42",
        reasoning="I calculated it",
        confidence=0.95,
    )

    with patch("src.orchestration.baseline_runner.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_client.max_tokens = 4096  # Initial value
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.return_value = (mock_solver_response, mock_token_usage)

        runner = BaselineRunner(baseline_config)
        result = await runner.run(
            problem_description="What is the meaning of life?",
        )

        # Verify solver was called with feedback=None
        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args[1]

        # Check that messages don't contain feedback
        messages = call_kwargs["messages"]
        # The prompt should not have feedback section populated
        prompt_text = messages[0]["content"]
        assert "Previous feedback from verifier:" not in prompt_text


@pytest.mark.asyncio
async def test_baseline_computes_ground_truth_reward(baseline_config, mock_token_usage, tmp_path):
    """Test that baseline computes ground truth reward for correct answer."""
    baseline_config.trajectory_output_dir = str(tmp_path / "trajectories")

    # Mock solver returns correct answer
    mock_solver_response = SolverResponse(
        solution="The answer is \\boxed{4}",
        reasoning="2 + 2 = 4",
        confidence=0.95,
    )

    with patch("src.orchestration.baseline_runner.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_client.max_tokens = 4096
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.return_value = (mock_solver_response, mock_token_usage)

        runner = BaselineRunner(baseline_config)
        result = await runner.run(
            problem_description="What is 2 + 2?",
            problem_metadata={"ground_truth": "The answer is \\boxed{4}"},
        )

        # Verify ground truth reward
        assert result.ground_truth_reward == 1.0  # Correct
        assert result.ground_truth_details is not None
        assert result.ground_truth_details["is_correct"] is True
        assert result.ground_truth_details["predicted_answer"] == "4"
        assert result.ground_truth_details["expected_answer"] == "4"
        assert result.passed_verification is True


@pytest.mark.asyncio
async def test_baseline_reward_incorrect(baseline_config, mock_token_usage, tmp_path):
    """Test that baseline computes ground truth reward as 0.0 for incorrect answer."""
    baseline_config.trajectory_output_dir = str(tmp_path / "trajectories")

    # Mock solver returns wrong answer
    mock_solver_response = SolverResponse(
        solution="The answer is \\boxed{5}",
        reasoning="Wrong calculation",
        confidence=0.5,
    )

    with patch("src.orchestration.baseline_runner.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_client.max_tokens = 4096
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.return_value = (mock_solver_response, mock_token_usage)

        runner = BaselineRunner(baseline_config)
        result = await runner.run(
            problem_description="What is 2 + 2?",
            problem_metadata={"ground_truth": "The answer is \\boxed{4}"},
        )

        # Verify ground truth reward
        assert result.ground_truth_reward == 0.0  # Incorrect
        assert result.ground_truth_details is not None
        assert result.ground_truth_details["is_correct"] is False
        assert result.ground_truth_details["predicted_answer"] == "5"
        assert result.ground_truth_details["expected_answer"] == "4"
        assert result.passed_verification is False


@pytest.mark.asyncio
async def test_baseline_trajectory_logged(baseline_config, mock_token_usage, tmp_path):
    """Test that baseline creates trajectory JSONL file with correct format."""
    baseline_config.trajectory_output_dir = str(tmp_path / "trajectories")

    mock_solver_response = SolverResponse(
        solution="The answer is \\boxed{42}",
        reasoning="Calculation",
        confidence=0.9,
    )

    with patch("src.orchestration.baseline_runner.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_client.max_tokens = 4096
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.return_value = (mock_solver_response, mock_token_usage)

        runner = BaselineRunner(baseline_config)
        result = await runner.run(
            problem_description="Test problem",
            problem_metadata={"ground_truth": "The answer is \\boxed{42}"},
        )

        # Verify trajectory file exists
        traj_path = Path(result.trajectory_path)
        assert traj_path.exists()

        # Read and verify entries
        entries = []
        with traj_path.open("r") as f:
            for line in f:
                entries.append(json.loads(line))

        # Should have 2 entries: solver and reward
        assert len(entries) == 2

        # First entry: solver with mode='baseline'
        assert entries[0]["agent"] == "solver"
        assert entries[0]["action"] == "generate"
        assert entries[0]["metadata"]["mode"] == "baseline"
        assert entries[0]["metadata"]["iteration"] == 1
        assert entries[0]["input"]["feedback"] is None

        # Second entry: reward with mode='baseline'
        assert entries[1]["agent"] == "reward"
        assert entries[1]["action"] == "ground_truth_verify"
        assert entries[1]["metadata"]["mode"] == "baseline"
        assert entries[1]["output"]["reward"] == 1.0


@pytest.mark.asyncio
async def test_baseline_iterations_is_one(baseline_config, mock_token_usage, tmp_path):
    """Test that baseline result always has iterations=1."""
    baseline_config.trajectory_output_dir = str(tmp_path / "trajectories")

    mock_solver_response = SolverResponse(
        solution="Test solution",
        reasoning="Test reasoning",
        confidence=0.8,
    )

    with patch("src.orchestration.baseline_runner.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_client.max_tokens = 4096
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.return_value = (mock_solver_response, mock_token_usage)

        runner = BaselineRunner(baseline_config)
        result = await runner.run(problem_description="Test problem")

        # Verify iterations is always 1
        assert result.iterations == 1


@pytest.mark.asyncio
async def test_pipeline_routes_to_baseline(baseline_config, mock_token_usage, tmp_path):
    """Test that pipeline routes to BaselineRunner when mode='baseline'."""
    baseline_config.trajectory_output_dir = str(tmp_path / "trajectories")

    mock_solver_response = SolverResponse(
        solution="Test solution",
        reasoning="Test reasoning",
        confidence=0.8,
    )

    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_client.max_tokens = 4096
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.return_value = (mock_solver_response, mock_token_usage)

        # Create pipeline with baseline config
        pipeline = SolverVerifierJudgePipeline(baseline_config)

        # Verify baseline runner was initialized
        assert hasattr(pipeline, "_baseline_runner")
        assert isinstance(pipeline._baseline_runner, BaselineRunner)

        # Run pipeline and verify it uses baseline
        result = await pipeline.run(problem_description="Test problem")

        assert isinstance(result, PipelineResult)
        assert result.iterations == 1
        assert result.judge_score == 0.0  # No judge in baseline


@pytest.mark.asyncio
async def test_baseline_token_budget(baseline_config, mock_token_usage, tmp_path):
    """Test that baseline runner applies token_budget parameter."""
    baseline_config.trajectory_output_dir = str(tmp_path / "trajectories")

    mock_solver_response = SolverResponse(
        solution="Test solution",
        reasoning="Test reasoning",
        confidence=0.8,
    )

    with patch("src.orchestration.baseline_runner.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_client.max_tokens = 4096  # Initial value
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.return_value = (mock_solver_response, mock_token_usage)

        runner = BaselineRunner(baseline_config)

        # Run with token budget
        token_budget = 2000
        result = await runner.run(
            problem_description="Test problem",
            token_budget=token_budget,
        )

        # Verify max_tokens was set during call
        # (It gets restored after, so we can't check final value)
        # But we can verify the call succeeded with budget
        assert result.iterations == 1

        # Verify max_tokens was restored to original
        assert mock_client.max_tokens == 4096


@pytest.mark.asyncio
async def test_baseline_no_judge_score(baseline_config, mock_token_usage, tmp_path):
    """Test that baseline does not run judge (judge_score=0.0)."""
    baseline_config.trajectory_output_dir = str(tmp_path / "trajectories")

    mock_solver_response = SolverResponse(
        solution="Test solution",
        reasoning="Test reasoning",
        confidence=0.8,
    )

    with patch("src.orchestration.baseline_runner.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_client.max_tokens = 4096
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.return_value = (mock_solver_response, mock_token_usage)

        runner = BaselineRunner(baseline_config)
        result = await runner.run(problem_description="Test problem")

        # Baseline should not run judge
        assert result.judge_score == 0.0


@pytest.mark.asyncio
async def test_baseline_cost_tracking(baseline_config, mock_token_usage, tmp_path):
    """Test that baseline tracks costs correctly."""
    baseline_config.trajectory_output_dir = str(tmp_path / "trajectories")

    mock_solver_response = SolverResponse(
        solution="Test solution",
        reasoning="Test reasoning",
        confidence=0.8,
    )

    with patch("src.orchestration.baseline_runner.LLMClient") as mock_llm_client_cls:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock()
        mock_client.max_tokens = 4096
        mock_llm_client_cls.return_value = mock_client

        mock_client.generate.return_value = (mock_solver_response, mock_token_usage)

        runner = BaselineRunner(baseline_config)
        result = await runner.run(problem_description="Test problem")

        # Verify cost tracking
        assert result.total_cost > 0.0
        assert result.token_usage.total_tokens == 150
        assert "total_cost_usd" in result.cost_summary
        assert "by_agent" in result.cost_summary
        assert "solver" in result.cost_summary["by_agent"]
