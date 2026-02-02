"""Tests for infrastructure services."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from src.infrastructure.cost_tracker import CostTracker
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.trajectory_logger import TrajectoryLogger
from src.models.trajectory import TokenUsage, TrajectoryEntry


class MockResponse(BaseModel):
    """Mock Pydantic response for testing."""

    answer: str
    confidence: float


@pytest.mark.asyncio
async def test_llm_client_generate():
    """Test LLMClient.generate returns parsed model and token usage."""
    client = LLMClient(model="command-r-plus", temperature=0.0, max_tokens=2048)

    # Create mock response with usage
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    mock_raw_response = MagicMock()
    mock_raw_response.usage = mock_usage

    mock_result = MockResponse(answer="The answer is 42", confidence=0.95)
    mock_result._raw_response = mock_raw_response

    # Mock instructor client
    with patch.object(
        client.client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_result

        messages = [{"role": "user", "content": "What is the answer?"}]
        response, token_usage = await client.generate(
            messages=messages,
            response_model=MockResponse,
            system_prompt="You are a helpful assistant.",
        )

        # Verify response
        assert isinstance(response, MockResponse)
        assert response.answer == "The answer is 42"
        assert response.confidence == 0.95

        # Verify token usage
        assert token_usage.prompt_tokens == 100
        assert token_usage.completion_tokens == 50
        assert token_usage.total_tokens == 150

        # Verify call was made correctly
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "command-r-plus"
        assert call_kwargs["response_model"] == MockResponse
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 2048
        assert len(call_kwargs["messages"]) == 2  # system + user


@pytest.mark.asyncio
async def test_llm_client_generate_text():
    """Test LLMClient.generate_text returns text and token usage."""
    client = LLMClient(model="command-r-plus", temperature=0.7, max_tokens=1024)

    # Create mock response
    mock_message = MagicMock()
    mock_message.content = "This is the generated text response."

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 75
    mock_usage.completion_tokens = 25
    mock_usage.total_tokens = 100

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    # Mock litellm.acompletion
    with patch("src.infrastructure.llm_client.acompletion", new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = mock_response

        messages = [{"role": "user", "content": "Generate text"}]
        text, token_usage = await client.generate_text(
            messages=messages,
            system_prompt="Be creative.",
        )

        # Verify response
        assert text == "This is the generated text response."

        # Verify token usage
        assert token_usage.prompt_tokens == 75
        assert token_usage.completion_tokens == 25
        assert token_usage.total_tokens == 100

        # Verify call was made correctly
        mock_ac.assert_called_once()
        call_kwargs = mock_ac.call_args.kwargs
        assert call_kwargs["model"] == "command-r-plus"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1024
        assert len(call_kwargs["messages"]) == 2  # system + user


def test_trajectory_logger_writes_jsonl(tmp_path: Path):
    """Test TrajectoryLogger writes JSONL with correct step IDs and flush."""
    output_file = tmp_path / "trajectory.jsonl"
    logger = TrajectoryLogger(
        output_path=output_file,
        run_id="test-run-123",
        config_hash="abc12345",
    )

    with logger:
        # Log 3 steps
        logger.log_step(
            agent="solver",
            action="generate_solution",
            input_data={"problem": "What is 2+2?"},
            output_data={"solution": "4"},
            metadata={"iteration": 1},
        )

        logger.log_step(
            agent="verifier",
            action="verify_solution",
            input_data={"solution": "4"},
            output_data={"passed": True},
            metadata={"iteration": 1},
        )

        logger.log_step(
            agent="judge",
            action="evaluate_solution",
            input_data={"solution": "4"},
            output_data={"score": 1.0},
            metadata={"iteration": 1},
        )

    # Verify file exists and is readable
    assert output_file.exists()

    # Read and parse JSONL
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 3

    # Parse each entry
    entries = [json.loads(line) for line in lines]

    # Verify step IDs increment
    assert entries[0]["step_id"] == 1
    assert entries[1]["step_id"] == 2
    assert entries[2]["step_id"] == 3

    # Verify run_id
    assert all(e["run_id"] == "test-run-123" for e in entries)

    # Verify agents
    assert entries[0]["agent"] == "solver"
    assert entries[1]["agent"] == "verifier"
    assert entries[2]["agent"] == "judge"

    # Verify each entry is parseable as TrajectoryEntry
    for entry_dict in entries:
        entry = TrajectoryEntry(**entry_dict)
        assert entry.run_id == "test-run-123"


def test_cost_tracker_accumulates_usage():
    """Test CostTracker adds usage and calculates costs."""
    tracker = CostTracker()

    # Add usage for command-r-plus
    usage1 = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost1 = tracker.add_usage(model="command-r-plus", usage=usage1)

    # Verify cost calculation (1000/1000 * 0.003) + (500/1000 * 0.015) = 0.003 + 0.0075 = 0.0105
    assert abs(cost1 - 0.0105) < 0.0001

    # Add usage for command-r
    usage2 = TokenUsage(prompt_tokens=2000, completion_tokens=1000, total_tokens=3000)
    cost2 = tracker.add_usage(model="command-r", usage=usage2)

    # Verify cost calculation (2000/1000 * 0.0005) + (1000/1000 * 0.0015) = 0.001 + 0.0015 = 0.0025
    assert abs(cost2 - 0.0025) < 0.0001

    # Verify total cost
    total_cost = tracker.total_cost()
    assert abs(total_cost - (cost1 + cost2)) < 0.0001

    # Verify total tokens
    total_tokens = tracker.total_tokens()
    assert total_tokens.prompt_tokens == 3000
    assert total_tokens.completion_tokens == 1500
    assert total_tokens.total_tokens == 4500

    # Verify summary
    summary = tracker.summary()
    assert summary["total_cost_usd"] > 0
    assert "command-r-plus" in summary["by_model"]
    assert "command-r" in summary["by_model"]


def test_cost_tracker_per_agent_tracking():
    """Test CostTracker tracks costs per agent."""
    tracker = CostTracker()

    # Add usage with agent labels
    usage1 = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    tracker.add_usage(model="command-r-plus", usage=usage1, agent="solver")

    usage2 = TokenUsage(prompt_tokens=1500, completion_tokens=750, total_tokens=2250)
    tracker.add_usage(model="command-r-plus", usage=usage2, agent="verifier")

    usage3 = TokenUsage(prompt_tokens=500, completion_tokens=250, total_tokens=750)
    tracker.add_usage(model="command-r", usage=usage3, agent="solver")

    # Verify summary has per-agent breakdown
    summary = tracker.summary()
    assert "by_agent" in summary
    assert "solver" in summary["by_agent"]
    assert "verifier" in summary["by_agent"]

    # Verify solver tokens (usage1 + usage3)
    solver_tokens = summary["by_agent"]["solver"]["tokens"]
    assert solver_tokens["prompt_tokens"] == 1500
    assert solver_tokens["completion_tokens"] == 750
    assert solver_tokens["total_tokens"] == 2250

    # Verify verifier tokens (usage2)
    verifier_tokens = summary["by_agent"]["verifier"]["tokens"]
    assert verifier_tokens["prompt_tokens"] == 1500
    assert verifier_tokens["completion_tokens"] == 750
    assert verifier_tokens["total_tokens"] == 2250

    # Verify costs are tracked per agent
    assert summary["by_agent"]["solver"]["cost_usd"] > 0
    assert summary["by_agent"]["verifier"]["cost_usd"] > 0
