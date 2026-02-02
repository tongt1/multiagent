"""Unit tests for batch execution and dataset loading."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.dataset_loader import DatasetLoader, Problem
from src.models.config import PipelineConfig
from src.models.trajectory import TokenUsage
from src.orchestration.batch_executor import BatchPipelineExecutor, BatchResult
from src.orchestration.pipeline import PipelineResult


class TestProblemModel:
    """Test Problem model creation and serialization."""

    def test_problem_creation(self) -> None:
        """Test creating a Problem instance."""
        problem = Problem(
            id="test_1",
            problem="What is 2+2?",
            ground_truth="4",
            domain="math",
        )

        assert problem.id == "test_1"
        assert problem.problem == "What is 2+2?"
        assert problem.ground_truth == "4"
        assert problem.domain == "math"
        assert problem.metadata == {}

    def test_problem_with_metadata(self) -> None:
        """Test Problem with metadata."""
        problem = Problem(
            id="test_2",
            problem="Test problem",
            metadata={"difficulty": "easy", "source": "test"},
            domain="general",
        )

        assert problem.metadata["difficulty"] == "easy"
        assert problem.metadata["source"] == "test"

    def test_problem_serialization(self) -> None:
        """Test Problem JSON serialization."""
        problem = Problem(
            id="test_3",
            problem="Test problem",
            domain="code",
        )

        json_str = problem.model_dump_json()
        assert "test_3" in json_str
        assert "Test problem" in json_str
        assert "code" in json_str


class TestDatasetLoader:
    """Test DatasetLoader with local files."""

    @pytest.fixture
    def temp_yaml_dataset(self, tmp_path: Path) -> Path:
        """Create a temporary YAML dataset file."""
        dataset_file = tmp_path / "test_dataset.yaml"
        dataset_file.write_text(
            """
- id: prob_1
  problem: "Solve x + 1 = 3"
  ground_truth: "x = 2"
  domain: math

- id: prob_2
  problem: "Write a function that adds two numbers"
  domain: code
  metadata:
    difficulty: easy

- problem: "What is the capital of France?"
  domain: general
"""
        )
        return dataset_file

    @pytest.fixture
    def temp_json_dataset(self, tmp_path: Path) -> Path:
        """Create a temporary JSON dataset file."""
        dataset_file = tmp_path / "test_dataset.json"
        dataset_file.write_text(
            """
[
  {
    "id": "json_1",
    "problem": "Test problem 1",
    "ground_truth": "answer 1",
    "domain": "math"
  },
  {
    "problem": "Test problem 2",
    "domain": "code"
  }
]
"""
        )
        return dataset_file

    def test_load_local_yaml(self, temp_yaml_dataset: Path) -> None:
        """Test loading local YAML dataset."""
        loader = DatasetLoader()
        problems = loader.load(str(temp_yaml_dataset))

        assert len(problems) == 3

        # Check first problem
        assert problems[0].id == "prob_1"
        assert problems[0].problem == "Solve x + 1 = 3"
        assert problems[0].ground_truth == "x = 2"
        assert problems[0].domain == "math"

        # Check second problem
        assert problems[1].id == "prob_2"
        assert problems[1].metadata["difficulty"] == "easy"

        # Check third problem (auto-generated ID)
        assert problems[2].id == "local_2"
        assert problems[2].domain == "general"

    def test_load_local_json(self, temp_json_dataset: Path) -> None:
        """Test loading local JSON dataset."""
        loader = DatasetLoader()
        problems = loader.load(str(temp_json_dataset))

        assert len(problems) == 2
        assert problems[0].id == "json_1"
        assert problems[1].id == "local_1"  # Auto-generated

    def test_load_nonexistent_file(self) -> None:
        """Test loading from nonexistent file raises error."""
        loader = DatasetLoader()

        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.yaml")

    def test_load_invalid_format(self, tmp_path: Path) -> None:
        """Test loading unsupported file format raises error."""
        bad_file = tmp_path / "dataset.txt"
        bad_file.write_text("not a dataset")

        loader = DatasetLoader()

        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(str(bad_file))


@pytest.mark.asyncio
class TestBatchPipelineExecutor:
    """Test BatchPipelineExecutor with mocked pipeline."""

    @pytest.fixture
    def mock_config(self) -> PipelineConfig:
        """Create a mock pipeline configuration."""
        return MagicMock(spec=PipelineConfig)

    @pytest.fixture
    def sample_problems(self) -> list[Problem]:
        """Create sample problems for testing."""
        return [
            Problem(id="1", problem="Problem 1", domain="math"),
            Problem(id="2", problem="Problem 2", domain="code"),
            Problem(id="3", problem="Problem 3", domain="general"),
        ]

    def create_mock_result(
        self, problem_id: str, score: float = 0.8
    ) -> PipelineResult:
        """Create a mock PipelineResult."""
        return PipelineResult(
            problem_description=f"Problem {problem_id}",
            solution=f"Solution {problem_id}",
            passed_verification=True,
            judge_score=score,
            iterations=2,
            total_cost=0.001,
            trajectory_path=f"/tmp/traj_{problem_id}.jsonl",
            token_usage=TokenUsage(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            ),
            cost_summary={},
        )

    async def test_run_one(
        self, mock_config: PipelineConfig, sample_problems: list[Problem]
    ) -> None:
        """Test running a single problem."""
        executor = BatchPipelineExecutor(mock_config, max_concurrent=5)
        problem = sample_problems[0]

        # Mock SolverVerifierJudgePipeline
        with patch(
            "src.orchestration.batch_executor.SolverVerifierJudgePipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.run = AsyncMock(
                return_value=self.create_mock_result("1")
            )
            mock_pipeline_class.return_value = mock_pipeline

            result = await executor.run_one(problem)

            assert isinstance(result, PipelineResult)
            assert result.judge_score == 0.8
            assert result.passed_verification is True

            # Verify pipeline was created with config
            mock_pipeline_class.assert_called_once_with(mock_config)

            # Verify run was called with problem data
            mock_pipeline.run.assert_called_once()
            call_args = mock_pipeline.run.call_args
            assert call_args[1]["problem_description"] == "Problem 1"
            assert call_args[1]["problem_metadata"]["problem_id"] == "1"

    async def test_run_batch_all_succeed(
        self, mock_config: PipelineConfig, sample_problems: list[Problem]
    ) -> None:
        """Test batch execution where all problems succeed."""
        executor = BatchPipelineExecutor(mock_config, max_concurrent=2)

        # Mock pipeline to return different scores
        with patch(
            "src.orchestration.batch_executor.SolverVerifierJudgePipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()

            # Return different results for each problem
            scores = [0.9, 0.7, 0.5]
            results = [self.create_mock_result(str(i + 1), score) for i, score in enumerate(scores)]

            mock_pipeline.run = AsyncMock(side_effect=results)
            mock_pipeline_class.return_value = mock_pipeline

            batch_result = await executor.run_batch(sample_problems)

            assert batch_result.total == 3
            assert batch_result.succeeded == 3
            assert batch_result.failed == 0
            assert len(batch_result.results) == 3
            assert len(batch_result.errors) == 0
            assert batch_result.elapsed_seconds > 0

    async def test_run_batch_partial_failure(
        self, mock_config: PipelineConfig, sample_problems: list[Problem]
    ) -> None:
        """Test batch execution with some failures (partial results)."""
        executor = BatchPipelineExecutor(mock_config, max_concurrent=2)

        with patch(
            "src.orchestration.batch_executor.SolverVerifierJudgePipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()

            # First succeeds, second fails, third succeeds
            mock_pipeline.run = AsyncMock(
                side_effect=[
                    self.create_mock_result("1", 0.9),
                    ValueError("Pipeline error"),
                    self.create_mock_result("3", 0.6),
                ]
            )
            mock_pipeline_class.return_value = mock_pipeline

            batch_result = await executor.run_batch(sample_problems)

            # Verify partial results (NOT all-or-nothing)
            assert batch_result.total == 3
            assert batch_result.succeeded == 2
            assert batch_result.failed == 1
            assert len(batch_result.results) == 2
            assert len(batch_result.errors) == 1

            # Check error details
            error = batch_result.errors[0]
            assert error["problem_id"] == "2"
            assert error["error_type"] == "ValueError"
            assert "Pipeline error" in error["error_message"]

    async def test_run_batch_with_callback(
        self, mock_config: PipelineConfig, sample_problems: list[Problem]
    ) -> None:
        """Test batch execution with on_complete callback."""
        executor = BatchPipelineExecutor(mock_config, max_concurrent=2)

        completed_problems: list[str] = []

        def on_complete(problem: Problem, result: Any) -> None:
            completed_problems.append(problem.id)

        with patch(
            "src.orchestration.batch_executor.SolverVerifierJudgePipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.run = AsyncMock(
                return_value=self.create_mock_result("test")
            )
            mock_pipeline_class.return_value = mock_pipeline

            await executor.run_batch(sample_problems, on_complete=on_complete)

            # Verify callback was called for all problems
            assert len(completed_problems) == 3
            assert set(completed_problems) == {"1", "2", "3"}

    async def test_semaphore_limits_concurrency(
        self, mock_config: PipelineConfig
    ) -> None:
        """Test that semaphore correctly limits concurrency."""
        max_concurrent = 2
        executor = BatchPipelineExecutor(mock_config, max_concurrent=max_concurrent)

        # Create 5 problems
        problems = [Problem(id=str(i), problem=f"Problem {i}", domain="math") for i in range(5)]

        concurrent_count = 0
        max_concurrent_observed = 0

        async def mock_run(*args: Any, **kwargs: Any) -> PipelineResult:
            nonlocal concurrent_count, max_concurrent_observed

            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)

            # Simulate work
            await asyncio.sleep(0.1)

            concurrent_count -= 1

            return self.create_mock_result("test")

        with patch(
            "src.orchestration.batch_executor.SolverVerifierJudgePipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.run = mock_run
            mock_pipeline_class.return_value = mock_pipeline

            await executor.run_batch(problems)

            # Verify concurrency was limited
            assert max_concurrent_observed <= max_concurrent


class TestBatchResultAggregation:
    """Test BatchResult aggregation logic."""

    def test_batch_result_counts(self) -> None:
        """Test that succeeded + failed = total."""
        result = BatchResult(
            total=10,
            succeeded=7,
            failed=3,
            results=[],
            errors=[],
            elapsed_seconds=5.0,
        )

        assert result.total == result.succeeded + result.failed
