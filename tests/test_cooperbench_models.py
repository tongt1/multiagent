"""Unit tests for CooperBench Pydantic models.

Tests serialization/deserialization, validation, defaults,
and edge cases for all CooperBench data models.
"""

import pytest
from pydantic import ValidationError

from src.evaluation.cooperbench.models import (
    CooperBenchConfig,
    CooperBenchEvalResult,
    CooperBenchJudgment,
    CooperBenchPipelineResult,
    CooperBenchProblem,
    CooperBenchResponse,
    CooperBenchVerification,
    FeatureResult,
    FeatureSpec,
)


# --- CooperBenchConfig Tests ---


class TestCooperBenchConfig:
    """Tests for CooperBenchConfig model."""

    def test_defaults(self) -> None:
        """Config should have sensible defaults."""
        config = CooperBenchConfig()
        assert config.dataset_path == "dataset/"
        assert config.subset == "lite"
        assert config.solver_model == "command-r-plus"
        assert config.verifier_model == "command-r-plus"
        assert config.temperature == 0.2
        assert config.max_tokens == 8192
        assert config.backend == "docker"
        assert config.timeout == 600
        assert config.mode == "coop"
        assert config.max_rounds == 3
        assert config.messaging_enabled is True
        assert config.git_enabled is False
        assert config.max_parallel_tasks == 4
        assert config.repo_filter is None
        assert config.task_filter is None

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = CooperBenchConfig(
            dataset_path="/data/cooperbench",
            subset="flash",
            solver_model="gpt-5",
            verifier_model="claude-sonnet-4.5",
            temperature=0.7,
            max_tokens=16384,
            backend="modal",
            timeout=1200,
            mode="solo",
            max_rounds=5,
            messaging_enabled=False,
            git_enabled=True,
            max_parallel_tasks=8,
            repo_filter=["llama_index", "dspy"],
        )
        assert config.dataset_path == "/data/cooperbench"
        assert config.subset == "flash"
        assert config.solver_model == "gpt-5"
        assert config.verifier_model == "claude-sonnet-4.5"
        assert config.temperature == 0.7
        assert config.max_tokens == 16384
        assert config.backend == "modal"
        assert config.mode == "solo"
        assert config.max_rounds == 5
        assert config.repo_filter == ["llama_index", "dspy"]

    def test_temperature_validation(self) -> None:
        """Temperature must be between 0.0 and 2.0."""
        with pytest.raises(ValidationError):
            CooperBenchConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            CooperBenchConfig(temperature=2.1)

    def test_max_tokens_validation(self) -> None:
        """Max tokens must be within bounds."""
        with pytest.raises(ValidationError):
            CooperBenchConfig(max_tokens=100)  # Below 256
        with pytest.raises(ValidationError):
            CooperBenchConfig(max_tokens=65536)  # Above 32768

    def test_max_rounds_validation(self) -> None:
        """Max rounds must be between 1 and 10."""
        with pytest.raises(ValidationError):
            CooperBenchConfig(max_rounds=0)
        with pytest.raises(ValidationError):
            CooperBenchConfig(max_rounds=11)

    def test_timeout_validation(self) -> None:
        """Timeout must be between 60 and 3600."""
        with pytest.raises(ValidationError):
            CooperBenchConfig(timeout=30)
        with pytest.raises(ValidationError):
            CooperBenchConfig(timeout=7200)

    def test_invalid_mode(self) -> None:
        """Mode must be 'coop' or 'solo'."""
        with pytest.raises(ValidationError):
            CooperBenchConfig(mode="invalid")

    def test_invalid_backend(self) -> None:
        """Backend must be 'docker', 'modal', or 'gcp'."""
        with pytest.raises(ValidationError):
            CooperBenchConfig(backend="kubernetes")

    def test_invalid_subset(self) -> None:
        """Subset must be 'lite', 'flash', or None."""
        with pytest.raises(ValidationError):
            CooperBenchConfig(subset="full")

    def test_serialization_roundtrip(self) -> None:
        """Config should survive JSON serialization roundtrip."""
        config = CooperBenchConfig(
            solver_model="gpt-5",
            mode="solo",
            max_rounds=5,
        )
        json_data = config.model_dump()
        restored = CooperBenchConfig(**json_data)
        assert restored == config

    def test_extra_fields_ignored(self) -> None:
        """Extra fields should be silently ignored."""
        config = CooperBenchConfig(
            solver_model="gpt-5",
            unknown_field="should_be_ignored",
        )
        assert config.solver_model == "gpt-5"


# --- FeatureSpec Tests ---


class TestFeatureSpec:
    """Tests for FeatureSpec model."""

    def test_basic(self) -> None:
        """FeatureSpec should hold feature metadata."""
        spec = FeatureSpec(
            feature_id=2,
            description="Add pagination support to list endpoint",
        )
        assert spec.feature_id == 2
        assert "pagination" in spec.description
        assert spec.patch_path is None
        assert spec.tests_patch_path is None

    def test_with_paths(self) -> None:
        """FeatureSpec should store patch paths."""
        spec = FeatureSpec(
            feature_id=3,
            description="Add caching",
            patch_path="/data/repo/task1/feature3/feature.patch",
            tests_patch_path="/data/repo/task1/feature3/tests.patch",
        )
        assert spec.patch_path is not None
        assert spec.tests_patch_path is not None


# --- CooperBenchProblem Tests ---


class TestCooperBenchProblem:
    """Tests for CooperBenchProblem model."""

    def _make_problem(self, **kwargs) -> CooperBenchProblem:
        """Helper to create a problem with defaults."""
        defaults = {
            "repo": "llama_index",
            "task_id": "task17244",
            "features": [2, 6],
            "feature_specs": {
                2: FeatureSpec(feature_id=2, description="Feature 2 spec"),
                6: FeatureSpec(feature_id=6, description="Feature 6 spec"),
            },
            "dataset_path": "/data/cooperbench",
        }
        defaults.update(kwargs)
        return CooperBenchProblem(**defaults)

    def test_basic_creation(self) -> None:
        """Problem should be created with required fields."""
        problem = self._make_problem()
        assert problem.repo == "llama_index"
        assert problem.task_id == "task17244"
        assert problem.features == [2, 6]
        assert len(problem.feature_specs) == 2

    def test_task_dir_property(self) -> None:
        """task_dir should combine dataset_path, repo, and task_id."""
        problem = self._make_problem()
        assert problem.task_dir == "/data/cooperbench/llama_index/task17244"

    def test_feature_pair_key(self) -> None:
        """feature_pair_key should be unique and sorted."""
        problem = self._make_problem(features=[6, 2])
        assert problem.feature_pair_key == "llama_index/task17244/features_2_6"

    def test_optional_fields_default_none(self) -> None:
        """Optional fields should default to None."""
        problem = self._make_problem()
        assert problem.setup_script is None
        assert problem.run_tests_script is None
        assert problem.runner_script is None
        assert problem.dockerfile is None
        assert problem.combined_patch is None
        assert problem.image_name is None

    def test_metadata_default_empty(self) -> None:
        """Metadata should default to empty dict."""
        problem = self._make_problem()
        assert problem.metadata == {}

    def test_serialization_roundtrip(self) -> None:
        """Problem should survive JSON roundtrip."""
        problem = self._make_problem(
            setup_script="/data/setup.sh",
            image_name="cooperbench-llama_index-task17244",
        )
        data = problem.model_dump()
        restored = CooperBenchProblem(**data)
        assert restored.repo == problem.repo
        assert restored.task_id == problem.task_id
        assert restored.features == problem.features
        assert restored.setup_script == problem.setup_script


# --- CooperBenchResponse Tests ---


class TestCooperBenchResponse:
    """Tests for CooperBenchResponse model."""

    def test_basic(self) -> None:
        """Response should hold patch and metadata."""
        response = CooperBenchResponse(
            patch="diff --git a/file.py b/file.py\n+new line",
            reasoning="Added new line for feature",
            confidence=0.85,
        )
        assert "diff --git" in response.patch
        assert response.confidence == 0.85
        assert response.messages_to_partner == []
        assert response.approach_summary == ""

    def test_with_messages(self) -> None:
        """Response should carry partner messages."""
        response = CooperBenchResponse(
            patch="patch content",
            reasoning="reasoning",
            confidence=0.7,
            messages_to_partner=["I'll modify file_a.py", "Please avoid changing utils.py"],
            approach_summary="Modifying file_a.py to add caching",
        )
        assert len(response.messages_to_partner) == 2
        assert response.approach_summary == "Modifying file_a.py to add caching"

    def test_confidence_bounds(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            CooperBenchResponse(
                patch="p", reasoning="r", confidence=1.5,
            )
        with pytest.raises(ValidationError):
            CooperBenchResponse(
                patch="p", reasoning="r", confidence=-0.1,
            )


# --- CooperBenchVerification Tests ---


class TestCooperBenchVerification:
    """Tests for CooperBenchVerification model."""

    def test_no_conflicts(self) -> None:
        """Verification with no conflicts."""
        v = CooperBenchVerification(has_conflicts=False)
        assert not v.has_conflicts
        assert v.conflict_description == ""
        assert v.suggestions == []
        assert v.confidence == 0.5

    def test_with_conflicts(self) -> None:
        """Verification detecting conflicts."""
        v = CooperBenchVerification(
            has_conflicts=True,
            conflict_description="Both agents modify utils.py:45",
            suggestions=["Move helper to a new file", "Use different function names"],
            confidence=0.9,
        )
        assert v.has_conflicts
        assert "utils.py" in v.conflict_description
        assert len(v.suggestions) == 2


# --- FeatureResult Tests ---


class TestFeatureResult:
    """Tests for FeatureResult model."""

    def test_passed(self) -> None:
        """Passing feature result."""
        fr = FeatureResult(feature_id=2, passed=True, test_output="OK")
        assert fr.passed
        assert fr.error is None

    def test_failed(self) -> None:
        """Failing feature result."""
        fr = FeatureResult(
            feature_id=3,
            passed=False,
            test_output="FAILED test_foo",
            error="AssertionError in test_foo",
        )
        assert not fr.passed
        assert "FAILED" in fr.test_output


# --- CooperBenchJudgment Tests ---


class TestCooperBenchJudgment:
    """Tests for CooperBenchJudgment model."""

    def test_both_passed(self) -> None:
        """Judgment where both features pass."""
        j = CooperBenchJudgment(
            both_passed=True,
            feature_results=[
                FeatureResult(feature_id=2, passed=True),
                FeatureResult(feature_id=6, passed=True),
            ],
            merge_status="clean",
            merge_strategy="naive",
        )
        assert j.both_passed
        assert len(j.feature_results) == 2

    def test_one_failed(self) -> None:
        """Judgment where one feature fails."""
        j = CooperBenchJudgment(
            both_passed=False,
            feature_results=[
                FeatureResult(feature_id=2, passed=True),
                FeatureResult(feature_id=6, passed=False, error="test failed"),
            ],
        )
        assert not j.both_passed

    def test_defaults(self) -> None:
        """Judgment defaults."""
        j = CooperBenchJudgment(both_passed=False)
        assert j.merge_status == "clean"
        assert j.merge_strategy == "naive"
        assert j.test_output == ""
        assert j.feature_results == []


# --- CooperBenchEvalResult Tests ---


class TestCooperBenchEvalResult:
    """Tests for CooperBenchEvalResult model."""

    def test_coop_result(self) -> None:
        """Coop evaluation result."""
        result = CooperBenchEvalResult(
            repo="llama_index",
            task_id="task17244",
            features=[2, 6],
            mode="coop",
            both_passed=True,
            feature_results=[
                FeatureResult(feature_id=2, passed=True),
                FeatureResult(feature_id=6, passed=True),
            ],
            patches=["patch1", "patch2"],
            merged_patch="merged",
            execution_time=45.2,
        )
        assert result.mode == "coop"
        assert result.both_passed
        assert len(result.patches) == 2
        assert result.merged_patch == "merged"

    def test_solo_result(self) -> None:
        """Solo evaluation result."""
        result = CooperBenchEvalResult(
            repo="dspy",
            task_id="task100",
            features=[1, 2],
            mode="solo",
            both_passed=False,
            patches=["single_patch"],
            execution_time=30.0,
        )
        assert result.mode == "solo"
        assert not result.both_passed
        assert len(result.patches) == 1
        assert result.merged_patch is None

    def test_failed_result(self) -> None:
        """Result with error."""
        result = CooperBenchEvalResult(
            repo="pillow",
            task_id="task50",
            features=[1, 3],
            mode="coop",
            both_passed=False,
            error="Docker build failed",
        )
        assert not result.both_passed
        assert result.error == "Docker build failed"

    def test_serialization_roundtrip(self) -> None:
        """EvalResult should survive JSON roundtrip."""
        result = CooperBenchEvalResult(
            repo="llama_index",
            task_id="task17244",
            features=[2, 6],
            mode="coop",
            both_passed=True,
            merge_status="clean",
            execution_time=10.0,
        )
        data = result.model_dump()
        restored = CooperBenchEvalResult(**data)
        assert restored.repo == result.repo
        assert restored.both_passed == result.both_passed


# --- CooperBenchPipelineResult Tests ---


class TestCooperBenchPipelineResult:
    """Tests for CooperBenchPipelineResult model."""

    def _make_problem(self) -> CooperBenchProblem:
        """Helper to create a test problem."""
        return CooperBenchProblem(
            repo="llama_index",
            task_id="task17244",
            features=[2, 6],
            feature_specs={
                2: FeatureSpec(feature_id=2, description="Feature 2"),
                6: FeatureSpec(feature_id=6, description="Feature 6"),
            },
            dataset_path="/data",
        )

    def test_successful_result(self) -> None:
        """Pipeline result for successful run."""
        problem = self._make_problem()
        eval_result = CooperBenchEvalResult(
            repo="llama_index",
            task_id="task17244",
            features=[2, 6],
            mode="coop",
            both_passed=True,
        )
        result = CooperBenchPipelineResult(
            problem=problem,
            eval_result=eval_result,
            reward=1.0,
            patches=["patch1", "patch2"],
            rounds_completed=3,
            messages_exchanged=4,
            total_cost=0.05,
            total_tokens=5000,
            solver_tokens=2500,
            verifier_tokens=2500,
            wall_time=120.0,
        )
        assert result.reward == 1.0
        assert result.rounds_completed == 3
        assert result.total_cost == 0.05
        assert result.error is None

    def test_failed_result(self) -> None:
        """Pipeline result for failed run."""
        problem = self._make_problem()
        result = CooperBenchPipelineResult(
            problem=problem,
            error="LLM API timeout",
        )
        assert result.reward == 0.0
        assert result.eval_result is None
        assert result.error == "LLM API timeout"

    def test_defaults(self) -> None:
        """Default values for pipeline result."""
        problem = self._make_problem()
        result = CooperBenchPipelineResult(problem=problem)
        assert result.reward == 0.0
        assert result.patches == []
        assert result.approach_summaries == []
        assert result.rounds_completed == 0
        assert result.messages_exchanged == 0
        assert result.total_cost == 0.0
        assert result.total_tokens == 0
        assert result.trajectory == []
        assert result.wall_time == 0.0
