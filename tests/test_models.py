"""Tests for data models."""

import json

import pytest
from pydantic import ValidationError

from src.models.config import AgentConfig, JudgeConfig, PipelineConfig
from src.models.evaluation import Judgment, VerificationResult
from src.models.trajectory import TrajectoryEntry


def test_agent_config_role_validation():
    """Test AgentConfig validates role constraint (solver/verifier only)."""
    # Valid roles
    config1 = AgentConfig(
        role="solver",
        model="command-r-plus",
        prompt_template="Solve: {problem}",
    )
    assert config1.role == "solver"

    config2 = AgentConfig(
        role="verifier",
        model="command-r-plus",
        prompt_template="Verify: {solution}",
    )
    assert config2.role == "verifier"

    # Invalid role
    with pytest.raises(ValidationError):
        AgentConfig(
            role="judge",  # type: ignore
            model="command-r-plus",
            prompt_template="Invalid",
        )


def test_judge_config_accepts_scoring_rubric():
    """Test JudgeConfig accepts valid scoring rubric."""
    rubric = "Score 0-1: correctness, clarity, completeness"
    config = JudgeConfig(
        model="command-r-plus",
        prompt_template="Judge: {solution}",
        scoring_rubric=rubric,
    )
    assert config.scoring_rubric == rubric


def test_pipeline_config_hash_deterministic():
    """Test PipelineConfig.config_hash() is deterministic."""
    solver = AgentConfig(
        role="solver",
        model="command-r-plus",
        temperature=0.0,
        max_tokens=4096,
        prompt_template="Solve: {problem}",
    )
    verifier = AgentConfig(
        role="verifier",
        model="command-r-plus",
        temperature=0.0,
        max_tokens=4096,
        prompt_template="Verify: {solution}",
    )
    judge = JudgeConfig(
        model="command-r-plus",
        temperature=0.0,
        max_tokens=4096,
        prompt_template="Judge: {solution}",
        scoring_rubric="Score 0-1",
    )

    config1 = PipelineConfig(
        solver=solver,
        verifier=verifier,
        judge=judge,
        max_iterations=7,
        config_version="1.0",
    )
    config2 = PipelineConfig(
        solver=solver,
        verifier=verifier,
        judge=judge,
        max_iterations=7,
        config_version="1.0",
    )

    hash1 = config1.config_hash()
    hash2 = config2.config_hash()

    assert hash1 == hash2
    assert len(hash1) == 8


def test_pipeline_config_hash_changes_on_modification():
    """Test PipelineConfig.config_hash() changes when config changes."""
    solver = AgentConfig(
        role="solver",
        model="command-r-plus",
        temperature=0.0,
        max_tokens=4096,
        prompt_template="Solve: {problem}",
    )
    verifier = AgentConfig(
        role="verifier",
        model="command-r-plus",
        temperature=0.0,
        max_tokens=4096,
        prompt_template="Verify: {solution}",
    )
    judge = JudgeConfig(
        model="command-r-plus",
        temperature=0.0,
        max_tokens=4096,
        prompt_template="Judge: {solution}",
        scoring_rubric="Score 0-1",
    )

    config1 = PipelineConfig(
        solver=solver,
        verifier=verifier,
        judge=judge,
        max_iterations=7,
        config_version="1.0",
    )
    config2 = PipelineConfig(
        solver=solver,
        verifier=verifier,
        judge=judge,
        max_iterations=10,  # Different
        config_version="1.0",
    )

    hash1 = config1.config_hash()
    hash2 = config2.config_hash()

    assert hash1 != hash2


def test_trajectory_entry_serializes_to_json():
    """Test TrajectoryEntry serializes to valid JSON."""
    entry = TrajectoryEntry(
        timestamp="2026-01-01T00:00:00Z",
        run_id="run-123",
        step_id=1,
        agent="solver",
        action="generate_solution",
        input={"problem": "What is 2+2?"},
        output={"solution": "4"},
        metadata={
            "model_version": "1.0",
            "prompt_version": "1.0",
            "config_hash": "abc12345",
            "tokens": 100,
            "cost_usd": 0.001,
            "iteration": 1,
        },
    )

    json_str = entry.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed["run_id"] == "run-123"
    assert parsed["step_id"] == 1
    assert parsed["agent"] == "solver"
    assert parsed["metadata"]["config_hash"] == "abc12345"


def test_judgment_score_validation():
    """Test Judgment score validation (rejects less than 0 and greater than 1)."""
    # Valid scores
    judgment1 = Judgment(
        score=0.0, reasoning="Poor", strengths=[], weaknesses=["Too short"]
    )
    assert judgment1.score == 0.0

    judgment2 = Judgment(
        score=1.0, reasoning="Perfect", strengths=["Complete"], weaknesses=[]
    )
    assert judgment2.score == 1.0

    judgment3 = Judgment(
        score=0.75, reasoning="Good", strengths=["Clear"], weaknesses=["Minor issues"]
    )
    assert judgment3.score == 0.75

    # Invalid scores
    with pytest.raises(ValidationError):
        Judgment(score=-0.1, reasoning="Invalid", strengths=[], weaknesses=[])

    with pytest.raises(ValidationError):
        Judgment(score=1.1, reasoning="Invalid", strengths=[], weaknesses=[])


def test_verification_result_defaults():
    """Test VerificationResult defaults."""
    result = VerificationResult(passed=True)

    assert result.passed is True
    assert result.critique is None
    assert result.scores == {}
    assert result.confidence == 0.5
