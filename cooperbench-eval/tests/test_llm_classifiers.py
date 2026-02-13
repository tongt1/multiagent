"""Tests for LLM-based failure mode classifiers.

Tests focus on:
1. Prompt construction (no API call needed)
2. Response parsing with known JSON
3. Skip behavior for solo tasks and missing API key
"""

from __future__ import annotations

import pytest

from src.classifiers.base import ClassificationResult, Severity
from src.data_loading.schemas import TaskData
from src.llm_judge.base import LLMClassifier
from src.llm_judge.broken_commitment import BrokenCommitmentClassifier
from src.llm_judge.dependency_access import DependencyAccessClassifier
from src.llm_judge.divergent_architecture import DivergentArchitectureClassifier
from src.llm_judge.parameter_flow import ParameterFlowClassifier
from src.llm_judge.timing_dependency import TimingDependencyClassifier
from src.llm_judge.unverifiable_claims import UnverifiableClaimsClassifier


ALL_LLM_CLASSIFIER_CLASSES = [
    DivergentArchitectureClassifier,
    UnverifiableClaimsClassifier,
    BrokenCommitmentClassifier,
    DependencyAccessClassifier,
    ParameterFlowClassifier,
    TimingDependencyClassifier,
]


class TestLLMClassifierBase:
    """Tests for LLMClassifier base class."""

    def test_all_llm_classifiers_have_unique_names(self):
        classifiers = [cls(api_key="fake") for cls in ALL_LLM_CLASSIFIER_CLASSES]
        names = [c.name for c in classifiers]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_all_llm_classifiers_have_descriptions(self):
        for cls in ALL_LLM_CLASSIFIER_CLASSES:
            c = cls(api_key="fake")
            assert c.description, f"{cls.__name__} has no description"

    def test_skip_solo_task(self, solo_task: TaskData):
        for cls in ALL_LLM_CLASSIFIER_CLASSES:
            c = cls(api_key="fake")
            result = c.classify(solo_task)
            assert result.skipped, f"{cls.__name__} should skip solo tasks"

    def test_skip_when_no_api_key(self, multi_agent_task: TaskData, monkeypatch):
        monkeypatch.delenv("CO_API_KEY", raising=False)
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        for cls in ALL_LLM_CLASSIFIER_CLASSES:
            c = cls(api_key="")
            result = c.classify(multi_agent_task)
            assert result.skipped, f"{cls.__name__} should skip with no API key"
            assert "CO_API_KEY" in result.skip_reason

    def test_parse_json_response_direct(self):
        text = '{"detected": true, "confidence": 0.8}'
        result = LLMClassifier.parse_json_response(text)
        assert result["detected"] is True
        assert result["confidence"] == 0.8

    def test_parse_json_response_markdown_block(self):
        text = """Here's the analysis:
```json
{"detected": false, "confidence": 0.1}
```"""
        result = LLMClassifier.parse_json_response(text)
        assert result["detected"] is False

    def test_parse_json_response_embedded(self):
        text = 'Based on my analysis: {"detected": true, "confidence": 0.5, "evidence": []} is the result.'
        result = LLMClassifier.parse_json_response(text)
        assert result["detected"] is True


class TestDivergentArchitectureClassifier:
    """Tests for divergent_architecture LLM classifier."""

    def test_prompt_contains_task_info(self, multi_agent_task: TaskData):
        c = DivergentArchitectureClassifier(api_key="fake")
        prompt = c.build_prompt(multi_agent_task)
        assert "DIVERGENT ARCHITECTURE" in prompt
        assert multi_agent_task.task_description in prompt

    def test_parse_detected_response(self, multi_agent_task: TaskData):
        c = DivergentArchitectureClassifier(api_key="fake")
        response = '{"detected": true, "confidence": 0.7, "severity": "medium", "evidence": [{"agents_involved": ["solver_0", "solver_1"], "description": "Different approaches", "agent1_approach": "approximation", "agent2_approach": "exact solution", "step_indices": [0, 1]}], "reasoning": "test"}'
        result = c.parse_response(response, multi_agent_task)
        assert result.detected
        assert result.severity == Severity.MEDIUM
        assert result.confidence == 0.7

    def test_parse_not_detected_response(self, multi_agent_task: TaskData):
        c = DivergentArchitectureClassifier(api_key="fake")
        response = '{"detected": false, "confidence": 0.1, "severity": "low", "evidence": [], "reasoning": "No divergence found"}'
        result = c.parse_response(response, multi_agent_task)
        assert not result.detected


class TestUnverifiableClaimsClassifier:
    """Tests for unverifiable_claims LLM classifier."""

    def test_prompt_contains_task_info(self, multi_agent_task: TaskData):
        c = UnverifiableClaimsClassifier(api_key="fake")
        prompt = c.build_prompt(multi_agent_task)
        assert "UNVERIFIABLE CLAIMS" in prompt

    def test_parse_detected_response(self, multi_agent_task: TaskData):
        c = UnverifiableClaimsClassifier(api_key="fake")
        response = '{"detected": true, "confidence": 0.6, "severity": "low", "evidence": [{"agent": "verifier", "claim": "solution has errors", "why_unverifiable": "no specific error shown", "step_index": 3}], "reasoning": "test"}'
        result = c.parse_response(response, multi_agent_task)
        assert result.detected
        assert len(result.evidence) >= 1


class TestBrokenCommitmentClassifier:
    """Tests for broken_commitment LLM classifier."""

    def test_prompt_contains_task_info(self, multi_agent_task: TaskData):
        c = BrokenCommitmentClassifier(api_key="fake")
        prompt = c.build_prompt(multi_agent_task)
        assert "BROKEN COMMITMENTS" in prompt


class TestDependencyAccessClassifier:
    """Tests for dependency_access LLM classifier."""

    def test_prompt_contains_task_info(self, multi_agent_task: TaskData):
        c = DependencyAccessClassifier(api_key="fake")
        prompt = c.build_prompt(multi_agent_task)
        assert "DEPENDENCY ACCESS" in prompt


class TestParameterFlowClassifier:
    """Tests for parameter_flow LLM classifier."""

    def test_prompt_contains_task_info(self, multi_agent_task: TaskData):
        c = ParameterFlowClassifier(api_key="fake")
        prompt = c.build_prompt(multi_agent_task)
        assert "PARAMETER FLOW" in prompt


class TestTimingDependencyClassifier:
    """Tests for timing_dependency LLM classifier."""

    def test_prompt_contains_task_info(self, multi_agent_task: TaskData):
        c = TimingDependencyClassifier(api_key="fake")
        prompt = c.build_prompt(multi_agent_task)
        assert "TIMING DEPENDENCY" in prompt
