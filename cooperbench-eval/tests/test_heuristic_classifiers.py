"""Tests for heuristic-based failure mode classifiers."""

from __future__ import annotations

import pytest

from src.classifiers.base import Severity
from src.classifiers.placeholder_misuse import PlaceholderMisuseClassifier
from src.classifiers.repetition import RepetitionClassifier
from src.classifiers.unresponsiveness import UnresponsivenessClassifier
from src.classifiers.work_overlap import WorkOverlapClassifier
from src.data_loading.schemas import TaskData


class TestWorkOverlapClassifier:
    """Tests for work_overlap classifier."""

    def test_skips_solo_task(self, solo_task: TaskData):
        classifier = WorkOverlapClassifier()
        result = classifier.classify(solo_task)
        assert result.skipped
        assert "Solo" in result.skip_reason

    def test_no_detection_without_patches(self, multi_agent_task: TaskData):
        classifier = WorkOverlapClassifier()
        result = classifier.classify(multi_agent_task)
        assert not result.detected

    def test_detects_file_overlap(self, patch_overlap_task: TaskData):
        classifier = WorkOverlapClassifier()
        result = classifier.classify(patch_overlap_task)
        assert result.detected
        assert result.confidence > 0
        assert len(result.evidence) > 0
        assert "src/solver.py" in result.evidence[0]

    def test_detects_function_overlap(self, patch_overlap_task: TaskData):
        classifier = WorkOverlapClassifier()
        result = classifier.classify(patch_overlap_task)
        assert result.detected
        # Should find function-level overlap on 'solve'
        assert any("solve" in e for e in result.evidence)


class TestRepetitionClassifier:
    """Tests for repetition classifier."""

    def test_skips_solo_task(self, solo_task: TaskData):
        classifier = RepetitionClassifier()
        result = classifier.classify(solo_task)
        assert result.skipped

    def test_no_detection_normal_conversation(self, multi_agent_task: TaskData):
        classifier = RepetitionClassifier()
        result = classifier.classify(multi_agent_task)
        assert not result.detected

    def test_detects_repetitive_messages(self, repetitive_task: TaskData):
        classifier = RepetitionClassifier()
        result = classifier.classify(repetitive_task)
        assert result.detected
        assert result.severity in (Severity.HIGH, Severity.MEDIUM)
        assert result.confidence > 0.5


class TestUnresponsivenessClassifier:
    """Tests for unresponsiveness classifier."""

    def test_skips_solo_task(self, solo_task: TaskData):
        classifier = UnresponsivenessClassifier()
        result = classifier.classify(solo_task)
        assert result.skipped

    def test_detects_unanswered_questions(self, unresponsive_task: TaskData):
        classifier = UnresponsivenessClassifier()
        result = classifier.classify(unresponsive_task)
        assert result.detected
        assert result.severity in (Severity.LOW, Severity.MEDIUM, Severity.HIGH)
        assert len(result.evidence) >= 2

    def test_normal_conversation_no_detection(self, multi_agent_task: TaskData):
        classifier = UnresponsivenessClassifier()
        result = classifier.classify(multi_agent_task)
        # Multi-agent task doesn't have questions
        assert not result.detected


class TestPlaceholderMisuseClassifier:
    """Tests for placeholder_misuse classifier."""

    def test_detects_todo_and_not_implemented(self, placeholder_task: TaskData):
        classifier = PlaceholderMisuseClassifier()
        result = classifier.classify(placeholder_task)
        assert result.detected
        assert result.severity == Severity.HIGH  # NotImplementedError is Tier 1
        assert len(result.evidence) >= 2  # TODO and NotImplementedError

    def test_no_detection_without_patches(self, multi_agent_task: TaskData):
        classifier = PlaceholderMisuseClassifier()
        result = classifier.classify(multi_agent_task)
        assert not result.detected

    def test_classifier_names_are_unique(self):
        classifiers = [
            WorkOverlapClassifier(),
            RepetitionClassifier(),
            UnresponsivenessClassifier(),
            PlaceholderMisuseClassifier(),
        ]
        names = [c.name for c in classifiers]
        assert len(names) == len(set(names)), f"Duplicate classifier names: {names}"
