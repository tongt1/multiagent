"""Base classifier interface for CooperBench failure mode detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.data_loading.schemas import TaskData


class Severity(str, Enum):
    """Severity levels for detected failure modes."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ClassificationResult:
    """Result of running a classifier on a task.

    Attributes:
        classifier_name: Name of the classifier that produced this result.
        detected: Whether the failure mode was detected.
        severity: Severity level if detected.
        confidence: Confidence score in [0.0, 1.0].
        evidence: Human-readable evidence supporting the detection.
        details: Structured details for programmatic consumption.
        skipped: Whether classification was skipped (e.g., solo task).
        skip_reason: Reason for skipping, if applicable.
    """

    classifier_name: str
    detected: bool = False
    severity: Severity = Severity.LOW
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def summary(self) -> str:
        """One-line summary of the result."""
        if self.skipped:
            return f"[{self.classifier_name}] SKIPPED: {self.skip_reason}"
        if self.detected:
            return f"[{self.classifier_name}] DETECTED ({self.severity.value}, conf={self.confidence:.2f})"
        return f"[{self.classifier_name}] not detected"


class BaseClassifier(ABC):
    """Abstract base class for all failure mode classifiers.

    Subclasses must implement `classify()` which receives a TaskData object
    and returns a ClassificationResult.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this classifier."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this classifier detects."""
        return ""

    @abstractmethod
    def classify(self, task: TaskData) -> ClassificationResult:
        """Run classification on a task.

        Args:
            task: Complete task data including messages, patches, and eval results.

        Returns:
            ClassificationResult with detection outcome and evidence.
        """
        ...

    def _no_detection(self) -> ClassificationResult:
        """Create a result indicating no failure mode detected."""
        return ClassificationResult(
            classifier_name=self.name,
            detected=False,
            severity=Severity.LOW,
            confidence=0.0,
        )

    def _skip_if_solo(self, task: TaskData) -> ClassificationResult | None:
        """Return a skip result if the task has only one agent.

        Multi-agent failure modes are not applicable to solo tasks.
        Returns None if the task is multi-agent (i.e., should NOT be skipped).
        """
        if task.is_solo:
            return ClassificationResult(
                classifier_name=self.name,
                detected=False,
                skipped=True,
                skip_reason="Solo task -- multi-agent failure mode not applicable",
            )
        return None
