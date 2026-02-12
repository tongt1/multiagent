"""Classifier registry for CooperBench failure mode detection."""

from src.classifiers.base import BaseClassifier, ClassificationResult, Severity
from src.classifiers.placeholder_misuse import PlaceholderMisuseClassifier
from src.classifiers.repetition import RepetitionClassifier
from src.classifiers.unresponsiveness import UnresponsivenessClassifier
from src.classifiers.work_overlap import WorkOverlapClassifier

# Heuristic (non-LLM) classifiers -- order matters for reporting
HEURISTIC_CLASSIFIERS: list[type[BaseClassifier]] = [
    RepetitionClassifier,
    UnresponsivenessClassifier,
    WorkOverlapClassifier,
    PlaceholderMisuseClassifier,
]

# All classifiers including LLM-based (to be added later)
ALL_CLASSIFIERS: list[type[BaseClassifier]] = list(HEURISTIC_CLASSIFIERS)

__all__ = [
    "ALL_CLASSIFIERS",
    "BaseClassifier",
    "ClassificationResult",
    "HEURISTIC_CLASSIFIERS",
    "PlaceholderMisuseClassifier",
    "RepetitionClassifier",
    "Severity",
    "UnresponsivenessClassifier",
    "WorkOverlapClassifier",
]
