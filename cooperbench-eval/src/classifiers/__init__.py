"""Classifier registry for CooperBench failure mode detection.

Provides both heuristic (fast, no API) and LLM-based (semantic, uses
Cohere API) classifiers. ALL_CLASSIFIERS contains all 10 classifiers
matching the CooperBench paper's failure mode taxonomy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.classifiers.base import BaseClassifier, ClassificationResult, Severity
from src.classifiers.placeholder_misuse import PlaceholderMisuseClassifier
from src.classifiers.repetition import RepetitionClassifier
from src.classifiers.unresponsiveness import UnresponsivenessClassifier
from src.classifiers.work_overlap import WorkOverlapClassifier

# Heuristic (non-LLM) classifiers -- fast, no API key needed
HEURISTIC_CLASSIFIERS: list[type[BaseClassifier]] = [
    WorkOverlapClassifier,            # 33.2%
    RepetitionClassifier,             # 14.7%
    UnresponsivenessClassifier,       # 8.7%
    PlaceholderMisuseClassifier,      # 1.5%
]

# CooperBench paper baseline prevalence rates (Figure 6)
BASELINE_RATES = {
    "work_overlap": 33.2,
    "divergent_architecture": 29.7,
    "repetition": 14.7,
    "unresponsiveness": 8.7,
    "unverifiable_claims": 4.3,
    "broken_commitment": 3.7,
    "dependency_access": 1.7,
    "placeholder_misuse": 1.5,
    "parameter_flow": 1.3,
    "timing_dependency": 1.1,
}


def get_all_classifiers() -> list[type[BaseClassifier]]:
    """Get all 10 classifiers (heuristic + LLM). Lazy import to avoid circular deps."""
    from src.llm_judge.broken_commitment import BrokenCommitmentClassifier
    from src.llm_judge.dependency_access import DependencyAccessClassifier
    from src.llm_judge.divergent_architecture import DivergentArchitectureClassifier
    from src.llm_judge.parameter_flow import ParameterFlowClassifier
    from src.llm_judge.timing_dependency import TimingDependencyClassifier
    from src.llm_judge.unverifiable_claims import UnverifiableClaimsClassifier

    return [
        WorkOverlapClassifier,            # 33.2% -- heuristic
        DivergentArchitectureClassifier,  # 29.7% -- LLM
        RepetitionClassifier,             # 14.7% -- heuristic
        UnresponsivenessClassifier,       # 8.7%  -- heuristic
        UnverifiableClaimsClassifier,     # 4.3%  -- LLM
        BrokenCommitmentClassifier,       # 3.7%  -- LLM
        DependencyAccessClassifier,       # 1.7%  -- LLM
        PlaceholderMisuseClassifier,      # 1.5%  -- heuristic
        ParameterFlowClassifier,          # 1.3%  -- LLM
        TimingDependencyClassifier,       # 1.1%  -- LLM
    ]


def get_llm_classifiers() -> list[type[BaseClassifier]]:
    """Get LLM-based classifiers only. Lazy import to avoid circular deps."""
    from src.llm_judge.broken_commitment import BrokenCommitmentClassifier
    from src.llm_judge.dependency_access import DependencyAccessClassifier
    from src.llm_judge.divergent_architecture import DivergentArchitectureClassifier
    from src.llm_judge.parameter_flow import ParameterFlowClassifier
    from src.llm_judge.timing_dependency import TimingDependencyClassifier
    from src.llm_judge.unverifiable_claims import UnverifiableClaimsClassifier

    return [
        DivergentArchitectureClassifier,  # 29.7%
        UnverifiableClaimsClassifier,     # 4.3%
        BrokenCommitmentClassifier,       # 3.7%
        DependencyAccessClassifier,       # 1.7%
        ParameterFlowClassifier,          # 1.3%
        TimingDependencyClassifier,       # 1.1%
    ]


# Backward-compatible module-level lists (lazy-populated on first access)
_ALL_CLASSIFIERS: list[type[BaseClassifier]] | None = None
_LLM_CLASSIFIERS: list[type[BaseClassifier]] | None = None


def __getattr__(name: str):
    """Lazy attribute access for ALL_CLASSIFIERS and LLM_CLASSIFIERS."""
    global _ALL_CLASSIFIERS, _LLM_CLASSIFIERS
    if name == "ALL_CLASSIFIERS":
        if _ALL_CLASSIFIERS is None:
            _ALL_CLASSIFIERS = get_all_classifiers()
        return _ALL_CLASSIFIERS
    if name == "LLM_CLASSIFIERS":
        if _LLM_CLASSIFIERS is None:
            _LLM_CLASSIFIERS = get_llm_classifiers()
        return _LLM_CLASSIFIERS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALL_CLASSIFIERS",
    "BASELINE_RATES",
    "BaseClassifier",
    "ClassificationResult",
    "HEURISTIC_CLASSIFIERS",
    "LLM_CLASSIFIERS",
    "PlaceholderMisuseClassifier",
    "RepetitionClassifier",
    "Severity",
    "UnresponsivenessClassifier",
    "WorkOverlapClassifier",
    "get_all_classifiers",
    "get_llm_classifiers",
]
