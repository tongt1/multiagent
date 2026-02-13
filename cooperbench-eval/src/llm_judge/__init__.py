"""LLM-based failure mode classifiers for CooperBench evaluation.

These classifiers use the Cohere API (CO_API_KEY) for semantic analysis
of multi-agent transcripts, detecting coordination failures that require
natural language understanding beyond simple heuristics.
"""

from src.llm_judge.base import LLMClassifier
from src.llm_judge.broken_commitment import BrokenCommitmentClassifier
from src.llm_judge.dependency_access import DependencyAccessClassifier
from src.llm_judge.divergent_architecture import DivergentArchitectureClassifier
from src.llm_judge.parameter_flow import ParameterFlowClassifier
from src.llm_judge.timing_dependency import TimingDependencyClassifier
from src.llm_judge.unverifiable_claims import UnverifiableClaimsClassifier

# All LLM-based classifiers in order of prevalence
LLM_CLASSIFIERS: list[type[LLMClassifier]] = [
    DivergentArchitectureClassifier,  # 29.7%
    UnverifiableClaimsClassifier,     # 4.3%
    BrokenCommitmentClassifier,       # 3.7%
    DependencyAccessClassifier,       # 1.7%
    ParameterFlowClassifier,          # 1.3%
    TimingDependencyClassifier,       # 1.1%
]

__all__ = [
    "BrokenCommitmentClassifier",
    "DependencyAccessClassifier",
    "DivergentArchitectureClassifier",
    "LLM_CLASSIFIERS",
    "LLMClassifier",
    "ParameterFlowClassifier",
    "TimingDependencyClassifier",
    "UnverifiableClaimsClassifier",
]
