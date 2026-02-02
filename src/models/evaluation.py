"""Evaluation and verification models."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Judgment(BaseModel):
    """Judge's evaluation of a solution."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    strengths: list[str]
    weaknesses: list[str]


class VerificationResult(BaseModel):
    """Result of verifier's assessment."""

    passed: bool
    critique: Optional[str] = None
    scores: dict = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class GroundTruthComparison(BaseModel):
    """Result of ground-truth verification."""

    domain: Literal["math", "code", "general"]
    reward: float
    is_correct: bool
    details: dict[str, Any]
