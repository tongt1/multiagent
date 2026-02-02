"""Evaluation and verification models."""

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
    critique: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
