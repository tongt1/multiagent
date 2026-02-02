"""Trajectory logging models."""

from typing import Any, Optional

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TrajectoryEntry(BaseModel):
    """A single entry in the trajectory log."""

    timestamp: str
    run_id: str
    step_id: int
    agent: str
    action: str
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]
    # RL training fields (backward compatible)
    reward: Optional[float] = None
    terminal: bool = False
    success: Optional[bool] = None
