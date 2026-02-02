"""Message and conversation models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Conversation(BaseModel):
    """A conversation with messages and metadata."""

    messages: list[Message]
    problem_description: str
    metadata: dict[str, Any] = Field(default_factory=dict)
