"""Data schemas for CooperBench evaluation traces."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    """Known agent roles in CooperBench traces."""

    AGENT_A = "agent_a"
    AGENT_B = "agent_b"
    UNKNOWN = "unknown"


INTERROGATIVE_PATTERNS = re.compile(
    r"\b(what|where|when|why|how|which|who|whom|whose|"
    r"can you|could you|would you|will you|do you|did you|"
    r"is there|are there|have you|has it|shall we|should we|"
    r"does this|does that|is this|is that)\b",
    re.IGNORECASE,
)


@dataclass
class Message:
    """A single message in a multi-agent conversation trace."""

    agent: str
    content: str
    index: int
    role: AgentRole = AgentRole.UNKNOWN
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_question(self) -> bool:
        """Detect if this message contains a question.

        Uses both '?' presence and interrogative word patterns for robustness.
        """
        if "?" in self.content:
            return True
        return bool(INTERROGATIVE_PATTERNS.search(self.content))

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Message(agent={self.agent!r}, idx={self.index}, preview={preview!r})"


@dataclass
class PatchInfo:
    """Parsed patch information for a single agent."""

    agent: str
    raw_diff: str
    files_modified: list[str] = field(default_factory=list)
    added_lines: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    functions_modified: dict[str, list[str]] = field(default_factory=dict)
    """Mapping of file path -> list of function/method names touched."""

    @property
    def total_changes(self) -> int:
        return len(self.added_lines) + len(self.removed_lines)


@dataclass
class EvalResult:
    """Ground-truth evaluation result for a CooperBench task."""

    task_id: str
    passed: bool
    score: float = 0.0
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskData:
    """Complete data for a single CooperBench task run.

    Aggregates conversation messages, patches, and evaluation results
    for downstream classifier consumption.
    """

    task_id: str
    run_id: str
    messages: list[Message] = field(default_factory=list)
    patches: list[PatchInfo] = field(default_factory=list)
    eval_result: EvalResult | None = None
    task_description: str = ""
    agents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_messages(self) -> dict[str, list[Message]]:
        """Group messages by agent."""
        groups: dict[str, list[Message]] = {}
        for msg in self.messages:
            groups.setdefault(msg.agent, []).append(msg)
        return groups

    @property
    def agent_patches(self) -> dict[str, PatchInfo]:
        """Map agent -> their patch."""
        return {p.agent: p for p in self.patches}

    @property
    def is_solo(self) -> bool:
        """True if only one agent participated."""
        return len(self.agents) <= 1

    def messages_by_agent(self, agent: str) -> list[Message]:
        """Get all messages from a specific agent."""
        return [m for m in self.messages if m.agent == agent]
