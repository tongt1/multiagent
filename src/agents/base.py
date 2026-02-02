"""Base agent class for solver, verifier, and judge agents."""

from abc import ABC
from typing import Any

from src.infrastructure.llm_client import LLMClient
from src.models.config import AgentConfig, JudgeConfig


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        config: AgentConfig | JudgeConfig,
        llm_client: LLMClient,
    ) -> None:
        """Initialize base agent.

        Args:
            config: Agent or Judge configuration
            llm_client: LLM client for API calls
        """
        self.config = config
        self.llm_client = llm_client

    @property
    def model(self) -> str:
        """Get model identifier for cost tracking.

        Returns:
            Model identifier string
        """
        return self.config.model

    def _build_messages(self, user_content: str) -> list[dict[str, Any]]:
        """Build message list for LLM call.

        Args:
            user_content: User message content

        Returns:
            List of message dicts with role and content
        """
        return [{"role": "user", "content": user_content}]
