"""Judge agent that scores final solutions."""

from typing import Any

from src.agents.base import BaseAgent
from src.infrastructure.llm_client import LLMClient
from src.models.config import JudgeConfig
from src.models.evaluation import Judgment
from src.models.trajectory import TokenUsage


class JudgeAgent(BaseAgent):
    """Agent that evaluates and scores final solutions."""

    def __init__(self, config: JudgeConfig, llm_client: LLMClient) -> None:
        """Initialize judge agent.

        Args:
            config: Judge configuration
            llm_client: LLM client for API calls
        """
        super().__init__(config, llm_client)
        self.config: JudgeConfig = config  # Type narrowing for mypy

    async def score(
        self,
        problem_description: str,
        solution: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> tuple[Judgment, TokenUsage]:
        """Score a solution's quality.

        Args:
            problem_description: The original problem
            solution: The solution to score
            conversation_history: Optional conversation history (currently unused)

        Returns:
            Tuple of (judgment, token usage)
        """
        # Format prompt using template with named placeholders
        # Note: conversation_history is accepted but not used in this implementation
        prompt = self.config.prompt_template.format(
            problem=problem_description,
            solution=solution,
            rubric=self.config.scoring_rubric,
        )

        # Build messages
        messages = self._build_messages(prompt)

        # Call LLM with structured output
        response, token_usage = await self.llm_client.generate(
            messages=messages,
            response_model=Judgment,
            system_prompt=self.config.system_prompt if self.config.system_prompt else None,
        )

        # Type assertion for mypy - response is guaranteed to be Judgment
        assert isinstance(response, Judgment)
        return response, token_usage
