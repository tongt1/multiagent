"""Verifier agent that validates solutions."""

from src.agents.base import BaseAgent
from src.infrastructure.llm_client import LLMClient
from src.models.config import AgentConfig
from src.models.evaluation import VerificationResult
from src.models.trajectory import TokenUsage


class VerifierAgent(BaseAgent):
    """Agent that validates and critiques solutions."""

    def __init__(self, config: AgentConfig, llm_client: LLMClient) -> None:
        """Initialize verifier agent.

        Args:
            config: Agent configuration with role="verifier"
            llm_client: LLM client for API calls

        Raises:
            ValueError: If config.role is not "verifier"
        """
        if config.role != "verifier":
            raise ValueError(f"VerifierAgent requires role='verifier', got '{config.role}'")
        super().__init__(config, llm_client)

    async def validate(
        self,
        problem_description: str,
        solution: str,
    ) -> tuple[VerificationResult, TokenUsage]:
        """Validate a solution against the problem.

        Args:
            problem_description: The original problem
            solution: The proposed solution to validate

        Returns:
            Tuple of (verification result, token usage)
        """
        # Format prompt using template with named placeholders
        prompt = self.config.prompt_template.format(
            problem=problem_description,
            solution=solution,
        )

        # Build messages
        messages = self._build_messages(prompt)

        # Call LLM with structured output
        response, token_usage = await self.llm_client.generate(
            messages=messages,
            response_model=VerificationResult,
            system_prompt=self.config.system_prompt if self.config.system_prompt else None,
        )

        # Type assertion for mypy - response is guaranteed to be VerificationResult
        assert isinstance(response, VerificationResult)
        return response, token_usage
