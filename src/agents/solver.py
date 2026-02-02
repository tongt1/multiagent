"""Solver agent that generates solutions to problems."""

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.infrastructure.llm_client import LLMClient
from src.models.config import AgentConfig
from src.models.trajectory import TokenUsage


class SolverResponse(BaseModel):
    """Structured response from solver agent."""

    solution: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class SolverAgent(BaseAgent):
    """Agent that generates solutions to problems."""

    def __init__(self, config: AgentConfig, llm_client: LLMClient) -> None:
        """Initialize solver agent.

        Args:
            config: Agent configuration with role="solver"
            llm_client: LLM client for API calls

        Raises:
            ValueError: If config.role is not "solver"
        """
        if config.role != "solver":
            raise ValueError(f"SolverAgent requires role='solver', got '{config.role}'")
        super().__init__(config, llm_client)

    async def generate(
        self,
        problem_description: str,
        feedback: str | None = None,
    ) -> tuple[SolverResponse, TokenUsage]:
        """Generate a solution to the problem.

        Args:
            problem_description: The problem to solve
            feedback: Optional feedback from previous iteration

        Returns:
            Tuple of (solver response, token usage)
        """
        # Build feedback section
        if feedback:
            feedback_section = f"Previous feedback from verifier:\n{feedback}"
        else:
            feedback_section = ""

        # Format prompt using template with named placeholders
        prompt = self.config.prompt_template.format(
            problem=problem_description,
            feedback_section=feedback_section,
        )

        # Build messages
        messages = self._build_messages(prompt)

        # Call LLM with structured output
        response, token_usage = await self.llm_client.generate(
            messages=messages,
            response_model=SolverResponse,
            system_prompt=self.config.system_prompt if self.config.system_prompt else None,
        )

        # Type assertion for mypy - response is guaranteed to be SolverResponse
        assert isinstance(response, SolverResponse)
        return response, token_usage
