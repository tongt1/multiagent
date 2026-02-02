"""LLM client with structured outputs using LiteLLM and Instructor."""

from typing import Any

import instructor
from litellm import acompletion
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.trajectory import TokenUsage


class LLMClient:
    """Async LLM client with structured output support."""

    def __init__(
        self, model: str, temperature: float = 0.0, max_tokens: int = 4096
    ) -> None:
        """Initialize LLM client.

        Args:
            model: Model identifier (e.g., "command-r-plus")
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = instructor.from_litellm(acompletion)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        system_prompt: str | None = None,
    ) -> tuple[BaseModel, TokenUsage]:
        """Generate structured output from LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Pydantic model class for structured output
            system_prompt: Optional system prompt to prepend

        Returns:
            Tuple of (parsed response model instance, token usage)
        """
        # Build full message list
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        logger.info(
            f"LLM call: model={self.model}, messages={len(full_messages)}, "
            f"response_model={response_model.__name__}"
        )

        # Call instructor-wrapped client
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            response_model=response_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=3,
        )

        # Extract token usage from raw response
        usage = response._raw_response.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

        logger.info(
            f"LLM response: tokens={token_usage.total_tokens} "
            f"(prompt={token_usage.prompt_tokens}, "
            f"completion={token_usage.completion_tokens})"
        )

        return response, token_usage

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_text(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> tuple[str, TokenUsage]:
        """Generate unstructured text output from LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt to prepend

        Returns:
            Tuple of (response text, token usage)
        """
        # Build full message list
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        logger.info(
            f"LLM text call: model={self.model}, messages={len(full_messages)}"
        )

        # Call LiteLLM directly (no instructor)
        response = await acompletion(
            model=self.model,
            messages=full_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Extract content and usage
        content = response.choices[0].message.content
        usage = response.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

        logger.info(
            f"LLM text response: tokens={token_usage.total_tokens} "
            f"(prompt={token_usage.prompt_tokens}, "
            f"completion={token_usage.completion_tokens})"
        )

        return content, token_usage
