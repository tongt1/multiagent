"""Base class for LLM-based failure mode classifiers.

Uses the Cohere API (via CO_API_KEY env var) for semantic analysis of
multi-agent transcripts. Provides prompt formatting, API calling, and
response parsing infrastructure shared by all LLM classifiers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import abstractmethod
from typing import Any

import httpx

from src.classifiers.base import BaseClassifier, ClassificationResult, Severity
from src.data_loading.schemas import Message, TaskData

logger = logging.getLogger(__name__)

# Cohere API configuration
# Use staging endpoint (production key is unauthorized; staging works with CO_API_KEY)
COHERE_API_URL = os.environ.get("COHERE_API_URL", "https://stg.api.cohere.com/v2/chat")
DEFAULT_MODEL = "command-r-plus-08-2024"
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


class LLMClassifier(BaseClassifier):
    """Base class for LLM-powered failure mode classifiers.

    Subclasses implement `build_prompt()` and `parse_response()` to define
    the classification logic. The base class handles API calls, retries,
    and error handling.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self._api_key = api_key or os.environ.get("CO_API_KEY", "")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        if not self._api_key:
            logger.warning("CO_API_KEY not set; LLM classifiers will return skip results.")

    @abstractmethod
    def build_prompt(self, task: TaskData) -> str:
        """Build the classification prompt for the LLM.

        Args:
            task: Complete task data to analyze.

        Returns:
            Formatted prompt string.
        """
        ...

    @abstractmethod
    def parse_response(self, response_text: str, task: TaskData) -> ClassificationResult:
        """Parse the LLM response into a ClassificationResult.

        Args:
            response_text: Raw text response from the LLM.
            task: The task data that was analyzed.

        Returns:
            Parsed ClassificationResult.
        """
        ...

    def classify(self, task: TaskData) -> ClassificationResult:
        """Run LLM-based classification on a task."""
        skip = self._skip_if_solo(task)
        if skip is not None:
            return skip

        if not self._api_key:
            return ClassificationResult(
                classifier_name=self.name,
                skipped=True,
                skip_reason="CO_API_KEY not set",
            )

        prompt = self.build_prompt(task)
        if not prompt:
            return self._no_detection()

        try:
            response_text = self._call_llm(prompt)
            return self.parse_response(response_text, task)
        except Exception as e:
            logger.error("LLM classification failed for %s on task %s: %s", self.name, task.task_id, e)
            return ClassificationResult(
                classifier_name=self.name,
                skipped=True,
                skip_reason=f"LLM call failed: {e}",
            )

    def _call_llm(self, prompt: str) -> str:
        """Call the Cohere API with retries.

        Args:
            prompt: The user prompt to send.

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at analyzing multi-agent system transcripts "
                        "for coordination failures. You respond in structured JSON format."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                # Rate limiting: small delay between calls to avoid 429/502
                if attempt > 0:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    time.sleep(0.5)  # Base delay between requests

                with httpx.Client(timeout=90.0) as client:
                    resp = client.post(COHERE_API_URL, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()

                    # Extract text from Cohere v2 response
                    content = data.get("message", {}).get("content", [])
                    if content and isinstance(content, list):
                        return content[0].get("text", "")
                    return str(content)

            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
                last_error = e
                logger.warning(
                    "LLM API attempt %d/%d failed: %s",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        raise RuntimeError(f"LLM API failed after {MAX_RETRIES} retries: {last_error}")

    @staticmethod
    def format_transcript(messages: list[Message], max_messages: int = 20) -> str:
        """Format messages into a readable transcript string for the LLM.

        Args:
            messages: List of Message objects.
            max_messages: Maximum number of messages to include.

        Returns:
            Formatted transcript string.
        """
        lines: list[str] = []
        for msg in messages[:max_messages]:
            role_info = f" ({msg.metadata.get('agent_role', '')})" if msg.metadata.get("agent_role") else ""
            reward = msg.metadata.get("reward", "")
            reward_str = f" [reward={reward}]" if reward != "" else ""
            lines.append(f"[Step {msg.index}] {msg.agent}{role_info}{reward_str}: {msg.content}")

        if len(messages) > max_messages:
            lines.append(f"... ({len(messages) - max_messages} more messages omitted)")

        return "\n".join(lines)

    @staticmethod
    def parse_json_response(text: str) -> dict[str, Any]:
        """Parse JSON from an LLM response, handling markdown code blocks.

        Args:
            text: Raw LLM response that may contain JSON.

        Returns:
            Parsed dict, or empty dict if parsing fails.
        """
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding any JSON object in the text
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON from LLM response: %s...", text[:200])
        return {}
