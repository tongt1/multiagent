"""Cost tracker for LLM API usage."""

from typing import Any

from src.models.trajectory import TokenUsage


class CostTracker:
    """Track token usage and costs across pipeline runs."""

    # Cost rates per 1K tokens (input, output) in USD
    DEFAULT_RATES = {
        "command-r-plus": (0.003, 0.015),  # $3/$15 per 1M tokens
        "command-r": (0.0005, 0.0015),  # $0.50/$1.50 per 1M tokens
        "command-r-plus-08-2024": (0.003, 0.015),
        "command-r-08-2024": (0.0005, 0.0015),
    }

    def __init__(self, custom_rates: dict[str, tuple[float, float]] | None = None) -> None:
        """Initialize cost tracker.

        Args:
            custom_rates: Optional dict of model -> (input_rate, output_rate) per 1K tokens
        """
        self.rates = {**self.DEFAULT_RATES}
        if custom_rates:
            self.rates.update(custom_rates)

        # Track usage by model and agent
        self.usage_by_model: dict[str, TokenUsage] = {}
        self.usage_by_agent: dict[str, TokenUsage] = {}
        self.cost_by_model: dict[str, float] = {}
        self.cost_by_agent: dict[str, float] = {}

    def add_usage(
        self, model: str, usage: TokenUsage, agent: str | None = None
    ) -> float:
        """Add token usage and calculate cost.

        Args:
            model: Model identifier
            usage: Token usage data
            agent: Optional agent identifier for per-agent tracking

        Returns:
            Cost in USD for this usage
        """
        # Calculate cost
        if model in self.rates:
            input_rate, output_rate = self.rates[model]
            cost = (
                (usage.prompt_tokens / 1000.0) * input_rate
                + (usage.completion_tokens / 1000.0) * output_rate
            )
        else:
            # Unknown model, use conservative estimate
            cost = (usage.total_tokens / 1000.0) * 0.01

        # Update model tracking
        if model not in self.usage_by_model:
            self.usage_by_model[model] = TokenUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            )
            self.cost_by_model[model] = 0.0

        self.usage_by_model[model] = TokenUsage(
            prompt_tokens=self.usage_by_model[model].prompt_tokens + usage.prompt_tokens,
            completion_tokens=self.usage_by_model[model].completion_tokens
            + usage.completion_tokens,
            total_tokens=self.usage_by_model[model].total_tokens + usage.total_tokens,
        )
        self.cost_by_model[model] += cost

        # Update agent tracking if provided
        if agent:
            if agent not in self.usage_by_agent:
                self.usage_by_agent[agent] = TokenUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                )
                self.cost_by_agent[agent] = 0.0

            self.usage_by_agent[agent] = TokenUsage(
                prompt_tokens=self.usage_by_agent[agent].prompt_tokens + usage.prompt_tokens,
                completion_tokens=self.usage_by_agent[agent].completion_tokens
                + usage.completion_tokens,
                total_tokens=self.usage_by_agent[agent].total_tokens + usage.total_tokens,
            )
            self.cost_by_agent[agent] += cost

        return cost

    def total_cost(self) -> float:
        """Get total accumulated cost in USD.

        Returns:
            Total cost across all models and agents
        """
        return sum(self.cost_by_model.values())

    def total_tokens(self) -> TokenUsage:
        """Get total accumulated token usage.

        Returns:
            TokenUsage summed across all models
        """
        total_prompt = sum(u.prompt_tokens for u in self.usage_by_model.values())
        total_completion = sum(u.completion_tokens for u in self.usage_by_model.values())
        total_all = sum(u.total_tokens for u in self.usage_by_model.values())

        return TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_all,
        )

    def summary(self) -> dict[str, Any]:
        """Get detailed cost and usage summary.

        Returns:
            Dict with breakdown by model and agent
        """
        return {
            "total_cost_usd": self.total_cost(),
            "total_tokens": self.total_tokens().model_dump(),
            "by_model": {
                model: {
                    "tokens": usage.model_dump(),
                    "cost_usd": self.cost_by_model[model],
                }
                for model, usage in self.usage_by_model.items()
            },
            "by_agent": {
                agent: {
                    "tokens": usage.model_dump(),
                    "cost_usd": self.cost_by_agent[agent],
                }
                for agent, usage in self.usage_by_agent.items()
            },
        }
