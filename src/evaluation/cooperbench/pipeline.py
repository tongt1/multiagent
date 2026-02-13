"""CooperBench cooperation pipeline.

Orchestrates the solver-verifier debate-as-cooperation loop for
CooperBench tasks. Two agents each implement a feature, exchange
approach summaries, revise their patches, and submit for evaluation.

Supports both cooperative (2 agents) and solo (1 agent) modes.
"""

import time
from typing import Any, Optional

from loguru import logger

from src.evaluation.cooperbench.evaluator import CooperBenchEvaluator
from src.evaluation.cooperbench.models import (
    CooperBenchConfig,
    CooperBenchEvalResult,
    CooperBenchPipelineResult,
    CooperBenchProblem,
    CooperBenchResponse,
)
from src.evaluation.cooperbench.reward import (
    compute_cooperbench_partial_reward,
    compute_cooperbench_reward,
)
from src.infrastructure.llm_client import LLMClient
from src.models.trajectory import TokenUsage


# Prompt templates for code cooperation agents

SOLVER_SYSTEM_PROMPT = """You are an expert software engineer working on a collaborative coding task.
You are Agent {agent_id} and your job is to implement a specific feature in a shared codebase.
Another agent is implementing a different feature in the same codebase simultaneously.

Your goal is to:
1. Implement your assigned feature correctly
2. Avoid conflicts with your partner's changes
3. Communicate your approach clearly so your partner can coordinate

Output your implementation as a git diff patch."""

SOLVER_PROMPT_TEMPLATE = """## Your Feature Assignment

{feature_spec}

## Repository Context

Repository: {repo}
Task: {task_id}

{partner_section}

{feedback_section}

## Instructions

1. Analyze the feature specification carefully
2. Plan your implementation approach
3. Generate a git diff patch that implements the feature
4. Provide a brief summary of your approach for your partner

Respond with:
- APPROACH: A 2-3 sentence summary of what files you'll modify and how
- PATCH: Your git diff patch (between ```diff and ``` markers)
- CONFIDENCE: A number between 0 and 1"""

SOLO_PROMPT_TEMPLATE = """## Feature Assignments

You must implement BOTH features in the same codebase.

### Feature 1:
{feature_spec_1}

### Feature 2:
{feature_spec_2}

## Repository Context

Repository: {repo}
Task: {task_id}

## Instructions

1. Analyze both feature specifications
2. Plan an implementation that satisfies both features without conflicts
3. Generate a single git diff patch covering both features

Respond with:
- APPROACH: Summary of your implementation plan
- PATCH: Your git diff patch (between ```diff and ``` markers)
- CONFIDENCE: A number between 0 and 1"""


class CooperBenchPipeline:
    """Orchestrates CooperBench solver-verifier cooperation.

    Implements the debate-as-cooperation paradigm:
    - Solver Agent = Feature Agent 1 (implements feature 1)
    - Verifier Agent = Feature Agent 2 (implements feature 2)
    - Debate rounds = Agent messaging (approach summaries)
    - Judge = Merge + test evaluation (ground truth)
    """

    def __init__(
        self,
        config: CooperBenchConfig,
        evaluator: Optional[CooperBenchEvaluator] = None,
    ) -> None:
        """Initialize the cooperation pipeline.

        Args:
            config: CooperBench configuration.
            evaluator: Optional evaluator instance (created from config if None).
        """
        self.config = config

        # Create LLM clients for agents
        self.solver_client = LLMClient(
            model=config.solver_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.verifier_client = LLMClient(
            model=config.verifier_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Create evaluator
        self.evaluator = evaluator or CooperBenchEvaluator(
            backend=config.backend,
            timeout=config.timeout,
        )

    async def run(
        self,
        problem: CooperBenchProblem,
    ) -> CooperBenchPipelineResult:
        """Run the cooperation pipeline on a problem.

        In coop mode: two agents each implement one feature with
        communication rounds. In solo mode: one agent implements both.

        Args:
            problem: CooperBench problem to solve.

        Returns:
            CooperBenchPipelineResult with patches, evaluation, and metrics.
        """
        start_time = time.monotonic()

        logger.info(
            f"Running {self.config.mode} pipeline for "
            f"{problem.repo}/{problem.task_id} features={problem.features}"
        )

        try:
            if self.config.mode == "solo":
                result = await self._run_solo(problem)
            else:
                result = await self._run_coop(problem)

            result.wall_time = time.monotonic() - start_time
            return result

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"Pipeline failed: {e}")
            return CooperBenchPipelineResult(
                problem=problem,
                wall_time=elapsed,
                error=str(e),
            )

    async def _run_coop(
        self,
        problem: CooperBenchProblem,
    ) -> CooperBenchPipelineResult:
        """Run cooperative (2-agent) pipeline.

        Loop:
        1. Both agents receive their feature specs
        2. Round 1: Both generate initial patches + approach summaries
        3. Round 2+: Agents revise patches based on partner's approach
        4. Final: Merge patches + run tests

        Args:
            problem: Problem to solve.

        Returns:
            Pipeline result.
        """
        trajectory: list[dict[str, Any]] = []
        total_solver_tokens = 0
        total_verifier_tokens = 0
        messages_exchanged = 0
        approach_summaries: list[str] = []

        # Get feature specs for each agent
        feature_ids = problem.features
        if len(feature_ids) < 2:
            return CooperBenchPipelineResult(
                problem=problem,
                error="Need at least 2 features for coop mode",
            )

        solver_feature_id = feature_ids[0]
        verifier_feature_id = feature_ids[1]
        solver_spec = problem.feature_specs[solver_feature_id].description
        verifier_spec = problem.feature_specs[verifier_feature_id].description

        solver_patch = ""
        verifier_patch = ""
        solver_approach = ""
        verifier_approach = ""

        for round_num in range(1, self.config.max_rounds + 1):
            logger.info(f"=== Cooperation Round {round_num} ===")

            # Build partner sections
            solver_partner_section = ""
            verifier_partner_section = ""

            if round_num > 1 and self.config.messaging_enabled:
                solver_partner_section = (
                    f"## Partner's Approach (Feature {verifier_feature_id})\n\n"
                    f"{verifier_approach}\n\n"
                    f"Coordinate your changes to avoid conflicts with their approach."
                )
                verifier_partner_section = (
                    f"## Partner's Approach (Feature {solver_feature_id})\n\n"
                    f"{solver_approach}\n\n"
                    f"Coordinate your changes to avoid conflicts with their approach."
                )
                messages_exchanged += 2

            # Generate solver patch
            solver_response, solver_usage = await self._generate_agent_patch(
                client=self.solver_client,
                agent_id=1,
                feature_spec=solver_spec,
                repo=problem.repo,
                task_id=problem.task_id,
                partner_section=solver_partner_section,
                feedback="" if round_num == 1 else f"This is revision round {round_num}. Refine your patch.",
            )
            solver_patch = solver_response.patch
            solver_approach = solver_response.approach_summary
            total_solver_tokens += solver_usage.total_tokens

            trajectory.append({
                "round": round_num,
                "agent": "solver",
                "action": "generate_patch",
                "approach": solver_approach,
                "patch_length": len(solver_patch),
                "confidence": solver_response.confidence,
                "tokens": solver_usage.model_dump(),
            })

            # Generate verifier patch
            verifier_response, verifier_usage = await self._generate_agent_patch(
                client=self.verifier_client,
                agent_id=2,
                feature_spec=verifier_spec,
                repo=problem.repo,
                task_id=problem.task_id,
                partner_section=verifier_partner_section,
                feedback="" if round_num == 1 else f"This is revision round {round_num}. Refine your patch.",
            )
            verifier_patch = verifier_response.patch
            verifier_approach = verifier_response.approach_summary
            total_verifier_tokens += verifier_usage.total_tokens

            trajectory.append({
                "round": round_num,
                "agent": "verifier",
                "action": "generate_patch",
                "approach": verifier_approach,
                "patch_length": len(verifier_patch),
                "confidence": verifier_response.confidence,
                "tokens": verifier_usage.model_dump(),
            })

            approach_summaries.extend([solver_approach, verifier_approach])

        # Evaluate merged patches
        eval_result = await self.evaluator.evaluate_coop(
            problem, solver_patch, verifier_patch
        )

        # Compute reward
        reward = compute_cooperbench_reward(eval_result)

        return CooperBenchPipelineResult(
            problem=problem,
            eval_result=eval_result,
            reward=reward,
            patches=[solver_patch, verifier_patch],
            approach_summaries=approach_summaries,
            rounds_completed=self.config.max_rounds,
            messages_exchanged=messages_exchanged,
            total_tokens=total_solver_tokens + total_verifier_tokens,
            solver_tokens=total_solver_tokens,
            verifier_tokens=total_verifier_tokens,
            trajectory=trajectory,
        )

    async def _run_solo(
        self,
        problem: CooperBenchProblem,
    ) -> CooperBenchPipelineResult:
        """Run solo (single-agent) pipeline.

        One agent implements both features in a single patch.

        Args:
            problem: Problem to solve.

        Returns:
            Pipeline result.
        """
        trajectory: list[dict[str, Any]] = []

        feature_ids = problem.features
        if len(feature_ids) < 2:
            return CooperBenchPipelineResult(
                problem=problem,
                error="Need at least 2 features",
            )

        spec_1 = problem.feature_specs[feature_ids[0]].description
        spec_2 = problem.feature_specs[feature_ids[1]].description

        # Build solo prompt
        prompt = SOLO_PROMPT_TEMPLATE.format(
            feature_spec_1=spec_1,
            feature_spec_2=spec_2,
            repo=problem.repo,
            task_id=problem.task_id,
        )

        messages = [{"role": "user", "content": prompt}]
        system_prompt = (
            "You are an expert software engineer. Implement both features "
            "in a single git diff patch. Ensure they work together without conflicts."
        )

        # Generate patch
        response_text, usage = await self.solver_client.generate_text(
            messages=messages,
            system_prompt=system_prompt,
        )

        # Parse response
        patch = self._extract_patch(response_text)
        approach = self._extract_approach(response_text)

        trajectory.append({
            "round": 1,
            "agent": "solo",
            "action": "generate_patch",
            "approach": approach,
            "patch_length": len(patch),
            "tokens": usage.model_dump(),
        })

        # Evaluate
        eval_result = await self.evaluator.evaluate_solo(problem, patch)
        reward = compute_cooperbench_reward(eval_result)

        return CooperBenchPipelineResult(
            problem=problem,
            eval_result=eval_result,
            reward=reward,
            patches=[patch],
            approach_summaries=[approach],
            rounds_completed=1,
            messages_exchanged=0,
            total_tokens=usage.total_tokens,
            solver_tokens=usage.total_tokens,
            verifier_tokens=0,
            trajectory=trajectory,
        )

    async def _generate_agent_patch(
        self,
        client: LLMClient,
        agent_id: int,
        feature_spec: str,
        repo: str,
        task_id: str,
        partner_section: str = "",
        feedback: str = "",
    ) -> tuple[CooperBenchResponse, TokenUsage]:
        """Generate a patch from an agent.

        Args:
            client: LLM client for this agent.
            agent_id: Agent identifier (1 or 2).
            feature_spec: Feature specification text.
            repo: Repository name.
            task_id: Task identifier.
            partner_section: Partner's approach information.
            feedback: Feedback from previous rounds.

        Returns:
            Tuple of (CooperBenchResponse, TokenUsage).
        """
        feedback_section = f"## Feedback\n\n{feedback}" if feedback else ""

        prompt = SOLVER_PROMPT_TEMPLATE.format(
            feature_spec=feature_spec,
            repo=repo,
            task_id=task_id,
            partner_section=partner_section,
            feedback_section=feedback_section,
        )

        system_prompt = SOLVER_SYSTEM_PROMPT.format(agent_id=agent_id)
        messages = [{"role": "user", "content": prompt}]

        response_text, usage = await client.generate_text(
            messages=messages,
            system_prompt=system_prompt,
        )

        # Parse the response into structured format
        patch = self._extract_patch(response_text)
        approach = self._extract_approach(response_text)
        confidence = self._extract_confidence(response_text)

        response = CooperBenchResponse(
            patch=patch,
            reasoning=response_text,
            confidence=confidence,
            approach_summary=approach,
        )

        return response, usage

    @staticmethod
    def _extract_patch(text: str) -> str:
        """Extract git diff patch from LLM response.

        Looks for content between ```diff and ``` markers.

        Args:
            text: Raw LLM response text.

        Returns:
            Extracted patch string, or empty string if not found.
        """
        # Try to extract from diff code block
        if "```diff" in text:
            start = text.index("```diff") + len("```diff")
            end = text.index("```", start)
            return text[start:end].strip()

        # Try generic code block
        if "```" in text:
            parts = text.split("```")
            for i in range(1, len(parts), 2):
                candidate = parts[i].strip()
                if candidate.startswith("diff") or candidate.startswith("---"):
                    return candidate

        # Look for raw diff content
        lines = text.split("\n")
        diff_lines = []
        in_diff = False
        for line in lines:
            if line.startswith("diff --git") or line.startswith("---"):
                in_diff = True
            if in_diff:
                diff_lines.append(line)

        if diff_lines:
            return "\n".join(diff_lines)

        return ""

    @staticmethod
    def _extract_approach(text: str) -> str:
        """Extract approach summary from LLM response.

        Args:
            text: Raw LLM response text.

        Returns:
            Approach summary string.
        """
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("APPROACH:"):
                return stripped[len("APPROACH:"):].strip()

        # Fallback: first non-empty line before any code block
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("```") and not stripped.startswith("#"):
                return stripped[:200]

        return ""

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Extract confidence score from LLM response.

        Args:
            text: Raw LLM response text.

        Returns:
            Confidence float between 0.0 and 1.0, default 0.5.
        """
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("CONFIDENCE:"):
                try:
                    value = float(stripped[len("CONFIDENCE:"):].strip())
                    return max(0.0, min(1.0, value))
                except ValueError:
                    pass

        return 0.5
