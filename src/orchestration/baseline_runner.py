"""Baseline runner for single-agent trajectory generation without debate."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.agents import SolverAgent
from src.infrastructure.cost_tracker import CostTracker
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.trajectory_logger import TrajectoryLogger
from src.models.config import PipelineConfig
from src.orchestration.pipeline import PipelineResult


class BaselineRunner:
    """Runs single-agent baseline: same solver, no debate, matched token budget."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize baseline runner with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Create LLM client for solver (same as debate mode)
        self.solver_client = LLMClient(
            model=config.solver.model,
            temperature=config.solver.temperature,
            max_tokens=config.solver.max_tokens,
        )

        # Create solver agent (same as debate mode)
        self.solver = SolverAgent(config.solver, self.solver_client)

        # Create cost tracker
        self.cost_tracker = CostTracker()

        # Compute config hash
        self.config_hash = config.config_hash()

    async def run(
        self,
        problem_description: str,
        problem_metadata: dict[str, Any] | None = None,
        token_budget: int | None = None,
    ) -> PipelineResult:
        """Run baseline trajectory generation (single solver call, no debate).

        Args:
            problem_description: The problem to solve
            problem_metadata: Optional metadata about the problem (e.g., ground_truth)
            token_budget: Optional token budget to match multi-agent consumption

        Returns:
            PipelineResult with baseline trajectory
        """
        # Generate run ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"baseline_{timestamp}_{self.config_hash}"
        logger.info(f"Starting baseline run: {run_id}")

        # Create trajectory path
        trajectory_dir = Path(self.config.trajectory_output_dir)
        trajectory_path = trajectory_dir / f"{run_id}.jsonl"

        # Apply token budget if specified
        original_max_tokens = self.solver_client.max_tokens
        if token_budget is not None:
            self.solver_client.max_tokens = token_budget
            logger.info(f"Applied token budget: {token_budget}")

        current_solution = ""
        passed_verification = False

        try:
            # Open trajectory logger
            async with TrajectoryLogger(trajectory_path, run_id, self.config_hash) as traj:
                # SINGLE SOLVER CALL (no feedback loop)
                try:
                    solver_response, solver_usage = await self.solver.generate(
                        problem_description=problem_description,
                        feedback=None,  # No feedback in baseline mode
                    )
                    current_solution = solver_response.solution

                    # Log solver step with mode="baseline" in metadata
                    traj.log_step(
                        agent="solver",
                        action="generate",
                        input_data={
                            "problem": problem_description,
                            "feedback": None,
                        },
                        output_data={
                            "solution": solver_response.solution,
                            "reasoning": solver_response.reasoning,
                            "confidence": solver_response.confidence,
                        },
                        metadata={
                            "iteration": 1,
                            "model": self.solver.model,
                            "tokens": solver_usage.model_dump(),
                            "mode": "baseline",  # Mark as baseline mode
                        },
                    )

                    # Track cost
                    cost = self.cost_tracker.add_usage(
                        model=self.solver.model,
                        usage=solver_usage,
                        agent="solver",
                    )
                    logger.info(f"Solver cost: ${cost:.6f}")

                except Exception as e:
                    logger.error(f"Solver failed in baseline run: {e}")
                    traj.log_error(
                        agent="solver",
                        action="generate",
                        error=e,
                        metadata={"iteration": 1, "mode": "baseline"},
                    )
                    current_solution = ""

                # GROUND TRUTH REWARD (when ground truth available)
                ground_truth_reward = None
                ground_truth_details = None
                if problem_metadata and "ground_truth" in problem_metadata:
                    from src.evaluation.math_verifier import compute_math_reward

                    try:
                        reward_score, verification_result = compute_math_reward(
                            solution=current_solution,
                            ground_truth_solution=problem_metadata["ground_truth"],
                        )
                        ground_truth_reward = reward_score  # Binary: 1.0 or 0.0
                        ground_truth_details = {
                            "is_correct": verification_result.is_correct,
                            "predicted_answer": verification_result.predicted_answer,
                            "expected_answer": verification_result.expected_answer,
                            "method": verification_result.method,
                        }

                        # Set passed_verification based on ground truth
                        passed_verification = verification_result.is_correct

                        # Log reward step in trajectory (identical format to debate mode)
                        traj.log_step(
                            agent="reward",
                            action="ground_truth_verify",
                            input_data={
                                "solution": current_solution,
                                "ground_truth": problem_metadata["ground_truth"],
                            },
                            output_data={
                                "reward": reward_score,
                                "is_correct": verification_result.is_correct,
                                "method": verification_result.method,
                            },
                            metadata={
                                "predicted_answer": verification_result.predicted_answer,
                                "expected_answer": verification_result.expected_answer,
                                "mode": "baseline",
                                "termination_reason": "single_iteration",
                                "termination_iteration": 1,
                            },
                        )
                        logger.info(
                            f"Ground truth reward: {reward_score} "
                            f"(method: {verification_result.method})"
                        )
                    except Exception as e:
                        logger.warning(f"Ground truth verification failed: {e}")

        finally:
            # Restore original max_tokens
            if token_budget is not None:
                self.solver_client.max_tokens = original_max_tokens

        # Get cost summary
        cost_summary = self.cost_tracker.summary()

        # Create result (iterations=1 for baseline)
        result = PipelineResult(
            problem_description=problem_description,
            solution=current_solution,
            passed_verification=passed_verification,
            judge_score=0.0,  # No judge in baseline mode
            iterations=1,  # Always 1 iteration
            total_cost=self.cost_tracker.total_cost(),
            trajectory_path=str(trajectory_path),
            token_usage=self.cost_tracker.total_tokens(),
            cost_summary=cost_summary,
            ground_truth_reward=ground_truth_reward,
            ground_truth_details=ground_truth_details,
        )

        logger.info(
            f"Baseline complete: iterations=1, "
            f"passed={passed_verification}, "
            f"cost=${result.total_cost:.6f}"
        )

        return result
