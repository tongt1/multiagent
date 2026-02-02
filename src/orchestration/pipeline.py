"""Pipeline orchestration for solver-verifier-judge loop."""

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from src.agents import JudgeAgent, SolverAgent, VerifierAgent
from src.infrastructure.cost_tracker import CostTracker
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.trajectory_logger import TrajectoryLogger
from src.models.config import PipelineConfig
from src.models.trajectory import TokenUsage
from src.orchestration.iteration import IterationController


class PipelineResult(BaseModel):
    """Result of a complete pipeline run."""

    problem_description: str
    solution: str
    passed_verification: bool
    judge_score: float
    iterations: int
    total_cost: float
    trajectory_path: str
    token_usage: TokenUsage
    cost_summary: dict[str, Any]


class SolverVerifierJudgePipeline:
    """Orchestrates the solver-verifier-judge pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Create LLM clients for each agent
        self.solver_client = LLMClient(
            model=config.solver.model,
            temperature=config.solver.temperature,
            max_tokens=config.solver.max_tokens,
        )
        self.verifier_client = LLMClient(
            model=config.verifier.model,
            temperature=config.verifier.temperature,
            max_tokens=config.verifier.max_tokens,
        )
        self.judge_client = LLMClient(
            model=config.judge.model,
            temperature=config.judge.temperature,
            max_tokens=config.judge.max_tokens,
        )

        # Create agents
        self.solver = SolverAgent(config.solver, self.solver_client)
        self.verifier = VerifierAgent(config.verifier, self.verifier_client)
        self.judge = JudgeAgent(config.judge, self.judge_client)

        # Create iteration controller
        self.iteration_controller = IterationController(config.max_iterations)

        # Compute config hash
        self.config_hash = config.config_hash()

        # Create cost tracker
        self.cost_tracker = CostTracker()

    async def run(
        self,
        problem_description: str,
        problem_metadata: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Run the complete pipeline on a problem.

        Args:
            problem_description: The problem to solve
            problem_metadata: Optional metadata about the problem

        Returns:
            PipelineResult with all execution details
        """
        # Generate run ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}_{self.config_hash}"
        logger.info(f"Starting pipeline run: {run_id}")

        # Create trajectory path
        trajectory_dir = Path(self.config.trajectory_output_dir)
        trajectory_path = trajectory_dir / f"{run_id}.jsonl"

        # Track iteration state
        current_solution = ""
        feedback: str | None = None
        passed_verification = False
        iterations_run = 0
        critiques: list[str] = []

        # Open trajectory logger
        async with TrajectoryLogger(trajectory_path, run_id, self.config_hash) as traj:
            # SOLVER-VERIFIER LOOP
            for iteration in range(1, self.config.max_iterations + 1):
                iterations_run = iteration
                logger.info(f"=== Iteration {iteration} ===")

                # SOLVER STEP
                try:
                    solver_response, solver_usage = await self.solver.generate(
                        problem_description=problem_description,
                        feedback=feedback,
                    )
                    current_solution = solver_response.solution

                    # Log solver step
                    traj.log_step(
                        agent="solver",
                        action="generate",
                        input_data={
                            "problem": problem_description,
                            "feedback": feedback,
                        },
                        output_data={
                            "solution": solver_response.solution,
                            "reasoning": solver_response.reasoning,
                            "confidence": solver_response.confidence,
                        },
                        metadata={
                            "iteration": iteration,
                            "model": self.solver.model,
                            "tokens": solver_usage.model_dump(),
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
                    logger.error(f"Solver failed in iteration {iteration}: {e}")
                    traj.log_error(
                        agent="solver",
                        action="generate",
                        error=e,
                        metadata={"iteration": iteration},
                    )
                    # Continue with empty solution
                    current_solution = ""

                # VERIFIER STEP
                if current_solution:  # Only verify if we have a solution
                    try:
                        verif_result, verif_usage = await self.verifier.validate(
                            problem_description=problem_description,
                            solution=current_solution,
                        )

                        # Log verifier step
                        traj.log_step(
                            agent="verifier",
                            action="validate",
                            input_data={
                                "problem": problem_description,
                                "solution": current_solution,
                            },
                            output_data={
                                "passed": verif_result.passed,
                                "critique": verif_result.critique,
                                "confidence": verif_result.confidence,
                                "scores": verif_result.scores,
                            },
                            metadata={
                                "iteration": iteration,
                                "model": self.verifier.model,
                                "tokens": verif_usage.model_dump(),
                            },
                        )

                        # Track cost
                        cost = self.cost_tracker.add_usage(
                            model=self.verifier.model,
                            usage=verif_usage,
                            agent="verifier",
                        )
                        logger.info(f"Verifier cost: ${cost:.6f}")

                        # Check if verification passed
                        passed_verification = verif_result.passed

                        # Record critique for circular detection
                        if verif_result.critique:
                            critiques.append(verif_result.critique)
                            self.iteration_controller.record_iteration(
                                iteration, verif_result.critique
                            )

                        # Check for circular critique
                        if self.iteration_controller.detect_circular_critique(critiques):
                            logger.warning(
                                "Circular critique pattern detected, stopping iteration"
                            )
                            break

                        # Check if should continue
                        if not self.iteration_controller.should_continue(
                            iteration, passed_verification
                        ):
                            break

                        # Set feedback for next iteration
                        feedback = verif_result.critique

                    except Exception as e:
                        logger.error(f"Verifier failed in iteration {iteration}: {e}")
                        traj.log_error(
                            agent="verifier",
                            action="validate",
                            error=e,
                            metadata={"iteration": iteration},
                        )
                        # Treat as failed verification, continue if possible
                        passed_verification = False
                        if not self.iteration_controller.should_continue(
                            iteration, passed_verification
                        ):
                            break

            # JUDGE SCORING (always runs)
            judge_score = 0.0
            try:
                judgment, judge_usage = await self.judge.score(
                    problem_description=problem_description,
                    solution=current_solution,
                )
                judge_score = judgment.score

                # Log judge step
                traj.log_step(
                    agent="judge",
                    action="score",
                    input_data={
                        "problem": problem_description,
                        "solution": current_solution,
                    },
                    output_data={
                        "score": judgment.score,
                        "reasoning": judgment.reasoning,
                        "strengths": judgment.strengths,
                        "weaknesses": judgment.weaknesses,
                    },
                    metadata={
                        "model": self.judge.model,
                        "tokens": judge_usage.model_dump(),
                    },
                )

                # Track cost
                cost = self.cost_tracker.add_usage(
                    model=self.judge.model,
                    usage=judge_usage,
                    agent="judge",
                )
                logger.info(f"Judge cost: ${cost:.6f}")

            except Exception as e:
                logger.error(f"Judge failed: {e}")
                traj.log_error(
                    agent="judge",
                    action="score",
                    error=e,
                )
                judge_score = 0.0

        # Get cost summary
        cost_summary = self.cost_tracker.summary()

        # Create result
        result = PipelineResult(
            problem_description=problem_description,
            solution=current_solution,
            passed_verification=passed_verification,
            judge_score=judge_score,
            iterations=iterations_run,
            total_cost=self.cost_tracker.total_cost(),
            trajectory_path=str(trajectory_path),
            token_usage=self.cost_tracker.total_tokens(),
            cost_summary=cost_summary,
        )

        logger.info(
            f"Pipeline complete: iterations={iterations_run}, "
            f"passed={passed_verification}, score={judge_score:.3f}, "
            f"cost=${result.total_cost:.6f}"
        )

        return result
