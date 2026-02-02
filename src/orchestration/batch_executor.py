"""Batch execution infrastructure for parallel problem processing."""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from src.data.dataset_loader import Problem
from src.models.config import PipelineConfig
from src.orchestration.pipeline import PipelineResult, SolverVerifierJudgePipeline


class BatchResult(BaseModel):
    """Result of batch execution."""

    total: int
    succeeded: int
    failed: int
    results: list[PipelineResult]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_seconds: float


class BatchPipelineExecutor:
    """Executes multiple problems in parallel through the pipeline."""

    def __init__(self, config: PipelineConfig, max_concurrent: int = 10) -> None:
        """Initialize batch executor.

        Args:
            config: Pipeline configuration
            max_concurrent: Maximum number of concurrent executions
        """
        self.config = config
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(self, problem: Problem) -> PipelineResult:
        """Run pipeline on a single problem.

        Creates a fresh pipeline instance to avoid shared state issues.

        Args:
            problem: Problem to solve

        Returns:
            PipelineResult from execution
        """
        async with self.semaphore:
            logger.info(f"Starting problem {problem.id} (domain: {problem.domain})")

            # Create fresh pipeline to avoid shared state with CostTracker/TrajectoryLogger
            pipeline = SolverVerifierJudgePipeline(self.config)

            # Build metadata for problem
            metadata = {
                "problem_id": problem.id,
                "domain": problem.domain,
                **problem.metadata,
            }

            if problem.ground_truth:
                metadata["ground_truth"] = problem.ground_truth

            # Run pipeline
            result = await pipeline.run(
                problem_description=problem.problem, problem_metadata=metadata
            )

            logger.info(
                f"Completed problem {problem.id}: "
                f"passed={result.passed_verification}, score={result.judge_score:.3f}"
            )

            return result

    async def run_batch(
        self,
        problems: list[Problem],
        on_complete: Callable[[Problem, PipelineResult | Exception], None] | None = None,
    ) -> BatchResult:
        """Run pipeline on multiple problems concurrently.

        Args:
            problems: List of problems to solve
            on_complete: Optional callback invoked after each problem completes
                         (called with problem and result or exception)

        Returns:
            BatchResult with aggregated statistics
        """
        start_time = time.monotonic()
        total = len(problems)

        logger.info(
            f"Starting batch execution: {total} problems, "
            f"max_concurrent={self.max_concurrent}"
        )

        # Create tasks for all problems
        tasks = [self.run_one(problem) for problem in problems]

        # Use gather with return_exceptions=True for partial results
        # (NOT TaskGroup - we want partial results even if some fail)
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successes from failures
        results: list[PipelineResult] = []
        errors: list[dict[str, Any]] = []

        for idx, (problem, result) in enumerate(zip(problems, raw_results)):
            if isinstance(result, Exception):
                error_entry = {
                    "problem_id": problem.id,
                    "error_type": type(result).__name__,
                    "error_message": str(result),
                }
                errors.append(error_entry)
                logger.error(f"Problem {problem.id} failed: {result}")
            else:
                # Type narrowing: result is PipelineResult here
                assert isinstance(result, PipelineResult)
                results.append(result)

            # Call on_complete callback if provided
            if on_complete:
                on_complete(problem, result)

        elapsed = time.monotonic() - start_time

        batch_result = BatchResult(
            total=total,
            succeeded=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
            elapsed_seconds=elapsed,
        )

        logger.info(
            f"Batch execution complete: {batch_result.succeeded}/{batch_result.total} succeeded, "
            f"elapsed={elapsed:.2f}s"
        )

        return batch_result
