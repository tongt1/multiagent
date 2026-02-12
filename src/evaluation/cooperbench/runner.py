"""Batch experiment runner for CooperBench evaluations.

Runs the CooperBench cooperation pipeline over a dataset with
parallel task execution, progress tracking, and results logging.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.evaluation.cooperbench.loader import load_cooperbench_dataset
from src.evaluation.cooperbench.models import (
    CooperBenchConfig,
    CooperBenchPipelineResult,
    CooperBenchProblem,
)
from src.evaluation.cooperbench.pipeline import CooperBenchPipeline
from src.evaluation.cooperbench.reward import (
    compute_cooperbench_partial_reward,
    compute_cooperbench_reward,
)


class ExperimentResults:
    """Aggregated results from a CooperBench experiment run."""

    def __init__(self) -> None:
        """Initialize empty experiment results."""
        self.results: list[CooperBenchPipelineResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def total_tasks(self) -> int:
        """Total number of tasks evaluated."""
        return len(self.results)

    @property
    def passed_tasks(self) -> int:
        """Number of tasks where both features passed."""
        return sum(
            1 for r in self.results
            if r.eval_result and r.eval_result.both_passed
        )

    @property
    def pass_rate(self) -> float:
        """Overall pass rate (both_passed / total)."""
        if self.total_tasks == 0:
            return 0.0
        return self.passed_tasks / self.total_tasks

    @property
    def total_cost(self) -> float:
        """Total LLM API cost in USD."""
        return sum(r.total_cost for r in self.results)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all tasks."""
        return sum(r.total_tokens for r in self.results)

    @property
    def avg_wall_time(self) -> float:
        """Average wall time per task in seconds."""
        if not self.results:
            return 0.0
        return sum(r.wall_time for r in self.results) / len(self.results)

    @property
    def wall_time(self) -> float:
        """Total experiment wall time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Dict with aggregated metrics.
        """
        # Per-repo breakdown
        repo_stats: dict[str, dict[str, int]] = {}
        for r in self.results:
            repo = r.problem.repo
            if repo not in repo_stats:
                repo_stats[repo] = {"total": 0, "passed": 0}
            repo_stats[repo]["total"] += 1
            if r.eval_result and r.eval_result.both_passed:
                repo_stats[repo]["passed"] += 1

        return {
            "total_tasks": self.total_tasks,
            "passed_tasks": self.passed_tasks,
            "pass_rate": self.pass_rate,
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "avg_wall_time_s": self.avg_wall_time,
            "total_wall_time_s": self.wall_time,
            "per_repo": {
                repo: {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "pass_rate": stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0,
                }
                for repo, stats in repo_stats.items()
            },
            "errors": sum(1 for r in self.results if r.error),
        }


class CooperBenchExperimentRunner:
    """Runs CooperBench experiments over a dataset with parallelism.

    Loads the dataset, runs the pipeline on each problem (optionally
    in parallel), collects results, and saves logs.
    """

    def __init__(
        self,
        config: CooperBenchConfig,
        output_dir: str = "cooperbench_results",
    ) -> None:
        """Initialize experiment runner.

        Args:
            config: CooperBench configuration.
            output_dir: Directory for saving results and logs.
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.pipeline = CooperBenchPipeline(config)

    async def run_experiment(
        self,
        problems: Optional[list[CooperBenchProblem]] = None,
    ) -> ExperimentResults:
        """Run experiment over the dataset.

        Loads dataset if problems not provided, runs pipeline on each
        problem with configurable parallelism, and collects results.

        Args:
            problems: Optional pre-loaded problems. If None, loads from config.

        Returns:
            ExperimentResults with all pipeline results and aggregated metrics.
        """
        experiment_results = ExperimentResults()
        experiment_results.start_time = time.monotonic()

        # Load dataset if not provided
        if problems is None:
            problems = load_cooperbench_dataset(
                dataset_path=self.config.dataset_path,
                subset=self.config.subset,
                repo_filter=self.config.repo_filter,
                task_filter=self.config.task_filter,
            )

        logger.info(
            f"Starting experiment: {len(problems)} tasks, "
            f"mode={self.config.mode}, "
            f"max_parallel={self.config.max_parallel_tasks}"
        )

        # Create output directory
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = run_dir / "config.json"
        config_path.write_text(
            json.dumps(self.config.model_dump(), indent=2),
            encoding="utf-8",
        )

        # Run tasks with parallelism
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def run_with_semaphore(problem: CooperBenchProblem) -> CooperBenchPipelineResult:
            async with semaphore:
                return await self._run_single(problem)

        tasks = [run_with_semaphore(p) for p in problems]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
                error_result = CooperBenchPipelineResult(
                    problem=problems[i],
                    error=str(result),
                )
                experiment_results.results.append(error_result)
            else:
                experiment_results.results.append(result)

        experiment_results.end_time = time.monotonic()

        # Save results
        await self._save_results(run_dir, experiment_results)

        # Log summary
        summary = experiment_results.summary()
        logger.info(
            f"Experiment complete: {summary['total_tasks']} tasks, "
            f"pass_rate={summary['pass_rate']:.2%}, "
            f"cost=${summary['total_cost_usd']:.4f}, "
            f"time={summary['total_wall_time_s']:.1f}s"
        )

        return experiment_results

    async def _run_single(
        self,
        problem: CooperBenchProblem,
    ) -> CooperBenchPipelineResult:
        """Run pipeline on a single problem with error handling.

        Args:
            problem: Problem to solve.

        Returns:
            Pipeline result.
        """
        try:
            result = await self.pipeline.run(problem)
            logger.info(
                f"Completed {problem.feature_pair_key}: "
                f"reward={result.reward:.1f}, "
                f"time={result.wall_time:.1f}s"
            )
            return result
        except Exception as e:
            logger.error(f"Failed {problem.feature_pair_key}: {e}")
            return CooperBenchPipelineResult(
                problem=problem,
                error=str(e),
            )

    async def _save_results(
        self,
        run_dir: Path,
        experiment_results: ExperimentResults,
    ) -> None:
        """Save experiment results to disk.

        Args:
            run_dir: Directory for this experiment run.
            experiment_results: Collected results.
        """
        # Save summary
        summary_path = run_dir / "summary.json"
        summary_path.write_text(
            json.dumps(experiment_results.summary(), indent=2),
            encoding="utf-8",
        )

        # Save per-task results
        results_path = run_dir / "results.jsonl"
        with results_path.open("w", encoding="utf-8") as f:
            for result in experiment_results.results:
                line = json.dumps({
                    "repo": result.problem.repo,
                    "task_id": result.problem.task_id,
                    "features": result.problem.features,
                    "reward": result.reward,
                    "both_passed": (
                        result.eval_result.both_passed
                        if result.eval_result else False
                    ),
                    "merge_status": (
                        result.eval_result.merge_status
                        if result.eval_result else None
                    ),
                    "rounds": result.rounds_completed,
                    "messages": result.messages_exchanged,
                    "tokens": result.total_tokens,
                    "wall_time": result.wall_time,
                    "error": result.error,
                })
                f.write(line + "\n")

        logger.info(f"Results saved to {run_dir}")
