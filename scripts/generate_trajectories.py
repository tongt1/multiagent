"""Generate MATH 500 trajectories for debate and baseline conditions.

Usage:
    python scripts/generate_trajectories.py --mode debate --config config/pipeline_math.yaml
    python scripts/generate_trajectories.py --mode baseline --config config/pipeline_math.yaml
    python scripts/generate_trajectories.py --mode both --config config/pipeline_math.yaml
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.data.math500 import load_math500, get_math500_stats
from src.models.config import PipelineConfig
from src.orchestration.batch_executor import BatchPipelineExecutor, BatchResult
from src.orchestration.baseline_runner import BaselineRunner
from src.data.dataset_loader import Problem


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate MATH 500 trajectories for debate and baseline conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["debate", "baseline", "both"],
        help="Generation mode: debate, baseline, or both",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline configuration YAML",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of problems (for testing; default: all 500)",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent API calls (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        default="trajectories",
        help="Base output directory (default: trajectories)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


async def run_debate_mode(
    problems: list[Problem],
    config: PipelineConfig,
    output_dir: Path,
    concurrency: int,
    console: Console,
) -> BatchResult:
    """Run trajectory generation in debate mode.

    Args:
        problems: List of MATH 500 problems
        config: Pipeline configuration
        output_dir: Output directory for debate trajectories
        concurrency: Max concurrent executions
        console: Rich console for output

    Returns:
        BatchResult from execution
    """
    logger.info(f"Running debate mode: {len(problems)} problems")

    # Ensure config is in debate mode
    config.mode = "debate"
    config.trajectory_output_dir = str(output_dir)

    # Create batch executor
    executor = BatchPipelineExecutor(config, max_concurrent=concurrency)

    # Run batch execution with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Debate mode...", total=len(problems))

        completed_count = 0

        def on_complete(problem: Problem, result: Any) -> None:
            """Callback for progress updates."""
            nonlocal completed_count
            completed_count += 1
            progress.update(task, advance=1)

        batch_result = await executor.run_batch(problems, on_complete=on_complete)

    return batch_result


async def run_baseline_mode(
    problems: list[Problem],
    config: PipelineConfig,
    output_dir: Path,
    concurrency: int,
    console: Console,
) -> BatchResult:
    """Run trajectory generation in baseline mode.

    Args:
        problems: List of MATH 500 problems
        config: Pipeline configuration
        output_dir: Output directory for baseline trajectories
        concurrency: Max concurrent executions
        console: Rich console for output

    Returns:
        BatchResult from execution
    """
    logger.info(f"Running baseline mode: {len(problems)} problems")

    # Ensure config is in baseline mode
    config.mode = "baseline"
    config.trajectory_output_dir = str(output_dir)

    # For baseline, we need to run each problem through BaselineRunner
    # and collect results manually since BaselineRunner doesn't have batch executor
    from src.orchestration.pipeline import PipelineResult

    semaphore = asyncio.Semaphore(concurrency)
    results: list[PipelineResult] = []
    errors: list[dict[str, Any]] = []

    start_time = time.monotonic()

    async def run_single_baseline(problem: Problem, progress: Progress, task_id: Any) -> None:
        """Run single baseline problem."""
        async with semaphore:
            try:
                # Create fresh baseline runner
                runner = BaselineRunner(config)

                # Build metadata
                metadata = {
                    "problem_id": problem.id,
                    "domain": problem.domain,
                    **problem.metadata,
                }
                if problem.ground_truth:
                    metadata["ground_truth"] = problem.ground_truth

                # Run baseline
                result = await runner.run(
                    problem_description=problem.problem,
                    problem_metadata=metadata,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Baseline failed for {problem.id}: {e}")
                errors.append({
                    "problem_id": problem.id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                })
            finally:
                progress.update(task_id, advance=1)

    # Run with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Baseline mode...", total=len(problems))

        tasks = [run_single_baseline(problem, progress, task) for problem in problems]
        await asyncio.gather(*tasks)

    elapsed = time.monotonic() - start_time

    # Create batch result
    batch_result = BatchResult(
        total=len(problems),
        succeeded=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
        elapsed_seconds=elapsed,
    )

    return batch_result


def print_summary(
    mode: str,
    batch_result: BatchResult,
    output_dir: Path,
    console: Console,
) -> None:
    """Print generation summary.

    Args:
        mode: Generation mode (debate or baseline)
        batch_result: Batch execution result
        output_dir: Output directory
        console: Rich console for output
    """
    from rich.table import Table

    # Create summary table
    table = Table(title=f"{mode.capitalize()} Mode Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total problems", str(batch_result.total))
    table.add_row("Succeeded", f"[green]{batch_result.succeeded}[/green]")
    table.add_row("Failed", f"[red]{batch_result.failed}[/red]")
    table.add_row("Success rate", f"{(batch_result.succeeded / batch_result.total * 100):.1f}%")
    table.add_row("Elapsed time", f"{batch_result.elapsed_seconds:.2f}s")

    # Calculate average metrics from results
    if batch_result.results:
        avg_cost = sum(r.total_cost for r in batch_result.results) / len(batch_result.results)
        avg_tokens = sum(r.token_usage.total_tokens for r in batch_result.results) / len(batch_result.results)

        # Count ground truth rewards
        correct_count = sum(1 for r in batch_result.results if r.ground_truth_reward == 1.0)
        incorrect_count = sum(1 for r in batch_result.results if r.ground_truth_reward == 0.0)

        table.add_row("Avg cost per problem", f"${avg_cost:.4f}")
        table.add_row("Avg tokens per problem", f"{avg_tokens:.0f}")
        table.add_row("Correct solutions", f"[green]{correct_count}[/green]")
        table.add_row("Incorrect solutions", f"[red]{incorrect_count}[/red]")
        table.add_row("Accuracy", f"{(correct_count / len(batch_result.results) * 100):.1f}%")

    table.add_row("Output directory", str(output_dir))

    console.print(table)

    # Show sample errors if any
    if batch_result.errors:
        console.print(f"\n[red]Errors encountered:[/red]")
        for error in batch_result.errors[:5]:  # Show first 5
            console.print(f"  - {error['problem_id']}: {error['error_message']}")
        if len(batch_result.errors) > 5:
            console.print(f"  ... and {len(batch_result.errors) - 5} more")


def save_generation_metadata(
    mode: str,
    batch_result: BatchResult,
    output_dir: Path,
    problems: list[Problem],
    args: argparse.Namespace,
) -> None:
    """Save generation metadata to JSON.

    Args:
        mode: Generation mode
        batch_result: Batch execution result
        output_dir: Output directory
        problems: List of problems
        args: Command line arguments
    """
    metadata = {
        "mode": mode,
        "total_problems": len(problems),
        "succeeded": batch_result.succeeded,
        "failed": batch_result.failed,
        "elapsed_seconds": batch_result.elapsed_seconds,
        "config_path": args.config,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    # Add accuracy if available
    if batch_result.results:
        correct_count = sum(1 for r in batch_result.results if r.ground_truth_reward == 1.0)
        metadata["correct_solutions"] = correct_count
        metadata["accuracy"] = correct_count / len(batch_result.results)
        metadata["avg_cost_per_problem"] = sum(r.total_cost for r in batch_result.results) / len(batch_result.results)
        metadata["avg_tokens_per_problem"] = sum(r.token_usage.total_tokens for r in batch_result.results) / len(batch_result.results)

    metadata_path = output_dir / "generation_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved generation metadata to {metadata_path}")


async def async_main() -> int:
    """Async main function."""
    args = parse_args()
    console = Console()

    # Setup logging
    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Set random seed
    random.seed(args.seed)

    # Load MATH 500 dataset
    console.print("[cyan]Loading MATH 500 dataset...[/cyan]")
    problems = load_math500(seed=args.seed)

    # Apply limit if specified
    if args.limit:
        problems = problems[:args.limit]
        console.print(f"[yellow]Limited to first {args.limit} problems[/yellow]")

    # Show dataset statistics
    stats = get_math500_stats(problems)
    console.print(f"[green]Loaded {stats['total']} problems[/green]")
    console.print(f"Distribution by level: {dict(stats['per_level'])}")

    # Assign sequential trajectory IDs to problems
    for idx, problem in enumerate(problems):
        problem.metadata["trajectory_id"] = idx

    # Load pipeline config
    from pydantic_yaml import parse_yaml_file_as
    config = parse_yaml_file_as(PipelineConfig, args.config)

    # Create base output directory
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Run generation based on mode
    start_time = time.time()

    try:
        if args.mode == "debate":
            output_dir = output_base / "debate"
            output_dir.mkdir(parents=True, exist_ok=True)

            batch_result = await run_debate_mode(
                problems=problems,
                config=config,
                output_dir=output_dir,
                concurrency=args.concurrency,
                console=console,
            )

            print_summary("debate", batch_result, output_dir, console)
            save_generation_metadata("debate", batch_result, output_dir, problems, args)

        elif args.mode == "baseline":
            output_dir = output_base / "baseline"
            output_dir.mkdir(parents=True, exist_ok=True)

            batch_result = await run_baseline_mode(
                problems=problems,
                config=config,
                output_dir=output_dir,
                concurrency=args.concurrency,
                console=console,
            )

            print_summary("baseline", batch_result, output_dir, console)
            save_generation_metadata("baseline", batch_result, output_dir, problems, args)

        elif args.mode == "both":
            # Run debate mode
            console.print("\n[bold cyan]===== DEBATE MODE =====[/bold cyan]")
            debate_dir = output_base / "debate"
            debate_dir.mkdir(parents=True, exist_ok=True)

            debate_result = await run_debate_mode(
                problems=problems,
                config=config,
                output_dir=debate_dir,
                concurrency=args.concurrency,
                console=console,
            )

            print_summary("debate", debate_result, debate_dir, console)
            save_generation_metadata("debate", debate_result, debate_dir, problems, args)

            # Run baseline mode
            console.print("\n[bold yellow]===== BASELINE MODE =====[/bold yellow]")
            baseline_dir = output_base / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            baseline_result = await run_baseline_mode(
                problems=problems,
                config=config,
                output_dir=baseline_dir,
                concurrency=args.concurrency,
                console=console,
            )

            print_summary("baseline", baseline_result, baseline_dir, console)
            save_generation_metadata("baseline", baseline_result, baseline_dir, problems, args)

            # Print combined summary
            console.print("\n[bold green]===== OVERALL SUMMARY =====[/bold green]")
            console.print(f"Total elapsed time: {time.time() - start_time:.2f}s")
            console.print(f"Debate: {debate_result.succeeded}/{debate_result.total} succeeded")
            console.print(f"Baseline: {baseline_result.succeeded}/{baseline_result.total} succeeded")

        console.print("\n[green]✓[/green] Generation complete!")
        return 0

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.exception("Generation failed")
        return 1


def main() -> None:
    """Main entry point."""
    try:
        exit_code = asyncio.run(async_main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
