"""CLI runner for executing pipelines."""

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.data.dataset_loader import DatasetLoader, Problem
from src.models.config import PipelineConfig
from src.orchestration.batch_executor import BatchPipelineExecutor, BatchResult
from src.orchestration.pipeline import PipelineResult, SolverVerifierJudgePipeline


async def run_single(
    config_path: str,
    problem: str,
    output_dir: str | None = None,
) -> PipelineResult:
    """Run a single pipeline execution.

    Args:
        config_path: Path to pipeline configuration YAML
        problem: Problem description string OR path to problem YAML file
        output_dir: Optional trajectory output directory (overrides config)

    Returns:
        PipelineResult with execution details
    """
    # Load pipeline configuration
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML manually and pass to PipelineConfig
    with open(config_file) as f:
        config_data = yaml.safe_load(f)

    # Override trajectory output dir if provided
    if output_dir:
        config_data["trajectory_output_dir"] = output_dir

    # Create PipelineConfig from loaded data
    config = PipelineConfig(**config_data)

    # Load problem
    problem_description: str
    problem_metadata: dict[str, Any] | None = None

    # Check if problem is a file path
    problem_path = Path(problem)
    if problem_path.exists() and problem_path.suffix in [".yaml", ".yml", ".json"]:
        with open(problem_path) as f:
            if problem_path.suffix == ".json":
                problem_data = json.load(f)
            else:
                problem_data = yaml.safe_load(f)

        problem_description = problem_data.get("description", "")
        problem_metadata = problem_data.get("metadata")
    else:
        # Treat as direct problem description
        problem_description = problem

    # Create and run pipeline
    pipeline = SolverVerifierJudgePipeline(config)
    result = await pipeline.run(problem_description, problem_metadata)

    return result


def display_result(result: PipelineResult, console: Console) -> None:
    """Display formatted pipeline result.

    Args:
        result: PipelineResult to display
        console: Rich Console instance
    """
    # Truncate problem description
    problem_display = result.problem_description
    if len(problem_display) > 200:
        problem_display = problem_display[:197] + "..."

    # Verification status
    if result.passed_verification:
        verify_icon = "[green]✓[/green]"
        verify_text = "[green]PASSED[/green]"
    else:
        verify_icon = "[red]✗[/red]"
        verify_text = "[red]FAILED[/red]"

    # Judge score with color
    score = result.judge_score
    if score >= 0.7:
        score_color = "green"
    elif score >= 0.4:
        score_color = "yellow"
    else:
        score_color = "red"
    score_text = f"[{score_color}]{score:.3f}[/{score_color}]"

    # Create main info panel
    info_lines = [
        f"[bold]Problem:[/bold] {problem_display}",
        f"[bold]Verification:[/bold] {verify_icon} {verify_text}",
        f"[bold]Judge Score:[/bold] {score_text}",
        f"[bold]Iterations:[/bold] {result.iterations}",
        f"[bold]Total Cost:[/bold] ${result.total_cost:.6f}",
        f"[bold]Trajectory:[/bold] {result.trajectory_path}",
    ]

    console.print(Panel("\n".join(info_lines), title="Pipeline Result", expand=False))

    # Per-agent cost breakdown table
    cost_summary = result.cost_summary
    by_agent = cost_summary.get("by_agent", {})

    if by_agent:
        agent_table = Table(title="Per-Agent Token and Cost Breakdown", show_header=True)
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Prompt Tokens", justify="right")
        agent_table.add_column("Completion Tokens", justify="right")
        agent_table.add_column("Total Tokens", justify="right")
        agent_table.add_column("Cost (USD)", justify="right")

        for agent_name in ["solver", "verifier", "judge"]:
            if agent_name in by_agent:
                agent_data = by_agent[agent_name]
                tokens = agent_data["tokens"]
                cost = agent_data["cost_usd"]

                agent_table.add_row(
                    agent_name.capitalize(),
                    f"{tokens['prompt_tokens']:,}",
                    f"{tokens['completion_tokens']:,}",
                    f"{tokens['total_tokens']:,}",
                    f"${cost:.6f}",
                )

        console.print(agent_table)

    # Per-model breakdown if multiple models used
    by_model = cost_summary.get("by_model", {})
    if len(by_model) > 1:
        model_table = Table(title="Per-Model Breakdown", show_header=True)
        model_table.add_column("Model", style="magenta")
        model_table.add_column("Total Tokens", justify="right")
        model_table.add_column("Cost (USD)", justify="right")

        for model_name, model_data in by_model.items():
            tokens = model_data["tokens"]
            cost = model_data["cost_usd"]

            model_table.add_row(
                model_name,
                f"{tokens['total_tokens']:,}",
                f"${cost:.6f}",
            )

        console.print(model_table)


async def run_batch(
    config_path: str,
    source: str,
    limit: int | None,
    concurrency: int,
    output_dir: str | None,
    console: Console,
) -> BatchResult:
    """Run batch pipeline execution.

    Args:
        config_path: Path to pipeline configuration YAML
        source: Dataset source ("math", "humaneval", or file path)
        limit: Maximum number of problems to process
        concurrency: Maximum concurrent executions
        output_dir: Optional trajectory output directory (overrides config)
        console: Rich Console for progress display

    Returns:
        BatchResult with execution statistics
    """
    # Load pipeline configuration
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML manually and pass to PipelineConfig
    with open(config_file) as f:
        config_data = yaml.safe_load(f)

    # Override trajectory output dir if provided
    if output_dir:
        config_data["trajectory_output_dir"] = output_dir

    # Create PipelineConfig from loaded data
    config = PipelineConfig(**config_data)

    # Load dataset
    loader = DatasetLoader()
    logger.info(f"Loading dataset from source: {source}")
    problems = loader.load(source, limit=limit)

    if not problems:
        raise ValueError(f"No problems loaded from source: {source}")

    logger.info(f"Loaded {len(problems)} problems")

    # Create batch executor
    executor = BatchPipelineExecutor(config, max_concurrent=concurrency)

    # Setup progress tracking
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    )

    completed_count = 0

    def on_complete(problem: Problem, result: PipelineResult | Exception) -> None:
        """Progress callback called after each problem completes."""
        nonlocal completed_count
        completed_count += 1

        if isinstance(result, Exception):
            status_msg = f"[red]✗[/red] {problem.id}: Failed"
        else:
            score = result.judge_score
            if score >= 0.7:
                icon = "[green]✓[/green]"
            elif score >= 0.4:
                icon = "[yellow]~[/yellow]"
            else:
                icon = "[red]✗[/red]"
            status_msg = f"{icon} {problem.id}: score={score:.2f}"

        progress.update(task_id, advance=1, description=status_msg)

    # Run batch with progress
    with progress:
        task_id = progress.add_task(
            "[cyan]Starting batch...", total=len(problems)
        )

        batch_result = await executor.run_batch(problems, on_complete=on_complete)

    return batch_result


def display_batch_result(result: BatchResult, console: Console) -> None:
    """Display formatted batch execution result.

    Args:
        result: BatchResult to display
        console: Rich Console instance
    """
    # Summary statistics
    success_rate = (result.succeeded / result.total * 100) if result.total > 0 else 0

    # Calculate average score from successful results
    avg_score = 0.0
    if result.succeeded > 0:
        avg_score = sum(r.judge_score for r in result.results) / len(result.results)

    # Calculate total cost
    total_cost = sum(r.total_cost for r in result.results)

    # Summary panel
    summary_lines = [
        f"[bold]Total Problems:[/bold] {result.total}",
        f"[bold]Succeeded:[/bold] [green]{result.succeeded}[/green]",
        f"[bold]Failed:[/bold] [red]{result.failed}[/red]",
        f"[bold]Success Rate:[/bold] {success_rate:.1f}%",
        f"[bold]Average Score:[/bold] {avg_score:.3f}",
        f"[bold]Total Cost:[/bold] ${total_cost:.6f}",
        f"[bold]Elapsed Time:[/bold] {result.elapsed_seconds:.2f}s",
    ]

    console.print(Panel("\n".join(summary_lines), title="Batch Execution Summary", expand=False))

    # Results distribution table
    if result.results:
        dist_table = Table(title="Score Distribution", show_header=True)
        dist_table.add_column("Score Range", style="cyan")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Percentage", justify="right")

        # Calculate distribution
        excellent = sum(1 for r in result.results if r.judge_score >= 0.7)
        good = sum(1 for r in result.results if 0.4 <= r.judge_score < 0.7)
        poor = sum(1 for r in result.results if r.judge_score < 0.4)

        total_succeeded = len(result.results)

        dist_table.add_row(
            "[green]Excellent (≥0.7)[/green]",
            str(excellent),
            f"{excellent / total_succeeded * 100:.1f}%",
        )
        dist_table.add_row(
            "[yellow]Good (0.4-0.7)[/yellow]",
            str(good),
            f"{good / total_succeeded * 100:.1f}%",
        )
        dist_table.add_row(
            "[red]Poor (<0.4)[/red]",
            str(poor),
            f"{poor / total_succeeded * 100:.1f}%",
        )

        console.print(dist_table)

    # Errors summary if any
    if result.errors:
        error_table = Table(title="Errors", show_header=True)
        error_table.add_column("Problem ID", style="red")
        error_table.add_column("Error Type")
        error_table.add_column("Error Message")

        for error in result.errors[:10]:  # Show first 10 errors
            error_table.add_row(
                error["problem_id"],
                error["error_type"],
                error["error_message"][:100] + ("..." if len(error["error_message"]) > 100 else ""),
            )

        if len(result.errors) > 10:
            error_table.caption = f"Showing first 10 of {len(result.errors)} errors"

        console.print(error_table)
