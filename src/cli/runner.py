"""CLI runner for executing pipelines."""

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.models.config import PipelineConfig
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
