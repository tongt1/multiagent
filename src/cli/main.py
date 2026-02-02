"""Main CLI entry point for running pipelines."""

import argparse
import asyncio
import sys

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

from src.cli.runner import (
    display_batch_result,
    display_job_status,
    display_result,
    run_batch,
    run_single,
    run_status,
    run_submit,
)


def setup_logging(verbose: bool) -> None:
    """Setup loguru with Rich handler.

    Args:
        verbose: Enable debug logging if True
    """
    # Remove default logger
    logger.remove()

    # Add Rich handler
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        RichHandler(console=Console(stderr=True), rich_tracebacks=True),
        format="{message}",
        level=log_level,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run multi-agent solver-verifier-judge pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ========== RUN SUBCOMMAND (single problem) ==========
    run_parser = subparsers.add_parser(
        "run",
        help="Run pipeline on a single problem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with direct problem description
  python -m src.cli.main run "What is 2+2?"

  # Run with problem file
  python -m src.cli.main run config/problems/example.yaml
        """,
    )

    run_parser.add_argument(
        "problem",
        help="Problem description string OR path to problem YAML file",
    )

    run_parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline configuration YAML (default: config/pipeline.yaml)",
    )

    run_parser.add_argument(
        "--output-dir",
        help="Trajectory output directory (overrides config)",
    )

    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # ========== BATCH SUBCOMMAND (multiple problems) ==========
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run pipeline on multiple problems concurrently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on MATH dataset (10 problems)
  python -m src.cli.main batch --source math --limit 10

  # Run on HumanEval dataset with high concurrency
  python -m src.cli.main batch --source humaneval --concurrency 20

  # Run on local dataset
  python -m src.cli.main batch --source ./problems.yaml
        """,
    )

    batch_parser.add_argument(
        "--source",
        required=True,
        help='Dataset source: "math", "humaneval", or path to local YAML/JSON file',
    )

    batch_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of problems to process (default: all)",
    )

    batch_parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent executions (default: 10)",
    )

    batch_parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline configuration YAML (default: config/pipeline.yaml)",
    )

    batch_parser.add_argument(
        "--output-dir",
        help="Trajectory output directory (overrides config)",
    )

    batch_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # ========== SUBMIT SUBCOMMAND (distributed job) ==========
    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit a batch job to distributed infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit job using batch config
  python -m src.cli.main submit --batch-config config/batch_config.yaml

  # Submit with custom executor
  python -m src.cli.main submit --batch-config config/batch_config.yaml --executor kjobs
        """,
    )

    submit_parser.add_argument(
        "--batch-config",
        required=True,
        help="Path to batch configuration YAML",
    )

    submit_parser.add_argument(
        "--executor",
        default="kubernetes",
        choices=["kubernetes", "kjobs"],
        help="Executor type (default: kubernetes)",
    )

    submit_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # ========== STATUS SUBCOMMAND (check job status) ==========
    status_parser = subparsers.add_parser(
        "status",
        help="Check status of a distributed job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check job status once
  python -m src.cli.main status multiagent-batch-20240201-120000

  # Watch job status (polls every 10s)
  python -m src.cli.main status multiagent-batch-20240201-120000 --watch

  # Custom poll interval
  python -m src.cli.main status multiagent-batch-20240201-120000 --watch --poll-interval 5
        """,
    )

    status_parser.add_argument(
        "job_id",
        help="Job ID returned from submit command",
    )

    status_parser.add_argument(
        "--executor",
        default="kubernetes",
        choices=["kubernetes", "kjobs"],
        help="Executor type (default: kubernetes)",
    )

    status_parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch job status until completion",
    )

    status_parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Poll interval in seconds for --watch (default: 10.0)",
    )

    status_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Show help if no subcommand provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    return args


async def async_main() -> int:
    """Async main function.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    setup_logging(args.verbose)

    # Load environment variables
    load_dotenv()

    console = Console()

    try:
        # Check for API keys (optional warning)
        import os

        if not os.getenv("COHERE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            logger.warning(
                "No API keys found in environment. "
                "Set COHERE_API_KEY or OPENAI_API_KEY in .env file."
            )

        # Route to appropriate subcommand
        if args.command == "run":
            # Single problem execution
            logger.info(f"Loading configuration from: {args.config}")
            logger.info(f"Problem: {args.problem[:100]}...")

            result = await run_single(
                config_path=args.config,
                problem=args.problem,
                output_dir=args.output_dir,
            )

            # Display results
            display_result(result, console)

        elif args.command == "batch":
            # Batch execution
            logger.info(f"Loading configuration from: {args.config}")
            logger.info(
                f"Batch execution: source={args.source}, "
                f"limit={args.limit}, concurrency={args.concurrency}"
            )

            batch_result = await run_batch(
                config_path=args.config,
                source=args.source,
                limit=args.limit,
                concurrency=args.concurrency,
                output_dir=args.output_dir,
                console=console,
            )

            # Display batch summary
            display_batch_result(batch_result, console)

        elif args.command == "submit":
            # Distributed job submission
            logger.info(f"Loading batch config from: {args.batch_config}")
            logger.info(f"Using executor: {args.executor}")

            job_id = await run_submit(
                batch_config_path=args.batch_config,
                executor_type=args.executor,
            )

            console.print(f"[green]✓[/green] Job submitted: [bold]{job_id}[/bold]")
            console.print(f"\nCheck status with: python -m src.cli.main status {job_id}")

        elif args.command == "status":
            # Job status check
            logger.info(f"Checking status for job: {args.job_id}")

            await run_status(
                job_id=args.job_id,
                executor_type=args.executor,
                watch=args.watch,
                poll_interval=args.poll_interval,
                console=console,
            )

        return 0

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}", style="bold red")
        return 1

    except KeyError as e:
        console.print(
            f"[red]Error:[/red] Missing configuration field: {e}",
            style="bold red",
        )
        return 1

    except Exception as e:
        console.print(f"[red]Pipeline Error:[/red] {e}", style="bold red")
        logger.exception("Pipeline execution failed")
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
