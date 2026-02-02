"""Main CLI entry point for running pipelines."""

import argparse
import asyncio
import sys

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

from src.cli.runner import display_result, run_single


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
        epilog="""
Examples:
  # Run with direct problem description
  python -m src.cli.main "What is 2+2?"

  # Run with problem file
  python -m src.cli.main config/problems/example.yaml

  # Use custom config and output directory
  python -m src.cli.main "Solve this..." --config my_config.yaml --output-dir ./outputs
        """,
    )

    parser.add_argument(
        "problem",
        help="Problem description string OR path to problem YAML file",
    )

    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline configuration YAML (default: config/pipeline.yaml)",
    )

    parser.add_argument(
        "--output-dir",
        help="Trajectory output directory (overrides config)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


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

        # Run pipeline
        logger.info(f"Loading configuration from: {args.config}")
        logger.info(f"Problem: {args.problem[:100]}...")

        result = await run_single(
            config_path=args.config,
            problem=args.problem,
            output_dir=args.output_dir,
        )

        # Display results
        display_result(result, console)

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
