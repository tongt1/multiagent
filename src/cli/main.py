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

    # ========== GENERATE SUBCOMMAND (MATH 500 trajectory generation) ==========
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate MATH 500 trajectories for debate and baseline modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate debate trajectories
  python -m src.cli.main generate --mode debate --config config/pipeline_math.yaml

  # Generate baseline trajectories
  python -m src.cli.main generate --mode baseline --config config/pipeline_math.yaml

  # Generate both modes
  python -m src.cli.main generate --mode both --config config/pipeline_math.yaml

  # Test with limited problems
  python -m src.cli.main generate --mode both --config config/pipeline_math.yaml --limit 10
        """,
    )

    generate_parser.add_argument(
        "--mode",
        required=True,
        choices=["debate", "baseline", "both"],
        help="Generation mode: debate, baseline, or both",
    )

    generate_parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline configuration YAML",
    )

    generate_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of problems (for testing; default: all 500)",
    )

    generate_parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent API calls (default: 10)",
    )

    generate_parser.add_argument(
        "--output-dir",
        default="trajectories",
        help="Base output directory (default: trajectories)",
    )

    generate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    generate_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # ========== EXPORT-MARTI SUBCOMMAND (trajectory export) ==========
    export_marti_parser = subparsers.add_parser(
        "export-marti",
        help="Export trajectories to MARTI-compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export trajectory to MARTI format
  python -m src.cli.main export-marti --input trajectories/run_001.jsonl --output marti_trajectories.jsonl

  # Export with custom number of solvers
  python -m src.cli.main export-marti --input trajectories/run_001.jsonl --output marti_trajectories.jsonl --num-solvers 5
        """,
    )

    export_marti_parser.add_argument(
        "--input",
        required=True,
        help="Path to input trajectory JSONL file",
    )

    export_marti_parser.add_argument(
        "--output",
        required=True,
        help="Path to output MARTI JSONL file",
    )

    export_marti_parser.add_argument(
        "--num-solvers",
        type=int,
        default=3,
        help="Number of solver agents (default: 3)",
    )

    export_marti_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # ========== TRAIN SUBCOMMAND (training job submission) ==========
    train_parser = subparsers.add_parser(
        "train",
        help="Submit multi-agent RL training job to Ray cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit training job
  python -m src.cli.main train trajectories/marti_train.jsonl \\
    --pretrain-path models/cohere-7b \\
    --save-path checkpoints/run_001

  # With custom algorithm and reward shaping
  python -m src.cli.main train trajectories/marti_train.jsonl \\
    --pretrain-path models/cohere-7b \\
    --save-path checkpoints/run_001 \\
    --algorithm ppo \\
    --reward-mode quality \\
    --alpha 0.7

  # Wait for completion
  python -m src.cli.main train trajectories/marti_train.jsonl \\
    --pretrain-path models/cohere-7b \\
    --save-path checkpoints/run_001 \\
    --wait
        """,
    )

    train_parser.add_argument(
        "trajectory_path",
        help="Path to MARTI trajectory JSONL file",
    )

    train_parser.add_argument(
        "--pretrain-path",
        required=True,
        help="Base model checkpoint path",
    )

    train_parser.add_argument(
        "--save-path",
        required=True,
        help="Output checkpoint directory",
    )

    train_parser.add_argument(
        "--ray-address",
        default="http://localhost:8265",
        help="Ray cluster dashboard address (default: http://localhost:8265)",
    )

    train_parser.add_argument(
        "--algorithm",
        default="reinforce",
        choices=["ppo", "grpo", "reinforce", "rloo"],
        help="Training algorithm (default: reinforce)",
    )

    train_parser.add_argument(
        "--reward-mode",
        default="margin",
        choices=["quality", "margin", "none"],
        help="Reward shaping mode (default: margin)",
    )

    train_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Reward shaping alpha parameter (default: 0.5)",
    )

    train_parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of training episodes (default: 5)",
    )

    train_parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        default=2,
        help="GPUs per node (default: 2)",
    )

    train_parser.add_argument(
        "--num-nodes",
        type=int,
        default=3,
        help="Number of nodes (default: 3)",
    )

    train_parser.add_argument(
        "--wandb-project",
        help="Weights & Biases project name (optional)",
    )

    train_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for job completion",
    )

    train_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # ========== TRAIN-STATUS SUBCOMMAND (training job status) ==========
    train_status_parser = subparsers.add_parser(
        "train-status",
        help="Check training job status and view logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check job status once
  python -m src.cli.main train-status marti-train-20260201-123456

  # Follow logs until completion
  python -m src.cli.main train-status marti-train-20260201-123456 --follow
        """,
    )

    train_status_parser.add_argument(
        "job_id",
        help="Training job ID",
    )

    train_status_parser.add_argument(
        "--ray-address",
        default="http://localhost:8265",
        help="Ray cluster dashboard address (default: http://localhost:8265)",
    )

    train_status_parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow logs until completion",
    )

    train_status_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # ========== COOPERBENCH SUBCOMMAND ==========
    cooperbench_parser = subparsers.add_parser(
        "cooperbench",
        help="Run CooperBench cooperative coding evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run lite subset (100 pairs) in coop mode
  python -m src.cli.main cooperbench --config configs/cooperbench_default.yaml

  # Run with custom dataset path
  python -m src.cli.main cooperbench --config configs/cooperbench_default.yaml --dataset-path repos/CooperBench/dataset

  # Run solo baseline
  python -m src.cli.main cooperbench --config configs/cooperbench_default.yaml --mode solo

  # Limit to 5 problems for testing
  python -m src.cli.main cooperbench --config configs/cooperbench_default.yaml --limit 5

  # Dry run (skip LLM calls, use mock patches)
  python -m src.cli.main cooperbench --config configs/cooperbench_default.yaml --dry-run --limit 3
        """,
    )

    cooperbench_parser.add_argument(
        "--config",
        default="configs/cooperbench_default.yaml",
        help="Path to CooperBench configuration YAML (default: configs/cooperbench_default.yaml)",
    )

    cooperbench_parser.add_argument(
        "--dataset-path",
        help="Override dataset path from config",
    )

    cooperbench_parser.add_argument(
        "--mode",
        choices=["coop", "solo"],
        help="Override mode from config (coop or solo)",
    )

    cooperbench_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of problems to evaluate",
    )

    cooperbench_parser.add_argument(
        "--output-dir",
        default="cooperbench_results",
        help="Output directory for results (default: cooperbench_results)",
    )

    cooperbench_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: load dataset and create pipeline but skip LLM calls",
    )

    cooperbench_parser.add_argument(
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

        elif args.command == "generate":
            # MATH 500 trajectory generation
            logger.info(f"Generating MATH 500 trajectories in {args.mode} mode")
            logger.info(f"Configuration: {args.config}")

            # Import the generation script's main logic
            import random
            import time
            from pathlib import Path
            from pydantic_yaml import parse_yaml_file_as

            from src.data.math500 import load_math500, get_math500_stats
            from src.orchestration.batch_executor import BatchPipelineExecutor
            from src.orchestration.baseline_runner import BaselineRunner

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
            config = parse_yaml_file_as(PipelineConfig, args.config)

            # Create base output directory
            output_base = Path(args.output_dir)
            output_base.mkdir(parents=True, exist_ok=True)

            # Import generation helper functions
            from scripts.generate_trajectories import (
                run_debate_mode,
                run_baseline_mode,
                print_summary,
                save_generation_metadata,
            )

            start_time = time.time()

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

        elif args.command == "export-marti":
            # MARTI trajectory export
            from pathlib import Path
            from src.data.training_export import export_to_marti_jsonl
            from src.training.marti_exporter import build_agent_graph

            logger.info(f"Exporting trajectories from: {args.input}")
            logger.info(f"Output to: {args.output}")
            logger.info(f"Number of solvers: {args.num_solvers}")

            # Build agent graph
            agent_graph = build_agent_graph(num_solvers=args.num_solvers)

            # Export trajectories
            num_exported = export_to_marti_jsonl(
                trajectory_path=Path(args.input),
                output_path=Path(args.output),
                agent_graph=agent_graph,
            )

            console.print(f"[green]✓[/green] Exported {num_exported} trajectories to MARTI format")
            console.print(f"Output file: {args.output}")

        elif args.command == "train":
            # Training job submission
            from pathlib import Path
            from src.infrastructure.ray_training_executor import submit_training_job
            from src.training.training_config import (
                build_default_config,
                RewardShapingConfig,
                RewardShapingMode,
                TrainingAlgorithm,
            )

            logger.info(f"Submitting training job")
            logger.info(f"Trajectory: {args.trajectory_path}")
            logger.info(f"Pretrain: {args.pretrain_path}")
            logger.info(f"Save path: {args.save_path}")
            logger.info(f"Ray address: {args.ray_address}")

            # Check trajectory file exists
            trajectory_path = Path(args.trajectory_path)
            if not trajectory_path.exists():
                console.print(f"[red]Error:[/red] Trajectory file not found: {args.trajectory_path}")
                return 1

            # Build training config
            config = build_default_config(
                pretrain_path=args.pretrain_path,
                trajectory_path=args.trajectory_path,
                save_path=args.save_path,
            )

            # Override with CLI parameters
            config.ray_config.num_nodes = args.num_nodes
            config.ray_config.num_gpus_per_node = args.num_gpus_per_node
            config.openrlhf_config.algorithm = TrainingAlgorithm(args.algorithm)
            config.openrlhf_config.num_episodes = args.num_episodes
            config.reward_shaping = RewardShapingConfig(
                mode=RewardShapingMode(args.reward_mode),
                alpha=args.alpha,
            )
            if args.wandb_project:
                config.wandb_project = args.wandb_project

            # Validate config
            from src.training.training_config import validate_config
            errors = validate_config(config)
            if errors:
                console.print(f"[red]Configuration Error:[/red]")
                for error in errors:
                    console.print(f"  - {error}")
                return 1

            # Submit job
            try:
                job_id = submit_training_job(config, ray_address=args.ray_address)

                console.print(f"[green]✓[/green] Training job submitted: [bold]{job_id}[/bold]")
                console.print(f"Ray dashboard: {args.ray_address}")
                console.print(f"\nCheck status with: python -m src.cli.main train-status {job_id}")

                # Wait for completion if requested
                if args.wait:
                    console.print("\n[yellow]Waiting for job completion...[/yellow]")
                    from src.infrastructure.ray_training_executor import (
                        RayTrainingExecutor,
                    )

                    executor = RayTrainingExecutor(args.ray_address)

                    import time
                    while True:
                        job_info = executor.get_training_job_status(job_id)

                        if job_info.status.value == "succeeded":
                            console.print(f"[green]✓[/green] Training job succeeded!")
                            break
                        elif job_info.status.value == "failed":
                            console.print(f"[red]✗[/red] Training job failed: {job_info.message}")
                            return 1
                        else:
                            console.print(f"Status: {job_info.status.value}")
                            time.sleep(10)

            except ImportError as e:
                console.print(f"[red]Error:[/red] {e}")
                console.print("Install with: pip install 'ray[client]'")
                return 1
            except Exception as e:
                console.print(f"[red]Training Error:[/red] {e}")
                logger.exception("Training job submission failed")
                return 1

        elif args.command == "train-status":
            # Training job status
            from src.infrastructure.ray_training_executor import RayTrainingExecutor
            from rich.table import Table

            logger.info(f"Checking training job status: {args.job_id}")

            try:
                executor = RayTrainingExecutor(args.ray_address)
                job_info = executor.get_training_job_status(args.job_id)

                # Display job info table
                table = Table(title=f"Training Job: {args.job_id}")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")

                table.add_row("Job Name", job_info.job_name)
                table.add_row("Status", job_info.status.value)
                if job_info.created_at:
                    table.add_row("Created", job_info.created_at)
                if job_info.completed_at:
                    table.add_row("Completed", job_info.completed_at)
                if job_info.message:
                    table.add_row("Message", job_info.message)

                console.print(table)

                # Follow logs if requested
                if args.follow:
                    console.print("\n[yellow]Following logs...[/yellow]\n")

                    import time
                    last_logs = ""

                    while True:
                        job_info = executor.get_training_job_status(args.job_id)
                        logs = executor.get_training_job_logs(args.job_id)

                        # Print new logs
                        if logs != last_logs:
                            new_content = logs[len(last_logs):]
                            console.print(new_content, end="")
                            last_logs = logs

                        # Check if job completed
                        if job_info.status.value in ("succeeded", "failed"):
                            if job_info.status.value == "succeeded":
                                console.print("\n[green]✓[/green] Training job succeeded!")
                            else:
                                console.print(f"\n[red]✗[/red] Training job failed: {job_info.message}")
                            break

                        time.sleep(10)

            except ImportError as e:
                console.print(f"[red]Error:[/red] {e}")
                console.print("Install with: pip install 'ray[client]'")
                return 1
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                logger.exception("Failed to get training job status")
                return 1

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
