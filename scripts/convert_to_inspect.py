#!/usr/bin/env python3
"""Convert trajectory JSONL files to Inspect AI .eval files.

Converts baseline trajectory data into .eval files viewable in ``inspect view``
with full conversation history, dual scorers (ground_truth + judge), and
problem metadata.

Usage examples:
    # Convert a single JSONL file:
    python scripts/convert_to_inspect.py \\
        --input-file data/sample_trajectories.jsonl \\
        --output-dir ./inspect_logs/

    # Convert all JSONL files in a directory:
    python scripts/convert_to_inspect.py \\
        --input-dir experiments/run_123/ \\
        --output-dir experiments/run_123/inspect_logs/

    # With optional overrides:
    python scripts/convert_to_inspect.py \\
        --input-file data/sample_trajectories.jsonl \\
        --output-dir ./inspect_logs/ \\
        --task-name my_eval \\
        --model-name gpt-4o \\
        --system-prompt "You are a math solver."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.infrastructure.inspect_logging.converter import convert_trajectories


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert trajectory JSONL files to Inspect .eval files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing .jsonl trajectory files. All .jsonl files are converted.",
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Single .jsonl trajectory file to convert.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for .eval files.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="baseline_eval",
        help="Task name for EvalSpec (default: baseline_eval).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system prompt to include as ChatMessageSystem.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name. If not provided, auto-detected from trajectory metadata.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the trajectory-to-Inspect conversion."""
    args = parse_args(argv)

    # Collect JSONL files to convert
    if args.input_file:
        if not args.input_file.exists():
            logger.error(f"Input file not found: {args.input_file}")
            return 1
        jsonl_files = [args.input_file]
    else:
        if not args.input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            return 1
        jsonl_files = sorted(args.input_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.error(f"No .jsonl files found in {args.input_dir}")
            return 1

    logger.info(f"Converting {len(jsonl_files)} file(s) to Inspect .eval format")

    total_samples = 0
    converted_count = 0
    errors = []

    for jsonl_path in jsonl_files:
        try:
            output_path = convert_trajectories(
                jsonl_path=jsonl_path,
                output_dir=args.output_dir,
                task_name=args.task_name,
                model_name=args.model_name,
                system_prompt=args.system_prompt,
            )
            # Read back to count samples for summary
            from inspect_ai.log import read_eval_log

            log = read_eval_log(str(output_path))
            n_samples = len(log.samples) if log.samples else 0
            total_samples += n_samples
            converted_count += 1
            logger.info(f"  {jsonl_path.name} -> {output_path.name} ({n_samples} samples)")
        except Exception as e:
            logger.error(f"  Failed to convert {jsonl_path.name}: {e}")
            errors.append((jsonl_path.name, str(e)))

    # Print summary
    print()
    print("=" * 60)
    print(f"Conversion complete:")
    print(f"  Files converted: {converted_count}/{len(jsonl_files)}")
    print(f"  Total samples:   {total_samples}")
    print(f"  Output dir:      {args.output_dir}")
    if errors:
        print(f"  Errors:          {len(errors)}")
        for fname, err in errors:
            print(f"    - {fname}: {err}")
    print("=" * 60)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
