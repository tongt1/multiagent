"""CLI for CooperBench failure mode evaluation pipeline.

Subcommands:
    classify  — Run all 10 classifiers on transcript data
    report    — Generate full analysis report with figures
    compare   — Compare failure rates across configurations
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("cooperbench-eval")


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="cooperbench-eval",
        description="CooperBench failure mode evaluation pipeline",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- classify ---
    classify_parser = subparsers.add_parser(
        "classify",
        help="Run failure mode classifiers on transcript data",
    )
    classify_parser.add_argument(
        "input",
        help="Path to JSONL trajectory file or directory of conversation.jsonl files",
    )
    classify_parser.add_argument(
        "-o", "--output",
        default="output/results.json",
        help="Output JSON path (default: output/results.json)",
    )
    classify_parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Run only heuristic classifiers (skip LLM-based)",
    )
    classify_parser.add_argument(
        "--llm-model",
        default="command-r-plus",
        help="Cohere model for LLM classifiers (default: command-r-plus)",
    )

    # --- report ---
    report_parser = subparsers.add_parser(
        "report",
        help="Generate analysis report from classification results",
    )
    report_parser.add_argument(
        "input",
        help="Path to JSONL trajectory file or classification results JSON",
    )
    report_parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Output directory for report artifacts (default: output/)",
    )
    report_parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Run only heuristic classifiers (skip LLM-based)",
    )
    report_parser.add_argument(
        "--llm-model",
        default="command-r-plus",
        help="Cohere model for LLM classifiers (default: command-r-plus)",
    )
    report_parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip figure generation",
    )

    # --- compare ---
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare failure rates across configurations",
    )
    compare_parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths to classification result JSON files (name:path format)",
    )
    compare_parser.add_argument(
        "-o", "--output",
        default="output/comparison.png",
        help="Output figure path (default: output/comparison.png)",
    )

    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "classify":
        return _cmd_classify(args)
    elif args.command == "report":
        return _cmd_report(args)
    elif args.command == "compare":
        return _cmd_compare(args)
    else:
        parser.print_help()
        return 1


def _load_tasks(input_path: str) -> list:
    """Load tasks from either JSONL trajectory file or directory."""
    from src.data_loading.trajectory_loader import load_trajectories

    path = Path(input_path)

    if path.is_file() and path.suffix == ".jsonl":
        return load_trajectories(path)
    elif path.is_dir():
        # Try loading as a run directory with conversation.jsonl files
        from src.data_loading.loader import load_run
        return load_run(path)
    else:
        logger.error("Input must be a .jsonl file or directory: %s", input_path)
        return []


def _build_classifiers(heuristic_only: bool = False, llm_model: str = "command-r-plus") -> list:
    """Build classifier instances."""
    from src.classifiers import ALL_CLASSIFIERS, HEURISTIC_CLASSIFIERS

    if heuristic_only:
        return [cls() for cls in HEURISTIC_CLASSIFIERS]

    classifiers = []
    for cls in ALL_CLASSIFIERS:
        # LLM classifiers need model parameter
        try:
            from src.llm_judge.base import LLMClassifier
            if issubclass(cls, LLMClassifier):
                classifiers.append(cls(model=llm_model))
            else:
                classifiers.append(cls())
        except (ImportError, TypeError):
            classifiers.append(cls())

    return classifiers


def _cmd_classify(args: argparse.Namespace) -> int:
    """Run classification pipeline."""
    from src.report.generator import export_json, run_classification

    tasks = _load_tasks(args.input)
    if not tasks:
        logger.error("No tasks loaded from %s", args.input)
        return 1

    logger.info("Loaded %d tasks", len(tasks))

    classifiers = _build_classifiers(
        heuristic_only=args.heuristic_only,
        llm_model=args.llm_model,
    )
    logger.info("Running %d classifiers", len(classifiers))

    report = run_classification(tasks, classifiers)

    output_path = export_json(report, args.output)
    logger.info("Results saved to %s", output_path)

    # Print summary
    rates = report.failure_rates
    print(f"\nClassified {report.total_tasks} tasks with {len(classifiers)} classifiers:")
    for name, rate in sorted(rates.items(), key=lambda x: -x[1]):
        print(f"  {name}: {rate:.1f}%")

    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Generate full report."""
    from src.report.generator import (
        export_json,
        generate_figure,
        generate_text_report,
        run_classification,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if input is already a results JSON or raw trajectories
    input_path = Path(args.input)
    if input_path.suffix == ".json":
        # Load pre-computed results
        logger.info("Loading pre-computed results from %s", input_path)
        # For simplicity, re-run classification from trajectories
        logger.warning("Re-running classification from raw data is recommended. Use .jsonl input.")

    tasks = _load_tasks(args.input)
    if not tasks:
        logger.error("No tasks loaded from %s", args.input)
        return 1

    logger.info("Loaded %d tasks", len(tasks))

    classifiers = _build_classifiers(
        heuristic_only=args.heuristic_only,
        llm_model=args.llm_model,
    )
    logger.info("Running %d classifiers", len(classifiers))

    report = run_classification(tasks, classifiers)

    # Generate text report
    text = generate_text_report(report)
    report_path = output_dir / "report.txt"
    report_path.write_text(text, encoding="utf-8")
    logger.info("Text report saved to %s", report_path)
    print(text)

    # Generate figure
    if not args.no_figure:
        try:
            fig_path = generate_figure(report, output_dir / "failure_modes.png")
            logger.info("Figure saved to %s", fig_path)
        except ImportError:
            logger.warning("matplotlib not available; skipping figure generation")

    # Export JSON
    json_path = export_json(report, output_dir / "results.json")
    logger.info("JSON results saved to %s", json_path)

    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare failure rates across configurations."""
    from src.report.generator import FullReport, generate_comparison_figure

    reports: dict[str, FullReport] = {}

    for input_spec in args.inputs:
        if ":" in input_spec:
            name, path = input_spec.split(":", 1)
        else:
            name = Path(input_spec).stem
            path = input_spec

        filepath = Path(path)
        if not filepath.exists():
            logger.error("File not found: %s", filepath)
            return 1

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct FullReport from JSON
        from src.classifiers.base import ClassificationResult, Severity
        from src.report.generator import TaskReport

        task_reports = []
        for task_data in data.get("tasks", []):
            results = []
            for r in task_data.get("results", []):
                sev = Severity(r["severity"]) if r.get("severity") else Severity.LOW
                results.append(ClassificationResult(
                    classifier_name=r["classifier"],
                    detected=r["detected"],
                    severity=sev,
                    confidence=r.get("confidence", 0.0),
                    evidence=r.get("evidence", []),
                    skipped=r.get("skipped", False),
                    skip_reason=r.get("skip_reason", ""),
                ))
            task_reports.append(TaskReport(
                task_id=task_data["task_id"],
                run_id=task_data.get("run_id", ""),
                task_description=task_data.get("task_description", ""),
                results=results,
                agents=task_data.get("agents", []),
            ))

        reports[name] = FullReport(task_reports=task_reports)

    if len(reports) < 2:
        logger.error("Need at least 2 configurations to compare")
        return 1

    try:
        fig_path = generate_comparison_figure(reports, args.output)
        logger.info("Comparison figure saved to %s", fig_path)
        print(f"Comparison figure saved to {fig_path}")
    except ImportError:
        logger.error("matplotlib required for comparison figures")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
