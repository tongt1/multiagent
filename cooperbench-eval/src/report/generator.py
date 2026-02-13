"""Report generator for CooperBench failure mode analysis.

Generates:
1. Failure mode frequency bar chart (comparable to CooperBench paper Figure 6)
2. Per-task failure analysis with transcript excerpts
3. Aggregate statistics across all transcripts
4. JSON export for programmatic consumption
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.classifiers.base import ClassificationResult, Severity
from src.data_loading.schemas import TaskData

logger = logging.getLogger(__name__)

# Display names for classifiers (Figure 6 style)
DISPLAY_NAMES = {
    "work_overlap": "Work Overlap",
    "divergent_architecture": "Divergent Architecture",
    "repetition": "Repetition",
    "unresponsiveness": "Unresponsiveness",
    "unverifiable_claims": "Unverifiable Claims",
    "broken_commitment": "Broken Commitment",
    "dependency_access": "Dependency Access",
    "placeholder_misuse": "Placeholder Misuse",
    "parameter_flow": "Parameter Flow",
    "timing_dependency": "Timing Dependency",
}

# CooperBench paper baseline rates for comparison
PAPER_BASELINE_RATES = {
    "Work Overlap": 33.2,
    "Divergent Architecture": 29.7,
    "Repetition": 14.7,
    "Unresponsiveness": 8.7,
    "Unverifiable Claims": 4.3,
    "Broken Commitment": 3.7,
    "Dependency Access": 1.7,
    "Placeholder Misuse": 1.5,
    "Parameter Flow": 1.3,
    "Timing Dependency": 1.1,
}


@dataclass
class TaskReport:
    """Classification results for a single task."""
    task_id: str
    run_id: str
    task_description: str
    results: list[ClassificationResult] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)

    @property
    def detected_failures(self) -> list[ClassificationResult]:
        return [r for r in self.results if r.detected]

    @property
    def failure_count(self) -> int:
        return len(self.detected_failures)


@dataclass
class FullReport:
    """Complete analysis report across all tasks."""
    task_reports: list[TaskReport] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tasks(self) -> int:
        return len(self.task_reports)

    @property
    def failure_rates(self) -> dict[str, float]:
        """Compute failure rate for each classifier as percentage of tasks."""
        if not self.task_reports:
            return {}
        counts: Counter[str] = Counter()
        for tr in self.task_reports:
            for result in tr.results:
                if result.detected:
                    counts[result.classifier_name] += 1
        return {
            name: round(count / self.total_tasks * 100, 1)
            for name, count in counts.items()
        }

    @property
    def severity_distribution(self) -> dict[str, dict[str, int]]:
        """Severity breakdown per classifier."""
        dist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for tr in self.task_reports:
            for result in tr.results:
                if result.detected:
                    dist[result.classifier_name][result.severity.value] += 1
        return {k: dict(v) for k, v in dist.items()}


def run_classification(
    tasks: list[TaskData],
    classifiers: list[Any],  # list of BaseClassifier instances
) -> FullReport:
    """Run all classifiers on all tasks and produce a FullReport.

    Args:
        tasks: List of TaskData objects to classify.
        classifiers: Instantiated classifier objects.

    Returns:
        FullReport with all results.
    """
    task_reports: list[TaskReport] = []

    for task in tasks:
        results: list[ClassificationResult] = []
        for classifier in classifiers:
            try:
                result = classifier.classify(task)
                results.append(result)
            except Exception as e:
                logger.error("Classifier %s failed on task %s: %s", classifier.name, task.task_id, e)
                results.append(ClassificationResult(
                    classifier_name=classifier.name,
                    skipped=True,
                    skip_reason=str(e),
                ))

        task_reports.append(TaskReport(
            task_id=task.task_id,
            run_id=task.run_id,
            task_description=task.task_description,
            results=results,
            agents=task.agents,
        ))

    return FullReport(task_reports=task_reports)


def generate_figure(report: FullReport, output_path: str | Path) -> Path:
    """Generate failure mode frequency bar chart (Figure 6 style).

    Args:
        report: Complete analysis report.
        output_path: Path to save the figure (PNG).

    Returns:
        Path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rates = report.failure_rates

    # Order by CooperBench paper prevalence
    ordered_names = [
        "work_overlap", "divergent_architecture", "repetition",
        "unresponsiveness", "unverifiable_claims", "broken_commitment",
        "dependency_access", "placeholder_misuse", "parameter_flow",
        "timing_dependency",
    ]

    display_labels = [DISPLAY_NAMES.get(n, n) for n in ordered_names]
    our_rates = [rates.get(n, 0.0) for n in ordered_names]
    paper_rates = [PAPER_BASELINE_RATES.get(DISPLAY_NAMES.get(n, n), 0.0) for n in ordered_names]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = range(len(ordered_names))
    bar_width = 0.35

    bars1 = ax.bar(
        [i - bar_width / 2 for i in x],
        paper_rates,
        bar_width,
        label="CooperBench Paper",
        color="#4A90D9",
        alpha=0.7,
        edgecolor="white",
    )
    bars2 = ax.bar(
        [i + bar_width / 2 for i in x],
        our_rates,
        bar_width,
        label="Our Pipeline",
        color="#E74C3C",
        alpha=0.8,
        edgecolor="white",
    )

    ax.set_xlabel("Failure Mode", fontsize=12, fontweight="bold")
    ax.set_ylabel("Detection Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Failure Mode Frequency — CooperBench Paper vs Our Pipeline",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#E74C3C",
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Figure saved to %s", output_path)
    return output_path


def generate_text_report(report: FullReport) -> str:
    """Generate a human-readable text report with transcript excerpts.

    Args:
        report: Complete analysis report.

    Returns:
        Formatted text report.
    """
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("CooperBench Failure Mode Analysis Report")
    lines.append("=" * 80)
    lines.append("")

    # Summary statistics
    lines.append("## Summary")
    lines.append(f"Total tasks analyzed: {report.total_tasks}")
    total_detections = sum(tr.failure_count for tr in report.task_reports)
    lines.append(f"Total failure detections: {total_detections}")
    tasks_with_failures = sum(1 for tr in report.task_reports if tr.failure_count > 0)
    lines.append(f"Tasks with at least one failure: {tasks_with_failures} ({tasks_with_failures / max(1, report.total_tasks) * 100:.1f}%)")
    lines.append("")

    # Failure rates table
    lines.append("## Failure Mode Frequency")
    lines.append(f"{'Failure Mode':<25} {'Our Rate':>10} {'Paper Rate':>12} {'Delta':>10}")
    lines.append("-" * 60)

    rates = report.failure_rates
    ordered = [
        "work_overlap", "divergent_architecture", "repetition",
        "unresponsiveness", "unverifiable_claims", "broken_commitment",
        "dependency_access", "placeholder_misuse", "parameter_flow",
        "timing_dependency",
    ]
    for name in ordered:
        display = DISPLAY_NAMES.get(name, name)
        our_rate = rates.get(name, 0.0)
        paper_rate = PAPER_BASELINE_RATES.get(display, 0.0)
        delta = our_rate - paper_rate
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        lines.append(f"{display:<25} {our_rate:>9.1f}% {paper_rate:>11.1f}% {delta_str:>10}")
    lines.append("")

    # Severity distribution
    lines.append("## Severity Distribution")
    severity_dist = report.severity_distribution
    for name in ordered:
        if name in severity_dist:
            display = DISPLAY_NAMES.get(name, name)
            dist = severity_dist[name]
            dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
            lines.append(f"  {display}: {dist_str}")
    lines.append("")

    # Per-task details with transcript excerpts
    lines.append("## Per-Task Analysis (showing tasks with failures)")
    lines.append("")

    for tr in report.task_reports:
        if tr.failure_count == 0:
            continue

        lines.append(f"### Task: {tr.task_id[:16]}...")
        lines.append(f"Problem: {tr.task_description[:120]}")
        lines.append(f"Agents: {', '.join(tr.agents)}")
        lines.append(f"Failures detected: {tr.failure_count}")
        lines.append("")

        for result in tr.detected_failures:
            display = DISPLAY_NAMES.get(result.classifier_name, result.classifier_name)
            lines.append(f"  * {display} ({result.severity.value}, conf={result.confidence:.2f})")
            for ev in result.evidence[:3]:  # Limit evidence lines
                lines.append(f"    - {ev[:150]}")
            lines.append("")

    lines.append("=" * 80)
    lines.append("End of Report")
    lines.append("=" * 80)

    return "\n".join(lines)


def export_json(report: FullReport, output_path: str | Path) -> Path:
    """Export report as structured JSON.

    Args:
        report: Complete analysis report.
        output_path: Path to save JSON file.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": {
            "total_tasks": report.total_tasks,
            "failure_rates": report.failure_rates,
            "severity_distribution": report.severity_distribution,
        },
        "tasks": [
            {
                "task_id": tr.task_id,
                "run_id": tr.run_id,
                "task_description": tr.task_description,
                "agents": tr.agents,
                "failure_count": tr.failure_count,
                "results": [
                    {
                        "classifier": r.classifier_name,
                        "detected": r.detected,
                        "severity": r.severity.value if r.detected else None,
                        "confidence": r.confidence,
                        "evidence": r.evidence,
                        "skipped": r.skipped,
                        "skip_reason": r.skip_reason if r.skipped else None,
                    }
                    for r in tr.results
                ],
            }
            for tr in report.task_reports
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("JSON report saved to %s", output_path)
    return output_path


def generate_comparison_figure(
    reports: dict[str, FullReport],
    output_path: str | Path,
) -> Path:
    """Generate comparison figure across multiple configurations.

    Args:
        reports: Mapping of config_name -> FullReport.
        output_path: Path to save the figure.

    Returns:
        Path to saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_names = [
        "work_overlap", "divergent_architecture", "repetition",
        "unresponsiveness", "unverifiable_claims", "broken_commitment",
        "dependency_access", "placeholder_misuse", "parameter_flow",
        "timing_dependency",
    ]
    display_labels = [DISPLAY_NAMES.get(n, n) for n in ordered_names]

    config_names = list(reports.keys())
    n_configs = len(config_names)
    n_modes = len(ordered_names)

    fig, ax = plt.subplots(figsize=(16, 8))

    bar_width = 0.8 / n_configs
    colors = plt.cm.Set2(np.linspace(0, 1, n_configs))

    for idx, (config_name, report) in enumerate(reports.items()):
        rates = report.failure_rates
        values = [rates.get(n, 0.0) for n in ordered_names]
        positions = [i + idx * bar_width - (n_configs - 1) * bar_width / 2 for i in range(n_modes)]
        ax.bar(positions, values, bar_width, label=config_name, color=colors[idx], alpha=0.8, edgecolor="white")

    ax.set_xlabel("Failure Mode", fontsize=12, fontweight="bold")
    ax.set_ylabel("Detection Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Failure Mode Comparison Across Configurations", fontsize=14, fontweight="bold")
    ax.set_xticks(range(n_modes))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Comparison figure saved to %s", output_path)
    return output_path
