"""Report generation for CooperBench failure mode analysis."""

from src.report.generator import (
    FullReport,
    TaskReport,
    export_json,
    generate_comparison_figure,
    generate_figure,
    generate_text_report,
    run_classification,
)

__all__ = [
    "FullReport",
    "TaskReport",
    "export_json",
    "generate_comparison_figure",
    "generate_figure",
    "generate_text_report",
    "run_classification",
]
