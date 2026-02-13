"""Tests for report generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.classifiers.base import ClassificationResult, Severity
from src.data_loading.schemas import TaskData
from src.report.generator import (
    FullReport,
    TaskReport,
    export_json,
    generate_text_report,
)


@pytest.fixture
def sample_report() -> FullReport:
    """A sample FullReport for testing."""
    task_reports = [
        TaskReport(
            task_id="task-001",
            run_id="run-001",
            task_description="Solve 2+2",
            agents=["solver_0", "solver_1", "verifier"],
            results=[
                ClassificationResult(
                    classifier_name="work_overlap",
                    detected=True,
                    severity=Severity.HIGH,
                    confidence=0.85,
                    evidence=["Agents overlap on solver.py"],
                ),
                ClassificationResult(
                    classifier_name="repetition",
                    detected=False,
                ),
                ClassificationResult(
                    classifier_name="divergent_architecture",
                    detected=True,
                    severity=Severity.MEDIUM,
                    confidence=0.7,
                    evidence=["Incompatible approaches"],
                ),
            ],
        ),
        TaskReport(
            task_id="task-002",
            run_id="run-002",
            task_description="Solve 3+3",
            agents=["solver_0", "solver_1"],
            results=[
                ClassificationResult(
                    classifier_name="work_overlap",
                    detected=False,
                ),
                ClassificationResult(
                    classifier_name="repetition",
                    detected=True,
                    severity=Severity.LOW,
                    confidence=0.6,
                    evidence=["Repeated message pattern"],
                ),
                ClassificationResult(
                    classifier_name="divergent_architecture",
                    detected=False,
                ),
            ],
        ),
    ]

    return FullReport(task_reports=task_reports)


class TestFullReport:
    """Tests for FullReport data class."""

    def test_total_tasks(self, sample_report: FullReport):
        assert sample_report.total_tasks == 2

    def test_failure_rates(self, sample_report: FullReport):
        rates = sample_report.failure_rates
        assert rates["work_overlap"] == 50.0  # 1/2 tasks
        assert rates["repetition"] == 50.0  # 1/2 tasks
        assert rates["divergent_architecture"] == 50.0  # 1/2 tasks

    def test_severity_distribution(self, sample_report: FullReport):
        dist = sample_report.severity_distribution
        assert dist["work_overlap"]["high"] == 1
        assert dist["repetition"]["low"] == 1
        assert dist["divergent_architecture"]["medium"] == 1


class TestTextReport:
    """Tests for text report generation."""

    def test_generates_text(self, sample_report: FullReport):
        text = generate_text_report(sample_report)
        assert "CooperBench Failure Mode Analysis Report" in text
        assert "Total tasks analyzed: 2" in text
        assert "Work Overlap" in text
        assert "task-001" in text

    def test_includes_failure_rates(self, sample_report: FullReport):
        text = generate_text_report(sample_report)
        assert "50.0%" in text

    def test_includes_evidence(self, sample_report: FullReport):
        text = generate_text_report(sample_report)
        assert "Agents overlap on solver.py" in text


class TestJsonExport:
    """Tests for JSON export."""

    def test_export_creates_file(self, sample_report: FullReport, tmp_path: Path):
        output = tmp_path / "results.json"
        path = export_json(sample_report, output)
        assert path.exists()

    def test_export_roundtrip(self, sample_report: FullReport, tmp_path: Path):
        import json

        output = tmp_path / "results.json"
        export_json(sample_report, output)

        with open(output) as f:
            data = json.load(f)

        assert data["summary"]["total_tasks"] == 2
        assert len(data["tasks"]) == 2
        assert data["summary"]["failure_rates"]["work_overlap"] == 50.0
