"""Integration tests for the full pipeline.

These tests run the complete pipeline on sample data to verify
end-to-end functionality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.classifiers import HEURISTIC_CLASSIFIERS, get_all_classifiers
from src.data_loading.trajectory_loader import load_trajectories
from src.report.generator import (
    export_json,
    generate_figure,
    generate_text_report,
    run_classification,
)


SAMPLE_DATA = Path("/home/terry_tong_cohere_com/multiagent/data/sample_trajectories.jsonl")


@pytest.fixture
def sample_tasks():
    if not SAMPLE_DATA.exists():
        pytest.skip("Sample trajectory data not available")
    return load_trajectories(SAMPLE_DATA)


class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_load_all_tasks(self, sample_tasks):
        assert len(sample_tasks) == 50
        for task in sample_tasks:
            assert task.task_id
            assert len(task.messages) > 0
            assert len(task.agents) > 1
            assert task.task_description

    def test_heuristic_pipeline(self, sample_tasks, tmp_path: Path):
        """Run heuristic classifiers on all tasks."""
        classifiers = [cls() for cls in HEURISTIC_CLASSIFIERS]
        report = run_classification(sample_tasks, classifiers)

        assert report.total_tasks == 50
        assert len(report.task_reports) == 50

        # Each task should have results from all classifiers
        for tr in report.task_reports:
            assert len(tr.results) == len(classifiers)

    def test_all_classifiers_registry(self):
        """Verify all 10 classifiers can be instantiated."""
        all_cls = get_all_classifiers()
        assert len(all_cls) == 10

        # Check names are unique
        names = set()
        for cls in all_cls:
            instance = cls() if not _needs_api_key(cls) else cls(api_key="test")
            assert instance.name not in names, f"Duplicate name: {instance.name}"
            names.add(instance.name)

    def test_report_generation(self, sample_tasks, tmp_path: Path):
        """Test full report generation pipeline."""
        classifiers = [cls() for cls in HEURISTIC_CLASSIFIERS]
        report = run_classification(sample_tasks[:5], classifiers)

        # Text report
        text = generate_text_report(report)
        assert "CooperBench" in text
        assert "Total tasks analyzed: 5" in text

        # JSON export
        json_path = export_json(report, tmp_path / "results.json")
        assert json_path.exists()

        # Figure
        fig_path = generate_figure(report, tmp_path / "figure.png")
        assert fig_path.exists()

    def test_all_classifiers_on_sample(self, sample_tasks, tmp_path: Path):
        """Run all 10 classifiers on a small sample (no real API calls).

        Uses fake API key so LLM classifiers return skip results.
        """
        all_cls = get_all_classifiers()
        classifiers = []
        for cls in all_cls:
            if _needs_api_key(cls):
                classifiers.append(cls(api_key=""))  # Will skip
            else:
                classifiers.append(cls())

        report = run_classification(sample_tasks[:3], classifiers)
        assert report.total_tasks == 3

        for tr in report.task_reports:
            assert len(tr.results) == 10  # All 10 classifiers ran


def _needs_api_key(cls):
    """Check if a classifier class is LLM-based."""
    try:
        from src.llm_judge.base import LLMClassifier
        return issubclass(cls, LLMClassifier)
    except ImportError:
        return False
