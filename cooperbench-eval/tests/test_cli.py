"""Tests for CLI interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli import main


class TestCLI:
    """Tests for the CLI subcommands."""

    def test_no_command_returns_error(self):
        ret = main([])
        assert ret == 1

    def test_help_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_classify_missing_input(self):
        with pytest.raises(SystemExit):
            main(["classify"])

    def test_classify_nonexistent_file(self):
        ret = main(["classify", "/nonexistent/file.jsonl"])
        assert ret == 1

    def test_classify_heuristic_only(self, tmp_path: Path):
        """Test classify with heuristic-only flag on real data."""
        data_path = Path("/home/terry_tong_cohere_com/multiagent/data/sample_trajectories.jsonl")
        if not data_path.exists():
            pytest.skip("Sample trajectory data not available")

        output = tmp_path / "results.json"
        ret = main([
            "classify",
            str(data_path),
            "--heuristic-only",
            "-o", str(output),
        ])
        assert ret == 0
        assert output.exists()

        with open(output) as f:
            data = json.load(f)
        assert data["summary"]["total_tasks"] == 50

    def test_report_missing_input(self):
        with pytest.raises(SystemExit):
            main(["report"])

    def test_report_heuristic_only(self, tmp_path: Path):
        """Test report generation with heuristic-only flag."""
        data_path = Path("/home/terry_tong_cohere_com/multiagent/data/sample_trajectories.jsonl")
        if not data_path.exists():
            pytest.skip("Sample trajectory data not available")

        ret = main([
            "report",
            str(data_path),
            "--heuristic-only",
            "-o", str(tmp_path),
        ])
        assert ret == 0
        assert (tmp_path / "report.txt").exists()
        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "failure_modes.png").exists()
