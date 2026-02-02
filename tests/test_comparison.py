"""Tests for comparison analysis module."""

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.comparison import (
    bootstrap_accuracy_delta,
    bootstrap_per_level,
    compute_normalization_metrics,
    generate_comparison_report,
    generate_learning_curves,
    load_eval_results,
)


def test_bootstrap_accuracy_delta_significant():
    """Test bootstrap with known significant difference."""
    # 70% vs 60% accuracy - should be significant
    debate = [1] * 70 + [0] * 30
    baseline = [1] * 60 + [0] * 40

    result = bootstrap_accuracy_delta(debate, baseline, n_resamples=1000, seed=42)

    assert result["debate_accuracy"] == 0.70
    assert result["baseline_accuracy"] == 0.60
    assert result["point_estimate"] == pytest.approx(0.10, abs=0.01)
    assert result["significant"] is True
    assert result["ci_low"] > 0  # CI should not include 0
    assert result["ci_high"] > 0
    assert result["n_resamples"] == 1000
    assert result["confidence_level"] == 0.95
    assert 0 < result["p_value"] < 1.0


def test_bootstrap_accuracy_delta_identical():
    """Test bootstrap with identical data - should not be significant."""
    # Both 60% accuracy
    debate = [1] * 60 + [0] * 40
    baseline = [1] * 60 + [0] * 40

    result = bootstrap_accuracy_delta(debate, baseline, n_resamples=1000, seed=42)

    assert result["debate_accuracy"] == 0.60
    assert result["baseline_accuracy"] == 0.60
    assert result["point_estimate"] == pytest.approx(0.0, abs=0.01)
    assert result["significant"] is False
    # CI should include 0 for identical data
    assert result["ci_low"] <= 0 <= result["ci_high"]


def test_bootstrap_per_level():
    """Test per-level bootstrap CI computation."""
    debate_by_level = {
        "1": [1] * 80 + [0] * 20,  # 80% accuracy
        "2": [1] * 70 + [0] * 30,  # 70% accuracy
        "3": [1] * 60 + [0] * 40,  # 60% accuracy
    }

    baseline_by_level = {
        "1": [1] * 75 + [0] * 25,  # 75% accuracy
        "2": [1] * 65 + [0] * 35,  # 65% accuracy
        "3": [1] * 55 + [0] * 45,  # 55% accuracy
    }

    results = bootstrap_per_level(debate_by_level, baseline_by_level, n_resamples=1000, seed=42)

    assert len(results) == 3
    assert "1" in results
    assert "2" in results
    assert "3" in results

    # Check level 1
    assert results["1"]["debate_accuracy"] == 0.80
    assert results["1"]["baseline_accuracy"] == 0.75
    assert results["1"]["point_estimate"] == pytest.approx(0.05, abs=0.01)


def test_compute_normalization_metrics():
    """Test compute normalization with known values."""
    result = compute_normalization_metrics(
        accuracy=0.7,
        total_tokens_generated=1_000_000,  # 1M tokens
        training_time_gpu_hours=10.0,
        model_params=8_000_000_000,  # 8B params
        condition="debate",
    )

    assert result["condition"] == "debate"
    assert result["accuracy"] == 0.7
    assert result["total_tokens_generated"] == 1_000_000
    assert result["training_time_gpu_hours"] == 10.0
    assert result["model_params"] == 8_000_000_000

    # accuracy_per_1M_tokens = 0.7 / (1M / 1M) = 0.7 / 1 = 0.7
    assert result["accuracy_per_1M_tokens"] == pytest.approx(0.7, abs=0.01)

    # accuracy_per_gpu_hour = 0.7 / 10.0 = 0.07
    assert result["accuracy_per_gpu_hour"] == pytest.approx(0.07, abs=0.01)

    # estimated_total_flops = 6 * 8B * 1M = 48e15
    expected_flops = 6 * 8_000_000_000 * 1_000_000
    assert result["estimated_total_flops"] == expected_flops

    # accuracy_per_petaflop = 0.7 / (48e15 / 1e15) = 0.7 / 48
    assert result["accuracy_per_petaflop"] == pytest.approx(0.7 / 48, abs=0.001)


def test_compute_normalization_metrics_zero_tokens():
    """Test normalization with zero tokens (edge case)."""
    result = compute_normalization_metrics(
        accuracy=0.5, total_tokens_generated=0, training_time_gpu_hours=10.0, model_params=8_000_000_000
    )

    assert result["accuracy_per_1M_tokens"] == 0.0
    assert result["accuracy_per_gpu_hour"] == pytest.approx(0.05, abs=0.01)
    assert result["accuracy_per_petaflop"] == 0.0


def test_generate_learning_curves_csv_fallback():
    """Test learning curves with CSV fallback (no matplotlib)."""
    debate_results = [
        {"step": 0, "overall_accuracy": 0.5, "by_difficulty": {"1": 0.6, "2": 0.5, "3": 0.4, "4": 0.3, "5": 0.2}},
        {"step": 50, "overall_accuracy": 0.6, "by_difficulty": {"1": 0.7, "2": 0.6, "3": 0.5, "4": 0.4, "5": 0.3}},
    ]

    baseline_results = [
        {"step": 0, "overall_accuracy": 0.45, "by_difficulty": {"1": 0.55, "2": 0.45, "3": 0.35, "4": 0.25, "5": 0.15}},
        {"step": 50, "overall_accuracy": 0.55, "by_difficulty": {"1": 0.65, "2": 0.55, "3": 0.45, "4": 0.35, "5": 0.25}},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "curves.png")
        result_path = generate_learning_curves(debate_results, baseline_results, output_path)

        # Should create a file (either PNG or CSV)
        assert Path(result_path).exists()

        # If CSV, verify content
        if result_path.endswith(".csv"):
            with open(result_path) as f:
                content = f.read()
                assert "step,condition,overall_accuracy" in content
                assert "debate" in content
                assert "baseline" in content


def test_generate_learning_curves_empty():
    """Test learning curves handles empty results gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "curves.png")
        result_path = generate_learning_curves([], [], output_path)

        # Should still create CSV with header only
        assert Path(result_path).exists()
        if result_path.endswith(".csv"):
            with open(result_path) as f:
                content = f.read()
                assert "step,condition,overall_accuracy" in content


def test_load_eval_results_missing_dir():
    """Test load_eval_results with missing directory."""
    results = load_eval_results("/nonexistent/path", "debate")
    assert results == []


def test_load_eval_results_with_all_results():
    """Test load_eval_results with all_results.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir) / "debate" / "eval_results"
        eval_dir.mkdir(parents=True)

        # Create all_results.json
        all_results = [
            {"step": 0, "overall_accuracy": 0.5},
            {"step": 50, "overall_accuracy": 0.6},
            {"step": 100, "overall_accuracy": 0.7},
        ]

        with open(eval_dir / "all_results.json", "w") as f:
            json.dump(all_results, f)

        results = load_eval_results(tmpdir, "debate")
        assert len(results) == 3
        assert results[0]["step"] == 0
        assert results[1]["step"] == 50
        assert results[2]["step"] == 100


def test_load_eval_results_with_individual_files():
    """Test load_eval_results with individual checkpoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir) / "baseline" / "eval_results"
        eval_dir.mkdir(parents=True)

        # Create individual checkpoint files
        for step in [0, 50, 100]:
            with open(eval_dir / f"ckpt-{step}_eval.json", "w") as f:
                json.dump({"overall_accuracy": 0.5 + step / 1000}, f)

        results = load_eval_results(tmpdir, "baseline")
        assert len(results) == 3
        assert results[0]["step"] == 0
        assert results[1]["step"] == 50
        assert results[2]["step"] == 100


def test_generate_comparison_report():
    """Test full comparison report generation."""
    debate_results = [
        {"step": 0, "overall_accuracy": 0.5, "by_difficulty": {"1": 0.6, "2": 0.5, "3": 0.4, "4": 0.3, "5": 0.2}},
        {"step": 100, "overall_accuracy": 0.7, "by_difficulty": {"1": 0.8, "2": 0.7, "3": 0.6, "4": 0.5, "5": 0.4}},
    ]

    baseline_results = [
        {"step": 0, "overall_accuracy": 0.45, "by_difficulty": {"1": 0.55, "2": 0.45, "3": 0.35, "4": 0.25, "5": 0.15}},
        {"step": 100, "overall_accuracy": 0.6, "by_difficulty": {"1": 0.7, "2": 0.6, "3": 0.5, "4": 0.4, "5": 0.3}},
    ]

    debate_compute = {"total_tokens": 10_000_000, "gpu_hours": 100.0, "model_params": 8_000_000_000}

    baseline_compute = {"total_tokens": 5_000_000, "gpu_hours": 50.0, "model_params": 8_000_000_000}

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, md_path = generate_comparison_report(
            debate_results, baseline_results, debate_compute, baseline_compute, tmpdir
        )

        # Verify JSON exists and has expected structure
        assert Path(json_path).exists()
        with open(json_path) as f:
            report_data = json.load(f)

        assert "overall_comparison" in report_data
        assert "per_level_comparison" in report_data
        assert "compute_normalization" in report_data
        assert "learning_curves" in report_data
        assert "metadata" in report_data

        # Verify Markdown exists and has expected content
        assert Path(md_path).exists()
        with open(md_path) as f:
            md_content = f.read()

        assert "Comparison Report: Debate vs Baseline" in md_content
        assert "Overall Accuracy Comparison" in md_content
        assert "Per-Difficulty Breakdown" in md_content
        assert "Compute Normalization" in md_content
        assert "Learning Curves" in md_content
        # Check for expected table headers
        assert "Level" in md_content
        assert "Debate" in md_content
        assert "Baseline" in md_content
        assert "Delta" in md_content


def test_markdown_report_contains_table_headers():
    """Test that Markdown report contains expected table structure."""
    debate_results = [
        {"step": 100, "overall_accuracy": 0.7, "by_difficulty": {"1": 0.8, "2": 0.7, "3": 0.6, "4": 0.5, "5": 0.4}}
    ]

    baseline_results = [
        {"step": 100, "overall_accuracy": 0.6, "by_difficulty": {"1": 0.7, "2": 0.6, "3": 0.5, "4": 0.4, "5": 0.3}}
    ]

    debate_compute = {"total_tokens": 1_000_000, "gpu_hours": 10.0}
    baseline_compute = {"total_tokens": 1_000_000, "gpu_hours": 10.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        _, md_path = generate_comparison_report(debate_results, baseline_results, debate_compute, baseline_compute, tmpdir)

        with open(md_path) as f:
            content = f.read()

        # Verify table headers
        assert "| Level | Debate | Baseline | Delta |" in content
        assert "| Axis | Debate | Baseline | Delta |" in content

        # Verify at least one difficulty level appears
        assert "| 1 |" in content or "| 2 |" in content
