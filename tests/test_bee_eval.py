"""Tests for BEE checkpoint evaluation module."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.evaluation import bee_eval


def test_bee_available_constant():
    """Test BEE_AVAILABLE constant is set correctly."""
    # Should be False in test environment (no BEE installed)
    assert isinstance(bee_eval.BEE_AVAILABLE, bool)


def test_evaluate_checkpoint_structure_without_bee():
    """Test evaluate_checkpoint returns correct structure when BEE not available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = bee_eval.evaluate_checkpoint(
            checkpoint_path="/fake/checkpoint/path",
            eval_data_path="/fake/eval/data.jsonl",
            output_dir=tmpdir,
        )

        # Check structure
        assert "status" in result
        assert result["status"] == "bee_not_available"
        assert "checkpoint" in result
        assert "overall_accuracy" in result
        assert "by_difficulty" in result
        assert "per_sample_results" in result
        assert "num_samples" in result
        assert "timestamp" in result

        # Check by_difficulty has levels 1-5
        assert set(result["by_difficulty"].keys()) == {"1", "2", "3", "4", "5"}

        # Check all accuracies are floats
        assert isinstance(result["overall_accuracy"], float)
        for level_acc in result["by_difficulty"].values():
            assert isinstance(level_acc, float)


def test_evaluate_checkpoint_with_mocked_bee():
    """Test evaluate_checkpoint with mocked BEE imports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock BEE imports and results
        mock_estimator = mock.MagicMock()
        mock_task = mock.MagicMock()

        # Mock task results
        mock_results = {
            "overall": {"accuracy": 0.65},
            "MATH_level_1": {"accuracy": 0.85},
            "MATH_level_2": {"accuracy": 0.72},
            "MATH_level_3": {"accuracy": 0.65},
            "MATH_level_4": {"accuracy": 0.55},
            "MATH_level_5": {"accuracy": 0.42},
            "samples": [
                {
                    "id": "problem_1",
                    "level": 1,
                    "correct": True,
                    "predicted": "42",
                    "expected": "42",
                },
                {
                    "id": "problem_2",
                    "level": 2,
                    "correct": False,
                    "predicted": "10",
                    "expected": "20",
                },
            ],
        }

        mock_task.run.return_value = mock_results
        mock_task.load_data.return_value = [{"id": f"problem_{i}"} for i in range(10)]

        with mock.patch("src.evaluation.bee_eval.BEE_AVAILABLE", True):
            with mock.patch("src.evaluation.bee_eval.load_estimator", return_value=mock_estimator):
                with mock.patch("src.evaluation.bee_eval.MATH", return_value=mock_task):
                    # Mock Config class
                    with mock.patch("src.evaluation.bee_eval.Config"):
                        result = bee_eval.evaluate_checkpoint(
                            checkpoint_path="/fake/ckpt-5",
                            eval_data_path="/fake/eval.jsonl",
                            output_dir=tmpdir,
                        )

        # Verify result structure
        assert result["status"] == "complete"
        assert result["overall_accuracy"] == 0.65
        assert result["by_difficulty"]["1"] == 0.85
        assert result["by_difficulty"]["2"] == 0.72
        assert result["by_difficulty"]["3"] == 0.65
        assert result["by_difficulty"]["4"] == 0.55
        assert result["by_difficulty"]["5"] == 0.42
        assert len(result["per_sample_results"]) == 2
        assert result["num_samples"] == 10

        # Verify file was created
        eval_files = list(Path(tmpdir).glob("*.json"))
        assert len(eval_files) == 1
        assert eval_files[0].name == "ckpt-5_eval.json"

        # Verify file contents
        with open(eval_files[0], "r", encoding="utf-8") as f:
            saved_result = json.load(f)
        assert saved_result["overall_accuracy"] == 0.65


def test_evaluate_all_checkpoints_finds_and_sorts():
    """Test evaluate_all_checkpoints finds and sorts checkpoints correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create experiment directory structure
        exp_dir = Path(tmpdir) / "experiment"
        ckpt_dir = exp_dir / "debate" / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        # Create checkpoint directories (out of order)
        (ckpt_dir / "ckpt-15").mkdir()
        (ckpt_dir / "ckpt-5").mkdir()
        (ckpt_dir / "ckpt-10").mkdir()

        # Mock evaluate_checkpoint to track calls
        with mock.patch(
            "src.evaluation.bee_eval.evaluate_checkpoint"
        ) as mock_eval:
            mock_eval.return_value = {
                "status": "complete",
                "checkpoint": "test",
                "overall_accuracy": 0.5,
                "by_difficulty": {str(i): 0.5 for i in range(1, 6)},
                "per_sample_results": [],
                "num_samples": 10,
                "timestamp": "2026-02-02T12:00:00Z",
            }

            results = bee_eval.evaluate_all_checkpoints(
                experiment_dir=str(exp_dir),
                eval_data_path="/fake/eval.jsonl",
                mode="debate",
            )

        # Verify 3 checkpoints were evaluated
        assert len(results) == 3
        assert mock_eval.call_count == 3

        # Verify checkpoints were called in sorted order (5, 10, 15)
        calls = mock_eval.call_args_list
        checkpoint_paths = [call[1]["checkpoint_path"] for call in calls]
        assert "ckpt-5" in checkpoint_paths[0]
        assert "ckpt-10" in checkpoint_paths[1]
        assert "ckpt-15" in checkpoint_paths[2]

        # Verify combined results file exists
        combined_file = exp_dir / "debate" / "eval_results" / "all_results.json"
        assert combined_file.exists()

        # Verify combined results content
        with open(combined_file, "r", encoding="utf-8") as f:
            combined_results = json.load(f)
        assert len(combined_results) == 3


def test_evaluate_all_checkpoints_missing_directory():
    """Test evaluate_all_checkpoints handles missing directory gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results = bee_eval.evaluate_all_checkpoints(
            experiment_dir=tmpdir,
            eval_data_path="/fake/eval.jsonl",
            mode="debate",
        )

        # Should return empty list without crashing
        assert results == []


def test_log_eval_to_wandb_handles_missing_wandb():
    """Test log_eval_to_wandb handles missing wandb gracefully."""
    with mock.patch("src.evaluation.bee_eval.wandb", None):
        # Should not crash when wandb is not available
        with tempfile.TemporaryFile() as tmpfile:
            bee_eval.log_eval_to_wandb(
                training_run_id="test_run",
                checkpoint_step=5,
                eval_results={
                    "overall_accuracy": 0.65,
                    "by_difficulty": {str(i): 0.5 for i in range(1, 6)},
                    "num_samples": 100,
                    "timestamp": "2026-02-02T12:00:00Z",
                },
                eval_results_path=tmpfile.name,
            )


def test_log_eval_to_wandb_with_mocked_wandb():
    """Test log_eval_to_wandb creates artifacts correctly."""
    mock_wandb = mock.MagicMock()
    mock_run = mock.MagicMock()
    mock_artifact = mock.MagicMock()

    mock_wandb.init.return_value = mock_run
    mock_wandb.Artifact.return_value = mock_artifact

    eval_results = {
        "overall_accuracy": 0.65,
        "by_difficulty": {"1": 0.85, "2": 0.72, "3": 0.65, "4": 0.55, "5": 0.42},
        "num_samples": 100,
        "timestamp": "2026-02-02T12:00:00Z",
    }

    with mock.patch("src.evaluation.bee_eval.wandb", mock_wandb):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmpfile:
            json.dump(eval_results, tmpfile)
            tmpfile.flush()

            bee_eval.log_eval_to_wandb(
                training_run_id="test_run_123",
                checkpoint_step=10,
                eval_results=eval_results,
                eval_results_path=tmpfile.name,
                wandb_project="test-project",
            )

    # Verify wandb.init was called correctly
    mock_wandb.init.assert_called_once()
    call_kwargs = mock_wandb.init.call_args[1]
    assert call_kwargs["project"] == "test-project"
    assert call_kwargs["id"] == "test_run_123"
    assert call_kwargs["resume"] == "allow"

    # Verify artifact was created
    mock_wandb.Artifact.assert_called_once()
    artifact_kwargs = mock_wandb.Artifact.call_args[1]
    assert artifact_kwargs["name"] == "eval-results-step-10"
    assert artifact_kwargs["type"] == "evaluation"

    # Verify artifact methods were called
    mock_artifact.add_file.assert_called_once()
    mock_run.log_artifact.assert_called_once_with(mock_artifact)

    # Verify metrics were logged
    mock_run.log.assert_called_once()
    logged_metrics = mock_run.log.call_args[0][0]
    assert "eval/overall_accuracy" in logged_metrics
    assert logged_metrics["eval/overall_accuracy"] == 0.65
    assert "eval/accuracy_level_1" in logged_metrics
    assert logged_metrics["eval/accuracy_level_1"] == 0.85

    # Verify run was finished
    mock_run.finish.assert_called_once()


def test_result_aggregation():
    """Test accuracy aggregation with known correct/incorrect samples."""
    # Create 10 samples with known outcomes
    per_sample_results = [
        {"problem_id": f"p{i}", "level": (i % 5) + 1, "correct": i < 6, "predicted": str(i), "expected": str(i)}
        for i in range(10)
    ]

    # 6 correct out of 10 = 0.6 overall accuracy
    correct_count = sum(1 for s in per_sample_results if s["correct"])
    overall_accuracy = correct_count / len(per_sample_results)
    assert overall_accuracy == 0.6

    # Per-level accuracy
    by_difficulty = {}
    for level in range(1, 6):
        level_samples = [s for s in per_sample_results if s["level"] == level]
        if level_samples:
            level_correct = sum(1 for s in level_samples if s["correct"])
            by_difficulty[str(level)] = level_correct / len(level_samples)
        else:
            by_difficulty[str(level)] = 0.0

    # Verify per-level accuracies
    # Level 1: samples 0, 5 -> 1 correct (sample 0) -> 0.5
    # Level 2: samples 1, 6 -> 1 correct (sample 1) -> 0.5
    # Level 3: samples 2, 7 -> 1 correct (sample 2) -> 0.5
    # Level 4: samples 3, 8 -> 1 correct (sample 3) -> 0.5
    # Level 5: samples 4, 9 -> 1 correct (sample 4) -> 0.5
    # Wait, that's not right. Let me recalculate.
    # i < 6 means samples 0-5 are correct
    # Levels: 0->1, 1->2, 2->3, 3->4, 4->5, 5->1, 6->2, 7->3, 8->4, 9->5
    # Level 1: i=0 (correct), i=5 (correct) -> 2/2 = 1.0
    # Level 2: i=1 (correct), i=6 (incorrect) -> 1/2 = 0.5
    # Level 3: i=2 (correct), i=7 (incorrect) -> 1/2 = 0.5
    # Level 4: i=3 (correct), i=8 (incorrect) -> 1/2 = 0.5
    # Level 5: i=4 (correct), i=9 (incorrect) -> 1/2 = 0.5

    assert by_difficulty["1"] == 1.0
    assert by_difficulty["2"] == 0.5
    assert by_difficulty["3"] == 0.5
    assert by_difficulty["4"] == 0.5
    assert by_difficulty["5"] == 0.5


def test_bee_not_available_path():
    """Test that BEE_AVAILABLE=False returns stub result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Force BEE_AVAILABLE to False
        with mock.patch("src.evaluation.bee_eval.BEE_AVAILABLE", False):
            result = bee_eval.evaluate_checkpoint(
                checkpoint_path="/fake/checkpoint",
                eval_data_path="/fake/eval.jsonl",
                output_dir=tmpdir,
            )

        # Should return stub result
        assert result["status"] == "bee_not_available"
        assert "message" in result
        assert result["overall_accuracy"] == 0.0
        assert all(acc == 0.0 for acc in result["by_difficulty"].values())
        assert result["num_samples"] == 0
