"""Tests for rollout_integration.py.

Tests the integration layer that bridges Flink actor output format to
rollout_table.py functions, debug_data_writer.py, and KL computation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.wandb_enrichment.rollout_integration import (
    build_rollout_records_from_batch,
    compute_kl_if_available,
    log_debate_rollout_table,
    write_debate_debug_data,
)
from src.training.wandb_enrichment.rollout_table import RolloutRecord


def test_build_rollout_records_basic():
    """Test basic conversion from Flink batch format to RolloutRecord."""
    trajectories = [
        "Solver: Let x=2\nVerifier: Correct\nJudge: Accept",
        "Solver: Let x=3\nVerifier: Wrong\nJudge: Reject",
        "Solver: Let x=4\nVerifier: Correct\nJudge: Accept",
        "Solver: Let x=5\nVerifier: Wrong\nJudge: Reject",
    ]
    rewards = [1.0, 0.0, 1.0, 0.0]
    unique_sample_ids = ["math:0", "math:0", "math:1", "math:1"]

    records = build_rollout_records_from_batch(
        trajectories=trajectories,
        rewards=rewards,
        unique_sample_ids=unique_sample_ids,
    )

    assert len(records) == 4
    assert all(isinstance(r, RolloutRecord) for r in records)

    # Verify first record
    assert records[0].prompt_id == "math:0"
    assert records[0].completion == trajectories[0]  # RAW text, not parsed
    assert records[0].reward == 1.0
    assert records[0].solver_reward is None  # Phase 6 will populate
    assert records[0].verifier_reward is None
    assert records[0].judge_reward is None
    assert records[0].role_assignments is None

    # Verify completion contains RAW multi-turn text (no parsing)
    assert "Solver:" in records[0].completion
    assert "Verifier:" in records[0].completion
    assert "Judge:" in records[0].completion


def test_build_rollout_records_filters_by_env():
    """Test that only math_debate env entries are included when env_names provided."""
    trajectories = ["debate1", "baseline1", "debate2", "baseline2"]
    rewards = [1.0, 0.5, 0.8, 0.3]
    unique_sample_ids = ["math:0", "math:1", "math:2", "math:3"]
    env_names = ["math_debate", "math", "math_debate", "math"]

    records = build_rollout_records_from_batch(
        trajectories=trajectories,
        rewards=rewards,
        unique_sample_ids=unique_sample_ids,
        env_names=env_names,
    )

    # Should only include math_debate entries (indices 0 and 2)
    assert len(records) == 2
    assert records[0].completion == "debate1"
    assert records[1].completion == "debate2"


def test_build_rollout_records_empty():
    """Test that empty inputs produce empty list."""
    records = build_rollout_records_from_batch(
        trajectories=[],
        rewards=[],
        unique_sample_ids=[],
    )
    assert records == []


def test_build_rollout_records_mismatched_lengths():
    """Test handling of mismatched input lengths."""
    # Trajectories longer than rewards
    trajectories = ["t1", "t2", "t3"]
    rewards = [1.0, 0.5]
    unique_sample_ids = ["math:0", "math:1"]

    records = build_rollout_records_from_batch(
        trajectories=trajectories,
        rewards=rewards,
        unique_sample_ids=unique_sample_ids,
    )

    # Should truncate to shortest length (2)
    assert len(records) == 2
    assert records[0].completion == "t1"
    assert records[1].completion == "t2"


def test_log_debate_rollout_table_calls_sampling():
    """Test that log_debate_rollout_table calls sampling functions correctly."""
    # Create mock actor output items
    class MockItem:
        def __init__(self, traj: str, reward: float, sample_id: str):
            self.metadata = {
                "trajectory": np.array(traj),
                "unique_sample_id": np.array(sample_id),
                "env_name": np.array("math_debate"),
            }
            self.data = {
                "rewards": np.array(reward),
            }

    items = [
        MockItem("traj1", 1.0, "math:0"),
        MockItem("traj2", 0.5, "math:0"),
        MockItem("traj3", 0.8, "math:1"),
        MockItem("traj4", 0.2, "math:1"),
    ]

    # Mock rollout_table functions
    with patch("src.training.wandb_enrichment.rollout_table.sample_rollouts_per_prompt") as mock_sample, \
         patch("src.training.wandb_enrichment.rollout_table.create_rollout_table") as mock_create, \
         patch("src.training.wandb_enrichment.rollout_table.add_sampled_rollouts_to_table") as mock_add, \
         patch("src.training.wandb_enrichment.rollout_table.log_rollout_table") as mock_log:

        # Setup mocks
        mock_wandb_run = MagicMock()
        mock_sample.return_value = [(RolloutRecord("math:0", "[Prompt math:0]", "traj1", 1.0), True)]
        mock_create.return_value = MagicMock()

        # Call function with explicit wandb_run to avoid import issues
        log_debate_rollout_table(items=items, step=100, wandb_run=mock_wandb_run, n_prompts=2, top_k=1, bottom_k=1)

        # Verify sample_rollouts_per_prompt was called
        mock_sample.assert_called_once()
        call_kwargs = mock_sample.call_args.kwargs
        assert call_kwargs["n_prompts"] == 2
        assert call_kwargs["top_k"] == 1
        assert call_kwargs["bottom_k"] == 1

        # Verify rollouts argument has correct structure
        rollouts_arg = call_kwargs["rollouts"]
        assert len(rollouts_arg) == 4
        assert all(isinstance(r, RolloutRecord) for r in rollouts_arg)

        # Verify W&B table functions called
        mock_create.assert_called_once_with(use_incremental=True)
        mock_add.assert_called_once()
        mock_log.assert_called_once()


def test_log_debate_rollout_table_no_wandb_run():
    """Test that log_debate_rollout_table handles missing W&B run gracefully."""
    items = []

    # Call with wandb_run=None explicitly
    # Should not raise, just log debug message
    log_debate_rollout_table(items=items, step=100, wandb_run=None)


def test_compute_kl_if_available_no_masks():
    """Test that compute_kl_if_available returns empty dict when no role_masks."""
    batch_metadata = {
        "some_other_key": "value",
    }

    result = compute_kl_if_available(batch_metadata)

    assert result == {}


def test_compute_kl_if_available_with_masks():
    """Test that compute_kl_if_available calls compute_per_role_kl when masks present."""
    # Mock role_masks and kl_per_token (Phase 6 format)
    kl_per_token = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    role_masks = {
        "solver": np.array([[True, True, False], [True, False, False]]),
        "verifier": np.array([[False, False, True], [False, True, True]]),
    }

    batch_metadata = {
        "debate/role_masks": role_masks,
        "kl_per_token": kl_per_token,
    }

    with patch("src.training.wandb_enrichment.debate_metrics.compute_per_role_kl") as mock_compute:
        mock_compute.return_value = {"debate/kl/solver": 0.25, "debate/kl/verifier": 0.45}

        result = compute_kl_if_available(batch_metadata)

        # Verify compute_per_role_kl was called with correct args
        mock_compute.assert_called_once()
        call_kwargs = mock_compute.call_args.kwargs
        assert np.array_equal(call_kwargs["kl_per_token"], kl_per_token)
        assert call_kwargs["role_masks"] == role_masks

        # Verify result
        assert result == {"debate/kl/solver": 0.25, "debate/kl/verifier": 0.45}


def test_compute_kl_if_available_missing_kl_per_token():
    """Test handling when role_masks present but kl_per_token missing."""
    batch_metadata = {
        "debate/role_masks": {"solver": np.array([[True, False]])},
        # Missing kl_per_token
    }

    result = compute_kl_if_available(batch_metadata)

    # Should return empty dict and log warning
    assert result == {}


def test_write_debate_debug_data_creates_parquet():
    """Test that write_debate_debug_data creates Parquet file with correct schema."""
    # Create mock actor output items
    class MockItem:
        def __init__(self, traj: str, reward: float, sample_id: str):
            self.metadata = {
                "trajectory": np.array(traj),
                "unique_sample_id": np.array(sample_id),
            }
            self.data = {
                "rewards": np.array(reward),
            }

    items = [
        MockItem("traj1", 1.0, "math:0"),
        MockItem("traj2", 0.5, "math:1"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = write_debate_debug_data(items=items, step=42, output_dir=tmpdir)

        # Verify file was created
        assert output_path is not None
        assert Path(output_path).exists()
        assert "batch_debug_data_train_42.parquet" in output_path

        # Verify file can be read
        import pandas as pd
        df = pd.read_parquet(output_path)

        # Verify schema and content
        assert len(df) == 2
        assert "trajectory" in df.columns
        assert "reward" in df.columns
        assert "unique_sample_id" in df.columns
        assert "schema_version" in df.columns

        # Verify data
        assert df["trajectory"].tolist() == ["traj1", "traj2"]
        assert df["reward"].tolist() == [1.0, 0.5]
        assert df["unique_sample_id"].tolist() == ["math:0", "math:1"]


def test_write_debate_debug_data_handles_errors():
    """Test that write_debate_debug_data handles errors gracefully."""
    items = []

    # Use invalid output directory
    output_path = write_debate_debug_data(items=items, step=42, output_dir="/invalid/path/that/does/not/exist")

    # Should return None on error, not raise
    # (Actually, write_debug_data_parquet creates the directory, so this test needs different error condition)
    # Let's test with items that have malformed data instead
    class BadItem:
        @property
        def metadata(self):
            raise AttributeError("Simulated error")

    items = [BadItem()]

    output_path = write_debate_debug_data(items=items, step=42, output_dir="/tmp")

    # Should return None and log warning, not crash
    assert output_path is None


def test_log_debate_rollout_table_with_explicit_wandb_run():
    """Test that log_debate_rollout_table accepts explicit wandb_run parameter."""
    class MockItem:
        def __init__(self, traj: str, reward: float, sample_id: str):
            self.metadata = {
                "trajectory": np.array(traj),
                "unique_sample_id": np.array(sample_id),
                "env_name": np.array("math_debate"),
            }
            self.data = {
                "rewards": np.array(reward),
            }

    items = [MockItem("traj1", 1.0, "math:0")]

    with patch("src.training.wandb_enrichment.rollout_table.sample_rollouts_per_prompt") as mock_sample, \
         patch("src.training.wandb_enrichment.rollout_table.create_rollout_table") as mock_create, \
         patch("src.training.wandb_enrichment.rollout_table.add_sampled_rollouts_to_table") as mock_add, \
         patch("src.training.wandb_enrichment.rollout_table.log_rollout_table") as mock_log:

        # Setup mocks
        mock_wandb_run = MagicMock()
        mock_sample.return_value = [(RolloutRecord("math:0", "[Prompt math:0]", "traj1", 1.0), True)]
        mock_create.return_value = MagicMock()

        # Call with explicit wandb_run
        log_debate_rollout_table(items=items, step=100, wandb_run=mock_wandb_run)

        # Verify log_rollout_table was called with the explicit run
        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs["wandb_run"] == mock_wandb_run


def test_build_rollout_records_raw_text_not_parsed():
    """Test that completion field contains raw text, not parsed debate turns."""
    # Multi-turn debate text with role markers
    raw_debate = """Solver: Let's solve this step by step.
First, we need to find x.
x = 2 + 3 = 5

Verifier: I'll verify this solution.
The calculation is correct.
x = 5 is the answer.

Judge: Both solver and verifier agree.
Final answer: x = 5"""

    trajectories = [raw_debate]
    rewards = [1.0]
    unique_sample_ids = ["math:42"]

    records = build_rollout_records_from_batch(
        trajectories=trajectories,
        rewards=rewards,
        unique_sample_ids=unique_sample_ids,
    )

    # Verify completion is EXACTLY the raw text (no parsing)
    assert records[0].completion == raw_debate
    assert "Solver:" in records[0].completion
    assert "Verifier:" in records[0].completion
    assert "Judge:" in records[0].completion

    # Verify no parsed structure like turns or role fields
    # (those would be added by Phase 7 Streamlit parsing, not here)
    assert not hasattr(records[0], "turns")
    assert not hasattr(records[0], "parsed_roles")
