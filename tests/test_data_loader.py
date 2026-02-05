"""Tests for data loading utilities."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from streamlit_viewer.lib.data_loader import (
    discover_parquet_files,
    compute_step_summary,
)


def test_discover_parquet_files():
    """Test Parquet file discovery with step extraction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock Parquet-named files (empty files for discovery test)
        Path(tmpdir, "batch_debug_data_train_100.parquet").touch()
        Path(tmpdir, "batch_debug_data_train_50.parquet").touch()
        Path(tmpdir, "batch_debug_data_train_200.parquet").touch()
        Path(tmpdir, "other_file.txt").touch()

        files = discover_parquet_files(tmpdir)

        # Should find 3 Parquet files, sorted by step
        assert len(files) == 3
        assert files[0]["step"] == 50
        assert files[1]["step"] == 100
        assert files[2]["step"] == 200
        assert all("path" in f for f in files)


def test_discover_parquet_files_nonexistent_dir():
    """Test discovery in non-existent directory."""
    files = discover_parquet_files("/nonexistent/directory")
    assert files == []


def test_discover_parquet_files_empty_dir():
    """Test discovery in empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = discover_parquet_files(tmpdir)
        assert files == []


def test_discover_parquet_files_no_match():
    """Test discovery with no matching files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "other_file.parquet").touch()
        Path(tmpdir, "data.csv").touch()

        files = discover_parquet_files(tmpdir)
        assert files == []


def test_compute_step_summary():
    """Test step summary computation with known data."""
    data = {
        "unique_sample_id": ["prompt1", "prompt1", "prompt2", "prompt2"],
        "reward": [1.0, 0.8, 0.6, 0.4],
        "trajectory": ["text1", "text2", "text3", "text4"],
    }
    df = pd.DataFrame(data)

    summary = compute_step_summary(df)

    assert summary["avg_reward"] == 0.7  # (1.0 + 0.8 + 0.6 + 0.4) / 4
    assert summary["num_prompts"] == 2  # 2 unique prompt IDs
    assert summary["num_rollouts"] == 4
    assert summary["reward_std"] > 0  # Has variance


def test_compute_step_summary_empty_df():
    """Test summary computation with empty DataFrame."""
    df = pd.DataFrame()

    summary = compute_step_summary(df)

    assert summary["avg_reward"] == 0.0
    assert summary["num_prompts"] == 0
    assert summary["num_rollouts"] == 0
    assert summary["reward_std"] == 0.0


def test_compute_step_summary_missing_columns():
    """Test summary computation with missing columns."""
    data = {
        "other_column": [1, 2, 3],
    }
    df = pd.DataFrame(data)

    summary = compute_step_summary(df)

    # Should handle missing columns gracefully
    assert summary["avg_reward"] == 0.0
    assert summary["num_prompts"] == 0
    assert summary["num_rollouts"] == 3


def test_compute_step_summary_single_prompt():
    """Test summary with single prompt multiple rollouts."""
    data = {
        "unique_sample_id": ["prompt1"] * 8,
        "reward": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.5, 0.7],
    }
    df = pd.DataFrame(data)

    summary = compute_step_summary(df)

    assert summary["num_prompts"] == 1
    assert summary["num_rollouts"] == 8
    assert 0 < summary["avg_reward"] < 1
