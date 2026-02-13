"""Tests for Parquet debug data writer."""

import pandas as pd
import pytest
import pyarrow.parquet as pq

from src.training.wandb_enrichment.debug_data_writer import (
    get_debug_data_path,
    read_debug_data_parquet,
    write_debug_data_parquet,
)
from src.training.wandb_enrichment.metric_schema import CURRENT_SCHEMA_VERSION


def test_write_and_read_basic(tmp_path):
    """Test writing and reading back basic debug data."""
    # Write data
    step = 100
    prompt_ids = ["p1", "p2", "p3", "p4"]
    prompt_texts = ["Problem 1", "Problem 2", "Problem 3", "Problem 4"]
    completions = ["Completion 1", "Completion 2", "Completion 3", "Completion 4"]
    rewards = [1.0, 0.8, 0.5, 0.0]

    output_path = write_debug_data_parquet(
        step=step,
        prompt_ids=prompt_ids,
        prompt_texts=prompt_texts,
        completions=completions,
        rewards=rewards,
        output_dir=str(tmp_path),
    )

    # Verify file exists
    assert output_path.endswith(f"batch_debug_data_train_{step}.parquet")
    assert (tmp_path / f"batch_debug_data_train_{step}.parquet").exists()

    # Read back
    df = read_debug_data_parquet(output_path)

    # Verify data
    assert len(df) == 4
    assert df["trajectory"].tolist() == completions
    assert df["reward"].tolist() == rewards
    assert df["env_name"].tolist() == ["math_debate"] * 4


def test_write_schema_version(tmp_path):
    """Verify schema_version column is present and correct."""
    step = 50
    output_path = write_debug_data_parquet(
        step=step,
        prompt_ids=["p1"],
        prompt_texts=["Problem"],
        completions=["Completion"],
        rewards=[1.0],
        output_dir=str(tmp_path),
    )

    df = read_debug_data_parquet(output_path)

    assert "schema_version" in df.columns
    assert df["schema_version"].iloc[0] == CURRENT_SCHEMA_VERSION
    assert all(df["schema_version"] == CURRENT_SCHEMA_VERSION)


def test_write_snappy_compression(tmp_path):
    """Verify Parquet file uses SNAPPY compression."""
    step = 75
    output_path = write_debug_data_parquet(
        step=step,
        prompt_ids=["p1", "p2"],
        prompt_texts=["Problem 1", "Problem 2"],
        completions=["Completion 1", "Completion 2"],
        rewards=[1.0, 0.5],
        output_dir=str(tmp_path),
    )

    # Read Parquet metadata
    parquet_file = pq.ParquetFile(output_path)
    metadata = parquet_file.metadata

    # Check compression
    # Note: PyArrow stores compression per column group
    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)
        for j in range(row_group.num_columns):
            column = row_group.column(j)
            # Compression enum: SNAPPY = 2
            assert column.compression in ["SNAPPY", 2]


def test_write_optional_columns_none(tmp_path):
    """Test writing with optional columns as None."""
    step = 25
    output_path = write_debug_data_parquet(
        step=step,
        prompt_ids=["p1", "p2"],
        prompt_texts=["Problem 1", "Problem 2"],
        completions=["Completion 1", "Completion 2"],
        rewards=[1.0, 0.5],
        output_dir=str(tmp_path),
        solver_rewards=None,
        verifier_rewards=None,
        judge_rewards=None,
        role_assignments=None,
        unique_sample_ids=None,
    )

    df = read_debug_data_parquet(output_path)

    # Verify columns exist with NaN/None
    assert "solver_reward" in df.columns
    assert "verifier_reward" in df.columns
    assert "judge_reward" in df.columns
    assert "role_assignments" in df.columns
    assert "unique_sample_id" in df.columns

    # Numerical columns should have NaN
    assert pd.isna(df["solver_reward"]).all()
    assert pd.isna(df["verifier_reward"]).all()
    assert pd.isna(df["judge_reward"]).all()

    # String columns should be None or empty
    assert df["unique_sample_id"].tolist() == ["", ""]


def test_write_mismatched_lengths(tmp_path):
    """Test that mismatched input lengths raise ValueError."""
    step = 10

    with pytest.raises(ValueError, match="Mismatched input lengths"):
        write_debug_data_parquet(
            step=step,
            prompt_ids=["p1", "p2", "p3"],  # 3 items
            prompt_texts=["Problem 1", "Problem 2"],  # 2 items - mismatch!
            completions=["Completion 1", "Completion 2", "Completion 3"],
            rewards=[1.0, 0.5, 0.0],
            output_dir=str(tmp_path),
        )


def test_write_creates_directory(tmp_path):
    """Test that output directory is created if it doesn't exist."""
    nested_dir = tmp_path / "nested" / "output" / "dir"
    assert not nested_dir.exists()

    step = 30
    output_path = write_debug_data_parquet(
        step=step,
        prompt_ids=["p1"],
        prompt_texts=["Problem"],
        completions=["Completion"],
        rewards=[1.0],
        output_dir=str(nested_dir),
    )

    # Verify directory was created
    assert nested_dir.exists()
    assert nested_dir.is_dir()

    # Verify file was written
    assert (nested_dir / f"batch_debug_data_train_{step}.parquet").exists()


def test_read_validates_schema_version(tmp_path):
    """Test that read_debug_data_parquet validates schema version."""
    step = 40
    output_path = write_debug_data_parquet(
        step=step,
        prompt_ids=["p1"],
        prompt_texts=["Problem"],
        completions=["Completion"],
        rewards=[1.0],
        output_dir=str(tmp_path),
    )

    # Reading current version should not produce warnings (in real scenario)
    df = read_debug_data_parquet(output_path)
    assert "schema_version" in df.columns
    assert df["schema_version"].iloc[0] == CURRENT_SCHEMA_VERSION


def test_get_debug_data_path():
    """Test get_debug_data_path returns correct path format."""
    path = get_debug_data_path("/tmp/output", 123)
    assert path == "/tmp/output/batch_debug_data_train_123.parquet"

    path = get_debug_data_path("/tmp/output", 0)
    assert path == "/tmp/output/batch_debug_data_train_0.parquet"


def test_write_large_text(tmp_path):
    """Test writing large completion text (5000+ chars)."""
    step = 60

    # Create a large completion (multi-turn debate)
    large_completion = "Turn 1: " + "A" * 2000 + "\nTurn 2: " + "B" * 2000 + "\nTurn 3: " + "C" * 1500

    output_path = write_debug_data_parquet(
        step=step,
        prompt_ids=["p1"],
        prompt_texts=["Problem"],
        completions=[large_completion],
        rewards=[1.0],
        output_dir=str(tmp_path),
    )

    # Read back
    df = read_debug_data_parquet(output_path)

    # Verify no truncation
    assert len(df["trajectory"].iloc[0]) == len(large_completion)
    assert df["trajectory"].iloc[0] == large_completion


def test_write_with_all_optional_fields(tmp_path):
    """Test writing with all optional fields populated."""
    step = 80
    output_path = write_debug_data_parquet(
        step=step,
        prompt_ids=["p1", "p2"],
        prompt_texts=["Problem 1", "Problem 2"],
        completions=["Completion 1", "Completion 2"],
        rewards=[1.0, 0.5],
        output_dir=str(tmp_path),
        solver_rewards=[0.9, 0.4],
        verifier_rewards=[0.95, 0.45],
        judge_rewards=[1.0, 0.5],
        role_assignments=["S:0,V:1,J:2", "S:1,V:2,J:0"],
        unique_sample_ids=["sample_001", "sample_002"],
    )

    df = read_debug_data_parquet(output_path)

    # Verify all fields are populated
    assert df["solver_reward"].tolist() == [0.9, 0.4]
    assert df["verifier_reward"].tolist() == [0.95, 0.45]
    assert df["judge_reward"].tolist() == [1.0, 0.5]
    assert df["role_assignments"].tolist() == ["S:0,V:1,J:2", "S:1,V:2,J:0"]
    assert df["unique_sample_id"].tolist() == ["sample_001", "sample_002"]
