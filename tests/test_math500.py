"""Tests for MATH 500 dataset loader."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.dataset_loader import Problem
from src.data.math500 import get_math500_stats, load_math500


@pytest.fixture
def mock_dataset():
    """Create a synthetic MATH dataset for testing.

    Returns 600 problems (120 per difficulty level 1-5) with mix of:
    - Problems with extractable \\boxed{} answers
    - Problems without \\boxed{} answers (to test filtering)
    """
    problems = []

    # Create problems for each level (1-5)
    for level in range(1, 6):
        # Create 120 problems per level
        for i in range(120):
            # Every 10th problem has no boxed answer (to test filtering)
            if i % 10 == 0:
                problem = {
                    "problem": f"Level {level} problem {i} (no boxed)",
                    "solution": "This solution has no boxed answer",
                    "level": f"Level {level}",
                    "type": "Algebra",
                }
            else:
                problem = {
                    "problem": f"Level {level} problem {i}",
                    "solution": f"The answer is \\boxed{{{i}}}",
                    "level": f"Level {level}",
                    "type": ["Algebra", "Geometry", "Number Theory", "Counting & Probability"][
                        i % 4
                    ],
                }
            problems.append(problem)

    return problems


@pytest.fixture
def temp_cache_path(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir / "math500_cache.json"


def test_load_math500_returns_500_problems(mock_dataset, temp_cache_path):
    """Test that load_math500 returns exactly 500 problems."""
    with patch("src.data.math500.load_dataset") as mock_load_dataset:
        # Mock the dataset
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_dataset))
        mock_load_dataset.return_value = mock_ds

        # Load MATH 500
        problems = load_math500(seed=42, cache_path=temp_cache_path)

        # Should return exactly 500 problems
        assert len(problems) == 500


def test_filtering_removes_no_boxed(mock_dataset, temp_cache_path):
    """Test that problems without \\boxed{} answers are filtered out."""
    with patch("src.data.math500.load_dataset") as mock_load_dataset:
        # Mock the dataset
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_dataset))
        mock_load_dataset.return_value = mock_ds

        # Load MATH 500
        problems = load_math500(seed=42, cache_path=temp_cache_path)

        # Check that all problems have extractable boxed answers
        from src.evaluation.math_verifier import extract_boxed_answer

        for problem in problems:
            extracted = extract_boxed_answer(problem.ground_truth)
            assert extracted is not None, f"Problem {problem.id} has no extractable answer"


def test_stratified_sampling(mock_dataset, temp_cache_path):
    """Test that problems are sampled approximately evenly across difficulty levels."""
    with patch("src.data.math500.load_dataset") as mock_load_dataset:
        # Mock the dataset
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_dataset))
        mock_load_dataset.return_value = mock_ds

        # Load MATH 500
        problems = load_math500(seed=42, cache_path=temp_cache_path)

        # Count problems per level
        level_counts = {}
        for problem in problems:
            level = problem.metadata.get("level")
            level_counts[level] = level_counts.get(level, 0) + 1

        # Should have 5 levels
        assert len(level_counts) == 5

        # Each level should have approximately 100 problems
        # Allow some deviation due to filtering and redistribution
        for level, count in level_counts.items():
            assert (
                50 <= count <= 150
            ), f"Level {level} has {count} problems, expected ~100"

        # Most common case: each level has exactly 100
        # But we allow for redistribution, so check that total is 500
        assert sum(level_counts.values()) == 500


def test_problem_fields(mock_dataset, temp_cache_path):
    """Test that each Problem has correct fields and format."""
    with patch("src.data.math500.load_dataset") as mock_load_dataset:
        # Mock the dataset
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_dataset))
        mock_load_dataset.return_value = mock_ds

        # Load MATH 500
        problems = load_math500(seed=42, cache_path=temp_cache_path)

        # Check first problem in detail
        problem = problems[0]

        # Check ID format
        assert problem.id.startswith("math500_")
        assert len(problem.id) == 12  # "math500_" + 4 digits = 8 + 4 = 12

        # Check problem text is non-empty
        assert len(problem.problem) > 0

        # Check ground truth is non-None
        assert problem.ground_truth is not None

        # Check domain
        assert problem.domain == "math"

        # Check metadata
        assert "level" in problem.metadata
        assert isinstance(problem.metadata["level"], int)
        assert 1 <= problem.metadata["level"] <= 5

        assert "type" in problem.metadata
        assert isinstance(problem.metadata["type"], str)

        assert "source" in problem.metadata
        assert problem.metadata["source"] == "MATH-500"

        assert "difficulty" in problem.metadata
        assert problem.metadata["difficulty"].startswith("Level ")


def test_reproducible_seed(mock_dataset, temp_cache_path):
    """Test that the same seed produces the same problem IDs."""
    with patch("src.data.math500.load_dataset") as mock_load_dataset:
        # Mock the dataset
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_dataset))
        mock_load_dataset.return_value = mock_ds

        # Load twice with same seed
        problems1 = load_math500(seed=42, cache_path=temp_cache_path, force_reload=True)

        # Need to reload dataset for second call
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_dataset))

        problems2 = load_math500(seed=42, cache_path=temp_cache_path, force_reload=True)

        # Should have same IDs in same order
        ids1 = [p.id for p in problems1]
        ids2 = [p.id for p in problems2]

        # IDs should be identical
        assert ids1 == ids2


def test_cache_roundtrip(mock_dataset, temp_cache_path):
    """Test that caching to JSON and reloading produces identical problems."""
    with patch("src.data.math500.load_dataset") as mock_load_dataset:
        # Mock the dataset
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_dataset))
        mock_load_dataset.return_value = mock_ds

        # Load first time (creates cache)
        problems1 = load_math500(seed=42, cache_path=temp_cache_path)

        # Cache file should exist
        assert temp_cache_path.exists()

        # Load second time (uses cache)
        problems2 = load_math500(seed=42, cache_path=temp_cache_path)

        # Should be identical
        assert len(problems1) == len(problems2)

        for p1, p2 in zip(problems1, problems2):
            assert p1.id == p2.id
            assert p1.problem == p2.problem
            assert p1.ground_truth == p2.ground_truth
            assert p1.domain == p2.domain
            assert p1.metadata == p2.metadata


def test_get_math500_stats(mock_dataset, temp_cache_path):
    """Test that get_math500_stats returns correct structure."""
    with patch("src.data.math500.load_dataset") as mock_load_dataset:
        # Mock the dataset
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_dataset))
        mock_load_dataset.return_value = mock_ds

        # Load MATH 500
        problems = load_math500(seed=42, cache_path=temp_cache_path)

        # Get stats
        stats = get_math500_stats(problems)

        # Check structure
        assert "total" in stats
        assert "per_level" in stats
        assert "per_type" in stats

        # Check total
        assert stats["total"] == 500

        # Check per_level is a dict with integer keys
        assert isinstance(stats["per_level"], dict)
        for level, count in stats["per_level"].items():
            assert isinstance(level, int)
            assert isinstance(count, int)
            assert count > 0

        # Check per_type is a dict with string keys
        assert isinstance(stats["per_type"], dict)
        for math_type, count in stats["per_type"].items():
            assert isinstance(math_type, str)
            assert isinstance(count, int)
            assert count > 0

        # Total from per_level should equal 500
        assert sum(stats["per_level"].values()) == 500


def test_force_reload_ignores_cache(mock_dataset, temp_cache_path):
    """Test that force_reload=True ignores existing cache."""
    with patch("src.data.math500.load_dataset") as mock_load_dataset:
        # Mock the dataset
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_dataset))
        mock_load_dataset.return_value = mock_ds

        # Load first time (creates cache)
        problems1 = load_math500(seed=42, cache_path=temp_cache_path)

        # Modify cache to have wrong data
        with temp_cache_path.open("w") as f:
            fake_problem = {
                "id": "fake_0000",
                "problem": "Fake problem",
                "ground_truth": "\\boxed{fake}",
                "domain": "math",
                "metadata": {"level": 1, "type": "Fake", "source": "MATH-500", "difficulty": "Level 1"},
            }
            json.dump([fake_problem], f)

        # Load with cache (should use cache)
        problems2 = load_math500(seed=42, cache_path=temp_cache_path)
        assert len(problems2) == 1
        assert problems2[0].id == "fake_0000"

        # Reload dataset for force_reload
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_dataset))

        # Load with force_reload (should ignore cache)
        problems3 = load_math500(seed=42, cache_path=temp_cache_path, force_reload=True)
        assert len(problems3) == 500
        assert problems3[0].id != "fake_0000"
