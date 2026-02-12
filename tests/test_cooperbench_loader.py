"""Unit tests for CooperBench dataset loader.

Tests dataset discovery, feature extraction, subset filtering,
feature pair generation, and edge cases using mock filesystems.
"""

import json
from pathlib import Path

import pytest

from src.evaluation.cooperbench.loader import (
    _discover_features,
    _load_subset_filter,
    _read_file_safe,
    load_cooperbench_dataset,
)
from src.evaluation.cooperbench.models import CooperBenchProblem


def _create_mock_dataset(tmp_path: Path, repos: dict) -> Path:
    """Create a mock CooperBench dataset directory structure.

    Args:
        tmp_path: Base temporary directory.
        repos: Dict mapping repo_name -> {task_id -> [feature_ids]}.

    Returns:
        Path to the mock dataset root.
    """
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    for repo_name, tasks in repos.items():
        repo_dir = dataset_root / repo_name
        repo_dir.mkdir()

        for task_id, feature_ids in tasks.items():
            task_dir = repo_dir / task_id
            task_dir.mkdir()

            # Create task-level files
            (task_dir / "setup.sh").write_text("#!/bin/bash\necho setup", encoding="utf-8")
            (task_dir / "run_tests.sh").write_text("#!/bin/bash\necho tests", encoding="utf-8")
            (task_dir / "runner.sh").write_text("#!/bin/bash\necho runner", encoding="utf-8")
            (task_dir / "Dockerfile").write_text("FROM ubuntu:22.04", encoding="utf-8")
            (task_dir / "combined.patch").write_text("diff --git combined", encoding="utf-8")

            # Create feature directories
            for fid in feature_ids:
                feature_dir = task_dir / f"feature{fid}"
                feature_dir.mkdir()
                (feature_dir / "feature.md").write_text(
                    f"# Feature {fid}\n\nImplement feature {fid} for {repo_name}/{task_id}.",
                    encoding="utf-8",
                )
                (feature_dir / "feature.patch").write_text(
                    f"diff --git feature {fid}", encoding="utf-8"
                )
                (feature_dir / "tests.patch").write_text(
                    f"diff --git tests {fid}", encoding="utf-8"
                )

    return dataset_root


# --- _read_file_safe Tests ---


class TestReadFileSafe:
    """Tests for _read_file_safe helper."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        """Should read existing file contents."""
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert _read_file_safe(f) == "hello world"

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Should return None for nonexistent file."""
        f = tmp_path / "nonexistent.txt"
        assert _read_file_safe(f) is None


# --- _discover_features Tests ---


class TestDiscoverFeatures:
    """Tests for _discover_features helper."""

    def test_discover_features(self, tmp_path: Path) -> None:
        """Should discover feature directories with feature.md."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {"task1": [2, 6, 10]},
        })
        task_dir = dataset / "repo1" / "task1"
        features = _discover_features(task_dir)

        assert len(features) == 3
        assert 2 in features
        assert 6 in features
        assert 10 in features
        assert features[2].feature_id == 2
        assert "Feature 2" in features[2].description

    def test_skip_missing_feature_md(self, tmp_path: Path) -> None:
        """Should skip features without feature.md."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {"task1": [1, 2]},
        })
        # Remove feature.md from feature1
        (dataset / "repo1" / "task1" / "feature1" / "feature.md").unlink()

        features = _discover_features(dataset / "repo1" / "task1")
        assert len(features) == 1
        assert 2 in features

    def test_skip_non_feature_dirs(self, tmp_path: Path) -> None:
        """Should skip directories not starting with 'feature'."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {"task1": [1, 2]},
        })
        # Create a non-feature directory
        (dataset / "repo1" / "task1" / "scripts").mkdir()
        (dataset / "repo1" / "task1" / "scripts" / "feature.md").write_text("not a feature")

        features = _discover_features(dataset / "repo1" / "task1")
        assert len(features) == 2

    def test_empty_task(self, tmp_path: Path) -> None:
        """Should return empty dict for task with no features."""
        dataset = _create_mock_dataset(tmp_path, {"repo1": {"task1": []}})
        features = _discover_features(dataset / "repo1" / "task1")
        assert len(features) == 0


# --- _load_subset_filter Tests ---


class TestLoadSubsetFilter:
    """Tests for _load_subset_filter helper."""

    def test_load_string_list(self, tmp_path: Path) -> None:
        """Should load subset file with string entries."""
        subset_file = tmp_path / "lite.json"
        data = ["repo1/task1", "repo1/task2", "repo2/task3"]
        subset_file.write_text(json.dumps(data), encoding="utf-8")

        allowed = _load_subset_filter(tmp_path, "lite")
        assert allowed is not None
        assert "repo1/task1" in allowed
        assert "repo2/task3" in allowed
        assert len(allowed) == 3

    def test_load_dict_list(self, tmp_path: Path) -> None:
        """Should load subset file with dict entries."""
        subset_file = tmp_path / "flash.json"
        data = [
            {"repo": "repo1", "task_id": "task1"},
            {"repo": "repo2", "task_id": "task2", "features": [1, 3]},
        ]
        subset_file.write_text(json.dumps(data), encoding="utf-8")

        allowed = _load_subset_filter(tmp_path, "flash")
        assert allowed is not None
        assert "repo1/task1" in allowed
        assert "repo2/task2" in allowed
        # Feature pair key should also be present
        assert "repo2/task2/features_1_3" in allowed

    def test_missing_subset_file(self, tmp_path: Path) -> None:
        """Should return None when subset file doesn't exist."""
        allowed = _load_subset_filter(tmp_path, "nonexistent")
        assert allowed is None

    def test_malformed_json(self, tmp_path: Path) -> None:
        """Should return None for malformed JSON."""
        subset_file = tmp_path / "lite.json"
        subset_file.write_text("{broken json", encoding="utf-8")

        allowed = _load_subset_filter(tmp_path, "lite")
        assert allowed is None


# --- load_cooperbench_dataset Tests ---


class TestLoadCooperBenchDataset:
    """Tests for load_cooperbench_dataset function."""

    def test_basic_loading(self, tmp_path: Path) -> None:
        """Should discover repos, tasks, and feature pairs."""
        dataset = _create_mock_dataset(tmp_path, {
            "llama_index": {"task17244": [2, 6]},
        })

        problems = load_cooperbench_dataset(str(dataset))

        assert len(problems) == 1
        p = problems[0]
        assert p.repo == "llama_index"
        assert p.task_id == "task17244"
        assert p.features == [2, 6]
        assert 2 in p.feature_specs
        assert 6 in p.feature_specs
        assert p.setup_script is not None
        assert p.runner_script is not None
        assert p.dockerfile is not None

    def test_multiple_repos(self, tmp_path: Path) -> None:
        """Should discover tasks across multiple repos."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo_a": {"task1": [1, 2]},
            "repo_b": {"task2": [3, 4], "task3": [5, 6]},
        })

        problems = load_cooperbench_dataset(str(dataset))
        assert len(problems) == 3

        repos = {p.repo for p in problems}
        assert repos == {"repo_a", "repo_b"}

    def test_feature_pair_generation(self, tmp_path: Path) -> None:
        """Should generate nC2 feature pairs for tasks with >2 features."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {"task1": [1, 2, 3]},  # 3C2 = 3 pairs
        })

        problems = load_cooperbench_dataset(str(dataset))
        assert len(problems) == 3

        feature_sets = [tuple(p.features) for p in problems]
        assert (1, 2) in feature_sets
        assert (1, 3) in feature_sets
        assert (2, 3) in feature_sets

    def test_four_features(self, tmp_path: Path) -> None:
        """Should generate 6 pairs for 4 features (4C2 = 6)."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {"task1": [1, 2, 3, 4]},
        })

        problems = load_cooperbench_dataset(str(dataset))
        assert len(problems) == 6

    def test_no_pairing(self, tmp_path: Path) -> None:
        """When pair_features=False, return one problem per task."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {"task1": [1, 2, 3]},
        })

        problems = load_cooperbench_dataset(str(dataset), pair_features=False)
        assert len(problems) == 1
        assert problems[0].features == [1, 2, 3]

    def test_repo_filter(self, tmp_path: Path) -> None:
        """Should filter by repository name."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo_a": {"task1": [1, 2]},
            "repo_b": {"task2": [3, 4]},
        })

        problems = load_cooperbench_dataset(str(dataset), repo_filter=["repo_a"])
        assert len(problems) == 1
        assert problems[0].repo == "repo_a"

    def test_task_filter(self, tmp_path: Path) -> None:
        """Should filter by task ID."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {
                "task1": [1, 2],
                "task2": [3, 4],
                "task3": [5, 6],
            },
        })

        problems = load_cooperbench_dataset(str(dataset), task_filter=["task2"])
        assert len(problems) == 1
        assert problems[0].task_id == "task2"

    def test_skip_single_feature_tasks(self, tmp_path: Path) -> None:
        """Should skip tasks with fewer than 2 features."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {
                "task1": [1],      # Only 1 feature - skip
                "task2": [2, 3],   # 2 features - include
            },
        })

        problems = load_cooperbench_dataset(str(dataset))
        assert len(problems) == 1
        assert problems[0].task_id == "task2"

    def test_skip_hidden_dirs(self, tmp_path: Path) -> None:
        """Should skip directories starting with '.'."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {"task1": [1, 2]},
        })
        (dataset / ".git").mkdir()

        problems = load_cooperbench_dataset(str(dataset))
        assert len(problems) == 1

    def test_skip_non_task_dirs(self, tmp_path: Path) -> None:
        """Should skip directories not starting with 'task'."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {"task1": [1, 2]},
        })
        (dataset / "repo1" / "scripts").mkdir()

        problems = load_cooperbench_dataset(str(dataset))
        assert len(problems) == 1

    def test_nonexistent_path(self) -> None:
        """Should raise FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_cooperbench_dataset("/nonexistent/path")

    def test_subset_filter(self, tmp_path: Path) -> None:
        """Should filter by subset file."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo_a": {"task1": [1, 2], "task2": [3, 4]},
            "repo_b": {"task3": [5, 6]},
        })

        # Create subset file allowing only repo_a/task1
        subset_data = ["repo_a/task1"]
        (dataset / "lite.json").write_text(json.dumps(subset_data), encoding="utf-8")

        problems = load_cooperbench_dataset(str(dataset), subset="lite")
        assert len(problems) == 1
        assert problems[0].repo == "repo_a"
        assert problems[0].task_id == "task1"

    def test_feature_spec_content(self, tmp_path: Path) -> None:
        """Feature specs should contain actual file content."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo1": {"task1": [1, 2]},
        })

        problems = load_cooperbench_dataset(str(dataset))
        assert len(problems) == 1

        spec = problems[0].feature_specs[1]
        assert "Feature 1" in spec.description
        assert "repo1/task1" in spec.description

    def test_problem_image_name(self, tmp_path: Path) -> None:
        """Image name should be derived from repo and task."""
        dataset = _create_mock_dataset(tmp_path, {
            "Llama_Index": {"task123": [1, 2]},
        })

        problems = load_cooperbench_dataset(str(dataset))
        assert len(problems) == 1
        # Image name should be lowercase
        assert problems[0].image_name == "cooperbench-llama_index-task123"

    def test_empty_dataset(self, tmp_path: Path) -> None:
        """Should return empty list for empty dataset."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        problems = load_cooperbench_dataset(str(dataset))
        assert problems == []

    def test_combined_filters(self, tmp_path: Path) -> None:
        """Should apply repo and task filters simultaneously."""
        dataset = _create_mock_dataset(tmp_path, {
            "repo_a": {"task1": [1, 2], "task2": [3, 4]},
            "repo_b": {"task1": [5, 6]},
        })

        problems = load_cooperbench_dataset(
            str(dataset),
            repo_filter=["repo_a"],
            task_filter=["task2"],
        )
        assert len(problems) == 1
        assert problems[0].repo == "repo_a"
        assert problems[0].task_id == "task2"
