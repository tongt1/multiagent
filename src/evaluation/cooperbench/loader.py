"""Dataset loader for CooperBench cooperative coding benchmark.

Discovers repos, tasks, and features from the CooperBench dataset
filesystem structure and returns structured problem objects.

Expected dataset layout:
    dataset/
      <repo_name>/
        task<id>/
          setup.sh
          run_tests.sh
          runner.sh
          Dockerfile
          combined.patch
          feature<N>/
            feature.md
            feature.patch
            tests.patch
"""

import json
from itertools import combinations
from pathlib import Path
from typing import Optional

from loguru import logger

from src.evaluation.cooperbench.models import (
    CooperBenchProblem,
    FeatureSpec,
)


def _read_file_safe(path: Path) -> Optional[str]:
    """Read a file, returning None if it doesn't exist.

    Args:
        path: Path to the file to read.

    Returns:
        File contents as string, or None if file doesn't exist.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None


def _discover_features(task_dir: Path) -> dict[int, FeatureSpec]:
    """Discover feature directories within a task.

    Args:
        task_dir: Path to the task directory.

    Returns:
        Dict mapping feature IDs to FeatureSpec objects.
    """
    features: dict[int, FeatureSpec] = {}

    for entry in sorted(task_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("feature"):
            continue

        # Extract feature ID from directory name (e.g., "feature2" -> 2)
        try:
            feature_id = int(entry.name.replace("feature", ""))
        except ValueError:
            logger.warning(f"Skipping non-numeric feature directory: {entry.name}")
            continue

        # Read feature specification
        feature_md_path = entry / "feature.md"
        description = _read_file_safe(feature_md_path)
        if description is None:
            logger.warning(f"Missing feature.md in {entry}, skipping")
            continue

        # Build feature spec
        patch_path = entry / "feature.patch"
        tests_patch_path = entry / "tests.patch"

        features[feature_id] = FeatureSpec(
            feature_id=feature_id,
            description=description,
            patch_path=str(patch_path) if patch_path.exists() else None,
            tests_patch_path=str(tests_patch_path) if tests_patch_path.exists() else None,
        )

    return features


def _load_subset_filter(dataset_path: Path, subset: str) -> Optional[set[str]]:
    """Load subset filter (lite.json or flash.json).

    Supports two filesystem layouts:
    1. dataset/<subset>.json (flat)
    2. dataset/subsets/<subset>.json (CooperBench standard)

    Supports multiple JSON formats:
    1. Flat list of strings: ["repo/task_id", ...]
    2. Flat list of dicts: [{"repo": ..., "task_id": ..., "features": [...]}, ...]
    3. CooperBench format: {"name": ..., "tasks": [{"repo": ..., "task_id": int, "pairs": [[a,b], ...]}, ...]}

    Args:
        dataset_path: Root path to CooperBench dataset.
        subset: Subset name ("lite" or "flash").

    Returns:
        Set of allowed task keys (repo/task_id and repo/task_id/features_X_Y), or None if no filter file.
    """
    # Try both possible locations
    subset_file = dataset_path / f"{subset}.json"
    if not subset_file.exists():
        subset_file = dataset_path / "subsets" / f"{subset}.json"
    if not subset_file.exists():
        logger.warning(f"Subset file not found in {dataset_path} or {dataset_path}/subsets/, loading all tasks")
        return None

    try:
        with subset_file.open() as f:
            subset_data = json.load(f)

        allowed = set()

        # CooperBench format: {"name": ..., "tasks": [...]}
        if isinstance(subset_data, dict) and "tasks" in subset_data:
            has_pairs = False
            for item in subset_data["tasks"]:
                repo = item.get("repo", "")
                task_id = item.get("task_id", "")
                if repo and task_id is not None:
                    # task_id may be int in JSON, directory is "task<int>"
                    task_dir_name = f"task{task_id}" if not str(task_id).startswith("task") else str(task_id)
                    pairs = item.get("pairs", [])
                    if pairs:
                        # When pairs are explicitly listed, ONLY add pair-level keys
                        # (not task-level) so pair filtering is enforced
                        has_pairs = True
                        for pair in pairs:
                            if len(pair) == 2:
                                feature_key = "_".join(str(f) for f in sorted(pair))
                                allowed.add(f"{repo}/{task_dir_name}/features_{feature_key}")
                    else:
                        # No pairs specified — allow all pairs from this task
                        allowed.add(f"{repo}/{task_dir_name}")
            logger.info(f"Loaded CooperBench subset '{subset}' with {len(allowed)} entries "
                        f"({subset_data.get('stats', {}).get('tasks', '?')} tasks, "
                        f"{subset_data.get('stats', {}).get('pairs', '?')} pairs), "
                        f"pair-level filtering={'yes' if has_pairs else 'no'}")
            return allowed if allowed else None

        # Flat list format
        if isinstance(subset_data, list):
            for item in subset_data:
                if isinstance(item, dict):
                    repo = item.get("repo", "")
                    task_id = item.get("task_id", "")
                    if repo and task_id is not None:
                        task_dir_name = f"task{task_id}" if not str(task_id).startswith("task") else str(task_id)
                        allowed.add(f"{repo}/{task_dir_name}")
                    features = item.get("features")
                    if features and repo and task_id:
                        feature_key = "_".join(str(f) for f in sorted(features))
                        allowed.add(f"{repo}/{task_dir_name}/features_{feature_key}")
                elif isinstance(item, str):
                    allowed.add(item)

        logger.info(f"Loaded subset filter '{subset}' with {len(allowed)} entries")
        return allowed if allowed else None

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse subset file {subset_file}: {e}")
        return None


def load_cooperbench_dataset(
    dataset_path: str,
    subset: Optional[str] = None,
    repo_filter: Optional[list[str]] = None,
    task_filter: Optional[list[str]] = None,
    pair_features: bool = True,
) -> list[CooperBenchProblem]:
    """Load CooperBench dataset from filesystem.

    Discovers repositories, tasks, and features from the dataset directory
    structure. Optionally filters by subset, repo, or task.

    When pair_features is True (default), generates all 2-feature
    combinations for tasks with more than 2 features. When False,
    returns one problem per task with all features.

    Args:
        dataset_path: Root path to CooperBench dataset directory.
        subset: Optional subset filter ("lite" or "flash").
        repo_filter: Optional list of repository names to include.
        task_filter: Optional list of task IDs to include.
        pair_features: If True, generate feature pairs (nC2 combinations).

    Returns:
        List of CooperBenchProblem objects.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
    """
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Load subset filter if specified
    subset_allowed: Optional[set[str]] = None
    if subset:
        subset_allowed = _load_subset_filter(root, subset)

    problems: list[CooperBenchProblem] = []

    # Discover repositories
    for repo_dir in sorted(root.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name.startswith("."):
            continue

        repo_name = repo_dir.name

        # Apply repo filter
        if repo_filter and repo_name not in repo_filter:
            continue

        # Discover tasks
        for task_dir in sorted(repo_dir.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("task"):
                continue

            task_id = task_dir.name

            # Apply task filter
            if task_filter and task_id not in task_filter:
                continue

            # Apply subset filter (repo/task level)
            task_key = f"{repo_name}/{task_id}"
            if subset_allowed and task_key not in subset_allowed:
                # Check if any feature pairs from this task are in the subset
                has_matching_pair = any(
                    key.startswith(task_key + "/") for key in subset_allowed
                )
                if not has_matching_pair and task_key not in subset_allowed:
                    continue

            # Discover features
            features = _discover_features(task_dir)
            if len(features) < 2:
                logger.debug(
                    f"Skipping {task_key}: only {len(features)} features (need >= 2)"
                )
                continue

            # Resolve task-level files
            setup_script = task_dir / "setup.sh"
            run_tests_script = task_dir / "run_tests.sh"
            runner_script = task_dir / "runner.sh"
            dockerfile = task_dir / "Dockerfile"
            combined_patch = task_dir / "combined.patch"

            # Build image name from repo and task
            image_name = f"cooperbench-{repo_name}-{task_id}".lower()

            # Generate feature pairs or single problem
            feature_ids = sorted(features.keys())

            if pair_features and len(feature_ids) >= 2:
                # Generate all 2-feature combinations
                for pair in combinations(feature_ids, 2):
                    pair_list = list(pair)
                    pair_specs = {fid: features[fid] for fid in pair_list}

                    # Apply subset filter at pair level
                    feature_key = "_".join(str(f) for f in pair_list)
                    pair_key = f"{task_key}/features_{feature_key}"
                    if subset_allowed and pair_key not in subset_allowed and task_key not in subset_allowed:
                        continue

                    problem = CooperBenchProblem(
                        repo=repo_name,
                        task_id=task_id,
                        features=pair_list,
                        feature_specs=pair_specs,
                        dataset_path=dataset_path,
                        setup_script=str(setup_script) if setup_script.exists() else None,
                        run_tests_script=str(run_tests_script) if run_tests_script.exists() else None,
                        runner_script=str(runner_script) if runner_script.exists() else None,
                        dockerfile=str(dockerfile) if dockerfile.exists() else None,
                        combined_patch=str(combined_patch) if combined_patch.exists() else None,
                        image_name=image_name,
                    )
                    problems.append(problem)
            else:
                # Single problem with all features
                problem = CooperBenchProblem(
                    repo=repo_name,
                    task_id=task_id,
                    features=feature_ids,
                    feature_specs=features,
                    dataset_path=dataset_path,
                    setup_script=str(setup_script) if setup_script.exists() else None,
                    run_tests_script=str(run_tests_script) if run_tests_script.exists() else None,
                    runner_script=str(runner_script) if runner_script.exists() else None,
                    dockerfile=str(dockerfile) if dockerfile.exists() else None,
                    combined_patch=str(combined_patch) if combined_patch.exists() else None,
                    image_name=image_name,
                )
                problems.append(problem)

    logger.info(
        f"Loaded {len(problems)} problems from CooperBench dataset "
        f"(path={dataset_path}, subset={subset})"
    )
    return problems
