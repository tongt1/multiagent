"""MATH 500 dataset loader with stratified sampling and ambiguity filtering.

Provides a curated subset of 500 MATH problems sampled across all difficulty levels
(1-5) with problems having ambiguous answer formats filtered out.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from loguru import logger

from src.data.dataset_loader import Problem
from src.evaluation.math_verifier import extract_boxed_answer


def load_math500(
    seed: int = 42,
    force_reload: bool = False,
    cache_path: Optional[Path] = None,
) -> list[Problem]:
    """Load MATH 500 dataset with stratified sampling and ambiguity filtering.

    Samples 500 problems from the MATH test set (100 per difficulty level 1-5),
    filtering out problems without extractable \\boxed{} answers.

    Args:
        seed: Random seed for reproducible sampling (default: 42)
        force_reload: If True, ignore cache and reload from source (default: False)
        cache_path: Custom cache file path (default: data/math500_cache.json)

    Returns:
        List of 500 Problem objects with balanced difficulty distribution

    Notes:
        - Primary source: HuggingFace hendrycks/competition_math
        - Fallback: Local JSON files at data/MATH/test/**/*.json
        - Problems without extractable \\boxed{} answers are filtered before sampling
        - Results are cached to data/math500_cache.json for fast subsequent loads
    """
    # Set default cache path
    if cache_path is None:
        cache_path = Path("data/math500_cache.json")

    # Check cache first
    if not force_reload and cache_path.exists():
        logger.info(f"Loading MATH 500 from cache: {cache_path}")
        return _load_from_cache(cache_path)

    # Set random seed for reproducibility
    random.seed(seed)

    # Load full MATH test dataset
    logger.info("Loading MATH test dataset for filtering and sampling...")
    raw_problems = _load_math_test_dataset()

    # Filter out problems without extractable boxed answers
    logger.info("Filtering problems without extractable \\boxed{} answers...")
    filtered_problems = _filter_ambiguous_problems(raw_problems)

    # Stratified sampling across difficulty levels
    logger.info("Performing stratified sampling across difficulty levels...")
    sampled_problems = _stratified_sample(filtered_problems, target_per_level=100, seed=seed)

    # Convert to Problem objects with proper IDs
    logger.info("Converting to Problem objects...")
    problems = _convert_to_problems(sampled_problems)

    # Cache results
    logger.info(f"Caching results to {cache_path}...")
    _save_to_cache(problems, cache_path)

    logger.info(f"Successfully loaded MATH 500 dataset: {len(problems)} problems")
    return problems


def get_math500_stats(problems: list[Problem]) -> dict:
    """Return distribution statistics for MATH 500 problems.

    Args:
        problems: List of Problem objects from load_math500()

    Returns:
        Dictionary with keys:
            - total: Total number of problems
            - per_level: Dict mapping level (1-5) to count
            - per_type: Dict mapping math type (Algebra, Geometry, etc.) to count
    """
    stats = {
        "total": len(problems),
        "per_level": defaultdict(int),
        "per_type": defaultdict(int),
    }

    for problem in problems:
        level = problem.metadata.get("level")
        if level is not None:
            stats["per_level"][level] += 1

        math_type = problem.metadata.get("type")
        if math_type:
            stats["per_type"][math_type] += 1

    # Convert defaultdicts to regular dicts for cleaner output
    stats["per_level"] = dict(stats["per_level"])
    stats["per_type"] = dict(stats["per_type"])

    return stats


def _load_math_test_dataset() -> list[dict]:
    """Load MATH test dataset from HuggingFace or local fallback.

    Returns:
        List of raw problem dicts from dataset

    Raises:
        RuntimeError: If both HuggingFace and local fallback fail
    """
    # Try HuggingFace first
    try:
        logger.info("Attempting to load from HuggingFace: hendrycks/competition_math")
        dataset = load_dataset("hendrycks/competition_math", split="test")
        items = list(dataset)
        logger.info(f"Loaded {len(items)} problems from HuggingFace")
        return items
    except Exception as e:
        logger.warning(f"HuggingFace load failed: {e}")
        logger.info("Attempting local fallback...")

    # Try local fallback
    local_path = Path("data/MATH/test")
    if not local_path.exists():
        raise RuntimeError(
            "Failed to load MATH dataset: HuggingFace unavailable and "
            f"local fallback directory not found at {local_path}"
        )

    # Load all JSON files from local directory
    items = []
    json_files = list(local_path.rglob("*.json"))

    if not json_files:
        raise RuntimeError(
            f"No JSON files found in local fallback directory: {local_path}"
        )

    logger.info(f"Loading {len(json_files)} problems from local files...")
    for json_file in json_files:
        with json_file.open() as f:
            item = json.load(f)
            items.append(item)

    logger.info(f"Loaded {len(items)} problems from local fallback")
    return items


def _filter_ambiguous_problems(raw_problems: list[dict]) -> list[dict]:
    """Filter out problems without extractable boxed answers.

    Args:
        raw_problems: List of raw problem dicts from dataset

    Returns:
        List of problems with extractable \\boxed{} answers
    """
    filtered = []
    filtered_count = 0

    for item in raw_problems:
        solution = item.get("solution", "")
        extracted = extract_boxed_answer(solution)

        if extracted is not None:
            filtered.append(item)
        else:
            filtered_count += 1

    logger.info(
        f"Filtered {filtered_count} problems without extractable answers. "
        f"Remaining: {len(filtered)}"
    )

    return filtered


def _stratified_sample(
    filtered_problems: list[dict],
    target_per_level: int,
    seed: int,
) -> list[dict]:
    """Sample problems with stratified distribution across difficulty levels.

    Args:
        filtered_problems: Problems after ambiguity filtering
        target_per_level: Target number of problems per level (typically 100)
        seed: Random seed for reproducibility

    Returns:
        List of sampled problems with balanced difficulty distribution
    """
    # Group by difficulty level
    by_level = defaultdict(list)
    for item in filtered_problems:
        level_str = item.get("level", "")
        # Extract numeric level from "Level X" format
        if level_str.startswith("Level "):
            try:
                level = int(level_str.split()[1])
                by_level[level].append(item)
            except (IndexError, ValueError):
                logger.warning(f"Could not parse level from: {level_str}")
                continue

    # Log distribution before sampling
    logger.info("Problems per level before sampling:")
    for level in sorted(by_level.keys()):
        logger.info(f"  Level {level}: {len(by_level[level])} problems")

    # Sample from each level
    sampled = []
    shortfall = 0

    for level in sorted(by_level.keys()):
        available = by_level[level]
        if len(available) >= target_per_level:
            # Enough problems, sample target amount
            sample = random.sample(available, target_per_level)
            sampled.extend(sample)
        else:
            # Not enough problems, take all and track shortfall
            sampled.extend(available)
            shortfall += target_per_level - len(available)
            logger.warning(
                f"Level {level} has only {len(available)} problems, "
                f"need {target_per_level}"
            )

    # Redistribute shortfall if needed
    if shortfall > 0:
        logger.info(f"Redistributing shortfall of {shortfall} problems...")
        # Get levels with surplus
        surplus_levels = [
            level for level in by_level.keys()
            if len(by_level[level]) > target_per_level
        ]

        if surplus_levels:
            # Calculate how many extra to take from each surplus level
            extra_per_level = shortfall // len(surplus_levels)
            remainder = shortfall % len(surplus_levels)

            for level in surplus_levels:
                available = by_level[level]
                # Get problems not already sampled
                already_sampled = [p for p in sampled if p.get("level") == f"Level {level}"]
                remaining = [p for p in available if p not in already_sampled]

                extra = extra_per_level + (1 if remainder > 0 else 0)
                if remainder > 0:
                    remainder -= 1

                if len(remaining) >= extra:
                    extra_sample = random.sample(remaining, extra)
                    sampled.extend(extra_sample)

    # Log final distribution
    final_by_level = defaultdict(int)
    for item in sampled:
        level_str = item.get("level", "")
        if level_str.startswith("Level "):
            try:
                level = int(level_str.split()[1])
                final_by_level[level] += 1
            except (IndexError, ValueError):
                continue

    logger.info("Final distribution after sampling:")
    for level in sorted(final_by_level.keys()):
        logger.info(f"  Level {level}: {final_by_level[level]} problems")

    return sampled


def _convert_to_problems(sampled_problems: list[dict]) -> list[Problem]:
    """Convert raw problem dicts to Problem objects with proper IDs.

    Args:
        sampled_problems: List of sampled raw problem dicts

    Returns:
        List of Problem objects with sequential IDs
    """
    problems = []

    for idx, item in enumerate(sampled_problems):
        # Parse level as integer
        level_str = item.get("level", "Level 1")
        level_int = 1
        if level_str.startswith("Level "):
            try:
                level_int = int(level_str.split()[1])
            except (IndexError, ValueError):
                logger.warning(f"Could not parse level from: {level_str}")

        problem = Problem(
            id=f"math500_{idx:04d}",
            problem=item.get("problem", ""),
            ground_truth=item.get("solution", ""),
            domain="math",
            metadata={
                "level": level_int,
                "type": item.get("type", "Unknown"),
                "source": "MATH-500",
                "difficulty": level_str,
            },
        )
        problems.append(problem)

    return problems


def _load_from_cache(cache_path: Path) -> list[Problem]:
    """Load problems from cache file.

    Args:
        cache_path: Path to cache JSON file

    Returns:
        List of Problem objects
    """
    with cache_path.open() as f:
        data = json.load(f)

    problems = [Problem(**item) for item in data]
    logger.info(f"Loaded {len(problems)} problems from cache")
    return problems


def _save_to_cache(problems: list[Problem], cache_path: Path) -> None:
    """Save problems to cache file.

    Args:
        problems: List of Problem objects to cache
        cache_path: Path to cache JSON file
    """
    # Ensure directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable dicts
    data = [problem.model_dump() for problem in problems]

    with cache_path.open("w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Cached {len(problems)} problems to {cache_path}")
