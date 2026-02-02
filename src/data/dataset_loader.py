"""Dataset loading utilities for MATH, HumanEval, and local datasets."""

import json
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel, Field


class Problem(BaseModel):
    """Represents a problem from a dataset."""

    id: str
    problem: str
    ground_truth: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    domain: Literal["math", "code", "general"] = "general"


def load_math_dataset(split: str = "test", limit: Optional[int] = None) -> list[Problem]:
    """Load MATH dataset from HuggingFace.

    Args:
        split: Dataset split to load (train/test)
        limit: Maximum number of problems to load

    Returns:
        List of Problem objects
    """
    logger.info(f"Loading MATH dataset (split={split}, limit={limit})")

    # Load dataset with streaming if limit is set
    if limit:
        dataset = load_dataset("hendrycks/competition_math", split=split, streaming=True)
        items = list(dataset.take(limit))
    else:
        dataset = load_dataset("hendrycks/competition_math", split=split)
        items = list(dataset)

    problems = []
    for idx, item in enumerate(items):
        problem = Problem(
            id=f"math_{split}_{idx}",
            problem=item["problem"],
            ground_truth=item.get("solution"),
            metadata={
                "level": item.get("level"),
                "type": item.get("type"),
                "source": "hendrycks/competition_math",
                "split": split,
            },
            domain="math",
        )
        problems.append(problem)

    logger.info(f"Loaded {len(problems)} problems from MATH dataset")
    return problems


def load_humaneval_dataset(limit: Optional[int] = None) -> list[Problem]:
    """Load HumanEval dataset from HuggingFace.

    Args:
        limit: Maximum number of problems to load

    Returns:
        List of Problem objects
    """
    logger.info(f"Loading HumanEval dataset (limit={limit})")

    # Load dataset
    if limit:
        dataset = load_dataset("openai/openai_humaneval", split="test", streaming=True)
        items = list(dataset.take(limit))
    else:
        dataset = load_dataset("openai/openai_humaneval", split="test")
        items = list(dataset)

    problems = []
    for item in items:
        problem = Problem(
            id=item["task_id"],
            problem=item["prompt"],
            ground_truth=item.get("canonical_solution"),
            metadata={
                "test": item.get("test"),
                "entry_point": item.get("entry_point"),
                "source": "openai/openai_humaneval",
            },
            domain="code",
        )
        problems.append(problem)

    logger.info(f"Loaded {len(problems)} problems from HumanEval dataset")
    return problems


def load_local_dataset(path: Path) -> list[Problem]:
    """Load dataset from local YAML or JSON file.

    Args:
        path: Path to YAML or JSON file

    Returns:
        List of Problem objects

    Expected format:
        - List of dicts with at minimum 'problem' field
        - Optional fields: id, ground_truth, metadata, domain
    """
    logger.info(f"Loading local dataset from {path}")

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Load file based on extension
    if path.suffix in [".yaml", ".yml"]:
        with path.open() as f:
            data = yaml.safe_load(f)
    elif path.suffix == ".json":
        with path.open() as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .yaml, .yml, or .json")

    if not isinstance(data, list):
        raise ValueError("Dataset file must contain a list of problem dicts")

    problems = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not a dict: {item}")

        if "problem" not in item:
            raise ValueError(f"Item {idx} missing required 'problem' field")

        problem = Problem(
            id=item.get("id", f"local_{idx}"),
            problem=item["problem"],
            ground_truth=item.get("ground_truth"),
            metadata=item.get("metadata", {}),
            domain=item.get("domain", "general"),
        )
        problems.append(problem)

    logger.info(f"Loaded {len(problems)} problems from local file")
    return problems


class DatasetLoader:
    """Main entry point for loading datasets from various sources."""

    def load(self, source: str, **kwargs: Any) -> list[Problem]:
        """Load dataset from specified source.

        Args:
            source: Dataset source - "math", "humaneval", or path to local file
            **kwargs: Additional arguments passed to specific loaders
                - For math: split, limit
                - For humaneval: limit
                - For local: no additional args

        Returns:
            List of Problem objects
        """
        if source == "math":
            return load_math_dataset(
                split=kwargs.get("split", "test"), limit=kwargs.get("limit")
            )
        elif source == "humaneval":
            return load_humaneval_dataset(limit=kwargs.get("limit"))
        else:
            # Treat as file path
            return load_local_dataset(Path(source))
