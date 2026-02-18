"""JSONL trajectory reader with run_id grouping.

Reads trajectory JSONL files and groups entries by run_id for per-problem
sample construction.
"""

import json
from collections import defaultdict
from pathlib import Path

from loguru import logger

from src.models.trajectory import TrajectoryEntry


def read_and_group_trajectories(jsonl_path: Path) -> dict[str, list[TrajectoryEntry]]:
    """Read JSONL file and group entries by run_id.

    Args:
        jsonl_path: Path to a trajectory JSONL file.

    Returns:
        Dict mapping run_id -> list of TrajectoryEntry in file order.
    """
    groups: dict[str, list[TrajectoryEntry]] = defaultdict(list)

    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entry = TrajectoryEntry.model_validate(data)
                groups[entry.run_id].append(entry)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Skipping malformed line {line_num}: {e}")

    return dict(groups)
