"""Inspect AI log generation from trajectory JSONL data.

Converts baseline trajectory data into valid Inspect EvalLog files
viewable in ``inspect view``.
"""

from .sample_builder import build_baseline_sample
from .trajectory_reader import read_and_group_trajectories

__all__ = [
    "build_baseline_sample",
    "read_and_group_trajectories",
]
