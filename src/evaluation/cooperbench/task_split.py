"""Deterministic 80/20 train/eval splits for CooperBench tasks.

Provides seed-based task splitting for multi-seed training runs.
Each seed produces a DIFFERENT split (rotated eval set), enabling
variance measurement across seeds while ensuring reproducibility.

The split is at the task level (not pair level) -- all pairs from
a given task go to either train or eval, never both.
"""

import random

# All 26 task IDs in the CooperBench lite subset (sorted).
# Each ID is "repo/taskN" matching the dataset directory structure.
LITE_V4_TASK_IDS: list[str] = [
    "dottxt_ai_outlines_task/task1655",
    "dottxt_ai_outlines_task/task1706",
    "dspy_task/task8394",
    "dspy_task/task8587",
    "dspy_task/task8635",
    "go_chi_task/task26",
    "go_chi_task/task27",
    "go_chi_task/task56",
    "huggingface_datasets_task/task3997",
    "huggingface_datasets_task/task6252",
    "llama_index_task/task17244",
    "llama_index_task/task18813",
    "openai_tiktoken_task/task0",
    "pallets_click_task/task2068",
    "pallets_click_task/task2800",
    "pallets_click_task/task2956",
    "pallets_jinja_task/task1465",
    "pallets_jinja_task/task1559",
    "pallets_jinja_task/task1621",
    "pillow_task/task25",
    "pillow_task/task290",
    "pillow_task/task68",
    "react_hook_form_task/task153",
    "react_hook_form_task/task85",
    "samuelcolvin_dirty_equals_task/task43",
    "typst_task/task6554",
]

DEFAULT_SEEDS = [42, 43, 44]


def get_train_eval_split(
    task_ids: list[str],
    seed: int,
    eval_fraction: float = 0.2,
) -> tuple[list[str], list[str]]:
    """Compute a deterministic train/eval split for CooperBench tasks.

    Uses ``random.Random(seed)`` for deterministic shuffling -- no numpy
    dependency required.  Each seed produces a *different* split so that
    rotating the held-out set across seeds lets every task contribute to
    evaluation variance measurement.

    Args:
        task_ids: Full list of task IDs to partition.
        seed: RNG seed for reproducibility (different seed = different split).
        eval_fraction: Fraction of tasks held out for evaluation (default 0.2).

    Returns:
        Tuple of ``(train_ids, eval_ids)`` where the union equals *task_ids*
        and intersection is empty.
    """
    rng = random.Random(seed)
    shuffled = list(task_ids)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - eval_fraction))
    train_ids = sorted(shuffled[:split_idx])
    eval_ids = sorted(shuffled[split_idx:])
    return train_ids, eval_ids


def get_default_splits(
    n_seeds: int = 3,
) -> dict[int, tuple[list[str], list[str]]]:
    """Return train/eval splits for the default seed set (42, 43, 44).

    Convenience function for quickly obtaining splits for all planned
    multi-seed training runs.

    Args:
        n_seeds: Number of seeds to use from ``DEFAULT_SEEDS``.

    Returns:
        Dict mapping seed -> (train_ids, eval_ids).
    """
    seeds = DEFAULT_SEEDS[:n_seeds]
    return {
        seed: get_train_eval_split(LITE_V4_TASK_IDS, seed=seed)
        for seed in seeds
    }
