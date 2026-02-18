"""EvalLog builder for assembling complete Inspect log files.

Aggregates EvalSample objects into a complete EvalLog with EvalResults
containing per-scorer metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalMetric,
    EvalPlan,
    EvalResults,
    EvalSample,
    EvalScore,
    EvalSpec,
    EvalStats,
)


def build_eval_log(
    samples: list[EvalSample],
    task_name: str = "baseline_eval",
    model_name: str = "unknown",
    dataset_name: str = "unknown",
) -> EvalLog:
    """Assemble a complete EvalLog from a list of EvalSample objects.

    Computes aggregate scores for EvalResults, creating EvalScore entries
    for each scorer found across samples (e.g., "ground_truth", "judge").

    Args:
        samples: List of EvalSample objects to include.
        task_name: Task identifier for EvalSpec.
        model_name: Model identifier for EvalSpec.
        dataset_name: Dataset name for EvalSpec.

    Returns:
        A fully populated EvalLog ready to write with write_eval_log.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Compute aggregate scores for EvalResults
    # Collect all score values per scorer name across all samples
    scorer_values: dict[str, list[float]] = {}
    for sample in samples:
        if sample.scores:
            for scorer_name, score in sample.scores.items():
                if scorer_name not in scorer_values:
                    scorer_values[scorer_name] = []
                if isinstance(score.value, (int, float)):
                    scorer_values[scorer_name].append(float(score.value))

    # Build EvalScore objects for each scorer that has values
    eval_scores: list[EvalScore] = []
    for scorer_name, values in scorer_values.items():
        if values:
            mean_val = sum(values) / len(values)
            eval_scores.append(
                EvalScore(
                    name=scorer_name,
                    scorer=scorer_name,
                    params={},
                    metrics={
                        "mean": EvalMetric(name="mean", value=mean_val),
                        "count": EvalMetric(name="count", value=len(values)),
                    },
                )
            )

    # Assemble the full EvalLog
    spec = EvalSpec(
        task=task_name,
        model=model_name,
        created=now,
        dataset=EvalDataset(name=dataset_name, samples=len(samples)),
        config=EvalConfig(),
    )

    plan = EvalPlan(name="baseline_eval")

    results = EvalResults(
        total_samples=len(samples),
        completed_samples=len(samples),
        scores=eval_scores,
    )

    stats = EvalStats(
        started_at=now,
        completed_at=now,
    )

    return EvalLog(
        version=2,
        status="success",
        eval=spec,
        plan=plan,
        results=results,
        stats=stats,
        samples=samples,
    )
