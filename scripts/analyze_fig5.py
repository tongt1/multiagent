#!/usr/bin/env python3
"""Compute Figure 5 metrics: success rates, merge conflict rates, speech acts, overhead.

Reads: data/results.json
Writes: data/fig5_metrics.json

Produces all data for Figure 5's three panels:
  (a) Comm vs no-comm success rate comparison
  (b) Merge conflict reduction from communication
  (c) Communication overhead breakdown with speech act classification

Usage:
    python scripts/analyze_fig5.py
    python scripts/analyze_fig5.py --input data/results.json --output data/fig5_metrics.json
"""

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "fig5_metrics.json"

# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, total: int) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds for the 95% CI.

    Verified edge cases:
        wilson_ci(0, 100)  -> (0.0, 0.036995)    # p=0
        wilson_ci(100, 100) -> (0.963005, 1.0)    # p=1
        wilson_ci(0, 0)    -> (0.0, 1.0)          # no data
        wilson_ci(2, 3)    -> (0.207655, 0.938510) # small n
    """
    if total == 0:
        return (0.0, 1.0)

    z = 1.96  # 95% CI
    p_hat = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (round(lower, 6), round(upper, 6))


# ---------------------------------------------------------------------------
# Speech act classification (regex heuristics)
# ---------------------------------------------------------------------------

PLAN_PATTERNS = re.compile(
    r'\b('
    r'I will|I am going to|I\'ll|I plan to|I intend to|'
    r'I will be|I\'m going to|my plan|my approach|my strategy|'
    r'Let me|I am modifying|I am implementing|I am adding|'
    r'I have implemented|I\'ve implemented|'
    r'I suggest|I recommend|I propose|'
    r'my task is|I\'m working on|I am working on'
    r')\b',
    re.IGNORECASE,
)

QUESTION_PATTERNS = re.compile(
    r'\?|'
    r'\b('
    r'can you|could you|would you|will you|do you|did you|'
    r'is there|are there|have you|shall we|should we|'
    r'please let me know|please confirm|please ensure|'
    r'does this|does that|is this|is that'
    r')\b',
    re.IGNORECASE,
)

UPDATE_PATTERNS = re.compile(
    r'\b('
    r'I have completed|I have finished|I\'m done|I\'ve done|'
    r'I completed|done with|completed the|finished the|'
    r'I have made|I have added|changes are complete|'
    r'changes done|modifications complete|is ready|'
    r'has been implemented|my task is complete|'
    r'I\'ve added|I\'ve made|I\'ve modified|'
    r'I have resolved|acknowledged|proceeding with'
    r')\b',
    re.IGNORECASE,
)


def classify_speech_act(message_text: str) -> str:
    """Classify a message into plan/question/update/other.

    Priority ordering: question > update > plan > other.
    Each message gets exactly ONE classification -- no double-counting.
    """
    if QUESTION_PATTERNS.search(message_text):
        return "question"
    if UPDATE_PATTERNS.search(message_text):
        return "update"
    if PLAN_PATTERNS.search(message_text):
        return "plan"
    return "other"


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------


def compute_success_rates(records: list[dict]) -> dict:
    """Compute comm vs no-comm success rates (FIG5-01).

    Excludes records with eval_error or infra_error.
    """
    result = {}
    for setting in ["coop-comm", "coop-nocomm"]:
        setting_key = setting.replace("-", "_")
        recs = [
            r for r in records
            if r["setting"] == setting
            and r.get("eval_error") is None
            and not r.get("infra_error", False)
        ]
        successes = sum(1 for r in recs if r["both_passed"])
        total = len(recs)
        rate = successes / total if total > 0 else 0.0
        ci_lower, ci_upper = wilson_ci(successes, total)
        result[setting_key] = {
            "successes": successes,
            "total": total,
            "rate": round(rate, 6),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    return result


def compute_merge_conflict_rates(records: list[dict]) -> dict:
    """Compute merge conflict rates with and without communication (FIG5-02).

    Uses ALL coop records (including eval_error) because merge outcomes are
    valid even when eval subsequently fails. The eval_error records all have
    merge_failed outcomes, as the merge failure causes the eval error.
    """
    result = {}
    for setting in ["coop-comm", "coop-nocomm"]:
        setting_key = setting.replace("-", "_")
        recs = [r for r in records if r["setting"] == setting]
        conflicts = sum(1 for r in recs if r["merge_outcome"] == "merge_failed")
        total = len(recs)
        rate = conflicts / total if total > 0 else 0.0
        ci_lower, ci_upper = wilson_ci(conflicts, total)
        result[setting_key] = {
            "conflicts": conflicts,
            "total": total,
            "rate": round(rate, 6),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    # Compute difference
    comm_rate = result["coop_comm"]["rate"]
    nocomm_rate = result["coop_nocomm"]["rate"]
    diff = nocomm_rate - comm_rate
    result["difference"] = {
        "value": round(diff, 6),
        "direction": (
            f"comm reduces conflicts by {abs(diff)*100:.1f}%"
            if diff > 0
            else f"comm increases conflicts by {abs(diff)*100:.1f}%"
        ),
    }
    return result


def compute_speech_acts(records: list[dict]) -> dict:
    """Classify agent messages into speech act types (FIG5-03).

    Collects ALL messages from coop-comm records (regardless of eval_error --
    messages exist even for errored records).
    """
    counts: Counter = Counter()
    total_messages = 0

    for r in records:
        if r["setting"] != "coop-comm":
            continue
        for msg in r.get("messages", []):
            text = msg.get("message", "")
            if not text:
                continue
            category = classify_speech_act(text)
            counts[category] += 1
            total_messages += 1

    result = {"total_messages": total_messages}

    # Compute percentages with largest-remainder rounding to ensure sum == 100.0
    categories = ["plan", "question", "update", "other"]
    raw_pcts = {}
    for cat in categories:
        count = counts.get(cat, 0)
        raw_pcts[cat] = count / total_messages * 100 if total_messages > 0 else 0.0

    # Round each to 1 decimal place, then adjust the largest remainder
    # to ensure the sum is exactly 100.0
    floored = {cat: math.floor(raw_pcts[cat] * 10) / 10 for cat in categories}
    remainder = {cat: raw_pcts[cat] - floored[cat] for cat in categories}
    floored_sum = sum(floored.values())
    deficit = round(100.0 - floored_sum, 1)
    # Distribute deficit to categories with largest remainders
    steps = int(round(deficit * 10))  # number of 0.1 increments to add
    sorted_cats = sorted(categories, key=lambda c: -remainder[c])
    rounded_pcts = dict(floored)
    for i in range(steps):
        rounded_pcts[sorted_cats[i % len(sorted_cats)]] = round(
            rounded_pcts[sorted_cats[i % len(sorted_cats)]] + 0.1, 1
        )

    for cat in categories:
        result[cat] = {
            "count": counts.get(cat, 0),
            "pct": rounded_pcts[cat],
        }

    return result


def compute_overhead(records: list[dict]) -> dict:
    """Compute communication overhead as percentage of action budget (FIG5-04).

    For each coop-comm record with messages_count > 0 and total_steps > 0:
        overhead_pct = messages_count / total_steps * 100
    """
    per_task = []
    for r in records:
        if r["setting"] != "coop-comm":
            continue
        messages_count = r.get("messages_count", 0)
        total_steps = r.get("total_steps", 0)
        if messages_count > 0 and total_steps and total_steps > 0:
            overhead_pct = messages_count / total_steps * 100
            per_task.append({
                "task_id": r.get("task_id"),
                "repo": r.get("repo"),
                "messages": messages_count,
                "total_steps": total_steps,
                "overhead_pct": round(overhead_pct, 1),
            })

    if not per_task:
        return {
            "mean_pct": 0.0,
            "median_pct": 0.0,
            "min_pct": 0.0,
            "max_pct": 0.0,
            "std_pct": 0.0,
            "n_tasks": 0,
            "per_task": [],
        }

    overheads = [t["overhead_pct"] for t in per_task]
    return {
        "mean_pct": round(statistics.mean(overheads), 1),
        "median_pct": round(statistics.median(overheads), 1),
        "min_pct": round(min(overheads), 1),
        "max_pct": round(max(overheads), 1),
        "std_pct": round(statistics.stdev(overheads), 1) if len(overheads) > 1 else 0.0,
        "n_tasks": len(per_task),
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary(metrics: dict) -> None:
    """Print human-readable summary of all Figure 5 metrics."""
    print("\n" + "=" * 60)
    print("FIGURE 5 METRICS SUMMARY")
    print("=" * 60)

    # Success rates
    sr = metrics["success_rates"]
    print("\n--- Success Rates (FIG5-01) ---")
    for key, label in [("coop_comm", "Coop-Comm"), ("coop_nocomm", "Coop-NoComm")]:
        d = sr[key]
        print(
            f"  {label}: {d['successes']}/{d['total']} = {d['rate']*100:.1f}% "
            f"[{d['ci_lower']*100:.1f}%, {d['ci_upper']*100:.1f}%]"
        )

    # Merge conflict rates
    mc = metrics["merge_conflict_rates"]
    print("\n--- Merge Conflict Rates (FIG5-02) ---")
    for key, label in [("coop_comm", "Coop-Comm"), ("coop_nocomm", "Coop-NoComm")]:
        d = mc[key]
        print(
            f"  {label}: {d['conflicts']}/{d['total']} = {d['rate']*100:.1f}% "
            f"[{d['ci_lower']*100:.1f}%, {d['ci_upper']*100:.1f}%]"
        )
    diff = mc["difference"]
    print(f"  Difference: {diff['direction']}")

    # Speech acts
    sa = metrics["speech_acts"]
    print(f"\n--- Speech Act Classification (FIG5-03) ---")
    print(f"  Total messages: {sa['total_messages']}")
    for cat in ["plan", "question", "update", "other"]:
        d = sa[cat]
        print(f"  {cat:>10}: {d['count']:>4} ({d['pct']:.1f}%)")
    check_sum = sum(sa[c]["pct"] for c in ["plan", "question", "update", "other"])
    print(f"  Sum check: {check_sum:.1f}% (should be ~100%)")

    # Overhead
    oh = metrics["overhead"]
    print(f"\n--- Communication Overhead (FIG5-04) ---")
    print(f"  Tasks analyzed: {oh['n_tasks']}")
    print(f"  Mean:   {oh['mean_pct']:.1f}%")
    print(f"  Median: {oh['median_pct']:.1f}%")
    print(f"  Min:    {oh['min_pct']:.1f}%")
    print(f"  Max:    {oh['max_pct']:.1f}%")
    print(f"  Std:    {oh['std_pct']:.1f}%")

    # Metadata
    md = metrics["metadata"]
    print(f"\n--- Metadata ---")
    print(f"  Input: {md['input_file']}")
    print(f"  Total records: {md['total_records']}")
    print(f"  Coop-comm records: {md['coop_comm_records']}")
    print(f"  Coop-nocomm records: {md['coop_nocomm_records']}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Figure 5 metrics: success rates, merge conflict rates, "
        "speech acts, and communication overhead."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input file path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    # Load data
    if not args.input.is_file():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {args.input}")

    # Compute all metrics
    success_rates = compute_success_rates(records)
    merge_conflict_rates = compute_merge_conflict_rates(records)
    speech_acts = compute_speech_acts(records)
    overhead = compute_overhead(records)

    # Metadata
    coop_comm_count = sum(1 for r in records if r["setting"] == "coop-comm")
    coop_nocomm_count = sum(1 for r in records if r["setting"] == "coop-nocomm")

    metrics = {
        "success_rates": success_rates,
        "merge_conflict_rates": merge_conflict_rates,
        "speech_acts": speech_acts,
        "overhead": overhead,
        "metadata": {
            "input_file": str(args.input),
            "total_records": len(records),
            "coop_comm_records": coop_comm_count,
            "coop_nocomm_records": coop_nocomm_count,
            "speech_act_method": (
                "regex heuristics with priority ordering "
                "(question > update > plan > other)"
            ),
        },
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics to {args.output}")

    # Print human-readable summary
    print_summary(metrics)


if __name__ == "__main__":
    main()
