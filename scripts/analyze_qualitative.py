#!/usr/bin/env python3
"""Compute qualitative transcript analysis metrics.

Reads: data/results.json
Writes: data/qualitative_metrics.json

Requirements: QUAL-01, QUAL-02, QUAL-03, QUAL-04

Computes Plan:Question ratios, first-turn planning detection, specificity
metrics (file path and line number mentions), and correlates all with merge
conflict outcomes using Mann-Whitney U and Fisher's exact tests.

Usage:
    python scripts/analyze_qualitative.py
    python scripts/analyze_qualitative.py --input data/results.json --output data/qualitative_metrics.json
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

# Import speech act classifier from Phase 3
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from analyze_fig5 import classify_speech_act  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "qualitative_metrics.json"

# ---------------------------------------------------------------------------
# Specificity regex patterns (QUAL-03)
# ---------------------------------------------------------------------------

# File path detection: path/to/file.ext format (requires at least one /)
FILE_PATH_PATTERN = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.\w{1,5}")

# Standalone filename with known code extensions
STANDALONE_FILE_PATTERN = re.compile(
    r"\b[\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|rb|yml|yaml|json|toml|cfg|txt|md|html|css|scss)\b"
)

# Line number patterns
LINE_NUMBER_PATTERNS = [
    re.compile(r"\bline\s+\d+", re.IGNORECASE),
    re.compile(r"\blines?\s+\d+\s*[-\u2013to]+\s*\d+", re.IGNORECASE),
    re.compile(r"\bL\d+\b"),
    re.compile(r"\bat\s+line\b", re.IGNORECASE),
]


def count_file_mentions(text: str) -> int:
    """Return 1 if any file pattern matches the message, 0 otherwise.

    Counts per-message (not per-match) to avoid inflation.
    """
    if FILE_PATH_PATTERN.search(text) or STANDALONE_FILE_PATTERN.search(text):
        return 1
    return 0


def count_line_mentions(text: str) -> int:
    """Return 1 if any line number pattern matches the message, 0 otherwise."""
    return 1 if any(p.search(text) for p in LINE_NUMBER_PATTERNS) else 0


# ---------------------------------------------------------------------------
# Per-trajectory metric computation
# ---------------------------------------------------------------------------


def compute_trajectory_metrics(record: dict) -> dict:
    """Compute all qualitative metrics for a single coop-comm trajectory.

    Returns dict with speech act counts, Plan:Question ratio, first-turn
    planning flag, and specificity counts.
    """
    messages = record.get("messages", [])

    # Speech act classification using Phase 3 classifier
    acts = Counter(
        classify_speech_act(msg["message"]) for msg in messages if msg.get("message")
    )
    plans = acts.get("plan", 0)
    questions = acts.get("question", 0)

    # Plan:Question ratio (QUAL-01)
    if questions > 0:
        pq_ratio = plans / questions
    elif plans > 0:
        pq_ratio = None  # Infinite: plans but no questions
    else:
        pq_ratio = 0.0

    # First-turn planning (QUAL-02)
    first_turn_is_plan = False
    if messages and messages[0].get("message"):
        first_turn_is_plan = classify_speech_act(messages[0]["message"]) == "plan"

    # Specificity metrics (QUAL-03)
    file_mentions = sum(
        count_file_mentions(msg["message"])
        for msg in messages
        if msg.get("message")
    )
    line_mentions = sum(
        count_line_mentions(msg["message"])
        for msg in messages
        if msg.get("message")
    )

    return {
        "task_id": record["task_id"],
        "repo": record["repo"],
        "merge_outcome": record["merge_outcome"],
        "messages_count": len(messages),
        "plan_count": plans,
        "question_count": questions,
        "update_count": acts.get("update", 0),
        "other_count": acts.get("other", 0),
        "plan_question_ratio": round(pq_ratio, 4) if pq_ratio is not None else None,
        "first_turn_is_plan": first_turn_is_plan,
        "file_mentions": file_mentions,
        "line_mentions": line_mentions,
    }


# ---------------------------------------------------------------------------
# Group comparison and statistical tests
# ---------------------------------------------------------------------------


def compute_plan_question_correlation(trajectories: list[dict]) -> dict:
    """QUAL-01: Correlate Plan:Question ratios with merge outcomes.

    Uses Mann-Whitney U test on finite ratios only. Reports infinite-ratio
    counts separately.
    """
    clean_ratios = []
    clean_infinite = 0
    failed_ratios = []
    failed_infinite = 0

    for t in trajectories:
        is_clean = t["merge_outcome"] == "merge_clean"
        ratio = t["plan_question_ratio"]
        if ratio is None:
            if is_clean:
                clean_infinite += 1
            else:
                failed_infinite += 1
        else:
            if is_clean:
                clean_ratios.append(ratio)
            else:
                failed_ratios.append(ratio)

    # Mann-Whitney U on finite ratios
    if clean_ratios and failed_ratios:
        stat, p_value = mannwhitneyu(
            clean_ratios, failed_ratios, alternative="two-sided"
        )
        stat = round(float(stat), 4)
        p_value = round(float(p_value), 4)
    else:
        stat = None
        p_value = None

    clean_mean = round(statistics.mean(clean_ratios), 4) if clean_ratios else None
    clean_median = round(statistics.median(clean_ratios), 4) if clean_ratios else None
    failed_mean = round(statistics.mean(failed_ratios), 4) if failed_ratios else None
    failed_median = (
        round(statistics.median(failed_ratios), 4) if failed_ratios else None
    )

    # Determine direction
    if clean_mean is not None and failed_mean is not None:
        if failed_mean > clean_mean:
            direction = "higher ratio correlates with more conflicts"
        elif clean_mean > failed_mean:
            direction = "higher ratio correlates with fewer conflicts"
        else:
            direction = "no directional difference"
    else:
        direction = "insufficient data"

    return {
        "no_conflict": {
            "mean": clean_mean,
            "median": clean_median,
            "n_finite": len(clean_ratios),
            "n_infinite": clean_infinite,
        },
        "conflict": {
            "mean": failed_mean,
            "median": failed_median,
            "n_finite": len(failed_ratios),
            "n_infinite": failed_infinite,
        },
        "test": "Mann-Whitney U",
        "statistic": stat,
        "p_value": p_value,
        "direction": direction,
    }


def compute_first_turn_correlation(trajectories: list[dict]) -> dict:
    """QUAL-02: Correlate first-turn planning with merge outcomes.

    Uses Fisher's exact test on 2x2 contingency table.
    """
    plan_clean = sum(
        1
        for t in trajectories
        if t["first_turn_is_plan"] and t["merge_outcome"] == "merge_clean"
    )
    plan_failed = sum(
        1
        for t in trajectories
        if t["first_turn_is_plan"] and t["merge_outcome"] == "merge_failed"
    )
    noplan_clean = sum(
        1
        for t in trajectories
        if not t["first_turn_is_plan"] and t["merge_outcome"] == "merge_clean"
    )
    noplan_failed = sum(
        1
        for t in trajectories
        if not t["first_turn_is_plan"] and t["merge_outcome"] == "merge_failed"
    )

    table = [[plan_clean, plan_failed], [noplan_clean, noplan_failed]]
    odds_ratio, p_value = fisher_exact(table)

    total_plan = plan_clean + plan_failed
    total_noplan = noplan_clean + noplan_failed

    return {
        "contingency_table": {
            "plan_first_clean": plan_clean,
            "plan_first_conflict": plan_failed,
            "no_plan_first_clean": noplan_clean,
            "no_plan_first_conflict": noplan_failed,
        },
        "plan_first_conflict_rate": (
            round(plan_failed / total_plan, 4) if total_plan > 0 else None
        ),
        "no_plan_first_conflict_rate": (
            round(noplan_failed / total_noplan, 4) if total_noplan > 0 else None
        ),
        "fishers_exact_odds_ratio": round(float(odds_ratio), 4),
        "fishers_exact_p_value": round(float(p_value), 4),
        "n_plan_first": total_plan,
        "n_no_plan_first": total_noplan,
        "note": (
            f"Small no-plan-first sample (n={total_noplan}). Some messages "
            "classified as non-plan contain planning language with contractions "
            "(e.g., 'I'm modifying') not captured by the Phase 3 classifier."
        ),
    }


def compute_specificity_metrics(trajectories: list[dict]) -> dict:
    """QUAL-03: Compute file/line mention metrics grouped by merge outcome."""
    clean = [t for t in trajectories if t["merge_outcome"] == "merge_clean"]
    failed = [t for t in trajectories if t["merge_outcome"] == "merge_failed"]

    # File mentions
    clean_files = [t["file_mentions"] for t in clean]
    failed_files = [t["file_mentions"] for t in failed]

    clean_file_mean = round(statistics.mean(clean_files), 4) if clean_files else 0.0
    clean_file_median = (
        round(statistics.median(clean_files), 4) if clean_files else 0.0
    )
    failed_file_mean = round(statistics.mean(failed_files), 4) if failed_files else 0.0
    failed_file_median = (
        round(statistics.median(failed_files), 4) if failed_files else 0.0
    )

    # Mann-Whitney U on file mentions
    if clean_files and failed_files:
        file_stat, file_p = mannwhitneyu(
            clean_files, failed_files, alternative="two-sided"
        )
        file_stat = round(float(file_stat), 4)
        file_p = round(float(file_p), 4)
    else:
        file_stat = None
        file_p = None

    # Line mentions
    clean_lines = [t["line_mentions"] for t in clean]
    failed_lines = [t["line_mentions"] for t in failed]

    clean_line_mean = round(statistics.mean(clean_lines), 4) if clean_lines else 0.0
    clean_line_median = (
        round(statistics.median(clean_lines), 4) if clean_lines else 0.0
    )
    failed_line_mean = round(statistics.mean(failed_lines), 4) if failed_lines else 0.0
    failed_line_median = (
        round(statistics.median(failed_lines), 4) if failed_lines else 0.0
    )

    # Total message-level counts for reporting
    total_messages = sum(t["messages_count"] for t in trajectories)
    total_file_msg = sum(t["file_mentions"] for t in trajectories)
    total_line_msg = sum(t["line_mentions"] for t in trajectories)

    return {
        "file_mentions": {
            "no_conflict": {"mean": clean_file_mean, "median": clean_file_median},
            "conflict": {"mean": failed_file_mean, "median": failed_file_median},
            "test": "Mann-Whitney U",
            "statistic": file_stat,
            "p_value": file_p,
            "total_messages_with_file_mention": total_file_msg,
            "total_messages": total_messages,
        },
        "line_mentions": {
            "no_conflict": {"mean": clean_line_mean, "median": clean_line_median},
            "conflict": {"mean": failed_line_mean, "median": failed_line_median},
            "test": "not applicable",
            "total_messages_with_line_mention": total_line_msg,
            "total_messages": total_messages,
            "note": (
                f"Line number mentions are too sparse ({total_line_msg}/{total_messages} "
                "messages, <1%) to serve as a differentiating metric between "
                "conflict and no-conflict groups."
            ),
        },
    }


def build_summary_table(
    pq_result: dict, ft_result: dict, spec_result: dict
) -> dict:
    """QUAL-04: Assemble summary comparison table with all metrics."""
    return {
        "plan_question_ratio": {
            "no_conflict": pq_result["no_conflict"],
            "conflict": pq_result["conflict"],
            "test": pq_result["test"],
            "p_value": pq_result["p_value"],
            "direction": pq_result["direction"],
        },
        "first_turn_planning": {
            "no_conflict": {
                "rate": (
                    round(
                        ft_result["contingency_table"]["plan_first_clean"]
                        / (
                            ft_result["contingency_table"]["plan_first_clean"]
                            + ft_result["contingency_table"]["no_plan_first_clean"]
                        ),
                        4,
                    )
                    if (
                        ft_result["contingency_table"]["plan_first_clean"]
                        + ft_result["contingency_table"]["no_plan_first_clean"]
                    )
                    > 0
                    else None
                ),
                "count": ft_result["contingency_table"]["plan_first_clean"],
                "total": (
                    ft_result["contingency_table"]["plan_first_clean"]
                    + ft_result["contingency_table"]["no_plan_first_clean"]
                ),
            },
            "conflict": {
                "rate": (
                    round(
                        ft_result["contingency_table"]["plan_first_conflict"]
                        / (
                            ft_result["contingency_table"]["plan_first_conflict"]
                            + ft_result["contingency_table"]["no_plan_first_conflict"]
                        ),
                        4,
                    )
                    if (
                        ft_result["contingency_table"]["plan_first_conflict"]
                        + ft_result["contingency_table"]["no_plan_first_conflict"]
                    )
                    > 0
                    else None
                ),
                "count": ft_result["contingency_table"]["plan_first_conflict"],
                "total": (
                    ft_result["contingency_table"]["plan_first_conflict"]
                    + ft_result["contingency_table"]["no_plan_first_conflict"]
                ),
            },
            "test": "Fisher's exact",
            "p_value": ft_result["fishers_exact_p_value"],
            "odds_ratio": ft_result["fishers_exact_odds_ratio"],
            "direction": (
                "first-turn planning correlates with more conflicts"
                if ft_result["plan_first_conflict_rate"] is not None
                and ft_result["no_plan_first_conflict_rate"] is not None
                and ft_result["plan_first_conflict_rate"]
                > ft_result["no_plan_first_conflict_rate"]
                else "first-turn planning correlates with fewer conflicts"
            ),
        },
        "file_mentions_per_trajectory": {
            "no_conflict": spec_result["file_mentions"]["no_conflict"],
            "conflict": spec_result["file_mentions"]["conflict"],
            "test": spec_result["file_mentions"]["test"],
            "p_value": spec_result["file_mentions"]["p_value"],
            "direction": (
                "no significant difference"
                if spec_result["file_mentions"]["p_value"] is not None
                and spec_result["file_mentions"]["p_value"] > 0.05
                else "significant difference"
            ),
        },
        "line_mentions_per_trajectory": {
            "no_conflict": spec_result["line_mentions"]["no_conflict"],
            "conflict": spec_result["line_mentions"]["conflict"],
            "test": spec_result["line_mentions"]["test"],
            "note": spec_result["line_mentions"]["note"],
        },
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary_table(metrics: dict) -> None:
    """Print human-readable comparison table."""
    print("\n" + "=" * 80)
    print("QUALITATIVE METRICS: CONFLICT vs NO-CONFLICT TRAJECTORIES")
    print("=" * 80)

    # QUAL-01: Plan:Question Ratio
    pq = metrics["plan_question_ratio"]
    print("\n--- Plan:Question Ratio (QUAL-01) ---")
    nc = pq["no_conflict"]
    print(f"  No conflict (finite, n={nc['n_finite']}):")
    if nc["mean"] is not None:
        print(f"    Mean: {nc['mean']:.2f}, Median: {nc['median']:.2f}")
    print(f"    Infinite ratio (plans, no questions): {nc['n_infinite']}")
    c = pq["conflict"]
    print(f"  Conflict (finite, n={c['n_finite']}):")
    if c["mean"] is not None:
        print(f"    Mean: {c['mean']:.2f}, Median: {c['median']:.2f}")
    print(f"    Infinite ratio (plans, no questions): {c['n_infinite']}")
    print(f"  Test: {pq['test']}, stat={pq['statistic']}, p={pq['p_value']:.4f}")
    print(f"  Direction: {pq['direction']}")

    # QUAL-02: First-Turn Planning
    ft = metrics["first_turn_planning"]
    print(f"\n--- First-Turn Planning (QUAL-02) ---")
    print(
        f"  Plan-first conflict rate: "
        f"{ft['plan_first_conflict_rate']*100:.1f}% "
        f"(n={ft['n_plan_first']})"
    )
    print(
        f"  No-plan-first conflict rate: "
        f"{ft['no_plan_first_conflict_rate']*100:.1f}% "
        f"(n={ft['n_no_plan_first']})"
    )
    print(
        f"  Fisher's exact: OR={ft['fishers_exact_odds_ratio']:.2f}, "
        f"p={ft['fishers_exact_p_value']:.4f}"
    )
    print(f"  Note: {ft['note']}")

    # QUAL-03: Specificity
    spec = metrics["specificity"]
    fm = spec["file_mentions"]
    print(f"\n--- Specificity: File Mentions (QUAL-03) ---")
    print(
        f"  No conflict: mean={fm['no_conflict']['mean']:.2f}, "
        f"median={fm['no_conflict']['median']:.1f}"
    )
    print(
        f"  Conflict:    mean={fm['conflict']['mean']:.2f}, "
        f"median={fm['conflict']['median']:.1f}"
    )
    print(
        f"  Mann-Whitney U: stat={fm['statistic']}, p={fm['p_value']:.4f}"
    )
    print(
        f"  Messages with file mentions: "
        f"{fm['total_messages_with_file_mention']}/{fm['total_messages']}"
    )

    lm = spec["line_mentions"]
    print(f"\n--- Specificity: Line Mentions (QUAL-03) ---")
    print(
        f"  No conflict: mean={lm['no_conflict']['mean']:.4f}, "
        f"median={lm['no_conflict']['median']:.1f}"
    )
    print(
        f"  Conflict:    mean={lm['conflict']['mean']:.4f}, "
        f"median={lm['conflict']['median']:.1f}"
    )
    print(
        f"  Messages with line mentions: "
        f"{lm['total_messages_with_line_mention']}/{lm['total_messages']}"
    )
    print(f"  Note: {lm['note']}")

    # QUAL-04: Summary Table
    st = metrics["summary_table"]
    print(f"\n--- Summary Comparison Table (QUAL-04) ---")
    print(f"{'Metric':<35} {'No Conflict':>15} {'Conflict':>15} {'p-value':>10}")
    print("-" * 80)

    pqs = st["plan_question_ratio"]
    nc_str = (
        f"{pqs['no_conflict']['mean']:.2f}" if pqs["no_conflict"]["mean"] else "N/A"
    )
    c_str = f"{pqs['conflict']['mean']:.2f}" if pqs["conflict"]["mean"] else "N/A"
    p_str = f"{pqs['p_value']:.4f}" if pqs["p_value"] else "N/A"
    print(f"{'P:Q ratio (mean, finite)':<35} {nc_str:>15} {c_str:>15} {p_str:>10}")

    fts = st["first_turn_planning"]
    nc_rate = (
        f"{fts['no_conflict']['rate']*100:.1f}%" if fts["no_conflict"]["rate"] else "N/A"
    )
    c_rate = (
        f"{fts['conflict']['rate']*100:.1f}%" if fts["conflict"]["rate"] else "N/A"
    )
    print(
        f"{'First-turn planning rate':<35} {nc_rate:>15} {c_rate:>15} "
        f"{fts['p_value']:.4f}"
    )

    fms = st["file_mentions_per_trajectory"]
    fm_nc = f"{fms['no_conflict']['mean']:.2f}"
    fm_c = f"{fms['conflict']['mean']:.2f}"
    fm_p = f"{fms['p_value']:.4f}" if fms["p_value"] else "N/A"
    print(f"{'File mentions/trajectory (mean)':<35} {fm_nc:>15} {fm_c:>15} {fm_p:>10}")

    lms = st["line_mentions_per_trajectory"]
    lm_nc = f"{lms['no_conflict']['mean']:.4f}"
    lm_c = f"{lms['conflict']['mean']:.4f}"
    print(f"{'Line mentions/trajectory (mean)':<35} {lm_nc:>15} {lm_c:>15} {'sparse':>10}")

    print("\n" + "=" * 80)

    # Metadata
    md = metrics["metadata"]
    print(f"\n--- Metadata ---")
    print(f"  Input: {md['input_file']}")
    print(f"  Coop-comm records: {md['n_coop_comm']}")
    print(f"  Groups: {md['n_clean']} clean, {md['n_failed']} failed")
    print(f"  Total messages: {md.get('total_messages', 'N/A')}")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute qualitative transcript analysis metrics "
        "(Plan:Question ratio, first-turn planning, specificity) "
        "and correlate with merge outcomes."
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

    # Filter to coop-comm only
    coop_comm = [r for r in records if r["setting"] == "coop-comm"]
    print(f"Coop-comm records: {len(coop_comm)}")

    # Compute per-trajectory metrics
    trajectories = [compute_trajectory_metrics(r) for r in coop_comm]
    print(f"Computed metrics for {len(trajectories)} trajectories")

    # Group comparisons and statistical tests
    pq_result = compute_plan_question_correlation(trajectories)
    ft_result = compute_first_turn_correlation(trajectories)
    spec_result = compute_specificity_metrics(trajectories)

    # Summary table (QUAL-04)
    summary_table = build_summary_table(pq_result, ft_result, spec_result)

    # Metadata
    n_clean = sum(1 for t in trajectories if t["merge_outcome"] == "merge_clean")
    n_failed = sum(1 for t in trajectories if t["merge_outcome"] == "merge_failed")
    total_messages = sum(t["messages_count"] for t in trajectories)

    # Assemble output
    output = {
        "trajectories": trajectories,
        "plan_question_ratio": pq_result,
        "first_turn_planning": ft_result,
        "specificity": spec_result,
        "summary_table": summary_table,
        "metadata": {
            "input_file": str(args.input),
            "n_coop_comm": len(coop_comm),
            "n_clean": n_clean,
            "n_failed": n_failed,
            "total_messages": total_messages,
            "speech_act_classifier": "imported from analyze_fig5.py (Phase 3)",
            "statistical_tests": [
                "Mann-Whitney U (scipy.stats.mannwhitneyu)",
                "Fisher's exact (scipy.stats.fisher_exact)",
            ],
        },
    }

    # Write JSON output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote metrics to {args.output}")

    # Print human-readable summary
    print_summary_table(output)


if __name__ == "__main__":
    main()
