#!/usr/bin/env python3
"""Collect benchmark results into unified data store.

Reads result.json, eval.json, and conversation.json from all run directories,
normalizes into flat records, computes difficulty scores and bucket assignments,
and writes to data/results.json.

Usage:
    python scripts/collect_results.py
    python scripts/collect_results.py --skip-difficulty
    python scripts/collect_results.py --cooperbench-dir /path/to/CooperBench
    python scripts/collect_results.py --output /path/to/output.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN_SETTINGS = {
    "command-a-solo": "solo",
    "command-a-solo-seed1": "solo",
    "command-a-solo-seed2": "solo",
    "command-a-coop-comm": "coop-comm",
    "command-a-coop-nocomm": "coop-nocomm",
}

RUN_SEEDS = {
    "command-a-solo": 0,
    "command-a-solo-seed1": 1,
    "command-a-solo-seed2": 2,
}

# Agent status priority: lower number = worse status (used for coop flattening)
STATUS_PRIORITY = {
    "Error": 0,
    "LimitsExceeded": 1,
    "Submitted": 2,
    "Unknown": 3,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_agent_status(result_data: dict, setting: str) -> str:
    """Extract agent status from result.json, flattened for coop mode.

    For solo mode, returns the single agent status directly.
    For coop mode, returns the *worst* status across both agents,
    using priority: Error > LimitsExceeded > Submitted > Unknown.
    """
    if setting == "solo":
        return result_data.get("agent", {}).get("status", "Unknown")
    else:
        agents = result_data.get("agents", {})
        if not agents:
            return "Unknown"
        statuses = [a.get("status", "Unknown") for a in agents.values()]
        return min(statuses, key=lambda s: STATUS_PRIORITY.get(s, 99))


def classify_merge_outcome(eval_data: dict, setting: str) -> str:
    """Map eval merge result to merge outcome category.

    Solo mode always returns 'merge_clean' (no merge step by definition).
    Coop mode classifies based on merge.status and merge.strategy:
      - clean + naive  -> merge_clean
      - clean + union  -> merge_union
      - anything else  -> merge_failed
    """
    if setting == "solo":
        return "merge_clean"
    merge = eval_data.get("merge")
    if merge is None:
        return "merge_failed"
    status = merge.get("status", "error")
    strategy = merge.get("strategy")
    if status == "clean" and strategy == "naive":
        return "merge_clean"
    elif status == "clean" and strategy == "union":
        return "merge_union"
    else:
        return "merge_failed"


# ---------------------------------------------------------------------------
# Per-run collection
# ---------------------------------------------------------------------------


def collect_run(run_name: str, setting: str, logs_dir: Path) -> list[dict]:
    """Collect all records from a single run directory.

    Traverses the log directory structure:
        logs/{run_name}/{solo|coop}/{repo}/{task_id}/f{X}_f{Y}/
    and reads result.json, eval.json, and conversation.json from each.

    Returns a list of normalized records.
    """
    records = []
    run_dir = logs_dir / run_name
    if not run_dir.is_dir():
        print(f"WARNING: Run directory not found: {run_dir}", file=sys.stderr)
        return records

    # Solo runs have solo/ subdir, coop runs have coop/ subdir
    mode_dir = run_dir / ("solo" if setting == "solo" else "coop")
    if not mode_dir.is_dir():
        print(f"WARNING: Mode directory not found: {mode_dir}", file=sys.stderr)
        return records

    seed = RUN_SEEDS.get(run_name, 0)
    skipped = 0

    # Traverse: {repo}/{task_id}/f{X}_f{Y}/
    for repo_dir in sorted(mode_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        for task_dir in sorted(repo_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for pair_dir in sorted(task_dir.iterdir()):
                if not pair_dir.is_dir():
                    continue

                result_path = pair_dir / "result.json"
                eval_path = pair_dir / "eval.json"

                if not result_path.is_file():
                    print(
                        f"WARNING: Missing result.json: {result_path}",
                        file=sys.stderr,
                    )
                    continue

                # eval.json is REQUIRED -- skip directory if missing
                if not eval_path.is_file():
                    print(
                        f"WARNING: Missing eval.json (skipping): {eval_path}",
                        file=sys.stderr,
                    )
                    skipped += 1
                    continue

                with open(result_path) as f:
                    result_data = json.load(f)
                with open(eval_path) as f:
                    eval_data = json.load(f)

                # Read conversation.json (optional, for coop-comm)
                conversation_path = pair_dir / "conversation.json"
                messages = []
                if conversation_path.is_file():
                    with open(conversation_path) as f:
                        messages = json.load(f)

                # Extract agent status
                agent_status = get_agent_status(result_data, setting)

                # Extract eval fields
                both_passed = eval_data.get("both_passed", False)
                feature1_passed = eval_data.get("feature1", {}).get("passed", False)
                feature2_passed = eval_data.get("feature2", {}).get("passed", False)

                # LimitsExceeded override (LOCKED DECISION):
                # Per user decision, LimitsExceeded pairs count as failure
                if agent_status == "LimitsExceeded":
                    both_passed = False

                # Classify merge outcome
                merge_outcome = classify_merge_outcome(eval_data, setting)
                merge_info = eval_data.get("merge")
                merge_status = merge_info.get("status") if merge_info else None
                merge_strategy = merge_info.get("strategy") if merge_info else None

                # Build normalized record
                record = {
                    "repo": result_data.get("repo"),
                    "task_id": result_data.get("task_id"),
                    "features": result_data.get("features"),
                    "setting": setting,
                    "seed": seed,
                    "run_name": run_name,
                    "model": result_data.get("model"),
                    "started_at": result_data.get("started_at"),
                    "duration_seconds": result_data.get("duration_seconds"),
                    "total_cost": result_data.get("total_cost"),
                    "total_steps": result_data.get("total_steps"),
                    "agent_status": agent_status,
                    "infra_error": result_data.get("infra_error", False),
                    "both_passed": both_passed,
                    "feature1_passed": feature1_passed,
                    "feature2_passed": feature2_passed,
                    "merge_outcome": merge_outcome,
                    "merge_status": merge_status,
                    "merge_strategy": merge_strategy,
                    "eval_error": eval_data.get("error"),
                    "messages": messages,
                    "messages_count": len(messages),
                    "difficulty": None,
                    "bucket": None,
                }
                records.append(record)

    if skipped > 0:
        print(
            f"  {run_name}: skipped {skipped} pairs (missing eval.json)",
            file=sys.stderr,
        )

    return records


# ---------------------------------------------------------------------------
# Difficulty computation
# ---------------------------------------------------------------------------


def compute_difficulty(records: list[dict]) -> None:
    """Compute per-pair difficulty from solo results across seeds.

    d(pair) = 1 - mean(both_passed across solo seeds)
    bucket  = min(floor(d * 10), 9)

    Difficulty and bucket are applied to ALL records (all settings, all seeds)
    for the same (repo, task_id, features) triple.
    """
    pair_key = lambda r: (r["repo"], r["task_id"], tuple(r["features"]))
    solo = [r for r in records if r["setting"] == "solo"]

    pair_passes: dict[tuple, list[bool]] = defaultdict(list)
    for r in solo:
        pair_passes[pair_key(r)].append(r["both_passed"])

    pair_difficulty: dict[tuple, tuple[float, int]] = {}
    for key, passes in pair_passes.items():
        d = 1.0 - sum(passes) / len(passes)
        bucket = min(int(d * 10), 9)
        pair_difficulty[key] = (round(d, 4), bucket)

    # Apply to ALL records (all settings, all seeds)
    applied = 0
    for r in records:
        key = pair_key(r)
        if key in pair_difficulty:
            r["difficulty"], r["bucket"] = pair_difficulty[key]
            applied += 1

    print(f"  Difficulty scores applied to {applied}/{len(records)} records")
    if applied < len(records):
        missing = len(records) - applied
        print(
            f"  WARNING: {missing} records missing difficulty (no solo data for pair)",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(records: list[dict]) -> None:
    """Print summary statistics of the unified results store."""
    print("\n" + "=" * 60)
    print("UNIFIED RESULTS STORE SUMMARY")
    print("=" * 60)

    # Total records by setting
    settings = Counter(r["setting"] for r in records)
    print(f"\nTotal records: {len(records)}")
    print("Records by setting:")
    for s, c in sorted(settings.items()):
        print(f"  {s}: {c}")

    # Records by run_name
    runs = Counter(r["run_name"] for r in records)
    print("\nRecords by run:")
    for rn, c in sorted(runs.items()):
        print(f"  {rn}: {c}")

    # Infra error count
    infra = sum(1 for r in records if r["infra_error"])
    print(f"\nInfra errors: {infra}")

    # LimitsExceeded override count
    limits = [r for r in records if r["agent_status"] == "LimitsExceeded"]
    print(f"LimitsExceeded overrides: {len(limits)}")

    # Eval error count
    eval_errors = sum(1 for r in records if r["eval_error"] is not None)
    print(f"Eval errors: {eval_errors}")

    # Pass rates by setting (excluding infra errors)
    print("\nPass rates by setting (excluding infra errors):")
    for setting_name in ["solo", "coop-comm", "coop-nocomm"]:
        setting_records = [
            r
            for r in records
            if r["setting"] == setting_name and not r["infra_error"]
        ]
        # For solo, report per-seed and aggregate
        if setting_name == "solo":
            for seed in sorted(set(r["seed"] for r in setting_records)):
                seed_records = [r for r in setting_records if r["seed"] == seed]
                passed = sum(1 for r in seed_records if r["both_passed"])
                rate = passed / len(seed_records) * 100 if seed_records else 0
                print(
                    f"  {setting_name} (seed {seed}): "
                    f"{passed}/{len(seed_records)} = {rate:.1f}%"
                )
            # Aggregate (seed 0 only, to avoid inflating denominator)
            seed0 = [r for r in setting_records if r["seed"] == 0]
            passed = sum(1 for r in seed0 if r["both_passed"])
            rate = passed / len(seed0) * 100 if seed0 else 0
            print(
                f"  {setting_name} (seed 0 only): "
                f"{passed}/{len(seed0)} = {rate:.1f}%"
            )
        else:
            passed = sum(1 for r in setting_records if r["both_passed"])
            rate = (
                passed / len(setting_records) * 100 if setting_records else 0
            )
            print(
                f"  {setting_name}: "
                f"{passed}/{len(setting_records)} = {rate:.1f}%"
            )

    # Difficulty distribution
    difficulties = [r["difficulty"] for r in records if r["difficulty"] is not None]
    if difficulties:
        unique_diffs = sorted(set(round(d, 4) for d in difficulties))
        diff_counts = Counter(round(d, 4) for d in difficulties)
        print(f"\nDifficulty distribution ({len(difficulties)} records):")
        print(f"  Unique values: {unique_diffs}")
        for d_val in unique_diffs:
            print(f"  d={d_val}: {diff_counts[d_val]} records")

    # Bucket population
    buckets = Counter(r["bucket"] for r in records if r["bucket"] is not None)
    if buckets:
        print("\nBucket population (per-record counts, all settings):")
        for b in range(10):
            count = buckets.get(b, 0)
            bar = "#" * (count // 5) if count > 0 else ""
            print(f"  bucket {b}: {count:>4} {bar}")

        # Per-pair bucket distribution
        pair_key = lambda r: (r["repo"], r["task_id"], tuple(r["features"]))
        unique_pairs: dict[tuple, int | None] = {}
        for r in records:
            k = pair_key(r)
            if k not in unique_pairs:
                unique_pairs[k] = r["bucket"]
        pair_buckets = Counter(v for v in unique_pairs.values() if v is not None)
        print(f"\nPer-pair bucket distribution ({len(unique_pairs)} unique pairs):")
        for b in range(10):
            count = pair_buckets.get(b, 0)
            print(f"  bucket {b}: {count} pairs")
        populated = sum(1 for b in range(10) if pair_buckets.get(b, 0) > 0)
        print(f"  Populated buckets: {populated}/10")

    # Merge outcome distribution
    merge_outcomes = Counter(r["merge_outcome"] for r in records)
    print("\nMerge outcome distribution:")
    for m, c in sorted(merge_outcomes.items()):
        print(f"  {m}: {c}")

    # Coop merge x test cross-product
    coop = [r for r in records if r["setting"] != "solo"]
    if coop:
        cross = Counter(
            (r["merge_outcome"], r["both_passed"]) for r in coop
        )
        print("\nMerge x Test cross-product (coop only):")
        for (m, t), c in sorted(cross.items()):
            print(f"  {m} x both_passed={t}: {c}")

    # Messages in coop-comm
    comm_records = [r for r in records if r["setting"] == "coop-comm"]
    if comm_records:
        with_messages = sum(1 for r in comm_records if r["messages_count"] > 0)
        total_msgs = sum(r["messages_count"] for r in comm_records)
        print(
            f"\nCoop-comm messages: {with_messages}/{len(comm_records)} "
            f"records have messages ({total_msgs} total messages)"
        )

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect benchmark results into unified data store."
    )
    parser.add_argument(
        "--skip-difficulty",
        action="store_true",
        help="Collect without computing difficulty scores",
    )
    parser.add_argument(
        "--cooperbench-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "repos" / "CooperBench",
        help="Path to CooperBench repo root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "results.json",
        help="Output file path",
    )
    args = parser.parse_args()

    logs_dir = args.cooperbench_dir / "logs"
    if not logs_dir.is_dir():
        print(f"ERROR: Logs directory not found: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    print("Collecting benchmark results...")
    print(f"  CooperBench dir: {args.cooperbench_dir}")
    print(f"  Logs dir: {logs_dir}")
    print(f"  Output: {args.output}")
    print()

    records: list[dict] = []
    for run_name, setting in RUN_SETTINGS.items():
        run_records = collect_run(run_name, setting, logs_dir)
        print(f"  {run_name} ({setting}): {len(run_records)} records")
        records.extend(run_records)

    print(f"\nTotal collected: {len(records)} records")

    if not args.skip_difficulty:
        print("\nComputing difficulty scores...")
        compute_difficulty(records)
    else:
        print("\nSkipping difficulty computation (--skip-difficulty)")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nWrote {len(records)} records to {args.output}")

    # Print summary
    print_summary(records)


if __name__ == "__main__":
    main()
