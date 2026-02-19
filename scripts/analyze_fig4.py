#!/usr/bin/env python3
"""Compute Figure 4 metrics: per-bucket success rates, Wilson CIs, AUC, retention.

Reads: data/results.json
Writes: data/fig4_metrics.json

Computes per-bucket solo and coop success rates stratified by difficulty,
Wilson 95% confidence intervals for all rates, AUC via trapezoidal integration
over populated buckets, and retention ratio (AUC_coop / AUC_solo).

Usage:
    python scripts/analyze_fig4.py
    python scripts/analyze_fig4.py --input data/results.json --output data/fig4_metrics.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "fig4_metrics.json"


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, total: int) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds for the 95% CI (z=1.96).

    Handles edge cases:
        wilson_ci(0, 100)  -> (0.0, ~0.037)  # p=0 with n=100
        wilson_ci(100, 100) -> (~0.963, 1.0)  # p=1
        wilson_ci(0, 0)    -> (0.0, 1.0)      # no data
        wilson_ci(2, 3)    -> (~0.208, ~0.939) # small n
        wilson_ci(1, 100)  -> (~0.002, ~0.054) # rare event
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


def trapezoidal_auc(x: list[float], y: list[float]) -> float:
    """Trapezoidal integration for AUC computation.

    x and y must be same length, sorted by x.
    Returns 0.0 if fewer than 2 points.
    """
    if len(x) < 2:
        return 0.0
    auc = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        auc += (y[i] + y[i + 1]) / 2 * dx
    return round(auc, 6)


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------


def load_and_filter(input_path: Path) -> tuple[list[dict], int]:
    """Load results.json and filter out eval_error and infra_error records.

    Returns (filtered_records, total_records_in_file).
    """
    with open(input_path) as f:
        records = json.load(f)

    total = len(records)
    filtered = [
        r for r in records
        if r.get("eval_error") is None and not r.get("infra_error", False)
    ]

    return filtered, total


# ---------------------------------------------------------------------------
# Per-bucket rate computation
# ---------------------------------------------------------------------------


def compute_bucket_rates(records: list[dict]) -> list[dict]:
    """Compute per-bucket success rates for each setting.

    For solo: uses seed=0 only (to match coop denominator of ~100).
    For coop: uses coop-comm and coop-nocomm.

    Returns list of bucket dicts for populated buckets only.
    """
    # Group by (setting_key, bucket)
    # setting_key maps: solo -> "solo", coop-comm -> "coop_comm", coop-nocomm -> "coop_nocomm"
    groups: dict[tuple[str, int], dict] = defaultdict(lambda: {"successes": 0, "total": 0})

    for r in records:
        setting = r["setting"]
        bucket = r.get("bucket")
        if bucket is None:
            continue

        # For solo: use seed=0 only
        if setting == "solo" and r.get("seed", 0) != 0:
            continue

        # Map setting names to output keys
        setting_key = setting.replace("-", "_")
        key = (setting_key, bucket)
        groups[key]["total"] += 1
        if r["both_passed"]:
            groups[key]["successes"] += 1

    # Find populated buckets (buckets where at least one setting has data)
    all_buckets = sorted(set(b for (_, b) in groups.keys()))

    result = []
    for bucket in all_buckets:
        center = round(0.05 + 0.1 * bucket, 2)
        entry: dict = {"bucket": bucket, "center": center}

        for setting_key in ["solo", "coop_comm", "coop_nocomm"]:
            key = (setting_key, bucket)
            if key in groups:
                g = groups[key]
                rate = g["successes"] / g["total"] if g["total"] > 0 else 0.0
                ci_lower, ci_upper = wilson_ci(g["successes"], g["total"])
                entry[setting_key] = {
                    "successes": g["successes"],
                    "total": g["total"],
                    "rate": round(rate, 6),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            else:
                # No data for this setting in this bucket
                entry[setting_key] = {
                    "successes": 0,
                    "total": 0,
                    "rate": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 1.0,
                }

        result.append(entry)

    return result


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------


def compute_auc(bucket_data: list[dict], setting_key: str) -> dict:
    """Compute AUC via trapezoidal integration over populated buckets.

    Only includes buckets where the setting has data (total > 0).
    Returns dict with value, n_points, and x_range.
    """
    points = []
    for b in bucket_data:
        setting_data = b.get(setting_key)
        if setting_data is not None and setting_data["total"] > 0:
            points.append((b["center"], setting_data["rate"]))

    if len(points) < 2:
        return {
            "value": 0.0,
            "n_points": len(points),
            "x_range": [points[0][0], points[0][0]] if points else [],
        }

    # Sort by x (difficulty center) -- should already be sorted
    points.sort()
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    auc = trapezoidal_auc(x, y)

    return {
        "value": auc,
        "n_points": len(points),
        "x_range": [x[0], x[-1]],
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze(input_path: Path, output_path: Path) -> dict:
    """Run the full Figure 4 analysis pipeline."""
    # Load and filter data
    records, total_records = load_and_filter(input_path)
    records_used = len(records)

    print(f"Loaded {total_records} records from {input_path}")
    print(f"After filtering (eval_error, infra_error): {records_used} records")

    # Compute per-bucket rates
    bucket_data = compute_bucket_rates(records)
    populated_buckets = [b["bucket"] for b in bucket_data]

    print(f"\nPopulated buckets: {populated_buckets}")
    print(f"Total buckets possible: 10")

    # Print per-bucket details
    for b in bucket_data:
        print(f"\n  Bucket {b['bucket']} (center={b['center']}):")
        for sk in ["solo", "coop_comm", "coop_nocomm"]:
            d = b[sk]
            rate_pct = d["rate"] * 100
            print(
                f"    {sk}: {d['successes']}/{d['total']} = {rate_pct:.1f}%"
                f"  CI [{d['ci_lower']:.4f}, {d['ci_upper']:.4f}]"
            )

    # Compute AUC for each setting
    auc_results = {}
    for setting_key in ["solo", "coop_comm", "coop_nocomm"]:
        auc_results[setting_key] = compute_auc(bucket_data, setting_key)

    print(f"\nAUC values:")
    for sk, auc_info in auc_results.items():
        print(f"  {sk}: {auc_info['value']:.6f} (n_points={auc_info['n_points']}, x_range={auc_info['x_range']})")

    # Compute retention
    auc_solo = auc_results["solo"]["value"]
    retention = {}
    for sk in ["coop_comm", "coop_nocomm"]:
        if auc_solo > 0:
            retention[sk] = round(auc_results[sk]["value"] / auc_solo, 6)
        else:
            retention[sk] = None

    print(f"\nRetention (AUC_coop / AUC_solo):")
    for sk, val in retention.items():
        print(f"  {sk}: {val}")

    # Count solo seed=0 records used
    solo_seed0_count = sum(
        1 for r in records
        if r["setting"] == "solo" and r.get("seed", 0) == 0
    )

    # Build output
    output = {
        "buckets": bucket_data,
        "auc": auc_results,
        "retention": retention,
        "metadata": {
            "input_file": str(input_path),
            "total_records": total_records,
            "records_used": records_used,
            "solo_seed": 0,
            "solo_seed0_records": solo_seed0_count,
            "populated_buckets": populated_buckets,
            "total_buckets": 10,
            "sparsity_warning": f"Only {len(populated_buckets)} of 10 buckets populated; AUC is a rough approximation",
        },
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote metrics to {output_path}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Figure 4 metrics: per-bucket rates, Wilson CIs, AUC, retention."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input results file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output metrics file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
