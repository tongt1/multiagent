#!/usr/bin/env python3
"""Tier 3: Parse GPU unit test logs and validate success criteria.

Takes a log directory from a sweep job and checks each success criterion,
printing PASS/FAIL for each item.

Usage:
    python tools/parse_sweep_logs.py --log-dir ~/sweep_jobs/gpu_unit_test/sweep_gpu_unit_test/<uid>/<trial>/
    python tools/parse_sweep_logs.py --log-file /path/to/main.log
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def find_log_file(log_dir: Path) -> Path | None:
    """Find the main log file in a sweep job directory."""
    # Common log file patterns
    candidates = [
        log_dir / "main.log",
        log_dir / "output.log",
        log_dir / "stdout.log",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Try any .log file
    logs = list(log_dir.glob("*.log"))
    if logs:
        return max(logs, key=lambda p: p.stat().st_size)

    # Try any text file
    txts = list(log_dir.glob("*.txt"))
    if txts:
        return max(txts, key=lambda p: p.stat().st_size)

    return None


def check_logs(content: str) -> list[CheckResult]:
    """Run all success criteria checks against log content."""
    results: list[CheckResult] = []

    # 1. Checkpoint loading
    ckpt_load = re.search(r"Loading checkpoint from '(gs://[^']+)'", content)
    if ckpt_load:
        results.append(CheckResult("Checkpoint loading", True, ckpt_load.group(1)))
    else:
        # Alternative patterns
        alt = re.search(r"(?:Loading|Restoring) (?:checkpoint|model|weights)", content, re.I)
        results.append(CheckResult(
            "Checkpoint loading",
            alt is not None,
            alt.group(0) if alt else "No checkpoint loading message found",
        ))

    # 2. Model parameter count
    param_match = re.search(r"Number of model parameters:\s*([\d,.]+)", content)
    if param_match:
        results.append(CheckResult("Parameter count", True, param_match.group(1)))
    else:
        param_alt = re.search(r"(?:model|total)\s*param(?:eter)?s?[:\s]*([\d,.]+[MBK]?)", content, re.I)
        results.append(CheckResult(
            "Parameter count",
            param_alt is not None,
            param_alt.group(0) if param_alt else "Not found",
        ))

    # 3. Training loop started
    train_start = re.search(
        r"(?:Starting fax training loop|Training loop started|Begin training)", content, re.I
    )
    results.append(CheckResult(
        "Training loop start",
        train_start is not None,
        train_start.group(0) if train_start else "Not found",
    ))

    # 4. HF export
    hf_export = re.search(r"Exporting HF model to", content, re.I)
    results.append(CheckResult(
        "HF model export",
        hf_export is not None,
        hf_export.group(0) if hf_export else "Not found",
    ))

    # 5. Export complete marker
    export_complete = "_HF_EXPORT_IS_COMPLETE" in content
    results.append(CheckResult(
        "Export completion marker",
        export_complete,
        "Found" if export_complete else "Not found",
    ))

    # 6. vLLM sidecar model reload
    sidecar_reload = re.search(r"Reloading model weights", content, re.I)
    results.append(CheckResult(
        "vLLM sidecar reload",
        sidecar_reload is not None,
        sidecar_reload.group(0) if sidecar_reload else "Not found",
    ))

    # 7. vLLM GPU worker count
    gpu_workers = re.search(r"Running vllm_workers=(\d+) with (\d+) GPUs per worker", content)
    if gpu_workers:
        results.append(CheckResult(
            "vLLM GPU workers",
            True,
            f"workers={gpu_workers.group(1)}, gpus_per_worker={gpu_workers.group(2)}",
        ))
    else:
        results.append(CheckResult("vLLM GPU workers", False, "Not found"))

    # 8. Training step completion
    step_matches = re.findall(r"Step (\d+)/(\d+) completed", content)
    if step_matches:
        last_step, total = step_matches[-1]
        results.append(CheckResult(
            "Training step completion",
            True,
            f"Completed step {last_step}/{total}",
        ))
    else:
        # Alternative step patterns
        alt_step = re.findall(r"step[:\s]*(\d+)", content, re.I)
        results.append(CheckResult(
            "Training step completion",
            len(alt_step) > 0,
            f"Found {len(alt_step)} step references" if alt_step else "No step completions found",
        ))

    # 9. No k8s failure events
    k8s_failures = []
    if "BackoffLimitExceeded" in content:
        k8s_failures.append("BackoffLimitExceeded")
    if "ReachedMaxRestarts" in content:
        k8s_failures.append("ReachedMaxRestarts")
    if "OOMKilled" in content:
        k8s_failures.append("OOMKilled")
    if "Error" in content and "FATAL" in content:
        k8s_failures.append("FATAL Error")

    results.append(CheckResult(
        "No k8s failures",
        len(k8s_failures) == 0,
        "Clean" if not k8s_failures else f"Found: {', '.join(k8s_failures)}",
    ))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse sweep job logs and validate success criteria")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--log-dir", type=Path, help="Directory containing log files")
    group.add_argument("--log-file", type=Path, help="Direct path to log file")
    args = parser.parse_args()

    if args.log_file:
        log_path = args.log_file
    else:
        log_path = find_log_file(args.log_dir)
        if log_path is None:
            print(f"FAIL: No log files found in {args.log_dir}")
            return 1

    if not log_path.exists():
        print(f"FAIL: Log file not found: {log_path}")
        return 1

    print(f"Parsing: {log_path}")
    print(f"Size: {log_path.stat().st_size / 1024:.1f} KB")
    print()

    content = log_path.read_text(errors="replace")
    results = check_logs(content)

    # Print results
    passed = 0
    failed = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "+" if r.passed else "-"
        print(f"  [{icon}] {status}: {r.name} — {r.detail}")
        if r.passed:
            passed += 1
        else:
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
