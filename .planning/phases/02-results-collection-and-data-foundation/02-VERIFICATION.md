---
phase: 02-results-collection-and-data-foundation
verified: 2026-02-18T22:19:39Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 2: Results Collection and Data Foundation Verification Report

**Phase Goal:** Raw benchmark logs are normalized into a unified data store with difficulty scores computed and validated, ready for downstream analysis.
**Verified:** 2026-02-18T22:19:39Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Docker eval backend creates containers that stay running (entrypoint override works) | VERIFIED | `entrypoint=""` present at line 96 of `repos/CooperBench/src/cooperbench/eval/backends/docker.py`; commit `6cff50f` in CooperBench sub-repo |
| 2 | Eval produces eval.json for all 100 solo runs with both_passed and feature-level results | VERIFIED | `find logs/command-a-solo -name eval.json | wc -l` = 100; sample eval.json has keys: both_passed, feature1, feature2, merge, error |
| 3 | Eval produces eval.json for all 100 coop-comm runs with merge status/strategy and feature-level results | VERIFIED | `find logs/command-a-coop-comm -name eval.json | wc -l` = 100; sample has merge.status="clean", merge.strategy="naive" |
| 4 | Eval produces eval.json for all 100 coop-nocomm runs with merge status/strategy and feature-level results | VERIFIED | `find logs/command-a-coop-nocomm -name eval.json | wc -l` = 100; sample has merge fields |
| 5 | Two additional solo seed runs exist with 100 pairs each, matching the original 100 pairs | VERIFIED | seed1: 100 result.json + 100 eval.json; seed2: 100 result.json + 100 eval.json; SUMMARY confirms 100% pair overlap |
| 6 | A unified JSON results store exists at data/results.json containing all benchmark outcomes | VERIFIED | `data/results.json` exists with 500 records (300 solo + 100 coop-comm + 100 coop-nocomm), 24 fields per record |
| 7 | Merge outcomes tracked as independent dimension from test outcomes | VERIFIED | merge_outcome and both_passed are separate fields; merge_clean(404) + merge_failed(96); solo always merge_clean |
| 8 | Per-pair difficulty scores d(pair) = 1 - mean(solo_both_passed) computed from 3 seeds | VERIFIED | All 500 records have non-null difficulty; values {0.3333, 0.6667, 1.0}; formula verified for sample pairs |
| 9 | Each record has a bucket assignment (0-9) from 10 equal-width buckets over [0,1] | VERIFIED | All 500 records have non-null bucket; buckets {3, 6, 9} populated (3 of 10); formula bucket=min(floor(d*10),9) confirmed |
| 10 | LimitsExceeded pairs marked as both_passed=false and infra errors flagged separately | VERIFIED | 4 LimitsExceeded records, all have both_passed=False; infra_error field on all records (0 infra errors); 27 eval_errors tracked |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `repos/CooperBench/src/cooperbench/eval/backends/docker.py` | Fixed Docker eval backend with entrypoint override | VERIFIED | Line 96: `entrypoint=""` present; commit 6cff50f in CooperBench repo |
| `repos/CooperBench/logs/command-a-solo/eval_summary.json` | Solo eval summary | VERIFIED | total=100, passed=1, failed=96, errors=3 |
| `repos/CooperBench/logs/command-a-coop-comm/eval_summary.json` | Coop-comm eval summary | VERIFIED | total=100, passed=0, failed=96, errors=4 |
| `repos/CooperBench/logs/command-a-coop-nocomm/eval_summary.json` | Coop-nocomm eval summary | VERIFIED | total=100, passed=0, failed=88, errors=12 |
| `repos/CooperBench/logs/command-a-solo-seed1/` | Solo seed 1 run results | VERIFIED | 100 result.json + 100 eval.json + eval_summary.json |
| `repos/CooperBench/logs/command-a-solo-seed2/` | Solo seed 2 run results | VERIFIED | 100 result.json + 100 eval.json + eval_summary.json |
| `repos/CooperBench/logs/command-a-solo-seed1/eval_summary.json` | Seed 1 eval summary | VERIFIED | total=100, passed=1, failed=95, errors=4 |
| `repos/CooperBench/logs/command-a-solo-seed2/eval_summary.json` | Seed 2 eval summary | VERIFIED | total=100, passed=1, failed=95, errors=4 |
| `scripts/collect_results.py` | Collection pipeline script (>100 lines) | VERIFIED | 465 lines, stdlib-only, has argparse CLI, difficulty computation, merge classification, LimitsExceeded override |
| `data/results.json` | Unified results store with all benchmark outcomes | VERIFIED | 500 records, 24 fields each, all settings, difficulties, buckets |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| cooperbench eval CLI | DockerBackend.create_sandbox() | entrypoint="" parameter | VERIFIED | Line 96 of docker.py: `entrypoint=""` in client.containers.run() |
| eval.json | result.json | same log_dir directory | VERIFIED | Both files co-located in each pair directory across all 500 runs |
| scripts/collect_results.py | repos/CooperBench/logs/*/result.json | pathlib traversal | VERIFIED | Line 133: `result_path = pair_dir / "result.json"` |
| scripts/collect_results.py | repos/CooperBench/logs/*/eval.json | pathlib traversal | VERIFIED | Line 134: `eval_path = pair_dir / "eval.json"` |
| scripts/collect_results.py | data/results.json | json.dump output | VERIFIED | Line 456: `json.dump(records, f, indent=2)` |
| data/results.json | Phase 3 analysis modules | downstream consumer | VERIFIED | File exists at expected path; 500 records with complete schema ready for consumption |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DATA-01 | 02-03 | Normalize upstream log directory structure into unified JSON results store | SATISFIED | `data/results.json` has 500 records from 5 run directories |
| DATA-02 | 02-01, 02-03 | Distinguish infrastructure errors from genuine test failures | SATISFIED | infra_error boolean field on all records; eval_error field; LimitsExceeded both_passed override applied |
| DATA-03 | 02-01, 02-03 | Track merge outcomes as separate dimension from test outcomes | SATISFIED | merge_outcome field (merge_clean/merge_failed) independent of both_passed; solo=merge_clean, coop cross-product tracked |
| FIG4-01 | 02-02, 02-03 | Compute per-task difficulty score d(t) = 1 - Solo(t) in [0,1] | SATISFIED | Difficulty computed from 3 solo seeds; values {0.3333, 0.6667, 1.0}; propagated to all 500 records |
| FIG4-02 | 02-03 | Partition tasks into 10 equal-width buckets over [0,1] | SATISFIED | bucket = min(floor(d*10), 9) applied to all records; buckets 3, 6, 9 populated (3/10 due to low pass rates) |

No orphaned requirements. All 5 phase requirements are claimed by at least one plan and verified as satisfied.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | -- | -- | -- | No anti-patterns found |

No TODO/FIXME/PLACEHOLDER/HACK comments, no empty implementations, no stub returns found in `scripts/collect_results.py` or `repos/CooperBench/src/cooperbench/eval/backends/docker.py`.

### Human Verification Required

None. All checks are programmatically verifiable:
- File existence and counts verified via filesystem operations
- Data schema and content verified via JSON parsing
- Code patterns verified via grep
- Commit history verified in CooperBench sub-repo

### Notes

1. **Difficulty bucket sparsity:** Only 3 of 10 difficulty buckets are populated (3, 6, 9) with 98% of pairs at maximum difficulty (d=1.0). This is a known limitation of single-model reproduction (vs the paper's multi-model approach) and does not indicate a bug. The collection script and bucket assignment logic are correct.

2. **No merge_union outcomes:** All coop merges were either merge_clean (naive strategy) or merge_failed. No union strategy merges were observed. This means agents' patches either applied cleanly or conflicted entirely.

3. **Commit 6cff50f:** This commit exists in the CooperBench sub-repo (`repos/CooperBench/`) rather than the main repo, which is expected since docker.py is in the CooperBench codebase. Commit 883cecb (collection script) is in the main repo.

4. **Eval errors (27):** These are captured in the eval_error field and tracked separately. They represent test execution failures (timeouts, compilation errors), not data collection issues.

---

_Verified: 2026-02-18T22:19:39Z_
_Verifier: Claude (gsd-verifier)_
